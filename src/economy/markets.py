"""Finite local market execution and inventory-driven price adjustment."""

import math
from typing import TYPE_CHECKING

from src.economy.catalog import COMMODITIES, NODES, stock_weight
from src.economy.model import Action, ActionResult, InvalidAction, Phase

if TYPE_CHECKING:
    from src.economy.world import Economy


def quote(
    world: "Economy", node_id: str, commodity: str,
    quantity: float, buying: bool,
) -> tuple[float, float]:
    """Return total cash and end log-price along an exponential depth curve.

    An immediate opposite order traverses the same curve backwards. Brokerage
    makes the round trip strictly loss-making at positive fees.
    """
    world.validate_quantity(quantity)
    if node_id not in world.nodes or commodity not in COMMODITIES:
        raise InvalidAction("unknown node or commodity")
    node = world.nodes[node_id]
    scale = world.config.market_depth / node.target_stock[commodity]
    displacement = scale * quantity
    if displacement > 20:
        raise InvalidAction("order exceeds quoted market depth")
    spot = math.exp(node.log_prices[commodity])
    direction = 1.0 if buying else -1.0
    if scale == 0:
        before_fee = spot * quantity
    else:
        before_fee = spot * math.expm1(direction * displacement)
        before_fee /= direction * scale
    amount = before_fee * (1 + direction * world.config.brokerage)
    return amount, node.log_prices[commodity] + direction * displacement


def trade(
    world: "Economy", enterprise_id: str, action: Action,
) -> ActionResult:
    if world.phase == Phase.NIGHT:
        raise InvalidAction("municipal markets are closed at night")
    firm = world.enterprises[enterprise_id]
    warehouse = world.warehouse(enterprise_id, action.node)
    buying = action.kind == "buy"
    amount, end_price = quote(
        world, action.node, action.commodity, action.quantity, buying,
    )
    node = world.nodes[action.node]
    quantity = action.quantity
    commodity = action.commodity
    fee = world.config.brokerage
    if buying:
        if node.stock[commodity] + 1e-9 < quantity:
            raise InvalidAction("market has insufficient physical stock")
        if firm.cash + 1e-9 < amount:
            raise InvalidAction("insufficient cash")
        free = warehouse.capacity - stock_weight(warehouse.inventory)
        free -= world.reserved_capacity(enterprise_id, action.node)
        if quantity * COMMODITIES[commodity].weight > free + 1e-9:
            raise InvalidAction("warehouse capacity is reserved or full")
        firm.cash -= amount
        node.market_cash += amount / (1 + fee)
        node.treasury += amount - amount / (1 + fee)
        node.stock[commodity] -= quantity
        warehouse.inventory[commodity] += quantity
    else:
        if warehouse.inventory[commodity] + 1e-9 < quantity:
            raise InvalidAction("insufficient warehouse stock")
        gross = amount / (1 - fee)
        if node.market_cash + 1e-9 < gross:
            raise InvalidAction("market cannot fund this purchase")
        node.market_cash -= gross
        firm.cash += amount
        node.treasury += gross - amount
        warehouse.inventory[commodity] -= quantity
        node.stock[commodity] += quantity
        # Supplying shortages builds local standing, without creating currency.
        if commodity in {"grain", "rations", "fuel"}:
            world.change_standing(firm.id, action.node, min(2, quantity / 50))
    node.log_prices[commodity] = end_price
    world.emit(
        "trade", enterprise=enterprise_id, node=action.node,
        commodity=commodity, quantity=quantity, buying=buying, cash=amount,
    )
    return ActionResult(True, "trade executed", amount if buying else -amount)


def consumer_price_index(world: "Economy", node_id: str) -> float:
    """Cost of the local household basket relative to catalog base prices."""
    from src.economy.production import DAILY_NEEDS

    current = 0.0
    baseline = 0.0
    for commodity, amount in DAILY_NEEDS.items():
        if commodity in {"grain", "rations"}:
            amount *= NODES[node_id].food_multiplier
        log_price = world.nodes[node_id].log_prices[commodity]
        current += amount * math.exp(log_price)
        baseline += amount * COMMODITIES[commodity].base_price
    return current / baseline


def update_prices(world: "Economy") -> None:
    """Move toward scarcity targets with small seeded residual shocks."""
    cfg = world.config
    for node in world.nodes.values():
        for commodity, spec in COMMODITIES.items():
            target = node.target_stock[commodity]
            available = max(node.stock[commodity], target * 0.02)
            scarcity = math.log(target / available)
            unmet = node.unmet_demand[commodity] / target
            log_target = math.log(spec.base_price)
            log_target += cfg.scarcity_elasticity * scarcity
            log_target += min(0.5, unmet) + 0.15 * node.unrest
            low = math.log(spec.base_price * 0.15)
            high = math.log(spec.base_price * 12.0)
            next_price = (
                node.log_prices[commodity]
                + cfg.price_adjustment * (
                    log_target - node.log_prices[commodity]
                )
                + cfg.price_noise * float(world.rng.standard_normal())
            )
            node.log_prices[commodity] = min(high, max(low, next_price))
        node.price_index = consumer_price_index(world, node.id)
        if world.phase == Phase.NIGHT:
            node.daily_inflation = (
                node.price_index / node.prior_day_price_index - 1
            )
            node.prior_day_price_index = node.price_index
