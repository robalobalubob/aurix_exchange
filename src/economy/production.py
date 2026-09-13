"""Physical production, household consumption, wages, and seasonal labor."""

import math
from typing import TYPE_CHECKING

from src.economy.catalog import COMMODITIES, NODES, RECIPES, stock_weight

if TYPE_CHECKING:
    from src.economy.world import Economy


# Units are wholesale lots per thousand residents per day.
DAILY_NEEDS = {
    "rations": 0.18, "grain": 0.08, "fuel": 0.035,
    "clothing": 0.008, "tools": 0.004,
}

# Public infrastructure, defense, and research consume real manufactured goods.
# Quantities are lots per node per day, paid from that municipality's treasury.
INSTITUTIONAL_NEEDS = {
    "belos": {"iron_ore": 1.0, "machinery": 0.35, "tools": 0.4},
    "byrin": {"byrinium": 0.8, "machinery": 0.4, "tools": 0.6},
    "sinder": {"machinery": 0.25, "clothing": 0.3},
    "oakhaven": {"iron_ore": 0.8, "tools": 0.8, "catalysts": 0.3},
    "crossing": {"weapons": 0.8, "byrinium": 0.15, "rations": 1.2},
    "wilbur": {"tools": 0.5, "clothing": 0.4, "machinery": 0.1},
}


def labor_factor(world: "Economy", node_id: str, recipe: str) -> float:
    node = world.nodes[node_id]
    if node.strike:
        return 0.0
    factor = max(0.1, node.food_satisfaction) * (1 - 0.7 * node.unrest)
    if world.season == "winter":
        if node_id == "wilbur":
            factor *= 0.6
        if recipe == "farm":
            factor *= 0.45
    elif world.season == "autumn" and recipe == "farm":
        factor *= 1.35
    if RECIPES[recipe].research_required:
        factor *= 0.35 + 0.65 * world.nodes["oakhaven"].research
    return factor


def run_production(world: "Economy") -> None:
    for node in world.nodes.values():
        node.last_output = dict.fromkeys(COMMODITIES, 0.0)
    for facility in world.facilities.values():
        facility.last_batches = 0.0
        if not facility.enabled:
            continue
        recipe = RECIPES[facility.recipe]
        node = world.nodes[facility.node]
        public = facility.owner == "public"
        firm = None if public else world.enterprises[facility.owner]
        if firm is not None and (
            firm.bankrupt or facility.node not in firm.warehouses
        ):
            continue
        stock = (
            node.stock if public else firm.warehouses[facility.node].inventory
        )
        cash = node.market_cash if public else firm.cash
        batches = facility.capacity * labor_factor(
            world, facility.node, facility.recipe,
        )
        for commodity, needed in recipe.inputs.items():
            batches = min(batches, stock[commodity] / needed)
        batches = min(batches, cash / recipe.labor_cost)
        if not public:
            output_weight = stock_weight(recipe.outputs)
            input_weight = stock_weight(recipe.inputs)
            delta_weight = output_weight - input_weight
            if delta_weight > 0:
                warehouse = firm.warehouses[facility.node]
                free = warehouse.capacity - stock_weight(stock)
                free -= world.reserved_capacity(firm.id, facility.node)
                batches = min(batches, max(0, free) / delta_weight)
        if batches < 1e-9:
            continue
        for commodity, needed in recipe.inputs.items():
            consumed = needed * batches
            stock[commodity] -= consumed
            world.goods_destroyed[commodity] += consumed
        for commodity, yield_per_batch in recipe.outputs.items():
            produced = yield_per_batch * batches
            stock[commodity] += produced
            world.goods_created[commodity] += produced
            node.last_output[commodity] += produced
        wages = recipe.labor_cost * batches
        if public:
            node.market_cash -= wages
        else:
            firm.cash -= wages
        node.household_cash += wages
        facility.last_batches = batches
        world.emit(
            "production", facility=facility.id, node=facility.node,
            owner=facility.owner, recipe=facility.recipe,
            batches=batches, wages=wages,
        )


def consume(world: "Economy") -> None:
    for node_id, node in world.nodes.items():
        population = NODES[node_id].population / 1000.0
        food_wanted = 0.0
        food_bought = 0.0
        consumed = {}
        expenditure = 0.0
        node.unmet_demand = dict.fromkeys(COMMODITIES, 0.0)
        # Food comes first when household purchasing power is constrained.
        for commodity, daily_amount in DAILY_NEEDS.items():
            need = daily_amount * population / 4.0
            if commodity in {"grain", "rations"}:
                need *= NODES[node_id].food_multiplier
                food_wanted += need
            if commodity in {"fuel", "clothing"} and world.season == "winter":
                need *= 1.6
            price = math.exp(node.log_prices[commodity])
            bought = min(
                need, node.stock[commodity], node.household_cash / price,
            )
            node.stock[commodity] -= bought
            node.household_cash -= bought * price
            node.market_cash += bought * price
            world.goods_destroyed[commodity] += bought
            consumed[commodity] = bought
            expenditure += bought * price
            node.unmet_demand[commodity] = need - bought
            if commodity in {"grain", "rations"}:
                food_bought += bought
        satisfaction = food_bought / food_wanted
        node.food_satisfaction = min(1.0, max(0.0, satisfaction))
        if satisfaction < 0.8:
            node.shortage_days += 0.25
            node.unrest = min(1.0, node.unrest + 0.05 * (1 - satisfaction))
        else:
            node.shortage_days = max(0.0, node.shortage_days - 0.5)
            node.unrest = max(0.0, node.unrest - 0.015)
        prior_strike = node.strike
        agitated = node.agitation_until > world.tick
        node.strike = node.shortage_days >= 2 or agitated
        if prior_strike != node.strike:
            world.emit("strike", node=node_id, active=node.strike)
        if node_id == "oakhaven":
            target = satisfaction * (1 - node.unrest)
            if node.strike:
                target = 0.0
            node.research += 0.15 * (target - node.research)
        world.emit(
            "consumption", node=node_id, sector="household",
            goods=consumed, cash=expenditure,
        )

        public_consumed = {}
        public_expenditure = 0.0
        for commodity, daily_amount in INSTITUTIONAL_NEEDS[node_id].items():
            need = daily_amount / 4
            if node_id == "crossing" and commodity == "weapons":
                need *= 1 + world.border_tension
            price = math.exp(node.log_prices[commodity])
            bought = min(need, node.stock[commodity], node.treasury / price)
            node.stock[commodity] -= bought
            node.treasury -= bought * price
            node.market_cash += bought * price
            world.goods_destroyed[commodity] += bought
            node.unmet_demand[commodity] += need - bought
            public_consumed[commodity] = bought
            public_expenditure += bought * price
        world.emit(
            "consumption", node=node_id, sector="public",
            goods=public_consumed, cash=public_expenditure,
        )


def daily_finances(world: "Economy") -> None:
    for node_id, node in world.nodes.items():
        # Tax/rebate circulation sustains household purchasing power without
        # an unaccounted faucet. Insufficient treasury funds limit the rebate.
        tax = node.market_cash * 0.002
        node.market_cash -= tax
        node.treasury += tax
        stipend = min(node.treasury, NODES[node_id].population / 1000 * 3.0)
        node.treasury -= stipend
        node.household_cash += stipend
        world.emit("public_finance", node=node_id, tax=tax, stipend=stipend)
    for firm in world.enterprises.values():
        if firm.bankrupt:
            continue
        for node_id, warehouse in firm.warehouses.items():
            maintenance = 2.0 * warehouse.plots ** 1.6
            paid = min(firm.cash, maintenance)
            firm.cash -= paid
            world.nodes[node_id].treasury += paid
            firm.arrears += maintenance - paid
            world.emit(
                "maintenance", enterprise=firm.id, node=node_id,
                charged=maintenance, paid=paid,
            )
        for route_id in list(firm.patrol_contracts):
            if firm.patrol_contracts[route_id] <= world.tick or firm.cash < 15:
                del firm.patrol_contracts[route_id]
                continue
            firm.cash -= 15
            world.nodes[world.routes[route_id].origin].treasury += 15
        if firm.arrears > 2000:
            firm.bankrupt = True
            world.emit("bankruptcy", enterprise=firm.id, arrears=firm.arrears)


def spoil(world: "Economy") -> None:
    stores = [node.stock for node in world.nodes.values()]
    stores.extend(
        warehouse.inventory
        for firm in world.enterprises.values()
        for warehouse in firm.warehouses.values()
    )
    stores.extend(shipment.cargo for shipment in world.shipments.values())
    destroyed = dict.fromkeys(COMMODITIES, 0.0)
    for stock in stores:
        for commodity, spec in COMMODITIES.items():
            lost = stock.get(commodity, 0.0) * spec.spoilage_daily
            stock[commodity] = stock.get(commodity, 0.0) - lost
            world.goods_destroyed[commodity] += lost
            destroyed[commodity] += lost
    world.emit("spoilage", goods=destroyed)
