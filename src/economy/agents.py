"""Deterministic rival enterprises acting through the public command API."""

import math
from typing import TYPE_CHECKING

from src.economy.catalog import COMMODITIES, NODES, RECIPES, stock_weight
from src.economy.model import Action, Phase
from src.economy.politics import BUILD_MATERIALS

if TYPE_CHECKING:
    from src.economy.world import Economy


def merchant_action(world: "Economy", enterprise_id: str) -> Action | None:
    """Finance deliveries, service audits, and trade visible local scarcity.

    This policy is a reproducible comparator, not a trained model or an oracle.
    It does not read hidden political thresholds or future market shocks.
    """
    firm = world.enterprises[enterprise_id]
    if firm.bankrupt:
        return None
    for audit in world.audits.values():
        if audit.enterprise == enterprise_id and not audit.resolved:
            if firm.cash >= audit.fine:
                return Action("comply", target=audit.id)
    if world.phase == Phase.NIGHT:
        return None

    # Establish a productive business before spending every morning's AP on
    # arbitrage. Material shortages must not trap it in rejected builds.
    owned = [
        facility for facility in world.facilities.values()
        if facility.owner == firm.id
    ]
    if world.phase == Phase.MORNING and not owned and firm.cash > 20000:
        for node_id in firm.warehouses:
            recipe_id = next(iter(NODES[node_id].production))
            materials = BUILD_MATERIALS["facility"]
            node = world.nodes[node_id]
            cost = RECIPES[recipe_id].build_cost + sum(
                quantity * math.exp(node.log_prices[commodity])
                for commodity, quantity in materials.items()
            )
            if cost < firm.cash * 0.6 and all(
                node.stock[key] >= quantity
                for key, quantity in materials.items()
            ):
                return Action("facility", node_id, recipe=recipe_id)

    # Realize delivered cargo before initiating another financed shipment.
    for node_id, warehouse in firm.warehouses.items():
        for commodity, quantity in warehouse.inventory.items():
            reserved = sum(
                RECIPES[facility.recipe].inputs.get(commodity, 0)
                * facility.capacity * 3
                for facility in world.facilities.values()
                if facility.owner == firm.id and facility.node == node_id
                and facility.enabled
            )
            quantity = max(0.0, quantity - reserved)
            price = math.exp(world.nodes[node_id].log_prices[commodity])
            fair_price = COMMODITIES[commodity].base_price * 0.95
            for facility in owned:
                recipe = RECIPES[facility.recipe]
                if facility.node != node_id or commodity not in recipe.outputs:
                    continue
                variable_cost = recipe.labor_cost + sum(
                    units * math.exp(world.nodes[node_id].log_prices[key])
                    for key, units in recipe.inputs.items()
                )
                minimum = variable_cost / recipe.outputs[commodity] * 1.05
                fair_price = min(fair_price, minimum)
            if quantity > 1 and price >= fair_price:
                quantity = min(quantity, 60.0)
                return Action("sell", node_id, commodity, quantity)

    if world.phase in {Phase.MORNING, Phase.DAY} and firm.cash > 1500:
        opportunities = []
        for route_id, route in world.routes.items():
            if route.origin not in firm.warehouses:
                continue
            for commodity, spec in COMMODITIES.items():
                buy = world.quote(route.origin, commodity, 1.0)
                sell = world.quote(route.destination, commodity, 1.0, False)
                freight = spec.weight * route.cost_per_weight
                margin = (sell - buy - freight) / buy
                stock = world.nodes[route.origin].stock[commodity]
                if margin < 0.18 or stock < 5:
                    continue
                if commodity in {"byrinium", "byrinium_ore"}:
                    if world.trade_ban or firm.federal_standing <= -300:
                        continue
                    if route.restricted and route_id not in firm.permits:
                        continue
                if world.route_load(route_id) >= route.capacity * 0.9:
                    continue
                opportunities.append((margin, route_id, commodity))
        opportunities.sort(reverse=True)
        for _, route_id, commodity in opportunities:
            route = world.routes[route_id]
            if route.destination not in firm.warehouses:
                if world.phase == Phase.MORNING and firm.cash > 8000:
                    return Action("warehouse", route.destination)
                continue
            source = firm.warehouses[route.origin]
            destination = firm.warehouses[route.destination]
            weight = COMMODITIES[commodity].weight
            free_dest = (
                destination.capacity - stock_weight(destination.inventory)
            )
            free_dest -= world.reserved_capacity(firm.id, route.destination)
            free_route = route.capacity - world.route_load(route_id)
            busy_fleet = sum(
                stock_weight(shipment.cargo)
                for shipment in world.shipments.values()
                if shipment.owner == firm.id and not shipment.public
            )
            quantity = min(
                40.0, free_dest / weight, free_route / weight,
                (firm.fleet_capacity - busy_fleet) / weight,
            )
            if quantity < 2:
                continue
            held = source.inventory[commodity]
            if held > 1:
                return Action(
                    "ship", commodity=commodity, quantity=min(held, quantity),
                    route=route_id,
                )
            free_source = source.capacity - stock_weight(source.inventory)
            free_source -= world.reserved_capacity(firm.id, route.origin)
            price = world.quote(route.origin, commodity, 1)
            quantity = min(
                quantity, free_source / weight,
                world.nodes[route.origin].stock[commodity],
                firm.cash * 0.15 / price,
            )
            if quantity >= 2:
                return Action("buy", route.origin, commodity, quantity)

    # Supply owned plants from their local depots using the same paid trades.
    if world.phase in {Phase.MORNING, Phase.EVENING}:
        for facility in world.facilities.values():
            if facility.owner != firm.id or not facility.enabled:
                continue
            inventory = firm.warehouses[facility.node].inventory
            for commodity, units in RECIPES[facility.recipe].inputs.items():
                missing = units * facility.capacity * 3 - inventory[commodity]
                if missing > 0.1:
                    return Action("buy", facility.node, commodity, missing)
    return None


def run_competitors(world: "Economy") -> None:
    rivals = [firm.id for firm in world.enterprises.values() if firm.npc]
    if not rivals:
        return
    offset = world.day % len(rivals)
    for enterprise_id in rivals[offset:] + rivals[:offset]:
        for _ in range(world.enterprises[enterprise_id].ap):
            action = merchant_action(world, enterprise_id)
            if action is None or not world.act(enterprise_id, action).success:
                break
