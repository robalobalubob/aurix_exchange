"""Directed, capacity-limited logistics with explicit in-transit ownership."""

import math
from typing import TYPE_CHECKING

from src.economy.catalog import COMMODITIES, stock_weight
from src.economy.model import (
    Action, ActionResult, InvalidAction, Phase, Shipment,
)

if TYPE_CHECKING:
    from src.economy.world import Economy


RESTRICTED_GOODS = {"byrinium", "byrinium_ore"}


def dispatch(
    world: "Economy", enterprise_id: str, action: Action,
) -> ActionResult:
    world.validate_quantity(action.quantity)
    if action.route not in world.routes or action.commodity not in COMMODITIES:
        raise InvalidAction("unknown route or commodity")
    if action.smuggle:
        if world.phase != Phase.NIGHT:
            raise InvalidAction("smuggling departures require night")
    elif world.phase not in {Phase.MORNING, Phase.DAY}:
        raise InvalidAction("legal departures require morning or day")
    route = world.routes[action.route]
    firm = world.enterprises[enterprise_id]
    origin = world.warehouse(enterprise_id, route.origin)
    destination = world.warehouse(enterprise_id, route.destination)
    quantity = action.quantity
    commodity = action.commodity
    weight = quantity * COMMODITIES[commodity].weight
    if origin.inventory[commodity] + 1e-9 < quantity:
        raise InvalidAction("insufficient origin cargo")
    free = destination.capacity - stock_weight(destination.inventory)
    free -= world.reserved_capacity(enterprise_id, route.destination)
    if free + 1e-9 < weight:
        raise InvalidAction("destination cannot reserve the delivery")
    loaded = sum(
        stock_weight(shipment.cargo)
        for shipment in world.shipments.values()
        if not shipment.public and shipment.owner == enterprise_id
    )
    if loaded + weight > firm.fleet_capacity + 1e-9:
        raise InvalidAction("fleet capacity is already committed")
    if world.route_load(action.route) + weight > route.capacity + 1e-9:
        raise InvalidAction("route capacity is already committed")
    restricted = commodity in RESTRICTED_GOODS
    if restricted and not action.smuggle:
        if world.trade_ban or firm.federal_standing <= -300:
            raise InvalidAction("federal Byrinium transit ban is active")
        if route.restricted and action.route not in firm.permits:
            raise InvalidAction("restricted cargo requires a transit permit")
    cost = weight * route.cost_per_weight * (2 if action.smuggle else 1)
    if cost > firm.cash:
        raise InvalidAction("insufficient freight funds")
    origin.inventory[commodity] -= quantity
    firm.cash -= cost
    world.nodes[route.origin].treasury += cost
    shipment_id = world.next_id("shipment")
    world.shipments[shipment_id] = Shipment(
        shipment_id, enterprise_id, action.route, {commodity: quantity},
        route.travel_days * 4, smuggled=action.smuggle,
    )
    world.emit(
        "dispatch", shipment=shipment_id, enterprise=enterprise_id,
        route=action.route, commodity=commodity, quantity=quantity,
        freight=cost, smuggled=action.smuggle,
    )
    return ActionResult(True, "cargo dispatched", cost, shipment_id)


def seizure_risk(world: "Economy", shipment: Shipment) -> float:
    if shipment.public:
        return 0.0
    route = world.routes[shipment.route]
    firm = world.enterprises[shipment.owner]
    if firm.patrol_contracts.get(shipment.route, -1) > world.tick:
        return 0.0
    restricted = any(
        shipment.cargo.get(commodity, 0) > 0
        for commodity in RESTRICTED_GOODS
    )
    if restricted and (
        world.trade_ban or firm.federal_standing <= -300
        or (route.restricted and shipment.route not in firm.permits)
    ):
        return world.config.seizure_probability
    if shipment.smuggled:
        return min(1.0, 0.08 + 0.12 * world.border_tension)
    return 0.0


def advance_shipments(world: "Economy") -> None:
    for shipment_id, shipment in list(world.shipments.items()):
        route = world.routes[shipment.route]
        if shipment.smuggled and world.phase != Phase.NIGHT:
            continue
        if (
            route.mode == "road" and world.phase == Phase.NIGHT
            and not shipment.smuggled
        ):
            continue
        if not shipment.inspection_done:
            risk = seizure_risk(world, shipment)
            shipment.inspection_done = True
            if risk > 0 and world.rng.random() < risk:
                for commodity, quantity in shipment.cargo.items():
                    world.nodes[route.destination].stock[commodity] += quantity
                firm = world.enterprises[shipment.owner]
                firm.federal_standing = max(-500, firm.federal_standing - 60)
                world.change_standing(firm.id, route.origin, -25)
                world.emit(
                    "cargo_seized", shipment=shipment_id,
                    enterprise=firm.id, node=route.destination,
                    cargo=dict(shipment.cargo),
                )
                del world.shipments[shipment_id]
                continue
        shipment.remaining = max(0.0, shipment.remaining - 1.0)
        if shipment.remaining <= 0:
            if shipment.public:
                destination_stock = world.nodes[route.destination].stock
            else:
                firm = world.enterprises[shipment.owner]
                warehouse = firm.warehouses[route.destination]
                destination_stock = warehouse.inventory
            for commodity, quantity in shipment.cargo.items():
                destination_stock[commodity] += quantity
            world.emit(
                "delivery", shipment=shipment_id, owner=shipment.owner,
                node=route.destination, cargo=dict(shipment.cargo),
            )
            del world.shipments[shipment_id]


def public_dispatches(world: "Economy") -> None:
    """Municipal procurement competes for the same route capacity as firms.

    Buyers pay the source depot and freight at dispatch. Goods are then owned
    by the destination municipality; no instantaneous remote inventory appears.
    """
    if not world.config.public_logistics:
        return
    for route_id, route in world.routes.items():
        origin = world.nodes[route.origin]
        destination = world.nodes[route.destination]
        available_weight = route.capacity - world.route_load(route_id)
        for commodity, spec in COMMODITIES.items():
            pending = sum(
                shipment.cargo.get(commodity, 0.0)
                for shipment in world.shipments.values()
                if shipment.public
                and world.routes[shipment.route].destination
                == route.destination
            )
            deficit = (
                destination.target_stock[commodity]
                - destination.stock[commodity] - pending
            )
            surplus = (
                origin.stock[commodity] - origin.target_stock[commodity] * 0.7
            )
            if deficit <= 1 or surplus <= 1:
                continue
            unit_price = math.exp(origin.log_prices[commodity])
            freight = spec.weight * route.cost_per_weight
            quantity = min(
                deficit, surplus, available_weight / spec.weight,
                destination.market_cash / (unit_price + freight),
                route.capacity * 0.25 / spec.weight,
            )
            if quantity < 1:
                continue
            origin.stock[commodity] -= quantity
            destination.market_cash -= quantity * (unit_price + freight)
            origin.market_cash += quantity * unit_price
            origin.treasury += quantity * freight
            shipment_id = world.next_id("shipment")
            world.shipments[shipment_id] = Shipment(
                shipment_id, "public", route_id, {commodity: quantity},
                route.travel_days * 4, public=True,
            )
            available_weight -= quantity * spec.weight
            world.emit(
                "public_dispatch", shipment=shipment_id, route=route_id,
                commodity=commodity, quantity=quantity,
            )
