"""Accounting across all markets, enterprises, and goods in transit."""

import math
from typing import TYPE_CHECKING

from src.economy.catalog import COMMODITIES, empty_stock, stock_weight

if TYPE_CHECKING:
    from src.economy.world import Economy


def goods_total(world: "Economy") -> dict[str, float]:
    total = empty_stock()
    stores = [node.stock for node in world.nodes.values()]
    stores.extend(
        warehouse.inventory
        for firm in world.enterprises.values()
        for warehouse in firm.warehouses.values()
    )
    stores.extend(shipment.cargo for shipment in world.shipments.values())
    for stock in stores:
        for commodity in COMMODITIES:
            total[commodity] += stock.get(commodity, 0.0)
    return total


def cash_total(world: "Economy") -> float:
    return sum(
        node.market_cash + node.household_cash + node.treasury
        for node in world.nodes.values()
    ) + sum(firm.cash for firm in world.enterprises.values())


def assert_invariants(world: "Economy") -> None:
    """Raise on unexplained creation, loss, debt, or capacity violations."""
    total = goods_total(world)
    for commodity, actual in total.items():
        expected = (
            world.initial_goods[commodity]
            + world.goods_created[commodity]
            - world.goods_destroyed[commodity]
        )
        if not math.isfinite(actual) or not math.isclose(
            actual, expected, rel_tol=1e-9, abs_tol=1e-6,
        ):
            raise AssertionError(
                f"{commodity} ledger mismatch: {actual} versus {expected}"
            )
    if not math.isclose(
        cash_total(world), world.initial_cash, rel_tol=1e-10, abs_tol=1e-5,
    ):
        raise AssertionError("currency creation or destruction")

    stores = [node.stock for node in world.nodes.values()]
    balances = []
    for node in world.nodes.values():
        balances.extend(
            [node.market_cash, node.household_cash, node.treasury]
        )
        if not 0 <= node.food_satisfaction <= 1:
            raise AssertionError("invalid food satisfaction")
        if not 0 <= node.unrest <= 1 or not 0 <= node.research <= 1:
            raise AssertionError("invalid social state")
        if not all(math.isfinite(value) for value in node.log_prices.values()):
            raise AssertionError("invalid prices")
        if (
            not math.isfinite(node.price_index) or node.price_index <= 0
            or not math.isfinite(node.daily_inflation)
        ):
            raise AssertionError("invalid price index")
    for firm in world.enterprises.values():
        balances.append(firm.cash)
        if not math.isfinite(firm.arrears) or firm.arrears < 0:
            raise AssertionError("invalid arrears")
        if not -500 <= firm.federal_standing <= 500:
            raise AssertionError("invalid federal standing")
        if any(not -500 <= value <= 500 for value in firm.standing.values()):
            raise AssertionError("invalid local standing")
        if not 0 <= firm.ap <= world.config.action_points:
            raise AssertionError("invalid action points")
        for node_id, warehouse in firm.warehouses.items():
            if warehouse.capacity <= 0 or warehouse.plots < 1:
                raise AssertionError("invalid warehouse")
            stores.append(warehouse.inventory)
            used = stock_weight(warehouse.inventory)
            reserved = world.reserved_capacity(firm.id, node_id)
            if used + reserved > warehouse.capacity + 1e-7:
                raise AssertionError("warehouse overcommitted")
        loaded = sum(
            stock_weight(shipment.cargo)
            for shipment in world.shipments.values()
            if shipment.owner == firm.id and not shipment.public
        )
        if loaded > firm.fleet_capacity + 1e-7:
            raise AssertionError("fleet overcommitted")
    stores.extend(shipment.cargo for shipment in world.shipments.values())
    for stock in stores:
        if any(
            not math.isfinite(value) or value < -1e-8
            for value in stock.values()
        ):
            raise AssertionError("invalid inventory")
    if any(not math.isfinite(value) or value < -1e-8 for value in balances):
        raise AssertionError("invalid monetary account")
    for route_id, route in world.routes.items():
        if world.route_load(route_id) > route.capacity + 1e-7:
            raise AssertionError("route overcommitted")
    for shipment in world.shipments.values():
        if not math.isfinite(shipment.remaining) or shipment.remaining < 0:
            raise AssertionError("invalid shipment clock")
    for facility in world.facilities.values():
        if not math.isfinite(facility.capacity) or facility.capacity <= 0:
            raise AssertionError("invalid facility capacity")
