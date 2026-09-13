"""Regulation, warehouse development, intelligence, and localized sanctions."""

import math
from typing import TYPE_CHECKING

from src.economy.catalog import NODES, RECIPES
from src.economy.model import (
    Action, ActionResult, Audit, Facility, InvalidAction, Phase, Warehouse,
)

if TYPE_CHECKING:
    from src.economy.world import Economy


def spend(world: "Economy", enterprise: str, node: str, amount: float) -> None:
    firm = world.enterprises[enterprise]
    if firm.cash < amount:
        raise InvalidAction("insufficient funds")
    firm.cash -= amount
    world.nodes[node].treasury += amount


BUILD_MATERIALS = {
    "warehouse": {"timber": 10.0, "tools": 2.0},
    "facility": {"tools": 4.0, "machinery": 1.0},
    "fleet": {"machinery": 2.0, "tools": 4.0, "timber": 5.0},
}


def fund_construction(
    world: "Economy", enterprise: str, node_id: str,
    kind: str, base_cost: float,
) -> float:
    """Procure actual local materials and pay a municipal building contract."""
    node = world.nodes[node_id]
    materials = BUILD_MATERIALS[kind]
    if any(node.stock[key] < value for key, value in materials.items()):
        raise InvalidAction("local construction materials are unavailable")
    material_cost = sum(
        quantity * math.exp(node.log_prices[commodity])
        for commodity, quantity in materials.items()
    )
    total = base_cost + material_cost
    firm = world.enterprises[enterprise]
    if firm.cash < total:
        raise InvalidAction("insufficient construction and material funds")
    firm.cash -= total
    node.treasury += base_cost
    node.market_cash += material_cost
    for commodity, quantity in materials.items():
        node.stock[commodity] -= quantity
        world.goods_destroyed[commodity] += quantity
    world.emit(
        "construction_materials", enterprise=enterprise, node=node_id,
        kind=kind, materials=dict(materials), material_cost=material_cost,
    )
    return total


def construct(
    world: "Economy", enterprise_id: str, action: Action,
) -> ActionResult:
    if world.phase != Phase.MORNING:
        raise InvalidAction("construction requests require morning")
    firm = world.enterprises[enterprise_id]
    if firm.standing[action.node] < -100 or firm.federal_standing <= -300:
        raise InvalidAction("council denied construction permission")
    if action.kind == "warehouse":
        occupied = sum(
            other.warehouses[action.node].plots
            for other in world.enterprises.values()
            if action.node in other.warehouses
        )
        maximum = NODES[action.node].plots
        if occupied >= maximum:
            raise InvalidAction("no municipal plots remain")
        cost = fund_construction(
            world, enterprise_id, action.node, action.kind,
            2000 * (1 + occupied / maximum) ** 2,
        )
        if action.node in firm.warehouses:
            warehouse = firm.warehouses[action.node]
            warehouse.capacity += world.config.warehouse_capacity
            warehouse.plots += 1
            warehouse.book_value += cost
        else:
            firm.warehouses[action.node] = Warehouse(
                action.node, world.config.warehouse_capacity, book_value=cost,
            )
        reference = action.node
    elif action.kind == "facility":
        world.warehouse(enterprise_id, action.node)
        if action.recipe not in RECIPES:
            raise InvalidAction("unknown facility recipe")
        recipe = RECIPES[action.recipe]
        if (
            recipe.extraction
            and action.recipe not in NODES[action.node].production
        ):
            raise InvalidAction("node lacks this extraction resource or land")
        cost = fund_construction(
            world, enterprise_id, action.node, action.kind, recipe.build_cost,
        )
        reference = world.next_id("facility")
        world.facilities[reference] = Facility(
            reference, enterprise_id, action.node, action.recipe,
            2.0, book_value=cost,
        )
    else:
        cost = fund_construction(
            world, enterprise_id, action.node, action.kind, 3000.0,
        )
        firm.fleet_capacity += 500
        firm.fleet_book_value += cost
        reference = enterprise_id
    world.emit(
        "construction", enterprise=enterprise_id, node=action.node,
        kind=action.kind, cost=cost, reference=reference,
    )
    return ActionResult(True, "construction completed", cost, reference)


def political_action(
    world: "Economy", enterprise_id: str, action: Action,
) -> ActionResult:
    firm = world.enterprises[enterprise_id]
    kind = action.kind
    charged_node = action.node
    if kind in {"intelligence", "agitate", "patrol_bribe"}:
        if world.phase != Phase.EVENING:
            raise InvalidAction("clandestine operations require evening")
    if kind == "permit":
        if world.phase != Phase.MORNING or action.route not in world.routes:
            raise InvalidAction("request a known route permit in the morning")
        if action.route in firm.permits:
            raise InvalidAction("permit already held")
        if firm.federal_standing < -100 or world.trade_ban:
            raise InvalidAction("federal council denied permit")
        cost = 500.0
        charged_node = "byrin"
        spend(world, enterprise_id, charged_node, cost)
        firm.permits.append(action.route)
    elif kind == "intelligence":
        cost = 250.0
        spend(world, enterprise_id, action.node, cost)
        firm.intelligence[action.node] = world.tick + 12
    elif kind == "agitate":
        target = world.enterprises.get(action.target)
        if target is None or action.target == enterprise_id:
            raise InvalidAction("agitation requires a rival enterprise")
        if action.node not in target.warehouses:
            raise InvalidAction("target has no presence in this node")
        cost = 600.0
        spend(world, enterprise_id, action.node, cost)
        world.nodes[action.node].agitation_until = world.tick + 12
        world.nodes[action.node].unrest = min(
            1.0, world.nodes[action.node].unrest + 0.25,
        )
        firm.corruption += 5
        world.change_standing(enterprise_id, action.node, -20)
    elif kind == "patrol_bribe":
        if action.route not in world.routes:
            raise InvalidAction("unknown patrol route")
        cost = 300.0
        charged_node = world.routes[action.route].origin
        spend(world, enterprise_id, charged_node, cost)
        firm.patrol_contracts[action.route] = world.tick + 28
        firm.corruption += 10
    elif kind in {"comply", "bribe_audit"}:
        audit = world.audits.get(action.target)
        if (
            audit is None or audit.resolved
            or audit.enterprise != enterprise_id
        ):
            raise InvalidAction("no matching open audit")
        cost = audit.fine * (0.6 if kind == "bribe_audit" else 1)
        charged_node = audit.node
        spend(world, enterprise_id, charged_node, cost)
        audit.resolved = True
        if kind == "bribe_audit":
            firm.corruption += 15
            firm.federal_standing = max(-500, firm.federal_standing - 10)
        else:
            world.change_standing(enterprise_id, audit.node, 15)
    elif kind == "pay_arrears":
        if firm.arrears <= 0:
            raise InvalidAction("no maintenance arrears")
        cost = firm.arrears
        spend(world, enterprise_id, action.node, cost)
        firm.arrears = 0.0
        firm.bankrupt = False
    else:
        raise InvalidAction("unknown political action")
    world.emit(
        "political_action", enterprise=enterprise_id, kind=kind,
        node=charged_node, cost=cost, target=action.target, route=action.route,
    )
    return ActionResult(True, "action completed", cost)


def seize_property(world: "Economy", enterprise_id: str, node_id: str) -> None:
    """Nationalize local stock and auction the plot, preserving global cash."""
    firm = world.enterprises[enterprise_id]
    warehouse = firm.warehouses.pop(node_id, None)
    if warehouse is None:
        return
    node = world.nodes[node_id]
    for commodity, quantity in warehouse.inventory.items():
        node.stock[commodity] += quantity
    warehouse.inventory = dict.fromkeys(warehouse.inventory, 0.0)
    for shipment in world.shipments.values():
        if (
            shipment.owner == enterprise_id and not shipment.public
            and world.routes[shipment.route].destination == node_id
        ):
            shipment.owner = "public"
            shipment.public = True
    for facility in world.facilities.values():
        if facility.owner == enterprise_id and facility.node == node_id:
            facility.owner = "public"
    price = warehouse.book_value * 0.5
    bidders = sorted(
        (
            other for other in world.enterprises.values()
            if other.id != enterprise_id and not other.bankrupt
            and other.cash >= price and other.standing[node_id] >= -100
        ),
        key=lambda other: (-other.cash, other.id),
    )
    buyer = None
    if bidders:
        winner = bidders[0]
        winner.cash -= price
        node.treasury += price
        buyer = winner.id
        if node_id in winner.warehouses:
            target = winner.warehouses[node_id]
            target.capacity += warehouse.capacity
            target.plots += warehouse.plots
            target.book_value += price
        else:
            warehouse.book_value = price
            winner.warehouses[node_id] = warehouse
    world.emit(
        "property_seized", enterprise=enterprise_id, node=node_id,
        auction_buyer=buyer, auction_price=price if buyer else 0,
    )


def update_politics(world: "Economy") -> None:
    world.border_tension = min(1.0, max(
        0.0, world.border_tension
        + float(world.rng.normal(0, 0.015))
        + 0.005 * world.nodes["crossing"].unrest - 0.001,
    ))
    prior_ban = world.trade_ban
    world.trade_ban = world.border_tension >= 0.8
    if prior_ban != world.trade_ban:
        world.emit("trade_ban", active=world.trade_ban)
    for node_id, node in world.nodes.items():
        # Less severe standing now triggers an audit during famine or war.
        node.audit_threshold = (
            -180 + 100 * (1 - node.food_satisfaction)
            + 50 * world.border_tension + 30 * node.unrest
            + min(50, max(0, node.daily_inflation) * 300)
        )
        for firm in world.enterprises.values():
            if node_id not in firm.warehouses:
                continue
            if firm.standing[node_id] <= -400 or firm.federal_standing <= -400:
                seize_property(world, firm.id, node_id)
                continue
            stock = firm.warehouses[node_id].inventory
            hoarded = sum(stock[key] for key in ("grain", "rations", "fuel"))
            deficit = (
                node.food_satisfaction < 0.8 or node.unmet_demand["fuel"] > 0
                or node.daily_inflation > 0.02
            )
            active = any(
                not audit.resolved and audit.enterprise == firm.id
                and audit.node == node_id for audit in world.audits.values()
            )
            threshold = node.target_stock["grain"] * 0.25
            if not active and (
                (deficit and hoarded > threshold)
                or firm.standing[node_id] < node.audit_threshold
            ):
                value = sum(
                    quantity * math.exp(node.log_prices[commodity])
                    for commodity, quantity in stock.items()
                )
                fine = max(100.0, value * (0.03 + 0.07 * node.unrest))
                audit_id = world.next_id("audit")
                world.audits[audit_id] = Audit(
                    audit_id, firm.id, node_id, fine, world.tick + 8,
                )
                world.emit(
                    "audit", audit=audit_id, enterprise=firm.id,
                    node=node_id, fine=fine,
                )
    for audit in world.audits.values():
        if audit.resolved or audit.due_tick > world.tick:
            continue
        firm = world.enterprises[audit.enterprise]
        warehouse = firm.warehouses.get(audit.node)
        if warehouse is not None:
            # Confiscate only enough inventory to satisfy the unpaid levy.
            remaining = audit.fine
            for commodity in sorted(warehouse.inventory):
                price = math.exp(world.nodes[audit.node].log_prices[commodity])
                taken = min(warehouse.inventory[commodity], remaining / price)
                warehouse.inventory[commodity] -= taken
                world.nodes[audit.node].stock[commodity] += taken
                remaining = max(0.0, remaining - taken * price)
            firm.arrears += remaining
        world.change_standing(firm.id, audit.node, -40)
        audit.resolved = True
        world.emit("audit_enforced", audit=audit.id, enterprise=firm.id)
