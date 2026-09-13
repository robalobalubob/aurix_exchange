"""Integrated supply-chain and adverse-policy scenarios, beyond unit checks."""

from dataclasses import replace
import json

import pytest

from src.economy import Action, Economy, EconomyConfig
from src.economy import politics, production
from src.economy.__main__ import interactive, parse_action
from src.economy.catalog import NODES
from src.economy.model import Audit, Phase


def act(world, kind, node="belos", **kwargs):
    result = world.act("player", Action(kind, node, **kwargs))
    assert result.success, result.message
    return result


def morning(world):
    while world.phase != Phase.MORNING:
        world.advance(1, competitors=False)


def deliver(world, shipment_id):
    for _ in range(100):
        if shipment_id not in world.shipments:
            return
        world.advance(1, competitors=False)
    pytest.fail("cargo did not finish its documented route")


def test_player_can_source_transport_refine_manufacture_and_sell():
    world = Economy(EconomyConfig(
        seed=24, npc_enterprises=0, action_points=20,
    ))
    for node in ("wilbur", "crossing", "sinder"):
        act(world, "warehouse", node)
    act(world, "buy", "wilbur", commodity="byrinium_ore", quantity=8)
    act(world, "buy", "wilbur", commodity="iron_ore", quantity=4)
    first = act(
        world, "ship", commodity="byrinium_ore", quantity=8,
        route="wilbur_crossing",
    )
    first_iron = act(
        world, "ship", commodity="iron_ore", quantity=4,
        route="wilbur_crossing",
    )
    deliver(world, first.reference)
    deliver(world, first_iron.reference)
    morning(world)
    act(world, "permit", route="crossing_sinder")
    second = act(
        world, "ship", commodity="byrinium_ore", quantity=8,
        route="crossing_sinder",
    )
    second_iron = act(
        world, "ship", commodity="iron_ore", quantity=4,
        route="crossing_sinder",
    )
    deliver(world, second.reference)
    deliver(world, second_iron.reference)
    morning(world)
    smelter = act(world, "facility", "sinder", recipe="smelter")
    act(world, "buy", "sinder", commodity="fuel", quantity=8)
    act(world, "buy", "sinder", commodity="catalysts", quantity=2)
    world.advance(2, competitors=False)
    warehouse = world.enterprises["player"].warehouses["sinder"]
    assert warehouse.inventory["byrinium"] > 0
    assert world.facilities[smelter.reference].last_batches > 0
    morning(world)
    forge = act(world, "facility", "sinder", recipe="forge")
    world.advance(2, competitors=False)
    assert world.facilities[forge.reference].last_batches > 0
    quantity = warehouse.inventory["tools"]
    assert quantity > 0
    before = world.enterprises["player"].cash
    result = act(world, "sell", "sinder", commodity="tools", quantity=quantity)
    assert world.enterprises["player"].cash > before
    assert result.cost < 0
    assert warehouse.inventory["tools"] == 0
    world.check()


def test_closed_supply_routes_degrade_mining_region_food_security():
    cfg = EconomyConfig(seed=8, npc_enterprises=0, price_noise=0)
    supplied = Economy(cfg)
    isolated = Economy(cfg)
    for world in (supplied, isolated):
        for commodity in ("grain", "rations"):
            world.supply_shock("wilbur", commodity, 0.2)
    for key, route in list(isolated.routes.items()):
        if route.destination == "wilbur":
            isolated.routes[key] = replace(route, capacity=0.0)
    supplied.advance(120, competitors=False)
    isolated.advance(120, competitors=False)
    supplied_food = supplied.nodes["wilbur"].food_satisfaction
    isolated_food = isolated.nodes["wilbur"].food_satisfaction
    assert supplied_food > isolated_food
    assert isolated.nodes["wilbur"].strike
    supplied.check()
    isolated.check()


def test_research_famine_reduces_downstream_refined_output():
    cfg = EconomyConfig(
        seed=2, npc_enterprises=0, public_logistics=False, price_noise=0,
    )
    supplied = Economy(cfg)
    hungry = Economy(cfg)
    # A research-hub food shock with farms/mills temporarily stopped isolates
    # the causal path from hunger to research to the identical Sinder inputs.
    for world in (supplied, hungry):
        for facility in world.facilities.values():
            if facility.node == "oakhaven":
                facility.enabled = False
    hungry.supply_shock("oakhaven", "grain", 0)
    hungry.supply_shock("oakhaven", "rations", 0)
    supplied.advance(24, competitors=False)
    hungry.advance(24, competitors=False)
    hungry_research = hungry.nodes["oakhaven"].research
    supplied_research = supplied.nodes["oakhaven"].research
    assert hungry_research < supplied_research
    for commodity in ("byrinium", "machinery"):
        assert (
            hungry.goods_created[commodity] < supplied.goods_created[commodity]
        )
    supplied.check()
    hungry.check()


def test_embargo_blocks_legal_cargo_even_with_permit():
    world = Economy(EconomyConfig(npc_enterprises=0, action_points=10))
    act(world, "warehouse", "oakhaven")
    act(world, "warehouse", "crossing")
    act(world, "buy", "oakhaven", commodity="byrinium", quantity=2)
    act(world, "permit", route="oakhaven_crossing")
    assert world.events[-1]["node"] == "byrin"
    assert world.events[-1]["route"] == "oakhaven_crossing"
    world.border_tension = 1
    politics.update_politics(world)
    assert world.trade_ban
    result = world.act("player", Action(
        "ship", commodity="byrinium", quantity=2,
        route="oakhaven_crossing",
    ))
    assert not result.success
    assert "ban" in result.message
    assert not world.shipments
    world.check()


def test_agitation_stops_work_and_spying_does_not_reveal_rival_ledgers():
    world = Economy(EconomyConfig(npc_enterprises=1, action_points=10))
    world.tick = 2
    act(world, "agitate", "oakhaven", target="rival_1")
    world.advance(1, competitors=False)
    assert world.nodes["oakhaven"].strike
    assert production.labor_factor(world, "oakhaven", "farm") == 0
    view = world.observe("player")
    assert "enterprises" not in view
    assert "rival_1" not in view["warehouses"]
    assert "audit_threshold" not in view["nodes"]["oakhaven"]
    world.check()


def test_warehouse_permissions_capacity_and_nonlinear_maintenance():
    world = Economy(EconomyConfig(npc_enterprises=0, action_points=40))
    firm = world.enterprises["player"]
    firm.standing["byrin"] = -150
    assert not world.act("player", Action("warehouse", "byrin")).success
    act(world, "warehouse", "belos")
    warehouse = firm.warehouses["belos"]
    assert warehouse.capacity == 2 * world.config.warehouse_capacity
    before = firm.cash
    production.daily_finances(world)
    assert before - firm.cash == pytest.approx(2 * 2 ** 1.6)
    assert before - firm.cash > 2 * 2
    warehouse.plots = NODES["belos"].plots
    assert not world.act("player", Action("warehouse", "belos")).success


def test_fleet_purchase_is_a_capital_asset():
    world = Economy(EconomyConfig(npc_enterprises=0))
    before = world.net_worth("player")
    capacity = world.enterprises["player"].fleet_capacity
    act(world, "fleet")
    assert world.net_worth("player") == pytest.approx(before)
    assert world.enterprises["player"].fleet_capacity == capacity + 500
    world.check()


def test_maintenance_arrears_can_be_settled_and_reopen_firm():
    world = Economy(EconomyConfig(npc_enterprises=0))
    firm = world.enterprises["player"]
    world.nodes["belos"].treasury += firm.cash
    firm.cash = 0
    firm.arrears = 2000
    production.daily_finances(world)
    assert firm.bankrupt
    assert not world.act("player", Action("warehouse")).success
    # A declared cash transfer from the municipality is a rescue loan/grant;
    # this scenario does not invent additional currency.
    world.nodes["belos"].treasury -= 3000
    firm.cash += 3000
    act(world, "pay_arrears")
    assert not firm.bankrupt
    assert firm.arrears == 0
    world.check()


def test_public_sector_consumes_manufactured_and_defense_goods():
    world = Economy(EconomyConfig(npc_enterprises=0))
    production.consume(world)
    assert world.goods_destroyed["machinery"] > 0
    assert world.goods_destroyed["weapons"] > 0
    assert world.goods_destroyed["byrinium"] > 0
    world.check()


@pytest.mark.parametrize("override", [
    {"seed": -1}, {"seed": 1.5}, {"action_points": 0},
    {"season_days": 1.5}, {"brokerage": 1}, {"price_noise": float("nan")},
    {"warehouse_capacity": 0}, {"fleet_capacity": float("inf")},
    {"npc_enterprises": 13}, {"seizure_probability": 1.1},
])
def test_invalid_configuration_is_rejected(override):
    with pytest.raises(ValueError):
        EconomyConfig(**override)


def test_console_grammar_builds_executable_commands_and_json_views():
    world = Economy(EconomyConfig(npc_enterprises=0))
    action = parse_action(["buy", "belos", "tools", "2.5"])
    assert world.act("player", action).success
    inventory = world.enterprises["player"].warehouses["belos"].inventory
    assert inventory["tools"] == 2.5
    shipment = parse_action(["ship", "belos_byrin", "tools", "2", "smuggle"])
    assert shipment.smuggle
    assert parse_action(["facility", "sinder", "forge"]).recipe == "forge"
    with pytest.raises(ValueError):
        parse_action(["ship", "belos_byrin", "tools", "2", "unexpected"])
    assert json.loads(json.dumps(world.observe()))["phase"] == "morning"


@pytest.mark.parametrize("constraint", ["fleet", "route"])
def test_dispatch_rejection_checks_each_capacity_without_mutation(constraint):
    world = Economy(EconomyConfig(npc_enterprises=0, action_points=10))
    act(world, "warehouse", "byrin")
    act(world, "buy", commodity="tools", quantity=5)
    if constraint == "fleet":
        world.enterprises["player"].fleet_capacity = 2
    else:
        world.routes["belos_byrin"] = replace(
            world.routes["belos_byrin"], capacity=2,
        )
    before = json.dumps(world.to_dict(), sort_keys=True)
    result = world.act("player", Action(
        "ship", commodity="tools", quantity=5, route="belos_byrin",
    ))
    assert not result.success
    assert constraint in result.message
    assert json.dumps(world.to_dict(), sort_keys=True) == before


def test_incoming_reservations_also_limit_private_production():
    world = Economy(EconomyConfig(npc_enterprises=0, action_points=10))
    act(world, "warehouse", "byrin")
    factory = act(world, "facility", "byrin", recipe="farm")
    act(world, "buy", commodity="grain", quantity=10)
    act(world, "ship", commodity="grain", quantity=10, route="belos_byrin")
    warehouse = world.enterprises["player"].warehouses["byrin"]
    warehouse.capacity = 11
    production.run_production(world)
    assert warehouse.inventory["grain"] == pytest.approx(1)
    batches = world.facilities[factory.reference].last_batches
    assert batches == pytest.approx(1 / 6)
    assert world.reserved_capacity("player", "byrin") == 10
    world.check()


def test_audit_bribe_costs_cash_and_increases_corruption():
    world = Economy(EconomyConfig(npc_enterprises=0))
    world.audits["test_audit"] = Audit(
        "test_audit", "player", "sinder", 1000, 8,
    )
    firm = world.enterprises["player"]
    before = firm.cash
    act(world, "bribe_audit", target="test_audit")
    assert firm.cash == before - 600
    assert firm.corruption == 15
    assert firm.federal_standing == -10
    assert world.audits["test_audit"].resolved
    assert world.events[-1]["node"] == "sinder"
    assert world.events[-1]["target"] == "test_audit"
    world.check()


def test_seizure_transfers_inbound_cargo_without_losing_ownership():
    world = Economy(EconomyConfig(npc_enterprises=0, action_points=10))
    act(world, "warehouse", "byrin")
    act(world, "buy", commodity="tools", quantity=5)
    shipment = act(
        world, "ship", commodity="tools", quantity=5, route="belos_byrin",
    )
    world.enterprises["player"].standing["byrin"] = -450
    politics.update_politics(world)
    cargo = world.shipments[shipment.reference]
    assert cargo.public and cargo.owner == "public"
    assert "byrin" not in world.enterprises["player"].warehouses
    deliver(world, shipment.reference)
    world.check()


def test_console_runs_complete_delivery_and_resumable_save(
    monkeypatch, capsys, tmp_path,
):
    world = Economy(EconomyConfig(npc_enterprises=0))
    path = (tmp_path / "console.json").as_posix()
    commands = iter([
        "warehouse byrin", "buy belos tools 5", "ship belos_byrin tools 5",
        "next 4", "sell byrin tools 5", "competitors", "routes",
        "status", f'save "{path}"', "quit",
    ])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    interactive(world)
    output = capsys.readouterr().out
    assert "Error:" not in output
    assert '"success": false' not in output
    assert Economy.load(path).to_dict() == world.to_dict()
    inventory = world.enterprises["player"].warehouses["byrin"].inventory
    assert inventory["tools"] == 0


def test_price_index_reports_supply_shock_as_local_inflation():
    world = Economy(EconomyConfig(
        npc_enterprises=0, public_logistics=False, price_noise=0,
    ))
    before = world.nodes["wilbur"].price_index
    world.supply_shock("wilbur", "grain", 0.01)
    world.supply_shock("wilbur", "rations", 0.01)
    world.advance(4, competitors=False)
    node = world.nodes["wilbur"]
    assert node.price_index > before
    assert node.daily_inflation > 0
    visible = world.observe()["nodes"]["wilbur"]
    assert visible["daily_inflation"] == node.daily_inflation
    world.check()


def test_expansion_consumes_materials_and_shortages_reject_atomically():
    world = Economy(EconomyConfig(npc_enterprises=0))
    before = world.nodes["belos"].stock["machinery"]
    act(world, "fleet")
    assert world.nodes["belos"].stock["machinery"] == before - 2
    assert world.goods_destroyed["machinery"] == 2
    world.supply_shock("belos", "timber", 0)
    state = json.dumps(world.to_dict(), sort_keys=True)
    result = world.act("player", Action("warehouse", "belos"))
    assert not result.success
    assert "materials" in result.message
    assert json.dumps(world.to_dict(), sort_keys=True) == state
    world.check()
