"""Causal and accounting tests for the independent detailed economy."""

from dataclasses import replace
import json
import math

import pytest

from src.economy import Action, Economy, EconomyConfig
from src.economy import logistics, markets, politics, production
from src.economy.catalog import COMMODITIES, NODES, RECIPES
from src.economy.model import Audit, Phase


@pytest.fixture
def world():
    return Economy(EconomyConfig(
        seed=11, npc_enterprises=0, public_logistics=False,
        price_noise=0, action_points=20,
    ))


def execute(world, kind, node="belos", **kwargs):
    result = world.act("player", Action(kind, node, **kwargs))
    assert result.success, result.message
    return result


def fingerprint(world):
    return json.dumps(world.to_dict(), sort_keys=True, allow_nan=False)


def private_facility(world, node, recipe):
    if node not in world.enterprises["player"].warehouses:
        execute(world, "warehouse", node)
    result = execute(world, "facility", node, recipe=recipe)
    return world.facilities[result.reference]


def test_six_specialized_nodes_and_independent_ledgers(world):
    assert len(world.nodes) == 6
    assert sum(node.population for node in NODES.values()) == 518000
    assert len(COMMODITIES) == 13
    assert len(world.facilities) > 15
    assert set(world.nodes["belos"].stock) == set(COMMODITIES)
    assert world.nodes["belos"].stock is not world.nodes["byrin"].stock
    world.check()


@pytest.mark.parametrize("quantity", [0, -1, float("nan"), float("inf"), True])
def test_invalid_quantities_are_atomic(world, quantity):
    before = fingerprint(world)
    result = world.act("player", Action("buy", quantity=quantity))
    assert not result.success
    assert fingerprint(world) == before


@pytest.mark.parametrize("action", [
    Action("buy", commodity="unknown", quantity=1),
    Action("buy", node="unknown", quantity=1),
    Action("sell", commodity="tools", quantity=1),
    Action("ship", route="unknown", quantity=1),
    Action("facility", recipe="unknown"),
    Action("agitate", target="absent"),
    Action("invalid"),
])
def test_rejected_commands_do_not_spend_or_mutate(world, action):
    before = fingerprint(world)
    assert not world.act("player", action).success
    assert fingerprint(world) == before


@pytest.mark.parametrize("commodity", list(COMMODITIES))
def test_immediate_round_trip_loses_only_spread(world, commodity):
    firm = world.enterprises["player"]
    initial_cash = firm.cash
    initial_stock = world.nodes["belos"].stock[commodity]
    initial_log_price = world.nodes["belos"].log_prices[commodity]
    buy = execute(world, "buy", commodity=commodity, quantity=5)
    sell = execute(world, "sell", commodity=commodity, quantity=5)
    assert firm.cash < initial_cash
    assert firm.cash == pytest.approx(initial_cash - buy.cost - sell.cost)
    node = world.nodes["belos"]
    assert node.stock[commodity] == pytest.approx(initial_stock)
    assert node.log_prices[commodity] == pytest.approx(initial_log_price)
    assert firm.warehouses["belos"].inventory[commodity] == 0
    world.check()


def test_market_stock_and_cash_are_real_constraints(world):
    before = fingerprint(world)
    quantity = world.nodes["belos"].stock["tools"] + 1
    action = Action("buy", commodity="tools", quantity=quantity)
    assert not world.act("player", action).success
    assert fingerprint(world) == before
    execute(world, "buy", commodity="tools", quantity=10)
    node = world.nodes["belos"]
    node.treasury += node.market_cash
    node.market_cash = 0
    before = fingerprint(world)
    action = Action("sell", commodity="tools", quantity=10)
    assert not world.act("player", action).success
    assert fingerprint(world) == before
    world.check()


def test_shortages_raise_prices_and_surpluses_lower_them(world):
    control = Economy.from_dict(world.to_dict())
    world.supply_shock("belos", "grain", 0.1)
    control.supply_shock("belos", "grain", 4.0)
    markets.update_prices(world)
    markets.update_prices(control)
    shortage_price = world.nodes["belos"].log_prices["grain"]
    surplus_price = control.nodes["belos"].log_prices["grain"]
    assert shortage_price > surplus_price
    world.check()
    control.check()


def test_production_requires_inputs_and_pays_actual_wages(world):
    facility = private_facility(world, "sinder", "forge")
    production.run_production(world)
    assert facility.last_batches == 0
    recipe = RECIPES["forge"]
    for commodity, units in recipe.inputs.items():
        execute(
            world, "buy", "sinder", commodity=commodity, quantity=units * 2,
        )
    firm = world.enterprises["player"]
    before = firm.cash
    production.run_production(world)
    assert facility.last_batches == pytest.approx(2)
    inventory = firm.warehouses["sinder"].inventory
    assert inventory["tools"] == pytest.approx(2)
    for commodity in recipe.inputs:
        assert inventory[commodity] == pytest.approx(0)
    assert firm.cash == pytest.approx(before - 2 * recipe.labor_cost)
    world.check()


def test_winter_and_research_have_causal_production_effects(world):
    spring = production.labor_factor(world, "wilbur", "iron_mine")
    world.tick = 3 * world.config.season_days * 4
    assert world.season == "winter"
    winter = production.labor_factor(world, "wilbur", "iron_mine")
    assert winter == pytest.approx(0.6 * spring)
    research_full = production.labor_factor(world, "sinder", "smelter")
    world.nodes["oakhaven"].research = 0
    impaired = production.labor_factor(world, "sinder", "smelter")
    assert impaired == pytest.approx(research_full * 0.35)


def test_food_shortage_stops_mining_and_aid_restores_work(world):
    world.supply_shock("wilbur", "grain", 0)
    world.supply_shock("wilbur", "rations", 0)
    world.advance(9, competitors=False)
    node = world.nodes["wilbur"]
    assert node.strike
    assert production.labor_factor(world, "wilbur", "iron_mine") == 0
    # Transfer relief stocks, not an unexplained inventory mutation.
    for commodity in ("grain", "rations"):
        world.nodes["belos"].stock[commodity] -= 100
        node.stock[commodity] += 100
    world.advance(8, competitors=False)
    assert not node.strike
    assert node.food_satisfaction == 1
    assert production.labor_factor(world, "wilbur", "iron_mine") > 0
    world.check()


def test_dispatch_reserves_capacity_and_keeps_assets_in_transit(world):
    execute(world, "warehouse", "byrin")
    execute(world, "buy", commodity="tools", quantity=10)
    before_worth = world.net_worth("player")
    result = execute(
        world, "ship", route="belos_byrin", commodity="tools", quantity=10,
    )
    firm = world.enterprises["player"]
    assert firm.warehouses["belos"].inventory["tools"] == 0
    assert firm.warehouses["byrin"].inventory["tools"] == 0
    assert world.reserved_capacity("player", "byrin") == 10
    expected_worth = before_worth - result.cost
    assert world.net_worth("player") == pytest.approx(expected_worth)
    assert result.reference in world.shipments
    world.advance(4, competitors=False)
    assert result.reference not in world.shipments
    assert firm.warehouses["byrin"].inventory["tools"] == 10
    world.check()


def test_destination_reservations_prevent_overbooking(world):
    execute(world, "warehouse", "byrin")
    execute(world, "buy", commodity="grain", quantity=400)
    execute(
        world, "ship", route="belos_byrin", commodity="grain", quantity=400,
    )
    world.supply_shock("byrin", "grain", 10)
    before = fingerprint(world)
    action = Action("buy", "byrin", "grain", 650)
    result = world.act("player", action)
    assert not result.success
    assert "capacity" in result.message
    assert fingerprint(world) == before


def test_road_curfew_pauses_time_and_rivers_are_asymmetric(world):
    execute(world, "warehouse", "wilbur")
    execute(world, "buy", commodity="tools", quantity=5)
    result = execute(
        world, "ship", route="belos_wilbur", commodity="tools", quantity=5,
    )
    shipment = world.shipments[result.reference]
    initial = shipment.remaining
    world.tick = 3
    world.advance(1, competitors=False)
    assert shipment.remaining == initial
    world.advance(1, competitors=False)
    assert shipment.remaining == initial - 1
    upstream = world.routes["belos_sinder"].travel_days
    downstream = world.routes["sinder_belos"].travel_days
    assert upstream > downstream


def test_restricted_cargo_needs_permit_and_seizure_is_localized(world):
    execute(world, "warehouse", "oakhaven")
    execute(world, "warehouse", "crossing")
    execute(world, "buy", "oakhaven", commodity="byrinium", quantity=5)
    before = fingerprint(world)
    action = Action(
        "ship", commodity="byrinium", quantity=5, route="oakhaven_crossing",
    )
    assert not world.act("player", action).success
    assert fingerprint(world) == before
    world.tick = 3
    world.config = replace(world.config, seizure_probability=1.0)
    action = replace(action, smuggle=True)
    result = world.act("player", action)
    assert result.success
    firm = world.enterprises["player"]
    cash = firm.cash
    world.advance(1, competitors=False)
    assert result.reference not in world.shipments
    maintenance = sum(
        2 * warehouse.plots ** 1.6 for warehouse in firm.warehouses.values()
    )
    assert firm.cash == pytest.approx(cash - maintenance)
    assert any(item["event"] == "cargo_seized" for item in world.events)
    assert not firm.bankrupt
    world.check()


def test_paid_patrol_contract_removes_restricted_risk_and_expires(world):
    execute(world, "warehouse", "oakhaven")
    execute(world, "warehouse", "crossing")
    execute(world, "buy", "oakhaven", commodity="byrinium", quantity=5)
    world.tick = 2
    execute(world, "patrol_bribe", route="oakhaven_crossing")
    world.tick = 3
    result = execute(
        world, "ship", commodity="byrinium", quantity=5,
        route="oakhaven_crossing", smuggle=True,
    )
    shipment = world.shipments[result.reference]
    assert logistics.seizure_risk(world, shipment) == 0
    world.tick = 31
    probability = world.config.seizure_probability
    assert logistics.seizure_risk(world, shipment) == probability


def test_intelligence_reveals_threshold_temporarily(world):
    assert "audit_threshold" not in world.observe()["nodes"]["belos"]
    world.tick = 2
    execute(world, "intelligence", "belos")
    assert "audit_threshold" in world.observe()["nodes"]["belos"]
    world.tick += 12
    assert "audit_threshold" not in world.observe()["nodes"]["belos"]


def test_hoarding_during_famine_triggers_audit_and_compliance(world):
    execute(world, "buy", commodity="grain", quantity=400)
    world.supply_shock("belos", "grain", 0)
    world.supply_shock("belos", "rations", 0)
    production.consume(world)
    politics.update_politics(world)
    audits = [audit for audit in world.audits.values() if not audit.resolved]
    assert len(audits) == 1
    cash = world.enterprises["player"].cash
    execute(world, "comply", target=audits[0].id)
    assert audits[0].resolved
    final_cash = world.enterprises["player"].cash
    assert final_cash == pytest.approx(cash - audits[0].fine)
    world.check()


def test_expired_audit_confiscates_only_local_assets(world):
    execute(world, "buy", commodity="tools", quantity=10)
    firm = world.enterprises["player"]
    cash = firm.cash
    world.audits["audit_test"] = Audit("audit_test", "player", "belos", 100, 0)
    politics.update_politics(world)
    assert firm.cash == cash
    assert 0 < firm.warehouses["belos"].inventory["tools"] < 10
    assert world.audits["audit_test"].resolved
    world.check()


def test_state_enemy_seizure_auctions_property_and_preserves_goods():
    world = Economy(EconomyConfig(npc_enterprises=1, public_logistics=False))
    execute(world, "buy", commodity="tools", quantity=10)
    firm = world.enterprises["player"]
    cash = firm.cash
    firm.federal_standing = -450
    politics.update_politics(world)
    assert "belos" not in firm.warehouses
    assert "belos" in world.enterprises["rival_1"].warehouses
    assert firm.cash == cash
    world.check()


def test_action_points_and_night_market_closure(world):
    firm = world.enterprises["player"]
    firm.ap = 1
    execute(world, "buy", commodity="tools", quantity=1)
    before = fingerprint(world)
    action = Action("buy", commodity="tools", quantity=1)
    assert not world.act("player", action).success
    assert fingerprint(world) == before
    world.advance(3, competitors=False)
    assert world.phase == Phase.NIGHT
    assert firm.ap == world.config.action_points
    assert not world.act("player", action).success


def test_save_resume_is_exact_and_isolated(tmp_path):
    world = Economy(EconomyConfig(seed=123))
    world.advance(24)
    save = tmp_path / "economy.json"
    world.save(save)
    loaded = Economy.load(save)
    assert fingerprint(loaded) == fingerprint(world)
    world.advance(40)
    loaded.advance(40)
    assert fingerprint(loaded) == fingerprint(world)
    state = world.to_dict()
    copy = Economy.from_dict(state)
    copy.nodes["belos"].stock["grain"] = 0
    assert world.nodes["belos"].stock["grain"] > 0


def test_corrupt_save_fails_closed(world):
    data = json.loads(fingerprint(world))
    data["nodes"]["belos"]["stock"]["grain"] += 1
    with pytest.raises(AssertionError, match="ledger mismatch"):
        Economy.from_dict(data)
    data["schema_version"] = -1
    with pytest.raises(ValueError, match="schema"):
        Economy.from_dict(data)


@pytest.mark.parametrize("seed", [0, 3, 17])
def test_seeded_year_has_real_production_delivery_and_balanced_books(seed):
    world = Economy(EconomyConfig(seed=seed))
    world.advance(120 * 4)
    world.check()
    events = world.summary()["event_counts"]
    assert events["production"] > 100
    assert events["delivery"] > 100
    assert events["trade"] > 100
    assert events["dispatch"] > 0
    assert any(
        event["event"] == "production" and event["owner"] != "public"
        and event["batches"] > 0
        for event in world.events
    )
    assert all(
        math.isfinite(world.net_worth(key)) for key in world.enterprises
    )
    assert all(
        0 <= node.food_satisfaction <= 1 for node in world.nodes.values()
    )
    assert world.season == "spring"
