"""Authoritative economic state, commands, phase clock, and persistence."""

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from src.economy import accounting, logistics, markets, politics, production
from src.economy.catalog import (
    COMMODITIES, NODES, RECIPES, ROUTES, RouteSpec, empty_stock, stock_weight,
)
from src.economy.model import (
    Action, ActionResult, Audit, EconomyConfig, Enterprise, Facility,
    InvalidAction, Node, Phase, Shipment, Warehouse,
)


SCHEMA_VERSION = 1


def catalog_hash() -> str:
    payload = {
        "commodities": {
            key: asdict(value) for key, value in COMMODITIES.items()
        },
        "recipes": {key: asdict(value) for key, value in RECIPES.items()},
        "nodes": {key: asdict(value) for key, value in NODES.items()},
        "routes": {key: asdict(value) for key, value in ROUTES.items()},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


class Economy:
    """One shared world with isolated enterprise ledgers and political views.

    ``act`` spends action points but never advances time. ``advance`` settles
    one phase, including competitors and public institutions. This separates
    human/agent decisions from the deterministic simulation clock.
    """

    def __init__(self, config: EconomyConfig | None = None) -> None:
        self.config = config or EconomyConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.tick = 0
        self.sequence = 0
        self.border_tension = 0.15
        self.trade_ban = False
        self.routes = dict(ROUTES)
        self.nodes: dict[str, Node] = {}
        self.enterprises: dict[str, Enterprise] = {}
        self.facilities: dict[str, Facility] = {}
        self.shipments: dict[str, Shipment] = {}
        self.audits: dict[str, Audit] = {}
        self.events: list[dict] = []
        self.goods_created = empty_stock()
        self.goods_destroyed = empty_stock()
        self._initialize_nodes()
        self._initialize_enterprises()
        self.initial_goods = accounting.goods_total(self)
        self.initial_cash = accounting.cash_total(self)
        self.check()

    def _initialize_nodes(self) -> None:
        for node_id, spec in NODES.items():
            needs = empty_stock()
            for commodity, amount in production.DAILY_NEEDS.items():
                multiplier = (
                    spec.food_multiplier
                    if commodity in {"grain", "rations"} else 1.0
                )
                needs[commodity] += (
                    amount * spec.population / 1000 * multiplier
                )
            for recipe_id, capacity in spec.production.items():
                for commodity, quantity in RECIPES[recipe_id].inputs.items():
                    needs[commodity] += quantity * capacity
            public_needs = production.INSTITUTIONAL_NEEDS[node_id]
            for commodity, quantity in public_needs.items():
                needs[commodity] += quantity
            target = {
                commodity: max(80.0, needs[commodity] * 12)
                for commodity in COMMODITIES
            }
            local_outputs = {
                commodity for recipe in spec.production
                for commodity in RECIPES[recipe].outputs
            }
            stock = {
                commodity: quantity * (
                    1.4 if commodity in local_outputs else 0.85
                )
                for commodity, quantity in target.items()
            }
            prices = {
                commodity: math.log(value.base_price)
                + self.config.scarcity_elasticity
                * math.log(target[commodity] / stock[commodity])
                for commodity, value in COMMODITIES.items()
            }
            self.nodes[node_id] = Node(
                node_id, stock, target, prices,
                household_cash=1000000.0, market_cash=1000000.0,
                treasury=2000000.0,
            )
            index = markets.consumer_price_index(self, node_id)
            self.nodes[node_id].price_index = index
            self.nodes[node_id].prior_day_price_index = index
            for recipe, capacity in spec.production.items():
                facility_id = self.next_id("facility")
                self.facilities[facility_id] = Facility(
                    facility_id, "public", node_id, recipe, capacity,
                )

    def _initialize_enterprises(self) -> None:
        homes = ["belos", "oakhaven", "sinder", "wilbur"]
        for index in range(self.config.npc_enterprises + 1):
            enterprise_id = "player" if index == 0 else f"rival_{index}"
            node_id = homes[index % len(homes)]
            self.enterprises[enterprise_id] = Enterprise(
                enterprise_id, self.config.initial_enterprise_cash,
                {node_id: Warehouse(node_id, self.config.warehouse_capacity)},
                self.config.fleet_capacity, self.config.action_points,
                npc=index > 0, standing=dict.fromkeys(NODES, 0.0),
            )

    @property
    def phase(self) -> Phase:
        return Phase(self.tick % 4)

    @property
    def day(self) -> int:
        return self.tick // 4

    @property
    def season(self) -> str:
        return ("spring", "summer", "autumn", "winter")[
            self.day // self.config.season_days % 4
        ]

    def next_id(self, prefix: str) -> str:
        self.sequence += 1
        return f"{prefix}_{self.sequence:06d}"

    def emit(self, event: str, **details) -> None:
        self.events.append({"tick": self.tick, "event": event, **details})

    @staticmethod
    def validate_quantity(quantity: float) -> None:
        if (
            isinstance(quantity, bool)
            or not isinstance(quantity, (int, float))
            or not math.isfinite(quantity) or quantity <= 0
        ):
            raise InvalidAction("quantity must be finite and positive")

    def warehouse(self, enterprise: str, node: str) -> Warehouse:
        firm = self.enterprises[enterprise]
        if node not in firm.warehouses:
            raise InvalidAction("enterprise has no warehouse at this node")
        return firm.warehouses[node]

    def reserved_capacity(self, enterprise: str, node: str) -> float:
        return sum(
            stock_weight(shipment.cargo)
            for shipment in self.shipments.values()
            if shipment.owner == enterprise and not shipment.public
            and self.routes[shipment.route].destination == node
        )

    def route_load(self, route: str) -> float:
        return sum(
            stock_weight(shipment.cargo)
            for shipment in self.shipments.values() if shipment.route == route
        )

    def change_standing(
        self, enterprise: str, node: str, delta: float,
    ) -> None:
        firm = self.enterprises[enterprise]
        firm.standing[node] = min(
            500.0, max(-500.0, firm.standing[node] + delta),
        )

    def quote(
        self, node: str, commodity: str, quantity: float, buying: bool = True,
    ) -> float:
        return markets.quote(self, node, commodity, quantity, buying)[0]

    def act(self, enterprise: str, action: Action) -> ActionResult:
        """Apply a validated atomic command; invalid commands cost no AP."""
        if enterprise not in self.enterprises:
            return ActionResult(False, "unknown enterprise")
        firm = self.enterprises[enterprise]
        if action.node not in self.nodes:
            return ActionResult(False, "unknown node")
        if firm.ap <= 0:
            return ActionResult(False, "no action points; advance the phase")
        if firm.bankrupt and action.kind != "pay_arrears":
            return ActionResult(False, "enterprise is insolvent")
        try:
            if action.kind in {"buy", "sell"}:
                result = markets.trade(self, enterprise, action)
            elif action.kind == "ship":
                result = logistics.dispatch(self, enterprise, action)
            elif action.kind in {"warehouse", "facility", "fleet"}:
                result = politics.construct(self, enterprise, action)
            elif action.kind == "hold":
                result = ActionResult(True, "action point reserved for rest")
            elif action.kind == "toggle_facility":
                facility = self.facilities.get(action.target)
                if facility is None or facility.owner != enterprise:
                    raise InvalidAction("no owned facility with that id")
                facility.enabled = not facility.enabled
                result = ActionResult(True, "facility operation toggled")
            else:
                result = politics.political_action(self, enterprise, action)
        except InvalidAction as error:
            return ActionResult(False, str(error))
        firm.ap -= 1
        if self.config.check_invariants:
            self.check()
        return result

    def advance(self, phases: int = 1, *, competitors: bool = True) -> None:
        if (
            isinstance(phases, bool) or not isinstance(phases, int)
            or phases < 0
        ):
            raise ValueError("phases must be a nonnegative integer")
        for _ in range(phases):
            if competitors:
                from src.economy.agents import run_competitors

                run_competitors(self)
            if self.phase == Phase.MORNING:
                logistics.public_dispatches(self)
            logistics.advance_shipments(self)
            if self.phase == Phase.DAY:
                production.run_production(self)
            production.consume(self)
            if self.phase == Phase.EVENING:
                politics.update_politics(self)
            if self.phase == Phase.NIGHT:
                production.spoil(self)
                production.daily_finances(self)
            markets.update_prices(self)
            self.tick += 1
            for firm in self.enterprises.values():
                firm.ap = self.config.action_points
            if self.config.check_invariants:
                self.check()

    def net_worth(self, enterprise: str) -> float:
        firm = self.enterprises[enterprise]
        worth = firm.cash - firm.arrears + firm.fleet_book_value
        for node_id, warehouse in firm.warehouses.items():
            worth += warehouse.book_value
            worth += sum(
                quantity * math.exp(self.nodes[node_id].log_prices[commodity])
                for commodity, quantity in warehouse.inventory.items()
            )
        for shipment in self.shipments.values():
            if shipment.owner != enterprise or shipment.public:
                continue
            # In-transit assets retain origin prices until delivered. Dispatch
            # cannot fabricate an immediate gain by repricing at destination.
            origin = self.nodes[self.routes[shipment.route].origin]
            worth += sum(
                quantity * math.exp(origin.log_prices[commodity])
                for commodity, quantity in shipment.cargo.items()
            )
        worth += sum(
            facility.book_value for facility in self.facilities.values()
            if facility.owner == enterprise
        )
        return worth

    def observe(self, enterprise: str = "player") -> dict:
        """Visible state, excluding rival ledgers and hidden thresholds."""
        firm = self.enterprises[enterprise]
        nodes = {}
        for node_id, node in self.nodes.items():
            visible = {
                "name": NODES[node_id].name,
                "population": NODES[node_id].population,
                "prices": {
                    key: math.exp(value)
                    for key, value in node.log_prices.items()
                },
                "market_stock": dict(node.stock),
                "food_satisfaction": node.food_satisfaction,
                "unrest": node.unrest, "strike": node.strike,
                "research": node.research, "standing": firm.standing[node_id],
                "price_index": node.price_index,
                "daily_inflation": node.daily_inflation,
            }
            if firm.intelligence.get(node_id, -1) > self.tick:
                visible["audit_threshold"] = node.audit_threshold
                visible["intelligence_expires"] = firm.intelligence[node_id]
            nodes[node_id] = visible
        return {
            "schema_version": SCHEMA_VERSION,
            "tick": self.tick, "day": self.day,
            "phase": self.phase.name.lower(), "season": self.season,
            "cash": firm.cash, "net_worth": self.net_worth(enterprise),
            "action_points": firm.ap, "bankrupt": firm.bankrupt,
            "arrears": firm.arrears, "fleet_capacity": firm.fleet_capacity,
            "federal_standing": firm.federal_standing,
            "corruption": firm.corruption, "trade_ban": self.trade_ban,
            "nodes": nodes,
            "warehouses": {
                key: asdict(value) for key, value in firm.warehouses.items()
            },
            "shipments": [
                asdict(value) for value in self.shipments.values()
                if value.owner == enterprise and not value.public
            ],
            "facilities": [
                asdict(value) for value in self.facilities.values()
                if value.owner == enterprise
            ],
            "audits": [
                asdict(value) for value in self.audits.values()
                if value.enterprise == enterprise and not value.resolved
            ],
            "permits": list(firm.permits),
            "patrol_contracts": dict(firm.patrol_contracts),
            "routes": {
                key: {**asdict(value), "load": self.route_load(key)}
                for key, value in self.routes.items()
            },
            "competitors": {
                key: {
                    "locations": list(value.warehouses),
                    "bankrupt": value.bankrupt,
                }
                for key, value in self.enterprises.items() if key != enterprise
            },
        }

    def supply_shock(
        self, node: str, commodity: str, multiplier: float,
    ) -> None:
        """Apply an administrative shock with an explicit source/sink entry."""
        if node not in self.nodes or commodity not in COMMODITIES:
            raise ValueError("unknown node or commodity")
        if not math.isfinite(multiplier) or multiplier < 0:
            raise ValueError("multiplier must be finite and nonnegative")
        before = self.nodes[node].stock[commodity]
        after = before * multiplier
        self.nodes[node].stock[commodity] = after
        if after >= before:
            self.goods_created[commodity] += after - before
        else:
            self.goods_destroyed[commodity] += before - after
        self.emit(
            "supply_shock", node=node, commodity=commodity,
            before=before, after=after,
        )
        self.check()

    def check(self) -> None:
        accounting.assert_invariants(self)

    def summary(self) -> dict:
        return {
            "tick": self.tick, "day": self.day, "season": self.season,
            "population": sum(value.population for value in NODES.values()),
            "cash_conservation_error": (
                accounting.cash_total(self) - self.initial_cash
            ),
            "goods_in_transit": len(self.shipments),
            "event_counts": dict(Counter(
                item["event"] for item in self.events
            )),
            "nodes": {
                key: {
                    "food_satisfaction": value.food_satisfaction,
                    "strike": value.strike, "unrest": value.unrest,
                    "research": value.research,
                    "price_index": value.price_index,
                    "daily_inflation": value.daily_inflation,
                    "market_stock_value": sum(
                        quantity * math.exp(value.log_prices[commodity])
                        for commodity, quantity in value.stock.items()
                    ),
                }
                for key, value in self.nodes.items()
            },
            "enterprises": {
                key: {
                    "cash": value.cash, "net_worth": self.net_worth(key),
                    "warehouses": len(value.warehouses),
                    "bankrupt": value.bankrupt, "arrears": value.arrears,
                }
                for key, value in self.enterprises.items()
            },
        }

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION, "catalog_hash": catalog_hash(),
            "config": asdict(self.config),
            "rng_state": self.rng.bit_generator.state,
            "tick": self.tick, "sequence": self.sequence,
            "border_tension": self.border_tension, "trade_ban": self.trade_ban,
            "nodes": {key: asdict(value) for key, value in self.nodes.items()},
            "routes": {
                key: asdict(value) for key, value in self.routes.items()
            },
            "enterprises": {
                key: asdict(value) for key, value in self.enterprises.items()
            },
            "facilities": {
                key: asdict(value) for key, value in self.facilities.items()
            },
            "shipments": {
                key: asdict(value) for key, value in self.shipments.items()
            },
            "audits": {
                key: asdict(value) for key, value in self.audits.items()
            },
            "events": self.events,
            "goods_created": self.goods_created,
            "goods_destroyed": self.goods_destroyed,
            "initial_goods": self.initial_goods,
            "initial_cash": self.initial_cash,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Economy":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported economy save schema")
        if payload.get("catalog_hash") != catalog_hash():
            raise ValueError("save uses a different economic catalog")
        # Isolate loaded state from the caller's mutable dictionary.
        payload = json.loads(json.dumps(payload, allow_nan=False))
        world = cls(EconomyConfig(**payload["config"]))
        world.rng.bit_generator.state = payload["rng_state"]
        for name in (
            "tick", "sequence", "border_tension", "trade_ban", "events",
            "goods_created", "goods_destroyed",
            "initial_goods", "initial_cash",
        ):
            setattr(world, name, payload[name])
        world.nodes = {
            key: Node(**value) for key, value in payload["nodes"].items()
        }
        world.routes = {
            key: RouteSpec(**value) for key, value in payload["routes"].items()
        }
        world.enterprises = {}
        for key, value in payload["enterprises"].items():
            value["warehouses"] = {
                node: Warehouse(**warehouse)
                for node, warehouse in value["warehouses"].items()
            }
            world.enterprises[key] = Enterprise(**value)
        for attribute, constructor in (
            ("facilities", Facility), ("shipments", Shipment),
            ("audits", Audit),
        ):
            setattr(world, attribute, {
                key: constructor(**value)
                for key, value in payload[attribute].items()
            })
        world.check()
        return world

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=destination.parent,
                prefix=f".{destination.name}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(self.to_dict(), stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path: str | Path) -> "Economy":
        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))
