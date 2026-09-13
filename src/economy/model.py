"""Serializable state and public command types for the detailed economy."""

from dataclasses import dataclass, field
from enum import IntEnum
import math

from src.economy.catalog import empty_stock


class Phase(IntEnum):
    MORNING = 0
    DAY = 1
    EVENING = 2
    NIGHT = 3


@dataclass(frozen=True)
class EconomyConfig:
    seed: int = 0
    action_points: int = 3
    initial_enterprise_cash: float = 50000.0
    warehouse_capacity: float = 1000.0
    fleet_capacity: float = 500.0
    brokerage: float = 0.01
    market_depth: float = 0.15
    price_adjustment: float = 0.15
    price_noise: float = 0.015
    scarcity_elasticity: float = 0.65
    season_days: int = 30
    seizure_probability: float = 0.45
    npc_enterprises: int = 3
    public_logistics: bool = True
    check_invariants: bool = True

    def __post_init__(self) -> None:
        for name in (
            "seed", "action_points", "season_days", "npc_enterprises",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        for name in (
            "initial_enterprise_cash", "warehouse_capacity", "fleet_capacity",
            "brokerage", "market_depth", "price_adjustment", "price_noise",
            "scarcity_elasticity", "seizure_probability",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.brokerage >= 1 or self.seizure_probability > 1:
            raise ValueError("fees or probabilities exceed their bounds")
        if not 0 < self.price_adjustment <= 1:
            raise ValueError("price_adjustment must be in (0, 1]")
        if self.warehouse_capacity <= 0 or self.fleet_capacity <= 0:
            raise ValueError("warehouse and fleet capacities must be positive")
        if self.action_points < 1 or self.season_days < 1:
            raise ValueError("action_points and season_days must be positive")
        if not 0 <= self.npc_enterprises <= 12:
            raise ValueError("npc_enterprises must be between 0 and 12")


@dataclass
class Node:
    id: str
    stock: dict[str, float]
    target_stock: dict[str, float]
    log_prices: dict[str, float]
    household_cash: float
    market_cash: float
    treasury: float
    food_satisfaction: float = 1.0
    shortage_days: float = 0.0
    unrest: float = 0.0
    strike: bool = False
    agitation_until: int = 0
    research: float = 1.0
    audit_threshold: float = -180.0
    price_index: float = 1.0
    prior_day_price_index: float = 1.0
    daily_inflation: float = 0.0
    last_output: dict[str, float] = field(default_factory=empty_stock)
    unmet_demand: dict[str, float] = field(default_factory=empty_stock)


@dataclass
class Warehouse:
    node: str
    capacity: float
    plots: int = 1
    book_value: float = 2000.0
    inventory: dict[str, float] = field(default_factory=empty_stock)


@dataclass
class Enterprise:
    id: str
    cash: float
    warehouses: dict[str, Warehouse]
    fleet_capacity: float
    ap: int
    npc: bool = False
    standing: dict[str, float] = field(default_factory=dict)
    federal_standing: float = 0.0
    corruption: float = 0.0
    permits: list[str] = field(default_factory=list)
    intelligence: dict[str, int] = field(default_factory=dict)
    patrol_contracts: dict[str, int] = field(default_factory=dict)
    arrears: float = 0.0
    bankrupt: bool = False
    fleet_book_value: float = 3000.0


@dataclass
class Shipment:
    id: str
    owner: str
    route: str
    cargo: dict[str, float]
    remaining: float
    smuggled: bool = False
    public: bool = False
    inspection_done: bool = False


@dataclass
class Facility:
    id: str
    owner: str
    node: str
    recipe: str
    capacity: float
    enabled: bool = True
    book_value: float = 0.0
    last_batches: float = 0.0


@dataclass
class Audit:
    id: str
    enterprise: str
    node: str
    fine: float
    due_tick: int
    resolved: bool = False


@dataclass(frozen=True)
class Action:
    kind: str
    node: str = "belos"
    commodity: str = "grain"
    quantity: float = 0.0
    route: str = ""
    recipe: str = ""
    target: str = ""
    smuggle: bool = False


@dataclass(frozen=True)
class ActionResult:
    success: bool
    message: str
    cost: float = 0.0
    reference: str = ""


class InvalidAction(ValueError):
    """Expected action rejection; the world must remain unchanged."""
