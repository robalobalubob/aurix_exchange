"""Static, inspectable geography and production technology for Aurix."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Commodity:
    name: str
    base_price: float
    weight: float = 1.0
    spoilage_daily: float = 0.0


COMMODITIES = {
    "grain": Commodity("Grain", 8.0, spoilage_daily=0.002),
    "livestock": Commodity("Livestock", 50.0, 4.0, 0.001),
    "rations": Commodity("Rations", 12.0, spoilage_daily=0.004),
    "timber": Commodity("Timber", 15.0, 2.0),
    "fuel": Commodity("Fuel", 18.0),
    "iron_ore": Commodity("Iron ore", 25.0, 2.0),
    "byrinium_ore": Commodity("Byrinium ore", 60.0, 2.0),
    "byrinium": Commodity("Refined Byrinium", 180.0),
    "tools": Commodity("Tools", 90.0),
    "machinery": Commodity("Machinery", 500.0, 5.0),
    "weapons": Commodity("Weapons", 250.0, 2.0),
    "clothing": Commodity("Winter clothing", 40.0),
    "catalysts": Commodity("Alchemical catalysts", 45.0),
}


@dataclass(frozen=True)
class Recipe:
    inputs: dict[str, float]
    outputs: dict[str, float]
    labor_cost: float
    build_cost: float
    research_required: bool = False
    extraction: bool = False


RECIPES = {
    "farm": Recipe({}, {"grain": 6.0}, 5.0, 2500.0, extraction=True),
    "logging": Recipe({}, {"timber": 3.0}, 6.0, 3000.0, extraction=True),
    "iron_mine": Recipe(
        {"rations": 0.1, "tools": 0.02}, {"iron_ore": 2.0},
        9.0, 6000.0, extraction=True,
    ),
    "byrinium_mine": Recipe(
        {"rations": 0.15, "tools": 0.03}, {"byrinium_ore": 1.0},
        12.0, 9000.0, extraction=True,
    ),
    "ranch": Recipe({"grain": 2.0}, {"livestock": 1.0}, 4.0, 3500.0),
    "slaughterhouse": Recipe(
        {"livestock": 1.0, "fuel": 0.2}, {"rations": 6.0}, 5.0, 4000.0,
    ),
    "mill": Recipe({"grain": 1.5}, {"rations": 2.0}, 3.0, 2500.0),
    "fuelworks": Recipe({"timber": 2.0}, {"fuel": 3.0}, 4.0, 3500.0),
    "alchemist": Recipe(
        {"grain": 0.5, "fuel": 0.3}, {"catalysts": 1.0}, 5.0, 5000.0,
    ),
    "smelter": Recipe(
        {"byrinium_ore": 2.0, "fuel": 1.0, "catalysts": 0.1},
        {"byrinium": 1.0}, 12.0, 10000.0, research_required=True,
    ),
    "forge": Recipe(
        {"iron_ore": 1.0, "fuel": 0.5, "byrinium": 0.1},
        {"tools": 1.0}, 8.0, 6500.0,
    ),
    "machine_shop": Recipe(
        {"tools": 2.0, "byrinium": 1.0, "fuel": 1.0},
        {"machinery": 1.0}, 20.0, 14000.0, research_required=True,
    ),
    "armory": Recipe(
        {"iron_ore": 2.0, "byrinium": 0.5, "fuel": 1.0},
        {"weapons": 1.0}, 15.0, 11000.0,
    ),
    "tailor": Recipe(
        {"livestock": 0.2, "tools": 0.03, "fuel": 0.1},
        {"clothing": 1.0}, 5.0, 3500.0,
    ),
}


@dataclass(frozen=True)
class NodeSpec:
    name: str
    population: int
    production: dict[str, float]
    plots: int
    food_multiplier: float = 1.0


NODES = {
    "belos": NodeSpec(
        "Belos", 125000, {"mill": 30.0, "slaughterhouse": 6.0}, 24,
    ),
    "byrin": NodeSpec(
        "Byrin", 85000, {"farm": 18.0, "tailor": 5.0}, 18,
    ),
    "sinder": NodeSpec(
        "Sinder", 68000,
        {"smelter": 14.0, "forge": 14.0, "machine_shop": 3.0,
         "fuelworks": 14.0, "logging": 12.0},
        20,
    ),
    "oakhaven": NodeSpec(
        "Oakhaven", 195000,
        {"farm": 55.0, "ranch": 10.0, "alchemist": 8.0,
         "logging": 12.0, "mill": 24.0},
        28,
    ),
    "crossing": NodeSpec(
        "River's Crossing", 27000, {"armory": 4.0, "mill": 7.0}, 10,
    ),
    "wilbur": NodeSpec(
        "Wilbur's Drop", 18000,
        {"iron_mine": 24.0, "byrinium_mine": 24.0}, 10, 1.4,
    ),
}


@dataclass(frozen=True)
class RouteSpec:
    origin: str
    destination: str
    mode: str
    travel_days: float
    capacity: float
    cost_per_weight: float
    restricted: bool = False


ROUTES = {
    "belos_wilbur": RouteSpec("belos", "wilbur", "road", 9.5, 600, 0.8),
    "wilbur_belos": RouteSpec("wilbur", "belos", "road", 7.0, 600, 0.6),
    "wilbur_crossing": RouteSpec(
        "wilbur", "crossing", "road", 7.0, 500, 0.5,
    ),
    "crossing_wilbur": RouteSpec(
        "crossing", "wilbur", "road", 9.5, 500, 0.7,
    ),
    "oakhaven_crossing": RouteSpec(
        "oakhaven", "crossing", "river", 2.0, 1000, 0.12, True,
    ),
    "crossing_oakhaven": RouteSpec(
        "crossing", "oakhaven", "river", 4.0, 1000, 0.24, True,
    ),
    "crossing_sinder": RouteSpec(
        "crossing", "sinder", "river", 2.0, 1000, 0.12, True,
    ),
    "sinder_crossing": RouteSpec(
        "sinder", "crossing", "river", 4.0, 1000, 0.24, True,
    ),
    "sinder_belos": RouteSpec("sinder", "belos", "river", 1.0, 1600, 0.10),
    "belos_sinder": RouteSpec("belos", "sinder", "river", 2.0, 1600, 0.20),
    "belos_byrin": RouteSpec("belos", "byrin", "river", 1.0, 1400, 0.10),
    "byrin_belos": RouteSpec("byrin", "belos", "river", 2.0, 1400, 0.20),
    "oakhaven_belos": RouteSpec("oakhaven", "belos", "road", 3.0, 800, 0.25),
    "belos_oakhaven": RouteSpec("belos", "oakhaven", "road", 4.0, 800, 0.35),
}


def empty_stock() -> dict[str, float]:
    """Create a separate complete commodity ledger."""
    return dict.fromkeys(COMMODITIES, 0.0)


def stock_weight(stock: dict[str, float]) -> float:
    return sum(COMMODITIES[key].weight * value for key, value in stock.items())
