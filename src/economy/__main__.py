"""Run, inspect, save, or interactively operate the Aurix economy."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shlex

from src.economy.catalog import COMMODITIES, NODES, RECIPES
from src.economy.model import Action, EconomyConfig
from src.economy.world import Economy


HELP = """Commands:
  status | market NODE | warehouses | routes | facilities | recipes | audits
  competitors
  buy NODE COMMODITY QUANTITY | sell NODE COMMODITY QUANTITY
  ship ROUTE COMMODITY QUANTITY [smuggle]
  warehouse NODE | facility NODE RECIPE | fleet NODE
  permit ROUTE | intelligence NODE | patrol_bribe ROUTE
  agitate NODE RIVAL | comply AUDIT | bribe_audit AUDIT
  toggle_facility FACILITY | pay_arrears NODE
  next [PHASES] | save PATH | help | quit

Nodes: belos, byrin, sinder, oakhaven, crossing, wilbur.
Each phase has action points. Use next to settle production and deliveries.
"""


def print_status(world: Economy, *, player_view: bool = False) -> None:
    summary = world.summary()
    print(
        f"Aurix | day {world.day} | {world.phase.name.lower()}"
        f" | {world.season}"
        f" | {summary['goods_in_transit']} shipments"
    )
    for node_id, node in world.nodes.items():
        print(
            f"  {NODES[node_id].name:18} food {node.food_satisfaction:5.0%}"
            f" | unrest {node.unrest:5.0%} | strike {str(node.strike):5}"
            f" | research {node.research:5.0%}"
            f" | daily inflation {node.daily_inflation:+6.1%}"
        )
    for key, firm in world.enterprises.items():
        if player_view and key != "player":
            print(f"  {key:18} locations: {', '.join(firm.warehouses)}")
            continue
        print(
            f"  {key:18} cash {firm.cash:12,.0f}"
            f" | worth {world.net_worth(key):12,.0f}"
            f" | warehouses {len(firm.warehouses)} | AP {firm.ap}"
        )
    error = summary["cash_conservation_error"]
    print(f"  Currency balance error: {error:.8f}")


def parse_action(words: list[str]) -> Action:
    """Translate the documented console grammar into typed commands."""
    kind, *args = words
    if kind in {"buy", "sell"} and len(args) == 3:
        return Action(kind, args[0], args[1], float(args[2]))
    if kind == "ship" and len(args) in {3, 4}:
        if len(args) == 4 and args[3] != "smuggle":
            raise ValueError("the optional shipping argument is smuggle")
        return Action(
            kind, commodity=args[1], quantity=float(args[2]), route=args[0],
            smuggle=len(args) == 4,
        )
    if kind in {"warehouse", "fleet", "intelligence", "pay_arrears"}:
        if len(args) == 1:
            return Action(kind, args[0])
    if kind == "facility" and len(args) == 2:
        return Action(kind, args[0], recipe=args[1])
    if kind in {"permit", "patrol_bribe"} and len(args) == 1:
        return Action(kind, route=args[0])
    if kind in {"comply", "bribe_audit", "toggle_facility"} and len(args) == 1:
        return Action(kind, target=args[0])
    if kind == "agitate" and len(args) == 2:
        return Action(kind, args[0], target=args[1])
    raise ValueError("invalid command; type help for the command grammar")


def interactive(world: Economy) -> None:
    print(HELP)
    print_status(world, player_view=True)
    while True:
        try:
            words = shlex.split(input("aurix> "))
            if not words:
                continue
            command = words[0]
            if command == "quit":
                break
            if command == "help":
                print(HELP)
            elif command == "status":
                print_status(world, player_view=True)
            elif command == "next":
                world.advance(int(words[1]) if len(words) > 1 else 1)
                print_status(world, player_view=True)
            elif command == "save" and len(words) == 2:
                world.save(words[1])
                print(f"Saved {words[1]}")
            elif command == "market" and len(words) == 2:
                view = world.observe()["nodes"][words[1]]
                for commodity in COMMODITIES:
                    print(
                        f"{commodity:16} "
                        f"price {view['prices'][commodity]:9.2f}"
                        f" stock {view['market_stock'][commodity]:10.2f}"
                    )
            elif command == "recipes":
                recipes = {
                    key: asdict(value) for key, value in RECIPES.items()
                }
                print(json.dumps(recipes, indent=2))
            elif command in {
                "warehouses", "routes", "facilities", "audits", "competitors",
            }:
                print(json.dumps(world.observe()[command], indent=2))
            else:
                result = world.act("player", parse_action(words))
                print(json.dumps(asdict(result)))
        except (ValueError, KeyError, IndexError, OSError) as error:
            print(f"Error: {error}")
        except (EOFError, KeyboardInterrupt):
            print()
            break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rivals", type=int, default=3)
    parser.add_argument("--load", type=Path)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    if args.days < 0:
        parser.error("--days must be nonnegative")
    world = Economy.load(args.load) if args.load else Economy(
        EconomyConfig(seed=args.seed, npc_enterprises=args.rivals),
    )
    if args.interactive:
        interactive(world)
    else:
        world.advance(args.days * 4)
        print_status(world)
    world.check()
    if args.save:
        world.save(args.save)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(world.summary(), indent=2, allow_nan=False),
            encoding="utf-8",
        )
    if args.events:
        args.events.parent.mkdir(parents=True, exist_ok=True)
        with args.events.open("w", encoding="utf-8") as stream:
            for event in world.events:
                stream.write(json.dumps(event, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
