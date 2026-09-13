# Running the detailed Aurix economy

`src.economy.Economy` implements the systemic economy described in `vision.md` as
an independently runnable Python simulation. It is separate from `OUTradingEnv`
and the stationary `AurixExchangeEnv`; their policies, observation schemas,
objectives, and result claims do not transfer to this simulation.

## Start and resume

From the repository root, using its Python environment:

```powershell
# Run one 120-day model year with three active competitors.
.\.venv\Scripts\python.exe -m src.economy --days 120 --seed 7

# Save the full state, administrative summary, and event journal.
.\.venv\Scripts\python.exe -m src.economy --days 30 --seed 7 --save exports/economy/world.json --report exports/economy/report.json --events exports/economy/events.jsonl

# Resume for another 30 days, including the saved random-generator state.
.\.venv\Scripts\python.exe -m src.economy --load exports/economy/world.json --days 30 --save exports/economy/world.json

# Operate the player enterprise interactively from day zero.
.\.venv\Scripts\python.exe -m src.economy --interactive --seed 7
```

`--days` is the number of additional days in batch mode. Interactive mode starts
at the current state without advancing those days. `--rivals 0` creates a world
without private competitors; municipal producers and trade continue operating.
Use `help` in the console for the complete command grammar. Node and commodity
IDs use lowercase names and underscores. Save paths with forward slashes also
work on Windows; quote paths containing spaces.

An introductory console sequence is:

```text
market belos
warehouse byrin
buy belos tools 5
ship belos_byrin tools 5
warehouses
next 4
warehouses
sell byrin tools 5
save exports/economy/player.json
```

This demonstrates a real delivery, not a guaranteed profit. Prices change, freight
and brokerage cost cash, routes have finite capacity, and other buyers may exhaust
local stock. An action rejection explains which constraint prevented execution.

## World and units

The six nodes retain the vision's populations, totaling 518,000 residents:
Belos, Byrin, Sinder, Oakhaven, River's Crossing (`crossing`), and Wilbur's Drop
(`wilbur`). There are thirteen goods: grain, livestock, rations, timber, fuel,
iron ore, Byrinium ore, refined Byrinium, tools, machinery, weapons, clothing,
and catalysts. Inventories are wholesale lots, not individual meals or citizens.
Household needs are specified per thousand residents per day.

Each node has independent market stock, market cash, household cash, and municipal
treasury accounts. Player and rival enterprises own cash, warehouse stock, fleets,
facilities, and cargo in transit. Everyone trades in the same finite markets.

The model year has four 30-day seasons. Every day has Morning, Day, Evening, and
Night. Each enterprise receives three action points per phase by default.
Commands spend AP; `next` advances time explicitly. No action can advance time
implicitly or partially execute after rejection.

| Phase | Operations |
|---|---|
| Morning | Legal trades, warehouse/facility/fleet purchases, permits, dispatch; municipal procurement |
| Day | Legal trades and dispatch; production, wage payments, and freight movement |
| Evening | Legal trades, intelligence, labor agitation, patrol contracts; audits and political updates |
| Night | Smuggling departures and movement; legal roads pause, legal rivers continue; spoilage, maintenance, taxes and rebates |

Households consume in every phase. Prices update at the end of every phase.
Audit compliance, audit bribery, and arrears settlement are available in any phase
when the enterprise has AP. Empty AP replenishes at the next phase, not on a
failed command.

## Markets and money

Local log prices move toward an inventory-dependent target:

```text
log target = log base price
           + elasticity * log(target stock / available stock)
           + unmet-demand pressure + unrest pressure
```

Available stock has a small numerical floor; end-of-phase prices are bounded to
0.15–12 times the base price. Within a phase, a block trade can temporarily move
execution prices beyond that band. There is a small seeded residual price shock.
This is a simplified
economic mechanism, not an empirical calibration or an exact financial benchmark.
Changing currency denomination does not change the strength of scarcity.

Each node reports a consumer price index based on its household basket and the
last completed day's inflation rate. Rising inflation also tightens the hidden
audit threshold. Hoarding during a day with inflation above 2% can trigger an
audit even before the node runs out of essential goods.

Private block trades integrate an exponential depth curve. Buys walk prices up,
sells walk them down, and brokerage transfers to the local treasury. An immediate
opposite trade retraces the same curve and loses brokerage. Both physical stock
and counterparty cash constrain execution. No unlimited market maker exists.

Currency is conserved across modeled accounts. Wages go to households; consumer
spending goes to markets; public procurement and institutional consumption spend
treasury or market funds; fees and maintenance go to treasuries. Construction
pays local markets for real materials and the treasury for the building contract.
Daily tax/rebate circulation redistributes existing money. The model has no bank
credit, interest, or automatic money creation.

## Production and social dependencies

Fourteen recipes implement farming, logging, mining, ranching, slaughtering,
milling, fuel production, alchemy, refining, forging, machinery, weapons, and
clothing. Use
`recipes` to inspect input quantities, yields, wages, and construction costs.

Facilities run during Day. Output is limited by input stock, affordable wages,
local labor conditions, and private warehouse capacity after incoming reservations.
Factories cannot consume inputs from another node or another owner's warehouse.
Municipal facilities follow the same recipes and labor rules as private ones.
Mining and farming are explicit material sources; all transformations declare
consumed inputs and created outputs in the accounting totals.

Private facilities have two batches/day of nameplate capacity. They can be paused
with `toggle_facility FACILITY_ID`. Extraction construction is restricted to nodes
with the corresponding natural resource or agricultural specialization.

- Wilbur's Drop consumes 1.4 times the standard food requirement and loses 40% of
  production efficiency in winter.
- Winter farms produce at 45% of normal seasonal capacity; autumn raises it to
  135%. These rates are configurable design choices in the simulation code.
- Sustained food shortages cause strikes, stopping local production. Delivering
  provisions can end the strike after the shortage recovers.
- Oakhaven's research falls during food shortages, unrest, and strikes. Sinder's
  smelters and machinery plants depend on that research and on physical catalysts.
- Governments consume machinery, tools, refined metal, and defense equipment;
  those manufactured outputs therefore have actual downstream demand.

The effect of research failure is partial degradation, with a 35% technology floor.
It does not grant free inputs. Shortage, labor, input, and technology limits multiply.

## Logistics and property

Fourteen directed routes represent roads and rivers. Direction matters: for example,
Belos→Wilbur has a 9.5-day movement requirement, while Wilbur→Belos requires seven.
Legal road freight pauses at night, so elapsed calendar duration exceeds its
uninterrupted movement requirement. River upstream routes are slower and costlier
than their downstream counterparts.

Each shipment occupies route capacity and its owner's fleet capacity until delivery
or seizure. Its full weight reserves space in the receiving warehouse. Buys,
production, and further dispatches cannot use that reserved space. Transit goods
remain in wealth accounting at current origin prices until delivery; dispatch
alone does not revalue goods at a distant, expensive market.

Municipal procurement also creates shipments and occupies the same route capacity.
Remote goods never teleport to a production input or a household. Perishable goods
spoil in markets, warehouses, and transit, with explicit material sink entries.

Warehouse construction requires council permission and a free municipal plot.
Additional plots expand capacity; purchase cost rises with local plot occupancy,
and daily maintenance grows as `2 * plots**1.6` for each enterprise/node warehouse.
Construction is immediate after approval in this version. Fleet and facility
purchases are capital assets with explicit book values. Freight is an expense.

Expansion consumes real materials bought from the local market: warehouses use
timber and tools, factories use tools and machinery, and fleets use machinery,
tools, and timber. A cash-rich firm cannot build where those goods are unavailable.
Construction records the consumed goods as an explicit material sink and includes
their cost in the new asset's book value.

## Politics, audits, and intelligence

Local and federal standing are separate values in [-500, 500]. Supplying essential
goods improves local standing. Corruption is tracked independently. Food stress,
unrest, inflation, and hidden border tension shift local audit thresholds.
Border tension can trigger a public federal Byrinium transit ban.

Hoarding essential goods during shortages can trigger an audit even with neutral
standing. An audit gives eight phases to comply or bribe the inspector. An unpaid
levy seizes local warehouse goods up to the assessed value and records an unpaid
remainder as arrears. It does not confiscate the enterprise's global cash.

State-enemy standing at or below -400 causes local property seizure. Stock becomes
municipal stock; inbound deliveries to the seized warehouse become municipal
property; facilities are nationalized. Eligible rivals can buy the vacant warehouse
at a deterministic auction. The transaction transfers cash to the treasury.

Restricted Byrinium cargo needs a permit. During a federal ban, legal dispatch is
blocked even with a permit. Night smuggling bypasses that dispatch prohibition but
risks inspection. Unauthorized restricted cargo has the vision's default 45%
seizure probability; a seizure transfers only the shipment's cargo to the local
state stock. Each shipment is inspected once, at its first eligible movement.

Evening operations include:

- `intelligence NODE`: reveal its audit threshold for twelve phases.
- `agitate NODE RIVAL`: disrupt a node where the rival has a warehouse for twelve
  phases. Other enterprises at that node are affected too.
- `patrol_bribe ROUTE`: a seven-day patrol contract suppresses that route's cargo
  inspection risk while active. It costs an upfront payment and continuing daily
  payments; nonpayment ends coverage.

Visible observations expose rival locations, not their ledgers or hidden political
thresholds. Batch summaries and full save files are administrative diagnostics,
not player observations. Debt here consists of unpaid maintenance and levies;
arrears above 2,000 suspend the firm until it can settle them.

## Programmatic use and reproducibility

```python
from src.economy import Action, Economy, EconomyConfig

world = Economy(EconomyConfig(seed=7))
result = world.act("player", Action("buy", "belos", "tools", 5.0))
assert result.success, result.message
world.advance(4)
view = world.observe("player")
world.check()
world.save("exports/economy/world.json")
resumed = Economy.load("exports/economy/world.json")
```

This is a command-driven simulator, not a new Gymnasium wrapper. It provides
structured observations and parameterized actions without forcing the economy into
either existing RL interface. Rivals use a deterministic merchant policy through
the same command API. They pay costs, reserve capacity, operate facilities, and
service audits; they are not trained policies or optimal opponents.

Snapshot schema 1 stores accounts, ownership, facilities, shipments, audits, the
calendar, event history, and the NumPy random-generator state. A catalog hash rejects
incompatible saves. Restore checks accounting before returning a usable world.
Saving uses an atomic replacement. Identical initial seeds and commands reproduce
the complete state; continuing a restored snapshot reproduces uninterrupted play.

`supply_shock(node, commodity, multiplier)` is an administrative scenario tool.
It records the intervention as a material source or sink; it is not a player action.
Developer scenarios may also close routes or pause institutions explicitly.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest src/tests/test_economy.py src/tests/test_economy_scenarios.py -q
.\.venv\Scripts\python.exe -m pytest src/tests -q
.\.venv\Scripts\python.exe -m flake8 src/economy src/tests/test_economy.py src/tests/test_economy_scenarios.py
```

Tests cover complete sourcing→transport→refining→manufacturing→sale, finite market
liquidity, conservation across every owner and shipment, invalid-action atomicity,
destination reservations, winter, famine, strikes, research dependency, embargoes,
seizures, auctions, intelligence expiry, corruption actions, arrears, and exact
save/resume. Extended seeded runs cover the full four-season model year. These prove
specified behavior and accounting; they do not establish empirical realism or
optimal economic balance.
