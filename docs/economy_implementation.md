# Detailed economy implementation

The active objective is a fully functional, detailed economic simulation grounded
in `vision.md`. This is a third track, separate from the exact OU benchmark and
the stationary M2 prototype. Existing checkpoints and regret claims do not apply.

## Completion requirements

- Six populated, specialized nodes with finite inventories and monetary accounts.
- Scarcity-driven local prices, executable finite-liquidity trades, fees, and
  distinct enterprise ledgers participating in the same world.
- Extraction, agriculture, livestock, refining, manufacturing, research-dependent
  production, labor needs, food shortages, seasons, and strikes.
- Directed roads/rivers, asymmetric travel times, capacity constraints, fleets,
  in-transit ownership, curfews, delivery reservations, and localized cargo seizures.
- Warehouse plots, council permission, construction, nonlinear maintenance, and
  facilities that actually consume inputs and produce goods.
- Local/federal standing, stress-dependent hidden audit thresholds, trade bans,
  hoarding audits, compliance/bribery/smuggling, and state-enemy seizure orders.
- Evening intelligence, labor agitation, and continuing patrol bribes with expiry.
- Independent competitor enterprises using the same actions, accounts, and rules.
- Four phases per day with action points and explicit time advancement.
- Inspectable observations, events, snapshots, deterministic save/resume, and a
  usable command-line simulation and interactive control interface.
- Automated accounting invariants, deterministic replay, economic causal tests,
  and extended seeded simulations; existing suites continue to pass.

The simulator is the deliverable for this objective. Training new learned policies,
ONNX/C# deployment, and a Godot presentation layer are downstream integrations;
their absence must remain explicit in project documentation.

## Implementation decisions

Goods and cash cannot appear through transfers. Extraction and recipes declare
their external sources and transformation sinks in the accounting journal;
household consumption and spoilage are explicit sinks. Fees, wages, taxes, and
construction costs transfer currency to another modeled account. Accounting is
checked across markets, warehouses, and shipments, including reserved deliveries.

Prices respond to physical local inventories and unmet demand. The proposed
vision formula multiplying a log-price by a shortage factor is replaced with an
additive log scarcity adjustment: changing the currency unit must not change the
strength of scarcity. Exact OU remains unchanged in its separate environment.

Public institutions and private enterprises share production and delivery rules.
Private actions are validated before mutation and consume action points only on
success. Hidden political thresholds remain absent from ordinary player views;
intelligence reveals them temporarily. Full save files are administrative state,
not agent observations.

## Status

Implemented and verified on 2026-09-13.
The requirement audit below covers the full simulator scope rather than only a
market or logistics slice. The existing two RL environments retain their own
contracts and experiments.

## Requirement-to-evidence audit

| Requirement | Authoritative implementation | Verification |
|---|---|---|
| Six specialized nodes, 518,000 residents, independent finite accounts | `catalog.py`, `model.py`, `world.py` initialization | `test_six_specialized_nodes_and_independent_ledgers`; full-year state inspection |
| Endogenous local prices, inflation, finite trades, fees, no free round trips | `markets.py`; `world.py::act` | Shortage/surplus price tests, local-inflation scenario, cash/stock constraints, thirteen commodity round trips, rejected-command atomicity |
| Extraction, agriculture, livestock, slaughtering, refining and manufacturing | Fourteen `RECIPES` and public/private facilities in `production.py` | All fourteen recipes observed producing in the saved year; private production/wage test; full player sourcing→transport→refining→manufacturing→sale scenario |
| Seasons, food, labor, strikes and research dependency | `production.py` consumption and labor factors | Winter reduces Wilbur capacity by 40%; food shortage stops mining and relief restores it; an Oakhaven food crisis reduces downstream Byrinium and machinery output |
| Directed travel, road curfew, asymmetric rivers and real transit ownership | Fourteen directed routes; `logistics.py` | Delivery timing, paused night road movement, upstream/downstream differences, transit wealth and localized seizure tests |
| Warehouse, fleet and route capacities, including arriving shipments | `accounting.py`, `logistics.py`, production space checks | Separate fleet/route rejection scenarios, incoming reservation prevents purchases and limits production; every step checks physical bounds |
| Council permission, limited plots, nonlinear upkeep, materials and capital assets | `politics.py` construction; `production.py` finances | Permission/plot tests, nonlinear maintenance, fleet asset valuation, real-material consumption and shortage rejection |
| Local/federal standing, inflation/stress audits, embargoes and state-enemy orders | `politics.py`, `logistics.py` | Hoarding audit, compliance, bribery, unpaid levy, embargo despite permit, property auction and inbound-cargo nationalization scenarios |
| Hidden thresholds, evening intelligence, agitation and continuing bribes | `world.py::observe`, `politics.py`, `logistics.py` | Temporary threshold reveal, rival-ledger exclusion, agitation stopping work, patrol suppression/expiry and corruption/payment accounting |
| Rival enterprises using the same economic rules | `agents.py` calls `world.act`; production has no private exception | Three seeded complete years, private production records, paid trades, shared-capacity dispatches and audit responses |
| Four daily phases, action points and usable interactive commands | `world.py::advance`, `__main__.py` | AP exhaustion/reset, phase restrictions, console grammar, complete interactive delivery/sale/save test |
| Auditable sources, sinks and currency transfers | `accounting.py` plus production, consumption, spoilage and construction journals | Invariants after successful commands and every phase; full-year and ten-year runs preserve accounting |
| Exact persistence, schema boundaries, administrative versus player views | `world.py` snapshots and observation methods | Exact uninterrupted/restored state equality, save isolation, corrupt/incompatible save rejection, public CLI resume |
| Existing research work remains valid | Existing OU, DP/MPC, DQN/PER and policy-gradient modules remain separate | Full repository test suite, including slow oracle tests |

## Reproduction evidence

The generated evidence lives under `exports/economy/` and is ignored by Git.
The durable operator contract is `docs/economy_guide.md`.

- Final full repository validation: **323 passed**, including **74 detailed
  economy tests**, in 74.59 seconds. The slow OU/DP/MPC checks are included.
- Final `flake8` validation passed for `src/economy` and both economy test modules.
- The interactive console test performs warehouse purchase, a buy, a real
  shipment, phase advancement, a sale, state inspection, and a resumable save.

- `development.json`: seed 7, 120 days, three rivals, full state and random
  generator state. All fourteen production recipes executed. The run recorded
  2,219 production events, including 281 private production events, 2,386 trades,
  4,844 deliveries, 17 material-backed construction events, and 87 audits.
- `development_report.json` and `development_events.jsonl`: inspectable summary
  and full journal for that state. Currency residual was approximately `1.49e-8`.
- `resumed.json` / `resumed_report.json`: the public CLI loaded the saved year
  and advanced it to day 121 successfully.
- `ten_year_report.json`: seed 41, 1,200 days (ten four-season model years), with
  automatic accounting checks enabled. Currency residual was approximately
  `3.58e-7`; all six towns were supplied at the endpoint. This endpoint does not
  assert that no shortages or strikes occurred earlier.

These runs verify functionality and numerical/accounting stability. They do not
claim real-world macroeconomic calibration, game balance, or learned-policy
optimality. Existing trained policies cannot be loaded into this command-driven
simulation. Godot, ONNX and training integrations remain downstream projects.
