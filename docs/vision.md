# **Aurix Exchange: Macroeconomic Immersive Sim Vision Document**

This document serves as the authoritative product vision and design anchor for the *Aurix Exchange* ecosystem. It bridges the gap between the immediate 1-month Deep Reinforcement Learning (DRL) MVP framework and the long-term design goals of a complex, systemic macroeconomic simulation. It provides a blueprint for an economy driven by endogenous scarcity, logistical friction, and political instability.

---

## **1. The Vision & Core Philosophy**

* **The Paradigm Shift:** Moving away from a standard marketplace arbitrage loop to a systemic supply-chain, production, and throughput simulation. The player operates a business enterprise embedded within a living, state-sanctioned command economy.

* **The Immersive Sim Objective:** Designing an economic environment where macro-level variables (like asset market prices) are the direct consequence of localized human behavior, physical infrastructure bottlenecks, and geographic constraints.

* **The Core Conflict:** Navigating the tight symbiotic bottleneck between academic innovation (Oakhaven's research patents), industrial execution (Sinder's manufacturing centers), resource extraction (Wilbur's Drop's mines), and state surveillance (Federal Council trade bans).

---

## **2. Demographic Registry & Node Architecture**

The Republic of Aurix spans approximately 518,000 citizens distributed across six highly specialized municipal and resource nodes. The transcontinental **Great Aralis River** serves as the vertical spine of global commerce, while the **Byrin River** links the industrial heartland directly to the central markets.

```txt
                  [Wilbur's Drop] (Alpine Ore Extraction)
                         │
                         │ Overland Highway (7.0 Days)
                         ▼
   [Oakhaven] ───> [River's Crossing] ───> [SINDER] ◄───(Byrin River)───► [BELOS Hub]
 (Grain/Research)    (Border Fortress)    (Forges/Maritime)                   │
                                                                              ▼
                                                                      [Byrin Capital]

```

### **The Node Matrix**

| Node Name | Primary Role | Est. Population | Core Surplus Production | Core Deficit Consumption |
| --- | --- | --- | --- | --- |
| **Belos** | Commercial Master Chokepoint | 125,000 | Currency Liquidity, Sorting Depots, Central Slaughterhouses | Raw Grain, Industrial Metal, Fuel Inputs |
| **Byrin** | Administrative Capital | 85,000 | Governance, Legislative Decrees, Floodplain Agriculture | Refined Byrinium, Manufactured Machinery, Tools |
| **Sinder** | Maritime Confluence & Manufacturing | 68,000 | Heavy Refined Alloys, Machinery, Logistical Barges/Wagons | Raw Ore, Timber, Rations, Fuel Sinks |
| **Oakhaven** | Academic Technocracy & Agriculture | 195,000 | Grain, Livestock, Alchemical Catalysts, Tech Patents | Refined Tools, Iron Ore, Basic Consumer Ware |
| **River's Crossing** | Frontier Fortress Shield | 27,000 | Border Customs Fees, Imperial Deterrence | Weapons, Forged Armor, High-Tier Provisions |
| **Wilbur's Drop** | Alpine Mining Outpost | 18,000 | Raw Byrinium Ore, Raw Iron Ore | External Foodstuffs, Forged Equipment, Winter Clothing |

---

## **3. The Phase-Based Operational Loop**

Gameplay transitions occur across a strict **4-Phase Day** (Morning, Day, Evening, Night). Players navigate operations using a pool of Action Points (AP).

### **Phase Breakdown**

* **Morning:** Tactical evaluation phase. Market ledger reviews, asset purchases, warehouse lot acquisition requests, and dispatching cargo transit pipelines.
* **Day:** Active logistical execution. Private fleets move across routes, refineries consume fuel inputs, and labor shifts progress.
* **Evening:** Financial reconciliation. Processing bulk transactions at municipal offices, monitoring local inflation metrics, and running espionage network sweeps.
* **Night:** The Curfew Phase. Most legal overland transit freezes. Black-market syndicates launch contraband operations, and the state runs stealth cargo inspections to prevent unlawful Byrinium exports.

### **Logistical Transit Mechanics & Real-Time Lags**

Instantaneous trades do not exist outside local municipal depots. Moving goods across the Republic forces items into a **Transit Pipeline** that locks up capital across explicit temporal steps:

* **Overland Highways:** Moving heavy wagon freight uphill from Belos to Wilbur's Drop is an intensive process, incurring a punishing **9.5-day base delay**.

* **The River Network:** Upstream towpath towing introduces a heavy time multiplier compared to rapid downstream river-barge drift.

* **The Restricted Channels:** Waterways connecting Oakhaven, River's Crossing, and Sinder are tightly monitored by state patrols, imposing a flat **45% cargo seizure hazard** for unauthorized or mislabeled Byrinium transit.

---

## **4. Advanced Systemic Mechanics**

### **The Dual-Sector Symbiosis**

Industrial execution in Sinder is physically bound to academic progress in Oakhaven. Refineries cannot process raw Byrinium ore at peak optimization through brute force alone. They require alchemical blueprints and stabilization metrics patented by the University. If political unrest or food shortages stall Oakhaven's academic guilds, Sinder's output degrades, triggering a downstream collapse in tool integration cycles across the entire country.

### **Labor Leverage: The Wilbur's Drop Strike Loop**

Because Wilbur's Drop faces an aggressive **40% production penalty during harsh alpine winters** alongside a high **1.4 food consumption multiplier**, the outpost is highly volatile. If players or competitor syndicates hoard food in Belos to force price spikes, the mining workforce will unionize and trigger a **Labor Strike**, cutting off all raw ore extraction to the lowlands until provisions are guaranteed.

### **The Warehouse & Hoarding Tax Framework**

Warehouses provide structural insulation against seasonal supply dips, but are bound by strict municipal restrictions:

* **Plot Constraints:** Construction requires local council approval and scales non-linearly in maintenance costs based on the density of the node.

* **Hoarding Tax & Audits:** Councils monitor warehouse volumes. If a player hoards critical survival resources (like grain or fuel) during an economic deficit, local inflation triggers a **State Audit**. Players must choose to comply and pay heavy penalties, bribe the local inspectors (spiking long-term corruption metrics), or execute a high-risk night smuggling breakout.

### **Split-Council Affinity & Hidden Threshold Drift**

Political standing is tracked through a visible metric (-500 to +500) split across independent Local Councils and the central Federal Council. However, the exact boundaries governing regulatory crackdowns are semi-opaque, drifting dynamically based on macro-economic stress signals:

```txt
[Macro-Economic Stress Variables]
Local Famine (Food Deficit) ──> Lowers Hidden Audit Thresholds ──> High Seizure Risk
Imperial Border Tension     ──> Triggers Global War Footing    ──> Mandatory Cargo Raids

```

If an enterprise drops into the *State Enemy* tier, the council issues a **Seizure Order**, nationalizing warehouse stocks or auctioning physical property slots directly to simulated competitors.

---

## **5. Espionage Layer: Unveiling the Opaque**

To navigate a system with semi-opaque political triggers, players invest capital and AP into clandestine operations:

* **Council Ledger Infiltration:** Dispatched during the Evening phase to temporarily map the exact numeric values of hidden *Threshold Modifiers*, identifying upcoming tax audits before the month transitions.
* **Labor Agitation:** Instigating strikes at competitor-controlled nodes, bottlenecking a rival's throughput while driving up the spot market value of hoarded inventory.

* **Patrol Bribes:** Paying ongoing capital maintenance to naval captains stationed at River's Crossing, dropping the baseline **45% restricted waterway seizure chance** down to zero for private black-market shipments.

---

## **6. MVP Rebalancing & Mathematical Alignment Guidelines**

To prepare your Python-based 1-month DRL framework to cleanly port into this complex Godot .NET immersive sim, implement these core mathematical adjustments within the environment script:

### **Transitioning to Endogenous Price Variance**

Modify the stationary **Geometric Ornstein-Uhlenbeck (GOU)** pricing equation. Let $X_t^i = \ln S_t^i$ track the log-price of commodity $i$. Replace the static, top-down long-term mean parameter ($\mu_i$) with a dynamic mean that shifts based on inventory pools:

$$\mu_i \leftarrow \mu_{i,\text{baseline}} \cdot \left(1.0 + \omega \cdot \left[\frac{\text{Deficit}_t^i - \text{Inventory}_t^i}{\text{Capacity}_t^i}\right]\right)$$

The temporal update density handles the step calculation:

$$X_{t+1}^i = \mu_i + (X_t^i - \mu_i)e^{-\theta_i} + \sigma_i \sqrt{\frac{1 - e^{-2\theta_i}}{2\theta_i}} \cdot \epsilon_i \quad \text{where } \epsilon_i \sim N(0,1)$$

* *Why this matters:* This forces the machine learning model to learn **price elasticity**. If the agent dumps huge hoards of Byrinium in a single step, the market parameters shift permanently, preventing basic reward hacking and preparing the network to operate alongside competitor AI agents in a shared environment.

### **Structural Array Decoupling**

Ensure that your 10-dimensional observation array ($o_t$) wraps all player transaction states separately from the global engine market indices. This ensures that when you clone the trained `.onnx` policy file across multiple distinct rival factions inside Godot later, each agent can run local inferences using its own personal ledger inputs against the shared worldwide node parameters.
