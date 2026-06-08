# Aurix Exchange — Simulation Rebalancing & Design Notes

**Status:** Living design document. Actively overrides previous specifications for mathematical logic and model structure to resolve degenerate agent behaviors.

---

## 1. Updated Asset Risk-Ladder & Structural Rescale

To eliminate the $330\times$ price level disparity where Echo Crystals acted as a high-stakes casino asset and Sovereign Herbs remained economically inert, the simulation is restructured. Per the updated long-term vision, **Echo Crystals are completely removed from the simulation** due to their extreme continental scarcity making them unviable for macro-scale corporate distribution. They are replaced in the asset risk-ladder by **Refined Machinery/Tools**, setting up the foundation for the Phase B manufacturing loop[cite: 2, 3].

Scarcity and economic utility are now modeled through **volumetric carrying capacity and structural volatility**, rather than raw unit price differentials.

### Revised Commodity Profile Matrix

| Commodity ($i$) | Target Node Source | Base Price ($\exp(\mu_i)$) | Reversion Speed ($\theta_i$) | Volatility ($\sigma_i$) | Maximum Inventory Capacity ($I_{\text{max}}^i$) |
| --- | --- | --- | --- | --- | --- |
| **Sovereign Herbs** | Oakhaven

 | $30$<br> | $0.50$<br> | $0.15$<br> | $2,500\text{ Units}$ |
| **Raw Byrinium** | Wilbur's Drop

 | $150$ | $0.10$<br> | $0.40$ | $500\text{ Units}$ |
| **Refined Tools** | Sinder

 | $600$ | $0.05$<br> | $0.25$ | $100\text{ Units}$ |

* **Herbs (High-Volume, Low-Margin):** Low price, fast reversion, and massive carrying capacity. This serves as a reliable operational safety valve for capital recycling.


* **Byrinium (The Industrial Workhorse):** Moderate price, slow reversion, fat cyclical swings, and mid-tier carrying capacity.


* **Refined Tools (High-Tier Speculation/Output):** High value, low reversion, and strict capacity limits. This asset reacts sharply to supply-chain blocks.



---

## 2. Revised Mathematical Logic

### 2.1 Endogenous Price Dynamics Engine (Anti-Self-Inflation Refactor)

To eliminate the exploit where the single-agent policy could infinitely inflate its asset values via the permanent impact parameter ($\mu \leftarrow \mu + y \cdot q$), the marketplace is refactored to simulate endogenous supply pooling. Prices still track a **Geometric Ornstein-Uhlenbeck (GOU)** process, but the long-term log-mean parameter ($\mu_{i,t}$) dynamically responds to global inventory deficits.

Let $X_t^i = \ln S_t^i$ represent the log-price of commodity $i$. The step-wise transition density equation is:

$$X_{t+1}^i = \mu_{i,t} + (X_t^i - \mu_{i,t})e^{-\theta_i} + \sigma_i \sqrt{\frac{1 - e^{-\2\theta_i}}{\2\theta_i}} \cdot \epsilon_i \quad \text{where } \epsilon_i \sim N(0,1)$$

The dynamic log-mean vector updates each step based on market inventory pools to simulate consumer demand elasticity:

$$\mu_{i,t} = \mu_{i,\text{baseline}} \cdot \left(1.0 + \omega_i \cdot \left[\frac{\text{Deficit}_t^i - I_{\text{market},t}^i}{I_{\text{market, target}}^i}\right]\right)$$

Where:

* $I_{\text{market},t}^i$ is the running total of commodity units held in the public sorting warehouses of Belos.


* $\omega_i$ is the price elasticity coefficient (set tightly for Byrinium to penalize over-harvesting, and loosely for Herbs).



### 2.2 Temporal Phase-Splitting Math (The 4-Phase Day)

The model structure shifts from abstract sequential steps to a cyclical **4-Phase Day** framework:

$$\text{Phase Index } \tau_t = t \pmod 4 \quad \text{where } \tau \in \{0=\text{Morning}, 1=\text{Day}, 2=\text{Evening}, 3=\text{Night}\}$$

This operational cycle introduces deterministic time-of-day constraints on the action space:

```
[Phase Constraints Engine]
Morning (τ=0) ──> Action Mask Allows: BUY, SELL, WAREHOUSE_PLOT
Day (τ=1)     ──> Action Mask Allows: LAUNCH_EXPEDITION, TRANSFER_GOODS
Evening (τ=2) ──> Action Mask Allows: EXECUTE_ESPIONAGE, RECONNAISSANCE
Night (τ=3)   ──> Action Mask Allows: BLACK_MARKET_SMUGGLE (Overland Frozen)

```

### 2.3 Expedition Sourcing Fees & Localized Failure Mechanics

Expeditions are stripped of their "free wealth" characteristics and configured as high-friction sourcing operations.

* **Upfront Operational Cost:** Executing `LAUNCH_EXPEDITION` decrements liquid cash immediately by a flat fee ($C_{\text{launch}}$), independent of the downstream outcome.


* **Localized Failure Matrix:** On a catastrophic hazard failure check ($P_{\text{fail}}$), the agent **never** forfeits their global cash ledger or separate node inventories. The penalty is tightly isolated:



$$\text{If } \text{Failure}: \quad C_{t+1} = C_t - C_{\text{launch}}, \quad I_{\text{pending\_harvest}} = 0, \quad T_{\text{cooldown}} = 2 \text{ Steps}$$

* **Action Masking Cooldown:** The tracking variable $T_{\text{cooldown}}$ decrements by 1 each step. While $T_{\text{cooldown}} > 0$, the `LAUNCH_EXPEDITION` index is hard-masked out of the action selection graph.



---

## 3. Structural Model & State Space Modifications

### 3.1 Observation Space Contract Preservation (Phase A)

To prevent breaking the downstream C# Godot engine struct contracts and the unmanaged ONNX inference wrappers during Phase A, the observation space **remains exactly 10-dimensional**. The model achieves this by repurposing existing features to incorporate the new temporal and structural variables:

$$o_t = [C_t, I_t^B, I_t^H, I_t^T, X_t^B, X_t^H, X_t^T, F_t, \tau_t, T_{\text{cooldown}}]^T$$

1. **`Idx 2 & 3`:** Echo Crystal features are swapped out for **Refined Tools ($I_t^T$)** inventory and standardized log-prices ($X_t^T$).


2. **`Idx 8 (Horizon Step)`:** Swapped out for the explicit **Phase Index ($\tau_t / 3.0$)** to let the network natively compute phase-conditioned trading and movement sequences.
3. **`Idx 9 (Net Worth Scale)`:** Swapped out for the **Expedition Cooldown ($T_{\text{cooldown}} / 2.0$)**, preserving strict Markovian state evaluation. Net worth tracking is moved entirely to internal reward step calculations.



### 3.2 Action Masking Matrix Architecture

Valid action paths must be asserted via an active boolean layer before computing value-network layers to suppress illegal actions across phases and cooldown locks:

```python
def get_action_mask(state):
    # state mapping: [Cash, InvB, InvH, InvT, PriceB, PriceH, PriceT, Fatigue, Phase, Cooldown]
    mask = np.ones(8, dtype=bool)
    phase = int(state[8] * 3.0)
    cooldown = int(state[9] * 2.0)
    cash = state[0]
    
    # Rule 1: Cooldown Lockout
    if cooldown > 0:
        mask[7] = False  # Block LAUNCH_EXPEDITION
        
    # Rule 2: Phase-Based Structural Bans
    if phase == 3:  # Night Phase Curfew
        mask[1] = mask[3] = mask[5] = False  # Block standard open market purchases
        mask[7] = False                      # Block legal daytime expeditions
        
    # Rule 3: Liquidity Checks
    if cash <= MIN_LAUNCH_COST:
        mask[7] = False  # Cannot afford sourcing overhead
        
    return mask

```

---

## 4. Phase A Evaluation Targets & Acceptance Criteria

To ensure the environment design is completely validated before exporting the network weights to the C# Godot wrapper, the Python test harness must verify the following benchmarks over 500 seeded training episodes:

* **Random Policy Survival Rate:** $\ge 90\%$. A random policy must safely navigate the environment without triggering asset seizure or corporate bankruptcy, establishing a stable gradient space for the network to optimize within.


* **Heuristic Baseline Outperformance:** The trained Double DQN policy must achieve a median terminal net worth at least $1.8\times$ higher than a rule-based mean-reversion script, proving that it has successfully mapped optimal opportunity costs between open-market scaling and expedition risk extraction.



---

### Next Step for Implementation Optimization

To initialize the Week 1 environment configuration files based on this structural shift, should the **upfront expedition launch fee ($C_{\text{launch}}$)** be mapped as a fixed flat rate, or should it scale dynamically based on your current **Party Fatigue ($F_t$)** metric to simulate the increasing logistical overhead of managing an exhausted mercenary crew?
