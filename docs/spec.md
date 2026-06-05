# Aurix_Exchange_Specification

## 1. Project Overview & Meta-Parameters

* **Project Name:** Aurix Exchange Simulation
* **Architecture:** Double DQN + Prioritized Experience Replay (PER) + Custom Action Masking
* **Training Framework:** Python 3.10+, Gymnasium, PyTorch
* **Production Environment:** Godot 4.NET Core (C#), Desktop Execution
* **Core Goal:** Solve a stationary, single-agent vertical integration/market arbitrage Markov Decision Process (MDP).

---

## 2. Microeconomic Framework & Asset Risk-Ladder

### 2.1 Commodity Profiles

1. **Raw Byrinium ($B$):** High-liquidity industrial asset sourced from Wilbur's Drop. Moderate volatility, stable drift. Negligible permanent market impact.
2. **Sovereign Botanicals/Herbs ($H$):** Low-volatility agricultural safety asset sourced from Oakhaven. High reversion speed. Low risk, highly predictable cycles.
3. **Echo Crystals ($C$):** Subterranean high-hazard asset. Extreme volatility, low reversion speed. **Harvesting blocked for exploration party** due to cognitive hazards; accessible *only* via market speculation. Heavy permanent market impact parameter.

### 2.2 Price Dynamics Engine

Spot prices follow a **Geometric Ornstein-Uhlenbeck (GOU) process** to preserve environment stationarity. Let $X_t^i = \ln S_t^i$ be the log-price of commodity $i$.

The continuous stochastic differential equation is:

$$dX_t^i = \theta_i(\mu_i - X_t^i)dt + \sigma_i dW_t^i$$

The exact, drift-free closed-form temporal update $(\Delta t = 1)$ used in the environment step is:

$$X_{t+1}^i = \mu_i + (X_t^i - \mu_i)e^{-\theta_i} + \sigma_i \sqrt{\frac{1 - e^{-2\theta_i}}{2\theta_i}} \cdot \epsilon_i \quad \text{where } \epsilon_i \sim N(0,1)$$

Execution spot market price: $S_t^i = \exp(X_t^i)$.

### 2.3 Market Impact & Slippage

* **Temporary Impact (Friction):**
$$P_{\text{exec}}^i = S_t^i \cdot (1 + \eta_t q_t^i) \cdot (1 \pm \varphi)$$

*(where $q_t^i > 0$ represents a buy, and $\varphi$ is the baseline percentage brokerage fee)*.

* **Permanent Impact (Equilibrium Shift):**

$$\mu_i \leftarrow \mu_i + y_i q_t^i$$

### 2.4 Fatigue & Catastrophic Expedition Hazards

Party fatigue maps to a continuous domain: $F_t \in [0, 100]$. Active expeditions increment $F_t$, while resting/sedentary steps induce decay. Complete asset forfeiture via catastrophic team failure is tracked via a logistic hazard rate:

$$P_{\text{fail}}(F_t, i) = P_{\text{base},i} + \frac{1 - P_{\text{base},i}}{1 + \exp(-\kappa_i(F_t - y_i))}$$

---

## 3. Markov Decision Process (MDP) Structuring

### 3.1 Observation Space (10-Dimensional Float32 Array)

$$o_t = [C_t, I_t^B, I_t^H, I_t^C, X_t^B, X_t^H, X_t^C, F_t, t, W_t]^T$$

| Index | Feature Name | Raw Definition | Fixed Normalization / Transformation Pipeline |
| --- | --- | --- | --- |
| 0 | Normalized Cash | Cash Balance ($C_t$) | $\ln(C_t + 1) / \ln(\text{Max\_Expected\_Cash})$ |
| 1 | Inv Byrinium | Inventory Units ($I_t^B$) | $I_t^B / \text{Max\_Capacity}_B$ |
| 2 | Inv Herbs | Inventory Units ($I_t^H$) | $I_t^H / \text{Max\_Capacity}_H$ |
| 3 | Inv Crystals | Inventory Units ($I_t^C$) | $I_t^C / \text{Max\_Capacity}_C$ |
| 4 | Price Byrinium | Log-Spot ($\ln S_t^B$) | $(\ln S_t^B - \mu_B) / \sigma_{\text{stat},B}$ where $\sigma_{\text{stat}} = \sigma / \sqrt{2\theta}$ |
| 5 | Price Herbs | Log-Spot ($\ln S_t^H$) | $(\ln S_t^H - \mu_H) / \sigma_{\text{stat},H}$ |
| 6 | Price Crystals | Log-Spot ($\ln S_t^C$) | $(\ln S_t^C - \mu_C) / \sigma_{\text{stat},C}$ |
| 7 | Party Fatigue | Fatigue Tracker ($F_t$) | $F_t / 100.0$ |
| 8 | Horizon Step | Step Counter ($t$) | $t / T_{\text{max}}$ |
| 9 | Net Worth Scale | Portfolio Net Worth ($W_t$) | $\ln(W_t + 1) / \ln(\text{Max\_Expected\_Cash})$ |

### 3.2 Action Space Configuration (Fixed-Fractional Engine)

A 1D discrete action mapping space containing exactly 8 choices:

1. **`HOLD` (Idx 0):** No trade/exploration. Active sedentary fatigue decay.
2. **`BUY_BYRINIUM` (Idx 1):** Allocate exactly 25% of liquid cash to purchase Raw Byrinium.
3. **`SELL_BYRINIUM` (Idx 2):** Liquidate exactly 100% of current Byrinium inventory.
4. **`BUY_HERBS` (Idx 3):** Allocate exactly 25% of liquid cash to purchase Sovereign Herbs.
5. **`SELL_HERBS` (Idx 4):** Liquidate exactly 100% of current Herb inventory.
6. **`BUY_CRYSTALS` (Idx 5):** Allocate exactly 25% of liquid cash to purchase Echo Crystals.
7. **`SELL_CRYSTALS` (Idx 6):** Liquidate exactly 100% of current Crystal inventory.
8. **`LAUNCH_EXPEDITION` (Idx 7):** Target extraction assets based on environment weights. Fully blocked if party status or cash assets flag an invalid conditional branch.

### 3.3 The Core Objective Reward Function

$$R_t = \text{clip}\left(\alpha \cdot \ln\left(\frac{W_t}{W_{t-1}}\right), -c, +c\right) - \Psi_{\text{insolvency}}(W_t) - \Psi_{\text{fatigue}}(F_t) - \delta \cdot \mathbb{1}_{\{a_t = \text{HOLD}\}}$$

* **Insolvency Defense Layer (net-worth-keyed):** $\Psi_{\text{insolvency}}(W_t) = \beta \cdot \left(\frac{W_{\text{crit}} - W_t}{W_{\text{crit}}}\right)^2 \cdot \mathbb{1}_{\{0 < W_t < W_{\text{crit}}\}} + \Omega \cdot \mathbb{1}_{\{W_t \le 0\}}$
* **Fatigue Barrier Layer:** $\Psi_{\text{fatigue}}(F_t) = \eta \cdot \exp(\xi(F_t - F_{\text{crit}}))$

> **Revision 2026-06-04:** the potential-based shaping term $\gamma\Phi(s_{t+1}) - \Phi(s_t)$ was **removed**, and the insolvency penalty was **re-keyed from cash $C_t$ to net worth $W_t$**. Decomposing the first trained checkpoint showed both terms were misaligned with the objective: with $\gamma < 1$, the shaping imposed a per-step drag $\approx (1-\gamma)\,w\ln W_t$ that *grew* with wealth, and the cash-based insolvency penalty fired continuously for an asset-rich/cash-poor agent. Together they drove mean episode reward to $-6.2$ despite ~19× net-worth growth. The clipped log-return already supplies a dense growth signal, so shaping is unnecessary; $\Psi_{\text{insolvency}}$ now only activates near true ruin ($W_t < W_{\text{crit}}$, default $W_{\text{crit}} = 1000$).

---

## 4. Production C# Godot 4 Native Interface

```csharp
using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

public enum GameAction : int
{
    HOLD = 0,
    BUY_BYRINIUM = 1,
    SELL_BYRINIUM = 2,
    BUY_HERBS = 3,
    SELL_HERBS = 4,
    BUY_CRYSTALS = 5,
    SELL_CRYSTALS = 6,
    LAUNCH_EXPEDITION = 7
}

public struct MarketEnvironmentState
{
    public float Cash;
    public float InvByrinium;
    public float InvHerbs;
    public float InvCrystals;
    public float PriceByrinium;
    public float PriceHerbs;
    public float PriceCrystals;
    public float Fatigue;
    public float StepIndex;
    public float TotalSteps;

    public float[] GetNormalizedVector()
    {
        float networth = Cash + (InvByrinium * PriceByrinium) + (InvHerbs * PriceHerbs) + (InvCrystals * PriceCrystals);
        return new float[]
        {
            MathF.Log(Cash + 1.0f) / 11.5f,
            InvByrinium / 100.0f,
            InvHerbs / 100.0f,
            InvCrystals / 100.0f,
            (MathF.Log(PriceByrinium) - 6.9f) / 0.5f, 
            (MathF.Log(PriceHerbs) - 3.4f) / 0.15f,
            (MathF.Log(PriceCrystals) - 9.2f) / 0.9f,
            Fatigue / 100.0f,
            StepIndex / TotalSteps,
            MathF.Log(networth + 1.0f) / 11.5f
        };
    }
}

public class PalanDecisionEngine : IDisposable
{
    private InferenceSession _session;
    private string _inputName;

    public PalanDecisionEngine(string onnxModelPath)
    {
        // SAFE DESKTOP PCK PACKAGING PATH: Read assembly payload through memory buffers
        byte[] modelBytes = Godot.FileAccess.GetFileAsBytes(onnxModelPath);
        _session = new InferenceSession(modelBytes);
        _inputName = _session.InputNames[0];
    }

    public GameAction GetBestAction(MarketEnvironmentState state, bool[] actionMask)
    {
        float[] inputFeatures = state.GetNormalizedVector();
        long[] shape = { 1, 10 };

        // Enforce strict unmanaged scope boundaries using structural disposables
        using var inputTensor = OrtValue.CreateTensorWithDataAsSpan(inputFeatures, shape);
        var inputs = new Dictionary<string, OrtValue> { { _inputName, inputTensor } };
        using var runOptions = new RunOptions();
        using var outputCollection = _session.Run(runOptions, inputs, _session.OutputNames);
        
        ReadOnlySpan<float> qValues = outputCollection[0].GetTensorDataAsSpan<float>();

        int bestActionIdx = 0;
        float maxQ = float.MinValue;

        for (int i = 0; i < qValues.Length; i++)
        {
            if (actionMask[i] && qValues[i] > maxQ)
            {
                maxQ = qValues[i];
                bestActionIdx = i;
            }
        }

        return (GameAction)bestActionIdx;
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}

```

> **Note:** the numeric literals in `GetNormalizedVector()` above (`11.5f`, `6.9f`,
> `0.5f`, `100.0f`, …) are *illustrative*. In production they must be loaded from the
> shared-constants sidecar defined in **Addendum A**, not hardcoded — otherwise the C#
> normaliser will silently drift from the Python training distribution whenever an env
> parameter changes.

---

## 5. Addendum A — Shared-Constants Sidecar (Normative)

The observation normaliser and action-mask thresholds are duplicated across two
implementations: the Python training environment (`src/env/aurix_env.py`) and the C#
deployment client (`PalanDecisionEngine` / `MarketEnvironmentState`). Any divergence in
these constants means the deployed model receives a different observation distribution
than it trained on — a silent, hard-to-diagnose correctness failure.

To eliminate that risk, **all MDP constants are owned by the Python `EnvConfig` dataclass
and serialised to a JSON sidecar** at training/export time via `export_config()`. The
sidecar is written next to the model artifacts (default `exports/aurix_config.json`) and
is the single source of truth.

**Schema:**

```json
{
  "obs_dim": 10,
  "n_actions": 8,
  "config": { "...": "every EnvConfig field, including buy_fraction, fatigue_max, expedition_fatigue_ceiling, buy_cash_epsilon, sell_inventory_epsilon" },
  "derived": {
    "sigma_stat":    [0.5, 0.15, 0.9],
    "gou_decay":     ["e^-theta per commodity"],
    "gou_noise_std": ["closed-form per-step noise std per commodity"]
  }
}
```

**Consumer contract:**

* The C# client loads the sidecar once at startup and reads `cash_norm`,
  `max_inventory`, `gou_mu`, and `derived.sigma_stat` for the normaliser, and
  `expedition_fatigue_ceiling`, `expedition_min_cash`, `buy_cash_epsilon`,
  `sell_inventory_epsilon` for the action-mask reconstruction.
* `derived.sigma_stat` is provided precomputed because the C# normaliser divides log-price
  deviations by it directly; the consumer must **not** recompute it from `sigma`/`theta`.
* The sidecar and the `.onnx` are a versioned pair — shipping one without regenerating the
  other is forbidden.

---

## 6. Addendum B — Multiplayer / Competitive Extension (Forward-Looking, Non-Normative)

This addendum sketches a future direction in which trained agents are embedded in the
Godot client to *play* the market — either for deployment-environment evaluation or for
competitive multi-agent scenarios. **It is non-normative:** the core training MDP (§1–§3)
remains a stationary, single-agent process. Nothing here changes the current pipeline.

### 6.1 Prerequisite — the game must become the environment

The current C# surface implements only the *consumer* of the policy (normaliser +
masked-argmax inference). To run an episode in-engine, the **market dynamics themselves**
must be ported to C#: the GOU step (§2.2), trade execution with temporary/permanent impact
(§2.3), and the fatigue/expedition hazard (§2.4). Until that port exists, agents cannot
"play" — they can only score externally-supplied observations.

Three parity surfaces must match the Python env exactly:

| Surface | Python source of truth | C# status |
| --- | --- | --- |
| Observation normalisation | `_normalize_obs()` | exists (`GetNormalizedVector`) |
| Action masking | `_action_mask()` | **missing** — must be reconstructed from sidecar thresholds |
| Market dynamics | `_gou_step` / `_execute_*` | **missing** — must be ported |

### 6.2 Validation strategy

Cross-language bit-exactness is not achievable (NumPy PCG64 vs. C# RNG, differing float
ops), so parity is validated in two decoupled halves:

1. **Inference parity (export correctness):** feed recorded Python observations through both
   the PyTorch net and the ONNX session; assert Q-values agree to ~`1e-4`. Independent of
   any RNG.
2. **Dynamics parity (port correctness):** drive both simulators with an identical action
   sequence and, where shareable, identical noise; assert trajectories track. Otherwise
   fall back to statistical comparison of return distributions across many seeds.

Both are best expressed as golden-trajectory fixtures generated from Python and asserted
against in C#.

### 6.3 The stationarity break (competitive case)

The single-agent MDP assumes a stationary market in which permanent impact
(`mu ← mu + y·q`) is driven by one agent. Placing *N* agents in one shared market violates
this: impact sums across agents and the environment each agent observes becomes
non-stationary. Two tiers follow:

* **Drop-in (available once §6.1 is done):** instantiate multiple `PalanDecisionEngine`s
  against one shared market. Agents will trade and the result is a compelling
  visualisation, but each was trained believing it acts alone — expect degraded or unstable
  behaviour when rivals move prices. Suitable for demos and qualitative evaluation only.
* **Genuinely competitive (research-level):** retrain under a multi-agent regime
  (independent learners, self-play, or PSRO-style population training) with the market
  modelled as multi-agent from the outset. This is a different training pipeline from the
  current Double-DQN-vs-stationary-GOU setup and a substantially larger undertaking.
