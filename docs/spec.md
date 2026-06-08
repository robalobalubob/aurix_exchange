# Aurix_Exchange_Specification.md

## 1. Project Overview & Meta-Parameters

* **Project Name:** Aurix Exchange Simulation
**Architecture:** Double DQN + Prioritized Experience Replay (PER) + Custom Action Masking

**Training Framework:** Python 3.10+, Gymnasium, PyTorch

**Production Environment:** Godot 4.NET Core (C#), Desktop Execution

**Core Goal:** Solve a stationary, single-agent endogenous market arbitrage and supply-chain Markov Decision Process (MDP).

---

## 2. Microeconomic Framework & Production Lifecycle

### 2.1 Commodity Profiles & Volumetric Scaling

The asset risk-ladder is structured around volumetric carrying capacity and structural volatility. Echo Crystals are completely removed from the environment due to extreme continental scarcity. They are replaced by **Refined Tools**, representing the output of the Sinder manufacturing confluence loop.

1. **Sovereign Herbs ($H$):** Sourced from the Oakhaven basin. Low-price, high-reversion, low-volatility safety asset. Represents a high-volume, low-margin agricultural commodity.

2. **Raw Byrinium ($B$):** Sourced from the Wilbur's Drop alpine mines. Moderate price, slow reversion speed, fat cyclical swings. Represents the primary industrial workhorse asset of the Republic.

3. **Refined Tools ($T$):** Sourced from the Sinder manufacturing and forge confluence node. High-value, low-reversion, mid-volatility manufactured asset. Represents the end-stage tool integration pipeline that drives structural economic feedback.

```txt
 [Wilbur's Drop: Ore] ──> [Belos: 5% Tax] ──> [Sinder: Forging] ──> [Refined Tools]
                                │                  ▲
                                ▼                  │ Blueprints / Patents
                         [Oakhaven: Academic Research]

```

### 2.2 Endogenous Price Dynamics Engine

Spot prices follow a **Geometric Ornstein-Uhlenbeck (GOU)** process. To ensure the simulator exhibits market demand elasticity and eliminates self-inflation value exploits, the long-term log-mean parameter ($\mu_{i,t}$) dynamically responds to global inventory deficits.

Let $X_t^i = \ln S_t^i$ be the log-price of commodity $i$. The exact closed-form temporal update $(\Delta t = 1)$ used in the environment step is:

$$X_{t+1}^i = \mu_{i,t} + (X_t^i - \mu_{i,t})e^{-\theta_i} + \sigma_i \sqrt{\frac{1 - e^{-2\theta_i}}{2\theta_i}} \cdot \epsilon_i \quad \text{where } \epsilon_i \sim N(0,1)$$

Execution spot market price: $S_t^i = \exp(X_t^i)$.

The dynamic long-term log-mean updates each step based on public sorting inventory metrics:

$$\mu_{i,t} = \mu_{i,\text{baseline}} \cdot \left(1.0 + \omega_i \cdot \left[\frac{\text{Deficit}_t^i - I_{\text{market},t}^i}{I_{\text{market, target}}^i}\right]\right)$$

Where:

$I_{\text{market},t}^i$ is the running volume of commodity units held in the public sorting warehouses of Belos.

* $\omega_i$ is the price elasticity coefficient of asset $i$.

### 2.3 Market Friction & Slidings

* **Temporary Impact (Execution Friction):**

$$P_{\text{exec}}^i = S_t^i \cdot (1 + \eta_t q_t^i) \cdot (1 \pm \varphi)$$

(where $q_t^i > 0$ represents a buy , and $\varphi$ is the baseline 5% global market transaction tax levied at Belos ).

### 2.4 Fatigue & Localized Expedition Hazards

Party fatigue maps to a continuous domain: $F_t \in [0, 100]$. Active expeditions increment $F_t$, while resting/sedentary steps induce decay. The hazard check tracks complete expedition failure:

$$P_{\text{fail}}(F_t, i) = P_{\text{base},i} + \frac{1 - P_{\text{base},i}}{1 + \exp(-\kappa_i(F_t - y_i))}$$

On a catastrophic hazard check failure, the agent **never** forfeits unrelated cash ledger assets or distinct warehouse stocks. The penalty is localized:

$$\text{If Failure}: \quad C_{t+1} = C_t - C_{\text{launch}}, \quad I_{\text{pending\_harvest}} = 0, \quad T_{\text{cooldown}} = 2 \text{ Steps}$$

---

## 3. Markov Decision Process (MDP) Structuring

### 3.1 Observation Space (10-Dimensional Float32 Array)

To maintain zero-copy compatibility with standard C# structures, the state vector is modeled as a fixed-size 10-dimensional float32 array:

$$o_t = [C_t, I_t^B, I_t^H, I_t^T, X_t^B, X_t^H, X_t^T, F_t, \tau_t, T_{\text{cooldown}}]^T$$

| Index | Feature Name | Raw Definition | Fixed Normalization / Transformation Pipeline |
| --- | --- | --- | --- |
| 0 | Normalized Cash | Cash Balance ($C_t$) | $\ln(C_t + 1) / \ln(\text{Max\_Expected\_Cash})$ |
| 1 | Inv Byrinium | Inventory Units ($I_t^B$) | $I_t^B / \text{Max\_Capacity}_B$ |
| 2 | Inv Herbs | Inventory Units ($I_t^H$) | $I_t^H / \text{Max\_Capacity}_H$ |
| 3 | Inv Tools | Inventory Units ($I_t^T$) | $I_t^T / \text{Max\_Capacity}_T$ |
| 4 | Price Byrinium | Log-Spot ($\ln S_t^B$) | $(\ln S_t^B - \mu_B) / \sigma_{\text{stat},B}$ where $\sigma_{\text{stat}} = \sigma / \sqrt{2\theta}$ |
| 5 | Price Herbs | Log-Spot ($\ln S_t^H$) | $(\ln S_t^H - \mu_H) / \sigma_{\text{stat},H}$ |
| 6 | Price Tools | Log-Spot ($\ln S_t^T$) | $(\ln S_t^T - \mu_T) / \sigma_{\text{stat},T}$ |
| 7 | Party Fatigue | Fatigue Tracker ($F_t$) | $F_t / 100.0$ |
| 8 | Phase Index | Current Turn Phase ($\tau_t$) | $\tau_t / 3.0 \quad \text{where } \tau_t = t \pmod 4$ |
| 9 | Expedition Cooldown | Forced Cooldown Ticks | $T_{\text{cooldown}} / 2.0$ |

### 3.2 Action Space Configuration (Fixed-Fractional Engine)

A 1D discrete action mapping space containing exactly 8 choices:

1. **`HOLD` (Idx 0):** No trade/exploration activity. Active sedentary fatigue decay.

2. **`BUY_BYRINIUM` (Idx 1):** Allocate exactly 25% of liquid cash to purchase Raw Byrinium.

3. **`SELL_BYRINIUM` (Idx 2):** Liquidate exactly 100% of current Byrinium inventory.

4. **`BUY_HERBS` (Idx 3):** Allocate exactly 25% of liquid cash to purchase Sovereign Herbs.

5. **`SELL_HERBS` (Idx 4):** Liquidate exactly 100% of current Herb inventory.

6. **`BUY_TOOLS` (Idx 5):** Allocate exactly 25% of liquid cash to purchase Refined Tools.
7. **`SELL_TOOLS` (Idx 6):** Liquidate exactly 100% of current Refined Tools inventory.
8. **`LAUNCH_EXPEDITION` (Idx 7):** Commits the party to active resource extraction. Costs upfront cash fee $C_{\text{launch}}$ regardless of outcome. Fully blocked if $T_{\text{cooldown}} > 0$.

### 3.3 Target Objective Reward Function

$$R_t = \text{clip}\left(\alpha \cdot \ln\left(\frac{W_t}{W_{t-1}}\right), -c, +c\right) - \Psi_{\text{insolvency}}(W_t) - \Psi_{\text{fatigue}}(F_t) - \delta \cdot \mathbb{1}_{\{a_t = \text{HOLD}\}}$$

* **Logarithmic Portfolio Return:** Measures growth rate of total network net worth $W_t = C_t + \sum(I_t^i \cdot S_t^i)$. Symmetric clamping to $[-c, +c]$ isolates optimization routines from destabilizing valuation spikes.

* **Insolvency Defense Layer (Net-Worth-Keyed):** Activates only near true ruin thresholds ($W_t < W_{\text{crit}}$):

$$\Psi_{\text{insolvency}}(W_t) = \beta \cdot \left(\frac{W_{\text{crit}} - W_t}{W_{\text{crit}}}\right)^2 \cdot \mathbb{1}_{\{0 < W_t < W_{\text{crit}}\}} + \Omega \cdot \mathbb{1}_{\{W_t \le 0\}}$$

* **Fatigue Barrier Layer:** Penalizes the agent heavily as fatigue approaches maximum thresholds, prompting proactive execution of the REST action:

$$\text{\Psi}_{\text{fatigue}}(F_t) = \eta \cdot \exp(\xi(F_t - F_{\text{crit}}))$$

---

## 4. Production C# Godot 4 Native Interface

This production class handles memory-safe unmanaged unrolling and execution parity within the Godot .NET framework:

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
    BUY_TOOLS = 5,
    SELL_TOOLS = 6,
    LAUNCH_EXPEDITION = 7
}

public struct MarketEnvironmentState
{
    public float Cash;
    public float InvByrinium;
    public float InvHerbs;
    public float InvTools;
    public float PriceByrinium;
    public float PriceHerbs;
    public float PriceTools;
    public float Fatigue;
    public float PhaseIndex; // Raw turn index % 4
    public float CooldownTicks;

    public float[] GetNormalizedVector()
    {
        float networth = Cash + (InvByrinium * PriceByrinium) + (InvHerbs * PriceHerbs) + (InvTools * PriceTools);
        return new float[]
        {
            MathF.Log(Cash + 1.0f) / 11.5f,
            InvByrinium / 100.0f,
            InvHerbs / 100.0f,
            InvTools / 100.0f,
            (MathF.Log(PriceByrinium) - 6.9f) / 0.5f, 
            (MathF.Log(PriceHerbs) - 3.4f) / 0.15f,
            (MathF.Log(PriceTools) - 5.8f) / 0.25f, // Standardized against tools mu
            Fatigue / 100.0f,
            PhaseIndex / 3.0f,
            CooldownTicks / 2.0f
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

---

## 5. Addendum A — Shared-Constants Sidecar (Normative)

To eliminate structural observation drift, all environment configuration bounds are owned by the Python `EnvConfig` dataclass and serialized to a JSON sidecar next to the model artifacts (`exports/aurix_config.json`).

The C# client reads this sidecar at initialization to dynamically rebuild normalcy weights and phase action masks:

```json
{
  "obs_dim": 10,
  "n_actions": 8,
  "config": {
    "global_market_tax_rate": 0.05,
    "max_capacity_byrinium": 500,
    "max_capacity_herbs": 2500,
    "max_capacity_tools": 100,
    "expedition_launch_cost": 150.0
  },
  "derived": {
    "sigma_stat": [0.40, 0.15, 0.25],
    "gou_mu": [5.01, 3.40, 6.40]
  }
}

```

---

## 6. Addendum B — Desktop Production Safeguards

1. **Prevention of Unmanaged Memory Leaks:** The underlying `InferenceSession`, `OrtValue`, and output payload collections operate completely outside the standard .NET Garbage Collector. You must explicitly isolate inference pipelines inside lexical `using` statements to prevent major native heap memory leakage.

2. **Virtual PCK Packaging Encapsulation:** Compiled Godot desktop application payloads aggregate raw file resources into internal virtual archives (`.pck` binaries). Passing a direct file string such as `"res://models/policy.onnx"` into the native ONNX runtime constructor will crash. Implement `Godot.FileAccess.GetFileAsBytes()` to safely load the model as a raw memory buffer stream.

3. **Graph Execution Parity & Structural Decoupling:** Do not compile network operations like `ArgMax` or dynamic input control-flow graphs directly into the exported `.onnx` binary asset. Keep your model outputs limited to raw float arrays representing action values, and perform index argmax resolution natively in C# to maximize cross-platform driver compatibility.
