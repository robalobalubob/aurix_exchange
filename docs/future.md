# Aurix Exchange: Phase A Technical Reference & Forward Roadmap

## 1. Project Overview & Meta-Parameters

* **Project Name:** Aurix Exchange Simulation
* **Architecture:** Double DQN + Prioritized Experience Replay (PER) + Custom Action Masking
* **Training Framework:** Python 3.10+, Gymnasium, PyTorch
* **Production Environment:** Godot 4.NET Core (C#), Desktop Execution
* **Core Goal:** Solve a strictly stationary, single-agent market arbitrage and resource extraction Markov Decision Process (MDP).

---

## 2. Phase A Reference Specification (Normative Baseline)

### 2.1 Price Dynamics Engine

Spot prices follow a strictly stationary **Geometric Ornstein-Uhlenbeck (GOU)** process with a constant, static long-term log-mean parameter ($\mu_i$). The market logic does not adapt to agent inventories or trades in Phase A, ensuring a stable baseline gradient space and closing the self-inflation exploit.

Let $X_t^i = \ln S_t^i$ be the log-price of commodity $i$. The exact closed-form temporal state update $(\Delta t = 1)$ executed during the environment step is:


$$X_{t+1}^i = \mu_i + (X_t^i - \mu_i)e^{-\theta_i} + \sigma_i \sqrt{\frac{1 - e^{-2\theta_i}}{2\theta_i}} \cdot \epsilon_i \quad \text{where } \epsilon_i \sim N(0,1)$$

Execution spot market price: $S_t^i = \exp(X_t^i)$.

### 2.2 MDP Configuration Contracts

#### Observation Space (10-Dimensional Float32 Array)

The state vector is exposed to the model as a fixed-size 10-dimensional array of float32 values:


$$o_t = [C_t, I_t^B, I_t^H, I_t^T, X_t^B, X_t^H, X_t^T, F_t, \tau_t, T_{\text{cooldown}}]^T$$

| Index | Feature Name | Raw Definition | Fixed Normalization / Transformation Pipeline |
| --- | --- | --- | --- |
| 0 | Normalized Cash | Cash Balance ($C_t$) | $\ln(C_t + 1) / \ln(\text{Max\_Expected\_Cash})$ |
| 1 | Inv Byrinium | Inventory Units ($I_t^B$) | $I_t^B / \text{Max\_Capacity}_B \quad \text{where } \text{Max\_Capacity}_B = 500$ |
| 2 | Inv Herbs | Inventory Units ($I_t^H$) | $I_t^H / \text{Max\_Capacity}_H \quad \text{where } \text{Max\_Capacity}_H = 2500$ |
| 3 | Inv Tools | Inventory Units ($I_t^T$) | $I_t^T / \text{Max\_Capacity}_T \quad \text{where } \text{Max\_Capacity}_T = 100$ |
| 4 | Price Byrinium | Log-Spot ($\ln S_t^B$) | $(\ln S_t^B - \mu_B) / \sigma_{\text{stat},B} \quad \text{where } \sigma_{\text{stat}} = \sigma / \sqrt{2\theta} \approx 0.8944$ |
| 5 | Price Herbs | Log-Spot ($\ln S_t^H$) | $(\ln S_t^H - \mu_H) / \sigma_{\text{stat},H} \quad \text{where } \sigma_{\text{stat}} = \sigma / \sqrt{2\theta} = 0.1500$ |
| 6 | Price Tools | Log-Spot ($\ln S_t^T$) | $(\ln S_t^T - \mu_T) / \sigma_{\text{stat},T} \quad \text{where } \sigma_{\text{stat}} = \sigma / \sqrt{2\theta} \approx 0.7905$ |
| 7 | Party Fatigue | Fatigue Tracker ($F_t$) | $F_t / 100.0$ |
| 8 | Phase Index | Current Turn Phase ($\tau_t$) | $\tau_t / 3.0 \quad \text{where } \tau_t = t \pmod 4$ |
| 9 | Expedition Cooldown | Forced Cooldown Ticks | $T_{\text{cooldown}} / 2.0$ |

#### Action Space Configurations

* `Idx 0`: **`HOLD`** — No trade or extraction activity. Promotes sedentary fatigue decay.
* `Idx 1-2`: **`BUY / SELL BYRINIUM`** — Allocate 25% cash / Liquidate 100% stock.
* `Idx 3-4`: **`BUY / SELL HERBS`** — Allocate 25% cash / Liquidate 100% stock.
* `Idx 5-6`: **`BUY / SELL TOOLS`** — Allocate 25% cash / Liquidate 100% stock.
* `Idx 7`: **`LAUNCH_EXPEDITION`** — Charges an upfront, fatigue-scaled launch fee paid regardless of operational outcome: $C_{\text{launch}} = \text{launch\_fee\_base} \cdot (1 + k \cdot F_t / F_{\max})$, where `launch_fee_base = 100.0` and `launch_fee_fatigue_k = 1.0` (i.e. 100.0 at zero fatigue, rising with party fatigue). **Eligibility is gated separately**: the action is masked out unless cash $\geq$ `expedition_min_cash = 200.0` *and* $T_{\text{cooldown}} = 0$. Note `expedition_min_cash` is the liquidity guardrail, not the fee.

### 2.3 Committed Action Masking Logic (Python)

```python
import numpy as np

def get_action_mask(state, expedition_min_cash=200.0, sell_inventory_epsilon=1e-5):
    """
    Computes valid boolean mask matching the committed environment logic.
    Excludes experimental curfew mechanics to maintain contract parity.
    """
    mask = np.ones(8, dtype=bool)
    
    # De-normalize parameters for rule validation
    cash = np.exp(state[0] * 11.5) - 1.0
    cooldown = int(state[9] * 2.0)
    
    inv_byrinium = state[1]
    inv_herbs = state[2]
    inv_tools = state[3]

    # Rule 1: Temporal Cooldown Lockout
    if cooldown > 0:
        mask[7] = False  # Block LAUNCH_EXPEDITION

    # Rule 2: Economic Liquidity Guardrail 
    if cash < expedition_min_cash:
        mask[7] = False

    # Rule 3: Robust Epsilon-Based Liquidation Invalidation
    if inv_byrinium <= sell_inventory_epsilon: mask[2] = False
    if inv_herbs <= sell_inventory_epsilon:    mask[4] = False
    if inv_tools <= sell_inventory_epsilon:    mask[6] = False

    return mask

```

---

## 3. Production C# Godot 4 Native Interface

To eliminate observation distribution drift between training and deployment, this class reads its scaling coefficients from the exported `aurix_config.json` sidecar instead of hardcoding them. The deserialization model mirrors the **nested** structure that `export_config` actually emits — a top-level object with `config` (raw `EnvConfig` fields) and `derived` (precomputed quantities like `sigma_stat`) sub-objects. Only genuinely structural constants that the sidecar does not parameterize (the 4-phase day divisor) remain as literals.

```csharp
using System;
using System.Collections.Generic;
using System.Text.Json;
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

// Mirrors the nested payload written by export_config():
//   { "obs_dim", "n_actions", "config": {...}, "derived": {...} }
// Property names are snake_case to bind directly to the JSON keys.
public class EnvParams
{
    public float[] max_inventory { get; set; }   // [B, H, T] capacities, e.g. [500, 2500, 100]
    public float[] gou_mu { get; set; }          // long-term log-means [B, H, T]
    public float cash_norm { get; set; }         // log-cash divisor (e.g. 11.5)
    public float fatigue_max { get; set; }       // fatigue normalizer (e.g. 100.0)
    public float expedition_cooldown_steps { get; set; } // cooldown normalizer (e.g. 2)
}

public class DerivedParams
{
    public float[] sigma_stat { get; set; }      // stationary std σ/√(2θ) [B, H, T]
}

public class SimulationConfig
{
    public int obs_dim { get; set; }
    public int n_actions { get; set; }
    public EnvParams config { get; set; }
    public DerivedParams derived { get; set; }
}

public class PalanDecisionEngine : IDisposable
{
    // Number of diurnal phases; structural (not parameterized in the sidecar).
    private const float PhaseDivisor = 3.0f; // N_PHASES - 1

    private InferenceSession _session;
    private string _inputName;
    private SimulationConfig _config;

    public PalanDecisionEngine(string onnxModelPath, string configJsonPath)
    {
        // Safe PCK stream reading prevents uncompressed file extraction crashes
        byte[] modelBytes = Godot.FileAccess.GetFileAsBytes(onnxModelPath);
        _session = new InferenceSession(modelBytes);
        _inputName = _session.InputNames[0];

        // Parse runtime configuration mappings directly from the shared config sidecar
        string jsonText = Godot.FileAccess.GetFileAsString(configJsonPath);
        _config = JsonSerializer.Deserialize<SimulationConfig>(jsonText);
    }

    public float[] NormalizeState(
        float cash, float invB, float invH, float invT, 
        float priceB, float priceH, float priceT, 
        float fatigue, float phaseIndex, float cooldownTicks)
    {
        EnvParams c = _config.config;
        float[] cap = c.max_inventory;
        float[] mu = c.gou_mu;
        float[] sig = _config.derived.sigma_stat;

        return new float[]
        {
            MathF.Log(cash + 1.0f) / c.cash_norm,
            invB / cap[0],
            invH / cap[1],
            invT / cap[2],
            (MathF.Log(priceB) - mu[0]) / sig[0],
            (MathF.Log(priceH) - mu[1]) / sig[1],
            (MathF.Log(priceT) - mu[2]) / sig[2],
            fatigue / c.fatigue_max,
            phaseIndex / PhaseDivisor,
            cooldownTicks / c.expedition_cooldown_steps
        };
    }

    public GameAction GetBestAction(float[] normalizedFeatures, bool[] actionMask)
    {
        long[] shape = { 1, 10 };

        // Enforce lexical using statements to guarantee execution outside .NET Garbage Collector
        using var inputTensor = OrtValue.CreateTensorWithDataAsSpan(normalizedFeatures, shape);
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

## 4. Part 4: Production Desktop Safeguards (Verified)

1. **Unmanaged Heap Protection:** `InferenceSession` and `OrtValue` allocations use unmanaged native memory backing. Wrap inference frames tightly within lexical or structural scopes to eliminate catastrophic native heap memory leaks.
2. **PCK File Virtualization Bounds:** Direct paths using `"res://"` strings break inside compiled `.pck` boundaries when targeted by unmanaged native C++ engines. Utilize `Godot.FileAccess` byte buffers to bridge the virtual filesystem layer safely.
3. **Index Policy Enforcement Layer:** `ArgMax` layers are banned within the exported `.onnx` graph to insulate inferences from hardware-specific engine driver interop bugs. Keep raw float tensors streaming into the C# host loop and calculate action choices via masked array parsing.

---

## 5. Phase B Evolution Roadmap (Future Implementations Only)

The following concepts are explicitly non-normative, un-implemented features deferred to the Phase B system refactor:

### 5.1 Non-Stationary Endogenous Supply Modeling

* **Concept:** Transition the fixed $\mu_i$ values to a dynamic elasticity vector responding to resource storage quantities inside the public depots of Belos:

$$\mu_{i,t} = \mu_{i,\text{baseline}} \cdot \left(1.0 + \omega_i \cdot \left[\frac{\text{Deficit}_t^i - I_{\text{market},t}^i}{I_{\text{market, target}}^i}\right]\right)$$


* **Systemic Friction:** This requires upgrading the training framework from a standard single-agent Double DQN to a multi-agent self-play architecture to stabilize training against policy-induced price shifts.

### 5.2 Chronological Phase Curfew Lockouts

* **Concept:** Layering true phase-conditioned action masking rule structures. When the phase index rolls into the Night phase ($\tau_t = 3$), standard open-market purchase indexes will be actively masked out, restricting legal transit lines and forcing interactions with localized black-market smuggling channels.
