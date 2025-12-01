
ASTRAL: Abstraction‑Slot Test‑time Reweighting for Adaptation in Latent RL

1. Problem and Motivation (Small‑Scale Version)

In‑context RL agents (e.g., AMAGO‑style sequence models) process a window of recent transitions and produce an adapted policy in one forward pass. Adaptation to task identity or environment changes is implicit in a single dense context embedding. This has two issues:

1. **No explicit strategies**: The agent’s internal “modes” (different tasks, dynamic regimes, exploration vs exploitation) are entangled and opaque.
2. **No controlled test‑time RL**: If the environment shifts, you can only rely on in‑context adaptation (no gradients) or full‑network fine‑tuning (unstable, expensive).

**Goal (small‑scale):**

On a simple non‑stationary benchmark, build an in‑context RL agent with:

* A small **bank of K abstraction vectors** (slots) that represent latent strategies.
* A **gating network** that picks a mixture over these slots from context.
* **Policy and value heads** that are **forced to depend** on these slots.
* **Test‑time RL** that updates only the light gating network (not the backbone).
* **Diagnostics and interventions** to test whether these abstractions:
  * Align with different environment modes,
  * Are actually used (no bypass),
  * Are not collapsed into a single slot.

This is ASTRAL: **Abstraction‑Slot Test‑time Reweighting for Adaptation in Latent RL**.

2. Benchmark: NonStationaryCartPole

### 2.1. Base environment

Use Gym’s `CartPole‑v1`:

* Observations: 4D continuous vector.
* Actions: 2 discrete (left/right).
* Reward: +1 per timestep until failure.

### 2.2. Non‑stationarity wrapper

Define `NonStationaryCartPole`:

* At the start of each episode, sample a hidden mode `m ∈ {0,1,2}`.
* Modes change **dynamics parameters** (gravity, pole length), e.g.:
  * `m=0`: gravity = 9.8, length = 0.5
  * `m=1`: gravity = 7.5, length = 0.7
  * `m=2`: gravity = 12.0, length = 0.4
* The agent **does not observe** `m`. It only sees the state and reward.
* For logging only, put `info["mode"] = m` in `step`.

Training: reset with `mode=None` → random mode each episode.
Testing: fix `mode = m*` to study adaptation to that specific regime.

This gives a fast, clear non‑stationary setting where an in‑context learner and the abstraction bank should differentiate regimes.

3. Architecture

### 3.1. Backbone: GRU‑based in‑context encoder

We use a small GRU as the in‑context RL backbone.

**Input per timestep t:**

* Current observation \(s_t\).
* Previous action \(a_{t-1}\) (one‑hot).
* Previous reward \(r_{t-1}\).

Concatenate to a vector:
\[
x_t = [s_t; \text{onehot}(a_{t-1}); r_{t-1}] \in \mathbb{R}^{d_{\text{in}}}.
\]

Pass through a linear layer to get to model dimension \(d\):
\[
\tilde{x}_t = \mathrm{MLP}_{\text{in}}(x_t) \in \mathbb{R}^{d}.
\]

Update GRU:
\[
h_t = \mathrm{GRUCell}(\tilde{x}_t, h_{t-1}), \quad h_t \in \mathbb{R}^d.
\]

* Initialize \(h_0 = 0\) at episode start.
* For \(t=0\), use a dummy previous action/reward (zeros or special token embedding).

**Interpretation:** \(h_t\) is the context embedding summarizing recent transitions; it’s where in‑context RL lives.

### 3.2. Abstraction Bank (Slots)

Let \(d\) be the GRU hidden size.

**Parameters:**

* Abstraction matrix \(A \in \mathbb{R}^{K \times d}\), rows \(a_k\) are slots.
* Gating MLP \(g: \mathbb{R}^d \to \mathbb{R}^K\).

Forward at time t:

1. Gating logits:
   \[
   \ell_t = g(h_t) \in \mathbb{R}^K.
   \]
2. Mixture weights with temperature \(\tau\):
   \[
   w_t = \mathrm{softmax}(\ell_t / \tau), \quad w_t \in \Delta^{K-1}.
   \]
3. Combined abstraction:
   \[
   z_t = \sum_{k=1}^K w_{t,k} a_k = w_t^{\top} A \in \mathbb{R}^d.
   \]

Default: \(K=3\) or \(5\), \(d=64\), \(\tau=1.0\).

### 3.3. Anti‑bypass design: FiLM as the default

To prevent the policy from ignoring abstractions, **FiLM modulation is the default** (not concatenation).

We use \(z_t\) to modulate \(h_t\):

* Compute \((\gamma_t, \beta_t) \in \mathbb{R}^{2d}\) from \(z_t\):
  \[
  [\gamma_t, \beta_t] = \mathrm{MLP}_{\text{film}}(z_t).
  \]
* Modulate context:
  \[
  h'_t = \gamma_t \odot h_t + \beta_t.
  \]

Only \(h'_t\) is fed to the policy and value heads. There is **no direct path** from raw \(h_t\) to the heads.

**Concatenation** \([h; z]\) is reserved as an **ablation baseline** to test bypass effect, not the main model.

### 3.4. Policy and Value Heads

Simple MLPs on \(h'_t\):

*PolicyHead*

```
class PolicyHead:
    def __init__(self, d_model, num_actions):
        self.net = MLP(d_model -> d_model -> num_actions)
    def forward(self, h_mod):
        return self.net(h_mod)  # logits
```

*ValueHead*

```
class ValueHead:
    def __init__(self, d_model):
        self.net = MLP(d_model -> d_model -> 1)
    def forward(self, h_mod):
        return self.net(h_mod).squeeze(-1)
```

4. Anti–Mode‑Collapse Regularization

To prevent the model from using only one slot or ignoring the bank, we add:

### 4.1. Per‑sample entropy on w

Encourage each state’s mixture to carry non‑zero entropy:
\[
L_{\text{w‑ent}} = -\lambda_{\text{w‑ent}} \mathbb{E}_t [H(w_t)].
\]

### 4.2. Load balancing over slots

Encourage average usage \(\bar{w}\) across a batch to be near‑uniform:
\[
L_{\text{lb}} = \lambda_{\text{lb}} \sum_k \bar{w}_k \log \bar{w}_k
\]
(negative entropy of \(\bar{w}\)).

### 4.3. Optional: Orthogonality/Diversity of A

Encourage abstraction vectors to be diverse:
\[
L_{\text{orth}} = \lambda_{\text{orth}} \big\| A A^{\top} - I_K \big\|^2_F.
\]

### 4.4. Monitoring and dead‑slot revival (debugging)

During training:

* Track average `w[:, k]` usage per slot over a moving window.
* If a slot’s usage stays below a threshold (e.g., < 1% average mass) for many updates, reinitialize that row of \(A\) from recent context embeddings or randomly.

This isn’t part of loss, but it’s a practical debugging tool if collapse is severe.

5. RL Training Setup

Use PPO or A2C; PPO is more robust, A2C is simpler. CartPole is cheap, so PPO is fine.

### 5.1. Rollout

For each parallel environment:

1. At time t:
   * Use \((s_t, a_{t-1}, r_{t-1})\) to update GRU and get \(h_t\).
   * Pass \(h_t\) through abstraction bank → \(z_t, w_t\).
   * Compute \(h'_t = \mathrm{FiLM}(h_t, z_t)\).
   * Compute policy logits and value.
   * Sample action \(a_t\) from \(\pi(a_t|s_t)\).
   * Step the env: get \((s_{t+1}, r_t, \text{done}, \text{info})\).

2. Store:
   * \(s_t, a_t, r_t\),
   * log‑prob of \(a_t\),
   * value estimate \(V(s_t)\),
   * mixture weights \(w_t\),
   * env mode from `info["mode"]`.

Roll for \(T\) steps; collect across \(N\) envs.

### 5.2. Loss

Standard PPO loss + abstraction regularizers:
\[
L = L_{\text{actor}} + c_v L_{\text{critic}} + L_{\text{policy‑ent}} + L_{\text{w‑ent}} + L_{\text{lb}} + L_{\text{orth}}.
\]

6. Test‑Time Adaptation (ASTRAL’s RL)

After training:

### 6.1. Freeze vs trainable

* **Frozen:**
  * GRU backbone and input encoder.
  * Abstraction vectors \(A\).
  * Policy head and value head.

* **Trainable:**
  * Only the **abstraction mixer** (gating MLP).
  * Optionally FiLM’s small MLP (if you want a slightly larger adaptation surface).

### 6.2. Adaptation protocol

For a fixed mode \(m^*\) (e.g., gravity = 12.0):

1. Initialize the mixer parameters to their trained values.
2. For each adaptation iteration:
   * Rollout for episodes using current policy.
   * Compute advantages and returns from these episodes.
   * Compute PPO (or A2C) loss **but backprop only through mixer** (and optional FiLM).
   * Apply gradient steps with a small learning rate.
3. Track performance curves.

**Baselines:**

* **No adaptation:** Mixer frozen; pure in‑context adaptation.
* **Full‑head adaptation** (optional baseline): Freeze GRU + \(A\); adapt policy/value heads; compare stability and speed.

7. Evaluation and Interpretability (Addressing Visualization Gotcha)

### 7.1. Correlational analysis

1. **Mode vs mixture heatmap:**
   * For each episode, compute mean \(w_t\) over time.
   * Group episodes by true env mode \(m\).
   * Plot heatmap: modes (rows) vs abstraction indices k (cols), value = mean \(w_k\).
   * Expect different modes to activate different slots.

2. **Entropy and load balancing:**
   * Plot distribution of \(\bar{w}\) across training and ensure no slot is starved.

### 7.2. Causal interventions (mandatory)

To avoid fake interpretability, do explicit interventions:

**Experiment A: Clamp a single slot**

* For each slot k:
  * Override \(w_t\) during rollout to be one‑hot at k (or highly peaked).
  * Run several episodes in each mode and measure performance and qualitative behavior.
* If slot k really corresponds to a “strategy,” clamping it should induce consistent behavioral patterns.

**Experiment B: Disable a slot**

* For each slot k:
  * During rollout, force \(w_t[k] = 0\) and renormalize the rest.
  * Compare performance and behavior to unmodified policy.
* If disabling a slot significantly harms performance in specific modes, that slot is functionally important there.

**Experiment C: Cross‑mode transfer**

* Identify a slot that lights up in mode \(m=0\).
* Clamp that slot in mode \(m=1\) or \(m=2\) and see if it “pulls” behavior toward what is typical of \(m=0\).
* This tests whether slots encode reusable strategies instead of arbitrary features.

8. Summary

ASTRAL (updated) is:

* A small in‑context RL agent (GRU‑based) on NonStationaryCartPole.
* Equipped with a **bank of K abstraction vectors** and a **gating network** that produces mixture weights.
* Uses **FiLM modulation** as the default, forcing the policy/value heads to depend on abstractions and reducing bypass.
* Uses entropy and load‑balancing regularizers (and optionally orthogonality/monitoring) to reduce mode collapse.
* Implements **test‑time RL** that updates only the gating network (and optionally FiLM), providing a low‑dimensional adaptation surface.
* Includes an explicit **intervention‑based interpretability protocol** (clamp/disable slots, cross‑mode tests) to move beyond pretty heatmaps.

This directly addresses bypass effect → FiLM bottleneck, mode collapse → entropy + load balancing + orthogonality + usage monitoring, and visualization ambiguity → causal interventions.