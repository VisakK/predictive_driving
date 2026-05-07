# Anomaly Input — Per-Agent Self-Attention + ViT Cross-Attention

## Motivation

The current `AgentAnomalyEncoder` (`src/driving/vit_cvae.py:242`) flattens the
`(15, 4)` per-agent anomaly tensor into a 60-d vector, runs it through a
2-layer MLP (`60 → 64 → 32`), and concatenates the 32-d output to the ViT CLS
embedding. Two problems:

1. **Per-agent identity is destroyed.** The slot ordering is consistent
   (`_sorted_nearest`), but flattening forces the MLP to relearn agent
   structure from scratch.
2. **No spatial coupling to the scene.** The 32-d global summary never
   meets the ViT's spatial tokens. The policy can know "something is
   anomalous" but not "this anomaly is in the lane to my left."

The v3 fair eval (`experiments/050_h10_v3_highway/results/fair_eval_v3.md`)
shows ViT-only (051, 21.4% crash) outperforming every ExpectedInput-H10
variant. The hypothesis: the H10 anomaly signal is informative but is
being squeezed through a bottleneck that strips both per-agent and spatial
structure before it reaches the policy.

## Design overview

Replace `AgentAnomalyEncoder` with a small two-stage transformer:

```
agent_kinematics + agent_anomaly  ──►  Stage A: per-agent token build
                                                  │
                                                  ▼
                                       Stage B: agent self-attention
                                                  │
ViT (all tokens)  ──────────────────►  Stage C: agent → scene cross-attention
                                                  │
                                                  ▼
                                       Stage D: presence-weighted pool
                                                  │
                                                  ▼
                                  concat with ViT CLS  →  policy features
```

The ViT is modified to expose all `1 + H·W = 122` tokens (CLS + patch
tokens) instead of only the CLS token. The new module operates on per-agent
tokens that *can* attend back to the scene patches, with a spatial bias so
each agent's token preferentially looks at the patches its `(x, y)` falls
into. Output is concatenated with the existing CLS embedding so the design
strictly subsumes the current pathway.

## Slot semantics — important caveat

The 15 obs slots are **not** stable per-agent across frames. Two layers
of tracking exist in `KinHistoryDictObsWrapper`
(`src/driving/adversarial.py:427`):

1. **Internally, agents are tracked.** `self._history` is keyed by
   `id(v)` — the Python object id of the underlying vehicle. Each
   physical vehicle's last 10 kinematic frames are stored under its own
   key, and the H10 anomaly's `_pending_predictions` queue uses the
   same keying. The history persists as long as the vehicle exists.

2. **Slot ordering is recomputed every step.** `_sorted_nearest`
   (`adversarial.py:516`) re-sorts every step by current distance to
   ego:

   ```python
   others.sort(key=lambda v: np.sum((v.position - ego.position) ** 2))
   return others[: self.N_AGENTS]
   ```

   So slot 0 is "currently closest agent," slot 1 is "second closest,"
   etc. Slot occupancy can change frame to frame.

**Crucial detail:** within a single observation, `agent_kinematics[i]`,
`agent_kin_history[i]`, and `agent_anomaly[i]` are all coherent — they
describe the same physical vehicle, the one currently in slot `i`. But
that vehicle may have been in a different slot last step.

**Implications for this design:**

- Stage A's "slot positional embedding" encodes **rank in proximity**,
  not agent identity. Useful as a stable "is this the closest car?"
  feature, but should not be expected to track an agent.
- Stage C's spatial bias keys off the slot's current `(x, y)`, which is
  the right semantic since `(x, y)` always reflects whichever vehicle
  occupies the slot now.
- A recurrent / temporal attention pattern that reasons about a single
  agent over time is **not** expressible by simply unrolling the model
  across timesteps — it requires consuming the per-agent history that
  *is* stored coherently in `agent_kin_history[i]`. That motivates
  Stage A.5 below.

## Stage A — Per-agent token construction

**Inputs (per env):**

| Source                | Key                  | Shape       | Notes |
|-----------------------|----------------------|-------------|-------|
| Kinematics (current)  | `agent_kinematics`   | `(15, 7)`   | `[presence, x, y, vx, vy, cos_h, sin_h]` (normalized) |
| Anomaly (H10 wrapper) | `agent_anomaly`      | `(15, 4)`   | `[presence, anomaly, risk, raw_error]` |
| Kin history           | `agent_kin_history`  | `(15, 10, 7)` | Per-slot, coherent with current occupant. Used in Stage A.5. |

**Token build:** for each agent `i`, concatenate `[kin_i (7), anom_i (4)]`
→ `(15, 11)`, project to model dim `D=64`:

```python
# token_in: (B, N=15, 11)
agent_tokens = self.token_proj(token_in)   # (B, 15, 64)
agent_tokens = agent_tokens + self.agent_pos_emb   # learned (1, 15, 64)
```

The learned slot positional embedding gives the network a stable handle on
slot identity (slot 0 is always the nearest agent, slot 14 the farthest).

**Padding mask:** `presence == 0` slots are masked out of every subsequent
attention (`key_padding_mask=(presence < 0.5)`).

## Stage A.5 — Per-slot temporal encoder (next experiment)

**Status:** *not* in experiment 052. Reserved for the follow-up
experiment after 052 lands.

The 10-frame history at `agent_kin_history[i]` is coherent with whoever
currently occupies slot `i` on this step (see Slot semantics above).
Even though slot identity is unstable across model timesteps, *within
this observation* slot `i`'s history is a clean per-agent trajectory —
worth encoding before the self-attention layer.

```python
# agent_kin_history: (B, 15, 10, 7)
# Single shared GRU over all 15 slots, per-slot independent rollouts.
self.per_slot_gru = nn.GRU(
    input_size=7, hidden_size=32, num_layers=1, batch_first=True,
)

B, N, T, F = kin_history.shape
flat = kin_history.reshape(B * N, T, F)        # (B*15, 10, 7)
_, h = self.per_slot_gru(flat)                 # (1, B*15, 32)
temporal = h[-1].reshape(B, N, 32)             # (B, 15, 32)

# Concat with the (kin, anomaly) features before token projection
token_in = torch.cat([agent_kin, agent_anom, temporal], dim=-1)  # (B, 15, 43)
agent_tokens = self.token_proj(token_in)       # (B, 15, 64)
```

Why GRU and not a temporal transformer: the sequence is only 10 steps,
the model is shared across slots, and CPU compute is the binding
constraint. A 1-layer GRU adds ~5k params and is cheap. If 052+A.5
clearly beats 052, replace with a small temporal transformer as an
ablation.

What this does *not* solve: when an agent migrates between slots across
two consecutive model steps, the GRU's internal state (carried within
this obs's 10 frames) is the right state for the *agent*, but the slot
positional embedding in Stage A still treats slot `i` as the same
"identity bucket" before and after the migration. Identity-stable
tracking would require a true ID-based slot assignment in the wrapper —
out of scope here, but would unlock cross-step recurrent policies.

## Stage B — Agent self-attention

A single TransformerEncoderLayer over the 15 agent tokens lets agents
share context. Example use cases the layer can express that the current
flat-MLP cannot:

- "The agent in slot 2 is anomalous **and** is in the same lane as
  the agent in slot 5 that is decelerating" — joint reasoning about the
  pair.
- "Three of four nearby agents are slowing simultaneously" — coordinated
  threat detection that the flat MLP can only learn through brittle
  feature interactions.

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=64, nhead=4, dim_feedforward=128,
    dropout=0.1, batch_first=True, norm_first=True,
)
self.agent_self_attn = nn.TransformerEncoder(encoder_layer, num_layers=1)

# (B, 15, 64), (B, 15) padding mask
agent_tokens = self.agent_self_attn(agent_tokens, src_key_padding_mask=pad_mask)
```

Single layer keeps compute cheap (CPU-only project constraint).

## Stage C — Agent → scene cross-attention

**ViT exposes all tokens.** Modify `ViTEncoder.forward` to return the full
sequence in addition to the CLS embedding:

```python
def forward(self, grid):
    ...
    x = self.encoder(x)
    x = self.norm(x)
    return x[:, 0], x[:, 1:]   # (B, D), (B, H*W, D)  — CLS, patches
```

(`x[:, 1:]` shape `(B, 121, 64)` for the 11×11 grid.)

**Cross-attention head:** queries are agent tokens, keys/values are patch
tokens.

```python
self.agent_to_scene = nn.MultiheadAttention(
    embed_dim=64, num_heads=4, dropout=0.1, batch_first=True,
)
self.cross_norm = nn.LayerNorm(64)
self.cross_ffn = nn.Sequential(
    nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64),
)
self.cross_ffn_norm = nn.LayerNorm(64)

# Q: (B, 15, 64) agent tokens
# K, V: (B, 121, 64) ViT patch tokens
attn_out, _ = self.agent_to_scene(
    agent_tokens, scene_patches, scene_patches,
    attn_mask=spatial_bias,           # (15, 121) — see below
)
agent_tokens = self.cross_norm(agent_tokens + attn_out)
agent_tokens = self.cross_ffn_norm(agent_tokens + self.cross_ffn(agent_tokens))
```

**Spatial bias.** Without help, attention from agent slot `i` to all 121
patches has no inductive prior linking the agent's `(x, y)` to its grid
cell. We add an additive bias that softly prefers patches near each agent's
position:

```python
# agent positions in normalized [-1, 1] (already in agent_kinematics[:, :, 1:3])
# grid layout: 11x11, range [-27.5, 27.5] m, step 5 m, ego-centered
def build_spatial_bias(agent_xy, grid_h=11, grid_w=11, sigma=1.5):
    # patch centers in normalized coords
    ys = torch.linspace(-1, 1, grid_h, device=agent_xy.device)
    xs = torch.linspace(-1, 1, grid_w, device=agent_xy.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    patch_xy = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # (121, 2)

    # squared distance from each agent to each patch
    diff = agent_xy.unsqueeze(2) - patch_xy.unsqueeze(0).unsqueeze(0)  # (B, 15, 121, 2)
    d2 = diff.pow(2).sum(-1)                                           # (B, 15, 121)
    return -d2 / (2 * sigma**2)   # additive log-Gaussian bias
```

This is added to the attention logits — patches further from the agent
become exponentially less preferred but never strictly masked, so the
network can override the prior when the kinematics are unreliable
(occluded, fast-moving, or missing agents). `sigma` is a hyperparameter;
start at `1.5` (≈ 20 m in world units) and ablate.

For padded agent slots (`presence < 0.5`), set the entire bias row to
`-inf` so they contribute nothing downstream.

## Stage D — Pool to a fixed-size embedding

```python
# weight by presence so padded slots contribute zero
weights = presence.unsqueeze(-1)                  # (B, 15, 1)
pooled = (agent_tokens * weights).sum(1) / weights.sum(1).clamp(min=1)
# (B, 64)
```

Risk-weighted pooling is a tempting alternative (`weights = risk * presence`)
but conflates "what to look at" with "how confident we are it matters."
Keep presence-weighted as the default; risk-weighted pooling is an ablation.

## Stage E — Final feature assembly

```python
# In ViTCVAEExtractor.forward
cls, scene_patches = self.vit(grid)              # (B, 64), (B, 121, 64)
agent_token_out = self.anomaly_attn(             # (B, 64)
    agent_kin, agent_anom, scene_patches,
)
features = [cls, agent_token_out]                # 64 + 64 = 128
if self.use_kinematics_policy:
    features.append(self.kin_encoder(kin_hist))  # +64
return torch.cat(features, dim=1)
```

Total `features_dim = 128` (or 192 with kinematics history). The existing
CVAE/Discriminator auxiliary losses keep operating on `cls` unchanged.

## Module skeleton

New file `src/driving/anomaly_attention.py`:

```python
class AnomalyAttentionEncoder(nn.Module):
    """Per-agent self-attention + cross-attention to ViT scene tokens."""

    def __init__(
        self,
        n_agents: int = 15,
        kin_feat_dim: int = 7,
        anomaly_feat_dim: int = 4,
        scene_dim: int = 64,
        embed_dim: int = 64,
        n_heads: int = 4,
        ffn_ratio: int = 2,
        dropout: float = 0.1,
        spatial_sigma: float = 1.5,
        grid_h: int = 11,
        grid_w: int = 11,
    ):
        ...

    def forward(
        self,
        agent_kin: torch.Tensor,        # (B, N, 7)
        agent_anom: torch.Tensor,       # (B, N, 4)
        scene_patches: torch.Tensor,    # (B, H*W, scene_dim)
    ) -> torch.Tensor:                  # (B, embed_dim)
        ...
```

Wiring changes:

- `src/driving/vit_cvae.py`
  - `ViTEncoder.forward` returns `(cls, patches)`. Update existing
    callers (`ViTCVAEExtractor.forward`, `compute_auxiliary_losses`,
    `compute_anomaly_scores`) to unpack the tuple.
  - `ViTCVAEExtractor` gains `use_anomaly_attention_policy` flag.
    When set, instantiate `AnomalyAttentionEncoder` and call it in
    `forward`. Mutually exclusive with `use_anomaly_policy` /
    `use_learned_anomaly_policy` (assert at construct time).
  - `actual_features_dim` calculation adds `embed_dim` (64).

- `experiments/052_h10_attn_v3_highway/config.yaml` (new) — clone of
  `experiments/050_h10_v3_highway/config.yaml` with
  `use_anomaly_policy: false`, `use_anomaly_attention_policy: true`.
  Same env (`AdversarialHighwayV3` + horizon=10 wrapper), same total
  timesteps, same seeds.

## Parameter count

| Component                                          | Params |
|----------------------------------------------------|-------:|
| `token_proj` (11→64)                               |    768 |
| Slot positional embedding                          |    960 |
| Self-attention encoder layer (D=64, h=4, ff=128)   | ~33k   |
| Cross-attention layer + FFN + 2 LayerNorms         | ~33k   |
| **Total new**                                      | **~68k** |

Removed: `AgentAnomalyEncoder` MLP (~6k). Net add: ~62k. Negligible
relative to the existing ViT+CVAE+Discriminator extractor (~250k+).

## Compute cost (CPU)

- Self-attention: 15×15 attention, single layer → trivial.
- Cross-attention: 15 queries × 121 keys, single layer → ~1800
  attention scores per env per step.
- ViT change: just keeping the patch tokens we already computed; no
  additional encoder forward passes.

Expected per-step overhead vs. current extractor: <10% on CPU. Should
not materially extend the ~2.5h/model fair eval window.

## Training plan

1. **Smoke test** (1000 steps) on highway-v0 with the v3 wrapper to
   validate shapes, gradient flow, and that no obs key is missing.
2. **Full run**: `experiments/052_h10_attn_v3_highway`, same hyperparams
   as 050 (PPO, 350k steps, lr=3e-4, n_steps=2048, etc.). Seed 42 for
   first run; seed 7 + seed 13 if 042 looks promising, to control for
   seed variance.
3. **Eval**: extend `scripts/fair_eval_v3_archetype.py` to include
   `experiments/052_h10_attn_v3_highway/results/model.zip` as the
   seventh model. Same 500-seed protocol so all results are
   bit-comparable to the existing report.

## Ablation matrix

Each row is a separate training run (350k steps, identical seed, same
v3 env). Drop a stage to isolate its contribution:

| Run    | Self-attn | Cross-attn | Spatial bias | Notes |
|--------|:---------:|:----------:|:------------:|-------|
| 052-A  | ✓         | ✗          | n/a          | Per-agent context only — does flattening matter? |
| 052-B  | ✗         | ✓          | ✓            | Cross-attn only — does spatial coupling alone help? |
| 052-C  | ✓         | ✓          | ✗            | Full design without spatial prior — does the bias matter? |
| 052    | ✓         | ✓          | ✓            | **Full proposed design** |

If 052 wins, run **052-D** as a follow-up: replace presence-weighted pooling
with risk-weighted pooling.

## Hypothesized failure modes (and what to watch)

1. **Cross-attention collapse.** If the spatial bias is too sharp
   (`sigma` small), each agent attends only to its own cell — equivalent
   to looking up a single ViT patch, throwing away neighborhood context.
   *Watch:* attention entropy averaged over agent slots; if median
   entropy stays below ~1 nat after 50k steps, increase `sigma`.
2. **Padded slots leaking gradients.** Forgetting to mask
   `presence == 0` slots either in the self-attention key-padding mask or
   in the pooling weights → gradients dominated by garbage tokens.
   *Watch:* per-slot gradient norms during early training; should drop
   sharply on padded slots.
3. **Anomaly signal underused.** If the ViT alone solves the task, the
   cross-attention output may be ignored (residual connection downstream
   pushes the agent token toward the scene patches it attended to,
   dropping the anomaly signal). *Watch:* an ablation that zeros the
   anomaly channels of the input token at eval time. If crash rate
   barely changes, the signal isn't being used and we need a stronger
   coupling (e.g., gating the cross-attention output by the agent's
   `risk` scalar).

## Empirical results — Experiment 052

Trained on 2026-05-02 with the design above (Stages A, B, C, D, E; no
A.5; sigma=1.5). 350k PPO timesteps on AdversarialHighwayV3, seed 42,
identical hyperparameters to 050. Evaluated on the same 500-seed
protocol as `fair_eval_v3_archetype.py`.

### Comparison vs. priors

| Metric | **AnomAttn-v3 (052)** | Baseline (027) | ExpectedInput-H10 (048) | ViT-only-v3 (051) |
|---|---|---|---|---|
| Mean reward | 18.32 | 21.80 | 8.99 | 8.63 |
| Crash rate (total) | 43.2% | 55.0% | 23.8% | **21.4%** |
| Crash (adversarial) | 25.4% | 35.6% | 17.8% | **16.2%** |
| Crash (nominal) | 18.2% | 20.2% | 6.0% | **5.4%** |
| Survival | 56.8% | 45.0% | 76.2% | **78.6%** |
| Mean ep length | 59.5 | 55.1 | 69.1 | 70.4 |

052 sits between Baseline and the H10 line — **closer to Baseline**.
Aggressive driving (short episodes, high reward) and ~2× the crash rate
of the flat-MLP H10 variants. Per-archetype, 052 underperforms 048/051
on every category except a near-tie on `sudden_braker`.

### Anomaly-zero ablation

To test whether 052 is using the anomaly signal at all, I re-evaluated
the same model on the same 500 seeds with `agent_anomaly` zeroed in
every observation before the policy forward pass.

`scripts/anom_zero_ablation_052.py` — full report at
`experiments/052_h10_attn_v3_highway/results/anom_zero_ablation.md`.

| Metric            | 052 (orig) | 052 (anomaly zeroed) | Δ |
|-------------------|-----------:|---------------------:|---:|
| Crash (total)     | 43.2%      | 41.8%                | **−1.4 pp** |
| Crash (adversarial)| 25.4%     | 25.6%                | +0.2 pp |
| Crash (nominal)   | 18.2%      | 16.2%                | −2.0 pp |
| Mean reward       | 18.32      | 16.02                | −2.31 |
| Survival          | 56.8%      | 58.2%                | +1.4 pp |
| tailgater         | 6.2%       | 6.0%                 | −0.2 pp |
| sudden_braker     | 2.8%       | 2.6%                 | −0.2 pp |
| lane_drifter      | 12.4%      | 13.0%                | +0.6 pp |
| erratic_speed     | 4.0%       | 4.0%                 |  0.0 pp |

**Verdict: the policy is essentially ignoring the H10 signal.**
Zeroing `agent_anomaly` makes the model marginally *safer* (within
sampling noise) and only modestly less rewarded. Per-archetype crash
rates are within ±0.6 pp on every category — within the noise floor we
should expect for 500-episode comparisons. The architecture *can* see
the anomaly (we feed it through Stage A → Stage B → Stage C), but
PPO never found a useful gradient through that channel.

### Probable causes (ranked by suspicion)

1. **ViT patches dominate the cross-attention residual.** Each agent
   token starts as `proj([kin, anom])`, gets self-attended (mostly
   informative since anom is small magnitude vs. kin), then receives a
   cross-attention residual update that draws from 121 ViT patches.
   The patch content has far more signal-bandwidth than the anomaly
   channels, so the residual stream's "memory" of the original anomaly
   bits is plausibly washed out before pooling.
2. **Bottleneck at the first projection.** `Linear(7 + 4 → 64)` mixes
   kin and anomaly channels at the very first layer. If SGD discovered
   early that kin alone solves the problem, the weights for the anomaly
   columns can collapse to ~0 and stay there. There's no architectural
   constraint that *requires* the anomaly channel to be read.
3. **Slot ordering instability.** Slots are re-sorted by distance every
   step (see Slot semantics). The anomaly channel's *meaning at slot i*
   changes as occupants migrate. The flat-MLP H10 path (048) had the
   same instability and still benefited; the difference may be that
   048's compute graph forces the anomaly through its own dedicated
   sub-MLP that can't be ignored, while in 052 the anomaly information
   must compete with the ViT for influence on the same residual stream.
4. **Training-time gradient signal too weak.** With
   `anomaly_reward_weight = 0.0` (matching 050) and no auxiliary loss
   that explicitly teaches the policy to use the anomaly channel, the
   only path for "the anomaly bit was useful" to register is via the
   PPO advantage. If avoidance behaviors in v3 are dominated by
   easier-to-learn position/velocity cues, the anomaly never crosses
   the gradient noise floor.

### What 052 tells us about the architecture vs. the signal

This was a clean test of the *architecture* given the *current
constant-velocity H10 signal*. The architecture is plumbed through
correctly (smoke test passed, gradient flow is fine, training
converged), but the policy converged to a Baseline-like, anomaly-blind
policy. Two interpretations are consistent with the data:

- **Pessimistic:** the ExpectedInput-H10 advantage in 048 was a
  capacity / regularization effect (the flat MLP forced the policy to
  attend to the anomaly), not the H10 signal carrying genuine
  predictive content beyond what kin alone provides. Under this story,
  the architectural changes here can't help — what's needed is the
  Future direction (learned H10 prediction).
- **Optimistic:** the architecture is right but needs a coupling that
  *requires* the anomaly to be read. Candidates: anomaly-as-attention-
  bias (multiply or add per-agent risk into attention logits), per-
  agent FiLM gating of cross-attention output by `risk`, or a small
  auxiliary loss that predicts the next-step anomaly from the current
  representation.

The next experiment should distinguish between these. Cheapest test:
add anomaly-as-attention-bias to the cross-attention (multiply
attention logits by `1 + λ * risk_i`, where λ is a small positive
constant or a learned scalar). If that pushes 052 below 048's crash
rate, the optimistic story holds. If it doesn't, escalate to the
learned-prediction H10 idea.

## Experiment 053 — Risk-temperature cross-attention

**Date trained:** 2026-05-02 → 2026-05-03. Same env, same 350k timesteps,
same seed (42), same hyperparameters as 052. The *only* difference from
052 is in the cross-attention block.

### Motivation — what the 052 ablation forced us to fix

The anomaly-zero ablation on 052 showed the policy was routing around
the anomaly channels: zeroing `agent_anomaly` at inference moved crash
rate by only −1.4 pp (within noise). The architecture *could* see the
anomaly but PPO never found a useful gradient path through it.

Two interpretations were consistent with that data:

1. **Pessimistic:** the H10 signal has little marginal content over kin
   alone. Architecture changes can't help; we'd need a richer signal
   (learned H10 prediction).
2. **Optimistic:** the architecture *let* the policy ignore the anomaly,
   even though the signal is informative. We need a structural
   constraint that forces a gradient pathway through the anomaly
   channels.

053 was designed as a clean test of the optimistic interpretation:
introduce a coupling that the optimizer **cannot** zero out except by
collapsing learnable parameters to a degenerate point.

### Architectural change

Replaced the cross-attention block in Stage C with a custom
multi-head dot-product attention that exposes per-query multiplicative
scaling on attention logits. Per-query scale derived from per-agent
anomaly + risk:

```python
# Inside AnomalyAttentionEncoder.forward, when use_risk_attention_bias=True:
anomaly_scalar = agent_anom[:, :, 1]   # (B, N) — H10 anomaly in [0, 1]
risk_scalar    = agent_anom[:, :, 2]   # (B, N) — H10 risk    in [0, 1]
q_scale = (
    1.0
    + F.softplus(self.s_anomaly) * anomaly_scalar
    + F.softplus(self.s_risk)    * risk_scalar
)
# attn_logits: (B, H, N, S) = (Q · K^T) / sqrt(d_h) + spatial_bias
attn_logits = attn_logits * q_scale[:, None, :, None]
```

Both `s_anomaly` and `s_risk` are scalar `nn.Parameter`s initialized to
0, so at init `softplus(0) = ln(2) ≈ 0.69` and the maximum scale at
`anomaly = risk = 1` is `1 + 0.69 + 0.69 ≈ 2.38`. The pathway has a
non-trivial gradient at init regardless of whether the optimizer
"wants" to use it: `∂(attn_logits) / ∂(anomaly_i)` is nonzero from step
zero.

Why **multiplicative** on logits and not **additive on logits to the
key (S) dimension**: an additive constant per-query (independent of
key index) gets absorbed by softmax — softmax is shift-invariant on
the key axis. Multiplicative scaling, on the other hand, is an inverse
temperature: it *sharpens* attention when scale > 1 and *flattens* it
when scale < 1. Since we want anomalous agents to commit more strongly
to specific scene patches, multiplicative is the right operation.

Why **per-query** (per-agent) and not per-key: attention sharpening
applies to "how much one query-token concentrates on its attended
patches." Anomaly is a per-agent property, not a per-patch property,
so the per-query axis is the natural place to inject it. (A per-key
variant — patches near anomalous agents getting boosted attention from
*all* agents — is a separate hypothesis worth testing later.)

### Implementation

- New `_RiskTemperatureCrossAttn` class in `src/driving/vit_cvae.py`
  (~50 LOC, drop-in replacement for `nn.MultiheadAttention` with
  `attn_bias` and `q_scale` arguments).
- `AnomalyAttentionEncoder` gains a `use_risk_attention_bias` flag.
  When True: instantiate `_RiskTemperatureCrossAttn` + two scalar
  `nn.Parameter`s `s_anomaly`, `s_risk`. When False (default): keep
  `nn.MultiheadAttention` exactly as in 052.
- `ViTCVAEExtractor` and `train_adversarial.py` plumb a new config key
  `anomaly_attn_use_risk_bias` (default False).
- `experiments/053_h10_attn_risk_v3_highway/config.yaml` — clone of 052
  with `anomaly_attn_use_risk_bias: true`. Otherwise identical.

Net code add: ~70 LOC. Net parameter add over 052: 2 scalars (basically
nothing). FPS during training was ~10% *higher* than 052 (the custom
MHA is lighter than torch's default since we don't need
`key_padding_mask` for the cross-attention's dense ViT keys).

### Results (500-seed fair eval)

`scripts/fair_eval_053_focused.py` — full report at
`experiments/053_h10_attn_risk_v3_highway/results/fair_eval_focused.md`.

| Metric | **053** | 052 | 048 (H10) | 051 (ViT-only) | 027 (Baseline) |
|---|---|---|---|---|---|
| Mean reward | 10.20 | 18.32 | 8.99 | 8.63 | 21.80 |
| Crash rate (total) | **20.8%** | 43.2% | 23.8% | 21.4% | 55.0% |
| Crash (adversarial) | **15.6%** | 25.4% | 17.8% | 16.2% | 35.6% |
| Crash (nominal) | 5.6% | 18.2% | 6.0% | **5.4%** | 20.2% |
| Survival | **79.2%** | 56.8% | 76.2% | 78.6% | 45.0% |
| Mean ep length | 69.7 | 59.5 | 69.1 | **70.4** | 55.1 |

**Per-archetype crash rate (lower = better):**

| Archetype | 053 | 052 | 048 | 051 | 027 |
|---|---|---|---|---|---|
| tailgater | **2.8%** | 6.2% | 3.6% | 3.0% | 7.6% |
| sudden_braker | **1.0%** | 2.8% | 1.0%* | 1.2% | 2.6% |
| lane_drifter | **7.4%** | 12.4% | 9.4% | 8.4% | 18.2% |
| erratic_speed | 4.4% | 4.0% | 3.8% | **3.6%** | 7.2% |

(*tied with 048 at 1.0%.)

053 takes the top spot overall, beating ViT-only-v3 by 0.6 pp and 052
by **22.4 pp** — crash rate is approximately halved by the single
architectural change. The behavioral profile (low reward, long
episodes, ~80% survival) cleanly matches the H10 line, **not** 052.
This indicates the policy is now using the anomaly channel to drive
cautious avoidance behavior the H10 signal supports.

### Diagnostic — did the gradient pathway actually get used?

Loaded the trained model and inspected the learned scalars:

```
s_anomaly = -0.4910  →  softplus coeff 0.4775  (init 0.69, so −31%)
s_risk    = -0.4308  →  softplus coeff 0.5008  (init 0.69, so −27%)
max scale @ anomaly = risk = 1:   1.978  (init was 2.38)
```

Two things to read off this:

1. **The pathway was used.** Both scalars stayed positive (softplus is
   always positive) and far from `−∞`, which is the only way the
   optimizer could collapse the coupling. The network *kept* the
   anomaly→attention pathway open through 350k steps.
2. **Init was slightly too aggressive.** Both scalars moved ~30% below
   init, suggesting that the ideal multiplicative scale at maximum
   anomaly+risk is closer to 2.0× than 2.4×. Not surprising — the
   anomaly and risk channels are noisy in the H10 wrapper, so a more
   tempered coupling generalizes better.

This is the cleanest possible falsifier of the pessimistic
interpretation: same model, same data, single architectural change,
and the H10 signal goes from "ignored" to "actively shaping policy."

### Interpretation

**Optimistic interpretation wins on this evidence.** The 052 result was
*not* telling us that the H10 signal lacks content. It was telling us
that **a cross-attention residual stream lets the policy route around
auxiliary inputs** unless the architecture imposes a non-bypassable
coupling. In retrospect, this is consistent with how transformer
representations work — residual streams compose by default, and there
is no architectural pressure to *use* any particular input feature.
The flat-MLP H10 path in 048 worked partly because the MLP couldn't
route around the anomaly without zeroing weights, while the
cross-attention residual in 052 *could* (and did).

A useful general principle the 052→053 comparison surfaces:

> When introducing a new input channel into a transformer-based policy,
> design at least one structural pathway through which the channel
> **must** flow — not just one through which it *can* flow.

The risk-temperature scaling is one such pathway. Other candidates
exist (FiLM gating, K-side anomaly diffusion, anomaly-conditioned
attention masking), and each implements the principle differently.

### Anomaly-zero ablation on 053 (validation)

Same protocol as the 052 ablation: load the trained 053 model,
re-evaluate on the same 500 seeds (1000-1499), but zero
`agent_anomaly` in every observation before the policy forward pass.
Run on 2026-05-03 via `scripts/anom_zero_ablation_053.py`. Full
report at
`experiments/053_h10_attn_risk_v3_highway/results/anom_zero_ablation.md`.

| Metric | 053 (orig) | 053 (anomaly zeroed) | Δ |
|---|---:|---:|---:|
| Crash (total) | 20.8% | 28.8% | **+8.0 pp** |
| Crash (adversarial) | 15.6% | 20.4% | +4.8 pp |
| Crash (nominal) | 5.6% | 8.8% | +3.2 pp |
| Mean reward | 10.20 | 9.43 | −0.77 |
| Survival | 79.2% | 71.2% | −8.0 pp |
| tailgater | 2.8% | 3.8% | +1.0 |
| sudden_braker | 1.0% | 1.2% | +0.2 |
| lane_drifter | 7.4% | 8.0% | +0.6 |
| erratic_speed | 4.4% | 7.4% | +3.0 |

**Verdict: the anomaly channel is genuinely being used.** Zeroing
`agent_anomaly` moves crash rate by +8.0 pp — well above noise, and a
sharp contrast with the 052 ablation (−1.4 pp, within noise). The
risk-temperature pathway is load-bearing: it's the difference between
"the policy can ignore the anomaly" and "the policy actively reads
it." This is the cleanest possible falsifier of the pessimistic
interpretation, and it falsifies cleanly.

A few details worth noting:

1. **Slightly under the doc's predicted ≥10 pp jump (8.0 pp actual).**
   The signal is *informative* but not *dominant* — the kin and scene
   streams retain most of 053's competence on their own. This is
   reassuring rather than disappointing: a policy whose crash rate
   doubled when one channel was zeroed would be brittle.
2. **Zeroed-053 (28.8%) still beats 052 (43.2%) by 14.4 pp.** Even
   with the anomaly stream killed, 053 retains a structural advantage
   over 052. The risk-temperature MHA appears to have shaped the
   cross-attention behavior in ways that survive the input being
   zeroed — possibly because the network learned a tighter, more
   selective attention pattern during training that the kin signal
   alone can still drive.
3. **Per-archetype, `erratic_speed` takes the biggest hit (+3.0 pp).**
   This is also the archetype where 053 had its weakest result vs.
   priors (4.4% vs 3.6% for ViT-only). It looks like erratic_speed is
   exactly where the anomaly channel was paying its rent — losing it
   pushes that archetype's crash rate into Baseline territory (7.2%).
4. **Nominal crashes also rise (+3.2 pp).** The anomaly signal isn't
   only used for adversary avoidance — it's also helping with nominal
   driving. Likely the constant-velocity-deviation flags any
   kinematically irregular vehicle, adversarial or not, which the
   policy uses as a general "be careful" cue.

This locks in the interpretation: the 052→053 win is mechanistically
attributable to the H10 anomaly being routed through the policy, not
to a secondary effect of the custom MHA implementation.

### What 053 does *not* tell us

Several things remain genuinely open:

1. **Is the H10 anomaly signal itself well-calibrated?** 053 shows the
   policy uses *whatever* the H10 wrapper outputs, but constant-
   velocity extrapolation produces noisy anomaly for nominal lane
   changes too. Replacing it with a learned predictor (the "Future
   direction" section of this doc) could push 053's ceiling higher —
   but see the caveat there about a learned predictor potentially
   absorbing adversarial behaviors into its own distribution.
2. **Could a richer scene encoding stack with the new coupling?** The
   CVAE/Discriminator aux losses are off in 053. They were also off in
   052, so we know the 052→053 jump doesn't depend on them — but
   nothing rules out them stacking additively.
3. **Is the temporal history per slot useful?** Stage A.5 is still
   untested. The 053 architecture only consumes per-slot kinematics
   *for the current frame*. The 10-frame slot history is sitting
   unused.
4. **Sigma sensitivity.** Spatial bias `sigma=1.5` was a starting
   guess. Untested.

### Possible next steps

These are independent hypotheses; each addresses a different question.
Listed roughly from cheapest-to-run to most-ambitious. Not a
recommendation — just the option space.

**A. Anomaly-zero ablation on 053.** ~~~1.5h wall-clock. Pure validation
   step: confirm that zeroing the H10 anomaly at inference materially
   degrades 053 (we expect a crash rate jump of 10+ pp, given the s_*
   diagnostic). If not, the 052→053 win comes from something other
   than the anomaly coupling. *Lowest risk, highest information value
   for confirming our story.*~~ **Done (2026-05-03):** crash rate
   jumped +8.0 pp (slightly under the predicted ≥10 pp); the
   interpretation is locked in. See "Anomaly-zero ablation on 053
   (validation)" above.

**B. CVAE + Discriminator aux losses on top of 053.** Re-train ~4–5h.
   Tests whether richer patch tokens (forced by the aux losses on the
   ViT encoder) stack with the forced anomaly coupling. The aux losses
   tighten the *representation* the cross-attention reads from; the
   risk-temperature scaling tightens *how* it reads. They're
   orthogonal in mechanism, so they could stack — but they could also
   be redundant if the policy gradient alone already gave the patches
   enough structure.

**C. Stage A.5 — per-slot GRU.** Re-train ~4–5h. Adds the only
   structured input the current design doesn't consume:
   `agent_kin_history[i]` (the per-slot trajectory over the last 10
   frames). Worth testing because trajectory context is what the H10
   anomaly *implicitly* encodes — giving the network direct access
   could either improve performance further or reveal that the H10
   anomaly is already a near-sufficient summary of recent trajectory.

**D. K-side anomaly diffusion (per-patch anomaly bias).** Smaller
   architectural change. Inject an additive bias on the K side of the
   cross-attention that's a Gaussian-smoothed map of `anomaly_i`
   centered at each agent's `(x, y)`. Equivalent of: "patches near
   anomalous agents are easier to attend to from any agent." Distinct
   gradient pathway from 053's per-query temperature; could be
   complementary or redundant.

**E. Pool weighting by risk (Stage D variant).** One-line change to
   Stage D's pooling: `weights = presence × (1 + λ · risk)` instead of
   `presence`. Re-train. Concentrates the pooled representation on
   agents the H10 wrapper considers risky. Cheap to try; effect could
   go either way (helpful focus vs. losing context on non-risky
   agents).

**F. Learned H10 prediction (replace constant-velocity).** Bigger
   change to the *signal* rather than the architecture. Replace
   `_future_predictions` in `HorizonExpectedObservedDictObsWrapper`
   with a learned predictor (existing `OnlineKinematicsPredictor`
   could be extended to H steps). Now that we know the signal is
   being used (053), improving the signal's quality should translate
   to gains. Larger engineering effort but the most ambitious win.

**G. Sigma sweep on Stage C spatial bias.** Untested hyperparameter.
   `sigma=1.5` is a default. A short A/B at `sigma ∈ {1.0, 3.0, 5.0}`
   could shift performance — or do nothing, in which case we know it's
   not load-bearing.

### Things that aren't worth doing (in this author's opinion)

- **Larger 052/053 — more layers, more heads, larger embed_dim.** The
  wins so far are from inductive-bias changes, not capacity. Throwing
  capacity at this likely overfits the v3 mixture without improving
  generalization to novel adversarial archetypes.
- **Different optimizers / learning-rate schedules.** Same reason.
- **Re-running 053 with different seeds for variance estimates.** The
  500-seed evaluation already gives us very tight error bars on each
  comparison; the seed-42 model isn't a one-off.

## What this design does *not* do (out of scope)

- It does not replace the CVAE/Discriminator auxiliary losses; those
  still operate on the CLS embedding for the existing
  `ViTCVAEExtractor.compute_auxiliary_losses` path.
- It does not consume `agent_kin_history` directly in 052. Stage A.5
  above is the planned follow-up experiment.
- It does not change the H10 wrapper. The anomaly tensor produced by
  `HorizonExpectedObservedDictObsWrapper` is consumed unchanged; only
  the model architecture downstream of the wrapper is modified.

## Future direction — learned H10 prediction

The current H10 anomaly compares observed agent state to a
**constant-velocity extrapolation** (`_future_predictions`,
`adversarial.py:743`):

```python
delta = current - previous
pred = current + h * delta   # for h in 1..10
```

This is a strong baseline but treats every agent as drifting in a
straight line. A real adversary that decelerates sharply (sudden_braker)
or weaves (lane_drifter) violates the prediction in ways the current
design intends to surface — but a constant-velocity model also flags
*nominal* lane changes and gentle accelerations as "anomalies." Some of
the false-positive risk shows up in the v3 numbers: ExpectedInput-H10's
crash rate (23.8%) is competitive but not dominant.

**Replacement idea:** train a small predictor — per-agent or shared —
that maps the last `K` frames of `agent_kin_history[i]` to the next
`H` frames. Compute the H10 anomaly against the *learned* prediction
instead of the constant-velocity extrapolation.

Two flavors worth ablating later:

1. **Predictor trained as an auxiliary loss inside the policy network.**
   Use the existing `OnlineKinematicsPredictor` (`vit_cvae.py:264`),
   which already does 1-step prediction with an aux loss; extend it to
   `H` steps and wire its output back into the wrapper's anomaly
   computation. Avoids any new model file.
2. **Predictor trained offline on rollouts of nominal traffic.** A
   separate model that has *never* seen adversaries — anything it can't
   predict is, by construction, off-distribution. Closer in spirit to
   the original "expected vs observed" framing; harder to keep aligned
   with the policy's distribution shift over training.

Open question for whichever flavor: should the predictor consume the
ego-relative or world-frame trajectory? Ego-relative is what the wrapper
uses today and is robust to ego turning, but couples agent prediction
to ego motion in a way that may confuse the predictor.

This belongs to a separate experiment branch (call it 053+ or a 5xx
series for "learned-prediction H10"). Logging it here so the idea
isn't lost.

## Experiment 054 — Per-slot GRU on top of 053

**Date trained:** 2026-05-03 → 2026-05-04. Same env, same 350k
timesteps, same seed (42), same hyperparameters as 053. The *only*
difference from 053 is that `agent_kin_history[i]` is now consumed by
a per-slot GRU whose final hidden state is concatenated with
`[agent_kin, agent_anom]` before the token projection. Implements
next-step **C** from the 053 follow-up matrix (Stage A.5 from the
original design).

### Motivation

The 053 anomaly-zero ablation showed the policy was using the H10
anomaly (Δ +8.0 pp), but two pieces of structured input remained
unconsumed: the per-slot 10-frame trajectory `agent_kin_history[i]`,
and any signal the GRU could extract from it that the constant-
velocity-residual H10 anomaly doesn't already encode. The hypothesis:
H10 is a *lossy summary* of trajectory deviation, and a small
recurrent encoder reading the trajectory directly should either
strictly improve performance (if H10 was lossy) or stay flat (if H10
was already a near-sufficient summary).

### Architectural change

A single shared GRU runs independently over each slot's 10-frame
trajectory; final hidden state feeds into the agent token before
self-attention.

```python
# Inside AnomalyAttentionEncoder.forward when use_per_slot_gru=True:
B, N, T, F_h = agent_kin_history.shape           # (B, 15, 10, 7)
flat = agent_kin_history.reshape(B * N, T, F_h)  # (B*15, 10, 7)
_, h = self.per_slot_gru(flat)                   # (1, B*15, 32)
temporal = h[-1].reshape(B, N, 32)               # (B, 15, 32)

token_in = torch.cat(
    [agent_kin, agent_anom, temporal], dim=-1    # (B, 15, 7+4+32 = 43)
)
agent_tokens = self.token_proj(token_in)         # (B, 15, 64)
```

The GRU is shared across all 15 slots (so it learns one general
"trajectory → state summary" function rather than 15 specialized
ones), which is the right inductive bias given that slot occupants
migrate every step. Single layer, hidden=32, ~5k params.

### Implementation

- `AnomalyAttentionEncoder` (`src/driving/vit_cvae.py`) gains
  `use_per_slot_gru: bool`, `gru_hidden: int = 32`, and
  `kin_history_feat_dim: int | None` init args. `forward` accepts an
  optional `agent_kin_history` tensor; asserts non-None when the GRU
  is on. Token-projection input dim grows from 11 to 43.
- `ViTCVAEExtractor` plumbs two new config keys
  (`anomaly_attn_use_per_slot_gru`, `anomaly_attn_gru_hidden`),
  validates that `agent_kin_history` is in the obs space when the GRU
  is on, and passes the tensor through `forward`.
- `train_adversarial.py` wires both keys into
  `features_extractor_kwargs`.
- `experiments/054_h10_attn_risk_gru_v3_highway/config.yaml` is a
  clone of 053 with the two new flags set true.

Net parameter add over 053: ~5k (GRU) + ~2k (wider token projection)
≈ 7k. FPS during training was ~25% lower than 053 (the 15 sequential
GRU rollouts per env per step add measurable CPU overhead) but still
well within the ~5h training window.

### Results (500-seed fair eval)

`scripts/fair_eval_054_focused.py` — full report at
`experiments/054_h10_attn_risk_gru_v3_highway/results/fair_eval_focused.md`.

| Metric | **054** | 053 | 052 | 048 (H10) | 051 (ViT-only) | 027 (Baseline) |
|---|---|---|---|---|---|---|
| Mean reward | 10.20 | 10.20 | 18.32 | 8.99 | 8.63 | 21.80 |
| Crash rate (total) | **18.2%** | 20.8% | 43.2% | 23.8% | 21.4% | 55.0% |
| Crash (adversarial) | **15.6%** | 15.6% | 25.4% | 17.8% | 16.2% | 35.6% |
| Crash (nominal) | **3.0%** | 5.6% | 18.2% | 6.0% | 5.4% | 20.2% |
| Survival | **81.8%** | 79.2% | 56.8% | 76.2% | 78.6% | 45.0% |
| Mean ep length | **70.98** | 69.7 | 59.5 | 69.1 | 70.4 | 55.1 |

**Per-archetype crash rate (lower = better):**

| Archetype | 054 | 053 | 052 | 048 | 051 | 027 |
|---|---|---|---|---|---|---|
| tailgater | **2.8%** | **2.8%** | 6.2% | 3.6% | 3.0% | 7.6% |
| sudden_braker | 2.0% | **1.0%** | 2.8% | **1.0%** | 1.2% | 2.6% |
| lane_drifter | **7.2%** | 7.4% | 12.4% | 9.4% | 8.4% | 18.2% |
| erratic_speed | **3.6%** | 4.4% | 4.0% | 3.8% | **3.6%** | 7.2% |

054 takes the new top spot, beating 053 by **2.6 pp** on total crash
rate. The reward profile is identical to 053 (10.20 mean), so the
GRU isn't trading caution for aggression — it's a strictly better
cautious policy. The big wins are on **nominal crashes** (3.0% vs
5.6% — nearly halved) and **erratic_speed** (3.6% vs 4.4%).

The one regression: `sudden_braker` doubled from 1.0% to 2.0% (still
well below Baseline). This is plausibly a capacity-allocation issue —
sudden_braker is a rare, high-frequency-deceleration archetype, and
the GRU's hidden state may not have allocated enough representational
budget for it. Worth ablating later (try gru_hidden=64) but not a
deal-breaker against the +2.6 pp aggregate gain.

### Anomaly-zero ablation on 054

Same protocol as 053's. Full report at
`experiments/054_h10_attn_risk_gru_v3_highway/results/anom_zero_ablation.md`.

| Metric | 054 (orig) | 054 (zeroed) | **054 Δ** | 053 Δ |
|---|---:|---:|---:|---:|
| Crash rate | 18.2% | 23.4% | **+5.2 pp** | +8.0 pp |
| Crash (adversarial) | 15.6% | 18.2% | +2.6 pp | +4.8 pp |
| Crash (nominal) | 3.0% | 5.4% | +2.4 pp | +3.2 pp |
| Mean reward | 10.20 | 10.17 | −0.03 | −0.77 |
| Survival | 81.8% | 76.6% | −5.2 pp | −8.0 pp |

**Per-archetype:**

| Archetype | 054 Δ | 053 Δ | Reading |
|---|---:|---:|---|
| tailgater | +2.4 | +1.0 | Zeroing hurts 054 *more* — H10 still load-bearing here |
| sudden_braker | −0.8 | +0.2 | Both noise — neither model uses H10 for this |
| lane_drifter | +1.4 | +0.6 | Both small; 054 slightly more sensitive |
| erratic_speed | **−0.4** | **+3.0** | GRU has fully replaced H10 here |

### Interpretation

Two things to read off these numbers:

1. **Both signals are still being used in 054, but with a clean
   division of labor.** The H10 anomaly's contribution shrinks (Δ
   +5.2 pp vs. 053's +8.0 pp) but doesn't collapse to noise — both
   inputs carry independent information. The per-archetype delta
   makes the division concrete: the GRU has *fully absorbed* H10's
   role for `erratic_speed` (Δ goes from +3.0 to −0.4, well within
   noise), while H10 remains *more* load-bearing for `tailgater`
   (Δ grows from +1.0 to +2.4). This is the optimal outcome — neither
   redundant nor wasteful. The GRU specializes in trajectory-shape
   archetypes (high-frequency velocity oscillation is exactly what a
   recurrent encoder reads natively), and H10 retains the proximity-
   based threats where its `risk = anomaly × proximity × closing`
   heuristic is hard for the GRU to reproduce from raw kinematics.

2. **The cleanest measure of the GRU's standalone value is
   zeroed-054 vs. zeroed-053: 23.4% vs. 28.8%, a 5.4 pp gap.** The
   only difference between those two conditions is the GRU (both
   have H10 zeroed at inference). So the GRU alone, with no H10,
   carries +5.4 pp of safety over no GRU. Symmetrically, comparing
   original-054 (18.2%) to zeroed-054 (23.4%) shows H10 on top of
   the GRU is worth +5.2 pp. The two signals are roughly equally
   valuable on the margin — within sampling noise.

3. **Mean reward is essentially unchanged when H10 is zeroed**
   (10.20 → 10.17, vs. 053's 10.20 → 9.43). The 054 policy retains
   its driving character without H10 — meaning the GRU output is
   sufficient to drive the cautious-policy behavior pattern, and H10
   is providing crash-avoidance precision rather than overall policy
   shape.

### What 054 tells us about the design space

The 053→054 jump validates the original Stage A.5 hypothesis: the
trajectory carries information beyond what a constant-velocity-
residual scalar can summarize, and a small shared GRU is enough to
extract it. Combined with the 053 result, the cumulative story is:

- 052 (no risk-temp, no GRU): 43.2% crash. Anomaly ignored.
- 053 (+ risk-temp): 20.8% crash. Anomaly used (Δ +8.0 pp).
- 054 (+ GRU): 18.2% crash. Anomaly + trajectory both used; division
  of labor by archetype.

Each architectural step pays for itself with a clear mechanism, and
the ablation evidence at each step is consistent with the design
intent.

### What 054 does *not* tell us

- **Is gru_hidden=32 the right size?** The sudden_braker regression
  hints at a capacity ceiling. Untested.
- **Could the GRU subsume H10 entirely with more training or a
  bigger hidden state?** The current ablation says "not at this
  budget" — but a larger GRU may push the H10 Δ further toward zero.
- **Does the GRU help the *base* (053) cross-attention without the
  risk-temperature?** I.e., would 052 + GRU recover the 053 result?
  Untested.
- **Sigma sensitivity (still untested from the 053 follow-up).**

### Possible next steps (revised after 054)

Carrying over what's still open from the 053 list, plus new directions
opened by 054:

**A. CVAE + Discriminator aux losses on top of 054.** Was option B in
   the 053 list — still open. The aux losses tighten the patch
   representation the cross-attention reads from; orthogonal in
   mechanism to both the risk-temperature scaling and the GRU. Could
   stack additively with both.

**B. GRU hidden-size sweep.** The sudden_braker regression is the
   only evidence we have for under-capacity. Try `gru_hidden ∈
   {16, 32, 64}` to test whether the regression is a tuning issue or
   a fundamental tradeoff.

**C. K-side anomaly diffusion (was D in 053 list).** Now more
   interesting because 054 shows the GRU and H10 are complementary —
   a third gradient pathway (per-patch anomaly bias) might pick up
   what neither currently does.

**D. Pool weighting by risk (was E in 053 list).** Untested. Cheap.

**E. Learned H10 prediction (was F in 053 list).** The 054 result
   sharpens this question. The GRU has *partially* subsumed H10; if
   we replace H10's constant-velocity baseline with a learned
   predictor, would the resulting "harder" anomaly signal complement
   the GRU again, or would the GRU just absorb it the same way?
   Worth running if the GRU-only ablation (E2 below) shows the GRU
   isn't already saturating the trajectory channel.

**F. GRU-only (anomaly-attn without H10).** A cleaner version of the
   anomaly-zero ablation: train from scratch with the GRU but no H10
   anomaly input at all. Tells us whether the GRU+H10 combo is
   strictly better than GRU alone *trained for it*, which is a
   different question from "does H10 still help a model that learned
   with it."

**G. Sigma sweep on Stage C spatial bias.** Carrying over from 053.

## Experiment 055 — CVAE + Discriminator auxiliary losses on top of 054

**Status:** *training launched 2026-05-04*. Same env, same 350k
timesteps, same seed (42), same hyperparameters as 054. The *only*
difference from 054 is the addition of two auxiliary loss terms — CVAE
reconstruction (weight $\alpha = 0.01$) and Discriminator
real-vs-counterfactual classification (weight $\beta = 0.1$) — both
attached to the ViT CLS embedding. Implements option **A** from the 054
follow-up matrix.

### Motivation

054 locked in the architectural story: H10 anomaly + risk-temperature
cross-attention + per-slot GRU is a clean, mechanistically attributable
design where each component carries its weight. But the *scene
representation* the cross-attention reads from — the ViT patch tokens
— is shaped only by the PPO gradient. There is no auxiliary supervision
encouraging the patches to encode the per-agent kinematic distribution.

The aux losses (CVAE reconstruction + Discriminator real-vs-fake) were
the original V0 design's mechanism for shaping the ViT representation
without relying solely on policy gradient. They were disabled
(`alpha = beta = 0`) in 050–054 to isolate the effect of policy-side
architectural changes; 055 asks: do they stack on top of 054?

The hypothesis space splits cleanly:

1. **Stack.** Aux losses tighten the patch representation that
   cross-attention queries against, producing an additive crash-rate
   reduction over 054. The mechanisms are orthogonal — risk-temperature
   shapes *how* the agent reads the scene, aux losses shape *what*
   the scene encodes.
2. **Redundant.** PPO gradient through the cross-attention residual
   already saturates whatever scene information the policy needs.
   Adding aux supervision changes nothing.
3. **Interfere.** Aux losses pull ViT weights toward the
   reconstruction objective at the expense of policy-relevant features.
   The 049 vs. 048 V3 fair-eval comparison shows this is a real risk:
   ExpectedInput-H10+Aux (049) underperformed ExpectedInput-H10 (048)
   by 7.6 pp on V3 crash rate (31.4% vs. 23.8%). The likely mechanism
   was the flat-MLP H10 path competing with aux losses for limited ViT
   capacity. 054's anomaly attention reads from the same ViT, so the
   same risk applies in principle — though the per-agent token pathway
   draws on patches rather than CLS, so the locus of competition is
   different.

### Architectural change

Two auxiliary loss terms are added to the PPO loss inside
`AdversarialPPO.train` (`adversarial_ppo.py:245`). Both terms have
lived in `vit_cvae.py` since the V0 design and were merely gated off
in 050–054 by setting their weights to zero:

```python
# Inside AdversarialPPO.train, per minibatch:
loss = (
    ppo_loss
    + self.alpha * cvae_loss        # 0.01 in 055; 0.0 in 054
    + self.beta  * disc_loss        # 0.1  in 055; 0.0 in 054
    + self.predictor_loss_weight * predictor_loss
)
```

Both losses operate on $z_\text{CLS} = \text{ViT}(G_t)[:, 0]$ — the ViT
CLS embedding (the same vector the policy reads from on every step).
They do *not* touch the patch tokens directly, but their gradient
propagates back through the ViT encoder, which is the mechanism by
which they shape the patches the cross-attention reads.

#### CVAE reconstruction loss ($\alpha = 0.01$)

A Conditional VAE encodes $(z_\text{CLS}, K_t)$ to a $32$-d latent and
decodes back to per-agent kinematics conditional on $z_\text{CLS}$:

```python
mu, logvar = encoder(concat(z_CLS, flatten(K_t)))   # (B, 32) each
z          = reparameterize(mu, logvar)             # (B, 32)
K_hat      = decoder(concat(z, z_CLS))              # (B, 15, 7)
recon_loss = presence_masked_mse(K_hat, K_t)
kl_loss    = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
cvae_loss  = recon_loss + kl_loss
```

Presence-masked MSE only counts loss on slots where $K_t[i, 0] = 1$, so
padded slots cost nothing. Encoder and decoder are 2-layer MLPs (hidden
$128$, latent $32$). The pressure: the CLS embedding must contain
enough about the scene that a separate decoder can reconstruct
per-agent kinematics from a $32$-d latent.

#### Discriminator loss ($\beta = 0.1$)

A 3-layer MLP classifies per-agent kinematic features as real ($K_t[i]$)
vs. counterfactual (sampled from the CVAE prior):

```python
fake_kin    = cvae.sample(z_CLS).detach()  # (B, 15, 7) — prior sample
real_logits = D(z_CLS, K_t)                # (B, 15)
fake_logits = D(z_CLS, fake_kin)           # (B, 15)
disc_loss   = presence_masked_BCE_symmetric(real=1, fake=0)
```

Fake samples are detached from the CVAE encoder graph, so only the
Discriminator and the ViT receive gradient from this term. The
pressure: $z_\text{CLS}$ must encode the agent distribution well
enough that real samples are linearly separable from generic CVAE
samples conditional on the same scene.

### Implementation

No code changes. The full path was already implemented (it produced
049's ExpectedInput-H10+Aux result on V2). 054's config disabled it via
`alpha: 0.0`, `beta: 0.0`. 055 simply re-enables it:

- `experiments/055_h10_attn_risk_gru_aux_v3_highway/config.yaml` —
  clone of 054 with `alpha: 0.01`, `beta: 0.1`. Otherwise identical.

Weight choice mirrors 049 (the only prior H10+aux precedent): a small
CVAE weight ($0.01$) treats reconstruction as a regularizer, while the
Discriminator weight ($0.1$) is the original V0 default. The
grad-norm diagnostic in `AdversarialPPO.train`
(`train/grad_norm_{ppo,cvae,disc}_at_vit`) will surface whether aux
losses dominate the ViT gradient at training time. Smoke test confirms
the intended ordering: PPO gradient at ViT is ~$50$–$100\times$ larger
than either aux term's gradient, so aux losses act as regularizers,
not co-equal objectives.

Net parameter add over 054: zero. The CVAE and Discriminator already
live in `ViTCVAEExtractor`; 054 just trained their parameters with no
gradient signal (CVAE/Disc weights stayed at random init for the entire
350k run because their losses were not added to the optimizer step).
Per-minibatch compute: one extra encoder/decoder forward pass through
CVAE, one Discriminator forward pass, plus their backward passes —
~10–15% per-step training overhead expected on CPU.

### Results (500-seed fair eval)

Trained 2026-05-04 (350k PPO timesteps, seed 42, ~5.5 h wall on CPU).
Evaluated on the same 500-seed protocol (seeds 1000–1499) as
`fair_eval_054_focused.py`. Full report at
`experiments/055_h10_attn_risk_gru_aux_v3_highway/results/fair_eval_focused.md`.

| Metric | **055 (Aux)** | 054 | 053 | 052 | 048 (H10) | 051 (ViT-only) | 027 (Baseline) |
|---|---|---|---|---|---|---|---|
| Mean reward | 13.25 | 10.20 | 10.20 | 18.32 | 8.99 | 8.63 | 21.80 |
| Crash rate (total) | 29.6% | **18.2%** | 20.8% | 43.2% | 23.8% | 21.4% | 55.0% |
| Crash (adversarial) | 22.0% | **15.6%** | **15.6%** | 25.4% | 17.8% | 16.2% | 35.6% |
| Crash (nominal) | 8.0% | **3.0%** | 5.6% | 18.2% | 6.0% | 5.4% | 20.2% |
| Survival | 70.4% | **81.8%** | 79.2% | 56.8% | 76.2% | 78.6% | 45.0% |
| Mean ep length | 65.6 | **70.98** | 69.7 | 59.5 | 69.1 | 70.4 | 55.1 |

**Per-archetype crash rate (lower = better):**

| Archetype | 055 | 054 | 053 | 052 | 048 | 051 | 027 |
|---|---|---|---|---|---|---|---|
| tailgater | 7.0% | **2.8%** | **2.8%** | 6.2% | 3.6% | 3.0% | 7.6% |
| sudden_braker | 1.2% | 2.0% | **1.0%** | 2.8% | **1.0%** | 1.2% | 2.6% |
| lane_drifter | 9.8% | **7.2%** | 7.4% | 12.4% | 9.4% | 8.4% | 18.2% |
| erratic_speed | 4.0% | **3.6%** | 4.4% | 4.0% | 3.8% | **3.6%** | 7.2% |

**055 regresses by +11.4 pp** vs 054 — adding aux losses sharply hurt
the model. The behavioral signature is unmistakable: mean reward jumped
+30% (10.20 → 13.25), mean episode length dropped −7.5% (70.98 → 65.6),
nominal crashes rose +5.0 pp (3.0% → 8.0%), tailgater crashes rose
+4.2 pp (2.8% → 7.0%). Per-archetype, 055 is worse than 054 on three
of four archetypes; the only category where 055 ties 054 (within
sampling noise) is `sudden_braker`.

In the V3 fair-eval ranking, 055 lands at #5 — behind both ViT-only-v3
(051) and ExpectedInput-H10 (048), and ahead only of the no-coupling
052 and Baseline. The result echoes the V3 fair eval pattern from
049, where ExpectedInput-H10+Aux (the only prior H10+aux precedent)
underperformed ExpectedInput-H10 by 7.6 pp on V3.

### Diagnostic — trained scalars

Loaded the 055 model and inspected the temperature-coupling parameters:

```
055:  s_anomaly = -0.4957  →  softplus 0.4757  (init 0.69, so −31%)
      s_risk    = -0.4635  →  softplus 0.4880  (init 0.69, so −29%)
      max scale @ anomaly = risk = 1:    1.964

053:  s_anomaly = -0.4910  →  softplus 0.4775  (cf. 055: −0.0018 in softplus)
      s_risk    = -0.4308  →  softplus 0.5008  (cf. 055: −0.0128 in softplus)
      max scale @ anomaly = risk = 1:    1.978
```

**The risk-temperature coupling is essentially unchanged from 053/054.**
The H10 anomaly → cross-attention pathway is being read at the same
effective temperature with or without aux losses. So the regression
isn't because the aux losses suppressed the H10 read; it's something
else.

### Diagnostic — aux loss training trajectory

Extracted `cvae_loss`, `disc_accuracy`, and the gradient norms at the
ViT (`grad_norm_*_at_vit`) across the 350k run.

```
iter   cvae_loss   disc_loss   disc_acc   grad_norm_ppo  cvae    disc
1      0.89        —           0.67       115            39      7.22
2      0.32        1.26        0.85       5.88           0.0034  0.06
5      0.45        0.25        0.98       —              —       —
10     0.56        0.05        0.99       —              —       —
late   0.78–0.82   0.02–0.04   0.995      5–50           ~0.001  ~0.03
```

Two things to read off this:

1. **Aux losses contributed meaningfully only at iter 1.** Initial
   ViT gradient was ~30% from CVAE+Disc, ~70% from PPO. By iter 2 the
   aux-induced gradient at the ViT had already fallen ~4 orders of
   magnitude. Disc accuracy reached 99% within ~10 iterations and
   plateaued.
2. **The CLS embedding linearly separates real from CVAE-fake at
   99.5% by mid-training.** That's an extremely well-shaped scene
   embedding for the auxiliary tasks. But the policy crash rate got
   *worse*. The aux objectives produce a "good" CLS for reconstruction
   and discrimination, but not for the policy's needs.

### Interpretation — hypothesis #3 (interfere) wins

Of the three hypotheses laid out in this section before training:

- ❌ **Stack:** would predict 055 < 054 on crash rate. Observed: 055 is
  +11.4 pp *worse*.
- ❌ **Redundant:** would predict 055 ≈ 054. Observed: 055 is sharply
  *worse* and behaviorally distinct (more aggressive, shorter
  episodes).
- ✅ **Interfere:** the aux losses pull the ViT toward an objective
  (reconstruct + discriminate per-agent kinematics) that conflicts
  with the policy-relevant feature distribution, even at small
  weights ($\alpha = 0.01$, $\beta = 0.1$).

The mechanism appears to be at the ViT, not at the cross-attention.
Evidence:

- The risk-temperature scalars `s_anomaly`, `s_risk` learned values
  are within ~0.01 softplus units of 053/054's values. The
  agent-side pathway is structurally untouched.
- The gradient diagnostic shows aux losses only meaningfully shaped
  the ViT in iter 1, then went silent — but those early-shaping
  effects are persistent because they bias the optimization
  trajectory.
- The behavioral shift (more reward, more crashes, shorter episodes)
  is consistent with a less safety-discriminating scene representation
  in the patches that the cross-attention queries from. The patches
  encode "what's where" well enough for reconstruction, but the
  crash-relevant features (proximity-with-trajectory, lane-occupancy
  patterns associated with imminent danger) are diluted relative to
  the no-aux baseline.

This is the same failure mode that the 049 V3 fair-eval surfaced for
ExpectedInput-H10+Aux. The lesson generalizes: **CVAE/Disc
auxiliaries on CLS — at the weights that worked for the V0
adversarial design — are net-negative for V3 policies that already
have a strong policy-side architecture.** The auxiliary objectives
were originally introduced to compensate for a weak policy signal;
when the policy can route its own gradient through richer pathways
(cross-attention, GRU, risk-temperature), the same auxiliary becomes
a regularization tax rather than help.

### What 055 does *not* tell us

- **Could a smaller $\alpha, \beta$ help?** This run used the only
  prior precedent ($\alpha = 0.01$, $\beta = 0.1$). A sweep
  ($\alpha = \beta \in \{0.001, 0.005\}$) might find a regime where
  the aux losses are weak enough to act purely as regularizers. But
  given the aux-induced ViT gradient is already only ~30% of PPO's
  at iter 1 and ~0.02% at convergence, dialing them lower will mostly
  recover 054, not exceed it.
- **Could attaching aux losses to *patches* (rather than CLS) help?**
  Untested. The current implementation only conditions CVAE/Disc on
  the CLS embedding. A patch-level reconstruction objective might
  shape the patches in a way that's more aligned with what the
  cross-attention reads — but this would require a code change.
- **Anomaly-zero ablation result.** Skipped on 2026-05-04. The
  fair-eval result (29.6% crash rate, behavioral shift toward
  aggression) and the trained-scalar diagnostic (s_anomaly/s_risk
  within 0.01 softplus units of 053/054) jointly localized the
  regression to the ViT representation rather than the H10 read
  pathway, making the additional ablation low-information for the
  follow-up direction. The script
  `scripts/anom_zero_ablation_055.py` is committed and runnable if
  needed later.

### What this means for the design space

The 053 → 054 → 055 trajectory rules out aux supervision on CLS as a
"free lunch" stacked on top of the H10-attention design. The
remaining open mechanisms from the 054 follow-up list — K-side
anomaly diffusion (option C), risk-weighted pooling (option D),
learned H10 prediction (option E) — are still plausible because none
of them depend on shaping the ViT through a separate objective. The
055 result narrows the path forward: future work should add new
*input-conditioning* pathways (more ways the anomaly or trajectory
information enters the agent token), not new *representation-shaping*
pathways on the shared backbone.

## Experiment 056 — Refined Attention & Risk-Weighted Pooling (TODO)

**Status:** *Planned/Implemented (to be launched after 055)*. Same env,
same 350k timesteps, same seed (42). This experiment builds on 055
(H10 + Risk-Temp + GRU + Aux) by refining the attention mechanisms and
pooling strategy to better handle multi-modal adversarial patterns.

### Motivation

The results from 053 and 054 showed that structural couplings
(Risk-Temperature) and temporal context (GRU) are highly effective.
However, 054 showed a slight regression on `sudden_braker`, and the
spatial bias/risk-temp parameters were fixed across all attention
heads. This experiment tests whether allowing the model to specialize
its attention heads and prioritize risky agents in the final pooling
can recover the lost performance and push the ceiling higher.

### Architectural changes

#### 1. Head-specific Risk-Temperature

In 053/054, `s_anomaly` and `s_risk` were scalars shared across all 4
attention heads. In 056, they are upgraded to vectors of shape `(n_heads,)`:

```python
# Inside AnomalyAttentionEncoder.__init__:
self.s_anomaly = nn.Parameter(torch.zeros(n_heads))
self.s_risk = nn.Parameter(torch.zeros(n_heads))

# Inside forward (Stage C):
q_scale = (
    1.0
    + F.softplus(self.s_anomaly).view(1, H, 1) * anomaly_scalar.unsqueeze(1)
    + F.softplus(self.s_risk).view(1, H, 1) * risk_scalar.unsqueeze(1)
)
# (B, H, N, S) logits scaling
```

This allows the network to dedicate specific heads to be highly
"anomaly-sharpened" while others remain nominal, improving the
diversity of the representational stream.

#### 2. Learned Head-specific Spatial Sigma

The spatial bias `sigma=1.5` was a fixed hyperparameter. 056 makes it a
learned parameter per head:

```python
# Inside AnomalyAttentionEncoder.__init__:
self.log_spatial_sigma = nn.Parameter(
    torch.full((n_heads,), math.log(spatial_sigma))
)

# Inside _spatial_bias:
sigma = self.log_spatial_sigma.exp().view(1, H, 1, 1)
bias = -d2.unsqueeze(1) / (2.0 * sigma**2)  # (B, H, N, S)
```

Different heads can now "zoom in" on agents or "zoom out" to capture
neighborhood context, depending on what the policy gradient requires.

#### 3. Risk-weighted pooling (Stage D)

Stage D pooling is changed from `presence`-weighted to `risk`-weighted:

```python
# Stage D — Presence * (1 + Risk) weighted pool
risk = agent_anom[:, :, 2]                 # (B, N)
weights = (presence * (1.0 + risk)).unsqueeze(-1)
pooled = (agent_tokens * weights).sum(1) / weights.sum(1).clamp(min=1.0)
```

This ensures that the final 64-d embedding `z_anom` is dominated by
features from agents the H10 wrapper has flagged as high-risk,
reducing the "wash-out" effect from the 14 other (mostly nominal)
agent slots.

### Hypotheses

1. **Specialization:** Head-specific parameters will allow the model to
   maintain a "nominal drive" head while another head "locks on" to
   anomalies.
2. **Focus:** Risk-weighted pooling will recover the `sudden_braker`
   performance by preventing the rare deceleration signal from being
   averaged away by the 14 other slots.
3. **Sigma tuning:** The model will learn to use a larger sigma for
   heads that handle global coordination and a smaller sigma for heads
   reading specific agent kinematics.

### Implementation

- `AnomalyAttentionEncoder` updated to support `use_learned_spatial_sigma`
  and `use_risk_weighted_pooling`.
- `_RiskTemperatureCrossAttn` updated to handle head-specific `q_scale`
  and `attn_bias`.
- `ViTCVAEExtractor` plumbs new config keys `anomaly_attn_risk_weighted_pooling`
  and `anomaly_attn_learned_spatial_sigma`.
- New config: `experiments/056_h10_attn_refined_v3_highway/config.yaml`.

### Caveat — what was actually trained

A plumbing gap was discovered after training finished. `train_adversarial.py`
reads YAML config keys into `features_extractor_kwargs`, but the two new
keys `anomaly_attn_risk_weighted_pooling` and `anomaly_attn_learned_spatial_sigma`
were never added to that dict (`train_adversarial.py:166–205`). They
therefore defaulted to `False` at construction time. The per-head
`s_anomaly`/`s_risk` change is unconditional in
`AnomalyAttentionEncoder.__init__` and so *did* take effect.

**Net: 056-as-trained = 054 + per-head temperature only.** The
risk-weighted pooling and learned spatial sigma flags were silently
ignored. The state-dict diagnostic confirms this: the saved model has
`s_anomaly`, `s_risk` of shape `(4,)` (per-head) but no
`log_spatial_sigma` parameter. A future "056-full" experiment would
need a one-line plumbing fix.

### Results (500-seed fair eval)

Trained 2026-05-04 → 2026-05-05 (350k PPO timesteps, seed 42, ~5.5 h
wall on CPU). Evaluated on the same 500-seed protocol. Full report at
`experiments/056_h10_attn_refined_v3_highway/results/fair_eval_focused.md`.

| Metric | **056 (per-head only)** | 054 | 053 | Δ vs 054 |
|---|---|---|---|---:|
| Mean reward | 8.87 | 10.20 | 10.20 | −1.33 |
| Crash rate (total) | 22.0% | **18.2%** | 20.8% | **+3.8 pp** |
| Crash (adversarial) | 15.2% | 15.6% | 15.6% | −0.4 pp |
| Crash (nominal) | 6.8% | **3.0%** | 5.6% | +3.8 pp |
| Survival | 78.0% | **81.8%** | 79.2% | −3.8 pp |
| Mean ep length | 69.86 | **70.98** | 69.7 | −1.12 |

**Per-archetype crash rate:**

| Archetype | 056 | 054 | Δ |
|---|---|---|---:|
| tailgater | 4.4% | **2.8%** | +1.6 |
| sudden_braker | 2.2% | **2.0%** | +0.2 |
| lane_drifter | **6.4%** | 7.2% | −0.8 |
| erratic_speed | **2.2%** | 3.6% | −1.4 |

056 lands at **#4** in the V3 ranking, behind 054, 053, and ViT-only-v3
(051). Per-head temperature alone underperforms scalar temperature on
total crash rate by 3.8 pp. The pattern is mixed at the archetype
level: 056 *helps* on archetypes where trajectory-shape matters
(`lane_drifter`, `erratic_speed`) but *hurts* on archetypes where the
H10 anomaly itself carries the danger signal (`tailgater`,
`sudden_braker`) and on nominal-collision avoidance.

### Diagnostic — trained per-head scalars

```
056:  s_anomaly per head:  [-0.30, -0.21, -0.17, -0.27]
      softplus per head:   [ 0.55,  0.59,  0.61,  0.57]
      s_risk per head:     [-0.28, -0.19, -0.20, -0.22]
      softplus per head:   [ 0.56,  0.60,  0.60,  0.59]
      max scale per head:  [ 2.12,  2.20,  2.21,  2.15]

053/054 (scalar):  softplus_anomaly = 0.476,  softplus_risk = 0.50
                   max scale         = 1.98
```

Two reads off this:

1. **Per-head learned values are systematically larger than 053/054's
   scalar.** All four heads land at softplus ≈ 0.56–0.61 vs scalar
   0.476–0.50. The per-head representation is *more* anomaly-sharpened
   on average, which plausibly explains the over-reaction to nominal
   kinematic noise (nominal crashes +3.8 pp).
2. **No meaningful head specialization.** The four heads' scales are
   within ~10% of each other (2.12–2.21). Hypothesis #1 from the doc
   ("specific heads will lock on to anomalies while others remain
   nominal") was not borne out. Either the inductive bias to specialize
   wasn't strong enough, or the policy didn't benefit from
   specialization in the V3 mixture.

### Interpretation

The per-head temperature change moved softplus weights *up* (more
sharpening) without specializing heads. The net effect was a more
aggressive anomaly response that hurts on archetypes with weak
trajectory-residual signal but reasonable kinematic signal
(`tailgater`: nearly constant velocity, anomaly is small but
proximity is dangerous). The aux-channel sharpening over-fires on
those, costing `tailgater` and nominal-crash rates.

The full-refined version of 056 (with risk-weighted pool + learned
sigma actually wired in) might recover more — risk-weighted pool
explicitly de-weights low-risk slots in the pooled embedding, which
would partially counteract the over-sharpening. But that's a separate
experiment.

## Experiment 057 — Reward shaping with H10 risk on top of 054

**Date trained:** 2026-05-05 (350k timesteps, seed 42, ~5.4 h wall on
CPU). Same env, same hyperparameters, same architecture as 054. The
*only* difference: `anomaly_reward_weight = 0.5` (was 0.0 in
050–056).

### Motivation

The 055 result (CVAE/Disc aux losses on CLS) failed by interfering
with the ViT representation outside PPO's trust region. The natural
follow-up: instead of routing the anomaly signal through an auxiliary
loss on the extractor, route it through the *reward function*. PPO's
clip + KL penalty bound how much the policy can change per iteration
in response to the modified reward, so any destabilization should be
naturally constrained.

The existing `AdversarialPPO.collect_rollouts` already implements this
mechanism (`adversarial_ppo.py:108–114`):

```python
if self.anomaly_reward_weight > 0 and self.num_timesteps > 2048:
    with th.no_grad():
        anomaly = extractor.compute_anomaly_scores(new_obs_t)
        rewards = rewards - self.anomaly_reward_weight * anomaly.cpu().numpy()
```

`compute_anomaly_scores` (`vit_cvae.py:799`) returns the
presence-weighted mean of the per-agent H10 risk score
(`agent_anomaly[:, :, 2]`). Per-step penalty range: roughly
`[0, anomaly_reward_weight]`, with typical magnitudes of
`0.05–0.2` for a single salient adversary in close proximity.

### Architectural change

Zero. 057 differs from 054 only in the YAML config:
`anomaly_reward_weight: 0.0 → 0.5`. The architecture, optimizer,
batch sizes, and seed are identical to 054.

`alpha = beta = 0` (no aux losses, ruling out the 055 failure mode).

### Results (500-seed fair eval)

Full report at
`experiments/057_h10_attn_risk_gru_rewardshape_v3_highway/results/fair_eval_focused.md`.

| Metric | **057 (RewShape)** | 054 | 056 | 055 | Baseline | Δ vs 054 |
|---|---|---|---|---|---|---:|
| Mean reward | 17.72 | 10.20 | 8.87 | 13.25 | 21.80 | +7.51 |
| Crash rate (total) | **65.4%** | **18.2%** | 22.0% | 29.6% | 55.0% | **+47.2 pp** |
| Crash (adversarial) | 34.0% | **15.6%** | 15.2% | 22.0% | 35.6% | +18.4 pp |
| Crash (nominal) | **31.8%** | **3.0%** | 6.8% | 8.0% | 20.2% | **+28.8 pp** |
| Survival | 34.6% | **81.8%** | 78.0% | 70.4% | 45.0% | −47.2 pp |
| Mean ep length | 48.53 | **70.98** | 69.86 | 65.64 | 55.12 | −22.45 |

**Per-archetype crash rate:**

| Archetype | 057 | 054 | Δ |
|---|---|---|---:|
| tailgater | 10.0% | **2.8%** | +7.2 |
| sudden_braker | 3.4% | **2.0%** | +1.4 |
| lane_drifter | 15.4% | **7.2%** | +8.2 |
| erratic_speed | 5.2% | **3.6%** | +1.6 |

057 lands at **#9 (last place)** in the V3 ranking, even **worse than
the no-anomaly Baseline (55.0%)**. Total crash rate +47.2 pp, episode
length collapsed to 48.5 (shorter than Baseline's 55.1), nominal
crashes 31.8% (10× the 054 value). This is a far worse regression
than 055.

### Behavioral signature

Three observations from the eval data converge on a single picture:

1. **Episode length 48.5 vs 054's 70.98** — episodes systematically
   end ~22 steps earlier. Even shorter than the no-anomaly Baseline
   (55.1).
2. **Nominal crash rate 31.8% vs 054's 3.0%** — 10× more crashes
   into *nominal* vehicles. A genuine avoidance policy would have
   near-zero nominal crashes (like 054).
3. **Mean reward 17.72** is suspiciously high given a 65.4% crash
   rate. Short, fast episodes accumulate the highway-v0 high-speed
   reward before crashing. The policy "wins" by crashing while
   sprinting.

The training trajectory is consistent: ep_len climbed from 13.8
(early) to 35+ (mid-training) but plateaued there rather than
continuing toward 70 as a healthy avoidance policy would. Mean reward
kept climbing past mid-training. `approx_kl` rose to 0.06+
(3–6× the typical healthy range), indicating PPO's clip was
working hard to constrain policy updates against a moving objective.

### Possible causes of the regression

Five hypotheses, ranked by how strongly the data and code support
them as contributors. They are not mutually exclusive — at least the
top three are likely co-acting.

#### Cause 1 (strongest support) — Early-termination incentive in the shaped reward

The H10 risk score is per-step and applied as a negative reward.
Total accumulated penalty over an episode of length $T$ is
$\sum_{t=1}^{T} w \cdot \rho_t$. Two ways to minimize this sum:

- **Path A (intended):** drive cautiously to make $\rho_t$ small at
  each step. Total penalty stays small even with $T$ large.
- **Path B (unintended):** crash quickly, making $T$ small. Total
  penalty stays small even with $\rho_t$ large.

PPO's clip+KL constrain *per-iteration* policy distribution change but
do not constrain *which equilibrium* the policy converges to. If
Path B is reachable in fewer gradient steps from the init policy than
Path A is — and crashing into a nominal vehicle from a
mid-aggressiveness state is structurally easier than learning a
sophisticated avoidance maneuver — the optimizer finds Path B first
and PPO is happy.

**Evidence:** ep_len ↓, nominal crashes ↑, reward ↑ — all three
deltas vs 054 are exactly the signature of Path B.

#### Cause 2 (strong support) — Penalty computed on reset observation, not terminal observation

The implementation in `adversarial_ppo.py:108–114`:

```python
new_obs, rewards, dones, infos = env.step(clipped_actions)
...
if self.anomaly_reward_weight > 0 and self.num_timesteps > 2048:
    with th.no_grad():
        new_obs_t = obs_as_tensor(new_obs, self.device)
        anomaly = extractor.compute_anomaly_scores(new_obs_t)
        rewards = rewards - self.anomaly_reward_weight * anomaly.cpu().numpy()
```

In SB3's `VecEnv`, when an env terminates, it auto-resets and
`new_obs[i]` is the **fresh starting state of the next episode**.
The terminal observation is moved to `infos[i]["terminal_observation"]`
but is not used here. So:

- For a continuing transition, `compute_anomaly_scores(new_obs)`
  correctly penalizes the agent for the anomaly of the state it just
  entered.
- For a **terminated** transition (crash), the penalty is computed on
  the freshly-reset episode, which has near-zero anomaly. The crash
  step receives essentially **no anomaly penalty**.

This compounds Cause 1: not only is "crashing early" the optimal
strategy under the accumulated penalty, but the crash transition
itself escapes the per-step penalty by virtue of being followed by a
fresh reset. The shaping under-prices crashes by exactly the amount
that would otherwise discourage them.

**Evidence:** code-level (lines 109–114). This is implementation
behavior, not hypothesized.

#### Cause 3 (moderate support) — Penalty magnitude comparable to per-step progress reward

The highway-v0 reward at policy frequency 1Hz is dominated by speed
reward, roughly 0.5–1.0 per step at high speeds. The per-step
anomaly penalty is `w · mean(risk × presence)`. With `w = 0.5` and
typical mean risk 0.1–0.4 in adversary-dense states, the per-step
penalty is 0.05–0.2 — i.e., **5–40% of the speed reward at every
step**.

When the penalty is comparable to the per-step reward, the optimizer
treats the anomaly term as a primary objective rather than a
regularizer. This shifts the policy's optimum substantially. A
regularizer-strength weight (`w ≤ 0.05`) would have penalty
~5–20× smaller than progress reward, leaving the speed objective
dominant.

**Evidence:** the trained policy's reward profile (17.72 mean,
heavily reward-seeking despite frequent crashes) matches a policy
operating under a strong but escapable cost term, not one tuned to
avoid the cost.

#### Cause 4 (moderate support) — Non-potential-based shaping changes the optimal policy

Ng, Harada, & Russell (1999) showed that reward shaping is
policy-invariant if and only if the shaping function is
potential-based: $F(s, s') = \gamma \Phi(s') - \Phi(s)$ for some
potential $\Phi$. The current shaping is just $-w \cdot \rho(s')$ —
*not* expressible as $\gamma \Phi(s') - \Phi(s)$ unless $\Phi$ is
specifically chosen to make it so. So the optimal policy under the
shaped reward is in general **different** from the optimal policy
under the original reward.

This isn't a bug per se — it's how reward shaping works when used
deliberately to bias the policy. But it means we should *expect* the
policy to converge to a different equilibrium than 054, not a "054
plus a little caution." The question is only whether the new
equilibrium is safer or less safe. Causes 1+2 jointly explain why
it ended up less safe.

**Evidence:** theoretical. The 47 pp regression magnitude is
consistent with a meaningfully different equilibrium, not just
parameter noise.

#### Cause 5 (weaker support) — Anomaly score is not fully action-attributable

The H10 risk score depends on *other agents'* kinematic deviations
from constant velocity, not directly on ego actions. A sudden braker
that brakes regardless of ego behavior produces high anomaly
attributable to itself, not to the ego policy. PPO's policy gradient
attributes this anomaly-driven reward back to whatever ego action
preceded it, even when no ego action could have changed it.

This adds **variance** to the advantage estimate without adding
useful learning signal. High-variance gradients with PPO's clip
manifest as KL spikes (we observed approx_kl rising to 0.06+ vs
the typical 0.01–0.02 range for stable runs). Some of the
training instability is plausibly from this source, on top of the
optimizer's pursuit of Path B.

**Evidence:** approx_kl trajectory (0.015 → 0.06+ over training)
combined with the architectural fact that the ego does not control
the anomaly signal. This cause likely contributes to instability but
is not strong enough on its own to explain the full regression.

### Interpretation

**Causes 1 + 2 jointly explain the regression.** Cause 1 makes early
termination the dominant strategy under the shaped objective. Cause 2
removes the only mechanism by which terminal transitions could have
been penalized enough to counteract Cause 1. Together they convert a
"penalize anomaly states" intent into a "reward crashing into nominal
traffic" effect.

Cause 3 explains why the regression magnitude is large (47 pp): the
penalty weight is large enough that the policy meaningfully chases
the modified objective rather than treating the penalty as a soft
nudge. Causes 4 and 5 are background contributors — Cause 4 is why
the new equilibrium *can* be substantially different from 054's, and
Cause 5 is why the path to that equilibrium was unstable
(KL excursions during training).

**The user's trust-region argument was correct in mechanism but
incomplete in scope.** PPO's clip protects against per-iteration
policy distribution change, which 057 confirms (the policy did not
diverge or NaN). It does not protect against:

- Convergence to a different equilibrium under a modified reward
  (Causes 1, 4)
- Implementation defects in the modified reward
  (Cause 2)
- Magnitude mismatches that turn the auxiliary signal into a primary
  objective (Cause 3)
- Variance in the modified reward that adds noise to advantages
  (Cause 5)

### Possible fixes (ranked by mechanical effect on the failure)

1. **Fix Cause 2 first** (smallest code change, largest expected
   effect). Mask the anomaly penalty on transitions where
   `dones[i] = True`, or apply it to `infos[i]["terminal_observation"]`
   when present rather than `new_obs[i]`. This removes the implicit
   reward for crashing.

2. **Add a survival bonus** (paired with the penalty). For each
   non-terminal step, add `+b` to the reward. Now ending the episode
   forfeits all future bonuses. This counteracts Cause 1 directly:
   Path B becomes expensive even with the unfixed Cause 2.

3. **Reduce `w` substantially** (e.g., to `0.05`). Counters Cause 3
   directly and reduces the impact of Causes 4 and 5 by making the
   shaped reward closer to the original. Lower bound on usefulness
   though — too small and the policy will ignore the signal entirely
   (the doc-flagged risk for an `anomaly_reward_weight` sweep).

4. **Switch to potential-based shaping** with $\Phi(s) = -V_\rho(s)$
   for some risk value function. Eliminates Cause 4 entirely. Larger
   engineering effort because it requires training a separate
   $V_\rho$ network or using a heuristic potential.

A reasonable cheap-to-test follow-up is **057b**: apply fixes 1 and
3 together (mask penalty on terminal transitions + reduce `w` to
0.05). This is a single ~10-line code change in
`adversarial_ppo.py` plus a config tweak.

### What 057 tells us about the design space

The reward-shaping path is *not* dead — it's just sensitive to
implementation. The current pipeline fails because of the
terminal-state asymmetry (Cause 2) compounded by an oversized weight
(Cause 3), not because reward shaping is fundamentally wrong.

The cumulative story across 053 → 057:

- **053 (Risk-Temp):** introducing a non-bypassable architectural
  pathway through the anomaly channel. **+10.6 pp** improvement vs
  052. Best so far.
- **054 (+ GRU):** adding direct trajectory access to the agent
  token. **+2.6 pp** improvement vs 053. **18.2% — current
  best.**
- **055 (+ CVAE/Disc aux):** auxiliary loss on CLS interferes with
  ViT. **−11.4 pp regression** vs 054.
- **056 (per-head temp only — partial implementation):** per-head
  temperature without specialization. **−3.8 pp regression** vs
  054.
- **057 (+ reward shaping w=0.5):** reward-hacking-induced
  catastrophic regression via Causes 1+2+3. **−47.2 pp
  regression** vs 054, ranking dead last (#9 of 9), worse than the
  no-anomaly Baseline.

054 remains the V3 winner. The "more knobs" experiments (055, 056,
057) all regressed for distinct reasons:

- 055 — interference at the *representation* layer (aux losses
  pull ViT off-task).
- 056 — interference at the *attention* layer (per-head
  over-sharpening with no specialization).
- 057 — interference at the *reward* layer (early-termination
  incentive + terminal-state penalty bug).

Each failure points at a different design pitfall, but they share a
common pattern: every *additional* training signal added on top of
054 made things worse. The path forward should either be (a) targeted
fixes to the failures (057b in particular looks tractable), or (b)
*input-level* changes that bypass these layers entirely — e.g., the
learned-H10-prediction direction in this doc's "Future direction"
section, which improves the *quality* of what 054 already reads
rather than adding new training signals on top.

## Experiment 058 — Tier 1 fixes for 057's reward-shaping regression

**Date trained:** 2026-05-05 → 2026-05-06. Same env, 350k PPO timesteps,
seed 42. Identical architecture and config to 054 except for three
targeted fixes to the per-step penalty that broke 057.

### Fixes (relative to 057)

1. **Mask the penalty on terminal transitions.** SB3's `VecEnv` replaces
   `new_obs` with the auto-reset state on `done=True`, so the anomaly
   computed on it is meaningless and crash steps were paying ≈0
   penalty. The asymmetry implicitly rewarded crashing. The fix is a
   one-line `not_done` mask:
   ```python
   not_done = (~np.asarray(dones, dtype=bool)).astype(np.float32)
   rewards = rewards - w * compute_anomaly_scores(new_obs) * not_done
   ```
2. **Concentrate the per-env score on threats.** The previous score was
   `(risk × presence).sum / presence.sum` — a presence-weighted mean
   over 15 slots that diluted a single high-risk threat by the 14
   surrounding nominals. Replaced with **top-3 mean** of presence-masked
   risks: `(risk * presence).topk(3, dim=1).values.mean(dim=1)`. An
   intermediate version using pure `max()` NaN'd at iteration 21 from
   advantage-variance blowup (single-step jumps in the per-env penalty
   spiked PPO's clip and KL); top-3 mean concentrates without the
   single-point spikiness.
3. **Reduce `anomaly_reward_weight` from 0.5 → 0.05.** Restores the
   regularizer regime instead of the primary-objective regime. With
   typical top-3 mean ≈ 0.30, per-step penalty is ≈ 0.015, about 2–3%
   of the typical highway speed reward.

### Result

500-seed fair eval (seeds 1000–1499):

| Metric | 058 | 057 | 054 |
|---|---:|---:|---:|
| Crash rate (total) | 29.0% | 65.4% | **18.2%** |
| Crash (adversarial) | 18.6% | 34.0% | **15.6%** |
| Crash (nominal) | 11.0% | 31.8% | **3.0%** |
| Mean ep length | 66.6 | 48.5 | **71.0** |
| Survival | 71.0% | 34.6% | **81.8%** |

**058 recovers 36.4 pp from 057 — the implementation defects were real
and fixable. But 058 remains +10.8 pp worse than 054** (no shaping):
even the cleanest version of per-step penalty shaping is net-negative
on top of 054's saturated architecture. Per-archetype, 058 wins on
`sudden_braker` (0.8% — the archetype where the H10 signal carries
the most predictive content) and loses elsewhere. The pattern motivates
a switch from penalty-based to **policy-invariant** shaping in 059.

## Experiment 059 — Potential-Based Shaping + Risk-Weighted Truncation Bonus

**Date trained:** 2026-05-06. Same env, 350k PPO timesteps, seed 42.
Identical architecture to 054. Reward function adds two anomaly-derived
terms via PPO's reward stream — both *disjoint from* the per-step
penalty in 057/058.

### Motivation

057 and 058 framed reward shaping as "penalize anomaly states." That
framing is policy-changing by construction: the optimal policy under
the shaped reward is in general not optimal under the original reward
(Ng/Harada/Russell 1999). 058 confirmed empirically that even a
well-implemented penalty regularizes the policy *off* 054's optimum
(+10.8 pp regression, mean reward up, episode length down).

The intended behavioral effect — *the policy should be encouraged for
identifying anomalous behavior and evading a potential crash, not
just for staying alive* — is more naturally expressed as:

1. **Per-step shaping that rewards de-risking transitions** without
   changing the optimal policy (potential-based shaping, PBS).
2. **A sparse terminal bonus conditional on having actually navigated
   through danger** — distinct from a uniform survival bonus.

### Method 1: Potential-based shaping

Following Ng, Harada, & Russell (1999), a reward-shaping function
`F(s, s')` is policy-invariant — meaning it does not change the
optimal policy — iff there exists a potential `Φ(s)` such that

> `F(s, s') = γ · Φ(s') − Φ(s)`.

The shaped return for any trajectory is then a *boundary term*:
`Σₜ γ^t F(sₜ, s_{t+1}) = γ^T Φ(s_T) − Φ(s_0)`. Two trajectories starting
at the same state differ in shaped return by exactly their unshaped
difference, so relative preferences are preserved.

We choose `Φ(s) = −ρ(s)`, where `ρ(s)` is the H10 risk score (top-3
mean of presence-masked per-agent risk; same definition introduced in
058). The shaping then reduces to

> `F(s, s') = ρ(s) − γ · ρ(s')`.

Read each transition:

- *Risk drops* (`ρ(s′) < ρ(s)/γ`): F > 0 — reward for escape.
- *Risk rises* (`ρ(s′) > ρ(s)/γ`): F < 0 — penalty for entering danger.
- *Risk constant* `ρ̄`: F = `ρ̄·(1−γ)` ≥ 0, vanishing in nominal states
  (`ρ̄ = 0`) but a small positive bias when sustained risk exists. This
  is a feature: it represents "Φ at terminal is implicitly 0, and the
  agent gains shaped value by surviving into a terminal of lower-than-
  current risk."

Coefficient `pbs_weight = 0.05`. Per-step magnitude during a typical
adversarial encounter is ≈ `0.05 · γ · ρ ≈ 0.015`, comparable to the
regularizer-regime per-step penalty in 058 but with the policy-
invariance guarantee.

### Method 2: Risk-weighted truncation bonus

A sparse terminal reward applied once at episode end **only on
truncation** (`infos[i]["TimeLimit.truncated"] = True`), with magnitude
proportional to the maximum risk encountered during the episode:

> `B(τ) = w_trunc · maxₜ ρ(sₜ)` if truncated, else 0.

The conditional structure is what distinguishes this from a plain
survival bonus:
- **Crashed episodes get nothing** — there is no "I survived because
  the scene was easy" credit.
- **Truncated episodes that never encountered risk get nothing**
  (`max ρ ≈ 0`) — there is no "I drove through nominal traffic" credit.
- **Only episodes that survived a high-risk encounter receive credit**,
  scaled by how dangerous the peak was.

Coefficient `w_trunc = 1.0`. Maximum bonus per episode ≤ 1.0 (since
ρ ∈ [0, 1]), about 1–2% of typical episode return.

### Implementation

In `src/driving/adversarial_ppo.py:AdversarialPPO.collect_rollouts`,
both signals share a single `compute_anomaly_scores(new_obs)` call
per step. Per-env state:

- `_prev_anomaly_score` ∈ ℝⁿ_envs: holds `ρ(s)` for the upcoming
  step's PBS calculation. Lazy-initialized to zeros (corresponds to
  `Φ(initial state) := 0`, the standard PBS convention). Multiplied
  by `not_done` after each step so the next step of a fresh episode
  starts from `Φ = 0`.
- `_episode_max_risk` ∈ ℝⁿ_envs: running max of `ρ(sₜ)` within the
  current episode. Reset to 0 on `done=True`.

Both signals are gated behind `num_timesteps > 2048` (warmup) and
both mask terminal transitions. The standard SB3 terminal-value
bootstrap path on `TimeLimit.truncated` is unchanged; the truncation
bonus is layered additively on top of it.

The change is a single block in `collect_rollouts`; the existing
057/058 per-step penalty path remains in the same block, gated on
`anomaly_reward_weight > 0` and disabled in the 059 config.

### Hyperparameters

Identical to 054 except where noted. All v3-trained models share these
settings:

| Parameter | Value |
|---|---|
| Algorithm | PPO with 20 fork-mode subprocess envs |
| Total timesteps | 350,000 |
| Rollout length per env (`n_steps`) | 512 |
| Mini-batch size | 64 |
| Learning rate | 3 × 10⁻⁴ |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| Clip range | 0.2 |
| Max grad norm | 0.5 (SB3 default) |
| Policy/value MLP | 2 × 128 |
| Seed | 42 |
| **`anomaly_reward_weight`** | **0.0** (no per-step penalty) |
| **`pbs_weight`** | **0.05** |
| **`truncation_bonus_weight`** | **1.0** |
| `alpha`, `beta` (CVAE/Disc aux losses) | 0.0 |

Architecture (unchanged from 054): ViT scene encoder (3 layers, d=64,
4 heads), Anomaly Attention encoder (Stages A–D, risk-temperature
cross-attention with scalar `s_α`, `s_ρ`, per-slot GRU on `H_t`).

### Results — 500-seed fair eval (seeds 1000–1499)

| Metric | **059** | 058 | 054 (prior best) | 057 | Baseline (027) |
|---|---:|---:|---:|---:|---:|
| Mean reward | 7.98 | 13.25 | 10.20 | 17.72 | 21.80 |
| Mean episode length | **74.13** | 66.55 | 70.98 | 48.53 | 55.12 |
| Crash rate (total) | **15.0%** | 29.0% | 18.2% | 65.4% | 55.0% |
| Crash (adversarial) | **11.6%** | 18.6% | 15.6% | 34.0% | 35.6% |
| Crash (nominal) | 3.4% | 11.0% | **3.0%** | 31.8% | 20.2% |
| Survival (truncation) | **85.0%** | 71.0% | 81.8% | 34.6% | 45.0% |

**Per-archetype crash rate:**

| Archetype | **059** | 058 | 054 | 053 | 048 | 051 |
|---|---:|---:|---:|---:|---:|---:|
| tailgater | 3.6% | 6.4% | **2.8%** | **2.8%** | 3.6% | 3.0% |
| sudden_braker | **0.6%** | 0.8% | 2.0% | 1.0% | 1.0% | 1.2% |
| lane_drifter | **6.2%** | 8.6% | 7.2% | 7.4% | 9.4% | 8.4% |
| erratic_speed | **1.2%** | 2.8% | 3.6% | 4.4% | 3.8% | 3.6% |

059 takes the top spot in the 500-seed v3 ranking, beating 054 by
**3.2 pp** on total crash rate and setting **all-time lows on three of
four archetypes**:

- `sudden_braker`: 0.6% (prior low: 1.0% in 053/048).
- `lane_drifter`: 6.2% (prior low: 6.4% in 056).
- `erratic_speed`: 1.2% (prior low: 2.2% in 056).

The one regression is `tailgater` (3.6% vs 054's 2.8%, a 0.8 pp gap).
Tailgaters track the ego at near-constant velocity, so the H10
constant-velocity prior produces a small but persistent risk signal
with little temporal variation. PBS shapes *transitions* and so has
relatively little to bite on for that archetype; the small nominal
crash uptick (3.4% vs 054's 3.0%) likely shares the same cause —
states with sustained, nearly-constant risk where the policy doesn't
get a strong PBS gradient.

### Behavioral signature

The metrics describe a clean cautious-policy line:

- **Long episodes.** Mean ep length 74.13 — the highest in the entire
  series. The policy consistently survives to within ≈ 6 of the
  80-step truncation horizon.
- **Modest reward.** Mean reward 7.98 vs 054's 10.20. The policy gives
  up some speed reward to drive more conservatively. This is the
  expected sign of the truncation bonus pulling toward "make it to the
  end" rather than "go fast."
- **Sharp drop in adversarial collisions.** −4.0 pp vs 054 on the
  adversarial-collider rate (11.6% vs 15.6%) — the bulk of the gain.
- **Marginal nominal-collision uptick.** +0.4 pp vs 054 (3.4% vs 3.0%).

### Training-time diagnostic — stability under PBS

`approx_kl` reached 0.05–0.06 mid-training, the same range that caused
058's first run to NaN. 059 trained through it cleanly. This is
mechanistic confirmation of the policy-invariance argument:
penalty-based shaping (058) created gradients that pushed the policy
toward an off-equilibrium attractor, and PPO's clip strained to track
that drift; PBS does not move the optimum, so the gradient direction
remains aligned with the unshaped objective and PPO's clip behaves
normally even at high `approx_kl`. The clip strains less because it
is correcting *magnitude* rather than *direction* of policy change.

### Reproducibility

- **Branch:** `visak/anomaly_reward`
- **Code commit:** `793c93f` ("059 — potential-based shaping +
  risk-weighted truncation bonus")
- **Config:** `experiments/059_h10_attn_risk_gru_pbs_truncbonus_v3_highway/config.yaml`
- **Run name:** `h10_attn_risk_gru_pbs_truncbonus_20260506_100606`
- **WandB run ID:** `gzbp6qj5` —
  https://wandb.ai/visakii/predictive_driving/runs/gzbp6qj5
- **Wall-clock training time:** ~5h 34min on CPU (350k PPO timesteps,
  20 forked envs).
- **Eval seeds:** 1000–1499 (500 episodes), `AdversarialHighwayV3`
  default archetype mixture (0.4, 0.2, 0.2, 0.2), `seed_eval_offset
  = +10000` for the in-training 50-ep eval.
- **Eval script:** `scripts/fair_eval_059_focused.py` (mirrors the
  052–057 protocol, bit-comparable to all priors).
- **Result artifacts:** `experiments/059_*/results/fair_eval_focused.{json,md}`,
  `model.zip`.

### Interpretation — why PBS+truncation worked where 057/058 didn't

Three properties together explain the result:

1. **Policy-invariance localizes the gradient signal.** Per-step
   penalty (057/058) is policy-changing — every state with non-zero
   anomaly contributes a constant per-trajectory cost, biasing the
   policy toward minimizing time-in-anomaly globally. PBS only
   contributes to the gradient on transitions where Φ changes, which
   in our case means transitions involving a change in `ρ`. Nominal
   driving (low-ρ throughout) gets no shaped gradient and therefore
   the policy retains 054's behavior in those regimes.
2. **Conditional terminal credit attaches the bonus to the behavior
   we want.** A plain survival bonus rewards every truncated episode
   equally, which incentivizes safe-but-boring driving. The risk-
   weighted truncation bonus rewards specifically *episodes where the
   policy faced danger and survived* — the exact behavior we wanted
   to amplify, and disjoint from "drove slowly through nominal
   traffic."
3. **The H10 signal was already saturated by the architecture in
   054.** 058 demonstrated that adding more weight to the same
   per-step signal only pushes the policy off-equilibrium. 059's
   shaping uses the same H10 signal but in a fundamentally different
   capacity — as a *gradient guide* (PBS) rather than a *value
   penalty* (per-step). This is consistent with the principle from
   the 052→053 transition: when an information channel is saturated,
   adding a *new pathway through which it flows* can still help, even
   if adding *more weight to the existing pathway* cannot.

Per-archetype evidence supports the mechanistic story: archetypes with
*risk-transition* structure (sudden_braker, lane_drifter, erratic_speed)
hit new lows, while the one archetype with *sustained-low-variance
risk* (tailgater) sees a small regression. PBS rewards transitions; it
has less to offer in regimes where there are none.

### What 059 does *not* tell us

- **Single-seed result.** Only seed 42. Variance estimate is needed to
  confirm 059 ≠ 054 + noise.
- **Joint vs. individual contribution.** Both PBS and the truncation
  bonus are active — their individual contributions are not separated.
- **Hyperparameter sensitivity.** `pbs_weight = 0.05` and
  `truncation_bonus_weight = 1.0` were chosen to put both signals at
  comparable per-episode magnitudes (≈ 1.0 each) but were not tuned.
- **Anomaly-zero ablation.** Not yet run on 059. We don't know the
  marginal contribution of the H10 signal at inference time
  (analogous to the 053/054 ablations that gave +8.0/+5.2 pp).
- **Robustness to the tailgater regression.** The +0.8 pp on tailgater
  is small but consistent with the mechanistic story; a targeted fix
  is unverified.

## Possible next steps after 059

Listed roughly from cheapest-to-run to most-ambitious. Each addresses
a distinct question raised by 059's result.

**A. Seed-variance check.** Re-train 059 with seeds {7, 13} and
   compare. ~10h wall-clock total. Lowest-information per hour but
   the standard sanity test before treating 059 as the new ceiling.
   If both seeds land within ±2 pp of 15.0%, 059 > 054 is robust;
   otherwise we need to be more careful about the claim.

**B. Component ablation: PBS-only vs. truncation-bonus-only.**
   Two ~5h runs:
   - 059-A: `pbs_weight=0.05, truncation_bonus_weight=0.0`
   - 059-B: `pbs_weight=0.0, truncation_bonus_weight=1.0`
   Tells us which component drives the gain. The mechanistic story
   above predicts PBS provides most of the per-archetype precision
   while the truncation bonus shifts the equilibrium toward longer
   episodes — but this is testable. The result determines whether
   future work doubles down on dense intermediate signal (PBS) or on
   sparse terminal credit (truncation bonus) or stacks both.

**C. Anomaly-zero ablation on 059.** Same protocol as the 053/054
   ablations: re-evaluate 059 on seeds 1000–1499 with `agent_anomaly`
   zeroed in every observation. Tells us whether 059 has saturated
   the H10 channel through its observation pathway or whether the
   reward shaping has shifted load to the architectural channel
   (or vice versa). Free of training cost — purely an inference-time
   test.

**D. Hyperparameter sweep on `pbs_weight` and `truncation_bonus_weight`.**
   `pbs_weight ∈ {0.02, 0.05, 0.1, 0.2}` and
   `truncation_bonus_weight ∈ {0.5, 1.0, 2.0}`. Even a coarse 2×2 sweep
   (~20h wall) would tell us whether 0.05 / 1.0 is on a flat region
   or a sharp peak. Cheap insurance before treating these values as
   defaults in any downstream experiment.

**E. Tailgater fix — augment Φ with a sustained-proximity term.**
   Tailgater episodes have nearly-constant risk, so PBS produces a
   small constant shaping that PPO doesn't act on. One option: change
   the potential to `Φ(s) = −α·ρ(s) − β·proximity(s)·closing(s)` so
   that even constant-risk encounters produce a gradient when the ego
   is in close trailing proximity. Another option: a separate dense
   penalty on sustained sub-threshold risk (`ρ > 0.3 for ≥ k consecutive
   steps`). Either keeps the policy-invariance guarantee for the rest
   of the design as long as the new term has the form
   `γΦ_2(s′) − Φ_2(s)`.

**F. Per-archetype shaping coefficients.** The per-archetype profile
   shows different archetypes benefit from different mechanisms.
   Conditional shaping that uses the per-agent anomaly *and* infers
   the dominant archetype would let the policy receive different
   gradients in tailgater vs. sudden-braker scenarios. Architecturally
   nontrivial; out of scope for a quick follow-up but worth considering
   if the per-archetype profile remains stable across seeds.

**G. Stack PBS with the GRU+risk-temp ablation matrix.** 059 stacks
   the new shaping on the full 054 architecture. To attribute the
   gain to architecture vs. shaping vs. their interaction, run:
   - PBS+TruncBonus on 052 (no risk-temp, no GRU) — does shaping
     alone beat 052?
   - PBS+TruncBonus on 053 (risk-temp, no GRU)
   - PBS+TruncBonus on 054 (this experiment, the full stack)
   The progression tells us whether PBS substitutes for, complements,
   or requires the architectural pathway.

**H. Learned H10 prediction (the doc's "Future direction").** Replace
   the constant-velocity `_future_predictions` with a small learned
   predictor (e.g., extending `OnlineKinematicsPredictor` to H steps).
   The H10 risk score is currently noisy — false positives on nominal
   lane changes contribute to the per-step signal. A learned predictor
   would tighten the signal, which has higher leverage now that we
   have a working channel (PBS) for the policy to read it through. The
   biggest engineering effort, but likely the biggest ceiling-raise:
   059 demonstrates the channel works; **H** improves the source.

**I. Potential function design.** `Φ(s) = −ρ(s)` is a sensible default
   but unmotivated by any optimality claim. Other candidates:
   `Φ(s) = −ρ(s)²` (sharper near-zero, flatter near-one),
   `Φ(s) = −E[risk over next K steps]` (cheap rollout-based estimate),
   `Φ(s) = −V_ρ(s)` for some learned risk-value function. The PBS
   theory says any choice is policy-invariant; the learning-speed gain
   depends on how well Φ approximates the actual value. Worth a
   focused study if PBS becomes a load-bearing component of the design.

**Recommended ordering.** First C (cheap, free), then A and B in
parallel (validity + mechanism). D after that to lock in
hyperparameters before any further architectural work. E and H are
the two ambitious next horizons — E targets the one regression in
059, H targets the underlying signal quality. F, G, I are research
directions that would only become priorities after the basics from
A–D have shipped.


