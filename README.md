# multi task ranking model

This note merges four parts into one mathematically detailed document:

1. Detailed ESMM gradient-flow derivation.
2. MMoE formulation and where conflict appears.
3. Entropy regularization math and its effect.
4. Conflict-aware gradient optimization focused on (y=1, z=0) samples.

It is aligned with the current implementation in:
- `train_rank.py`
- `src/models/ranking.py`

---

## 1. Problem Setup

We model user behavior chain: impression -> click -> conversion.

- x: feature vector
- y in {0,1}: click label
- z in {0,1}: conversion label
- p = P(y=1|x): CTR prediction
- q = P(z=1|y=1,x): CVR prediction
- t = y*z in {0,1}: CTCVR label
- p_ctcvr = p*q

In ESMM mode, the task losses are:

L_ctr = -[ y log p + (1-y) log(1-p) ]

L_ctcvr = -[ t log(pq) + (1-t) log(1-pq) ]

Total objective (without extra regularizer):

L_task = w_ctr * L_ctr + w_cvr * L_ctcvr

In this project, monitor metric uses:

combined_auc = ctr_auc * cvr_auc

which corresponds to the project-side CTCVR-style combined target.

---

## 2. Detailed Gradient Derivation in ESMM

### 2.1 General derivatives

For L_ctr:

dL_ctr/dp = -y/p + (1-y)/(1-p)

For L_ctcvr:

dL_ctcvr/dp = -t/p + (1-t) * q/(1-pq)

dL_ctcvr/dq = -t/q + (1-t) * p/(1-pq)

So:

dL/dp = w_ctr * [ -y/p + (1-y)/(1-p) ] + w_cvr * [ -t/p + (1-t) * q/(1-pq) ]

dL/dq = w_cvr * [ -t/q + (1-t) * p/(1-pq) ]

Below we focus on direction/sign with positive weights.

---

### 2.2 Case A: (y=0, z=0) => t=0

Loss terms:
- L_ctr = -log(1-p)
- L_ctcvr = -log(1-pq)

Derivatives:

dL/dp = 1/(1-p) + q/(1-pq) > 0

dL/dq = p/(1-pq) > 0

Interpretation:
- Gradient descent will decrease p and q.
- Direction is consistent; no conflict.

---

### 2.3 Case B: (y=1, z=1) => t=1

Loss terms:
- L_ctr = -log p
- L_ctcvr = -log(pq)

Derivatives:

dL/dp = -1/p - 1/p = -2/p < 0

dL/dq = -1/q < 0

Interpretation:
- Gradient descent increases both p and q.
- Direction is consistent; no conflict.

---

### 2.4 Case C: (y=1, z=0) => t=0 (critical)

Loss terms:
- L_ctr = -log p
- L_ctcvr = -log(1-pq)

Derivatives:

dL/dp = -1/p + q/(1-pq)

dL/dq = p/(1-pq) > 0

Key points:
- dL/dq is always positive: q is pushed down (reasonable for non-conversion).
- dL/dp has opposing terms:
  - -1/p from CTR (push p up)
  - +q/(1-pq) from CTCVR (push p down)

Set dL/dp = 0:

q/(1-pq) = 1/p  =>  p*q = 0.5

So we get the classic boundary:
- p*q < 0.5: net effect tends to increase p
- p*q > 0.5: net effect tends to decrease p

This is the conflict region that motivates targeted surgery.

---

### 2.5 Gradient table by sample type

- (0,0):
  - dL/dp: down signal
  - dL/dq: down signal
  - conflict: no

- (1,1):
  - dL/dp: up signal
  - dL/dq: up signal
  - conflict: no

- (1,0):
  - dL/dp: up vs down opposition (boundary near pq=0.5)
  - dL/dq: down signal
  - conflict: yes (mainly on p path)

---

### 2.6 Logit-space view (for optimization intuition)

If p = sigmoid(a), q = sigmoid(b):

dL/da = (dL/dp) * p(1-p)

dL/db = (dL/dq) * q(1-q)

So even when sign is fixed, magnitude is modulated by saturation terms p(1-p), q(1-q).
This is one reason sparse positive conversion signal can be weak in practice.

---

## 3. MMoE: Mathematical Form and Conflict Location

### 3.1 MMoE equations

Let experts be e_i(x), i=1..E.

Task-specific gate distributions:

g_ctr(x) = softmax(W_ctr x + b_ctr)

g_cvr(x) = softmax(W_cvr x + b_cvr)

Task representations:

h_ctr = sum_i g_ctr_i * e_i(x)

h_cvr = sum_i g_cvr_i * e_i(x)

Then task towers output logits for CTR/CVR.

### 3.2 Why MMoE helps

Compared with hard sharing, MMoE allows:
- different expert mixtures for CTR and CVR,
- partial separation of task features,
- reduced negative transfer.

### 3.3 Where conflict still exists

Experts are still shared parameters. For expert params theta_exp:

g_exp = w_ctr * grad_theta_exp(L_ctr) + w_cvr * grad_theta_exp(L_ctcvr) + lambda_ent * grad_theta_exp(L_ent)

So gradient interference still exists on shared experts; MMoE reduces but does not remove it.

In current code, asymmetric projection in MMoE is applied exactly on experts:
- `_get_asym_proj_params()` returns `self.experts.parameters()`.

---

## 4. Entropy Regularization: Objective and Effect

### 4.1 Gate entropy

For a gate vector g in simplex:

H(g) = - sum_i g_i log(g_i + eps)

Project implementation uses average gate entropy (MMoE):

H_avg = (H_ctr + H_cvr) / 2

Regularizer is implemented as:

L_ent = - E_batch[H_avg]

Total objective:

L_total = w_ctr L_ctr + w_cvr L_ctcvr + lambda_ent L_ent

Because L_ent = -H, minimizing L_total increases entropy.

### 4.2 Gradient intuition

For L_ent = sum_i g_i log g_i (up to batch average/sign convention):

dL_ent/dg_i = log(g_i) + 1

This penalizes overly peaked distributions and encourages broader expert usage,
which helps avoid early gate collapse and improves robustness.

---

## 5. Conflict-Aware Gradient Optimization Focused on (1,0)

In this project, `grad_scope on10` means conflict-aware optimization only on:

M_10 = {k | click_k = 1 and purchase_k = 0}

Code mask:

mask_10 = (ctr_labels > 0.5) & (cvr_labels < 0.5)

---

### 5.1 TorchJD conflict-aware optimization (UPGrad / PCGrad)

Enabled by:
- `--grad_surgery upgrad` or `--grad_surgery pcgrad`

For on10 mode, code effectively splits losses into two parts:

L_ctr = L_ctr_10 + L_ctr_other
L_ctcvr = L_ctcvr_10 + L_ctcvr_other

Then:
1. Apply aggregator A (UPGrad or PCGrad) on (L_ctr_10, L_ctcvr_10):

g_10 = A( grad(L_ctr_10), grad(L_ctcvr_10) )

2. Add normal gradients from non-(1,0) region:

g_other = grad(L_ctr_other + L_ctcvr_other)

3. Final update uses g_10 + g_other.

This concentrates conflict handling where theory predicts strongest interference.

---

### 5.2 Asymmetric projection conflict-aware optimization

Enabled by:
- `--grad_surgery asymmetric_projection`

On projection parameter set P (MMoE: experts), compute:

g_ctr = grad_P( w_ctr * L_ctr )

g_cvr_proj = grad_P( w_cvr * L_ctcvr_10 )   (for on10 mode)

g_cvr_other = grad_P( w_cvr * L_ctcvr_other )

Conflict statistics:

dot = <g_cvr_proj, g_ctr>
cos = dot / ( ||g_cvr_proj|| * ||g_ctr|| + eps )

Apply projection when:
- dot < 0,
- norms > 0,
- cos < -tau (`--asym_proj_tau`).

Projection rule:

coeff = dot / (||g_ctr||^2 + eps)

g_cvr_proj <- g_cvr_proj - lambda * coeff * g_ctr

where lambda is `--asym_proj_lambda`.

Optional norm restoration (`--asym_proj_restore_norm`):

scale = ||g_before|| / (||g_after|| + eps)
scale = clamp(scale, max=`--asym_proj_restore_max_scale`)

g_cvr_proj <- scale * g_cvr_proj

Then:

g_cvr = g_cvr_other + g_cvr_proj

g_final_on_P = g_ctr + g_cvr

Non-projection parameters keep normal backprop gradients.

---

### 5.3 Relation to stop-gradient idea

A strict stop-gradient variant on (1,0):

p_ctcvr = stop_grad(p) * q

would enforce:

dL_ctcvr/dp = 0 on (1,0)

This is an extreme form of decoupling.
Current project does not use this exact conditional detach by default in MMoE path,
but asymmetric projection and TorchJD-on10 serve a similar purpose: reduce harmful
CTR-CTCVR interference on the conflict-heavy subset.

---

## 6. Implementation Mapping (Current Project)

Key CLI switches:

- MMoE:
  - `--model mmoe --num_experts ... --expert_hidden_dims ...`

- Entropy regularization:
  - `--regularization entropy --lambda_entropy ...`

- Conflict-aware optimization method (CLI key remains `grad_surgery`):
  - `--grad_surgery upgrad|pcgrad|asymmetric_projection`

- Surgery scope:
  - `--grad_scope on10|all`

- Asymmetric projection controls:
  - `--asym_proj_lambda`
  - `--asym_proj_tau`
  - `--asym_proj_restore_norm`
  - `--asym_proj_restore_max_scale`

Validation/test in this project monitor:
- `val_ctr_auc`, `val_cvr_auc`, `val_combined_auc`
- `test_ctr_auc`, `test_cvr_auc`, `test_combined_auc`

and test is run on best checkpoint by `val_combined_auc` in `train_rank.py`.

---

## 7. Practical Interpretation

- ESMM conflict is theoretically concentrated at (1,0), especially on the p path.
- MMoE improves representational decoupling but shared experts still receive mixed gradients.
- Entropy regularization stabilizes gate usage and reduces expert collapse risk.
- on10 conflict-aware optimization is a targeted intervention that matches the analytical conflict region.

Together, these form a coherent optimization strategy for CTR/CVR/CTCVR training in this project.

---

## 8. Experimental Results (Current Snapshot)

### 8.1 Data source and scope

This snapshot is computed from completed 2-seed runs (seed=42, 2027) for all 8 settings:

- run-level table:
  - `experiments/eight_settings_dualnode/summaries/results_with_best_ckpt_ctcvr_latest_20260401_122144.tsv`
- setting-level summary:
  - `experiments/eight_settings_dualnode/summaries/results_with_best_ckpt_ctcvr_by_setting_latest_20260401_122144.tsv`
- ranking file:
  - `experiments/eight_settings_dualnode/summaries/results_rankings_latest_20260401_122144.txt`

Note:
- extra-seed jobs (`seed=3407`) are still running and are not included in this snapshot.

### 8.2 Setting-level mean +/- std (n=2 per setting)

| setting | test CTR (mean +/- std) | test CVR (mean +/- std) | test CTCVR (mean +/- std) |
|---|---:|---:|---:|
| 01_share_bottom | 0.618118852 +/- 0.000156238 | 0.650957227 +/- 0.000283143 | 0.402368918 +/- 0.000073314 |
| 02_mmoe | 0.614763618 +/- 0.003471804 | 0.643996090 +/- 0.009822281 | 0.395922408 +/- 0.008274206 |
| 03_mmoe_entropy | 0.617619574 +/- 0.000248414 | 0.658505976 +/- 0.005440062 | 0.406705514 +/- 0.003196312 |
| 04_mmoe_entropy_upgrad_all | 0.615951300 +/- 0.000458052 | 0.652493179 +/- 0.006262263 | 0.401905462 +/- 0.004156122 |
| 05_mmoe_entropy_upgrad_on10 | 0.616009057 +/- 0.000291150 | 0.641322434 +/- 0.006247090 | 0.395061329 +/- 0.004034992 |
| 06_mmoe_entropy_asym_all | 0.617426991 +/- 0.000282047 | 0.657656759 +/- 0.004514644 | 0.406054407 +/- 0.002601978 |
| 07_mmoe_entropy_asym_on10 | 0.617874473 +/- 0.000098666 | 0.655931354 +/- 0.000063895 | 0.405283242 +/- 0.000025246 |
| 08_mmoe_entropy_asym_on10_restore_norm | 0.617699623 +/- 0.000755440 | 0.656828731 +/- 0.004971179 | 0.405720979 +/- 0.002574498 |

### 8.3 Rankings (by mean AUC)

- By test CTCVR:
  1. `03_mmoe_entropy` (0.406705514)
  2. `06_mmoe_entropy_asym_all` (0.406054407)
  3. `08_mmoe_entropy_asym_on10_restore_norm` (0.405720979)

- By test CTR:
  1. `01_share_bottom` (0.618118852)
  2. `07_mmoe_entropy_asym_on10` (0.617874473)
  3. `08_mmoe_entropy_asym_on10_restore_norm` (0.617699623)

- By test CVR:
  1. `03_mmoe_entropy` (0.658505976)
  2. `06_mmoe_entropy_asym_all` (0.657656759)
  3. `08_mmoe_entropy_asym_on10_restore_norm` (0.656828731)

### 8.4 Quick interpretation

- If optimizing CTCVR first, current best is `03_mmoe_entropy`, with `06_mmoe_entropy_asym_all` very close.
- `01_share_bottom` remains strongest on CTR only, but lags in CTCVR.
- Asymmetric projection variants improve CTCVR/CVR while preserving strong CTR.
