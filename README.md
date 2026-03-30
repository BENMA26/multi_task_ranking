# Multi-Task Ranking: ESMM + MMoE + Entropy Regularization + Asymmetric Projection

This repository studies multi-task ranking for joint CTR/CVR prediction, with Ali-CCP style data as the default setup.

Our main project direction is:

- `ESMM` for entire-space supervision
- `MMoE` for soft task-specific representation routing
- `Entropy Regularization` to reduce expert/gate polarization
- `Asymmetric Projection` to mitigate harmful task-gradient interference

The project is research-oriented and focused on reproducible experiments rather than production deployment.

## 1. Motivation

In conversion modeling, user behavior is sequential: exposure -> click -> purchase.

Direct CVR training suffers from:

- sample selection bias (train on clicked samples, infer on all exposed samples)
- severe data sparsity for positive conversion labels

ESMM addresses train/serve mismatch via `pCTCVR = pCTR * pCVR`, but training can still be unstable because of coupled gradients between CTR and CVR signals.

Our method targets this with architecture-level and optimization-level improvements.

## 2. Proposed Method

### 2.1 ESMM Backbone

We train with:

- CTR loss on `click`
- CTCVR loss on `click * purchase`

and use:

- `pCTCVR = pCTR * pCVR`

This keeps supervision on the entire exposure space.

### 2.2 MMoE for Task Decoupling

`MMoE` provides shared experts plus task-specific gating, allowing CTR and CVR heads to select different expert mixtures.

This reduces hard parameter sharing pressure compared with plain shared-bottom designs.

### 2.3 Entropy Regularization for Gate Polarization

For `MOE / MMOE / PLE`, we add entropy regularization on gate distributions:

- improves expert utilization balance
- avoids early collapse to a few experts
- improves stability across seeds

### 2.4 Asymmetric Projection for Gradient Conflict

Asymmetric projection is applied to supported models (`share_bottom / moe / mmoe`) to suppress harmful conflicting gradient components while preserving useful task-specific updates.

It is particularly useful when click and conversion objectives compete on partially overlapping samples.

## 3. Gradient-Flow View (Project Intuition)

Under ESMM, three sample patterns are critical:

- `(y=0, z=0)`: exposure not clicked
- `(y=1, z=0)`: clicked not purchased
- `(y=1, z=1)`: clicked and purchased

In practice, `(1,0)` samples are the main source of optimization tension: CTR and CTCVR signals can push shared parameters in opposite directions under some probability regimes.

Our method combination addresses this at different levels:

- ESMM: training-space consistency
- MMoE: representation-level decoupling
- Entropy regularization: gate health and expert diversity
- Asymmetric projection: gradient-level conflict reduction

## 4. Implemented Models and Features

Main training entry: `train_rank.py`

Supported models:

- `share_bottom`
- `moe`
- `mmoe`
- `ple`
- `adaftr` (extended branch/feature)

Implemented capabilities:

- standard CTR/CVR multi-task training
- ESMM training (`--esmm`)
- TorchJD (`--use_torchjd`, methods: `upgrad`, `mgda`, `pcgrad`, `graddrop`)
- entropy regularization (`--use_entropy_reg --lambda_entropy`)
- asymmetric projection (`--use_asym_proj` and related options)
- EMA (`--use_ema`)
- DCN-v2 (`--use_dcn`)

## 5. Key Constraints

- `--use_torchjd` and `--use_asym_proj` are mutually exclusive.
- `--use_asym_proj` currently supports `share_bottom`, `moe`, `mmoe`.
- `--use_entropy_reg` currently supports `moe`, `mmoe`, `ple`.
- `--use_dcn` currently supports `share_bottom`, `mmoe`, `ple`, `adaftr`.

## 6. Data and Paths

Default Ali-CCP-style parquet paths are configured in `train_rank.py`:

- `/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_train.parquet`
- `/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_val.parquet`
- `/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_test.parquet`

Labels:

- `click`
- `purchase`

## 7. Quick Start

Install:

```bash
python -m pip install -e .
```

Basic ESMM + MMoE run:

```bash
python train_rank.py \
  --model mmoe \
  --esmm \
  --num_experts 6 \
  --expert_hidden_dims 256 128 \
  --batch_size 1024 \
  --num_workers 16 \
  --max_epochs 20 \
  --learning_rate 1e-3 \
  --dropout 0.2 \
  --accelerator gpu \
  --devices 4 \
  --strategy ddp_find_unused_parameters_false
```

Add entropy regularization:

```bash
python train_rank.py \
  --model mmoe \
  --esmm \
  --use_entropy_reg \
  --lambda_entropy 0.01
```

Add asymmetric projection:

```bash
python train_rank.py \
  --model mmoe \
  --esmm \
  --use_asym_proj \
  --asym_proj_lambda 1.0 \
  --asym_proj_tau 0.0
```

## 8. Final Experiment Suite

Directory:

- `experiments/final_ali_ccp/`

Run final suite:

```bash
bash experiments/final_ali_ccp/run_final_ali_ccp_suite.sh
```

Project pipeline (recommended): hyperparameter search -> ablation

```bash
bash experiments/final_ali_ccp/run_project_hpo_then_ablation.sh
```

Key output files:

- `experiments/final_ali_ccp/outputs/project_compare/hp/hp_search_grouped.csv`
- `experiments/final_ali_ccp/outputs/project_compare/hp/hp_best_by_stage_model.csv`
- `experiments/final_ali_ccp/outputs/project_compare/ablation/ablation_grouped.csv`
- `experiments/final_ali_ccp/outputs/project_compare/ablation/ablation_best_by_model.csv`

## 9. Repository Structure

```text
multi_task_ranking/
|- train_rank.py
|- src/
|  |- data/dataset.py
|  |- models/ranking.py
|  |- utils/constants.py
|  `- utils/metrics.py
|- experiments/
|  |- final_ali_ccp/
|  `- ...
`- setup.py
```

## 10. Notes

- This repo is experiment-heavy and assumes a GPU training environment.
- Default scripts are tuned for multi-GPU or cluster settings.
- For small smoke runs, reduce `TRAIN_SUBSET_FRAC`, `MAX_EPOCHS`, and set `MAX_EXPERIMENTS` in experiment scripts.
