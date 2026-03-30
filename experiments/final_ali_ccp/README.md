# Final Ali-CCP: ESMM + MMoE + Entropy Regularization + Asymmetric Projection

This directory contains the final experiment suite for our CTR/CVR ranking project on Ali-CCP style data.

The main method we propose is:

- `ESMM` for entire-space supervision of conversion behavior
- `MMoE` for soft task-specific feature routing
- `Entropy Regularization` to avoid gate polarization
- `Asymmetric Projection` to reduce harmful gradient interference

## Why this combination

In ESMM training, samples with `(click=1, purchase=0)` can create conflicting gradient pressure on the CTR pathway through the `pCTR * pCVR` coupling term. At the same time, CVR learning is sparse and unstable in early training.

Our design addresses these issues from complementary levels:

- `ESMM` fixes train/serve space mismatch for CVR-related learning
- `MMoE` decouples task representation with per-task gates over shared experts
- `Entropy Regularization` keeps expert utilization balanced and prevents gate collapse
- `Asymmetric Projection` suppresses conflicting gradient components while keeping useful task signal

## Supported models and options

Core models in this suite:

- `share_bottom`
- `moe`
- `mmoe`
- `ple`

Training options:

- `ESMM` (`--esmm`)
- `Entropy Regularization` (`--use_entropy_reg --lambda_entropy ...`)
- `TorchJD` (`--use_torchjd --aggregation_method ...`)
- `Asymmetric Projection` (`--use_asym_proj ...`)

Important constraint:

- `use_torchjd` and `use_asym_proj` are mutually exclusive in one run.

## Run the full final suite

```bash
bash experiments/final_ali_ccp/run_final_ali_ccp_suite.sh
```

Default stages:

- `baseline`
- `entropy`
- `torchjd`
- `asymproj`

Run only selected stages:

```bash
STAGES=baseline,entropy bash experiments/final_ali_ccp/run_final_ali_ccp_suite.sh
```

## Project pipeline: Hyperparameter Search -> Ablation

For the project comparison focusing on `share_bottom / mmoe / ple`:

```bash
bash experiments/final_ali_ccp/run_project_hpo_then_ablation.sh
```

This pipeline does:

1. hyperparameter search over architecture and method settings
2. seed aggregation and best-setting selection
3. ablation runs using selected best settings

## Standalone scripts

Hyperparameter search:

```bash
bash experiments/final_ali_ccp/run_project_hp_search.sh
```

Ablation only (requires HP best CSV):

```bash
bash experiments/final_ali_ccp/run_project_ablation.sh
```

Subset search (generic, not project-specific):

```bash
bash experiments/final_ali_ccp/run_subset_param_search.sh
```

## Key outputs

Hyperparameter search outputs:

- `experiments/final_ali_ccp/outputs/project_compare/hp/hp_search_summary.csv`
- `experiments/final_ali_ccp/outputs/project_compare/hp/hp_search_grouped.csv`
- `experiments/final_ali_ccp/outputs/project_compare/hp/hp_best_by_stage_model.csv`

Ablation outputs:

- `experiments/final_ali_ccp/outputs/project_compare/ablation/ablation_summary.csv`
- `experiments/final_ali_ccp/outputs/project_compare/ablation/ablation_grouped.csv`
- `experiments/final_ali_ccp/outputs/project_compare/ablation/ablation_best_by_model.csv`

General final-suite summary:

- `experiments/final_ali_ccp/ali_ccp_summary.csv`

## Common environment variables

- `ALI_CCP_TRAIN_PATH`, `ALI_CCP_VAL_PATH`, `ALI_CCP_TEST_PATH`
- `SEARCH_PROFILE=mini|quick|full`
- `SEEDS=42,2027,3407`
- `TRAIN_SUBSET_FRAC=0.1`
- `MAX_EPOCHS=4` (HP), `MAX_EPOCHS=6` (ablation), or custom values
- `BATCH_SIZE`, `NUM_WORKERS`
- `MAX_EXPERIMENTS=0` (`>0` for smoke runs)
