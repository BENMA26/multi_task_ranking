# Multi-Task Ranking: ESMM + MMOE + Entropy Regularization + Gradient Surgery

This repository studies multi-task ranking for joint CTR/CVR prediction on Ali-CCP style data.

Current supported model set:

- `share_bottom`
- `mmoe`
- `ple`

Current supported optimization/regularization set:

- `Entropy Regularization`
- `Gradient surgery`: `normal`, `pcgrad`, `upgrad`, `asymmetric_projection`
- Scope control for gradient surgery: `all` or `(1,0)` samples (`on10`)

## Hierarchical CLI

Main training entry: `train_rank.py`

Hierarchical interface:

- `train` (subcommand)
- `--model`
- `--regularization` (`none|entropy`)
- `--grad_surgery` (`normal|pcgrad|upgrad|asymmetric_projection`)
- `--grad_scope` (`all|on10`)

Example:

```bash
python train_rank.py train \
  --model mmoe \
  --esmm \
  --regularization entropy \
  --lambda_entropy 0.01 \
  --grad_surgery upgrad \
  --grad_scope on10
```

## Key Ideas

- `ESMM`: entire-space supervision via `pCTCVR = pCTR * pCVR`
- `MMOE`: soft task-specific expert routing
- `Entropy Regularization`: avoid gate polarization
- `Gradient surgery`: reduce harmful task-gradient conflicts, especially around `(click=1,purchase=0)` samples

## Data Paths (default)

- `/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_train.parquet`
- `/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_val.parquet`
- `/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_test.parquet`

## Experiments

Final pipeline scripts are under:

- `experiments/final_ali_ccp/`

Main entry scripts:

- `run_project_hp_search.sh`
- `run_project_ablation.sh`
- `run_final_ali_ccp_suite.sh`

All of them are aligned to:

- models: `share_bottom/mmoe/ple`
- gradient surgery: `normal/pcgrad/upgrad/asymmetric_projection`
- scope: `all/on10`
