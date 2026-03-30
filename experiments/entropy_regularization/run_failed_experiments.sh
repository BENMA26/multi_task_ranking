#!/bin/bash
#SBATCH --job-name=mtl_entropy_failed
#SBATCH --output=logs/mtl_entropy_failed_%j.out
#SBATCH --error=logs/mtl_entropy_failed_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=144:00:00
#SBATCH --partition=normal
#SBATCH --exclude=g01n01

mkdir -p logs

TRAIN_PY=/work/home/maben/project/rec_sys/projects/multi_task_ranking/train_rank.py
OUTPUT_ROOT=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/entropy_regularization/outputs

# ── 实验 2: entropy_only ──────────────────────────────────────────────
echo "=========================================="
echo "Experiment [1/2]: mmoe_esmm_entropy_only"
echo "Model: mmoe | Entropy: true | UPGrad: false"
echo "=========================================="

python -u ${TRAIN_PY} \
    --model mmoe \
    --batch_size 1024 \
    --num_workers 32 \
    --max_epochs 20 \
    --learning_rate 1e-3 \
    --dropout 0.2 \
    --embedding_dim 32 \
    --expert_hidden_dims 256 128 \
    --num_experts 6 \
    --tower_hidden_dims 64 \
    --accelerator gpu \
    --devices 4 \
    --esmm \
    --sigmoid 1 \
    --strategy ddp_find_unused_parameters_false \
    --early_stop_patience 3 \
    --use_entropy_reg --lambda_entropy 0.01 \
    --exp_dir ${OUTPUT_ROOT}/mmoe_esmm_entropy_only

if [ $? -ne 0 ]; then
    echo "[ERROR] mmoe_esmm_entropy_only failed! Continuing..."
else
    echo "[OK] mmoe_esmm_entropy_only completed."
fi

# ── 实验 4: entropy_upgrad ────────────────────────────────────────────
echo "=========================================="
echo "Experiment [2/2]: mmoe_esmm_entropy_upgrad"
echo "Model: mmoe | Entropy: true | UPGrad: true"
echo "=========================================="

python -u ${TRAIN_PY} \
    --model mmoe \
    --batch_size 1024 \
    --num_workers 32 \
    --max_epochs 20 \
    --learning_rate 1e-3 \
    --dropout 0.2 \
    --embedding_dim 32 \
    --expert_hidden_dims 256 128 \
    --num_experts 6 \
    --tower_hidden_dims 64 \
    --accelerator gpu \
    --devices 4 \
    --esmm \
    --sigmoid 1 \
    --strategy ddp_find_unused_parameters_false \
    --early_stop_patience 3 \
    --use_entropy_reg --lambda_entropy 0.01 \
    --use_torchjd --aggregation_method upgrad \
    --exp_dir ${OUTPUT_ROOT}/mmoe_esmm_entropy_upgrad

if [ $? -ne 0 ]; then
    echo "[ERROR] mmoe_esmm_entropy_upgrad failed!"
else
    echo "[OK] mmoe_esmm_entropy_upgrad completed."
fi

echo "All 2 experiments finished."
