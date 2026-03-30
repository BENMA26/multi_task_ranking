#!/bin/bash
#SBATCH --job-name=mtl_ema_stability
#SBATCH --output=logs/mtl_ema_%j.out
#SBATCH --error=logs/mtl_ema_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=144:00:00
#SBATCH --partition=normal
#SBATCH --exclude=g01n01

mkdir -p logs

TRAIN_PY=/work/home/maben/project/rec_sys/projects/multi_task_ranking/train_rank.py
OUTPUT_ROOT=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/ema_stability/outputs

# 测试 3 个不同的随机种子，验证 EMA 的稳定性
SEEDS=(42 123 2024)

for SEED in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Experiment: mmoe_esmm_ema_seed_${SEED}"
    echo "Model: mmoe | ESMM: true | EMA: true | Seed: ${SEED}"
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
        --use_ema --ema_decay 0.999 \
        --exp_dir ${OUTPUT_ROOT}/mmoe_esmm_ema_seed_${SEED}

    if [ $? -ne 0 ]; then
        echo "[ERROR] mmoe_esmm_ema_seed_${SEED} failed! Continuing..."
    else
        echo "[OK] mmoe_esmm_ema_seed_${SEED} completed."
    fi
    echo ""
done

echo "All EMA stability experiments finished."
