#!/bin/bash
#SBATCH --job-name=mtl_contra
#SBATCH --output=logs/mtl_contra_%j.out
#SBATCH --error=logs/mtl_contra_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=144:00:00
#SBATCH --partition=normal
#SBATCH --exclude=g01n01

mkdir -p logs

# ── 实验矩阵 ──────────────────────────────────────────────────────────────
# 模型：share_bottom / moe
# 对比学习：关闭（baseline） / 开启（+contrastive）
# 共 4 组实验

MODELS=("share_bottom" "moe")
USE_CONTRASTIVES=("true")

TOTAL=$(( ${#MODELS[@]} * ${#USE_CONTRASTIVES[@]} ))
CURRENT=0

TRAIN_PY=/work/home/maben/project/rec_sys/projects/multi_task_ranking/train_rank.py
OUTPUT_ROOT=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/contrastive_experiment/outputs

for MODEL in "${MODELS[@]}"; do
    for USE_CONTRA in "${USE_CONTRASTIVES[@]}"; do
        CURRENT=$(( CURRENT + 1 ))

        if [ "$USE_CONTRA" == "true" ]; then
            EXP_NAME="${MODEL}_esmm_contra"
            CONTRA_FLAGS="--use_contrastive --proj_dim 128 --bank_size 4096 --tau 0.07 --lambda_contra 0.1"
        else
            EXP_NAME="${MODEL}_esmm_baseline"
            CONTRA_FLAGS=""
        fi

        echo "=========================================="
        echo "Experiment [${CURRENT}/${TOTAL}]: ${EXP_NAME}"
        echo "Model: ${MODEL} | Contrastive: ${USE_CONTRA}"
        echo "=========================================="

        python -u ${TRAIN_PY} \
            --model ${MODEL} \
            --batch_size 1024 \
            --num_workers 32 \
            --max_epochs 20 \
            --learning_rate 1e-3 \
            --dropout 0.2 \
            --embedding_dim 32 \
            --expert_hidden_dims 256 128 \
            --num_experts 6 \
            --shared_hidden_dims 256 128 \
            --tower_hidden_dims 64 \
            --accelerator gpu \
            --devices 4 \
            --esmm \
            --sigmoid 1 \
            --strategy ddp_find_unused_parameters_false \
            --early_stop_patience 3 \
            ${CONTRA_FLAGS} \
            --exp_dir ${OUTPUT_ROOT}/${EXP_NAME}

        if [ $? -ne 0 ]; then
            echo "[ERROR] Experiment ${EXP_NAME} failed! Continuing to next..."
        else
            echo "[OK] Experiment ${EXP_NAME} completed."
        fi

        echo ""
    done
done

echo "All ${TOTAL} experiments finished."
