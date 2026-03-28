#!/bin/bash
#SBATCH --job-name=mtl_dcnv2_entropy
#SBATCH --output=logs/mtl_dcnv2_entropy_%j.out
#SBATCH --error=logs/mtl_dcnv2_entropy_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=144:00:00
#SBATCH --partition=normal
#SBATCH --exclude=g01n01

mkdir -p logs

# 目标：在 DCN-v2 条件下复用 entropy_regularization 的实验矩阵
# 模型固定 mmoe（有 gate），并开启 --use_dcn
# 组合：
#   1) baseline        : no entropy + no upgrad
#   2) entropy_only    : entropy + no upgrad
#   3) upgrad_only     : no entropy + upgrad
#   4) entropy_upgrad  : entropy + upgrad

TRAIN_PY=/work/home/maben/project/rec_sys/projects/multi_task_ranking/train_rank.py
OUTPUT_ROOT=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/dcn_v2_feature_cross/outputs

USE_ENTROPY_REGS=("false" "true" "false" "true")
USE_TORCHJ_DS=("false" "false" "true" "true")

TOTAL=${#USE_ENTROPY_REGS[@]}
CURRENT=0

for i in "${!USE_ENTROPY_REGS[@]}"; do
    CURRENT=$(( CURRENT + 1 ))
    USE_ENTROPY=${USE_ENTROPY_REGS[$i]}
    USE_TORCHJD=${USE_TORCHJ_DS[$i]}

    if [ "$USE_ENTROPY" == "true" ] && [ "$USE_TORCHJD" == "true" ]; then
        EXP_NAME="mmoe_esmm_dcnv2_entropy_upgrad"
        ENTROPY_FLAGS="--use_entropy_reg --lambda_entropy 0.01"
        TORCHJD_FLAGS="--use_torchjd --aggregation_method upgrad"
    elif [ "$USE_ENTROPY" == "true" ]; then
        EXP_NAME="mmoe_esmm_dcnv2_entropy_only"
        ENTROPY_FLAGS="--use_entropy_reg --lambda_entropy 0.01"
        TORCHJD_FLAGS=""
    elif [ "$USE_TORCHJD" == "true" ]; then
        EXP_NAME="mmoe_esmm_dcnv2_upgrad_only"
        ENTROPY_FLAGS=""
        TORCHJD_FLAGS="--use_torchjd --aggregation_method upgrad"
    else
        EXP_NAME="mmoe_esmm_dcnv2_baseline"
        ENTROPY_FLAGS=""
        TORCHJD_FLAGS=""
    fi

    echo "=========================================="
    echo "Experiment [${CURRENT}/${TOTAL}]: ${EXP_NAME}"
    echo "Model: mmoe + dcnv2 | Entropy: ${USE_ENTROPY} | UPGrad: ${USE_TORCHJD}"
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
        --use_dcn --dcn_num_layers 2 --dcn_dropout 0.0 \
        ${ENTROPY_FLAGS} \
        ${TORCHJD_FLAGS} \
        --exp_dir ${OUTPUT_ROOT}/${EXP_NAME}

    if [ $? -ne 0 ]; then
        echo "[ERROR] Experiment ${EXP_NAME} failed! Continuing to next..."
    else
        echo "[OK] Experiment ${EXP_NAME} completed."
    fi

    echo ""
done

echo "All ${TOTAL} DCN-v2 entropy experiments finished."
