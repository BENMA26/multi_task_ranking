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

TRAIN_PY=/work/home/maben/project/rec_sys/projects/multi_task_ranking/train_rank.py
OUTPUT_ROOT=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/dcn_v2_feature_cross/outputs

# 可通过环境变量覆盖
# 例：DCN_RANK=32 LAMBDA_ENTROPY=0.003 bash run_dcn_entropy_experiments.sh
DCN_RANK=${DCN_RANK:-64}
LAMBDA_ENTROPY=${LAMBDA_ENTROPY:-0.003}

USE_ENTROPY_REGS=("false" "true" "false" "true")
USE_TORCHJ_DS=("false" "false" "true" "true")

TOTAL=${#USE_ENTROPY_REGS[@]}
CURRENT=0

for i in "${!USE_ENTROPY_REGS[@]}"; do
    CURRENT=$(( CURRENT + 1 ))
    USE_ENTROPY=${USE_ENTROPY_REGS[$i]}
    USE_TORCHJD=${USE_TORCHJ_DS[$i]}

    if [ "$USE_ENTROPY" == "true" ] && [ "$USE_TORCHJD" == "true" ]; then
        EXP_NAME="mmoe_esmm_dcnv2_rank${DCN_RANK}_entropy_upgrad"
        ENTROPY_FLAGS="--use_entropy_reg --lambda_entropy ${LAMBDA_ENTROPY}"
        TORCHJD_FLAGS="--use_torchjd --aggregation_method upgrad"
    elif [ "$USE_ENTROPY" == "true" ]; then
        EXP_NAME="mmoe_esmm_dcnv2_rank${DCN_RANK}_entropy_only"
        ENTROPY_FLAGS="--use_entropy_reg --lambda_entropy ${LAMBDA_ENTROPY}"
        TORCHJD_FLAGS=""
    elif [ "$USE_TORCHJD" == "true" ]; then
        EXP_NAME="mmoe_esmm_dcnv2_rank${DCN_RANK}_upgrad_only"
        ENTROPY_FLAGS=""
        TORCHJD_FLAGS="--use_torchjd --aggregation_method upgrad"
    else
        EXP_NAME="mmoe_esmm_dcnv2_rank${DCN_RANK}_baseline"
        ENTROPY_FLAGS=""
        TORCHJD_FLAGS=""
    fi

    echo "=========================================="
    echo "Experiment [${CURRENT}/${TOTAL}]: ${EXP_NAME}"
    echo "Model: mmoe + dcnv2(rank=${DCN_RANK}) | Entropy: ${USE_ENTROPY} | UPGrad: ${USE_TORCHJD}"
    echo "=========================================="

    python -u ${TRAIN_PY} \
        --model mmoe \
        --batch_size 1024 \
        --num_workers 32 \
        --max_epochs 20 \
        --learning_rate 7e-4 \
        --dropout 0.25 \
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
        --use_dcn --dcn_num_layers 2 --dcn_dropout 0.10 --dcn_rank ${DCN_RANK} \
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
