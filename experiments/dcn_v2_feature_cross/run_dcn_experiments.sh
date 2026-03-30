#!/bin/bash
#SBATCH --job-name=mtl_dcnv2
#SBATCH --output=logs/mtl_dcnv2_%j.out
#SBATCH --error=logs/mtl_dcnv2_%j.err
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

# 主对照：MMOE baseline + DCN rank sweep(0/32/64)
DCN_RANKS=(0 32 64)

COMMON_ARGS="
    --batch_size 1024
    --num_workers 32
    --max_epochs 20
    --learning_rate 7e-4
    --dropout 0.25
    --embedding_dim 32
    --tower_hidden_dims 64
    --accelerator gpu
    --devices 4
    --esmm
    --sigmoid 1
    --strategy ddp_find_unused_parameters_false
    --early_stop_patience 3
"

run_exp() {
    local name=$1
    local extra=$2
    echo "=========================================="
    echo "Experiment: ${name}"
    echo "=========================================="
    python -u ${TRAIN_PY} ${COMMON_ARGS} ${extra} \
        --exp_dir ${OUTPUT_ROOT}/${name}
    if [ $? -ne 0 ]; then
        echo "[ERROR] ${name} failed! Continuing..."
    else
        echo "[OK] ${name} completed."
    fi
    echo ""
}

# 1) MMOE baseline（无 DCN）
run_exp "mmoe_esmm_baseline" "
    --model mmoe
    --expert_hidden_dims 256 128
    --num_experts 6
"

# 2) MMOE + DCN-v2 rank sweep
for RANK in "${DCN_RANKS[@]}"; do
    if [ "$RANK" -eq 0 ]; then
        EXP_NAME="mmoe_esmm_dcnv2_rank0_full"
    else
        EXP_NAME="mmoe_esmm_dcnv2_rank${RANK}"
    fi

    run_exp "${EXP_NAME}" "
        --model mmoe
        --expert_hidden_dims 256 128
        --num_experts 6
        --use_dcn --dcn_num_layers 2 --dcn_dropout 0.10 --dcn_rank ${RANK}
    "
done

# 3) ShareBottom 简单对照（可选）
run_exp "share_bottom_baseline" "
    --model share_bottom
    --shared_hidden_dims 256 128
"

run_exp "share_bottom_dcnv2_rank64" "
    --model share_bottom
    --shared_hidden_dims 256 128
    --use_dcn --dcn_num_layers 2 --dcn_dropout 0.10 --dcn_rank 64
"

echo "All DCN-v2 experiments finished."
