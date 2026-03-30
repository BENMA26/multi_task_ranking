#!/bin/bash
#SBATCH --job-name=mtl_hard_refine
#SBATCH --output=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/hard_sample_discovery/logs/mtl_hard_refine_%j.out
#SBATCH --error=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/hard_sample_discovery/logs/mtl_hard_refine_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=144:00:00
#SBATCH --partition=normal
#SBATCH --exclude=g01n01

set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

mkdir -p logs

TRAIN_PY=/work/home/maben/project/rec_sys/projects/multi_task_ranking/train_rank.py
OUTPUT_ROOT=/work/home/maben/project/rec_sys/projects/multi_task_ranking/experiments/hard_sample_discovery/outputs_refine_grid

# 默认关闭，若想同批次补跑 baseline:
# RUN_BASELINE=true sbatch run_hard_sample_refine_experiments.sh
RUN_BASELINE=${RUN_BASELINE:-false}

COMMON_ARGS="
    --model mmoe
    --batch_size 1024
    --num_workers 32
    --max_epochs 20
    --learning_rate 7e-4
    --dropout 0.25
    --embedding_dim 32
    --expert_hidden_dims 256 128
    --num_experts 6
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
    if python -u "${TRAIN_PY}" ${COMMON_ARGS} ${extra} --exp_dir "${OUTPUT_ROOT}/${name}"; then
        echo "[OK] ${name} completed."
    else
        echo "[ERROR] ${name} failed! Continuing..."
    fi
    echo ""
}

RATIOS=("0.05:r05" "0.10:r10" "0.15:r15")
WEIGHTS=("1.2:w12" "1.5:w15" "1.8:w18")
WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}

TOTAL=$(( ${#RATIOS[@]} * ${#WEIGHTS[@]} ))
if [ "${RUN_BASELINE}" = "true" ]; then
    TOTAL=$(( TOTAL + 1 ))
fi
CURRENT=0

if [ "${RUN_BASELINE}" = "true" ]; then
    CURRENT=$((CURRENT + 1))
    echo "Progress [${CURRENT}/${TOTAL}]"
    run_exp "mmoe_esmm_baseline" ""
fi

for ratio_info in "${RATIOS[@]}"; do
    IFS=':' read -r ratio ratio_tag <<< "${ratio_info}"
    for weight_info in "${WEIGHTS[@]}"; do
        IFS=':' read -r weight weight_tag <<< "${weight_info}"
        CURRENT=$((CURRENT + 1))
        EXP_NAME="mmoe_esmm_hard_${ratio_tag}_${weight_tag}"
        echo "Progress [${CURRENT}/${TOTAL}]"
        run_exp "${EXP_NAME}" "
            --use_hard_sample
            --hard_sample_ratio ${ratio}
            --hard_sample_weight ${weight}
            --hard_sample_warmup_epochs ${WARMUP_EPOCHS}
        "
    done
done

echo "All hard sample refine-grid experiments finished."
