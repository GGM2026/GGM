#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="../../data/cifar10_download"
CHECKPOINT_DIR="./checkpoints"
RESULTS_DIR="./results"
STDOUT_DIR="./stdout_logs"

BASE_RUN="cifar10_vgg16"
BASE_SEED=13

mkdir -p "$CHECKPOINT_DIR" "$RESULTS_DIR" "$STDOUT_DIR"

N_SCALE=2.5

RESULTS_FILE="${RESULTS_DIR}/nscale_${N_SCALE}.csv"

# Uncomment this if you want to start fresh each time for this N_scale
# rm -f "$RESULTS_FILE"

SEED=$((BASE_SEED))
RUN_NAME="${BASE_RUN}_N${N_SCALE}_seed${SEED}"
OUT_LOG="${STDOUT_DIR}/${RUN_NAME}.log"

echo "=== N_scale=${N_SCALE} | seed=${SEED} ==="

python main_cifar10.py \
  --data_root "$DATA_ROOT" \
  --batch_size 256 \
  --epochs 300 \
  --size 16 \
  --base_lr 3e-3 \
  --weight_decay 0.05 \
  --label_smoothing 0.05 \
  --num_workers 20 \
  --run_name "$RUN_NAME" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --no-compile \
  --amp \
  --use_ggd \
  --N_scale "$N_SCALE" \
  --seed "$SEED" \
  --results_file "$RESULTS_FILE" \
  2>&1 | tee "$OUT_LOG"
 