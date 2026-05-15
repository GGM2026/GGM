#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="./data"
CKPT_DIR="./checkpoints"

python scripts/prepare_tinyimagenet.py "${DATA_ROOT}"


NPROC_PER_NODE="1"

RUN_NAME="ggd_resnet"
RUN_NAME_GGD_N1="${RUN_NAME}_18_N1_accum"

EPOCHS=90

# --- THE CRITICAL FIX ---
# Process 128 images at a time, but update the model every 4 steps (128 * 4 = 512)
BS_PER_GPU=128
ACCUMULATION_STEPS=4

NUM_WORKERS=8
# Keep 0.004! Because world_size=1, your onecycle scheduler 
# will correctly use exactly 0.004 as the peak LR.
BASE_LR=0.004 

DATASET="tinyimagenet"
IMG_SIZE=64

LABEL_SMOOTHING=0.1
WEIGHT_DECAY=0.05
MIXUP=0.8
CUTMIX=1.0
DROP_PATH=0.1
AMP_FLAG="--amp"

echo "=========================================================="
echo " Starting: GGD ResNet-18 (Single GPU w/ Grad Accumulation)"
echo "=========================================================="

torchrun --nproc_per_node=${NPROC_PER_NODE} train.py \
  --dataset ${DATASET} \
  --data_root "${DATA_ROOT}" \
  --ckpt_dir "${CKPT_DIR}" \
  --run_name "${RUN_NAME_GGD_N1}" \
  --model "resnet" \
  --size "18" \
  --N_scale 1 \
  --epochs "${EPOCHS}" \
  --batch_size "${BS_PER_GPU}" \
  --accumulation_steps "${ACCUMULATION_STEPS}" \
  --num_workers "${NUM_WORKERS}" \
  --base_lr "${BASE_LR}" \
  --img_size ${IMG_SIZE} \
  --label_smoothing ${LABEL_SMOOTHING} \
  --weight_decay ${WEIGHT_DECAY} \
  --num_runs 1 \
  --mixup ${MIXUP} \
  --cutmix ${CUTMIX} \
  --drop_path ${DROP_PATH} \
  --scheduler "cosine" \
  ${AMP_FLAG}

echo "=========================================================="
echo "EXPERIMENTS COMPLETED"