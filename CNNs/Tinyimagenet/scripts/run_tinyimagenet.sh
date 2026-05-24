set -euo pipefail


DATA_ROOT="./data"
CKPT_DIR="./checkpoints"

python scripts/prepare_tinyimagenet.py "${DATA_ROOT}"

ulimit -l unlimited || echo "Warning: could not raise memlock (need higher hard limit / sudo / limits.conf)"

NPROC_PER_NODE="4"

RUN_NAME="ggm_resnet"
RUN_NAME_GGD_N1="${RUN_NAME}_18_N1"

EPOCHS=90
BS_PER_GPU=128      
NUM_WORKERS=16      
BASE_LR=0.001     

DATASET="tinyimagenet"
IMG_SIZE=64         

LABEL_SMOOTHING=0.1
WEIGHT_DECAY=0.05
MIXUP=0.8
CUTMIX=1.0
DROP_PATH=0.1
AMP_FLAG="--amp"

echo "=========================================================="
echo " Starting: GGD ResNet-18" 
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
  --num_workers "${NUM_WORKERS}" \
  --base_lr "${BASE_LR}" \
  --img_size ${IMG_SIZE} \
  --label_smoothing ${LABEL_SMOOTHING} \
  --weight_decay ${WEIGHT_DECAY} \
  --num_runs 1 \
  --mixup ${MIXUP} \
  --cutmix ${CUTMIX} \
  --drop_path ${DROP_PATH} \
  ${AMP_FLAG}

echo "=========================================================="
echo "EXPERIMENTS COMPLETED"
