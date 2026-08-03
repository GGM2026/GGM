#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="../data/"
TASK_NAME="SST-2"

TEACHER_MODEL="../dynabert/SST-2"
STUDENT_MODEL="../dynabert/SST-2"

RATIO=1.0

OUTPUT_DIR="./outputs/"
LOG_DIR="./logs/${TASK_NAME}/${RATIO}"

mkdir -p "${OUTPUT_DIR}/${TASK_NAME}"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/$(date '+%Y-%m-%d-%H-%M-%S')-${TASK_NAME}-ggm-${RATIO}.log"

CUDA_VISIBLE_DEVICES=0 python quant_task_glue.py \
  --data_dir "${TASK_DIR}" \
  --teacher_model "${TEACHER_MODEL}" \
  --student_model "${STUDENT_MODEL}" \
  --task_name "${TASK_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs 50 \
  --seed 43 \
  --learning_rate 1e-4 \
  --weight_bits 1 \
  --embedding_bits 1 \
  --input_bits 1 \
  --batch_size 32 \
  --pred_distill \
  --intermediate_distill \
  --value_distill \
  --key_distill \
  --query_distill \
  --save_fp_model \
  --use_ggm \
  --ggm_ratio ${RATIO} \
  2>&1 | tee "${LOG_FILE}"
