#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="/nfs/hpc/share/pourmans/ggm-sst/data"
TASK_NAME="SST-2"

TEACHER_MODEL="/nfs/hpc/share/pourmans/ggm-sst/dynabert/SST-2"
STUDENT_MODEL="/nfs/hpc/share/pourmans/ggm-sst/dynabert/SST-2"

OUTPUT_DIR="/nfs/hpc/share/pourmans/ggm-sst/results/BiBERT_GGM_eval"
LOG_DIR="/nfs/hpc/share/pourmans/ggm-sst/results/BiBERT_GGM_eval/logs"

mkdir -p "${OUTPUT_DIR}/${TASK_NAME}"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/$(date '+%Y-%m-%d-%H-%M-%S')-${TASK_NAME}-ggm.log"

CUDA_VISIBLE_DEVICES=0 python quant_task_glue.py \
  --data_dir "${TASK_DIR}" \
  --teacher_model "${TEACHER_MODEL}" \
  --student_model "${STUDENT_MODEL}" \
  --task_name "${TASK_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs 50 \
  --seed 42 \
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
  --ggm_n_factor 5.0 \
  --ggm_eps 1e-5 \
  2>&1 | tee "${LOG_FILE}"
