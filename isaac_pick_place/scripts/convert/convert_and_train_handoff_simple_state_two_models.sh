#!/usr/bin/env bash
set -euo pipefail

# One-command pipeline for two simplified-state Handoff V2 trainings:
#   1. 26D state + 14D EE delta action
#   2. 26D state + 18D absolute post joint_pos action
#
# This script reuses the existing raw V2 demos and does not collect new data.

PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/Workspace/seven_dof_pick_place_lbm}"
SCRIPT_DIR="${PROJECT_ROOT}/isaac_pick_place/scripts/convert"

TRAIN_EPISODES="${TRAIN_EPISODES:-180}"
VAL_EPISODES="${VAL_EPISODES:-20}"

DO_CONVERT="${DO_CONVERT:-1}"
DO_TRAIN_DELTA14="${DO_TRAIN_DELTA14:-1}"
DO_TRAIN_ABSJOINT18="${DO_TRAIN_ABSJOINT18:-1}"

STEPS="${STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
TENSORBOARD="${TENSORBOARD:-1}"

DELTA_RUN_NAME="${DELTA_RUN_NAME:-hf_mtdp_handoff_v2_simple_state_delta14_${TRAIN_EPISODES}train${VAL_EPISODES}val_bs${BATCH_SIZE}acc${GRAD_ACCUM_STEPS}_50k}"
ABSJOINT_RUN_NAME="${ABSJOINT_RUN_NAME:-hf_mtdp_handoff_v2_simple_state_absjoint18_${TRAIN_EPISODES}train${VAL_EPISODES}val_bs${BATCH_SIZE}acc${GRAD_ACCUM_STEPS}_50k}"

DELTA_TRAIN_DIR="${PROJECT_ROOT}/experiments/lerobot_datasets/lerobot_handoff_v2_simple_state_delta14_train${TRAIN_EPISODES}"
DELTA_VAL_DIR="${PROJECT_ROOT}/experiments/lerobot_datasets/lerobot_handoff_v2_simple_state_delta14_val${VAL_EPISODES}"
ABSJOINT_TRAIN_DIR="${PROJECT_ROOT}/experiments/lerobot_datasets/lerobot_handoff_v2_simple_state_absjoint18_train${TRAIN_EPISODES}"
ABSJOINT_VAL_DIR="${PROJECT_ROOT}/experiments/lerobot_datasets/lerobot_handoff_v2_simple_state_absjoint18_val${VAL_EPISODES}"

echo "[PIPELINE] project=${PROJECT_ROOT}"
echo "[PIPELINE] train_episodes=${TRAIN_EPISODES} val_episodes=${VAL_EPISODES}"
echo "[PIPELINE] do_convert=${DO_CONVERT} do_train_delta14=${DO_TRAIN_DELTA14} do_train_absjoint18=${DO_TRAIN_ABSJOINT18}"
echo "[PIPELINE] steps=${STEPS} batch=${BATCH_SIZE} accum=${GRAD_ACCUM_STEPS}"
echo "[PIPELINE] delta_run=${DELTA_RUN_NAME}"
echo "[PIPELINE] absjoint_run=${ABSJOINT_RUN_NAME}"
echo

cd "${PROJECT_ROOT}"

if [[ "${DO_CONVERT}" == "1" ]]; then
  TRAIN_EPISODES="${TRAIN_EPISODES}" \
  VAL_EPISODES="${VAL_EPISODES}" \
  bash "${SCRIPT_DIR}/convert_handoff_simple_state_datasets.sh"
else
  echo "[PIPELINE] skipping conversion"
fi

if [[ "${DO_TRAIN_DELTA14}" == "1" ]]; then
  TRAIN_EPISODES="${TRAIN_EPISODES}" \
  VAL_EPISODES="${VAL_EPISODES}" \
  DATASET_DIR="${DELTA_TRAIN_DIR}" \
  VAL_DATASET_DIR="${DELTA_VAL_DIR}" \
  RUN_NAME="${DELTA_RUN_NAME}" \
  STEPS="${STEPS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS}" \
  TENSORBOARD="${TENSORBOARD}" \
  bash "${PROJECT_ROOT}/isaac_pick_place/scripts/train/train_handoff_simple_state_delta14_256_mtdp.sh"
else
  echo "[PIPELINE] skipping delta14 training"
fi

if [[ "${DO_TRAIN_ABSJOINT18}" == "1" ]]; then
  TRAIN_EPISODES="${TRAIN_EPISODES}" \
  VAL_EPISODES="${VAL_EPISODES}" \
  DATASET_DIR="${ABSJOINT_TRAIN_DIR}" \
  VAL_DATASET_DIR="${ABSJOINT_VAL_DIR}" \
  RUN_NAME="${ABSJOINT_RUN_NAME}" \
  STEPS="${STEPS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS}" \
  TENSORBOARD="${TENSORBOARD}" \
  bash "${PROJECT_ROOT}/isaac_pick_place/scripts/train/train_handoff_simple_state_absjoint18_256_mtdp.sh"
else
  echo "[PIPELINE] skipping absjoint18 training"
fi

echo
echo "[DONE] simplified-state two-model pipeline complete."
echo "[DONE] delta14 dataset: ${DELTA_TRAIN_DIR}"
echo "[DONE] absjoint18 dataset: ${ABSJOINT_TRAIN_DIR}"
if [[ "${DO_TRAIN_DELTA14}" == "1" ]]; then
  echo "[DONE] delta14 checkpoint: ${PROJECT_ROOT}/experiments/training_runs/${DELTA_RUN_NAME}/final_model"
fi
if [[ "${DO_TRAIN_ABSJOINT18}" == "1" ]]; then
  echo "[DONE] absjoint18 checkpoint: ${PROJECT_ROOT}/experiments/training_runs/${ABSJOINT_RUN_NAME}/final_model"
fi
