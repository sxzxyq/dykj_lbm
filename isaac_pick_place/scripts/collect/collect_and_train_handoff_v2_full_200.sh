#!/usr/bin/env bash
set -euo pipefail

# One-command Handoff V2 full pipeline:
#   1. collect 200 successful demos by default
#   2. convert into train180 / val20 LeRobot datasets
#   3. train the 49D BiRelPose+Time MultiTask DiT policy
#
# Useful overrides:
#   DO_COLLECT=0 bash isaac_pick_place/scripts/collect_and_train_handoff_v2_full_200.sh
#   DO_TRAIN=0 bash isaac_pick_place/scripts/collect_and_train_handoff_v2_full_200.sh
#   STEPS=60000 bash isaac_pick_place/scripts/collect_and_train_handoff_v2_full_200.sh

PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/Workspace/seven_dof_pick_place_lbm}"
COLLECT_DIR="${PROJECT_ROOT}/isaac_pick_place/scripts/collect"
TRAIN_DIR="${PROJECT_ROOT}/isaac_pick_place/scripts/train"

FULL_TRAIN_EPISODES="${FULL_TRAIN_EPISODES:-180}"
FULL_VAL_EPISODES="${FULL_VAL_EPISODES:-20}"
FULL_RUN_NAME="${FULL_RUN_NAME:-handoff_v2_full_${FULL_TRAIN_EPISODES}train${FULL_VAL_EPISODES}val}"

DO_COLLECT="${DO_COLLECT:-1}"
DO_TRAIN="${DO_TRAIN:-1}"

STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
HORIZON="${HORIZON:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-40}"
ACTION_REPRESENTATION="${ACTION_REPRESENTATION:-relative_current_pose_chunk}"
ACTION_HORIZON="${ACTION_HORIZON:-${HORIZON}}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-hf_mtdp_handoff_v2_full_cube5_preaction_birelpose_time_aug_clip_relchunk_h${HORIZON}_a${N_ACTION_STEPS}_${FULL_TRAIN_EPISODES}train${FULL_VAL_EPISODES}val_bs${BATCH_SIZE}acc${GRAD_ACCUM_STEPS}_${STEPS}steps}"

TRAIN_DATASET_DIR="${PROJECT_ROOT}/experiments/lerobot_datasets/lerobot_handoff_v2_full_train${FULL_TRAIN_EPISODES}"
VAL_DATASET_DIR="${PROJECT_ROOT}/experiments/lerobot_datasets/lerobot_handoff_v2_full_val${FULL_VAL_EPISODES}"

echo "[PIPELINE] project=${PROJECT_ROOT}"
echo "[PIPELINE] full_run_name=${FULL_RUN_NAME}"
echo "[PIPELINE] train_dataset=${TRAIN_DATASET_DIR}"
echo "[PIPELINE] val_dataset=${VAL_DATASET_DIR}"
echo "[PIPELINE] train_run_name=${TRAIN_RUN_NAME}"
echo "[PIPELINE] do_collect=${DO_COLLECT} do_train=${DO_TRAIN}"
echo "[PIPELINE] horizon=${HORIZON} n_action_steps=${N_ACTION_STEPS}"
echo "[PIPELINE] action_representation=${ACTION_REPRESENTATION} action_horizon=${ACTION_HORIZON}"
echo

cd "${PROJECT_ROOT}"

if [[ "${DO_COLLECT}" == "1" ]]; then
  CLEAN_EPISODES=0 \
  FULL_TRAIN_EPISODES="${FULL_TRAIN_EPISODES}" \
  FULL_VAL_EPISODES="${FULL_VAL_EPISODES}" \
  FULL_RUN_NAME="${FULL_RUN_NAME}" \
  HEADLESS="${HEADLESS:-1}" \
  LOG_EVERY="${LOG_EVERY:-100}" \
  ACTION_REPRESENTATION="${ACTION_REPRESENTATION}" \
  ACTION_HORIZON="${ACTION_HORIZON}" \
  bash "${COLLECT_DIR}/collect_handoff_v2_clean_full_to_lerobot.sh"
else
  echo "[PIPELINE] skipping collection/conversion"
fi

if [[ "${DO_TRAIN}" == "1" ]]; then
  DATASET_DIR="${TRAIN_DATASET_DIR}" \
  VAL_DATASET_DIR="${VAL_DATASET_DIR}" \
  RUN_NAME="${TRAIN_RUN_NAME}" \
  STEPS="${STEPS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS}" \
  HORIZON="${HORIZON}" \
  N_ACTION_STEPS="${N_ACTION_STEPS}" \
  ACTION_REPRESENTATION="${ACTION_REPRESENTATION}" \
  TENSORBOARD="${TENSORBOARD:-1}" \
  bash "${TRAIN_DIR}/train_handoff_birelpose_time_256_mtdp.sh"
else
  echo "[PIPELINE] skipping training"
fi

echo
echo "[DONE] Handoff V2 full pipeline complete."
echo "[DONE] train dataset: ${TRAIN_DATASET_DIR}"
echo "[DONE] val dataset: ${VAL_DATASET_DIR}"
if [[ "${DO_TRAIN}" == "1" ]]; then
  echo "[DONE] checkpoint: ${PROJECT_ROOT}/experiments/training_runs/${TRAIN_RUN_NAME}/final_model"
  echo "[DONE] tensorboard: ${PROJECT_ROOT}/experiments/training_runs/${TRAIN_RUN_NAME}/tensorboard"
fi
