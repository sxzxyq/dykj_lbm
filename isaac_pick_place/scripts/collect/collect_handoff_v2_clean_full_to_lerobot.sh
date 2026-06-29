#!/usr/bin/env bash
set -euo pipefail

# Sequentially collect Handoff V2 datasets, then convert full into train/val
# LeRobot datasets. Clean-control is optional and disabled by default because
# the full dataset is the current training priority.

PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/Workspace/seven_dof_pick_place_lbm}"
COLLECT_DIR="${PROJECT_ROOT}/isaac_pick_place/scripts/collect"
CONVERT_DIR="${PROJECT_ROOT}/isaac_pick_place/scripts/convert"

CLEAN_EPISODES="${CLEAN_EPISODES:-0}"
FULL_TRAIN_EPISODES="${FULL_TRAIN_EPISODES:-180}"
FULL_VAL_EPISODES="${FULL_VAL_EPISODES:-20}"
FULL_EPISODES="${FULL_EPISODES:-$((FULL_TRAIN_EPISODES + FULL_VAL_EPISODES))}"

CLEAN_RUN_NAME="${CLEAN_RUN_NAME:-handoff_v2_clean_control_${CLEAN_EPISODES}}"
FULL_RUN_NAME="${FULL_RUN_NAME:-handoff_v2_full_${FULL_TRAIN_EPISODES}train${FULL_VAL_EPISODES}val}"

HEADLESS="${HEADLESS:-1}"
MAX_ATTEMPTS_CLEAN="${MAX_ATTEMPTS_CLEAN:-0}"
MAX_ATTEMPTS_FULL="${MAX_ATTEMPTS_FULL:-0}"
SEED_CLEAN="${SEED_CLEAN:-3100}"
SEED_FULL="${SEED_FULL:-4100}"
ACTION_REPRESENTATION="${ACTION_REPRESENTATION:-relative_current_pose_chunk}"
ACTION_HORIZON="${ACTION_HORIZON:-50}"

COMMON_ENV=(
  PROJECT_ROOT="${PROJECT_ROOT}"
  HEADLESS="${HEADLESS}"
  RECORD_IMAGE_EVERY="${RECORD_IMAGE_EVERY:-1}"
  STATE_LAYOUT="handoff_joint_ee_birelpose_time"
  STATE_TIMING="exact_pre_action"
)

if (( CLEAN_EPISODES > 0 )); then
  echo "[V2] collecting clean-control dataset (${CLEAN_EPISODES} successes)"
  env "${COMMON_ENV[@]}" \
    RUN_NAME="${CLEAN_RUN_NAME}" \
    EPISODES="${CLEAN_EPISODES}" \
    MAX_ATTEMPTS="${MAX_ATTEMPTS_CLEAN}" \
    SEED="${SEED_CLEAN}" \
    DATASET_VERSION="handoff_v2_clean_control" \
    SPLIT_NAME="clean" \
    CUBE_SIZE_M="0.04" \
    CUBE_RADIUS_RANGE="0.0,0.02" \
    CUBE_ANGLE_RANGE_DEG="-180,180" \
    CUBE_YAW_RANGE_DEG="0,0" \
    RANDOMIZATION_PROFILE="clean" \
    IMAGE_NORMALIZATION="dataset_stats" \
    IMAGE_AUGMENTATION="none" \
    ACTION_REPRESENTATION="${ACTION_REPRESENTATION}" \
    ACTION_HORIZON="${ACTION_HORIZON}" \
    LEROBOT_DIR="${PROJECT_ROOT}/experiments/lerobot_datasets/lerobot_handoff_v2_clean_control_${CLEAN_EPISODES}" \
    bash "${COLLECT_DIR}/collect_handoff_demos_to_lerobot.sh"
else
  echo "[V2] skipping clean-control collection (CLEAN_EPISODES=0)"
fi

echo "[V2] collecting full raw dataset (${FULL_EPISODES} successes)"
env "${COMMON_ENV[@]}" \
  RUN_NAME="${FULL_RUN_NAME}" \
  EPISODES="${FULL_EPISODES}" \
  MAX_ATTEMPTS="${MAX_ATTEMPTS_FULL}" \
  SEED="${SEED_FULL}" \
  DATASET_VERSION="handoff_v2_full" \
  SPLIT_NAME="full_raw" \
  CUBE_SIZE_M="0.05" \
  CUBE_RADIUS_RANGE="0.0,0.10" \
  CUBE_ANGLE_RANGE_DEG="-180,180" \
  CUBE_YAW_RANGE_DEG="-180,180" \
  RANDOMIZATION_PROFILE="full" \
  IMAGE_NORMALIZATION="clip" \
  IMAGE_AUGMENTATION="handoff_v2_full" \
  ACTION_REPRESENTATION="${ACTION_REPRESENTATION}" \
  ACTION_HORIZON="${ACTION_HORIZON}" \
  CONVERT_AFTER_COLLECT=0 \
  bash "${COLLECT_DIR}/collect_handoff_demos_to_lerobot.sh"

RAW_FULL_DIR="${PROJECT_ROOT}/experiments/raw_demos/raw_handoff_${FULL_RUN_NAME}"
CONVERT_PYTHON="${CONVERT_PYTHON:-/home/ubuntu/Workspace/multitask_dit_policy/.venv/bin/python}"

echo "[V2] converting full train split (${FULL_TRAIN_EPISODES})"
"${CONVERT_PYTHON}" "${CONVERT_DIR}/convert_handoff_raw_demos_to_lerobot.py" \
  --raw-dir "${RAW_FULL_DIR}" \
  --output-dir "${PROJECT_ROOT}/experiments/lerobot_datasets/lerobot_handoff_v2_full_train${FULL_TRAIN_EPISODES}" \
  --repo-id "local/seven_dof_pick_place_lbm_handoff_v2_full_train${FULL_TRAIN_EPISODES}" \
  --state-layout handoff_joint_ee_birelpose_time \
  --state-timing exact_pre_action \
  --dataset-version handoff_v2_full \
  --split-name train \
  --cube-size-m 0.05 \
  --image-normalization clip \
  --image-augmentation handoff_v2_full \
  --action-representation "${ACTION_REPRESENTATION}" \
  --action-horizon "${ACTION_HORIZON}" \
  --skip-failed \
  --max-episodes "${FULL_TRAIN_EPISODES}" \
  --require-episodes "${FULL_TRAIN_EPISODES}" \
  --overwrite

echo "[V2] converting full val split (${FULL_VAL_EPISODES})"
"${CONVERT_PYTHON}" "${CONVERT_DIR}/convert_handoff_raw_demos_to_lerobot.py" \
  --raw-dir "${RAW_FULL_DIR}" \
  --output-dir "${PROJECT_ROOT}/experiments/lerobot_datasets/lerobot_handoff_v2_full_val${FULL_VAL_EPISODES}" \
  --repo-id "local/seven_dof_pick_place_lbm_handoff_v2_full_val${FULL_VAL_EPISODES}" \
  --state-layout handoff_joint_ee_birelpose_time \
  --state-timing exact_pre_action \
  --dataset-version handoff_v2_full \
  --split-name val \
  --cube-size-m 0.05 \
  --image-normalization clip \
  --image-augmentation handoff_v2_full \
  --action-representation "${ACTION_REPRESENTATION}" \
  --action-horizon "${ACTION_HORIZON}" \
  --skip-failed \
  --skip-episodes-count "${FULL_TRAIN_EPISODES}" \
  --max-episodes "${FULL_VAL_EPISODES}" \
  --require-episodes "${FULL_VAL_EPISODES}" \
  --overwrite

echo
if (( CLEAN_EPISODES > 0 )); then
  echo "[DONE] clean dataset: ${PROJECT_ROOT}/experiments/lerobot_datasets/lerobot_handoff_v2_clean_control_${CLEAN_EPISODES}"
fi
echo "[DONE] full train: ${PROJECT_ROOT}/experiments/lerobot_datasets/lerobot_handoff_v2_full_train${FULL_TRAIN_EPISODES}"
echo "[DONE] full val: ${PROJECT_ROOT}/experiments/lerobot_datasets/lerobot_handoff_v2_full_val${FULL_VAL_EPISODES}"
