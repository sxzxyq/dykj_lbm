#!/usr/bin/env bash
set -euo pipefail

# Convert the existing Handoff V2 raw demos into two simplified-state dataset
# families:
#   1. 26D state + 14D end-effector delta action
#   2. 26D state + 18D absolute post-action joint_pos target

PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/Workspace/seven_dof_pick_place_lbm}"
SCRIPT_DIR="${PROJECT_ROOT}/isaac_pick_place/scripts/convert"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/Workspace/multitask_dit_policy/.venv/bin/python}"

RAW_DIR="${RAW_DIR:-${PROJECT_ROOT}/experiments/raw_demos/raw_handoff_handoff_v2_full_180train20val}"
TRAIN_EPISODES="${TRAIN_EPISODES:-180}"
VAL_EPISODES="${VAL_EPISODES:-20}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"

IMAGE_NORMALIZATION="${IMAGE_NORMALIZATION:-clip}"
IMAGE_AUGMENTATION="${IMAGE_AUGMENTATION:-handoff_v2_full}"
STATE_LAYOUT="${STATE_LAYOUT:-handoff_joint_tcp_pos_gripper}"
STATE_TIMING="${STATE_TIMING:-exact_pre_action}"
CUBE_SIZE_M="${CUBE_SIZE_M:-0.05}"

DELTA_DATASET_VERSION="${DELTA_DATASET_VERSION:-handoff_v2_simple_state_delta14}"
ABSJOINT_DATASET_VERSION="${ABSJOINT_DATASET_VERSION:-handoff_v2_simple_state_absjoint18}"

DELTA_TRAIN_DIR="${DELTA_TRAIN_DIR:-${PROJECT_ROOT}/experiments/lerobot_datasets/lerobot_handoff_v2_simple_state_delta14_train${TRAIN_EPISODES}}"
DELTA_VAL_DIR="${DELTA_VAL_DIR:-${PROJECT_ROOT}/experiments/lerobot_datasets/lerobot_handoff_v2_simple_state_delta14_val${VAL_EPISODES}}"
ABSJOINT_TRAIN_DIR="${ABSJOINT_TRAIN_DIR:-${PROJECT_ROOT}/experiments/lerobot_datasets/lerobot_handoff_v2_simple_state_absjoint18_train${TRAIN_EPISODES}}"
ABSJOINT_VAL_DIR="${ABSJOINT_VAL_DIR:-${PROJECT_ROOT}/experiments/lerobot_datasets/lerobot_handoff_v2_simple_state_absjoint18_val${VAL_EPISODES}}"

COMMON_ARGS=(
  "${SCRIPT_DIR}/convert_handoff_raw_demos_to_lerobot.py"
  --raw-dir "${RAW_DIR}"
  --state-layout "${STATE_LAYOUT}"
  --state-timing "${STATE_TIMING}"
  --cube-size-m "${CUBE_SIZE_M}"
  --image-normalization "${IMAGE_NORMALIZATION}"
  --image-augmentation "${IMAGE_AUGMENTATION}"
  --action-horizon "${ACTION_HORIZON}"
  --skip-failed
  --overwrite
)

echo "[CONVERT] project=${PROJECT_ROOT}"
echo "[CONVERT] raw_dir=${RAW_DIR}"
echo "[CONVERT] train_episodes=${TRAIN_EPISODES} val_episodes=${VAL_EPISODES}"
echo "[CONVERT] state_layout=${STATE_LAYOUT} state_timing=${STATE_TIMING}"
echo

cd "${PROJECT_ROOT}"

echo "[CONVERT] delta14 train -> ${DELTA_TRAIN_DIR}"
"${PYTHON_BIN}" "${COMMON_ARGS[@]}" \
  --output-dir "${DELTA_TRAIN_DIR}" \
  --repo-id "local/seven_dof_pick_place_lbm_handoff_v2_simple_state_delta14_train${TRAIN_EPISODES}" \
  --dataset-version "${DELTA_DATASET_VERSION}" \
  --split-name train \
  --action-layout ee_delta_14 \
  --action-representation relative_current_pose_chunk \
  --max-episodes "${TRAIN_EPISODES}" \
  --require-episodes "${TRAIN_EPISODES}"

echo "[CONVERT] delta14 val -> ${DELTA_VAL_DIR}"
"${PYTHON_BIN}" "${COMMON_ARGS[@]}" \
  --output-dir "${DELTA_VAL_DIR}" \
  --repo-id "local/seven_dof_pick_place_lbm_handoff_v2_simple_state_delta14_val${VAL_EPISODES}" \
  --dataset-version "${DELTA_DATASET_VERSION}" \
  --split-name val \
  --action-layout ee_delta_14 \
  --action-representation relative_current_pose_chunk \
  --skip-episodes-count "${TRAIN_EPISODES}" \
  --max-episodes "${VAL_EPISODES}" \
  --require-episodes "${VAL_EPISODES}"

echo "[CONVERT] absjoint18 train -> ${ABSJOINT_TRAIN_DIR}"
"${PYTHON_BIN}" "${COMMON_ARGS[@]}" \
  --output-dir "${ABSJOINT_TRAIN_DIR}" \
  --repo-id "local/seven_dof_pick_place_lbm_handoff_v2_simple_state_absjoint18_train${TRAIN_EPISODES}" \
  --dataset-version "${ABSJOINT_DATASET_VERSION}" \
  --split-name train \
  --action-layout abs_joint_pos_18 \
  --action-representation absolute_joint_pos \
  --max-episodes "${TRAIN_EPISODES}" \
  --require-episodes "${TRAIN_EPISODES}"

echo "[CONVERT] absjoint18 val -> ${ABSJOINT_VAL_DIR}"
"${PYTHON_BIN}" "${COMMON_ARGS[@]}" \
  --output-dir "${ABSJOINT_VAL_DIR}" \
  --repo-id "local/seven_dof_pick_place_lbm_handoff_v2_simple_state_absjoint18_val${VAL_EPISODES}" \
  --dataset-version "${ABSJOINT_DATASET_VERSION}" \
  --split-name val \
  --action-layout abs_joint_pos_18 \
  --action-representation absolute_joint_pos \
  --skip-episodes-count "${TRAIN_EPISODES}" \
  --max-episodes "${VAL_EPISODES}" \
  --require-episodes "${VAL_EPISODES}"

echo
echo "[DONE] delta14 train: ${DELTA_TRAIN_DIR}"
echo "[DONE] delta14 val: ${DELTA_VAL_DIR}"
echo "[DONE] absjoint18 train: ${ABSJOINT_TRAIN_DIR}"
echo "[DONE] absjoint18 val: ${ABSJOINT_VAL_DIR}"
