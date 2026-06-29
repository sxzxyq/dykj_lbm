#!/usr/bin/env bash
set -euo pipefail

# Collect sequential dual-arm handoff demos and convert them to a local LeRobotDataset.
#
# Example:
#   bash isaac_pick_place/scripts/collect_handoff_demos_to_lerobot.sh
#   HEADLESS=0 EPISODES=1 MAX_ATTEMPTS=1 RUN_NAME=handoff_gui_smoke bash isaac_pick_place/scripts/collect_handoff_demos_to_lerobot.sh

PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/Workspace/seven_dof_pick_place_lbm}"
ISAACLAB_DIR="${ISAACLAB_DIR:-/home/ubuntu/Workspace/IsaacLab}"
CONDA_BIN="${CONDA_BIN:-/home/ubuntu/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
CONVERT_PYTHON="${CONVERT_PYTHON:-/home/ubuntu/Workspace/multitask_dit_policy/.venv/bin/python}"

TASK="${TASK:-Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0}"
EPISODES="${EPISODES:-5}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"
SEED="${SEED:-2000}"
MAX_STEPS="${MAX_STEPS:-2600}"
PHASE_TIMEOUT="${PHASE_TIMEOUT:-320}"
LOG_EVERY="${LOG_EVERY:-100}"
DEVICE="${DEVICE:-cuda:0}"
HEADLESS="${HEADLESS:-1}"
RECORD_IMAGE_EVERY="${RECORD_IMAGE_EVERY:-1}"
FPS="${FPS:-50}"
VCODEC="${VCODEC:-h264}"
STATE_LAYOUT="${STATE_LAYOUT:-handoff_joint_ee_birelpose_time}"
STATE_TIMING="${STATE_TIMING:-exact_pre_action}"
CONVERT_AFTER_COLLECT="${CONVERT_AFTER_COLLECT:-1}"
DATASET_VERSION="${DATASET_VERSION:-handoff_v1}"
SPLIT_NAME="${SPLIT_NAME:-train}"
CUBE_SIZE_M="${CUBE_SIZE_M:-0.04}"
CUBE_RADIUS_RANGE="${CUBE_RADIUS_RANGE:-0.0,0.10}"
CUBE_ANGLE_RANGE_DEG="${CUBE_ANGLE_RANGE_DEG:--180,180}"
CUBE_YAW_RANGE_DEG="${CUBE_YAW_RANGE_DEG:-0,0}"
RANDOMIZATION_PROFILE="${RANDOMIZATION_PROFILE:-none}"
IMAGE_NORMALIZATION="${IMAGE_NORMALIZATION:-dataset_stats}"
IMAGE_AUGMENTATION="${IMAGE_AUGMENTATION:-none}"
ACTION_REPRESENTATION="${ACTION_REPRESENTATION:-delta_step}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"
SKIP_EPISODES_COUNT="${SKIP_EPISODES_COUNT:-0}"

RUN_NAME="${RUN_NAME:-handoff_lerobot_${EPISODES}eps_$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_ROOT}/experiments}"
RAW_DIR="${RAW_DIR:-${EXPERIMENT_DIR}/raw_demos/raw_handoff_${RUN_NAME}}"
LEROBOT_DIR="${LEROBOT_DIR:-${EXPERIMENT_DIR}/lerobot_datasets/lerobot_handoff_${RUN_NAME}}"
REPORT="${REPORT:-${EXPERIMENT_DIR}/reports/${RUN_NAME}_handoff_report.txt}"
REPO_ID="${REPO_ID:-local/seven_dof_pick_place_lbm_handoff_${RUN_NAME}}"

YELLOW_XY="${YELLOW_XY:-0.50,0.00}"
RED_XY="${RED_XY:-0.50,0.30}"
IFS=, read -r YELLOW_X YELLOW_Y <<< "${YELLOW_XY}"
IFS=, read -r RED_X RED_Y <<< "${RED_XY}"

echo "[CONFIG] project=${PROJECT_ROOT}"
echo "[CONFIG] task=${TASK}"
echo "[CONFIG] run_name=${RUN_NAME}"
echo "[CONFIG] target_success_episodes=${EPISODES} max_attempts=${MAX_ATTEMPTS} seed=${SEED}"
echo "[CONFIG] raw_dir=${RAW_DIR}"
echo "[CONFIG] lerobot_dir=${LEROBOT_DIR}"
echo "[CONFIG] repo_id=${REPO_ID}"
echo "[CONFIG] report=${REPORT}"
echo "[CONFIG] convert_python=${CONVERT_PYTHON}"
echo "[CONFIG] headless=${HEADLESS}"
echo "[CONFIG] yellow_xy_world=${YELLOW_XY}"
echo "[CONFIG] red_xy_world=${RED_XY}"
echo "[CONFIG] record_image_every=${RECORD_IMAGE_EVERY}"
echo "[CONFIG] state_layout=${STATE_LAYOUT} state_timing=${STATE_TIMING}"
echo "[CONFIG] convert_after_collect=${CONVERT_AFTER_COLLECT}"
echo "[CONFIG] dataset_version=${DATASET_VERSION} split=${SPLIT_NAME} cube_size_m=${CUBE_SIZE_M}"
echo "[CONFIG] cube_radius_range=${CUBE_RADIUS_RANGE} cube_angle_range_deg=${CUBE_ANGLE_RANGE_DEG} cube_yaw_range_deg=${CUBE_YAW_RANGE_DEG}"
echo "[CONFIG] randomization_profile=${RANDOMIZATION_PROFILE} image_normalization=${IMAGE_NORMALIZATION} image_augmentation=${IMAGE_AUGMENTATION}"
echo "[CONFIG] action_representation=${ACTION_REPRESENTATION} action_horizon=${ACTION_HORIZON}"
echo

if [[ "${RECORD_IMAGE_EVERY}" != "1" ]]; then
  echo "[WARN] LeRobot conversion expects one image per recorded step. RECORD_IMAGE_EVERY=1 is recommended."
fi

cd "${ISAACLAB_DIR}"

SCRIPTED_ARGS=(
  "${PROJECT_ROOT}/isaac_pick_place/scripts/collect/scripted_handoff_collect.py" \
  --task "${TASK}" \
  --num_envs 1 \
  --episodes "${EPISODES}" \
  --success-episodes "${EPISODES}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  --max-steps "${MAX_STEPS}" \
  --seed "${SEED}" \
  --yellow-x "${YELLOW_X}" \
  --yellow-y "${YELLOW_Y}" \
  --red-x "${RED_X}" \
  --red-y "${RED_Y}" \
  --cube-size-m "${CUBE_SIZE_M}" \
  --dataset-version "${DATASET_VERSION}" \
  --randomization-profile "${RANDOMIZATION_PROFILE}" \
  --enable_cameras \
  --device "${DEVICE}" \
  --phase-timeout "${PHASE_TIMEOUT}" \
  --log-every "${LOG_EVERY}" \
  --record-dir "${RAW_DIR}" \
  --record-warmup-steps 2 \
  --record-image-every "${RECORD_IMAGE_EVERY}" \
  --refresh-camera-xform \
  --report "${REPORT}"
)

if [[ "${HEADLESS}" == "1" ]]; then
  SCRIPTED_ARGS+=(--headless)
fi

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" env \
  TERM=xterm \
  PYTHONUNBUFFERED=1 \
  CUBE_SIZE_M="${CUBE_SIZE_M}" \
  CUBE_RADIUS_RANGE="${CUBE_RADIUS_RANGE}" \
  CUBE_ANGLE_RANGE_DEG="${CUBE_ANGLE_RANGE_DEG}" \
  CUBE_YAW_RANGE_DEG="${CUBE_YAW_RANGE_DEG}" \
  ./isaaclab.sh -p "${SCRIPTED_ARGS[@]}"

cd "${PROJECT_ROOT}"

if [[ ! -d "${RAW_DIR}" ]]; then
  echo "[ERROR] Raw demo directory was not created: ${RAW_DIR}" >&2
  echo "[ERROR] Collection likely failed before recording any episode; check the Isaac log above and ${REPORT}." >&2
  exit 1
fi

if [[ "${CONVERT_AFTER_COLLECT}" == "0" ]]; then
  echo
  echo "[DONE] raw handoff demos: ${RAW_DIR}"
  echo "[DONE] report: ${REPORT}"
  exit 0
fi

"${CONVERT_PYTHON}" "${PROJECT_ROOT}/isaac_pick_place/scripts/convert/convert_handoff_raw_demos_to_lerobot.py" \
  --raw-dir "${RAW_DIR}" \
  --output-dir "${LEROBOT_DIR}" \
  --repo-id "${REPO_ID}" \
  --fps "${FPS}" \
  --vcodec "${VCODEC}" \
  --state-layout "${STATE_LAYOUT}" \
  --state-timing "${STATE_TIMING}" \
  --dataset-version "${DATASET_VERSION}" \
  --split-name "${SPLIT_NAME}" \
  --cube-size-m "${CUBE_SIZE_M}" \
  --image-normalization "${IMAGE_NORMALIZATION}" \
  --image-augmentation "${IMAGE_AUGMENTATION}" \
  --action-representation "${ACTION_REPRESENTATION}" \
  --action-horizon "${ACTION_HORIZON}" \
  --skip-failed \
  --skip-episodes-count "${SKIP_EPISODES_COUNT}" \
  --max-episodes "${EPISODES}" \
  --require-episodes "${EPISODES}" \
  --overwrite

echo
echo "[DONE] raw handoff demos: ${RAW_DIR}"
echo "[DONE] LeRobot dataset: ${LEROBOT_DIR}"
echo "[DONE] report: ${REPORT}"
