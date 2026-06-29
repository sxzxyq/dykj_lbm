#!/usr/bin/env bash
set -euo pipefail

# Collect random-cube scripted demos and convert them to a local LeRobotDataset.
#
# Example:
#   bash isaac_pick_place/scripts/collect_random_demos_to_lerobot.sh
#
# Optional overrides:
#   EPISODES=10 SEED=1200 bash isaac_pick_place/scripts/collect_random_demos_to_lerobot.sh
#   EPISODES=10 MAX_ATTEMPTS=25 bash isaac_pick_place/scripts/collect_random_demos_to_lerobot.sh
#   RUN_NAME=my_test bash isaac_pick_place/scripts/collect_random_demos_to_lerobot.sh
#   HEADLESS=0 EPISODES=1 MAX_ATTEMPTS=1 bash isaac_pick_place/scripts/collect_random_demos_to_lerobot.sh
#   TARGET_XY=0.62,0.00 bash isaac_pick_place/scripts/collect_random_demos_to_lerobot.sh

PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/Workspace/seven_dof_pick_place_lbm}"
ISAACLAB_DIR="${ISAACLAB_DIR:-/home/ubuntu/Workspace/IsaacLab}"
CONDA_BIN="${CONDA_BIN:-/home/ubuntu/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
CONVERT_PYTHON="${CONVERT_PYTHON:-/home/ubuntu/Workspace/multitask_dit_policy/.venv/bin/python}"

EPISODES="${EPISODES:-5}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"
SEED="${SEED:-1000}"
MAX_STEPS="${MAX_STEPS:-2500}"
PHASE_TIMEOUT="${PHASE_TIMEOUT:-700}"
LOG_EVERY="${LOG_EVERY:-250}"
DEVICE="${DEVICE:-cuda:0}"
TASK="${TASK:-Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0}"
HEADLESS="${HEADLESS:-1}"
TARGET_XY="${TARGET_XY:-0.50,0.00}"
CUBE_RESET_TARGET_XY="${CUBE_RESET_TARGET_XY:-}"
CUBE_RADIUS_RANGE="${CUBE_RADIUS_RANGE:-}"
CUBE_ANGLE_RANGE_DEG="${CUBE_ANGLE_RANGE_DEG:-}"
CUBE_ANGLE_RANGE_RAD="${CUBE_ANGLE_RANGE_RAD:-}"
IFS=, read -r TARGET_X TARGET_Y <<< "${TARGET_XY}"

RUN_NAME="${RUN_NAME:-random_cube_v0_${EPISODES}eps_$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_ROOT}/experiments}"
RAW_DIR="${RAW_DIR:-${EXPERIMENT_DIR}/raw_demos/raw_demos_${RUN_NAME}}"
LEROBOT_DIR="${LEROBOT_DIR:-${EXPERIMENT_DIR}/lerobot_datasets/lerobot_${RUN_NAME}}"
REPORT="${REPORT:-${EXPERIMENT_DIR}/reports/${RUN_NAME}_scripted_report.txt}"
REPO_ID="${REPO_ID:-local/seven_dof_pick_place_lbm_${RUN_NAME}}"

echo "[CONFIG] project=${PROJECT_ROOT}"
echo "[CONFIG] run_name=${RUN_NAME}"
echo "[CONFIG] target_success_episodes=${EPISODES} max_attempts=${MAX_ATTEMPTS:-0} seed=${SEED}"
echo "[CONFIG] raw_dir=${RAW_DIR}"
echo "[CONFIG] lerobot_dir=${LEROBOT_DIR}"
echo "[CONFIG] repo_id=${REPO_ID}"
echo "[CONFIG] convert_python=${CONVERT_PYTHON}"
echo "[CONFIG] headless=${HEADLESS}"
echo "[CONFIG] target_xy=${TARGET_XY}"
echo "[CONFIG] cube_reset_target_xy=${CUBE_RESET_TARGET_XY:-<target_xy>}"
echo "[CONFIG] cube_radius_range=${CUBE_RADIUS_RANGE:-<task default>}"
echo "[CONFIG] cube_angle_range_deg=${CUBE_ANGLE_RANGE_DEG:-<task default>}"
echo "[CONFIG] cube_angle_range_rad=${CUBE_ANGLE_RANGE_RAD:-<task default>}"
echo

cd "${ISAACLAB_DIR}"

SCRIPTED_ARGS=(
  "${PROJECT_ROOT}/isaac_pick_place/scripts/collect/scripted_pick_place.py" \
  --task "${TASK}" \
  --num_envs 1 \
  --episodes "${EPISODES}" \
  --success-episodes "${EPISODES}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  --max-steps "${MAX_STEPS}" \
  --seed "${SEED}" \
  --target-x "${TARGET_X}" \
  --target-y "${TARGET_Y}" \
  --enable_cameras \
  --device "${DEVICE}" \
  --phase-timeout "${PHASE_TIMEOUT}" \
  --log-every "${LOG_EVERY}" \
  --record-dir "${RAW_DIR}" \
  --record-warmup-steps 2 \
  --record-image-every 1 \
  --refresh-camera-xform \
  --report "${REPORT}"
)

if [[ "${HEADLESS}" == "1" ]]; then
  SCRIPTED_ARGS+=(--headless)
fi

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" env TERM=xterm PYTHONUNBUFFERED=1 ./isaaclab.sh -p "${SCRIPTED_ARGS[@]}"

cd "${PROJECT_ROOT}"

"${CONVERT_PYTHON}" "${PROJECT_ROOT}/isaac_pick_place/scripts/convert/convert_raw_demos_to_lerobot.py" \
    --raw-dir "${RAW_DIR}" \
    --output-dir "${LEROBOT_DIR}" \
    --repo-id "${REPO_ID}" \
    --fps 50 \
    --vcodec h264 \
    --skip-failed \
    --max-episodes "${EPISODES}" \
    --require-episodes "${EPISODES}" \
    --overwrite

echo
echo "[DONE] raw demos: ${RAW_DIR}"
echo "[DONE] LeRobot dataset: ${LEROBOT_DIR}"
echo "[DONE] report: ${REPORT}"
