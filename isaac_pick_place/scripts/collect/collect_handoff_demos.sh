#!/usr/bin/env bash
set -euo pipefail

# Collect raw scripted demos for the sequential dual-arm handoff task.
#
# Example:
#   bash isaac_pick_place/scripts/collect_handoff_demos.sh
#   HEADLESS=0 EPISODES=1 MAX_ATTEMPTS=1 RUN_NAME=handoff_gui_smoke bash isaac_pick_place/scripts/collect_handoff_demos.sh

PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/Workspace/seven_dof_pick_place_lbm}"
ISAACLAB_DIR="${ISAACLAB_DIR:-/home/ubuntu/Workspace/IsaacLab}"
CONDA_BIN="${CONDA_BIN:-/home/ubuntu/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

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
RUN_NAME="${RUN_NAME:-handoff_${EPISODES}eps_$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_ROOT}/experiments}"
RAW_DIR="${RAW_DIR:-${EXPERIMENT_DIR}/raw_demos/raw_handoff_${RUN_NAME}}"
REPORT="${REPORT:-${EXPERIMENT_DIR}/reports/${RUN_NAME}_handoff_report.txt}"

YELLOW_XY="${YELLOW_XY:-0.50,0.00}"
RED_XY="${RED_XY:-0.50,0.30}"
IFS=, read -r YELLOW_X YELLOW_Y <<< "${YELLOW_XY}"
IFS=, read -r RED_X RED_Y <<< "${RED_XY}"

echo "[CONFIG] project=${PROJECT_ROOT}"
echo "[CONFIG] task=${TASK}"
echo "[CONFIG] run_name=${RUN_NAME}"
echo "[CONFIG] target_success_episodes=${EPISODES} max_attempts=${MAX_ATTEMPTS} seed=${SEED}"
echo "[CONFIG] raw_dir=${RAW_DIR}"
echo "[CONFIG] report=${REPORT}"
echo "[CONFIG] headless=${HEADLESS}"
echo "[CONFIG] yellow_xy_world=${YELLOW_XY}"
echo "[CONFIG] red_xy_world=${RED_XY}"
echo "[CONFIG] record_image_every=${RECORD_IMAGE_EVERY}"
echo

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

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" env TERM=xterm PYTHONUNBUFFERED=1 ./isaaclab.sh -p "${SCRIPTED_ARGS[@]}"

echo
echo "[DONE] raw handoff demos: ${RAW_DIR}"
echo "[DONE] report: ${REPORT}"
