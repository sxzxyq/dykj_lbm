#!/usr/bin/env bash
set -euo pipefail

# Open TensorBoard in Chromium with a clean profile and optional forced page refresh.
#
# This is useful on remote desktop sessions where TensorBoard's frontend polling can
# get stuck in an inactive/browser-throttled state even though the backend is
# reading fresh event data.
#
# Example:
#   bash isaac_pick_place/scripts/open_tensorboard_live_chromium.sh

PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/Workspace/seven_dof_pick_place_lbm}"
TENSORBOARD_BIN="${TENSORBOARD_BIN:-/home/ubuntu/Workspace/multitask_dit_policy/.venv/bin/tensorboard}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/Workspace/multitask_dit_policy/.venv/bin/python}"
CHROMIUM_BIN="${CHROMIUM_BIN:-chromium-browser}"

RUN_NAME="${RUN_NAME:-hf_mtdp_handoff_3cam_joint_ee_100success_bs16acc4_30k}"
LOGDIR="${LOGDIR:-${PROJECT_ROOT}/experiments/training_runs/${RUN_NAME}/tensorboard}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6006}"
RELOAD_INTERVAL="${RELOAD_INTERVAL:-5}"

OPEN_BROWSER="${OPEN_BROWSER:-1}"
START_TENSORBOARD="${START_TENSORBOARD:-1}"
FORCE_PAGE_RELOAD="${FORCE_PAGE_RELOAD:-1}"
PAGE_RELOAD_INTERVAL="${PAGE_RELOAD_INTERVAL:-8}"
PROFILE_DIR="${PROFILE_DIR:-/tmp/tensorboard-live-chromium-${PORT}}"
RESET_PROFILE="${RESET_PROFILE:-1}"
WRAPPER_PATH="${WRAPPER_PATH:-${PROJECT_ROOT}/isaac_pick_place/scripts/tools/tensorboard_auto_refresh_wrapper.html}"
TB_URL="${TB_URL:-http://localhost:${PORT}/#timeseries}"
TB_LOG="${TB_LOG:-${PROJECT_ROOT}/experiments/reports/tensorboard_${RUN_NAME}_${PORT}.log}"

if [[ ! -d "${LOGDIR}" ]]; then
  echo "[ERROR] TensorBoard logdir does not exist: ${LOGDIR}" >&2
  exit 1
fi
if [[ ! -x "${TENSORBOARD_BIN}" ]]; then
  echo "[ERROR] TensorBoard executable not found: ${TENSORBOARD_BIN}" >&2
  exit 1
fi
if [[ ! -f "${WRAPPER_PATH}" ]]; then
  echo "[ERROR] Auto-refresh wrapper not found: ${WRAPPER_PATH}" >&2
  exit 1
fi

echo "[CONFIG] logdir=${LOGDIR}"
echo "[CONFIG] port=${PORT} reload_interval=${RELOAD_INTERVAL}"
echo "[CONFIG] tensorboard_bin=${TENSORBOARD_BIN}"
echo "[CONFIG] chromium_bin=${CHROMIUM_BIN}"
echo "[CONFIG] force_page_reload=${FORCE_PAGE_RELOAD} page_reload_interval=${PAGE_RELOAD_INTERVAL}"
echo "[CONFIG] profile_dir=${PROFILE_DIR} reset_profile=${RESET_PROFILE}"
echo

if ! curl -fsS "http://127.0.0.1:${PORT}/data/environment" >/dev/null 2>&1; then
  if [[ "${START_TENSORBOARD}" != "1" ]]; then
    echo "[ERROR] TensorBoard is not responding on port ${PORT}, and START_TENSORBOARD=0." >&2
    exit 1
  fi
  mkdir -p "$(dirname "${TB_LOG}")"
  echo "[INFO] starting TensorBoard on port ${PORT}; log=${TB_LOG}"
  nohup "${TENSORBOARD_BIN}" \
    --logdir "${LOGDIR}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --reload_interval "${RELOAD_INTERVAL}" \
    >"${TB_LOG}" 2>&1 &

  for _ in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:${PORT}/data/environment" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

if ! curl -fsS "http://127.0.0.1:${PORT}/data/environment" >/dev/null 2>&1; then
  echo "[ERROR] TensorBoard did not become ready on port ${PORT}." >&2
  exit 1
fi
echo "[OK] TensorBoard backend is responding: http://localhost:${PORT}"

if [[ "${OPEN_BROWSER}" != "1" ]]; then
  exit 0
fi

if [[ "${RESET_PROFILE}" == "1" ]]; then
  case "${PROFILE_DIR}" in
    /tmp/*)
      rm -rf "${PROFILE_DIR}"
      ;;
    *)
      echo "[WARN] Refusing to remove non-/tmp Chromium profile: ${PROFILE_DIR}" >&2
      ;;
  esac
fi
mkdir -p "${PROFILE_DIR}"

if [[ "${FORCE_PAGE_RELOAD}" == "1" ]]; then
  OPEN_URL="$("${PYTHON_BIN}" -c \
    'import pathlib, sys, urllib.parse; print(pathlib.Path(sys.argv[1]).resolve().as_uri() + "?url=" + urllib.parse.quote(sys.argv[2], safe="") + "&seconds=" + urllib.parse.quote(sys.argv[3], safe=""))' \
    "${WRAPPER_PATH}" "${TB_URL}" "${PAGE_RELOAD_INTERVAL}")"
else
  OPEN_URL="${TB_URL}"
fi

echo "[INFO] opening ${OPEN_URL}"
"${CHROMIUM_BIN}" \
  --user-data-dir="${PROFILE_DIR}" \
  --disable-gpu \
  --disable-software-rasterizer \
  --no-sandbox \
  --disable-background-timer-throttling \
  --disable-renderer-backgrounding \
  --disable-backgrounding-occluded-windows \
  --disable-features=CalculateNativeWinOcclusion,IntensiveWakeUpThrottling \
  --new-window \
  "${OPEN_URL}" >/tmp/tensorboard-live-chromium.log 2>&1 &

echo "[DONE] Chromium launched. Log: /tmp/tensorboard-live-chromium.log"
