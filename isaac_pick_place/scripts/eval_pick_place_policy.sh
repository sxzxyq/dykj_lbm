#!/usr/bin/env bash
set -euo pipefail

# Run closed-loop visual evaluation for a trained MultiTask DiT checkpoint.
#
# Example:
#   CHECKPOINT=/path/to/checkpoint_30000 bash isaac_pick_place/scripts/eval_pick_place_policy.sh
#
# Common overrides:
#   EPISODES=10 MAX_STEPS=900 N_ACTION_STEPS=8 bash isaac_pick_place/scripts/eval_pick_place_policy.sh
#   OUTPUT_DIR=/path/to/eval_out RECORD_IMAGE_EVERY=1 SAVE_VIDEO=1 bash isaac_pick_place/scripts/eval_pick_place_policy.sh
#   HEADLESS=0 SAVE_VIDEO=0 bash isaac_pick_place/scripts/eval_pick_place_policy.sh
#   FIXED_CUBE_XY=0.50,-0.10 bash isaac_pick_place/scripts/eval_pick_place_policy.sh
#   FIXED_CUBE_XY_LIST='0.36,-0.15;0.40,-0.11' bash isaac_pick_place/scripts/eval_pick_place_policy.sh

PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/Workspace/seven_dof_pick_place_lbm}"
ISAACLAB_DIR="${ISAACLAB_DIR:-/home/ubuntu/Workspace/IsaacLab}"
CONDA_BIN="${CONDA_BIN:-/home/ubuntu/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
LEROBOT_SITE_PACKAGES="${LEROBOT_SITE_PACKAGES:-/home/ubuntu/Workspace/multitask_dit_policy/.venv/lib/python3.12/site-packages}"
LEROBOT_PY311_SHIM="${LEROBOT_PY311_SHIM:-/tmp/lerobot_py311}"

CHECKPOINT="${CHECKPOINT:-}"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "[ERROR] CHECKPOINT is required, for example:"
  echo "  CHECKPOINT=${PROJECT_ROOT}/experiments/training_runs/hf_mtdp_random_cube_256_v0_100success_bs32acc2_30k/checkpoint_30000 \\"
  echo "    bash ${PROJECT_ROOT}/isaac_pick_place/scripts/eval_pick_place_policy.sh"
  exit 2
fi

RUN_NAME="${RUN_NAME:-eval_policy_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/experiments/eval_videos/${RUN_NAME}}"
TASK="${TASK:-Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0}"
EPISODES="${EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-900}"
SEED="${SEED:-2000}"
DEVICE="${DEVICE:-cuda:0}"
N_ACTION_STEPS="${N_ACTION_STEPS:-}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"
RECORD_IMAGE_EVERY="${RECORD_IMAGE_EVERY:-5}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
VIDEO_FPS="${VIDEO_FPS:-20}"
LOG_EVERY="${LOG_EVERY:-25}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
HEADLESS="${HEADLESS:-1}"
FIXED_CUBE_XY="${FIXED_CUBE_XY:-}"
FIXED_CUBE_XY_LIST="${FIXED_CUBE_XY_LIST:-}"

echo "[CONFIG] project=${PROJECT_ROOT}"
echo "[CONFIG] checkpoint=${CHECKPOINT}"
echo "[CONFIG] output=${OUTPUT_DIR}"
echo "[CONFIG] task=${TASK}"
echo "[CONFIG] episodes=${EPISODES} max_steps=${MAX_STEPS} seed=${SEED}"
echo "[CONFIG] device=${DEVICE} n_action_steps=${N_ACTION_STEPS:-<checkpoint>} num_inference_steps=${NUM_INFERENCE_STEPS:-<checkpoint>}"
echo

mkdir -p "${LEROBOT_PY311_SHIM}"
rm -rf "${LEROBOT_PY311_SHIM}/lerobot"
cp -a "${LEROBOT_SITE_PACKAGES}/lerobot" "${LEROBOT_PY311_SHIM}/lerobot"
python -c "from pathlib import Path; p=Path('${LEROBOT_PY311_SHIM}/lerobot/utils/io_utils.py'); text=p.read_text(); text=text.replace('def deserialize_json_into_object[T: JsonLike](fpath: Path, obj: T) -> T:', 'def deserialize_json_into_object(fpath: Path, obj):'); p.write_text(text)"
export PYTHONPATH="${LEROBOT_PY311_SHIM}${PYTHONPATH:+:${PYTHONPATH}}"

set +e
"${CONDA_BIN}" run -n "${CONDA_ENV}" python -c "import importlib.util, sys; missing=[name for name in ('lerobot','transformers','diffusers','safetensors','draccus','einops','PIL') if importlib.util.find_spec(name) is None]; print('[OK] policy inference dependencies available' if not missing else '[ERROR] Missing policy inference dependencies in the Isaac env: ' + ', '.join(missing)); sys.exit(0 if not missing else 3)"
preflight_status=$?
set -e
if [[ "${preflight_status}" -ne 0 ]]; then
  echo "[HINT] Install the LeRobot/MultiTask-DiT inference stack into CONDA_ENV=${CONDA_ENV}, then rerun this command."
  exit "${preflight_status}"
fi

cd "${ISAACLAB_DIR}"

ARGS=(
  "${PROJECT_ROOT}/isaac_pick_place/scripts/eval_pick_place_policy.py"
  --checkpoint "${CHECKPOINT}"
  --task "${TASK}"
  --output-dir "${OUTPUT_DIR}"
  --episodes "${EPISODES}"
  --max-steps "${MAX_STEPS}"
  --seed "${SEED}"
  --enable_cameras
  --device "${DEVICE}"
  --record-image-every "${RECORD_IMAGE_EVERY}"
  --video-fps "${VIDEO_FPS}"
  --warmup-steps "${WARMUP_STEPS}"
  --log-every "${LOG_EVERY}"
  --refresh-camera-xform
)

if [[ "${HEADLESS}" == "1" ]]; then
  ARGS+=(--headless)
fi
if [[ -n "${N_ACTION_STEPS}" ]]; then
  ARGS+=(--n-action-steps "${N_ACTION_STEPS}")
fi
if [[ -n "${NUM_INFERENCE_STEPS}" ]]; then
  ARGS+=(--num-inference-steps "${NUM_INFERENCE_STEPS}")
fi
if [[ -n "${FIXED_CUBE_XY}" ]]; then
  ARGS+=(--fixed-cube-xy "${FIXED_CUBE_XY}")
fi
if [[ -n "${FIXED_CUBE_XY_LIST}" ]]; then
  ARGS+=(--fixed-cube-xy-list "${FIXED_CUBE_XY_LIST}")
fi
if [[ "${SAVE_VIDEO}" == "1" ]]; then
  ARGS+=(--save-video)
fi

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" env TERM=xterm PYTHONUNBUFFERED=1 ./isaaclab.sh -p "${ARGS[@]}"

echo
echo "[DONE] eval output: ${OUTPUT_DIR}"
