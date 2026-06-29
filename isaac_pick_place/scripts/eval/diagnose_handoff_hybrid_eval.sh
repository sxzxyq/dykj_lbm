#!/usr/bin/env bash
set -euo pipefail

# Run 2x2 expert/policy hybrid diagnostics for the dual-arm handoff task.
#
# Example:
#   CHECKPOINT=/path/to/checkpoint_23000 \
#     bash isaac_pick_place/scripts/eval/diagnose_handoff_hybrid_eval.sh
#
# Common overrides:
#   COMBOS=expert_expert EPISODES=1 HEADLESS=1 bash isaac_pick_place/scripts/eval/diagnose_handoff_hybrid_eval.sh
#   RECORD_IMAGE_EVERY=5 RUN_NAME=hybrid_debug bash isaac_pick_place/scripts/eval/diagnose_handoff_hybrid_eval.sh

PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/Workspace/seven_dof_pick_place_lbm}"
ISAACLAB_DIR="${ISAACLAB_DIR:-/home/ubuntu/Workspace/IsaacLab}"
CONDA_BIN="${CONDA_BIN:-/home/ubuntu/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
LEROBOT_SITE_PACKAGES="${LEROBOT_SITE_PACKAGES:-/home/ubuntu/Workspace/multitask_dit_policy/.venv/lib/python3.12/site-packages}"
LEROBOT_PY311_SHIM="${LEROBOT_PY311_SHIM:-/tmp/lerobot_py311}"

CHECKPOINT="${CHECKPOINT:-}"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "[ERROR] CHECKPOINT is required, for example:"
  echo "  CHECKPOINT=${PROJECT_ROOT}/experiments/training_runs/.../checkpoint_23000 \\"
  echo "    bash ${PROJECT_ROOT}/isaac_pick_place/scripts/eval/diagnose_handoff_hybrid_eval.sh"
  exit 2
fi

RUN_NAME="${RUN_NAME:-diagnose_handoff_hybrid_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/experiments/eval_videos/${RUN_NAME}}"
TASK="${TASK:-Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0}"
TASK_TEXT="${TASK_TEXT:-Right arm moves the blue cube to the yellow handoff area, then left arm moves it to the red target area.}"
COMBOS="${COMBOS:-expert_expert,expert_policy,policy_expert,policy_policy}"
EPISODES="${EPISODES:-3}"
MAX_STEPS="${MAX_STEPS:-2600}"
SEED="${SEED:-2000}"
DEVICE="${DEVICE:-cuda:0}"
N_ACTION_STEPS="${N_ACTION_STEPS:-}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"
POLICY_INFERENCE_SEED="${POLICY_INFERENCE_SEED:-123}"
POLICY_INFERENCE_SEED_MODE="${POLICY_INFERENCE_SEED_MODE:-each_chunk}"
HANDOFF_TIME_TOTAL_STEPS="${HANDOFF_TIME_TOTAL_STEPS:-1845}"
STABLE_STEPS="${STABLE_STEPS:-20}"
REST_STEPS="${REST_STEPS:-20}"
CLOSE_STEPS="${CLOSE_STEPS:-35}"
OPEN_STEPS="${OPEN_STEPS:-35}"
PHASE_TIMEOUT="${PHASE_TIMEOUT:-320}"
LOG_EVERY="${LOG_EVERY:-25}"
RECORD_IMAGE_EVERY="${RECORD_IMAGE_EVERY:-0}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
HEADLESS="${HEADLESS:-1}"

echo "[CONFIG] project=${PROJECT_ROOT}"
echo "[CONFIG] checkpoint=${CHECKPOINT}"
echo "[CONFIG] output=${OUTPUT_DIR}"
echo "[CONFIG] task=${TASK}"
echo "[CONFIG] combos=${COMBOS}"
echo "[CONFIG] episodes=${EPISODES} max_steps=${MAX_STEPS} seed=${SEED}"
echo "[CONFIG] device=${DEVICE} n_action_steps=${N_ACTION_STEPS:-<checkpoint>} num_inference_steps=${NUM_INFERENCE_STEPS:-<checkpoint>}"
echo "[CONFIG] policy_inference_seed=${POLICY_INFERENCE_SEED} policy_inference_seed_mode=${POLICY_INFERENCE_SEED_MODE}"
echo "[CONFIG] stable_steps=${STABLE_STEPS} phase_timeout=${PHASE_TIMEOUT}"
echo "[CONFIG] record_image_every=${RECORD_IMAGE_EVERY} headless=${HEADLESS}"
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
  "${PROJECT_ROOT}/isaac_pick_place/scripts/eval/diagnose_handoff_hybrid_eval.py"
  --checkpoint "${CHECKPOINT}"
  --task "${TASK}"
  --task-text "${TASK_TEXT}"
  --output-dir "${OUTPUT_DIR}"
  --combos "${COMBOS}"
  --episodes "${EPISODES}"
  --max-steps "${MAX_STEPS}"
  --seed "${SEED}"
  --enable_cameras
  --device "${DEVICE}"
  --policy-inference-seed "${POLICY_INFERENCE_SEED}"
  --policy-inference-seed-mode "${POLICY_INFERENCE_SEED_MODE}"
  --handoff-time-total-steps "${HANDOFF_TIME_TOTAL_STEPS}"
  --stable-steps "${STABLE_STEPS}"
  --rest-steps "${REST_STEPS}"
  --close-steps "${CLOSE_STEPS}"
  --open-steps "${OPEN_STEPS}"
  --phase-timeout "${PHASE_TIMEOUT}"
  --record-image-every "${RECORD_IMAGE_EVERY}"
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

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" env TERM=xterm PYTHONUNBUFFERED=1 ./isaaclab.sh -p "${ARGS[@]}"

echo
echo "[DONE] hybrid diagnostic output: ${OUTPUT_DIR}"
