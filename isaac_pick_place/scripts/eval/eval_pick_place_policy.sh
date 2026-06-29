#!/usr/bin/env bash
set -euo pipefail

# Run closed-loop visual evaluation for a trained MultiTask DiT checkpoint.
#
# Example:
#   CHECKPOINT=/path/to/checkpoint_30000 bash isaac_pick_place/scripts/eval/eval_pick_place_policy.sh
#
# Common overrides:
#   EPISODES=10 MAX_STEPS=900 N_ACTION_STEPS=8 bash isaac_pick_place/scripts/eval/eval_pick_place_policy.sh
#   OUTPUT_DIR=/path/to/eval_out RECORD_IMAGE_EVERY=1 SAVE_VIDEO=1 bash isaac_pick_place/scripts/eval/eval_pick_place_policy.sh
#   HEADLESS=0 SAVE_VIDEO=0 bash isaac_pick_place/scripts/eval/eval_pick_place_policy.sh
#   FIXED_CUBE_XY=0.50,-0.10 bash isaac_pick_place/scripts/eval/eval_pick_place_policy.sh
#   FIXED_CUBE_XY_LIST='0.36,-0.15;0.40,-0.11' bash isaac_pick_place/scripts/eval/eval_pick_place_policy.sh

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
  echo "    bash ${PROJECT_ROOT}/isaac_pick_place/scripts/eval/eval_pick_place_policy.sh"
  exit 2
fi

RUN_NAME="${RUN_NAME:-eval_policy_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/experiments/eval_videos/${RUN_NAME}}"
TASK="${TASK:-Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0}"
TASK_TEXT="${TASK_TEXT:-Pick up the cube and place it on the red target area.}"
EPISODES="${EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-900}"
SEED="${SEED:-2000}"
DEVICE="${DEVICE:-cuda:0}"
N_ACTION_STEPS="${N_ACTION_STEPS:-}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"
POLICY_INFERENCE_SEED="${POLICY_INFERENCE_SEED:-}"
POLICY_INFERENCE_SEED_MODE="${POLICY_INFERENCE_SEED_MODE:-first_call}"
RECORD_IMAGE_EVERY="${RECORD_IMAGE_EVERY:-5}"
RECORD_POLICY_INPUTS="${RECORD_POLICY_INPUTS:-0}"
RECORD_POLICY_INPUT_TENSORS="${RECORD_POLICY_INPUT_TENSORS:-0}"
POLICY_INPUT_TENSOR_STEPS="${POLICY_INPUT_TENSOR_STEPS:-}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
VIDEO_FPS="${VIDEO_FPS:-20}"
LOG_EVERY="${LOG_EVERY:-25}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
WARMUP_OPEN_GRIPPER="${WARMUP_OPEN_GRIPPER:-1}"
HANDOFF_RIGHT_RETREAT_STEPS="${HANDOFF_RIGHT_RETREAT_STEPS:-300}"
HANDOFF_SCRIPTED_RIGHT_RETREAT="${HANDOFF_SCRIPTED_RIGHT_RETREAT:-0}"
HANDOFF_ACTIVE_ARM_MASK="${HANDOFF_ACTIVE_ARM_MASK:-1}"
FORCE_HANDOFF_ACTIVE_ARM_MASK="${FORCE_HANDOFF_ACTIVE_ARM_MASK:-0}"
HANDOFF_TIME_TOTAL_STEPS="${HANDOFF_TIME_TOTAL_STEPS:-1845}"
HEADLESS="${HEADLESS:-1}"
FIXED_CUBE_XY="${FIXED_CUBE_XY:-}"
FIXED_CUBE_XY_LIST="${FIXED_CUBE_XY_LIST:-}"
TEACHER_FORCED_DATASET_DIR="${TEACHER_FORCED_DATASET_DIR:-}"
TEACHER_FORCED_RAW_DIR="${TEACHER_FORCED_RAW_DIR:-}"
TEACHER_FORCED_EPISODE="${TEACHER_FORCED_EPISODE:-0}"
TEACHER_FORCED_RAW_EPISODE="${TEACHER_FORCED_RAW_EPISODE:-}"
TEACHER_FORCED_START_FRAME="${TEACHER_FORCED_START_FRAME:-0}"
TEACHER_FORCED_VIDEO_BACKEND="${TEACHER_FORCED_VIDEO_BACKEND:-pyav}"
TEACHER_FORCED_USE_DATASET_TASK="${TEACHER_FORCED_USE_DATASET_TASK:-0}"
TEACHER_FORCED_IMAGES_ONLY="${TEACHER_FORCED_IMAGES_ONLY:-0}"
TEACHER_FORCED_LIVE_IMAGE_KEYS="${TEACHER_FORCED_LIVE_IMAGE_KEYS:-}"

echo "[CONFIG] project=${PROJECT_ROOT}"
echo "[CONFIG] checkpoint=${CHECKPOINT}"
echo "[CONFIG] output=${OUTPUT_DIR}"
echo "[CONFIG] task=${TASK}"
echo "[CONFIG] task_text=${TASK_TEXT}"
echo "[CONFIG] episodes=${EPISODES} max_steps=${MAX_STEPS} seed=${SEED}"
echo "[CONFIG] device=${DEVICE} n_action_steps=${N_ACTION_STEPS:-<checkpoint>} num_inference_steps=${NUM_INFERENCE_STEPS:-<checkpoint>}"
echo "[CONFIG] policy_inference_seed=${POLICY_INFERENCE_SEED:-<none>} policy_inference_seed_mode=${POLICY_INFERENCE_SEED_MODE}"
echo "[CONFIG] warmup_steps=${WARMUP_STEPS} warmup_open_gripper=${WARMUP_OPEN_GRIPPER}"
echo "[CONFIG] record_policy_inputs=${RECORD_POLICY_INPUTS}"
echo "[CONFIG] record_policy_input_tensors=${RECORD_POLICY_INPUT_TENSORS} policy_input_tensor_steps=${POLICY_INPUT_TENSOR_STEPS:-<all>}"
echo "[CONFIG] handoff_right_retreat_steps=${HANDOFF_RIGHT_RETREAT_STEPS} handoff_time_total_steps=${HANDOFF_TIME_TOTAL_STEPS}"
echo "[CONFIG] force_handoff_active_arm_mask=${FORCE_HANDOFF_ACTIVE_ARM_MASK} handoff_active_arm_mask=${HANDOFF_ACTIVE_ARM_MASK}"
if [[ -n "${TEACHER_FORCED_DATASET_DIR}" ]]; then
  echo "[CONFIG] teacher_forced_dataset_dir=${TEACHER_FORCED_DATASET_DIR}"
  echo "[CONFIG] teacher_forced_episode=${TEACHER_FORCED_EPISODE} teacher_forced_start_frame=${TEACHER_FORCED_START_FRAME}"
  echo "[CONFIG] teacher_forced_images_only=${TEACHER_FORCED_IMAGES_ONLY}"
  echo "[CONFIG] teacher_forced_live_image_keys=${TEACHER_FORCED_LIVE_IMAGE_KEYS:-<none>}"
fi
if [[ -n "${TEACHER_FORCED_RAW_DIR}" ]]; then
  echo "[CONFIG] teacher_forced_raw_dir=${TEACHER_FORCED_RAW_DIR}"
  echo "[CONFIG] teacher_forced_raw_episode=${TEACHER_FORCED_RAW_EPISODE:-${TEACHER_FORCED_EPISODE}} teacher_forced_start_frame=${TEACHER_FORCED_START_FRAME}"
  echo "[CONFIG] teacher_forced_live_image_keys=${TEACHER_FORCED_LIVE_IMAGE_KEYS:-<none>}"
fi
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
  "${PROJECT_ROOT}/isaac_pick_place/scripts/eval/eval_pick_place_policy.py"
  --checkpoint "${CHECKPOINT}"
  --task "${TASK}"
  --task-text "${TASK_TEXT}"
  --output-dir "${OUTPUT_DIR}"
  --episodes "${EPISODES}"
  --max-steps "${MAX_STEPS}"
  --seed "${SEED}"
  --enable_cameras
  --device "${DEVICE}"
  --record-image-every "${RECORD_IMAGE_EVERY}"
  --video-fps "${VIDEO_FPS}"
  --warmup-steps "${WARMUP_STEPS}"
  --handoff-right-retreat-steps "${HANDOFF_RIGHT_RETREAT_STEPS}"
  --handoff-time-total-steps "${HANDOFF_TIME_TOTAL_STEPS}"
  --log-every "${LOG_EVERY}"
  --refresh-camera-xform
)

if [[ "${WARMUP_OPEN_GRIPPER}" == "1" ]]; then
  ARGS+=(--warmup-open-gripper)
fi
if [[ "${HEADLESS}" == "1" ]]; then
  ARGS+=(--headless)
fi
if [[ -n "${N_ACTION_STEPS}" ]]; then
  ARGS+=(--n-action-steps "${N_ACTION_STEPS}")
fi
if [[ -n "${NUM_INFERENCE_STEPS}" ]]; then
  ARGS+=(--num-inference-steps "${NUM_INFERENCE_STEPS}")
fi
if [[ -n "${POLICY_INFERENCE_SEED}" ]]; then
  ARGS+=(--policy-inference-seed "${POLICY_INFERENCE_SEED}" --policy-inference-seed-mode "${POLICY_INFERENCE_SEED_MODE}")
fi
if [[ -n "${FIXED_CUBE_XY}" ]]; then
  ARGS+=(--fixed-cube-xy "${FIXED_CUBE_XY}")
fi
if [[ -n "${FIXED_CUBE_XY_LIST}" ]]; then
  ARGS+=(--fixed-cube-xy-list "${FIXED_CUBE_XY_LIST}")
fi
if [[ -n "${TEACHER_FORCED_DATASET_DIR}" ]]; then
  ARGS+=(
    --teacher-forced-dataset-dir "${TEACHER_FORCED_DATASET_DIR}"
    --teacher-forced-episode "${TEACHER_FORCED_EPISODE}"
    --teacher-forced-start-frame "${TEACHER_FORCED_START_FRAME}"
    --teacher-forced-video-backend "${TEACHER_FORCED_VIDEO_BACKEND}"
  )
fi
if [[ -n "${TEACHER_FORCED_RAW_DIR}" ]]; then
  ARGS+=(
    --teacher-forced-raw-dir "${TEACHER_FORCED_RAW_DIR}"
    --teacher-forced-episode "${TEACHER_FORCED_EPISODE}"
    --teacher-forced-start-frame "${TEACHER_FORCED_START_FRAME}"
  )
  if [[ -n "${TEACHER_FORCED_RAW_EPISODE}" ]]; then
    ARGS+=(--teacher-forced-raw-episode "${TEACHER_FORCED_RAW_EPISODE}")
  fi
fi
if [[ "${TEACHER_FORCED_USE_DATASET_TASK}" == "1" ]]; then
  ARGS+=(--teacher-forced-use-dataset-task)
fi
if [[ "${TEACHER_FORCED_IMAGES_ONLY}" == "1" ]]; then
  ARGS+=(--teacher-forced-images-only)
fi
if [[ -n "${TEACHER_FORCED_LIVE_IMAGE_KEYS}" ]]; then
  ARGS+=(--teacher-forced-live-image-keys "${TEACHER_FORCED_LIVE_IMAGE_KEYS}")
fi
if [[ "${HANDOFF_SCRIPTED_RIGHT_RETREAT}" == "1" ]]; then
  ARGS+=(--handoff-scripted-right-retreat)
fi
if [[ "${HANDOFF_ACTIVE_ARM_MASK}" == "0" ]]; then
  ARGS+=(--disable-handoff-active-arm-mask)
fi
if [[ "${FORCE_HANDOFF_ACTIVE_ARM_MASK}" == "1" ]]; then
  ARGS+=(--force-handoff-active-arm-mask)
fi
if [[ "${SAVE_VIDEO}" == "1" ]]; then
  ARGS+=(--save-video)
fi
if [[ "${RECORD_POLICY_INPUTS}" == "1" ]]; then
  ARGS+=(--record-policy-inputs)
fi
if [[ "${RECORD_POLICY_INPUT_TENSORS}" == "1" ]]; then
  ARGS+=(--record-policy-input-tensors)
fi
if [[ -n "${POLICY_INPUT_TENSOR_STEPS}" ]]; then
  ARGS+=(--policy-input-tensor-steps "${POLICY_INPUT_TENSOR_STEPS}")
fi

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" env TERM=xterm PYTHONUNBUFFERED=1 ./isaaclab.sh -p "${ARGS[@]}"

echo
echo "[DONE] eval output: ${OUTPUT_DIR}"
