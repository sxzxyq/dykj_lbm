#!/usr/bin/env bash
set -euo pipefail

# Run two step-0 diagnostics for the 49D handoff policy:
#   1) dataset/live image-state input split
#   2) diffusion sampling and action queue/chunk checks

PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/Workspace/seven_dof_pick_place_lbm}"
ISAACLAB_DIR="${ISAACLAB_DIR:-/home/ubuntu/Workspace/IsaacLab}"
CONDA_BIN="${CONDA_BIN:-/home/ubuntu/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
LEROBOT_SITE_PACKAGES="${LEROBOT_SITE_PACKAGES:-/home/ubuntu/Workspace/multitask_dit_policy/.venv/lib/python3.12/site-packages}"
LEROBOT_PY311_SHIM="${LEROBOT_PY311_SHIM:-/tmp/lerobot_py311}"

CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/experiments/training_runs/hf_mtdp_handoff_3cam_joint_ee_birelpose_time_100success_bs16acc4_30k/final_model}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/experiments/lerobot_datasets/lerobot_handoff_handoff_100_joint_ee_3cam_v1_birelpose_time}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/experiments/reports}"
TASK="${TASK:-Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0}"
TASK_TEXT="${TASK_TEXT:-Right arm moves the blue cube to the yellow handoff area, then left arm moves it to the red target area.}"
SEED="${SEED:-2000}"
DEVICE="${DEVICE:-cuda:0}"
DATASET_EPISODE="${DATASET_EPISODE:-0}"
DATASET_FRAME="${DATASET_FRAME:-0}"
DATASET_VIDEO_BACKEND="${DATASET_VIDEO_BACKEND:-pyav}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
WARMUP_OPEN_GRIPPER="${WARMUP_OPEN_GRIPPER:-0}"
HANDOFF_TIME_TOTAL_STEPS="${HANDOFF_TIME_TOTAL_STEPS:-1845}"
SAME_SEED_REPEATS="${SAME_SEED_REPEATS:-10}"
SEED_SWEEP_REPEATS="${SEED_SWEEP_REPEATS:-20}"
N_ACTION_STEPS="${N_ACTION_STEPS:-}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"
HEADLESS="${HEADLESS:-1}"

echo "[CONFIG] project=${PROJECT_ROOT}"
echo "[CONFIG] checkpoint=${CHECKPOINT}"
echo "[CONFIG] dataset=${DATASET_DIR}"
echo "[CONFIG] output_root=${OUTPUT_ROOT}"
echo "[CONFIG] task=${TASK}"
echo "[CONFIG] seed=${SEED} dataset_episode=${DATASET_EPISODE} dataset_frame=${DATASET_FRAME}"
echo "[CONFIG] warmup_steps=${WARMUP_STEPS} warmup_open_gripper=${WARMUP_OPEN_GRIPPER}"
echo "[CONFIG] same_seed_repeats=${SAME_SEED_REPEATS} seed_sweep_repeats=${SEED_SWEEP_REPEATS}"
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
  echo "[HINT] Install the LeRobot/MultiTask-DiT inference stack into CONDA_ENV=${CONDA_ENV}, then rerun."
  exit "${preflight_status}"
fi

cd "${ISAACLAB_DIR}"

ARGS=(
  "${PROJECT_ROOT}/isaac_pick_place/scripts/eval/diagnose_handoff_step0_policy.py"
  --checkpoint "${CHECKPOINT}"
  --dataset-dir "${DATASET_DIR}"
  --dataset-video-backend "${DATASET_VIDEO_BACKEND}"
  --task "${TASK}"
  --task-text "${TASK_TEXT}"
  --output-root "${OUTPUT_ROOT}"
  --seed "${SEED}"
  --dataset-episode "${DATASET_EPISODE}"
  --dataset-frame "${DATASET_FRAME}"
  --warmup-steps "${WARMUP_STEPS}"
  --handoff-time-total-steps "${HANDOFF_TIME_TOTAL_STEPS}"
  --same-seed-repeats "${SAME_SEED_REPEATS}"
  --seed-sweep-repeats "${SEED_SWEEP_REPEATS}"
  --enable_cameras
  --device "${DEVICE}"
  --refresh-camera-xform
)

if [[ "${HEADLESS}" == "1" ]]; then
  ARGS+=(--headless)
fi
if [[ "${WARMUP_OPEN_GRIPPER}" == "1" ]]; then
  ARGS+=(--warmup-open-gripper)
fi
if [[ -n "${N_ACTION_STEPS}" ]]; then
  ARGS+=(--n-action-steps "${N_ACTION_STEPS}")
fi
if [[ -n "${NUM_INFERENCE_STEPS}" ]]; then
  ARGS+=(--num-inference-steps "${NUM_INFERENCE_STEPS}")
fi

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" env TERM=xterm PYTHONUNBUFFERED=1 ./isaaclab.sh -p "${ARGS[@]}"
