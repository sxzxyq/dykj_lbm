#!/usr/bin/env bash
set -euo pipefail

# Train Handoff V2 simplified-state policy with the existing 14D dual-arm
# end-effector delta action target.

PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/Workspace/seven_dof_pick_place_lbm}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/Workspace/multitask_dit_policy/.venv/bin/python}"

TRAIN_EPISODES="${TRAIN_EPISODES:-180}"
VAL_EPISODES="${VAL_EPISODES:-20}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/experiments/lerobot_datasets/lerobot_handoff_v2_simple_state_delta14_train${TRAIN_EPISODES}}"
VAL_DATASET_DIR="${VAL_DATASET_DIR:-${PROJECT_ROOT}/experiments/lerobot_datasets/lerobot_handoff_v2_simple_state_delta14_val${VAL_EPISODES}}"
RUN_NAME="${RUN_NAME:-hf_mtdp_handoff_v2_simple_state_delta14_${TRAIN_EPISODES}train${VAL_EPISODES}val_bs16acc4_50k}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/experiments/training_runs/${RUN_NAME}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

STEPS="${STEPS:-50000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-17}"
LR="${LR:-2.0e-5}"

HORIZON="${HORIZON:-32}"
N_OBS_STEPS="${N_OBS_STEPS:-2}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
NUM_LAYERS="${NUM_LAYERS:-6}"
NUM_HEADS="${NUM_HEADS:-8}"
NUM_TRAIN_TIMESTEPS="${NUM_TRAIN_TIMESTEPS:-100}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
IMAGE_KEYS="${IMAGE_KEYS:-auto}"
IMAGE_NORMALIZATION="${IMAGE_NORMALIZATION:-clip}"
IMAGE_AUGMENTATION="${IMAGE_AUGMENTATION:-handoff_v2_full}"
ACTION_REPRESENTATION="${ACTION_REPRESENTATION:-relative_current_pose_chunk}"
ACTION_LAYOUT="${ACTION_LAYOUT:-ee_delta_14}"
STATE_MODE="${STATE_MODE:-handoff_joint_tcp_pos_gripper}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
LOG_EVERY="${LOG_EVERY:-1}"
VAL_EVERY="${VAL_EVERY:-500}"
VAL_BATCHES="${VAL_BATCHES:-8}"
REQUIRE_MANIFEST="${REQUIRE_MANIFEST:-1}"
EXPECTED_DATASET_VERSION="${EXPECTED_DATASET_VERSION:-handoff_v2_simple_state_delta14}"
EXPECTED_STATE_TIMING="${EXPECTED_STATE_TIMING:-exact_pre_action}"
EXPECTED_IMAGE_NORMALIZATION="${EXPECTED_IMAGE_NORMALIZATION:-${IMAGE_NORMALIZATION}}"
EXPECTED_IMAGE_AUGMENTATION="${EXPECTED_IMAGE_AUGMENTATION:-${IMAGE_AUGMENTATION}}"
EXPECTED_ACTION_LAYOUT="${EXPECTED_ACTION_LAYOUT:-${ACTION_LAYOUT}}"
TENSORBOARD="${TENSORBOARD:-1}"
TENSORBOARD_LOG_DIR="${TENSORBOARD_LOG_DIR:-${OUTPUT_DIR}/tensorboard}"
TENSORBOARD_FLUSH_EVERY="${TENSORBOARD_FLUSH_EVERY:-10}"
TENSORBOARD_FLUSH_SECS="${TENSORBOARD_FLUSH_SECS:-5}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

echo "[CONFIG] project=${PROJECT_ROOT}"
echo "[CONFIG] dataset=${DATASET_DIR}"
echo "[CONFIG] val_dataset=${VAL_DATASET_DIR:-<none>}"
echo "[CONFIG] output=${OUTPUT_DIR}"
echo "[CONFIG] checkpoint_path=${CHECKPOINT_PATH:-<fresh>}"
echo "[CONFIG] steps=${STEPS} micro_batch_size=${BATCH_SIZE} grad_accum_steps=${GRAD_ACCUM_STEPS} effective_batch_size=$((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "[CONFIG] horizon=${HORIZON} n_obs_steps=${N_OBS_STEPS} n_action_steps=${N_ACTION_STEPS}"
echo "[CONFIG] state_mode=${STATE_MODE} action_layout=${ACTION_LAYOUT} action_representation=${ACTION_REPRESENTATION}"
echo "[CONFIG] image_normalization=${IMAGE_NORMALIZATION} image_augmentation=${IMAGE_AUGMENTATION}"
echo

cd "${PROJECT_ROOT}"

TRAIN_ARGS=(
  "${PROJECT_ROOT}/isaac_pick_place/scripts/train/train_hf_mtdp_smoke.py"
  --dataset-dir "${DATASET_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --save-freq "${SAVE_FREQ}"
  --steps "${STEPS}"
  --batch-size "${BATCH_SIZE}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --horizon "${HORIZON}"
  --n-obs-steps "${N_OBS_STEPS}"
  --n-action-steps "${N_ACTION_STEPS}"
  --hidden-dim "${HIDDEN_DIM}"
  --num-layers "${NUM_LAYERS}"
  --num-heads "${NUM_HEADS}"
  --num-train-timesteps "${NUM_TRAIN_TIMESTEPS}"
  --lr "${LR}"
  --image-size "${IMAGE_SIZE}"
  --image-keys "${IMAGE_KEYS}"
  --image-normalization "${IMAGE_NORMALIZATION}"
  --image-augmentation "${IMAGE_AUGMENTATION}"
  --action-representation "${ACTION_REPRESENTATION}"
  --log-every "${LOG_EVERY}"
  --state-mode "${STATE_MODE}"
  --video-backend "${VIDEO_BACKEND}"
  --val-every "${VAL_EVERY}"
  --val-batches "${VAL_BATCHES}"
)

if [[ -n "${VAL_DATASET_DIR}" ]]; then
  TRAIN_ARGS+=(--val-dataset-dir "${VAL_DATASET_DIR}")
fi
if [[ "${REQUIRE_MANIFEST}" == "1" ]]; then
  TRAIN_ARGS+=(
    --require-manifest
    --expected-dataset-version "${EXPECTED_DATASET_VERSION}"
    --expected-state-timing "${EXPECTED_STATE_TIMING}"
    --expected-image-normalization "${EXPECTED_IMAGE_NORMALIZATION}"
    --expected-image-augmentation "${EXPECTED_IMAGE_AUGMENTATION}"
    --expected-action-layout "${EXPECTED_ACTION_LAYOUT}"
  )
fi
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  TRAIN_ARGS+=(--checkpoint-path "${CHECKPOINT_PATH}")
fi
if [[ "${TENSORBOARD}" == "1" ]]; then
  TRAIN_ARGS+=(
    --tensorboard-log-dir "${TENSORBOARD_LOG_DIR}"
    --tensorboard-flush-every "${TENSORBOARD_FLUSH_EVERY}"
    --tensorboard-flush-secs "${TENSORBOARD_FLUSH_SECS}"
  )
fi

"${PYTHON_BIN}" "${TRAIN_ARGS[@]}"

echo
echo "[DONE] checkpoint: ${OUTPUT_DIR}/final_model"
if [[ "${TENSORBOARD}" == "1" ]]; then
  echo "[DONE] tensorboard: ${TENSORBOARD_LOG_DIR}"
fi
