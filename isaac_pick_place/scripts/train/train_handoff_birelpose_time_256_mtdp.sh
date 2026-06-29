#!/usr/bin/env bash
set -euo pipefail

# Train the HF/LeRobot MultiTask DiT policy on the dual-arm handoff
# bidirectional-relative-pose + time-progress dataset.
#
# Example:
#   bash isaac_pick_place/scripts/train_handoff_birelpose_time_256_mtdp.sh
#
# Common overrides:
#   STEPS=10000 BATCH_SIZE=8 GRAD_ACCUM_STEPS=8 bash isaac_pick_place/scripts/train_handoff_birelpose_time_256_mtdp.sh
#   RUN_NAME=my_handoff_birelpose_time_run bash isaac_pick_place/scripts/train_handoff_birelpose_time_256_mtdp.sh
#   CHECKPOINT_PATH=/path/to/final_model bash isaac_pick_place/scripts/train_handoff_birelpose_time_256_mtdp.sh
#   TENSORBOARD=0 bash isaac_pick_place/scripts/train_handoff_birelpose_time_256_mtdp.sh

PROJECT_ROOT="${PROJECT_ROOT:-/home/ubuntu/Workspace/seven_dof_pick_place_lbm}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/Workspace/multitask_dit_policy/.venv/bin/python}"

DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/experiments/lerobot_datasets/lerobot_handoff_v2_full_train180}"
VAL_DATASET_DIR="${VAL_DATASET_DIR:-${PROJECT_ROOT}/experiments/lerobot_datasets/lerobot_handoff_v2_full_val20}"
RUN_NAME="${RUN_NAME:-hf_mtdp_handoff_v2_full_cube5_preaction_birelpose_time_aug_clip_relchunk_h50_a40_180train20val_bs16acc4_30k}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/experiments/training_runs/${RUN_NAME}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

STEPS="${STEPS:-30000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-17}"
LR="${LR:-2.0e-5}"

HORIZON="${HORIZON:-50}"
N_OBS_STEPS="${N_OBS_STEPS:-2}"
N_ACTION_STEPS="${N_ACTION_STEPS:-40}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
NUM_LAYERS="${NUM_LAYERS:-6}"
NUM_HEADS="${NUM_HEADS:-8}"
NUM_TRAIN_TIMESTEPS="${NUM_TRAIN_TIMESTEPS:-100}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
IMAGE_KEYS="${IMAGE_KEYS:-auto}"
IMAGE_NORMALIZATION="${IMAGE_NORMALIZATION:-clip}"
IMAGE_AUGMENTATION="${IMAGE_AUGMENTATION:-handoff_v2_full}"
ACTION_REPRESENTATION="${ACTION_REPRESENTATION:-relative_current_pose_chunk}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
LOG_EVERY="${LOG_EVERY:-1}"
STATE_MODE="${STATE_MODE:-handoff_joint_ee_birelpose_time}"
VAL_EVERY="${VAL_EVERY:-500}"
VAL_BATCHES="${VAL_BATCHES:-8}"
REQUIRE_MANIFEST="${REQUIRE_MANIFEST:-1}"
EXPECTED_DATASET_VERSION="${EXPECTED_DATASET_VERSION:-handoff_v2_full}"
EXPECTED_STATE_TIMING="${EXPECTED_STATE_TIMING:-exact_pre_action}"
EXPECTED_IMAGE_NORMALIZATION="${EXPECTED_IMAGE_NORMALIZATION:-${IMAGE_NORMALIZATION}}"
EXPECTED_IMAGE_AUGMENTATION="${EXPECTED_IMAGE_AUGMENTATION:-${IMAGE_AUGMENTATION}}"
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
echo "[CONFIG] save_freq=${SAVE_FREQ}"
echo "[CONFIG] device=${DEVICE} seed=${SEED}"
echo "[CONFIG] horizon=${HORIZON} n_obs_steps=${N_OBS_STEPS} n_action_steps=${N_ACTION_STEPS}"
echo "[CONFIG] hidden_dim=${HIDDEN_DIM} layers=${NUM_LAYERS} heads=${NUM_HEADS} diffusion_steps=${NUM_TRAIN_TIMESTEPS}"
echo "[CONFIG] image_size=${IMAGE_SIZE} image_keys=${IMAGE_KEYS} image_normalization=${IMAGE_NORMALIZATION} image_augmentation=${IMAGE_AUGMENTATION} video_backend=${VIDEO_BACKEND}"
echo "[CONFIG] action_representation=${ACTION_REPRESENTATION}"
echo "[CONFIG] state_mode=${STATE_MODE}"
echo "[CONFIG] val_every=${VAL_EVERY} val_batches=${VAL_BATCHES} require_manifest=${REQUIRE_MANIFEST}"
echo "[CONFIG] log_every=${LOG_EVERY} tensorboard=${TENSORBOARD} tensorboard_log_dir=${TENSORBOARD_LOG_DIR}"
echo "[CONFIG] tensorboard_flush_every=${TENSORBOARD_FLUSH_EVERY} tensorboard_flush_secs=${TENSORBOARD_FLUSH_SECS}"
echo

cd "${PROJECT_ROOT}"

TRAIN_ARGS=(
  "${PROJECT_ROOT}/isaac_pick_place/scripts/train/train_hf_mtdp_smoke.py"
  --dataset-dir "${DATASET_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --save-freq "${SAVE_FREQ}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --horizon "${HORIZON}" \
  --n-obs-steps "${N_OBS_STEPS}" \
  --n-action-steps "${N_ACTION_STEPS}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --num-layers "${NUM_LAYERS}" \
  --num-heads "${NUM_HEADS}" \
  --num-train-timesteps "${NUM_TRAIN_TIMESTEPS}" \
  --lr "${LR}" \
  --image-size "${IMAGE_SIZE}" \
  --image-keys "${IMAGE_KEYS}" \
  --image-normalization "${IMAGE_NORMALIZATION}" \
  --image-augmentation "${IMAGE_AUGMENTATION}" \
  --action-representation "${ACTION_REPRESENTATION}" \
  --log-every "${LOG_EVERY}" \
  --state-mode "${STATE_MODE}" \
  --video-backend "${VIDEO_BACKEND}" \
  --val-every "${VAL_EVERY}" \
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
