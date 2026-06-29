"""Open-loop action regression evaluation for local LeRobot MultiTask-DiT checkpoints.

This follows the same broad pattern as AgiBot-World's ``openloop_eval.py``:
iterate over a LeRobot dataset at a fixed inference interval, predict an action
chunk from dataset observations, compare it against the expert action chunk,
and plot predicted-vs-GT action traces.

The script is intentionally offline-only: it does not start Isaac. It uses the
checkpoint config/stats and the same preprocessing helpers as the local training
script, so the tensors entering the model match training/eval conventions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for local_dir in (SCRIPTS_DIR, SCRIPTS_ROOT / "common", SCRIPTS_ROOT / "train"):
    if str(local_dir) not in sys.path:
        sys.path.insert(0, str(local_dir))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from handoff_v2_utils import (  # noqa: E402
    ACTION_MOTION_SLICES,
    ACTION_REPRESENTATION_ABSOLUTE_JOINT_POS,
    ACTION_REPRESENTATION_DELTA_STEP,
    ACTION_REPRESENTATION_RELATIVE_CURRENT_POSE_CHUNK,
    ACTION_REPRESENTATIONS,
    load_manifest,
)
from train_hf_mtdp_smoke import (  # noqa: E402
    STATE_MODES,
    _build_features,
    _delta_timestamps,
    _mock_groot_imports,
    _normalize_or_unnormalize_action,
    _prepare_batch,
    _resolve_image_keys,
    _seed_everything,
    _state_mode_spec,
    _stats_to_tensors,
    _validate_state_feature_names,
)


DEFAULT_RUN_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "training_runs"
    / "hf_mtdp_handoff_v2_full_cube5_preaction_birelpose_time_aug_clip_relchunk_180train20val_bs16acc4_30000steps"
)
DEFAULT_CHECKPOINT = DEFAULT_RUN_ROOT / "checkpoint_24000"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "experiments" / "lerobot_datasets" / "lerobot_handoff_v2_full_val20"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "reports" / f"openloop_eval_{time.strftime('%Y%m%d_%H%M%S')}"

ACTION_DIM_NAMES = (
    "left_dx",
    "left_dy",
    "left_dz",
    "left_drx",
    "left_dry",
    "left_drz",
    "left_gripper",
    "right_dx",
    "right_dy",
    "right_dz",
    "right_drx",
    "right_dry",
    "right_drz",
    "right_gripper",
)
ABS_JOINT_ACTION_DIM_NAMES = (
    *(f"left_joint_pos.{index}" for index in range(9)),
    *(f"right_joint_pos.{index}" for index in range(9)),
)
STATE_MODE_BY_DIM = {
    16: "joint_ee",
    7: "ee_only",
    26: "handoff_joint_tcp_pos_gripper",
    34: "handoff_joint_ee",
    41: "handoff_joint_ee_relpose",
    43: "handoff_joint_ee_subtask",
    49: "handoff_joint_ee_birelpose_time",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Policy checkpoint directory.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="LeRobot dataset directory.")
    parser.add_argument("--repo-id", type=str, default=None, help="LeRobot repo id. Defaults to dataset dir name.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for open-loop artifacts.")
    parser.add_argument("--device", type=str, default="cuda", help="Policy device. Falls back to CPU if CUDA is unavailable.")
    parser.add_argument("--seed", type=int, default=123, help="Base RNG seed.")
    parser.add_argument(
        "--policy-inference-seed",
        type=int,
        default=123,
        help="If set, reseed before each policy batch as seed+batch_index for deterministic diffusion sampling.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Open-loop inference batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--video-backend", type=str, default="torchcodec", help="LeRobot video backend.")
    parser.add_argument("--image-keys", type=str, default="auto", help="Image keys, or auto.")
    parser.add_argument(
        "--image-normalization",
        type=str,
        default=None,
        choices=("dataset_stats", "clip", "imagenet"),
        help="Eval image preprocessing mode. Defaults to checkpoint manifest value, else dataset_stats.",
    )
    parser.add_argument(
        "--action-representation",
        type=str,
        default=None,
        choices=ACTION_REPRESENTATIONS,
        help="Checkpoint action representation. Defaults to checkpoint manifest value, else delta_step.",
    )
    parser.add_argument(
        "--state-mode",
        type=str,
        default=None,
        choices=tuple(STATE_MODES.keys()),
        help="State slice mode. Defaults from checkpoint observation.state dim.",
    )
    parser.add_argument("--sample-stride", type=int, default=None, help="Dataset index stride. Defaults to n_action_steps.")
    parser.add_argument("--start-index", type=int, default=0, help="Global dataset start index, or frame offset if --episode is set.")
    parser.add_argument("--max-chunks", type=int, default=0, help="Maximum chunks to evaluate. 0 means all selected chunks.")
    parser.add_argument("--episode", type=int, default=None, help="Evaluate one LeRobot episode index only.")
    parser.add_argument(
        "--sample-episodes",
        type=int,
        default=0,
        help="Randomly sample this many full episode_index values. Cannot be combined with --episode.",
    )
    parser.add_argument("--plot-space", choices=("env_delta", "model_action", "both"), default="both")
    parser.add_argument("--max-plot-points", type=int, default=20000, help="Downsample action plots above this many rows.")
    parser.add_argument("--write-csv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-npz", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True, help="Set HF/Transformers offline env vars.")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _checkpoint_weight_file(checkpoint: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.safetensors"):
        path = checkpoint / name
        if path.exists():
            return path
    candidates = sorted(checkpoint.glob("*.safetensors"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No safetensors weight file found in {checkpoint}")


def _remap_transformers5_clip_keys(state_dict: dict[str, torch.Tensor], model_keys: set[str]) -> tuple[dict[str, torch.Tensor], int]:
    remapped: dict[str, torch.Tensor] = {}
    remap_count = 0
    replacements = (
        (
            "observation_encoder.vision_encoder.model.",
            "observation_encoder.vision_encoder.model.vision_model.",
        ),
        (
            "observation_encoder.text_encoder.text_encoder.",
            "observation_encoder.text_encoder.text_encoder.text_model.",
        ),
    )
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in replacements:
            if not key.startswith(old_prefix):
                continue
            if key.startswith(new_prefix):
                break
            candidate = new_prefix + key[len(old_prefix) :]
            if candidate in model_keys:
                new_key = candidate
                remap_count += 1
                break
        remapped[new_key] = value
    return remapped, remap_count


def _load_policy_weights(policy: torch.nn.Module, checkpoint: Path, device: torch.device) -> dict[str, Any]:
    weight_file = _checkpoint_weight_file(checkpoint)
    state_dict = load_safetensors_file(str(weight_file), device=str(device))
    state_dict, remap_count = _remap_transformers5_clip_keys(state_dict, set(policy.state_dict().keys()))
    missing_keys, unexpected_keys = policy.load_state_dict(state_dict, strict=False)
    return {
        "weight_file": str(weight_file),
        "compat_key_remaps": int(remap_count),
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
    }


def _load_config_from_json(checkpoint: Path, config_cls, feature_type_cls, policy_feature_cls, device: torch.device):
    raw = _load_json(checkpoint / "config.json")
    raw.pop("type", None)
    raw["device"] = str(device)
    raw["input_features"] = {
        key: policy_feature_cls(type=feature_type_cls(value["type"]), shape=tuple(value["shape"]))
        for key, value in raw["input_features"].items()
    }
    raw["output_features"] = {
        key: policy_feature_cls(type=feature_type_cls(value["type"]), shape=tuple(value["shape"]))
        for key, value in raw["output_features"].items()
    }
    for key in ("image_resize_shape", "image_crop_shape", "optimizer_betas"):
        if isinstance(raw.get(key), list):
            raw[key] = tuple(raw[key])
    return config_cls(**raw)


def _state_dim_from_config(config) -> int:
    shape = tuple(config.input_features["observation.state"].shape)
    if len(shape) != 1:
        raise ValueError(f"Expected 1D observation.state shape, got {shape}")
    return int(shape[0])


def _action_dim_from_config(config) -> int:
    shape = tuple(config.output_features["action"].shape)
    if len(shape) != 1:
        raise ValueError(f"Expected 1D action shape, got {shape}")
    return int(shape[0])


def _action_dim_names(action_dim: int) -> tuple[str, ...]:
    if action_dim == 14:
        return ACTION_DIM_NAMES
    if action_dim == 18:
        return ABS_JOINT_ACTION_DIM_NAMES
    return tuple(f"action.{index}" for index in range(action_dim))


def _load_checkpoint_stats(checkpoint: Path, device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    stats_path = checkpoint / "dataset_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing checkpoint stats: {stats_path}")
    return _stats_to_tensors(_load_json(stats_path), device)


def _relative_current_chunk_to_delta_step_torch(actions: torch.Tensor) -> torch.Tensor:
    if actions.shape[-1] != 14:
        raise ValueError(f"Expected 14D actions, got {tuple(actions.shape)}")
    if actions.ndim != 3:
        raise ValueError(f"Expected [B,T,14] actions, got {tuple(actions.shape)}")
    output = actions.clone()
    for start, end in ACTION_MOTION_SLICES:
        output[:, 0, start:end] = actions[:, 0, start:end]
        if actions.shape[1] > 1:
            output[:, 1:, start:end] = actions[:, 1:, start:end] - actions[:, :-1, start:end]
    return output


class IndexedDataset(Dataset):
    def __init__(self, dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        dataset_index = int(self.indices[index])
        sample = dict(self.dataset[dataset_index])
        sample["__dataset_index"] = torch.tensor(dataset_index, dtype=torch.long)
        return sample


def _column_as_numpy(dataset, column: str) -> np.ndarray:
    values = dataset.hf_dataset[column]
    return np.asarray(values)


def _selected_indices(
    dataset,
    *,
    episode: int | None,
    sample_episodes: int,
    seed: int,
    start_index: int,
    stride: int,
    max_chunks: int,
) -> tuple[list[int], list[int]]:
    if stride <= 0:
        raise ValueError("--sample-stride must be positive")
    if episode is not None and sample_episodes > 0:
        raise ValueError("Use either --episode or --sample-episodes, not both.")
    selected_episode_ids: list[int] = []
    if sample_episodes > 0:
        episode_indices = _column_as_numpy(dataset, "episode_index").astype(np.int64)
        unique_episodes = np.unique(episode_indices)
        if sample_episodes > len(unique_episodes):
            raise ValueError(f"--sample-episodes={sample_episodes} exceeds dataset episode count {len(unique_episodes)}")
        rng = np.random.default_rng(seed)
        selected_episode_ids = sorted(int(value) for value in rng.choice(unique_episodes, size=sample_episodes, replace=False))
        indices: list[int] = []
        for episode_id in selected_episode_ids:
            matches = np.flatnonzero(episode_indices == int(episode_id))
            first = int(matches[0]) + int(start_index)
            last = int(matches[-1]) + 1
            if first < last:
                indices.extend(range(first, last, stride))
    elif episode is None:
        first = int(start_index)
        last = len(dataset)
        indices = list(range(first, last, stride))
    else:
        episode_indices = _column_as_numpy(dataset, "episode_index")
        matches = np.flatnonzero(episode_indices == int(episode))
        if matches.size == 0:
            raise ValueError(f"Dataset has no episode_index={episode}")
        first = int(matches[0]) + int(start_index)
        last = int(matches[-1]) + 1
        if first >= last:
            raise ValueError(f"--start-index {start_index} is outside episode {episode} length {len(matches)}")
        indices = list(range(first, last, stride))
        selected_episode_ids = [int(episode)]
    if max_chunks > 0:
        indices = indices[: int(max_chunks)]
    return indices, selected_episode_ids


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _valid_action_mask(batch: dict, start: int, steps: int, device: torch.device) -> torch.Tensor:
    mask = batch.get("action_is_pad")
    if mask is None:
        return torch.ones((batch["action"].shape[0], steps), dtype=torch.bool, device=device)
    mask = mask.to(device=device, dtype=torch.bool)
    return ~mask[:, start : start + steps]


def _masked_flat(values: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    return _to_numpy(values)[_to_numpy(mask).astype(bool)]


def _repeat_metadata(batch: dict, start: int, steps: int, mask: torch.Tensor) -> dict[str, np.ndarray]:
    valid = _to_numpy(mask).astype(bool).reshape(-1)
    batch_size = int(mask.shape[0])
    offsets = np.arange(start, start + steps, dtype=np.int64)
    metadata: dict[str, np.ndarray] = {
        "dataset_index": np.repeat(_to_numpy(batch["__dataset_index"]).astype(np.int64), steps),
        "chunk_step": np.tile(np.arange(steps, dtype=np.int64), batch_size),
        "target_offset": np.tile(offsets, batch_size),
    }
    for key in ("episode_index", "frame_index", "timestamp"):
        if key not in batch:
            continue
        value = _to_numpy(batch[key])
        if value.ndim > 1:
            value = value.reshape(value.shape[0], -1)[:, 0]
        metadata[key] = np.repeat(value, steps)
    if "frame_index" in metadata:
        metadata["target_frame_index"] = metadata["frame_index"].astype(np.int64) + metadata["target_offset"]
    if "timestamp" in metadata:
        metadata["target_timestamp"] = metadata["timestamp"].astype(np.float64) + metadata["target_offset"] / 50.0
    return {key: value[valid] for key, value in metadata.items()}


def _concat_or_empty(items: list[np.ndarray], shape_tail: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    if not items:
        return np.empty((0, *shape_tail), dtype=dtype)
    return np.concatenate(items, axis=0)


def _metrics(pred: np.ndarray, gt: np.ndarray, dim_names: tuple[str, ...]) -> dict[str, Any]:
    if pred.size == 0:
        return {}
    diff = pred - gt
    result = {
        "rows": int(pred.shape[0]),
        "mse": float(np.mean(diff**2)),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "per_dim_mse": {name: float(value) for name, value in zip(dim_names, np.mean(diff**2, axis=0), strict=True)},
        "per_dim_mae": {name: float(value) for name, value in zip(dim_names, np.mean(np.abs(diff), axis=0), strict=True)},
    }
    if pred.shape[-1] == 14:
        left_motion = diff[:, 0:6]
        right_motion = diff[:, 7:13]
        gripper = diff[:, [6, 13]]
        result["left_motion_mse"] = float(np.mean(left_motion**2))
        result["right_motion_mse"] = float(np.mean(right_motion**2))
        result["gripper_mse"] = float(np.mean(gripper**2))
        left_gt_norm = np.linalg.norm(gt[:, 0:6], axis=-1)
        right_gt_norm = np.linalg.norm(gt[:, 7:13], axis=-1)
        inactive_values = []
        if np.any(left_gt_norm < 1.0e-4):
            inactive_values.append(np.linalg.norm(pred[:, 0:6], axis=-1)[left_gt_norm < 1.0e-4])
        if np.any(right_gt_norm < 1.0e-4):
            inactive_values.append(np.linalg.norm(pred[:, 7:13], axis=-1)[right_gt_norm < 1.0e-4])
        if inactive_values:
            all_inactive = np.concatenate(inactive_values, axis=0)
            result["inactive_arm_action_norm_mean"] = float(np.mean(all_inactive))
            result["inactive_arm_action_norm_max"] = float(np.max(all_inactive))
        else:
            result["inactive_arm_action_norm_mean"] = 0.0
            result["inactive_arm_action_norm_max"] = 0.0
    elif pred.shape[-1] == 18:
        result["left_action_mse"] = float(np.mean(diff[:, 0:9] ** 2))
        result["right_action_mse"] = float(np.mean(diff[:, 9:18] ** 2))
        result["gripper_mse"] = float(np.mean(diff[:, [7, 8, 16, 17]] ** 2))
    else:
        half = pred.shape[-1] // 2
        result["left_action_mse"] = float(np.mean(diff[:, :half] ** 2))
        result["right_action_mse"] = float(np.mean(diff[:, half:] ** 2))
    return result


def _downsample_for_plot(pred: np.ndarray, gt: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or pred.shape[0] <= max_points:
        return pred, gt
    indices = np.linspace(0, pred.shape[0] - 1, num=max_points, dtype=np.int64)
    return pred[indices], gt[indices]


def _plot_actions(
    pred: np.ndarray,
    gt: np.ndarray,
    title: str,
    output_path: Path,
    max_points: int,
    dim_names: tuple[str, ...],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_plot, gt_plot = _downsample_for_plot(pred, gt, max_points)
    nrows = 4
    ncols = math.ceil(len(dim_names) / nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 2.8 * nrows), squeeze=False)
    fig.suptitle(title)
    x = np.arange(pred_plot.shape[0])
    for dim, name in enumerate(dim_names):
        ax = axes.reshape(-1)[dim]
        ax.set_title(f"{dim}: {name}")
        ax.plot(x, pred_plot[:, dim], color="#1f77b4", label="pred", linewidth=1.0)
        ax.plot(x, gt_plot[:, dim], color="#2ca02c", linestyle="--", label="gt", linewidth=1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes.reshape(-1)[len(dim_names) :]:
        ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_actions_csv(
    path: Path,
    metadata: dict[str, np.ndarray],
    pred: np.ndarray,
    gt: np.ndarray,
    dim_names: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_keys = list(metadata.keys())
    fieldnames = [
        *meta_keys,
        *(f"pred_{name}" for name in dim_names),
        *(f"gt_{name}" for name in dim_names),
        *(f"diff_{name}" for name in dim_names),
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        diff = pred - gt
        for row_index in range(pred.shape[0]):
            row = {key: metadata[key][row_index].item() for key in meta_keys}
            row.update({f"pred_{name}": float(pred[row_index, dim]) for dim, name in enumerate(dim_names)})
            row.update({f"gt_{name}": float(gt[row_index, dim]) for dim, name in enumerate(dim_names)})
            row.update({f"diff_{name}": float(diff[row_index, dim]) for dim, name in enumerate(dim_names)})
            writer.writerow(row)


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.dataset_dir.exists():
        raise FileNotFoundError(args.dataset_dir)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    _mock_groot_imports()

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
    from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy
    from transformers import CLIPTokenizer

    _seed_everything(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint_manifest = load_manifest(args.checkpoint)
    action_representation = (
        args.action_representation
        or (checkpoint_manifest or {}).get("action_representation")
        or ACTION_REPRESENTATION_DELTA_STEP
    )
    image_normalization = (
        args.image_normalization
        or (checkpoint_manifest or {}).get("image_normalization")
        or "dataset_stats"
    )

    config = _load_config_from_json(args.checkpoint, MultiTaskDiTConfig, FeatureType, PolicyFeature, device)
    state_dim = _state_dim_from_config(config)
    action_dim = _action_dim_from_config(config)
    action_dim_names = _action_dim_names(action_dim)
    state_mode = args.state_mode or STATE_MODE_BY_DIM.get(state_dim)
    if state_mode is None:
        raise ValueError(f"Cannot infer state_mode from checkpoint observation.state dim={state_dim}; pass --state-mode.")
    state_indices, state_names = _state_mode_spec(state_mode)
    if len(state_indices) != state_dim:
        raise ValueError(
            f"Checkpoint state dim={state_dim}, but state_mode={state_mode!r} keeps {len(state_indices)} dims."
        )

    meta = LeRobotDatasetMetadata(repo_id=args.repo_id or args.dataset_dir.name, root=args.dataset_dir)
    _validate_state_feature_names(meta, state_indices, state_names, state_mode)
    image_keys = _resolve_image_keys(meta, args.image_keys)
    input_features, output_features = _build_features(meta, state_indices, image_keys)
    config.input_features = input_features
    config.output_features = output_features

    dataset = LeRobotDataset(
        repo_id=args.repo_id or args.dataset_dir.name,
        root=args.dataset_dir,
        delta_timestamps=_delta_timestamps(config, meta.fps),
        video_backend=args.video_backend,
    )
    sample_stride = int(args.sample_stride or config.n_action_steps)
    selected_indices, selected_episode_ids = _selected_indices(
        dataset,
        episode=args.episode,
        sample_episodes=args.sample_episodes,
        seed=args.seed,
        start_index=args.start_index,
        stride=sample_stride,
        max_chunks=args.max_chunks,
    )
    indexed_dataset = IndexedDataset(dataset, selected_indices)
    dataloader = DataLoader(
        indexed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    stats = _load_checkpoint_stats(args.checkpoint, device)
    tokenizer = CLIPTokenizer.from_pretrained(config.text_encoder_name, local_files_only=True)
    policy = MultiTaskDiTPolicy(config).to(device)
    load_report = _load_policy_weights(policy, args.checkpoint, device)
    policy.eval()

    start = config.n_obs_steps - 1
    steps = config.n_action_steps
    pred_model_rows: list[np.ndarray] = []
    gt_model_rows: list[np.ndarray] = []
    pred_delta_rows: list[np.ndarray] = []
    gt_delta_rows: list[np.ndarray] = []
    metadata_rows: dict[str, list[np.ndarray]] = {}

    for batch_index, batch in enumerate(tqdm(dataloader, desc="openloop-eval")):
        if args.policy_inference_seed is not None:
            _seed_everything(int(args.policy_inference_seed) + batch_index)
        prepared = _prepare_batch(
            batch,
            config,
            tokenizer,
            stats,
            device,
            state_indices,
            image_keys,
            image_normalization=image_normalization,
            image_augmentation="none",
            action_representation=action_representation,
            train=False,
        )
        model_batch = policy._prepare_batch({key: value for key, value in prepared.items() if key != "action"})
        pred_norm = policy._generate_actions(model_batch)
        target_norm = prepared["action"][:, start : start + steps]
        if pred_norm.shape[:2] != target_norm.shape[:2]:
            raise RuntimeError(f"Pred shape {tuple(pred_norm.shape)} does not match target {tuple(target_norm.shape)}")

        pred_model = _normalize_or_unnormalize_action(pred_norm, stats, config, unnormalize=True)
        gt_model = _normalize_or_unnormalize_action(target_norm, stats, config, unnormalize=True)
        if action_representation == ACTION_REPRESENTATION_RELATIVE_CURRENT_POSE_CHUNK:
            pred_delta = _relative_current_chunk_to_delta_step_torch(pred_model)
            gt_delta = _relative_current_chunk_to_delta_step_torch(gt_model)
        elif action_representation == ACTION_REPRESENTATION_DELTA_STEP:
            pred_delta = pred_model
            gt_delta = gt_model
        elif action_representation == ACTION_REPRESENTATION_ABSOLUTE_JOINT_POS:
            pred_delta = pred_model
            gt_delta = gt_model
        else:
            raise ValueError(f"Unsupported action_representation={action_representation!r}")

        mask = _valid_action_mask(batch, start, steps, device)
        pred_model_rows.append(_masked_flat(pred_model, mask))
        gt_model_rows.append(_masked_flat(gt_model, mask))
        pred_delta_rows.append(_masked_flat(pred_delta, mask))
        gt_delta_rows.append(_masked_flat(gt_delta, mask))
        batch_meta = _repeat_metadata(batch, start, steps, mask)
        for key, value in batch_meta.items():
            metadata_rows.setdefault(key, []).append(value)

    pred_model_np = _concat_or_empty(pred_model_rows, (action_dim,))
    gt_model_np = _concat_or_empty(gt_model_rows, (action_dim,))
    pred_delta_np = _concat_or_empty(pred_delta_rows, (action_dim,))
    gt_delta_np = _concat_or_empty(gt_delta_rows, (action_dim,))
    metadata_np = {
        key: np.concatenate(values, axis=0) if values else np.empty((0,), dtype=np.float32)
        for key, values in metadata_rows.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.write_npz:
        np.savez_compressed(
            args.output_dir / "actions.npz",
            pred_model_action=pred_model_np,
            gt_model_action=gt_model_np,
            pred_env_delta=pred_delta_np,
            gt_env_delta=gt_delta_np,
            **{f"meta_{key}": value for key, value in metadata_np.items()},
        )
    if args.write_csv:
        _write_actions_csv(args.output_dir / "actions_env_delta.csv", metadata_np, pred_delta_np, gt_delta_np, action_dim_names)
        _write_actions_csv(args.output_dir / "actions_model_action.csv", metadata_np, pred_model_np, gt_model_np, action_dim_names)
    if args.plot_space in ("env_delta", "both"):
        _plot_actions(
            pred_delta_np,
            gt_delta_np,
            f"Open-loop env delta pred vs GT\n{args.checkpoint}",
            args.output_dir / "pred_vs_gt_env_delta.png",
            args.max_plot_points,
            action_dim_names,
        )
    if args.plot_space in ("model_action", "both"):
        _plot_actions(
            pred_model_np,
            gt_model_np,
            f"Open-loop model-action pred vs GT\n{args.checkpoint}",
            args.output_dir / "pred_vs_gt_model_action.png",
            args.max_plot_points,
            action_dim_names,
        )

    dataset_manifest = load_manifest(args.dataset_dir)
    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "seed": args.seed,
        "policy_inference_seed": args.policy_inference_seed,
        "dataset_frames": len(dataset),
        "selected_chunks": len(selected_indices),
        "valid_action_rows": int(pred_delta_np.shape[0]),
        "sample_stride": sample_stride,
        "start_index": args.start_index,
        "episode": args.episode,
        "sample_episodes": args.sample_episodes,
        "selected_episode_ids": selected_episode_ids,
        "batch_size": args.batch_size,
        "fps": meta.fps,
        "horizon": config.horizon,
        "n_obs_steps": config.n_obs_steps,
        "n_action_steps": config.n_action_steps,
        "target_start_in_action_chunk": start,
        "image_keys": list(image_keys),
        "state_mode": state_mode,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_dim_names": list(action_dim_names),
        "image_normalization": image_normalization,
        "action_representation": action_representation,
        "checkpoint_manifest": checkpoint_manifest,
        "dataset_manifest": dataset_manifest,
        "load_report": {
            "weight_file": load_report["weight_file"],
            "compat_key_remaps": load_report["compat_key_remaps"],
            "missing_key_count": len(load_report["missing_keys"]),
            "unexpected_key_count": len(load_report["unexpected_keys"]),
            "first_missing_keys": load_report["missing_keys"][:10],
            "first_unexpected_keys": load_report["unexpected_keys"][:10],
        },
        "metrics_env_delta": _metrics(pred_delta_np, gt_delta_np, action_dim_names),
        "metrics_model_action": _metrics(pred_model_np, gt_model_np, action_dim_names),
    }
    _write_json(args.output_dir / "summary.json", summary)
    _write_json(args.output_dir / "selected_indices.json", {"indices": selected_indices})

    print(f"[DONE] output_dir={args.output_dir}")
    print(
        "[METRIC] env_delta "
        f"mse={summary['metrics_env_delta'].get('mse', float('nan')):.6g} "
        f"mae={summary['metrics_env_delta'].get('mae', float('nan')):.6g} "
        f"rows={summary['metrics_env_delta'].get('rows', 0)}"
    )
    print(
        "[METRIC] model_action "
        f"mse={summary['metrics_model_action'].get('mse', float('nan')):.6g} "
        f"mae={summary['metrics_model_action'].get('mae', float('nan')):.6g}"
    )


if __name__ == "__main__":
    main()
