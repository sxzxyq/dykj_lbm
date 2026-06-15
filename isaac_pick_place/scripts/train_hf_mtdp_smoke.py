"""Smoke-train HF/LeRobot MultiTask DiT on a local LeRobot dataset.

This is intentionally small and explicit. It uses the official LeRobot
``MultiTaskDiTPolicy`` while slicing ``observation.state`` down to one of:

    joint_ee: joint_pos(9) + ee_position(3) + ee_quat(4)
    ee_only:  ee_position(3) + ee_quat(4)

The dataset may contain richer state vectors; this script keeps the model input
surface aligned with the first visuomotor training plan.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import signal
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


DEFAULT_DATASET_DIR = Path(
    "/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/lerobot_datasets/lerobot_ee_pose_smoke"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_ee_pose_smoke"
)

STATE_MODES = {
    "joint_ee": {
        "indices": tuple(range(0, 9)) + tuple(range(18, 25)),
        "names": (
            *(f"joint_pos.{i}" for i in range(9)),
            *(f"ee_position.{i}" for i in range(3)),
            *(f"ee_quat.{i}" for i in range(4)),
        ),
    },
    "ee_only": {
        "indices": tuple(range(18, 25)),
        "names": (
            *(f"ee_position.{i}" for i in range(3)),
            *(f"ee_quat.{i}" for i in range(4)),
        ),
    },
}
IMAGE_KEYS = (
    "observation.images.wrist_rgb",
    "observation.images.observer_wrist_rgb",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--repo-id", type=str, default=None, help="LeRobot repo id. Defaults to dataset dir name.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional existing policy checkpoint directory to resume weights from.",
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--save-freq", type=int, default=0, help="Save checkpoint_N every N optimizer steps. Use 0 to disable.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Accumulate gradients over this many micro-batches per optimizer step.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--n-action-steps", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-train-timesteps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--state-mode",
        type=str,
        default="joint_ee",
        choices=tuple(STATE_MODES),
        help="Low-dimensional state slice to feed the policy.",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        type=Path,
        default=None,
        help="Optional TensorBoard event output directory. Disabled when omitted.",
    )
    parser.add_argument(
        "--tensorboard-flush-every",
        type=int,
        default=10,
        help="Flush TensorBoard events every N optimizer steps. Use 0 to flush only at checkpoints/end.",
    )
    parser.add_argument(
        "--tensorboard-flush-secs",
        type=int,
        default=5,
        help="TensorBoard SummaryWriter flush_secs value.",
    )
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use local Hugging Face cache for CLIP weights/tokenizer.",
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="torchcodec",
        choices=("torchcodec", "pyav"),
        help="LeRobot video decoder backend.",
    )
    return parser.parse_args()


def _mock_groot_imports() -> None:
    """Work around a Python 3.12 dataclass issue in lerobot.policies.groot."""
    groot_pkg = MagicMock(__path__=[])
    sys.modules["lerobot.policies.groot"] = groot_pkg
    sys.modules["lerobot.policies.groot.configuration_groot"] = MagicMock()
    sys.modules["lerobot.policies.groot.modeling_groot"] = MagicMock()
    sys.modules["lerobot.policies.groot.groot_n1"] = MagicMock()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _feature_shape_hwc_to_chw(shape: tuple[int, ...]) -> tuple[int, int, int]:
    if len(shape) != 3:
        raise ValueError(f"Expected HWC image feature shape, got {shape}")
    h, w, c = shape
    return (c, h, w)


def _state_mode_spec(mode: str) -> tuple[tuple[int, ...], tuple[str, ...]]:
    spec = STATE_MODES[mode]
    return spec["indices"], spec["names"]


def _build_features(meta, state_indices: tuple[int, ...]):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.utils.constants import ACTION, OBS_STATE

    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(len(state_indices),)),
    }
    for key in IMAGE_KEYS:
        if key not in meta.features:
            raise KeyError(f"Dataset is missing required image feature: {key}")
        input_features[key] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=_feature_shape_hwc_to_chw(tuple(meta.features[key]["shape"])),
        )

    if ACTION not in meta.features:
        raise KeyError(f"Dataset is missing required action feature: {ACTION}")
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=tuple(meta.features[ACTION]["shape"])),
    }
    return input_features, output_features


def _delta_timestamps(config, fps: int) -> dict[str, list[float]]:
    keys = list(config.input_features) + list(config.output_features)
    result = {}
    for key in keys:
        if key in config.output_features:
            indices = config.action_delta_indices
        else:
            indices = config.observation_delta_indices
        result[key] = [idx / fps for idx in indices]
    return result


def _state_index_tensor(device: torch.device, state_indices: tuple[int, ...]) -> torch.Tensor:
    return torch.tensor(state_indices, dtype=torch.long, device=device)


def _to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    return value


def _stats_to_tensors(stats: dict, device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    tensor_stats: dict[str, dict[str, torch.Tensor]] = {}
    for key, values in stats.items():
        tensor_stats[key] = {}
        for stat_key, stat_value in values.items():
            if stat_key == "count":
                continue
            tensor_stats[key][stat_key] = torch.tensor(stat_value, dtype=torch.float32, device=device)
    return tensor_stats


def _slice_state_stats(stats: dict[str, dict[str, torch.Tensor]], state_indices: torch.Tensor) -> dict:
    state_stats = stats["observation.state"]
    for key, value in list(state_stats.items()):
        if value.ndim == 1 and value.shape[0] >= int(state_indices.max().item()) + 1:
            state_stats[key] = value.index_select(0, state_indices)
    return stats


def _normalization_mode(config, feature_key: str):
    if feature_key in config.input_features:
        feature_type = config.input_features[feature_key].type
    else:
        feature_type = config.output_features[feature_key].type
    feature_type = getattr(feature_type, "value", feature_type)
    return config.normalization_mapping[feature_type]


def _normalize_tensor(tensor: torch.Tensor, stat: dict[str, torch.Tensor], mode, eps: float = 1.0e-6) -> torch.Tensor:
    mode = getattr(mode, "value", mode)
    if mode == "IDENTITY" or not stat:
        return tensor
    if mode == "MIN_MAX":
        min_val = stat["min"]
        max_val = stat["max"]
        return 2.0 * (tensor - min_val) / torch.clamp(max_val - min_val, min=eps) - 1.0
    if mode == "MEAN_STD":
        std = stat["std"]
        # Tiny smoke datasets can have zero image std. Keep the smoke numerically tame.
        std = torch.where(std.abs() < eps, torch.ones_like(std), std)
        return (tensor - stat["mean"]) / std
    raise ValueError(f"Unsupported normalization mode: {mode}")


def _prepare_batch(
    batch: dict,
    config,
    tokenizer,
    stats: dict,
    device: torch.device,
    state_indices: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

    state_index_tensor = _state_index_tensor(device, state_indices)
    prepared: dict[str, torch.Tensor] = {}

    state = batch[OBS_STATE].to(device, non_blocking=True).index_select(-1, state_index_tensor)
    prepared[OBS_STATE] = _normalize_tensor(state, stats[OBS_STATE], _normalization_mode(config, OBS_STATE))

    action = batch[ACTION].to(device, non_blocking=True)
    prepared[ACTION] = _normalize_tensor(action, stats[ACTION], _normalization_mode(config, ACTION))

    for key in IMAGE_KEYS:
        image = batch[key].to(device, non_blocking=True)
        prepared[key] = _normalize_tensor(image, stats[key], _normalization_mode(config, key))

    tasks = batch.get("task")
    if tasks is None:
        tasks = ["Pick up the cube and place it on the red target area."] * state.shape[0]
    tokens = tokenizer(
        list(tasks),
        max_length=config.tokenizer_max_length,
        padding=config.tokenizer_padding,
        truncation=config.tokenizer_truncation,
        return_tensors="pt",
    )
    prepared[OBS_LANGUAGE_TOKENS] = tokens["input_ids"].to(device, non_blocking=True)
    prepared[OBS_LANGUAGE_ATTENTION_MASK] = tokens["attention_mask"].to(device, non_blocking=True)

    return prepared


def _write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _build_summary_writer(log_dir: Path | None, flush_secs: int):
    if log_dir is None:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard logging was requested, but the 'tensorboard' package is not installed "
            "in the training Python environment. Install it in "
            "/home/ubuntu/Workspace/multitask_dit_policy/.venv, or rerun with TENSORBOARD=0."
        ) from exc

    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir), flush_secs=flush_secs)
    atexit.register(writer.close)
    print(f"[INFO] tensorboard_log_dir={log_dir}", flush=True)
    return writer


def _tensorboard_config_text(
    args: argparse.Namespace,
    repo_id: str,
    config,
    meta,
    state_names: tuple[str, ...],
) -> str:
    rows = {
        "dataset_dir": str(args.dataset_dir),
        "repo_id": repo_id,
        "output_dir": str(args.output_dir),
        "checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path is not None else "<fresh>",
        "fps": meta.fps,
        "steps": args.steps,
        "save_freq": args.save_freq,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(config.device),
        "horizon": config.horizon,
        "n_obs_steps": config.n_obs_steps,
        "n_action_steps": config.n_action_steps,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "num_train_timesteps": config.num_train_timesteps,
        "image_size": args.image_size,
        "video_backend": args.video_backend,
        "state_mode": args.state_mode,
        "state_keep_names": ", ".join(state_names),
    }
    return "\n".join(f"- **{key}**: `{value}`" for key, value in rows.items())


def _stats_to_jsonable(stats: dict[str, dict[str, torch.Tensor]]) -> dict:
    result = {}
    for key, values in stats.items():
        result[key] = {}
        for stat_key, value in values.items():
            result[key][stat_key] = value.detach().cpu().tolist()
    return result


def _save_policy_dir(policy, stats: dict[str, dict[str, torch.Tensor]], checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(checkpoint_dir)
    _write_summary(checkpoint_dir / "dataset_stats.json", _stats_to_jsonable(stats))


def main() -> None:
    args = _parse_args()
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be positive.")
    if args.save_freq < 0:
        raise ValueError("--save-freq must be non-negative.")
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.checkpoint_path is not None and not args.checkpoint_path.exists():
        raise FileNotFoundError(args.checkpoint_path)

    _mock_groot_imports()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
    from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy
    from transformers import CLIPTokenizer

    if not args.dataset_dir.exists():
        raise FileNotFoundError(args.dataset_dir)

    _seed_everything(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    state_indices, state_names = _state_mode_spec(args.state_mode)

    repo_id = args.repo_id or args.dataset_dir.name
    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=args.dataset_dir)
    input_features, output_features = _build_features(meta, state_indices)

    config = MultiTaskDiTConfig(
        n_obs_steps=args.n_obs_steps,
        horizon=args.horizon,
        n_action_steps=args.n_action_steps,
        input_features=input_features,
        output_features=output_features,
        device=str(device),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_train_timesteps=args.num_train_timesteps,
        optimizer_lr=args.lr,
        image_resize_shape=(args.image_size, args.image_size),
        image_crop_shape=(args.image_size, args.image_size),
        image_crop_is_random=False,
        push_to_hub=False,
    )

    delta_timestamps = _delta_timestamps(config, meta.fps)
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=args.dataset_dir,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    stats = _slice_state_stats(_stats_to_tensors(meta.stats, device), _state_index_tensor(device, state_indices))
    tokenizer = CLIPTokenizer.from_pretrained(config.text_encoder_name)
    if args.checkpoint_path is None:
        policy = MultiTaskDiTPolicy(config).to(device)
    else:
        policy = MultiTaskDiTPolicy.from_pretrained(
            args.checkpoint_path,
            config=config,
            local_files_only=True,
        ).to(device)
        print(f"[INFO] resumed policy weights from {args.checkpoint_path}", flush=True)
    policy.train()

    optimizer = torch.optim.Adam(policy.get_optim_params(), lr=args.lr, betas=config.optimizer_betas, eps=config.optimizer_eps)
    writer = _build_summary_writer(args.tensorboard_log_dir, args.tensorboard_flush_secs)
    if writer is not None:
        writer.add_text("config/summary", _tensorboard_config_text(args, repo_id, config, meta, state_names), 0)
        writer.add_scalar("config/effective_batch_size", args.batch_size * args.grad_accum_steps, 0)
        writer.add_scalar("config/lr", args.lr, 0)
        writer.add_scalar("config/state_dim", len(state_indices), 0)
        writer.add_scalar("dataset/fps", meta.fps, 0)
        writer.add_scalar("dataset/frames", len(dataset), 0)
        writer.flush()

    stop_requested = {"value": False, "signal": ""}

    def _handle_stop(signum, _frame):
        stop_requested["value"] = True
        stop_requested["signal"] = signal.Signals(signum).name
        print(f"[WARN] received {stop_requested['signal']}; will save an interrupted checkpoint after this step", flush=True)

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    losses: list[float] = []
    iterator = iter(dataloader)
    completed_steps = 0
    interrupted_by = None
    progress = tqdm(range(args.steps), desc="hf-mtdp-smoke")
    for step in progress:
        if stop_requested["value"]:
            interrupted_by = stop_requested["signal"]
            break

        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        for _ in range(args.grad_accum_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)

            train_batch = _prepare_batch(batch, config, tokenizer, stats, device, state_indices)
            loss, _ = policy(train_batch)
            (loss / args.grad_accum_steps).backward()
            micro_losses.append(float(loss.detach().cpu().item()))

        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        loss_value = float(np.mean(micro_losses))
        losses.append(loss_value)
        progress.set_postfix(loss=f"{loss_value:.4f}")
        global_step = step + 1
        completed_steps = global_step
        if writer is not None:
            writer.add_scalar("train/loss", loss_value, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("train/grad_norm", float(grad_norm.detach().cpu().item()), global_step)
            writer.add_scalar("train/micro_loss_min", float(np.min(micro_losses)), global_step)
            writer.add_scalar("train/micro_loss_max", float(np.max(micro_losses)), global_step)
            if args.tensorboard_flush_every > 0 and global_step % args.tensorboard_flush_every == 0:
                writer.flush()
        if (step + 1) % args.log_every == 0:
            print(
                f"[STEP {step + 1}/{args.steps}] loss={loss_value:.6f} "
                f"micro_batch={args.batch_size} grad_accum={args.grad_accum_steps}",
                flush=True,
            )
        if args.save_freq > 0 and (step + 1) % args.save_freq == 0:
            checkpoint_dir = args.output_dir / f"checkpoint_{step + 1}"
            _save_policy_dir(policy, stats, checkpoint_dir)
            print(f"[SAVE] wrote checkpoint to {checkpoint_dir}", flush=True)
            if writer is not None:
                writer.add_text("checkpoint/latest", str(checkpoint_dir), step + 1)
                writer.flush()
        if stop_requested["value"]:
            interrupted_by = stop_requested["signal"]
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if interrupted_by is None:
        model_dir = args.output_dir / "final_model"
        summary_name = "smoke_summary.json"
    else:
        model_dir = args.output_dir / f"interrupted_step_{completed_steps}"
        summary_name = "interrupted_summary.json"

    _save_policy_dir(policy, stats, model_dir)
    _write_summary(
        args.output_dir / summary_name,
        {
            "dataset_dir": str(args.dataset_dir),
            "repo_id": repo_id,
            "output_dir": str(args.output_dir),
            "checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path is not None else None,
            "interrupted": interrupted_by is not None,
            "interrupted_by": interrupted_by,
            "completed_steps": completed_steps,
            "steps": args.steps,
            "save_freq": args.save_freq,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
            "tensorboard_log_dir": str(args.tensorboard_log_dir) if args.tensorboard_log_dir is not None else None,
            "losses": losses,
            "state_mode": args.state_mode,
            "state_keep_indices": list(state_indices),
            "state_keep_names": list(state_names),
            "input_features": {key: {"type": value.type.value, "shape": list(value.shape)} for key, value in input_features.items()},
            "output_features": {
                key: {"type": value.type.value, "shape": list(value.shape)} for key, value in output_features.items()
            },
            "config": {
                "horizon": config.horizon,
                "n_obs_steps": config.n_obs_steps,
                "n_action_steps": config.n_action_steps,
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_layers,
                "num_heads": config.num_heads,
                "num_train_timesteps": config.num_train_timesteps,
            },
        },
    )
    if writer is not None:
        tag = "checkpoint/final_model" if interrupted_by is None else "checkpoint/interrupted_model"
        writer.add_text(tag, str(model_dir), completed_steps)
        writer.flush()
        writer.close()
    if interrupted_by is None:
        print(f"[DONE] wrote smoke checkpoint to {model_dir}", flush=True)
    else:
        print(f"[INTERRUPTED] wrote interrupted checkpoint to {model_dir}", flush=True)


if __name__ == "__main__":
    main()
