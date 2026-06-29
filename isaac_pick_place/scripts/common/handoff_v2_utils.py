"""Shared Handoff V2 utilities for dataset conversion, training, and eval.

The helpers in this module intentionally avoid Isaac imports so they can be used
from both the LeRobot training venv and the IsaacLab runtime.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGENET_IMAGE_STD = (0.229, 0.224, 0.225)

IMAGE_NORMALIZATION_STATS = {
    "clip": (CLIP_IMAGE_MEAN, CLIP_IMAGE_STD),
    "imagenet": (IMAGENET_IMAGE_MEAN, IMAGENET_IMAGE_STD),
}

STATE_TIMING_EXACT_PRE_ACTION = "exact_pre_action"
DATASET_VERSION_V2_CLEAN = "handoff_v2_clean_control"
DATASET_VERSION_V2_FULL = "handoff_v2_full"
ACTION_REPRESENTATION_DELTA_STEP = "delta_step"
ACTION_REPRESENTATION_RELATIVE_CURRENT_POSE_CHUNK = "relative_current_pose_chunk"
ACTION_REPRESENTATION_ABSOLUTE_JOINT_POS = "absolute_joint_pos"
ACTION_REPRESENTATIONS = (
    ACTION_REPRESENTATION_DELTA_STEP,
    ACTION_REPRESENTATION_RELATIVE_CURRENT_POSE_CHUNK,
    ACTION_REPRESENTATION_ABSOLUTE_JOINT_POS,
)
ACTION_MOTION_SLICES = ((0, 6), (7, 13))


def normalize_quat_wxyz_np(quat: np.ndarray, source: str = "quat") -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-8:
        raise ValueError(f"{source} has near-zero norm")
    return (quat / norm).astype(np.float32)


def canonicalize_quat_wxyz_np(quat: np.ndarray, source: str = "quat") -> np.ndarray:
    quat = normalize_quat_wxyz_np(quat, source)
    pivot = int(np.argmax(np.abs(quat)))
    if float(quat[pivot]) < 0.0:
        quat = -quat
    return quat.astype(np.float32)


def image_stats_for_normalization(mode: str) -> dict[str, list[list[list[float]]]]:
    mode = mode.lower()
    if mode not in IMAGE_NORMALIZATION_STATS:
        raise ValueError(f"Unsupported image normalization {mode!r}; choose one of {sorted(IMAGE_NORMALIZATION_STATS)}")
    mean, std = IMAGE_NORMALIZATION_STATS[mode]
    return {
        "min": [[[0.0]], [[0.0]], [[0.0]]],
        "max": [[[1.0]], [[1.0]], [[1.0]]],
        "mean": [[[float(value)]] for value in mean],
        "std": [[[float(value)]] for value in std],
    }


def load_manifest(dataset_dir: Path) -> dict[str, Any] | None:
    path = dataset_dir / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(dataset_dir: Path, manifest: dict[str, Any]) -> None:
    path = dataset_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def median_episode_length(lengths: list[int], default: int = 1845) -> int:
    if not lengths:
        return int(default)
    values = sorted(int(length) for length in lengths)
    return int(values[len(values) // 2])


def quat_continuity_report(states: list[np.ndarray], state_names: list[str]) -> dict[str, dict[str, float | int]]:
    """Return adjacent quaternion continuity statistics for named state vectors."""
    if not states:
        return {}
    stacked = np.stack(states, axis=0).astype(np.float32)
    name_to_index = {name: index for index, name in enumerate(state_names)}
    prefixes = (
        "left_tcp_quat_w",
        "right_tcp_quat_w",
        "right_tcp_quat_in_left_tcp_frame",
        "left_tcp_quat_in_right_tcp_frame",
    )
    report: dict[str, dict[str, float | int]] = {}
    for prefix in prefixes:
        indices = [name_to_index.get(f"{prefix}.{i}") for i in range(4)]
        if any(index is None for index in indices):
            continue
        quat = stacked[:, [int(index) for index in indices]]
        norms = np.linalg.norm(quat, axis=1)
        dots = np.sum(quat[1:] * quat[:-1], axis=1) if len(quat) > 1 else np.asarray([], dtype=np.float32)
        adjacent_diff = np.linalg.norm(quat[1:] - quat[:-1], axis=1) if len(quat) > 1 else np.asarray([], dtype=np.float32)
        report[prefix] = {
            "count": int(len(quat)),
            "sign_flip_count": int(np.sum(dots < 0.0)),
            "min_w": float(np.min(quat[:, 0])),
            "min_norm": float(np.min(norms)),
            "max_norm": float(np.max(norms)),
            "max_adjacent_diff": float(np.max(adjacent_diff)) if adjacent_diff.size else 0.0,
        }
    return report


def action_delta_chunk_to_relative_current_np(actions: np.ndarray) -> np.ndarray:
    """Convert per-step delta action chunks to current-observation relative chunks.

    Raw actions remain the environment's single-step delta command. For action
    chunks, each future motion command is accumulated so every chunk element is
    represented relative to the observation time. Gripper commands are kept as
    per-step absolute open/close commands.
    """
    values = np.asarray(actions, dtype=np.float32)
    if values.shape[-1] != 14:
        raise ValueError(f"Expected action dim 14, got shape={values.shape}")
    if values.ndim < 2:
        return values.copy()
    output = values.copy()
    for start, end in ACTION_MOTION_SLICES:
        output[..., start:end] = np.cumsum(values[..., start:end], axis=-2)
    return output.astype(np.float32)


def action_delta_chunk_to_relative_current_torch(actions):
    """Torch equivalent of :func:`action_delta_chunk_to_relative_current_np`."""
    if actions.shape[-1] != 14:
        raise ValueError(f"Expected action dim 14, got shape={tuple(actions.shape)}")
    if actions.ndim < 3:
        return actions
    output = actions.clone()
    for start, end in ACTION_MOTION_SLICES:
        output[..., start:end] = torch_cumsum(actions[..., start:end], dim=-2)
    return output


def torch_cumsum(values, dim: int):
    import torch

    return torch.cumsum(values, dim=dim)


def relative_current_action_to_delta_step_torch(current_relative_action, previous_relative_action=None):
    """Convert one queued current-relative action back to an env delta command."""
    import torch

    if current_relative_action.shape[-1] != 14:
        raise ValueError(f"Expected action dim 14, got shape={tuple(current_relative_action.shape)}")
    if previous_relative_action is None:
        previous_relative_action = torch.zeros_like(current_relative_action)
    env_action = current_relative_action.clone()
    for start, end in ACTION_MOTION_SLICES:
        env_action[..., start:end] = current_relative_action[..., start:end] - previous_relative_action[..., start:end]
    next_previous = current_relative_action.detach().clone()
    return env_action, next_previous


def resize_with_pad_torch(images, size: int = 224, pad_value: float = 0.0):
    """Resize channel-first torch images to a square canvas with padding.

    Accepts shapes [N,C,H,W] and returns [N,C,size,size].
    """
    import torch
    import torch.nn.functional as F

    if images.ndim != 4:
        raise ValueError(f"Expected [N,C,H,W] images, got shape={tuple(images.shape)}")
    _, _, height, width = images.shape
    scale = min(size / float(height), size / float(width))
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    resized = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
    pad_top = (size - new_h) // 2
    pad_bottom = size - new_h - pad_top
    pad_left = (size - new_w) // 2
    pad_right = size - new_w - pad_left
    return F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=float(pad_value))


def preprocess_eval_images_torch(images, image_normalization: str = "clip", size: int = 224):
    """Deterministic eval image preprocessing before policy normalization."""
    flat, shape = _flatten_camera_batch(images)
    flat = resize_with_pad_torch(flat, size=size)
    restored = _unflatten_camera_batch(flat.clamp(0.0, 1.0), shape)
    return _restore_input_rank(restored, shape)


def augment_train_images_torch(
    images,
    feature_key: str,
    image_normalization: str = "clip",
    augmentation: str = "none",
    size: int = 224,
):
    """Apply V2 train-time image processing and optional light augmentation.

    The same random parameters are used for all temporal frames of one sample.
    """
    import torch
    import torch.nn.functional as F

    flat, shape = _flatten_camera_batch(images)
    images_5d = _ensure_batched_temporal(images)
    batch, steps, channels, _, _ = images_5d.shape
    x = resize_with_pad_torch(flat, size=size).reshape(batch, steps, channels, size, size)
    if augmentation not in ("abc_top", "handoff_v2_full"):
        return _restore_input_rank(x.clamp(0.0, 1.0), shape)

    term_name = feature_key.removeprefix("observation.images.")
    if term_name == "global_rgb":
        angles = (torch.rand(batch, device=x.device, dtype=x.dtype) * 4.0 - 2.0) * math.pi / 180.0
        x = _rotate_temporal_batch(x, angles)
        x = _random_crop_resize_temporal(x, min_scale=0.95, max_scale=0.95, size=size)
    elif term_name in ("wrist_rgb", "observer_wrist_rgb"):
        x = _random_crop_resize_temporal(x, min_scale=0.95, max_scale=1.0, size=size)

    brightness = torch.empty(batch, 1, 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(0.7, 1.3)
    contrast = torch.empty(batch, 1, 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(0.6, 1.4)
    saturation = torch.empty(batch, 1, 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(0.5, 1.5)
    x = x * brightness
    channel_mean = x.mean(dim=(-3, -2, -1), keepdim=True)
    x = (x - channel_mean) * contrast + channel_mean
    gray = (0.2989 * x[:, :, 0:1] + 0.5870 * x[:, :, 1:2] + 0.1140 * x[:, :, 2:3])
    x = (x - gray) * saturation + gray
    x = x.clamp(0.0, 1.0)
    return _restore_input_rank(x, shape)


def _ensure_batched_temporal(images):
    if images.ndim == 5:
        return images
    if images.ndim == 4:
        return images.unsqueeze(1)
    raise ValueError(f"Expected image tensor [B,S,C,H,W] or [B,C,H,W], got shape={tuple(images.shape)}")


def _flatten_camera_batch(images):
    x = _ensure_batched_temporal(images)
    batch, steps, channels, height, width = x.shape
    return x.reshape(batch * steps, channels, height, width), (images.ndim, batch, steps, channels, height, width)


def _unflatten_camera_batch(flat, shape):
    _, batch, steps, channels, height, width = shape
    return flat.reshape(batch, steps, channels, flat.shape[-2], flat.shape[-1])


def _restore_input_rank(x, shape):
    original_ndim = shape[0]
    if original_ndim == 4:
        return x[:, 0]
    return x


def _rotate_temporal_batch(images, angles):
    import torch
    import torch.nn.functional as F

    batch, steps, channels, height, width = images.shape
    cos = torch.cos(angles).repeat_interleave(steps)
    sin = torch.sin(angles).repeat_interleave(steps)
    theta = torch.zeros((batch * steps, 2, 3), device=images.device, dtype=images.dtype)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    flat = images.reshape(batch * steps, channels, height, width)
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    rotated = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return rotated.reshape(batch, steps, channels, height, width)


def _random_crop_resize_temporal(images, min_scale: float, max_scale: float, size: int):
    import torch
    import torch.nn.functional as F

    batch, steps, channels, height, width = images.shape
    output = torch.empty((batch, steps, channels, size, size), device=images.device, dtype=images.dtype)
    scales = torch.empty(batch, device=images.device).uniform_(min_scale, max_scale)
    for batch_index in range(batch):
        crop_h = max(1, int(round(float(height) * float(scales[batch_index]))))
        crop_w = max(1, int(round(float(width) * float(scales[batch_index]))))
        top_max = height - crop_h
        left_max = width - crop_w
        top = int(torch.randint(0, top_max + 1, (1,), device=images.device).item()) if top_max > 0 else 0
        left = int(torch.randint(0, left_max + 1, (1,), device=images.device).item()) if left_max > 0 else 0
        crop = images[batch_index, :, :, top : top + crop_h, left : left + crop_w]
        flat = crop.reshape(steps, channels, crop_h, crop_w)
        output[batch_index] = F.interpolate(
            flat,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return output
