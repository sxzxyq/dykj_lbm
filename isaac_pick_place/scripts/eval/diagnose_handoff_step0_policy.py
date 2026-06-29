"""Diagnose step-0 handoff policy inputs, sampling, and action queue behavior."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import types
from typing import Any
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "experiments"
    / "training_runs"
    / "hf_mtdp_handoff_3cam_joint_ee_birelpose_time_100success_bs16acc4_30k"
    / "final_model"
)
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "lerobot_datasets"
    / "lerobot_handoff_handoff_100_joint_ee_3cam_v1_birelpose_time"
)
DEFAULT_TASK = "Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0"
DEFAULT_TASK_TEXT = (
    "Right arm moves the blue cube to the yellow handoff area, then left arm moves it to the red target area."
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "reports"
IMAGE_TERM_BY_FEATURE = {
    "observation.images.wrist_rgb": "wrist_rgb",
    "observation.images.observer_wrist_rgb": "observer_wrist_rgb",
    "observation.images.global_rgb": "global_rgb",
}
CAMERA_NAME_BY_FEATURE = {
    "observation.images.wrist_rgb": "wrist_cam",
    "observation.images.observer_wrist_rgb": "observer_wrist_cam",
    "observation.images.global_rgb": "global_cam",
}
ACTION = "action"
OBS_STATE = "observation.state"
OBS_IMAGES = "observation.images"
OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"
TCP_OFFSET = (0.0, 0.0, 0.107)
OPEN_ACTION = 1.0
HANDOFF_TIME_TOTAL_STEPS = 1845
STATE_SEGMENTS_49D = (
    ("left_joint_pos", 0, 9),
    ("left_tcp_pos_w", 9, 12),
    ("left_tcp_quat_w", 12, 16),
    ("left_gripper_opening", 16, 17),
    ("right_joint_pos", 17, 26),
    ("right_tcp_pos_w", 26, 29),
    ("right_tcp_quat_w", 29, 33),
    ("right_gripper_opening", 33, 34),
    ("right_tcp_pos_in_left_tcp_frame", 34, 37),
    ("right_tcp_quat_in_left_tcp_frame", 37, 41),
    ("left_tcp_pos_in_right_tcp_frame", 41, 44),
    ("left_tcp_quat_in_right_tcp_frame", 44, 48),
    ("episode_progress", 48, 49),
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="49D policy checkpoint directory.")
parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="Matching LeRobot dataset root.")
parser.add_argument("--dataset-video-backend", choices=("pyav", "torchcodec"), default="pyav")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Isaac Lab task id.")
parser.add_argument("--task-text", type=str, default=DEFAULT_TASK_TEXT, help="Text instruction passed to the policy.")
parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Parent directory for report folders.")
parser.add_argument("--seed", type=int, default=2000, help="Seed for env reset and fixed-seed policy checks.")
parser.add_argument("--dataset-episode", type=int, default=0, help="Dataset episode index.")
parser.add_argument("--dataset-frame", type=int, default=0, help="Frame inside the dataset episode.")
parser.add_argument("--warmup-steps", type=int, default=2, help="Warmup env steps after reset.")
parser.add_argument(
    "--warmup-open-gripper",
    action="store_true",
    default=False,
    help="Set both grippers to OPEN_ACTION during warmup. Omit for all-zero warmup actions.",
)
parser.add_argument("--handoff-time-total-steps", type=int, default=HANDOFF_TIME_TOTAL_STEPS)
parser.add_argument("--same-seed-repeats", type=int, default=10, help="Repeat count for deterministic fixed-seed check.")
parser.add_argument("--seed-sweep-repeats", type=int, default=20, help="Repeat count for diffusion seed sweep.")
parser.add_argument("--n-action-steps", type=int, default=None, help="Optional checkpoint n_action_steps override.")
parser.add_argument("--num-inference-steps", type=int, default=None, help="Optional diffusion denoising-step override.")
parser.add_argument("--refresh-camera-xform", action="store_true", default=False)
parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True, help="Use local HF cache.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

_MISSING_POLICY_DEPS = [
    name
    for name in ("lerobot", "transformers", "diffusers", "safetensors", "draccus", "einops", "PIL")
    if importlib.util.find_spec(name) is None
]
if _MISSING_POLICY_DEPS:
    raise SystemExit(
        "[ERROR] Missing policy inference dependencies before launching Isaac: " + ", ".join(_MISSING_POLICY_DEPS)
    )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors_file

import isaaclab_tasks  # noqa: F401
import isaac_pick_place.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils import math as math_utils


def _mock_groot_imports() -> None:
    groot_pkg = MagicMock(__path__=[])
    sys.modules["lerobot.policies.groot"] = groot_pkg
    sys.modules["lerobot.policies.groot.configuration_groot"] = MagicMock()
    sys.modules["lerobot.policies.groot.modeling_groot"] = MagicMock()
    sys.modules["lerobot.policies.groot.groot_n1"] = MagicMock()


def _patch_lerobot_namespace_imports() -> None:
    import lerobot

    lerobot_root = Path(lerobot.__file__).resolve().parent
    for package_name in ("policies", "datasets", "optim"):
        full_name = f"lerobot.{package_name}"
        package = types.ModuleType(full_name)
        package.__path__ = [str(lerobot_root / package_name)]
        sys.modules[full_name] = package
    mtdp_package = types.ModuleType("lerobot.policies.multi_task_dit")
    mtdp_package.__path__ = [str(lerobot_root / "policies" / "multi_task_dit")]
    sys.modules["lerobot.policies.multi_task_dit"] = mtdp_package

    train_config_module = types.ModuleType("lerobot.configs.train")

    class TrainPipelineConfig:
        pass

    train_config_module.TrainPipelineConfig = TrainPipelineConfig
    sys.modules["lerobot.configs.train"] = train_config_module


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _chw_float_to_uint8_hwc(chw: torch.Tensor) -> np.ndarray:
    image = chw.detach().cpu().float().clamp(0.0, 1.0)
    if image.ndim == 4:
        image = image[0]
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    if image.shape[0] != 3:
        raise ValueError(f"Expected CHW image with 3 channels, got shape={tuple(image.shape)}")
    return (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def _save_chw_image(chw: torch.Tensor, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_chw_float_to_uint8_hwc(chw)).save(path)


def _save_image_comparison(dataset_chw: torch.Tensor, live_chw: torch.Tensor, output_dir: Path, stem: str) -> None:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_img = _chw_float_to_uint8_hwc(dataset_chw)
    live_img = _chw_float_to_uint8_hwc(live_chw)

    diff = (live_chw.detach().cpu().float() - dataset_chw.detach().cpu().float()).abs()
    if diff.ndim == 4:
        diff = diff[0]
    diff_gray = diff.mean(dim=0)
    diff_max = float(diff_gray.max().item())
    diff_vis = diff_gray / max(diff_max, 1.0 / 255.0)
    heatmap = np.zeros((*diff_vis.shape, 3), dtype=np.uint8)
    heatmap[..., 0] = (diff_vis.numpy() * 255.0).round().astype(np.uint8)
    heatmap[..., 1] = (diff_vis.numpy() * 64.0).round().astype(np.uint8)

    separator = np.full((dataset_img.shape[0], 6, 3), 255, dtype=np.uint8)
    side_by_side = np.concatenate([dataset_img, separator, live_img, separator, heatmap], axis=1)

    Image.fromarray(dataset_img).save(output_dir / f"{stem}_dataset.png")
    Image.fromarray(live_img).save(output_dir / f"{stem}_live.png")
    Image.fromarray(heatmap).save(output_dir / f"{stem}_absdiff_heatmap.png")
    Image.fromarray(side_by_side).save(output_dir / f"{stem}_side_by_side.png")


def _as_list(value: torch.Tensor) -> list[float]:
    return value.detach().cpu().reshape(-1).tolist()


def _stats(values: torch.Tensor) -> dict[str, float]:
    value = values.detach().cpu().float().reshape(-1)
    if value.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "max_abs": 0.0}
    return {
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "max_abs": float(value.abs().max().item()),
    }


def _load_stats(checkpoint: Path, device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    stats_path = checkpoint / "dataset_stats.json"
    raw = json.loads(stats_path.read_text(encoding="utf-8"))
    stats: dict[str, dict[str, torch.Tensor]] = {}
    for key, values in raw.items():
        stats[key] = {}
        for stat_key, stat_value in values.items():
            if stat_key == "count":
                continue
            stats[key][stat_key] = torch.tensor(stat_value, dtype=torch.float32, device=device)
    return stats


def _load_config_from_json(checkpoint: Path, config_cls, feature_type_cls, policy_feature_cls, device: torch.device):
    raw = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
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
        std = torch.where(stat["std"].abs() < eps, torch.ones_like(stat["std"]), stat["std"])
        return (tensor - stat["mean"]) / std
    raise ValueError(f"Unsupported normalization mode: {mode}")


def _unnormalize_tensor(tensor: torch.Tensor, stat: dict[str, torch.Tensor], mode) -> torch.Tensor:
    mode = getattr(mode, "value", mode)
    if mode == "IDENTITY" or not stat:
        return tensor
    if mode == "MIN_MAX":
        min_val = stat["min"]
        max_val = stat["max"]
        return (tensor + 1.0) * 0.5 * (max_val - min_val) + min_val
    if mode == "MEAN_STD":
        return tensor * stat["std"] + stat["mean"]
    raise ValueError(f"Unsupported normalization mode: {mode}")


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
        ("observation_encoder.vision_encoder.model.", "observation_encoder.vision_encoder.model.vision_model."),
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
    model_keys = set(policy.state_dict().keys())
    state_dict, remap_count = _remap_transformers5_clip_keys(state_dict, model_keys)
    missing_keys, unexpected_keys = policy.load_state_dict(state_dict, strict=False)
    return {
        "weight_file": str(weight_file),
        "compat_key_remaps": remap_count,
        "missing_keys": len(missing_keys),
        "unexpected_keys": len(unexpected_keys),
        "first_missing_keys": list(missing_keys)[:10],
        "first_unexpected_keys": list(unexpected_keys)[:10],
    }


def _policy_obs(obs) -> dict:
    if isinstance(obs, dict) and "policy" in obs:
        return obs["policy"]
    return {}


def _image_to_chw_float(image: torch.Tensor, device: torch.device) -> torch.Tensor:
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = image.to(device=device, dtype=torch.float32)
    if image.max() > 1.5:
        image = image / 255.0
    return image.permute(2, 0, 1).contiguous()


def _dataset_image_to_chw_float(image: torch.Tensor, device: torch.device) -> torch.Tensor:
    image = image.to(device=device, dtype=torch.float32)
    if image.ndim != 3:
        raise ValueError(f"Expected dataset image as CHW or HWC, got shape={tuple(image.shape)}")
    if image.shape[0] in (1, 3, 4):
        chw = image[:3] if image.shape[0] == 4 else image
    elif image.shape[-1] in (1, 3, 4):
        image = image[..., :3] if image.shape[-1] == 4 else image
        chw = image.permute(2, 0, 1).contiguous()
    else:
        raise ValueError(f"Could not infer channel dimension for dataset image shape={tuple(image.shape)}")
    if chw.max() > 1.5:
        chw = chw / 255.0
    return chw.contiguous()


def _asset(env, arm_name: str):
    return env.unwrapped.scene[arm_name]


def _body_id(env, arm_name: str, body_name: str = "panda_hand") -> int:
    body_ids, body_names = _asset(env, arm_name).find_bodies(body_name)
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected one body for {arm_name}:{body_name}, got {body_names}")
    return body_ids[0]


def _arm_gripper_joint_ids(env, arm_name: str) -> list[int]:
    joint_ids, joint_names = _asset(env, arm_name).find_joints(["panda_finger.*"])
    if not joint_ids:
        raise RuntimeError(f"Could not resolve gripper joints for {arm_name}: {joint_names}")
    return joint_ids


def _tcp_pos_w(env, arm_name: str) -> torch.Tensor:
    robot = _asset(env, arm_name)
    body_idx = _body_id(env, arm_name)
    hand_pos = robot.data.body_pos_w[:, body_idx, :]
    hand_quat = robot.data.body_quat_w[:, body_idx, :]
    offset = torch.tensor(TCP_OFFSET, device=env.unwrapped.device, dtype=hand_pos.dtype).repeat(env.unwrapped.num_envs, 1)
    return hand_pos + math_utils.quat_apply(hand_quat, offset)


def _tcp_quat_w(env, arm_name: str) -> torch.Tensor:
    robot = _asset(env, arm_name)
    body_idx = _body_id(env, arm_name)
    return robot.data.body_quat_w[:, body_idx, :]


def _normalize_quat_wxyz(quat: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return quat / torch.clamp(torch.linalg.vector_norm(quat, dim=-1, keepdim=True), min=eps)


def _canonicalize_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return torch.where(quat[..., :1] < 0.0, -quat, quat)


def _quat_conjugate_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat[..., :1], -quat[..., 1:]], dim=-1)


def _quat_multiply_wxyz(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dim=-1,
    )


def _relative_tcp_pose(env, frame_arm_name: str, target_arm_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    frame_pos = _tcp_pos_w(env, frame_arm_name)
    target_pos = _tcp_pos_w(env, target_arm_name)
    frame_quat = _normalize_quat_wxyz(_tcp_quat_w(env, frame_arm_name))
    target_quat = _normalize_quat_wxyz(_tcp_quat_w(env, target_arm_name))
    frame_inv = _quat_conjugate_wxyz(frame_quat)
    rel_pos = math_utils.quat_apply(frame_inv, target_pos - frame_pos)
    rel_quat = _quat_multiply_wxyz(frame_inv, target_quat)
    rel_quat = _canonicalize_quat_wxyz(_normalize_quat_wxyz(rel_quat))
    return rel_pos, rel_quat


def _arm_gripper_opening(env, arm_name: str) -> torch.Tensor:
    robot = _asset(env, arm_name)
    joint_ids = _arm_gripper_joint_ids(env, arm_name)
    return torch.sum(torch.abs(robot.data.joint_pos[:, joint_ids]), dim=1)


def _build_handoff_birelpose_time_state(env, device: torch.device, episode_progress: float) -> torch.Tensor:
    pieces = []
    for arm_name in ("robot", "observer_robot"):
        robot = _asset(env, arm_name)
        pieces.extend(
            [
                robot.data.joint_pos.to(device=device, dtype=torch.float32),
                _tcp_pos_w(env, arm_name).to(device=device, dtype=torch.float32),
                _tcp_quat_w(env, arm_name).to(device=device, dtype=torch.float32),
                _arm_gripper_opening(env, arm_name).unsqueeze(-1).to(device=device, dtype=torch.float32),
            ]
        )
    right_in_left_pos, right_in_left_quat = _relative_tcp_pose(env, "robot", "observer_robot")
    left_in_right_pos, left_in_right_quat = _relative_tcp_pose(env, "observer_robot", "robot")
    progress = torch.full((env.unwrapped.num_envs, 1), episode_progress, device=device, dtype=torch.float32)
    state = torch.cat(
        [
            *pieces,
            right_in_left_pos.to(device=device, dtype=torch.float32),
            right_in_left_quat.to(device=device, dtype=torch.float32),
            left_in_right_pos.to(device=device, dtype=torch.float32),
            left_in_right_quat.to(device=device, dtype=torch.float32),
            progress,
        ],
        dim=-1,
    )
    if state.shape[-1] != 49:
        raise ValueError(f"Expected 49D handoff state, got {tuple(state.shape)}")
    return state


def _refresh_camera_xforms(env, camera_names: list[str]) -> None:
    from pxr import Gf, UsdGeom

    from isaaclab.utils.math import convert_camera_frame_orientation_convention

    for camera_name in camera_names:
        if camera_name not in env.unwrapped.scene.sensors:
            print(f"[CAMERA refresh] {camera_name}: missing from scene sensors", flush=True)
            continue
        camera = env.unwrapped.scene[camera_name]
        cfg = camera.cfg
        rot = torch.tensor(cfg.offset.rot, dtype=torch.float32, device="cpu").unsqueeze(0)
        rot_offset = convert_camera_frame_orientation_convention(rot, origin=cfg.offset.convention, target="opengl")[0]
        orient = Gf.Quatd(
            float(rot_offset[0]),
            Gf.Vec3d(float(rot_offset[1]), float(rot_offset[2]), float(rot_offset[3])),
        )
        translate = Gf.Vec3d(*[float(value) for value in cfg.offset.pos])
        refreshed = 0
        for prim in getattr(camera, "_sensor_prims", []):
            xformable = UsdGeom.Xformable(prim)
            translate_op = None
            orient_op = None
            for op in xformable.GetOrderedXformOps():
                if op.GetOpName() == "xformOp:translate":
                    translate_op = op
                elif op.GetOpName() == "xformOp:orient":
                    orient_op = op
            if translate_op is None:
                translate_op = xformable.AddTranslateOp()
            if orient_op is None:
                orient_op = xformable.AddOrientOp()
            translate_op.Set(translate)
            orient_op.Set(orient)
            refreshed += 1
        print(f"[CAMERA refresh] {camera_name}: refreshed {refreshed} prim(s)", flush=True)


def _episode_bounds(dataset, episode_index: int, frame_index: int) -> tuple[int, dict]:
    for row in dataset.meta.episodes:
        if int(row["episode_index"]) == episode_index:
            if frame_index < 0 or frame_index >= int(row["length"]):
                raise ValueError(f"Dataset frame {frame_index} outside episode length {row['length']}")
            return int(row["dataset_from_index"]) + frame_index, dict(row)
    raise ValueError(f"Episode {episode_index} not found in dataset")


def _token_batch(tokenizer, config, device: torch.device, task_text: str) -> dict[str, torch.Tensor]:
    tokens = tokenizer(
        [task_text],
        max_length=config.tokenizer_max_length,
        padding=config.tokenizer_padding,
        truncation=config.tokenizer_truncation,
        return_tensors="pt",
    )
    return {
        OBS_LANGUAGE_TOKENS: tokens["input_ids"].to(device),
        OBS_LANGUAGE_ATTENTION_MASK: tokens["attention_mask"].to(device),
    }


def _normalize_policy_batch(
    unnormalized: dict[str, torch.Tensor],
    config,
    tokenizer,
    stats: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
    task_text: str,
) -> dict[str, torch.Tensor]:
    batch = {
        OBS_STATE: _normalize_tensor(
            unnormalized[OBS_STATE].to(device=device, dtype=torch.float32),
            stats[OBS_STATE],
            _normalization_mode(config, OBS_STATE),
        )
    }
    for feature_key in config.image_features:
        batch[feature_key] = _normalize_tensor(
            unnormalized[feature_key].to(device=device, dtype=torch.float32),
            stats[feature_key],
            _normalization_mode(config, feature_key),
        )
    batch.update(_token_batch(tokenizer, config, device, task_text))
    return batch


def _action_summary(action: torch.Tensor) -> dict[str, Any]:
    row = action.detach().cpu().float().reshape(-1)
    return {
        "full_action": row.tolist(),
        "left_xyz_norm": float(torch.linalg.vector_norm(row[0:3]).item()),
        "left_gripper": float(row[6].item()),
        "right_action_z": float(row[9].item()),
        "right_gripper": float(row[13].item()),
        "right_xyz_norm": float(torch.linalg.vector_norm(row[7:10]).item()),
    }


def _run_single_action(
    policy,
    batch: dict[str, torch.Tensor],
    stats: dict[str, dict[str, torch.Tensor]],
    config,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _seed_everything(seed)
    policy.reset()
    with torch.no_grad():
        action_norm = policy.select_action(dict(batch))
    action = _unnormalize_tensor(action_norm, stats[ACTION], _normalization_mode(config, ACTION))
    return action_norm.detach().clone(), action.detach().clone()


def _predict_executed_chunk(
    policy,
    batch: dict[str, torch.Tensor],
    stats: dict[str, dict[str, torch.Tensor]],
    config,
    seed: int,
):
    from lerobot.policies.utils import populate_queues

    _seed_everything(seed)
    policy.reset()
    with torch.no_grad():
        prepared = policy._prepare_batch(dict(batch))
        policy._queues = populate_queues(policy._queues, prepared)
        chunk_norm = policy.predict_action_chunk(prepared)
    chunk = _unnormalize_tensor(chunk_norm, stats[ACTION], _normalization_mode(config, ACTION))
    return chunk_norm.detach().clone(), chunk.detach().clone()


def _select_action_sequence(
    policy,
    batch: dict[str, torch.Tensor],
    stats: dict[str, dict[str, torch.Tensor]],
    config,
    seed: int,
    count: int,
) -> torch.Tensor:
    _seed_everything(seed)
    policy.reset()
    actions = []
    with torch.no_grad():
        for _ in range(count):
            action_norm = policy.select_action(dict(batch))
            action = _unnormalize_tensor(action_norm, stats[ACTION], _normalization_mode(config, ACTION))
            actions.append(action.detach().clone())
    return torch.stack(actions, dim=1)


def _action_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    full = torch.tensor([row["full_action"] for row in rows], dtype=torch.float32)
    return {
        "count": len(rows),
        "left_xyz_norm": _stats(torch.tensor([row["left_xyz_norm"] for row in rows])),
        "left_gripper": _stats(torch.tensor([row["left_gripper"] for row in rows])),
        "right_action_z": _stats(torch.tensor([row["right_action_z"] for row in rows])),
        "right_gripper": _stats(torch.tensor([row["right_gripper"] for row in rows])),
        "full_action_mean": full.mean(dim=0).tolist(),
        "full_action_std": full.std(dim=0, unbiased=False).tolist(),
        "full_action_min": full.min(dim=0).values.tolist(),
        "full_action_max": full.max(dim=0).values.tolist(),
    }


def _chunk_step_summary(chunks: list[torch.Tensor]) -> dict[str, Any]:
    if not chunks:
        return {}
    stacked = torch.cat([chunk.detach().cpu().float() for chunk in chunks], dim=0)
    # stacked shape: [num_seeds, n_action_steps, action_dim]
    left_xyz_norm = torch.linalg.vector_norm(stacked[..., 0:3], dim=-1)
    return {
        "count": int(stacked.shape[0]),
        "chunk_shape": list(stacked.shape[1:]),
        "right_action_z_by_step": {
            "mean": stacked[..., 9].mean(dim=0).tolist(),
            "std": stacked[..., 9].std(dim=0, unbiased=False).tolist(),
            "min": stacked[..., 9].min(dim=0).values.tolist(),
            "max": stacked[..., 9].max(dim=0).values.tolist(),
        },
        "left_xyz_norm_by_step": {
            "mean": left_xyz_norm.mean(dim=0).tolist(),
            "std": left_xyz_norm.std(dim=0, unbiased=False).tolist(),
            "min": left_xyz_norm.min(dim=0).values.tolist(),
            "max": left_xyz_norm.max(dim=0).values.tolist(),
        },
        "left_gripper_by_step": {
            "mean": stacked[..., 6].mean(dim=0).tolist(),
            "std": stacked[..., 6].std(dim=0, unbiased=False).tolist(),
            "min": stacked[..., 6].min(dim=0).values.tolist(),
            "max": stacked[..., 6].max(dim=0).values.tolist(),
        },
        "right_gripper_by_step": {
            "mean": stacked[..., 13].mean(dim=0).tolist(),
            "std": stacked[..., 13].std(dim=0, unbiased=False).tolist(),
            "min": stacked[..., 13].min(dim=0).values.tolist(),
            "max": stacked[..., 13].max(dim=0).values.tolist(),
        },
    }


def main() -> None:
    if args_cli.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not args_cli.checkpoint.exists():
        raise FileNotFoundError(args_cli.checkpoint)
    if not args_cli.dataset_dir.exists():
        raise FileNotFoundError(args_cli.dataset_dir)
    if args_cli.same_seed_repeats <= 0 or args_cli.seed_sweep_repeats <= 0:
        raise ValueError("Repeat counts must be positive")
    if args_cli.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args_cli.handoff_time_total_steps <= 0:
        raise ValueError("--handoff-time-total-steps must be positive")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    input_split_dir = args_cli.output_root / f"input_split_step0_{timestamp}"
    sampling_dir = args_cli.output_root / f"sampling_queue_step0_{timestamp}"
    input_split_dir.mkdir(parents=True, exist_ok=True)
    sampling_dir.mkdir(parents=True, exist_ok=True)

    _patch_lerobot_namespace_imports()
    _mock_groot_imports()
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
    from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy
    from transformers import CLIPTokenizer

    device = torch.device(args_cli.device if args_cli.device == "cpu" or torch.cuda.is_available() else "cpu")
    _seed_everything(args_cli.seed)

    config = _load_config_from_json(args_cli.checkpoint, MultiTaskDiTConfig, FeatureType, PolicyFeature, device)
    if args_cli.n_action_steps is not None:
        config.n_action_steps = args_cli.n_action_steps
    if args_cli.num_inference_steps is not None:
        config.num_inference_steps = args_cli.num_inference_steps
    if tuple(config.input_features[OBS_STATE].shape) != (49,):
        raise ValueError(f"This diagnostic expects a 49D state checkpoint, got {config.input_features[OBS_STATE].shape}")

    stats = _load_stats(args_cli.checkpoint, device)
    tokenizer = CLIPTokenizer.from_pretrained(config.text_encoder_name)
    policy = MultiTaskDiTPolicy(config).to(device)
    weight_info = _load_policy_weights(policy, args_cli.checkpoint, device)
    policy.eval()

    dataset = LeRobotDataset(
        repo_id=args_cli.dataset_dir.name,
        root=args_cli.dataset_dir,
        video_backend=args_cli.dataset_video_backend,
    )
    dataset_index, episode_row = _episode_bounds(dataset, args_cli.dataset_episode, args_cli.dataset_frame)
    sample = dataset[dataset_index]

    dataset_unorm: dict[str, torch.Tensor] = {
        OBS_STATE: sample[OBS_STATE].to(device=device, dtype=torch.float32).unsqueeze(0)
    }
    for feature_key in config.image_features:
        dataset_unorm[feature_key] = _dataset_image_to_chw_float(sample[feature_key], device).unsqueeze(0)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        reset_out = env.reset(seed=args_cli.seed)
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        camera_names = [CAMERA_NAME_BY_FEATURE[key] for key in config.image_features]
        if args_cli.refresh_camera_xform:
            _refresh_camera_xforms(env, camera_names)
        if args_cli.warmup_steps > 0:
            warmup_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            if args_cli.warmup_open_gripper and warmup_actions.shape[1] > 6:
                warmup_actions[:, 6] = OPEN_ACTION
            if args_cli.warmup_open_gripper and warmup_actions.shape[1] > 13:
                warmup_actions[:, 13] = OPEN_ACTION
            for _ in range(args_cli.warmup_steps):
                obs, _, terminated, truncated, _ = env.step(warmup_actions)
                if terminated.any() or truncated.any():
                    break

        policy_obs = _policy_obs(obs)
        episode_progress = min(args_cli.dataset_frame / args_cli.handoff_time_total_steps, 1.0)
        live_unorm: dict[str, torch.Tensor] = {
            OBS_STATE: _build_handoff_birelpose_time_state(env, device, episode_progress)
        }
        for feature_key in config.image_features:
            term_name = IMAGE_TERM_BY_FEATURE[feature_key]
            image = policy_obs.get(term_name)
            if image is None:
                raise KeyError(f"Missing live image observation term: {term_name}")
            live_unorm[feature_key] = torch.stack([_image_to_chw_float(frame, device) for frame in image], dim=0)

        def _image_mix_unorm(live_image_features: set[str], state: torch.Tensor) -> dict[str, torch.Tensor]:
            mixed = {OBS_STATE: state}
            for feature_key in config.image_features:
                mixed[feature_key] = live_unorm[feature_key] if feature_key in live_image_features else dataset_unorm[feature_key]
            return mixed

        variants_unorm = {
            "A_dataset_image_dataset_state": {
                OBS_STATE: dataset_unorm[OBS_STATE],
                **{key: dataset_unorm[key] for key in config.image_features},
            },
            "B_live_image_dataset_state": {
                OBS_STATE: dataset_unorm[OBS_STATE],
                **{key: live_unorm[key] for key in config.image_features},
            },
            "C_dataset_image_live_state": {
                OBS_STATE: live_unorm[OBS_STATE],
                **{key: dataset_unorm[key] for key in config.image_features},
            },
            "D_live_image_live_state": {
                OBS_STATE: live_unorm[OBS_STATE],
                **{key: live_unorm[key] for key in config.image_features},
            },
            "E_all_dataset_image_live_state": _image_mix_unorm(set(), live_unorm[OBS_STATE]),
            "F_wrist_live_only_live_state": _image_mix_unorm(
                {"observation.images.wrist_rgb"}, live_unorm[OBS_STATE]
            ),
            "G_observer_live_only_live_state": _image_mix_unorm(
                {"observation.images.observer_wrist_rgb"}, live_unorm[OBS_STATE]
            ),
            "H_global_live_only_live_state": _image_mix_unorm(
                {"observation.images.global_rgb"}, live_unorm[OBS_STATE]
            ),
            "I_all_live_image_live_state": _image_mix_unorm(set(config.image_features.keys()), live_unorm[OBS_STATE]),
        }
        variants = {
            name: _normalize_policy_batch(unorm, config, tokenizer, stats, device, args_cli.task_text)
            for name, unorm in variants_unorm.items()
        }

        state_diff: dict[str, Any] = {"segments": {}, "full": {}}
        raw_diff = live_unorm[OBS_STATE] - dataset_unorm[OBS_STATE]
        norm_dataset_state = _normalize_tensor(dataset_unorm[OBS_STATE], stats[OBS_STATE], _normalization_mode(config, OBS_STATE))
        norm_live_state = _normalize_tensor(live_unorm[OBS_STATE], stats[OBS_STATE], _normalization_mode(config, OBS_STATE))
        norm_diff = norm_live_state - norm_dataset_state
        state_diff["full"] = {"raw": _stats(raw_diff), "normalized": _stats(norm_diff)}
        for name, start, end in STATE_SEGMENTS_49D:
            state_diff["segments"][name] = {
                "dataset": _as_list(dataset_unorm[OBS_STATE][0, start:end]),
                "live": _as_list(live_unorm[OBS_STATE][0, start:end]),
                "raw_diff": _stats(raw_diff[:, start:end]),
                "normalized_diff": _stats(norm_diff[:, start:end]),
            }

        image_diff: dict[str, Any] = {}
        image_compare_dir = input_split_dir / "image_compare"
        for feature_key in config.image_features:
            diff = live_unorm[feature_key] - dataset_unorm[feature_key]
            term_name = IMAGE_TERM_BY_FEATURE[feature_key]
            _save_image_comparison(
                dataset_unorm[feature_key],
                live_unorm[feature_key],
                image_compare_dir,
                term_name,
            )
            image_diff[feature_key] = {
                "dataset_shape": list(dataset_unorm[feature_key].shape),
                "live_shape": list(live_unorm[feature_key].shape),
                "pixel_0_1_abs_diff": _stats(diff.abs()),
                "images": {
                    "dataset": str((image_compare_dir / f"{term_name}_dataset.png").relative_to(input_split_dir)),
                    "live": str((image_compare_dir / f"{term_name}_live.png").relative_to(input_split_dir)),
                    "absdiff_heatmap": str(
                        (image_compare_dir / f"{term_name}_absdiff_heatmap.png").relative_to(input_split_dir)
                    ),
                    "side_by_side": str(
                        (image_compare_dir / f"{term_name}_side_by_side.png").relative_to(input_split_dir)
                    ),
                },
            }

        action_rows = []
        variant_summaries = {}
        for variant_name, batch in variants.items():
            action_norm, action = _run_single_action(policy, batch, stats, config, args_cli.seed)
            summary = _action_summary(action)
            summary["variant"] = variant_name
            summary["seed"] = args_cli.seed
            summary["normalized_action"] = _as_list(action_norm)
            action_rows.append(summary)
            variant_summaries[variant_name] = summary

        queue_checks: dict[str, Any] = {}
        for variant_name, batch in variants.items():
            chunk_norm, chunk = _predict_executed_chunk(policy, batch, stats, config, args_cli.seed)
            queue_checks[variant_name] = {
                "horizon": int(config.horizon),
                "n_obs_steps": int(config.n_obs_steps),
                "n_action_steps": int(config.n_action_steps),
                "executed_chunk_shape": list(chunk.shape),
                "chunk_first_action": _action_summary(chunk[:, 0, :]),
                "chunk_actions": chunk.detach().cpu().tolist(),
                "chunk_actions_normalized": chunk_norm.detach().cpu().tolist(),
            }

        chunk = torch.tensor(queue_checks["A_dataset_image_dataset_state"]["chunk_actions"], device=device)
        selected_sequence = _select_action_sequence(
            policy,
            variants["A_dataset_image_dataset_state"],
            stats,
            config,
            args_cli.seed,
            config.n_action_steps,
        )
        queue_check = {
            "horizon": int(config.horizon),
            "n_obs_steps": int(config.n_obs_steps),
            "n_action_steps": int(config.n_action_steps),
            "executed_chunk_shape": list(chunk.shape),
            "selected_sequence_shape": list(selected_sequence.shape),
            "selected_vs_chunk_max_abs": float((selected_sequence - chunk).abs().max().detach().cpu().item()),
            "chunk_first_action": _action_summary(chunk[:, 0, :]),
            "selected_first_action": _action_summary(selected_sequence[:, 0, :]),
            "chunk_actions": queue_checks["A_dataset_image_dataset_state"]["chunk_actions"],
            "chunk_actions_normalized": queue_checks["A_dataset_image_dataset_state"]["chunk_actions_normalized"],
        }
        queue_checks["A_dataset_image_dataset_state"]["selected_sequence_shape"] = list(selected_sequence.shape)
        queue_checks["A_dataset_image_dataset_state"]["selected_vs_chunk_max_abs"] = queue_check["selected_vs_chunk_max_abs"]
        queue_checks["A_dataset_image_dataset_state"]["selected_first_action"] = queue_check["selected_first_action"]

        with (input_split_dir / "actions.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "variant",
                "seed",
                "left_xyz_norm",
                "left_gripper",
                "right_action_z",
                "right_gripper",
                "right_xyz_norm",
                "full_action",
                "normalized_action",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in action_rows:
                csv_row = dict(row)
                csv_row["full_action"] = json.dumps(csv_row["full_action"])
                csv_row["normalized_action"] = json.dumps(csv_row["normalized_action"])
                writer.writerow({key: csv_row[key] for key in fieldnames})

        fixed_rows = []
        sweep_rows = []
        chunk_sweep_rows = []
        sampling_summary: dict[str, Any] = {}
        chunk_sweep_summary: dict[str, Any] = {}
        chunk_variant_pairs = (
            ("A", "A_dataset_image_dataset_state"),
            ("B", "B_live_image_dataset_state"),
            ("C", "C_dataset_image_live_state"),
            ("D", "D_live_image_live_state"),
            ("E", "E_all_dataset_image_live_state"),
            ("F", "F_wrist_live_only_live_state"),
            ("G", "G_observer_live_only_live_state"),
            ("H", "H_global_live_only_live_state"),
            ("I", "I_all_live_image_live_state"),
        )
        for variant_short, variant_name in chunk_variant_pairs:
            # Keep the original scalar first-action statistics for A/D to preserve the
            # previous report shape, and add full-chunk seed sweeps for all variants.
            if variant_short in ("A", "D"):
                fixed_variant_rows = []
                for repeat in range(args_cli.same_seed_repeats):
                    _, action = _run_single_action(policy, variants[variant_name], stats, config, args_cli.seed)
                    row = _action_summary(action)
                    row.update({"variant": variant_name, "test": "same_seed", "repeat": repeat, "seed": args_cli.seed})
                    fixed_rows.append(row)
                    fixed_variant_rows.append(row)
                sweep_variant_rows = []
                for repeat in range(args_cli.seed_sweep_repeats):
                    seed = args_cli.seed + repeat
                    _, action = _run_single_action(policy, variants[variant_name], stats, config, seed)
                    row = _action_summary(action)
                    row.update({"variant": variant_name, "test": "seed_sweep", "repeat": repeat, "seed": seed})
                    sweep_rows.append(row)
                    sweep_variant_rows.append(row)
                sampling_summary[variant_short] = {
                    "same_seed": _action_stats(fixed_variant_rows),
                    "seed_sweep": _action_stats(sweep_variant_rows),
                }

            chunks_for_variant = []
            for repeat in range(args_cli.seed_sweep_repeats):
                seed = args_cli.seed + repeat
                _, chunk = _predict_executed_chunk(policy, variants[variant_name], stats, config, seed)
                chunks_for_variant.append(chunk)
                for chunk_step in range(chunk.shape[1]):
                    action_summary = _action_summary(chunk[:, chunk_step, :])
                    chunk_sweep_rows.append(
                        {
                            "variant": variant_name,
                            "repeat": repeat,
                            "seed": seed,
                            "chunk_step": chunk_step,
                            **action_summary,
                        }
                    )
            chunk_sweep_summary[variant_short] = _chunk_step_summary(chunks_for_variant)

        with (sampling_dir / "actions.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "test",
                "variant",
                "repeat",
                "seed",
                "left_xyz_norm",
                "left_gripper",
                "right_action_z",
                "right_gripper",
                "right_xyz_norm",
                "full_action",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in [*fixed_rows, *sweep_rows]:
                csv_row = dict(row)
                csv_row["full_action"] = json.dumps(csv_row["full_action"])
                writer.writerow({key: csv_row[key] for key in fieldnames})

        with (sampling_dir / "chunk_seed_sweep.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "variant",
                "repeat",
                "seed",
                "chunk_step",
                "left_xyz_norm",
                "left_gripper",
                "right_action_z",
                "right_gripper",
                "right_xyz_norm",
                "full_action",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in chunk_sweep_rows:
                csv_row = dict(row)
                csv_row["full_action"] = json.dumps(csv_row["full_action"])
                writer.writerow({key: csv_row[key] for key in fieldnames})

        common_meta = {
            "checkpoint": str(args_cli.checkpoint),
            "dataset_dir": str(args_cli.dataset_dir),
            "dataset_episode": args_cli.dataset_episode,
            "dataset_frame": args_cli.dataset_frame,
            "dataset_index": dataset_index,
            "dataset_episode_row": episode_row,
            "task": args_cli.task,
            "task_text": args_cli.task_text,
            "seed": args_cli.seed,
            "warmup_steps": args_cli.warmup_steps,
            "warmup_open_gripper": args_cli.warmup_open_gripper,
            "handoff_time_total_steps": args_cli.handoff_time_total_steps,
            "policy_device": str(device),
            "env_device": str(env.unwrapped.device),
            "image_features": list(config.image_features.keys()),
            "state_shape": list(config.input_features[OBS_STATE].shape),
            "action_shape": list(config.output_features[ACTION].shape),
            "n_obs_steps": int(config.n_obs_steps),
            "horizon": int(config.horizon),
            "n_action_steps": int(config.n_action_steps),
            "num_inference_steps": config.num_inference_steps,
            "dataset_video_backend": args_cli.dataset_video_backend,
            "weight_info": weight_info,
        }
        _write_json(input_split_dir / "summary.json", {**common_meta, "variant_actions": variant_summaries})
        _write_json(input_split_dir / "state_diff.json", state_diff)
        _write_json(input_split_dir / "image_diff.json", image_diff)
        _write_json(input_split_dir / "queue_check_A.json", queue_check)
        _write_json(input_split_dir / "queue_checks_by_variant.json", queue_checks)
        _write_json(
            sampling_dir / "summary.json",
            {**common_meta, "sampling": sampling_summary, "chunk_seed_sweep": chunk_sweep_summary},
        )
        _write_json(sampling_dir / "chunk_seed_sweep_summary.json", chunk_sweep_summary)
        _write_json(sampling_dir / "queue_check.json", queue_check)
        _write_json(sampling_dir / "queue_checks_by_variant.json", queue_checks)

        print(f"[OK] input split report: {input_split_dir}", flush=True)
        print(f"[OK] sampling/queue report: {sampling_dir}", flush=True)
        print("[A]", json.dumps(variant_summaries["A_dataset_image_dataset_state"], ensure_ascii=False), flush=True)
        print("[D]", json.dumps(variant_summaries["D_live_image_live_state"], ensure_ascii=False), flush=True)
        print(f"[QUEUE] selected_vs_chunk_max_abs={queue_check['selected_vs_chunk_max_abs']:.8f}", flush=True)
    finally:
        try:
            env.close()
        finally:
            simulation_app.close()


if __name__ == "__main__":
    main()
