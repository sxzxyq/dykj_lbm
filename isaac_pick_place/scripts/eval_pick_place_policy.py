"""Evaluate a trained HF/LeRobot MultiTask DiT policy in Isaac Lab.

The script runs closed-loop rollouts for the red-target cube pick-place task,
saves per-step metrics, and records policy-camera frames for visual debugging.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import types
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher


DEFAULT_RUN_NAME = time.strftime("eval_policy_%Y%m%d_%H%M%S")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "eval_videos" / DEFAULT_RUN_NAME
DEFAULT_TASK = "Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0"
DEFAULT_TASK_TEXT = "Pick up the cube and place it on the red target area."
IMAGE_TERM_BY_CAMERA = {
    "wrist_cam": "wrist_rgb",
    "observer_wrist_cam": "observer_wrist_rgb",
}
OPEN_ACTION = 1.0
REPORT_LINES: list[str] = []
STATE_MODE_BY_DIM = {
    16: "joint_ee",
    7: "ee_only",
}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True, help="Policy checkpoint directory.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Isaac Lab task id.")
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for eval artifacts.")
parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate.")
parser.add_argument("--max-steps", type=int, default=900, help="Maximum steps per episode.")
parser.add_argument("--seed", type=int, default=2000, help="Base seed. Episode i uses seed+i.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs. This first eval recorder supports 1.")
parser.add_argument("--task-text", type=str, default=DEFAULT_TASK_TEXT, help="Text instruction passed to the policy.")
parser.add_argument("--n-action-steps", type=int, default=None, help="Override checkpoint n_action_steps at inference.")
parser.add_argument("--num-inference-steps", type=int, default=None, help="Override diffusion denoising steps.")
parser.add_argument("--record-image-every", type=int, default=5, help="Save camera PNGs every N env steps. Use 0 to disable.")
parser.add_argument("--save-video", action="store_true", default=False, help="Also encode recorded PNGs to mp4 if imageio is installed.")
parser.add_argument("--video-fps", type=int, default=20, help="FPS for optional mp4 encoding.")
parser.add_argument("--warmup-steps", type=int, default=2, help="Open-gripper no-op steps after reset/camera refresh.")
parser.add_argument("--camera-names", type=str, default="wrist_cam,observer_wrist_cam", help="Scene cameras to record.")
parser.add_argument(
    "--fixed-cube-xy",
    type=str,
    default=None,
    help="Optional cube center xy in robot-root frame, formatted as 'x,y'. Example: '0.50,-0.08'.",
)
parser.add_argument(
    "--fixed-cube-xy-list",
    type=str,
    default=None,
    help="Optional semicolon-separated cube xy list in robot-root frame. Example: '0.36,-0.15;0.40,-0.11'.",
)
parser.add_argument("--refresh-camera-xform", action="store_true", default=False, help="Rewrite camera local xforms after reset.")
parser.add_argument("--log-every", type=int, default=25, help="Log status every N steps.")
parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True, help="Use local HF cache.")
parser.add_argument("--report", type=Path, default=None, help="Text report path. Defaults to output-dir/eval_report.txt.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

_MISSING_POLICY_DEPS = [
    name
    for name in ("lerobot", "transformers", "diffusers", "safetensors", "draccus", "einops", "PIL")
    if importlib.util.find_spec(name) is None
]
if _MISSING_POLICY_DEPS:
    raise SystemExit(
        "[ERROR] Missing policy inference dependencies before launching Isaac: "
        + ", ".join(_MISSING_POLICY_DEPS)
        + "\n[HINT] Install the LeRobot/MultiTask-DiT inference stack into the Isaac Python environment."
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


ACTION = "action"
OBS_STATE = "observation.state"
OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"


def _log(message: str) -> None:
    print(message, flush=True)
    REPORT_LINES.append(message)


def _is_shutdown_race_error(exc: Exception) -> bool:
    text = str(exc)
    shutdown_fragments = (
        "Simulation view object is invalidated",
        "Failed to get DOF velocities from backend",
        "physics.tensors simulationView was invalidated",
        "was deleted while being used by a shape in a tensor view class",
    )
    return any(fragment in text for fragment in shutdown_fragments)


def _mock_groot_imports() -> None:
    """Work around a Python 3.12 dataclass issue in lerobot.policies.groot."""
    groot_pkg = MagicMock(__path__=[])
    sys.modules["lerobot.policies.groot"] = groot_pkg
    sys.modules["lerobot.policies.groot.configuration_groot"] = MagicMock()
    sys.modules["lerobot.policies.groot.modeling_groot"] = MagicMock()
    sys.modules["lerobot.policies.groot.groot_n1"] = MagicMock()


def _patch_lerobot_namespace_imports() -> None:
    """Skip heavy LeRobot package __init__ modules that are not needed for policy eval.

    This lets the Isaac Python 3.11 env reuse the LeRobot 0.5.1 pure-Python
    package from the training Python 3.12 venv. The MultiTask-DiT modules we
    need are compatible, but some package-level imports touch Python 3.12-only
    dataset modules.
    """
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


def _camera_names() -> list[str]:
    return [name.strip() for name in args_cli.camera_names.split(",") if name.strip()]


def _camera_term_name(camera_name: str) -> str:
    return IMAGE_TERM_BY_CAMERA.get(camera_name, camera_name)


def _policy_obs(obs) -> dict:
    if isinstance(obs, dict) and "policy" in obs:
        return obs["policy"]
    return {}


def _obs_image(policy_obs: dict, term_name: str) -> torch.Tensor | None:
    value = policy_obs.get(term_name)
    if value is None:
        return None
    return value[0]


def _tensor_row(tensor: torch.Tensor, env_id: int = 0):
    value = tensor[env_id].detach().cpu()
    if value.ndim == 0:
        return value.item()
    return value.tolist()


def _success_term(env) -> torch.Tensor:
    try:
        return env.unwrapped.termination_manager.get_term("success").clone()
    except Exception:
        return torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device, dtype=torch.bool)


def _cube_pos_w(env) -> torch.Tensor:
    return env.unwrapped.scene["object"].data.root_pos_w[:, :3]


def _parse_xy(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected --fixed-cube-xy as 'x,y', got: {value!r}")
    return float(parts[0]), float(parts[1])


def _parse_xy_list(value: str | None) -> list[tuple[float, float]]:
    if not value:
        return []
    return [_parse_xy(item) for item in value.split(";") if item.strip()]


def _set_cube_xy_in_robot_frame(env, xy: tuple[float, float], object_center_z: float = 0.0205) -> None:
    robot = env.unwrapped.scene["robot"]
    object_asset = env.unwrapped.scene["object"]
    env_ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
    xy_t = torch.tensor(xy, device=env.unwrapped.device, dtype=robot.data.root_pos_w.dtype)

    positions = torch.zeros((env.unwrapped.num_envs, 3), device=env.unwrapped.device, dtype=robot.data.root_pos_w.dtype)
    positions[:, :2] = robot.data.root_pos_w[:, :2] + xy_t
    positions[:, 2] = robot.data.root_pos_w[:, 2] + object_center_z
    orientations = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device, dtype=robot.data.root_pos_w.dtype)
    orientations[:, 0] = 1.0
    velocities = torch.zeros((env.unwrapped.num_envs, 6), device=env.unwrapped.device, dtype=robot.data.root_pos_w.dtype)

    object_asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    object_asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
    _log(f"[INFO] fixed_cube_xy_robot={xy} fixed_cube_pos_w={positions.detach().cpu().tolist()}")


def _ee_pos_w(env) -> torch.Tensor:
    ee_frame = env.unwrapped.scene["ee_frame"]
    return ee_frame.data.target_pos_w[..., 0, :]


def _gripper_opening(env, gripper_joint_ids) -> torch.Tensor:
    robot = env.unwrapped.scene["robot"]
    return torch.sum(torch.abs(robot.data.joint_pos[:, gripper_joint_ids]), dim=1)


def _refresh_camera_xforms(env) -> None:
    from pxr import Gf, UsdGeom

    from isaaclab.utils.math import convert_camera_frame_orientation_convention

    for name in _camera_names():
        if name not in env.unwrapped.scene.sensors:
            _log(f"[CAMERA refresh] {name}: missing from scene sensors")
            continue
        camera = env.unwrapped.scene[name]
        cfg = camera.cfg
        rot = torch.tensor(cfg.offset.rot, dtype=torch.float32, device="cpu").unsqueeze(0)
        rot_offset = convert_camera_frame_orientation_convention(
            rot, origin=cfg.offset.convention, target="opengl"
        )[0]
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
        _log(f"[CAMERA refresh] {name}: refreshed {refreshed} prim(s)")


def _save_rgb_image(image: torch.Tensor, image_path: Path) -> None:
    from PIL import Image

    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = image.detach().cpu()
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != torch.uint8:
        if image.max() <= 1.0:
            image = image * 255.0
        image = image.clamp(0, 255).to(torch.uint8)
    Image.fromarray(image.numpy()).save(image_path)


def _maybe_write_video(image_paths: list[Path], video_path: Path, fps: int) -> str | None:
    if not image_paths:
        return None
    try:
        import imageio.v3 as iio
    except ImportError:
        return None

    frames = [iio.imread(path) for path in image_paths]
    video_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(video_path, frames, fps=fps)
    return str(video_path)


def _load_stats(checkpoint: Path, device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    stats_path = checkpoint / "dataset_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing checkpoint stats: {stats_path}")
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
    config_path = checkpoint / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
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


def _image_to_chw_float(image: torch.Tensor, device: torch.device) -> torch.Tensor:
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = image.to(device=device, dtype=torch.float32)
    if image.max() > 1.5:
        image = image / 255.0
    return image.permute(2, 0, 1).contiguous()


def _state_dim_from_config(config) -> int:
    state_shape = tuple(config.input_features[OBS_STATE].shape)
    if len(state_shape) != 1:
        raise ValueError(f"Expected 1D observation.state feature, got shape={state_shape}")
    return int(state_shape[0])


def _state_terms_for_dim(state_dim: int) -> tuple[str, ...]:
    if state_dim == 16:
        return ("joint_pos", "ee_position", "ee_quat")
    if state_dim == 7:
        return ("ee_position", "ee_quat")
    raise ValueError(
        f"Unsupported checkpoint observation.state dim={state_dim}; "
        "expected 16 for joint_ee or 7 for ee_only."
    )


def _build_state(policy_obs: dict, device: torch.device, state_dim: int) -> torch.Tensor:
    pieces = []
    for term in _state_terms_for_dim(state_dim):
        value = policy_obs.get(term)
        if value is None:
            raise KeyError(f"Missing observation term: {term}")
        pieces.append(value.to(device=device, dtype=torch.float32))
    return torch.cat(pieces, dim=-1)


def _build_policy_batch(obs, config, tokenizer, stats, device: torch.device, task_text: str) -> dict[str, torch.Tensor]:
    policy_obs = _policy_obs(obs)
    state_dim = _state_dim_from_config(config)
    state = _build_state(policy_obs, device, state_dim)
    batch: dict[str, torch.Tensor] = {
        OBS_STATE: _normalize_tensor(state, stats[OBS_STATE], _normalization_mode(config, OBS_STATE)),
    }

    for feature_key in config.image_features:
        term_name = feature_key.removeprefix("observation.images.")
        image = policy_obs.get(term_name)
        if image is None:
            raise KeyError(f"Missing image observation term: {term_name}")
        chw = torch.stack([_image_to_chw_float(frame, device) for frame in image], dim=0)
        batch[feature_key] = _normalize_tensor(chw, stats[feature_key], _normalization_mode(config, feature_key))

    tokens = tokenizer(
        [task_text] * state.shape[0],
        max_length=config.tokenizer_max_length,
        padding=config.tokenizer_padding,
        truncation=config.tokenizer_truncation,
        return_tensors="pt",
    )
    batch[OBS_LANGUAGE_TOKENS] = tokens["input_ids"].to(device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = tokens["attention_mask"].to(device)
    return batch


def _clamp_action_for_env(action: torch.Tensor, env) -> torch.Tensor:
    low = torch.as_tensor(env.action_space.low, dtype=action.dtype, device=action.device)
    high = torch.as_tensor(env.action_space.high, dtype=action.dtype, device=action.device)
    return torch.max(torch.min(action, high), low)


class EvalRecorder:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.episode_dir: Path | None = None
        self.steps_file = None
        self.image_paths: dict[str, list[Path]] = {}
        self.image_counts: dict[str, int] = {}
        self.recorded_steps = 0

    def start_episode(self, episode: int, env, config, fixed_cube_xy: tuple[float, float] | None = None) -> None:
        self.episode_dir = self.root_dir / f"episode_{episode:06d}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.steps_file = (self.episode_dir / "steps.jsonl").open("w", encoding="utf-8")
        self.image_paths = {}
        self.image_counts = {}
        self.recorded_steps = 0
        for name in _camera_names():
            term_name = _camera_term_name(name)
            (self.episode_dir / term_name).mkdir(parents=True, exist_ok=True)
            self.image_paths[term_name] = []
            self.image_counts[term_name] = 0
        meta = {
            "episode": episode,
            "seed": args_cli.seed + episode,
            "task": args_cli.task,
            "checkpoint": str(args_cli.checkpoint),
            "task_text": args_cli.task_text,
            "max_steps": args_cli.max_steps,
            "n_action_steps": config.n_action_steps,
            "num_inference_steps": config.num_inference_steps,
            "state_dim": _state_dim_from_config(config),
            "state_mode": STATE_MODE_BY_DIM.get(_state_dim_from_config(config), "unknown"),
            "action_space_shape": tuple(env.action_space.shape),
            "camera_names": _camera_names(),
            "record_image_every": args_cli.record_image_every,
            "fixed_cube_xy_robot": list(fixed_cube_xy) if fixed_cube_xy is not None else None,
        }
        (self.episode_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def record_step(
        self,
        obs,
        step: int,
        action: torch.Tensor,
        raw_model_action: torch.Tensor,
        reward: torch.Tensor,
        success: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        cube_pos: torch.Tensor,
        ee_pos: torch.Tensor,
        gripper_opening: torch.Tensor,
    ) -> None:
        if self.episode_dir is None or self.steps_file is None:
            return
        policy_obs = _policy_obs(obs)
        images = {}
        if args_cli.record_image_every > 0 and step % args_cli.record_image_every == 0:
            for name in _camera_names():
                term_name = _camera_term_name(name)
                image = _obs_image(policy_obs, term_name)
                if image is None:
                    continue
                image_index = self.image_counts[term_name]
                image_path = self.episode_dir / term_name / f"{image_index:06d}.png"
                _save_rgb_image(image, image_path)
                self.image_paths[term_name].append(image_path)
                self.image_counts[term_name] += 1
                images[term_name] = str(image_path.relative_to(self.episode_dir))

        row = {
            "step": step,
            "action": _tensor_row(action),
            "model_action_normalized": _tensor_row(raw_model_action),
            "reward": _tensor_row(reward),
            "success": bool(success[0].detach().cpu().item()),
            "terminated": bool(terminated[0].detach().cpu().item()),
            "truncated": bool(truncated[0].detach().cpu().item()),
            "cube_pos_w": _tensor_row(cube_pos),
            "ee_pos_w": _tensor_row(ee_pos),
            "gripper_opening": _tensor_row(gripper_opening),
            "images": images,
        }
        self.steps_file.write(json.dumps(row) + "\n")
        self.recorded_steps += 1

    def finish_episode(
        self,
        success: torch.Tensor,
        steps: int,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        fixed_cube_xy: tuple[float, float] | None = None,
    ) -> dict:
        videos = {}
        if args_cli.save_video and self.episode_dir is not None:
            for term_name, paths in self.image_paths.items():
                video_path = self.episode_dir / f"{term_name}.mp4"
                written = _maybe_write_video(paths, video_path, args_cli.video_fps)
                if written is not None:
                    videos[term_name] = written
        summary = {
            "success": bool(success[0].detach().cpu().item()),
            "steps": steps,
            "terminated": bool(terminated[0].detach().cpu().item()),
            "truncated": bool(truncated[0].detach().cpu().item()),
            "fixed_cube_xy_robot": list(fixed_cube_xy) if fixed_cube_xy is not None else None,
            "recorded_steps": self.recorded_steps,
            "image_counts": self.image_counts,
            "videos": videos,
        }
        if self.episode_dir is not None:
            (self.episode_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if self.steps_file is not None:
            self.steps_file.close()
            self.steps_file = None
        return summary


def _write_report(output_dir: Path) -> None:
    report_path = args_cli.report or (output_dir / "eval_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(REPORT_LINES) + "\n", encoding="utf-8")


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
    """Map CLIP key names saved with Transformers 5.x to the 4.x module layout."""
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


def _load_policy_weights(policy: torch.nn.Module, checkpoint: Path, device: torch.device) -> None:
    weight_file = _checkpoint_weight_file(checkpoint)
    state_dict = load_safetensors_file(str(weight_file), device=str(device))
    model_keys = set(policy.state_dict().keys())
    state_dict, remap_count = _remap_transformers5_clip_keys(state_dict, model_keys)
    missing_keys, unexpected_keys = policy.load_state_dict(state_dict, strict=False)
    _log(f"[INFO] loaded_weights={weight_file}")
    _log(f"[INFO] compat_key_remaps={remap_count}")
    _log(f"[INFO] missing_keys={len(missing_keys)} unexpected_keys={len(unexpected_keys)}")
    if missing_keys:
        _log(f"[WARN] first_missing_keys={list(missing_keys)[:10]}")
    if unexpected_keys:
        _log(f"[WARN] first_unexpected_keys={list(unexpected_keys)[:10]}")


def main() -> None:
    if args_cli.num_envs != 1:
        raise ValueError("This first visual eval recorder supports --num_envs 1 only.")
    if args_cli.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args_cli.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    if args_cli.record_image_every < 0:
        raise ValueError("--record-image-every must be non-negative.")
    fixed_cube_xy_list = _parse_xy_list(args_cli.fixed_cube_xy_list)
    if fixed_cube_xy_list and args_cli.fixed_cube_xy is not None:
        raise ValueError("Use either --fixed-cube-xy or --fixed-cube-xy-list, not both.")
    if fixed_cube_xy_list:
        args_cli.episodes = len(fixed_cube_xy_list)
    if args_cli.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not args_cli.checkpoint.exists():
        raise FileNotFoundError(args_cli.checkpoint)

    _patch_lerobot_namespace_imports()
    _mock_groot_imports()
    from lerobot.configs.types import FeatureType, PolicyFeature
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
    state_dim = _state_dim_from_config(config)
    state_mode = STATE_MODE_BY_DIM.get(state_dim, f"unknown_{state_dim}d")
    stats = _load_stats(args_cli.checkpoint, device)
    tokenizer = CLIPTokenizer.from_pretrained(config.text_encoder_name)
    policy = MultiTaskDiTPolicy(config).to(device)
    _load_policy_weights(policy, args_cli.checkpoint, device)
    policy.eval()

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_device = env.unwrapped.device
    gripper_joint_ids, _ = env.unwrapped.scene["robot"].find_joints(["panda_finger.*"])
    recorder = EvalRecorder(args_cli.output_dir)

    _log(f"[INFO] task={args_cli.task}")
    _log(f"[INFO] checkpoint={args_cli.checkpoint}")
    _log(f"[INFO] output_dir={args_cli.output_dir}")
    _log(f"[INFO] policy_device={device} env_device={env_device}")
    _log(f"[INFO] action_space={env.action_space}")
    _log(f"[INFO] n_action_steps={config.n_action_steps} num_inference_steps={config.num_inference_steps}")
    _log(f"[INFO] image_features={list(config.image_features.keys())}")
    _log(f"[INFO] state_dim={state_dim} state_mode={state_mode}")

    summaries = []
    try:
        for episode in range(args_cli.episodes):
            if not simulation_app.is_running():
                break
            _seed_everything(args_cli.seed + episode)
            policy.reset()
            reset_out = env.reset(seed=args_cli.seed + episode)
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            fixed_cube_xy = None
            if fixed_cube_xy_list:
                fixed_cube_xy = fixed_cube_xy_list[episode]
            elif args_cli.fixed_cube_xy is not None:
                fixed_cube_xy = _parse_xy(args_cli.fixed_cube_xy)
            if fixed_cube_xy is not None:
                _set_cube_xy_in_robot_frame(env, fixed_cube_xy)
            if args_cli.refresh_camera_xform:
                _refresh_camera_xforms(env)
            if args_cli.warmup_steps > 0:
                warmup_actions = torch.zeros(env.action_space.shape, device=env_device)
                warmup_actions[:, 6] = OPEN_ACTION
                for _ in range(args_cli.warmup_steps):
                    obs, _, warmup_terminated, warmup_truncated, _ = env.step(warmup_actions)
                    if warmup_terminated.any() or warmup_truncated.any():
                        break

            recorder.start_episode(episode, env, config, fixed_cube_xy)
            success = torch.zeros(env.unwrapped.num_envs, device=env_device, dtype=torch.bool)
            terminated = torch.zeros_like(success)
            truncated = torch.zeros_like(success)
            reward = torch.zeros(env.unwrapped.num_envs, device=env_device)
            last_step = 0

            for step in range(args_cli.max_steps):
                if not simulation_app.is_running():
                    break
                with torch.no_grad():
                    batch = _build_policy_batch(obs, config, tokenizer, stats, device, args_cli.task_text)
                    model_action = policy.select_action(batch)
                    env_action = _unnormalize_tensor(
                        model_action,
                        stats[ACTION],
                        _normalization_mode(config, ACTION),
                    ).to(env_device)
                    env_action = _clamp_action_for_env(env_action, env)
                    cube_pos = _cube_pos_w(env).clone()
                    ee_pos = _ee_pos_w(env).clone()
                    opening = _gripper_opening(env, gripper_joint_ids).clone()
                    pre_obs = obs
                    obs, reward, terminated, truncated, _ = env.step(env_action)
                    success = _success_term(env)
                    recorder.record_step(
                        pre_obs,
                        step,
                        env_action,
                        model_action,
                        reward,
                        success,
                        terminated,
                        truncated,
                        cube_pos,
                        ee_pos,
                        opening,
                    )
                    last_step = step + 1

                    if step == 0 or (step + 1) % args_cli.log_every == 0 or success.any():
                        _log(
                            f"[EP {episode + 1} STEP {step + 1}] "
                            f"reward={reward.detach().cpu().tolist()} "
                            f"success={success.detach().cpu().tolist()} "
                            f"terminated={terminated.detach().cpu().tolist()} "
                            f"truncated={truncated.detach().cpu().tolist()} "
                            f"cube_pos={cube_pos.detach().cpu().tolist()}"
                        )

                    if success.any() or terminated.any() or truncated.any():
                        break

            summary = recorder.finish_episode(success, last_step, terminated, truncated, fixed_cube_xy)
            summaries.append(summary)
            _log(
                f"[EP {episode + 1}] final_success={summary['success']} "
                f"terminated={summary['terminated']} truncated={summary['truncated']} steps={summary['steps']}"
            )

        total_successes = sum(int(item["success"]) for item in summaries)
        aggregate = {
            "checkpoint": str(args_cli.checkpoint),
            "task": args_cli.task,
            "episodes": len(summaries),
            "successes": total_successes,
            "success_rate": total_successes / len(summaries) if summaries else 0.0,
            "max_steps": args_cli.max_steps,
            "seed": args_cli.seed,
            "n_action_steps": config.n_action_steps,
            "state_dim": state_dim,
            "state_mode": state_mode,
            "fixed_cube_xy_list": [list(xy) for xy in fixed_cube_xy_list],
            "episode_summaries": summaries,
        }
        args_cli.output_dir.mkdir(parents=True, exist_ok=True)
        (args_cli.output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
        _log(f"[SUMMARY] successes={total_successes}/{len(summaries)} rate={aggregate['success_rate']:.3f}")
        _log("[OK] Policy eval completed.")
    except KeyboardInterrupt:
        _log("[INFO] Interrupted by user.")
    except Exception as exc:
        if _is_shutdown_race_error(exc):
            _log("[INFO] Isaac Sim shutdown invalidated the physics tensor view; treating as exit.")
        else:
            _log(f"[ERROR] {type(exc).__name__}: {exc}")
            raise
    finally:
        try:
            env.close()
        except Exception as exc:
            if _is_shutdown_race_error(exc):
                _log("[INFO] Ignored tensor-view invalidation during env.close().")
            else:
                raise
        _write_report(args_cli.output_dir)


if __name__ == "__main__":
    main()
    simulation_app.close()
