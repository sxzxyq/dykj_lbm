"""Hybrid expert/policy handoff diagnostics for the dual-arm yellow-to-red task.

This script intentionally does not import eval_pick_place_policy.py because that
module launches Isaac at import time. It duplicates the small set of helpers
needed to load a MultiTask-DiT checkpoint, build live policy observations, run a
scripted handoff expert, and record per-step failure diagnostics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import types
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = SCRIPTS_ROOT / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher


DEFAULT_RUN_NAME = time.strftime("diagnose_handoff_hybrid_%Y%m%d_%H%M%S")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "eval_videos" / DEFAULT_RUN_NAME
DEFAULT_TASK = "Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0"
DEFAULT_TASK_TEXT = (
    "Right arm moves the blue cube to the yellow handoff area, then left arm moves it to the red target area."
)
COMBO_SOURCES = {
    "expert_expert": ("expert", "expert"),
    "expert_policy": ("expert", "policy"),
    "policy_expert": ("policy", "expert"),
    "policy_policy": ("policy", "policy"),
}
IMAGE_TERM_BY_CAMERA = {
    "wrist_cam": "wrist_rgb",
    "observer_wrist_cam": "observer_wrist_rgb",
    "global_cam": "global_rgb",
}
LEFT_ARM = "robot"
RIGHT_ARM = "observer_robot"
OPEN_ACTION = 1.0
CLOSE_ACTION = -1.0
TCP_OFFSET = (0.0, 0.0, 0.107)
HANDOFF_YELLOW_CENTER_W = (0.50, 0.00, 0.0205)
HANDOFF_RED_CENTER_W = (0.50, 0.30, 0.0205)
HANDOFF_AREA_SIZE_XY = (0.12, 0.12)
HANDOFF_HEIGHT_TOLERANCE = 0.03
HANDOFF_HAS_CUBE_TCP_DISTANCE = 0.10
HANDOFF_HAS_CUBE_MIN_Z = 0.075
HANDOFF_GRIPPER_HOLD_THRESHOLD = 0.055
GRIPPER_OPEN_COMMAND = 0.04
GRIPPER_OPEN_THRESHOLD = 0.01
HANDOFF_TIME_TOTAL_STEPS = 1845
DEFAULT_SIM_DT = 0.02

RIGHT_PICK_CUBE = 0
RIGHT_PLACE_YELLOW = 1
WAIT_YELLOW_STABLE = 2
LEFT_PICK_FROM_YELLOW = 3
LEFT_PLACE_RED = 4
DONE_HOLD = 5
ACTIVE_LEFT = 0
ACTIVE_RIGHT = 1
ACTIVE_NONE = 2
SUBTASK_NAMES = {
    RIGHT_PICK_CUBE: "RIGHT_PICK_CUBE",
    RIGHT_PLACE_YELLOW: "RIGHT_PLACE_YELLOW",
    WAIT_YELLOW_STABLE: "WAIT_YELLOW_STABLE",
    LEFT_PICK_FROM_YELLOW: "LEFT_PICK_FROM_YELLOW",
    LEFT_PLACE_RED: "LEFT_PLACE_RED",
    DONE_HOLD: "DONE_HOLD",
}
ACTIVE_ARM_NAMES = {
    ACTIVE_LEFT: "ACTIVE_LEFT",
    ACTIVE_RIGHT: "ACTIVE_RIGHT",
    ACTIVE_NONE: "ACTIVE_NONE",
}
SUBTASK_ACTIVE_ARM_ID = {
    RIGHT_PICK_CUBE: ACTIVE_RIGHT,
    RIGHT_PLACE_YELLOW: ACTIVE_RIGHT,
    WAIT_YELLOW_STABLE: ACTIVE_NONE,
    LEFT_PICK_FROM_YELLOW: ACTIVE_LEFT,
    LEFT_PLACE_RED: ACTIVE_LEFT,
    DONE_HOLD: ACTIVE_NONE,
}
STATE_MODE_BY_DIM = {
    34: "handoff_joint_ee",
    41: "handoff_joint_ee_relpose",
    43: "handoff_joint_ee_subtask",
    49: "handoff_joint_ee_birelpose_time",
}
PHASES = (
    "right_open_rest",
    "right_move_above_cube",
    "right_descend_to_grasp",
    "right_close_gripper",
    "right_lift_cube",
    "right_move_above_yellow",
    "right_descend_to_yellow",
    "right_release_on_yellow",
    "right_retreat",
    "wait_yellow_stable",
    "left_open_rest",
    "left_move_above_cube",
    "left_descend_to_grasp",
    "left_close_gripper",
    "left_lift_cube",
    "left_move_above_red",
    "left_descend_to_red",
    "left_release_on_red",
    "left_retreat",
    "wait_red_stable",
    "done",
)
LEFT_START_PHASE = PHASES.index("left_open_rest")

OBS_STATE = "observation.state"
ACTION = "action"
OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"
REPORT_LINES: list[str] = []
BODY_ID_CACHE: dict[tuple[str, str], int] = {}
JOINT_ID_CACHE: dict[str, list[int]] = {}
CHECKPOINT_MANIFEST: dict | None = None


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True, help="Policy checkpoint directory.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Isaac Lab task id.")
parser.add_argument("--task-text", type=str, default=DEFAULT_TASK_TEXT, help="Text instruction passed to the policy.")
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for diagnostics.")
parser.add_argument(
    "--combos",
    type=str,
    default="expert_expert,expert_policy,policy_expert,policy_policy",
    help=f"Comma-separated combos. Choices: {','.join(COMBO_SOURCES)}",
)
parser.add_argument("--episodes", type=int, default=3, help="Episodes per combo.")
parser.add_argument("--max-steps", type=int, default=2600, help="Maximum env steps per episode.")
parser.add_argument("--seed", type=int, default=2000, help="Base seed. Episode i uses seed+i.")
parser.add_argument("--num_envs", type=int, default=1, help="Only 1 env is supported.")
parser.add_argument("--n-action-steps", type=int, default=None, help="Override checkpoint n_action_steps.")
parser.add_argument("--num-inference-steps", type=int, default=None, help="Override diffusion denoising steps.")
parser.add_argument("--policy-inference-seed", type=int, default=123, help="Optional policy inference seed.")
parser.add_argument(
    "--policy-inference-seed-mode",
    type=str,
    default="each_chunk",
    choices=("first_call", "each_chunk", "each_step"),
    help="When to reseed policy inference.",
)
parser.add_argument("--handoff-time-total-steps", type=int, default=HANDOFF_TIME_TOTAL_STEPS)
parser.add_argument("--stable-steps", type=int, default=20, help="Predicate stable frames required for stage success.")
parser.add_argument("--rest-steps", type=int, default=20, help="Scripted expert open-gripper rest steps.")
parser.add_argument("--close-steps", type=int, default=35, help="Scripted expert close hold steps.")
parser.add_argument("--open-steps", type=int, default=35, help="Scripted expert release hold steps.")
parser.add_argument("--phase-timeout", type=int, default=320, help="Scripted expert phase timeout.")
parser.add_argument("--pos-threshold", type=float, default=0.015, help="Scripted expert waypoint threshold.")
parser.add_argument("--max-delta", type=float, default=0.018, help="Scripted expert max Cartesian delta per step.")
parser.add_argument("--arm-action-scale", type=float, default=0.5, help="IK action scale.")
parser.add_argument("--grasp-z", type=float, default=0.015, help="Scripted grasp TCP z in world frame.")
parser.add_argument("--release-z", type=float, default=0.085, help="Scripted release TCP z in world frame.")
parser.add_argument("--hover-z", type=float, default=0.20, help="Scripted hover TCP z in world frame.")
parser.add_argument("--lift-z", type=float, default=0.19, help="Scripted lift TCP z in world frame.")
parser.add_argument("--record-image-every", type=int, default=0, help="Save camera PNGs every N steps. 0 disables.")
parser.add_argument(
    "--camera-names",
    type=str,
    default="wrist_cam,observer_wrist_cam,global_cam",
    help="Scene cameras to record.",
)
parser.add_argument("--warmup-steps", type=int, default=2, help="Open-gripper warmup steps after reset.")
parser.add_argument("--refresh-camera-xform", action="store_true", default=False, help="Rewrite camera local xforms.")
parser.add_argument("--log-every", type=int, default=25, help="Log status every N env steps.")
parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True, help="Use local HF cache.")
parser.add_argument("--report", type=Path, default=None, help="Text report path. Defaults to output-dir/diagnostic_report.txt.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
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
from isaaclab.utils import math as math_utils
from isaaclab_tasks.utils import parse_env_cfg

from handoff_v2_utils import (
    ACTION_REPRESENTATION_DELTA_STEP,
    ACTION_REPRESENTATION_RELATIVE_CURRENT_POSE_CHUNK,
    load_manifest,
    preprocess_eval_images_torch,
    relative_current_action_to_delta_step_torch,
)


def _log(message: str) -> None:
    print(message, flush=True)
    REPORT_LINES.append(message)


def _is_shutdown_race_error(exc: Exception) -> bool:
    text = str(exc)
    fragments = (
        "Simulation view object is invalidated",
        "Failed to get DOF velocities from backend",
        "physics.tensors simulationView was invalidated",
        "was deleted while being used by a shape in a tensor view class",
    )
    return any(fragment in text for fragment in fragments)


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


def _tensor_row(tensor: torch.Tensor | None, env_id: int = 0):
    if tensor is None:
        return None
    value = tensor[env_id].detach().cpu()
    if value.ndim == 0:
        return value.item()
    return value.tolist()


def _scalar_bool(tensor: torch.Tensor) -> bool:
    return bool(tensor[0].detach().cpu().item())


def _asset(env, arm_name: str):
    return env.unwrapped.scene[arm_name]


def _body_id(env, arm_name: str, body_name: str = "panda_hand") -> int:
    key = (arm_name, body_name)
    if key not in BODY_ID_CACHE:
        body_ids, body_names = _asset(env, arm_name).find_bodies(body_name)
        if len(body_ids) != 1:
            raise RuntimeError(f"Expected one body for {arm_name}:{body_name}, got {body_names}")
        BODY_ID_CACHE[key] = body_ids[0]
    return BODY_ID_CACHE[key]


def _arm_gripper_joint_ids(env, arm_name: str) -> list[int]:
    if arm_name not in JOINT_ID_CACHE:
        joint_ids, joint_names = _asset(env, arm_name).find_joints(["panda_finger.*"])
        if not joint_ids:
            raise RuntimeError(f"Could not resolve gripper joints for {arm_name}: {joint_names}")
        JOINT_ID_CACHE[arm_name] = joint_ids
    return JOINT_ID_CACHE[arm_name]


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


def _cube_pos_w(env) -> torch.Tensor:
    return env.unwrapped.scene["object"].data.root_pos_w[:, :3]


def _cube_quat_w(env) -> torch.Tensor:
    return env.unwrapped.scene["object"].data.root_quat_w[:, :4]


def _cube_lin_vel_w(env) -> torch.Tensor:
    data = env.unwrapped.scene["object"].data
    if hasattr(data, "root_lin_vel_w"):
        return data.root_lin_vel_w[:, :3]
    if hasattr(data, "root_vel_w"):
        return data.root_vel_w[:, :3]
    return torch.zeros((env.unwrapped.num_envs, 3), device=env.unwrapped.device)


def _normalize_quat_wxyz(quat: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return quat / torch.clamp(torch.linalg.vector_norm(quat, dim=-1, keepdim=True), min=eps)


def _canonicalize_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    pivot = torch.argmax(torch.abs(quat), dim=-1, keepdim=True)
    pivot_value = torch.gather(quat, dim=-1, index=pivot)
    return torch.where(pivot_value < 0.0, -quat, quat)


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


def _quat_yaw_wxyz(quat: torch.Tensor) -> torch.Tensor:
    quat = _normalize_quat_wxyz(quat)
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _relative_tcp_pose(env, frame_arm_name: str, target_arm_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    frame_pos = _tcp_pos_w(env, frame_arm_name)
    target_pos = _tcp_pos_w(env, target_arm_name)
    frame_quat = _normalize_quat_wxyz(_tcp_quat_w(env, frame_arm_name))
    target_quat = _normalize_quat_wxyz(_tcp_quat_w(env, target_arm_name))
    frame_inv = _quat_conjugate_wxyz(frame_quat)
    rel_pos = math_utils.quat_apply(frame_inv, target_pos - frame_pos)
    rel_quat = _quat_multiply_wxyz(frame_inv, target_quat)
    return rel_pos, _canonicalize_quat_wxyz(_normalize_quat_wxyz(rel_quat))


def _relative_tcp_pose_right_in_left(env) -> tuple[torch.Tensor, torch.Tensor]:
    return _relative_tcp_pose(env, LEFT_ARM, RIGHT_ARM)


def _relative_tcp_pose_left_in_right(env) -> tuple[torch.Tensor, torch.Tensor]:
    return _relative_tcp_pose(env, RIGHT_ARM, LEFT_ARM)


def _arm_gripper_opening(env, arm_name: str) -> torch.Tensor:
    robot = _asset(env, arm_name)
    joint_ids = _arm_gripper_joint_ids(env, arm_name)
    return torch.sum(torch.abs(robot.data.joint_pos[:, joint_ids]), dim=1)


def _arm_gripper_is_open(env, arm_name: str) -> torch.Tensor:
    robot = _asset(env, arm_name)
    joint_ids = _arm_gripper_joint_ids(env, arm_name)
    finger_pos = torch.abs(robot.data.joint_pos[:, joint_ids])
    return torch.all(finger_pos >= GRIPPER_OPEN_COMMAND - GRIPPER_OPEN_THRESHOLD, dim=1)


def _arm_has_cube(env, arm_name: str) -> torch.Tensor:
    cube_pos = _cube_pos_w(env)
    tcp_pos = _tcp_pos_w(env, arm_name)
    near_tcp = torch.linalg.norm(cube_pos - tcp_pos, dim=1) <= HANDOFF_HAS_CUBE_TCP_DISTANCE
    lifted = cube_pos[:, 2] >= HANDOFF_HAS_CUBE_MIN_Z
    closed_on_object = _arm_gripper_opening(env, arm_name) <= HANDOFF_GRIPPER_HOLD_THRESHOLD
    return near_tcp & lifted & closed_on_object


def _object_on_handoff_area(env, center_w: tuple[float, float, float], gripper_arm: str) -> torch.Tensor:
    cube_pos = _cube_pos_w(env)
    center = torch.tensor(center_w, device=env.unwrapped.device, dtype=cube_pos.dtype).unsqueeze(0)
    xy_error = torch.abs(cube_pos[:, :2] - center[:, :2])
    inside = torch.logical_and(
        xy_error[:, 0] <= HANDOFF_AREA_SIZE_XY[0] * 0.5,
        xy_error[:, 1] <= HANDOFF_AREA_SIZE_XY[1] * 0.5,
    )
    low = torch.abs(cube_pos[:, 2] - center[:, 2]) <= HANDOFF_HEIGHT_TOLERANCE
    released = _arm_gripper_is_open(env, gripper_arm)
    return inside & low & released


def _one_hot_batch(index: int, size: int, env, device: torch.device) -> torch.Tensor:
    values = torch.zeros((env.unwrapped.num_envs, size), device=device, dtype=torch.float32)
    values[:, index] = 1.0
    return values


def _stage_scheduler_status(env, stage: str) -> dict:
    if stage == "right_to_yellow":
        subtask_id = RIGHT_PLACE_YELLOW if _scalar_bool(_arm_has_cube(env, RIGHT_ARM)) else RIGHT_PICK_CUBE
    elif stage == "left_to_red":
        subtask_id = LEFT_PLACE_RED if _scalar_bool(_arm_has_cube(env, LEFT_ARM)) else LEFT_PICK_FROM_YELLOW
    else:
        subtask_id = DONE_HOLD
    active_arm_id = SUBTASK_ACTIVE_ARM_ID[subtask_id]
    device = torch.device(env.unwrapped.device)
    return {
        "subtask_id": torch.full((env.unwrapped.num_envs,), subtask_id, device=device, dtype=torch.long),
        "active_arm_id": torch.full((env.unwrapped.num_envs,), active_arm_id, device=device, dtype=torch.long),
        "subtask_name": SUBTASK_NAMES[subtask_id],
        "active_arm_name": ACTIVE_ARM_NAMES[active_arm_id],
    }


def _sim_time(env, fallback_step: int) -> float:
    sim = getattr(env.unwrapped, "sim", None)
    for attr in ("current_time", "time", "sim_time"):
        value = getattr(sim, attr, None) if sim is not None else None
        if value is not None:
            try:
                return float(value)
            except TypeError:
                pass
    return float(fallback_step) * DEFAULT_SIM_DT


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


def _checkpoint_uses_fixed_image_preprocess() -> bool:
    if not CHECKPOINT_MANIFEST:
        return False
    return CHECKPOINT_MANIFEST.get("image_normalization") in ("clip", "imagenet")


def _image_to_chw_float(image: torch.Tensor, device: torch.device) -> torch.Tensor:
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = image.to(device=device, dtype=torch.float32)
    if image.max() > 1.5:
        image = image / 255.0
    return image.permute(2, 0, 1).contiguous()


def _preprocess_policy_image(chw_or_batched: torch.Tensor, feature_key: str) -> torch.Tensor:
    del feature_key
    if not _checkpoint_uses_fixed_image_preprocess():
        return chw_or_batched
    if chw_or_batched.ndim == 3:
        return preprocess_eval_images_torch(chw_or_batched.unsqueeze(0), size=224)[0]
    return preprocess_eval_images_torch(chw_or_batched, size=224)


def _state_dim_from_config(config) -> int:
    state_shape = tuple(config.input_features[OBS_STATE].shape)
    if len(state_shape) != 1:
        raise ValueError(f"Expected 1D observation.state feature, got shape={state_shape}")
    return int(state_shape[0])


def _build_handoff_state(env, device: torch.device) -> torch.Tensor:
    pieces = []
    for arm_name in (LEFT_ARM, RIGHT_ARM):
        robot = _asset(env, arm_name)
        pieces.extend(
            [
                robot.data.joint_pos.to(device=device, dtype=torch.float32),
                _tcp_pos_w(env, arm_name).to(device=device, dtype=torch.float32),
                _canonicalize_quat_wxyz(_normalize_quat_wxyz(_tcp_quat_w(env, arm_name))).to(
                    device=device, dtype=torch.float32
                ),
                _arm_gripper_opening(env, arm_name).unsqueeze(-1).to(device=device, dtype=torch.float32),
            ]
        )
    state = torch.cat(pieces, dim=-1)
    if state.shape[-1] != 34:
        raise ValueError(f"Expected handoff state dim 34, got shape={tuple(state.shape)}")
    return state


def _build_handoff_subtask_state(env, device: torch.device, scheduler_status: dict) -> torch.Tensor:
    base_state = _build_handoff_state(env, device)
    subtask_id = int(scheduler_status["subtask_id"][0].detach().cpu().item())
    active_arm_id = int(scheduler_status["active_arm_id"][0].detach().cpu().item())
    state = torch.cat(
        [
            base_state,
            _one_hot_batch(subtask_id, len(SUBTASK_NAMES), env, device),
            _one_hot_batch(active_arm_id, len(ACTIVE_ARM_NAMES), env, device),
        ],
        dim=-1,
    )
    if state.shape[-1] != 43:
        raise ValueError(f"Expected handoff subtask state dim 43, got shape={tuple(state.shape)}")
    return state


def _build_handoff_relpose_state(env, device: torch.device) -> torch.Tensor:
    base_state = _build_handoff_state(env, device)
    rel_pos, rel_quat = _relative_tcp_pose_right_in_left(env)
    state = torch.cat([base_state, rel_pos.to(device=device, dtype=torch.float32), rel_quat.to(device=device, dtype=torch.float32)], dim=-1)
    if state.shape[-1] != 41:
        raise ValueError(f"Expected handoff relpose state dim 41, got shape={tuple(state.shape)}")
    return state


def _build_handoff_birelpose_time_state(env, device: torch.device, episode_progress: float) -> torch.Tensor:
    base_state = _build_handoff_state(env, device)
    right_in_left_pos, right_in_left_quat = _relative_tcp_pose_right_in_left(env)
    left_in_right_pos, left_in_right_quat = _relative_tcp_pose_left_in_right(env)
    progress = torch.full((env.unwrapped.num_envs, 1), float(episode_progress), device=device, dtype=torch.float32)
    state = torch.cat(
        [
            base_state,
            right_in_left_pos.to(device=device, dtype=torch.float32),
            right_in_left_quat.to(device=device, dtype=torch.float32),
            left_in_right_pos.to(device=device, dtype=torch.float32),
            left_in_right_quat.to(device=device, dtype=torch.float32),
            progress,
        ],
        dim=-1,
    )
    if state.shape[-1] != 49:
        raise ValueError(f"Expected handoff birelpose+time state dim 49, got shape={tuple(state.shape)}")
    return state


def _build_state(env, device: torch.device, state_dim: int, scheduler_status: dict, episode_progress: float | None) -> torch.Tensor:
    if state_dim == 34:
        return _build_handoff_state(env, device)
    if state_dim == 41:
        return _build_handoff_relpose_state(env, device)
    if state_dim == 43:
        return _build_handoff_subtask_state(env, device, scheduler_status)
    if state_dim == 49:
        if episode_progress is None:
            raise ValueError("49D handoff state requires episode_progress")
        return _build_handoff_birelpose_time_state(env, device, episode_progress)
    raise ValueError(f"Hybrid handoff diagnostic only supports handoff state dims 34/41/43/49, got {state_dim}")


def _build_policy_batch(
    obs,
    config,
    tokenizer,
    stats,
    device: torch.device,
    task_text: str,
    env,
    scheduler_status: dict,
    episode_progress: float | None,
) -> dict[str, torch.Tensor]:
    policy_obs = _policy_obs(obs)
    state_dim = _state_dim_from_config(config)
    state = _build_state(env, device, state_dim, scheduler_status, episode_progress)
    batch: dict[str, torch.Tensor] = {
        OBS_STATE: _normalize_tensor(state, stats[OBS_STATE], _normalization_mode(config, OBS_STATE)),
    }
    for feature_key in config.image_features:
        term_name = feature_key.removeprefix("observation.images.")
        image = policy_obs.get(term_name)
        if image is None:
            raise KeyError(f"Missing image observation term: {term_name}")
        chw = torch.stack([_image_to_chw_float(frame, device) for frame in image], dim=0)
        chw = _preprocess_policy_image(chw, feature_key)
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


def _policy_action_queue_is_empty(policy) -> bool:
    queue = getattr(policy, "_queues", {}).get(ACTION)
    return queue is None or len(queue) == 0


def _should_seed_policy_inference(policy, mode: str, seed_event_count: int) -> bool:
    if mode == "first_call":
        return seed_event_count == 0
    if mode == "each_chunk":
        return _policy_action_queue_is_empty(policy)
    if mode == "each_step":
        return True
    raise ValueError(f"Unsupported policy inference seed mode: {mode}")


def _clamp_action_for_env(action: torch.Tensor, env) -> torch.Tensor:
    low = torch.as_tensor(env.action_space.low, dtype=action.dtype, device=action.device)
    high = torch.as_tensor(env.action_space.high, dtype=action.dtype, device=action.device)
    return torch.max(torch.min(action, high), low)


def _hold_action(env) -> torch.Tensor:
    action = torch.zeros((env.unwrapped.num_envs, env.action_space.shape[-1]), device=env.unwrapped.device)
    if action.shape[-1] > 6:
        action[:, 6] = OPEN_ACTION
    if action.shape[-1] > 13:
        action[:, 13] = OPEN_ACTION
    return action


def _stage_filter_action(env_action: torch.Tensor, stage: str) -> tuple[torch.Tensor, str, str]:
    filtered = _hold_action_tensor(env_action)
    if stage == "right_to_yellow":
        filtered[:, 7:14] = env_action[:, 7:14]
        return filtered, "hold_open", "policy"
    if stage == "left_to_red":
        filtered[:, 0:7] = env_action[:, 0:7]
        return filtered, "policy", "hold_open"
    return filtered, "hold_open", "hold_open"


def _hold_action_tensor(reference: torch.Tensor) -> torch.Tensor:
    action = torch.zeros_like(reference)
    if action.shape[-1] > 6:
        action[:, 6] = OPEN_ACTION
    if action.shape[-1] > 13:
        action[:, 13] = OPEN_ACTION
    return action


def _stage_for_phase(phase_name: str) -> str:
    if phase_name.startswith("right") or phase_name == "wait_yellow_stable":
        return "right_to_yellow"
    if phase_name.startswith("left") or phase_name == "wait_red_stable":
        return "left_to_red"
    return "done"


def _active_arm_for_phase(phase_name: str) -> str | None:
    if phase_name.startswith("right"):
        return RIGHT_ARM
    if phase_name.startswith("left"):
        return LEFT_ARM
    return None


def _target_area_for_phase(phase_name: str) -> str | None:
    if "yellow" in phase_name:
        return "yellow"
    if "red" in phase_name:
        return "red"
    return None


def _gripper_for_phase(phase_name: str) -> float:
    if "close" in phase_name or "lift" in phase_name or "move_above_yellow" in phase_name:
        return CLOSE_ACTION
    if "descend_to_yellow" in phase_name or "move_above_red" in phase_name or "descend_to_red" in phase_name:
        return CLOSE_ACTION
    return OPEN_ACTION


def _area_center(name: str, env) -> torch.Tensor:
    if name == "yellow":
        center = HANDOFF_YELLOW_CENTER_W
    elif name == "red":
        center = HANDOFF_RED_CENTER_W
    else:
        raise ValueError(f"Unknown area: {name}")
    return torch.tensor(center, device=env.unwrapped.device, dtype=torch.float32).unsqueeze(0)


def _park_positions(env) -> dict[str, torch.Tensor]:
    parks = {
        LEFT_ARM: _tcp_pos_w(env, LEFT_ARM).clone(),
        RIGHT_ARM: _tcp_pos_w(env, RIGHT_ARM).clone(),
    }
    for value in parks.values():
        value[:, 2] = torch.maximum(value[:, 2], torch.full_like(value[:, 2], args_cli.hover_z))
    return parks


def _desired_pos_for_phase(env, phase_name: str, park_positions: dict[str, torch.Tensor]) -> torch.Tensor | None:
    active_arm = _active_arm_for_phase(phase_name)
    if active_arm is None:
        return None
    desired = _tcp_pos_w(env, active_arm).clone()
    cube_pos = _cube_pos_w(env)
    area_name = _target_area_for_phase(phase_name)
    target = _area_center(area_name, env) if area_name else None
    if "move_above_cube" in phase_name:
        desired[:, :2] = cube_pos[:, :2]
        desired[:, 2] = args_cli.hover_z
    elif "descend_to_grasp" in phase_name or "close_gripper" in phase_name:
        desired[:, :2] = cube_pos[:, :2]
        desired[:, 2] = args_cli.grasp_z
    elif "lift_cube" in phase_name:
        desired[:, :2] = cube_pos[:, :2]
        desired[:, 2] = args_cli.lift_z
    elif "move_above_yellow" in phase_name or "move_above_red" in phase_name:
        desired[:, :2] = target[:, :2]
        desired[:, 2] = args_cli.lift_z
    elif "descend_to_yellow" in phase_name or "descend_to_red" in phase_name:
        desired[:, :2] = target[:, :2]
        desired[:, 2] = args_cli.release_z
    elif "release_on_yellow" in phase_name or "release_on_red" in phase_name:
        desired[:, :2] = target[:, :2]
        desired[:, 2] = args_cli.release_z
    elif "retreat" in phase_name:
        desired = park_positions[active_arm].clone()
    return desired


def _arm_delta_action(env, arm_name: str, desired_pos_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    robot = _asset(env, arm_name)
    tcp_pos = _tcp_pos_w(env, arm_name)
    delta_w = desired_pos_w - tcp_pos
    distance = torch.linalg.vector_norm(delta_w, dim=1)
    scale = torch.clamp(args_cli.max_delta / (distance + 1.0e-8), max=1.0).unsqueeze(-1)
    clipped_delta_w = delta_w * scale
    clipped_delta_b = math_utils.quat_apply_inverse(robot.data.root_quat_w, clipped_delta_w)
    arm_action = torch.zeros((env.unwrapped.num_envs, 6), device=env.unwrapped.device)
    arm_action[:, :3] = clipped_delta_b / args_cli.arm_action_scale
    return arm_action, distance


def _compute_expert_action(
    env,
    phase_name: str,
    park_positions: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, str | None]:
    actions = _hold_action(env)
    active_arm = _active_arm_for_phase(phase_name)
    desired_pos = _desired_pos_for_phase(env, phase_name, park_positions)
    distance = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    if active_arm is None or desired_pos is None:
        return actions, distance, desired_pos, active_arm
    arm_action, distance = _arm_delta_action(env, active_arm, desired_pos)
    gripper = _gripper_for_phase(phase_name)
    if active_arm == LEFT_ARM:
        actions[:, 0:6] = arm_action
        actions[:, 6] = gripper
    else:
        actions[:, 7:13] = arm_action
        actions[:, 13] = gripper
    return actions, distance, desired_pos, active_arm


def _advance_phase(
    phase_idx: int,
    phase_steps: int,
    reached: bool,
    yellow_stable_steps: int,
    red_stable_steps: int,
) -> tuple[int, int]:
    phase_name = PHASES[phase_idx]
    advance = False
    if phase_name.endswith("open_rest"):
        advance = phase_steps >= args_cli.rest_steps
    elif "close_gripper" in phase_name:
        advance = phase_steps >= args_cli.close_steps
    elif "release_on" in phase_name:
        advance = phase_steps >= args_cli.open_steps
    elif "retreat" in phase_name:
        advance = reached
    elif phase_name == "wait_yellow_stable":
        advance = yellow_stable_steps >= args_cli.stable_steps or phase_steps >= args_cli.phase_timeout
    elif phase_name == "wait_red_stable":
        advance = red_stable_steps >= args_cli.stable_steps or phase_steps >= args_cli.phase_timeout
    elif phase_name == "done":
        advance = False
    else:
        advance = reached or phase_steps >= args_cli.phase_timeout
    if advance:
        return min(phase_idx + 1, len(PHASES) - 1), 0
    return phase_idx, phase_steps + 1


class ScriptedExpertController:
    def __init__(self, start_stage: str, park_positions: dict[str, torch.Tensor]):
        if start_stage == "right_to_yellow":
            self.phase_idx = 0
        elif start_stage == "left_to_red":
            self.phase_idx = LEFT_START_PHASE
        else:
            raise ValueError(f"Unsupported scripted start stage={start_stage}")
        self.phase_steps = 0
        self.park_positions = park_positions

    @property
    def phase_name(self) -> str:
        return PHASES[self.phase_idx]

    def compute(self, env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, str | None]:
        return _compute_expert_action(env, self.phase_name, self.park_positions)

    def advance(self, reached: bool, yellow_stable_steps: int, red_stable_steps: int) -> None:
        self.phase_idx, self.phase_steps = _advance_phase(
            self.phase_idx,
            self.phase_steps,
            reached,
            yellow_stable_steps,
            red_stable_steps,
        )


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
        ("observation_encoder.text_encoder.text_encoder.", "observation_encoder.text_encoder.text_encoder.text_model."),
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


class EpisodeDiagnostics:
    def __init__(self) -> None:
        self.left_start_step: int | None = None
        self.left_gripper_close_step: int | None = None
        self.left_tcp_cube_min_dist_before_close: float | None = None
        self.left_tcp_cube_dist_at_close: float | None = None
        self.cube_pos_at_left_start: list[float] | None = None
        self.cube_yaw_at_left_start: float | None = None
        self.cube_vel_at_left_start: list[float] | None = None
        self.right_tcp_cube_dist_at_left_start: float | None = None
        self.cube_z_after_left_close: float | None = None
        self.cube_z_after_left_lift: float | None = None
        self.left_has_cube_seen = False
        self.left_has_cube_lost = False
        self._left_has_cube_previous = False
        self._cube_xy_at_left_start: torch.Tensor | None = None

    def mark_left_start(self, step: int, env) -> None:
        self.left_start_step = step
        cube_pos = _cube_pos_w(env).detach().clone()
        cube_quat = _cube_quat_w(env).detach().clone()
        cube_vel = _cube_lin_vel_w(env).detach().clone()
        self.cube_pos_at_left_start = _tensor_row(cube_pos)
        self.cube_yaw_at_left_start = float(_quat_yaw_wxyz(cube_quat)[0].detach().cpu().item())
        self.cube_vel_at_left_start = _tensor_row(cube_vel)
        self.right_tcp_cube_dist_at_left_start = float(
            torch.linalg.norm(_tcp_pos_w(env, RIGHT_ARM) - cube_pos, dim=1)[0].detach().cpu().item()
        )
        self._cube_xy_at_left_start = cube_pos[:, :2]

    def update_left_stage(self, step: int, env, env_action: torch.Tensor) -> None:
        if self.left_start_step is None:
            self.mark_left_start(step, env)
        cube_pos = _cube_pos_w(env)
        left_dist = float(torch.linalg.norm(_tcp_pos_w(env, LEFT_ARM) - cube_pos, dim=1)[0].detach().cpu().item())
        if self.left_gripper_close_step is None:
            if self.left_tcp_cube_min_dist_before_close is None:
                self.left_tcp_cube_min_dist_before_close = left_dist
            else:
                self.left_tcp_cube_min_dist_before_close = min(self.left_tcp_cube_min_dist_before_close, left_dist)
        closing = env_action.shape[-1] > 6 and float(env_action[0, 6].detach().cpu().item()) <= 0.0
        if closing and self.left_gripper_close_step is None:
            self.left_gripper_close_step = step
            self.left_tcp_cube_dist_at_close = left_dist
        if self.left_gripper_close_step is not None:
            cube_z = float(cube_pos[0, 2].detach().cpu().item())
            self.cube_z_after_left_close = cube_z if self.cube_z_after_left_close is None else max(self.cube_z_after_left_close, cube_z)
        left_has_cube = _scalar_bool(_arm_has_cube(env, LEFT_ARM))
        if left_has_cube:
            self.left_has_cube_seen = True
            cube_z = float(cube_pos[0, 2].detach().cpu().item())
            self.cube_z_after_left_lift = cube_z if self.cube_z_after_left_lift is None else max(self.cube_z_after_left_lift, cube_z)
        elif self._left_has_cube_previous and not left_has_cube:
            self.left_has_cube_lost = True
        self._left_has_cube_previous = left_has_cube

    def pushed_cube_away(self, env) -> bool:
        if self._cube_xy_at_left_start is None or self.left_has_cube_seen:
            return False
        delta = torch.linalg.norm(_cube_pos_w(env)[:, :2] - self._cube_xy_at_left_start, dim=1)
        return float(delta[0].detach().cpu().item()) > 0.04

    def failure_label(self, env, yellow_seen: bool, red_success: bool) -> str:
        if yellow_seen and red_success:
            return "success"
        if not yellow_seen:
            return "right_never_reached_yellow"
        min_before_close = self.left_tcp_cube_min_dist_before_close
        close_dist = self.left_tcp_cube_dist_at_close
        if min_before_close is not None and min_before_close > 0.03:
            return "left_not_near_cube"
        if close_dist is not None and close_dist > 0.02:
            return "left_close_far"
        if close_dist is not None and close_dist <= 0.02 and not self.left_has_cube_seen:
            return "close_near_no_grasp"
        if self.pushed_cube_away(env):
            return "pushed_cube_away"
        if self.left_has_cube_seen and self.left_has_cube_lost:
            return "grasp_then_drop"
        if self.right_tcp_cube_dist_at_left_start is not None and self.right_tcp_cube_dist_at_left_start < 0.08:
            return "right_blocking_left"
        return "unknown_left_failure"

    def to_summary_fields(self) -> dict:
        return {
            "left_start_step": self.left_start_step,
            "left_gripper_close_step": self.left_gripper_close_step,
            "left_tcp_cube_min_dist_before_close": self.left_tcp_cube_min_dist_before_close,
            "left_tcp_cube_dist_at_close": self.left_tcp_cube_dist_at_close,
            "cube_pos_at_left_start": self.cube_pos_at_left_start,
            "cube_yaw_at_left_start": self.cube_yaw_at_left_start,
            "cube_vel_at_left_start": self.cube_vel_at_left_start,
            "right_tcp_cube_dist_at_left_start": self.right_tcp_cube_dist_at_left_start,
            "cube_z_after_left_close": self.cube_z_after_left_close,
            "cube_z_after_left_lift": self.cube_z_after_left_lift,
        }


class HybridRecorder:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.steps_file = None
        self.episode_dir: Path | None = None
        self.image_counts: dict[str, int] = {}
        self.recorded_steps = 0

    def start_episode(self, combo: str, episode: int, config, action_representation: str) -> None:
        self.episode_dir = self.root_dir / combo / f"episode_{episode:06d}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.steps_file = (self.episode_dir / "steps.jsonl").open("w", encoding="utf-8")
        self.image_counts = {}
        self.recorded_steps = 0
        for name in _camera_names():
            term_name = _camera_term_name(name)
            (self.episode_dir / term_name).mkdir(parents=True, exist_ok=True)
            self.image_counts[term_name] = 0
        meta = {
            "combo": combo,
            "episode": episode,
            "seed": args_cli.seed + episode,
            "task": args_cli.task,
            "checkpoint": str(args_cli.checkpoint),
            "task_text": args_cli.task_text,
            "max_steps": args_cli.max_steps,
            "stable_steps": args_cli.stable_steps,
            "n_action_steps": config.n_action_steps,
            "num_inference_steps": config.num_inference_steps,
            "policy_inference_seed": args_cli.policy_inference_seed,
            "policy_inference_seed_mode": args_cli.policy_inference_seed_mode,
            "state_dim": _state_dim_from_config(config),
            "state_mode": STATE_MODE_BY_DIM.get(_state_dim_from_config(config), "unknown"),
            "handoff_time_total_steps": args_cli.handoff_time_total_steps,
            "action_representation": action_representation,
            "camera_names": _camera_names(),
            "record_image_every": args_cli.record_image_every,
        }
        (self.episode_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def record_step(
        self,
        env,
        obs,
        step: int,
        combo: str,
        stage: str,
        stage_source: str,
        expert_phase: str | None,
        action_source_left: str,
        action_source_right: str,
        env_action: torch.Tensor,
        model_action_normalized: torch.Tensor | None,
        model_action_unnormalized: torch.Tensor | None,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        yellow_success: torch.Tensor,
        red_success: torch.Tensor,
        yellow_seen: bool,
        yellow_stable_steps: int,
        red_stable_steps: int,
        episode_progress: float | None,
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
                self.image_counts[term_name] += 1
                images[term_name] = str(image_path.relative_to(self.episode_dir))
        cube_pos = _cube_pos_w(env)
        cube_quat = _canonicalize_quat_wxyz(_normalize_quat_wxyz(_cube_quat_w(env)))
        cube_yaw = _quat_yaw_wxyz(cube_quat)
        row = {
            "step": int(step),
            "combo": combo,
            "stage": stage,
            "stage_source": stage_source,
            "expert_phase": expert_phase,
            "action_source_left": action_source_left,
            "action_source_right": action_source_right,
            "episode_progress": episode_progress,
            "env_action": _tensor_row(env_action),
            "model_action_normalized": _tensor_row(model_action_normalized),
            "model_action_unnormalized": _tensor_row(model_action_unnormalized),
            "cube_pos_w": _tensor_row(cube_pos),
            "cube_quat_w": _tensor_row(cube_quat),
            "cube_yaw": _tensor_row(cube_yaw),
            "cube_lin_vel_w": _tensor_row(_cube_lin_vel_w(env)),
            "left_tcp_pos_w": _tensor_row(_tcp_pos_w(env, LEFT_ARM)),
            "right_tcp_pos_w": _tensor_row(_tcp_pos_w(env, RIGHT_ARM)),
            "left_gripper_opening": _tensor_row(_arm_gripper_opening(env, LEFT_ARM).unsqueeze(-1)),
            "right_gripper_opening": _tensor_row(_arm_gripper_opening(env, RIGHT_ARM).unsqueeze(-1)),
            "left_has_cube": _scalar_bool(_arm_has_cube(env, LEFT_ARM)),
            "right_has_cube": _scalar_bool(_arm_has_cube(env, RIGHT_ARM)),
            "yellow_success": _scalar_bool(yellow_success),
            "yellow_seen": yellow_seen,
            "red_success": _scalar_bool(red_success),
            "yellow_stable_steps": yellow_stable_steps,
            "red_stable_steps": red_stable_steps,
            "reward": _tensor_row(reward),
            "terminated": _scalar_bool(terminated),
            "truncated": _scalar_bool(truncated),
            "images": images,
        }
        self.steps_file.write(json.dumps(row) + "\n")
        self.recorded_steps += 1

    def finish_episode(self, summary: dict) -> None:
        if self.episode_dir is not None:
            summary = dict(summary)
            summary["recorded_steps"] = self.recorded_steps
            summary["image_counts"] = self.image_counts
            (self.episode_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if self.steps_file is not None:
            self.steps_file.close()
            self.steps_file = None


def _write_report(output_dir: Path) -> None:
    report_path = args_cli.report or (output_dir / "diagnostic_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(REPORT_LINES) + "\n", encoding="utf-8")


def _parse_combos(value: str) -> list[str]:
    combos = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [combo for combo in combos if combo not in COMBO_SOURCES]
    if unknown:
        raise ValueError(f"Unknown combo(s): {unknown}; valid={sorted(COMBO_SOURCES)}")
    return combos


def _run_episode(
    *,
    combo: str,
    episode: int,
    env,
    policy,
    config,
    tokenizer,
    stats,
    policy_device: torch.device,
    action_representation: str,
    recorder: HybridRecorder,
) -> dict:
    right_source, left_source = COMBO_SOURCES[combo]
    _seed_everything(args_cli.seed + episode)
    policy.reset()
    BODY_ID_CACHE.clear()
    JOINT_ID_CACHE.clear()
    reset_out = env.reset(seed=args_cli.seed + episode)
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    if args_cli.refresh_camera_xform:
        _refresh_camera_xforms(env)
    if args_cli.warmup_steps > 0:
        warmup_actions = _hold_action(env)
        for _ in range(args_cli.warmup_steps):
            obs, _, warmup_terminated, warmup_truncated, _ = env.step(warmup_actions)
            if warmup_terminated.any() or warmup_truncated.any():
                break

    park_positions = _park_positions(env)
    expert_controllers = {
        "right_to_yellow": ScriptedExpertController("right_to_yellow", park_positions),
        "left_to_red": ScriptedExpertController("left_to_red", park_positions),
    }
    recorder.start_episode(combo, episode, config, action_representation)

    stage = "right_to_yellow"
    stage_switched = False
    diagnostics = EpisodeDiagnostics()
    yellow_seen = False
    red_stage_success = False
    yellow_stable_steps = 0
    red_stable_steps = 0
    policy_seed_event_count = 0
    previous_relative_action = None
    terminated = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device, dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    reward = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    last_step = 0

    for step in range(args_cli.max_steps):
        if not simulation_app.is_running():
            break
        if stage == "left_to_red" and diagnostics.left_start_step is None:
            diagnostics.mark_left_start(step, env)
        source = right_source if stage == "right_to_yellow" else left_source
        scheduler_status = _stage_scheduler_status(env, stage)
        episode_progress = min(step / args_cli.handoff_time_total_steps, 1.0) if _state_dim_from_config(config) == 49 else None
        model_action_normalized = None
        model_action_unnormalized = None
        expert_phase = None

        with torch.no_grad():
            if source == "expert":
                controller = expert_controllers[stage]
                expert_phase = controller.phase_name
                env_action, distance, _, active_arm = controller.compute(env)
                reached = bool((distance < args_cli.pos_threshold).all().detach().cpu().item())
                action_source_left = "expert" if active_arm == LEFT_ARM else "hold_open"
                action_source_right = "expert" if active_arm == RIGHT_ARM else "hold_open"
            else:
                will_infer_policy = _policy_action_queue_is_empty(policy)
                if args_cli.policy_inference_seed is not None and _should_seed_policy_inference(
                    policy, args_cli.policy_inference_seed_mode, policy_seed_event_count
                ):
                    inference_seed = int(args_cli.policy_inference_seed) + policy_seed_event_count
                    _seed_everything(inference_seed)
                    policy_seed_event_count += 1
                    if step == 0 or will_infer_policy:
                        _log(
                            f"[{combo} EP {episode + 1} STEP {step}] "
                            f"policy_inference_seed={inference_seed} mode={args_cli.policy_inference_seed_mode}"
                        )
                batch = _build_policy_batch(
                    obs,
                    config,
                    tokenizer,
                    stats,
                    policy_device,
                    args_cli.task_text,
                    env,
                    scheduler_status,
                    episode_progress,
                )
                model_action_normalized = policy.select_action(batch)
                model_action_unnormalized = _unnormalize_tensor(
                    model_action_normalized,
                    stats[ACTION],
                    _normalization_mode(config, ACTION),
                ).to(env.unwrapped.device)
                env_action = model_action_unnormalized
                if action_representation == ACTION_REPRESENTATION_RELATIVE_CURRENT_POSE_CHUNK:
                    if will_infer_policy:
                        previous_relative_action = None
                    env_action, previous_relative_action = relative_current_action_to_delta_step_torch(
                        env_action,
                        previous_relative_action,
                    )
                env_action, action_source_left, action_source_right = _stage_filter_action(env_action, stage)
                reached = False
            env_action = _clamp_action_for_env(env_action, env)
            obs, reward, terminated, truncated, _ = env.step(env_action)
            yellow_success = _object_on_handoff_area(env, HANDOFF_YELLOW_CENTER_W, RIGHT_ARM)
            red_success = _object_on_handoff_area(env, HANDOFF_RED_CENTER_W, LEFT_ARM)
            yellow_now = _scalar_bool(yellow_success)
            red_now = _scalar_bool(red_success)
            yellow_seen = yellow_seen or yellow_now
            yellow_stable_steps = yellow_stable_steps + 1 if yellow_now else 0
            red_stable_steps = red_stable_steps + 1 if red_now else 0
            red_stage_success = red_stage_success or red_stable_steps >= args_cli.stable_steps
            if source == "expert":
                expert_controllers[stage].advance(reached, yellow_stable_steps, red_stable_steps)
            if stage == "left_to_red":
                diagnostics.update_left_stage(step, env, env_action)

        recorder.record_step(
            env,
            obs,
            step,
            combo,
            stage,
            source,
            expert_phase,
            action_source_left,
            action_source_right,
            env_action,
            model_action_normalized,
            model_action_unnormalized,
            reward,
            terminated,
            truncated,
            yellow_success,
            red_success,
            yellow_seen,
            yellow_stable_steps,
            red_stable_steps,
            episode_progress,
        )
        last_step = step + 1

        if step == 0 or (step + 1) % args_cli.log_every == 0 or red_stage_success:
            _log(
                f"[{combo} EP {episode + 1} STEP {step + 1}] stage={stage} source={source} "
                f"yellow_stable={yellow_stable_steps} red_stable={red_stable_steps} "
                f"yellow_seen={yellow_seen} red_success={red_stage_success} "
                f"cube={_tensor_row(_cube_pos_w(env))}"
            )

        if stage == "right_to_yellow" and yellow_stable_steps >= args_cli.stable_steps:
            stage = "left_to_red"
            stage_switched = True
            diagnostics.mark_left_start(step + 1, env)
            policy.reset()
            previous_relative_action = None
            _log(f"[{combo} EP {episode + 1} STEP {step + 1}] stage_change right_to_yellow -> left_to_red")
        if red_stage_success or terminated.any() or truncated.any():
            break

    summary = {
        "combo": combo,
        "episode": episode,
        "success": bool(yellow_seen and red_stage_success),
        "yellow_seen": bool(yellow_seen),
        "red_success": bool(red_stage_success),
        "steps": last_step,
        "stage_switched_to_left": stage_switched,
        "final_stage": stage,
        "yellow_stable_steps": yellow_stable_steps,
        "red_stable_steps": red_stable_steps,
        "terminated": _scalar_bool(terminated),
        "truncated": _scalar_bool(truncated),
    }
    summary.update(diagnostics.to_summary_fields())
    summary["failure_label"] = diagnostics.failure_label(env, yellow_seen, red_stage_success)
    recorder.finish_episode(summary)
    return summary


def main() -> None:
    global CHECKPOINT_MANIFEST
    if args_cli.num_envs != 1:
        raise ValueError("diagnose_handoff_hybrid_eval.py supports --num_envs 1 only.")
    if args_cli.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args_cli.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    if args_cli.handoff_time_total_steps <= 0:
        raise ValueError("--handoff-time-total-steps must be positive.")
    if args_cli.record_image_every < 0:
        raise ValueError("--record-image-every must be non-negative.")
    combos = _parse_combos(args_cli.combos)
    if args_cli.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not args_cli.checkpoint.exists():
        raise FileNotFoundError(args_cli.checkpoint)
    CHECKPOINT_MANIFEST = load_manifest(args_cli.checkpoint)

    _patch_lerobot_namespace_imports()
    _mock_groot_imports()
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
    from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy
    from transformers import CLIPTokenizer

    policy_device = torch.device(args_cli.device if args_cli.device == "cpu" or torch.cuda.is_available() else "cpu")
    _seed_everything(args_cli.seed)
    config = _load_config_from_json(args_cli.checkpoint, MultiTaskDiTConfig, FeatureType, PolicyFeature, policy_device)
    if args_cli.n_action_steps is not None:
        config.n_action_steps = args_cli.n_action_steps
    if args_cli.num_inference_steps is not None:
        config.num_inference_steps = args_cli.num_inference_steps
    state_dim = _state_dim_from_config(config)
    if state_dim not in STATE_MODE_BY_DIM:
        raise ValueError(f"Hybrid diagnostic supports only handoff checkpoints with state dim 34/41/43/49, got {state_dim}")
    if (
        state_dim == 49
        and CHECKPOINT_MANIFEST is not None
        and args_cli.handoff_time_total_steps == HANDOFF_TIME_TOTAL_STEPS
        and CHECKPOINT_MANIFEST.get("recommended_handoff_time_total_steps")
    ):
        args_cli.handoff_time_total_steps = int(CHECKPOINT_MANIFEST["recommended_handoff_time_total_steps"])
    action_representation = (
        CHECKPOINT_MANIFEST.get("action_representation", ACTION_REPRESENTATION_DELTA_STEP)
        if CHECKPOINT_MANIFEST is not None
        else ACTION_REPRESENTATION_DELTA_STEP
    )
    if action_representation not in (
        ACTION_REPRESENTATION_DELTA_STEP,
        ACTION_REPRESENTATION_RELATIVE_CURRENT_POSE_CHUNK,
    ):
        raise ValueError(f"Unsupported checkpoint action_representation={action_representation!r}")
    stats = _load_stats(args_cli.checkpoint, policy_device)
    tokenizer = CLIPTokenizer.from_pretrained(config.text_encoder_name)
    policy = MultiTaskDiTPolicy(config).to(policy_device)
    _load_policy_weights(policy, args_cli.checkpoint, policy_device)
    policy.eval()

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    recorder = HybridRecorder(args_cli.output_dir)

    _log(f"[INFO] task={args_cli.task}")
    _log(f"[INFO] checkpoint={args_cli.checkpoint}")
    _log(f"[INFO] output_dir={args_cli.output_dir}")
    _log(f"[INFO] combos={combos}")
    _log(f"[INFO] policy_device={policy_device} env_device={env.unwrapped.device}")
    _log(f"[INFO] action_space={env.action_space}")
    _log(f"[INFO] state_dim={state_dim} state_mode={STATE_MODE_BY_DIM[state_dim]}")
    _log(f"[INFO] n_action_steps={config.n_action_steps} num_inference_steps={config.num_inference_steps}")
    _log(f"[INFO] action_representation={action_representation}")
    _log(f"[INFO] handoff_time_total_steps={args_cli.handoff_time_total_steps}")
    if CHECKPOINT_MANIFEST is not None:
        _log(
            "[INFO] checkpoint_manifest="
            f"dataset_version={CHECKPOINT_MANIFEST.get('dataset_version')} "
            f"state_timing={CHECKPOINT_MANIFEST.get('state_timing')} "
            f"image_normalization={CHECKPOINT_MANIFEST.get('image_normalization')} "
            f"image_augmentation={CHECKPOINT_MANIFEST.get('image_augmentation')} "
            f"action_representation={CHECKPOINT_MANIFEST.get('action_representation')}"
        )

    all_summaries: list[dict] = []
    try:
        for combo in combos:
            combo_summaries = []
            for episode in range(args_cli.episodes):
                if not simulation_app.is_running():
                    break
                summary = _run_episode(
                    combo=combo,
                    episode=episode,
                    env=env,
                    policy=policy,
                    config=config,
                    tokenizer=tokenizer,
                    stats=stats,
                    policy_device=policy_device,
                    action_representation=action_representation,
                    recorder=recorder,
                )
                combo_summaries.append(summary)
                all_summaries.append(summary)
                _log(
                    f"[{combo} EP {episode + 1}] success={summary['success']} "
                    f"yellow_seen={summary['yellow_seen']} red_success={summary['red_success']} "
                    f"failure_label={summary['failure_label']} steps={summary['steps']}"
                )
            successes = sum(int(item["success"]) for item in combo_summaries)
            combo_aggregate = {
                "combo": combo,
                "episodes": len(combo_summaries),
                "successes": successes,
                "success_rate": successes / len(combo_summaries) if combo_summaries else 0.0,
                "failure_labels": {},
            }
            for item in combo_summaries:
                label = item["failure_label"]
                combo_aggregate["failure_labels"][label] = combo_aggregate["failure_labels"].get(label, 0) + 1
            combo_dir = args_cli.output_dir / combo
            combo_dir.mkdir(parents=True, exist_ok=True)
            (combo_dir / "summary.json").write_text(json.dumps(combo_aggregate, indent=2) + "\n", encoding="utf-8")
            _log(f"[{combo} SUMMARY] successes={successes}/{len(combo_summaries)} labels={combo_aggregate['failure_labels']}")

        total_successes = sum(int(item["success"]) for item in all_summaries)
        aggregate = {
            "checkpoint": str(args_cli.checkpoint),
            "task": args_cli.task,
            "combos": combos,
            "episodes_per_combo": args_cli.episodes,
            "total_episodes": len(all_summaries),
            "total_successes": total_successes,
            "overall_success_rate": total_successes / len(all_summaries) if all_summaries else 0.0,
            "state_dim": state_dim,
            "state_mode": STATE_MODE_BY_DIM[state_dim],
            "action_representation": action_representation,
            "handoff_time_total_steps": args_cli.handoff_time_total_steps,
        }
        args_cli.output_dir.mkdir(parents=True, exist_ok=True)
        (args_cli.output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
        _log(f"[SUMMARY] total_successes={total_successes}/{len(all_summaries)}")
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
    try:
        main()
    except KeyboardInterrupt:
        _log("[INFO] Interrupted by user.")
    except Exception as exc:
        if _is_shutdown_race_error(exc):
            _log("[INFO] Isaac Sim shutdown invalidated the physics tensor view; treating as exit.")
        else:
            _log(f"[ERROR] {type(exc).__name__}: {exc}")
            raise
    finally:
        simulation_app.close()
