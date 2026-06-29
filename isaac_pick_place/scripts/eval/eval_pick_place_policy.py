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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = SCRIPTS_ROOT / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
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
    "global_cam": "global_rgb",
}
OPEN_ACTION = 1.0
HANDOFF_TASK_ID = "Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0"
TCP_OFFSET = (0.0, 0.0, 0.107)
HANDOFF_YELLOW_CENTER_W = (0.50, 0.00, 0.0205)
HANDOFF_RED_CENTER_W = (0.50, 0.30, 0.0205)
HANDOFF_AREA_SIZE_XY = (0.12, 0.12)
HANDOFF_HEIGHT_TOLERANCE = 0.03
GRIPPER_OPEN_COMMAND = 0.04
GRIPPER_OPEN_THRESHOLD = 0.01
HANDOFF_HAS_CUBE_TCP_DISTANCE = 0.10
HANDOFF_HAS_CUBE_MIN_Z = 0.075
HANDOFF_GRIPPER_HOLD_THRESHOLD = 0.055
HANDOFF_STABLE_STEPS = 20
HANDOFF_RIGHT_RETREAT_STEPS = 300
HANDOFF_TIME_TOTAL_STEPS = 1845
HANDOFF_SCRIPTED_RETREAT_Z = 0.22
HANDOFF_SCRIPTED_RETREAT_MAX_DELTA = 0.018
HANDOFF_SCRIPTED_RETREAT_ACTION_SCALE = 0.5
HANDOFF_SCRIPTED_RETREAT_POS_THRESHOLD = 0.03
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
REPORT_LINES: list[str] = []
STATE_MODE_BY_DIM = {
    16: "joint_ee",
    7: "ee_only",
    26: "handoff_joint_tcp_pos_gripper",
    34: "handoff_joint_ee",
    41: "handoff_joint_ee_relpose",
    43: "handoff_joint_ee_subtask",
    49: "handoff_joint_ee_birelpose_time",
}
BODY_ID_CACHE: dict[tuple[str, str], int] = {}
JOINT_ID_CACHE: dict[str, list[int]] = {}
CHECKPOINT_MANIFEST: dict | None = None


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
parser.add_argument(
    "--policy-inference-seed",
    type=int,
    default=None,
    help=(
        "Optional seed to set immediately before selected policy inference calls. "
        "Useful for diagnosing diffusion sampling/RNG effects in closed-loop eval."
    ),
)
parser.add_argument(
    "--policy-inference-seed-mode",
    type=str,
    default="first_call",
    choices=("first_call", "each_chunk", "each_step"),
    help=(
        "When --policy-inference-seed is set, choose whether to reseed before the first policy call, "
        "before each newly generated action chunk, or before every select_action call."
    ),
)
parser.add_argument("--record-image-every", type=int, default=5, help="Save camera PNGs every N env steps. Use 0 to disable.")
parser.add_argument(
    "--record-policy-inputs",
    action="store_true",
    default=False,
    help="Save the image tensors actually passed to policy inference whenever a new action chunk is generated.",
)
parser.add_argument(
    "--record-policy-input-tensors",
    action="store_true",
    default=False,
    help="Save exact normalized policy batch tensors before select_action. Use --policy-input-tensor-steps to limit size.",
)
parser.add_argument(
    "--policy-input-tensor-steps",
    type=str,
    default="",
    help="Optional comma-separated env steps for --record-policy-input-tensors. Empty means every new action chunk.",
)
parser.add_argument("--save-video", action="store_true", default=False, help="Also encode recorded PNGs to mp4 if imageio is installed.")
parser.add_argument("--video-fps", type=int, default=20, help="FPS for optional mp4 encoding.")
parser.add_argument("--warmup-steps", type=int, default=2, help="No-op steps after reset/camera refresh.")
parser.add_argument(
    "--warmup-open-gripper",
    action="store_true",
    default=False,
    help="Set gripper actions to OPEN_ACTION during warmup. Omit for all-zero warmup actions.",
)
parser.add_argument(
    "--handoff-right-retreat-steps",
    type=int,
    default=HANDOFF_RIGHT_RETREAT_STEPS,
    help=(
        "For 43D handoff-subtask eval, keep RIGHT_PLACE_YELLOW active for this many steps after "
        "the cube is released on yellow before switching to the left arm. This lets the learned "
        "right_retreat behavior run although right_retreat is folded into RIGHT_PLACE_YELLOW."
    ),
)
parser.add_argument(
    "--handoff-scripted-right-retreat",
    action="store_true",
    default=False,
    help=(
        "Debug mode for 43D handoff eval: after the cube is released on yellow, override the "
        "policy for the right arm only and command a scripted upward retreat before activating "
        "the left arm. This is not a pure policy evaluation."
    ),
)
parser.add_argument(
    "--handoff-time-total-steps",
    type=int,
    default=HANDOFF_TIME_TOTAL_STEPS,
    help=(
        "For 49D handoff birelpose+time checkpoints, compute episode_progress as "
        "min(current_env_step / this value, 1.0)."
    ),
)
parser.add_argument(
    "--disable-handoff-active-arm-mask",
    action="store_true",
    default=False,
    help=(
        "For 43D handoff-subtask eval, do not zero the inactive arm action. The active-arm one-hot "
        "state is still provided because it is part of the checkpoint input."
    ),
)
parser.add_argument(
    "--force-handoff-active-arm-mask",
    action="store_true",
    default=False,
    help=(
        "Eval-only debug mode for handoff checkpoints without active-arm state. Run the handoff "
        "scheduler only for execution-time masking, zeroing the inactive arm action without adding "
        "subtask or active-arm fields to observation.state."
    ),
)
parser.add_argument(
    "--camera-names",
    type=str,
    default="wrist_cam,observer_wrist_cam,global_cam",
    help="Scene cameras to record.",
)
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
parser.add_argument(
    "--teacher-forced-dataset-dir",
    type=Path,
    default=None,
    help=(
        "Optional LeRobot dataset root. When set, policy inputs come from this dataset episode's "
        "recorded observations while actions are executed in the live Isaac environment."
    ),
)
parser.add_argument(
    "--teacher-forced-raw-dir",
    type=Path,
    default=None,
    help=(
        "Optional raw demo root or episode directory produced by scripted_handoff_collect.py. "
        "When set, images are loaded directly from raw PNGs while observation.state is built "
        "from the live Isaac env."
    ),
)
parser.add_argument(
    "--teacher-forced-episode",
    type=int,
    default=0,
    help="Dataset episode index to use for --teacher-forced-dataset-dir.",
)
parser.add_argument(
    "--teacher-forced-start-frame",
    type=int,
    default=0,
    help="Frame offset inside the dataset episode for teacher-forced inputs.",
)
parser.add_argument(
    "--teacher-forced-raw-episode",
    type=int,
    default=None,
    help=(
        "Raw episode index to use with --teacher-forced-raw-dir. Defaults to "
        "--teacher-forced-episode. Ignored if --teacher-forced-raw-dir points at an episode directory."
    ),
)
parser.add_argument(
    "--teacher-forced-video-backend",
    type=str,
    default="torchcodec",
    choices=("torchcodec", "pyav"),
    help="LeRobot video backend used to decode teacher-forced dataset images.",
)
parser.add_argument(
    "--teacher-forced-use-dataset-task",
    action="store_true",
    default=False,
    help="Use the dataset sample's task text instead of --task-text in teacher-forced mode.",
)
parser.add_argument(
    "--teacher-forced-images-only",
    action="store_true",
    default=False,
    help=(
        "In teacher-forced mode, use dataset images but build observation.state from the live Isaac env. "
        "This diagnoses image-only open-loop forcing while actions still execute in simulation."
    ),
)
parser.add_argument(
    "--teacher-forced-live-image-keys",
    type=str,
    default="",
    help=(
        "Comma-separated image terms/features to source from live Isaac observations in "
        "--teacher-forced-images-only mode. Empty means all images come from the dataset. "
        "Examples: observer_wrist_rgb or observation.images.global_rgb,wrist_rgb."
    ),
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
from isaaclab.utils import math as math_utils

from handoff_v2_utils import (
    ACTION_REPRESENTATION_ABSOLUTE_JOINT_POS,
    ACTION_REPRESENTATION_DELTA_STEP,
    ACTION_REPRESENTATION_RELATIVE_CURRENT_POSE_CHUNK,
    load_manifest,
    preprocess_eval_images_torch,
    relative_current_action_to_delta_step_torch,
)


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


def _policy_action_queue_is_empty(policy) -> bool:
    queue = getattr(policy, "_queues", {}).get(ACTION)
    return queue is None or len(queue) == 0


def _parse_int_set(value: str) -> set[int]:
    if not value:
        return set()
    result = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        result.add(int(item))
    return result


def _should_record_policy_input_tensor_step(step: int) -> bool:
    steps = _parse_int_set(args_cli.policy_input_tensor_steps)
    return not steps or int(step) in steps


def _should_seed_policy_inference(policy, step: int, mode: str) -> bool:
    if mode == "first_call":
        return step == 0
    if mode == "each_chunk":
        return _policy_action_queue_is_empty(policy)
    if mode == "each_step":
        return True
    raise ValueError(f"Unsupported policy inference seed mode: {mode}")


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


def _relative_tcp_pose_right_in_left(env) -> tuple[torch.Tensor, torch.Tensor]:
    return _relative_tcp_pose(env, "robot", "observer_robot")


def _relative_tcp_pose_left_in_right(env) -> tuple[torch.Tensor, torch.Tensor]:
    return _relative_tcp_pose(env, "observer_robot", "robot")


def _arm_gripper_opening(env, arm_name: str) -> torch.Tensor:
    robot = _asset(env, arm_name)
    joint_ids = _arm_gripper_joint_ids(env, arm_name)
    return torch.sum(torch.abs(robot.data.joint_pos[:, joint_ids]), dim=1)


def _arm_gripper_is_open(env, arm_name: str) -> torch.Tensor:
    robot = _asset(env, arm_name)
    joint_ids = _arm_gripper_joint_ids(env, arm_name)
    finger_pos = torch.abs(robot.data.joint_pos[:, joint_ids])
    return torch.all(finger_pos >= GRIPPER_OPEN_COMMAND - GRIPPER_OPEN_THRESHOLD, dim=1)


def _scene_has_asset(env, asset_name: str) -> bool:
    try:
        env.unwrapped.scene[asset_name]
    except KeyError:
        return False
    return True


def _arm_state_snapshot(env, arm_name: str) -> dict | None:
    if not _scene_has_asset(env, arm_name):
        return None
    robot = _asset(env, arm_name)
    body_idx = _body_id(env, arm_name)
    gripper_joint_ids = _arm_gripper_joint_ids(env, arm_name)
    snapshot = {
        "joint_pos": _tensor_row(robot.data.joint_pos),
        "tcp_pos_w": _tensor_row(_tcp_pos_w(env, arm_name)),
        "tcp_quat_w": _tensor_row(_canonicalize_quat_wxyz(_normalize_quat_wxyz(_tcp_quat_w(env, arm_name)))),
        "hand_pos_w": _tensor_row(robot.data.body_pos_w[:, body_idx, :]),
        "hand_quat_w": _tensor_row(
            _canonicalize_quat_wxyz(_normalize_quat_wxyz(robot.data.body_quat_w[:, body_idx, :]))
        ),
        "gripper_opening": _tensor_row(_arm_gripper_opening(env, arm_name)),
        "gripper_joint_pos": _tensor_row(robot.data.joint_pos[:, gripper_joint_ids]),
    }
    joint_vel = getattr(robot.data, "joint_vel", None)
    if joint_vel is not None:
        snapshot["joint_vel"] = _tensor_row(joint_vel)
    return snapshot


def _env_state_snapshot(env, episode_progress: float | None = None) -> dict:
    snapshot = {
        "cube_pos_w": _tensor_row(_cube_pos_w(env)),
        "episode_progress": None if episode_progress is None else float(episode_progress),
        "arms": {
            "robot": _arm_state_snapshot(env, "robot"),
            "observer_robot": _arm_state_snapshot(env, "observer_robot"),
        },
    }
    if snapshot["arms"]["robot"] is not None and snapshot["arms"]["observer_robot"] is not None:
        right_in_left_pos, right_in_left_quat = _relative_tcp_pose_right_in_left(env)
        left_in_right_pos, left_in_right_quat = _relative_tcp_pose_left_in_right(env)
        snapshot["relative_tcp"] = {
            "right_in_left_pos": _tensor_row(right_in_left_pos),
            "right_in_left_quat": _tensor_row(right_in_left_quat),
            "left_in_right_pos": _tensor_row(left_in_right_pos),
            "left_in_right_quat": _tensor_row(left_in_right_quat),
        }
    return snapshot


def _handoff_area_center(device: str, center_w: tuple[float, float, float]) -> torch.Tensor:
    return torch.tensor(center_w, device=device, dtype=torch.float32).unsqueeze(0)


def _object_on_handoff_area(env, center_w: tuple[float, float, float], gripper_arm: str) -> torch.Tensor:
    cube_pos = _cube_pos_w(env)
    center = _handoff_area_center(env.unwrapped.device, center_w).to(dtype=cube_pos.dtype)
    xy_error = torch.abs(cube_pos[:, :2] - center[:, :2])
    inside = torch.logical_and(
        xy_error[:, 0] <= HANDOFF_AREA_SIZE_XY[0] * 0.5,
        xy_error[:, 1] <= HANDOFF_AREA_SIZE_XY[1] * 0.5,
    )
    low = torch.abs(cube_pos[:, 2] - center[:, 2]) <= HANDOFF_HEIGHT_TOLERANCE
    released = _arm_gripper_is_open(env, gripper_arm)
    return inside & low & released


class HandoffSuccessTracker:
    def __init__(self, env):
        self.yellow_seen = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device, dtype=torch.bool)

    def update(self, env) -> dict[str, torch.Tensor]:
        yellow_success = _object_on_handoff_area(env, HANDOFF_YELLOW_CENTER_W, "observer_robot")
        self.yellow_seen |= yellow_success
        red_success = _object_on_handoff_area(env, HANDOFF_RED_CENTER_W, "robot")
        success = self.yellow_seen & red_success
        return {
            "yellow_success": yellow_success.clone(),
            "yellow_seen": self.yellow_seen.clone(),
            "red_success": red_success.clone(),
            "success": success.clone(),
        }


def _arm_has_cube(env, arm_name: str) -> torch.Tensor:
    cube_pos = _cube_pos_w(env)
    tcp_pos = _tcp_pos_w(env, arm_name)
    near_tcp = torch.linalg.norm(cube_pos - tcp_pos, dim=1) <= HANDOFF_HAS_CUBE_TCP_DISTANCE
    lifted = cube_pos[:, 2] >= HANDOFF_HAS_CUBE_MIN_Z
    closed_on_object = _arm_gripper_opening(env, arm_name) <= HANDOFF_GRIPPER_HOLD_THRESHOLD
    return near_tcp & lifted & closed_on_object


def _one_hot_batch(index: int, size: int, env, device: torch.device) -> torch.Tensor:
    values = torch.zeros((env.unwrapped.num_envs, size), device=device, dtype=torch.float32)
    values[:, index] = 1.0
    return values


class HandoffSubtaskScheduler:
    def __init__(
        self,
        env,
        right_retreat_steps: int = HANDOFF_RIGHT_RETREAT_STEPS,
        scripted_right_retreat: bool = False,
    ):
        self.subtask_id = RIGHT_PICK_CUBE
        self.yellow_stable_steps = 0
        self.red_stable_steps = 0
        self.right_retreat_steps = max(0, int(right_retreat_steps))
        self.right_retreat_elapsed = 0
        self.scripted_right_retreat = bool(scripted_right_retreat)
        self.scripted_right_retreat_active = False
        self.right_retreat_reached = False
        self.right_retreat_target_w = _tcp_pos_w(env, "observer_robot").detach().clone()
        self.right_retreat_target_w[:, 2] = torch.maximum(
            self.right_retreat_target_w[:, 2],
            torch.full_like(self.right_retreat_target_w[:, 2], HANDOFF_SCRIPTED_RETREAT_Z),
        )
        self._device = env.unwrapped.device

    def update(self, env) -> dict:
        if self.subtask_id == RIGHT_PICK_CUBE:
            if bool(_arm_has_cube(env, "observer_robot")[0].detach().cpu().item()):
                self.subtask_id = RIGHT_PLACE_YELLOW
        elif self.subtask_id == RIGHT_PLACE_YELLOW:
            if self.scripted_right_retreat_active:
                self.right_retreat_elapsed += 1
                retreat_distance = torch.linalg.vector_norm(_tcp_pos_w(env, "observer_robot") - self.right_retreat_target_w, dim=1)
                self.right_retreat_reached = bool(
                    (retreat_distance <= HANDOFF_SCRIPTED_RETREAT_POS_THRESHOLD)[0].detach().cpu().item()
                )
                if self.right_retreat_reached or self.right_retreat_elapsed >= self.right_retreat_steps:
                    self.subtask_id = WAIT_YELLOW_STABLE
                    self.yellow_stable_steps = 0
                    self.scripted_right_retreat_active = False
            elif bool(_object_on_handoff_area(env, HANDOFF_YELLOW_CENTER_W, "observer_robot")[0].detach().cpu().item()):
                if self.scripted_right_retreat:
                    self.scripted_right_retreat_active = True
                    self.right_retreat_elapsed = 0
                    self.right_retreat_reached = False
                else:
                    self.right_retreat_elapsed += 1
                    if self.right_retreat_elapsed >= self.right_retreat_steps:
                        self.subtask_id = WAIT_YELLOW_STABLE
                        self.yellow_stable_steps = 0
            elif not self.scripted_right_retreat:
                self.right_retreat_elapsed = 0
        elif self.subtask_id == WAIT_YELLOW_STABLE:
            if bool(_object_on_handoff_area(env, HANDOFF_YELLOW_CENTER_W, "observer_robot")[0].detach().cpu().item()):
                self.yellow_stable_steps += 1
            else:
                self.yellow_stable_steps = 0
            if self.yellow_stable_steps >= HANDOFF_STABLE_STEPS:
                self.subtask_id = LEFT_PICK_FROM_YELLOW
        elif self.subtask_id == LEFT_PICK_FROM_YELLOW:
            if bool(_arm_has_cube(env, "robot")[0].detach().cpu().item()):
                self.subtask_id = LEFT_PLACE_RED
        elif self.subtask_id == LEFT_PLACE_RED:
            if bool(_object_on_handoff_area(env, HANDOFF_RED_CENTER_W, "robot")[0].detach().cpu().item()):
                self.subtask_id = DONE_HOLD
                self.red_stable_steps = 0
        elif self.subtask_id == DONE_HOLD:
            if bool(_object_on_handoff_area(env, HANDOFF_RED_CENTER_W, "robot")[0].detach().cpu().item()):
                self.red_stable_steps += 1
            else:
                self.red_stable_steps = 0

        active_arm_id = SUBTASK_ACTIVE_ARM_ID[self.subtask_id]
        device = torch.device(self._device)
        return {
            "subtask_id": torch.full((env.unwrapped.num_envs,), self.subtask_id, device=device, dtype=torch.long),
            "active_arm_id": torch.full((env.unwrapped.num_envs,), active_arm_id, device=device, dtype=torch.long),
            "subtask_name": SUBTASK_NAMES[self.subtask_id],
            "active_arm_name": ACTIVE_ARM_NAMES[active_arm_id],
            "yellow_stable_steps": self.yellow_stable_steps,
            "red_stable_steps": self.red_stable_steps,
            "right_retreat_elapsed": self.right_retreat_elapsed,
            "right_retreat_steps": self.right_retreat_steps,
            "scripted_right_retreat_active": self.scripted_right_retreat_active,
            "scripted_right_retreat": self.scripted_right_retreat,
            "right_retreat_reached": self.right_retreat_reached,
        }

    def scripted_retreat_action(self, env, action_shape) -> torch.Tensor | None:
        if not self.scripted_right_retreat_active:
            return None
        robot = _asset(env, "observer_robot")
        tcp_pos = _tcp_pos_w(env, "observer_robot")
        delta_w = self.right_retreat_target_w.to(device=tcp_pos.device, dtype=tcp_pos.dtype) - tcp_pos
        distance = torch.linalg.vector_norm(delta_w, dim=1)
        scale = torch.clamp(HANDOFF_SCRIPTED_RETREAT_MAX_DELTA / (distance + 1.0e-8), max=1.0).unsqueeze(-1)
        clipped_delta_w = delta_w * scale
        clipped_delta_b = math_utils.quat_apply_inverse(robot.data.root_quat_w, clipped_delta_w)

        action = torch.zeros(action_shape, device=env.unwrapped.device)
        action[:, 7:10] = clipped_delta_b / HANDOFF_SCRIPTED_RETREAT_ACTION_SCALE
        action[:, 13] = OPEN_ACTION
        return action


def _merge_handoff_status(
    success_status: dict[str, torch.Tensor] | None,
    scheduler_status: dict | None,
) -> dict | None:
    if success_status is None and scheduler_status is None:
        return None
    merged: dict = {}
    if success_status is not None:
        merged.update(success_status)
    if scheduler_status is not None:
        merged.update(scheduler_status)
    return merged


def _mask_inactive_handoff_arm(action: torch.Tensor, scheduler_status: dict | None) -> torch.Tensor:
    if scheduler_status is None:
        return action
    action = action.clone()
    active_arm_id = int(scheduler_status["active_arm_id"][0].detach().cpu().item())
    if active_arm_id == ACTIVE_RIGHT:
        action[:, 0:6] = 0.0
        action[:, 6] = 0.0
    elif active_arm_id == ACTIVE_LEFT:
        action[:, 7:13] = 0.0
        action[:, 13] = 0.0
    elif active_arm_id == ACTIVE_NONE:
        action[:, 0:6] = 0.0
        action[:, 6] = 0.0
        action[:, 7:13] = 0.0
        action[:, 13] = 0.0
    else:
        raise ValueError(f"Unknown active_arm_id={active_arm_id}")
    return action


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


def _save_chw_float_image(chw: torch.Tensor, image_path: Path) -> None:
    image = chw.detach().cpu()
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image, got shape={tuple(image.shape)}")
    if image.shape[0] == 4:
        image = image[:3]
    if image.shape[0] not in (1, 3):
        raise ValueError(f"Expected 1/3-channel CHW image, got shape={tuple(image.shape)}")
    image = image.clamp(0.0, 1.0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    image = image.permute(1, 2, 0).contiguous()
    _save_rgb_image(image, image_path)


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


def _checkpoint_uses_fixed_image_preprocess() -> bool:
    if not CHECKPOINT_MANIFEST:
        return False
    return CHECKPOINT_MANIFEST.get("image_normalization") in ("clip", "imagenet")


def _preprocess_policy_image(chw_or_batched: torch.Tensor, feature_key: str) -> torch.Tensor:
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


def _state_terms_for_dim(state_dim: int) -> tuple[str, ...]:
    if state_dim == 16:
        return ("joint_pos", "ee_position", "ee_quat")
    if state_dim == 7:
        return ("ee_position", "ee_quat")
    if state_dim == 26:
        return ()
    if state_dim == 34:
        return ()
    if state_dim == 41:
        return ()
    if state_dim == 43:
        return ()
    if state_dim == 49:
        return ()
    raise ValueError(
        f"Unsupported checkpoint observation.state dim={state_dim}; "
        "expected 16 for joint_ee, 7 for ee_only, 26 for handoff_joint_tcp_pos_gripper, "
        "34 for handoff_joint_ee, "
        "41 for handoff_joint_ee_relpose, 43 for handoff_joint_ee_subtask, "
        "or 49 for handoff_joint_ee_birelpose_time."
    )


def _build_handoff_state(env, device: torch.device) -> torch.Tensor:
    pieces = []
    for arm_name in ("robot", "observer_robot"):
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


def _build_handoff_joint_tcp_pos_gripper_state(env, device: torch.device) -> torch.Tensor:
    pieces = []
    for arm_name in ("robot", "observer_robot"):
        robot = _asset(env, arm_name)
        pieces.extend(
            [
                robot.data.joint_pos.to(device=device, dtype=torch.float32),
                _tcp_pos_w(env, arm_name).to(device=device, dtype=torch.float32),
                _arm_gripper_opening(env, arm_name).unsqueeze(-1).to(device=device, dtype=torch.float32),
            ]
        )
    state = torch.cat(pieces, dim=-1)
    if state.shape[-1] != 26:
        raise ValueError(f"Expected handoff joint/tcp/gripper state dim 26, got shape={tuple(state.shape)}")
    return state


def _current_abs_joint_pos_action(env, device: torch.device) -> torch.Tensor:
    state = torch.cat(
        [
            _asset(env, "robot").data.joint_pos.to(device=device, dtype=torch.float32),
            _asset(env, "observer_robot").data.joint_pos.to(device=device, dtype=torch.float32),
        ],
        dim=-1,
    )
    if state.shape[-1] != 18:
        raise ValueError(f"Expected current absolute joint action dim 18, got shape={tuple(state.shape)}")
    return state


def _build_handoff_subtask_state(env, device: torch.device, scheduler_status: dict | None) -> torch.Tensor:
    if scheduler_status is None:
        raise ValueError("handoff_joint_ee_subtask state requires scheduler status")
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
    state = torch.cat(
        [
            base_state,
            rel_pos.to(device=device, dtype=torch.float32),
            rel_quat.to(device=device, dtype=torch.float32),
        ],
        dim=-1,
    )
    if state.shape[-1] != 41:
        raise ValueError(f"Expected handoff relpose state dim 41, got shape={tuple(state.shape)}")
    return state


def _build_handoff_birelpose_time_state(
    env,
    device: torch.device,
    episode_progress: float,
) -> torch.Tensor:
    base_state = _build_handoff_state(env, device)
    right_in_left_pos, right_in_left_quat = _relative_tcp_pose_right_in_left(env)
    left_in_right_pos, left_in_right_quat = _relative_tcp_pose_left_in_right(env)
    progress = torch.full(
        (env.unwrapped.num_envs, 1),
        float(episode_progress),
        device=device,
        dtype=torch.float32,
    )
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


def _build_state(
    policy_obs: dict,
    device: torch.device,
    state_dim: int,
    env=None,
    scheduler_status: dict | None = None,
    episode_progress: float | None = None,
) -> torch.Tensor:
    if state_dim in (26, 34, 41, 43, 49):
        if env is None:
            raise ValueError("handoff state requires env scene access")
        if state_dim == 26:
            return _build_handoff_joint_tcp_pos_gripper_state(env, device)
        if state_dim == 34:
            return _build_handoff_state(env, device)
        if state_dim == 41:
            return _build_handoff_relpose_state(env, device)
        if state_dim == 49:
            if episode_progress is None:
                raise ValueError("handoff_joint_ee_birelpose_time state requires episode_progress")
            return _build_handoff_birelpose_time_state(env, device, episode_progress)
        return _build_handoff_subtask_state(env, device, scheduler_status)

    pieces = []
    for term in _state_terms_for_dim(state_dim):
        value = policy_obs.get(term)
        if value is None:
            raise KeyError(f"Missing observation term: {term}")
        pieces.append(value.to(device=device, dtype=torch.float32))
    return torch.cat(pieces, dim=-1)


def _build_policy_batch(
    obs,
    config,
    tokenizer,
    stats,
    device: torch.device,
    task_text: str,
    env=None,
    scheduler_status: dict | None = None,
    episode_progress: float | None = None,
) -> dict[str, torch.Tensor]:
    policy_obs = _policy_obs(obs)
    state_dim = _state_dim_from_config(config)
    state = _build_state(
        policy_obs,
        device,
        state_dim,
        env=env,
        scheduler_status=scheduler_status,
        episode_progress=episode_progress,
    )
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


def _dataset_image_to_chw_float(image: torch.Tensor, device: torch.device) -> torch.Tensor:
    image = image.to(device=device, dtype=torch.float32)
    if image.ndim != 3:
        raise ValueError(f"Expected dataset image as CHW or HWC, got shape={tuple(image.shape)}")
    if image.shape[0] in (1, 3, 4):
        if image.shape[0] == 4:
            image = image[:3]
        chw = image
    elif image.shape[-1] in (1, 3, 4):
        if image.shape[-1] == 4:
            image = image[..., :3]
        chw = image.permute(2, 0, 1).contiguous()
    else:
        raise ValueError(f"Could not infer channel dimension for dataset image shape={tuple(image.shape)}")
    if chw.max() > 1.5:
        chw = chw / 255.0
    return chw.contiguous()


def _image_term_from_feature_key(feature_key: str) -> str:
    return feature_key.removeprefix("observation.images.")


def _canonical_image_key(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    return _image_term_from_feature_key(value)


def _parse_image_key_set(value: str) -> set[str]:
    return {
        key
        for key in (_canonical_image_key(piece) for piece in value.split(","))
        if key
    }


def _build_policy_batch_from_dataset_sample(
    sample: dict,
    config,
    tokenizer,
    stats,
    device: torch.device,
    task_text: str,
) -> dict[str, torch.Tensor]:
    state = sample.get(OBS_STATE)
    if state is None:
        raise KeyError(f"Dataset sample is missing {OBS_STATE}")
    state = state.to(device=device, dtype=torch.float32).unsqueeze(0)
    expected_state_dim = _state_dim_from_config(config)
    if state.shape[-1] != expected_state_dim:
        raise ValueError(
            f"Teacher-forced dataset state dim={state.shape[-1]}, checkpoint expects {expected_state_dim}"
        )

    batch: dict[str, torch.Tensor] = {
        OBS_STATE: _normalize_tensor(state, stats[OBS_STATE], _normalization_mode(config, OBS_STATE)),
    }

    for feature_key in config.image_features:
        image = sample.get(feature_key)
        if image is None:
            raise KeyError(f"Dataset sample is missing image feature: {feature_key}")
        chw = _dataset_image_to_chw_float(image, device).unsqueeze(0)
        chw = _preprocess_policy_image(chw, feature_key)
        batch[feature_key] = _normalize_tensor(chw, stats[feature_key], _normalization_mode(config, feature_key))

    tokens = tokenizer(
        [task_text],
        max_length=config.tokenizer_max_length,
        padding=config.tokenizer_padding,
        truncation=config.tokenizer_truncation,
        return_tensors="pt",
    )
    batch[OBS_LANGUAGE_TOKENS] = tokens["input_ids"].to(device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = tokens["attention_mask"].to(device)
    return batch


def _build_policy_batch_from_dataset_images_live_state(
    sample: dict,
    obs,
    config,
    tokenizer,
    stats,
    device: torch.device,
    task_text: str,
    env,
    scheduler_status: dict | None = None,
    episode_progress: float | None = None,
    live_image_keys: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    policy_obs = _policy_obs(obs)
    live_image_keys = live_image_keys or set()
    state_dim = _state_dim_from_config(config)
    state = _build_state(
        policy_obs,
        device,
        state_dim,
        env=env,
        scheduler_status=scheduler_status,
        episode_progress=episode_progress,
    )
    batch: dict[str, torch.Tensor] = {
        OBS_STATE: _normalize_tensor(state, stats[OBS_STATE], _normalization_mode(config, OBS_STATE)),
    }

    for feature_key in config.image_features:
        term_name = _image_term_from_feature_key(feature_key)
        if term_name in live_image_keys:
            image = policy_obs.get(term_name)
            if image is None:
                raise KeyError(f"Missing live image observation term: {term_name}")
            chw = torch.stack([_image_to_chw_float(frame, device) for frame in image], dim=0)
        else:
            image = sample.get(feature_key)
            if image is None:
                raise KeyError(f"Dataset sample is missing image feature: {feature_key}")
            chw = _dataset_image_to_chw_float(image, device).unsqueeze(0)
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


def _teacher_forced_raw_episode_dir(raw_dir: Path, episode_index: int) -> Path:
    if (raw_dir / "steps.jsonl").exists():
        return raw_dir
    episode_dir = raw_dir / f"episode_{episode_index:06d}"
    if not (episode_dir / "steps.jsonl").exists():
        raise FileNotFoundError(
            f"Could not find raw steps.jsonl in {raw_dir} or {episode_dir}"
        )
    return episode_dir


def _load_teacher_forced_raw_rows(
    raw_dir: Path,
    episode_index: int,
    start_frame: int,
) -> tuple[Path, list[dict]]:
    if start_frame < 0:
        raise ValueError("--teacher-forced-start-frame must be non-negative")
    episode_dir = _teacher_forced_raw_episode_dir(raw_dir, episode_index)
    steps_path = episode_dir / "steps.jsonl"
    rows = [json.loads(line) for line in steps_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if start_frame >= len(rows):
        raise ValueError(
            f"--teacher-forced-start-frame={start_frame} is outside raw episode length {len(rows)}"
        )
    return episode_dir, rows[start_frame:]


def _raw_demo_image_to_chw_float(
    episode_dir: Path,
    row: dict,
    feature_key: str,
    device: torch.device,
) -> torch.Tensor:
    term_name = _image_term_from_feature_key(feature_key)
    images = row.get("pre_images") if CHECKPOINT_MANIFEST and CHECKPOINT_MANIFEST.get("state_timing") == "exact_pre_action" else None
    if not isinstance(images, dict):
        images = row.get("images", {})
    relative_path = images.get(term_name)
    if relative_path is None:
        raise KeyError(f"Raw row is missing image term {term_name!r}: available={sorted(images)}")
    image_path = episode_dir / relative_path
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    tensor = torch.from_numpy(np.array(image))
    return _dataset_image_to_chw_float(tensor, device)


def _build_policy_batch_from_raw_images_live_state(
    raw_row: dict,
    raw_episode_dir: Path,
    obs,
    config,
    tokenizer,
    stats,
    device: torch.device,
    task_text: str,
    env,
    scheduler_status: dict | None = None,
    episode_progress: float | None = None,
    live_image_keys: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    policy_obs = _policy_obs(obs)
    live_image_keys = live_image_keys or set()
    state_dim = _state_dim_from_config(config)
    state = _build_state(
        policy_obs,
        device,
        state_dim,
        env=env,
        scheduler_status=scheduler_status,
        episode_progress=episode_progress,
    )
    batch: dict[str, torch.Tensor] = {
        OBS_STATE: _normalize_tensor(state, stats[OBS_STATE], _normalization_mode(config, OBS_STATE)),
    }

    for feature_key in config.image_features:
        term_name = _image_term_from_feature_key(feature_key)
        if term_name in live_image_keys:
            image = policy_obs.get(term_name)
            if image is None:
                raise KeyError(f"Missing live image observation term: {term_name}")
            chw = torch.stack([_image_to_chw_float(frame, device) for frame in image], dim=0)
        else:
            chw = _raw_demo_image_to_chw_float(raw_episode_dir, raw_row, feature_key, device).unsqueeze(0)
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


def _sample_scalar(sample: dict, key: str, default=None):
    value = sample.get(key, default)
    if torch.is_tensor(value):
        return value.detach().cpu().item()
    return value


def _sample_tensor_list(sample: dict, key: str):
    value = sample.get(key)
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _teacher_forced_episode_bounds(dataset, episode_index: int, start_frame: int) -> tuple[int, int, dict]:
    if episode_index < 0:
        raise ValueError("--teacher-forced-episode must be non-negative")
    if start_frame < 0:
        raise ValueError("--teacher-forced-start-frame must be non-negative")

    episode_row = None
    for row in dataset.meta.episodes:
        if int(row["episode_index"]) == episode_index:
            episode_row = row
            break
    if episode_row is None:
        raise ValueError(
            f"Teacher-forced episode {episode_index} not found in dataset with {dataset.num_episodes} episode(s)"
        )

    episode_length = int(episode_row["length"])
    if start_frame >= episode_length:
        raise ValueError(
            f"--teacher-forced-start-frame={start_frame} is outside episode length {episode_length}"
        )
    start_index = int(episode_row["dataset_from_index"]) + start_frame
    end_index = int(episode_row["dataset_to_index"])
    return start_index, end_index, dict(episode_row)


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
        self.policy_inputs_file = None
        self.image_paths: dict[str, list[Path]] = {}
        self.image_counts: dict[str, int] = {}
        self.recorded_steps = 0

    def start_episode(self, episode: int, env, config, fixed_cube_xy: tuple[float, float] | None = None) -> None:
        self.close()
        self.episode_dir = self.root_dir / f"episode_{episode:06d}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.steps_file = (self.episode_dir / "steps.jsonl").open("w", encoding="utf-8")
        self.policy_inputs_file = (self.episode_dir / "policy_input_index.jsonl").open("w", encoding="utf-8")
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
            "policy_inference_seed": args_cli.policy_inference_seed,
            "policy_inference_seed_mode": args_cli.policy_inference_seed_mode,
            "state_dim": _state_dim_from_config(config),
            "state_mode": STATE_MODE_BY_DIM.get(_state_dim_from_config(config), "unknown"),
            "handoff_time_total_steps": args_cli.handoff_time_total_steps,
            "action_space_shape": tuple(env.action_space.shape),
            "camera_names": _camera_names(),
            "record_image_every": args_cli.record_image_every,
            "record_policy_inputs": args_cli.record_policy_inputs,
            "record_policy_input_tensors": args_cli.record_policy_input_tensors,
            "policy_input_tensor_steps": args_cli.policy_input_tensor_steps,
            "fixed_cube_xy_robot": list(fixed_cube_xy) if fixed_cube_xy is not None else None,
            "teacher_forced_dataset_dir": (
                str(args_cli.teacher_forced_dataset_dir) if args_cli.teacher_forced_dataset_dir is not None else None
            ),
            "teacher_forced_raw_dir": (
                str(args_cli.teacher_forced_raw_dir) if args_cli.teacher_forced_raw_dir is not None else None
            ),
            "teacher_forced_episode": args_cli.teacher_forced_episode,
            "teacher_forced_raw_episode": args_cli.teacher_forced_raw_episode,
            "teacher_forced_start_frame": args_cli.teacher_forced_start_frame,
            "teacher_forced_images_only": args_cli.teacher_forced_images_only,
            "teacher_forced_live_image_keys": sorted(_parse_image_key_set(args_cli.teacher_forced_live_image_keys)),
        }
        (self.episode_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def record_policy_inputs(
        self,
        step: int,
        batch: dict[str, torch.Tensor],
        config,
        stats: dict[str, dict[str, torch.Tensor]],
        extra_fields: dict | None = None,
    ) -> None:
        if (
            not args_cli.record_policy_inputs
            and not args_cli.record_policy_input_tensors
        ) or self.episode_dir is None or self.policy_inputs_file is None:
            return
        image_paths = {}
        if args_cli.record_policy_inputs:
            for feature_key in config.image_features:
                if feature_key not in batch:
                    continue
                term_name = _image_term_from_feature_key(feature_key)
                image = _unnormalize_tensor(
                    batch[feature_key],
                    stats[feature_key],
                    _normalization_mode(config, feature_key),
                )
                if image.ndim != 4 or image.shape[0] < 1:
                    raise ValueError(f"Expected batched image for {feature_key}, got shape={tuple(image.shape)}")
                image_path = self.episode_dir / "policy_inputs" / term_name / f"{step:06d}.png"
                _save_chw_float_image(image[0], image_path)
                image_paths[term_name] = str(image_path.relative_to(self.episode_dir))
        row = {
            "step": int(step),
            "images": image_paths,
        }
        if args_cli.record_policy_input_tensors and _should_record_policy_input_tensor_step(step):
            tensor_path = self.episode_dir / "policy_input_tensors" / f"{step:06d}.pt"
            tensor_path.parent.mkdir(parents=True, exist_ok=True)
            tensors = {
                key: value.detach().cpu()
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            torch.save(tensors, tensor_path)
            row["batch_tensor"] = str(tensor_path.relative_to(self.episode_dir))
        if extra_fields is not None:
            row.update(extra_fields)
        self.policy_inputs_file.write(json.dumps(row) + "\n")

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
        handoff_status: dict[str, torch.Tensor] | None = None,
        extra_fields: dict | None = None,
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
        if handoff_status is not None:
            if "yellow_success" in handoff_status:
                row.update(
                    {
                        "yellow_success": bool(handoff_status["yellow_success"][0].detach().cpu().item()),
                        "yellow_seen": bool(handoff_status["yellow_seen"][0].detach().cpu().item()),
                        "red_success": bool(handoff_status["red_success"][0].detach().cpu().item()),
                    }
                )
            if "subtask_id" in handoff_status:
                row.update(
                    {
                        "subtask_id": int(handoff_status["subtask_id"][0].detach().cpu().item()),
                        "subtask_name": handoff_status["subtask_name"],
                        "active_arm_id": int(handoff_status["active_arm_id"][0].detach().cpu().item()),
                        "active_arm_name": handoff_status["active_arm_name"],
                        "yellow_stable_steps": int(handoff_status["yellow_stable_steps"]),
                        "red_stable_steps": int(handoff_status["red_stable_steps"]),
                        "right_retreat_elapsed": int(handoff_status.get("right_retreat_elapsed", 0)),
                        "right_retreat_steps": int(handoff_status.get("right_retreat_steps", 0)),
                    }
                )
        if extra_fields is not None:
            row.update(extra_fields)
        self.steps_file.write(json.dumps(row) + "\n")
        self.recorded_steps += 1

    def finish_episode(
        self,
        success: torch.Tensor,
        steps: int,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        fixed_cube_xy: tuple[float, float] | None = None,
        handoff_status: dict[str, torch.Tensor] | None = None,
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
        if handoff_status is not None:
            if "yellow_success" in handoff_status:
                summary.update(
                    {
                        "yellow_success": bool(handoff_status["yellow_success"][0].detach().cpu().item()),
                        "yellow_seen": bool(handoff_status["yellow_seen"][0].detach().cpu().item()),
                        "red_success": bool(handoff_status["red_success"][0].detach().cpu().item()),
                    }
                )
            if "subtask_id" in handoff_status:
                summary.update(
                    {
                        "subtask_id": int(handoff_status["subtask_id"][0].detach().cpu().item()),
                        "subtask_name": handoff_status["subtask_name"],
                        "active_arm_id": int(handoff_status["active_arm_id"][0].detach().cpu().item()),
                        "active_arm_name": handoff_status["active_arm_name"],
                        "yellow_stable_steps": int(handoff_status["yellow_stable_steps"]),
                        "red_stable_steps": int(handoff_status["red_stable_steps"]),
                        "right_retreat_elapsed": int(handoff_status.get("right_retreat_elapsed", 0)),
                        "right_retreat_steps": int(handoff_status.get("right_retreat_steps", 0)),
                    }
                )
        if self.episode_dir is not None:
            (self.episode_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        self.close()
        return summary

    def close(self) -> None:
        if self.steps_file is not None:
            self.steps_file.close()
            self.steps_file = None
        if self.policy_inputs_file is not None:
            self.policy_inputs_file.close()
            self.policy_inputs_file = None


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
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint weights do not fully match policy model: "
            f"missing_keys={len(missing_keys)} unexpected_keys={len(unexpected_keys)}"
        )


def main() -> None:
    global CHECKPOINT_MANIFEST
    if args_cli.num_envs != 1:
        raise ValueError("This first visual eval recorder supports --num_envs 1 only.")
    if args_cli.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args_cli.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    if args_cli.record_image_every < 0:
        raise ValueError("--record-image-every must be non-negative.")
    if args_cli.handoff_time_total_steps <= 0:
        raise ValueError("--handoff-time-total-steps must be positive.")
    if args_cli.teacher_forced_dataset_dir is not None and args_cli.teacher_forced_raw_dir is not None:
        raise ValueError("Use either --teacher-forced-dataset-dir or --teacher-forced-raw-dir, not both.")
    teacher_forced_enabled = args_cli.teacher_forced_dataset_dir is not None or args_cli.teacher_forced_raw_dir is not None
    if teacher_forced_enabled and args_cli.episodes != 1:
        raise ValueError("Teacher-forced visualization currently supports --episodes 1 only.")
    if args_cli.teacher_forced_images_only and args_cli.teacher_forced_dataset_dir is None:
        raise ValueError("--teacher-forced-images-only currently applies to --teacher-forced-dataset-dir.")
    if args_cli.teacher_forced_live_image_keys and not (
        args_cli.teacher_forced_images_only or args_cli.teacher_forced_raw_dir is not None
    ):
        raise ValueError(
            "--teacher-forced-live-image-keys requires --teacher-forced-images-only or --teacher-forced-raw-dir."
        )
    if args_cli.teacher_forced_dataset_dir is not None and not args_cli.teacher_forced_dataset_dir.exists():
        raise FileNotFoundError(args_cli.teacher_forced_dataset_dir)
    if args_cli.teacher_forced_raw_dir is not None and not args_cli.teacher_forced_raw_dir.exists():
        raise FileNotFoundError(args_cli.teacher_forced_raw_dir)
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
    CHECKPOINT_MANIFEST = load_manifest(args_cli.checkpoint)

    _patch_lerobot_namespace_imports()
    _mock_groot_imports()
    LeRobotDataset = None
    if args_cli.teacher_forced_dataset_dir is not None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
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
        ACTION_REPRESENTATION_ABSOLUTE_JOINT_POS,
        ACTION_REPRESENTATION_DELTA_STEP,
        ACTION_REPRESENTATION_RELATIVE_CURRENT_POSE_CHUNK,
    ):
        raise ValueError(f"Unsupported checkpoint action_representation={action_representation!r}")
    stats = _load_stats(args_cli.checkpoint, device)
    tokenizer = CLIPTokenizer.from_pretrained(config.text_encoder_name)
    policy = MultiTaskDiTPolicy(config).to(device)
    _load_policy_weights(policy, args_cli.checkpoint, device)
    policy.eval()

    teacher_forced_dataset = None
    teacher_forced_start_index = None
    teacher_forced_end_index = None
    teacher_forced_episode_row = None
    teacher_forced_raw_episode_dir = None
    teacher_forced_raw_rows = None
    if args_cli.teacher_forced_dataset_dir is not None:
        if LeRobotDataset is None:
            raise RuntimeError("LeRobotDataset import unexpectedly unavailable")
        teacher_forced_dataset = LeRobotDataset(
            repo_id=args_cli.teacher_forced_dataset_dir.name,
            root=args_cli.teacher_forced_dataset_dir,
            video_backend=args_cli.teacher_forced_video_backend,
        )
        teacher_forced_start_index, teacher_forced_end_index, teacher_forced_episode_row = (
            _teacher_forced_episode_bounds(
                teacher_forced_dataset,
                args_cli.teacher_forced_episode,
                args_cli.teacher_forced_start_frame,
            )
        )
        args_cli.max_steps = min(args_cli.max_steps, teacher_forced_end_index - teacher_forced_start_index)
        _log(f"[INFO] teacher_forced_dataset_dir={args_cli.teacher_forced_dataset_dir}")
        _log(f"[INFO] teacher_forced_images_only={args_cli.teacher_forced_images_only}")
        _log(
            "[INFO] teacher_forced_episode="
            f"{args_cli.teacher_forced_episode} dataset_index_range="
            f"[{teacher_forced_start_index}, {teacher_forced_end_index}) max_steps={args_cli.max_steps}"
        )
    if args_cli.teacher_forced_raw_dir is not None:
        raw_episode_index = (
            args_cli.teacher_forced_episode
            if args_cli.teacher_forced_raw_episode is None
            else args_cli.teacher_forced_raw_episode
        )
        teacher_forced_raw_episode_dir, teacher_forced_raw_rows = _load_teacher_forced_raw_rows(
            args_cli.teacher_forced_raw_dir,
            raw_episode_index,
            args_cli.teacher_forced_start_frame,
        )
        args_cli.max_steps = min(args_cli.max_steps, len(teacher_forced_raw_rows))
        teacher_forced_episode_row = {
            "raw_episode_dir": str(teacher_forced_raw_episode_dir),
            "raw_episode_index": raw_episode_index,
            "start_frame": args_cli.teacher_forced_start_frame,
            "length": len(teacher_forced_raw_rows),
        }
        _log(f"[INFO] teacher_forced_raw_dir={args_cli.teacher_forced_raw_dir}")
        _log(f"[INFO] teacher_forced_raw_episode_dir={teacher_forced_raw_episode_dir}")
        _log(
            "[INFO] teacher_forced_raw_episode="
            f"{raw_episode_index} start_frame={args_cli.teacher_forced_start_frame} "
            f"frames={len(teacher_forced_raw_rows)} max_steps={args_cli.max_steps}"
        )

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
    _log(
        "[INFO] policy_inference_seed="
        f"{args_cli.policy_inference_seed} mode={args_cli.policy_inference_seed_mode}"
    )
    _log(f"[INFO] image_features={list(config.image_features.keys())}")
    _log(f"[INFO] state_dim={state_dim} state_mode={state_mode}")
    if CHECKPOINT_MANIFEST is not None:
        _log(
            "[INFO] checkpoint_manifest="
            f"dataset_version={CHECKPOINT_MANIFEST.get('dataset_version')} "
            f"state_timing={CHECKPOINT_MANIFEST.get('state_timing')} "
            f"image_normalization={CHECKPOINT_MANIFEST.get('image_normalization')} "
            f"image_augmentation={CHECKPOINT_MANIFEST.get('image_augmentation')} "
            f"action_representation={CHECKPOINT_MANIFEST.get('action_representation')}"
        )
    _log(f"[INFO] action_representation={action_representation}")
    if action_representation == ACTION_REPRESENTATION_ABSOLUTE_JOINT_POS and env.action_space.shape[-1] != 18:
        raise ValueError(
            "absolute_joint_pos checkpoints require an 18D joint-position action environment. "
            "Use TASK=Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-Joint-Pos-Visuomotor-v0."
        )
    if state_dim == 49:
        _log(f"[INFO] handoff_time_total_steps={args_cli.handoff_time_total_steps}")
    handoff_progress_denominator = max(args_cli.handoff_time_total_steps - 1, 1)
    if args_cli.force_handoff_active_arm_mask:
        _log("[INFO] force_handoff_active_arm_mask=True (eval-only execution mask)")
    teacher_forced_live_image_keys = _parse_image_key_set(args_cli.teacher_forced_live_image_keys)
    valid_image_keys = {_image_term_from_feature_key(feature_key) for feature_key in config.image_features}
    unknown_live_image_keys = teacher_forced_live_image_keys - valid_image_keys
    if unknown_live_image_keys:
        raise ValueError(
            "--teacher-forced-live-image-keys contains unknown image term(s): "
            f"{sorted(unknown_live_image_keys)}; valid terms are {sorted(valid_image_keys)}"
        )
    if teacher_forced_live_image_keys:
        _log(f"[INFO] teacher_forced_live_image_keys={sorted(teacher_forced_live_image_keys)}")

    summaries = []
    try:
        for episode in range(args_cli.episodes):
            if not simulation_app.is_running():
                break
            _seed_everything(args_cli.seed + episode)
            _log(f"[EP {episode + 1}] reset_policy_start")
            policy.reset()
            _log(f"[EP {episode + 1}] env_reset_start")
            reset_out = env.reset(seed=args_cli.seed + episode)
            _log(f"[EP {episode + 1}] env_reset_done")
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
                if action_representation == ACTION_REPRESENTATION_ABSOLUTE_JOINT_POS:
                    warmup_actions = _current_abs_joint_pos_action(env, env_device)
                    if args_cli.warmup_open_gripper:
                        warmup_actions[:, [7, 8, 16, 17]] = GRIPPER_OPEN_COMMAND
                else:
                    warmup_actions = torch.zeros(env.action_space.shape, device=env_device)
                if (
                    action_representation != ACTION_REPRESENTATION_ABSOLUTE_JOINT_POS
                    and args_cli.warmup_open_gripper
                    and warmup_actions.shape[1] > 6
                ):
                    warmup_actions[:, 6] = OPEN_ACTION
                if (
                    action_representation != ACTION_REPRESENTATION_ABSOLUTE_JOINT_POS
                    and args_cli.warmup_open_gripper
                    and warmup_actions.shape[1] > 13
                ):
                    warmup_actions[:, 13] = OPEN_ACTION
                for _ in range(args_cli.warmup_steps):
                    obs, _, warmup_terminated, warmup_truncated, _ = env.step(warmup_actions)
                    if warmup_terminated.any() or warmup_truncated.any():
                        break

            recorder.start_episode(episode, env, config, fixed_cube_xy)
            success = torch.zeros(env.unwrapped.num_envs, device=env_device, dtype=torch.bool)
            terminated = torch.zeros_like(success)
            truncated = torch.zeros_like(success)
            reward = torch.zeros(env.unwrapped.num_envs, device=env_device)
            handoff_tracker = HandoffSuccessTracker(env) if state_dim in (26, 34, 41, 43, 49) else None
            use_handoff_scheduler = state_dim == 43 or (
                args_cli.force_handoff_active_arm_mask and state_dim in (26, 34, 41, 49)
            )
            if action_representation == ACTION_REPRESENTATION_ABSOLUTE_JOINT_POS:
                use_handoff_scheduler = False
            subtask_scheduler = (
                HandoffSubtaskScheduler(
                    env,
                    right_retreat_steps=args_cli.handoff_right_retreat_steps,
                    scripted_right_retreat=args_cli.handoff_scripted_right_retreat,
                )
                if use_handoff_scheduler
                else None
            )
            success_status = handoff_tracker.update(env) if handoff_tracker is not None else None
            scheduler_status = subtask_scheduler.update(env) if subtask_scheduler is not None else None
            handoff_status = _merge_handoff_status(success_status, scheduler_status)
            last_step = 0
            policy_seed_event_count = 0
            previous_relative_action = None

            for step in range(args_cli.max_steps):
                if not simulation_app.is_running():
                    break
                episode_progress = min(step / handoff_progress_denominator, 1.0) if state_dim == 49 else None
                with torch.no_grad():
                    teacher_forced_sample = None
                    teacher_forced_extra = None
                    if teacher_forced_dataset is not None:
                        dataset_index = teacher_forced_start_index + step
                        if dataset_index >= teacher_forced_end_index:
                            break
                        teacher_forced_sample = teacher_forced_dataset[dataset_index]
                        teacher_task_text = (
                            teacher_forced_sample.get("task", args_cli.task_text)
                            if args_cli.teacher_forced_use_dataset_task
                            else args_cli.task_text
                        )
                        if args_cli.teacher_forced_images_only:
                            batch = _build_policy_batch_from_dataset_images_live_state(
                                teacher_forced_sample,
                                obs,
                                config,
                                tokenizer,
                                stats,
                                device,
                                teacher_task_text,
                                env=env,
                                scheduler_status=scheduler_status,
                                episode_progress=episode_progress,
                                live_image_keys=teacher_forced_live_image_keys,
                            )
                        else:
                            batch = _build_policy_batch_from_dataset_sample(
                                teacher_forced_sample,
                                config,
                                tokenizer,
                                stats,
                                device,
                                teacher_task_text,
                            )
                        teacher_forced_extra = {
                            "teacher_forced_images_only": args_cli.teacher_forced_images_only,
                            "teacher_forced_live_image_keys": sorted(teacher_forced_live_image_keys),
                            "teacher_forced_dataset_index": int(dataset_index),
                            "teacher_forced_episode_index": int(
                                _sample_scalar(teacher_forced_sample, "episode_index", args_cli.teacher_forced_episode)
                            ),
                            "teacher_forced_frame_index": int(
                                _sample_scalar(teacher_forced_sample, "frame_index", step)
                            ),
                            "teacher_forced_timestamp": float(
                                _sample_scalar(teacher_forced_sample, "timestamp", step / 50.0)
                            ),
                            "teacher_forced_progress": float(
                                teacher_forced_sample[OBS_STATE][-1].detach().cpu().item()
                            ),
                            "teacher_forced_expert_action": _sample_tensor_list(teacher_forced_sample, ACTION),
                        }
                    elif teacher_forced_raw_rows is not None:
                        if step >= len(teacher_forced_raw_rows):
                            break
                        raw_row = teacher_forced_raw_rows[step]
                        batch = _build_policy_batch_from_raw_images_live_state(
                            raw_row,
                            teacher_forced_raw_episode_dir,
                            obs,
                            config,
                            tokenizer,
                            stats,
                            device,
                            args_cli.task_text,
                            env=env,
                            scheduler_status=scheduler_status,
                            episode_progress=episode_progress,
                            live_image_keys=teacher_forced_live_image_keys,
                        )
                        teacher_forced_extra = {
                            "teacher_forced_raw_dir": str(args_cli.teacher_forced_raw_dir),
                            "teacher_forced_raw_episode_dir": str(teacher_forced_raw_episode_dir),
                            "teacher_forced_raw_row_index": int(step + args_cli.teacher_forced_start_frame),
                            "teacher_forced_frame_index": int(raw_row.get("record_step", raw_row.get("step", step))),
                            "teacher_forced_timestamp": float(raw_row.get("timestamp", step / 50.0)),
                            "teacher_forced_phase": raw_row.get("phase"),
                            "teacher_forced_stage": raw_row.get("stage"),
                            "teacher_forced_active_arm": raw_row.get("active_arm"),
                            "teacher_forced_live_image_keys": sorted(teacher_forced_live_image_keys),
                            "teacher_forced_expert_action": raw_row.get(ACTION),
                        }
                    else:
                        batch = _build_policy_batch(
                            obs,
                            config,
                            tokenizer,
                            stats,
                            device,
                            args_cli.task_text,
                            env=env,
                            scheduler_status=scheduler_status,
                            episode_progress=episode_progress,
                        )
                    will_infer_policy = _policy_action_queue_is_empty(policy)
                    if (
                        args_cli.record_policy_inputs
                        or args_cli.record_policy_input_tensors
                    ) and will_infer_policy:
                        policy_input_extra = {
                            "policy_queue_empty": True,
                            "source": "live",
                            "episode_progress": float(episode_progress) if episode_progress is not None else None,
                        }
                        if teacher_forced_dataset is not None:
                            policy_input_extra["source"] = (
                                "lerobot_dataset_images_live_state"
                                if args_cli.teacher_forced_images_only
                                else "lerobot_dataset"
                            )
                        elif teacher_forced_raw_rows is not None:
                            policy_input_extra["source"] = "raw_png_images_live_state"
                        if teacher_forced_extra is not None:
                            for key in (
                                "teacher_forced_dataset_index",
                                "teacher_forced_raw_row_index",
                                "teacher_forced_episode_index",
                                "teacher_forced_frame_index",
                                "teacher_forced_timestamp",
                                "teacher_forced_progress",
                                "teacher_forced_phase",
                                "teacher_forced_stage",
                                "teacher_forced_active_arm",
                                "teacher_forced_images_only",
                                "teacher_forced_live_image_keys",
                            ):
                                if key in teacher_forced_extra:
                                    policy_input_extra[key] = teacher_forced_extra[key]
                        recorder.record_policy_inputs(step, batch, config, stats, policy_input_extra)
                    if args_cli.policy_inference_seed is not None and _should_seed_policy_inference(
                        policy, step, args_cli.policy_inference_seed_mode
                    ):
                        inference_seed = int(args_cli.policy_inference_seed) + policy_seed_event_count
                        _seed_everything(inference_seed)
                        if step == 0 or args_cli.policy_inference_seed_mode != "each_step":
                            _log(
                                f"[EP {episode + 1} STEP {step}] policy_inference_seed={inference_seed} "
                                f"mode={args_cli.policy_inference_seed_mode}"
                            )
                        policy_seed_event_count += 1

                    model_action = policy.select_action(batch)
                    env_action = _unnormalize_tensor(
                        model_action,
                        stats[ACTION],
                        _normalization_mode(config, ACTION),
                    ).to(env_device)
                    if action_representation == ACTION_REPRESENTATION_RELATIVE_CURRENT_POSE_CHUNK:
                        if will_infer_policy:
                            previous_relative_action = None
                        env_action, previous_relative_action = relative_current_action_to_delta_step_torch(
                            env_action,
                            previous_relative_action,
                        )
                    action_handoff_status = handoff_status
                    scripted_action = (
                        subtask_scheduler.scripted_retreat_action(env, env.action_space.shape)
                        if subtask_scheduler is not None
                        else None
                    )
                    if scripted_action is not None:
                        env_action = scripted_action
                    elif subtask_scheduler is not None and not args_cli.disable_handoff_active_arm_mask:
                        env_action = _mask_inactive_handoff_arm(env_action, scheduler_status)
                    env_action = _clamp_action_for_env(env_action, env)
                    cube_pos = _cube_pos_w(env).clone()
                    ee_pos = _ee_pos_w(env).clone()
                    opening = _gripper_opening(env, gripper_joint_ids).clone()
                    step_extra = {"env_state": _env_state_snapshot(env, episode_progress)}
                    if teacher_forced_extra is not None:
                        step_extra.update(teacher_forced_extra)
                    pre_obs = obs
                    obs, reward, terminated, truncated, _ = env.step(env_action)
                    if handoff_tracker is not None:
                        success_status = handoff_tracker.update(env)
                        success = success_status["success"]
                    else:
                        success = _success_term(env)
                    if subtask_scheduler is not None:
                        prev_subtask_id = int(scheduler_status["subtask_id"][0].detach().cpu().item())
                        scheduler_status = subtask_scheduler.update(env)
                        next_subtask_id = int(scheduler_status["subtask_id"][0].detach().cpu().item())
                        if next_subtask_id != prev_subtask_id:
                            policy.reset()
                            previous_relative_action = None
                            _log(
                                f"[EP {episode + 1} STEP {step + 1}] subtask_change "
                                f"{SUBTASK_NAMES[prev_subtask_id]} -> {SUBTASK_NAMES[next_subtask_id]}"
                            )
                    handoff_status = _merge_handoff_status(success_status, scheduler_status)
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
                        action_handoff_status,
                        step_extra,
                    )
                    last_step = step + 1

                    if step == 0 or (step + 1) % args_cli.log_every == 0 or success.any():
                        handoff_log = ""
                        if handoff_status is not None:
                            parts = []
                            if "yellow_seen" in handoff_status:
                                parts.append(f"yellow_seen={handoff_status['yellow_seen'].detach().cpu().tolist()}")
                                parts.append(f"red_success={handoff_status['red_success'].detach().cpu().tolist()}")
                            if "subtask_id" in handoff_status:
                                parts.append(f"subtask={handoff_status['subtask_name']}")
                                parts.append(f"active_arm={handoff_status['active_arm_name']}")
                                parts.append(
                                    "right_retreat="
                                    f"{handoff_status.get('right_retreat_elapsed', 0)}/"
                                    f"{handoff_status.get('right_retreat_steps', 0)}"
                                )
                            handoff_log = " " + " ".join(parts) if parts else ""
                        _log(
                            f"[EP {episode + 1} STEP {step + 1}] "
                            f"reward={reward.detach().cpu().tolist()} "
                            f"success={success.detach().cpu().tolist()} "
                            f"terminated={terminated.detach().cpu().tolist()} "
                            f"truncated={truncated.detach().cpu().tolist()} "
                            f"cube_pos={cube_pos.detach().cpu().tolist()}"
                            f"{handoff_log}"
                        )

                    if success.any() or terminated.any() or truncated.any():
                        break

            summary = recorder.finish_episode(success, last_step, terminated, truncated, fixed_cube_xy, handoff_status)
            summaries.append(summary)
            handoff_summary = ""
            if handoff_status is not None:
                pieces = []
                if "yellow_seen" in summary:
                    pieces.append(f"yellow_seen={summary.get('yellow_seen')}")
                    pieces.append(f"red_success={summary.get('red_success')}")
                if "subtask_name" in summary:
                    pieces.append(f"subtask={summary.get('subtask_name')}")
                    pieces.append(f"active_arm={summary.get('active_arm_name')}")
                handoff_summary = " " + " ".join(pieces) if pieces else ""
            _log(
                f"[EP {episode + 1}] final_success={summary['success']} "
                f"terminated={summary['terminated']} truncated={summary['truncated']} steps={summary['steps']}"
                f"{handoff_summary}"
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
            "action_representation": action_representation,
            "policy_inference_seed": args_cli.policy_inference_seed,
            "policy_inference_seed_mode": args_cli.policy_inference_seed_mode,
            "state_dim": state_dim,
            "state_mode": state_mode,
            "handoff_time_total_steps": args_cli.handoff_time_total_steps,
            "fixed_cube_xy_list": [list(xy) for xy in fixed_cube_xy_list],
            "teacher_forced_dataset_dir": (
                str(args_cli.teacher_forced_dataset_dir) if args_cli.teacher_forced_dataset_dir is not None else None
            ),
            "teacher_forced_raw_dir": (
                str(args_cli.teacher_forced_raw_dir) if args_cli.teacher_forced_raw_dir is not None else None
            ),
            "teacher_forced_episode": args_cli.teacher_forced_episode,
            "teacher_forced_raw_episode": args_cli.teacher_forced_raw_episode,
            "teacher_forced_start_frame": args_cli.teacher_forced_start_frame,
            "teacher_forced_images_only": args_cli.teacher_forced_images_only,
            "teacher_forced_live_image_keys": sorted(_parse_image_key_set(args_cli.teacher_forced_live_image_keys)),
            "teacher_forced_episode_row": teacher_forced_episode_row,
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
        recorder.close()
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
