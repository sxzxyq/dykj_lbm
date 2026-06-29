"""Collect scripted sequential dual-arm handoff demos for the yellow-to-red task."""

import argparse
import copy
import json
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Scripted dual-arm handoff collector.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0",
    help="Isaac Lab task id.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of vectorized environments.")
parser.add_argument("--episodes", type=int, default=1, help="Number of episode attempts.")
parser.add_argument("--success-episodes", type=int, default=None, help="Stop after this many successful episodes.")
parser.add_argument("--max-attempts", type=int, default=0, help="Attempt cap for success-driven collection.")
parser.add_argument("--max-steps", type=int, default=2600, help="Maximum steps per episode.")
parser.add_argument("--seed", type=int, default=2000, help="Base seed for env resets.")
parser.add_argument("--yellow-x", type=float, default=0.50, help="Yellow handoff x in world frame.")
parser.add_argument("--yellow-y", type=float, default=0.00, help="Yellow handoff y in world frame.")
parser.add_argument("--red-x", type=float, default=0.50, help="Red target x in world frame.")
parser.add_argument("--red-y", type=float, default=0.30, help="Red target y in world frame.")
parser.add_argument("--grasp-z", type=float, default=0.015, help="TCP z target for grasping in world frame.")
parser.add_argument("--release-z", type=float, default=0.085, help="TCP z target for releasing above target.")
parser.add_argument("--hover-z", type=float, default=0.20, help="TCP z target for pre-grasp and retreat.")
parser.add_argument("--lift-z", type=float, default=0.19, help="TCP z target for transport.")
parser.add_argument("--pos-threshold", type=float, default=0.015, help="Waypoint position threshold in meters.")
parser.add_argument("--max-delta", type=float, default=0.018, help="Maximum Cartesian delta per env step.")
parser.add_argument("--arm-action-scale", type=float, default=0.5, help="IK action scale.")
parser.add_argument("--rest-steps", type=int, default=20, help="Open-gripper rest steps.")
parser.add_argument("--close-steps", type=int, default=35, help="Close-gripper hold steps.")
parser.add_argument("--open-steps", type=int, default=35, help="Release hold steps.")
parser.add_argument("--stable-steps", type=int, default=12, help="Stable area steps required for stage success.")
parser.add_argument("--phase-timeout", type=int, default=320, help="Maximum steps before forcing a phase transition.")
parser.add_argument("--cube-size-m", type=float, default=0.04, help="Cube side length in meters.")
parser.add_argument("--dataset-version", type=str, default="handoff_v1", help="Dataset/version label stored in raw metadata.")
parser.add_argument(
    "--randomization-profile",
    type=str,
    default="none",
    choices=("none", "clean", "full"),
    help="Episode-level scripted expert jitter profile. full jitters speed/timing and records V2 metadata.",
)
parser.add_argument("--speed-jitter", type=float, default=0.20, help="Relative max_delta jitter for full profile.")
parser.add_argument("--timing-jitter-steps", type=int, default=5, help="Open/close/rest jitter range for full profile.")
parser.add_argument("--log-every", type=int, default=25, help="Log one status line every N steps.")
parser.add_argument(
    "--report",
    type=str,
    default="/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/reports/scripted_handoff_report.txt",
    help="Path to write rollout report.",
)
parser.add_argument(
    "--record-dir",
    type=str,
    default="/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/raw_demos/raw_handoff_debug",
    help="Directory to write raw scripted handoff demos.",
)
parser.add_argument("--record-warmup-steps", type=int, default=2, help="Zero-action warmup steps after reset.")
parser.add_argument("--record-image-every", type=int, default=1, help="Save camera images every N recorded steps.")
parser.add_argument(
    "--camera-names",
    type=str,
    default="wrist_cam,observer_wrist_cam,global_cam",
    help="Comma-separated scene camera sensor names to refresh and record.",
)
parser.add_argument(
    "--refresh-camera-xform",
    action="store_true",
    default=True,
    help="Rewrite camera local xform ops from cfg after reset before recording.",
)
parser.add_argument(
    "--no-refresh-camera-xform",
    action="store_false",
    dest="refresh_camera_xform",
    help="Disable camera xform refresh.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.grasp_z == 0.015:
    args_cli.grasp_z = max(0.005, args_cli.cube_size_m * 0.5 - 0.005)
if args_cli.release_z == 0.085:
    args_cli.release_z = args_cli.cube_size_m * 0.5 + 0.065

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import isaac_pick_place.tasks  # noqa: F401
from isaaclab.utils import math as math_utils
from isaaclab_tasks.utils import parse_env_cfg


LEFT_ARM = "robot"
RIGHT_ARM = "observer_robot"
OPEN_ACTION = 1.0
CLOSE_ACTION = -1.0
TCP_OFFSET = (0.0, 0.0, 0.107)
AREA_SIZE_XY = (0.12, 0.12)
OBJECT_CENTER_Z = args_cli.cube_size_m * 0.5 + 0.0005
HEIGHT_TOLERANCE = 0.03
GRIPPER_OPEN_COMMAND = 0.04
GRIPPER_OPEN_THRESHOLD = 0.01
DEFAULT_SIM_DT = 0.02

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

REPORT_LINES: list[str] = []
BODY_ID_CACHE: dict[tuple[str, str], int] = {}
JOINT_ID_CACHE: dict[str, list[int]] = {}


def _log(message: str):
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


def _camera_names() -> list[str]:
    return [name.strip() for name in args_cli.camera_names.split(",") if name.strip()]


def _camera_term_name(camera_name: str) -> str:
    if camera_name == "wrist_cam":
        return "wrist_rgb"
    if camera_name == "observer_wrist_cam":
        return "observer_wrist_rgb"
    if camera_name == "global_cam":
        return "global_rgb"
    return camera_name


def _tensor_row(tensor: torch.Tensor, env_id: int = 0):
    value = tensor[env_id].detach().cpu()
    if value.ndim == 0:
        return value.item()
    return value.tolist()


def _policy_obs(obs):
    if isinstance(obs, dict) and "policy" in obs:
        return obs["policy"]
    return {}


def _obs_image(policy_obs: dict, term_name: str):
    value = policy_obs.get(term_name)
    if value is None:
        return None
    return value[0]


def _save_rgb_image(image: torch.Tensor, image_path: Path):
    from PIL import Image

    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = image.detach().cpu()
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != torch.uint8:
        image = image.clamp(0, 255).to(torch.uint8)
    Image.fromarray(image.numpy()).save(image_path)


def _refresh_camera_xforms(env):
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


def _gripper_joint_ids(env, arm_name: str) -> list[int]:
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


def _gripper_is_open(env, arm_name: str) -> torch.Tensor:
    robot = _asset(env, arm_name)
    joint_ids = _gripper_joint_ids(env, arm_name)
    finger_pos = torch.abs(robot.data.joint_pos[:, joint_ids])
    return torch.all(finger_pos >= GRIPPER_OPEN_COMMAND - GRIPPER_OPEN_THRESHOLD, dim=1)


def _gripper_opening(env, arm_name: str) -> torch.Tensor:
    robot = _asset(env, arm_name)
    joint_ids = _gripper_joint_ids(env, arm_name)
    return torch.sum(torch.abs(robot.data.joint_pos[:, joint_ids]), dim=1)


def _area_center(args, name: str, device: str) -> torch.Tensor:
    if name == "yellow":
        xy = (args.yellow_x, args.yellow_y)
    elif name == "red":
        xy = (args.red_x, args.red_y)
    else:
        raise ValueError(f"Unknown area: {name}")
    center = torch.zeros((1, 3), device=device)
    center[:, 0] = xy[0]
    center[:, 1] = xy[1]
    center[:, 2] = OBJECT_CENTER_Z
    return center


def _object_on_area(env, center_w: torch.Tensor, gripper_arm: str) -> torch.Tensor:
    cube_pos = _cube_pos_w(env)
    xy_error = torch.abs(cube_pos[:, :2] - center_w[:, :2])
    inside = torch.logical_and(xy_error[:, 0] <= AREA_SIZE_XY[0] * 0.5, xy_error[:, 1] <= AREA_SIZE_XY[1] * 0.5)
    low = torch.abs(cube_pos[:, 2] - center_w[:, 2]) <= HEIGHT_TOLERANCE
    released = _gripper_is_open(env, gripper_arm)
    return inside & low & released


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


def _park_positions(env, args) -> dict[str, torch.Tensor]:
    parks = {
        LEFT_ARM: _tcp_pos_w(env, LEFT_ARM).clone(),
        RIGHT_ARM: _tcp_pos_w(env, RIGHT_ARM).clone(),
    }
    for value in parks.values():
        value[:, 2] = torch.maximum(value[:, 2], torch.full_like(value[:, 2], args.hover_z))
    return parks


def _desired_pos_for_phase(
    env,
    phase_name: str,
    args,
    park_positions: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    active_arm = _active_arm_for_phase(phase_name)
    if active_arm is None:
        return None

    desired = _tcp_pos_w(env, active_arm).clone()
    cube_pos = _cube_pos_w(env)
    area_name = _target_area_for_phase(phase_name)
    target = _area_center(args, area_name, env.unwrapped.device) if area_name else None

    if "move_above_cube" in phase_name:
        desired[:, :2] = cube_pos[:, :2]
        desired[:, 2] = args.hover_z
    elif "descend_to_grasp" in phase_name or "close_gripper" in phase_name:
        desired[:, :2] = cube_pos[:, :2]
        desired[:, 2] = args.grasp_z
    elif "lift_cube" in phase_name:
        desired[:, :2] = cube_pos[:, :2]
        desired[:, 2] = args.lift_z
    elif "move_above_yellow" in phase_name or "move_above_red" in phase_name:
        desired[:, :2] = target[:, :2]
        desired[:, 2] = args.lift_z
    elif "descend_to_yellow" in phase_name or "descend_to_red" in phase_name:
        desired[:, :2] = target[:, :2]
        desired[:, 2] = args.release_z
    elif "release_on_yellow" in phase_name or "release_on_red" in phase_name:
        desired[:, :2] = target[:, :2]
        desired[:, 2] = args.release_z
    elif "retreat" in phase_name:
        desired = park_positions[active_arm].clone()

    return desired


def _arm_delta_action(env, arm_name: str, desired_pos_w: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor]:
    robot = _asset(env, arm_name)
    tcp_pos = _tcp_pos_w(env, arm_name)
    delta_w = desired_pos_w - tcp_pos
    distance = torch.linalg.vector_norm(delta_w, dim=1)
    scale = torch.clamp(args.max_delta / (distance + 1.0e-8), max=1.0).unsqueeze(-1)
    clipped_delta_w = delta_w * scale
    clipped_delta_b = math_utils.quat_apply_inverse(robot.data.root_quat_w, clipped_delta_w)

    arm_action = torch.zeros((env.unwrapped.num_envs, 6), device=env.unwrapped.device)
    arm_action[:, :3] = clipped_delta_b / args.arm_action_scale
    return arm_action, distance


def _compute_action(
    env,
    phase_name: str,
    args,
    park_positions: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, str | None]:
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    active_arm = _active_arm_for_phase(phase_name)
    distance = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    desired_pos = _desired_pos_for_phase(env, phase_name, args, park_positions)
    gripper = _gripper_for_phase(phase_name)

    if active_arm is None or desired_pos is None:
        return actions, distance, desired_pos, active_arm

    arm_action, distance = _arm_delta_action(env, active_arm, desired_pos, args)
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
    args,
) -> tuple[int, int]:
    phase_name = PHASES[phase_idx]
    advance = False
    if phase_name.endswith("open_rest"):
        advance = phase_steps >= args.rest_steps
    elif "close_gripper" in phase_name:
        advance = phase_steps >= args.close_steps
    elif "release_on" in phase_name:
        advance = phase_steps >= args.open_steps
    elif "retreat" in phase_name:
        advance = reached
    elif phase_name == "wait_yellow_stable":
        advance = yellow_stable_steps >= args.stable_steps or phase_steps >= args.phase_timeout
    elif phase_name == "wait_red_stable":
        advance = red_stable_steps >= args.stable_steps or phase_steps >= args.phase_timeout
    elif phase_name == "done":
        advance = False
    else:
        advance = reached or phase_steps >= args.phase_timeout

    if advance:
        return min(phase_idx + 1, len(PHASES) - 1), 0
    return phase_idx, phase_steps + 1


def _split_action(action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return action[:, 0:7], action[:, 7:14]


def _arm_state(env, arm_name: str) -> dict:
    robot = _asset(env, arm_name)
    return {
        "joint_pos": _tensor_row(robot.data.joint_pos),
        "joint_vel": _tensor_row(robot.data.joint_vel),
        "tcp_pos_w": _tensor_row(_tcp_pos_w(env, arm_name)),
        "tcp_quat_w": _tensor_row(_tcp_quat_w(env, arm_name)),
        "gripper_opening": _tensor_row(_gripper_opening(env, arm_name)),
        "gripper_open": bool(_gripper_is_open(env, arm_name)[0].detach().cpu().item()),
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


def _scene_snapshot(env, args, step: int) -> dict:
    return {
        "timestamp": _sim_time(env, step),
        "cube_pos_w": _tensor_row(_cube_pos_w(env)),
        "yellow_area_pos_w": [args.yellow_x, args.yellow_y, OBJECT_CENTER_Z],
        "red_area_pos_w": [args.red_x, args.red_y, OBJECT_CENTER_Z],
        "arms": {
            "left": _arm_state(env, LEFT_ARM),
            "right": _arm_state(env, RIGHT_ARM),
        },
    }


def _episode_args(base_args, episode: int):
    episode_args = copy.copy(base_args)
    seed = int(base_args.seed) + int(episode) * 1009
    rng = random.Random(seed)
    randomization = {
        "profile": base_args.randomization_profile,
        "seed": seed,
        "max_delta_scale": 1.0,
        "rest_steps_delta": 0,
        "close_steps_delta": 0,
        "open_steps_delta": 0,
    }
    if base_args.randomization_profile == "full":
        max_delta_scale = rng.uniform(1.0 - base_args.speed_jitter, 1.0 + base_args.speed_jitter)
        episode_args.max_delta = max(0.001, base_args.max_delta * max_delta_scale)
        jitter = max(0, int(base_args.timing_jitter_steps))
        rest_delta = rng.randint(-jitter, jitter)
        close_delta = rng.randint(-jitter, jitter)
        open_delta = rng.randint(-jitter, jitter)
        episode_args.rest_steps = max(1, base_args.rest_steps + rest_delta)
        episode_args.close_steps = max(1, base_args.close_steps + close_delta)
        episode_args.open_steps = max(1, base_args.open_steps + open_delta)
        randomization.update(
            {
                "max_delta_scale": max_delta_scale,
                "rest_steps_delta": rest_delta,
                "close_steps_delta": close_delta,
                "open_steps_delta": open_delta,
            }
        )
    episode_args.randomization = randomization
    return episode_args


class RawHandoffRecorder:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.episode_dir: Path | None = None
        self.steps_file = None
        self.step_count = 0
        self.image_counts: dict[str, int] = {}

    def start_episode(self, episode: int, env, args):
        self.episode_dir = self.root_dir / f"episode_{episode:06d}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        (self.episode_dir / "quality_check").mkdir(parents=True, exist_ok=True)
        for name in _camera_names():
            (self.episode_dir / _camera_term_name(name)).mkdir(parents=True, exist_ok=True)
        self.steps_file = (self.episode_dir / "steps.jsonl").open("w", encoding="utf-8")
        self.step_count = 0
        self.image_counts = {name: 0 for name in _camera_names()}
        meta = {
            "episode": episode,
            "dataset_version": args.dataset_version,
            "task": args.task,
            "seed": args.seed + episode,
            "num_envs": env.unwrapped.num_envs,
            "cube_size_m": args.cube_size_m,
            "object_center_z": OBJECT_CENTER_Z,
            "state_timing": "exact_pre_action",
            "image_timing": "pre_action",
            "action_timing": "action_from_pre_to_post",
            "randomization": getattr(args, "randomization", {"profile": args.randomization_profile}),
            "camera_names": _camera_names(),
            "camera_refresh_enabled": args.refresh_camera_xform,
            "record_image_every": args.record_image_every,
            "action_space_shape": tuple(env.action_space.shape),
            "action_layout": {
                "left_action": [0, 7],
                "left_arm_delta_pose": [0, 6],
                "left_gripper": 6,
                "right_action": [7, 14],
                "right_arm_delta_pose": [7, 13],
                "right_gripper": 13,
            },
            "phases": list(PHASES),
            "areas": {
                "yellow_area_pos_w": [args.yellow_x, args.yellow_y, OBJECT_CENTER_Z],
                "red_area_pos_w": [args.red_x, args.red_y, OBJECT_CENTER_Z],
                "area_size_xy": list(AREA_SIZE_XY),
            },
        }
        (self.episode_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def save_quality_check(self, obs):
        if self.episode_dir is None:
            return
        policy_obs = _policy_obs(obs)
        for name in _camera_names():
            term_name = _camera_term_name(name)
            image = _obs_image(policy_obs, term_name)
            if image is None:
                _log(f"[RECORD] missing quality-check image term {term_name}")
                continue
            _save_rgb_image(image, self.episode_dir / "quality_check" / f"{term_name}_reset.png")

    def record_step(
        self,
        obs,
        step: int,
        phase_name: str,
        active_arm: str | None,
        actions: torch.Tensor,
        desired_pos_w: torch.Tensor | None,
        distance: torch.Tensor,
        pre_snapshot: dict,
        post_snapshot: dict,
        action_timestamp: float,
        env,
        reward: torch.Tensor,
        success_yellow: torch.Tensor,
        success_red: torch.Tensor,
        yellow_stable_steps: int,
        red_stable_steps: int,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        args,
    ):
        if self.episode_dir is None or self.steps_file is None:
            return

        policy_obs = _policy_obs(obs)
        image_paths = {}
        if args.record_image_every > 0 and self.step_count % args.record_image_every == 0:
            for name in _camera_names():
                term_name = _camera_term_name(name)
                image = _obs_image(policy_obs, term_name)
                if image is None:
                    continue
                image_index = self.image_counts[name]
                image_path = self.episode_dir / term_name / f"{image_index:06d}.png"
                _save_rgb_image(image, image_path)
                image_paths[term_name] = str(image_path.relative_to(self.episode_dir))
                self.image_counts[name] += 1

        left_action, right_action = _split_action(actions)
        row = {
            "step": step,
            "record_step": self.step_count,
            "step_index": self.step_count,
            "sim_time": pre_snapshot["timestamp"],
            "pre_state_timestamp": pre_snapshot["timestamp"],
            "action_timestamp": action_timestamp,
            "post_state_timestamp": post_snapshot["timestamp"],
            "stage": _stage_for_phase(phase_name),
            "phase": phase_name,
            "active_arm": active_arm,
            "action": _tensor_row(actions),
            "left_action": _tensor_row(left_action),
            "right_action": _tensor_row(right_action),
            "desired_ee_pos_w": _tensor_row(desired_pos_w) if desired_pos_w is not None else None,
            "distance_to_waypoint": _tensor_row(distance),
            "pre_cube": {"pos_w": pre_snapshot["cube_pos_w"]},
            "post_cube": {"pos_w": post_snapshot["cube_pos_w"]},
            "pre_targets": {
                "yellow_area_pos_w": pre_snapshot["yellow_area_pos_w"],
                "red_area_pos_w": pre_snapshot["red_area_pos_w"],
            },
            "post_targets": {
                "yellow_area_pos_w": post_snapshot["yellow_area_pos_w"],
                "red_area_pos_w": post_snapshot["red_area_pos_w"],
            },
            "pre_arms": pre_snapshot["arms"],
            "post_arms": post_snapshot["arms"],
            "cube_pos_w": post_snapshot["cube_pos_w"],
            "yellow_area_pos_w": post_snapshot["yellow_area_pos_w"],
            "red_area_pos_w": post_snapshot["red_area_pos_w"],
            "success_yellow": bool(success_yellow[0].detach().cpu().item()),
            "success_red": bool(success_red[0].detach().cpu().item()),
            "yellow_stable_steps": yellow_stable_steps,
            "red_stable_steps": red_stable_steps,
            "reward": _tensor_row(reward),
            "terminated": bool(terminated[0].detach().cpu().item()),
            "truncated": bool(truncated[0].detach().cpu().item()),
            "arms": {
                "left": post_snapshot["arms"]["left"],
                "right": post_snapshot["arms"]["right"],
            },
            "pre_images": image_paths,
            "images": image_paths,
        }
        self.steps_file.write(json.dumps(row) + "\n")
        self.step_count += 1

    def finish_episode(
        self,
        success: bool,
        red_stage_success: bool,
        steps: int,
        final_phase: str,
        yellow_stable_steps: int,
        red_stable_steps: int,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ):
        if self.episode_dir is not None:
            summary = {
                "success": success,
                "red_stage_success": red_stage_success,
                "steps": steps,
                "final_phase": final_phase,
                "yellow_stable_steps": yellow_stable_steps,
                "red_stable_steps": red_stable_steps,
                "terminated": bool(terminated[0].detach().cpu().item()),
                "truncated": bool(truncated[0].detach().cpu().item()),
                "recorded_steps": self.step_count,
                "image_counts": self.image_counts,
            }
            (self.episode_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if self.steps_file is not None:
            self.steps_file.close()
            self.steps_file = None


def _write_report():
    report_path = Path(args_cli.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(REPORT_LINES) + "\n", encoding="utf-8")


def main():
    if args_cli.num_envs != 1:
        raise ValueError("scripted_handoff_collect.py currently supports --num_envs 1 only.")
    if args_cli.success_episodes is not None and args_cli.success_episodes <= 0:
        raise ValueError("--success-episodes must be positive when set.")
    if args_cli.max_attempts < 0:
        raise ValueError("--max-attempts must be non-negative.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device
    recorder = RawHandoffRecorder(Path(args_cli.record_dir))

    _log(f"[INFO] task={args_cli.task}")
    _log(f"[INFO] device={device}")
    _log(f"[INFO] action_space={env.action_space}")
    _log(f"[INFO] action_terms={env.unwrapped.action_manager.active_terms}")
    _log(f"[INFO] action_term_dim={env.unwrapped.action_manager.action_term_dim}")
    _log(f"[INFO] record_dir={args_cli.record_dir}")
    if env.action_space.shape[-1] != 14:
        raise RuntimeError(f"Expected 14-dim handoff action space, got {env.action_space.shape}")

    attempt_limit = args_cli.episodes
    if args_cli.success_episodes is not None:
        attempt_limit = args_cli.max_attempts if args_cli.max_attempts > 0 else None
        _log(
            "[INFO] success-driven collection="
            f"target_success_episodes={args_cli.success_episodes}, "
            f"max_attempts={'unlimited' if attempt_limit is None else attempt_limit}"
        )

    yellow_center = _area_center(args_cli, "yellow", device)
    red_center = _area_center(args_cli, "red", device)
    total_successes = 0
    attempted_episodes = 0
    failure_phases: dict[str, int] = {}

    try:
        episode = 0
        while simulation_app.is_running():
            if args_cli.success_episodes is not None and total_successes >= args_cli.success_episodes:
                break
            if attempt_limit is not None and attempted_episodes >= attempt_limit:
                break

            reset_out = env.reset(seed=args_cli.seed + episode)
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            episode_args = _episode_args(args_cli, episode)
            BODY_ID_CACHE.clear()
            JOINT_ID_CACHE.clear()
            _log(f"[EP {episode + 1}] reset obs_keys={list(obs.keys()) if isinstance(obs, dict) else type(obs)}")

            if args_cli.refresh_camera_xform:
                _refresh_camera_xforms(env)
            recorder.start_episode(episode, env, episode_args)

            if args_cli.record_warmup_steps > 0:
                warmup_actions = torch.zeros(env.action_space.shape, device=device)
                for _ in range(args_cli.record_warmup_steps):
                    obs, _, warmup_terminated, warmup_truncated, _ = env.step(warmup_actions)
                    if warmup_terminated.any() or warmup_truncated.any():
                        break
            park_positions = _park_positions(env, episode_args)
            _log(
                f"[EP {episode + 1}] park_left={park_positions[LEFT_ARM].detach().cpu().tolist()} "
                f"park_right={park_positions[RIGHT_ARM].detach().cpu().tolist()}"
            )
            _log(f"[EP {episode + 1}] randomization={getattr(episode_args, 'randomization', {})}")
            recorder.save_quality_check(obs)

            phase_idx = 0
            phase_steps = 0
            yellow_stable_steps = 0
            red_stable_steps = 0
            terminated = torch.zeros(env.unwrapped.num_envs, device=device, dtype=torch.bool)
            truncated = torch.zeros_like(terminated)
            reward = torch.zeros(env.unwrapped.num_envs, device=device)
            episode_success = False
            red_stage_success = False
            last_step = 0
            last_executed_phase = PHASES[phase_idx]

            for step in range(args_cli.max_steps):
                if not simulation_app.is_running():
                    break

                phase_name = PHASES[phase_idx]
                last_executed_phase = phase_name
                pre_obs = obs
                with torch.no_grad():
                    pre_snapshot = _scene_snapshot(env, episode_args, step)
                    actions, distance, desired_pos_w, active_arm = _compute_action(
                        env, phase_name, episode_args, park_positions
                    )
                    action_timestamp = _sim_time(env, step)
                    obs, reward, terminated, truncated, _ = env.step(actions)
                    post_snapshot = _scene_snapshot(env, episode_args, step + 1)

                    success_yellow = _object_on_area(env, yellow_center, RIGHT_ARM)
                    success_red = _object_on_area(env, red_center, LEFT_ARM)
                    yellow_now = bool(success_yellow[0].detach().cpu().item())
                    red_now = bool(success_red[0].detach().cpu().item())
                    yellow_stable_steps = yellow_stable_steps + 1 if yellow_now else 0
                    red_stable_steps = red_stable_steps + 1 if red_now else 0
                    red_stage_success = red_stage_success or red_stable_steps >= args_cli.stable_steps

                    reached = bool((distance < args_cli.pos_threshold).all().detach().cpu().item())
                    recorder.record_step(
                        pre_obs,
                        step,
                        phase_name,
                        active_arm,
                        actions,
                        desired_pos_w,
                        distance,
                        pre_snapshot,
                        post_snapshot,
                        action_timestamp,
                        env,
                        reward,
                        success_yellow,
                        success_red,
                        yellow_stable_steps,
                        red_stable_steps,
                        terminated,
                        truncated,
                        episode_args,
                    )
                    phase_idx, phase_steps = _advance_phase(
                        phase_idx,
                        phase_steps,
                        reached,
                        yellow_stable_steps,
                        red_stable_steps,
                        episode_args,
                    )
                    episode_success = (
                        phase_name == "left_retreat"
                        and reached
                        and red_stable_steps >= args_cli.stable_steps
                    )

                last_step = step + 1
                if step == 0 or (step + 1) % args_cli.log_every == 0 or red_stage_success or episode_success:
                    _log(
                        f"[EP {episode + 1} STEP {step + 1}] "
                        f"phase={phase_name} active_arm={active_arm} "
                        f"dist={distance.detach().cpu().tolist()} "
                        f"cube={_cube_pos_w(env).detach().cpu().tolist()} "
                        f"yellow_stable={yellow_stable_steps} red_stable={red_stable_steps} "
                        f"red_stage_success={red_stage_success} success={episode_success} "
                        f"terminated={terminated.detach().cpu().tolist()} "
                        f"truncated={truncated.detach().cpu().tolist()}"
                    )

                if episode_success or terminated.any() or truncated.any() or PHASES[phase_idx] == "done":
                    break

            final_phase = last_executed_phase
            if episode_success:
                total_successes += 1
            else:
                failure_phases[final_phase] = failure_phases.get(final_phase, 0) + 1
            recorder.finish_episode(
                episode_success,
                red_stage_success,
                last_step,
                final_phase,
                yellow_stable_steps,
                red_stable_steps,
                terminated,
                truncated,
            )
            _log(
                f"[EP {episode + 1}] success={episode_success} final_phase={final_phase} "
                f"yellow_stable={yellow_stable_steps} red_stable={red_stable_steps} steps={last_step}"
            )
            attempted_episodes += 1
            episode += 1

        expected = args_cli.success_episodes if args_cli.success_episodes is not None else attempted_episodes
        _log(f"[SUMMARY] successes={total_successes}/{expected} attempts={attempted_episodes}")
        if failure_phases:
            _log(f"[SUMMARY] failure_phases={failure_phases}")
        _log("[OK] Scripted handoff rollout completed.")
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
        _write_report()


if __name__ == "__main__":
    main()
    simulation_app.close()
