"""Run a waypoint-based scripted expert for the red-target cube pick-and-place task."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Scripted pick-place expert for the red-target cube task.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0",
    help="Isaac Lab task id.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of vectorized environments.")
parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
parser.add_argument(
    "--success-episodes",
    type=int,
    default=None,
    help="Stop after this many successful episodes. If unset, run exactly --episodes attempts.",
)
parser.add_argument(
    "--max-attempts",
    type=int,
    default=0,
    help="Maximum episode attempts when --success-episodes is set. Use 0 for no explicit cap.",
)
parser.add_argument("--max-steps", type=int, default=1200, help="Maximum steps per episode.")
parser.add_argument("--seed", type=int, default=42, help="Base seed for env resets.")
parser.add_argument("--target-x", type=float, default=0.50, help="Red target x position in robot root frame.")
parser.add_argument("--target-y", type=float, default=0.00, help="Red target y position in robot root frame.")
parser.add_argument("--grasp-z", type=float, default=0.015, help="TCP z target for grasping in robot root frame.")
parser.add_argument(
    "--release-z",
    type=float,
    default=0.085,
    help="TCP z target where the gripper opens above the red target, letting the cube fall freely.",
)
parser.add_argument(
    "--place-z",
    type=float,
    default=None,
    help="Deprecated alias for --release-z, kept for older commands.",
)
parser.add_argument("--hover-z", type=float, default=0.20, help="TCP z target for pre-grasp and retreat.")
parser.add_argument("--lift-z", type=float, default=0.19, help="TCP z target for transport.")
parser.add_argument("--pos-threshold", type=float, default=0.015, help="Waypoint position threshold in meters.")
parser.add_argument("--max-delta", type=float, default=0.018, help="Maximum processed Cartesian delta per env step.")
parser.add_argument(
    "--arm-action-scale",
    type=float,
    default=0.5,
    help="Scale used by the arm IK action. Raw actions are divided by this value.",
)
parser.add_argument("--rest-steps", type=int, default=20, help="Steps to hold the initial open-gripper phase.")
parser.add_argument("--close-steps", type=int, default=35, help="Steps to hold the close-gripper phase.")
parser.add_argument("--open-steps", type=int, default=35, help="Steps to hold the release phase.")
parser.add_argument("--phase-timeout", type=int, default=260, help="Maximum steps before forcing a phase transition.")
parser.add_argument("--log-every", type=int, default=25, help="Log one status line every N steps.")
parser.add_argument(
    "--report",
    type=str,
    default="/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/scripted_pick_place_report.txt",
    help="Path to write the scripted rollout report.",
)
parser.add_argument(
    "--record-dir",
    type=str,
    default=None,
    help="Directory to write raw scripted demo episodes. Recording currently supports num_envs=1.",
)
parser.add_argument(
    "--record-warmup-steps",
    type=int,
    default=2,
    help="Ignored zero-arm steps after reset/camera refresh before writing demo data.",
)
parser.add_argument(
    "--record-image-every",
    type=int,
    default=1,
    help="Save camera images every N recorded steps.",
)
parser.add_argument(
    "--camera-names",
    type=str,
    default="wrist_cam,observer_wrist_cam",
    help="Comma-separated scene camera sensor names to refresh and quality-check.",
)
parser.add_argument(
    "--refresh-camera-xform",
    action="store_true",
    default=False,
    help="Rewrite camera local xform ops from cfg after reset before recording.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import isaac_pick_place.tasks  # noqa: F401
from isaaclab.utils import math as math_utils
from isaaclab_tasks.utils import parse_env_cfg


PHASES = (
    "open_gripper_rest",
    "move_above_cube",
    "descend_to_grasp",
    "close_gripper",
    "lift_cube",
    "move_above_red_target",
    "descend_to_release",
    "release_gripper",
    "retreat",
    "done",
)
# Isaac Lab BinaryJointAction maps negative scalar actions to close_command and non-negative actions to open_command.
OPEN_ACTION = 1.0
CLOSE_ACTION = -1.0
REPORT_LINES: list[str] = []


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


def _target_center_w(env, target_xy: torch.Tensor, z: float) -> torch.Tensor:
    robot = env.unwrapped.scene["robot"]
    target = torch.zeros((env.unwrapped.num_envs, 3), device=env.unwrapped.device)
    target[:, :2] = robot.data.root_pos_w[:, :2] + target_xy
    target[:, 2] = robot.data.root_pos_w[:, 2] + z
    return target


def _ee_pos_w(env) -> torch.Tensor:
    ee_frame = env.unwrapped.scene["ee_frame"]
    return ee_frame.data.target_pos_w[..., 0, :]


def _cube_pos_w(env) -> torch.Tensor:
    return env.unwrapped.scene["object"].data.root_pos_w[:, :3]


def _gripper_opening(env, gripper_joint_ids) -> torch.Tensor:
    robot = env.unwrapped.scene["robot"]
    return torch.sum(torch.abs(robot.data.joint_pos[:, gripper_joint_ids]), dim=1)


def _make_waypoint(env, phase: torch.Tensor, target_xy: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor]:
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    cube_pos = _cube_pos_w(env)
    ee_pos = _ee_pos_w(env)
    target_pos = _target_center_w(env, target_xy, args.release_z)
    desired = ee_pos.clone()
    gripper = torch.full((num_envs,), OPEN_ACTION, device=device)

    move_above_cube = phase == 1
    desired[move_above_cube, 0:2] = cube_pos[move_above_cube, 0:2]
    desired[move_above_cube, 2] = args.hover_z

    descend_to_grasp = phase == 2
    desired[descend_to_grasp, 0:2] = cube_pos[descend_to_grasp, 0:2]
    desired[descend_to_grasp, 2] = args.grasp_z

    close_gripper = phase == 3
    desired[close_gripper, 0:2] = cube_pos[close_gripper, 0:2]
    desired[close_gripper, 2] = args.grasp_z
    gripper[close_gripper] = CLOSE_ACTION

    lift_cube = phase == 4
    desired[lift_cube, 0:2] = cube_pos[lift_cube, 0:2]
    desired[lift_cube, 2] = args.lift_z
    gripper[lift_cube] = CLOSE_ACTION

    move_target = phase == 5
    desired[move_target, 0:2] = target_pos[move_target, 0:2]
    desired[move_target, 2] = args.lift_z
    gripper[move_target] = CLOSE_ACTION

    descend_to_release = phase == 6
    desired[descend_to_release, 0:2] = target_pos[descend_to_release, 0:2]
    desired[descend_to_release, 2] = args.release_z
    gripper[descend_to_release] = CLOSE_ACTION

    release_gripper = phase == 7
    desired[release_gripper, 0:2] = target_pos[release_gripper, 0:2]
    desired[release_gripper, 2] = args.release_z

    retreat = phase == 8
    desired[retreat, 0:2] = target_pos[retreat, 0:2]
    desired[retreat, 2] = args.hover_z

    return desired, gripper


def _advance_phase(phase: torch.Tensor, phase_steps: torch.Tensor, reached: torch.Tensor, args) -> torch.Tensor:
    advance = torch.zeros_like(reached)
    advance |= (phase == 0) & (phase_steps >= args.rest_steps)
    advance |= (phase == 1) & (reached | (phase_steps >= args.phase_timeout))
    advance |= (phase == 2) & (reached | (phase_steps >= args.phase_timeout))
    advance |= (phase == 3) & (phase_steps >= args.close_steps)
    advance |= (phase == 4) & (reached | (phase_steps >= args.phase_timeout))
    advance |= (phase == 5) & (reached | (phase_steps >= args.phase_timeout))
    advance |= (phase == 6) & (reached | (phase_steps >= args.phase_timeout))
    advance |= (phase == 7) & (phase_steps >= args.open_steps)
    advance |= (phase == 8) & (reached | (phase_steps >= args.phase_timeout))
    new_phase = torch.where(advance, torch.clamp(phase + 1, max=len(PHASES) - 1), phase)
    phase_steps[:] = torch.where(advance, torch.zeros_like(phase_steps), phase_steps + 1)
    return new_phase


def _compute_action(env, desired_pos_w: torch.Tensor, gripper: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.unwrapped.scene["robot"]
    ee_pos_w = _ee_pos_w(env)
    delta_w = desired_pos_w - ee_pos_w
    distance = torch.linalg.vector_norm(delta_w, dim=1)
    scale = torch.clamp(args.max_delta / (distance + 1.0e-8), max=1.0).unsqueeze(-1)
    clipped_delta_w = delta_w * scale
    clipped_delta_b = math_utils.quat_apply_inverse(robot.data.root_quat_w, clipped_delta_w)

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    actions[:, :3] = clipped_delta_b / args.arm_action_scale
    actions[:, 6] = gripper
    return actions, distance


def _success_term(env) -> torch.Tensor:
    try:
        return env.unwrapped.termination_manager.get_term("success").clone()
    except Exception:
        return torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device, dtype=torch.bool)


def _camera_names() -> list[str]:
    return [name.strip() for name in args_cli.camera_names.split(",") if name.strip()]


def _tensor_row(tensor: torch.Tensor, env_id: int = 0):
    value = tensor[env_id].detach().cpu()
    if value.ndim == 0:
        return value.item()
    return value.tolist()


def _phase_name(phase: torch.Tensor, env_id: int = 0) -> str:
    return PHASES[int(phase[env_id].detach().cpu().item())]


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


def _save_rgb_image(image: torch.Tensor, image_path: Path):
    from PIL import Image

    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = image.detach().cpu()
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != torch.uint8:
        image = image.clamp(0, 255).to(torch.uint8)
    Image.fromarray(image.numpy()).save(image_path)


def _policy_obs(obs):
    if isinstance(obs, dict) and "policy" in obs:
        return obs["policy"]
    return {}


def _obs_image(policy_obs: dict, term_name: str):
    value = policy_obs.get(term_name)
    if value is None:
        return None
    return value[0]


def _obs_lowdim(policy_obs: dict, term_name: str):
    value = policy_obs.get(term_name)
    if value is None:
        return None
    return _tensor_row(value)


def _camera_term_name(camera_name: str) -> str:
    if camera_name == "wrist_cam":
        return "wrist_rgb"
    if camera_name == "observer_wrist_cam":
        return "observer_wrist_rgb"
    return camera_name


class RawDemoRecorder:
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
            "task": args.task,
            "seed": args.seed + episode,
            "num_envs": env.unwrapped.num_envs,
            "camera_names": _camera_names(),
            "camera_refresh_enabled": args.refresh_camera_xform,
            "record_warmup_steps": args.record_warmup_steps,
            "record_image_every": args.record_image_every,
            "action_space_shape": tuple(env.action_space.shape),
            "phases": list(PHASES),
            "waypoints": {
                "target_xy": [args.target_x, args.target_y],
                "grasp_z": args.grasp_z,
                "release_z": args.release_z,
                "hover_z": args.hover_z,
                "lift_z": args.lift_z,
                "max_delta": args.max_delta,
                "arm_action_scale": args.arm_action_scale,
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
        phase: torch.Tensor,
        actions: torch.Tensor,
        desired_pos_w: torch.Tensor,
        distance: torch.Tensor,
        pre_cube_pos: torch.Tensor,
        pre_cube_target_xy_error: torch.Tensor,
        pre_opening: torch.Tensor,
        reward: torch.Tensor,
        success: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ):
        if self.episode_dir is None or self.steps_file is None:
            return
        policy_obs = _policy_obs(obs)
        image_paths = {}
        if args_cli.record_image_every > 0 and self.step_count % args_cli.record_image_every == 0:
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

        row = {
            "step": step,
            "record_step": self.step_count,
            "phase": _phase_name(phase),
            "action": _tensor_row(actions),
            "desired_ee_pos_w": _tensor_row(desired_pos_w),
            "distance_to_waypoint": _tensor_row(distance),
            "cube_pos_w": _tensor_row(pre_cube_pos),
            "cube_target_xy_error": _tensor_row(pre_cube_target_xy_error),
            "gripper_opening": _tensor_row(pre_opening),
            "reward": _tensor_row(reward),
            "success": bool(success[0].detach().cpu().item()),
            "terminated": bool(terminated[0].detach().cpu().item()),
            "truncated": bool(truncated[0].detach().cpu().item()),
            "obs": {
                "joint_pos": _obs_lowdim(policy_obs, "joint_pos"),
                "joint_vel": _obs_lowdim(policy_obs, "joint_vel"),
                "ee_position": _obs_lowdim(policy_obs, "ee_position"),
                "ee_quat": _obs_lowdim(policy_obs, "ee_quat"),
                "object_position": _obs_lowdim(policy_obs, "object_position"),
                "target_area_position": _obs_lowdim(policy_obs, "target_area_position"),
                "actions": _obs_lowdim(policy_obs, "actions"),
            },
            "images": image_paths,
        }
        self.steps_file.write(json.dumps(row) + "\n")
        self.step_count += 1

    def finish_episode(self, success: torch.Tensor, steps: int, terminated: torch.Tensor, truncated: torch.Tensor):
        if self.episode_dir is not None:
            summary = {
                "success": bool(success[0].detach().cpu().item()),
                "steps": steps,
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
    if args_cli.place_z is not None:
        args_cli.release_z = args_cli.place_z

    if args_cli.record_dir is not None and args_cli.num_envs != 1:
        raise ValueError("Raw demo recording currently supports --num_envs 1 only.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device
    target_xy = torch.tensor((args_cli.target_x, args_cli.target_y), device=device)
    gripper_joint_ids, _ = env.unwrapped.scene["robot"].find_joints(["panda_finger.*"])

    _log(f"[INFO] task={args_cli.task}")
    _log(f"[INFO] device={device}")
    _log(f"[INFO] num_envs={env.unwrapped.num_envs}")
    _log(f"[INFO] action_space={env.action_space}")
    _log(f"[INFO] phases={list(PHASES)}")
    _log(
        "[INFO] waypoints="
        f"grasp_z={args_cli.grasp_z}, release_z={args_cli.release_z}, "
        f"hover_z={args_cli.hover_z}, lift_z={args_cli.lift_z}, target_xy=({args_cli.target_x}, {args_cli.target_y})"
    )
    recorder = RawDemoRecorder(Path(args_cli.record_dir)) if args_cli.record_dir is not None else None
    if recorder is not None:
        _log(f"[INFO] recording raw demos to {args_cli.record_dir}")

    if args_cli.success_episodes is not None and args_cli.success_episodes <= 0:
        raise ValueError("--success-episodes must be positive when set.")
    if args_cli.max_attempts < 0:
        raise ValueError("--max-attempts must be non-negative.")
    if args_cli.success_episodes is not None:
        attempt_limit = args_cli.max_attempts if args_cli.max_attempts > 0 else None
        _log(
            "[INFO] success-driven collection="
            f"target_success_episodes={args_cli.success_episodes}, "
            f"max_attempts={'unlimited' if attempt_limit is None else attempt_limit}"
        )
    else:
        attempt_limit = args_cli.episodes

    total_successes = 0
    attempted_episodes = 0
    try:
        episode = 0
        while True:
            if not simulation_app.is_running():
                break
            if args_cli.success_episodes is not None and total_successes >= args_cli.success_episodes:
                break
            if attempt_limit is not None and attempted_episodes >= attempt_limit:
                break
            reset_out = env.reset(seed=args_cli.seed + episode)
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            _log(f"[EP {episode + 1}] reset obs_keys={list(obs.keys()) if isinstance(obs, dict) else type(obs)}")
            if args_cli.refresh_camera_xform:
                _refresh_camera_xforms(env)
            if recorder is not None:
                recorder.start_episode(episode, env, args_cli)

            if args_cli.record_warmup_steps > 0:
                warmup_actions = torch.zeros(env.action_space.shape, device=device)
                warmup_actions[:, 6] = OPEN_ACTION
                for warmup_step in range(args_cli.record_warmup_steps):
                    obs, _, warmup_terminated, warmup_truncated, _ = env.step(warmup_actions)
                    if warmup_terminated.any() or warmup_truncated.any():
                        _log(
                            f"[EP {episode + 1}] warmup ended early at step {warmup_step + 1}: "
                            f"terminated={warmup_terminated.detach().cpu().tolist()} "
                            f"truncated={warmup_truncated.detach().cpu().tolist()}"
                        )
                        break
            if recorder is not None:
                recorder.save_quality_check(obs)

            phase = torch.zeros(env.unwrapped.num_envs, device=device, dtype=torch.long)
            phase_steps = torch.zeros_like(phase)
            success = torch.zeros(env.unwrapped.num_envs, device=device, dtype=torch.bool)
            terminated = torch.zeros_like(success)
            truncated = torch.zeros_like(success)

            for step in range(args_cli.max_steps):
                if not simulation_app.is_running():
                    break
                with torch.no_grad():
                    pre_obs = obs
                    phase_before = phase.clone()
                    desired_pos_w, gripper = _make_waypoint(env, phase, target_xy, args_cli)
                    actions, distance = _compute_action(env, desired_pos_w, gripper, args_cli)
                    pre_cube_pos = _cube_pos_w(env).clone()
                    pre_target_pos = _target_center_w(env, target_xy, args_cli.release_z)
                    pre_cube_target_xy_error = torch.linalg.vector_norm(
                        pre_cube_pos[:, :2] - pre_target_pos[:, :2], dim=1
                    )
                    pre_opening = _gripper_opening(env, gripper_joint_ids).clone()
                    obs, reward, terminated, truncated, info = env.step(actions)
                    success = _success_term(env)
                    reached = distance < args_cli.pos_threshold
                    phase = _advance_phase(phase, phase_steps, reached, args_cli)
                    if recorder is not None:
                        recorder.record_step(
                            pre_obs,
                            step,
                            phase_before,
                            actions,
                            desired_pos_w,
                            distance,
                            pre_cube_pos,
                            pre_cube_target_xy_error,
                            pre_opening,
                            reward,
                            success,
                            terminated,
                            truncated,
                        )

                    if step == 0 or (step + 1) % args_cli.log_every == 0 or success.any():
                        reward_list = reward.detach().cpu().tolist()
                        phase_names = [PHASES[int(idx)] for idx in phase.detach().cpu().tolist()]
                        _log(
                            f"[EP {episode + 1} STEP {step + 1}] phase={phase_names} "
                            f"dist={distance.detach().cpu().tolist()} "
                            f"cube_target_xy_error={pre_cube_target_xy_error.detach().cpu().tolist()} "
                            f"cube_z={pre_cube_pos[:, 2].detach().cpu().tolist()} "
                            f"gripper_opening={pre_opening.detach().cpu().tolist()} "
                            f"reward={reward_list} success={success.detach().cpu().tolist()} "
                            f"terminated={terminated.detach().cpu().tolist()} truncated={truncated.detach().cpu().tolist()}"
                        )

                    if success.any() or terminated.any() or truncated.any() or torch.all(phase == len(PHASES) - 1):
                        break

            total_successes += int(success.sum().item())
            if recorder is not None:
                recorder.finish_episode(success, step + 1, terminated, truncated)
            _log(
                f"[EP {episode + 1}] final_success={success.detach().cpu().tolist()} "
                f"terminated={terminated.detach().cpu().tolist()} truncated={truncated.detach().cpu().tolist()} "
                f"steps={step + 1}"
            )
            attempted_episodes += 1
            episode += 1

        if args_cli.success_episodes is not None:
            _log(
                f"[SUMMARY] successes={total_successes}/{args_cli.success_episodes} "
                f"attempts={attempted_episodes}"
            )
            if total_successes < args_cli.success_episodes:
                _log(
                    f"[WARN] stopped before reaching target successes: "
                    f"{total_successes}/{args_cli.success_episodes}"
                )
        else:
            _log(f"[SUMMARY] successes={total_successes}/{attempted_episodes * env.unwrapped.num_envs}")
        _log("[OK] Scripted rollout completed.")
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
