"""Replay recorded dual-arm handoff expert actions in the Isaac Lab environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher


DEFAULT_RAW_DIR = PROJECT_ROOT / "experiments" / "raw_demos" / "raw_handoff_handoff_v2_full_180train20val"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "experiments" / "reports" / f"replay_handoff_raw_{time.strftime('%Y%m%d_%H%M%S')}"
HANDOFF_TASK = "Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Raw handoff dataset directory.")
parser.add_argument(
    "--episodes",
    type=str,
    default="",
    help="Comma-separated raw episode indices/names to replay. Empty means sample successful episodes.",
)
parser.add_argument("--sample-count", type=int, default=3, help="Number of successful episodes to sample when --episodes is empty.")
parser.add_argument("--task", type=str, default=HANDOFF_TASK, help="Isaac Lab task id.")
parser.add_argument("--max-steps", type=int, default=0, help="Replay cap. 0 means the full recorded episode.")
parser.add_argument("--warmup-steps", type=int, default=2, help="Zero-action warmup steps after reset, matching V2 collection.")
parser.add_argument("--stable-steps", type=int, default=12, help="Stable area steps required for replay success.")
parser.add_argument("--log-every", type=int, default=200, help="Log replay status every N steps.")
parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Directory for replay reports.")
parser.add_argument("--force-initial-state", action="store_true", help="Write first raw pre-state joints/cube position after reset.")
parser.add_argument("--cube-size-m", type=float, default=0.05, help="Cube side length for env construction.")
parser.add_argument("--cube-radius-range", type=str, default="0.0,0.10", help="CUBE_RADIUS_RANGE for env reset.")
parser.add_argument("--cube-angle-range-deg", type=str, default="-180,180", help="CUBE_ANGLE_RANGE_DEG for env reset.")
parser.add_argument("--cube-yaw-range-deg", type=str, default="-180,180", help="CUBE_YAW_RANGE_DEG for env reset.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

os.environ["CUBE_SIZE_M"] = str(args_cli.cube_size_m)
os.environ["CUBE_RADIUS_RANGE"] = args_cli.cube_radius_range
os.environ["CUBE_ANGLE_RANGE_DEG"] = args_cli.cube_angle_range_deg
os.environ["CUBE_YAW_RANGE_DEG"] = args_cli.cube_yaw_range_deg

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import isaac_pick_place.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


LEFT_ARM = "robot"
RIGHT_ARM = "observer_robot"
HANDOFF_YELLOW_CENTER_W = (0.50, 0.00, 0.0255)
HANDOFF_RED_CENTER_W = (0.50, 0.30, 0.0255)
HANDOFF_AREA_SIZE_XY = (0.12, 0.12)
HANDOFF_HEIGHT_TOLERANCE = 0.03
GRIPPER_OPEN_THRESHOLD = 0.01


def _log(message: str) -> None:
    print(message, flush=True)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rows(episode_dir: Path) -> list[dict]:
    rows = []
    with (episode_dir / "steps.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{episode_dir}/steps.jsonl is empty")
    return rows


def _successful_episode_dirs(raw_dir: Path) -> list[Path]:
    episode_dirs = []
    for summary_path in sorted(raw_dir.glob("episode_*/summary.json")):
        summary = _read_json(summary_path)
        if summary.get("success"):
            episode_dirs.append(summary_path.parent)
    return episode_dirs


def _episode_dirs(raw_dir: Path, episodes: str, sample_count: int) -> list[Path]:
    if raw_dir.name.startswith("episode_") and (raw_dir / "steps.jsonl").exists():
        return [raw_dir]
    if episodes.strip():
        result = []
        for item in episodes.split(","):
            item = item.strip()
            if not item:
                continue
            name = item if item.startswith("episode_") else f"episode_{int(item):06d}"
            episode_dir = raw_dir / name
            if not episode_dir.exists():
                raise FileNotFoundError(episode_dir)
            result.append(episode_dir)
        return result
    successful = _successful_episode_dirs(raw_dir)
    if not successful:
        raise RuntimeError(f"No successful raw episodes found in {raw_dir}")
    if sample_count >= len(successful):
        return successful
    if sample_count <= 1:
        return [successful[0]]
    # Spread samples across the successful list so we do not only test easy early episodes.
    indices = [round(i * (len(successful) - 1) / (sample_count - 1)) for i in range(sample_count)]
    return [successful[i] for i in indices]


def _cube_pos_w(env) -> torch.Tensor:
    return env.unwrapped.scene["object"].data.root_pos_w[:, :3]


def _gripper_opening(env, arm_name: str) -> torch.Tensor:
    robot = env.unwrapped.scene[arm_name]
    joint_ids, _ = robot.find_joints(["panda_finger.*"])
    return robot.data.joint_pos[:, joint_ids].sum(dim=1)


def _gripper_is_open(env, arm_name: str) -> torch.Tensor:
    return _gripper_opening(env, arm_name) >= GRIPPER_OPEN_THRESHOLD


def _object_on_area(env, center_w: tuple[float, float, float], gripper_arm: str) -> torch.Tensor:
    cube_pos = _cube_pos_w(env)
    center = torch.tensor(center_w, device=env.unwrapped.device, dtype=cube_pos.dtype).unsqueeze(0)
    xy_error = torch.abs(cube_pos[:, :2] - center[:, :2])
    inside = torch.logical_and(
        xy_error[:, 0] <= HANDOFF_AREA_SIZE_XY[0] * 0.5,
        xy_error[:, 1] <= HANDOFF_AREA_SIZE_XY[1] * 0.5,
    )
    low = torch.abs(cube_pos[:, 2] - center[:, 2]) <= HANDOFF_HEIGHT_TOLERANCE
    released = _gripper_is_open(env, gripper_arm)
    return inside & low & released


def _write_initial_state(env, first_row: dict) -> None:
    device = env.unwrapped.device
    pre_arms = first_row.get("pre_arms") or first_row.get("arms") or {}
    for scene_name, raw_name in ((LEFT_ARM, "left"), (RIGHT_ARM, "right")):
        arm = pre_arms.get(raw_name)
        if not arm:
            continue
        robot = env.unwrapped.scene[scene_name]
        joint_pos = torch.tensor(arm["joint_pos"], device=device, dtype=robot.data.joint_pos.dtype).unsqueeze(0)
        joint_vel = torch.tensor(
            arm.get("joint_vel", [0.0] * len(arm["joint_pos"])),
            device=device,
            dtype=robot.data.joint_vel.dtype,
        ).unsqueeze(0)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

    cube = first_row.get("pre_cube") or {}
    cube_pos = cube.get("pos_w") or first_row.get("cube_pos_w")
    if cube_pos is not None:
        object_asset = env.unwrapped.scene["object"]
        pos = torch.tensor(cube_pos[:3], device=device, dtype=object_asset.data.root_pos_w.dtype).unsqueeze(0)
        quat = object_asset.data.root_quat_w.clone()
        pose = torch.cat([pos, quat], dim=-1)
        velocity = torch.zeros((1, 6), device=device, dtype=pos.dtype)
        object_asset.write_root_pose_to_sim(pose)
        object_asset.write_root_velocity_to_sim(velocity)

    scene = getattr(env.unwrapped, "scene", None)
    if scene is not None and hasattr(scene, "write_data_to_sim"):
        scene.write_data_to_sim()
    sim = getattr(env.unwrapped, "sim", None)
    if sim is not None and hasattr(sim, "forward"):
        sim.forward()


def _initial_diff(env, first_row: dict) -> dict:
    raw_cube = torch.tensor(first_row.get("pre_cube", {}).get("pos_w", first_row.get("cube_pos_w"))[:3])
    live_cube = _cube_pos_w(env)[0].detach().cpu()
    result = {"cube_pos_l2": float(torch.linalg.norm(live_cube - raw_cube).item())}
    pre_arms = first_row.get("pre_arms") or first_row.get("arms") or {}
    for scene_name, raw_name in ((LEFT_ARM, "left"), (RIGHT_ARM, "right")):
        arm = pre_arms.get(raw_name)
        if not arm:
            continue
        robot = env.unwrapped.scene[scene_name]
        raw_joint = torch.tensor(arm["joint_pos"])
        live_joint = robot.data.joint_pos[0, : raw_joint.numel()].detach().cpu()
        result[f"{raw_name}_joint_l2"] = float(torch.linalg.norm(live_joint - raw_joint).item())
    return result


def _action_tensor(env, row: dict) -> torch.Tensor:
    action = torch.tensor(row["action"], device=env.unwrapped.device, dtype=torch.float32).unsqueeze(0)
    if action.shape[-1] != env.action_space.shape[-1]:
        raise ValueError(f"Raw action shape={tuple(action.shape)}, env action_space={env.action_space}")
    return action


def _run_episode(env, episode_dir: Path, report_dir: Path) -> dict:
    meta = _read_json(episode_dir / "meta.json")
    summary = _read_json(episode_dir / "summary.json")
    rows = _load_rows(episode_dir)
    seed = int(meta.get("seed", 0))
    max_steps = len(rows) if args_cli.max_steps <= 0 else min(args_cli.max_steps, len(rows))

    reset_out = env.reset(seed=seed)
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    if args_cli.warmup_steps > 0:
        zero = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for _ in range(args_cli.warmup_steps):
            obs, _, terminated, truncated, _ = env.step(zero)
            if terminated.any() or truncated.any():
                break
    initial_before = _initial_diff(env, rows[0])
    if args_cli.force_initial_state:
        _write_initial_state(env, rows[0])
    initial_after = _initial_diff(env, rows[0])

    yellow_stable = 0
    red_stable = 0
    yellow_seen = False
    red_seen = False
    terminated = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device, dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    reward = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    last_step = 0

    trace_path = report_dir / f"{episode_dir.name}_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as trace:
        for step, row in enumerate(rows[:max_steps]):
            action = _action_tensor(env, row)
            obs, reward, terminated, truncated, _ = env.step(action)
            was_yellow_seen = yellow_seen
            was_red_seen = red_seen
            yellow_now = bool(_object_on_area(env, HANDOFF_YELLOW_CENTER_W, RIGHT_ARM)[0].detach().cpu().item())
            red_now = bool(_object_on_area(env, HANDOFF_RED_CENTER_W, LEFT_ARM)[0].detach().cpu().item())
            yellow_stable = yellow_stable + 1 if yellow_now else 0
            red_stable = red_stable + 1 if red_now else 0
            yellow_seen = yellow_seen or yellow_stable >= args_cli.stable_steps
            red_seen = red_seen or red_stable >= args_cli.stable_steps
            cube_pos = _cube_pos_w(env)[0].detach().cpu().tolist()
            last_step = step + 1
            if (
                step == 0
                or (step + 1) % args_cli.log_every == 0
                or (yellow_seen and not was_yellow_seen)
                or (red_seen and not was_red_seen)
            ):
                _log(
                    f"[{episode_dir.name} step {step + 1}/{max_steps}] "
                    f"cube={cube_pos} yellow_stable={yellow_stable} red_stable={red_stable} "
                    f"yellow_seen={yellow_seen} red_seen={red_seen}"
                )
            trace.write(
                json.dumps(
                    {
                        "step": step,
                        "raw_phase": row.get("phase"),
                        "cube_pos_w": cube_pos,
                        "yellow_stable": yellow_stable,
                        "red_stable": red_stable,
                        "terminated": bool(terminated[0].detach().cpu().item()),
                        "truncated": bool(truncated[0].detach().cpu().item()),
                        "reward": float(reward[0].detach().cpu().item()),
                    }
                )
                + "\n"
            )
            if terminated.any() or truncated.any():
                break

    replay_success = bool(yellow_seen and red_seen)
    result = {
        "episode": episode_dir.name,
        "raw_success": bool(summary.get("success")),
        "raw_steps": int(summary.get("steps", len(rows))),
        "raw_final_phase": summary.get("final_phase"),
        "seed": seed,
        "replay_success": replay_success,
        "yellow_seen": bool(yellow_seen),
        "red_seen": bool(red_seen),
        "yellow_stable_final": int(yellow_stable),
        "red_stable_final": int(red_stable),
        "steps_replayed": int(last_step),
        "terminated": bool(terminated[0].detach().cpu().item()),
        "truncated": bool(truncated[0].detach().cpu().item()),
        "initial_diff_before_force": initial_before,
        "initial_diff_after_force": initial_after,
        "force_initial_state": bool(args_cli.force_initial_state),
        "trace": str(trace_path),
    }
    _log(f"[RESULT] {json.dumps(result, ensure_ascii=False)}")
    return result


def main() -> None:
    if args_cli.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    if not args_cli.raw_dir.exists():
        raise FileNotFoundError(args_cli.raw_dir)
    args_cli.report_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    _log(f"[INFO] task={args_cli.task}")
    _log(f"[INFO] raw_dir={args_cli.raw_dir}")
    _log(f"[INFO] report_dir={args_cli.report_dir}")
    _log(f"[INFO] action_space={env.action_space}")

    episode_dirs = _episode_dirs(args_cli.raw_dir, args_cli.episodes, args_cli.sample_count)
    _log(f"[INFO] replay_episodes={[p.name for p in episode_dirs]}")
    results = []
    try:
        for episode_dir in episode_dirs:
            if not simulation_app.is_running():
                break
            results.append(_run_episode(env, episode_dir, args_cli.report_dir))
    finally:
        env.close()

    successes = sum(1 for item in results if item["replay_success"])
    summary = {
        "raw_dir": str(args_cli.raw_dir),
        "task": args_cli.task,
        "episodes": results,
        "successes": successes,
        "attempts": len(results),
        "success_rate": successes / len(results) if results else 0.0,
    }
    (args_cli.report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(f"[SUMMARY] replay_successes={successes}/{len(results)} report={args_cli.report_dir / 'summary.json'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
