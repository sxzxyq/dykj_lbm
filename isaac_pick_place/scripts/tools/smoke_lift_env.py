"""Smoke-test an Isaac Lab Franka lift environment for reset/step and space inspection."""

import argparse
import asyncio
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


parser = argparse.ArgumentParser(description="Smoke-test an Isaac Lab environment for this project.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Rel-v0", help="Isaac Lab task id.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of vectorized environments.")
parser.add_argument(
    "--steps",
    type=int,
    default=20,
    help="Number of steps to run. Use -1 to keep running until the window is closed or Ctrl+C is pressed.",
)
parser.add_argument(
    "--action-mode",
    choices=["zero", "random"],
    default="zero",
    help="Action source for stepping the environment.",
)
parser.add_argument(
    "--report",
    type=str,
    default="/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/smoke_lift_env_report.txt",
    help="Path to write the smoke-test report.",
)
parser.add_argument(
    "--camera-names",
    type=str,
    default="wrist_cam,observer_wrist_cam,global_cam",
    help="Comma-separated scene camera sensor names to debug.",
)
parser.add_argument(
    "--print-camera-pose",
    action="store_true",
    default=False,
    help="Print camera world poses from Isaac Lab sensor data.",
)
parser.add_argument(
    "--save-camera-frame-dir",
    type=str,
    default=None,
    help="Directory to save RGB frames from camera sensor data.",
)
parser.add_argument(
    "--save-viewport-frame-dir",
    type=str,
    default=None,
    help="Directory to save captures from the active Isaac Sim Viewport.",
)
parser.add_argument(
    "--camera-frame-step",
    type=int,
    default=1,
    help="Environment step at which to save camera frames. Use 0 to save immediately after reset.",
)
parser.add_argument(
    "--viewport-frame-step",
    type=int,
    default=1,
    help="Environment step at which to save the active viewport. Use 0 to save immediately after reset.",
)
parser.add_argument(
    "--viewport-camera-name",
    type=str,
    default=None,
    help="Optional scene camera sensor name or USD prim path to switch the Viewport to before capture.",
)
parser.add_argument(
    "--viewport-resolution",
    type=str,
    default=None,
    help="Optional Viewport capture resolution as WIDTHxHEIGHT, for example 1280x720.",
)
parser.add_argument(
    "--refresh-camera-xform",
    action="store_true",
    default=False,
    help="Rewrite camera local xform ops from cfg after reset to test USD/viewport transform refresh issues.",
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
from isaaclab_tasks.utils import parse_env_cfg


REPORT_LINES = []


def _log(message):
    print(message, flush=True)
    REPORT_LINES.append(message)


def _summarize_obs(obs, prefix="obs"):
    if isinstance(obs, dict):
        _log(f"{prefix}: dict keys={list(obs.keys())}")
        for key, value in obs.items():
            _summarize_obs(value, prefix=f"{prefix}.{key}")
    elif hasattr(obs, "shape"):
        _log(f"{prefix}: shape={tuple(obs.shape)}, dtype={getattr(obs, 'dtype', type(obs))}")
    else:
        _log(f"{prefix}: type={type(obs)}")


def _is_shutdown_race_error(exc: Exception) -> bool:
    """Return True for Isaac/PhysX errors caused by closing the GUI while stepping."""
    text = str(exc)
    shutdown_fragments = (
        "Simulation view object is invalidated",
        "Failed to get DOF velocities from backend",
        "physics.tensors simulationView was invalidated",
        "was deleted while being used by a shape in a tensor view class",
    )
    return any(fragment in text for fragment in shutdown_fragments)


def _camera_names():
    return [name.strip() for name in args_cli.camera_names.split(",") if name.strip()]


def _format_tensor_row(tensor):
    row = tensor[0].detach().cpu().tolist()
    return "[" + ", ".join(f"{value:.6f}" for value in row) + "]"


def _log_camera_poses(env, label):
    for name in _camera_names():
        if name not in env.unwrapped.scene.sensors:
            _log(f"[CAMERA {label}] {name}: missing from scene sensors")
            continue
        camera = env.unwrapped.scene[name]
        data = camera.data
        _log(f"[CAMERA {label}] {name}.pos_w={_format_tensor_row(data.pos_w)}")
        _log(f"[CAMERA {label}] {name}.quat_w_world={_format_tensor_row(data.quat_w_world)}")
        if hasattr(data, "quat_w_ros"):
            _log(f"[CAMERA {label}] {name}.quat_w_ros={_format_tensor_row(data.quat_w_ros)}")


def _save_camera_frames(env, label):
    if args_cli.save_camera_frame_dir is None:
        return
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Saving camera frames requires pillow/PIL in the Isaac Lab environment.") from exc

    output_dir = Path(args_cli.save_camera_frame_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in _camera_names():
        if name not in env.unwrapped.scene.sensors:
            _log(f"[CAMERA {label}] {name}: missing from scene sensors, skip frame save")
            continue
        camera = env.unwrapped.scene[name]
        rgb = camera.data.output.get("rgb")
        if rgb is None:
            _log(f"[CAMERA {label}] {name}: no rgb output, available={list(camera.data.output.keys())}")
            continue
        image = rgb[0].detach().cpu()
        if image.shape[-1] == 4:
            image = image[..., :3]
        if image.dtype != torch.uint8:
            image = image.clamp(0, 255).to(torch.uint8)
        image_path = output_dir / f"{label}_{name}.png"
        Image.fromarray(image.numpy()).save(image_path)
        _log(f"[CAMERA {label}] saved {name} rgb to {image_path}")


def _resolve_viewport_camera_path(env):
    if not args_cli.viewport_camera_name:
        return None
    camera_name = args_cli.viewport_camera_name.strip()
    if not camera_name:
        return None
    if camera_name.startswith("/"):
        return camera_name
    if camera_name not in env.unwrapped.scene.sensors:
        _log(f"[VIEWPORT] camera sensor {camera_name!r} missing from scene sensors")
        return None
    camera = env.unwrapped.scene[camera_name]
    sensor_prims = getattr(camera, "_sensor_prims", [])
    if not sensor_prims:
        _log(f"[VIEWPORT] camera sensor {camera_name!r} has no USD sensor prims")
        return None
    return sensor_prims[0].GetPath().pathString


def _parse_viewport_resolution():
    if args_cli.viewport_resolution is None:
        return None
    text = args_cli.viewport_resolution.lower().replace(",", "x")
    try:
        width_text, height_text = text.split("x", 1)
        resolution = (int(width_text), int(height_text))
    except ValueError as exc:
        raise ValueError("--viewport-resolution must be WIDTHxHEIGHT, for example 1280x720") from exc
    if resolution[0] <= 0 or resolution[1] <= 0:
        raise ValueError("--viewport-resolution values must be positive")
    return resolution


def _save_viewport_frame(env, label):
    if args_cli.save_viewport_frame_dir is None:
        return

    async def _capture_async(viewport, image_path):
        from omni.kit.viewport.utility import capture_viewport_to_file, next_viewport_frame_async
        import omni.kit.app
        import omni.renderer_capture

        await next_viewport_frame_async(viewport)
        capture = capture_viewport_to_file(viewport, file_path=str(image_path))
        result = await capture.wait_for_result(completion_frames=30)
        omni.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
        for _ in range(3):
            await omni.kit.app.get_app().next_update_async()
        return result

    try:
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Sdf
    except ImportError as exc:
        _log(f"[VIEWPORT {label}] unavailable in this Isaac Sim session: {exc}")
        return

    viewport = get_active_viewport()
    if viewport is None:
        _log(f"[VIEWPORT {label}] no active viewport; run without --headless to capture the GUI viewpoint")
        return

    camera_path = _resolve_viewport_camera_path(env)
    if camera_path is not None:
        if hasattr(viewport, "set_active_camera"):
            viewport.set_active_camera(camera_path)
        else:
            viewport.camera_path = Sdf.Path(camera_path)
        _log(f"[VIEWPORT {label}] active_camera={camera_path}")

    resolution = _parse_viewport_resolution()
    if resolution is not None:
        viewport.resolution = resolution
        _log(f"[VIEWPORT {label}] resolution={resolution[0]}x{resolution[1]}")

    output_dir = Path(args_cli.save_viewport_frame_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{label}_viewport.png"
    result = asyncio.run(_capture_async(viewport, image_path))
    _log(f"[VIEWPORT {label}] saved capture to {image_path}, result={result}")


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
        _log(
            f"[CAMERA refresh] {name}: refreshed {refreshed} prim(s), "
            f"pos={tuple(float(value) for value in cfg.offset.pos)}, "
            f"opengl_rot={_format_tensor_row(rot_offset.unsqueeze(0))}"
        )


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    _log(f"[INFO] task={args_cli.task}")
    _log(f"[INFO] device={env.unwrapped.device}")
    _log(f"[INFO] num_envs={env.unwrapped.num_envs}")
    _log(f"[INFO] observation_space={env.observation_space}")
    _log(f"[INFO] action_space={env.action_space}")
    _log(f"[INFO] scene_keys={env.unwrapped.scene.keys()}")
    _log(f"[INFO] scene_sensors={list(env.unwrapped.scene.sensors.keys())}")

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    _summarize_obs(obs)
    if args_cli.refresh_camera_xform:
        _refresh_camera_xforms(env)
    if args_cli.print_camera_pose:
        _log_camera_poses(env, "reset")
    if args_cli.camera_frame_step == 0:
        _save_camera_frames(env, "step_000000")
    if args_cli.viewport_frame_step == 0:
        _save_viewport_frame(env, "step_000000")

    action_shape = env.action_space.shape
    actions = torch.zeros(action_shape, device=env.unwrapped.device)
    _log(f"[INFO] action_mode={args_cli.action_mode}")
    _log(f"[INFO] action_shape={tuple(actions.shape)}")

    step = 0
    try:
        while simulation_app.is_running() and (args_cli.steps < 0 or step < args_cli.steps):
            if args_cli.action_mode == "random":
                actions = 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1
            step_out = env.step(actions)
            obs, reward, terminated, truncated, info = step_out
            should_log = step == 0 or (args_cli.steps > 0 and step == args_cli.steps - 1)
            should_log = should_log or (args_cli.steps < 0 and (step + 1) % 1000 == 0)
            if should_log:
                reward_cpu = reward.detach().cpu().tolist() if hasattr(reward, "detach") else reward
                _log(
                    f"[STEP {step + 1}] reward={reward_cpu}, "
                    f"terminated={terminated}, truncated={truncated}, info_keys={list(info.keys())}"
                )
            if args_cli.print_camera_pose and step + 1 == max(1, args_cli.camera_frame_step):
                _log_camera_poses(env, f"step_{step + 1:06d}")
            if step + 1 == args_cli.camera_frame_step:
                _save_camera_frames(env, f"step_{step + 1:06d}")
            if step + 1 == args_cli.viewport_frame_step:
                _save_viewport_frame(env, f"step_{step + 1:06d}")
            step += 1
    except KeyboardInterrupt:
        _log(f"[INFO] Interrupted by user at step {step}.")
    except Exception as exc:
        if _is_shutdown_race_error(exc):
            _log(f"[INFO] Isaac Sim shutdown invalidated the physics tensor view at step {step}; treating as exit.")
        else:
            _log(f"[ERROR] {type(exc).__name__}: {exc}")
            raise
    finally:
        _log("[OK] Smoke test completed.")
        try:
            env.close()
        except Exception as exc:
            if _is_shutdown_race_error(exc):
                _log("[INFO] Ignored tensor-view invalidation during env.close().")
            else:
                raise
        report_path = Path(args_cli.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(REPORT_LINES) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
    simulation_app.close()
