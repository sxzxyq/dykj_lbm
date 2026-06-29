"""Convert raw scripted Isaac pick-place demos into a LeRobotDataset v3 dataset.

The raw demo format is produced by ``scripted_pick_place.py --record-dir``. This converter
uses the official LeRobotDataset writer when available, so the output follows the installed
LeRobot version's on-disk format rather than a local approximation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image


DEFAULT_RAW_DIR = (
    "/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/raw_demos/raw_demos_target_aligned_v0_2eps"
)
DEFAULT_OUTPUT_DIR = (
    "/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/lerobot_datasets/lerobot_target_aligned_v0_2eps"
)
DEFAULT_REPO_ID = "local/seven_dof_pick_place_lbm_target_aligned_v0_2eps"
DEFAULT_TASK = "Pick up the cube and place it on the red target area."

ORDERED_IMAGE_KEY_MAP = (
    ("wrist_rgb", "observation.images.wrist_rgb"),
    ("observer_wrist_rgb", "observation.images.observer_wrist_rgb"),
    ("global_rgb", "observation.images.global_rgb"),
)

STATE_COMPONENTS = (
    ("joint_pos", 9),
    ("joint_vel", 9),
    ("ee_position", 3),
    ("ee_quat", 4),
    ("object_position", 3),
    ("target_area_position", 3),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path(DEFAULT_RAW_DIR), help="Raw demo directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Output LeRobotDataset directory.",
    )
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help="Local LeRobot repo id.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Task prompt saved in the dataset.")
    parser.add_argument("--fps", type=int, default=50, help="Dataset frame rate. Isaac env dt=0.02s => 50 FPS.")
    parser.add_argument("--robot-type", type=str, default="franka_panda_isaaclab", help="Robot type metadata.")
    parser.add_argument("--vcodec", type=str, default="h264", help="Video codec passed to LeRobot.")
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="Background image writer threads used by LeRobot before video encoding.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Remove an existing output directory before conversion.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        default=False,
        help="Convert failed raw episodes too. Defaults to success-only.",
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        default=False,
        help="Skip failed raw episodes instead of raising. Ignored when --include-failed is set.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of selected episodes to convert after failed-episode filtering.",
    )
    parser.add_argument(
        "--require-episodes",
        type=int,
        default=None,
        help="Require at least this many selected episodes after filtering.",
    )
    parser.add_argument(
        "--no-load-check",
        action="store_true",
        default=False,
        help="Skip the final LeRobotDataset reload smoke test.",
    )
    parser.add_argument(
        "--skip-image-stats-repair",
        action="store_true",
        default=False,
        help="Skip post-conversion repair of LeRobot video image mean/std stats.",
    )
    return parser.parse_args()


def _import_lerobot():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: lerobot. Run this script in an environment with LeRobot installed, "
            "for example with uv using cached packages:\n"
            "  uv run --with lerobot --with pyarrow --with datasets --with av "
            "python isaac_pick_place/scripts/convert_raw_demos_to_lerobot.py"
        ) from exc
    return LeRobotDataset


def _episode_dirs(raw_dir: Path) -> list[Path]:
    episodes = sorted(path for path in raw_dir.glob("episode_*") if path.is_dir())
    if not episodes:
        raise FileNotFoundError(f"No episode_* directories found in {raw_dir}")
    return episodes


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_steps(episode_dir: Path):
    steps_path = episode_dir / "steps.jsonl"
    with steps_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def _state_names() -> list[str]:
    names: list[str] = []
    for key, size in STATE_COMPONENTS:
        names.extend(f"{key}.{i}" for i in range(size))
    return names


def _features(image_shapes: dict[str, tuple[int, int, int]]) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(_state_names()),),
            "names": _state_names(),
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": [
                "delta_x",
                "delta_y",
                "delta_z",
                "delta_roll",
                "delta_pitch",
                "delta_yaw",
                "gripper",
            ],
        },
    }
    for _, lerobot_key in ORDERED_IMAGE_KEY_MAP:
        if lerobot_key not in image_shapes:
            continue
        features[lerobot_key] = {
            "dtype": "video",
            "shape": image_shapes[lerobot_key],
            "names": ["height", "width", "channels"],
        }
    return features


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _state_from_step(step: dict, episode_dir: Path, line_number: int) -> np.ndarray:
    obs = step.get("obs", {})
    chunks: list[np.ndarray] = []
    for key, expected_size in STATE_COMPONENTS:
        if key not in obs:
            raise KeyError(f"{episode_dir}/steps.jsonl:{line_number} missing obs.{key}")
        values = np.asarray(obs[key], dtype=np.float32)
        if values.shape != (expected_size,):
            raise ValueError(
                f"{episode_dir}/steps.jsonl:{line_number} obs.{key} has shape {values.shape}, "
                f"expected {(expected_size,)}"
            )
        chunks.append(values)
    return np.concatenate(chunks).astype(np.float32)


def _image_shapes_from_first_episode(episode_dir: Path) -> dict[str, tuple[int, int, int]]:
    for _, step in _iter_steps(episode_dir):
        images = step.get("images", {})
        image_shapes: dict[str, tuple[int, int, int]] = {}
        for raw_key, lerobot_key in ORDERED_IMAGE_KEY_MAP:
            raw_rel = images.get(raw_key)
            if raw_rel is None:
                continue
            image = _load_rgb(episode_dir / raw_rel)
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected HWC RGB image for {raw_key}, got shape {image.shape}")
            image_shapes[lerobot_key] = tuple(int(dim) for dim in image.shape)
        if "observation.images.wrist_rgb" not in image_shapes:
            raise KeyError(f"{episode_dir}/steps.jsonl first step missing images.wrist_rgb")
        return image_shapes
    raise ValueError(f"{episode_dir}/steps.jsonl is empty")


def _validate_episode(episode_dir: Path, include_failed: bool) -> dict:
    summary_path = episode_dir / "summary.json"
    meta_path = episode_dir / "meta.json"
    steps_path = episode_dir / "steps.jsonl"
    for path in (summary_path, meta_path, steps_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required raw demo file: {path}")
    summary = _read_json(summary_path)
    if not include_failed and not summary.get("success", False):
        raise ValueError(f"{episode_dir} is not successful; pass --include-failed to convert it anyway")
    return summary


def _select_episodes(args: argparse.Namespace) -> tuple[list[Path], list[dict]]:
    selected_episodes: list[Path] = []
    summaries: list[dict] = []
    for episode_dir in _episode_dirs(args.raw_dir):
        try:
            summary = _validate_episode(episode_dir, args.include_failed)
        except ValueError as exc:
            if args.skip_failed and not args.include_failed and "is not successful" in str(exc):
                print(f"[SKIP] {episode_dir.name}: not successful", flush=True)
                continue
            raise
        selected_episodes.append(episode_dir)
        summaries.append(summary)
        if args.max_episodes is not None and len(selected_episodes) >= args.max_episodes:
            break

    if not selected_episodes:
        raise ValueError(f"No episodes selected from {args.raw_dir}")
    if args.require_episodes is not None and len(selected_episodes) < args.require_episodes:
        raise ValueError(
            f"Selected {len(selected_episodes)} episode(s), but --require-episodes={args.require_episodes}"
        )
    return selected_episodes, summaries


def _frame_from_step(
    episode_dir: Path,
    line_number: int,
    step: dict,
    task: str,
    image_keys: tuple[tuple[str, str], ...],
) -> dict:
    action = np.asarray(step.get("action"), dtype=np.float32)
    if action.shape != (7,):
        raise ValueError(f"{episode_dir}/steps.jsonl:{line_number} action has shape {action.shape}, expected (7,)")

    frame = {
        "observation.state": _state_from_step(step, episode_dir, line_number),
        "action": action,
        "task": task,
    }
    images = step.get("images", {})
    for raw_key, lerobot_key in image_keys:
        rel_path = images.get(raw_key)
        if rel_path is None:
            raise KeyError(f"{episode_dir}/steps.jsonl:{line_number} missing images.{raw_key}")
        image_path = episode_dir / rel_path
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image referenced by steps.jsonl: {image_path}")
        frame[lerobot_key] = _load_rgb(image_path)
    return frame


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)


def _convert(args: argparse.Namespace) -> dict:
    LeRobotDataset = _import_lerobot()
    episodes, summaries = _select_episodes(args)
    image_shapes = _image_shapes_from_first_episode(episodes[0])
    image_keys = tuple(
        (raw_key, lerobot_key) for raw_key, lerobot_key in ORDERED_IMAGE_KEY_MAP if lerobot_key in image_shapes
    )

    _prepare_output_dir(args.output_dir, args.overwrite)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output_dir,
        fps=args.fps,
        robot_type=args.robot_type,
        features=_features(image_shapes),
        use_videos=True,
        image_writer_threads=args.image_writer_threads,
        vcodec=args.vcodec,
    )

    converted: list[dict] = []
    try:
        for episode_index, episode_dir in enumerate(episodes):
            frame_count = 0
            for line_number, step in _iter_steps(episode_dir):
                dataset.add_frame(_frame_from_step(episode_dir, line_number, step, args.task, image_keys))
                frame_count += 1
            dataset.save_episode(parallel_encoding=True)
            converted.append(
                {
                    "raw_episode": episode_dir.name,
                    "episode_index": episode_index,
                    "frames": frame_count,
                    "raw_summary": summaries[episode_index],
                }
            )
            print(f"[OK] converted {episode_dir.name}: {frame_count} frames", flush=True)
    finally:
        dataset.finalize()

    image_stats_repair = None
    if not args.skip_image_stats_repair:
        from repair_lerobot_image_stats import repair_image_stats

        image_stats_repair = repair_image_stats(args.output_dir)

    conversion_summary = {
        "raw_dir": str(args.raw_dir),
        "output_dir": str(args.output_dir),
        "repo_id": args.repo_id,
        "fps": args.fps,
        "task": args.task,
        "image_shape": list(next(iter(image_shapes.values()))),
        "image_shapes": {key: list(shape) for key, shape in image_shapes.items()},
        "image_features": [lerobot_key for _, lerobot_key in image_keys],
        "state_names": _state_names(),
        "image_stats_repair": image_stats_repair,
        "episodes": converted,
        "total_frames": int(sum(item["frames"] for item in converted)),
    }
    summary_path = args.output_dir / "conversion_summary.json"
    summary_path.write_text(json.dumps(conversion_summary, indent=2) + "\n", encoding="utf-8")
    return conversion_summary


def _load_check(args: argparse.Namespace, summary: dict) -> None:
    if args.no_load_check:
        return
    LeRobotDataset = _import_lerobot()
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.output_dir, download_videos=False)
    if len(dataset) != summary["total_frames"]:
        raise RuntimeError(f"Reloaded dataset length {len(dataset)} != expected {summary['total_frames']}")
    sample = dataset[0]
    required_keys = {
        "observation.state",
        "action",
        *summary["image_features"],
    }
    missing = required_keys.difference(sample)
    if missing:
        raise RuntimeError(f"Reloaded sample missing keys: {sorted(missing)}")
    print(f"[OK] reload check passed: {len(dataset)} frames, keys={sorted(sample.keys())}", flush=True)


def main() -> None:
    args = _parse_args()
    if not args.raw_dir.exists():
        raise FileNotFoundError(f"Raw demo directory does not exist: {args.raw_dir}")
    summary = _convert(args)
    _load_check(args, summary)
    print(
        f"[DONE] wrote {summary['total_frames']} frames from {len(summary['episodes'])} episode(s) to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
