"""Repair LeRobot video image statistics computed with uint8 overflow.

LeRobot 0.5.1 can write zero per-episode image std values when stats are
computed on uint8 frames. This script recomputes dataset-level image stats
from the encoded MP4 videos using float64 accumulators and updates
``meta/stats.json`` in place.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


StatsDict = dict[str, dict[str, Any]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="LeRobotDataset root directory.")
    parser.add_argument(
        "--backup-name",
        type=str,
        default="stats.before_image_repair.json",
        help="Backup filename under meta/. Existing backups are left untouched.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _video_features(info: dict) -> tuple[str, ...]:
    features = info.get("features", {})
    video_keys = tuple(key for key, spec in features.items() if spec.get("dtype") == "video")
    if not video_keys:
        raise ValueError("No video features found in meta/info.json")
    return video_keys


def _feature_video_paths(dataset_dir: Path, feature_key: str) -> list[Path]:
    feature_dir = dataset_dir / "videos" / feature_key
    paths = sorted(feature_dir.glob("chunk-*/*.mp4"))
    if not paths:
        raise FileNotFoundError(f"No MP4 videos found for feature {feature_key}: {feature_dir}")
    return paths


def _as_channel_stats(values: np.ndarray) -> list[list[list[float]]]:
    return [[[float(value)]] for value in values.reshape(3)]


def _stats_to_flat_array(stats: StatsDict, feature_key: str, stat_key: str) -> np.ndarray | None:
    if feature_key not in stats or stat_key not in stats[feature_key]:
        return None
    return np.asarray(stats[feature_key][stat_key], dtype=np.float64).reshape(-1)


def _decode_video_stats(paths: list[Path]) -> dict[str, Any]:
    try:
        return _decode_video_stats_cv2(paths)
    except ImportError:
        return _decode_video_stats_av(paths)


def _decode_video_stats_cv2(paths: list[Path]) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is not available for video stats repair.") from exc

    pixel_count = 0
    frame_count = 0
    channel_sum_bgr = np.zeros(3, dtype=np.float64)
    channel_sum_sq_bgr = np.zeros(3, dtype=np.float64)
    channel_min_bgr = np.full(3, np.inf, dtype=np.float64)
    channel_max_bgr = np.full(3, -np.inf, dtype=np.float64)

    for path in paths:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video with OpenCV: {path}")

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            pixels = int(frame.shape[0] * frame.shape[1])
            mean_bgr, std_bgr = cv2.meanStdDev(frame)
            mean_bgr = mean_bgr.reshape(3)
            std_bgr = std_bgr.reshape(3)

            channel_sum_bgr += mean_bgr * pixels
            channel_sum_sq_bgr += (np.square(std_bgr) + np.square(mean_bgr)) * pixels
            flat = frame.reshape(-1, 3)
            channel_min_bgr = np.minimum(channel_min_bgr, flat.min(axis=0))
            channel_max_bgr = np.maximum(channel_max_bgr, flat.max(axis=0))
            pixel_count += pixels
            frame_count += 1

            if frame_count % 25000 == 0:
                print(f"[IMAGE STATS] decoded {frame_count} frames from {path.parent.parent.name}", flush=True)

        capture.release()

    if frame_count == 0 or pixel_count == 0:
        raise ValueError(f"No frames decoded from videos: {[str(path) for path in paths]}")

    # OpenCV decodes color frames as BGR. Convert accumulated channels to RGB.
    return _finalize_channel_stats(
        pixel_count=pixel_count,
        frame_count=frame_count,
        channel_sum=channel_sum_bgr[::-1],
        channel_sum_sq=channel_sum_sq_bgr[::-1],
        channel_min=channel_min_bgr[::-1],
        channel_max=channel_max_bgr[::-1],
    )


def _decode_video_stats_av(paths: list[Path]) -> dict[str, Any]:
    try:
        import av
    except ImportError as exc:
        raise ImportError("Either OpenCV or PyAV is required to repair video image stats.") from exc

    pixel_count = 0
    frame_count = 0
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    channel_min = np.full(3, np.inf, dtype=np.float64)
    channel_max = np.full(3, -np.inf, dtype=np.float64)

    for path in paths:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                rgb = frame.to_ndarray(format="rgb24")
                flat = rgb.reshape(-1, 3).astype(np.float64)
                pixel_count += flat.shape[0]
                frame_count += 1
                channel_sum += flat.sum(axis=0)
                channel_sum_sq += np.square(flat).sum(axis=0)
                channel_min = np.minimum(channel_min, flat.min(axis=0))
                channel_max = np.maximum(channel_max, flat.max(axis=0))

    if frame_count == 0 or pixel_count == 0:
        raise ValueError(f"No frames decoded from videos: {[str(path) for path in paths]}")

    return _finalize_channel_stats(
        pixel_count=pixel_count,
        frame_count=frame_count,
        channel_sum=channel_sum,
        channel_sum_sq=channel_sum_sq,
        channel_min=channel_min,
        channel_max=channel_max,
    )


def _finalize_channel_stats(
    *,
    pixel_count: int,
    frame_count: int,
    channel_sum: np.ndarray,
    channel_sum_sq: np.ndarray,
    channel_min: np.ndarray,
    channel_max: np.ndarray,
) -> dict[str, Any]:
    scale = 255.0
    mean = channel_sum / pixel_count / scale
    mean_sq = channel_sum_sq / pixel_count / (scale * scale)
    variance = np.maximum(mean_sq - np.square(mean), 0.0)
    std = np.sqrt(variance)

    return {
        "frame_count": frame_count,
        "pixel_count": pixel_count,
        "min": channel_min / scale,
        "max": channel_max / scale,
        "mean": mean,
        "std": std,
    }


def repair_image_stats(dataset_dir: Path, backup_name: str = "stats.before_image_repair.json") -> dict[str, Any]:
    dataset_dir = dataset_dir.expanduser().resolve()
    meta_dir = dataset_dir / "meta"
    info_path = meta_dir / "info.json"
    stats_path = meta_dir / "stats.json"
    backup_path = meta_dir / backup_name

    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot info file: {info_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing LeRobot stats file: {stats_path}")

    info = _load_json(info_path)
    stats: StatsDict = _load_json(stats_path)
    video_keys = _video_features(info)

    if not backup_path.exists():
        shutil.copy2(stats_path, backup_path)
        backup_written = True
    else:
        backup_written = False

    report: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "stats_path": str(stats_path),
        "backup_path": str(backup_path),
        "backup_written": backup_written,
        "features": {},
    }

    for feature_key in video_keys:
        old_mean = _stats_to_flat_array(stats, feature_key, "mean")
        old_std = _stats_to_flat_array(stats, feature_key, "std")
        video_paths = _feature_video_paths(dataset_dir, feature_key)
        fixed = _decode_video_stats(video_paths)

        feature_stats = stats.setdefault(feature_key, {})
        feature_stats["min"] = _as_channel_stats(fixed["min"])
        feature_stats["max"] = _as_channel_stats(fixed["max"])
        feature_stats["mean"] = _as_channel_stats(fixed["mean"])
        feature_stats["std"] = _as_channel_stats(fixed["std"])
        if "count" not in feature_stats:
            feature_stats["count"] = [int(fixed["frame_count"])]

        report["features"][feature_key] = {
            "videos": [str(path) for path in video_paths],
            "frame_count": int(fixed["frame_count"]),
            "pixel_count": int(fixed["pixel_count"]),
            "old_mean": None if old_mean is None else old_mean.tolist(),
            "old_std": None if old_std is None else old_std.tolist(),
            "new_mean": fixed["mean"].tolist(),
            "new_std": fixed["std"].tolist(),
            "new_min": fixed["min"].tolist(),
            "new_max": fixed["max"].tolist(),
        }

        print(
            "[IMAGE STATS] "
            f"{feature_key}: old_std={None if old_std is None else np.round(old_std, 6).tolist()} "
            f"new_std={np.round(fixed['std'], 6).tolist()} frames={fixed['frame_count']}",
            flush=True,
        )

    _write_json(stats_path, stats)
    print(f"[IMAGE STATS] wrote repaired stats to {stats_path}", flush=True)
    if backup_written:
        print(f"[IMAGE STATS] wrote backup to {backup_path}", flush=True)
    else:
        print(f"[IMAGE STATS] kept existing backup at {backup_path}", flush=True)

    return report


def main() -> None:
    args = _parse_args()
    report = repair_image_stats(args.dataset_dir, backup_name=args.backup_name)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
