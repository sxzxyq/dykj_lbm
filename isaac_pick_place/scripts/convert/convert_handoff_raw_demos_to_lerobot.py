"""Convert raw sequential dual-arm handoff demos into a LeRobotDataset v3 dataset.

The raw demo format is produced by ``scripted_handoff_collect.py --record-dir``.
This converter is intentionally separate from the single-arm converter because the
handoff task has a 14-D dual-arm action and a dual-arm low-dimensional state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = SCRIPTS_ROOT / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from handoff_v2_utils import (
    ACTION_REPRESENTATION_ABSOLUTE_JOINT_POS,
    ACTION_REPRESENTATION_DELTA_STEP,
    ACTION_REPRESENTATION_RELATIVE_CURRENT_POSE_CHUNK,
    ACTION_REPRESENTATIONS,
    DATASET_VERSION_V2_CLEAN,
    DATASET_VERSION_V2_FULL,
    STATE_TIMING_EXACT_PRE_ACTION,
    action_delta_chunk_to_relative_current_np,
    canonicalize_quat_wxyz_np,
    image_stats_for_normalization,
    median_episode_length,
    quat_continuity_report,
    write_manifest,
)


DEFAULT_RAW_DIR = (
    "/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/raw_demos/raw_handoff_debug"
)
DEFAULT_OUTPUT_DIR = (
    "/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/lerobot_datasets/lerobot_handoff_debug"
)
DEFAULT_REPO_ID = "local/seven_dof_pick_place_lbm_handoff_debug"
DEFAULT_TASK = (
    "Right arm moves the blue cube to the yellow handoff area, then left arm moves it to the red target area."
)

ORDERED_IMAGE_KEY_MAP = (
    ("wrist_rgb", "observation.images.wrist_rgb"),
    ("observer_wrist_rgb", "observation.images.observer_wrist_rgb"),
    ("global_rgb", "observation.images.global_rgb"),
)

ACTION_NAMES = (
    "left_delta_x",
    "left_delta_y",
    "left_delta_z",
    "left_delta_roll",
    "left_delta_pitch",
    "left_delta_yaw",
    "left_gripper",
    "right_delta_x",
    "right_delta_y",
    "right_delta_z",
    "right_delta_roll",
    "right_delta_pitch",
    "right_delta_yaw",
    "right_gripper",
)
ABS_JOINT_ACTION_NAMES = (
    *(f"left_joint_pos.{i}" for i in range(9)),
    *(f"right_joint_pos.{i}" for i in range(9)),
)

STATE_LAYOUT_HANDOFF_JOINT_EE_SUBTASK = "handoff_joint_ee_subtask"
STATE_LAYOUT_HANDOFF_JOINT_EE_RELPOSE = "handoff_joint_ee_relpose"
STATE_LAYOUT_HANDOFF_JOINT_EE_BIRELPOSE_TIME = "handoff_joint_ee_birelpose_time"
STATE_LAYOUT_HANDOFF_JOINT_TCP_POS_GRIPPER = "handoff_joint_tcp_pos_gripper"
STATE_LAYOUTS = (
    STATE_LAYOUT_HANDOFF_JOINT_EE_SUBTASK,
    STATE_LAYOUT_HANDOFF_JOINT_EE_RELPOSE,
    STATE_LAYOUT_HANDOFF_JOINT_EE_BIRELPOSE_TIME,
    STATE_LAYOUT_HANDOFF_JOINT_TCP_POS_GRIPPER,
)
ACTION_LAYOUT_EE_DELTA_14 = "ee_delta_14"
ACTION_LAYOUT_ABS_JOINT_POS_18 = "abs_joint_pos_18"
ACTION_LAYOUTS = (
    ACTION_LAYOUT_EE_DELTA_14,
    ACTION_LAYOUT_ABS_JOINT_POS_18,
)
STATE_TIMING_POST_ACTION = "post_action"
STATE_TIMING_PRE_ACTION_FROM_PREVIOUS_POST = "previous_post_as_pre"
STATE_TIMINGS = (
    STATE_TIMING_EXACT_PRE_ACTION,
    STATE_TIMING_POST_ACTION,
    STATE_TIMING_PRE_ACTION_FROM_PREVIOUS_POST,
)

STAGE_TO_ID = {
    "right_to_yellow": 0.0,
    "left_to_red": 1.0,
    "done": 2.0,
}
ACTIVE_ARM_TO_ID = {
    None: -1.0,
    "robot": 0.0,
    "observer_robot": 1.0,
}
SUBTASKS = (
    "Use observer_robot, the right arm, to pick up the blue cube.",
    "Use observer_robot, the right arm, to place the blue cube on the yellow handoff area.",
    "Hold both arms still while the cube stabilizes on the yellow handoff area.",
    "Use robot, the left arm, to pick up the cube from the yellow handoff area.",
    "Use robot, the left arm, to place the cube on the red target area.",
    "Hold both arms still because the task is complete.",
)
SUBTASK_NAME_TO_ID = {
    "RIGHT_PICK_CUBE": 0,
    "RIGHT_PLACE_YELLOW": 1,
    "WAIT_YELLOW_STABLE": 2,
    "LEFT_PICK_FROM_YELLOW": 3,
    "LEFT_PLACE_RED": 4,
    "DONE_HOLD": 5,
}
ACTIVE_LEFT = 0
ACTIVE_RIGHT = 1
ACTIVE_NONE = 2
SUBTASK_ACTIVE_ARM_ID = {
    SUBTASK_NAME_TO_ID["RIGHT_PICK_CUBE"]: ACTIVE_RIGHT,
    SUBTASK_NAME_TO_ID["RIGHT_PLACE_YELLOW"]: ACTIVE_RIGHT,
    SUBTASK_NAME_TO_ID["WAIT_YELLOW_STABLE"]: ACTIVE_NONE,
    SUBTASK_NAME_TO_ID["LEFT_PICK_FROM_YELLOW"]: ACTIVE_LEFT,
    SUBTASK_NAME_TO_ID["LEFT_PLACE_RED"]: ACTIVE_LEFT,
    SUBTASK_NAME_TO_ID["DONE_HOLD"]: ACTIVE_NONE,
}
PHASE_TO_SUBTASK_ID = {
    "right_open_rest": SUBTASK_NAME_TO_ID["RIGHT_PICK_CUBE"],
    "right_move_above_cube": SUBTASK_NAME_TO_ID["RIGHT_PICK_CUBE"],
    "right_descend_to_grasp": SUBTASK_NAME_TO_ID["RIGHT_PICK_CUBE"],
    "right_close_gripper": SUBTASK_NAME_TO_ID["RIGHT_PICK_CUBE"],
    "right_lift_cube": SUBTASK_NAME_TO_ID["RIGHT_PICK_CUBE"],
    "right_move_above_yellow": SUBTASK_NAME_TO_ID["RIGHT_PLACE_YELLOW"],
    "right_descend_to_yellow": SUBTASK_NAME_TO_ID["RIGHT_PLACE_YELLOW"],
    "right_release_on_yellow": SUBTASK_NAME_TO_ID["RIGHT_PLACE_YELLOW"],
    "right_retreat": SUBTASK_NAME_TO_ID["RIGHT_PLACE_YELLOW"],
    "wait_yellow_stable": SUBTASK_NAME_TO_ID["WAIT_YELLOW_STABLE"],
    "left_open_rest": SUBTASK_NAME_TO_ID["LEFT_PICK_FROM_YELLOW"],
    "left_move_above_cube": SUBTASK_NAME_TO_ID["LEFT_PICK_FROM_YELLOW"],
    "left_descend_to_grasp": SUBTASK_NAME_TO_ID["LEFT_PICK_FROM_YELLOW"],
    "left_close_gripper": SUBTASK_NAME_TO_ID["LEFT_PICK_FROM_YELLOW"],
    "left_lift_cube": SUBTASK_NAME_TO_ID["LEFT_PICK_FROM_YELLOW"],
    "left_move_above_red": SUBTASK_NAME_TO_ID["LEFT_PLACE_RED"],
    "left_descend_to_red": SUBTASK_NAME_TO_ID["LEFT_PLACE_RED"],
    "left_release_on_red": SUBTASK_NAME_TO_ID["LEFT_PLACE_RED"],
    "left_retreat": SUBTASK_NAME_TO_ID["LEFT_PLACE_RED"],
    "wait_red_stable": SUBTASK_NAME_TO_ID["DONE_HOLD"],
    "done": SUBTASK_NAME_TO_ID["DONE_HOLD"],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path(DEFAULT_RAW_DIR), help="Raw handoff demo directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Output LeRobotDataset directory.",
    )
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help="Local LeRobot repo id.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Task prompt saved in the dataset.")
    parser.add_argument("--fps", type=int, default=50, help="Dataset frame rate. Isaac env dt=0.02s => 50 FPS.")
    parser.add_argument("--robot-type", type=str, default="dual_franka_panda_isaaclab", help="Robot type metadata.")
    parser.add_argument("--vcodec", type=str, default="h264", help="Video codec passed to LeRobot.")
    parser.add_argument(
        "--state-layout",
        type=str,
        default=STATE_LAYOUT_HANDOFF_JOINT_EE_SUBTASK,
        choices=STATE_LAYOUTS,
        help=(
            "Low-dimensional state layout to write. handoff_joint_ee_subtask preserves the existing "
            "54D state with subtask/active-arm one-hots. handoff_joint_ee_relpose writes a 41D state "
            "with dual-arm joint/TCP state plus right TCP pose relative to left TCP, without stage or "
            "active-arm fields. handoff_joint_ee_birelpose_time writes a 49D state with dual-arm "
            "joint/TCP state, bidirectional TCP relative poses, and one continuous episode progress scalar. "
            "handoff_joint_tcp_pos_gripper writes a 26D simplified state with joint_pos, tcp_pos_w, "
            "and gripper_opening for both arms."
        ),
    )
    parser.add_argument(
        "--state-timing",
        type=str,
        default=STATE_TIMING_EXACT_PRE_ACTION,
        choices=STATE_TIMINGS,
        help=(
            "Which raw row supplies observation.state. exact_pre_action uses the current row's pre_arms/pre_images "
            "and fails if they are missing. post_action preserves the legacy behavior and uses the "
            "same raw row as the action. previous_post_as_pre uses the previous raw row's post-action arm state "
            "as an approximation of the current frame's pre-action state, aligning low-dimensional state with "
            "the pre-action image stored in the current row. Frame 0 falls back to its own row."
        ),
    )
    parser.add_argument("--dataset-version", type=str, default="handoff_v1", help="Dataset version stored in manifest.")
    parser.add_argument("--split-name", type=str, default="train", help="Logical split name stored in manifest.")
    parser.add_argument("--cube-size-m", type=float, default=None, help="Cube side length in meters stored in manifest.")
    parser.add_argument(
        "--image-normalization",
        type=str,
        default="dataset_stats",
        choices=("dataset_stats", "clip", "imagenet"),
        help="Image normalization contract stored in manifest and optionally used to overwrite image stats.",
    )
    parser.add_argument(
        "--image-augmentation",
        type=str,
        default="none",
        help="Image augmentation contract stored in manifest.",
    )
    parser.add_argument(
        "--action-representation",
        type=str,
        default=ACTION_REPRESENTATION_DELTA_STEP,
        choices=ACTION_REPRESENTATIONS,
        help=(
            "Model-side action chunk representation. Raw LeRobot frames still store single-step env deltas; "
            "relative_current_pose_chunk recomputes action stats for cumulative future offsets sharing the "
            "current observation as reference."
        ),
    )
    parser.add_argument(
        "--action-layout",
        type=str,
        default=ACTION_LAYOUT_EE_DELTA_14,
        choices=ACTION_LAYOUTS,
        help=(
            "Action target layout written to LeRobot. ee_delta_14 keeps the raw dual-arm end-effector delta "
            "command. abs_joint_pos_18 writes post_arms.left/right joint_pos as an 18D absolute joint target."
        ),
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=32,
        help="Action horizon used when computing relative_current_pose_chunk action stats.",
    )
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
        "--skip-episodes-count",
        type=int,
        default=0,
        help="Skip this many selected successful episodes before applying --max-episodes.",
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
            "for example:\n"
            "  /home/ubuntu/Workspace/multitask_dit_policy/.venv/bin/python "
            "isaac_pick_place/scripts/convert_handoff_raw_demos_to_lerobot.py"
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


def _vector(
    values,
    expected_size: int,
    source: str,
    *,
    default: tuple[float, ...] | None = None,
) -> np.ndarray:
    if values is None:
        if default is None:
            raise KeyError(f"Missing required value: {source}")
        values = default
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (expected_size,):
        raise ValueError(f"{source} has shape {array.shape}, expected {(expected_size,)}")
    return array


def _scalar(value, source: str, default: float | None = None) -> np.ndarray:
    if value is None:
        if default is None:
            raise KeyError(f"Missing required value: {source}")
        value = default
    return np.asarray([value], dtype=np.float32)


def _int_scalar(value: int, source: str) -> np.ndarray:
    if value is None:
        raise KeyError(f"Missing required value: {source}")
    return np.asarray([value], dtype=np.int64)


def _one_hot(index: int, size: int, source: str) -> np.ndarray:
    if index < 0 or index >= size:
        raise ValueError(f"{source}={index} is outside one-hot size {size}")
    values = np.zeros((size,), dtype=np.float32)
    values[index] = 1.0
    return values


def _normalize_quat_wxyz(quat: np.ndarray, source: str) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-8:
        raise ValueError(f"{source} has near-zero norm")
    return (quat / norm).astype(np.float32)


def _canonicalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    return canonicalize_quat_wxyz_np(quat)


def _arm_tcp_quat(arm: dict, source: str) -> np.ndarray:
    return canonicalize_quat_wxyz_np(
        _vector(
            arm.get("tcp_quat_w"),
            4,
            source,
            default=(1.0, 0.0, 0.0, 0.0),
        ),
        source,
    )


def _quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    return np.asarray([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float32)


def _quat_multiply_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float32,
    )


def _quat_apply_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q_xyz = quat[1:4]
    t = 2.0 * np.cross(q_xyz, vector)
    return (vector + quat[0] * t + np.cross(q_xyz, t)).astype(np.float32)


def _relative_tcp_pose(
    frame_arm: dict,
    target_arm: dict,
    frame_name: str,
    target_name: str,
    source: str,
) -> tuple[np.ndarray, np.ndarray]:
    frame_pos = _vector(frame_arm.get("tcp_pos_w"), 3, f"{source}.{frame_name}.tcp_pos_w")
    target_pos = _vector(target_arm.get("tcp_pos_w"), 3, f"{source}.{target_name}.tcp_pos_w")
    frame_quat = _arm_tcp_quat(frame_arm, f"{source}.{frame_name}.tcp_quat_w")
    target_quat = _arm_tcp_quat(target_arm, f"{source}.{target_name}.tcp_quat_w")
    frame_inv = _quat_conjugate_wxyz(frame_quat)
    rel_pos = _quat_apply_wxyz(frame_inv, target_pos - frame_pos)
    rel_quat = _quat_multiply_wxyz(frame_inv, target_quat)
    rel_quat = _canonicalize_quat_wxyz(
        _normalize_quat_wxyz(rel_quat, f"{source}.{target_name}_tcp_quat_in_{frame_name}_tcp_frame")
    )
    return rel_pos, rel_quat


def _relative_tcp_pose_right_in_left(left: dict, right: dict, source: str) -> tuple[np.ndarray, np.ndarray]:
    return _relative_tcp_pose(left, right, "left", "right", source)


def _relative_tcp_pose_left_in_right(left: dict, right: dict, source: str) -> tuple[np.ndarray, np.ndarray]:
    return _relative_tcp_pose(right, left, "right", "left", source)


def _subtask_id_from_phase(phase: str, source: str) -> int:
    if phase not in PHASE_TO_SUBTASK_ID:
        raise ValueError(f"{source} unknown phase={phase!r}")
    return PHASE_TO_SUBTASK_ID[phase]


def _base_arm_state_names() -> list[str]:
    names: list[str] = []
    for prefix in ("left", "right"):
        names.extend(f"{prefix}_joint_pos.{i}" for i in range(9))
        names.extend(f"{prefix}_tcp_pos_w.{i}" for i in range(3))
        names.extend(f"{prefix}_tcp_quat_w.{i}" for i in range(4))
        names.append(f"{prefix}_gripper_opening.0")
    return names


def _state_names(state_layout: str) -> list[str]:
    names = _base_arm_state_names()
    if state_layout == STATE_LAYOUT_HANDOFF_JOINT_TCP_POS_GRIPPER:
        simple_names: list[str] = []
        for prefix in ("left", "right"):
            simple_names.extend(f"{prefix}_joint_pos.{i}" for i in range(9))
            simple_names.extend(f"{prefix}_tcp_pos_w.{i}" for i in range(3))
            simple_names.append(f"{prefix}_gripper_opening.0")
        return simple_names
    if state_layout == STATE_LAYOUT_HANDOFF_JOINT_EE_RELPOSE:
        names.extend(f"right_tcp_pos_in_left_tcp_frame.{i}" for i in range(3))
        names.extend(f"right_tcp_quat_in_left_tcp_frame.{i}" for i in range(4))
        return names
    if state_layout == STATE_LAYOUT_HANDOFF_JOINT_EE_BIRELPOSE_TIME:
        names.extend(f"right_tcp_pos_in_left_tcp_frame.{i}" for i in range(3))
        names.extend(f"right_tcp_quat_in_left_tcp_frame.{i}" for i in range(4))
        names.extend(f"left_tcp_pos_in_right_tcp_frame.{i}" for i in range(3))
        names.extend(f"left_tcp_quat_in_right_tcp_frame.{i}" for i in range(4))
        names.append("episode_progress.0")
        return names
    if state_layout != STATE_LAYOUT_HANDOFF_JOINT_EE_SUBTASK:
        raise ValueError(f"Unsupported state_layout={state_layout!r}")
    names.extend(f"cube_pos_w.{i}" for i in range(3))
    names.extend(f"yellow_area_pos_w.{i}" for i in range(3))
    names.extend(f"red_area_pos_w.{i}" for i in range(3))
    names.append("stage_id.0")
    names.append("active_arm_id.0")
    names.extend(f"subtask_onehot.{i}" for i in range(len(SUBTASKS)))
    names.extend(f"active_arm_onehot.{i}" for i in range(3))
    return names


def _action_names(action_layout: str) -> tuple[str, ...]:
    if action_layout == ACTION_LAYOUT_EE_DELTA_14:
        return ACTION_NAMES
    if action_layout == ACTION_LAYOUT_ABS_JOINT_POS_18:
        return ABS_JOINT_ACTION_NAMES
    raise ValueError(f"Unsupported action_layout={action_layout!r}")


def _features(image_shapes: dict[str, tuple[int, int, int]], state_layout: str, action_layout: str) -> dict:
    state_names = _state_names(state_layout)
    action_names = _action_names(action_layout)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": list(action_names),
        },
        "subtask_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
        "subtask": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
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
    arms = step.get("arms", {})
    left = arms.get("left")
    right = arms.get("right")
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise KeyError(f"{episode_dir}/steps.jsonl:{line_number} missing arms.left/right")

    chunks: list[np.ndarray] = []
    for arm_name, arm in (("left", left), ("right", right)):
        prefix = f"{episode_dir}/steps.jsonl:{line_number} arms.{arm_name}"
        chunks.append(_vector(arm.get("joint_pos"), 9, f"{prefix}.joint_pos"))
        chunks.append(_vector(arm.get("tcp_pos_w"), 3, f"{prefix}.tcp_pos_w"))
        chunks.append(_arm_tcp_quat(arm, f"{prefix}.tcp_quat_w"))
        chunks.append(_scalar(arm.get("gripper_opening"), f"{prefix}.gripper_opening", default=0.0))

    chunks.append(_vector(step.get("cube_pos_w"), 3, f"{episode_dir}/steps.jsonl:{line_number} cube_pos_w"))
    chunks.append(
        _vector(step.get("yellow_area_pos_w"), 3, f"{episode_dir}/steps.jsonl:{line_number} yellow_area_pos_w")
    )
    chunks.append(_vector(step.get("red_area_pos_w"), 3, f"{episode_dir}/steps.jsonl:{line_number} red_area_pos_w"))

    stage = step.get("stage")
    active_arm = step.get("active_arm")
    if stage not in STAGE_TO_ID:
        raise ValueError(f"{episode_dir}/steps.jsonl:{line_number} unknown stage={stage!r}")
    if active_arm not in ACTIVE_ARM_TO_ID:
        raise ValueError(f"{episode_dir}/steps.jsonl:{line_number} unknown active_arm={active_arm!r}")
    chunks.append(_scalar(STAGE_TO_ID[stage], "stage_id"))
    chunks.append(_scalar(ACTIVE_ARM_TO_ID[active_arm], "active_arm_id"))
    subtask_id = _subtask_id_from_phase(step.get("phase"), f"{episode_dir}/steps.jsonl:{line_number}")
    active_arm_id = SUBTASK_ACTIVE_ARM_ID[subtask_id]
    chunks.append(_one_hot(subtask_id, len(SUBTASKS), "subtask_id"))
    chunks.append(_one_hot(active_arm_id, 3, "active_arm_id"))

    return np.concatenate(chunks).astype(np.float32)


def _relpose_state_from_step(step: dict, episode_dir: Path, line_number: int) -> np.ndarray:
    arms = step.get("arms", {})
    left = arms.get("left")
    right = arms.get("right")
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise KeyError(f"{episode_dir}/steps.jsonl:{line_number} missing arms.left/right")

    chunks: list[np.ndarray] = []
    for arm_name, arm in (("left", left), ("right", right)):
        prefix = f"{episode_dir}/steps.jsonl:{line_number} arms.{arm_name}"
        chunks.append(_vector(arm.get("joint_pos"), 9, f"{prefix}.joint_pos"))
        chunks.append(_vector(arm.get("tcp_pos_w"), 3, f"{prefix}.tcp_pos_w"))
        chunks.append(_arm_tcp_quat(arm, f"{prefix}.tcp_quat_w"))
        chunks.append(_scalar(arm.get("gripper_opening"), f"{prefix}.gripper_opening", default=0.0))

    source = f"{episode_dir}/steps.jsonl:{line_number} arms"
    rel_pos, rel_quat = _relative_tcp_pose_right_in_left(left, right, source)
    chunks.append(rel_pos)
    chunks.append(rel_quat)
    return np.concatenate(chunks).astype(np.float32)


def _birelpose_time_state_from_step(
    step: dict,
    episode_dir: Path,
    line_number: int,
    episode_progress: float,
) -> np.ndarray:
    arms = step.get("arms", {})
    left = arms.get("left")
    right = arms.get("right")
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise KeyError(f"{episode_dir}/steps.jsonl:{line_number} missing arms.left/right")

    chunks: list[np.ndarray] = []
    for arm_name, arm in (("left", left), ("right", right)):
        prefix = f"{episode_dir}/steps.jsonl:{line_number} arms.{arm_name}"
        chunks.append(_vector(arm.get("joint_pos"), 9, f"{prefix}.joint_pos"))
        chunks.append(_vector(arm.get("tcp_pos_w"), 3, f"{prefix}.tcp_pos_w"))
        chunks.append(_arm_tcp_quat(arm, f"{prefix}.tcp_quat_w"))
        chunks.append(_scalar(arm.get("gripper_opening"), f"{prefix}.gripper_opening", default=0.0))

    source = f"{episode_dir}/steps.jsonl:{line_number} arms"
    right_in_left_pos, right_in_left_quat = _relative_tcp_pose_right_in_left(left, right, source)
    left_in_right_pos, left_in_right_quat = _relative_tcp_pose_left_in_right(left, right, source)
    chunks.extend(
        [
            right_in_left_pos,
            right_in_left_quat,
            left_in_right_pos,
            left_in_right_quat,
            _scalar(episode_progress, "episode_progress"),
        ]
    )
    state = np.concatenate(chunks).astype(np.float32)
    if state.shape != (49,):
        raise ValueError(f"{episode_dir}/steps.jsonl:{line_number} birelpose_time state shape={state.shape}, expected (49,)")
    return state


def _joint_tcp_pos_gripper_state_from_step(step: dict, episode_dir: Path, line_number: int) -> np.ndarray:
    arms = step.get("arms", {})
    left = arms.get("left")
    right = arms.get("right")
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise KeyError(f"{episode_dir}/steps.jsonl:{line_number} missing arms.left/right")

    chunks: list[np.ndarray] = []
    for arm_name, arm in (("left", left), ("right", right)):
        prefix = f"{episode_dir}/steps.jsonl:{line_number} arms.{arm_name}"
        chunks.append(_vector(arm.get("joint_pos"), 9, f"{prefix}.joint_pos"))
        chunks.append(_vector(arm.get("tcp_pos_w"), 3, f"{prefix}.tcp_pos_w"))
        chunks.append(_scalar(arm.get("gripper_opening"), f"{prefix}.gripper_opening", default=0.0))
    state = np.concatenate(chunks).astype(np.float32)
    if state.shape != (26,):
        raise ValueError(f"{episode_dir}/steps.jsonl:{line_number} simple state shape={state.shape}, expected (26,)")
    return state


def _state_from_step_for_layout(
    step: dict,
    episode_dir: Path,
    line_number: int,
    state_layout: str,
    episode_progress: float,
) -> np.ndarray:
    if state_layout == STATE_LAYOUT_HANDOFF_JOINT_TCP_POS_GRIPPER:
        return _joint_tcp_pos_gripper_state_from_step(step, episode_dir, line_number)
    if state_layout == STATE_LAYOUT_HANDOFF_JOINT_EE_RELPOSE:
        return _relpose_state_from_step(step, episode_dir, line_number)
    if state_layout == STATE_LAYOUT_HANDOFF_JOINT_EE_BIRELPOSE_TIME:
        return _birelpose_time_state_from_step(step, episode_dir, line_number, episode_progress)
    if state_layout == STATE_LAYOUT_HANDOFF_JOINT_EE_SUBTASK:
        return _state_from_step(step, episode_dir, line_number)
    raise ValueError(f"Unsupported state_layout={state_layout!r}")


def _state_source_for_timing(
    episode_steps: list[tuple[int, dict]],
    frame_index: int,
    state_timing: str,
) -> tuple[int, dict]:
    line_number, step = episode_steps[frame_index]
    if state_timing == STATE_TIMING_EXACT_PRE_ACTION:
        return line_number, _pre_action_step(step, f"steps.jsonl:{line_number}")
    if state_timing == STATE_TIMING_POST_ACTION:
        return episode_steps[frame_index]
    if state_timing == STATE_TIMING_PRE_ACTION_FROM_PREVIOUS_POST:
        source_index = max(frame_index - 1, 0)
        return episode_steps[source_index]
    raise ValueError(f"Unsupported state_timing={state_timing!r}")


def _pre_action_step(step: dict, source: str) -> dict:
    pre_arms = step.get("pre_arms")
    pre_images = step.get("pre_images")
    pre_cube = step.get("pre_cube", {})
    pre_targets = step.get("pre_targets", {})
    if not isinstance(pre_arms, dict) or not isinstance(pre_images, dict):
        raise KeyError(f"{source} missing pre_arms/pre_images required by state_timing={STATE_TIMING_EXACT_PRE_ACTION}")
    exact = dict(step)
    exact["arms"] = pre_arms
    exact["images"] = pre_images
    exact["cube_pos_w"] = pre_cube.get("pos_w", step.get("cube_pos_w"))
    exact["yellow_area_pos_w"] = pre_targets.get("yellow_area_pos_w", step.get("yellow_area_pos_w"))
    exact["red_area_pos_w"] = pre_targets.get("red_area_pos_w", step.get("red_area_pos_w"))
    return exact


def _images_for_step(step: dict, state_timing: str) -> dict:
    if state_timing == STATE_TIMING_EXACT_PRE_ACTION:
        images = step.get("pre_images")
        if not isinstance(images, dict):
            raise KeyError(f"Raw row missing pre_images required by state_timing={STATE_TIMING_EXACT_PRE_ACTION}")
        return images
    return step.get("images", {})


def _image_shapes_from_first_episode(episode_dir: Path, state_timing: str) -> dict[str, tuple[int, int, int]]:
    for _, step in _iter_steps(episode_dir):
        images = _images_for_step(step, state_timing)
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
            raise FileNotFoundError(f"Missing required raw handoff demo file: {path}")
    summary = _read_json(summary_path)
    if not include_failed and not summary.get("success", False):
        raise ValueError(f"{episode_dir} is not successful; pass --include-failed to convert it anyway")
    return summary


def _select_episodes(args: argparse.Namespace) -> tuple[list[Path], list[dict]]:
    all_selected_episodes: list[Path] = []
    all_summaries: list[dict] = []
    for episode_dir in _episode_dirs(args.raw_dir):
        try:
            summary = _validate_episode(episode_dir, args.include_failed)
        except ValueError as exc:
            if args.skip_failed and not args.include_failed and "is not successful" in str(exc):
                print(f"[SKIP] {episode_dir.name}: not successful", flush=True)
                continue
            raise
        all_selected_episodes.append(episode_dir)
        all_summaries.append(summary)

    if args.skip_episodes_count < 0:
        raise ValueError("--skip-episodes-count must be non-negative")
    selected_episodes = all_selected_episodes[args.skip_episodes_count :]
    summaries = all_summaries[args.skip_episodes_count :]
    if args.max_episodes is not None:
        selected_episodes = selected_episodes[: args.max_episodes]
        summaries = summaries[: args.max_episodes]

    if not selected_episodes:
        raise ValueError(f"No episodes selected from {args.raw_dir}")
    if args.require_episodes is not None and len(selected_episodes) < args.require_episodes:
        raise ValueError(
            f"Selected {len(selected_episodes)} episode(s), but --require-episodes={args.require_episodes}"
        )
    return selected_episodes, summaries


def _action_from_step(step: dict, episode_dir: Path, line_number: int, action_layout: str) -> np.ndarray:
    if action_layout == ACTION_LAYOUT_EE_DELTA_14:
        action = np.asarray(step.get("action"), dtype=np.float32)
        if action.shape != (14,):
            raise ValueError(f"{episode_dir}/steps.jsonl:{line_number} action has shape {action.shape}, expected (14,)")
        return action
    if action_layout == ACTION_LAYOUT_ABS_JOINT_POS_18:
        post_arms = step.get("post_arms")
        if not isinstance(post_arms, dict):
            raise KeyError(f"{episode_dir}/steps.jsonl:{line_number} missing post_arms for {action_layout}")
        chunks: list[np.ndarray] = []
        for arm_name in ("left", "right"):
            arm = post_arms.get(arm_name)
            if not isinstance(arm, dict):
                raise KeyError(f"{episode_dir}/steps.jsonl:{line_number} missing post_arms.{arm_name}")
            chunks.append(_vector(arm.get("joint_pos"), 9, f"post_arms.{arm_name}.joint_pos"))
        action = np.concatenate(chunks).astype(np.float32)
        if action.shape != (18,):
            raise ValueError(f"{episode_dir}/steps.jsonl:{line_number} action has shape {action.shape}, expected (18,)")
        return action
    raise ValueError(f"Unsupported action_layout={action_layout!r}")


def _frame_from_step(
    episode_dir: Path,
    line_number: int,
    step: dict,
    state_line_number: int,
    state_step: dict,
    task: str,
    image_keys: tuple[tuple[str, str], ...],
    state_layout: str,
    action_layout: str,
    episode_progress: float,
    state_timing: str,
    state_vector: np.ndarray | None = None,
) -> dict:
    action = _action_from_step(step, episode_dir, line_number, action_layout)

    frame = {
        "observation.state": state_vector
        if state_vector is not None
        else _state_from_step_for_layout(state_step, episode_dir, state_line_number, state_layout, episode_progress),
        "action": action,
        "subtask_index": _int_scalar(
            _subtask_id_from_phase(step.get("phase"), f"{episode_dir}/steps.jsonl:{line_number}"),
            "subtask_index",
        ),
        "task": task,
    }
    frame["subtask"] = SUBTASKS[int(frame["subtask_index"][0])]
    images = _images_for_step(step, state_timing)
    for raw_key, lerobot_key in image_keys:
        rel_path = images.get(raw_key)
        if rel_path is None:
            raise KeyError(
                f"{episode_dir}/steps.jsonl:{line_number} missing images.{raw_key}. "
                "Use RECORD_IMAGE_EVERY=1 when collecting LeRobot data."
            )
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


def _write_subtasks_metadata(output_dir: Path) -> None:
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"subtask_index": index, "subtask": text} for index, text in enumerate(SUBTASKS)]
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(meta_dir / "subtasks.parquet", index=False)
    except Exception:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(rows), meta_dir / "subtasks.parquet")


def _overwrite_fixed_image_stats(output_dir: Path, image_features: list[str], image_normalization: str) -> dict | None:
    if image_normalization == "dataset_stats":
        return None
    stats_path = output_dir / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    fixed_stats = image_stats_for_normalization(image_normalization)
    for feature_key in image_features:
        current = dict(stats.get(feature_key, {}))
        current.update(fixed_stats)
        stats[feature_key] = current
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return {"mode": image_normalization, "features": image_features}


def _array_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D stats array, got shape={values.shape}")
    return {
        "min": np.min(values, axis=0).astype(float).tolist(),
        "max": np.max(values, axis=0).astype(float).tolist(),
        "mean": np.mean(values, axis=0).astype(float).tolist(),
        "std": np.std(values, axis=0).astype(float).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).astype(float).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).astype(float).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).astype(float).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).astype(float).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(float).tolist(),
    }


def _relative_current_action_stats(episode_actions: list[np.ndarray], horizon: int) -> dict:
    if horizon <= 0:
        raise ValueError("--action-horizon must be positive")
    chunks: list[np.ndarray] = []
    offsets = np.arange(horizon, dtype=np.int64)
    for actions in episode_actions:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"Expected episode actions [T,14], got shape={actions.shape}")
        if actions.shape[0] == 0:
            continue
        indices = np.minimum(
            np.arange(actions.shape[0], dtype=np.int64)[:, None] + offsets[None, :],
            actions.shape[0] - 1,
        )
        delta_chunks = actions[indices]
        relative_chunks = action_delta_chunk_to_relative_current_np(delta_chunks)
        chunks.append(relative_chunks.reshape(-1, 14))
    if not chunks:
        raise ValueError("No converted actions available for action stats")
    return _array_stats(np.concatenate(chunks, axis=0))


def _overwrite_action_stats(
    output_dir: Path,
    episode_actions: list[np.ndarray],
    action_representation: str,
    action_horizon: int,
) -> dict | None:
    if action_representation in (ACTION_REPRESENTATION_DELTA_STEP, ACTION_REPRESENTATION_ABSOLUTE_JOINT_POS):
        return None
    if action_representation != ACTION_REPRESENTATION_RELATIVE_CURRENT_POSE_CHUNK:
        raise ValueError(f"Unsupported action_representation={action_representation!r}")
    stats_path = output_dir / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    before = dict(stats.get("action", {}))
    stats["action"] = _relative_current_action_stats(episode_actions, action_horizon)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return {
        "mode": action_representation,
        "horizon": int(action_horizon),
        "before_std": before.get("std"),
        "after_std": stats["action"]["std"],
        "count": stats["action"]["count"],
    }


def _manifest_from_summary(args: argparse.Namespace, summary: dict, state_qa: dict) -> dict:
    lengths = [int(item["frames"]) for item in summary["episodes"]]
    cube_size = args.cube_size_m
    if cube_size is None:
        for item in summary["episodes"]:
            raw_meta = item.get("raw_meta", {})
            if raw_meta.get("cube_size_m") is not None:
                cube_size = float(raw_meta["cube_size_m"])
                break
    return {
        "dataset_version": args.dataset_version,
        "split_name": args.split_name,
        "state_timing": args.state_timing,
        "image_timing": "pre_action" if args.state_timing == STATE_TIMING_EXACT_PRE_ACTION else "legacy",
        "action_timing": "action_from_pre_to_post" if args.state_timing == STATE_TIMING_EXACT_PRE_ACTION else "legacy",
        "action_representation": args.action_representation,
        "action_horizon_for_stats": int(args.action_horizon),
        "state_layout": args.state_layout,
        "state_dim": summary["state_dim"],
        "action_layout": args.action_layout,
        "action_dim": len(_action_names(args.action_layout)),
        "tcp_quat_format": "wxyz",
        "tcp_quat_normalized": True,
        "tcp_quat_canonicalization": "dominant_component_nonnegative",
        "cube_size_m": cube_size,
        "image_normalization": args.image_normalization,
        "image_augmentation": args.image_augmentation,
        "fps": args.fps,
        "episode_count": len(summary["episodes"]),
        "total_frames": summary["total_frames"],
        "episode_length_min": int(min(lengths)) if lengths else 0,
        "episode_length_max": int(max(lengths)) if lengths else 0,
        "episode_length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "episode_length_median": median_episode_length(lengths),
        "recommended_handoff_time_total_steps": median_episode_length(lengths),
        "video_frame_count_checked": not args.no_load_check,
        "state_qa": state_qa,
    }


def _convert(args: argparse.Namespace) -> dict:
    LeRobotDataset = _import_lerobot()
    if args.action_layout == ACTION_LAYOUT_ABS_JOINT_POS_18 and args.action_representation != ACTION_REPRESENTATION_ABSOLUTE_JOINT_POS:
        raise ValueError(
            f"--action-layout {ACTION_LAYOUT_ABS_JOINT_POS_18} requires "
            f"--action-representation {ACTION_REPRESENTATION_ABSOLUTE_JOINT_POS}"
        )
    if args.action_layout == ACTION_LAYOUT_EE_DELTA_14 and args.action_representation == ACTION_REPRESENTATION_ABSOLUTE_JOINT_POS:
        raise ValueError(
            f"--action-representation {ACTION_REPRESENTATION_ABSOLUTE_JOINT_POS} requires "
            f"--action-layout {ACTION_LAYOUT_ABS_JOINT_POS_18}"
        )
    if (
        args.state_timing == STATE_TIMING_PRE_ACTION_FROM_PREVIOUS_POST
        and args.state_layout == STATE_LAYOUT_HANDOFF_JOINT_EE_SUBTASK
    ):
        raise ValueError(
            "--state-timing previous_post_as_pre is only intended for arm-state-only layouts "
            f"({STATE_LAYOUT_HANDOFF_JOINT_EE_RELPOSE}, {STATE_LAYOUT_HANDOFF_JOINT_EE_BIRELPOSE_TIME}); "
            f"got --state-layout {args.state_layout}"
        )
    episodes, summaries = _select_episodes(args)
    image_shapes = _image_shapes_from_first_episode(episodes[0], args.state_timing)
    image_keys = tuple(
        (raw_key, lerobot_key) for raw_key, lerobot_key in ORDERED_IMAGE_KEY_MAP if lerobot_key in image_shapes
    )

    _prepare_output_dir(args.output_dir, args.overwrite)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output_dir,
        fps=args.fps,
        robot_type=args.robot_type,
        features=_features(image_shapes, args.state_layout, args.action_layout),
        use_videos=True,
        image_writer_threads=args.image_writer_threads,
        vcodec=args.vcodec,
    )

    converted: list[dict] = []
    converted_states: list[np.ndarray] = []
    converted_actions_by_episode: list[np.ndarray] = []
    try:
        for episode_index, episode_dir in enumerate(episodes):
            frame_count = 0
            raw_meta = _read_json(episode_dir / "meta.json")
            episode_steps = list(_iter_steps(episode_dir))
            episode_denominator = max(len(episode_steps) - 1, 1)
            episode_actions: list[np.ndarray] = []
            for frame_index, (line_number, step) in enumerate(episode_steps):
                episode_progress = frame_index / episode_denominator
                state_line_number, state_step = _state_source_for_timing(
                    episode_steps, frame_index, args.state_timing
                )
                state_vector = _state_from_step_for_layout(
                    state_step, episode_dir, state_line_number, args.state_layout, episode_progress
                )
                converted_states.append(state_vector)
                frame = _frame_from_step(
                    episode_dir,
                    line_number,
                    step,
                    state_line_number,
                    state_step,
                    args.task,
                    image_keys,
                    args.state_layout,
                    args.action_layout,
                    episode_progress,
                    args.state_timing,
                    state_vector,
                )
                episode_actions.append(np.asarray(frame["action"], dtype=np.float32).copy())
                dataset.add_frame(frame)
                frame_count += 1
            dataset.save_episode(parallel_encoding=True)
            converted_actions_by_episode.append(np.stack(episode_actions, axis=0))
            converted.append(
                {
                    "raw_episode": episode_dir.name,
                    "episode_index": episode_index,
                    "frames": frame_count,
                    "raw_summary": summaries[episode_index],
                    "raw_meta": raw_meta,
                }
            )
            print(f"[OK] converted {episode_dir.name}: {frame_count} frames", flush=True)
    finally:
        dataset.finalize()
    _write_subtasks_metadata(args.output_dir)
    image_stats_repair = None
    if not args.skip_image_stats_repair:
        from repair_lerobot_image_stats import repair_image_stats

        image_stats_repair = repair_image_stats(args.output_dir)
    fixed_image_stats = _overwrite_fixed_image_stats(
        args.output_dir,
        [lerobot_key for _, lerobot_key in image_keys],
        args.image_normalization,
    )
    action_stats_override = _overwrite_action_stats(
        args.output_dir,
        converted_actions_by_episode,
        args.action_representation,
        args.action_horizon,
    )
    state_qa = quat_continuity_report(converted_states, _state_names(args.state_layout))

    conversion_summary = {
        "raw_dir": str(args.raw_dir),
        "output_dir": str(args.output_dir),
        "repo_id": args.repo_id,
        "fps": args.fps,
        "task": args.task,
        "image_shape": list(next(iter(image_shapes.values()))),
        "image_shapes": {key: list(shape) for key, shape in image_shapes.items()},
        "image_features": [lerobot_key for _, lerobot_key in image_keys],
        "state_layout": args.state_layout,
        "state_timing": args.state_timing,
        "action_layout": args.action_layout,
        "action_representation": args.action_representation,
        "action_horizon_for_stats": int(args.action_horizon),
        "state_names": _state_names(args.state_layout),
        "state_dim": len(_state_names(args.state_layout)),
        "action_names": list(_action_names(args.action_layout)),
        "stage_to_id": STAGE_TO_ID,
        "active_arm_to_id": {str(key): value for key, value in ACTIVE_ARM_TO_ID.items()},
        "subtasks": [{"subtask_index": index, "subtask": text} for index, text in enumerate(SUBTASKS)],
        "subtask_name_to_id": SUBTASK_NAME_TO_ID,
        "phase_to_subtask_id": PHASE_TO_SUBTASK_ID,
        "subtask_active_arm_id": SUBTASK_ACTIVE_ARM_ID,
        "image_stats_repair": image_stats_repair,
        "fixed_image_stats": fixed_image_stats,
        "action_stats_override": action_stats_override,
        "state_qa": state_qa,
        "episodes": converted,
        "total_frames": int(sum(item["frames"] for item in converted)),
    }
    summary_path = args.output_dir / "conversion_summary.json"
    summary_path.write_text(json.dumps(conversion_summary, indent=2) + "\n", encoding="utf-8")
    manifest = _manifest_from_summary(args, conversion_summary, state_qa)
    write_manifest(args.output_dir, manifest)
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
        "subtask_index",
        "subtask",
        *summary["image_features"],
    }
    missing = required_keys.difference(sample)
    if missing:
        raise RuntimeError(f"Reloaded sample missing keys: {sorted(missing)}")
    print(f"[OK] reload check passed: {len(dataset)} frames, keys={sorted(sample.keys())}", flush=True)


def main() -> None:
    args = _parse_args()
    if not args.raw_dir.exists():
        raise FileNotFoundError(f"Raw handoff demo directory does not exist: {args.raw_dir}")
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
