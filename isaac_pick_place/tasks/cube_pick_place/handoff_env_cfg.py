"""Dual-arm handoff scene built on top of the red-target cube pick-place task."""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg
from isaaclab.utils import configclass

from . import mdp as pick_place_mdp
from .env_cfg import CubePickPlaceRedTargetFrankaIKRelVisuomotorEnvCfg
from .env_cfg import _cube_yaw_range_from_env
from .env_cfg import _parse_float_pair_env


def _handoff_cube_angle_range_from_env() -> tuple[float, float]:
    radians = os.environ.get("CUBE_ANGLE_RANGE_RAD")
    if radians:
        return _parse_float_pair_env("CUBE_ANGLE_RANGE_RAD", (-math.pi, math.pi))
    degrees = _parse_float_pair_env("CUBE_ANGLE_RANGE_DEG", (-180.0, 180.0))
    return math.radians(degrees[0]), math.radians(degrees[1])


def _xy_world_to_robot_root_xy(world_xy: tuple[float, float], robot_root_xy: tuple[float, float]) -> tuple[float, float]:
    return world_xy[0] - robot_root_xy[0], world_xy[1] - robot_root_xy[1]


PANDA_JOINT_POS_ACTION_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
]


@configclass
class CubeHandoffYellowRedDualFrankaIKRelVisuomotorEnvCfg(CubePickPlaceRedTargetFrankaIKRelVisuomotorEnvCfg):
    """Scene layout for a staged dual-arm handoff task.

    The right-side static arm starts with the blue cube in front of it, the yellow staging zone sits
    between the arms, and the red final target stays in front of the left-side actor arm.
    """

    def __post_init__(self):
        super().__post_init__()

        self.task_name = "cube_handoff_yellow_red_dual_franka"
        self.instruction = (
            "First place the blue cube on the yellow middle handoff area, then place it on the red target area."
        )
        self.actions.observer_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="observer_robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
        self.actions.observer_gripper_action = BinaryJointPositionActionCfg(
            asset_name="observer_robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # World-frame layout sketch:
        # observer/right arm at y=-0.30 -> cube starts near (0.50, -0.30)
        # middle staging area      -> yellow at (0.50,  0.00)
        # actor/left arm at y=0.30 -> red at    (0.50,  0.30)
        self.yellow_area_center_world_xy = (0.50, 0.00)
        self.red_area_center_world_xy = (0.50, 0.30)
        self.cube_init_center_world_xy = (0.50, -0.30)
        self.handoff_area_size_xy = (0.12, 0.12)

        actor_root_xy = (self.scene.robot.init_state.pos[0], self.scene.robot.init_state.pos[1])
        observer_root_xy = (self.scene.observer_robot.init_state.pos[0], self.scene.observer_robot.init_state.pos[1])
        red_target_xy_actor = _xy_world_to_robot_root_xy(self.red_area_center_world_xy, actor_root_xy)
        cube_center_xy_observer = _xy_world_to_robot_root_xy(self.cube_init_center_world_xy, observer_root_xy)

        self.target_area_center_xy = red_target_xy_actor
        self.cube_reset_target_xy = cube_center_xy_observer
        self.target_area_size_xy = (0.12, 0.12)
        self.observations.policy.target_area_position.params["target_xy"] = red_target_xy_actor

        self.scene.object.init_state.pos = (
            self.cube_init_center_world_xy[0],
            self.cube_init_center_world_xy[1],
            self.object_center_z,
        )
        self.scene.target_area.init_state.pos = (
            self.red_area_center_world_xy[0],
            self.red_area_center_world_xy[1],
            0.0015,
        )
        self.scene.target_area.spawn.semantic_tags = [("class", "red_target_area")]

        self.scene.yellow_area = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/YellowHandoffArea",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(self.yellow_area_center_world_xy[0], self.yellow_area_center_world_xy[1], 0.0016)
            ),
            spawn=sim_utils.CuboidCfg(
                size=(self.handoff_area_size_xy[0], self.handoff_area_size_xy[1], 0.001),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.85, 0.0),
                    roughness=0.5,
                    metallic=0.0,
                ),
                collision_props=CollisionPropertiesCfg(collision_enabled=False),
                semantic_tags=[("class", "yellow_handoff_area")],
            ),
        )

        self.events.reset_object_position = EventTerm(
            func=pick_place_mdp.reset_object_pose_around_target,
            mode="reset",
            params={
                "target_xy": cube_center_xy_observer,
                "radius_range": _parse_float_pair_env("CUBE_RADIUS_RANGE", (0.0, 0.10)),
                "angle_range": _handoff_cube_angle_range_from_env(),
                "object_center_z": self.object_center_z,
                "yaw_range": _cube_yaw_range_from_env(),
                "robot_cfg": SceneEntityCfg("observer_robot"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        for reward_name in ("object_goal_tracking", "object_goal_tracking_fine_grained"):
            getattr(self.rewards, reward_name).params["target_xy"] = red_target_xy_actor

        self.rewards.placed_on_target.params["target_xy"] = red_target_xy_actor
        self.rewards.placed_on_target.params["target_size_xy"] = self.target_area_size_xy
        # The scripted handoff collector owns success: red placement alone is not enough, because
        # the left arm still needs to retreat to park after releasing the cube.
        self.terminations.success = None


@configclass
class CubeHandoffYellowRedDualFrankaJointPosVisuomotorEnvCfg(CubeHandoffYellowRedDualFrankaIKRelVisuomotorEnvCfg):
    """Handoff variant with absolute joint-position actions for both Franka arms.

    The action is 18D and matches the abs-joint model target layout:
    left/actor joint_pos(9) followed by right/observer joint_pos(9).
    """

    def __post_init__(self):
        super().__post_init__()

        self.task_name = "cube_handoff_yellow_red_dual_franka_joint_pos"
        self.actions.arm_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=PANDA_JOINT_POS_ACTION_NAMES,
            scale=1.0,
            offset=0.0,
            preserve_order=True,
            use_default_offset=False,
        )
        self.actions.gripper_action = None
        self.actions.observer_arm_action = JointPositionActionCfg(
            asset_name="observer_robot",
            joint_names=PANDA_JOINT_POS_ACTION_NAMES,
            scale=1.0,
            offset=0.0,
            preserve_order=True,
            use_default_offset=False,
        )
        self.actions.observer_gripper_action = None


@configclass
class CubeHandoffYellowRedDualFrankaOppositeGlobalCamIKRelVisuomotorEnvCfg(
    CubeHandoffYellowRedDualFrankaIKRelVisuomotorEnvCfg
):
    """Handoff variant with the fixed global camera moved to the opposite side of the table."""

    def __post_init__(self):
        super().__post_init__()

        self.task_name = "cube_handoff_yellow_red_dual_franka_opposite_global_cam"
        self.scene.global_cam.offset = CameraCfg.OffsetCfg(
            pos=(0.97, 0.00, 1.00),
            rot=(0.69527, 0.12886, 0.12886, 0.69527),
            convention="opengl",
        )
