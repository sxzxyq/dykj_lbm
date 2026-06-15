"""Register the red-target cube pick-and-place Franka task."""

import gymnasium as gym


gym.register(
    id="Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "isaac_pick_place.tasks.cube_pick_place.env_cfg:"
            "CubePickPlaceRedTargetFrankaIKRelVisuomotorEnvCfg"
        ),
    },
    disable_env_checker=True,
)

