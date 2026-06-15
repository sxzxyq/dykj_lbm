# Seven-DoF Pick-and-Place LBM Experiment

Isaac Sim experiment workspace for training an HF/LeRobot MultiTask DiT policy to pick up a cube from a white tabletop and place it on a red target area.

## Layout

```text
isaac_pick_place/
  envs/       Isaac Sim environment wrappers
  assets/     Local robot/object assets if needed
  scripts/    Data collection, conversion, training, eval, and visualization scripts
  configs/    Environment and training configs

outputs_pick_place/
  datasets/    Generated or converted LeRobot/HF datasets
  checkpoints/ Trained policy checkpoints
  eval_videos/ Evaluation rollouts and diagnostics
```

Design document:

```text
/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/SEVEN_DOF_GRASP_EXPERIMENT_PLAN.md
```
