# 七自由度机械臂抓取并放置方块仿真实验设计 v0.2

> 目标：基于当前 PushT 模仿学习实验经验，设计一个可逐步落地的“七自由度机械臂 + 爪夹 + 手腕相机”方块抓取并放置仿真实验。任务是在白色桌面上从非目标区域抓起方块，并放到红色目标区域内。本文先定义实验方案、数据规格、模型接口、评估指标和风险检查项，暂不开始实验。

---

## 0. 已确认方案

本版文档根据你当前确认的信息定稿为以下默认方案：

1. **任务终点**：抓起方块后放到桌面红色目标区域内，成功条件以放置结果为准。
2. **机械臂资产**：直接使用 Isaac Sim 自带 7 自由度机械臂资产，优先使用内置 Franka/Panda 类机械臂；如果本机 Isaac Sim 版本资产路径不同，则选同等内置 7-DoF manipulator。
3. **末端执行器**：使用该机械臂自带夹爪，初期只需要开合控制，不引入复杂力控策略。
4. **相机配置**：actor 机械臂使用腕部 RGB 相机；另有一只暂时静止的 observer 机械臂，其腕部相机作为全局观察视角。
5. **控制接口**：策略输出末端笛卡尔空间相对增量，再由 IK/控制器映射到关节命令。
6. **sim-to-real**：暂时不考虑真实机械臂迁移，优先把 Isaac Sim 仿真实验跑通。
7. **模型路线**：使用当前 HF/LeRobot 文档里的 LBM / MultiTask DiT 路线，尽量复用现有项目经验。

### 0.1 当前实现状态

截至 `2026-06-09 11:06 CST`，第一版 Isaac Lab 环境和 scripted expert 已经落地到：

- Gym task id：`Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0`
- Env config：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/tasks/cube_pick_place/env_cfg.py`
- Task MDP terms：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/tasks/cube_pick_place/mdp.py`
- Smoke 脚本：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py`
- Scripted expert：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/scripted_pick_place.py`

已通过 headless smoke 验证：

- Franka/Panda relative IK 7D action 可创建、reset、step。
- 场景包含白色桌面、纯蓝方块、红色桌面目标区域、actor 腕部 RGB 相机。
- `Command Manager` 已移除 Lift 的随机 `object_pose` command，当前目标为固定红色区域。
- `policy` observation 为非拼接 dict，当前实际字段为：
  - `joint_pos: (1, 9)`
  - `joint_vel: (1, 9)`
  - `object_position: (1, 3)`
  - `actions: (1, 7)`
  - `target_area_position: (1, 3)`
  - `wrist_rgb: (1, 200, 200, 3), torch.uint8`
- Reward Manager 当前包含红色目标区域放置奖励 `placed_on_target`。
- Termination Manager 当前包含稳定放置成功项 `success`。
- Waypoint-based scripted expert 已在 `num_envs=1, seed=42, max_steps=1200` 下完成 `1/1` headless rollout success，成功触发 step 为 `952`。
- Scripted expert 已按 GUI 反馈调整为更低抓取 `grasp_z=0.015`，并在红色区域上方以 `release_z=0.085` 半空释放；此调整已通过 `py_compile`，但当前会话缺少 CUDA/GPU 和本地 Franka USD 缓存，仍需回到 GUI/Isaac 会话做运行时确认。
- Wrist camera 挂载在 `{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam`，即 `panda_hand` link 下。当前 actor eye-in-hand 外参为：`offset_pos=(0.085, 0.0, -0.0)`, `offset_rot_ros=(0.68301, -0.18301, -0.18301, 0.68301)`, `focal_length=18.0`, `clipping_range=(0.03, 2.0)`。
- 已新增第二只静止 Franka `observer_robot`，放在 `{ENV_REGEX_NS}/ObserverRobot`，当前与 actor 并排在桌子同一长边；actor 初始位姿显式设为 `pos=(0.0, 0.30, 0.0)`, `rot=(1.0, 0.0, 0.0, 0.0)`，并显式设置较高 reset view 的关节姿态：`joint2=-0.750`, `joint4=-2.650`, `joint6=3.058`；observer 初始位姿为 `pos=(0.0, -0.50, 0.0)`, `rot=(1.0, 0.0, 0.0, 0.0)`；桌子 x/y scale 扩大为 `(1.35, 1.80, 1.0)`。Observer arm 暂不进入 action/reward，只通过 `{ENV_REGEX_NS}/ObserverRobot/panda_hand/observer_wrist_cam` 提供全局观察视角，并配置为 `obs.policy.observer_wrist_rgb`。该新增项已通过 `py_compile` 和 runtime smoke，`obs.policy.observer_wrist_rgb: shape=(1, 200, 200, 3), dtype=torch.uint8`。
- 已禁用 Lift 基类的 `time_out` termination，避免长开 GUI 时每 `episode_length_s=30.0` 自动 reset 整个 scene、导致静止 observer arm 周期性跳动；当前仍保留 `success` 和 `object_dropping` termination。
- 为了调 observer camera 和双臂布局，已临时禁用 Lift 继承来的 `reset_object_position` event，cube reset pose 固定为 `pos=(0.50, 0.00, 0.0205)`；采 demo 或训练前需要恢复/重设随机 cube pose。
- 已确认 `wrist_cam` sensor RGB 在启动后可能出现相机画面斜转/未刷新问题。`smoke_lift_env.py --refresh-camera-xform` 会在 reset 后按当前 `CameraCfg.OffsetCfg` 重写 camera prim 的 local `translate/orient`，可使 `step_000001_wrist_cam.png` 恢复正常；后续 dataset/demo collector 应在每次 reset 后执行同类 refresh，并丢弃第一帧或从 refresh 后的下一帧开始写数据。
- 已采集第一批 raw scripted demo smoke 数据：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/raw_demos_smoke_v0_retry`。该批数据 `3/3` success，每条 `860` recorded steps，两路相机各 `860` 张 PNG，并包含 `steps.jsonl`、`meta.json`、`summary.json` 和 `manifest.json`。由于 cube pose 当前固定，该批仅用于验证采集链路和后续格式转换，不作为正式训练集。

当前成功条件第一版实现为：方块中心位于红色区域内、方块高度接近桌面、夹爪释放，并连续满足 `10` step 后触发 `success` termination。

---

## 1. 当前 PushT 实验给新任务的经验

当前项目里最重要的经验不是“训练更久”，而是**训练/评估接口必须完全一致**。

PushT 中已经验证的结论：

- 模型输入必须使用训练同款 observation 图，而不是可视化渲染图。
- `gym_pusht` 的 `env.render()` 会返回 680x680 可视化图，并可能叠加 action 标记；这曾经导致评估输入污染。
- 修正为训练同款 96x96 observation 图后，6-layer HF/LeRobot 模型从旧口径 `0/30` 提升到 `20/30 = 66.7%`（当前最佳推理配置：`max_steps=900, n_action_steps=12`）。
- 推理 `n_action_steps` 对闭环表现影响很大：过短会抖，过长会错过纠偏窗口。
- 评估必须固定 episode seed，并让 diffusion 采样也按 episode 重置 seed，否则不同 `max_steps` 不可公平比较。

迁移到七自由度抓取时，对应原则：

- 仿真相机图、数据集图、训练预处理、评估预处理必须逐像素语义一致。
- 状态坐标系、动作坐标系、单位、旋转表示必须写死并测试。
- 先做可观测、可复现、可诊断的简单抓取，再扩展随机化和复杂场景。
- 评估脚本要从第一天就记录 success、接触、抓取高度、末端误差、碰撞和超时，而不是只看视频。

---

## 2. 实验目标与范围

### 2.1 第一阶段目标

在 Isaac Sim 中训练一个视觉语言/视觉状态条件的策略，使 7-DoF 机械臂能从白色桌面的非目标区域抓起随机位置和随机姿态的方块，并放置到红色目标区域内。

第一阶段成功标准：

- 方块随机放置在白色桌面的非红色区域内。
- 红色目标区域固定在桌面上，作为明确可见的放置区域。
- 初始机械臂姿态固定或小范围随机。
- 策略根据手腕相机图像和机器人状态输出末端相对增量。
- 在限定步数内夹住方块、移动到红色区域上方、释放方块，并让方块稳定落在红色区域内。
- 同 seed 固定评估集上成功率达到可复现阈值。

### 2.2 暂不纳入第一阶段

- 多物体遮挡。
- 任意形状物体泛化。
- 真实机械臂部署。
- 双臂或灵巧手。
- 完整语言多任务。
- 多目标区域或复杂语言条件。

这些可以作为第二/三阶段扩展。

---

## 3. 暂定系统配置

### 3.1 硬件/仿真对象

| 模块 | 默认方案 | 备注 |
|---|---|---|
| 机械臂 | Isaac Sim 自带 7-DoF 机械臂，优先 Franka/Panda 类资产 | 具体资产路径随本机 Isaac Sim 版本确认 |
| 末端执行器 | 机械臂自带平行二指夹爪 | 开合控制范围 |
| 相机 | 单个手腕 RGB 相机 | 分辨率、视场角、安装外参 |
| 物体 | 立方体方块 | 尺寸、质量、摩擦、颜色 |
| 场景 | 白色桌面 + 红色目标区域 + 简单背景 | 目标区域尺寸和位置 |

### 3.2 仿真环境

默认选择：

- Isaac Sim：用于物理仿真、机器人资产、传感器、渲染和合成数据。
- Isaac Lab：如需批量并行环境、RL/IL 训练环境管理和任务封装，可作为上层框架。

设计原则：

- 初期不要过度追求照片级真实，先追求状态、相机、动作和成功判据一致。
- 先单环境 debug，再并行采集。
- 从一开始保留 replay/video，以便定位失败原因。

参考资料：

- NVIDIA Isaac Sim 官方页面：https://developer.nvidia.com/isaac-sim
- Isaac Sim NGC 容器页面：https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim
- Hugging Face LeRobot 文档：https://huggingface.co/docs/lerobot
- LeRobot Dataset v3 说明：https://github.com/huggingface/lerobot/blob/main/docs/source/lerobot-dataset-v3.mdx
- MultiTask DiT / LBM 相关 HF 文档：https://huggingface.co/docs/lerobot/multi_task_dit

---

## 4. 任务定义

### 4.1 任务名称

建议任务文本：

```text
Pick up the cube and place it on the red target area.
```

如果要对齐中文实验记录，也可以保留中文说明，但模型条件建议先用英文文本，和当前 CLIP text encoder 习惯一致。

### 4.2 初始状态随机化

第一阶段随机化范围建议保守：

- 方块位置：桌面中心附近非红色区域，例如 `x in [0.35, 0.55] m, y in [-0.20, 0.20] m`，并排除红色目标区。
- 红色目标区域：第一版固定在桌面一侧，例如中心 `target_xy = [0.50, 0.22] m`，尺寸 `12 cm x 12 cm`。
- 方块 yaw：`[-pi, pi]`，如果是立方体 yaw 不重要，但可保留。
- 方块尺寸：先固定，例如边长 `4 cm` 或 `5 cm`。
- 机械臂初始姿态：先固定 home pose；模型稳定后再小幅随机。
- 光照/材质：先固定；模型能抓后再随机化。

### 4.3 成功判定

第一阶段 success 建议同时满足：

- episode 结束时方块中心投影落在红色目标区域内。
- 方块底面稳定接触桌面，且高度接近桌面高度，不是被夹爪悬空带过目标区。
- 夹爪已释放或方块与夹爪无持续强接触。
- 方块在最近 `N` 步内位置稳定，没有滚出/滑出目标区。
- 机械臂没有发生严重碰撞或关节越界。

辅助指标：

- `placed_in_target`
- `final_cube_xy_error_to_target`
- `final_cube_target_margin`
- `max_cube_height`
- `grasp_contact_steps`
- `release_step`
- `eef_cube_distance_min`
- `episode_length`
- `collision_count`
- `timeout`

---

## 5. 输入/输出接口设计

### 5.1 观测输入

建议第一阶段输入：

```text
observation.image.wrist: RGB, 96x96 或 224x224
observation.eef_pose: 7D, [x, y, z, qx, qy, qz, qw]
observation.gripper: 1D, gripper opening
observation.joint_positions: 7D, optional
```

是否加入 joint state：

- 建议**保留 joint_positions**，至少作为可选输入，虽然用户侧主输入重点是手腕相机和末端姿态。
- 末端位姿能表达任务空间状态，但 joint state 能帮助模型知道冗余自由度、关节极限附近情况和 IK 状态。
- 如果模型输入维度压力不大，第一版就用 `eef_pose + gripper + joint_positions`。

历史帧建议：

- `n_obs_steps=2` 起步，沿用 PushT 经验。
- 手腕相机在接近物体时视角变化快，后续可试 `n_obs_steps=3/4`。
- 若控制频率较高，历史帧能帮助估计相对运动和接触前状态。

图像分辨率建议：

- 数据存储：优先 224x224 或 128x128。
- 如果沿用现有 CLIP ViT-B/16，模型内部通常会 resize/crop 到 224。
- 关键不是分辨率本身，而是训练/评估/可视化输入不要混用。

### 5.2 动作输出

你提出“输出相对末端执行器当前位姿的增量”，本文默认动作空间：

```text
action:
  delta_position: [dx, dy, dz]
  delta_rotation: rotation delta
  gripper_target: 1D
```

推荐第一版动作维度：

```text
7D action = [dx, dy, dz, d_rx, d_ry, d_rz, gripper]
```

其中：

- `dx,dy,dz`：单位米，限幅例如 `[-0.02, 0.02] m/step`。
- `d_rx,d_ry,d_rz`：轴角/rotation vector，小角度增量，限幅例如 `[-0.10, 0.10] rad/step`。
- `gripper`：建议用 target，而不是 delta。`-1=open, +1=close` 或实际开合宽度归一化。
- 执行方式：由当前末端位姿加上 Cartesian delta 得到下一步末端目标位姿，再通过 Isaac Sim/Isaac Lab 的 IK 或 operational space controller 转成关节命令。

坐标系需要明确：

| 方案 | 优点 | 风险 |
|---|---|---|
| base frame delta | 和世界/物体坐标更直观 | wrist camera 视角下空间关系不完全局部 |
| end-effector local delta | 更贴近手眼伺服 | 数据生成和执行时坐标变换必须严格一致 |

建议第一版：

- 平移 delta 用 base frame 表示，目标位姿仍然是“相对当前末端位姿”的增量更新，即 `target_ee_position = current_ee_position + delta_xyz_base`。
- 旋转 delta 先保留接口，但实际任务中可以强约束夹爪保持近似垂直向下，减少第一版学习难度。
- 文档和 dataset metadata 明确写入 `action.frame = base`，避免训练/评估坐标系错位。

### 5.3 控制频率与 action chunk

建议初始控制频率：

```text
control_hz = 10 Hz
```

与当前 PushT 实验一致，方便复用经验。

初始模型配置建议：

```text
n_obs_steps = 2
horizon = 32
n_action_steps = 8 或 12
max_episode_steps = 600-900
```

PushT 中 `n_action_steps=12` 在修正后表现最好，但机械臂接触任务可能不同：

- 接近阶段可以更长 chunk。
- 接触/闭合阶段可能需要更频繁重规划。

第一阶段建议直接做小网格：

```text
n_action_steps: 4, 8, 12
max_episode_steps: 600, 900
```

---

## 6. 数据集设计

### 6.1 数据格式

建议使用 Hugging Face / LeRobot dataset 格式，原因：

- 当前项目已经跑通 HF/LeRobot 风格训练与评估。
- 支持图像、状态、动作、任务文本、episode metadata。
- 后续可以上传到 HF Hub 或本地复用。

建议 features：

```yaml
observation.image.wrist:
  dtype: video
  shape: [H, W, 3]

observation.eef_pose:
  dtype: float32
  shape: [7]

observation.gripper:
  dtype: float32
  shape: [1]

observation.joint_positions:
  dtype: float32
  shape: [7]

action:
  dtype: float32
  shape: [7]

next.reward:
  dtype: float32
  shape: [1]

next.success:
  dtype: bool
  shape: [1]

task_index:
  dtype: int64
  shape: [1]
```

### 6.2 数据采集方式

建议按三阶段采集：

#### Phase A: Scripted Expert

用解析策略或 motion planning 生成演示：

1. 移动到方块上方 pre-grasp pose。
2. 对齐夹爪姿态。
3. 下降到抓取高度。
4. 闭合夹爪。
5. 抬升到安全高度。
6. 移动到红色目标区域上方。
7. 下降到释放高度。
8. 打开夹爪释放方块。
9. 抬起末端并等待方块稳定。

优点：

- 快速产生大量干净数据。
- 成功率高，便于验证模型接口和训练流程。

风险：

- 动作轨迹过于单一，模型可能对偏差恢复能力弱。

#### Phase B: Scripted + Noise / Recovery

加入扰动：

- pre-grasp 位置噪声。
- 下降时横向偏差。
- 抬升后移动到目标区时的路径扰动。
- 释放高度和释放时机扰动。
- 方块位置/材质/摩擦随机化。
- 失败恢复片段，例如夹偏后重新调整、移动过程中方块滑动后重新抓取。

优点：

- 提供闭环纠偏样本。
- 更接近真实策略需要的状态分布。

#### Phase C: Human Teleoperation / Hand Collection

如果 Isaac Sim 内有 VR、SpaceMouse、键鼠或其他遥操作接口，可采集仿真人类示教。

优点：

- 轨迹多样，有更多纠偏行为。
- 能补 scripted expert 过于规则的问题。

风险：

- 数据质量不稳定。
- 需要统一动作定义和控制频率。

### 6.3 数据数量建议

第一阶段数量建议：

| 阶段 | 目的 | episode 数 |
|---|---|---:|
| smoke | 验证 dataset/training/eval 闭环 | 20-50 |
| first train | 单方块固定外观，有限随机位置和固定目标区 | 500-1,000 |
| robust train | 加姿态、光照、摩擦、相机/释放扰动 | 2,000-5,000 |
| optional future | 若以后扩大任务，再加多目标区/多物体/更强随机化 | 暂不规划 |

当前 PushT 只有 206 episodes，6-layer 已能通过修正评估达到 66.7%，说明**几百条高质量数据可以跑出可诊断结果**；但机械臂 pick-and-place 比单纯推块或只抓起更长、更容易累积误差，建议正式第一版不要低于 500 条成功演示，最好从 1,000 条左右开始。

### 6.4 数据切分

建议固定：

```text
train: 80%
val: 10%
test/eval seeds: 10%
```

不要只随机 frame 切分；应按 episode/场景 seed 切分，避免同一初始状态泄漏。

---

## 7. 模型方案

### 7.1 初始模型

暂定使用 LBM / MultiTask DiT 路线：

- Vision encoder: CLIP ViT 或同类视觉 backbone。
- Text encoder: CLIP text，可先单任务固定 task string。
- State encoder: MLP for eef pose / joint state / gripper.
- Action decoder: diffusion / DiT 输出 action chunk。

建议从当前 HF/LeRobot `MultiTaskDiTPolicy` 适配，而不是从零写模型。

### 7.2 输入输出维度改动

当前 PushT：

```text
observation.image: [3, 96, 96]
observation.state: [2]
action: [2]
```

新抓取任务建议：

```text
observation.image.wrist: [3, H, W]
observation.state: [eef_pose(7), gripper(1), joints(7)] = [15]
action: [dx, dy, dz, d_rx, d_ry, d_rz, gripper_target] = [7]
```

如果要更保守：

```text
observation.state: [eef_position(3), eef_quat(4), gripper(1)] = [8]
action: [dx, dy, dz, gripper_target] = [4]
```

建议第一版是否真正释放旋转自由度取决于 Isaac Sim 自带机械臂/夹爪控制器的稳定性：

- 数据集和模型接口先按 7D 预留，便于后续扩展。
- 控制执行时可以先把 `d_rx,d_ry,d_rz` 限得很小，或用固定垂直抓取姿态生成 scripted demo。
- 如果第一版 IK/控制器调试复杂，可以临时切到 4D action 做 smoke test，但正式文档默认 7D Cartesian delta。

### 7.3 训练配置起点

参考当前最佳 PushT 配置：

```yaml
num_layers: 6
hidden_dim: 512
num_heads: 8
use_rope: true
horizon: 32
n_obs_steps: 2
n_action_steps: 8 or 12
batch_size: 64
steps: 30k smoke baseline; 100k+ for robust version
```

建议实验顺序：

1. 先固定 `num_layers=6`。
2. 先用 `batch_size=64` 确保不 OOM。
3. 不急着调 4/8 层，先把数据和评估接口做对。
4. 对 action chunk 做推理期 ablation：`4/8/12/16`。

---

## 8. 实验阶段计划

### Stage 0: 仿真环境最小闭环

交付物：

- Isaac Sim 场景：白色桌面、红色目标区域、方块、Isaac Sim 自带 7-DoF arm、夹爪、手腕相机。
- Python reset/step API。
- 能执行一个手写 scripted pick-and-place。
- 每个 step 输出 observation/action/reward/success。
- 能保存 episode replay/video。

通过标准：

- Scripted expert 成功率 >= 95%。
- 随机 100 个初始 cube pose 不崩溃，且不采样到红色目标区域内。
- 相机图与 dataset 图完全一致。

### Stage 1: LeRobot dataset smoke

交付物：

- 20-50 条 scripted demo。
- 转成 LeRobot/HF dataset。
- 可读取、可可视化、可 compute stats。
- 训练 100-1000 steps smoke。

通过标准：

- 模型能 overfit 小数据。
- 训练 loss 下降。
- eval 脚本能跑通并生成视频。

### Stage 2: 第一版 BC/LBM 训练

交付物：

- 200-500 条 demo。
- 6-layer MultiTask DiT baseline。
- 固定评估集 30/50 episodes。

通过标准：

- 成功率超过 scripted-free 随机策略。
- 能稳定把固定尺寸方块从非目标区域放到红色区域。
- 失败可分类：没看见、没对准、夹不住、搬运中掉落、没放进目标区、释放失败、碰撞、超时。

### Stage 3: 鲁棒性扩展

加入：

- 方块位置/尺寸/颜色随机。
- 光照/材质随机。
- 手腕相机外参微扰。
- 夹爪摩擦/方块质量随机。
- scripted recovery 数据。
- 红色目标区域位置/尺寸小幅随机，作为后续泛化实验。

通过标准：

- 固定评估集成功率提升。
- randomization eval 和 train-like eval 分开统计。

### Stage 4: 后续扩展（暂不执行）

如果以后重新考虑真实机械臂或更复杂任务，再加入：

- 真实相机标定。
- 真机安全限幅。
- 动作滤波。
- 少量真实示教 fine-tune。
- sim/real 图像分布对比。
- 多目标区域和语言条件，例如 "place the cube on the red/blue/green area"。

---

## 9. 评估协议

### 9.1 固定评估命令结构

所有评估都应该记录：

```text
checkpoint
dataset version
env version
robot asset version
control_hz
max_steps
n_action_steps
seed
episodes
success
avg_steps
max_cube_height
grasp_contact_steps
collision_count
placed_in_target
final_cube_xy_error_to_target
release_step
```

### 9.2 固定 seed

沿用 PushT 修正经验：

- 全局 seed 固定。
- 每个 episode 用 `seed + ep`。
- diffusion 采样也在 episode 开始前重新 seed。

### 9.3 评估集

建议维护三套：

```text
eval_easy: 30 episodes, train-like cube positions, fixed red target
eval_medium: 50 episodes, wider cube pose and lighting, fixed red target
eval_hard: 50 episodes, randomized friction/camera/material, optional target jitter
```

不要用训练随机 seed 当最终指标。

---

## 10. 风险清单与对应检查

| 风险 | PushT 中的对应教训 | 检查方法 |
|---|---|---|
| 模型输入图和训练图不一致 | `env.render()` 带 action 十字导致旧评估失真 | 保存训练帧和 eval 输入帧逐像素/统计对比 |
| 动作坐标系错 | 2D action 已验证需反归一化和 clip | 单步 action replay，验证 delta frame |
| 环境 TimeLimit 固定 | PushT 曾固定 300 step | env 创建时显式传 `max_episode_steps` |
| diffusion 采样不可复现 | max_steps 比较需 episode seed | episode 开始前重置 torch/numpy/random seed |
| 成功率虚高 | 只看 reward 不够 | success 需物理条件判定，不只 reward |
| scripted 数据过窄 | 模型无法 recovery | 加扰动和失败恢复样本 |
| 夹爪控制不稳定 | 接触任务比 PushT 更敏感 | 限速、滤波、接触/滑落日志 |
| wrist camera 盲区 | 接近后方块或红色目标区可能出画 | 调整手腕相机外参、FOV 和轨迹高度 |

---

## 11. 推荐初始配置

如果今天就开始做最小实验，我建议：

```yaml
robot:
  arm: Isaac Sim built-in 7-DoF arm, prefer Franka/Panda asset
  gripper: built-in parallel jaw gripper
  camera: wrist RGB

task:
  name: cube_pick_place_red_target
  success: cube center inside red target area and stable after release
  control_hz: 10
  max_episode_steps: 600 initially, then 900

observation:
  image:
    wrist: RGB 224x224 stored or 96x96 stored + model resize
  state:
    eef_pose: 7
    gripper: 1
    joint_positions: 7
  n_obs_steps: 2

action:
  representation: relative eef delta
  frame: base frame initially
  dim: 7
  fields: [dx, dy, dz, d_rx, d_ry, d_rz, gripper_target]
  limits:
    translation: +/- 0.02 m per step
    rotation: +/- 0.10 rad per step

model:
  policy: MultiTaskDiT / LBM-style diffusion transformer
  layers: 6
  hidden_dim: 512
  heads: 8
  horizon: 32
  n_action_steps: 8, 12 ablation
  batch_size: 64

dataset:
  smoke: 20-50 episodes
  first_train: 500-1000 episodes
  robust_train: 2000-5000 episodes
```

---

## 12. 初始目录建议

建议新增：

```text
isaac_pick_place/
  envs/
    cube_pick_place_env.py
  assets/
    robots/
    objects/
  scripts/
    collect_scripted_demos.py
    convert_to_lerobot.py
    train_pick_place_lbm.py
    eval_pick_place_policy.py
    live_viz_pick_place.py
  configs/
    cube_pick_place_env.yaml
    train_mtdp_pick_place.yaml

outputs_pick_place/
  datasets/
  checkpoints/
  eval_videos/
```

如果不想污染当前 PushT 工程，也可以新建 sibling repo：

```text
/home/ubuntu/Workspace/seven_dof_pick_place_lbm
```

---

## 13. 仍需实施时确认的细节

大方向已经确定，后面开始落地时只需要确认实现细节：

1. Scripted expert 的失败恢复策略，以及随机 cube pose 后每阶段容差是否需要自适应。
2. 手腕相机外参是否足以在抓取、搬运、释放阶段看到方块和红色区域。
3. 数据采集时是否存 200x200 原图，还是在 dataset 转换阶段 resize 到 96x96/224x224。
4. 是否在第一版 scripted expert 中把旋转 delta 强约束为近似 0，先用固定垂直抓取姿态降低难度。
5. 是否把当前 9 维 Franka joint observation 保留原样，还是导出时拆成 arm 7D + gripper 2D/1D。
6. Success 判定是否需要额外加入接触/速度稳定项，避免极端情况下滑过红区误判。
7. 数据采集器 reset 后必须 refresh `wrist_cam` / `observer_wrist_cam` local xform，并保存首帧 PNG 做质量检查，避免 RTX/Replicator camera transform 初始化刷新问题污染 dataset。

---

## 14. 下一步建议

下一步实现建议按这个顺序继续：

1. 用 GUI 命令肉眼检查调整后的 `scripted_pick_place.py`：确认 `observer_wrist_cam` 能看到蓝色方块、红色目标区和 actor arm；同时确认 actor `wrist_cam` 近场抓取视角、夹爪是否夹在方块中下部、红区上方是否在 `release_z` 半空松爪并让方块自由下落。
2. 写 raw demo 到 HF/LeRobot dataset 的转换脚本，并用 `raw_demos_smoke_v0_retry` 做格式 smoke。
3. 恢复或重写 cube pose randomization 后，基于 `scripted_pick_place.py` 采集 20-50 条 smoke train dataset。
4. 定义 HF/LeRobot dataset `features` JSON。
5. 增加多 episode 失败重试和更丰富 rollout metrics。
6. 写第一版 eval 脚本，固定 seed 测 scripted expert 成功率，再接模型训练。
