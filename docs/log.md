# 七自由度抓取放置实验日志

本文档用于维护 `/home/ubuntu/Workspace/seven_dof_pick_place_lbm` 的实验记录。

## 记录维护规则

1. **新记录写在 `实验记录` 章节最上方**，按时间倒序排列。
2. **所有可运行实验都要单独记录**，包括失败运行、烟测、数据采集、训练、评估和可视化检查。
3. **运行过命令就记录精确命令和路径**，避免后续无法复现。
4. **配置和结果必须同时记录**。只有结果没有配置不可比较；只有配置没有结果只能算备注。
5. **默认使用中文记录**。命令、路径、包名、日志原文和必要的英文报错可以保留原文。
6. **不要覆盖旧结论**。如果后续实验推翻了旧结论，新增一条记录说明原因。
7. **区分观察和解释**：
   - 观察：原始指标、报错、视频路径、checkpoint 路径。
   - 解释：为什么可能出现这个结果。
8. **认真运行时记录复现字段**：git 状态/commit、seed、dataset 版本、checkpoint 路径、Isaac Sim 版本、robot asset、max steps、action chunk、device。
9. **TODO 尽量放在对应记录里**；只有变成长期任务时再提升为项目级 TODO。

## 当前基线方案

- 任务：从白色桌面抓起方块，并放置到红色目标区域。
- 仿真器：Isaac Sim，使用内置 7 自由度机械臂，优先选择 Franka/Panda 类机械臂。
- 末端执行器：内置平行夹爪。
- 相机：单个腕部 RGB 相机。
- 观测：腕部图像 + 末端位姿 + 夹爪状态 + 可选关节位置。
- 动作：相对末端笛卡尔增量，通过 IK/controller 执行。
- 模型路线：HF/LeRobot MultiTask DiT / LBM-style policy。
- 真实机器人：暂不纳入当前范围。
- 设计文档：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/docs/SEVEN_DOF_GRASP_EXPERIMENT_PLAN.md`

## 目录结构

- `raw_demos/`：Isaac 原始 demonstration 录制结果。
- `lerobot_datasets/`：转换后的 LeRobot 数据集。
- `training_runs/`：HF/LeRobot MultiTask DiT 训练输出和 checkpoint。
- `eval_videos/`：策略评估输出、抽帧和 mp4。
- `camera_debug_runs/`：相机调试图像和相关报告。
- `smoke_runs/`：环境或任务 smoke test 产物。
- `reports/`：零散 txt/log 报告。
- `docs/`：设计文档和实验说明。

## 记录模板

````markdown
### YYYY-MM-DD HH:MM CST - Short Title

**Type:** design | env | data | train | eval | viz | debug

**Goal**
- 本次运行或变更要回答什么问题。

**Setup**
- 代码路径：
- 数据集：
- Checkpoint：
- 机器人资产：
- 相机：
- Seed：
- Device：
- 关键配置：

**Command**
```bash
# 这里记录精确命令
```

**Result**
- 指标：
- 产物：
- 错误：

**Interpretation**
- 这个结果可能意味着什么。

**Next**
- 下一步具体动作。
````

## 实验记录

### 2026-06-22 14:10 CST - Handoff BiRelPose Time 30k Ep1 Headless Eval

**Type:** eval

**Goal**
- 使用训练完成的 49D `handoff_joint_ee_birelpose_time` 模型跑 1 条 handoff 在线推理，并保存三路视频。

**Setup**
- Checkpoint：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_handoff_3cam_joint_ee_birelpose_time_100success_bs16acc4_30k/final_model`
- Task：`Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0`
- State mode：`handoff_joint_ee_birelpose_time`
- State dim：`49`
- Action dim：`14`
- 三路图像：`wrist_rgb`、`observer_wrist_rgb`、`global_rgb`
- `HANDOFF_TIME_TOTAL_STEPS=1845`
- `MAX_STEPS=3000`
- `RECORD_IMAGE_EVERY=5`

**Command**
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm

CHECKPOINT=/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_handoff_3cam_joint_ee_birelpose_time_100success_bs16acc4_30k/final_model \
TASK=Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0 \
TASK_TEXT="Right arm moves the blue cube to the yellow handoff area, then left arm moves it to the red target area." \
RUN_NAME=eval_handoff_birelpose_time_30k_ep1 \
HEADLESS=1 \
EPISODES=1 \
MAX_STEPS=3000 \
SAVE_VIDEO=1 \
RECORD_IMAGE_EVERY=5 \
LOG_EVERY=25 \
HANDOFF_TIME_TOTAL_STEPS=1845 \
bash isaac_pick_place/scripts/eval_pick_place_policy.sh
```

**Result**
- Isaac env 正常启动，policy 权重加载成功：
```text
missing_keys=0 unexpected_keys=0
state_dim=49 state_mode=handoff_joint_ee_birelpose_time
```
- 评估输出：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/eval_videos/eval_handoff_birelpose_time_30k_ep1`
- 结果：
```text
successes=0/1
success_rate=0.0
steps=3000
yellow_seen=False
red_success=False
```
- 三路视频均已生成：
```text
episode_000000/wrist_rgb.mp4
episode_000000/observer_wrist_rgb.mp4
episode_000000/global_rgb.mp4
```
- 每路抽帧 `600` 张。
- rollout 过程中 cube 从初始约 `(0.448, -0.284, 0.017)` 被轻微推到约 `(0.483, -0.279, 0.017)` 后基本停住，未进入黄色区域，也没有形成最终红区成功。

**Interpretation**
- 49D 完整动作模型可以在线加载和执行，但当前这条测试未学出有效第一阶段 handoff 行为。
- 相比 43D subtask/active-arm 版本，去掉显式阶段和 active-arm 提示后，单一时间进度 + 双向相对位姿不足以在这条随机初始状态上稳定启动右臂抓取。
- 后续需要看视频确认失败形态：是右臂接触/推走方块、夹爪时序错误，还是时间进度驱动下动作幅度衰减。

**Next**
- 先查看 `global_rgb.mp4` 和两路腕部视频，确认失败动作。
- 若失败是“没有明确进入右臂抓取阶段”，考虑恢复更弱的阶段提示，或把 `episode_progress` 改为非饱和/分段连续进度，而不是 1845 step 后固定 1.0。

### 2026-06-22 13:59 CST - Handoff BiRelPose Time 30k 推理前 GPU 阻塞检查

**Type:** eval | debug

**Goal**
- 使用训练完成的 49D `handoff_joint_ee_birelpose_time` 模型跑 handoff 在线推理并保存视频。

**Setup**
- Checkpoint：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_handoff_3cam_joint_ee_birelpose_time_100success_bs16acc4_30k/final_model`
- 预期任务：`Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0`
- 预期 state/action：`observation.state.shape=[49]`，`action.shape=[14]`
- 图像输入：`wrist_rgb`、`observer_wrist_rgb`、`global_rgb`

**Command**
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm

nvidia-smi

python - <<'PY'
from pathlib import Path
import json
ckpt = Path('/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_handoff_3cam_joint_ee_birelpose_time_100success_bs16acc4_30k/final_model')
config = json.loads((ckpt / 'config.json').read_text())
print('state_shape=', config['input_features']['observation.state']['shape'])
print('action_shape=', config['output_features']['action']['shape'])
print('image_features=', [k for k in config['input_features'] if k.startswith('observation.images.')])
PY

lsmod | rg '^nvidia|^nouveau' || true
dkms status 2>/dev/null | rg 'nvidia|NVIDIA' || true
lspci | rg -i 'nvidia|vga|3d' || true
cat /proc/driver/nvidia/version 2>/dev/null || true
find /proc/driver/nvidia -maxdepth 3 -type f -name information -print -exec cat {} \; 2>/dev/null || true
```

**Result**
- Checkpoint 文件存在，`config.json` 验证通过：
  - `state_shape=[49]`
  - `action_shape=[14]`
  - 三路图像输入均存在。
- `nvidia-smi` 失败：
```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```
- 内核模块存在：
```text
nvidia_uvm
nvidia_drm
nvidia_modeset
nvidia
```
- DKMS 显示 `nvidia/580.65.06` 已安装到 `6.5.0-18-generic` 和 `6.8.0-124-generic`。
- `/proc/driver/nvidia/version` 显示 `NVIDIA UNIX Open Kernel Module 580.65.06`。
- `/proc/driver/nvidia/gpus/0000:01:00.0/information` 能看到 GPU UUID，`GPU Excluded: No`。
- 因为 NVML 当前不可通信，未启动 Isaac eval；预计会在环境创建/CUDA 初始化阶段失败。

**Interpretation**
- 49D checkpoint 本身可读，模型配置与新推理脚本匹配。
- 当前阻塞点是系统 GPU/NVIDIA driver runtime 状态，不是模型、数据集或 eval 逻辑。
- 需要先恢复 `nvidia-smi` 正常，再跑 Isaac 推理。

**Next**
- 恢复 GPU 驱动后，使用同一 checkpoint 跑 49D handoff eval：
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm

CHECKPOINT=/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_handoff_3cam_joint_ee_birelpose_time_100success_bs16acc4_30k/final_model \
TASK=Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0 \
TASK_TEXT="Right arm moves the blue cube to the yellow handoff area, then left arm moves it to the red target area." \
RUN_NAME=eval_handoff_birelpose_time_30k_ep1 \
HEADLESS=1 \
EPISODES=1 \
MAX_STEPS=3000 \
SAVE_VIDEO=1 \
RECORD_IMAGE_EVERY=5 \
LOG_EVERY=25 \
HANDOFF_TIME_TOTAL_STEPS=1845 \
bash isaac_pick_place/scripts/eval_pick_place_policy.sh
```

### 2026-06-18 14:30 CST - Handoff BiRelPose Time 49D State Layout Implementation Smoke

**Type:** data | train | eval | debug

**Goal**
- 新增 handoff `handoff_joint_ee_birelpose_time` 观测版本：三路图像 + 两臂关节/末端状态 + 双向 TCP 相对位姿 + 单一连续时间进度。
- 不在 `observation.state` 中使用 `stage_id`、`active_arm_id`、`subtask_onehot`、`active_arm_onehot`、`cube_pos_w`、`yellow_area_pos_w`、`red_area_pos_w`。
- 输出继续保持完整 `14D` 双臂 action；在线推理时不启用 subtask scheduler 或 inactive-arm mask。

**Setup**
- 代码路径：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm`
- Raw 数据：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/raw_demos/raw_handoff_handoff_100_joint_ee_3cam_v1`
- Smoke LeRobot 数据集：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/lerobot_datasets/lerobot_handoff_birelpose_time_smoke`
- Smoke 训练输出：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_handoff_birelpose_time_smoke`
- Eval smoke 输出：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/eval_videos/eval_handoff_birelpose_time_smoke`
- 推理时间标量：`episode_progress = min(current_env_step / HANDOFF_TIME_TOTAL_STEPS, 1.0)`，默认 `HANDOFF_TIME_TOTAL_STEPS=1845`

**Command**
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm

python -m py_compile \
  isaac_pick_place/scripts/convert_handoff_raw_demos_to_lerobot.py \
  isaac_pick_place/scripts/train_hf_mtdp_smoke.py \
  isaac_pick_place/scripts/eval_pick_place_policy.py

bash -n isaac_pick_place/scripts/train_handoff_birelpose_time_256_mtdp.sh
bash -n isaac_pick_place/scripts/eval_pick_place_policy.sh
git diff --check -- \
  isaac_pick_place/scripts/convert_handoff_raw_demos_to_lerobot.py \
  isaac_pick_place/scripts/train_hf_mtdp_smoke.py \
  isaac_pick_place/scripts/eval_pick_place_policy.py \
  isaac_pick_place/scripts/eval_pick_place_policy.sh \
  isaac_pick_place/scripts/train_handoff_birelpose_time_256_mtdp.sh

/home/ubuntu/Workspace/multitask_dit_policy/.venv/bin/python \
  isaac_pick_place/scripts/convert_handoff_raw_demos_to_lerobot.py \
  --raw-dir /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/raw_demos/raw_handoff_handoff_100_joint_ee_3cam_v1 \
  --output-dir /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/lerobot_datasets/lerobot_handoff_birelpose_time_smoke \
  --repo-id local/seven_dof_pick_place_lbm_handoff_birelpose_time_smoke \
  --state-layout handoff_joint_ee_birelpose_time \
  --skip-failed \
  --max-episodes 2 \
  --require-episodes 2 \
  --overwrite

DATASET_DIR=/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/lerobot_datasets/lerobot_handoff_birelpose_time_smoke \
RUN_NAME=hf_mtdp_handoff_birelpose_time_smoke \
STEPS=2 \
BATCH_SIZE=1 \
GRAD_ACCUM_STEPS=1 \
TENSORBOARD=0 \
bash isaac_pick_place/scripts/train_handoff_birelpose_time_256_mtdp.sh

CHECKPOINT=/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_handoff_birelpose_time_smoke/final_model \
TASK=Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0 \
TASK_TEXT="Right arm moves the blue cube to the yellow handoff area, then left arm moves it to the red target area." \
RUN_NAME=eval_handoff_birelpose_time_smoke \
HEADLESS=1 \
EPISODES=1 \
MAX_STEPS=5 \
SAVE_VIDEO=0 \
RECORD_IMAGE_EVERY=0 \
LOG_EVERY=1 \
HANDOFF_TIME_TOTAL_STEPS=1845 \
bash isaac_pick_place/scripts/eval_pick_place_policy.sh
```

**Result**
- 静态检查通过：`py_compile`、`bash -n`、`git diff --check` 均无报错。
- 转换 smoke 成功：2 条 episode，`3634` 帧；reload check 通过。
- `meta/info.json` 中 `observation.state.shape == [49]`，包含：
  - `right_tcp_pos_in_left_tcp_frame.*`
  - `right_tcp_quat_in_left_tcp_frame.*`
  - `left_tcp_pos_in_right_tcp_frame.*`
  - `left_tcp_quat_in_right_tcp_frame.*`
  - `episode_progress.0`
- 49D state names 不包含 `stage`、`active_arm`、`subtask`、`cube_pos_w`、`yellow_area_pos_w`、`red_area_pos_w`。
- 抽查 episode 0：`episode_progress` 第一帧为 `0.0`，最后一帧为 `1.0`。
- 训练 smoke 成功：`STEPS=2`，生成 checkpoint `/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_handoff_birelpose_time_smoke/final_model`。
- Checkpoint config 验证：三路图像输入，`observation.state.shape=[49]`，`action.shape=[14]`。
- Eval smoke 加载 checkpoint 成功，权重兼容映射后 `missing_keys=0 unexpected_keys=0`；Isaac 环境创建阶段失败，原因是当前系统 CUDA/NVIDIA 驱动不可用：
```text
RuntimeError: No CUDA GPUs are available
NVML_ERROR_DRIVER_NOT_LOADED: NVIDIA driver is not loaded.
```

**Interpretation**
- 49D 数据布局、训练入口、checkpoint 配置和推理侧 state 识别已经打通。
- 当前 eval 失败是机器 GPU/驱动状态阻塞，不是 49D checkpoint 或权重加载问题。
- 这版模型不再依赖显式阶段/active-arm 提示，时间进度只提供一个连续标量；能否学会完整时序需要完整数据集训练后再用可用 GPU 评估。

**Next**
- GPU/驱动恢复后，先用 smoke checkpoint 跑 5 step eval 验证 49D 在线 state 构造可执行。
- 转换完整 100 条到 `lerobot_handoff_handoff_100_joint_ee_3cam_v1_birelpose_time` 后，使用 `train_handoff_birelpose_time_256_mtdp.sh` 跑正式训练。

### 2026-06-18 12:20 CST - Handoff RelPose 41D State Layout Implementation Smoke

**Type:** data | train | eval | debug

**Goal**
- 新增 handoff `handoff_joint_ee_relpose` 观测版本：三路图像 + 两臂关节/末端状态 + 右臂 TCP 相对左臂 TCP 位姿。
- 不在 `observation.state` 中使用 `stage_id`、`active_arm_id`、`subtask_onehot`、`active_arm_onehot`、`cube_pos_w`、`yellow_area_pos_w`、`red_area_pos_w`。
- 输出继续保持完整 `14D` 双臂 action。

**Setup**
- 代码路径：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm`
- Raw 数据：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/raw_demos/raw_handoff_handoff_100_joint_ee_3cam_v1`
- Smoke LeRobot 数据集：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/lerobot_datasets/lerobot_handoff_relpose_smoke`
- Smoke 训练输出：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_handoff_relpose_smoke`

**Command**
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm

python -m py_compile \
  isaac_pick_place/scripts/convert_handoff_raw_demos_to_lerobot.py \
  isaac_pick_place/scripts/train_hf_mtdp_smoke.py \
  isaac_pick_place/scripts/eval_pick_place_policy.py

bash -n isaac_pick_place/scripts/train_handoff_relpose_256_mtdp.sh
bash -n isaac_pick_place/scripts/eval_pick_place_policy.sh
git diff --check -- \
  isaac_pick_place/scripts/convert_handoff_raw_demos_to_lerobot.py \
  isaac_pick_place/scripts/train_hf_mtdp_smoke.py \
  isaac_pick_place/scripts/eval_pick_place_policy.py \
  isaac_pick_place/scripts/train_handoff_relpose_256_mtdp.sh

/home/ubuntu/Workspace/multitask_dit_policy/.venv/bin/python \
  isaac_pick_place/scripts/convert_handoff_raw_demos_to_lerobot.py \
  --raw-dir /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/raw_demos/raw_handoff_handoff_100_joint_ee_3cam_v1 \
  --output-dir /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/lerobot_datasets/lerobot_handoff_relpose_smoke \
  --repo-id local/seven_dof_pick_place_lbm_handoff_relpose_smoke \
  --state-layout handoff_joint_ee_relpose \
  --skip-failed \
  --max-episodes 2 \
  --require-episodes 2 \
  --overwrite

DATASET_DIR=/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/lerobot_datasets/lerobot_handoff_relpose_smoke \
RUN_NAME=hf_mtdp_handoff_relpose_smoke \
STEPS=2 \
BATCH_SIZE=1 \
GRAD_ACCUM_STEPS=1 \
TENSORBOARD=0 \
bash isaac_pick_place/scripts/train_handoff_relpose_256_mtdp.sh

CHECKPOINT=/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_handoff_relpose_smoke/final_model \
TASK=Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0 \
TASK_TEXT="Right arm moves the blue cube to the yellow handoff area, then left arm moves it to the red target area." \
RUN_NAME=eval_handoff_relpose_smoke \
HEADLESS=1 \
EPISODES=1 \
MAX_STEPS=5 \
SAVE_VIDEO=0 \
RECORD_IMAGE_EVERY=0 \
LOG_EVERY=1 \
bash isaac_pick_place/scripts/eval_pick_place_policy.sh
```

**Result**
- 静态检查通过：`py_compile`、`bash -n`、`git diff --check` 均无输出。
- 转换 smoke 成功：2 条 episode，共 `3634` frames。
- 新数据集 `observation.state.shape=[41]`，state names 不包含 `stage`、`active_arm`、`subtask`、`cube_pos_w`、`yellow_area_pos_w`、`red_area_pos_w`。
- 训练 smoke 成功：`STEPS=2`，loss 从 `1.166565` 到 `0.997926`，checkpoint 写入 `final_model`。
- Smoke checkpoint config 确认为三路图像 + `observation.state.shape=[41]` + `action.shape=[14]`。
- 在线 eval smoke：checkpoint 权重加载成功，`missing_keys=0 unexpected_keys=0`；但当前运行环境没有可用 CUDA GPU，Isaac 建环境时报 `RuntimeError: No CUDA GPUs are available`，未完成 env step。

**Interpretation**
- 41D relpose 数据转换和训练入口已打通。
- 在线评估代码已能识别并加载 41D checkpoint；实际闭环 step 还需在可用 CUDA/Isaac 环境下再跑一次。

**Next**
- 在 GPU 可用时重新跑 `eval_handoff_relpose_smoke`，确认 41D state 实时构造和 14D 完整动作闭环执行。
- 若 smoke eval 通过，再用完整 relpose 数据集启动 30k 训练。

### 2026-06-18 11:33 CST - Exp1 RIGHT_PLACE_YELLOW 300-Step Retreat Window 10-Episode Eval

**Type:** eval | debug

**Goal**
- 沿用 300 step 右臂撤离窗口设置，将 Exp1 handoff policy 再跑 10 条 headless 评估并保存三路视频。
- 重点观察：延长 `RIGHT_PLACE_YELLOW + ACTIVE_RIGHT` 是否能稳定让右臂自行撤离，并让左臂完成黄区到红区接力。

**Setup**
- 代码路径：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm`
- Checkpoint：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_handoff_3cam_joint_ee_subtask_100success_bs16acc4_30k/final_model`
- Task：`Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0`
- 输入：三路图像 + `handoff_joint_ee_subtask` 43D state
- `EPISODES=10`
- `MAX_STEPS=2600`
- `HANDOFF_RIGHT_RETREAT_STEPS=300`
- `HANDOFF_SCRIPTED_RIGHT_RETREAT=0`
- `HANDOFF_ACTIVE_ARM_MASK=1`
- `RECORD_IMAGE_EVERY=5`

**Command**
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm

CHECKPOINT=/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_handoff_3cam_joint_ee_subtask_100success_bs16acc4_30k/final_model \
TASK=Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0 \
TASK_TEXT="First place the blue cube on the yellow middle handoff area, then place it on the red target area." \
RUN_NAME=eval_handoff_subtask_30k_retreat300_10eps \
HEADLESS=1 \
EPISODES=10 \
MAX_STEPS=2600 \
SAVE_VIDEO=1 \
RECORD_IMAGE_EVERY=5 \
LOG_EVERY=100 \
HANDOFF_RIGHT_RETREAT_STEPS=300 \
HANDOFF_SCRIPTED_RIGHT_RETREAT=0 \
HANDOFF_ACTIVE_ARM_MASK=1 \
bash isaac_pick_place/scripts/eval_pick_place_policy.sh
```

**Result**
- 输出目录：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/eval_videos/eval_handoff_subtask_30k_retreat300_10eps`
- 总成功率：`2/10`，`success_rate=0.2`
- `yellow_seen=5/10`
- `red_success=2/10`
- 结束子任务统计：
  - `RIGHT_PLACE_YELLOW`：`7/10`
  - `RIGHT_PICK_CUBE`：`1/10`
  - `LEFT_PICK_FROM_YELLOW`：`1/10`
  - `DONE_HOLD`：`1/10`
- 每条 episode 均保存三路视频：`wrist_rgb.mp4`、`observer_wrist_rgb.mp4`、`global_rgb.mp4`。
- EP6 虽然环境判定成功，但最终仍停在 `RIGHT_PLACE_YELLOW`，`right_retreat_elapsed=0/300`；这更像右臂直接/误打误撞把方块送到了红区，不是完整 handoff。
- EP8 是真正完整流程成功：`RIGHT_PLACE_YELLOW -> WAIT_YELLOW_STABLE -> LEFT_PICK_FROM_YELLOW -> LEFT_PLACE_RED -> DONE_HOLD`，`right_retreat_elapsed=300/300`。
- EP9 完成右臂黄区放置和 300 step 撤离，但左臂在 `LEFT_PICK_FROM_YELLOW` 阶段把方块拨离黄区，未能抓起。
- EP10 曾达到 `right_retreat_elapsed=220/300`，随后方块被右臂继续带离黄区并抬高，计数归零，最终仍卡在 `RIGHT_PLACE_YELLOW`。

**Interpretation**
- 300 step 右臂撤离窗口确实可以让少数样本完成完整 handoff，但整体不稳定。
- 主要失败仍发生在右臂阶段：右臂经常在 `RIGHT_PLACE_YELLOW` 条件下继续夹/推/带动方块，导致黄区稳定计数无法累计到 300。
- 新出现的次要失败是左臂黄区抓取不稳：即使右臂完成撤离，左臂也可能把方块拨离黄区。
- 当前结果继续支持前一条结论：单纯延长 `RIGHT_PLACE_YELLOW` 窗口不是可靠修法，下一版应把 `RIGHT_RETREAT` 显式建成子任务/状态，或对右臂释放后的动作监督做更明确的阶段拆分。

**Next**
- 回看 EP8 成功视频，对比 EP9/EP10 的黄区释放和左臂抓取细节。
- 下一版数据/训练优先加入显式 `RIGHT_RETREAT` 子任务，而不是继续增加等待窗口。

### 2026-06-18 11:03 CST - Exp1 RIGHT_PLACE_YELLOW 300-Step Retreat Window Eval

**Type:** eval | debug

**Goal**
- 将在线 scheduler 的右臂撤离窗口从 `60` step 改为 `300` step。
- 不使用脚本右臂撤离，保留 active-arm action mask，测试模型在更长 `RIGHT_PLACE_YELLOW + ACTIVE_RIGHT` 条件下是否能自行完成右臂撤离并让左臂接力。

**Setup**
- 代码路径：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm`
- Checkpoint：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_handoff_3cam_joint_ee_subtask_100success_bs16acc4_30k/final_model`
- Task：`Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0`
- 输入：三路图像 + `handoff_joint_ee_subtask` 43D state
- `HANDOFF_RIGHT_RETREAT_STEPS=300`
- `HANDOFF_SCRIPTED_RIGHT_RETREAT=0`
- `HANDOFF_ACTIVE_ARM_MASK=1`

**Command**
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm

CHECKPOINT=/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_handoff_3cam_joint_ee_subtask_100success_bs16acc4_30k/final_model \
TASK=Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0 \
TASK_TEXT="First place the blue cube on the yellow middle handoff area, then place it on the red target area." \
RUN_NAME=eval_handoff_subtask_30k_retreat300_3eps \
HEADLESS=1 \
EPISODES=3 \
MAX_STEPS=2600 \
SAVE_VIDEO=1 \
RECORD_IMAGE_EVERY=5 \
LOG_EVERY=100 \
HANDOFF_RIGHT_RETREAT_STEPS=300 \
HANDOFF_SCRIPTED_RIGHT_RETREAT=0 \
HANDOFF_ACTIVE_ARM_MASK=1 \
bash isaac_pick_place/scripts/eval_pick_place_policy.sh
```

**Result**
- 输出目录：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/eval_videos/eval_handoff_subtask_30k_retreat300_3eps`
- 成功率：`1/3`
- 每条均保存三路视频：`wrist_rgb.mp4`、`observer_wrist_rgb.mp4`、`global_rgb.mp4`
- EP1：`yellow_seen=True`，但最终仍卡在 `RIGHT_PLACE_YELLOW`，`right_retreat_elapsed=0/300`；方块被右臂带/推离黄区，最终约 `(0.576, 0.315, 0.024)`。
- EP2：成功，`right_retreat_elapsed=300/300`，`LEFT_PICK_FROM_YELLOW -> LEFT_PLACE_RED` at step 1327，`LEFT_PLACE_RED -> DONE_HOLD` at step 1596。
- EP3：`yellow_seen=True`，但最终仍卡在 `RIGHT_PLACE_YELLOW`，`right_retreat_elapsed=0/300`；方块被右臂带/推离黄区，最终约 `(0.601, 0.313, 0.038)`。

**Interpretation**
- 300 step 撤离窗口可以在某些初始点上成功让右臂保持黄区稳定足够久，并切到左臂完成任务。
- 但该策略不稳定：EP1/EP3 中右臂在 `RIGHT_PLACE_YELLOW` 下继续拖动方块，导致黄区稳定计数无法累计到 300，最终一直不切左臂。
- 说明单纯延长粗子任务窗口不是可靠修法；`right_retreat` 仍应作为明确子任务/阶段建模，或使用更明确的撤离控制逻辑。

**Next**
- 回看 EP2 成功视频和 EP1/EP3 失败视频，对比右臂放黄后的末端高度和是否持续接触方块。
- 下一版优先拆出 `RIGHT_RETREAT` 子任务，而不是继续增大 `RIGHT_PLACE_YELLOW` 等待窗口。

### 2026-06-18 10:50 CST - Exp1 No Active-Arm Action Mask Eval

**Type:** eval | debug

**Goal**
- 关闭在线评估中的 active-arm action mask，让模型输出的 14D 双臂动作完整进入环境。
- 不使用脚本右臂撤离，测试模型是否可以自行让右臂撤离并完成 handoff。

**Setup**
- 代码路径：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm`
- Checkpoint：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_handoff_3cam_joint_ee_subtask_100success_bs16acc4_30k/final_model`
- Task：`Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0`
- 输入：三路图像 + `handoff_joint_ee_subtask` 43D state
- 改动：
  - `eval_pick_place_policy.py` 增加 `--disable-handoff-active-arm-mask`
  - `eval_pick_place_policy.sh` 增加 `HANDOFF_ACTIVE_ARM_MASK=0`
- 注意：active-arm onehot 仍保留在 state 中，因为 checkpoint 输入维度固定为 43D；本次只关闭 action mask。
- 脚本右臂撤离：关闭，`HANDOFF_SCRIPTED_RIGHT_RETREAT=0`

**Command**
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm

CHECKPOINT=/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_handoff_3cam_joint_ee_subtask_100success_bs16acc4_30k/final_model \
TASK=Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0 \
TASK_TEXT="First place the blue cube on the yellow middle handoff area, then place it on the red target area." \
RUN_NAME=eval_handoff_subtask_30k_no_active_mask_2eps \
HEADLESS=1 \
EPISODES=2 \
MAX_STEPS=2600 \
SAVE_VIDEO=1 \
RECORD_IMAGE_EVERY=5 \
LOG_EVERY=100 \
HANDOFF_ACTIVE_ARM_MASK=0 \
HANDOFF_SCRIPTED_RIGHT_RETREAT=0 \
HANDOFF_RIGHT_RETREAT_STEPS=60 \
bash isaac_pick_place/scripts/eval_pick_place_policy.sh
```

**Result**
- 输出目录：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/eval_videos/eval_handoff_subtask_30k_no_active_mask_2eps`
- 成功率：`0/2`
- `yellow_seen=True`：`0/2`
- 两条均最终卡在 `RIGHT_PICK_CUBE`，没有进入 `RIGHT_PLACE_YELLOW`。
- EP1 最终方块约在 `(0.592, -0.225, 0.017)`，未抓起。
- EP2 最终方块约在 `(0.537, -0.240, 0.017)`，未抓起。
- 每条均保存三路视频：`wrist_rgb.mp4`、`observer_wrist_rgb.mp4`、`global_rgb.mp4`。

**Interpretation**
- 关闭 active-arm action mask 后表现更差，两个 episode 都失败在右臂抓取阶段，无法测试后续“右臂是否自行撤离”。
- 这说明 action mask 在当前模型上不是单纯限制能力，反而可能在早期阶段抑制了另一只手/无关 action 的干扰。
- 仅靠关闭 active-arm mask 不能解决右臂撤离问题；更可信的方向仍是把 `right_retreat` 单独建模为明确子任务，或使用更细粒度阶段条件。

**Next**
- 回看 `global_rgb.mp4`，确认关闭 mask 后左臂或右臂多余动作如何干扰右抓。
- 下一版训练/转换优先考虑把 `right_retreat` 从 `RIGHT_PLACE_YELLOW` 中拆出来。

### 2026-06-18 10:36 CST - Exp1 Scripted Right Retreat Oracle Check

**Type:** eval | debug

**Goal**
- 验证用户在视频中观察到的失败原因：右臂放置后没有抬起撤离，挡住左臂接力抓取。
- 使用调试开关 `HANDOFF_SCRIPTED_RIGHT_RETREAT=1`，在黄区释放后脚本化控制右臂抬高撤离，再把控制权切给左臂。

**Setup**
- 代码路径：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm`
- Checkpoint：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_handoff_3cam_joint_ee_subtask_100success_bs16acc4_30k/final_model`
- Task：`Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0`
- 输入：三路图像 + `handoff_joint_ee_subtask` 43D state
- 调试干预：`HANDOFF_SCRIPTED_RIGHT_RETREAT=1`
- 右臂撤离上限：`HANDOFF_RIGHT_RETREAT_STEPS=120`
- 注意：该运行不是纯策略评估，右臂撤离阶段由脚本覆盖右臂 action。

**Command**
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm

CHECKPOINT=/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_handoff_3cam_joint_ee_subtask_100success_bs16acc4_30k/final_model \
TASK=Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0 \
TASK_TEXT="First place the blue cube on the yellow middle handoff area, then place it on the red target area." \
RUN_NAME=eval_handoff_subtask_30k_scripted_right_retreat_ep1 \
HEADLESS=1 \
EPISODES=1 \
MAX_STEPS=2600 \
SAVE_VIDEO=1 \
RECORD_IMAGE_EVERY=5 \
LOG_EVERY=100 \
HANDOFF_RIGHT_RETREAT_STEPS=120 \
HANDOFF_SCRIPTED_RIGHT_RETREAT=1 \
bash isaac_pick_place/scripts/eval_pick_place_policy.sh
```

**Result**
- 输出目录：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/eval_videos/eval_handoff_subtask_30k_scripted_right_retreat_ep1`
- 三路视频：
  - `episode_000000/wrist_rgb.mp4`
  - `episode_000000/observer_wrist_rgb.mp4`
  - `episode_000000/global_rgb.mp4`
- 成功率：`1/1`
- 最终阶段：`DONE_HOLD`
- `yellow_seen=True`
- `red_success=True`
- `right_retreat_elapsed=120/120`
- 阶段关键点：
  - `RIGHT_PICK_CUBE -> RIGHT_PLACE_YELLOW` at step 381
  - `RIGHT_PLACE_YELLOW -> WAIT_YELLOW_STABLE` at step 872
  - `WAIT_YELLOW_STABLE -> LEFT_PICK_FROM_YELLOW` at step 892
  - `LEFT_PICK_FROM_YELLOW -> LEFT_PLACE_RED` at step 1851
  - `LEFT_PLACE_RED -> DONE_HOLD` at step 2144

**Interpretation**
- 用户观察成立：纯策略失败的关键原因之一是右臂放黄后没有抬起撤离，阻挡左臂。
- 在仅脚本化右臂撤离的情况下，同一个 30k Exp1 模型可以完成左臂接力抓取和放红区，说明左臂策略并非完全不会，主要被右臂低位遮挡/碰撞破坏。
- 当前粗子任务设计把 `right_release_on_yellow` 和 `right_retreat` 都压在 `RIGHT_PLACE_YELLOW` 中，模型没有可靠学出“释放后抬起撤离”的时序。后续更干净的方案是把 `RIGHT_RETREAT` 独立成子任务类别，重新转换并训练；短期可用脚本化撤离做 oracle/debug。

**Next**
- 回看 `global_rgb.mp4` 对比纯策略和脚本撤离版本，确认右臂抬起后左臂路径恢复。
- 下一版训练建议把 `right_retreat` 单独作为子任务，或在 state/task condition 中显式加入 release/retreat 阶段，避免粗子任务内部多模态动作混淆。

### 2026-06-18 10:30 CST - Exp1 Eval Scheduler Right Retreat Window Test

**Type:** eval | debug

**Goal**
- 修复在线推理 scheduler 在黄色区稳定后过早切到左臂，导致右臂冻结在放置姿态的问题。
- 快速测试 `RIGHT_PLACE_YELLOW` 额外保持一段右臂撤离窗口后，左臂接力是否改善。

**Setup**
- 代码路径：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm`
- Checkpoint：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_handoff_3cam_joint_ee_subtask_100success_bs16acc4_30k/final_model`
- Task：`Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0`
- 改动：
  - `eval_pick_place_policy.py` 增加 `--handoff-right-retreat-steps`
  - `eval_pick_place_policy.sh` 增加 `HANDOFF_RIGHT_RETREAT_STEPS`
  - 在线 scheduler 在检测到黄色区释放后继续保持 `RIGHT_PLACE_YELLOW + ACTIVE_RIGHT`，默认可调；本次有效测试使用 `60` step。

**Command**
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm

CHECKPOINT=/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_handoff_3cam_joint_ee_subtask_100success_bs16acc4_30k/final_model \
TASK=Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0 \
TASK_TEXT="First place the blue cube on the yellow middle handoff area, then place it on the red target area." \
RUN_NAME=eval_handoff_subtask_30k_retreat60_ep1 \
HEADLESS=1 \
EPISODES=1 \
MAX_STEPS=2600 \
SAVE_VIDEO=1 \
RECORD_IMAGE_EVERY=5 \
LOG_EVERY=100 \
HANDOFF_RIGHT_RETREAT_STEPS=60 \
bash isaac_pick_place/scripts/eval_pick_place_policy.sh
```

**Result**
- 输出目录：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/eval_videos/eval_handoff_subtask_30k_retreat60_ep1`
- 三路视频已生成：
  - `episode_000000/wrist_rgb.mp4`
  - `episode_000000/observer_wrist_rgb.mp4`
  - `episode_000000/global_rgb.mp4`
- 成功率：`0/1`
- `yellow_seen=True`
- `red_success=False`
- `right_retreat_elapsed=60/60`
- 阶段切换：
  - `RIGHT_PICK_CUBE -> RIGHT_PLACE_YELLOW` at step 309
  - `RIGHT_PLACE_YELLOW -> WAIT_YELLOW_STABLE` at step 745
  - `WAIT_YELLOW_STABLE -> LEFT_PICK_FROM_YELLOW` at step 765
- 方块在 step 800 仍在黄色区附近：约 `(0.530, 0.019, 0.017)`。
- 左臂阶段未抓起方块，反而逐步把方块推到约 `(0.645, -0.172, 0.017)`。

**Interpretation**
- 原先“右臂刚放完就被冻结”的 scheduler 问题已被修正：本次在线推理确实让 `RIGHT_PLACE_YELLOW` 多跑了 60 step，右臂撤离窗口生效。
- 但完整任务仍失败，说明“右臂遮挡”不是唯一瓶颈；当前左臂接力抓取本身仍不稳，可能存在左臂接近方向、黄色区抓取姿态或训练分布不足的问题。
- `HANDOFF_RIGHT_RETREAT_STEPS=200` 曾短暂试跑，但窗口太长，右臂继续在 `RIGHT_PLACE_YELLOW` 下会把方块推出黄区，因此中止，未作为正式评估结果。

**Next**
- 回看 `eval_handoff_subtask_30k_retreat60_ep1/episode_000000/global_rgb.mp4`，重点确认右臂撤离距离和左臂夹取时的碰撞/对准情况。
- 后续可能需要单独强化左臂从黄色区抓取，或把 `right_retreat` 单独作为子任务类别重新转换/训练。

### 2026-06-18 10:08 CST - Exp1 Subtask Handoff 30k Headless Eval

**Type:** eval

**Goal**
- 测试 Exp1 子任务条件输入模型 30k `final_model` 在双 Franka handoff 任务上的在线推理表现。
- 重点观察 `subtask_onehot + active_arm_onehot` 是否改善左右臂时序和第一阶段黄色中转放置。

**Setup**
- 代码路径：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm`
- Checkpoint：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_handoff_3cam_joint_ee_subtask_100success_bs16acc4_30k/final_model`
- Task：`Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0`
- Task text：`First place the blue cube on the yellow middle handoff area, then place it on the red target area.`
- 输入：三路图像 + `handoff_joint_ee_subtask` 43D state
- 动作：14D 双臂动作，`n_action_steps=8`
- Seed：`2000`
- Device：`cuda:0`
- Episodes：`5`
- Max steps：`2600`

**Command**
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm

CHECKPOINT=/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_handoff_3cam_joint_ee_subtask_100success_bs16acc4_30k/final_model \
TASK=Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0 \
TASK_TEXT="First place the blue cube on the yellow middle handoff area, then place it on the red target area." \
RUN_NAME=eval_handoff_subtask_30k_5eps_headless_video_retry \
EPISODES=5 \
MAX_STEPS=2600 \
HEADLESS=1 \
SAVE_VIDEO=1 \
RECORD_IMAGE_EVERY=5 \
LOG_EVERY=100 \
bash isaac_pick_place/scripts/eval_pick_place_policy.sh
```

**Result**
- 输出目录：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/eval_videos/eval_handoff_subtask_30k_5eps_headless_video_retry`
- 总成功率：`0/5`
- `yellow_seen=True`：`3/5`
- `yellow_success=True`：`2/5`
- `red_success=True`：`0/5`
- 每个 episode 均保存三路视频：`wrist_rgb.mp4`、`observer_wrist_rgb.mp4`、`global_rgb.mp4`
- 分 episode 结果：
  - EP1：到过黄色区，最终卡在 `LEFT_PICK_FROM_YELLOW`，左臂抓取失败并把方块推偏到约 `(0.523, -0.085, 0.017)`。
  - EP2：右臂抓起后放置过冲，最终卡在 `RIGHT_PLACE_YELLOW`，没有黄色稳定。
  - EP3：黄色放置成功，最终卡在 `LEFT_PICK_FROM_YELLOW`，方块稳定在黄色区附近但左臂未抓起。
  - EP4：初始点偏右下，右臂抓取失败，最终卡在 `RIGHT_PICK_CUBE`。
  - EP5：黄色放置成功，最终卡在 `LEFT_PICK_FROM_YELLOW`，左臂未抓起。

**Interpretation**
- 相比未加子任务条件的 30k 模型，Exp1 明显改善了阶段顺序：右臂先执行、到黄后才切左臂，未再出现左臂一开始乱动到黄色区的主要问题。
- 当前主要瓶颈已经转移到左臂接力抓取：即使方块准确稳定地放在黄色区，左臂也常常无法形成有效抓取。
- 右臂也仍有边界泛化问题：部分初始点会在抓取或放黄阶段失败，但整体第一阶段已能成功触发。

**Next**
- 优先回看 EP3/EP5 的 `global_rgb.mp4` 和 `wrist_rgb.mp4`，确认左臂在黄色区抓取失败时的末端位置、夹爪闭合和视野遮挡。
- 下一轮可考虑：单独强化左臂从黄色区到红区的数据比例，或先做阶段化训练/评估，把右臂到黄和左臂到红拆开定位。

### 2026-06-17 16:41 CST - Handoff Exp 1 子任务条件输入实现与数据集转换

**Type:** data | train-smoke | eval-smoke | debug

**Goal**
- 针对 30k handoff baseline 仍然失败的问题，验证“粗粒度子任务 + 激活机械臂”条件输入是否可以降低双臂阶段歧义。
- 只做 Exp 1：模型输入增加 `subtask_onehot(6)` 和 `active_arm_onehot(3)`，不加入 cube/yellow/red oracle position。

**Setup**
- 代码路径：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm`
- Baseline checkpoint：
  - `/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_handoff_3cam_joint_ee_resume_from_14569_to_30k_bs16acc4/final_model`
- Baseline eval 输出：
  - `/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/eval_videos/eval_handoff_30k_resume_headless_video`
- Raw handoff 数据：
  - `/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/raw_demos/raw_handoff_handoff_100_joint_ee_3cam_v1`
- 新 LeRobot 数据集：
  - `/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/lerobot_datasets/lerobot_handoff_handoff_100_joint_ee_3cam_v1_subtask`
- 任务：`Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0`
- 相机：`wrist_rgb`、`observer_wrist_rgb`、`global_rgb`
- 动作：14D dual-arm action，保持不变

**Command**
```bash
# 30k baseline headless 视频评估
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm

CHECKPOINT=/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_handoff_3cam_joint_ee_resume_from_14569_to_30k_bs16acc4/final_model \
TASK=Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0 \
TASK_TEXT="First place the blue cube on the yellow middle handoff area, then place it on the red target area." \
RUN_NAME=eval_handoff_30k_resume_headless_video \
EPISODES=10 \
MAX_STEPS=2600 \
HEADLESS=1 \
SAVE_VIDEO=1 \
RECORD_IMAGE_EVERY=5 \
LOG_EVERY=100 \
bash isaac_pick_place/scripts/eval_pick_place_policy.sh

# 重新转换 handoff 数据集，加入 coarse subtask 和 active arm 条件
/home/ubuntu/Workspace/multitask_dit_policy/.venv/bin/python \
  isaac_pick_place/scripts/convert_handoff_raw_demos_to_lerobot.py \
  --raw-dir /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/raw_demos/raw_handoff_handoff_100_joint_ee_3cam_v1 \
  --output-dir /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/lerobot_datasets/lerobot_handoff_handoff_100_joint_ee_3cam_v1_subtask \
  --repo-id local/seven_dof_pick_place_lbm_handoff_handoff_100_joint_ee_3cam_v1_subtask \
  --overwrite

# 43D subtask state 训练 smoke
DATASET_DIR=/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/lerobot_datasets/lerobot_handoff_subtask_convert_smoke \
RUN_NAME=handoff_subtask_train_smoke \
STATE_MODE=handoff_joint_ee_subtask \
STEPS=2 \
SAVE_FREQ=0 \
BATCH_SIZE=1 \
GRAD_ACCUM_STEPS=1 \
TENSORBOARD=0 \
NUM_WORKERS=0 \
bash isaac_pick_place/scripts/train_handoff_256_mtdp.sh

# 43D subtask checkpoint Isaac eval smoke
CHECKPOINT=/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/handoff_subtask_train_smoke/final_model \
TASK=Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0 \
TASK_TEXT="First place the blue cube on the yellow middle handoff area, then place it on the red target area." \
RUN_NAME=eval_handoff_subtask_smoke_10steps \
EPISODES=1 \
MAX_STEPS=10 \
HEADLESS=1 \
SAVE_VIDEO=0 \
RECORD_IMAGE_EVERY=0 \
LOG_EVERY=5 \
bash isaac_pick_place/scripts/eval_pick_place_policy.sh
```

**Result**
- 30k baseline 评估手动跑完前 3 个完整 episode 后中断剩余重复失败样本：
  - `final_success=0/3`
  - `yellow_seen=0/3`
  - `red_success=0/3`
  - 每个完整 episode 跑满 `2600` steps
  - 三路视频已保存到 `eval_handoff_30k_resume_headless_video/episode_000000..000002`
- 失败模式：
  - 第 1 轮右臂只轻微推动方块，未抓起。
  - 第 2、3 轮右臂把方块向远离任务区方向推走，后半段有时带起方块，但已偏离黄色区域。
  - 左臂阶段基本还没有进入有效评估，主要瓶颈仍在右臂第一阶段。
- 已实现 Exp 1 数据和推理改造：
  - 从 raw `phase` 映射 6 个 coarse subtask：
    - `RIGHT_PICK_CUBE`
    - `RIGHT_PLACE_YELLOW`
    - `WAIT_YELLOW_STABLE`
    - `LEFT_PICK_FROM_YELLOW`
    - `LEFT_PLACE_RED`
    - `DONE_HOLD`
  - active arm 编码：
    - `ACTIVE_LEFT`
    - `ACTIVE_RIGHT`
    - `ACTIVE_NONE`
  - converter 保留原 45D full state 前缀，并在末尾追加：
    - `subtask_onehot(6)`
    - `active_arm_onehot(3)`
  - 新 LeRobot full state 为 `54D`。
  - 训练 Exp 1 实际使用 `43D`：
    - 前 `34D` 左右臂关节/TCP/夹爪状态
    - 末尾 `9D` 子任务和激活机械臂 one-hot
  - eval 对 `43D` checkpoint 自动启用 oracle coarse scheduler、inactive arm mask、subtask change 时 `policy.reset()`。
- 新数据集转换检查：
  - episodes：`100`
  - total frames：`184675`
  - `observation.state`：`54D`
  - action：`14D`
  - 三路图像 feature 均存在
  - `meta/subtasks.parquet` 存在
  - 抽查首帧/中间帧/末帧：
    - `subtask_onehot.sum() = 1.0`
    - `active_arm_onehot.sum() = 1.0`
- 训练 smoke：
  - `STATE_MODE=handoff_joint_ee_subtask` 跑通，checkpoint config 中 `observation.state.shape=[43]`
  - 旧 `STATE_MODE=handoff_joint_ee` 兼容 smoke 跑通，checkpoint config 中 `observation.state.shape=[34]`
- eval smoke：
  - 43D smoke checkpoint 在 Isaac headless 中跑通 10 steps
  - 日志中确认 scheduler 输出：
```text
subtask=RIGHT_PICK_CUBE active_arm=ACTIVE_RIGHT
```

**Checks**
```bash
python -m py_compile \
  isaac_pick_place/scripts/convert_handoff_raw_demos_to_lerobot.py \
  isaac_pick_place/scripts/train_hf_mtdp_smoke.py \
  isaac_pick_place/scripts/eval_pick_place_policy.py

bash -n isaac_pick_place/scripts/train_handoff_256_mtdp.sh
bash -n isaac_pick_place/scripts/eval_pick_place_policy.sh
git diff --check
```

**Interpretation**
- 当前 30k baseline 的失败不只是训练步数不足，至少明显存在阶段/手臂条件不清晰的问题。
- 固定 global language 对每帧来说几乎是 constant task label，不能告诉 policy 当前该右臂抓、右臂放黄区、等待、左臂抓还是左臂放红区。
- Exp 1 将 hidden FSM 压缩成少量 semantic subtask，并显式告诉 policy active arm；这应该优先改善：
  - 左臂提前动作
  - 左夹爪开局夹空气
  - 左右臂 action 混合
  - action chunk 跨阶段残留
- 如果 Exp 1 仍然右臂抓不稳，下一步再转向 grasp 控制、action scale、gripper command convention、相机和数据质量排查。

**Next**
- 用新数据集正式训练 Exp 1：
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm

DATASET_DIR=/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/lerobot_datasets/lerobot_handoff_handoff_100_joint_ee_3cam_v1_subtask \
RUN_NAME=hf_mtdp_handoff_3cam_joint_ee_subtask_100success_bs16acc4_30k \
STATE_MODE=handoff_joint_ee_subtask \
BATCH_SIZE=16 \
GRAD_ACCUM_STEPS=4 \
STEPS=30000 \
bash isaac_pick_place/scripts/train_handoff_256_mtdp.sh
```
- 训练完成后跑 10 episode headless 视频评估，优先比较 `yellow_seen` 是否从 0 提升。

### 2026-06-15 16:36 CST - 双臂 Handoff raw demo 采集脚本

**Type:** env | data | scripted-expert

**Goal**
- 为黄色中转区 handoff 任务新增独立 raw demo 采集流程。
- 右侧 `observer_robot` 先把 cube 放到黄色区域，左侧 `robot` 再把 cube 放到红色区域。

**Result**
- handoff task action space 扩为双臂末端增量控制：
  - `action[0:6]`：左侧/actor `robot` 末端 6D delta pose
  - `action[6]`：左侧/actor `robot` gripper
  - `action[7:13]`：右侧/observer `observer_robot` 末端 6D delta pose
  - `action[13]`：右侧/observer `observer_robot` gripper
- 新增 `isaac_pick_place/scripts/scripted_handoff_collect.py`：
  - 顺序执行 `right_to_yellow -> left_to_red`
  - 同时只移动一只手，另一只手 action 保持 0
  - raw demo 每步记录 `stage`、`phase`、`active_arm`、14 维 action、左右手 7 维 action slice、cube/area world pose、yellow/red stage success 和三路图像路径
- 新增 `isaac_pick_place/scripts/collect_handoff_demos.sh`：
  - 默认只采 raw demo，不转换 LeRobot
  - 默认输出 `experiments/raw_demos/raw_handoff_<RUN_NAME>`

**Notes**
- Franka 本体仍是 7 自由度；这里每只手 action 的 `6+1` 是任务空间控制：6D 末端相对位姿增量经 Differential IK 映射到 7 个关节，再加 1D 夹爪。
- 修正右臂撤离逻辑：`right_retreat` 不再停在黄色区正上方，而是回到 episode reset 后记录的右臂 park TCP，避免左臂从黄色区抓取时撞到右臂。
- 修正左臂收尾逻辑：红区稳定达标后不立即结束 episode，而是继续执行 `left_retreat` 回到左臂 park TCP；retreat 阶段不能靠 timeout 跳过，最终 `success=True` 要求红区仍稳定且左臂已回 park；`summary.json` 额外记录 `red_stage_success`。
- handoff 环境关闭继承自旧单臂任务的内置 `terminations.success`，避免红区刚稳定就被 Isaac 自动 reset，导致左臂撤回动作被截断。

**Checks**
```bash
python -m py_compile \
  isaac_pick_place/tasks/cube_pick_place/handoff_env_cfg.py \
  isaac_pick_place/scripts/scripted_handoff_collect.py

bash -n isaac_pick_place/scripts/collect_handoff_demos.sh
git diff --check
```

### 2026-06-15 16:08 CST - 双臂黄色中转区环境草稿

**Type:** env | task-design

**Goal**
- 基于当前三相机双 Franka 场景，搭一个更像双臂协作的几何布局：
  - 右侧/observer arm 前方生成蓝色方块；
  - 两臂中间放黄色中转区域；
  - 左侧/actor arm 前方保留红色最终目标区域。

**Setup**
- 新 task id：`Isaac-Cube-Handoff-Yellow-Red-Dual-Franka-IK-Rel-Visuomotor-v0`
- 新环境配置：`isaac_pick_place/tasks/cube_pick_place/handoff_env_cfg.py`
- 继承：`CubePickPlaceRedTargetFrankaIKRelVisuomotorEnvCfg`
- world-frame 关键位置：
  - `cube_init_center_world_xy=(0.50, -0.30)`
  - `yellow_area_center_world_xy=(0.50, 0.00)`
  - `red_area_center_world_xy=(0.50, 0.30)`

**Result**
- 新增视觉黄色区域 `YellowHandoffArea`，大小 `0.12m x 0.12m`，无碰撞。
- 红色区域继续作为最终目标，位置固定到 world `(0.50, 0.30)`。
- 方块 reset 改为以 `observer_robot` 为参考，在其前方 world `(0.50, -0.30)` 附近随机，默认半径 `0.0-0.10m`。
- 当前只先搭场景和 reset 分布；右臂控制、双阶段奖励/终止、交接专家脚本尚未实现。

**Checks**
- `python -m py_compile isaac_pick_place/tasks/cube_pick_place/handoff_env_cfg.py isaac_pick_place/tasks/cube_pick_place/__init__.py`
- `git diff --check`

### 2026-06-15 15:21 CST - 三相机正式化代码改造

**Type:** env | data | train | eval | debug

**Goal**
- 将现有 `wrist_cam` 与 `observer_wrist_cam` 正式定义为两台机械臂各自的腕部相机。
- 新增真正的固定全局相机 `global_cam`，并让第三路图像进入后续数据集和模型输入。

**Setup**
- 代码路径：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm`
- 任务：`Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0`
- 关键文件：
  - `isaac_pick_place/tasks/cube_pick_place/env_cfg.py`
  - `isaac_pick_place/scripts/scripted_pick_place.py`
  - `isaac_pick_place/scripts/convert_raw_demos_to_lerobot.py`
  - `isaac_pick_place/scripts/train_hf_mtdp_smoke.py`
  - `isaac_pick_place/scripts/eval_pick_place_policy.py`

**Result**
- `observer_robot` 默认关节姿态已与主 `robot` 对齐。
- `observer_wrist_cam` 已改为与 `wrist_cam` 相同的手腕 local offset 和相机内参。
- 新增 `global_cam`：
  - `pos=(0.20, 0.00, 1.00)`
  - `rot=(0.69527, 0.12886, -0.12886, -0.69527)`，来自 Isaac UI 的 WXYZ 四元数
  - `convention="opengl"`，保证代码中的四元数和 IsaacSim Transform 面板中的 `Orient` 保持一致
  - `focal_length=14.0`
  - `clipping_range=(0.05, 3.0)`
- policy observation 新增 `global_rgb`。
- raw demo、LeRobot 转换、训练、评估均支持三路图像：
  - `observation.images.wrist_rgb`
  - `observation.images.observer_wrist_rgb`
  - `observation.images.global_rgb`
- 训练脚本新增 `--image-keys`，wrapper 对应 `IMAGE_KEYS`，默认 `auto`：
  - 旧两路数据集自动使用两路；
  - 新三路数据集自动使用三路。

**Checks**
```bash
python -m py_compile \
  isaac_pick_place/tasks/cube_pick_place/env_cfg.py \
  isaac_pick_place/scripts/scripted_pick_place.py \
  isaac_pick_place/scripts/smoke_lift_env.py \
  isaac_pick_place/scripts/eval_pick_place_policy.py \
  isaac_pick_place/scripts/convert_raw_demos_to_lerobot.py \
  isaac_pick_place/scripts/train_hf_mtdp_smoke.py

bash -n isaac_pick_place/scripts/train_random_cube_256_mtdp.sh
bash -n isaac_pick_place/scripts/collect_random_demos_to_lerobot.sh
bash -n isaac_pick_place/scripts/eval_pick_place_policy.sh
```
- 以上静态检查通过。
- 用临时 raw demo 片段验证转换逻辑：
  - 两路 raw demo 保持输出两路 image features。
  - 三路 raw demo 输出 `wrist_rgb + observer_wrist_rgb + global_rgb`。
- 用训练 venv 验证 `IMAGE_KEYS=auto`：
  - 旧两路 dataset metadata 解析为两路；
  - 新三路 dataset metadata 解析为三路。

**Blocked Runtime Check**
- 尝试运行 Isaac headless 相机 smoke 时，当前 Codex 执行环境无法看到 CUDA/NVIDIA driver：
  - `NVML_ERROR_DRIVER_NOT_LOADED`
  - `RuntimeError: No CUDA GPUs are available`
- 因此三路相机的实际画面质量仍需在正常服务器终端里运行 smoke 命令确认。

**Next**
- 在正常终端运行三路相机 smoke，检查 `global_rgb` 是否同时覆盖两台机械臂、蓝色方块和红色目标区域。
- 确认视角后再重新采集三路数据集并训练三路图像模型。

### 2026-06-12 10:05 CST - front_back_1 全状态模型随机闭环评估

**Type:** eval | train

**Goal**
- 记录 `front_back_1` 数据集训练完成后的全状态模型，在与数据集随机范围对齐的 Isaac 闭环评估结果。
- 为下一版“去掉关节角，只保留两路图像 + 末端状态”的消融实验提供基线。

**Setup**
- 代码路径：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm`
- 数据集：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/lerobot_datasets/lerobot_front_back_1`
- Checkpoint：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_back_side_256_v0_100success_bs32acc2_30k/final_model`
- 训练完成信息：`30000/30000` steps，最终训练 loss `0.002321`，`final_model` 已写出。
- Eval 输出目录：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/eval_videos/eval_front_back_gui_aligned_random`
- State 输入：`joint_pos(9) + ee_position(3) + ee_quat(4) = 16D`
- 图像输入：`wrist_rgb` + `observer_wrist_rgb`
- 关键环境随机化：
  - `TARGET_XY=0.67,0.00`
  - `CUBE_RESET_TARGET_XY=0.55,0.00`
  - `CUBE_RADIUS_RANGE=0.0,0.10`
  - `CUBE_ANGLE_RANGE_DEG=-210,-150`

**Command**
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm

TARGET_XY=0.67,0.00 \
CUBE_RESET_TARGET_XY=0.55,0.00 \
CUBE_RADIUS_RANGE=0.0,0.10 \
CUBE_ANGLE_RANGE_DEG=-210,-150 \
CHECKPOINT=/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_back_side_256_v0_100success_bs32acc2_30k/final_model \
RUN_NAME=eval_front_back_gui_aligned_random \
EPISODES=10 \
MAX_STEPS=3000 \
HEADLESS=1 \
SAVE_VIDEO=1 \
RECORD_IMAGE_EVERY=0 \
bash isaac_pick_place/scripts/eval_pick_place_policy.sh
```

**Result**
- `summary.json` 记录的实际 `max_steps=1500`，以评估产物为准。
- 成功率：`9/10 = 0.900`
- 成功 episode 步数：`506, 511, 504, 501, 531, 525, 484, 529, 498`
- 失败 episode：第 8 个 episode 跑满 `1500` step 未成功。
- `n_action_steps=8`，seed `2000`。
- 当前目录未看到 mp4；`summary.json` 中每个 episode 的 `image_counts` 都是 0，`videos={}`。这次命令里 `RECORD_IMAGE_EVERY=0` 导致没有抽帧，因此即使 `SAVE_VIDEO=1` 也没有可编码的视频帧。

**Interpretation**
- 新的 front/back 随机范围相比右侧遮挡数据更适合当前策略学习，闭环成功率已到 `90%`。
- 仍存在少量失败，下一步可以用同样随机范围对比 `ee_only` 消融，判断关节角低维输入是否是必要信息。
- 若需要保存 headless 评估视频，后续命令应设置 `RECORD_IMAGE_EVERY=5` 或 `RECORD_IMAGE_EVERY=1`。

**Next**
- 新增训练参数 `STATE_MODE=ee_only`，让模型低维输入从 16D 改为 `ee_position(3) + ee_quat(4) = 7D`，输出 action 保持不变。
- 评估脚本需要自动根据 checkpoint 的 `observation.state` 维度选择 16D 或 7D state，避免新模型推理时喂错输入。

### 2026-06-11 15:50 CST - 末端姿态与 action 旋转表示说明

**Type:** design | debug

**Goal**
- 明确当前模型输入 state 和输出 action 中的末端旋转表示，避免后续误解 action 维度含义。

**Setup**
- 代码路径：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm`
- 环境配置：`isaac_pick_place/tasks/cube_pick_place/env_cfg.py`
- MDP 观测：`isaac_pick_place/tasks/cube_pick_place/mdp.py`
- 数据转换：`isaac_pick_place/scripts/convert_raw_demos_to_lerobot.py`
- Scripted expert：`isaac_pick_place/scripts/scripted_pick_place.py`
- 训练脚本：`isaac_pick_place/scripts/train_hf_mtdp_smoke.py`

**Result**
- 当前模型实际低维 state 输入为：
  - `joint_pos(9) + ee_position(3) + ee_quat(4) = 16D`
- 其中末端旋转观测是 `ee_quat`，即四元数，4 维。
- LeRobot action 为 7 维：
  - `delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper`
- 因此 action 的旋转增量是 3 维；第 7 维是夹爪开合，不是旋转。
- 当前 scripted expert 实际主要只写平移和夹爪：
  - `actions[:, :3] = clipped_delta_b / args.arm_action_scale`
  - `actions[:, 6] = gripper`
  - `actions[:, 3:6]` 旋转增量基本保持 0。

**Interpretation**
- 绝对末端姿态用 quaternion 表示更稳定，避免欧拉角万向节锁和角度跳变。
- 小的相对旋转 action 用 3 维增量更适合 IK controller 和学习；如果用 quaternion delta，会引入单位范数约束以及 `q`/`-q` 双重表示问题。
- 当前任务主要是平移抓放，暂时不需要主动学习腕部旋转；如果后续任务需要复杂抓取角度，再让 expert 生成非零旋转增量，并同步检查数据分布。

**Next**
- 暂时保持当前表示不变。
- 若引入需要旋腕的任务，重新评估 action 旋转维度、normalization 和 scripted expert 的旋转轨迹。

### 2026-06-11 14:26 CST - 训练后闭环可视化评估与左侧随机数据集方案

**Type:** eval | viz | data | debug

**Goal**
- 对训练完成的 `hf_mtdp_random_cube_256_v0_100success_bs32acc2_30k` checkpoint 做 Isaac 闭环可视化评估。
- 验证固定 cube 初始位置下策略是否能成功，并判断此前失败是否与相机遮挡红色目标区域有关。
- 准备下一版数据采集方案：重新采集 100 条成功 demo，让蓝色方块随机在红色区域左侧，减少抓取阶段机械臂对红色区域的遮挡。

**Setup**
- 代码路径：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm`
- Eval wrapper：`isaac_pick_place/scripts/eval_pick_place_policy.sh`
- Eval Python：`isaac_pick_place/scripts/eval_pick_place_policy.py`
- 数据采集 wrapper：`isaac_pick_place/scripts/collect_random_demos_to_lerobot.sh`
- Checkpoint：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_random_cube_256_v0_100success_bs32acc2_30k/final_model`
- 训练数据集：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/lerobot_datasets/lerobot_random_cube_256_v0_100success`
- Device：`cuda`
- Eval 图像：`wrist_rgb` + `observer_wrist_rgb`
- Eval 默认输出：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/eval_videos/${RUN_NAME}`
- 原始数据集 FPS：`50`
- 模型实际低维状态输入：`joint_pos(9) + ee_position(3) + ee_quat(4) = 16` 维；不包含低维 `object_position` 或 `target_area_position`。

**Code Changes**
- `eval_pick_place_policy.py`
  - 增加 `--fixed-cube-xy`，可指定单个 cube 初始 XY。
  - 增加 `--fixed-cube-xy-list`，可用分号分隔多个固定点，在同一个 Isaac 进程里连续评估。
  - 增加 safetensors 权重加载兼容逻辑，并对 Transformers 5.x/4.x CLIP key 做 remap。评估日志显示 `compat_key_remaps=395`，`missing_keys=0`，`unexpected_keys=0`。
- `eval_pick_place_policy.sh`
  - 增加 `FIXED_CUBE_XY`、`FIXED_CUBE_XY_LIST` 和 `HEADLESS=0` 支持。
- `env_cfg.py`
  - 增加 cube 随机化范围环境变量：
    - `CUBE_RADIUS_RANGE=min,max`
    - `CUBE_ANGLE_RANGE_DEG=min,max`
    - `CUBE_ANGLE_RANGE_RAD=min,max`
  - 默认随机范围保持旧数据集不变。
- `collect_random_demos_to_lerobot.sh`
  - 输出采集时使用的 cube 半径和角度随机化配置，便于复现。
- `experiments/` 目录已整理为：
  - `raw_demos/`
  - `lerobot_datasets/`
  - `training_runs/`
  - `eval_videos/`
  - `camera_debug_runs/`
  - `smoke_runs/`
  - `reports/`
  - `docs/`

**Commands**
```bash
# 3000 step 闭环评估，保存视频
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm

CHECKPOINT=/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_random_cube_256_v0_100success_bs32acc2_30k/final_model \
RUN_NAME=eval_final_remap_ep1_3000 \
EPISODES=1 \
MAX_STEPS=3000 \
SAVE_VIDEO=1 \
RECORD_IMAGE_EVERY=5 \
bash isaac_pick_place/scripts/eval_pick_place_policy.sh

# 用户给出的固定点，成功
CHECKPOINT=/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/training_runs/hf_mtdp_random_cube_256_v0_100success_bs32acc2_30k/final_model \
RUN_NAME=eval_fixed_040_m011_ep1_3000 \
EPISODES=1 \
MAX_STEPS=3000 \
FIXED_CUBE_XY=0.40,-0.11 \
SAVE_VIDEO=1 \
RECORD_IMAGE_EVERY=5 \
bash isaac_pick_place/scripts/eval_pick_place_policy.sh

# 新数据集采集与自动转换命令，待正式执行
CUBE_ANGLE_RANGE_DEG=120,240 \
EPISODES=100 \
MAX_ATTEMPTS=180 \
RUN_NAME=random_cube_left_side_256_v0_100success \
bash isaac_pick_place/scripts/collect_random_demos_to_lerobot.sh
```

**Result**
- 训练完成后最新 checkpoint 已确认在：
  - `experiments/training_runs/hf_mtdp_random_cube_256_v0_100success_bs32acc2_30k/final_model`
- `eval_final_remap_ep1_3000`
  - Result：失败。
  - 现象：cube 曾被轻微抬起，`max cube z ~= 0.0655`，约在 step `1939`；最终掉回桌面，未放置成功。
- `eval_fixed_040_m011_ep1_3000`
  - 固定 cube XY：`0.40,-0.11`
  - Result：成功 `1/1`。
  - 成功步数：`1077`
  - 最高 cube z：约 `0.2537`，约在 step `976`
  - 最终 cube position：约 `[0.4943, 0.3014, 0.0170]`
  - 视频：
    - `/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/eval_videos/eval_fixed_040_m011_ep1_3000/episode_000000/observer_wrist_rgb.mp4`
    - `/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/eval_videos/eval_fixed_040_m011_ep1_3000/episode_000000/wrist_rgb.mp4`
- 固定点网格评估尝试：
  - 中心：`0.40,-0.11`
  - 3x3 点：`0.36,-0.15;0.40,-0.15;0.44,-0.15;0.36,-0.11;0.40,-0.11;0.44,-0.11;0.36,-0.07;0.40,-0.07;0.44,-0.07`
  - `MAX_STEPS=2000`
  - 用户中途要求停止，因此没有完整网格结论。
  - 已观察：`0.36,-0.15` 失败；`0.40,-0.15` 运行到约 750 step 时表现也较差，未记录最终完整结果。
- 旧数据集 cube 随机化范围：
  - `target_xy=(0.50,0.00)`
  - `radius_range=(0.12,0.22)`
  - `angle_range=(-150 deg, -30 deg)`
  - robot-root frame 近似范围：`x=0.309~0.691`，`y=-0.220~-0.060`
  - world frame 近似 `y=0.080~0.240`，因为 robot root y offset 为 `0.30`。
- 新计划左侧随机范围：
  - `CUBE_ANGLE_RANGE_DEG=120,240`
  - 半径默认 `0.12~0.22`
  - robot-root frame 近似范围：`x=0.28~0.44`，`y=-0.1905~0.1905`
  - world frame 近似 `y=0.109~0.491`

**Interpretation**
- checkpoint 能在至少一个固定点 `0.40,-0.11` 成功，说明模型并非完全不会执行 pick-place。
- 但默认随机位置和部分周边固定点失败，说明当前策略泛化区域较窄，且对 cube 初始位置/视觉遮挡比较敏感。
- 一个重要数据问题是：旧随机范围把蓝色方块主要放在红色区域右侧；抓取时机械臂/夹爪容易挡住全局相机对红色区域的观测。由于模型低维输入不包含目标位置，放置阶段更依赖图像中能否看到红色目标区域。
- 下一版把 cube 随机到红色区域左侧，可以让抓取阶段更少遮挡红色目标，有助于验证失败是否主要来自视觉遮挡和数据分布偏置。

**Next**
- 正式执行左侧随机 100 条成功 demo 采集与自动转换：
  - Raw 输出：`experiments/raw_demos/raw_demos_random_cube_left_side_256_v0_100success`
  - LeRobot 输出：`experiments/lerobot_datasets/lerobot_random_cube_left_side_256_v0_100success`
  - Report 输出：`experiments/reports/random_cube_left_side_256_v0_100success_scripted_report.txt`
- 使用新数据集训练下一版 checkpoint。
- 训练后重复固定点和小网格评估，对比旧数据集 checkpoint 的成功区域。

### 2026-06-10 13:58 CST - 补记：100 条随机方块 256px 数据采集与 LeRobot 转换

**Type:** data

**Goal**
- 补记 100 条随机蓝色方块 pick-place 成功 demo 的采集与 LeRobot 转换结果，作为当前正式训练数据集来源。

**Setup**
- 代码路径：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm`
- 采集脚本：`isaac_pick_place/scripts/collect_random_demos_to_lerobot.sh`
- Expert 脚本：`isaac_pick_place/scripts/scripted_pick_place.py`
- 转换脚本：`isaac_pick_place/scripts/convert_raw_demos_to_lerobot.py`
- Task：`Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0`
- Device：`cuda:0`
- Seed：`1000`
- 相机：`wrist_rgb` + `observer_wrist_rgb`
- 图像分辨率：`256x256`

**Command**
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm

EPISODES=100 MAX_ATTEMPTS=150 RUN_NAME=random_cube_256_v0_100success \
  bash isaac_pick_place/scripts/collect_random_demos_to_lerobot.sh
```

**Result**
- Raw 数据：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/raw_demos_random_cube_256_v0_100success`
- LeRobot 数据：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/lerobot_random_cube_256_v0_100success`
- 采集 report：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/random_cube_256_v0_100success_scripted_report.txt`
- 转换 summary：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/lerobot_random_cube_256_v0_100success/conversion_summary.json`
- 采集结果：`successes=100/100 attempts=113`
- 转换 episode 数：`100`
- 总帧数：`97041`
- 图像 shape：`[256, 256, 3]`
- 单 episode 帧数范围：`894` 到 `1343`，平均 `970.41`
- 第一个转换 raw episode：`episode_000000`
- 最后一个转换 raw episode：`episode_000112`
- LeRobot reload check 已通过。

**Interpretation**
- 成功驱动采集逻辑正常工作：脚本在 113 次尝试后凑齐 100 条成功 demo，转换阶段只选择成功 episode。
- 这批数据是当前正式训练 `hf_mtdp_random_cube_256_v0_100success_bs32acc2_30k` 的数据来源。

**Next**
- 用该数据集继续正式训练，并在训练记录中补充 loss、显存峰值、耗时和 checkpoint。
- 训练后实现 Isaac 闭环 eval。

### 2026-06-10 13:45 CST - MultiTask DiT 100 条随机方块正式训练配置

**Type:** train

**Goal**
- 使用已采集并转换完成的 100 条随机蓝色方块 pick-place 成功 demo，启动第一版正式 MultiTask DiT 训练。
- 训练配置对齐此前 PushT 成功的 HF/LeRobot MultiTask DiT 基线：6-layer、hidden_dim 512、horizon 32、diffusion timesteps 100。

**Setup**
- 代码路径：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm`
- 训练脚本：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/train_random_cube_256_mtdp.sh`
- 底层 Python trainer：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/train_hf_mtdp_smoke.py`
- 数据集：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/lerobot_random_cube_256_v0_100success`
- 数据规模：100 successful episodes，`97041` frames
- 图像：两路 RGB，相机项为 `wrist_rgb` 和 `observer_wrist_rgb`，LeRobot feature shape 为 `[3, 256, 256]`
- State 输入：`joint_pos(9) + ee_position(3) + ee_quat(4) = 16` 维
- Action 输出：7 维相对末端 action，`delta_x/y/z + delta_roll/pitch/yaw + gripper`
- Device：`cuda`
- Seed：`17`

**Command**
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm

BATCH_SIZE=32 GRAD_ACCUM_STEPS=2 RUN_NAME=hf_mtdp_random_cube_256_v0_100success_bs32acc2_30k \
  bash isaac_pick_place/scripts/train_random_cube_256_mtdp.sh
```

**Config**
- `STEPS=30000`
- `SAVE_FREQ=1000`
- `BATCH_SIZE=32`
- `GRAD_ACCUM_STEPS=2`
- `Effective batch size: 64`
- `HORIZON=32`
- `N_OBS_STEPS=2`
- `N_ACTION_STEPS=8`
- `HIDDEN_DIM=512`
- `NUM_LAYERS=6`
- `NUM_HEADS=8`
- `NUM_TRAIN_TIMESTEPS=100`
- `LR=2e-5`
- `IMAGE_SIZE=224`
- `VIDEO_BACKEND=torchcodec`
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `PYTORCH_ALLOC_CONF=expandable_segments:True`

**Result**
- 待运行或待补充。
- 预期输出目录：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/hf_mtdp_random_cube_256_v0_100success_bs32acc2_30k`
- 预期最终 checkpoint：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/hf_mtdp_random_cube_256_v0_100success_bs32acc2_30k/final_model`
- 预期中间 checkpoint：`checkpoint_1000`, `checkpoint_2000`, ...

**Interpretation**
- `BATCH_SIZE=32, GRAD_ACCUM_STEPS=2` 用 micro-batch 32 降低显存峰值，同时保持与 PushT 基线相同的 effective batch 64。
- 相比 PushT，该任务使用两路相机和 2 帧历史，因此不能直接使用单次 `batch_size=64`；此前单次 batch 64 已触发 CUDA OOM。

**Next**
- 运行后记录最终 loss、训练耗时、显存峰值和 checkpoint 路径。
- 训练完成后实现/运行 Isaac 闭环 eval，测试 `n_action_steps=8/12` 的推理效果差异。

### 2026-06-09 15:03 CST - 目标对齐 raw demos 转 LeRobot v3

**Type:** data

**Goal**
- 将修正红色目标中心后的 2 个 scripted raw demo 转换为官方 `LeRobotDataset` v3 目录，用于后续训练读数 smoke test。

**Setup**
- 代码路径：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm`
- Raw 数据：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/raw_demos_target_aligned_v0_2eps`
- 输出数据：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/lerobot_target_aligned_v0_2eps`
- Repo id：`local/seven_dof_pick_place_lbm_target_aligned_v0_2eps`
- LeRobot：`0.5.1`
- Python：`uv` managed `3.12`
- FPS：`50`
- 视频编码：`h264`
- Feature：
  - `observation.state`: 24 维，`joint_pos(9) + joint_vel(9) + object_position(3) + target_area_position(3)`
  - `action`: 7 维，末端相对增量 + gripper
  - `observation.images.wrist_rgb`: video, `200x200x3`
  - `observation.images.observer_wrist_rgb`: video, `200x200x3`

**Command**
```bash
uv run --python 3.12 \
  --with lerobot==0.5.1 \
  --with pyarrow \
  --with datasets \
  --with av \
  --with pillow \
  --with 'evdev==1.7.1' \
  python /home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/convert_raw_demos_to_lerobot.py \
    --raw-dir /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/raw_demos_target_aligned_v0_2eps \
    --output-dir /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/lerobot_target_aligned_v0_2eps \
    --repo-id local/seven_dof_pick_place_lbm_target_aligned_v0_2eps \
    --fps 50 \
    --vcodec h264
```

**Result**
- 转换脚本：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/convert_raw_demos_to_lerobot.py`
- 输出目录大小：`2.3M`
- Raw episodes：`2`
- 总帧数：`2038`
- 每个 episode：`1019` frames
- 输出文件：
  - `data/chunk-000/file-000.parquet`
  - `meta/info.json`
  - `meta/stats.json`
  - `meta/tasks.parquet`
  - `meta/episodes/chunk-000/file-000.parquet`
  - `videos/observation.images.wrist_rgb/chunk-000/file-000.mp4`
  - `videos/observation.images.observer_wrist_rgb/chunk-000/file-000.mp4`
- 视频验证：
  - 两路视频均为 `200x200`
  - `r_frame_rate=50/1`
  - `nb_frames=2038`
- 官方 loader smoke test：
```text
len 2038
observation.state torch.Size([24]) torch.float32
action torch.Size([7]) torch.float32
observation.images.wrist_rgb torch.Size([3, 200, 200]) torch.float32
observation.images.observer_wrist_rgb torch.Size([3, 200, 200]) torch.float32
```

**Interpretation**
- 已生成官方 `LeRobotDataset` v3 结构，不是手写近似格式。
- `evdev==1.9.3` 在本机内核头上编译失败，因此转换命令锁定 `evdev==1.7.1`。
- 旧目录 `raw_demos_camera_tuned_v0_long_timeout` 的 5 集数据存在目标中心错位，不用于训练。

**Next**
- 用该 2 episode LeRobot 数据集跑 dataloader/训练 smoke test。
- 确认训练脚本能读后，再恢复/加入 cube pose randomization，并采集更大规模目标对齐数据。

### 2026-06-09 12:05 CST - Actor 初始姿态更高并让 Wrist Camera 低头

**Type:** env | viz | debug

**Goal**
- 在上一版保守高位姿态基础上，继续抬高 actor 初始末端，并让 wrist camera 低头一点，扩大 reset wrist view。

**Setup**
- Env config：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/tasks/cube_pick_place/env_cfg.py`
- Camera smoke output：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/camera_debug_actor_higher_lookdown/step_000001_wrist_cam.png`
- Camera smoke report：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/camera_debug_actor_higher_lookdown_report.txt`

**Command**
```bash
cd /home/ubuntu/Workspace/IsaacLab
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

TERM=xterm ./isaaclab.sh -p /home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py \
  --task Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0 \
  --num_envs 1 \
  --steps 2 \
  --headless \
  --enable_cameras \
  --device cuda:0 \
  --action-mode zero \
  --refresh-camera-xform \
  --print-camera-pose \
  --save-camera-frame-dir /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/camera_debug_actor_higher_lookdown \
  --camera-frame-step 1 \
  --report /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/camera_debug_actor_higher_lookdown_report.txt
```

**Result**
- `py_compile` 通过。
- Actor 初始关节姿态更新为：
  - `panda_joint2=-0.850`
  - `panda_joint4=-2.550`
  - `panda_joint6=3.058`
- Actor wrist camera 低头后的 ROS offset rot：
  - `rot=(0.66014, -0.25340, -0.25340, 0.66014)`
- Smoke 图像显示：wrist 初始视野更高、更宽，没有手掌遮挡；但 cube/red target 仍不在 actor wrist 初始画面内，初始全局信息继续依赖 `observer_wrist_rgb`。

**Interpretation**
- 这版比大幅 ready-high 姿态和上一版保守高位姿态更适合继续 GUI 调参。
- 如果希望 actor wrist 初始也看到 cube/red target，下一步应在 GUI 里用关节控制手动找一组更合适的 joint pose，再回填 `env_cfg.py`，不要只靠盲调 quaternion。

**Next**
- 在 Isaac Sim GUI 中用 Articulation Inspector 或 joint drive target 手动调整 actor arm 初始关节，找到满意视角后记录各 `panda_joint*` 数值。
- 若手动调好后，回填 `self.scene.robot.init_state.joint_pos` 并重新 smoke。

### 2026-06-09 11:58 CST - 调整 Actor 初始关节姿态以抬高 Wrist Reset View

**Type:** env | viz | debug

**Goal**
- 让 actor wrist camera 初始视角不要过低/过局部，方便 reset 后看到更大的桌面区域。

**Setup**
- Env config：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/tasks/cube_pick_place/env_cfg.py`
- Camera smoke output：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/camera_debug_actor_high_ready_conservative/step_000001_wrist_cam.png`
- Camera smoke report：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/camera_debug_actor_high_ready_conservative_report.txt`

**Command**
```bash
cd /home/ubuntu/Workspace/IsaacLab
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

TERM=xterm ./isaaclab.sh -p /home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py \
  --task Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0 \
  --num_envs 1 \
  --steps 2 \
  --headless \
  --enable_cameras \
  --device cuda:0 \
  --action-mode zero \
  --refresh-camera-xform \
  --print-camera-pose \
  --save-camera-frame-dir /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/camera_debug_actor_high_ready_conservative \
  --camera-frame-step 1 \
  --report /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/camera_debug_actor_high_ready_conservative_report.txt
```

**Result**
- `py_compile` 通过。
- Actor arm 新增显式初始关节姿态：
  - `panda_joint2=-0.750`
  - `panda_joint4=-2.650`
  - `panda_joint6=3.058`
  - 其他 arm joints 接近 Franka 默认值，gripper open。
- 第一版大幅 ready-high 姿态会让手掌/手背挡住 wrist view，已弃用。
- 当前保守姿态的 reset wrist view 更高、更宽，能看到更大的桌面区域；但初始帧里 cube/red target 暂时不在 actor wrist 画面内，需要 GUI 再确认是否符合预期。

**Interpretation**
- 这个改动更适合解决“初始 wrist view 太贴桌面/太局部”的问题。
- 初始目标信息仍主要依赖 `observer_wrist_rgb`；actor wrist 负责靠近和操作阶段的局部视觉。

**Next**
- 用 GUI 长观察命令确认 actor arm 初始姿态是否自然、无碰撞，并切到 `wrist_cam` 看初始视野是否满意。
- 若希望 actor wrist 初始也看到 cube/red target，应再小幅调整 actor root/初始关节姿态或 wrist camera pitch，而不是继续大幅抬高手腕。

### 2026-06-09 11:40 CST - 采集 3 条 Raw Scripted Demo Smoke 数据

**Type:** data | debug

**Goal**
- 先按 raw demos 格式采一小批 scripted expert 数据，验证 `wrist_rgb`、`observer_wrist_rgb`、7D raw action、phase 和 metrics 能稳定落盘。

**Setup**
- Script：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/scripted_pick_place.py`
- Valid raw dataset：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/raw_demos_smoke_v0_retry`
- Manifest：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/raw_demos_smoke_v0_retry/manifest.json`
- Report：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/raw_demos_smoke_v0_retry_report.txt`
- Episodes：`3`
- Seed：`42`
- Device：`cuda:0`
- Camera refresh：enabled
- Warmup：`2` steps after reset/refresh, not recorded
- Cube pose：当前仍为临时 fixed pose

**Command**
```bash
cd /home/ubuntu/Workspace/IsaacLab
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

TERM=xterm ./isaaclab.sh -p /home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/scripted_pick_place.py \
  --task Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0 \
  --num_envs 1 \
  --episodes 3 \
  --max-steps 1200 \
  --seed 42 \
  --headless \
  --enable_cameras \
  --device cuda:0 \
  --refresh-camera-xform \
  --record-dir /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/raw_demos_smoke_v0_retry \
  --record-warmup-steps 2 \
  --report /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/raw_demos_smoke_v0_retry_report.txt
```

**Result**
- `scripted_pick_place.py` 新增 raw demo recorder，并通过 `py_compile`。
- 非沙箱运行可访问 GPU；沙箱内运行失败，原因是 `No CUDA GPUs are available` / `NVML_ERROR_DRIVER_NOT_LOADED`。
- 第一次输出目录 `raw_demos_smoke_v0` 在第 1 条成功后，第 2 条 reset 触发 `torch.inference_mode` 导致的 Isaac Lab buffer 写入错误；已将 rollout 包裹从 `torch.inference_mode()` 改为 `torch.no_grad()`。
- 第二次有效采集目录 `raw_demos_smoke_v0_retry` 完成：
  - Success：`3/3`
  - Episode steps：`860, 860, 860`
  - 每条 `steps.jsonl` 行数：`860`
  - 每条 `wrist_rgb` 图片数：`860`
  - 每条 `observer_wrist_rgb` 图片数：`860`
  - 每条 `summary.json` 均为 `success=true`
- 抽查 quality-check PNG：
  - `episode_000000/quality_check/wrist_rgb_reset.png` 正常，红区居中，没有启动斜转问题。
  - `episode_000000/quality_check/observer_wrist_rgb_reset.png` 正常，可见 actor arm、cube 和 red target。

**Interpretation**
- Raw demo 数据链路已经可用：reset 后 camera xform refresh、warmup、图像保存、7D action/phase/metrics/jsonl 都正常。
- 当前 3 条 episode 因 cube pose 固定，内容高度相似，只能作为 smoke dataset，不适合作为训练集。
- 第一条失败目录 `raw_demos_smoke_v0` 不作为有效数据集使用；后续以 `raw_demos_smoke_v0_retry` 为准。

**Next**
- 写 raw demo 到 HF/LeRobot dataset 的转换脚本。
- 恢复或重写 cube pose randomization 后，再采 20-50 条 smoke train dataset。

### 2026-06-09 11:06 CST - 确认 Camera Xform Refresh Workaround

**Type:** viz | debug

**Goal**
- 验证 `wrist_cam` 刚启动时 sensor RGB 画面异常是否可以通过重写相同 local camera xform 修复。

**Setup**
- Script：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py`
- Env config：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/tasks/cube_pick_place/env_cfg.py`
- Debug image before refresh：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/camera_debug/step_000001_wrist_cam.png`
- Debug image after refresh：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/camera_debug_refresh/step_000001_wrist_cam.png`

**Command**
```bash
cd /home/ubuntu/Workspace/IsaacLab
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

DISPLAY=:1 TERM=xterm ./isaaclab.sh -p /home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py \
  --task Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0 \
  --num_envs 1 \
  --steps 5 \
  --enable_cameras \
  --device cuda:0 \
  --action-mode zero \
  --print-camera-pose \
  --refresh-camera-xform \
  --save-camera-frame-dir /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/camera_debug_refresh \
  --camera-frame-step 1
```

**Result**
- `--refresh-camera-xform` 已修正 `GfQuatd/GfQuatf` 类型不匹配问题，并通过 `py_compile`。
- Refresh 后 `step_000001_wrist_cam.png` 明显恢复正常：红色目标区居中，画面不再出现启动时的斜转状态。
- Report 中 refresh 前后 camera world pose 数值保持一致，例如 `wrist_cam.pos_w` 和 `wrist_cam.quat_w_world` 未变化。
- 新增 debug 参数：
  - `--refresh-camera-xform`

**Interpretation**
- 这更像 RTX/Replicator/render product 或 USD xform dirty 标记没有在 camera prim 初始化后正确刷新，而不是 `env_cfg.py` 里的 camera local `pos/rot` 本身错误。
- 后续采集器应在每次 `env.reset()` 后主动 refresh camera local xform，并丢弃第一帧或至少从 refresh 后的下一帧开始写 dataset。

**Next**
- 在 dataset/demo collector 中集成 camera xform refresh。
- 采集 smoke demos 前先保存一张 refresh 后的 `wrist_cam` 和 `observer_wrist_cam` PNG，作为数据质量检查。

### 2026-06-09 10:57 CST - 增加 Wrist Camera Sensor Debug 输出

**Type:** viz | debug

**Goal**
- 区分 Isaac Sim GUI viewport 的 camera 刷新问题和 Isaac Lab camera sensor 的真实输出。
- 启动 GUI 后保存 `wrist_cam` / `observer_wrist_cam` 的 sensor RGB，并打印 camera world pose。

**Setup**
- Script：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py`
- Camera sensors：`wrist_cam,observer_wrist_cam`

**Command**
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm
/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python -m py_compile \
  isaac_pick_place/scripts/smoke_lift_env.py
```

**Result**
- `py_compile` 通过。
- `PIL` 可用，可直接保存 `.png`。
- 新增参数：
  - `--print-camera-pose`
  - `--save-camera-frame-dir`
  - `--camera-frame-step`
  - `--camera-names`

**Interpretation**
- 如果 GUI viewport 刚打开画面异常，但保存出来的 sensor RGB 正常，则问题更可能是 viewport camera transform/cache 刷新，不影响训练图像。
- 如果 sensor RGB 也异常，再继续查 camera prim 初始化、parent link world pose 或 reset 后第一帧 sensor update。

**Next**
- 用 GUI 命令保存启动后的真实 wrist camera 图像，对比 viewport 画面。

### 2026-06-08 17:44 CST - 同步 Reset Joint Targets 以消除 Observer 启动抖动

**Type:** env | debug

**Goal**
- 解释并修复 observer 机械臂在 GUI 启动开头轻微动一下的问题。

**Setup**
- Env config：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/tasks/cube_pick_place/env_cfg.py`
- Observer robot：`{ENV_REGEX_NS}/ObserverRobot`
- Reset event：`self.events.reset_all`

**Command**
```bash
cd /home/ubuntu
/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python -m py_compile \
  Workspace/seven_dof_pick_place_lbm/isaac_pick_place/tasks/cube_pick_place/env_cfg.py
```

**Result**
- `py_compile` 通过。
- 在 reset event 上设置：
  - `self.events.reset_all.params = {"reset_joint_targets": True}`

**Interpretation**
- Isaac Lab 默认 reset 会把 articulation joint state 写回默认值，但不一定同步写入 actuator joint target。
- Actor arm 每步有 action/controller 刷新目标，所以不明显；observer arm 当前没有 action term，作为静态相机臂时会被 PD 控制器短暂拉向旧 target，表现为开头动一下。
- reset 时同步 joint targets 后，observer 的 PD target 和 init joint state 一致，启动抖动应消失。

**Next**
- 在 GUI 中重新启动 env，确认 observer arm 第一帧不再 twitch。
- 若仍有细微 settling，再检查 observer 初始关节位姿是否存在自碰撞/桌面接触或重力下不可平衡的姿态。

### 2026-06-08 17:26 CST - 临时固定 Cube Reset Pose

**Type:** env | viz | debug

**Goal**
- 调 observer camera 和双臂布局期间，先固定方块初始位置，避免每次 reset 随机方块位置干扰视角判断。

**Setup**
- Env config：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/tasks/cube_pick_place/env_cfg.py`
- Cube fixed pose：`pos=(0.50, 0.00, 0.0205)`

**Command**
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm
/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python -m py_compile \
  isaac_pick_place/tasks/cube_pick_place/env_cfg.py
```

**Result**
- `py_compile` 通过。
- 临时关闭 Lift 继承来的 cube pose randomization：
  - `self.events.reset_object_position = None`

**Interpretation**
- 之后 reset 不再给 cube 初始位置加 `x/y` 随机偏移。
- 这是调相机/布局用的临时 debug 设置；后续采 demo 或训练前要恢复随机 cube pose。

**Next**
- observer camera 和双臂布局调好后，恢复 `reset_object_position` 或改成我们自己的随机范围。

### 2026-06-08 17:04 CST - 显式化 Actor Arm 初始位姿

**Type:** env | debug

**Goal**
- 让 actor/observer 两只机械臂的位置和朝向都能在当前 `env_cfg.py` 同一处调整，不再需要追到 Isaac Lab Lift 基类里找 actor 默认位姿。

**Setup**
- Env config：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/tasks/cube_pick_place/env_cfg.py`
- Actor robot：`{ENV_REGEX_NS}/Robot`
- Observer robot：`{ENV_REGEX_NS}/ObserverRobot`

**Command**
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm
/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python -m py_compile \
  isaac_pick_place/tasks/cube_pick_place/env_cfg.py
```

**Result**
- `py_compile` 通过。
- 新增 actor arm 显式初始位姿：
  - `self.scene.robot.init_state.pos = (0.0, 0.0, 0.0)`
  - `self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)`
- Observer arm 位姿仍在相邻代码段：
  - `self.scene.observer_robot.init_state.pos = (0.0, -0.38, 0.0)`
  - `self.scene.observer_robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)`

**Interpretation**
- 后续双臂布局可以直接改 `env_cfg.py` 中 actor/observer 的 `init_state.pos` 和 `init_state.rot`。
- 当前值保持 actor 原默认位姿，不改变 scripted expert 的坐标假设。

**Next**
- GUI 中继续微调两臂间距和 observer wrist camera 视角。

### 2026-06-08 16:51 CST - 禁用周期性 Time-Out Reset

**Type:** env | viz | debug

**Goal**
- 删除长开 GUI 时每隔一段时间触发 reset、导致新加的 observer arm 动一下的逻辑。

**Setup**
- Env config：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/tasks/cube_pick_place/env_cfg.py`
- 触发方式：GUI 使用 `smoke_lift_env.py --steps -1` 长开时，Lift 基类 `time_out` termination 会按 `episode_length_s=30.0` 周期触发整场景 reset。

**Command**
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm
/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python -m py_compile \
  isaac_pick_place/tasks/cube_pick_place/env_cfg.py
```

**Result**
- `py_compile` 通过。
- 在 env config 中设置：
  - `self.terminations.time_out = None`
- 保留：
  - `object_dropping`
  - `success`
- Observer arm 本身仍为静止 `observer_robot`，不进 action space。

**Interpretation**
- 之前 observer arm 周期性动一下，不是它有单独控制逻辑，而是 episode timeout 后 `reset_scene_to_default` 重置整场景。
- 禁用 `time_out` 后，长开 GUI 不会再按 30 秒周期 reset；scripted expert 仍可通过 `success` 或脚本 `--max-steps` 结束。

**Next**
- 重新用 `--steps -1` 打开 GUI，观察 observer arm 是否还会周期性跳动。

### 2026-06-08 16:40 CST - 调整双臂桌面布局与 Observer 相机视角

**Type:** env | viz | debug

**Goal**
- 修正 GUI 检查发现的两个问题：
  - 新加的 observer arm 漂浮在桌面外，双臂布局太拥挤。
  - `observer_wrist_rgb` 的桌面视角明显斜。

**Setup**
- Env config：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/tasks/cube_pick_place/env_cfg.py`
- Actor robot：`{ENV_REGEX_NS}/Robot`
- Observer robot：`{ENV_REGEX_NS}/ObserverRobot`
- Observer wrist camera：`{ENV_REGEX_NS}/ObserverRobot/panda_hand/observer_wrist_cam`

**Command**
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm
/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python -m py_compile \
  isaac_pick_place/tasks/cube_pick_place/env_cfg.py
```

GUI 检查建议命令：

```bash
cd /home/ubuntu/Workspace/IsaacLab
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
TERM=xterm ./isaaclab.sh -p /home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py \
  --task Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0 \
  --num_envs 1 \
  --steps -1 \
  --enable_cameras \
  --device cuda:0 \
  --action-mode zero \
  --report /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/smoke_dual_arm_observer_report.txt
```

**Result**
- `py_compile` 通过。
- 桌子 USD 在 x/y 方向放大，z 不变：
  - `self.scene.table.spawn.scale = (1.35, 1.80, 1.0)`
- Observer arm 从桌外拉回到 actor 同一长边并排：
  - old `pos=(0.05, -0.75, 0.0)`
  - new `pos=(0.0, -0.38, 0.0)`
- Observer arm 朝向改为与 actor 一致：
  - old `rot=(0.70711, 0.0, 0.0, 0.70711)`
  - new `rot=(1.0, 0.0, 0.0, 0.0)`
- Observer wrist camera 视野进一步放宽：
  - `focal_length: 16.0 -> 14.0`

**Interpretation**
- 当前策略是先保留 actor 坐标系不动，避免破坏 scripted pick-place；只移动 observer arm 解决漂浮和视角问题。
- 桌面通过 USD x/y scale 扩展，比单独加 free-floating table camera 更符合后续双臂场景。
- `observer_wrist_rgb` 是否足够正，还需要 GUI 里切换到相机视角肉眼确认；如果仍然斜，下一步优先微调 observer robot 的 yaw 或 observer wrist camera roll。

**Next**
- GUI 中检查两臂是否都落在桌面长边上，observer arm 是否挡 actor 轨迹。
- 检查 `observer_wrist_rgb` 是否同时覆盖蓝色方块、红色目标区和 actor arm。

### 2026-06-08 16:22 CST - 新增 Static Observer Arm 与 Observer Wrist Camera

**Type:** env | viz | data

**Goal**
- 按后续双机械臂路线，先加入第二只静止 Franka，让它的腕部相机充当全局观察相机，补足 actor wrist camera 看不到初始方块和红色目标区域的问题。

**Setup**
- Env config：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/tasks/cube_pick_place/env_cfg.py`
- YAML draft：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/configs/cube_pick_place_env_draft.yaml`
- Actor robot scene name：`robot`
- Actor wrist camera：`{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam`
- Observer robot scene name：`observer_robot`
- Observer robot prim：`{ENV_REGEX_NS}/ObserverRobot`
- Observer robot init pose：
  - `pos=(0.05, -0.75, 0.0)`
  - `rot=(0.70711, 0.0, 0.0, 0.70711)`
- Observer wrist camera：`{ENV_REGEX_NS}/ObserverRobot/panda_hand/observer_wrist_cam`
- New observation term：`obs.policy.observer_wrist_rgb`

**Command**
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm
/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python -m py_compile \
  isaac_pick_place/tasks/cube_pick_place/env_cfg.py
```

GUI/runtime smoke 建议命令：

```bash
cd /home/ubuntu/Workspace/IsaacLab
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
TERM=xterm ./isaaclab.sh -p /home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py \
  --task Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0 \
  --num_envs 1 \
  --steps 2 \
  --enable_cameras \
  --device cuda:0 \
  --action-mode zero \
  --report /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/smoke_dual_arm_observer_report.txt
```

**Result**
- `py_compile` 通过。
- 新增 `observer_robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/ObserverRobot")`。
- `observer_robot` 第一版不接入 action space、reward、termination；当前 task action 仍只控制 actor `robot`。
- 新增 `observer_wrist_cam`，挂在 `observer_robot/panda_hand` 下，作为全局观察视角。
- `policy` observation 配置新增：
  - `wrist_rgb`
  - `observer_wrist_rgb`
- `image_obs_list` 更新为：
  - `wrist_cam`
  - `observer_wrist_cam`
- 2026-06-08 16:26 CST GUI/runtime smoke 已通过，`--steps 2` 正常跑完后退出：
  - `scene_keys=['terrain', 'robot', 'observer_robot', 'object', 'ee_frame', 'wrist_cam', 'observer_wrist_cam', 'table', 'plane', 'light', 'target_area']`
  - `scene_sensors=['ee_frame', 'wrist_cam', 'observer_wrist_cam']`
  - `obs.policy.wrist_rgb: shape=(1, 200, 200, 3), dtype=torch.uint8`
  - `obs.policy.observer_wrist_rgb: shape=(1, 200, 200, 3), dtype=torch.uint8`
  - `Action Manager` 仍只有 actor arm/gripper 的 7D action，不包含 observer arm action。

**Interpretation**
- 这不是普通 free-floating table camera，而是为后续双臂任务提前引入的静止 observer arm。
- Dataset 字段建议使用 `observation.image.actor_wrist` 和 `observation.image.observer_wrist`，以后第二臂开始动时语义仍然连续。
- 第一版训练仍只预测 actor action；observer arm 的用途是提供全局视觉上下文。
- 本次“显示一下就退出”是 smoke 脚本 `--steps 2` 的预期行为，不是崩溃。

**Next**
- 在 GUI 中确认第二臂位置不挡 actor scripted trajectory。
- 若 observer wrist 视角没有覆盖红区/方块，优先调 `observer_robot.init_state.pos/rot`，其次调 `observer_wrist_cam` 外参和 `focal_length`。

### 2026-06-08 15:22 CST - 修正 Wrist Camera 视角对准 TCP

**Type:** env | viz | debug

**Goal**
- 回答 wrist camera 挂载位置，并修正 GUI wrist camera 视角全流程看不到方块和红色区域的问题。

**Setup**
- Env config：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/tasks/cube_pick_place/env_cfg.py`
- YAML draft：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/configs/cube_pick_place_env_draft.yaml`
- Camera prim：`{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam`
- Mount parent：Franka `panda_hand`

**Command**
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm
/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python -m py_compile \
  isaac_pick_place/tasks/cube_pick_place/env_cfg.py
```

**Result**
- `py_compile` 通过。
- 确认当前 wrist camera 挂在 `panda_hand` link 下，不是 world/table camera。
- 旧参数来自 Isaac Lab stack visuomotor task：
  - `offset_pos=(0.13, 0.0, -0.15)`
  - `offset_rot_ros=(-0.70614, 0.03701, 0.03701, -0.70614)`
  - `focal_length=24.0`
  - `clipping_range=(0.1, 2.0)`
- 旧姿态在 `panda_hand` frame 中的 ROS forward 约为 `[-0.105, 0.0, 0.995]`，只轻微偏向 TCP，GUI wrist view 反馈全流程看不到方块和红色区域。
- 新参数：
  - `offset_pos=(0.13, 0.0, -0.15)`
  - `offset_rot_ros=(0.68734, -0.16602, -0.16602, 0.68734)`
  - `focal_length=18.0`
  - `clipping_range=(0.03, 2.0)`
- 新姿态在 `panda_hand` frame 中的 ROS forward 约为 `[-0.456, 0.0, 0.890]`，从手侧更明确地看向 TCP/夹爪中心；镜头也从 24mm 改为 18mm，视野更宽。

**Interpretation**
- 问题不是 observation 没接入，而是 camera extrinsic 对当前单 wrist-camera pick-place 任务不合适。
- 之前直接复用了 stack task 的 wrist camera，但 stack task 还有 `table_cam` 兜底；当前任务只有 `wrist_cam`，所以必须保证 eye-in-hand 视角能看到 cube 和 red target。

**Next**
- 在 GUI wrist camera 视图中重新跑 scripted trajectory，确认：
  - pre-grasp / descend / close 阶段能看到蓝色方块。
  - transport / release 阶段能看到红色目标区域，至少在释放前后进入视野。
- 如果仍看不到，下一步优先调 `offset_rot_ros` 的 aim 角度，再调 `focal_length` 或相机挂载位置。

### 2026-06-08 15:12 CST - 调整 Scripted Expert 抓取高度与半空释放

**Type:** env | debug | viz

**Goal**
- 修正 scripted expert 的两个轨迹问题：
  - 爪夹夹取位置偏方块上方，抓取 TCP 再往下。
  - 到红色区域上方后只下降到半空释放高度，打开夹爪让方块自由落到红色区域，而不是把方块怼到桌面再松开。

**Setup**
- Script：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/scripted_pick_place.py`
- Task：`Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0`
- Seed：`42`
- Num envs：`1`
- 关键默认参数更新：
  - `grasp_z: 0.025 -> 0.015`
  - 新增 `release_z: 0.085`
  - `place_z` 保留为旧命令兼容别名，会覆盖 `release_z`
- Phase 名称更新：
  - `descend_to_place -> descend_to_release`
  - `open_gripper -> release_gripper`

**Command**
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm
/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python -m py_compile \
  isaac_pick_place/scripts/scripted_pick_place.py

cd /home/ubuntu/Workspace/IsaacLab
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
TERM=xterm ./isaaclab.sh -p /home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/scripted_pick_place.py \
  --task Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0 \
  --num_envs 1 \
  --episodes 1 \
  --max-steps 1200 \
  --seed 42 \
  --headless \
  --enable_cameras \
  --device cuda:0 \
  --log-every 200 \
  --report /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/scripted_pick_place_report.txt

TERM=xterm ./isaaclab.sh -p /home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/scripted_pick_place.py \
  --task Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0 \
  --num_envs 1 \
  --episodes 1 \
  --max-steps 1200 \
  --seed 42 \
  --headless \
  --device cpu \
  --log-every 200 \
  --report /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/scripted_pick_place_cpu_report.txt
```

GUI 可视化建议命令：

```bash
cd /home/ubuntu/Workspace/IsaacLab
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
TERM=xterm ./isaaclab.sh -p /home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/scripted_pick_place.py \
  --task Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0 \
  --num_envs 1 \
  --episodes 1 \
  --max-steps 1200 \
  --seed 42 \
  --enable_cameras \
  --device cuda:0 \
  --log-every 25 \
  --report /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/scripted_pick_place_gui_report.txt
```

**Result**
- `py_compile` 通过。
- CUDA/headless runtime 未完成：当前会话报 `NVIDIA driver is not loaded` / `No CUDA GPUs are available`。
- CPU/no-camera runtime 也未完成：当前会话没有本地 Franka USD 缓存，并且无法访问 `https://omniverse-content-production.s3-us-west-2.amazonaws.com/.../panda_instanceable.usd`。
- 因此本次只完成代码级调整和静态检查；新的轨迹需要在有 GPU 和 Isaac 资产缓存的 GUI 会话里肉眼确认。

**Interpretation**
- 轨迹语义已经改成“更低抓取 + 红区上方半空释放”。
- `release_z` 是后续主要调参旋钮：如果方块落下偏移过大可以降低，例如 `--release-z 0.070`；如果仍像贴桌放置可以升高，例如 `--release-z 0.100`。
- `grasp_z` 是抓取高度旋钮：如果 GUI 看起来还偏高，可以试 `--grasp-z 0.012`。

**Next**
- 在 GUI 机器运行可视化命令，确认夹取位置和释放高度。
- 若 free-fall 后成功率下降，优先微调 `release_z`，再考虑增加目标上方等待或更慢下降。

### 2026-06-08 14:54 CST - Scripted Expert 完成 1 Env Pick-Place

**Type:** env | eval | viz

**Goal**
- 补一个可复用的 waypoint scripted expert，第一版目标是单 env 能稳定完成可视化 pick-place，并输出可复现 report。

**Setup**
- Script：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/scripted_pick_place.py`
- Report：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/scripted_pick_place_report.txt`
- Task：`Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0`
- Device：`cuda:0`
- Seed：`42`
- Num envs：`1`
- Max steps：`1200`
- 关键参数：`target_xy=(0.50, 0.22)`, `grasp_z=0.025`, `place_z=0.025`, `hover_z=0.22`, `lift_z=0.24`, `max_delta=0.018`, `arm_action_scale=0.5`

**Command**
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm
/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python -m py_compile \
  isaac_pick_place/scripts/scripted_pick_place.py

cd /home/ubuntu/Workspace/IsaacLab
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
TERM=xterm ./isaaclab.sh -p /home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/scripted_pick_place.py \
  --task Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0 \
  --num_envs 1 \
  --episodes 1 \
  --max-steps 1200 \
  --seed 42 \
  --headless \
  --enable_cameras \
  --device cuda:0 \
  --log-every 200 \
  --report /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/scripted_pick_place_report.txt
```

GUI 可视化命令：

```bash
cd /home/ubuntu/Workspace/IsaacLab
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
TERM=xterm ./isaaclab.sh -p /home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/scripted_pick_place.py \
  --task Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0 \
  --num_envs 1 \
  --episodes 1 \
  --max-steps 1200 \
  --seed 42 \
  --enable_cameras \
  --device cuda:0 \
  --log-every 25 \
  --report /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/scripted_pick_place_gui_report.txt
```

**Result**
- `py_compile` 通过。
- Headless scripted rollout 成功：`[SUMMARY] successes=1/1`。
- 成功触发 step：`952`。
- 最终成功前日志：
  - `cube_target_xy_error ~= 0.0175 m`
  - `cube_z ~= 0.0205 m`
  - `gripper_opening ~= 0.0795`
  - `success=True`
- 脚本按 phase 状态机执行：
  - `open_gripper_rest`
  - `move_above_cube`
  - `descend_to_grasp`
  - `close_gripper`
  - `lift_cube`
  - `move_above_red_target`
  - `descend_to_place`
  - `open_gripper`
  - `retreat`

**Interpretation**
- 第一版“GUI 里稳定完成 1 个 env 的 pick-place”的核心链路已经具备。
- Franka IK action 的前 6 维会被 action term 乘 `scale=0.5`，脚本已用 `--arm-action-scale 0.5` 反向补偿。
- 当前环境里夹爪 action 语义是 `+1` 打开、`-1` 闭合，脚本已按实际接口处理。
- Report 中的关键 metrics 改为 `env.step()` 前采样，避免 success termination 后自动 reset 污染最后一行状态读数。

**Next**
- 先用 GUI 命令肉眼确认轨迹和相机视角。
- 基于 `scripted_pick_place.py` 扩展 dataset collection/export。
- 后续再加随机 cube pose、多 episode、失败重试和更丰富 rollout metrics。

### 2026-06-08 14:42 CST - 替换 Red-Target Pick-Place Reward 与 Success Termination

**Type:** env | debug

**Goal**
- 把当前自定义环境从 Lift 的随机 `object_pose` 目标，改成固定红色目标区域的 pick-place reward 和 success 判定。

**Setup**
- Env config：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/tasks/cube_pick_place/env_cfg.py`
- MDP terms：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/tasks/cube_pick_place/mdp.py`
- Smoke 脚本：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py`
- Report：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/smoke_custom_env_report.txt`
- Task：`Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0`
- Device：`cuda:0`

**Command**
```bash
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm
/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python -m py_compile \
  isaac_pick_place/tasks/cube_pick_place/env_cfg.py \
  isaac_pick_place/tasks/cube_pick_place/mdp.py

cd /home/ubuntu/Workspace/IsaacLab
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
TERM=xterm ./isaaclab.sh -p /home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py \
  --task Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0 \
  --num_envs 1 \
  --steps 3 \
  --headless \
  --enable_cameras \
  --device cuda:0 \
  --action-mode zero \
  --report /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/smoke_custom_env_report.txt
```

**Result**
- `py_compile` 通过。
- 第一次 runtime smoke 暴露了 Isaac Lab manager 不接受 `**kwargs` 的严格签名问题；已把 reward/termination 函数参数显式展开。
- 修正后 headless smoke 成功，输出 `[OK] Smoke test completed.`。
- Command Manager 现在为 `0 active terms`，不再使用 Lift 的随机 `object_pose` command。
- Observation Manager 中 `policy` 组包含：
  - `joint_pos`
  - `joint_vel`
  - `object_position`
  - `actions`
  - `target_area_position`
  - `wrist_rgb`
- Termination Manager 中包含：
  - `time_out`
  - `object_dropping`
  - `success`
- Reward Manager 中包含：
  - `reaching_object`
  - `lifting_object`
  - `object_goal_tracking`
  - `object_goal_tracking_fine_grained`
  - `action_rate`
  - `joint_vel`
  - `placed_on_target`
- Report 中实际 observation：
  - `obs.policy.target_area_position: shape=(1, 3), dtype=torch.float32`
  - `obs.policy.wrist_rgb: shape=(1, 200, 200, 3), dtype=torch.uint8`

**Interpretation**
- 环境已经从“Lift smoke 环境”推进到“固定红色目标区域 pick-place 环境”的第一版。
- Success 判定现在要求方块位于红色区域、接近桌面高度、夹爪释放，并连续满足 `10` step 后才触发 `success` termination。
- 当前 smoke 用 zero action，只验证 manager 配置和 step 链路，不验证策略能完成任务。

**Next**
- 写 scripted expert，让 GUI/数据采集里出现真正的抓取、移动、释放轨迹。

### 2026-06-08 14:31 CST - Wrist RGB Observation 运行时验证通过

**Type:** env | debug

**Goal**
- 验证 `wrist_rgb` 不只是配置里存在，而是真的在 Isaac camera pipeline 启动后进入 `obs["policy"]`。

**Setup**
- Env config：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/tasks/cube_pick_place/env_cfg.py`
- Smoke 脚本：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py`
- Report：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/smoke_custom_env_report.txt`
- Task：`Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0`
- Device：`cuda:0`
- Camera：`--enable_cameras`

**Command**
```bash
cd /home/ubuntu/Workspace/IsaacLab
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
TERM=xterm ./isaaclab.sh -p /home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py \
  --task Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0 \
  --num_envs 1 \
  --steps 2 \
  --headless \
  --enable_cameras \
  --device cuda:0 \
  --action-mode zero \
  --report /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/smoke_custom_env_report.txt
```

**Result**
- Smoke 运行成功，输出 `[OK] Smoke test completed.`。
- Observation Manager 中 `policy` 组包含：
  - `joint_pos`
  - `joint_vel`
  - `object_position`
  - `target_object_position`
  - `actions`
  - `wrist_rgb`
- Report 中实际 reset observation：
  - `obs.policy.wrist_rgb: shape=(1, 200, 200, 3), dtype=torch.uint8`
- `scene_sensors=['ee_frame', 'wrist_cam']`，说明 wrist camera 已经作为 sensor 进入场景。

**Interpretation**
- 腕部 RGB 图像已经正式接入 policy observation。
- 现在 observation 是非拼接 dict 结构，适合后续导出成 HF/LeRobot 风格的 `observation.image.wrist` + 低维状态。

**Next**
- 下一步可以开始替换 Lift 的 reward/termination，写红色目标区域 pick-place 的 success 判定。

### 2026-06-05 - 接入 Wrist RGB Observation 配置

**Type:** env | data

**Goal**
- 将已经挂在 scene 里的 `wrist_cam` RGB 图像接入 policy observation，使环境后续能输出视觉输入。

**Setup**
- Env config：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/tasks/cube_pick_place/env_cfg.py`
- Smoke 脚本：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py`
- Task：`Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0`

**Command**
```bash
# 静态检查
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm
/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python -m py_compile \
  isaac_pick_place/tasks/cube_pick_place/env_cfg.py \
  isaac_pick_place/scripts/smoke_lift_env.py

# 运行时验证命令，需启动 Isaac camera pipeline
cd /home/ubuntu/Workspace/IsaacLab
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
TERM=xterm ./isaaclab.sh -p /home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py \
  --task Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0 \
  --num_envs 1 \
  --steps 2 \
  --headless \
  --enable_cameras \
  --device cuda:0 \
  --action-mode zero \
  --report /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/smoke_custom_env_report.txt
```

**Expected Runtime Evidence**
- Report 中应出现：
  - `obs.policy: dict keys=[...]`
  - `obs.policy.wrist_rgb: shape=(1, 200, 200, 3)`
  - `dtype=torch.uint8` 或相机返回的等价 RGB dtype

**Result**
- 新增 `PickPlaceObservationsCfg`。
- `policy.concatenate_terms = False`，避免把图像和低维状态强行拼接。
- `policy.wrist_rgb = ObsTerm(func=base_mdp.image, sensor_cfg="wrist_cam", data_type="rgb", normalize=False)`。
- 低维 Lift 观测项仍保留为命名字段。
- `py_compile` 通过。
- 本次由助手启动 Isaac/Camera runtime smoke 时被系统额度限制拦截，尚未获得运行时 image shape 证据。

**Interpretation**
- 配置层面已经接入 `wrist_rgb`。
- 完整完成还需要运行上面的 smoke 命令，确认 observation report 里真的出现 `obs.policy.wrist_rgb`。

**Next**
- 运行 smoke 命令验证图像 shape；通过后再实现 red-target success 判定。

### 2026-06-05 16:52 CST - 处理 GUI 关闭时 PhysX Tensor View 失效

**Type:** debug | viz

**Goal**
- 处理 `--steps -1` 长开 GUI 时，关闭窗口可能触发的 PhysX tensor view invalidation traceback。

**Setup**
- Smoke 脚本：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py`
- 触发方式：GUI 长开后关闭 Isaac Sim 窗口，环境可能还在 `env.step()` 中读取关节速度。

**Observed Error**
```text
prim '/World/envs/env_0/Object/geometry/mesh' was deleted while being used by a shape in a tensor view class.
Simulation view object is invalidated and cannot be used again to call getDofVelocities
Exception: Failed to get DOF velocities from backend
```

**Result**
- 在 smoke 脚本中加入 `_is_shutdown_race_error()`。
- 对以下关闭窗口相关错误做优雅退出处理：
  - `Simulation view object is invalidated`
  - `Failed to get DOF velocities from backend`
  - `physics.tensors simulationView was invalidated`
  - `was deleted while being used by a shape in a tensor view class`
- 其他未知异常仍然会被重新抛出，避免掩盖真实环境错误。
- `py_compile` 通过。

**Interpretation**
- 这是 GUI/stage 关闭时的退出竞态，不是蓝方块或红色目标区本身损坏。
- `gpu.foundation.plugin ... IMemoryBudgetManagerFactory` 是 performance warning，暂时不影响实验。

**Next**
- 后续长开 GUI 可以继续用 `--steps -1`；关闭窗口时应写 report 并干净退出。

### 2026-06-05 16:43 CST - 关闭画面中的 Command Pose 坐标轴

**Type:** env | viz

**Goal**
- 关闭 GUI 画面中显眼的两个坐标轴，避免后续视觉观察和采集数据被 debug marker 污染。

**Setup**
- Env config：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/tasks/cube_pick_place/env_cfg.py`
- Task：`Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0`
- Device：`cuda:0`

**Command**
```bash
# Headless smoke
cd /home/ubuntu/Workspace/IsaacLab
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
TERM=xterm ./isaaclab.sh -p /home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py \
  --task Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0 \
  --num_envs 1 \
  --steps 1 \
  --headless \
  --enable_cameras \
  --device cuda:0 \
  --action-mode zero \
  --report /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/smoke_custom_env_report.txt
```

**可视化命令**
```bash
cd /home/ubuntu/Workspace/IsaacLab
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

DISPLAY=:1 TERM=xterm ./isaaclab.sh -p /home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py \
  --task Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0 \
  --num_envs 1 \
  --steps 1200 \
  --enable_cameras \
  --device cuda:0 \
  --action-mode zero \
  --report /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/smoke_custom_env_gui_report.txt
```

**Result**
- 在 `CubePickPlaceRedTargetFrankaIKRelVisuomotorEnvCfg.__post_init__` 中加入：
  - `self.commands.object_pose.debug_vis = False`
- `py_compile` 通过。
- Headless smoke 成功，输出 `[OK] Smoke test completed.`。
- GUI smoke 成功，输出 `[OK] Smoke test completed.`。
- 本次日志中不再出现之前与 command pose marker 相关的 `/Visuals/Command/body_pose` 和 `/Visuals/Command/goal_pose` point-instancer warning。

**Interpretation**
- 两个显眼坐标轴来自 Lift 默认 `object_pose` command debug visualization。
- 关闭后画面更干净，适合后续接入 wrist RGB observation 和采集演示数据。

**Next**
- 继续接入 wrist RGB observation。

### 2026-06-05 16:38 CST - 修正目标区高度并替换纯蓝色方块

**Type:** env | viz | debug

**Goal**
- 修正两个视觉问题：
  - 红色目标区域看起来漂浮在桌面上。
  - 默认 DexCube 方块颜色/纹理较花，可能污染后续视觉训练。

**Setup**
- Env config：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/tasks/cube_pick_place/env_cfg.py`
- 草案文件：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/configs/cube_pick_place_env_draft.yaml`
- Task：`Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0`
- Device：`cuda:0`

**Command**
```bash
# Headless smoke
cd /home/ubuntu/Workspace/IsaacLab
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
TERM=xterm ./isaaclab.sh -p /home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py \
  --task Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0 \
  --num_envs 1 \
  --steps 1 \
  --headless \
  --enable_cameras \
  --device cuda:0 \
  --action-mode zero \
  --report /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/smoke_custom_env_report.txt
```

**可视化命令**
```bash
cd /home/ubuntu/Workspace/IsaacLab
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

DISPLAY=:1 TERM=xterm ./isaaclab.sh -p /home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py \
  --task Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0 \
  --num_envs 1 \
  --steps 1200 \
  --enable_cameras \
  --device cuda:0 \
  --action-mode zero \
  --report /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/smoke_custom_env_gui_report.txt
```

**Result**
- 将 `scene.object` 从 textured `DexCube` 替换为纯蓝色 `sim_utils.CuboidCfg`：
  - size：`0.04 x 0.04 x 0.04 m`
  - mass：`0.05 kg`
  - color：`diffuse_color=(0.0, 0.15, 1.0)`
- 将 `target_area` 从较高的视觉块下压为桌面薄片：
  - center：`(0.50, 0.22, 0.0015)`
  - size：`0.12 x 0.12 x 0.001 m`
  - collision：disabled
- Headless smoke 成功，输出 `[OK] Smoke test completed.`。
- GUI smoke 成功，输出 `Creating window for environment.` 和 `[OK] Smoke test completed.`。
- Scene keys 仍包含：
  - `object`
  - `target_area`
  - `wrist_cam`
- YAML 解析通过。

**Interpretation**
- 环境创建没有被视觉资产替换破坏。
- 后续视觉训练输入会更干净：方块是纯蓝色，红色目标区域贴近桌面。
- 仍建议用 GUI 命令肉眼确认红色薄片是否有轻微 z-fighting；如果出现闪烁，再把 `target_area` 的 `z` 从 `0.0015` 微调到 `0.0020`。

**Next**
- 你确认 GUI 视觉效果后，再接入 wrist RGB observation。

### 2026-06-05 16:35 CST - 注册并创建自定义 Red Target Pick-and-Place 环境

**Type:** env | debug | viz

**Goal**
- 将 `cube_pick_place_env_draft.yaml` 的核心设计落成真正的 Isaac Lab Python env config module。
- 先做到自定义 task id 可以被 gym 注册、`parse_env_cfg` 解析、`gym.make` 创建、`reset/step` 正常运行。

**Setup**
- 自定义 task id：`Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0`
- 注册入口：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/tasks/cube_pick_place/__init__.py`
- Env config：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/tasks/cube_pick_place/env_cfg.py`
- Smoke 脚本：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py`
- Isaac Lab 路径：`/home/ubuntu/Workspace/IsaacLab`
- Conda 环境：`env_isaaclab`
- Device：`cuda:0`

**Command**
```bash
# 注册检查
cd /home/ubuntu/Workspace/seven_dof_pick_place_lbm
/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python - <<'PY'
import sys
sys.path.insert(0, '/home/ubuntu/Workspace/seven_dof_pick_place_lbm')
import isaac_pick_place.tasks  # noqa
import gymnasium as gym
spec = gym.spec('Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0')
print(spec.id)
print(spec.entry_point)
print(spec.kwargs['env_cfg_entry_point'])
PY

# Headless 创建/reset/step 检查
cd /home/ubuntu/Workspace/IsaacLab
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
TERM=xterm ./isaaclab.sh -p /home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py \
  --task Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0 \
  --num_envs 1 \
  --steps 1 \
  --headless \
  --enable_cameras \
  --device cuda:0 \
  --action-mode zero \
  --report /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/smoke_custom_env_report.txt
```

**可视化命令**
```bash
cd /home/ubuntu/Workspace/IsaacLab
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

DISPLAY=:1 TERM=xterm ./isaaclab.sh -p /home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py \
  --task Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0 \
  --num_envs 1 \
  --steps 1200 \
  --enable_cameras \
  --device cuda:0 \
  --action-mode zero \
  --report /home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/smoke_custom_env_gui_report.txt
```

**Result**
- Gym 注册检查通过：
  - id：`Isaac-Cube-Pick-Place-Red-Target-Franka-IK-Rel-Visuomotor-v0`
  - entry point：`isaaclab.envs:ManagerBasedRLEnv`
  - env cfg entry point：`isaac_pick_place.tasks.cube_pick_place.env_cfg:CubePickPlaceRedTargetFrankaIKRelVisuomotorEnvCfg`
- Headless smoke 成功，输出 `[OK] Smoke test completed.`。
- GUI smoke 成功，Isaac Lab 输出 `Creating window for environment.`，并正常跑到 1200 step 后退出。
- Headless 报告文件：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/smoke_custom_env_report.txt`
- GUI 报告文件：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/smoke_custom_env_gui_report.txt`
- Observation space：`Dict('policy': Box(-inf, inf, (1, 35), float32))`
- Action space：`Box(-inf, inf, (1, 7), float32)`
- Scene keys：
  - `terrain`
  - `robot`
  - `object`
  - `ee_frame`
  - `wrist_cam`
  - `table`
  - `plane`
  - `light`
  - `target_area`
- Scene sensors：
  - `ee_frame`
  - `wrist_cam`

**Interpretation**
- 自定义环境已经不是草案：现在可以被 Isaac Lab 注册、创建和 step。
- 红色目标区 `target_area` 和腕部相机 `wrist_cam` 都已经进入场景。
- 当前 policy observation 仍沿用 Lift 的 35 维低维观测；`wrist_cam` 已经作为 sensor 存在，但还没有接入 policy observation 输出。
- 当前 reward/termination 仍沿用 Lift；真正的“方块稳定放入红色区域”success/reward 还没有实现。

**Next**
- 接入 wrist RGB observation，确认 `obs["policy"]` 或独立 observation group 中能拿到图像张量。
- 写 red target success 判定：方块中心落在目标区域、释放、稳定若干步。
- 再写 scripted expert，可视化有意义的抓取/放置轨迹。

### 2026-06-05 16:27 CST - Franka Lift IK GUI 可视化烟测

**Type:** env | viz | debug

**Goal**
- 用 Isaac Lab 原生 GUI 可视化现成 Franka Lift IK smoke 环境，确认图形窗口、渲染、场景加载和 step 循环都正常。

**Setup**
- 代码路径：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py`
- Isaac Lab 路径：`/home/ubuntu/Workspace/IsaacLab`
- Conda 环境：`env_isaaclab`
- Task：`Isaac-Lift-Cube-Franka-IK-Rel-v0`
- Display：`:1`
- Device：`cuda:0`
- 关键配置：`num_envs=1`，`steps=2000`，`action_mode=random`，GUI 非 headless

**Command**
```bash
cd /home/ubuntu/Workspace/IsaacLab
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
DISPLAY=:1 TERM=xterm ./isaaclab.sh -p /home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py \
  --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num_envs 1 \
  --steps 2000 \
  --device cuda:0 \
  --action-mode random
```

**Result**
- GUI 启动成功，Isaac Lab 输出 `Creating window for environment.`。
- 脚本运行到第 2000 步后正常退出，输出 `[OK] Smoke test completed.`。
- 报告文件：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/smoke_lift_env_report.txt`
- Observation space：`Dict('policy': Box(-inf, inf, (1, 35), float32))`
- Action space：`Box(-inf, inf, (1, 7), float32)`
- Step 1 reward：`0.3461155593395233`
- Step 2000 reward：`-4.5150958612794057e-05`
- Step 2000 状态：`terminated=False`，`truncated=True`，符合 episode time limit 结束预期。

**Interpretation**
- 原生 GUI 可视化链路已经可用。
- 这次使用 random action，只用于确认可视化和环境运行，不代表专家策略或模型表现。
- 后续如果要更适合肉眼检查，需要用 state machine/scripted expert，而不是 random action。

**Next**
- 增加一个 scripted pick/lift 或 pick/place 可视化脚本，让 GUI 里看到有意义的抓取动作。

### 2026-06-05 16:23 CST - Pick-and-Place 环境派生路线草案

**Type:** design | env

**Goal**
- 确定自定义红色目标区 pick-and-place 环境应该从 Lift 还是 Stack Visuomotor 派生，并形成第一版配置草案。

**Setup**
- 草案文件：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/configs/cube_pick_place_env_draft.yaml`
- 参考 Lift 配置：
  - `/home/ubuntu/Workspace/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/ik_rel_env_cfg.py`
  - `/home/ubuntu/Workspace/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/lift_env_cfg.py`
- 参考 Stack Visuomotor 配置：
  - `/home/ubuntu/Workspace/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/stack_ik_rel_visuomotor_env_cfg.py`
  - `/home/ubuntu/Workspace/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/stack_env_cfg.py`

**Command**
```bash
sed -n '1,260p' source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/joint_pos_env_cfg.py
sed -n '1,280p' source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/lift_env_cfg.py
sed -n '1,280p' source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/stack_ik_rel_visuomotor_env_cfg.py
sed -n '1,260p' source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/stack_joint_pos_env_cfg.py
sed -n '1,280p' source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/stack_env_cfg.py
```

**Result**
- 草案文件已创建：`isaac_pick_place/configs/cube_pick_place_env_draft.yaml`
- 结论：第一版自定义环境以 Lift IK 为主基底。
- 从 Lift 借用：
  - 单方块、桌面、Franka、Panda gripper 的简单场景结构。
  - `FRANKA_PANDA_HIGH_PD_CFG`
  - `DifferentialInverseKinematicsActionCfg`
  - 7D action：6D arm relative pose + 1D gripper。
- 从 Stack Visuomotor 借用：
  - wrist camera 的 `CameraCfg`
  - 非拼接 policy observation group
  - `eef_pos/eef_quat/gripper_pos/last_action` 等状态观测组织方式。
- 暂不继承：
  - table camera
  - 三方块 stack 任务逻辑
  - 视觉域随机化
  - 多 subtask observation groups

**Interpretation**
- 这样比直接改 Stack Visuomotor 更稳，因为我们的任务只有一个方块和一个红色目标区域。
- 同时比纯 Lift 更接近最终需求，因为后续会补入 wrist camera 和 image observation。

**Next**
- 根据草案创建真正的 Python env config module，并先做到能注册/创建环境，不急着写 scripted expert。

### 2026-06-05 16:21 CST - Franka Lift IK 环境 Headless 烟测

**Type:** env | debug

**Goal**
- 验证 Isaac Lab 现成 Franka relative IK lift 环境能否在本机 headless 模式下创建、reset、step，并确认 observation/action space。

**Setup**
- 代码路径：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py`
- Isaac Lab 路径：`/home/ubuntu/Workspace/IsaacLab`
- Conda 环境：`env_isaaclab`
- Task：`Isaac-Lift-Cube-Franka-IK-Rel-v0`
- 机器人资产：Isaac Lab Franka Panda，高 PD 配置，relative differential IK
- 相机：无，本次只测低维 Lift 环境
- Seed：未设置，Isaac Lab 日志提示 environment seed 为 `None`
- Device：`cuda:0`
- 关键配置：`num_envs=1`，`steps=5`，`headless=True`

**Command**
```bash
cd /home/ubuntu/Workspace/IsaacLab
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
TERM=xterm ./isaaclab.sh -p /home/ubuntu/Workspace/seven_dof_pick_place_lbm/isaac_pick_place/scripts/smoke_lift_env.py \
  --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num_envs 1 \
  --steps 5 \
  --headless \
  --device cuda:0
```

**Result**
- 运行成功，脚本输出 `[OK] Smoke test completed.`。
- 调试中遇到两个启动前置问题：
  - 未设置 `TERM=xterm` 时，`./isaaclab.sh` 在 `tabs` 调用处失败：`terminal type 'dumb' cannot reset tabs`。
  - 未激活 `env_isaaclab` 时，`./isaaclab.sh` 使用 base Python，报错：`ModuleNotFoundError: No module named 'isaaclab'`。
- 报告文件：`/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/smoke_lift_env_report.txt`
- Observation space：`Dict('policy': Box(-inf, inf, (1, 35), float32))`
- Action space：`Box(-inf, inf, (1, 7), float32)`
- Active action terms：
  - `arm_action`: 6D
  - `gripper_action`: 1D
- Active observation terms in policy group：
  - `joint_pos`: 9D
  - `joint_vel`: 9D
  - `object_position`: 3D
  - `target_object_position`: 7D
  - `actions`: 7D
- Step 1 reward：`0.3350559175014496`
- Step 5 reward：`1.477122259530006e-05`
- 没有 terminated/truncated。

**Interpretation**
- Isaac Lab runtime、Franka Panda 资产、relative IK action 和夹爪 action 都能正常工作。
- 这个 Lift 环境适合作为第一版工程烟测和控制接口参考。
- 当前 Lift 环境没有手腕图像观测；视觉部分需要后续从 Stack Visuomotor 或 Isaac Lab camera sensor 配置中引入。
- action 维度正好符合当前设计的 7D：6D 末端位姿增量 + 1D 夹爪控制。

**Next**
- 对比 Lift IK 和 Stack IK Visuomotor 的配置文件，确定我们的 `cube_pick_place_red_target` 环境从哪个配置派生最少改动。
- 下一步先写环境配置草案，不急着训练模型。

### 2026-06-05 16:12 CST - Isaac Lab 和 Franka 资产清点

**Type:** env

**Goal**
- 确认第一步应该基于哪套 Isaac Sim/Isaac Lab 入口、机器人资产和控制接口开始。

**Setup**
- Project path: `/home/ubuntu/Workspace/seven_dof_pick_place_lbm`
- Isaac Lab path: `/home/ubuntu/Workspace/IsaacLab`
- Conda env: `/home/ubuntu/miniconda3/envs/env_isaaclab`
- Design source: `/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/SEVEN_DOF_GRASP_EXPERIMENT_PLAN.md`

**Command**
```bash
find /home/ubuntu -maxdepth 4 \( -iname '*isaac*' -o -iname '*Isaac*' \) 2>/dev/null
rg -n "FRANKA|Franka|Panda|PANDA|franka" source apps scripts -S
sed -n '1,220p' source/isaaclab_assets/isaaclab_assets/robots/franka.py
sed -n '1,140p' source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/ik_rel_env_cfg.py
sed -n '1,130p' source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/__init__.py
```

**Result**
- 本机已有 Isaac Lab：`/home/ubuntu/Workspace/IsaacLab`。
- 本机已有 Isaac Lab conda 环境：`/home/ubuntu/miniconda3/envs/env_isaaclab`。
- 该环境中存在 `isaacsim` 可执行入口。
- Isaac Lab 提供 Franka Panda 配置：
  - `FRANKA_PANDA_CFG`
  - `FRANKA_PANDA_HIGH_PD_CFG`
  - USD: `{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd`
- `FRANKA_PANDA_HIGH_PD_CFG` 是为 task-space / differential IK 跟踪调硬的版本。
- 现成 IK 相对控制参考任务：
  - `Isaac-Lift-Cube-Franka-IK-Rel-v0`
  - `Isaac-Stack-Cube-Franka-IK-Rel-v0`
  - `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0`
- `Isaac-Lift-Cube-Franka-IK-Rel-v0` 使用：
  - `DifferentialInverseKinematicsActionCfg`
  - `DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")`
  - `body_name="panda_hand"`

**Interpretation**
- 第一版不应该从裸 Isaac Sim 场景从零写起。
- 最稳路线是基于 Isaac Lab 的 Franka + differential IK 相对位姿控制任务改造。
- pick-and-place 任务优先参考 stack visuomotor 环境，因为它已经更接近“视觉 + 方块 + 放置/堆叠”的任务形态。

**Next**
- 写一个最小环境设计草案：选 `FRANKA_PANDA_HIGH_PD_CFG`、relative IK action、白色桌面、红色目标区域、单 wrist camera，并确定从 lift 还是 stack visuomotor 配置派生。

### 2026-06-05 16:02 CST - 通过 NVIDIA R580 驱动修复 IsaacLab 原生 GUI 渲染

**Type:** env | debug | viz

**Goal**
- 解决 IsaacLab/Isaac Sim 原生 GUI 模式启动后在 RTX renderer 处段错误的问题，使后续七自由度抓取放置实验可以进行可视化检查。

**Setup**
- Code path: `/home/ubuntu/Workspace/IsaacLab`
- IsaacLab：v2.3.2 repo，Python package `isaaclab-0.54.2`
- Isaac Sim：5.1.0 pip install，Kit 107.3.3
- Conda 环境：`env_isaaclab`
- GPU：RTX 5090 D v2；降级驱动后 `nvidia-smi` 显示为 `NVIDIA Graphics Device`
- 修复前驱动：`595.71.05`
- 修复后驱动：`580.65.06`，`nvidia-smi` 显示 CUDA Version `13.0`
- OS/kernel：Ubuntu 22.04.5，`6.8.0-124-generic`
- Display：Xorg display `:1`
- Device: `cuda:0`
- 关键配置：原生 GUI，`--rendering_mode performance`

**Command**
```bash
# 降级驱动前失败的 GUI 烟测
cd /home/ubuntu/Workspace/IsaacLab
conda activate env_isaaclab
DISPLAY=:1 ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --rendering_mode performance

# 安装固定版本的 NVIDIA R580 驱动包
cd /home/ubuntu
./install_nvidia_580_65_06.sh

# 修复 gcc 版本不匹配导致的 DKMS 编译失败
./repair_nvidia_580_dkms.sh

# 重启并验证驱动版本
sudo reboot
nvidia-smi

# 最终 GUI 烟测
cd /home/ubuntu/Workspace/IsaacLab
conda activate env_isaaclab
DISPLAY=:1 ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --rendering_mode performance
```

**Result**
- 修复前，原生 GUI 在仿真脚本进入有效运行前反复段错误。
- crash stack 稳定指向 RTX 渲染路径：
  - `librtx.scenedb.plugin.so`
  - `libcarb.scenerenderer-rtx.plugin.so`
  - `libomni.hydra.rtx.plugin.so`
- Headless 模式正常，因此问题被定位到原生 GUI/RTX renderer 初始化，而不是 Python task 脚本本身。
- 将 NVIDIA 驱动从 `595.71.05` 降级到 Isaac Sim 5.1 测试过的 R580 分支，并固定到 `580.65.06`。
- 第一次驱动安装后，`nvidia-dkms-580-open` 处于半配置状态，原因是 DKMS 使用 `gcc-11`，而当前内核由 `gcc-12` 编译。
- DKMS 报错：
```text
cc: error: unrecognized command-line option '-ftrivial-auto-var-init=zero'
The kernel was built by: x86_64-linux-gnu-gcc-12
You are using: cc gcc-11
```
- 安装 `gcc-12 g++-12` 后重新执行 package configuration，DKMS 编译成功：
```text
nvidia/580.65.06, 6.5.0-18-generic, x86_64: installed
nvidia/580.65.06, 6.8.0-124-generic, x86_64: installed
```
- 重启后 `nvidia-smi` 显示：
```text
NVIDIA-SMI 580.65.06
Driver Version: 580.65.06
CUDA Version: 13.0
```
- 最终 IsaacLab GUI 烟测成功：
```text
Simulation App Startup Complete
[INFO][IsaacLab]: Logging to file: /tmp/isaaclab/logs/isaaclab_2026-06-05_16-02-14.log
[INFO]: Setup complete...
```

**Interpretation**
- 原始 native GUI crash 很可能是 Isaac Sim 5.1 的 RTX renderer stack 与当前 GPU/系统上的 NVIDIA driver `595.71.05` 存在兼容性问题或回归。
- 将驱动对齐到 Isaac Sim 5.1 测试过的 R580 分支后，`librtx.scenedb.plugin.so` crash 消失。
- 降级后 `nvidia-smi` 中的 `NVIDIA Graphics Device` 名称大概率是旧驱动对新设备 ID 的名称识别不完整；Xorg、显存识别和 IsaacLab 原生 GUI 都已可用。
- CPU powersave 和 IOMMU 仍只是性能提醒，不是启动阻塞项。

**Next**
- 后续数据采集前，先用 native GUI smoke test 检查场景、相机和资产加载。
- 除非有意识测试新驱动，否则保持 R580 驱动包 hold 状态。
- 如果后续 `apt upgrade` 触碰 NVIDIA 包，长实验前重新验证 `dkms status`、`nvidia-smi` 和 IsaacLab GUI smoke test。

### 2026-06-05 - 初始化实验日志格式

**Type:** design

**Goal**
- 在开始 Isaac Sim 实验前，先确定后续实验记录格式。

**Setup**
- Project path: `/home/ubuntu/Workspace/seven_dof_pick_place_lbm`
- Design source: `/home/ubuntu/Workspace/seven_dof_pick_place_lbm/experiments/SEVEN_DOF_GRASP_EXPERIMENT_PLAN.md`

**Command**
```bash
# No experiment command yet.
```

**Result**
- 创建实验日志结构和维护规则。

**Interpretation**
- 只要后续每条记录都保留命令、配置、指标和产物路径，实验之间就能比较和回溯。

**Next**
- 定义第一版 Isaac Sim 环境配置和 scripted expert 方案。
