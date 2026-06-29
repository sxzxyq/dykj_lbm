# Scripts README

```text
专家策略采集 raw demos
  -> 转换为 LeRobotDataset
    -> 训练 HF/LeRobot MultiTask DiT
      -> open-loop / closed-loop / hybrid 评估诊断
```

## 目录结构

```text
isaac_pick_place/scripts/
├── README.md
├── collect/
│   ├── collect_and_train_handoff_v2_full_200.sh
│   ├── collect_handoff_demos.sh
│   ├── collect_handoff_demos_to_lerobot.sh
│   ├── collect_handoff_v2_clean_full_to_lerobot.sh
│   ├── collect_random_demos_to_lerobot.sh
│   ├── scripted_handoff_collect.py
│   └── scripted_pick_place.py
├── convert/
│   ├── convert_and_train_handoff_simple_state_two_models.sh
│   ├── convert_handoff_raw_demos_to_lerobot.py
│   ├── convert_handoff_simple_state_datasets.sh
│   ├── convert_raw_demos_to_lerobot.py
│   └── repair_lerobot_image_stats.py
├── train/
│   ├── train_handoff_256_mtdp.sh
│   ├── train_handoff_birelpose_time_256_mtdp.sh
│   ├── train_handoff_relpose_256_mtdp.sh
│   ├── train_handoff_simple_state_absjoint18_256_mtdp.sh
│   ├── train_handoff_simple_state_delta14_256_mtdp.sh
│   ├── train_hf_mtdp_smoke.py
│   └── train_random_cube_256_mtdp.sh
├── eval/
│   ├── diagnose_handoff_hybrid_eval.py
│   ├── diagnose_handoff_hybrid_eval.sh
│   ├── diagnose_handoff_step0_policy.py
│   ├── diagnose_handoff_step0_policy.sh
│   ├── eval_pick_place_policy.py
│   ├── eval_pick_place_policy.sh
│   ├── openloop_eval_policy.py
│   └── replay_handoff_raw_demo.py
├── tools/
│   ├── open_tensorboard_live_chromium.sh
│   ├── smoke_lift_env.py
│   └── tensorboard_auto_refresh_wrapper.html
├── common/
│   └── handoff_v2_utils.py
└── __pycache__/
    Python 运行/编译产生的缓存目录，不是源码入口。
```

## 主线流程

```text
单臂 pick-place
├── collect/collect_random_demos_to_lerobot.sh
│   ├── collect/scripted_pick_place.py
│   └── convert/convert_raw_demos_to_lerobot.py
├── train/train_random_cube_256_mtdp.sh
│   └── train/train_hf_mtdp_smoke.py
└── eval/eval_pick_place_policy.sh
    └── eval/eval_pick_place_policy.py

双臂 handoff V2 主线
├── collect/collect_and_train_handoff_v2_full_200.sh
│   ├── collect/collect_handoff_v2_clean_full_to_lerobot.sh
│   │   ├── collect/collect_handoff_demos_to_lerobot.sh
│   │   │   ├── collect/scripted_handoff_collect.py
│   │   │   └── convert/convert_handoff_raw_demos_to_lerobot.py
│   │   └── convert/convert_handoff_raw_demos_to_lerobot.py
│   └── train/train_handoff_birelpose_time_256_mtdp.sh
│       └── train/train_hf_mtdp_smoke.py
├── eval/eval_pick_place_policy.sh
│   └── eval/eval_pick_place_policy.py
├── eval/openloop_eval_policy.py
└── eval/diagnose_handoff_*.sh / eval/diagnose_handoff_*.py

双臂 handoff 简化状态实验
├── convert/convert_and_train_handoff_simple_state_two_models.sh
│   ├── convert/convert_handoff_simple_state_datasets.sh
│   │   └── convert/convert_handoff_raw_demos_to_lerobot.py
│   ├── train/train_handoff_simple_state_delta14_256_mtdp.sh
│   │   └── train/train_hf_mtdp_smoke.py
│   └── train/train_handoff_simple_state_absjoint18_256_mtdp.sh
│       └── train/train_hf_mtdp_smoke.py
└── eval/openloop_eval_policy.py
```

## 脚本含义

```text
collect/
├── scripted_pick_place.py
│   单臂 Franka waypoint 专家。执行抓取蓝色方块并放到红色目标区，可记录 raw demos。
├── scripted_handoff_collect.py
│   双臂 handoff waypoint 专家。右臂放到黄色交接区，左臂再放到红色目标区，可记录 raw demos 和相机帧。
├── collect_random_demos_to_lerobot.sh
│   单臂一键采集并转换为 LeRobotDataset。
├── collect_handoff_demos.sh
│   只采集双臂 handoff raw demos，不做转换。
├── collect_handoff_demos_to_lerobot.sh
│   双臂 handoff 采集并转换为 LeRobotDataset，可配置 cube size、随机化、state layout、action representation。
├── collect_handoff_v2_clean_full_to_lerobot.sh
│   Handoff V2 数据集流水线。可选 clean-control，主要生成 full raw，并拆成 train/val。
└── collect_and_train_handoff_v2_full_200.sh
    当前 V2 主流水线。默认采集 200 个成功 demo，转换为 train180/val20，然后训练 49D 模型。

convert/
├── convert_raw_demos_to_lerobot.py
│   单臂 raw demo 转 LeRobotDataset v3，写 state/action/三路图像，并做 reload check。
├── convert_handoff_raw_demos_to_lerobot.py
│   双臂 handoff raw demo 转 LeRobotDataset。支持多种 state layout、14D EE delta、18D absolute joint action、manifest 和 action stats。
├── convert_handoff_simple_state_datasets.sh
│   从已有 Handoff V2 raw demo 生成两套简化状态数据集：26D+14D delta、26D+18D abs joint。
├── convert_and_train_handoff_simple_state_two_models.sh
│   简化状态双模型流水线：先转换，再训练 delta14 和 absjoint18 两个模型。
└── repair_lerobot_image_stats.py
    重新解码视频计算图像 mean/std，修复 LeRobot video image stats 可能出现的 uint8 溢出问题。

train/
├── train_hf_mtdp_smoke.py
│   通用 MultiTask DiT 训练入口。支持 state slice、image keys、图像归一化/增强、val loss、TensorBoard 和 checkpoint。
├── train_random_cube_256_mtdp.sh
│   单臂 random-cube 256px 数据集训练封装。
├── train_handoff_256_mtdp.sh
│   双臂 handoff 34D joint+TCP+gripper 状态训练封装。
├── train_handoff_relpose_256_mtdp.sh
│   双臂 handoff 41D 状态训练封装，额外加入 right TCP 在 left TCP 坐标系下的相对位姿。
├── train_handoff_birelpose_time_256_mtdp.sh
│   当前 Handoff V2 主训练封装。49D 状态包含双向相对位姿和 episode progress，默认 horizon=50、n_action_steps=40。
├── train_handoff_simple_state_delta14_256_mtdp.sh
│   26D 简化状态 + 14D 双臂 EE delta action 训练封装。
└── train_handoff_simple_state_absjoint18_256_mtdp.sh
    26D 简化状态 + 18D absolute joint target 训练封装。主要用于离线/open-loop 分析，闭环需要匹配 18D joint-pos 环境。

eval/
├── eval_pick_place_policy.py
│   Isaac closed-loop 评估入口。支持单臂和 handoff checkpoint，记录 rollout、summary、policy input、teacher-forced 调试等。
├── eval_pick_place_policy.sh
│   closed-loop 评估启动脚本，负责 IsaacLab conda 环境和 LeRobot Python 兼容 shim。
├── openloop_eval_policy.py
│   离线 open-loop 动作回归评估。不启动 Isaac，直接在 LeRobotDataset 上比较模型预测和专家动作。
├── replay_handoff_raw_demo.py
│   在 Isaac 中重放已采集 handoff raw demo 的动作，检查 raw action 是否能复现成功轨迹。
├── diagnose_handoff_step0_policy.py
│   49D handoff step-0 诊断。比较 dataset/live 图像与状态组合、扩散采样、action queue/chunk 行为。
├── diagnose_handoff_step0_policy.sh
│   step-0 诊断启动脚本。
├── diagnose_handoff_hybrid_eval.py
│   expert/policy hybrid handoff 诊断。右阶段和左阶段分别可用 expert 或 policy，用于定位失败阶段。
└── diagnose_handoff_hybrid_eval.sh
    hybrid 诊断启动脚本。

tools/
├── smoke_lift_env.py
│   Isaac 环境 smoke test。检查 reset/step、obs/action space、相机 pose、相机帧和 viewport 截图。
├── open_tensorboard_live_chromium.sh
│   启动 TensorBoard 并用 Chromium 打开，带独立 profile 和自动刷新。
└── tensorboard_auto_refresh_wrapper.html
    TensorBoard 自动刷新 iframe 页面，由 open_tensorboard_live_chromium.sh 使用。

common/
└── handoff_v2_utils.py
    Handoff V2 公共工具。包含 manifest、CLIP/ImageNet image stats、action chunk 表示转换、图像预处理/增强、四元数连续性检查。
```

## 常用入口

```bash
# 单臂：采集并转换
bash isaac_pick_place/scripts/collect/collect_random_demos_to_lerobot.sh

# 单臂：训练
bash isaac_pick_place/scripts/train/train_random_cube_256_mtdp.sh

# Handoff V2：采集 200 个 demo、拆分 train/val、训练 49D 主模型
bash isaac_pick_place/scripts/collect/collect_and_train_handoff_v2_full_200.sh

# Handoff V2：训练当前使用的 26D simple-state + 18D absolute joint 模型
HORIZON=50 \
N_ACTION_STEPS=40 \
RUN_NAME=hf_mtdp_handoff_v2_simple_state_absjoint18_h50_a40_180train20val_bs16acc4_50k \
bash isaac_pick_place/scripts/train/train_handoff_simple_state_absjoint18_256_mtdp.sh

# 闭环评估
CHECKPOINT=/path/to/final_model bash isaac_pick_place/scripts/eval/eval_pick_place_policy.sh

# 离线 open-loop 评估，需要 LeRobot/MultiTask-DiT Python 环境
/home/ubuntu/Workspace/multitask_dit_policy/.venv/bin/python isaac_pick_place/scripts/eval/openloop_eval_policy.py \
  --checkpoint /path/to/final_model \
  --dataset-dir /path/to/lerobot_dataset
```
