# simblame 中文说明

**仿真物理的 `git blame`。** 同一个机器人 policy，昨天在仿真里成功率 98%，今天变成 90%。simblame 回答：

1. **下降是真的吗？** 两个 setup 跑同样的 seed，只看结果翻转的那些 seed，用精确 McNemar 检验判断。
2. **问题出在仿真吗？** 先检查确定性（同一个 seed 换个顺序重跑，轨迹是否逐位相同），再看两边的 rollout 是否逐位相同。逐位相同但分数不同，说明问题不在物理，而在打分或代码。
3. **是哪一处物理改动造成的？** 对编译后的 `mjModel` 做 diff，按类别（接触、惯性、几何、执行器、求解器…）和具体实体分组。然后把每组改动直接写进正在运行的模型里，做两个反事实实验：
   - **reproduces（复现）**：只把这组改动加到 A 上，有多少个退化的 seed 也跟着退化；
   - **rescues（挽回）**：只在 B 里撤销这组改动，有多少个退化的 seed 恢复成功。

   两者综合，得到这组改动对总变化的贡献占比（share）和角色（主因、被放大的主因、只在组合时有害、无关……）。
4. **从哪里开始分叉？** 定位轨迹第一次出现差异的步数、物体和接触对，以及最终偏离最大的物体。开环回放（B 直接执行 A 录下的动作）可以把物理差异和 policy 的反应分开。
5. **是哪个 MuJoCo 版本？** `versions` 会复制你当前的 Python 环境、只替换 `mujoco` 版本，在每个版本上跑你的评估，扫描或二分出第一个改变结果的版本。

`simblame run` 一次做完 1–4，生成 `report.html`（可交互）、`report.md` 和 `report.json`。

## 真实数据验证

在答案已公开的真实基准上做盲测（完整命令和输出见 [validation.md](validation.md)）：

| 案例 | 已知答案 | simblame 的结论 | |
|---|---|---|---|
| **LIBERO** `libero_spatial/5`，50 个官方初始状态，MuJoCo 3.2.7 … 3.8.1 共 15 个版本 | MuJoCo ≥ 3.4.0 下 reset 后碗斜卡在小碟边缘（lerobot#4390） | 第一个改变结果的版本是 **3.3.7 → 3.4.0**（100% → 0%，p=1.8e-15）；编译后模型完全相同，所以是引擎本身的变化；偏移最大的物体正是**碗**（4.1 cm） | ✅ |
| **Walker2d v5 vs v4**，Minari 公开 SAC *medium* policy，100 个 seed | 两版之间唯一的物理改动：右脚摩擦 0.9 → 1.9 | 98% → 90%（p=0.039）；**主因：`foot_geom` 摩擦**，贡献 100% | ✅ |
| **Walker2d v5 vs v4**，SAC *expert* policy | 同上 | 100% → 72%（p=7.5e-9）；**主因：`foot_geom` 摩擦**，贡献 100% | ✅ |
| **Walker2d v5 vs v4**，较早的 *medium* checkpoint | 同上 | 99% → 30%（p=6.1e-20）；**主因：`foot_geom` 摩擦**，贡献 100% | ✅ |
| **Hopper v5 vs v4**（阴性对照） | 没有物理改动 | **IDENTICAL**：100/100 个 rollout 逐位相同，模型 diff 为空 | ✅ |
| **HalfCheetah v5 vs v4**（阴性对照） | 没有物理改动 | **IDENTICAL**：100/100 个 rollout 逐位相同，模型 diff 为空 | ✅ |
| **Meta-World** pick-place，一次注入 5 处资产改动 | 夹爪增益改动（kp 400 → 60） | 100% → 22%；**主因：执行器**（两个夹爪增益），指垫摩擦、质量、求解器被标为 weak / minor | ✅ |

## 快速开始

```bash
git clone https://github.com/isospin01/simblame && cd simblame
pip install -e ".[gym]"
simblame run examples/toy_push/harness.py        # 约 2 秒，无额外依赖

# 不写代码：直接比较两个 gymnasium 环境，policy 从 HuggingFace 下载
simblame run --gym Walker2d-v5 Walker2d-v4 \
    --policy farama-minari/Walker2d-v5-SAC-medium --success survived --seeds 100

# 不写代码：LIBERO 任务在多个 MuJoCo 版本间扫描
pip install -e ".[libero]"
simblame versions --libero libero_spatial/5 --mujoco 3.3.0..3.8.1
```

在自己的项目上使用：`simblame init` 生成一个带注释的 harness 模板，填好 A、B 两个 setup 怎么创建（约 10 行），然后 `simblame run my_harness.py`。也可以在 Python 里调用 `simblame.investigate(harness, seeds)`。

## 代码结构

五层，每层只依赖上面的层，详见 [ARCHITECTURE.md](../ARCHITECTURE.md)：

```
model     编译后模型的快照、对齐、diff、原地 patch
harness   如何创建、reset、step、判定两个 setup（gym / LIBERO 适配器、policy 加载）
analysis  统计比较、归因、分叉定位、版本扫描
report    终端文本、Markdown、HTML 报告（所有面向用户的文字都在 report/text.py）
cli       参数解析，每个命令都是 analysis + report 的薄封装
```

## 当前限制

- 只支持 CPU 上的 MuJoCo（MJX、MuJoCo Warp、Isaac Lab 在 [ROADMAP](../ROADMAP.md) 里）。
- 归因只覆盖编译后模型的**参数**。拓扑类改动（增删物体、执行器换目标、网格顶点数变化等）会报告并标记 `(!)`，但不能原地 patch。引擎本身的变化（如 LIBERO 案例）由 `versions` 定位，而不是 `attribute`。
- patch 只作用于运行中的 `MjModel`。环境在构建时复制到 Python 里的状态（如 gymnasium 的 `init_qpos`、keyframe）不会被 patch，所以初始姿态类改动的贡献会被低估。
- 贡献占比是一阶估计，只在被评估的 seed 上计算。
- `versions` 第一次运行需要 `uv` 和网络。
