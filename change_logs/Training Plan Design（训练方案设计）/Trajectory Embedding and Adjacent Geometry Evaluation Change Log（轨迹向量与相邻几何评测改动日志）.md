# Trajectory Embedding and Adjacent Geometry Evaluation Change Log（轨迹向量与相邻几何评测改动日志）

## 1. 改动类型

- 数据：不改变 COCO 5K、LCS 10K 或训练数据；读取固定 Probe，并永久保存每个 trajectory point 的 raw pre-L2 float32 embedding artifact。
- 模型：不改变模型结构或权重；把训练 backend 的 `model.*` state 严格加载到独立 M0 eval adapter，复用相同 preprocessing。
- 脚本：新增 completed-branch trajectory evaluator、checkpoint loader、公共 raw exporter、单点 geometry state 与相邻 transition 计算。
- 配置：新增 trajectory evaluation protocol；batch size 由服务器 Pilot 决定，只影响运行资源。

## 2. 改动位置

- `src/evaluation/embedding_export.py`：M0/trajectory 共用的图片加载、batching、raw float32 导出和 runtime metadata。
- `src/evaluation/trajectory.py`：checkpoint 解析、SHA/provenance 校验、state-dict 映射、point output、相邻 transition 与评测状态更新。
- `src/metrics/six_metrics.py`：可复用 point metrics、geometry state、Spearman、Neighbor Overlap@10 和浮点 metric delta。
- `scripts/evaluation/evaluate_trajectory.py`：单个 completed branch 的两 Probe、五轨迹点批量入口。
- `scripts/evaluation/extract_m0.py`：改为复用公共 raw exporter，不改变 M0 artifact schema。
- `src/datasets/probes.py`：允许 trajectory evaluator 显式使用完整 LCS `images/` 根目录，同时保留本地 M0 probe-images 默认路径。
- `configs/evaluation/TRAJECTORY_EVALUATION_PROTOCOL.md`、`configs/training/CHECKPOINT_PROTOCOL.md`、`TRAINING_RUNBOOK.md`：记录输入、输出、比较边界与运行方法。
- `tests/evaluation/`、`tests/metrics/`：覆盖导出顺序、snapshot 身份与加载、相邻 delta、Spearman、Overlap 和 M0 wrapper 回归。

## 3. 改动逻辑

每个 branch 完成后，从 1%、5%、20%、50% model-only snapshot 与 100% final full checkpoint 中只加载 `model` state。对两个固定 Probe 分别导出 raw `e_I/e_T`，生成 point metrics 与可复用 geometry state，再按 `M0 → 1% → 5% → 20% → 50% → 100%` 计算相邻变化。normalized embedding 只在指标函数内部派生，不重复保存。

现有有效 artifact 仅在 checkpoint、Probe 和 artifact SHA 全部一致时复用；部分输出或身份冲突会 fail-fast。当前不锁定分支间、跨模型或其他非相邻比较，完整 raw artifact 为训练结束后的分析保留扩展空间。

## 4. 改动效果

- 一个 completed branch 可以通过单一命令产生全部轨迹 raw embedding、point metrics、geometry states、相邻 transitions 和 trajectory index。
- 评测不会修改训练中的 model、optimizer、ALBEF queue、RNG 或数据流。
- trajectory model-only 与 final full checkpoint 都能使用相同 adapter 路径严格加载。
- M0 extraction 与 trajectory extraction 共享同一个 raw artifact 生成逻辑。
- 本地只进行 fake adapter、synthetic metric、checkpoint-schema 和 CPU 单元测试；真实四模型两 Probe 的 GPU 提取与吞吐仍需 AutoDL Pilot 验证。
