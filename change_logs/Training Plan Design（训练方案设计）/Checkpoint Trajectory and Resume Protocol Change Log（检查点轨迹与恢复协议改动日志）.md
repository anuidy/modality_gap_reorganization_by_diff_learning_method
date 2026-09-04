# Checkpoint Trajectory and Resume Protocol Change Log（检查点轨迹与恢复协议改动日志）

## 1. 改动类型

- 数据：未修改 LCS、COCO、Probe manifest 或训练样本顺序。
- 模型：未修改模型结构、M0 初始化权重或 Standard/Mixed/ALBEF objective；新增的 trajectory artifact 仅保存训练后 model state。
- 脚本：新增进度式 checkpoint plan、完整 resume checkpoint 轮换、trajectory model snapshot、final checkpoint 标记与 provenance index。
- 配置：固定正式训练的 `1/5/20/50/100%` trajectory progress 和最新两份 resume retention；当前 `20%` full resume progress 是 Pilot 候选，Pilot 后才正式冻结；移除旧的自由 `checkpoint_interval`。

## 2. 改动位置

- `src/training/checkpoint_plan.py`：将进度比例精确映射为 optimizer steps 的纯计划模块。
- `src/training/config.py`：加载并验证 checkpointing policy，写入 `RunConfig` 与 config signature。
- `src/training/engine.py`：保存 full resume、trajectory model snapshot、final marker、hash/provenance index，并在 full resume steps 运行既有只读 Validation。
- `configs/training/train_runs.yaml`：锁定 checkpointing policy。
- `configs/training/CHECKPOINT_PROTOCOL.md`：目录、恢复、轨迹和 Future Full GCL 规则。
- `configs/training/EXPERIMENT_PROTOCOL.md`、`VALIDATION_PROTOCOL.md`、`TRAINING_RUNBOOK.md`、`experiment_comparisons.yaml`、`configs/data/DATASET_RUNBOOK.md`：同步训练/验证/运行说明。
- `tests/training/test_checkpoint_plan.py`、`test_engine.py`、`test_validation_engine_integration.py`：覆盖计划解析、trajectory 保存、latest-two retention、final metadata、resume 与 validation 调度。

## 3. 改动逻辑

M0 作为 0% 不可变基线。正式 branch 在 1%、5%、20%、50% 保存 model-only snapshot，供 branch 完成后重新加载并批量计算 COCO 5K、LCS 10K 的六项指标与相对 M0 的几何变化。完整 resume state 在 20%、40%、60%、80%、100% 保存，且只在专用目录中保留最新两份，因此能精确恢复训练又不会无限累积 optimizer state。

100% full checkpoint 不复制 model-only 文件，而通过 `final.json` 同时标记为最终轨迹点和最终评测输入。每个 artifact 都记录 M0/config/probe hash、训练进度、Git commit、artifact SHA-256 与待评测状态。Validation 继续只读执行，但改为仅在完整 resume checkpoint 后运行。

## 4. 改动效果

- 八个主分支和后续 Full GCL 都可使用统一、非均匀的几何轨迹采样协议。
- 中间几何变化可观察，且不改变正在训练的模型、optimizer、RNG 或数据顺序。
- resume 与 trajectory artifact 的用途、目录和保留策略明确分离。
- final checkpoint 是唯一正式结果输入，不引入 best-checkpoint selection。
- 使用项目 PyTorch 环境运行 checkpoint plan、training engine 与 validation integration 的定向测试通过；完整 `tests/training` 回归仍将在本次改动完成后执行。
