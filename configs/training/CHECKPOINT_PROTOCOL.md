# Checkpoint 与几何轨迹协议

## 目的

当前正式训练只保存用于观察表示几何重组的模型轨迹快照，不保存optimizer（优化器）、RNG（随机状态）或数据流恢复状态。`save_resume_checkpoints: false`为当前正式配置。旧完整恢复检查点的读取能力保留用于兼容，但不代表本轮会创建resume文件。

## 固定进度计划

M0 是 0% 基线。对于总 optimizer step 数为 `S` 的每个正式 branch：

```text
Trajectory progress:  1%, 5%, 20%, 50%, 100%
Validation progress: 20%, 40%, 60%, 80%, 100%
```

目标step使用四舍五入`ROUND_HALF_UP(progress × S)`计算。若预算过短导致轨迹点重复，则拒绝启动。Validation按独立的`validation_progress_interval: 0.2`执行，不因停止保存恢复检查点而改变频率。

## 文件角色与目录

```text
outputs/training/formal_full_v1/<template_run>/seed_42/
├── checkpoints/
│   ├── trajectory/
│   │   ├── step_<step>_p001_model.pt
│   │   ├── step_<step>_p001_model.json
│   │   └── ... p005 / p020 / p050 / p100
│   └── final.json
├── checkpoint_index.jsonl
├── run_manifest.json
├── train_metrics.jsonl
└── validation_metrics.jsonl
```

- `trajectory/*.pt`：只含模型状态及来源元数据，不包含优化器、随机状态或数据流位置；不能精确续训。
- 当前不创建`resume/`及`latest.json`。
- 旧20%诊断的`outputs/training/formal_v1/`仅保留日志与元数据，72份权重已完成评测后按用户指令退役。原路径继续用于历史追溯，本次正式训练使用上面的独立目录。
- 100%的模型快照也位于`trajectory/`，由`final.json`引用；不重复复制权重。
- `checkpoint_index.jsonl`：按保存/淘汰事件记录 artifact path、SHA-256、训练进度、M0/config/probe hash、code commit 与评测状态。

## 评测与训练隔离

- 训练中只创建 snapshot；不加载 snapshot 回写当前训练模型。
- branch完成后，trajectory evaluator从磁盘重新加载各进度的模型快照；同时兼容历史完整检查点格式。通用A/B入口已支持显式选择已保存的点，但当前自动诊断队列仍只调度到20%，全程调度需另行接入。
- 每个点保存 raw embedding、point metrics 与 geometry state；跨时间点的正式比较由独立 trajectory evaluation protocol 定义。
- 指标只能观察轨迹，不能用于 early stopping、学习率调整、checkpoint selection 或改变后续训练数据。

当前实现保存每个点的 raw pre-L2 float32 `e_I/e_T`，并自动计算 `M0 → 1% → 5% → 20% → 50% → 100%` 相邻变化。其他比较组留到正式训练结束后再定义；完整 artifact 与输出布局见 `configs/evaluation/TRAJECTORY_EVALUATION_PROTOCOL.md`。

## 正式预算与gate暂停

S=15003，轨迹步数为150/750/3001/7502/15003；Validation步数为3001/6001/9002/12002/15003。40%、60%、80%运行验证时没有对应新模型快照，日志中的checkpoint为null。

`--gate`或`--stop-after-step=3001`在20%模型快照保存后停止，不改变15003步学习率调度，不写final.json。manifest状态为paused，使用trajectory_checkpoint字段并声明exact_resume_supported=false。每项任务到20%共保存1%、5%、20%三份权重。

模型快照可用于评测，但不能用`--resume`精确接续训练；若以后需要继续训练，必须另行明确训练协议，不能静默重置优化器后冒充连续训练。
