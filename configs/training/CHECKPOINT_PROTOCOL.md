# Checkpoint 与几何轨迹协议

## 目的

正式训练同时需要两类中间产物：可精确恢复训练的完整状态，以及用于观察表示几何重组过程的不可变模型样本。两者不能混用，因为 trajectory analysis 不需要 optimizer/RNG state，而无限保留完整状态会不必要地扩大磁盘占用。

## 固定进度计划

M0 是 0% 基线。对于总 optimizer step 数为 `S` 的每个正式 branch：

```text
Trajectory progress:  1%, 5%, 20%, 50%, 100%
Resume progress:     current Pilot candidate = 20%, 40%, 60%, 80%, 100%
```

目标 step 使用 `ceil(progress × S)` 计算。若正式 `S` 小到使两个 trajectory progress 映射到相同 step，配置会拒绝启动；Pilot 可以使用单独的较短测试 schedule，但不能作为正式分析协议。trajectory fractions 与 resume retention 已锁定；当前 20% resume interval 仅为 Pilot 候选，必须在正式训练前根据 Pilot 重新确认。

## 文件角色与目录

```text
outputs/training/<run_id>/
├── checkpoints/
│   ├── trajectory/
│   │   ├── step_<step>_p001_model.pt
│   │   ├── step_<step>_p001_model.json
│   │   └── ... p005 / p020 / p050
│   ├── resume/
│   │   ├── step_<step>.pt
│   │   └── latest.json
│   └── final.json
├── checkpoint_index.jsonl
├── run_manifest.json
├── train_metrics.jsonl
└── validation_metrics.jsonl
```

- `trajectory/*.pt`：只含 model state；永久保留，用于 branch 完成后的 representation-metric 批量评测；不可精确 resume。
- `resume/*.pt`：含 model、optimizer、RNG 与 data-stream state；始终只保留最新两份；可以精确 resume。
- 100% 的 `resume/*.pt` 同时由 `final.json` 标记为 final full checkpoint，不复制 model-only 文件。
- `checkpoint_index.jsonl`：按保存/淘汰事件记录 artifact path、SHA-256、训练进度、M0/config/probe hash、code commit 与评测状态。

## 评测与训练隔离

- 训练中只创建 snapshot；不加载 snapshot 回写当前训练模型。
- branch 完成后，trajectory evaluator 从磁盘重新加载 `1/5/20/50%` trajectory snapshot 与 100% final full checkpoint，提取固定 COCO 5K 和 LCS 10K 的 `e_I/e_T`。
- 每个点计算六项指标、相对 M0 的变化和 `Geometry Preservation(M0, T_k)`。
- 指标只能观察轨迹，不能用于 early stopping、学习率调整、checkpoint selection 或改变后续训练数据。

## Future Full GCL

Full GCL 仅在八组主实验完成后作为独立 reference 启动。它从对应 M0 source checkpoint 重新初始化，使用相同 progress schedule、metadata schema 和目录隔离规则，不能从 Standard 或 Mixed 的 checkpoint 继续训练。
