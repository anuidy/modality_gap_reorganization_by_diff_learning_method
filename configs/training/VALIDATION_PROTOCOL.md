# Validation 监控协议

## 数据角色

- 数据：固定 LCS Validation 8K。
- manifest：`data/processed/lcs_558k/manifests/validation_v1.jsonl`。
- Validation 不参与参数更新、early stopping、checkpoint selection 或表示指标。
- LCS 10K 与 COCO 5K 仍是独立 Probe，训练过程中不运行。

## 执行频率

- 按独立的`validation_progress_interval: 0.2`在20%、40%、60%、80%、100%运行Validation；不依赖恢复检查点的保存。
- 轨迹模型快照在1%、5%、20%、50%、100%保存。两项计划重合时，先保存模型快照，再进行Validation。
- 40%、60%、80%没有对应模型快照，验证记录的checkpoint为null，结果来自当时训练模型的只读验证。
- 100%保存最终模型快照并运行最终Validation。
- 所有结果写入每个 run 的 `validation_metrics.jsonl`。

## 公共指标

CLIP、VISTA 和 BEiT-3 的九个分支都计算相同的：

```text
common/I<->T/loss
common/I<->T/I->T
common/I<->T/T->I
```

因此同一模型的 所有分支 可以比较同一种 Validation objective。

ALBEF 的两个分支都计算：

```text
common/loss
common/ITC
```

这里使用 ALBEF 原生 ITC 公式、当前 momentum encoder 和当前 queue，但 Validation 不执行 momentum update、不 enqueue/dequeue，也不原地截断 temperature。

## 分支诊断

非Standard主模型分支额外记录自身目标的`diagnostic/<branch>/loss`及方向损失；Mixed诊断使用固定的同一步分组，且完整36实例候选。公共I↔T始终使用相同目标模态池，不能把分支诊断loss与公共loss混作同一指标。Full ALBEF另记录ITM、MLM和full_total。VISTA使用原生IT，其余主模型使用raw sum后归一化。

## 确定性与状态隔离

- Validation manifest 顺序固定，不 shuffle。
- 使用与训练相同的 physical micro-batch size。
- 固定清单保持8,000条；batch=36、drop_last=True，丢弃固定末尾8条，实际7,992条/222批。日志分别记录manifest_sample_count、sample_count、dropped_sample_count、batch_size和drop_last。
- 每个 Validation batch 使用固定 augmentation seed、MLM mask seed 和 hard-negative sampling seed。
- 每次 Validation 使用相同随机视图，且完成后恢复 Python、NumPy、CPU Torch 与 CUDA RNG 状态。
- Validation 前切换 `eval()`，结束后恢复原训练模式。
- 使用 `torch.inference_mode()`，不创建梯度，不调用 optimizer。
