# 训练实验协议（已冻结部分）

## 主要实验矩阵

### Relation intervention

CLIP、VISTA、BEiT-3 从各自 M0 checkpoint 分叉：

```text
M0 → Standard CL
M0 → Count-Matched Mixed CL
```

- Standard 每 step 训练 `I→T` 与 `T→I`。
- Count-Matched Mixed 按 `I↔T → I↔IT → T↔IT` 轮换。
- 两条分支匹配每 step positive supervision 数和每 query negative 数。
- 同一 semantic instance 的其他 representation 不得作为 negative。
- 每个 query 恰有 1 个 positive；其余 N-1 个 candidates 只来自当前 relation 的 target modality。

`e_IT` 规则：

- CLIP：`normalize(e_I + e_T)`，无额外 forward。
- VISTA：原生 joint encoder `encode_multimodal(I,T)`；额外 joint forward 必须作为分析限制记录。
- BEiT-3：`normalize(e_I + e_T)`，无额外 joint forward。

### ALBEF native objective intervention

ALBEF 从同一 M0 checkpoint 分叉：

```text
M0 → ITC-only       L = L_ITC
M0 → Full ALBEF     L = L_ITC + L_ITM + L_MLM
```

ALBEF 不构造人工 Mixed `e_IT`，不增加 pooling/projection。两条分支训练后只重新提取 `e_I/e_T`，并计算与其他模型相同的六项指标。

Full ALBEF 是 native objective/cross-attention intervention，不等价于前三模型的 Count-Matched Mixed relation intervention。

## 共同分析规则

- 所有训练分支从已冻结 M0 checkpoint 开始。
- 所有分支使用固定 COCO 5K 与 LCS 10K probe。
- 主要分析是同模型内部的 `M0 → branch` 指标变化量。
- Raw absolute value 不作为跨架构优劣结论。
- Full GCL 仅作为 CLIP/VISTA 的 reference，不属于主要 count-matched 机制对照。

## Checkpoint 与几何轨迹协议

- M0 是 0% 轨迹基线；不复制或重新保存 M0 checkpoint。
- 每个正式 branch 在训练后的 `1% → 5% → 20% → 50% → 100%` optimizer progress 保存几何轨迹样本。
- `1%`、`5%`、`20%`、`50%` 保存 model-only trajectory snapshot；它们只用于训练后批量提取 `e_I/e_T` 和几何指标，不可用于精确 resume。
- 当前 Pilot 候选方案为每 20% progress 保存完整 resume checkpoint；完整 checkpoint 含 model、optimizer、RNG 与 data-stream state，运行中只保留最新两份。Pilot 后可在正式训练开始前重新冻结 resume progress interval。
- 100% 保存的 final full resume checkpoint 同时是最终轨迹点、最终评测输入和永久保留的 branch 结果。
- 每个轨迹点均使用固定 COCO 5K、LCS 10K、pair indices 与 kNN reference；指标仅作观察性记录，不参与 early stopping、学习率调整或 checkpoint selection。
- Full GCL 后续使用同一轨迹规则，但独立从对应 M0 checkpoint 初始化。

## 尚未冻结

- learning rate
- global batch size
- gradient accumulation
- warmup / scheduler
- training steps
- resume checkpoint progress interval after Pilot
- multi-seed
