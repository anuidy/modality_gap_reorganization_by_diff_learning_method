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

## 尚未冻结

- learning rate
- global batch size
- gradient accumulation
- warmup / scheduler
- training steps
- checkpoint interval
- multi-seed
- Mixed negative modality ratio
