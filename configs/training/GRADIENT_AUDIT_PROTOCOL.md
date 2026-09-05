# AutoDL Pilot Preflight 与模块梯度审计

## 范围

独立入口 `scripts/training/audit_gradients.py` 从各模型 M0 初始化，只读取固定 LCS Train 的一个真实批次。执行训练模式的 forward/backward（前向计算与反向传播），不创建 optimizer（参数优化器），不调用参数优化步骤，不保存模型 checkpoint。

本审计不要求先填写正式学习率、训练步数、累积步数或 warmup（预热计划）。诊断 seed（随机种子）、physical batch size（单次实际批次大小）和 augmentation（图像预处理）由调用者显式提供；ALBEF 还必须显式指定诊断 alpha（动量软目标混合系数）。这些参数不会写回正式配置。

## 身份与输入检查

- 复用八组实验矩阵校验、M0 SHA-256（文件内容校验值）和完整数据身份锁。
- Validation、LCS Probe、COCO Probe 只核对身份，不加载为审计批次。
- 复用确定性采样器，取 epoch 0 的首个完整批次；同一 seed、batch size、预处理设置下，各关系及同模型两分支共用相同样本和图像视图。
- 不自动缩小 batch、不重采样、不重试 MLM mask（被遮盖的文本位置）。非有限损失会明确报告。
- 单进程、单 GPU；默认 BF16（脑浮点16位精度），不支持时明确拒绝。`--precision fp32 --device cpu` 只用于 CPU 诊断，不能替代 A800 BF16 验证。
- CLIP/VISTA 复用原生严格加载器；BEiT-3/ALBEF 保留现有加载行为，但审计读取完整 missing/unexpected keys（缺失或多余权重名称），任一非空即停止。不得自动豁免缺失参数或用随机初始化继续宣告通过。

## 独立检查项

| 分支 | 检查项 |
|---|---|
| Standard | I↔T（图像↔文本） |
| Count-Matched Mixed | I↔T、I↔IT（图像↔联合表示）、T↔IT（文本↔联合表示）分别检查 |
| ALBEF ITC-only | ITC（图文对比损失） |
| Full ALBEF | 总损失，以及 ITC、ITM（图文匹配）、MLM（掩码语言建模）各自的梯度 |

Full ALBEF 的四项检查各自重新执行相同随机种子的完整 forward，再对指定损失 backward。它们不累计梯度；各损失检查的显存仍包含完整 forward 的计算图，不能当成该损失单独实现的显存成本。

## 参数分组与判定

| 模型 | 参数组 |
|---|---|
| CLIP | 视觉编码器、文本编码器、两侧 projection（输出投影）、可训练 logit scale（相似度缩放系数） |
| VISTA | EVA 视觉模块、visual projection、BGE embeddings（输入嵌入）、BGE encoder（编码器）；另列未使用 pooler（池化层）、EVA 输出层及其 logit scale |
| BEiT-3 | 两侧 embedding、attention（注意力）、FFN（前馈网络）、归一化层、输出 head（输出映射层）、logit scale；另列未使用 mask token（遮盖标记） |
| ALBEF | 视觉编码器、文本 embedding、文本编码部分、fusion layers（融合层）、cross-attention（跨模态注意力）、两侧 projection、ITM head、MLM head、温度；四类 momentum（动量）模块分别列出 |

BEiT-3 当前 torchscale 实现中，attention、FFN 和归一化参数具有 A/B 两路，分别报告图像/文本侧，不能统称全部共享。参数名称由实际结构匹配；未知参数和缺失的必需参数组直接标记失败。

VISTA 的图像编码本身也调用原生 `encode_mm`，使用空文本提示并经过 BGE。检查中额外记录方法调用次数：按 `encode_image / encode_text / encode_mm` 顺序，I↔T 应为 `1/1/1`，I↔IT 为 `1/0/2`，T↔IT 为 `0/1/1`。这验证所执行的方法路径；参数梯度不进一步分解到每次共享模块调用。实际 temperature 是固定数值，不要求梯度。

| 预期状态 | 通过要求 |
|---|---|
| active（本次损失应反传） | 组内参数可训练，所有已产生梯度 finite（无 NaN/无穷值），且至少一个参数梯度非零 |
| inactive（本次损失不应反传） | 清空旧梯度后 `.grad` 全部为 `None`；全零张量也不算无梯度 |
| frozen（动量参数） | `requires_grad=False` 且 `.grad=None` |
| 非参数 | 固定温度标为不适用；queue buffers（队列状态张量）单独检查 |

每个参数记录名称、共享别名、元素数、可训练状态、梯度分类及 L2 norm（梯度长度）。每组汇总无梯度、全零、非零、非有限的参数数量。组内个别参数无梯度不自动导致活跃组失败，但完整名称始终保留供人工审阅。全零活跃组表示本批次未通过、需诊断，不直接证明代码有错误。

ALBEF 逐损失的预期：ITC 不应给融合层、cross-attention、ITM/MLM 独立参数反传；ITM 不应给两侧 projection、温度、MLM 独立参数反传；MLM 不应给两侧 projection、温度、ITM head 反传。MLM decoder（解码器）与词嵌入共享的权重只计入词嵌入组一次，并保留别名。

## 状态隔离

每项开始前恢复同一模型初始 tensor state（参数与状态张量），清空梯度、重置 forward 随机种子。预处理只执行一次。

ALBEF 复用现有累积窗口：开始时执行一次显式 momentum update，forward 收集队列特征，结束时调用 abort（丢弃暂存状态），不 flush（提交队列更新）。因此 momentum 与温度允许在 forward 阶段发生既有显式变化，queue 和其他参数不得变化；backward 阶段所有参数及状态张量均不得变化。

报告分别记录 forward 与 backward 的状态变化名称。每项结束后清空梯度；正常结束或异常退出均恢复初始参数、状态张量、模块 train/eval 模式及审计函数进入时的 RNG（随机数状态）。这是独立工具，不可传入正在训练的模型；既有梯度会被清空。不会保存或覆盖 M0 文件。

## 运行方法

先由用户确定此次诊断参数，再在 AutoDL 设置下列环境变量。这里不给数值默认值，以免把示例误当作已冻结参数：

```bash
python -u scripts/training/audit_gradients.py \
  --run clip_count_matched_mixed \
  --seed "$AUDIT_SEED" \
  --batch-size "$AUDIT_BATCH_SIZE" \
  --augmentation "$AUDIT_AUGMENTATION" \
  --flip-probability "$AUDIT_FLIP_PROBABILITY" \
  --device cuda --precision bf16
```

`AUDIT_AUGMENTATION` 可选 `resize_center_crop` 或 `random_resized_crop`；后者还需追加 `--crop-scale "$AUDIT_CROP_MIN" "$AUDIT_CROP_MAX"`。ALBEF 两分支追加 `--alpha "$AUDIT_ALPHA"`。同模型两分支必须使用相同的上述诊断参数。

追加 `--validate-only` 时，仅检查文件和数据身份，不加载模型、不解码训练图片、不验证 GPU 或梯度。该结果明确标记为 `preflight_identity_only`，不能当成完整审计通过。替换 `--run` 可依次检查全部八组；不自动开始正式训练。

## 输出与边界

每次独立保存到：

```text
outputs/pilot/gradient_audit/<run_id>/<UTC timestamp>/report.json
```

不会覆盖已有目录。报告含实际输入参数、样本身份、模型/数据/config hash、代码提交与相关源码 hash、运行环境、加载诊断、各项损失/梯度/状态检查、模型加载和预处理时间、forward/backward 时间及峰值显存。峰值显存分别记录 allocated（已分配）与 reserved（缓存预留），CPU 时为 null。

时间统计不包含状态复制/校验及梯度统计；为了隔离检查，主机内存额外保存一份模型 tensor state。这里的显存不包含 optimizer 状态，也不验证 gradient accumulation（梯度累积）、实际参数更新、checkpoint 写入或 resume（恢复训练）一致性。

退出码 `0` 表示当前模式全部通过；`2` 表示失败或运行异常。模型运行异常记录后停止后续项；数值或模块判定失败仍可报告后续独立项。真实四模型 BF16、吞吐与显存结论必须来自 AutoDL A800 运行结果。
