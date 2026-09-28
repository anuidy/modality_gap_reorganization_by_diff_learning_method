> 先验阶段知识参考。当前训练参数和执行方式以[主README](../README.md)及正式协议为准。文中的交接/修改日志仅保留服务器；历史假设和建议不代表新正式协议。原始模长不等时，不可套用等模长的平方根恒等式。
>
> 文中先验专用脚本与旧配置路径指归档内原路径，已退出当前执行目录；归档位置及参考方法见[主README](../README.md)。

# 训练与评测实现的事实核验

> **用途**：对已完成 14 条训练实验的**代码事实**做一次冻结式核验，供外部分析者准确理解"代码实际上做了什么"。
> **配套文档**：`docs/Project Deconstruction and Experiment Design（项目解构与实验设计说明）.md`（项目结构、实验设计、实测指标数据）。
> **核验日期**：2026-09-14（UTC）。**核验方式**：逐行读代码 + 实测产物 + 进程内真实前向。
> **核验约束**：不修改任何代码、配置、checkpoint、指标产物；不启动训练；不重跑轨迹评测；不做机制解释与合理性判断。
> **未落盘任何文件**：所有计算都在临时进程内完成，结论中的每个数字都可按 §13 的证据索引复现。

---

## 0. 一页结论

按对后续解释的影响排序，四件必须立刻知道的事：

1. **VISTA 训练用的是 T = 0.02（logit scale = 50），全程恒定，不是 T = 1.0。** `normlized=False → temperature=1.0` 只发生在 M0 导出用的那个适配器实例上，且该值在其路径上不被消费。（§2）
2. **CLIP 的评测侧 embedding 是 fp16 权重算出来的，训练侧是 fp32。** 两侧权重值逐位相同，但适配器没有 `.float()`，291/446 个张量保持 checkpoint 原生的 fp16。同输入相对差 3.4e-3（image），cos 0.999992。BEiT-3 与 ALBEF 两侧**逐位一致**；VISTA 只差目标函数那一次 L2（cos = 1.0）。（§3）
3. **Fixed GCL-2 的实际分母是 2，不是 6；`没有训练过 Fixed-/6 checkpoint`。** Rotating 也是 2，Full GCL 是 6 —— 全部由 run 自己的日志反解得到，不引用设计文档。（§4）
4. **存在同实例非目标负例**：六个方向各有 1 个/实例，全部留在 softmax 分母里，且可成为 hardest negative；而训练日志里的 `same_instance_false_negatives` 是**硬编码常量 0**，完全没有统计它们。（§5）

另有两条会直接影响解释的代码事实：**VISTA/ALBEF 训练时 dropout(p=0.1) 是开启的**（CLIP/BEiT-3 不是）；**最早完成的 5 条 run 用的是旧日志 schema**，没有 `total_loss` / `loss_directions` / `logit_scale` 字段。（§3.4、§4.3）

---

## 1. 核验方法与可靠性

**三类证据，权重依次递减**：

| 类别 | 做法 | 例 |
|---|---|---|
| A. 产物实测 | 直接读 run 的日志/清单/checkpoint | 分母、logit scale、dtype 分布 |
| B. 进程内前向 | 用仓库真实代码在同进程构造两侧模型，喂同一张量 | 训练侧↔评测侧一致性 |
| C. 代码行 | 只用于解释 A/B 的成因，不单独作为结论依据 | 掩码规则、L2 发生位置 |

**噪声基线（可靠性前提）**：同一路径重复两次，输出**逐位一致**（max|Δ| = 0.0）。因此下文所有非零差异都是真实差异，不是测量抖动。

**核验过程中被我自己纠正的两处测量失误**（记录在此，便于评估本报告的可靠性）：

1. 子核验报告曾给出 VISTA `cos(pre-L2, L2(pre-L2)) = 0.891`。这在数学上不可能（只差一次 `F.normalize` 时余弦必为 1）。复核后实测为 **cos = 1.0、`max|L2(raw) − normalized| = 0.0`**，原值为测量失误。
2. 我最初比较两侧权重时用了 `tensor.float()`，这会**把 dtype 差异抹掉**，导致"权重逐位相同"的假象。改为逐张量比较 dtype 后才发现 CLIP 的 291 个 fp16 张量。

---

## 2. VISTA 的 temperature（核验项 1）

### 2.1 `normlized=False` 在哪里、如何设置 temperature

`third_party/flag_embedding/research/visual_bge/visual_bge/modeling.py`：

```
:33    temperature: float = 0.02,   # 构造参数默认值
:76    self.normlized = normlized
:78    self.temperature = temperature
:79-81 if not normlized:
           self.temperature = 1.0    # 强制覆盖，不是"默认 1.0"
           logger.info("reset temperature = 1.0 due to using inner product to compute similarity")
:91    self.load_model(model_weight)
```

`self.temperature` 全库只有 4 处引用（`:33`、`:78`、`:80`、`:341`），唯一消费点是 `Visualized_BGE.forward`（`:341`）——而**本仓库没有任何代码调用这个 `forward`**。

### 2.2 赋值与 load 的先后 / checkpoint 是否保存 temperature

- **顺序**：`:78/80` 设温度 → `:91` load checkpoint。**先设温度，后 load。**
- **但 load 不影响它**：`self.temperature` 是普通 Python 属性，**不是 buffer/Parameter，不进 state_dict**。实测 `'temperature' in model.state_dict()` 为 `False`。
- **checkpoint 里也没有该键**：`BGE_EVA_Token_S1.pth` 共 488 个键，含 `temperature`/`temp` 的键为 **0 个**。唯一温度相关键是 `model_visual.logit_scale = 4.6055`（`exp ≈ 100.03`，EVA-CLIP 自带的对比温度，本项目路径不消费）。

### 2.3 训练实际使用的 temperature（实测）

| 路径 | 构造 | `self.temperature` | 传入目标函数的 scale |
|---|---|---|---|
| 训练 backend | `backends.py:279` `normlized=True`、`:280` `temperature=options["temperature"]` | **0.02** | `backends.py:321` → `1.0/0.02 = **50.0**` |
| M0 导出适配器 | `model_adapters/vista.py:41-47`，`normlized=False`，未传 temperature | **1.0** | 不消费 |

**日志实测**（`outputs/training/<run>/train_metrics.jsonl`）：

| run | 记录数 | `logit_scale` | 取到的值 |
|---|---|---|---|
| `vista_gcl2_fixed` | 1690 | 有 | **{50.0}**（唯一值，全程不变） |
| `vista_gcl2_rotating` | 1690 | 有 | **{50.0}** |
| `vista_full_gcl` | 1690 | 有 | **{50.0}** |
| `vista_standard` | 1688 | **无此字段**（旧日志 schema，见 §4.3） | — |

配置侧一致：四条 run 的 `run_manifest.json` 的 `config.model_options.temperature` 均为 `0.02`；`configs/training/vista_standard_execute.yaml` 的 `options.temperature: 0.02`。

### 2.4 logits 是不是 `cos / T`

是。`src/objectives/contrastive.py:142-145`（standard / GCL-2）与 `src/objectives/gcl.py:105-112`（Full GCL）：

```python
left_normalized  = F.normalize(left.float(),  dim=-1)
right_normalized = F.normalize(right.float(), dim=-1)
scale  = _scale_tensor(logit_scale, left_normalized)
logits = scale * left_normalized @ right_normalized.transpose(0, 1)
```

即 `logits = scale × cos = cos / T`，`T = self.model.temperature = 0.02`。

### 2.5 结论

**VISTA 训练全程 T = 0.02（scale 50），不是 T = 1.0。** T = 1.0 只存在于 M0 导出适配器实例，而该实例只调用 `encode_image/encode_text`，温度消费点不可达，因此对导出的 embedding 无影响。

### 2.6 `normlized=False` 的全部行为差异

`self.normlized` 在全库仅 3 处被使用：

| 行 | 作用 |
|---|---|
| `:79-81` | **temperature 重置为 1.0**（唯一的非 L2 副作用） |
| `:220-221` | `encode_text` 末尾的 `F.normalize` |
| `:292-293` | `encode_mm` 末尾的 `F.normalize` |

`img_token_embedding`（`:301-306`）里的 `model_visual.encode_image(images, normalize=False)` 是**硬编码**的，与 `self.normlized` 无关；`sentence_embedding`、`compute_similarity`、`forward`、`save`/`load_model` 均不读该属性。

> ⚠️ **对既有文档的更正**：`docs/Project Deconstruction and Experiment Design（项目解构与实验设计说明）.md` 的 §2.5 与 §4.2 提到"`normlized=False` 会把 temperature 重置为 1.0"，但**没有写明训练侧用的是 `normlized=True` + 0.02**。读者可能据此误得"VISTA 训练用 T = 1.0"。该文档尚未修改，待确认。

---

## 3. 训练侧 ↔ 评测侧数值一致性（核验项 2）

### 3.1 方法

两侧都加载**同一条 M0 checkpoint**，用**同一张预处理后的张量**（评测侧 preprocess 产出的那张），BEiT-3/VISTA/ALBEF 的文本用**同一组 token**。分四组配置：

- **V0**（噪声基线）：同一路径重复两次。
- **V1**（纯代码路径）：双方 fp32、双方 eval、无 autocast。
- **V2**（生产精度）：评测侧原生 fp16 autocast vs 训练侧 bf16 autocast，仍双方 eval。
- **V3**（生产现实）：V2 + 训练侧 `train()`。

### 3.2 结果

| 模型 | V0 基线 | V1 纯代码路径 | V2/V3 生产精度 |
|---|---|---|---|
| **clip** | 0.0（逐位） | **不一致**：max\|Δ\|=2.41e-2，相对 3.4e-3(img)/1.1e-3(txt)，cos 0.999992/0.999999 | 相对 1.65e-2 / 6.35e-3 |
| **beit3** | 0.0 | **逐位一致**（0.0） | 相对 9.05e-3 / 8.03e-3 |
| **vista** | — | **方向一致**：cos=1.0，`max\|L2(eval_raw) − train\|=0.0`，仅范数不同（1.0 vs 16.4） | 见 §3.4 |
| **albef** | — | **逐位一致**（max\|Δ\|=0.0，rel=0.0） | 未单独测（两侧同为 fp32） |

### 3.3 CLIP 不一致的根因（决定性证据）

两处构造的差别只有一个 `.float()`：

| 路径 | 代码 | 结果 |
|---|---|---|
| 评测适配器 | `src/model_adapters/clip_openai.py:43` `clip.load(str(checkpoint), device=..., jit=False)`（**无 `.float()`**） | state_dict：**291 fp16 + 155 fp32** |
| 训练 backend | `src/training/backends.py:224` `self.model = model.float().to(device)` | state_dict：**446 fp32** |

- 两侧权重**数值**（把 fp16 升到 fp32 后）**446/446 逐位相同**，`logit_scale` 也相同（4.605170249938965）。
- 所以差异**不是"两套代码算法不同"，而是"评测侧用 fp16 权重、训练侧用 fp32 权重"**。
- 适配器自己的 metadata 也自报 `inference_autocast_dtype: float16`。
- 注意：CLIP 的官方 `ViT-L-14.pt` 原生就是 fp16；`.float()` 把权重无损升到 fp32，而适配器保留了 fp16 并在 fp16 autocast 下计算。

### 3.4 训练模式（train()）的架构不对称 —— 新发现

`engine.py:445` 在训练开始时 `backend.train()`；验证时 `validation.py:362` 临时 `backend.eval()`，并在 `:384` 的 `finally` 里 `backend.train(was_training)` 恢复。**即整个训练过程模型处于 train 模式。** 而各模型实际处于"开启"状态的 Dropout：

| 模型 | p>0 的 Dropout 模块数 | p |
|---|---|---|
| clip | **0** | — |
| beit3 | 0（73 个模块但 p 全为 0） | — |
| **vista** | **37** | 0.1 |
| **albef** | **98** | 0.1 |

对 VISTA 的实测后果（同一输入）：

- 训练侧连续两次前向之间：**cos = 0.82–0.84（image）/ 0.88–0.89（text）**
- 训练侧 vs 评测侧（方向对齐后）：**cos = 0.886（image）/ 0.936（text）**，相对差 4.7e-1 / 3.6e-1

即：**CLIP/BEiT-3 的 `train()` 是空操作，而 VISTA/ALBEF 每一步训练看到的都是带 dropout 的随机样本**。这属于架构层面的不对称，与已知的"VISTA IT 走联合编码器"confound 是两件独立的事。

### 3.5 另一条配置不等价（代码事实，影响未测）

文本 tokenizer 的最大长度：

| | 训练侧 | 评测侧 |
|---|---|---|
| clip | 77（truncate） | 77 |
| beit3 | 64 | 64 |
| albef | 30 | 30 |
| **vista** | **64** | **512** |

COCO 描述普遍短于 64 token，本轮**实际影响应为零**，但该配置本身不等价；若将来换用长文本 probe，此处会产生真实差异。

### 3.6 决策记录：保持现状（方案 A，用户 2026-09-14 决定）

针对"CLIP 评测侧保留 fp16 权重"这一点，用户决定 **不修改，保持现状**，并把以下内容作为已知条件记录在案。

**（1）成因不是设计选择，而是继承了官方 loader 的默认行为**

- 评测侧 `src/model_adapters/clip_openai.py:43` 传 `device="cuda"` → 官方 `clip.load()` 里 `build_model()` **必定**调用 `convert_weights`（Linear/Conv/MultiheadAttention/proj → fp16），而唯一的 `.float()` 位于 **CPU 分支** → 于是得到 291 fp16 + 155 fp32。
- 训练侧 `src/training/backends.py:223-224` 传 `device="cpu"` 再 `.to(cuda)` → 命中 CPU 分支 → 446 fp32（因此那行 `.float()` 实为 no-op）。
- 即：两侧精度都是被 `clip.load` 的 device 参数间接决定的，**没有一处是显式的精度选择**。

**（2）去掉它没有功能收益，也没有功能必要**

适配器契约本身要求返回 float32（"Return CPU float32 raw embeddings"）、出口即 `.detach().float().cpu()`；指标层全程 numpy（协方差/特征值还升到 float64）。评测实测峰值仅 2.42 GiB、单条 run 约 13 分钟，fp16 的速度收益不是瓶颈。fp16 属于性能默认值。

**（3）保持现状的核心依据：导出产物内部自洽**

用"存盘值往返 fp16 是否精确"作为指纹检验（取前 2000 行）：

| 产物 | 可被 fp16 精确表示的比例 |
|---|---|
| CLIP M0（更早在 Windows 上导出） | **100.0000%** |
| CLIP 轨迹 p100（本机导出） | **100.0000%** |
| BEiT-3 M0 | **100.0000%**（来自适配器 fp16 autocast，非权重） |
| VISTA M0 | 0.0139%（最后一步不是 Linear） |

两点含义：① **所有 CLIP 导出向量带同一种 fp16 印记，跨机器、跨时间一致**；② 本项目主分析是**同模型 Δ**（M0 → 分支），两侧同源，该印记在一阶上相消。此外，精度差（同输入相对 3.4e-3、cos 0.999992）比要测的效应量（centroid gap 差 3 倍、effective rank 225→20、anisotropy 0.28→0.92）小 2–3 个数量级。

**（4）若日后改为 fp32，必须整批重导**

只改适配器代码而不重导，会让新导出向量与既有产物差 ~1e-3 相对量级，破坏自洽。需要重导的范围是 **CLIP 全部**：M0 两个 probe + 4 条 run × 5 个轨迹点 × 2 个 probe（约 1 小时 GPU），并重算相应指标。BEiT-3 / VISTA / ALBEF 无需重导（分析是同模型内比较）。

**（5）注意权重与 compute 精度是两件事**

适配器固定 `torch.autocast(cuda, fp16)`（`clip_openai.py:48-51`），而训练侧是 bf16 autocast。因此即使把 CLIP 权重改成 fp32，仍会剩 fp16-vs-bf16 的计算精度差——BEiT-3 的生产级差异（9.05e-3）就**全部**来自这一项（它权重本来就是 fp32）。所以"改成 fp32"若要彻底，必须同时改权重与 autocast。

---

## 4. 三种分支的实际分母（核验项 3）

### 4.1 方法

训练日志里同时记录两个量：`raw_total_loss` = 各方向损失的**未除之和**，`total_loss` = 实际回传的损失（`engine.py:698-701` 与 `:736-737`）。二者之比直接暴露分母。

### 4.2 结果（全部来自 run 自己的产物）

| run | 记录数 | `total_loss / raw_total_loss` | 实际分母 |
|---|---|---|---|
| `clip_gcl2_fixed` / `beit3_gcl2_fixed` / `vista_gcl2_fixed` | 各 1690 | **0.500000**（min = max） | **2** |
| `clip_gcl2_rotating` / `beit3_gcl2_rotating` / `vista_gcl2_rotating` | 各 1690 | 0.500000 | 2 |
| `clip_full_gcl` / `beit3_full_gcl` / `vista_full_gcl` | 各 1690 | **0.166667** | 6 |
| `clip_standard` / `beit3_standard` / `vista_standard` | 各 1688 | `metrics.loss/(L1+L2)` = 0.500000 | 2 |

**结论：`没有训练过 Fixed-/6 checkpoint。`** 三条 `*_gcl2_fixed` 全部在 2N 下训练。

### 4.3 五条最早完成的 run 使用旧日志 schema（重要）

| run 组 | 完成时间 | 日志字段 |
|---|---|---|
| `clip_standard` / `beit3_standard` / `vista_standard` / `albef_itc_only` / `albef_full` | 2026-09-08 ~ 09-09 | **6 个字段**：`completed_steps, data_stream, gradient_norm, learning_rate, metrics, relation_audit` |
| 6 条 GCL-2 + 3 条 full_gcl | 2026-09-10 之后 | **25 个字段**，含 `total_loss`、`raw_total_loss`、`loss_denominator`、`logit_scale`、`loss_direction_1..6`、`loss_directions` |

原因：写这些字段的引擎代码改动于 2026-09-10 16:41（+0800），晚于前五条 run。

**影响**：跨分支比较 loss 时，标准分支的每方向损失须从 `metrics["loss/I->T"]`、`metrics["loss/T->I"]` 读取（存在且可用，实测比值 0.5 精确）；`total_loss` 对应 `metrics["loss"]`；`vista_standard` 的 `logit_scale` 不可读。

---

## 5. 候选池与同实例非目标负例（核验项 4）

池顺序固定 `MIXED_POOL_ORDER = ("I","T","IT")`（`contrastive.py:17`），槽位索引 `I_i=0N+i`、`T_i=1N+i`、`IT_i=2N+i`。唯一置真的掩码元素是 `exclude[row, query_indices]`（`contrastive.py:226`），即**只 mask query 自己的槽**。

| 方向 | query 槽 | positive 槽 | masked 槽 | same-instance non-target 槽 | 留在分母 | 有效候选 | 负例 |
|---|---|---|---|---|---|---|---|
| I→T | i | N+i | i | **IT_i** | **是** | 3N−1 | 3N−2 |
| T→I | N+i | i | N+i | **IT_i** | 是 | 3N−1 | 3N−2 |
| I→IT | i | 2N+i | i | **T_i** | 是 | 3N−1 | 3N−2 |
| IT→I | 2N+i | i | 2N+i | **T_i** | 是 | 3N−1 | 3N−2 |
| T→IT | N+i | 2N+i | N+i | **I_i** | 是 | 3N−1 | 3N−2 |
| IT→T | 2N+i | N+i | 2N+i | **I_i** | 是 | 3N−1 | 3N−2 |

**六个方向全部存在**同实例非目标槽，每方向每实例恰好 1 个，且全部未 mask、参与 softmax 分母。它们**也参与 hardest-negative 统计**（`contrastive.py:233-236` 只额外屏蔽 query 槽与 positive 槽）。

**`same_instance_false_negatives = 0` 是硬编码，不是测量值**（本结论为 2026-09-14 的核验快照；该字段已于 2026-09-15 整体移除）：

- `contrastive.py:245` `return loss, debug, 0`
- `contrastive.py:282` = `0 + 0`；`:382` 直接写字面量 `0`；`:61` 默认值 0
- 写入日志处 `engine.py:655`
- 实测：所有 GCL 分支 1690 条记录该值**恒为 0**，与 standard 完全一致。

→ 该字段**不能**用于主张"Mixed 分支不存在同实例假负例"。

---

## 6. Standard → Fixed 实际改变了哪些变量（核验项 5）

| 项 | Standard | Fixed GCL-2 | 是否变化 |
|---|---|---|---|
| positive relation | I↔T（双向） | I↔T（fixed 恒选） | 端点未变，但损失对象由单模态池换成混合池 |
| query 数量 | 2N | 2N | 未变 |
| directional loss 数量 | 2 | 2 | 未变 |
| **candidate pool size** | **N**（仅对侧模态） | **3N**（I+T+IT） | **变了** |
| **negative 数量** | **N−1** | **3N−2** | **变了** |
| **candidate modality composition** | 单一模态 | {I, T, IT} | **变了** |
| **IT 是否参与** | 否（representations 仅 {I,T}） | 是（`e_I + e_T`，或 VISTA 的 native） | **变了** |
| **same-instance non-target candidate** | 不存在 | 存在（每 query 1 个，入分母） | **变了** |
| loss averaging（分母） | 0.5·(l2r+r2l) | (l2r+r2l)/2.0 | 未变（均为 2） |
| temperature / logit scale 来源 | backend 传入，与 branch 无关 | 同左 | 未变 |
| optimizer / batch / 数据 / augmentation | AdamW、32×1、同一 manifest、RRC 0.9–1.0 + hflip 0.5 | 同左 | 未变 |
| trainable parameters | 全参数 | 全参数 | 未变（两侧 `trainable_parameter_name_sha256` 相同） |

**结论：Standard → Fixed 不是"只换了 relation"。** 它在保持 relation 端点（都是 I↔T）的同时，把候选池扩到 3N、负例从 N−1 扩到 3N−2、引入 IT、并新增了同实例假负例。

---

## 7. VISTA 的 raw IT 能否无副作用导出（核验项 6）

- **明确的 pre-L2 张量存在**：
  - `encode_mm`：局部变量 **`prompt_img_reps`**，`modeling.py:291`，下一行 `:292-293` 才 `F.normalize` 并重新绑定同名变量。
  - 对称地，`encode_text` 的是 **`t_reps`**（`:219`，L2 在 `:220-221`）。
  - `encode_image`（`:308-318`）内部就是 `return self.encode_mm(...)`（`:317`），因此 I/T/IT 三个 raw 点同源。
- **能否不改行为地取到**：`F.normalize` 返回新张量、不原地修改，所以在 `:291` 之后接一个返回值**不改变任何现有行为**（`normlized`、`temperature`、`logit_scale`、优化器、loss 全不受影响）。但这需要改 `third_party` 源码；不动它就只能挂 forward hook 并自行复刻 pooling（含 `prom_img_attention_mask` 的构造，见 `:269-270`、`:291`）。
- **不能走 `normlized=False` 这条现成路**：那必然把 `self.temperature` 从 0.02 改成 1.0（§2.1），而训练侧正是为此选了 `normlized=True`（`backends.py:305-311` 的注释即此意）。
- 实测佐证：`normlized=False` 输出范数 ≈ 16.3/16.2（即 `:291` 的 pre-L2），`normlized=True` 输出范数 = 1.0；`L2(raw)` 与 `normlized=True` 输出**逐位相同**。

---

## 8. CLIP / BEiT-3 的 IT 能否由现有 artifact 精确重建（核验项 7）

**可以。** 逐项确认：

| 检查项 | 结论 | 位置 |
|---|---|---|
| 定义 | `return image + text`，系数均为 1，仅做形状校验 | `contrastive.py:430-439` |
| 唯一训练调用点 | `multimodal = joint_encoder() if ... else additive_multimodal_embedding(image, text)` | `backends.py:123` |
| 额外 projection | 无（CLIP 取 `encode_image` 原始投影；BEiT-3 取 `vision_head`/`language_head` 直出） | `backends.py:240-241`、`:406-417` |
| weighting | 无 | `contrastive.py:439` |
| detach | 无 | 同上 |
| normalize-before-sum | **无**。归一化发生在求和之后，由目标函数对池内成员逐个执行 | `contrastive.py:192` |
| model-specific 额外操作 | CLIP：`encode_image` 结尾 `x @ self.proj`，无 L2（`clip/model.py:235-240`）；BEiT-3：绕开官方 `forward` 的 `F.normalize`（`modeling_finetune.py:252-253/264-265`），head 为无 norm 的 `nn.Linear` | 左述 |

CLIP 与 BEiT-3 走同一条路（`joint_encoder=None`，`backends.py:249` / `:425`）；VISTA 走 native `encode_mm`；ALBEF 没有 IT（不进 `_relation_step`）。

**附带条件**：现有 npz 只存 `sample_ids` / `image_embeddings_raw` / `text_embeddings_raw`（可选开关 `--save-probe-embeddings` 在 14 条评测中**从未开启**），因此 IT 由 `e_I_raw + e_T_raw` 重建 —— 这正是训练时的定义。但 **CLIP 的存盘 raw 来自 fp16 权重的适配器**（§3.3），重建出的 IT 会继承该精度。

---

## 9. Probe 的实际身份（核验项 8）

### COCO 5K（`coco_2017_val_5k`，SHA `d8133889432c…`）

| 问题 | 结论 |
|---|---|
| 是否严格 5000 / 5000 unique image | **是**。`image_id`、`image_relpath`、`caption_id`、`sample_id`、`semantic_id` 均 5000 unique |
| 每图几条 caption | **1 条**，规则 `minimum_caption_id_per_image`（取 `id` 最小的 annotation，`probes.py:53`；硬校验 `:55-58`） |
| id 是否固定在 manifest | 是。字段：`sample_id, semantic_id, image_id, caption_id, image_relpath, caption` |
| 顺序 | **image_id 升序**（`probes.py:62`），加载侧按原序构造，不 shuffle |
| 可否完全复用同一批 pair | **可以**。manifest SHA 被 configs 与 run_manifest 钉死，评测前重算比对，不一致直接 raise；已导出 npz 保持 manifest 原序 |

⚠️ **唯一需要知情处**：caption **文本**有 38 组重复（unique text = **4962** < 5000），即不同图片使用了完全相同的句子。做 score 级分析时"跨模态文本"集合里会出现重复串。

### LCS 10K（`lcs_558k_in_domain_10k`，SHA `3895e647e218…`）

| 问题 | 结论 |
|---|---|
| 是否严格 10000 pair | **是**（加载侧强制 `!= 10000` 报错） |
| 重复 | **零**。`source_index`/`id`/`image`/`caption` 各 10000 unique；空 caption 0 条 |
| pair mapping | 固定。manifest 字段 `source_index, id, image, caption`；`semantic_id` 由加载时构造 `lcs_558k:{id}` |
| 抽样规则 | `random.sample(seed=20260825)` 从 558128 条源抽 10000 索引；manifest 已存在则不再重抽 |
| 与训练集重叠 | **0**（probe ∩ train = 0，probe ∩ validation = 0，按 index 与 id 双重实测） |

---

## 10. checkpoint 里的 temperature / logit scale（核验项 9）

| 模型 | 键 | M0 → p100 能否恢复 | 说明 |
|---|---|---|---|
| CLIP | `model.logit_scale`（M0 为模块属性 `logit_scale`） | ✅ 能 | 六点均有 |
| BEiT-3 | `model.logit_scale` | ✅ 能 | 六点均有 |
| ALBEF | `model.temp` | ✅ 能 | 另有 `config.temp = 0.07`（构造初值，非 tensor） |
| **VISTA** | **state_dict 中不存在** | ❌ **不能** | 生效值来自构造属性 `temperature=0.02`，来源是 config / `run_manifest.json` 的 `config.model_options` |

补充：**p100 不是 trajectory checkpoint**，而是 `checkpoints/resume/step_00016879.pt`（`checkpoint_kind: full_resume`），由 `checkpoints/final.json` 指向；trajectory 目录只保存 1/5/20/50%。

---

## 11. 事实冻结表

| 项 | 结论 | 标记 | 证据位置 |
|---|---|---|---|
| **A. VISTA temperature** | 训练全程 **T = 0.02（scale 50）**，恒定；T = 1.0 只在 M0 适配器的 `normlized=False` 实例且不被消费 | **confirmed** | `modeling.py:33/76/78/79-81/91`；`backends.py:279/280/321`；3 run × 1690 条日志 ≡ 50.0；checkpoint 无 temperature 键 |
| **B. 训练/评测 embedding 等价** | clip **fail**（评测侧 291 张量 fp16 vs 训练侧 fp32；相对 3.4e-3，cos 0.999992）；beit3 **pass**（逐位）；albef **pass**（逐位）；vista **pass**（仅差目标函数的 L2，cos = 1.0、`max\|L2(raw)−norm\| = 0.0`） | **confirmed** | `clip_openai.py:43` vs `backends.py:224`；446/446 值相同但 dtype 不同；V0 基线逐位 |
| **C. Fixed 实际分母** | **2** | **confirmed** | 1690 条 `total/raw = 0.500000`（min = max） |
| **D. Rotating 实际分母** | **2** | **confirmed** | 同上 |
| **E. Full 实际分母** | **6** | **confirmed** | 1690 条 `= 0.166667` |
| **F. 同实例非目标负例** | **存在**：六方向各 1 个/实例，全部留在分母，可成为 hardest negative；日志字段为硬编码 0 | **confirmed** | `contrastive.py:226-228/233-236/245/282/382`；`engine.py:655` |
| **G. Standard→Fixed 变化变量** | 变：池 N→3N、负例 N−1→3N−2、模态组成、IT 参与、同实例负例；不变：端点/query 数/方向数/分母/温度来源/optimizer/batch/数据/可训练参数 | **confirmed** | §6 表；`beit3_standard` 与 `beit3_gcl2_fixed` 的 run_manifest 逐项对照 |
| **H. VISTA raw IT** | pre-L2 点存在（`prompt_img_reps`，`modeling.py:291`），但取出须改 `third_party` 或走 hook；`normlized=False` 必然改温度 | **confirmed** | `modeling.py:219-221/291-293/317`；实测 `L2(raw) ≡ normlized=True` 输出逐位相同 |
| **I. CLIP/BEiT-3 IT 重建** | **可精确重建**（`e_I+e_T`，无任何附加算子）；CLIP 会继承评测侧 fp16 精度 | **confirmed** | `contrastive.py:430-439`；`backends.py:123/249/425` |
| **J. Probe 身份** | 两个 probe 均可直接复用到 A 类；COCO 有 38 组重复 caption 文本（4962 unique） | **confirmed** | `probes.py:53/55-58/62`；实测 unique 统计；split lock 交集 0 |
| **K. checkpoint 温度/scale** | clip/beit3/albef **能**（`logit_scale`/`temp`）；**vista 不能** | clip/beit3/albef **confirmed**；vista **conflict** | 见 §10；vista 的值只在 config / run_manifest |

---

## 12. 两条不在核验项内、但会直接影响解释的发现

1. **train mode 的架构不对称**：`engine.py:445` 全程 `backend.train()`（验证临时切 eval 并在 `validation.py:384` 恢复）。VISTA 有 37 个、ALBEF 有 98 个 `Dropout(p=0.1)` 处于开启；CLIP 0 个、BEiT-3 全为 p=0。VISTA 同输入两次 train-mode 前向 cos 仅 0.82–0.84（image）/ 0.88–0.89（text）。
2. **五条最早完成的 run 使用旧日志 schema**，无 `total_loss`/`loss_directions`/`loss_denominator`/`logit_scale` 字段；其每方向损失在 `metrics["loss/I->T"]` 等键下。

---

## 13. 证据索引

**代码**

```
src/objectives/contrastive.py      :17 池顺序 | :41 GCL2 分母常量 | :42 FULL 分母常量
                                   :142-145 standard logits | :162 声明计数
                                   :192 池内逐个归一化 | :226-228 query-only mask
                                   :233-236 hardest-negative 计算 | :245 硬编码 0
                                   :259-260 GCL-2 query 再归一化 | :263-266 槽位索引
                                   :277 GCL-2 loss | :279-281 声明计数 | :282 audit 字段
                                   :360-368 Full GCL 循环 | :371 Full loss | :380-382 audit
                                   :425-427 fixed/rotating 关系分配 | :430-439 additive IT
src/training/backends.py           :72-104 训练 transform | :107-147 _relation_step
                                   :123 additive IT 调用点 | :205-255 CLIP backend
                                   :223-224 clip.load + .float() | :246 logit_scale.exp()
                                   :258-326 VISTA backend | :279 normlized=True | :280 temperature
                                   :305-311 注释 | :321 1.0/temperature
                                   :336-432 BEiT-3 backend | :425 joint_encoder=None
                                   :496-653 ALBEF backend | :592-597 归一化位置 | :629-653 forward
src/training/engine.py             :445 backend.train() | :655 audit 写日志
                                   :698-701 raw_total_loss | :702-706 loss_denominator
                                   :736-737 scaled/total loss | :758 run_validation
src/training/validation.py         :357/362/384 train-eval 切换与恢复
src/model_adapters/clip_openai.py  :43 clip.load（无 .float()）| :50 fp16 autocast
src/model_adapters/beit3.py        :54 strict=False | :58 to(device).eval()
src/model_adapters/vista.py        :41-47 构造（normlized=False）| :56-68 encode_*
src/model_adapters/albef.py        :134/144-166
src/datasets/probes.py             :53 minimum caption id | :55-58 硬校验 | :62 image_id 升序
third_party/flag_embedding/research/visual_bge/visual_bge/modeling.py
                                   :33/76/78/79-81/91 temperature | :219-221 encode_text L2
                                   :291-293 encode_mm L2 | :317 encode_image 委托
```

**产物**

```
outputs/training/<run>/train_metrics.jsonl    分母反解、logit_scale、梯度范数、旧/新 schema
outputs/training/<run>/run_manifest.json      config.model_options.temperature、control_guarantees
outputs/training/<run>/checkpoints/           trajectory/*_model.pt、resume/step_*.pt、final.json
data/splits/coco_2017_val_probe_v1.json       COCO probe（5000）
data/splits/lcs_558k_in_domain_probe_v1.json  LCS probe（10000）
data/raw/models/{clip_sf,beit3,vista,albef}/  M0 权重与其 state_dict 键
```
