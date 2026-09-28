> 先验阶段知识参考。当前训练参数和执行方式以[主README](../README.md)及正式协议为准。文中的交接/修改日志仅保留服务器；历史假设和建议不代表新正式协议。原始模长不等时，不可套用等模长的平方根恒等式。
>
> 文中先验专用脚本与旧配置路径指归档内原路径，已退出当前执行目录；归档位置及参考方法见[主README](../README.md)。

# 项目解构与实验设计说明

> **本文件的用途**：自包含的项目说明，供**没有仓库访问权限**的外部读者（人或模型）理解本项目的实验设计、实现方式、实测数据与当前状态。
> **生成时间**：2026-09-13（UTC）。**生成方式**：逐文件读取代码、配置与产物后撰写；所有数字均从产物中现取，非记忆复述。
> **本文件不含研究结论**。所有对比表都只给数值与算术差，不做机制判断——判断留给讨论。

---

## 0. 一页摘要

**一句话**：把 4 个预训练好的视觉-语言模型当作不可变起点，只改变**对比学习的关系监督结构**（不改架构、不加融合模块、不换数据），观察模态间表征几何如何被"重组"。

**执行规模**：4 个模型 / **17 条**训练 run（每条 16879 步 = 一个 epoch；其中 3 条为 2026-09-14 的 Fixed-6N 重训）/ 每条 run 的轨迹评测覆盖 2 个 probe × 6 个进度点（M0 + 5 个 checkpoint）= 10 份点指标 + 10 份相邻 transition。

**核心对照**：同一模型内，`standard`（经典 I↔T 双向对比）vs `count_matched_mixed`（引入第三模态 `e_IT`，候选池扩到 3N，但**监督计数保持不变**；关系在 I↔T / I↔IT / T↔IT 间轮换）。另有 ALBEF 的原生融合对照与 Full GCL 参考组。

**度量**：八个指标，分三类空间（raw 表征几何 / cosine 几何 / score 级），全部以 **M0 → 分支的变化量**为读数，禁止跨架构比较绝对值。

**当前状态**：17 条训练 + 17 条评测全部完成（170 份点指标、170 份 transition，schema 统一）；同模型 Δ 派生数据完成但**呈现口径未定**；正式矩阵**已冻结为 17 条**。

---

## 1. 研究对象与假设

### 1.1 问题

视觉-语言对比学习（如 CLIP 类模型）训练出的图像表征与文本表征之间存在系统性的**模态间隔（modality gap）**：两个模态的嵌入在共享空间中占据互不重叠的锥形区域，正样本对的相似度显著低于同一模态内的任意两样本。这个间隔不是随机初始化造成的，而是对比目标的固有几何后果。

本项目问的是：**这个间隔能否被"主动重组"？** 具体地，在不动架构、不引入跨模态融合模块、不换预训练数据的条件下，**仅改变关系监督的结构**（监督哪些模态对、以什么计数方式监督），能否改变间隔的大小与几何形态，以及这种改变是否可控、是否可复现于不同模型族。

### 1.2 为什么用"分叉 + 变化量"的设计

四个模型的预训练历史、维度、尺度、tokenizer、甚至是否使用原生联合编码器都完全不同。因此：

- **跨模型的绝对指标值不可比**（raw 空间尤甚：范数量级差 10 倍以上，见 §7.3）。
- 唯一干净的读数是**同一模型内，分支相对其自身不可变起点 M0 的变化量**（`primary_analysis: checkpoint_delta_from_m0`）。

这条规则写进了设计文档与交接文档，是本项目所有结论的前提。

### 1.3 四个起点模型

| 代号 | 权重 | 维度 | 模态融合方式 | checkpoint SHA-256（前 8 位） |
|---|---|---|---|---|
| `clip` | OpenAI CLIP ViT-L/14 | 768 | 双塔，无融合 | `b8cca3fd` |
| `vista` | VISTA stage-1（BGE-base-en-v1.5 + EVA02-CLIP-B-16） | 768 | **原生联合编码器** | `05a4ce30` |
| `beit3` | BEiT-3 base (ITC, patch16, 224) | 768 | 双塔，无融合 | `d483a41d` |
| `albef` | ALBEF 14M | 256 | **原生 cross-attention 融合** | `41c10616` |

---

## 2. 实验设计

设计声明在 `configs/training/experiment_comparisons.yaml`（状态 `locked`），包含三个对照族。

### 2.1 主族：`relation_intervention`（clip / vista / beit3）

| 分支 | 监督的关系 | 每 query 正例数 | 每 query 负例数 | 每步方向数 | loss 分母 |
|---|---|---|---|---|---|
| `standard` | I↔T | 1 | N−1 | 2 | 2N |
| `count_matched_mixed` | 池 = {I, T, IT}，关系∈{I↔T, I↔IT, T↔IT} | 1 | 3N−2 | 2 | 2N |
| `full_gcl`（参考组） | 池 = {I, T, IT}，全部 6 个有序方向 | 1 | 3N−2 | 6 | 6N |

**第三模态 `e_IT` 的定义**（三种模型各不相同，这是关键）：

- `clip` / `beit3`：**加法**，`e_IT = normalize(e_I + e_T)`。原始和在上游完成，L2 归一化由目标函数对池内每个成员分别执行。**零额外前向**。
- `vista`：**原生联合编码器**，`e_IT = normalize(encode_mm(I, T))`。**多一次前向**。

### 2.2 "计数匹配"的确切含义

`count_matched_mixed` 与 `standard` 的差异被刻意限制在"监督哪一对关系"上，其余全部对齐：

- 每个 query 仍然只有 **1 个正例**（`one_positive_per_query: true`）；
- 候选池扩大到 `{I₁..I_N, T₁..T_N, IT₁..IT_N}`（3N 个条目）；
- **只 mask query 自身在池中的那一个槽**（`masked_candidates_per_query = 1`）；
- 因此有效候选 = 3N−1（含正例），负例 = 3N−2；
- 每步仍然只有 **2 个有向 loss**，positive term 数仍是 2N。

即：**监督密度（每步正例数、方向数、分母）保持不变，只有"关系种类"变化**。这是让 `standard` 与 `count_matched_mixed` 可比的唯一方式。

`fixed` vs `rotating` 的区别**仅**在关系如何分配到 step：

- `fixed`：每一步都用 `I↔T`；
- `rotating`：按 **optimizer step** 轮换，`step % 3` → `I↔T` / `I↔IT` / `T↔IT`。

两者共用同一目标函数、同一候选池、同一 mask 规则。因此 `standard → gcl2_fixed` 隔离出"引入 e_IT 但只监督 I-T"，`gcl2_fixed → gcl2_rotating` 隔离出"把监督扩展到含 IT 的关系"。

### 2.3 ALBEF 族：`albef_native_objective_intervention`

| 分支 | 目标 | 融合 |
|---|---|---|
| `itc_only` | 仅 ITC（含 momentum queue，宽度 N+65536 的软目标对比） | 关闭 |
| `full_albef` | ITC + ITM + MLM | 开启原生 cross-attention |

设计文档明确标注它 **`explicitly_not_equivalent_to`** `count_matched_mixed`，理由：ALBEF **故意不构造人工 `e_IT`**。它考的是"原生跨模态融合训练"这条完全不同的路径，不参与关系干预族的比较。

### 2.4 参考组：`full_gcl`

`full_gcl` 的声明是 `primary_mechanism_comparison: false`、`count_matched: false`。原因：它每步监督 6 个方向、分母 6N，**监督密度与 standard 不同**（是 3 倍）。它的用途仅为 `align_with_prior_gcl_results` 与 `implementation_reference`。**不应把它当作主结论的证据。**

值得注意的是：`rotating` 模式跑满 3 个 step 的**监督方向集合**与 `full_gcl` 的一个 step 相同（都是六方向的并集），但 `rotating` 每步只除以 2N、`full_gcl` 每步除以 6N。这构成一个有意思的对照：**同样的方向覆盖，不同的监督密度**。

### 2.5 已声明的 confound 与不可比性

1. **VISTA 的 IT 混淆**（设计文档自带 `analysis_warning`）：VISTA 的 `e_IT` 来自原生联合编码器，比 CLIP/BEiT-3 多一次前向计算。因此 **VISTA 的关系变化与"联合前向算力"混淆**，其结论的解释力弱于另外两个模型。
2. **VISTA 的"归一化开关"要分清两个实例，不要混淆。** 项目里有两个独立的 VISTA 对象：
   - **评测适配器**（`src/model_adapters/vista.py`）为了拿到 pre-L2 的原始向量，构造时用 `normlized=False`；这个开关有一个附带效果——源码 `modeling.py` 里 `if not normlized: self.temperature = 1.0`，把类属性温度重置为 1.0。**但适配器只调用 `encode_image` / `encode_text` / `encode_mm`，从不调用 `Visualized_BGE.forward`，而温度的唯一消费点就在 `forward` 里。因此这个 1.0 在评测路径上是惰性值，对导出的向量没有任何影响。**
   - **训练 backend**（`src/training/backends.py`）用 `normlized=True` + `temperature=0.02`，训练时把 `1 / temperature = 50` 作为 logit scale。实测四条 VISTA run 的日志，`logit_scale` 恒为 **50.0**（三条 GCL 系 run 各 1690 条记录，取值集合只有 `{50.0}`）。

   **即：VISTA 训练用的是 0.02（比 1.0 更"尖锐"），不是 1.0。** 这个 50 倍的尺度差在跨模型比较训练动力学时必须记住。
3. **跨架构绝对值禁止比较**：见 §1.2。
4. **单 seed**：全部 run 使用 seed 42，`multi_seed` 仍列在 `unresolved_training_hyperparameters` 中。

---

## 3. 执行矩阵：17 条 run

> 下列三张表列出最初的 14 条 run；2026-09-14 追加的三条 **Fixed-6N** 重训（`clip/beit3/vista_gcl2_fixed_6n`，回到原规格 6N 基准，与 2N 的 Fixed 并存作对照）使用同一组控制变量，故未重复列出。

全部 run：`seed=42`、`learning_rate=1e-5`、`weight_decay=0.05`、`warmup_steps=338`、`min_lr_ratio=0.1`、`max_steps=16879`、`micro_batch_size=32`、`gradient_accumulation=1`、`precision=bf16`、`gradient_clip_norm=1.0`、AdamW(0.9, 0.999, 1e-8)、cosine schedule。

`16879 = ceil(540128 / 32) × 1`，即 LCS-558k 训练集恰好 **一个 epoch**。

| run 目录名 | 模型 | 分支 | 完成步数 | 状态 |
|---|---|---|---|---|
| `clip_standard` | clip | standard | 16879 | complete |
| `clip_gcl2_fixed` | clip | count_matched_mixed (fixed) | 16879 | complete |
| `clip_gcl2_rotating` | clip | count_matched_mixed (rotating) | 16879 | complete |
| `clip_full_gcl` | clip | full_gcl | 16879 | complete |
| `beit3_standard` | beit3 | standard | 16879 | complete |
| `beit3_gcl2_fixed` | beit3 | count_matched_mixed (fixed) | 16879 | complete |
| `beit3_gcl2_rotating` | beit3 | count_matched_mixed (rotating) | 16879 | complete |
| `beit3_full_gcl` | beit3 | full_gcl | 16879 | complete |
| `vista_standard` | vista | standard | 16879 | complete |
| `vista_gcl2_fixed` | vista | count_matched_mixed (fixed) | 16879 | complete |
| `vista_gcl2_rotating` | vista | count_matched_mixed (rotating) | 16879 | complete |
| `vista_full_gcl` | vista | full_gcl | 16879 | complete |
| `albef_itc_only` | albef | itc_only | 16879 | complete |
| `albef_full` | albef | full_albef | 16879 | complete |

**配置的冻结状态**（重要）：`configs/training/train_runs.yaml` 是**正式矩阵的唯一权威声明**，2026-09-15 冻结为**17 条 run**（按已执行事实声明，超参已填实，不再有 `null`）。各 `*_execute.yaml` 是历史执行配置，其 `controls`/`models` 与正式矩阵逐字段一致。

---

## 4. 模型层：四个架构的实现差异

### 4.1 统一契约

所有模型适配器实现同一个抽象基类，契约只有一句话：**`encode_image` / `encode_text` 返回 raw、pre-L2 的 embedding**。归一化交给上游目标函数。每个适配器另外提供 `metadata()`（记录 checkpoint 路径与 SHA-256、预处理版本、存储字段、dtype 等）。

### 4.2 逐模型实现路径

**CLIP**：用官方 `clip` 包加载（`jit=False`）。官方实现的 L2 归一化发生在 `forward()` 内部，因此适配器**绕开 `forward`，直接调用 `encode_image` / `encode_text`** 拿到投影层输出。文本用 CLIP 自带 BPE（`truncate=True`），预处理用官方 transform（Resize + CenterCrop + CLIP 均值方差，224）。

**VISTA**：用作者托管的第三方源码构造 `Visualized_BGE`，构造参数 `model_name_bge="bge-base-en-v1.5"`、`normlized=False`、`from_pretrained=<本地 BGE backbone>`。`normlized=False` 的作用是关掉源码里的 `F.normalize`。**但 image 分支并不是纯视觉塔**：`encode_image` 内部把 image token 前缀拼到空文本 prompt 上再送进 BGE，也就是说 VISTA 的 image embedding 本身就经过联合编码器。文本用 BGE tokenizer（max 512）。预处理用官方 `preprocess_val`。**以上描述的是评测适配器的构造；训练侧的构造不同**（`normlized=True` + `temperature=0.02`，即 logit scale = 50，见 §2.5），两者不是同一个实例。

**BEiT-3**：用 `third_party/unilm/beit3`。torch 2.x 下缺 `torch._six`，需运行时伪造。官方 wrapper 的 `forward` 里有 `F.normalize`，因此适配器**绕开 wrapper**，直接调用底层 `beit3(...)` 再手动过 `vision_head` / `language_head`。文本用手写 ids 组装 + padding mask（max 64 tokens，`beit3.spm`）。预处理手写（Resize 224 BICUBIC + IMAGENET_INCEPTION 均值方差），与官方 `datasets.py` 对齐。权重 `strict=False`，但**不允许 unexpected keys**（遇 unexpected 直接抛错），missing keys 记入 metadata。

**ALBEF**：用官方源码 `third_party/albef`。官方 L2 同样在 `forward` 里，适配器**绕开 `forward`**，直接走 `visual_encoder + vision_proj` / `text_encoder.bert(mode="text") + text_proj`。维度 256、分辨率 256、文本 `_pre_caption` 截断到 30 词。权重加载 `strict=False`，不允许 unexpected。两个历史坑：① 官方包也叫 `models`，与本项目包名冲突，最终**把项目包改名为 `model_adapters`**，并在导入时拒绝覆盖已存在的同名包；② 官方实现的单卡 gather 与 MLM mask 在现代 PyTorch/CUDA 下有兼容问题，修正放在项目侧运行时，不改第三方源码。

### 4.3 差异对照

| | clip | vista | beit3 | albef |
|---|---|---|---|---|
| raw 的取法 | 绕开 `forward` | **评测侧**构造时 `normlized=False`（训练侧为 `normlized=True` + `temperature=0.02`） | 绕开 wrapper | 绕开 `forward` |
| 维度 | 768 | 768 | 768 | **256** |
| 分辨率 | 224 | 224 | 224 | **256** |
| 文本最大长度 | 77（truncate） | 512 | 64 | 30 |
| 有 e_IT？ | 加法 | **原生联合编码器** | 加法 | **无（故意）** |
| IT 额外前向 | 无 | **有（confound）** | 无 | — |
| 第三方依赖 | `clip` 包 | FlagEmbedding | UNILM/beit3 | 官方 albef |
| 权重加载 | strict | strict | strict=False，禁 unexpected | strict=False，禁 unexpected |

### 4.4 训练侧与评测侧是两套代码

模型的加载与前向在项目里有**两条独立路径**：评测侧（`src/model_adapters/`，用于导出 M0 与轨迹 embedding）与训练侧（`src/training/backends.py`）。两者严格性不同、实现不复用。这一点在阅读代码时容易误判。

---

## 5. 目标函数层

记一个 micro-batch 内有 N 个**唯一语义 id**（批次内重复的 semantic_id 直接报错），`s = logit_scale`（可学习，指数化后使用），`ẑ = z / ‖z‖₂`。

### 5.1 `standard`

```
L = ½ · [ CE( s · ẑ_I ẑ_Tᵀ , arange(N) ) + CE( s · ẑ_T ẑ_Iᵀ , arange(N) ) ]
```

- 方向数 2；分母 2N（由外层的 ½ 隐式给出）；候选 N、负例 N−1；**无 mask**（批次内语义 id 唯一，不存在同实例假负例）。
- CLIP/BEiT-3 的 `logit_scale` 每步后 clamp 到 ≤ log 100。

### 5.2 `count_matched_mixed`

池 `P = [ ẑ_I ; ẑ_T ; ẑ_IT ] ∈ R^{3N×d}`（每个成员**独立** L2 归一化后拼接）。

单方向：

```
L_{q→p} = CE( mask_self( s · ẑ_q Pᵀ ) , pos_idx )
```

`mask_self` 只把 query 自己在池中的槽位置为 `-inf`。

```
L = ( L_{l→r} + L_{r→l} ) / 2
```

- 方向数 2（当前 relation 的两个方向）；分母 2N；候选 3N−1；负例 3N−2；`masked_candidates_per_query = 1`。
- **注意**：由于只 mask query 自身槽，同一语义实例的**其他模态**条目（如 query 为 `I_i` 时的 `IT_i`）**保留在 softmax 分母中，被当作负例**。这是设计选择（见 §11.2）。

### 5.3 `full_gcl`

同一池，对 6 个有序方向逐个计算同一套 loss：

```
L = (1/6) · Σ_{k=1..6} L_{q_k→p_k}
```

方向顺序**冻结**为：`I→T, T→I, I→IT, IT→I, T→IT, IT→T`。

- 方向数 6；分母 6N；候选 3N−1；负例 3N−2。

### 5.4 ALBEF

- `itc_only`：`sim_i2t = ẑ_I · [ẑ_T^momentumᵀ ; queue_T] / temp`，软目标 `targets = α·softmax(sim^m) + (1−α)·diag`，损失为两个方向的软交叉熵平均。**分母是行宽 N + 65536（队列），不是 2N**，不进入 GCL 的分母体系。`temp` 可学习，forward 内 clamp 到 [0.001, 0.5]。
- `full_albef`：`L = ITC + ITM + MLM`。ITM 用 ITC 权重做难负例采样（3N 行 CE）；MLM 用 momentum encoder 的软标签，`α` 按 warmup 线性上升。

### 5.5 训练日志里的目标函数字段

每一步记录：`loss_direction_1..6`（按冻结顺序填，缺位补 `null`）、`loss_directions`（字典）、`loss_denominator`（**声明值**，standard=2.0 / GCL-2=2.0 / full_gcl=6.0，不是实测值）、`raw_total_loss`（各方向之和）、`scaled_total_loss` / `total_loss`、`logit_scale`、`relation_audit`（仅 GCL 系非空）、`training_branch` / `mixed_mode` / `gcl_mode`。

---

## 6. 训练引擎

### 6.1 一个 optimizer step 的流程

```
设定 lr（warmup 线性 + cosine 退火到 min_lr_ratio）
→ zero_grad
→ [ALBEF] 准备 momentum 更新
→ micro-step 循环：
     取 batch → 派生 augmentation seed → 预处理
     → 派生 forward seed
     → autocast(bf16) 下前向 → loss
     → scaled_loss = loss / gradient_accumulation
     → 非有限 loss 直接抛 FloatingPointError
     → backward
→ [ALBEF] 把各 micro-batch 特征拼接后一次性入队
→ relation audit 合并校验（累积窗口内 relation 必须唯一，计数必须一致）
→ 梯度裁剪（clip norm = 1.0）
→ optimizer.step()
→ [CLIP/BEiT-3] clamp logit_scale
```

### 6.2 控制变量：随机性全部显式派生

这是"分支之间可比"的技术基础。四类随机性都被绑定到确定的位置，而不是消耗全局 RNG：

| 来源 | 派生式 | 效果 |
|---|---|---|
| 数据顺序 | sampler = `randperm(seed + epoch)` | 各分支同 epoch 同顺序 |
| 数据增强 | `seed + 1_000_003 × epoch + batch_index`，**在父进程派生** | `num_workers` 不影响结果 |
| 模型前向（dropout / drop-path） | `seed + 2_000_003 + step × accum + micro_step` | 与累积窗口位置绑定 |
| 验证 | `seed + 5_000_003 + batch_index`，用完还原 RNG | 验证不扰动训练流 |

增强算子：`RandomResizedCrop(224, scale 0.9–1.0, bicubic)` + `RandomHorizontalFlip(0.5)` + ToTensor + Normalize（ALBEF 为 256）。

### 6.3 Checkpoint 协议

两类 checkpoint，用途完全不同：

| 类型 | 触发点 | 内容 | 保留 |
|---|---|---|---|
| `trajectory_model` | 进度 1% / 5% / 20% / 50% / 100% | **仅模型权重** + provenance | 永久，不清理 |
| `full_resume` | 每 20%（及末步） | 模型 + 优化器 + RNG(python/numpy/torch/cuda) + 数据流位置 | 保留最近 2 个 |

- 100% 的轨迹点由 `checkpoints/final.json` 指向末步 resume checkpoint。
- `trajectory_index.json` **不是训练写的**，是评测完成每个快照后写出的。
- resume 会校验 `run_id` 与 `config_sha256` 一致，并按 `sampler.set_epoch(epoch)` + 快进 `batch_index` 恢复到精确位置。

### 6.4 验证

只在 20% 间隔与末步触发。验证**不更新参数、不选择 checkpoint**，在 `inference_mode` 与保存/还原 RNG 下运行。共同考试恒为 `I↔T`；Mixed 分支额外记录 `I↔IT`、`T↔IT` 诊断项；ALBEF 用只读的原生 ITC + ITM/MLM 诊断。

---

## 7. 评测层

### 7.1 轨迹结构

每条 run 的评测点序列：`M0 → p001 → p005 → p020 → p050 → p100`（进度百分比）。

对每个点 × 每个 probe（`coco_2017_val_5k`，5000 样本；`lcs_558k_in_domain_10k`，10000 样本）：

1. 导出 raw embedding（图像 + 文本，**pre-L2**），存为不可变 npz（含样本 id 顺序、metadata、SHA）；
2. 计算八项指标；
3. 与相邻点、以及与 M0 计算 transition（指标差的完整记录 + 几何保真度）。

每个 npz 的存储字段是 `sample_ids` / `image_embeddings_raw` / `text_embeddings_raw`（float32）。此外有一个**可选的点云导出开关**（`--save-probe-embeddings`）：开启时会额外写出 raw 与归一化后的图像、文本矩阵，以及 clip/beit3 的加法 IT 向量（`e_I + e_T` 的原始和）。**该开关在全部 17 条评测中从未开启**，因此现有产物里没有 IT 向量，IT 只在训练侧的目标函数内部使用。

### 7.2 八项指标的精确定义

**① Centroid Gap**（raw 为主、L2 为辅）
```
centroid_gap = ‖ mean_i(e_I,i) − mean_j(e_T,j) ‖₂
```
两个模态质心的欧氏距离，不除以维度。

**② Covariance Gap**（raw 为主、L2 为辅）——相对 Frobenius
```
Σ = 样本协方差（rowvar=False, ddof=1）
covariance_gap = ‖Σ_I − Σ_T‖_F / ( ‖Σ_I‖_F + ‖Σ_T‖_F )
```
分母以 `1e-12` 兜底（仅防退化，不改定义）。

**③ Effective Rank**（每个模态各算，raw 与 L2 各一份）

对协方差矩阵做 `eigvalsh`，负特征值截断为 0，丢弃 `λ ≤ 1e-12` 的分量，令 `p_k = λ_k / Σλ`：
```
effective_rank = exp( − Σ_k p_k log p_k )
```
即"归一化协方差谱的熵的指数"。若总能量 ≤ 1e-12 返回 0.0。

**④ Cross-modal Alignment**（cosine 几何）

行 L2 归一化后，配对样本的欧氏距离：
```
alignment = mean_i ‖ ê_I,i − ê_T,i ‖₂
```
另记录分布（std / q25 / median / q75）与配对 cosine 的 mean/std 作为辅助。

**⑤ Intra-modal Geometry Preservation**（cosine 几何，**M0 ↔ 当前 checkpoint**）

在**固定的上三角索引对集合**（1,000,000 对，seed 20_260_825，按 probe manifest SHA 命名）上，取每个模态的 pair cosine 向量，计算 M0 与 checkpoint 之间的 **Spearman 秩相关**：
```
intra_geometry = spearman( cos_pairs(M0), cos_pairs(ckpt) )
```
两个状态共用同一份 pair 索引，因此样本对应关系不会因 batch 顺序或分块漂移。Neighbor Overlap@10 作为辅助指标（不计入八项）。**注意：M0 行该值为 1.0（参考与自身比较），是定义使然，不是测量结果。**

**⑥ Score Gap**（`category = score_level`）

对每个方向（image-query：`S_same(i,j) = ê_I,i · ê_I,j`，`S_cross(i,j) = ê_I,i · ê_T,j`，遍历所有 `j ≠ i`，排除自身与配对正例），取两个分布的 **Wasserstein-1 距离**（等样本量下即"排序后逐元素的平均绝对差"）：
```
W1 = mean_k | sort(S_same)_k − sort(S_cross)_k |
score_gap_mean = ( W1_image_query + W1_text_query ) / 2
```
另记录 `directional_bias = mean(S_same − S_cross)`。**注意**：当前实现是双模态兼容版，A 类实验所需的 I/T/IT **三模态** Score Gap **尚未实现**。

**⑦ Norm Imbalance**（**只算 raw**，L2 下所有行范数为 1，无意义）
```
norm_imbalance = | log( E‖e_I‖ / E‖e_T‖ ) |
```
同时输出 `image_text_norm_ratio`（保留方向）与两个模态的 norm 均值/标准差。

**⑧ Anisotropy**（cosine 几何，每个模态各一份）
```
anisotropy = E_{i ≠ j}[ cos(e_i, e_j) ]
```
分块计算（block 512），**对角线从求和与计数中同时剔除**。因 cosine 本身按方向归一化，raw 与 L2 空间给出同一个值。另出 `anisotropy_gap = |anisotropy_image − anisotropy_text|`。

### 7.3 raw / L2 空间规则（为什么必须区分）

四个模型导出的存储向量范数**全部远离 1**，即都是真正的 pre-L2 输出：

| 模型 | image 范数 | text 范数 | text/image |
|---|---|---|---|
| clip | 18.57 | 13.23 | ≈ 0.71 |
| vista | 16.31 | 16.20 | ≈ 0.99 |
| beit3 | 38.72 | 112.33 | ≈ 2.90 |
| albef | 8.94 | 5.68 | ≈ 0.64 |

规则：**Centroid / Covariance / Effective Rank 出 raw + L2 两份；Alignment / Intra-geometry / Score Gap / Anisotropy 走 cosine 几何；Norm Imbalance 只算 raw。** 四个模型的 `raw_available` 全为 `true`（本来为 VISTA 准备的 fallback 未启用）。

### 7.4 产物与 schema

每个点指标 JSON 同时包含**扁平键**（`centroid_gap_raw`、`anisotropy_image`、`norm_imbalance` …，供画图与分析消费）、**既有嵌套键**（`centroid_gap.{raw, l2_normalized}` …，兼容早期产物与验证器）、`auxiliary_metrics`、`metric_protocol`（记录 epsilon、分块大小、类别映射，供复现）、以及 `identity`（model / training_regime / checkpoint / global_step / training_progress / dataset / split / sample_count）。

---

## 8. 数据与不可变基线

### 8.1 数据

| 用途 | 规模 |
|---|---|
| 训练集（LCS-558k） | 540128 对 |
| 验证集 | 8000 |
| in-domain probe | 10000（LCS） |
| COCO probe | 5000（COCO 2017 val） |

数据身份通过 split lock 与逐文件 SHA-256 在训练前校验。

### 8.2 M0 与 SHA 锚点

M0 是四个模型**未经过任何训练**的原始表征，是整个实验的不可变参照。M0 基线由 **32 个文件**组成——每个模型 × probe（4 × 2 = 8 组）各 4 个：原始 embedding（`.npz` + 元数据 `.json`）、指标文档（`m0_*_metrics.json`）与几何参考（`m0_*_geometry_reference.npz`）。这 32 个文件由 `data/metadata/m0_sha256.txt` 覆盖，评测前后都会用 `sha256sum -c` 校验，用于证明"起点从未被改动"。历史上发生过一次迁移脚本就地覆盖派生文件的事故，已回滚并补上机制性防护（见 §11.1）。

### 8.3 M0 基线文件

M0 的指标只有**一份**文档：`outputs/metrics/m0/<model>/<probe>/m0_*_metrics.json`，其 schema 与各 checkpoint 的点指标一致，所以"从 M0 出发的变化量"两侧同构、无需任何转换。该文档由 `scripts/evaluation/validate_m0_outputs.py` 做语义校验（样本 ID 顺序、形状、dtype、probe 身份、指标自洽、几何参考指向固定 pair index）。

**M0 基线是冻结数据，不重算**：用同一份 embedding 重新计算会在末位产生 ~1e-15 的相对漂移（跨平台/多线程 BLAS 归约顺序），因此本项目的规则是"基线只读取、只复用"。该基线的 8 条记录如下（COCO 5k / LCS 10k）：

| 模型 | probe | centroid_raw | rank_I | norm_imbalance | anisotropy_I |
|---|---|---|---|---|---|
| clip | COCO 5k / LCS 10k | 13.87 / 12.04 | 225.7 / 310.6 | 0.339 / 0.273 | 0.495 / 0.395 |
| vista | COCO 5k / LCS 10k | 6.99 / 5.55 | 144.0 / 262.7 | 0.007 / 0.046 | 0.263 / 0.160 |
| beit3 | COCO 5k / LCS 10k | 63.89 / 65.34 | 110.9 / 179.6 | 1.065 / 1.202 | 0.512 / 0.295 |
| albef | COCO 5k / LCS 10k | 6.82 / 6.23 | 114.0 / 133.5 | 0.453 / 0.539 | 0.269 / 0.188 |

---

## 9. 实测数据

> 以下为 p100（训练终点）与 M0 的全部八项指标值。**只给数值，不给结论。** 读数规则：同模型内比较 M0 → 各分支；跨模型不比绝对值。

### 9.1 COCO 2017 val 5K（5000 样本）

| run | 点 | centroid_raw | centroid_l2 | cov_raw | rank_I | rank_T | align | norm_imb | aniso_I | aniso_T | intra_I | intra_T | score_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clip_standard | M0 | 13.87 | 0.8214 | 0.6122 | 225.7 | 158.5 | 1.220 | 0.3392 | 0.4947 | 0.3503 | 1 | 1 | 0.3373 |
| clip_standard | p100 | 11.12 | 0.7691 | 0.5306 | 181.7 | 135.9 | 1.183 | 0.2454 | 0.2813 | 0.3021 | 0.6715 | 0.7527 | 0.2957 |
| clip_gcl2_fixed | M0 | 13.87 | 0.8214 | 0.6122 | 225.7 | 158.5 | 1.220 | 0.3392 | 0.4947 | 0.3503 | 1 | 1 | 0.3373 |
| clip_gcl2_fixed | p100 | 33.04 | 0.0220 | 0.9330 | 20.49 | 19.90 | 0.1197 | 1.679 | 0.9217 | 0.9225 | 0.2746 | 0.2612 | 0.0004445 |
| clip_gcl2_rotating | M0 | 13.87 | 0.8214 | 0.6122 | 225.7 | 158.5 | 1.220 | 0.3392 | 0.4947 | 0.3503 | 1 | 1 | 0.3373 |
| clip_gcl2_rotating | p100 | 1.158 | 0.0708 | 0.2445 | 90.85 | 72.92 | 0.3021 | 0.00785 | 0.8601 | 0.8657 | 0.2918 | 0.3616 | 0.003072 |
| clip_full_gcl | M0 | 13.87 | 0.8214 | 0.6122 | 225.7 | 158.5 | 1.220 | 0.3392 | 0.4947 | 0.3503 | 1 | 1 | 0.3373 |
| clip_full_gcl | p100 | 0.6841 | 0.0256 | 0.0824 | 29.39 | 28.76 | 0.1526 | 0.0191 | 0.9017 | 0.9037 | 0.2808 | 0.2860 | 0.001022 |
| beit3_standard | M0 | 63.89 | 0.6491 | 0.8641 | 110.9 | 99.69 | 0.9837 | 1.065 | 0.5123 | 0.4405 | 1 | 1 | 0.2106 |
| beit3_standard | p100 | 74.11 | 0.6372 | 0.8444 | 103.9 | 89.74 | 0.9756 | 1.232 | 0.3404 | 0.4986 | 0.7981 | 0.8792 | 0.2030 |
| beit3_gcl2_fixed | M0 | 63.89 | 0.6491 | 0.8641 | 110.9 | 99.69 | 0.9837 | 1.065 | 0.5123 | 0.4405 | 1 | 1 | 0.2106 |
| beit3_gcl2_fixed | p100 | 209.4 | 0.0484 | 0.9953 | 21.41 | 25.18 | 0.1445 | 2.982 | 0.9224 | 0.9120 | 0.4318 | 0.3838 | 0.005174 |
| beit3_gcl2_rotating | M0 | 63.89 | 0.6491 | 0.8641 | 110.9 | 99.69 | 0.9837 | 1.065 | 0.5123 | 0.4405 | 1 | 1 | 0.2106 |
| beit3_gcl2_rotating | p100 | 10.44 | 0.1746 | 0.3830 | 68.64 | 66.83 | 0.3635 | 0.0896 | 0.8705 | 0.8121 | 0.5419 | 0.4195 | 0.02928 |
| beit3_full_gcl | M0 | 63.89 | 0.6491 | 0.8641 | 110.9 | 99.69 | 0.9837 | 1.065 | 0.5123 | 0.4405 | 1 | 1 | 0.2106 |
| beit3_full_gcl | p100 | 5.377 | 0.0761 | 0.1827 | 29.11 | 31.51 | 0.1881 | 0.0541 | 0.9041 | 0.8831 | 0.4546 | 0.3777 | 0.01051 |
| vista_standard | M0 | 6.988 | 0.4314 | 0.4240 | 144.0 | 145.5 | 0.9611 | 0.00705 | 0.2629 | 0.4016 | 1 | 1 | 0.0931 |
| vista_standard | p100 | 4.552 | 0.2852 | 0.3171 | 119.2 | 111.0 | 0.8346 | 0.00404 | 0.2807 | 0.3149 | 0.7563 | 0.6887 | 0.04062 |
| vista_gcl2_fixed | M0 | 6.988 | 0.4314 | 0.4240 | 144.0 | 145.5 | 0.9611 | 0.00705 | 0.2629 | 0.4016 | 1 | 1 | 0.0931 |
| vista_gcl2_fixed | p100 | 1.962 | 0.1227 | 0.2407 | 114.0 | 106.5 | 0.7274 | 0.00192 | 0.3391 | 0.3477 | 0.7195 | 0.6413 | 0.007472 |
| vista_gcl2_rotating | M0 | 6.988 | 0.4314 | 0.4240 | 144.0 | 145.5 | 0.9611 | 0.00705 | 0.2629 | 0.4016 | 1 | 1 | 0.0931 |
| vista_gcl2_rotating | p100 | 1.306 | 0.0799 | 0.2126 | 65.81 | 60.03 | 0.4106 | 0.000866 | 0.7402 | 0.7475 | 0.6326 | 0.5399 | 0.003974 |
| vista_full_gcl | M0 | 6.988 | 0.4314 | 0.4240 | 144.0 | 145.5 | 0.9611 | 0.00705 | 0.2629 | 0.4016 | 1 | 1 | 0.0931 |
| vista_full_gcl | p100 | 1.288 | 0.0803 | 0.1906 | 66.07 | 60.61 | 0.4068 | 0.00747 | 0.7247 | 0.7311 | 0.6723 | 0.5108 | 0.003620 |
| albef_itc_only | M0 | 6.825 | 1.062 | 0.9864 | 114.0 | 33.73 | 1.346 | 0.4534 | 0.2687 | 0.9944 | 1 | 1 | 0.5643 |
| albef_itc_only | p100 | 22.57 | 1.137 | 0.6327 | 104.2 | 82.23 | 1.289 | 0.6909 | 0.5309 | 0.9666 | 0.8183 | 0.4784 | 0.6468 |
| albef_full | M0 | 6.825 | 1.062 | 0.9864 | 114.0 | 33.73 | 1.346 | 0.4534 | 0.2687 | 0.9944 | 1 | 1 | 0.5643 |
| albef_full | p100 | 26.03 | 1.157 | 0.5634 | 104.4 | 77.69 | 1.322 | 0.9869 | 0.4913 | 0.9700 | 0.8276 | 0.4764 | 0.6696 |

### 9.2 LCS 558k in-domain 10K（10000 样本）

| run | 点 | centroid_raw | centroid_l2 | cov_raw | rank_I | rank_T | align | norm_imb | aniso_I | aniso_T | intra_I | intra_T | score_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clip_standard | M0 | 12.04 | 0.7281 | 0.6297 | 310.6 | 287.5 | 1.201 | 0.2727 | 0.3949 | 0.2673 | 1 | 1 | 0.2650 |
| clip_standard | p100 | 10.26 | 0.7145 | 0.5670 | 271.9 | 278.4 | 1.176 | 0.2152 | 0.1916 | 0.2346 | 0.7029 | 0.7043 | 0.2552 |
| clip_gcl2_fixed | M0 | 12.04 | 0.7281 | 0.6297 | 310.6 | 287.5 | 1.201 | 0.2727 | 0.3949 | 0.2673 | 1 | 1 | 0.2650 |
| clip_gcl2_fixed | p100 | 32.13 | 0.00624 | 0.9296 | 33.20 | 32.95 | 0.1139 | 1.656 | 0.8971 | 0.8971 | 0.1858 | 0.2177 | 0.0000852 |
| clip_gcl2_rotating | M0 | 12.04 | 0.7281 | 0.6297 | 310.6 | 287.5 | 1.201 | 0.2727 | 0.3949 | 0.2673 | 1 | 1 | 0.2650 |
| clip_gcl2_rotating | p100 | 0.6344 | 0.0382 | 0.1307 | 150.0 | 143.0 | 0.3006 | 0.00780 | 0.8360 | 0.8395 | 0.1836 | 0.3137 | 0.001774 |
| clip_full_gcl | M0 | 12.04 | 0.7281 | 0.6297 | 310.6 | 287.5 | 1.201 | 0.2727 | 0.3949 | 0.2673 | 1 | 1 | 0.2650 |
| clip_full_gcl | p100 | 0.5123 | 0.00790 | 0.0482 | 50.61 | 50.62 | 0.1510 | 0.0241 | 0.8732 | 0.8741 | 0.1895 | 0.2347 | 0.000477 |
| beit3_standard | M0 | 65.34 | 0.5085 | 0.8703 | 179.6 | 173.2 | 0.9692 | 1.202 | 0.2945 | 0.4001 | 1 | 1 | 0.1293 |
| beit3_standard | p100 | 76.08 | 0.5534 | 0.8518 | 169.6 | 164.3 | 0.9661 | 1.316 | 0.2075 | 0.4654 | 0.8138 | 0.7901 | 0.1531 |
| beit3_gcl2_fixed | M0 | 65.34 | 0.5085 | 0.8703 | 179.6 | 173.2 | 0.9692 | 1.202 | 0.2945 | 0.4001 | 1 | 1 | 0.1293 |
| beit3_gcl2_fixed | p100 | 209.5 | 0.00800 | 0.9949 | 34.83 | 35.35 | 0.1176 | 2.985 | 0.8944 | 0.8946 | 0.3768 | 0.2381 | 0.000134 |
| beit3_gcl2_rotating | M0 | 65.34 | 0.5085 | 0.8703 | 179.6 | 173.2 | 0.9692 | 1.202 | 0.2945 | 0.4001 | 1 | 1 | 0.1293 |
| beit3_gcl2_rotating | p100 | 2.517 | 0.0461 | 0.1409 | 116.1 | 114.4 | 0.2836 | 0.00623 | 0.8458 | 0.8426 | 0.4328 | 0.3470 | 0.001630 |
| beit3_full_gcl | M0 | 65.34 | 0.5085 | 0.8703 | 179.6 | 173.2 | 0.9692 | 1.202 | 0.2945 | 0.4001 | 1 | 1 | 0.1293 |
| beit3_full_gcl | p100 | 1.211 | 0.00902 | 0.0499 | 49.29 | 48.73 | 0.1486 | 0.0185 | 0.8731 | 0.8743 | 0.3927 | 0.2604 | 0.000607 |
| vista_standard | M0 | 5.547 | 0.3574 | 0.3300 | 262.7 | 248.5 | 0.9079 | 0.0464 | 0.1598 | 0.4186 | 1 | 1 | 0.1294 |
| vista_standard | p100 | 3.304 | 0.2081 | 0.2524 | 211.2 | 212.2 | 0.8347 | 0.0153 | 0.1975 | 0.2603 | 0.6893 | 0.6469 | 0.03167 |
| vista_gcl2_fixed | M0 | 5.547 | 0.3574 | 0.3300 | 262.7 | 248.5 | 0.9079 | 0.0464 | 0.1598 | 0.4186 | 1 | 1 | 0.1294 |
| vista_gcl2_fixed | p100 | 0.9730 | 0.0616 | 0.1398 | 208.9 | 198.6 | 0.7268 | 0.00676 | 0.2847 | 0.3076 | 0.6682 | 0.5531 | 0.01147 |
| vista_gcl2_rotating | M0 | 5.547 | 0.3574 | 0.3300 | 262.7 | 248.5 | 0.9079 | 0.0464 | 0.1598 | 0.4186 | 1 | 1 | 0.1294 |
| vista_gcl2_rotating | p100 | 0.7682 | 0.0475 | 0.1244 | 105.6 | 102.9 | 0.4224 | 0.00417 | 0.6926 | 0.7112 | 0.5496 | 0.5407 | 0.009309 |
| vista_full_gcl | M0 | 5.547 | 0.3574 | 0.3300 | 262.7 | 248.5 | 0.9079 | 0.0464 | 0.1598 | 0.4186 | 1 | 1 | 0.1294 |
| vista_full_gcl | p100 | 0.6801 | 0.0431 | 0.1088 | 113.1 | 105.3 | 0.4154 | 0.00909 | 0.6691 | 0.6840 | 0.6131 | 0.5157 | 0.007474 |
| albef_itc_only | M0 | 6.231 | 1.015 | 0.9816 | 133.5 | 31.70 | 1.338 | 0.5390 | 0.1880 | 0.9918 | 1 | 1 | 0.5157 |
| albef_itc_only | p100 | 22.93 | 1.133 | 0.6466 | 128.8 | 92.89 | 1.291 | 0.6051 | 0.5207 | 0.9627 | 0.6997 | 0.3185 | 0.6416 |
| albef_full | M0 | 6.231 | 1.015 | 0.9816 | 133.5 | 31.70 | 1.338 | 0.5390 | 0.1880 | 0.9918 | 1 | 1 | 0.5157 |
| albef_full | p100 | 26.43 | 1.142 | 0.6125 | 129.6 | 73.90 | 1.327 | 0.9674 | 0.4473 | 0.9658 | 0.7049 | 0.3512 | 0.6522 |

### 9.3 轨迹的完整中间点

上述只是两个端点。每条 run 的实际轨迹为 6 个点（M0 / 1% / 5% / 20% / 50% / 100%），每对相邻点之间都有完整的指标差与几何保真度记录，可用来判断变化是渐进的还是相变式的。此外还有 `m0 → p001` 的 transition，记录了"第一步训练之后几何变化了多少"。

---

## 10. 当前状态

| 事项 | 状态 |
|---|---|
| 17 条训练 run | ✅ 全部 complete，16879 步 |
| 17 条轨迹评测（2 probe × 6 点） | ✅ 全部完成（170 份点指标 + 170 份 transition，schema 统一） |
| 八项指标迁移（早期六项 → 八项） | ✅ 完成 |
| M0 基线（单一八项文档 + 32 行锚点） | ✅ 完成，`sha256sum -c` 32/32 通过 |
| 同模型 Δ 分析 | ✅ 派生数据完成（17 run × 2 probe × 5 点 × 25 键）；**呈现口径未定** |
| 正式矩阵冻结 | ✅ 已冻结为 **17 条**（`train_runs.yaml`，按已执行事实声明） |
| 多 seed 重复 | ❌ 未做（全部 seed 42） |
| A–G preflight 报告 | ⛔ 已作废（2026-09-15，所有者决定；目的已由实际完成的长程训练覆盖） |
| 三模态 Score Gap（I/T/IT） | ❌ 未实现（当前为双模态兼容版） |

硬件与环境：单卡 RTX 4080 SUPER（32 GB，训练中途由 RTX 5090 迁移而来）。跨机器的 BLAS 归约顺序会带来 ~1e-11 的相对末位差异，远小于效应量，但**不应声称跨机器逐位可复现**。

---

## 11. 已知问题与未决事项

### 11.1 已修复的历史事故（记录在案）

1. **M0 锚点被覆盖**：早期的一次指标迁移把 16 个被 SHA 锚点覆盖的派生文件就地覆盖。已回滚（锚点恢复 32/32），并把重算路径重定向到锚点目录之外 + 加断言拦截。
2. **并行驱动器中途死亡**：一个 `local -n` nameref 写法使 `set -u` 触发致命错误，导致第三个训练 run 未自动启动。已手动补跑并修复脚本。教训：SSH 断连对 screen 无害，**容器重启会杀掉一切**。

### 11.2 待讨论 / 待决事项

> **落实状态（2026-09-14；A 于 2026-09-15 进一步处置）**：下述 **A、B 已处理**（A → 在 `configs/training/TRAINING_RUNBOOK.md` 与 `EXPERIMENT_PROTOCOL.md` 中改写为"先讲 GCL 训练设定、再给结论：该字段对 GCL 分支不适用"；B → `experiment_comparisons.yaml` 已改为与实现一致并留修订注记）。**C–F 仍未决**。另：VISTA 温度的表述（本文件 §2.5 / §4.2 / §4.3）已于同日补充"评测适配器 vs 训练 backend 两个实例"的区分。

**A. `same_instance_false_negatives` 审计字段恒为 0，不是测量值。**
该字段在目标函数里被硬编码为 `0`，再写入每一步的训练日志。实测：所有 GCL 分支的 1690 条日志记录该值全为 0，与 standard 分支完全一致。但按现行 mask 规则（只 mask query 自身槽），每个 query 在每个方向上**恰好有 1 个同实例非目标负例**（query 为 `I_i` 时池中的 `IT_i` 就是一个负例）。因此该字段**不能**用作"Mixed 分支不存在同实例假负例"的证据。
**2026-09-15 处置**：该字段已从 `RelationAudit`、日志与代码中**整体移除**（不再输出一个硬编码的假数字）；同实例条目留在分母这一事实改由文档（§11.2 第 4 条、`TRAINING_RUNBOOK.md`）与行为断言（`tests/objectives/test_count_matched_mixed.py::test_same_instance_views_are_active_negatives_not_masked`）固定。

**B. 设计文档中两行措辞与实现不一致。**
`experiment_comparisons.yaml`（状态 `locked`）在 `count_matched_mixed_cl` 下写着 `candidate_pool: current_relation_target_modality_only` 与 `same_semantic_instance_excluded_from_negatives: true`。而现行实现是**全池 3N** 且**不排除**同实例其他模态条目（只有 query 自身槽被 mask）。这两行看起来是早期 pilot 时代的措辞残留，但因为它在一份 `locked` 文档里，容易被直接引用为方法描述。

**C. Full GCL 的方法学可引用性。**
`full_gcl` 的监督密度（每步 6 方向、6N 分母）与 standard 不同，设计文档已声明它不是主机制比较。但它与 `rotating` 的关系（同样的方向集合、不同的密度）是否足以支撑"密度 vs 覆盖面"的推论，仍是开放问题。

**D. VISTA 的 confound 是否可消除。**
VISTA 的 `e_IT` 必须走原生联合编码器（多一次前向），因此它的"关系变化"与"额外算力"混淆。是否存在可行的对照来分离这两者，尚未讨论。

**E. 单 seed 的效力。**
所有 run 单 seed 42。跨 se 方差未知，因此对"小幅变化"的解释力有限（尤其是 VISTA 与 ALBEF 上那些量级较小的变化）。

**F. 分析口径尚未确定。**
主要分析是"同模型 Δ"，但具体呈现方式（端点差 vs 轨迹形状 vs 各指标的联合模式）尚未定义。

### 11.2 已核实的已知条件与限制（任何报告都必须声明）

以下各项均已逐条核验（2026-09-15），属于"已知条件"而不是待修缺陷。

| # | 事实 | 对结论的影响 |
|---|---|---|
| 1 | **`code_commit` 不代表训练源码**：17 条 run 的 provenance 都记 `1fcb037`，但该提交之后工作区还有大量未提交改动（配置侧有 `config_sha256` 兜底，源码侧没有对应物） | 无法仅凭 git 复现每条 run 的确切源码；源码事实以 `change_logs/` 为准 |
| 2 | **Fixed-2N 无法用当前代码直接复现**：现行代码把 `fixed` 映射为 6N，2N 只存在于产物与改动日志中（数学上是同一个函数、只差一个常数因子） | 2N 那三条只能作为历史产物使用，不能再生成同名条件 |
| 3 | **硬件混杂**：三条 2N Fixed 训练于 RTX 5090（09-10），三条 Fixed-6N 训练于 RTX 4080 SUPER（09-14） | 跨机器末位差 1e-11–1e-3；实测两者终态差 0.03%–2.7%，量级更大，故硬件不是主要因素，但仍是混杂 |
| 4 | **同实例非目标条目是负例**：现行 mask 规则只排除 query 自身槽，因此每个 query、每个方向恰好有 1 个同实例条目留在分母里（GCL-2 每步 2N、Full GCL 每步 6N），可成为 hardest negative | 这是**设计选择**（方案 A"排除同实例"经 pilot 后未采用）。原先的日志字段 `same_instance_false_negatives` 已于 2026-09-15 移除（它硬编码 0、会误导）；该事实由单元测试的行为断言固定 |
| 5 | **`/2` 与 `/6` 在本配置下等效**：AdamW 的更新按坐标归一化（量级 ≈ lr，与梯度幅度无关），因此把方向 loss 的外层平均基数从 `/2` 改成 `/6` 几乎不改变更新（实测：同一步 `raw_total_loss` 比值 1.000、`gradient_norm` 比值 2.93 ≈ 3；20 步对照中仅缩放损失的参数相对差 **1.19e-09**）。梯度裁剪叠加其上、同样是尺度不变操作（同对照中"只加裁剪"的参数相对差 **0.000e+00**：`clip_grad_norm_` 只做整体缩放，随后被 AdamW 的归一化抵消）。 | 该对照**不是**"归一化有无影响"的实验，而是"在本优化器下该变量被归一化掉"的验证；若要让它成为可观测变量，需换对幅度敏感的优化器（如 SGD），或改测**等效学习率**（AdamW 下 `/6` 等价于把有效步长缩小 3 倍） |
| 6 | **单 seed**：全部 run 只有 seed 42 | 小幅变化不可解释（跨 seed 方差未知） |
| 7 | **CLIP 评测侧 fp16 / 训练侧 fp32**（已定案） | 相对差 ~3.4e-3，比效应量小 2–3 个数量级；作为已知条件，不再作为缺陷讨论 |
| 8 | **历史 `checkpoint_index.jsonl` 的 trajectory 事件只写文件名**（09-15 起新 run 写 run 相对路径） | 旧索引解析需按 `checkpoints/trajectory/` 补全；历史索引不修改，以免破坏 SHA 审计链 |
| 9 | **5 条最早的 run 已无法按原配置签名续训**：三条 Standard 与两条 ALBEF 的记录里没有 `mixed_mode` 字段（该字段晚于它们启动才加入 `RunConfig`），今天的序列化结果与之不同 | 引擎的 "different controlled configuration" 检查会拒绝 `--resume`；其余 12 条可复现（6 条由冻结矩阵、6 条需用对应 execute 配置 + `--mixed-mode/num_workers` 覆盖） |

---

## 12. 附录

### 12.1 目录地图

```
src/                        七个包，6499 行
  datasets/                 正式数据 manifest、probe、训练对与语义 id
  model_adapters/           四个模型的评测侧加载与前向（M0 用）
  objectives/               目标函数（contrastive.py 是训练路径；gcl.py 仅诊断）
  training/                 训练后端、引擎、配置、checkpoint、验证
  evaluation/               轨迹解析、embedding 导出
  embeddings/               不可变 artifact 读写 + SHA
  metrics/                  八项指标 + 几何状态 + 比较

scripts/                    17 个 py + 8 个 sh + 1 个 ps1，共 3566 行（薄驱动层）
  training/                 入口 train.py、detached/screen 驱动、preflight 与梯度审计
  evaluation/               轨迹评测主程序、M0 导出、smoke 与校验
  data/                     数据准备与 probe 下载（含 Windows 侧 ps1）
  analysis/                 M0 八项行现算与 transition 审计

tests/                      43 个 py，132 个 unittest 用例全部通过

configs/
  training/                 正式矩阵（已冻结，17 条 run）+ 自包含 execute 配置 + 协议 md
  evaluation/               八项指标协议、轨迹评测协议
  data/ models/             数据身份锁、模型 checkpoint 声明

outputs/
  training/<run>/           训练产物（run_manifest、日志、metrics jsonl、checkpoints）
  embeddings/{m0,trajectory}/   不可变 embedding（npz + metadata json）
  metrics/{m0,trajectory}/      点指标、几何状态、transition、trajectory_index
  verification/             评测/校验批处理日志
  analysis/                 现算冻结的分析行
  pilot/ preflight/         非正式 pilot 与引擎 smoke

third_party/                上游源码（pinned，不修改）
```

### 12.2 术语表

| 术语 | 含义 |
|---|---|
| **M0** | 四个模型未经训练的原点表征；不可变参照 |
| **p001 / p005 / p020 / p050 / p100** | 训练进度 1% / 5% / 20% / 50% / 100% 的轨迹点 |
| **`e_I` / `e_T` / `e_IT`** | 图像 / 文本 / 联合（第三）模态表征 |
| **raw / pre-L2** | 未做 L2 归一化的编码器输出（本项目的主空间） |
| **additive IT** | `e_IT = normalize(e_I + e_T)`，仅 clip / beit3 |
| **count-matched** | 每 query 正例数、负例数、每步方向数与 standard 对齐的控制 |
| **relation** | 参与对比的模态对，如 `I↔T`、`I↔IT`、`T↔IT` |
| **transition** | 两个轨迹点之间的指标差 + 几何保真度记录 |
| **intra-geometry** | 同一模态内成对相似度结构的保持度（M0 vs ckpt 的 Spearman） |
| **SHA 锚点** | 覆盖 M0 的 32 个文件、用于证明起点未被改动的 SHA-256 清单 |
| **execute 配置** | 自包含的、值已填满的执行配置（相对正式矩阵的 null 配置） |

### 12.3 可复现性要点

- 全部 run：seed 42、16879 步、bf16、单卡。
- 数据、probe、M0 embedding、模型 checkpoint 全部有 SHA-256 记录并在训练/评测前校验。
- 数据增强与 dropout 的随机性按 §6.2 的显式规则派生，与 `num_workers` 无关。
- 目标函数的 epsilon、分块大小、pair index 参数（1,000,000 对 / seed 20_260_825）都写进了每个指标 JSON 的 `metric_protocol` 字段。
- 已知不可复现项：跨机器的浮点末位（~1e-11）；未做多 seed。
