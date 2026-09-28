# 八项指标协议（实现与定义）

本文件保留已有八指标算子与冻结先验产物的定义对照。**当前A0–A6策略诊断以[诊断运行手册](AB_DIAGNOSTIC_RUNBOOK.md)为准**：Norm Dynamics已改为逐样本log范数比及Delta_out，本文旧Norm Imbalance仅保留为辅助；其余既有数学定义继续复用。历史改动记录见
`change_logs/Training Plan Design（训练方案设计）/Eight Metric Protocol Implementation Change Log（八项指标实现改动日志）.md`。
本批次只改指标统计，未触碰训练代码、loss、retrieval evaluation 与 checkpoint。

---

## 1. 表示空间规则

**Raw（pre-L2）是 representation analysis 的主空间，L2-normalized 是辅助诊断空间。**

| 空间 | 指标 |
|---|---|
| Raw + L2（Raw 主、L2 辅） | Centroid Gap、Covariance Gap、Effective Rank |
| Cosine 几何（L2 行输入，数学上与 raw 等价） | Cross-modal Alignment、Intra-modal Geometry Preservation、Score Gap、Anisotropy |
| **仅 Raw** | Norm Imbalance |

Norm Imbalance 不计算 L2 版本：L2 归一化后每行范数恒为 1，该量无分析意义。代码中不存在
`norm_imbalance_l2` 键（有单测锁定）。

Score Gap 属于 **score 层**，不是纯 representation geometry：所有输出都带
`category = score_level`（`metric_protocol.categories.score_gap` 与 `score_gap.category`），其余七项为
`representation_geometry`。

---

## 2. Embedding 输入与 raw 边界

`e_I = f_I(I)`、`e_T = f_T(T)` 取模型用于表示构造**之前**的输出。需要 mixed 表示且实验定义为
additive 融合时：`e_IT = e_I + e_T`，**先在 raw 空间相加**，随后才各自 L2 归一化。

```text
raw:  e_I, e_T, e_IT = e_I + e_T
l2 :  ê_I = e_I/‖e_I‖₂,  ê_T = e_T/‖e_T‖₂,  ê_IT = e_IT/‖e_IT‖₂
```

原始 raw tensor 不被覆盖；L2 版本只在计算时派生。

取得位置：`adapter.encode_image / encode_text`（`src/model_adapters/`）→
`export_raw_embeddings()`（`src/evaluation/embedding_export.py`）→ 落盘
`image_embeddings_raw` / `text_embeddings_raw`。

### 各模型的 raw 边界（2026-09-11 实测）

| 模型 | image ‖e‖ 均值 | text ‖e‖ 均值 | 归一化后 | `representation_boundary` | `raw_available` |
|---|---|---|---|---|---|---|
| CLIP ViT-L/14 | 18.57 (COCO 5K) | 13.23 | 1.0 | `encoder_output_pre_l2` | true |
| BEiT-3 base | 38.72 | **112.33** | 1.0 | `retrieval_head_output_pre_l2` | true |
| VISTA stage-1 | 16.31 | 16.20 | 1.0 | `native_encoder_output_normlized_false` | true |
| ALBEF 14M | 8.94 | 5.68 | 1.0 | `encoder_output_pre_l2` | true |

**结论：四个模型（含 VISTA）都能提供真正的 pre-L2 raw embedding。** VISTA 的评测适配器以
`normlized=False` 实例化原生模型，实测范数 16.3/16.2，远离 1。因此原规格中为 VISTA 准备的
fallback（`raw_available=false` + `representation_boundary=native_normalized`）**不需要启用**。

注意方向性差异：BEiT-3 的 text 范数远大于 image（比值 0.33），CLIP/ALBEF 相反（1.29 / 1.57），
VISTA 接近平衡（1.01）。这正是 Norm Imbalance 要量化的量。

> 训练侧的 VISTA 是另一条边界：`normlized=False` 会把原生 `temperature` 从 0.02 改成 1.0，
> 所以**训练**保持原生归一化表示，`control_guarantees.representation_boundary =
> vista_native_normalized_encoders`。评测与训练两处不冲突，各自记录。

### IT 规则

| 模型 | 评测侧 IT | 说明 |
|---|---|---|
| CLIP / BEiT-3 | additive `e_I + e_T`（raw 相加） | 与训练侧实验定义一致 |
| VISTA | 原生joint encoder输出 | 现有适配器已暴露encode_multimodal；A类使用原生raw IT，不以加法代替 |
| ALBEF | 不构造 | 协议明确 ALBEF 无人工 IT |

IT 只用于 §13 的可选点云导出，八项指标本身只用 `e_I`、`e_T`。

---

## 3. 八项指标：定义与实现

### 3.1 Centroid Gap

```
μ_I = (1/N) Σ_i x_i^I ;  μ_T = (1/N) Σ_i x_i^T
centroid_gap = ‖μ_I − μ_T‖₂
```

实现 `centroid_gap()`；输出 `centroid_gap_raw`（主）、`centroid_gap_l2`（辅）。

### 3.2 Covariance Gap

```
Σ = (1/(N−1)) Σ_i (x_i − μ)(x_i − μ)ᵀ
covariance_gap = ‖Σ_I − Σ_T‖_F / (‖Σ_I‖_F + ‖Σ_T‖_F)
```

**相对 Frobenius 定义，不是绝对距离。** 实现 `covariance_gap()`；输出
`covariance_gap_raw` / `covariance_gap_l2`。数值保护：分母以
`COVARIANCE_GAP_EPSILON = 1e-12` 兜底（只防 inf/nan，不改变正常情形下的数学），epsilon 写入
`metric_protocol.covariance_gap_epsilon`。协方差用 ddof=1。

### 3.3 Effective Rank

```
λ_1..λ_d = 非负特征值(Σ)
p_k = λ_k / Σ_j λ_j
effective_rank = exp( −Σ_k p_k log p_k )
```

Image / Text **分别**计算。实现 `effective_rank()`；输出 `effective_rank_image_raw`、
`effective_rank_text_raw`、`effective_rank_image_l2`、`effective_rank_text_l2`。
保护：特征值浮点负值 clamp 到0；实现跳过`λ ≤ EFFECTIVE_RANK_EPSILON = 1e-12`对应的熵项，epsilon写入
`metric_protocol.effective_rank_epsilon`。

### 3.4 Cross-modal Alignment

使用 L2 归一化后的 matched pairs：

```
cross_modal_alignment = (1/N) Σ_i ‖ê_i^I − ê_i^T‖₂
```

实现 `cross_modal_alignment()`（均值为主结果；std / 四分位进 `auxiliary_metrics`）。辅助输出
`matched_pair_cosine_mean`、`matched_pair_cosine_std`（实现 `matched_pair_cosine_summary()`）。
因 `‖ê_I − ê_T‖² = 2 − 2cos(I,T)`，不额外定义基于 raw 欧氏距离的主指标。

### 3.5 Intra-modal Geometry Preservation

与 **M0** 在同一批 probe samples 上比较单模态内部几何：

```
S_I^{M0}, S_I^{ckpt} = 同模态 pairwise cosine（固定上三角 pair 集合）
intra_geometry_image = ρ_Spearman( vec(S_I^{M0}), vec(S_I^{ckpt}) )
intra_geometry_text  = ρ_Spearman( vec(S_T^{M0}), vec(S_T^{ckpt}) )
```

实现 `intra_modal_geometry_preservation()` → `compare_geometry_states()`；输出
`intra_geometry_image` / `intra_geometry_text`。

- **pair 集合固定**：1,000,000 对上三角抽样，`GEOMETRY_SEED = 20_260_825`，索引文件按
  `probe_manifest_sha256[:12]` 命名并由 M0 与所有 checkpoint 共享 → 不存在 batch/chunk 导致的配对错位。
  规格允许"其他固定的上三角有效元素"，因此抽样合规；改为全上三角只需调整 pair_count（10K probe 为
  5000 万对，计算与存储上升约 50 倍）。
- self-similarity 不计入（pair 集合本身即 i<j）。
- Spearman 使用平均秩处理并列值。Neighbor Overlap@10 为**辅助**指标，放在
  `auxiliary_metrics.image_neighbor_overlap_k10` / `text_neighbor_overlap_k10`，不是八项之一。
- **相邻点漂移**仍单独记录在 transition 产物里（`point_metric_delta_target_minus_source` +
  `intra_modal_geometry_preservation`），与 M0↔ckpt 的语义区分开。

### 3.6 Score Gap（score 层，当前为双模态兼容版本）

Image query：`D_{I→I}`（同模态背景分）与 `D_{I→T}`（跨模态分），**排除** `I_i→I_i`（self）与
matched positive `I_i→T_i`；Text query 同理：

```
score_gap_image = W1(D_{I→I}, D_{I→T})
score_gap_text  = W1(D_{T→T}, D_{T→I})
score_gap_mean  = (score_gap_image + score_gap_text) / 2
```

Wasserstein-1 对等样本量的一维经验分布等价于排序后平均绝对差（实现即如此，样本量均为 N(N−1)）。
实现 `score_gap()` / `_directional_score_gap()`；输出 `score_gap_image`、`score_gap_text`、
`score_gap_mean`，并标记 `category = score_level`。

当前新增A2实现逐query的I/T/IT背景分布比较，A6实现六方向概率/CE，见`src/evaluation/score_analysis.py`；本文A0的总体分布W1仍单独保留，不能相互替代。

### 3.7 Norm Imbalance（新增）

```
n_I = (1/N) Σ_i ‖e_i^I‖₂ ,  n_T = (1/N) Σ_i ‖e_i^T‖₂
image_text_norm_ratio = n_I / n_T
norm_imbalance = | log(image_text_norm_ratio) |
```

实现 `norm_imbalance()`；输出 `image_norm_mean`、`image_norm_std`、`text_norm_mean`、
`text_norm_std`（std 用 ddof=1）、`image_text_norm_ratio`、`norm_imbalance`。ratio 保留方向，避免
绝对值丢失 `‖e_I‖ > ‖e_T‖` 还是相反。仅 raw。

### 3.8 Anisotropy（新增）

```
anisotropy = E_{i≠j}[ cos(e_i, e_j) ]
```

实现 `anisotropy()`；输出 `anisotropy_image`、`anisotropy_text`，以及辅助的
`anisotropy_gap = |A_I − A_T|`（**不**升级为第九项主指标）。

- 对角线（self cosine = 1）显式置 0 且从分母剔除，分母为 `N·(N−1)`。
- 分块计算（默认 block 512），不构造常驻 N×N 矩阵。
- cosine 自带方向归一化，raw 与 L2 数学等价，因此**只算一次**（输入 L2 行），metadata 记
  `auxiliary_metrics.anisotropy_space = cosine_geometry_raw_equals_l2`。
- 不定义"Raw anisotropy / L2 anisotropy"两个主指标。

---

## 4. Probe 数据一致性

每条 observation 记录：`model`、`training_regime`、`checkpoint`、`global_step`、
`training_progress`、`dataset`、`split`、`sample_count`，以及 `probe_name`、
`probe_manifest_path`、`probe_manifest_sha256`、`training_config_sha256`、
`representation_boundary`、`raw_available`。

一致性保障：同一 probe manifest SHA（跨点校验）＋共享固定 pair index（§3.5）＋相邻点 sample-ID
顺序全等校验（`compute_adjacent_transition`）。

---

## 5. 计算性能

所有 pairwise 统计分块流式，临时矩阵上限 `block × N`（512×10000×4B ≈ 20 MB）。

| probe | N | 每方向候选对数 | 单次 observation 实测 |
|---|---|---|---|
| `coco_2017_val_5k` | 5,000 | 24,995,000 | 3–6 秒 |
| `lcs_558k_in_domain_10k` | 10,000 | 99,990,000 | 11–17 秒 |

未采样：全部候选对都参与（无 random sampling，因此也不需要额外固定 sampling index）。

---

## 6. 输出 schema

`compute_point_metrics()` 同时输出两套键：

- **扁平八项**（本协议主 schema）：`centroid_gap_raw`、`centroid_gap_l2`、`covariance_gap_raw`、
  `covariance_gap_l2`、`effective_rank_image_raw`、`effective_rank_text_raw`、
  `effective_rank_image_l2`、`effective_rank_text_l2`、`cross_modal_alignment`、
  `matched_pair_cosine_mean`、`matched_pair_cosine_std`、`score_gap_image`、`score_gap_text`、
  `score_gap_mean`、`image_norm_mean`、`image_norm_std`、`text_norm_mean`、`text_norm_std`、
  `image_text_norm_ratio`、`norm_imbalance`、`anisotropy_image`、`anisotropy_text`、
  `anisotropy_gap`。
- **既有嵌套键**（向后兼容）：`centroid_gap.{raw,l2_normalized}`、
  `covariance_gap.{...}`、`effective_rank.{raw,l2_normalized}.{image,text}`、`score_gap.*`（含
  `overall_wasserstein_1`）、`cross_modal_alignment_distribution`。

辅助指标统一放 `auxiliary_metrics`：alignment 分布与四分位、matched-pair cosine、score gap 细节、
neighbor overlap、anisotropy_gap、intra-geometry 参考说明、各类 epsilon 与协议字符串见
`metric_protocol`。

**唯一的键类型变化**：`cross_modal_alignment` 由"分布 dict"变为"标量均值"（协议 §3.4 的主结果），
原分布移入 `auxiliary_metrics.cross_modal_alignment_distribution`。仓库内无消费者把它当 dict 使用
（已 grep 确认）；`validate_m0_outputs.py` 只读 `score_gap.*.candidate_pair_count`，不受影响。

---

## 7. 可选点云导出

`--save-probe-embeddings`（默认关闭）额外写出每个 point × probe 的
`raw_image_embeddings`、`raw_text_embeddings`、`normalized_image_embeddings`、
`normalized_text_embeddings`，以及 CLIP/BEiT-3 的 `raw_multimodal_embeddings` /
`normalized_multimodal_embeddings`（`it_definition = additive_raw_e_I_plus_e_T`）。VISTA可导出原生raw/L2 IT；ALBEF不构造人工IT。新诊断默认只保存raw，不重复存L2或可重建的加法IT矩阵。
10K probe 下每份 dump 约 40 MB/矩阵，故不默认开启。

---

## 8. 验证

- 单元测试 `tests/metrics/test_eight_metrics.py`（14 项）：公式对照、L2 不参与 Norm Imbalance、
  anisotropy 去对角线且与 brute force 一致、block size 无关、covariance gap 相对 Frobenius 与退化兜底、
  effective rank 熵公式、alignment 的 2−2cos 恒等式、score gap 排除 self 与 matched positive、
  flat schema 完整性。全量 **129 passed**。
- 历史Smoke结果（先验归档内`scripts/evaluation/smoke_eight_metrics.py`，CPU，只读既有 artifact）：
  **coco_2017_val_5k 与 lcs_558k_in_domain_10k 各 10/10 通过**。
  报告：`outputs/verification/eight_metrics_smoke/20260911T123544Z/report.json`（5K）、
  `.../20260911T123615Z/report.json`（10K）。

### Smoke 数值（CLIP，M0 → Standard p100 → GCL-2 fixed p100）

| probe / observation | centroid raw→l2 | covariance raw→l2 | rank_I raw (l2) | align | norm imb | aniso_I | score | intra_I |
|---|---|---|---|---|---|---|---|---|
| coco 5K M0 | 13.87 → 0.821 | 0.612 → 0.614 | 225.7 (232.7) | 1.220 | 0.339 | 0.4947 | 0.337 | — |
| coco 5K standard | 11.12 → 0.769 | 0.531 → 0.494 | 181.7 (184.6) | 1.183 | 0.245 | 0.2813 | 0.296 | 0.671 |
| coco 5K gcl2-fixed | 33.04 → 0.022 | 0.933 → 0.058 | 20.5 (19.9) | 0.120 | 1.679 | 0.9217 | 0.000 | 0.275 |
| lcs 10K M0 | 12.04 → 0.728 | 0.630 → 0.603 | 310.6 (326.9) | 1.201 | 0.273 | 0.3949 | 0.265 | — |
| lcs 10K standard | 10.27 → 0.714 | 0.567 → 0.490 | 271.9 (284.8) | 1.176 | 0.215 | 0.1916 | 0.255 | 0.703 |
| lcs 10K gcl2-fixed | 32.13 → 0.006 | 0.930 → 0.018 | 33.2 (32.8) | 0.114 | 1.656 | 0.8971 | 0.000 | 0.186 |

（单 checkpoint、单点观测，非结论。CLIP 的 GCL-2 fixed 分支呈现强各向异性、有效秩塌缩与 raw 范数
失衡，方向上与 additive `e_IT` 捷径的预期一致，留待正式分析。）

---

## 9. 与旧实现的差异 / 尚未应用的部分

| 项 | 旧 | 新 |
|---|---|---|
| Intra-modal Geometry | 相邻点之间（p001→p005…） | **M0 ↔ checkpoint**（§3.5）；相邻漂移仍存于 transition |
| Norm Imbalance / Anisotropy | 不存在 | 已实现 |
| 扁平 schema | 不存在 | 已输出（与嵌套键并存） |
| raw/L2 边界标记 | 无 | `raw_available` + `representation_boundary` |
| epsilon 记录 | 无 | `metric_protocol.*_epsilon` |

**schema 现状**：全部 17 条 run 的 point metrics 与 transition 均已统一到本文档描述的 schema；
M0 基线只有一份文档 `outputs/metrics/m0/<model>/<probe>/m0_*_metrics.json`，由
`data/metadata/m0_sha256.txt`（32 行）锚定，`scripts/evaluation/validate_m0_outputs.py` 校验其语义。

---

## 10. 代码索引

| 位置 | 内容 |
|---|---|
| `src/metrics/representation_metrics.py` | 八项指标实现（`compute_point_metrics`）、geometry state 与 Spearman |
| `src/evaluation/trajectory.py` | `REPRESENTATION_BOUNDARY`、`ADDITIVE_IT_MODELS`、`PROBE_DATASET/SPLIT`、`compute_point_output`、`compute_adjacent_transition` |
| `src/evaluation/embedding_export.py` | `export_raw_embeddings`、`save_probe_embedding_dump` |
| `scripts/evaluation/evaluate_trajectory.py` | 轨迹评测入口、`--save-probe-embeddings` |
| `scripts/evaluation/validate_m0_outputs.py` | M0 基线语义校验（embedding / 指标 / 几何参考） |
| 先验归档内`scripts/evaluation/smoke_eight_metrics.py` | 对历史已存产物的八指标检查；已退出当前入口，归档位置见主README |
| `tests/metrics/test_eight_metrics.py` | 指标单元测试 |

---

## 附：术语澄清（2026-09-15 修订，不改动上文历史表述）

下述第2、3条描述2026-09-15的先验训练定义，只用于解释历史结果。当前九分支的候选池与监督量以[正式训练协议](../training/EXPERIMENT_PROTOCOL.md)为准；本附录不是当前训练设置。

1. **Score Gap 的定义（重申，防止被误述）**：它比较的是**同一 query 模态下的两个"背景分布"**——
   `D_{同模态背景}`（如 image→image，遍历 j≠i）与 `D_{跨模态背景}`（如 image→text，**排除** matched positive i→i）
   ——之间的 Wasserstein-1 距离；**不是**"positive 分数分布 vs negative 分数分布"。
   姊妹文档 `docs/Project Deconstruction…md` §8 与本文 §3.6 的定义一致。
   （2026-09-15 的一份对外说明文稿曾把它误述为 positive-vs-negative，已作废；以本节与 §3.6 为准。）

2. **`/2`、`/6` 的正式名称是 outer directional normalization**，不是候选池大小。
   作用点是"把 2 个（或 6 个）方向 loss 求和之后再做平均"这一步。
   **候选池规模与此无关**：`standard` 的池是单一目标模态 `N`；所有 `mixed` 分支（fixed/rotating/full）的池**恒为 3N**。

3. **`count-matched` 的准确含义**：匹配的是
   （a）每步**有向 loss 的数量**、（b）每 query 的**正例数**（各 1 个）、（c）**平均协议**（GCL-2 用 `/2`）。
   **不匹配**的是候选数与负例数：`standard` 每 query 有 `N−1` 个负例，`mixed` 有 `3N−2` 个。
   日志字段 `negative_count_per_query_matched` 已于 2026-09-15 更正为 `false`。
