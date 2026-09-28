> 先验阶段知识参考。当前训练参数和执行方式以[主README](../README.md)及正式协议为准。文中的交接/修改日志仅保留服务器；历史假设和建议不代表新正式协议。原始模长不等时，不可套用等模长的平方根恒等式。
>
> 文中先验专用脚本与旧配置路径指归档内原路径，已退出当前执行目录；归档位置及参考方法见[主README](../README.md)。

# 重训手册与实验复盘（2026-09-17）

用途：在重新训练之前，用一份文档讲清楚 ① 我们的实验流程是什么 ② 哪里出了问题 ③ 重训要改什么 ④ 哪些现成资产可以直接复用。

---

## 一、实验流程（现状）

```
LCS-558K（540,128 训练对 / 8,000 验证 / 10,000 域内探针）
        │  + COCO 5K 外部探针（仅评测）
        ▼
训练（17 条 run：clip / vista / beit3 × {Standard, Fixed-2N, Fixed-6N, Rotating, Full GCL} + albef × 2）
        │  seed 42；1 epoch = 16,879 步；AdamW lr 1e-5 / WD 0.05 / bf16 / batch 32 / clip 1.0
        │  轨迹点：p001(169) / p005(844) / p020(3376) / p050(8440) / p100(16879) + 末步 resume
        ▼
几何评测（每 run × 2 探针 × 5 点）
        │  八项指标：centroid gap / covariance gap / effective rank / cross-modal alignment /
        │            intra-modal geometry / score gap / norm imbalance / anisotropy(+gap)
        │  产出：170 份点指标 + 170 份相邻差分（schema 94 键）
        ▼
下游评测（MMEB 12 个检索任务，R@1/R@5；M0 + p100）
        │  分组口径：cross-modal(6) / mixed-composite(5) / NIGHTS(1)
        ▼
分组汇总（scripts/analysis/mmeb_group_summary.py）
```

关键产物位置：

| 内容 | 路径 |
|---|---|
| 训练 run | `outputs/training/<run>/`（含 `run_manifest.json`、`train_metrics.jsonl`、`checkpoints/`） |
| 几何指标 | `outputs/metrics/trajectory/<run>/<probe>/points/` |
| M0 基线（32 行锚点） | `outputs/metrics/m0/` + `data/metadata/m0_sha256.txt` |
| MMEB 结果 | `outputs/analysis/mmeb_official_v2/`（21 个 JSON + CSV + 分组汇总） |
| MMEB 数据 | `data/raw/mmeb_eval/`（8.2 GB，已解压） |
| 评测脚本 | `scripts/evaluation/mmeb_retrieval.py`、`scripts/analysis/mmeb_group_summary.py` |

---

## 二、问题出在哪（三层）

### 1) 机制层：这个目标函数本身有结构性缺陷

- **加法捷径**：`e_IT = L2(e_I + e_T)` 使 `cos(I, IT) = sqrt((1+cos(I,T))/2)` 恒成立 → M0 时 I→IT 方向就饱和（loss≈0），IT 方向没有梯度
- **同实例假负例**：3N 池只 mask query 自身槽 → 每个 query 恰好 1 个同实例非目标负例；对 I↔T 而言，最难负例恒为同实例 IT 槽位（因 `sqrt((1+c)/2) > c` 恒成立）→ **损失存在不可消除的下限**，且梯度的一部分在把 I 和 T 推离
- **表示坍缩**：训练后各向异性从 0.19→0.90、matched-pair 余弦→0.99、有效秩 285→33、centroid gap→0.006 → 同模态/跨模态分数不可分（score_gap→0.0001）
- **方向性后果**：文本→图像（t2i）崩、图像→文本（i2t）反升（clip：0.018→0.33）——几何指标与 t2i 秩相关 ±0.77~1.00，与 i2t 反号

### 2) 配方层：与官方不可比

| | 官方 GCL | 我们 |
|---|---|---|
| epochs / lr | 5 / 5e-6 | 1 / 1e-5 |
| β₂ | 0.95 | 0.999 |
| CLIP 训练参数 | 只训视觉塔 | 全参数 |
| batch（VISTA） | 128 | 32 |

副作用：官方 MMEB 的 CLIP 基线（CL+Pairwise 29.9）比其 pretrained（39.3）还低 9.4 点，属被配方压低的对照；我们的 Standard 在同一评测上是 39.97。

另：`/2` vs `/6` 的外层分母在本配置下**不是有效变量**（4 臂实验：AdamW 吸收 3 倍尺度差，相对参数差 2.5e-4；而裁剪开关本身带来 9.6e-4）。

### 3) 评测层：已排除

M0 基线与论文逐任务吻合（clip 0.4067 vs 0.393；vista 0.3998 vs 0.403；VISTA 的 12 个任务全部在 ±0.032 内）→ 数据、融合、模板、打分链路正确。下游崩塌不是评测造成的。

---

## 三、重训要改什么（按优先级）

### P0：修目标函数（决定论文能不能立住）

1. **排除同实例非目标槽位**（方案 A）：mask 掉同一 semantic instance 的其它表示，只保留 query 自身槽以外的跨实例负例 → 直接消除结构性下限
2. **分级监督**：把 I↔T（已被验证是"好方向"）与 IT 相关方向分开加权，不要让 I→IT 的免费捷径主导梯度
3. **IT 表示的选择**：优先用原生联合编码器；若必须用加法，至少改为"先归一化再平均"或加入与 I/T 的耦合约束，避免 `sqrt` 恒等式带来的饱和
4. **加一个短程 pilot 判据**：2k 步后检查 ① I→IT 的 loss 是否仍为 0 ② 各向异性是否开始上升 ③ t2i 是否掉——三项都健康才开全量

### P1：对齐官方配方（用于公平对比）

- 复刻一组"官方配方"run：CLIP 只训视觉塔、5 epoch、lr 5e-6、β₂0.95、bs32
- 若条件允许，再做因子分解：`{全参数, 仅视觉塔} × {1 epoch, 5 epoch}` 四格，定位到底是"训什么"还是"训多久"在起作用

### P2：重训时的统一纪律（每条 run 必做）

- **同配方唯一变量**：同一模型族内，除监督结构外所有超参逐字段一致（配置里显式写死，禁止逐 run 覆盖）
- **seed ≥3**：任何"分支 A 优于分支 B"的结论必须跨 seed 稳定
- **训练中几何监控**：每 2k 步在 10K 域内探针上算 `anisotropy / matched-pair cosine / effective rank / score_gap`（纯 CPU 几分钟），作为早停与诊断信号
- **评测分组**：只报 12 任务总平均会掩盖方向性差异，必须给 cross-modal / mixed / NIGHTS 三组

### P3：训练环境（待定）

重训将在**最终选定的训练机器**上进行；本文件不记录任何针对特定过渡实例的适配参数。唯一与硬件无关、可提前确定的工程结论：

- 评测是 **CPU-bound**（实测 GPU 利用率仅 0–10%），选机器时 CPU 核心数与磁盘 IO 比 GPU 显存更关键
- 若评测量级上升到百万候选（如 M-BEIR global），提速手段按性价比排序：**预缩放图片缓存 > GPU 解码（DALI/nvJPEG）> 多进程 dataloader**

---

## 四、可直接复用的资产（不需要重建）

- MMEB 评测脚本与数据（`mmeb_retrieval.py` + `data/raw/mmeb_eval/`）——新 run 只需换 checkpoint 路径
- 几何评测管线与 170 份历史产物（作为对照基线）
- 分组汇总工具（`mmeb_group_summary.py`）
- M0 锚点（32 行 SHA256）与 split lock（数据切分不可变）
- LCS-558K 训练/验证/探针切分与图像
- 训练引擎（含轨迹点、resume、审计日志）——只需改目标函数与配置

---

## 五、当前结论（一句话）

在**同配方**下，GCL 家族的三种改造（Fixed 只训 I↔T / Rotating / 六方向 Full GCL）对下游检索**普遍有害**，且伤害与融合方式强相关：加法 IT 的 clip（−0.242）与 beit3（−0.167）最重，原生 IT 的 vista（−0.081）最轻；唯一三个骨干上一致的结论是"只监督 I↔T 会崩"。

**下一步不是继续堆 run，而是先把"排除同实例假负例"的修复版做出来并 pilot 验证。**
