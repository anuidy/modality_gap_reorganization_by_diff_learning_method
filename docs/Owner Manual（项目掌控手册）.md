> 先验阶段知识参考。当前训练参数和执行方式以[主README](../README.md)及正式协议为准。文中的交接/修改日志仅保留服务器；历史假设和建议不代表新正式协议。原始模长不等时，不可套用等模长的平方根恒等式。
>
> 文中先验专用脚本与旧配置路径指归档内原路径，已退出当前执行目录；归档位置及参考方法见[主README](../README.md)。

# 项目掌控手册

> **写给**：项目所有者本人。目标不是"介绍项目"，而是让你**不问助手**就能回答四个问题：
> ① 现在是什么状态？② 哪些东西我能改、改了会怎样？③ 哪些东西动了会出事、谁在拦？④ 结论依赖什么？
>
> **配套文档**（本手册不重复其内容，只做交叉引用）：
> - `docs/Project Deconstruction and Experiment Design（项目解构与实验设计说明）.md` —— 实验设计、四架构差异、公式、全量指标数据
> - `docs/Implementation Fact Verification（训练与评测实现事实核验）.md` —— 10 项实现事实核验 + A–K 冻结表
> - `docs/CLIP Float Upcast Verification Q&A（CLIP训练侧float升精度核验问答）.md` —— 权重精度问答与决议
>
> 更新时间：2026-09-14（含当日新增的 Fixed-6N 一轮）。

---

## 0. 一页速查

**现在的规模**：16 个 run 目录 = **14 条已完成** + **2 条正在跑**（Fixed-6N 的 beit3/vista）+ **1 条待跑**（clip，等前两条跑完）；3 份说明文档；36 份改动日志；136 项单元测试。

**训练跑在哪里**：detached screen 会话 `gcl2_fixed_6n`（排班：round 1 = beit3+vista 并行 → round 2 = clip 单独）。

**三条最该记住的规矩**：

1. `outputs/metrics/m0/` 与 `outputs/embeddings/m0/` 是**不可变锚点**（32 个文件有 SHA 清单），任何写入都是事故。
2. **同一模型的 `-2N` 与 `-6N` run 不可混用**：分母不同，`total_loss` 差 3 倍（日志里 `loss_denominator` 是唯一权威）。
3. **配置文件的措辞可能落后于代码**——判断"实际怎么跑的"一律以该 run 自己的 `run_manifest.json` + `train_metrics.jsonl` 为准。

---

## 1. 三张地图

### 1.1 数据地图：什么在哪、谁写的、能不能删

| 位置 | 内容 | 谁写的 | 能否删 |
|---|---|---|---|
| `outputs/training/<run>/` | 一次训练的全部产物：`run_manifest.json`（配置+状态+控制指纹）、`train.log`、`train_metrics.jsonl`、`validation_metrics.jsonl`、`checkpoint_index.jsonl`、`checkpoints/` | 训练引擎 | **分三类**：`checkpoints/resume/step_*末步*` 与 `checkpoints/trajectory/*` **必须留**；80% 的 resume 已于 09-14 删除；日志/manifest 极小（每条 ~6 MB）建议永留 |
| `outputs/embeddings/{m0,trajectory}/` | 不可变 raw 向量（`sample_ids` + `image_embeddings_raw` + `text_embeddings_raw`，float32） | 评测适配器 | 不能删——这是所有表征分析的原始输入 |
| `outputs/metrics/{m0,trajectory}/` | 点指标 JSON、几何状态 npz、transition JSON、`trajectory_index.json` | 指标层 | 不能删（可重算，但要重跑评测） |
| `outputs/analysis/` | 现算冻结的 M0 八项行、transition 修复日志与备份 | 分析脚本 | 可删（可重算） |
| `outputs/pilot/` | 非正式试跑（含 09-14 的 20 步 smoke 记录，检查点已删） | pilot 脚本 | 可删 |
| `outputs/verification/` | 批处理日志、M0 SHA 前后校验日志、786 MiB 的迁移前备份 tar | 驱动脚本 | 备份 tar 可删（回滚窗口已过） |
| `data/` | 权重 5.8 G + LCS 图像 28 G + COCO 1.8 G + manifest | 数据脚本 | **不能删**（数据身份锁依赖其 SHA） |
| `configs/` | 12 份 YAML（1 份正式矩阵 + 6 份执行配置 + 1 份 locked 设计 + 4 份 pilot/参考） | 见 §2.1 | 执行配置对应已完成的 run，**建议留作复现依据** |
| `change_logs/` `docs/` | 改动记录与说明 | 助手 | 记录类，建议永留 |

### 1.2 权威地图：哪类问题该信哪个文件

**这张表是"掌控"的核心。** 遇到冲突时按它判断谁说了算：

| 你想知道 | **权威来源** | **不要信** |
|---|---|---|
| 某条 run 实际用的超参 / 分支 / mode | `outputs/training/<run>/run_manifest.json` 的 `config` | 配置文件的默认值（可能被覆盖） |
| 某条 run 的 loss 尺度是 2N 还是 6N | 该 run 的 `train_metrics.jsonl` 的 `loss_denominator` 字段 + `total_loss/raw_total_loss` 比值（2N→0.5，6N→0.166667） | `configs/training/experiment_comparisons.yaml`（**已陈旧**，见 §3 表末行） |
| 某条 run 的分支特性（mask/分母/轮换单位） | 该 run 的 `run_manifest.json` 的 `control_guarantees` | — |
| 训练用的起点权重 | `run_manifest.json` 的 `config.checkpoint_sha256` | 记忆或文档 |
| 某个评测点对应哪份权重 | 1/5/20/50% → `checkpoints/trajectory/*_pNNN_model.pt`；100% → `checkpoints/final.json` 指向的 `resume/step_*末步*.pt` | — |
| 八项指标的定义 | `configs/evaluation/EIGHT_METRIC_PROTOCOL.md` + `src/metrics/representation_metrics.py`（代码为准） | — |
| 某个指标的具体数值 | `outputs/metrics/trajectory/<run>/<probe>/points/*_metrics.json` | — |
| M0 是否被改过 | `sha256sum -c data/metadata/m0_sha256.txt`（应 32/32） | — |
| 环境与硬件事实 | 本条：GPU = RTX 4080 SUPER 32760 MiB；python `/root/miniconda3/bin/python`；torch 2.8.0+cu128 | `requirements.txt`（写的是 2.7.1） |

### 1.3 守卫地图：系统会自动拦住什么

| 守卫 | 位置 | 拦住什么 | **不拦什么** |
|---|---|---|---|
| M0 SHA 锚点 | `m0_sha256.txt`（32 文件） | 只在你主动跑校验时生效；驱动脚本会在训练前后自动比对 | 不阻止写入——**它只能事后发现**，所以"绝不写 m0 目录"是纪律而非机制 |
| 授权 run 集合 | `src/training/config.py` `ALLOWED_RUN_SETS` | 配置声明了非授权的 run 集合 | 允许的三个集合是：8 条正式矩阵 / 8+3 条 Full GCL / 3 条 Fixed-6N |
| run 条目字段白名单 | `ALLOWED_RUN_KEYS = {model, branch, output_dir}` | 想用 run 条目偷偷覆盖超参 | `controls:` 段可自由加字段（`mixed_mode` 就走这里） |
| run_id ↔ model/branch 锁定 | `KNOWN_RUNS` 注册表 | run_id 与模型/分支对不上 | — |
| 已完成 run 的防覆盖守卫 | 引擎在 `run_training` 入口检查 `output_dir/run_manifest.json` | 直接对已完成的 run 再训练（报 `FileExistsError`，要 `--resume`） | 新建输出目录不受影响 |
| 不覆盖已有 run | `train.py` / 引擎：`run_manifest.json` 存在即拒绝 | 意外重跑覆盖已完成 run | 换输出目录即可再跑 |
| 数据身份锁 | `src/training/data_control.py` | 换数据/manifest/切分 | — |
| 起点一致性 | 引擎：适配器 M0 SHA 必须等于 run 配置的 checkpoint SHA | 用错起点 | — |
| 指标路径重定向 | `evaluate_trajectory.py` 的 `M0_METRICS_ROOT` / `M0_RECOMPUTED_ROOT` | `--recompute-metrics` 写进 M0 锚点目录（违反即 `RuntimeError`） | — |
| 单元测试 | `tests/`（136 项） | 代码行为回退 | 需要你主动跑 |

---

## 2. 控制面：你能拧的旋钮

### 2.1 配置层（最常用）

一份执行配置分三段，改动的影响面完全不同：

```yaml
controls:        # 全局共享：seed / precision / data / optimizer / scheduler / budget / checkpointing
                 #   + 可选 mixed_mode（fixed | rotating）
models:          # 每个模型的权重路径、SHA、batch、augmentation、options（分辨率/文本长度/温度等）
runs:            # 只能写三个键：model / branch / output_dir
```

**关键规则**：`runs:` 段只能声明 model/branch/output_dir（有守卫）；超参一律在 `controls:` 或 `models:` 里。所以**同一份配置里所有 run 共享同一套超参**——这正是"控制变量"的实现方式。

现有配置一览（按 `status` 字段区分用途）：

| 配置 | status | 用途 |
|---|---|---|
| `train_runs.yaml` | `frozen_formal_matrix_as_executed` | **正式矩阵唯一权威声明**：17 条 run，超参填实（2026-09-15 冻结） |
| `<model>_standard_execute.yaml` ×4 | `*_execute_not_formal_matrix` | 自包含执行配置（各只填自己模型的块） |
| `full_gcl_execute.yaml` | 同上 | 8+3 条（Full GCL 参考组）；`controls`/`models` 与正式矩阵逐字段一致 |
| `gcl2_fixed_6n_execute.yaml` | 同上 | **本次新增**：3 条 Fixed-6N，`controls.mixed_mode: fixed` |
| `gcl2_fixed_6n_smoke.yaml` | `*_smoke_not_formal` | 20 步 smoke（输出到 `outputs/pilot/`） |
| `experiment_comparisons.yaml` | **`locked`** | 实验设计的声明文档（**不参与运行**，且有两行陈旧，见 §3） |
| `pilot_update_runs.yaml` / `preflight_smoke_runs.yaml` | `*_not_formal` | 历史 pilot |

### 2.2 代码里的"物理常量"（改了后果最重的一层）

| 常量 / 机制 | 位置 | 现值 | 改了会怎样 |
|---|---|---|---|
| `FIXED_GCL2_LOSS_DENOMINATOR` | `src/objectives/contrastive.py` | **6.0** | 改 Fixed 的 loss 尺度 → 新 run 与既有 Fixed run 不可直接比 |
| `ROTATING_GCL2_LOSS_DENOMINATOR` | 同上 | **2.0** | 同上（Rotating） |
| `FULL_GCL_LOSS_DENOMINATOR` | 同上 | 6.0 | Full GCL 的尺度 |
| `MIXED_POOL_ORDER` | 同上 | `("I","T","IT")` | 改顺序会改变池槽位索引与日志语义 → 新旧 run 不可比 |
| `FULL_GCL_DIRECTIONS` / `DIRECTION_ORDER` | 同上 | 六方向（顺序冻结） | 日志 `loss_direction_1..6` 按它索引；改了会让新旧日志字段错位 |
| `COUNT_MATCHED_RELATION_CYCLE` | 同上 | `I↔T → I↔IT → T↔IT` | 改轮换顺序 |
| `denominator` 参数 | `mixed_pool_bidirectional_contrastive` / `count_matched_mixed_objective` | **必需参数、无默认值** | 设计如此：漏传直接报错，不会静默用错尺度 |
| `gradient_clip_norm` | 执行配置 `controls.optimizer` | 1.0 | 裁剪是否生效直接改变优化轨迹（GCL 分支实测 100% 触发） |
| 温度 / logit scale | `src/training/backends.py` | CLIP/BEiT-3 `exp(logit_scale)`（≤100）；VISTA `1/temperature = 50`；ALBEF `temp` clamp [0.001, 0.5] | 训练语义 |
| seed 派生三处 | `src/training/engine.py` | aug `seed+1_000_003·epoch+batch`；forward `seed+2_000_003+step·accum+micro`；validation `seed+5_000_003+batch` | 这是"控制变量"的技术基础，改了会让分支间不再可比 |
| M0 路径常量 | `scripts/evaluation/evaluate_trajectory.py` | `M0_METRICS_ROOT` / `M0_RECOMPUTED_ROOT` | 防止重算写进锚点 |

### 2.3 脚本层

| 类别 | 入口 | 说明 |
|---|---|---|
| 训练入口 | `scripts/training/train.py` | `--run`（必须是注册表里的 run_id）、`--config`、`--validate-only`、`--resume`、`--mixed-mode`、`--output-dir` |
| 训练驱动 | `run_clip_standard_detached.sh` / `run_albef_concurrent_detached.sh` / `run_gcl2_paired_detached.sh` / `run_eval_then_full_gcl_detached.sh` / **`run_gcl2_fixed_6n_detached.sh`** | 都是"预检 + `screen -dmS`";`--drive` 是内层体 |
| 评测 | `scripts/evaluation/evaluate_trajectory.py` | `--run --batch-size --device`；`--save-probe-embeddings`（可选点云，从未开启） |
| 分析 | `scripts/analysis/within_model_delta.py` | 由 M0 基线与各 run 点指标现算同模型 Δ 表（纯 CPU，不 import torch） |
| 校验 | `scripts/evaluation/validate_m0_outputs.py` / `smoke_eight_metrics.py` | M0 产物校验 / 八项指标 smoke |

---

## 3. 不变量与"动了会出事"

| 不变量 | 内容 | 破了会怎样 | 现状 |
|---|---|---|---|
| **M0 锚点** | `outputs/{metrics,embeddings}/m0/` 32 文件逐字节不变 | 起点不可信 → 所有 Δ 结论作废；只能从备份回滚 | ✅ 32/32 |
| **run 身份唯一** | 每条 run 一个独立 `run_id` + 独立输出目录 | 两条 run 的产物互相覆盖（2026-09-11 曾发生，已修） | ✅ Fixed 与 Rotating 现已分开 |
| **产物 ↔ 配置可追溯** | `run_manifest.json` 记录 config SHA、checkpoint SHA、控制指纹 | 无法证明某条 run 是怎么跑的 | ✅ 每条 run 都有 |
| **数据身份锁** | 训练 manifest / probe / split lock 的 SHA 在训练前校验 | 换数据而不自知 | ✅ 每个 run 记录 |
| **实验设计文档 vs 实现** | 应当一致 | 方法描述会写错 | ⚠️ **`experiment_comparisons.yaml`(locked) 有两行陈旧**：`candidate_pool: current_relation_target_modality_only` 与 `same_semantic_instance_excluded_from_negatives: true`，而实现是**全池 3N**且**不排除**同实例其他模态条目（只 mask query 自身槽）。该文件不被任何代码消费，但被文档反复引用为设计来源 → **引用前先看 §1.2 权威地图** |
| **日志审计字段可信** | 审计字段应当是测量值 | 会被误引用 | ⚠️ 曾被 `same_instance_false_negatives`（硬编码 0）破坏；该字段已于 2026-09-15 **整体移除**，同实例负例的事实改由单元测试断言，不再逐 step 记录 |

---

## 4. 自助核验命令（全部实测可用，可直接粘贴）

在 `/root/autodl-tmp/modality-gap-reorganization` 下运行。

**① 全部 run 一览（名字 / 状态 / 分支 / mode / 步数）**
```bash
python3 - <<'PY'
import json,os
for d in sorted(os.listdir("outputs/training")):
    p=f"outputs/training/{d}/run_manifest.json"
    if not os.path.isfile(p): continue
    m=json.load(open(p)); c=m["config"]
    print(f"{d:<24}{m['status']:<10}{c['branch']:<21}{c.get('mixed_mode') or '-':<10}{m.get('completed_steps')}")
PY
```

**② M0 锚点是否被动过（期望 32）**
```bash
sha256sum -c data/metadata/m0_sha256.txt | grep -c ": OK"
```

**③ 训练是否还活在 screen 里（进程 SID 应等于 screen 的 SID）**
```bash
screen -ls | grep -E "gcl2_fixed_6n|Detached"; ps -eo pid,sid,cmd | grep "[t]rain.py --run"
```

**④ 某条 run 的进度与损失尺度（分母 6.0=Fixed-6N，2.0=2N）**
```bash
tail -1 outputs/training/vista_gcl2_fixed_6n/train_metrics.jsonl | python3 -c "
import sys,json; d=json.loads(sys.stdin.read())
print(d['completed_steps'], d['loss_denominator'], d['gcl_mode'], d['total_loss'], d['total_loss']/d['raw_total_loss'])"
```

**⑤ 八项指标是否齐全（对某条 run 的 p100 点）**
```bash
python3 -c "
import json; d=json.load(open('outputs/metrics/trajectory/clip_standard/coco_2017_val_5k/points/trajectory_clip_standard_p100_coco_2017_val_5k_metrics.json'))
keys=['centroid_gap_raw','covariance_gap_raw','effective_rank_image_raw','cross_modal_alignment','intra_geometry_image','score_gap_mean','norm_imbalance','anisotropy_image']
print({k:(k in d) for k in keys})"
```

**⑥ 磁盘 / GPU**
```bash
df -h /root/autodl-tmp | tail -1; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```

**⑦ 改动代码后是否回退（136 项）**
```bash
OMP_NUM_THREADS=8 /root/miniconda3/bin/python -m unittest discover -s tests -t .
```

**⑧ 训练前干跑校验（不加载模型）**
```bash
/root/miniconda3/bin/python scripts/training/train.py --run vista_gcl2_fixed_6n --config configs/training/gcl2_fixed_6n_execute.yaml --validate-only
```

---

## 5. 决定台账

### 5.1 已决议（不要重复讨论）

| 决定 | 日期 | 内容 |
|---|---|---|
| M0 基线统一为单一八项文档 | **09-15** | ~~09-13 的 Option A（八项行另存 `outputs/analysis/`）~~ 已被结构化重构取代：M0 指标只有一份，位于 `outputs/metrics/m0/<model>/<probe>/m0_*_metrics.json`，由 `data/metadata/m0_sha256.txt` 锚定（32 行）。**M0 基线是冻结数据，不重算**（重算不可逐位复现，实测差 ~1e-15） |
| CLIP 评测侧 fp16 保持现状 | 09-14 | 不改代码、不重导；该精度状态是**已知条件**，不得当作缺陷反复重提。若日后必须改，**必须整批重导 CLIP 全部产物** |
| Fixed 恢复 6N 并重训 | 09-14 | 三条 `*_gcl2_fixed_6n`（进行中），与既有 2N 的 Fixed 并存作对照；run 结构分离 |
| 删除 14 个 80% resume 检查点 | 09-14 | 释放 43.88 GiB；末步与 trajectory 全部保留 |

### 5.2 待你决定

**已于 2026-09-14 决定并落实的三项**（保留在此以便追溯）：

| 事项 | 决定 | 落实位置 |
|---|---|---|
| 解构文档的 VISTA 温度表述 | **回改**：在 §2.5 / §4.2 / §4.3 明确区分"评测适配器的惰性 1.0"与"训练 backend 的 0.02（scale 50）" | `docs/Project Deconstruction…md`（三处） |
| `experiment_comparisons.yaml` 两行陈旧 | **改成与实现一致**：`candidate_pool: full_I_T_IT_pool_3N`、`masking_rule: query_slot_only`、`masked_candidates_per_query: 1`、`same_semantic_instance_excluded_from_negatives: false`，并留修订注记 | 该 yaml（YAML 解析已验证） |
| `same_instance_false_negatives` 硬编码 | 09-14 **先只改文档**（讲清设定再给结论）；09-15 **进一步整体删除该字段**，并以行为断言固定"同实例条目是活跃负例" | `TRAINING_RUNBOOK.md`、解构文档 §11.2、`tests/objectives/test_count_matched_mixed.py` |

**仍待你决定**：

| 事项 | 现状 | 需要你定的 |
|---|---|---|
| 同模型 Δ 分析 | 数据已齐（14 条 run 的八项指标 100% 覆盖，见 §6 下方），尚未产出分析 | 是否启动；呈现口径（端点差 / 轨迹形状 / 联合模式） |
| Fixed-6N 的评测 | 三条 run 仍在训练 | 训练完成后是否立即跑轨迹评测（约 40 分钟 GPU） |
| 正式 8-run 矩阵 | `train_runs.yaml` 超参仍 `null` | 何时冻结 |
| A–G preflight 报告 | 从未出具 | 是否仍需要 |

### 5.3 已授权未完成

- **Fixed-6N 三条训练**（进行中）→ 完成后需跑轨迹评测（约 40 分钟 GPU），然后才可做 2N vs 6N 对照。

---

## 6. 结论依赖什么（分析层的控制点）

以后无论谁来解释这批实验，以下六条决定了结论的强度：

1. **同模型 Δ 是唯一可比读数**：跨架构绝对指标不可比（raw 量级差 10 倍以上、维度/预训练史都不同）。
2. **raw 为主、L2 为辅**；`norm_imbalance` 只在 raw 上有意义。
3. **VISTA 的两个已知特殊性**：① `e_IT` 走原生联合编码器（多一次前向 → 与算力混淆）；② 训练时 dropout(p=0.1) 开启（CLIP/BEiT-3 无）→ 它的每步表征是随机样本。
4. **Fixed 的 2N vs 6N 是一对真实对照**（本轮结束后具备），但它只差一个常数尺度，而在 gradient clipping 全程生效的情况下该差异很可能被约掉 —— **这是推理，需用这两条对照实测验证**。
5. **单 seed**：全部 run 用 seed 42，跨 seed 方差未知 → 小幅变化不可解释。
6. **CLIP 的精度已定案**：评测侧 fp16 / 训练侧 fp32，相对差 3.4e-3，比效应量小 2–3 个数量级 → 不影响主结论。

**另有一张"已核实限制"清单**（8 项，含 `code_commit` 不代表源码、2N/6N 被裁剪抵消、假负例字段恒为 0、硬件混杂等）：见
`docs/Project Deconstruction and Experiment Design（项目解构与实验设计说明）.md` §11.2。**任何对外报告都必须带上那张表。**

### 6.1 测量覆盖现状（"都测完了吗"的答案）

**已测完（17 条 run × 2 个 probe）**：每条 run 的 5 个 checkpoint 点（1%/5%/20%/50%/100%）各有完整的点指标 JSON、几何状态与 5 份 transition，**全部含八项指标**（核验命令见 §4 第 ⑤ 条）；第 6 个时间点 **M0** 不在各 run 目录里，它来自冻结的 M0 基线 `outputs/metrics/m0/<model>/<probe>/m0_*_metrics.json`（8 份 = 4 模型 × 2 probe，每份含 25 个扁平指标键）。

**分支数并不都是 4**：clip / beit3 / vista 各有 4 个分支（Standard、GCL2-Fixed、GCL2-Rotating、Full GCL）；**ALBEF 是另一族，只有 2 条**（`albef_itc_only`、`albef_full`），不存在 4 分支结构。

**唯一缺口**：Fixed-6N 三条（训练中）——跑完后这三个模型各自会多出**第 5 个分支**，需要再跑一次轨迹评测（约 40 分钟 GPU）才能补上该列；在此之前，任何"分支 × 时间点"的表里都应把 Fixed-6N 标为待补。

---

## 7. 你要改东西时的标准流程

```
1. 先判断改动属于哪一层：
   数据(data/) / 配置(configs/) / 代码(src,scripts) / 产物(outputs/)
   产物层再看：是不是锚点（m0）？是不是已完成 run 的产物？

2. 配置改动 → 先跑 --validate-only（不加载模型）
3. 代码改动 → 先跑单测（136 项）
4. 要跑训练 → 先 20 步 smoke（配 *_smoke.yaml，输出到 outputs/pilot/），
   看到 loss_denominator / gcl_mode / relation 都对，再上长跑
5. 要写驱动脚本 → 进程等待逻辑必须先做隔离测试（成功/失败/缺失变量三条路径）
6. 任何持久改动 → 写一份 change_logs/ 记录 + 更新 HANDOFF_CURRENT.md
7. 训练/评测结束后 → 立刻核验 M0 SHA 与产物完整性
```

**为什么第 5 条是硬要求**：2026-09-11 与 09-14 各有一次驱动脚本因 bash 写法（`local -n` 的 nameref 泄漏、`local var=X pid="${!var}"` 的求值顺序）而让整个 campaign 陪葬——其中一次把正在跑的训练一起杀掉。这类 bug 只有在隔离测试里才会暴露。
