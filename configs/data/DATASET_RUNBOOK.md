# 正式数据集运行手册

## 已冻结的数据角色

| 数据 | 精确数量 | 允许用途 | 禁止用途 |
|---|---:|---|---|
| LCS Train | 540,128 | 参数更新 | Probe、跨分支重采样 |
| LCS Validation | 8,000 | 训练过程监控 | 参数更新、checkpoint selection、八项表示指标 |
| LCS In-domain Probe | 10,000 | M0/训练后八项表示指标 | 训练、early stopping、调参、checkpoint selection |
| COCO 2017 Val Probe | 5,000 | 外部分布八项表示指标 | 训练、early stopping、调参、checkpoint selection |

所有正式训练分支共用同一组 manifest。`Standard`、`Mixed`、`ITC-only`、`Full ALBEF` 只改变模型目标或 representation relation，不改变样本集合。

## 固定拆分

LCS 原始标注共有 558,128 条，源 SHA-256 为：

```text
fe0a999808f30dba5067297a1283d0937a0f519101298ad639d178d8564dfcd6
```

已有的 10K Probe 保持不变：

- seed：`20260825`
- manifest：`data/splits/lcs_558k_in_domain_probe_v1.json`
- SHA-256：`3895e647e218d8565dc51841c5de7cc57079568dcc8b69d33d2e5e08bd4f9aa1`

Validation 从排除 Probe 后的源索引中，以 seed `20260825` 对非 Probe 的 `source_index` 计算版本化 SHA-256 排名并固定取前 8,000 条；Train 是 Validation 与 Probe 的补集。两个生成 manifest 均按原始 `source_index` 排序，训练阶段再由统一训练 seed 产生 epoch 顺序。

版本化身份锁为 `data/splits/lcs_558k_split_lock_v1.json`，SHA-256：

```text
2e4f3cb62be0e2c6737dfdb2b6136c96ade8565b326a65cefa96d942cddf9f1e
```

它锁定了拆分算法、角色、数量和以下生成结果：

```text
Train       540128  e25de59aedd14edb460874c70ed62426d1ffe1c380f46461c5d331b335898190
Validation    8000  9b4727f88c8ec613a18791813401832cd8829b08b3fff0dbf61dbccec52110c2
```

COCO Probe 使用全部 5,000 张 `val2017` 图片，每张图片选择最小 `caption_id`；固定 manifest SHA-256 为：

```text
d8133889432c37f3a436b7dcff769f87cf9b35eb6ea49a6fba00e7c32d129f85
```

## AutoDL 目录

将真实数据放成以下结构。LCS ZIP 自带 `images/` 顶层目录时，应解压到 `data/raw/lcs_558k/`，不要再套一层目录。

```text
data/raw/lcs_558k/
├── blip_laion_cc_sbu_558k.json
└── images/
    ├── 00000/...
    └── ...

data/raw/coco_2017_val/extracted/
├── annotations/captions_val2017.json
└── val2017/*.jpg
```

完整 LCS 图片目录同时服务 Train、Validation 和 LCS Probe，不需要为 Probe 复制 10K 张图片。

## 生成与验证

在项目根目录执行：

```bash
python scripts/data/prepare_formal_datasets.py
```

该命令不会重新抽取已固定 Probe，也不会覆盖不匹配的 manifest。Git只保存无样本内容的协议模板`configs/data/lcs_558k_split_lock_v1.json`。源码新副本先运行`python scripts/data/restore_protocol_metadata.py`恢复运行时锁文件；实际probe清单仍需从服务器资源包取得。该命令依据运行时split lock生成：

```text
data/processed/lcs_558k/manifests/train_v1.jsonl
data/processed/lcs_558k/manifests/validation_v1.jsonl
```

整个`data/`目录不进入Git；本地只保留probe相关资源，Train/Validation和下游评测数据仅保留服务器。`data/processed/`不进入Git；在 AutoDL 上由同一源标注确定性重建，输出 SHA 必须与本手册一致。

只校验清单和拆分：

```bash
python scripts/data/prepare_formal_datasets.py --validate-only
```

正式训练前检查全部 LCS 与 COCO 图片确实存在：

```bash
python scripts/data/prepare_formal_datasets.py --validate-only --verify-images
```

最后一个命令需要扫描 563,128 个图片引用，耗时属于正常现象。

## 训练端保护

`configs/training/train_runs.yaml` 已锁定 Train、Validation、两套 Probe 和 split lock 的路径及 SHA。训练启动前会验证这些身份；文件被修改、不同分支换用其他 Train 清单、或把 Probe 路径换成 Train 路径时都会失败。

训练引擎只从 LCS Train 创建参数更新 DataLoader，因此 Probe 不可能进入梯度更新。固定 Validation 8K 使用独立 DataLoader，按配置中的 validation interval 只读运行，结果写入 `validation_metrics.jsonl`。已完成的三轮计划每20%执行验证、每50%保存完整恢复点，两者互相独立；后续五轮配置尚未实施。验证batch=36，固定丢弃清单末尾8条，实际7992条/222批。所有分支报告公共指标，Mixed/Full 额外报告分支诊断；Validation 始终不用于选择 checkpoint。完整定义见 `configs/training/VALIDATION_PROTOCOL.md`。

## 存储与资源身份

服务器保留全部原始数据与生成清单。本地不保留完整源标注和Train/Validation清单；LCS probe可使用`data/raw/lcs_558k/probe_images/`。COCO保留probe图片与caption标注。原始/最终权重可留本地但不得提交GitHub。协议模板和资源SHA是配置知识，保存在`configs/data/`与`configs/resources/`，不包含真实样本。
