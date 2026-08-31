# Formal Dataset Preparation Pipeline Change Log（正式数据集准备流程改动日志）

## 目的

将 LCS-558K / LLaVA-Pretrain 与 MS COCO 2017 Validation 处理成正式训练和表示空间 Probe 可以共同复用、可校验且不可在分支间重新采样的数据清单。

## 已完成

- 核验 LCS 原始标注为 558,128 条，ID 与图片路径均无重复。
- 逐条核验既有 LCS 10K Probe 与源标注一致，保留 seed `20260825`，未重新抽样。
- 从非 Probe 样本中以 seed `20260825` 对 source index 做跨平台 SHA-256 排名，固定取 8,000 条 Validation。
- 将其余 540,128 条固定为 Train，并保持源索引顺序写入 manifest。
- 建立 Git 跟踪的 `data/splits/lcs_558k_split_lock_v1.json`，锁定算法、角色、数量与 SHA-256。
- 核验 COCO 5K manifest 覆盖全部 val2017 图片，且每图使用最小 `caption_id`。
- 新增跨平台数据准备入口 `scripts/data/prepare_formal_datasets.py`。
- 新增清单互斥、穷尽、源数据一致性、不可覆盖、COCO caption 选择与 AutoDL 首次生成场景测试。
- 训练启动前增加 Train、Validation、LCS Probe、COCO Probe 和 split lock 身份校验。
- JSONL 训练清单改为逐行解析，降低 540K 清单加载时的峰值内存。

## 固定结果

```text
LCS source       558128  fe0a999808f30dba5067297a1283d0937a0f519101298ad639d178d8564dfcd6
LCS Train        540128  e25de59aedd14edb460874c70ed62426d1ffe1c380f46461c5d331b335898190
LCS Validation     8000  9b4727f88c8ec613a18791813401832cd8829b08b3fff0dbf61dbccec52110c2
LCS Probe         10000  3895e647e218d8565dc51841c5de7cc57079568dcc8b69d33d2e5e08bd4f9aa1
LCS split lock            2e4f3cb62be0e2c6737dfdb2b6136c96ade8565b326a65cefa96d942cddf9f1e
COCO Probe         5000  d8133889432c37f3a436b7dcff769f87cf9b35eb6ea49a6fba00e7c32d129f85
```

## 尚未完成

- 本地没有完整 LCS 558K 图片目录，因此尚未执行全量图片存在性检查。
- Validation 公共指标、分支诊断、checkpoint 周期执行与独立日志已经接入；仍需在 A800 pilot 上验证真实显存和吞吐。
- 数据下载、完整图片解压、真实模型训练及训练后 checkpoint 表示提取将在 AutoDL 完成。
