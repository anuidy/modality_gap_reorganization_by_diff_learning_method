# CLIP M0闭环实现改动日志

## 1. 改动类型

- 数据：生成固定 COCO 5K manifest，保存 CLIP 在 COCO/LCS 上的 M0 artifact、metrics 与 geometry reference。
- 模型：接入官方 OpenAI CLIP ViT-L/14 adapter。
- 脚本：增加 COCO manifest、CLIP M0 embedding 导出与六指标计算脚本。
- 配置：增加 CLIP M0 runtime 记录与 artifact metadata 约定。

## 2. 改动位置

- `src/datasets/probes.py`
- `src/models/base.py`
- `src/models/clip_openai.py`
- `src/embeddings/artifact.py`
- `src/metrics/six_metrics.py`
- `scripts/data/create_coco_probe_manifest.py`
- `scripts/evaluation/extract_clip_m0.py`
- `scripts/evaluation/compute_six_metrics.py`
- `data/splits/coco_2017_val_probe_v1.json`
- `data/metadata/geometry_references/`
- `outputs/embeddings/m0/openai_clip_vit_l14/`
- `outputs/metrics/m0/openai_clip_vit_l14/`

## 3. 改动逻辑

adapter 输出 projection head 后、L2 normalize 前的 raw `Z_I/Z_T`。artifact 保存 sample IDs、raw embeddings 与 JSON metadata；normalized embedding 由 metric 临时派生。指标遵守锁定协议：中心化样本协方差 `ddof=1`、1,000,000 个固定 upper-triangle pair、Neighbor Overlap@10 reference，以及所有 `j != i` 的 score distribution。

## 4. 改动效果

CLIP M0 最小闭环已完整运行。COCO artifact 为 `[5000, 768]`，LCS artifact 为 `[10000, 768]`，均为 float32、无 NaN/Inf、sample ID 顺序匹配 manifest。两套六指标 JSON、固定 pair index、M0 pairwise cosine 与 kNN reference 已保存。Geometry Preservation 的 before/after 比较保留到后续训练 checkpoint。
