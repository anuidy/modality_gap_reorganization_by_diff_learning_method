# Data and Model Resource Preparation Change Log（数据与模型资源准备改动日志）

## 1. 改动类型

- 数据：下载并验证 COCO 2017 val probe、LCS-558K 标注与固定 10K LCS probe 图像。
- 模型：下载四个候选 pretrained checkpoint 及其必要 tokenizer/backbone 文件。
- 脚本：增加可断点续传的下载脚本与 LCS HTTP range probe-image 下载脚本。
- 配置：增加下载清单和 SHA-256 清单。

## 2. 改动位置

- `data/raw/coco_2017_val/`：COCO 图像、标注及解压内容。
- `data/raw/lcs_558k/`：LCS 标注与 `probe_images/` 下的 10K 图像。
- `data/raw/models/`：CLIP、VISTA、ALBEF、BEiT-3 与 ALBEF BERT backbone。
- `data/splits/lcs_558k_in_domain_probe_v1.json`
- `data/metadata/download_manifest.md`
- `data/metadata/checksums.sha256`
- `scripts/data/download_m0_models_and_probe_metadata.ps1`
- `scripts/data/download_lcs_probe_images.py`

## 3. 改动逻辑

COCO 使用完整 5K val；LCS 用固定 seed `20260825` 选择 10K pair。LCS 官方只提供全量图像 ZIP，因此通过 ZIP central directory 与 HTTP byte range 仅提取被选中的图像条目，并对每个条目进行 CRC 校验。权重与归档文件均记录 SHA-256。

## 4. 改动效果

本机拥有 CLIP M0 与后续三模型 M0 所需的 probe 与权重，不需要下载完整 LCS 训练图像包。COCO 解压后验证有 5,000 张图像；LCS probe 验证有 10,000 张图像且 CRC 匹配官方 ZIP metadata。
