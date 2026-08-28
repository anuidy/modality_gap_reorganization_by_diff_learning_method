# VISTA Stage-1检查点修正改动日志

## 1. 改动类型

- 数据：更新 VISTA checkpoint checksum 与下载 metadata。
- 模型：将错误的 VISTA Stage-2/final checkpoint 替换为 Stage-1 checkpoint。
- 脚本：更新下载脚本中的 VISTA checkpoint 地址。
- 配置：新增 VISTA checkpoint metadata，并指明 Stage-1 pretraining history。

## 2. 改动位置

- 新权重：`data/raw/models/vista/BGE_EVA_Token_S1.pth`
- 删除旧权重：`data/raw/models/vista/Visualized_base_en_v1.5.pth`
- `configs/models/vista/checkpoint.yaml`
- `configs/models/vista/m0.yaml`
- `data/metadata/checksums.sha256`
- `data/metadata/download_manifest.md`
- `scripts/data/download_m0_models_and_probe_metadata.ps1`

## 3. 改动逻辑

研究设计要求 VISTA 从 Stage-1 初始化分叉为 Standard、Count-Matched Mixed 与 Full GCL。Stage-2/final 权重已包含 mixed-modality training，会污染该对照，因此改用 `JUNJIE99/VISTA` 的 `BGE_EVA_Token_S1.pth`，并校验 SHA-256 为 `05a4ce3033d52f96032c2cb79958b361ea9f378aa4dac44069476b433d7281cf`。

## 4. 改动效果

VISTA 初始化已与研究设计一致。新 Stage-1 state_dict 可加载，包含文本 BGE 与 EVA 视觉模块参数；已确认并删除本地旧 Stage-2 文件。此改动不启动任何 VISTA 训练。
