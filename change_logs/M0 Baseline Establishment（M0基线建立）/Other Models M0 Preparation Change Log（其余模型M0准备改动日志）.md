# Other Models M0 Preparation Change Log（其余模型M0准备改动日志）

## 1. 改动类型

- 数据：补齐 ALBEF 所需本地 `bert-base-uncased` backbone，并更新 checksum/下载清单。
- 模型：实现 VISTA、ALBEF、BEiT-3 的 M0 adapter 与统一 adapter factory。
- 脚本：增加通用 M0 导出入口与四模型 smoke test 入口。
- 配置：增加三模型 M0 config 与运行环境配置。

## 2. 改动位置

- `src/models/vista.py`
- `src/models/albef.py`
- `src/models/beit3.py`
- `src/models/factory.py`
- `scripts/evaluation/extract_m0.py`
- `scripts/evaluation/smoke_m0_adapter.py`
- `configs/models/vista/m0.yaml`
- `configs/models/albef/m0.yaml`
- `configs/models/beit3/m0.yaml`
- `configs/evaluation/m0_runtime.yaml`
- `third_party/flag_embedding/`、`third_party/albef/`、`third_party/unilm/`
- `data/raw/models/albef/bert-base-uncased/`

## 3. 改动逻辑

三模型均复用 CLIP 的 probe、artifact 与指标接口，只替换模型适配层。VISTA 以 `normlized=False` 导出 Stage-1 raw embedding；ALBEF 直接取 ITC `vision_proj`/`text_proj` 的 pre-L2 输出；BEiT-3 直接取 `vision_head`/`language_head` 的 pre-L2 输出。对 ALBEF/BEiT-3 官方旧源码，仅增加与当前 Transformers/PyTorch 的导入兼容 shim，不修改网络或权重。

## 4. 改动效果

VISTA、ALBEF、BEiT-3 已分别通过真实 checkpoint 的单图/单文本 smoke test：VISTA 输出 768 维，ALBEF 输出 256 维，BEiT-3 输出 768 维；三个 checkpoint 均无 missing keys。三者尚未执行完整 COCO/LCS M0 artifact 与指标计算。
