# M0 Metrics Summary Creation Change Log（M0指标汇总文件建立改动日志）

## 1. 改动类型

- 数据：将 8 个 model-probe 的原始 M0 metrics/metadata JSON 汇总为统一宽表与长表。
- 模型：无模型结构、权重或 adapter 改动。
- 脚本：未重算 embedding 或指标；仅读取已验证结果生成派生汇总。
- 配置：新增汇总文件说明和 provenance/协议记录。

## 2. 改动位置

- 人工查看：`outputs/m0_metrics_summary/M0模型指标汇总.xlsx`
- 机器读取：`analysis/m0/m0_metrics_long.csv`
- 完整归档：`analysis/m0/m0_metrics_consolidated.json`
- 说明：`analysis/m0/README.md`

## 3. 改动逻辑

原始单模型 JSON 保持不变并继续作为权威来源。汇总工作簿包含总览、宽表数据、指标长表、Provenance 和协议五个 sheet；CSV 使用 tidy long-form 结构；JSON 同时保存 wide/long records 与原始 source objects。所有汇总值直接来自 `outputs/metrics/m0/`，不进行重新估计或人工抄写。

## 4. 改动效果

四模型 × 两 probe 的 M0 指标现已具备人工审阅、表格筛选、程序读取和 provenance 审计三种使用方式。工作簿公式检查无错误，8 条宽表记录与 248 条长表记录已和原始 JSON 对齐。未覆盖任何原始 M0 artifact 或 metrics 文件。
