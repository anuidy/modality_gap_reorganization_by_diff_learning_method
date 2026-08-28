# M0 指标汇总

本目录保存四个模型在 COCO 5K 与 LCS 10K probe 上的跨模型汇总结果。

## 文件

- `m0_metrics_long.csv`：长表格式；每行是一个 model-probe-metric-statistic 记录，适合统计软件和后续训练分析读取。
- `m0_metrics_consolidated.json`：包含 8 个 model-probe 的 wide records、248 条 long records，以及原始 metadata/metrics JSON 的完整归档。
- 人工查看工作簿：`outputs/m0_metrics_summary/M0模型指标汇总.xlsx`。

## 权威来源

汇总文件由以下原始结果生成，不修改原始值：

- `outputs/embeddings/m0/<model>/<probe>/*.json`
- `outputs/metrics/m0/<model>/<probe>/*_six_metrics.json`

原始单模型 JSON 仍是权威来源；本目录文件是便于比较和程序读取的派生视图。

## 使用约束

- Raw centroid/covariance 等绝对值受模型维度、尺度和预训练历史影响，不能直接用于判断跨架构优劣。
- 正式机制结论应比较同一模型的 `M0 → Standard` 与 `M0 → Count-Matched Mixed` 变化量。
- M0 geometry 仅保存 pairwise cosine 与 kNN@10 reference；尚不存在 before/after preservation 结果。
