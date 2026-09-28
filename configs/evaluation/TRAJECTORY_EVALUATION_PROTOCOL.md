# Trajectory Embedding 与相邻几何变化评测协议

> 指标的正式定义（八项指标的空间规则、公式与实现对照）见 `configs/evaluation/EIGHT_METRIC_PROTOCOL.md`。

## 输入

每个完成的训练 branch 必须具备：

- `run_manifest.json`，状态为 `complete`；
- 1%、5%、20%、50%、100%的模型轨迹快照；
- `final.json`引用100%模型快照；读取器也兼容先验实验保留的最终完整检查点；
- 同一模型在 COCO 5K 与 LCS 10K 上已经冻结的 M0 raw embedding artifact；
- 固定 upper-triangle pair indices 与 M0 geometry reference。

评测器逐项校验 run、model、branch、optimizer step、M0/config/Probe SHA-256、checkpoint artifact SHA-256 和 state-dict namespace。任一身份不一致时停止，不静默加载或覆盖。

## Raw embedding artifact

每个 `checkpoint × probe` 保存一个 `.npz + .json`：

```text
sample_ids
image_embeddings_raw: [N, D] float32
text_embeddings_raw:  [N, D] float32
```

这里的 embedding 是 projection/head 输出后、L2 normalize 前的主要研究产物。normalized embedding 不落盘，由指标函数运行时派生。图片与文本预处理严格复用相应 M0 adapter 的固定 eval preprocessing，不使用训练 augmentation。

## Point metrics 与 geometry state

每个 trajectory point 分别在 COCO 5K 和 LCS 10K 上计算 centroid gap、covariance gap、effective rank、cross-modal alignment 和 score gap，并保存当前点的固定 pair cosine 与 Neighbor@10 indices。单点 geometry state 本身不解释为 preservation；preservation 只在两个时间点之间定义。

## 当前自动比较

当前只自动生成相邻轨迹链：

```text
M0 → 1% → 5% → 20% → 50% → 100%
```

每条 transition 输出：

- 所有匹配浮点 point metrics 的 `target - source` delta；
- image/text fixed-pair cosine 的 Spearman；
- image/text Neighbor Overlap@10；
- source/target raw embedding artifact SHA-256；
- Probe manifest SHA-256。

Standard/Mixed、ITC-only/Full、跨模型或非相邻时间点的正式比较组不在当前代码中冻结。因为所有 raw embedding 都永久保存，训练完成后可以从 artifact 派生其他比较而无需重新运行模型。

## 执行与状态

评测只在 branch 完成后运行，从磁盘 checkpoint 加载到独立 adapter，不接触训练进程中的 model、optimizer、queue、RNG 或 data-stream state。完整输出后，checkpoint sidecar 的 `evaluation.status` 更新为 `complete`；已有 artifact 只有在 checkpoint/Probe/artifact SHA 全部匹配时才会复用。

运行示例：

```bash
python -u scripts/evaluation/evaluate_trajectory.py \
  --run clip_standard \
  --batch-size 128 \
  --device cuda
```

正式 batch size 由服务器 Pilot 决定，只影响评测吞吐与显存，不改变 artifact 或指标定义。
