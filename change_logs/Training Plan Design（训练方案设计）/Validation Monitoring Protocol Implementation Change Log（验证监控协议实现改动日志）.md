# Validation Monitoring Protocol Implementation Change Log（验证监控协议实现改动日志）

## 改动目的

将固定 LCS Validation 8K 接入正式训练，同时保证 Standard/Mixed 与 ITC-only/Full 之间存在可比较的公共监控指标，并严格隔离 Probe、参数更新和 checkpoint selection。

## 实现内容

- Validation interval 与 `checkpoint_interval` 绑定，最终 checkpoint 保证至少验证一次。
- CLIP、VISTA、BEiT-3 统一报告公共 `I <-> T` loss。
- Count-Matched Mixed 额外报告 `I <-> IT` 与 `T <-> IT` 双向诊断。
- ALBEF 两个分支统一报告只读原生 ITC。
- Full ALBEF 额外报告 ITM、MLM 与完整目标总和。
- ALBEF Validation 不执行 momentum update、queue enqueue/dequeue 或 temperature 原地修改。
- 固定 Validation 数据顺序、augmentation seed、MLM mask 与 hard-negative sampling RNG。
- Validation 完成后恢复 Python、NumPy、Torch CPU/CUDA RNG 与训练模式。
- Validation 使用 `torch.inference_mode()`，不反向传播、不调用 optimizer。
- 独立输出 `validation_metrics.jsonl`，记录 `checkpoint_selection=false`。
- 配置锁定 `validation_sample_count=8000`，并要求 physical micro-batch size 整除 8,000。

## 测试

- 公共 `I <-> T` 在 Standard/Mixed 间一致。
- Mixed 只额外产生两组 relation diagnostics。
- Validation 不改变参数、模型模式或后续训练 RNG。
- ALBEF ITC Validation 不改变 temperature、momentum encoder 或 queue。
- checkpoint interval 与最终 step 均能写入独立 Validation 日志。
- 非整除 Validation 样本数的 micro-batch 配置会被拒绝。

## 尚待服务器验证

- 四个真实官方模型在 A800 BF16 下的 Validation forward。
- Full ALBEF 8K Validation 的显存峰值和运行耗时。
- 正式 micro-batch 选择与 `checkpoint_interval` 的训练成本权衡。
