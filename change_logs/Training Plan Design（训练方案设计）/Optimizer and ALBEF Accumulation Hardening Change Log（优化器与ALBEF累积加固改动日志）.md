# Optimizer and ALBEF Accumulation Hardening Change Log（优化器与ALBEF累积加固改动日志）

## 1. 改动类型

- 数据：未修改训练数据、Probe manifest 或数据采样协议。
- 模型：未修改四个模型的网络结构或初始化权重；修正 ALBEF 原生 momentum encoder 与 queue 在梯度累积窗口内的状态更新语义。
- 脚本：新增 ALBEF 状态延迟控制器和 AdamW 参数分组工具，并接入训练 backend 与 engine。
- 配置：将 ALBEF queue 整除约束从单个 micro-batch 改为 effective batch；新增 optimizer 参数组 provenance 记录。

## 2. 改动位置

- `src/training/albef_accumulation.py`：ALBEF 梯度累积窗口状态控制器。
- `src/training/optim.py`：AdamW decay/no-decay 参数分组器。
- `src/training/backends.py`：backend 生命周期钩子、ALBEF 状态拦截、模型原生 no-weight-decay 参数声明和 BEiT-3 特殊参数规则。
- `src/training/config.py`：ALBEF effective batch 与 native queue size 的整除校验。
- `src/training/engine.py`：optimizer 参数组接入、ALBEF accumulation window 调度和 provenance 输出。
- `tests/training/test_albef_accumulation_controller.py`：纯控制器行为测试。
- `tests/training/test_albef_accumulation_torch_model.py`：`nn.Module` 方法拦截与恢复测试。
- `tests/training/test_albef_accumulation_wiring.py`：engine/backend 调用顺序测试。
- `tests/training/test_albef_queue_config.py`：effective batch queue 约束测试。
- `tests/training/test_optimizer_groups.py`：参数组完整性、互斥性和分类测试。
- `tests/training/test_optimizer_special_parameters.py`：位置编码与特殊 token 自动 no-decay 测试。
- `tests/training/test_optimizer_weight_decay_effect.py`：真实 AdamW step 的选择性衰减测试。

## 3. 改动逻辑

ALBEF 原生 forward 会调用 momentum update 并把当前 batch 的 image/text features 写入负样本队列。如果直接在 gradient accumulation 的每个 micro-batch 中执行，模型状态会每个 micro-batch 更新一次，而主参数只在 optimizer step 更新一次。现在每个 accumulation window 开始时只执行一次 momentum update；各 micro-batch 的 queue features 被暂存，在 `optimizer.step()` 前合并并统一 enqueue 一次。ITC-only 与 Full ALBEF 两条分支使用相同规则。

AdamW 不再对全部可训练参数统一施加 weight decay。普通权重进入 decay 组；bias、一维参数、归一化参数、位置编码、特殊 token、temperature/logit scale、relative position bias 以及模型原生声明的 no-weight-decay 参数进入 no-decay 组。全局 optimizer weight decay 设为 0，由参数组携带各自数值，避免重复应用。

## 4. 改动效果

- 一个 ALBEF optimizer step 对应一次 momentum update 和一次合并后的 queue update。
- queue size 约束与真实 effective batch 对齐，避免 queue pointer 与梯度累积语义不一致。
- 特殊参数不会被 AdamW 错误衰减，四模型共享一致且可审计的 optimizer 构造路径。
- 运行 metadata 会记录各 optimizer 参数组的 weight decay、tensor 数量和参数量。
- 使用 `D:\conda\envs\modality-gap\python.exe -m unittest discover -s tests/training -p 'test_*.py' -v` 验证，29 项训练测试全部通过。
- 未包含真实四模型 GPU forward/backward、显存压力和 checkpoint resume；这些仍属于 AutoDL pilot 验证范围。
