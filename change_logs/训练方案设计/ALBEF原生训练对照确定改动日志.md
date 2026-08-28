# ALBEF原生训练对照确定改动日志

## 1. 改动类型

- 数据：无数据集、manifest 或 probe 改动。
- 模型：正式取消 ALBEF 的人工 Mixed `e_IT` 构造方案，保留其原生 cross-attention/fusion 特性。
- 脚本：无训练代码或 M0 导出脚本改动。
- 配置：新增跨模型实验对照配置，明确两类不同 intervention family。

## 2. 改动位置

- `configs/training/experiment_comparisons.yaml`
- `change_logs/训练方案设计/ALBEF原生训练对照确定改动日志.md`

## 3. 改动逻辑

CLIP、VISTA、BEiT-3 继续比较 `Standard CL vs Count-Matched Mixed CL`，其主要干预变量是 representation relation。ALBEF 不强造与 ITC space 对齐的 `e_IT`，而从同一 M0 checkpoint 分叉为：

- `ITC-only`：`L = L_ITC`
- `Full ALBEF`：`L = L_ITC + L_ITM + L_MLM`

两条 ALBEF 分支训练后都只重新提取 `e_I/e_T` 并计算相同六项指标。Full ALBEF 的干预是启用模型原生 ITM/MLM 与 cross-attention fusion training，不是前三个模型的 Mixed relation intervention。

## 4. 改动效果

ALBEF 不再受人工 pooling/projection/normalization 定义影响，避免人为构造 `e_IT` 引入新的混杂。项目现在有两条明确分开的机制比较：前三模型的 relation intervention，以及 ALBEF 的 native objective/cross-attention intervention。实际训练超参数、数据规模与 seed 仍待后续锁定；本次未启动训练。
