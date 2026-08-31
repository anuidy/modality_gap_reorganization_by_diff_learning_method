# Final Experiment Matrix Freeze Change Log（最终实验矩阵冻结改动日志）

## 1. 改动类型

- 数据：无 manifest 或数据内容改动。
- 模型：锁定四模型 M0 checkpoint 与三种 `e_IT` 规则；ALBEF 明确无人工 `e_IT`。
- 脚本：无训练代码执行。
- 配置：扩展正式实验矩阵 YAML，并增加人类可读训练协议。

## 2. 改动位置

- `configs/training/experiment_comparisons.yaml`
- `configs/training/EXPERIMENT_PROTOCOL.md`
- `change_logs/Training Plan Design（训练方案设计）/Final Experiment Matrix Freeze Change Log（最终实验矩阵冻结改动日志）.md`

## 3. 改动逻辑

主要实验分为 relation intervention 与 ALBEF native objective intervention。CLIP/VISTA/BEiT-3 比较 Standard 和 Count-Matched Mixed；ALBEF 比较 ITC-only 和 Full ALBEF。Full GCL 只保留为 CLIP/VISTA reference。模型初始化、branch ID、`e_IT` 定义、count matching 与 interpretation boundary 全部显式记录。

## 4. 改动效果

训练代码后续可直接从配置读取固定实验矩阵，不再需要在模型实现中猜测 branch 含义。ALBEF Full 不会被错误归类为 Mixed CL；VISTA joint-forward 混杂也被写入协议。超参数仍保持配置化并留待服务器 pilot 确定。
