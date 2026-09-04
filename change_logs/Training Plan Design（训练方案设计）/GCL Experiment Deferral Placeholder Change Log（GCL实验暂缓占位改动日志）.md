# GCL Experiment Deferral Placeholder Change Log（GCL实验暂缓占位改动日志）

## 1. 改动类型

- 数据：未定义或生成新的 GCL 数据。
- 模型：未增加模型分支、projection 或训练 intervention。
- 脚本：未增加 GCL 可执行训练代码。
- 配置：新增非执行性说明文件，记录 GCL-like 对照尚未进入正式实验矩阵。

## 2. 改动位置

- `configs/training/GCL_EXPERIMENT_PLACEHOLDER.md`：GCL-like 实验的暂缓状态、未决问题和启用条件。

## 3. 改动逻辑

当前正式实验矩阵继续锁定为八个 run：CLIP、VISTA、BEiT-3 各自进行 Standard 与 Count-Matched Mixed 对照，ALBEF 进行 ITC-only 与 Full ALBEF 对照。占位文件仅保留未来讨论入口，不提供 YAML 配置，也不得被训练入口读取。只有在论文/实现来源、损失定义、公平控制、计算预算、数据构造和评测协议全部经过独立审查后，才能增加正式配置与代码。

## 4. 改动效果

- 明确 GCL placeholder 不是第九个正式 run。
- 防止后续从不完整需求中推断默认超参数或误启动训练。
- 保持当前八组实验协议和训练代码不变。
- 该文件为文档性约束，不产生运行产物，也不需要模型或数据验证。
