# 改动日志约定

本目录记录项目中的持久改动。先按实际项目阶段建立子目录，再为每一个独立的逻辑改动批次建立一份 Markdown 文档。

当前阶段目录：

- `Project Initialization and Resource Preparation（项目初始化与资源准备）/`：项目骨架、数据/权重准备与初始化 checkpoint 修正。
- `M0 Baseline Establishment（M0基线建立）/`：M0 adapter、embedding extraction、指标与基线结果。
- `Training Plan Design（训练方案设计）/`：训练 intervention、分支定义、控制变量与正式训练前决策。

日志文件以实际改动内容命名，格式为：

`English Change Log（中文改动日志）.md`

每份日志固定说明四项内容：

1. 改动类型：数据、模型、脚本、配置。
2. 改动位置：项目目录位置与文件名。
3. 改动逻辑：为什么这样改、如何与实验设计对应。
4. 改动效果：产出、已验证范围和未包含的事项。

后续进入新的研究阶段时，先创建对应的阶段子目录；不要以数字给阶段或日志文件命名。运行产物、下载权重和第三方源码不进入 Git；日志只记录其路径、来源和验证状态。
