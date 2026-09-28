# 当前实验状态 / Current Experiment Status

更新：2026-09-28。此文件区分已实现的历史协议和下一轮决策。历史配置及结果身份保留，不重命名已有结果。

## 已完成 / Completed

- CLIP：seed=42七个一轮分支及对应A/B1/Local评测；六分支汇总包是其子集，结果不进入Git。
- BEiT-3：seed=42六分支三轮，45,009步；410项A/B1/Local评测完成。Standard前两轮采用两轮余弦，第三轮改用三轮余弦，仅供参考；其他五个分支从M0使用完整三轮余弦。
- VISTA：先验和20%诊断完成，新完整多轮矩阵尚未执行。
- 当前训练和评测于2026-09-24结束，暂停到新环境和协议准备完成。
- A0–A6、B1和十二项Local已有实测。B3/Global代码、C1/C2及C3数据入口已有；全量B3/Global实测和C3方法/干预尚未完成。

## 下一轮方向 / Next campaign

- CLIP、BEiT-3、VISTA各九分支，FN-on重新纳入；每分支seeds=[42,43,44]，包括VISTA Fixed-2M，合计81项。
- 计划每项五轮：100%=15,003步，500%=75,015步；从M0采用完整五轮余弦。当前三轮结果不能改名冒充五轮从头训练结果。
- 公共控制沿用batch36、accumulation1、BF16、全参数、AdamW、初始LR1e-5/最低1e-6、warmup300、WD0.05、betas=(0.9,0.999)、epsilon1e-8、梯度裁剪关闭；验证batch36，实际7992条。
- 两张4090各跑独立任务，无DDP；500GB是工作盘，需要评测产物和Global资源的外部存储规划。
- 按seed依次完成。拟每20%轨迹评测、每50%完整恢复，滚动保留最近两份；未完成规定评测的轨迹不能删除。所需A/B和后续分析材料完整保存并校验后，可以周转该seed权重。
- C需要跨seed结果，不要求永久保留所有权重；需权重的中间点干预必须在清理前确定。
- GitHub只存代码、配置和说明，不包含任何模型、数据或实验结果。

## 尚未就绪 / Not ready to execute

- 五轮YAML、81项矩阵和新滚动归档队列尚未实施。
- 当前 `run_independent.py --matrix` 仍是79项历史矩阵（VISTA Fixed-2M只有首个seed），不能用于81项计划。
- `beit3_three_epoch.yaml`及专用协调器是已完成实验的实现；该协调器选择当时的六个分支。
- Global/B3是否覆盖全部25点未决定；只完成A/B1/Local不能标记全部ABC完成。
- Global全量吞吐、解压占用、双卡硬件实测、外部存储和安全清理流程还需核对。
- 新机器准备完成不等于自动开训。

English: Training is paused after the completed CLIP/BEiT-3 campaigns. The next proposal restores all nine branches and all three seeds (81 runs) for five epochs. Five-epoch configs, the new matrix and rolling archival/retention are not implemented. The old matrix still has 79 jobs. B3/Global coverage, C3 interventions and external storage remain unresolved. Preserve complete analysis inputs before retiring weights; no datasets, weights or results belong in GitHub.
