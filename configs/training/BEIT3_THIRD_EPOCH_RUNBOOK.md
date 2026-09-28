# BEiT-3三轮余弦调度：正式分支与Standard参考续训

**状态更新（2026-09-28）：训练已暂停，已完成实验和下一轮决策以[当前实验状态](CURRENT_EXPERIMENT_STATUS.md)为准。下一轮恢复九分支并计划五轮，但新配置、81项矩阵及滚动保存尚未实施。下文中的旧轮数、FN取消、79项矩阵和“当前运行”均属于当时协议记录，不是新机器的自动执行计划。**

最新确认：所有后续正式分支从M0开始使用完整三轮余弦调度。已完成两轮的Standard从200%完整状态继续，但改变了余弦时间跨度，只作为参考，用户后续另行从M0重训正式Standard。

## 执行顺序

1. Standard参考续训：恢复30006步模型、AdamW状态、随机状态和数据位置，切到三轮余弦曲线，续至45009步。
2. Fixed-2M：原两轮调度的运行已停止在7502步完整检查点，保留作参考；正式三轮任务从M0重新开始。
3. Fixed-3M FN-off、Mixed-2M、Mixed-3M FN-off、Full GCL FN-off：依次从M0各训练完整三轮。

三个FN-on分支仍取消。每个分支完成三轮才轮到下一个；不再采用“先全部两轮，再全部补一轮”。旧固定1e-6的追加方案已取消，后台等待程序不再启用。

## 学习率与进度

- 100%=1epoch=15003步；最终300%=3epoch=45009步。
- AdamW、初始LR=1e-5、warmup=300、最低LR=1e-6；余弦衰减跨度45009步。
- Standard参考恢复时约3.28e-6，之后继续下降到1e-6。这与其前两轮末端1e-6不连续，不能视为从头三轮的正式对照。
- 其余五项从第0步开始使用同一三轮曲线，不改batch36、BF16、权重衰减、随机种子42、数据划分或目标函数。
- `--retime-cosine-for-extension`必须显式启用，并强制标记`analysis_role=reference_only`；模型、优化器和数据控制仍逐字段校验。

## 保存与评测

每20%保存轨迹并验证，每50%保留完整恢复状态。全程轨迹20/40/…/300%，完整恢复50/100/150/200/250/300%。300%完整状态兼作最终评测快照。纯轨迹仅在A/B1/Local全部完成并回读SHA和来源身份后删除，完整恢复文件长期保留。

Standard参考任务只产生200/220/240/260/280/300%的本阶段结果，早期记录保留原位置。其余五项各有十五个轨迹点，另共享M0，合计410个评测子任务。训练一进程、评测最多一进程；C/Global/B3仍暂缓。

运行清单、训练日志和评测checkpoint身份均标记Standard为reference_only；正式C类收集器不接受这些结果填充正式Standard对照，必须等待正式重训。

## 文件与状态

- 正式三轮配置：`configs/training/beit3_three_epoch.yaml`，其中Standard默认输出保留给未来M0正式重训。
- 当前混合队列冻结配置仅把Standard输出改到`outputs/training/beit3_three_epoch_v1/beit3_standard_reference/seed_42/`。
- 其他正式输出：`outputs/training/beit3_three_epoch_v1/<run>/seed_42/`。
- 评测根：`outputs/evaluation/beit3_three_epoch_v1/`；参考属性进入每份结果身份。
- 活动队列：`outputs/training_queues/beit3_three_epoch_cosine_seed42_20260923/`。
- 入口：`scripts/training/run_beit3_campaign.py`。
- 原两轮目录、Fixed-2M 50%完整点和Standard 200%来源完整点均保留，不覆盖。

## English

Use a three-epoch cosine schedule from M0 for subsequent formal branches: 45,009 updates, LR 1e-5 with 300 warmup steps, cosine decay to 1e-6. Standard first resumes its completed two-epoch checkpoint with the horizon explicitly retimed to three epochs; this produces an LR increase at the boundary and is reference-only. The old partially trained Fixed-2M is preserved at its 50% full state, while its formal run restarts from M0.

Finish each branch to 300% before starting the next. FN-on runs remain cancelled. Keep full states every 50% and evaluate model trajectories every 20%; retire only verified model-only snapshots. The Standard reference role is propagated to evaluation artifacts and excluded from formal C inputs. The previous constant-LR extension and wait-for-all-two-epoch ordering are cancelled. Standard's clean three-epoch baseline will be retrained separately at the user's instruction.
