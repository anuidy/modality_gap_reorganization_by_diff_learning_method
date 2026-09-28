# BEiT-3两轮训练：按epoch标记进度 / Epoch-based progress

**状态更新（2026-09-28）：训练已暂停，已完成实验和下一轮决策以[当前实验状态](CURRENT_EXPERIMENT_STATUS.md)为准。下一轮恢复九分支并计划五轮，但新配置、81项矩阵及滚动保存尚未实施。下文中的旧轮数、FN取消、79项矩阵和“当前运行”均属于当时协议记录，不是新机器的自动执行计划。**

2026-09-23修正：**1个epoch=100%=15003步；2个epoch=200%=30006步。** 原先把整个两轮计划当成100%是口径错误，历史结果保留原始标签，并通过步数解释，不篡改。

## 当前执行

BEiT-3、seed42、六个启用分支，单卡顺序训练、最多并行一个评测进程。Standard从旧队列15003步的完整恢复点（正确口径100%）恢复到新目录；其余五项从M0开始。暂停后未保存的约1700步重新执行。总预算、LR计划、warmup300、batch36、BF16、AdamW及随机流保持一致。

恢复入口`train.py --resume ... --resume-source-manifest ...`只允许改变输出目录、保存/验证间隔及进度基准；若LR、种子、数据、模型、batch、总步数等任何训练控制不一致则拒绝。检查来源manifest身份和完整权重SHA，恢复模型、优化器、随机状态、数据位置。旧目录不覆盖，来源完整权重不修改、不删除。

## 保存与评测

| 进度 | epoch | 更新步数 | 保存 |
|---|---:|---:|---|
| 20% | 0.2 | 3001 | 模型轨迹 |
| 40% | 0.4 | 6001 | 模型轨迹 |
| 50% | 0.5 | 7502 | 完整恢复 |
| 60% | 0.6 | 9002 | 模型轨迹 |
| 80% | 0.8 | 12002 | 模型轨迹 |
| 100% | 1.0 | 15003 | 模型轨迹＋完整恢复 |
| 120% | 1.2 | 18004 | 模型轨迹 |
| 140% | 1.4 | 21004 | 模型轨迹 |
| 150% | 1.5 | 22505 | 完整恢复 |
| 160% | 1.6 | 24005 | 模型轨迹 |
| 180% | 1.8 | 27005 | 模型轨迹 |
| 200% | 2.0 | 30006 | 完整恢复兼最终评测 |

完整恢复点50/100/150/200%长期保留，retention=4。200%复用完整权重进行评测，避免重复保存。纯模型轨迹在A/B1/Local全部完成并回读SHA、核对来源身份后删除；保留JSON、日志、原始向量、结果及删除凭据。

Standard已错过的20–80%轨迹不补造、不重训第一轮。旧6001/12002步的评测对应正确口径40/80%，仍保留在旧结果目录。新目录会导出恢复时100%轨迹，评测100/120/140/160/180/200%。其余五分支各评测全部十个轨迹点，另共享一个M0，共285个评测子任务（每个Local子任务含十二项任务）。没有续训来源的全新六分支计划应有305个子任务。

Validation也每20%（0.2epoch）执行，batch36、固定丢8条、实际7992条，不用于早停或选权重。训练日志同时记录`progress_percent`（epoch口径）和`budget_completion_percent`（占两轮总预算比例），防止再混淆。

## 文件与运行

- 训练配置：`configs/training/beit3_two_epoch.yaml`；`checkpointing.progress_reference_steps=15003`。
- 评测配置：`configs/evaluation/beit3_two_epoch_ab.yaml`，points为m0及p020…p200。
- 调度入口：`scripts/training/run_beit3_two_epoch.py`。
- 新训练/评测根：`outputs/training/beit3_two_epoch_v2/`、`outputs/evaluation/beit3_two_epoch_v2/`。
- 新队列：`outputs/training_queues/beit3_two_epoch_seed42_fn_off_20260923/`。
- 旧v1目录及冻结配置只作为历史与恢复来源。旧暂停进程经确认恢复文件有效后退出，不再解冻旧队列。

GPU评测沿用原A0–A6、B1、十二项Local实现。训练启动至少120秒且确实完成至少10步更新后才允许并行评测；至少5GiB显存空闲，评测预算3GiB，总显存超过22500MiB则延后评测，不改训练batch。C、Global、B3不在本队列。

完整恢复点数量增加，按历史权重大小估计本轮新增完整权重约58.7GB（Standard的100%来源另保留），全部纯轨迹若不删除约44.4GB；采用逐点评测后删除可降低常驻量。启动前至少115GiB空闲，后续启动分支至少20GiB、评测至少15GiB。独立session与文件日志保证断开SSH仍运行；源代码/配置冻结校验失败时停止，不静默换参数。

## English

Progress is measured in epochs: 100% = one epoch = 15,003 updates; 200% = two epochs. Evaluate/save model trajectories every 20%, and retain full recovery checkpoints every 50%. The final 200% full state doubles as the final evaluation snapshot. Validation follows the same epoch-based 20% interval.

Continue Standard from its verified 15,003-step full state with unchanged optimizer, RNG, data stream, LR schedule and total budget, using a new output directory. Preserve its historical first-epoch results without relabeling or fabrication; evaluate the restored 100% point and all later trajectory points. Five remaining branches start from M0. The correction campaign schedules 285 A/B1/Local subjobs; a fresh six-run campaign would schedule 305. The old controller stays retired, and the new detached queue is authoritative. Global/B3/C remain deferred.

## 2026-09-23分支取消

用户更正为取消FN-on：Fixed-3M FN-on、Mixed-3M FN-on、Full GCL FN-on不再排队。保留Standard、Fixed-2M、Fixed-3M FN-off、Mixed-2M、Mixed-3M FN-off、Full GCL FN-off。Standard本项不变；完成后新队列接管已完成训练和评测，继续其余五项。本节描述前两轮阶段；随后用户已授权以1e-6追加第三轮，见[第三轮接续](BEIT3_THIRD_EPOCH_RUNBOOK.md)。原九分支定义和旧结果保留用于复现，不从历史文件删除。

FN-on runs are cancelled, while FN-off runs remain active. The current Standard run finishes unchanged; a replacement controller adopts its verified outputs and runs the remaining five branches. This section describes the initial two-epoch stage; the authorized appended epoch is documented in BEIT3_THIRD_EPOCH_RUNBOOK.md.
