# 正式训练实验协议

**状态更新（2026-09-28）：训练已暂停，已完成实验和下一轮决策以[当前实验状态](CURRENT_EXPERIMENT_STATUS.md)为准。下一轮恢复九分支并计划五轮，但新配置、81项矩阵及滚动保存尚未实施。下文中的旧轮数、FN取消、79项矩阵和“当前运行”均属于当时协议记录，不是新机器的自动执行计划。**

版本：formal_v1，2026-09-20。本协议替代先验阶段的八/十七组训练入口；先验结果保留，不混入正式统计。

## 研究目的

研究监督关系、候选组成、同实例竞争与融合架构如何改变表征，并通过后续 A/B/C 实验连接评分、排序和下游表现。研究结论以正式结果为依据，不预设 FN-off 或 Mixed 一定改善性能。

## 公共控制

LCS train 540,128 对；每任务 batch=36、accumulation=1、drop_last=True，每轮使用540,108对、丢20对；1轮15,003次更新。AdamW：LR=1e-5、WD=0.05、betas=(0.9,0.999)、epsilon=1e-8；预热300步、余弦衰减、最低比例0.1；BF16、全参数、梯度裁剪关闭。random resized crop scale=0.9–1.0、hflip=0.5。

验证仍读完整8000条清单，batch与训练一致，drop_last=True，固定顺序丢末尾8条，实际7992条/222批；不更新参数、不选权重、不早停。见 [验证协议](VALIDATION_PROTOCOL.md)。

单4090依次训练任务；双4090每卡独立训练一个任务，均WORLD_SIZE=1，不使用DDP、跨卡gather或梯度合并。每任务候选实例数一直为36。

只保存1%、5%、20%、50%、100%的模型轨迹快照，不保存完整resume状态。Validation独立按每20%执行。已完成的20%诊断为24项seed42任务，停在3001步；其72份轨迹在完整评测后按用户授权退役，日志、向量和结果保留。准备中的完整单seed训练使用`formal_single_seed.yaml`，每任务从M0运行到15003步，不使用`--gate`或3001步停止参数。

## 九个主模型分支

| branch | 监督分配 | 屏蔽前候选数 | 有效候选数/查询 | 查询数/步 |
|---|---|---:|---:|---:|
| standard | 全批I↔T | 36 | 36 | 72 |
| fixed_2m | 全批I↔T | 72 | 71 | 72 |
| fixed_3m_fn_off | 全批I↔T | 108 | 106 | 72 |
| fixed_3m_fn_on | 全批I↔T | 108 | 107 | 72 |
| mixed_2m | 同一步三组各12实例 | 72 | 71 | 72 |
| mixed_3m_fn_off | 同上 | 108 | 106 | 72 |
| mixed_3m_fn_on | 同上 | 108 | 107 | 72 |
| full_3m_fn_off | 每实例六方向 | 108 | 106 | 216 |
| full_3m_fn_on | 每实例六方向 | 108 | 107 | 216 |

Mixed三组分别I↔T、I↔IT、T↔IT，只划分查询，候选来自完整36实例。2M使用当前关系的两种模态；3M使用I/T/IT。查询自身始终排除；FN-off额外排除同实例第三种非目标表示，FN-on保留为竞争者；每个方向只有一个正确目标。总损失按所有有效有方向查询取平均。所有正式分支的similarity/logits与CE计算边界统一为float32，编码器在BF16 autocast下计算；Standard数学定义不变。

CLIP/BEiT-3先获取原始head输出，raw eIT=eI+eT，再在目标函数中分别L2归一化；梯度不detach。VISTA使用原生encode_mm；其训练编码接口本身归一化，保留原生temperature=0.02，不改为加法。

旧Rotating和Fixed-6N不再是可执行分支，也不保留命令行别名。

## 随机性、任务身份和预算

一个主seed按固定SHA-256规则派生data、augmentation、group、model随机流，使用epoch/step/batch位置寻址；Mixed分组使用独立CPU Generator。卡号和任务顺序不影响种子。当前模型轨迹不保存优化器及RNG状态，不支持精确续训；日志保留训练步数和数据流位置用于核对。

每个模板run运行时的身份为`<model>_<branch>_seed_<seed>`。本次完整训练使用`outputs/training/formal_full_v1/<model>_<branch>/seed_42/`；旧`outputs/training/formal_v1/`保留20%诊断记录，不再用于本次正式输出。日志及评测还须按config/checkpoint SHA区分不同运行，不能只用run_id匹配跨轮结果。

主矩阵：CLIP 9×3、BEiT-3 9×3、VISTA 8×3+Fixed-2M一seed，共79任务。正式seed列表已确认42/43/44；默认单任务和gate使用42，VISTA Fixed-2M的单seed也使用42。可显式传`--seed`/`--seeds`选择任务。

20%诊断原计划27项，最终完成每模型8项、合计24项；三个Full GCL FN-on未执行。完整单seed计划按每模型九分支合计27项，于2026-09-22授权启动独立后台队列，先校验后顺序训练；当前进度见`outputs/training_queues/formal_full_seed42_20260922/queue_state.json`和各任务记录。不能把未跑过的三项写成已实测通过。`--matrix`仍是三seed主矩阵入口；单seed按`--runs`显式选择，见运行手册。当前不保存resume状态，不从20%快照冷启动优化器冒充连续训练。

## ALBEF辅助实验

仅itc_only与full_albef（原生ITC+ITM+MLM），不人工构造IT。保持queue=65536、momentum=0.995、alpha=0.4；队列使用环形写入支持batch=36，跨末尾时接着写入开头，动量每更新一次、入队每更新一次的语义不变。使用相同公共seed列表42/43/44，两个分支独立安排，不计入79个主模型任务。

## 验证范围与后续评测

三个模型各8项已在4090上完成batch=36的3001步诊断，另三项Full GCL FN-on暂无本轮GPU记录。A0–A6、COCO4407、Norm Dynamics和MMEB十二项Local已实现并完成当前诊断；完整epoch自动调度、其他B任务、Global及C仍需后续工作。训练身份锁继续引用原始COCO5K源清单，派生4407子集不能覆盖原文件。已删除的诊断权重以归档退役记录为准，原M0及所有数值结果保留。
