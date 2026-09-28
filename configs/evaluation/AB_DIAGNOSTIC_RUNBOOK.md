# A/B训练策略诊断

**状态更新（2026-09-28）：训练已暂停，已完成实验和下一轮决策以[当前实验状态](../training/CURRENT_EXPERIMENT_STATUS.md)为准。下一轮恢复九分支并计划五轮，但新配置、81项矩阵及滚动保存尚未实施。下文中的旧轮数、FN取消、79项矩阵和“当前运行”均属于当时协议记录，不是新机器的自动执行计划。**

当前用途：检查单seed、20%训练是否重复先验中的表征或检索退化。A0–A6与现有MMEB十二项Local检索已实现；C类统计机制分析、Global、缺失图像的Karpathy/Flickr完整检索和B3分类暂缓。本文件与`ab_diagnostic.yaml`是本轮执行口径，不将诊断结果冒称完整论文实验。

## 输入与产物

- 只读取状态为paused/complete的训练任务；明确请求m0、p001、p005、p020。未来p050/p100不作缺失错误，训练状态及checkpoint侧车文件不被改写。
- A使用完整LCS10K和COCO4407，保留最后不足推理批次的样本。COCO4407由原5000清单减去Karpathy test的593个重叠图像ID得到，保留原顺序、caption、父文件SHA；数量不符时失败，不替换定义。
- 本轮M0与训练点使用相同GPU图像预处理，M0重新抽取到新命名空间。原M0的32文件锚点、原5000清单及历史结果不覆盖。
- CLIP/BEiT-3保存pre-L2 raw I/T，IT由raw I+T重建；VISTA另外保存原生raw IT。编码沿用已验证适配器FP16推理，归一化和评分采用FP32。训练BF16、batch36等设置不变。
- 输出在`outputs/evaluation/ab_diagnostic_v1/`，分为embeddings、A、B_Local、inputs、queue。每项记录checkpoint/config/数据/代码SHA及样本顺序；complete标记包含实际产物SHA，损坏缓存拒绝复用。
- 原始向量、逐样本统计和分布采样保留，完整N×N分数矩阵不落盘。按已确认的300GB单项周转计划归档；队列不会自动删除任何权重或上传网盘。

## A0–A6定义

| 模块 | 本轮实现 |
|---|---|
| A0 | 既有centroid、相对covariance、effective rank的raw/L2口径；alignment、全局Score Gap、anisotropy、相对M0的几何保持；新增逐样本Norm Dynamics |
| A1 | 六方向匹配正例及j≠i负例分布；全量负例参与均值、样本标准差和100格直方图；固定seed抽取最多100000个有向负例，仅用于绘图 |
| A2 | 每个query对I/T/IT背景分布的三组精确W1与有符号均值差；全部排除同实例，不能用A0混合全部query的W1替代 |
| A3 | 排除所有同实例表示后取最强错误候选；保存M_first、M_all、负间隔比例、最难负例模态和模态并列数 |
| A4 | 统一3N池，只屏蔽query本身；六个有向目标分别排序；主rank=严格高于目标的候选数+1，另存悲观rank以揭示并列 |
| A5 | K=1/5/10；同时报inclusive与background。background排除同实例全部表示；减去各自真实候选模态先验。边界并列按比例分摊Top-K名额，避免模态排列顺序制造偏置 |
| A6 | 同A4的固定候选范围，六方向分别计算概率及CE；主温度0.1，另报原生温度。CLIP/BEiT-3取该点logit_scale；VISTA取训练配置0.02，不取raw导出时被重置的temperature |

Norm Dynamics：先逐样本计算`r_i=log(norm(I_i)/norm(T_i))`及`delta_out_i=sign(r_M0_i)*(r_i-r_M0_i)`，再统计mean/median/std，ddof=1。旧`abs(log(mean_norm_I/mean_norm_T))`只保留为辅助字段。加法模型另存D_IT。零模长明确失败；常量向量的Spearman记null/undefined，有效秩等仍可输出。

几何保持使用固定seed=20260825的一百万个上三角样本对；小型单测取全部可用对。邻居重叠是辅助量。计算分块不改变候选范围；大规模相似度、排序和编码走GPU，协方差谱采用原有float64 CPU算子，全局Score Gap最后的完整分布排序在CPU进行，以限制显存。

`run_step1_diagnostic.py`单独复现CLIP/BEiT-3的首个训练批次：复用数据顺序、训练增强、模型随机流、BF16和train模式，从全新M0前向但不反向、不更新参数。保存首批身份、实际raw相加dtype以及固定/原生温度下的低温近似关系残差；它不是“已更新一步”的检查点。该项使用统一A6的3N候选池，不冒称所选分支的实际训练loss；36行的小规模统计在CPU计算。VISTA原生joint不套用加法近似。该辅助项不作为训练通过/失败条件，不用probe结果冒充step-1。

## B类当前范围

只使用服务器已有MMEB十二项检索任务。每任务1000查询、每查询原生1000候选，首候选ID为标注答案；保留原有official文本清理，不移除有实际内容的任务指令。相同候选ID的重复出现视为同一答案。

报告R@1/5/10、逐查询rank及并列数。旧6/5/1分组保持：cross-modal为VisDial、VisualNews_t2i/i2t、MSCOCO_t2i/i2t、Wiki-SS-NQ；mixed-composite为CIRR、WebQA、OVEN、FashionIQ、EDIS；同模态为NIGHTS。缺任务时不生成伪完整的十二任务均值。

这些MMEB任务中的MSCOCO不是完整COCO Karpathy检索测试，不互相替代。ALBEF不在当前三模型诊断入口范围，也不恢复旧先验的人工IT兜底。

已有MMEB准备缓存与新读取器的查询、候选和文本内容已逐项核对。存在该缓存时校验原始parquet SHA、固定准备器SHA与缓存SHA后复用；缓存缺失时由公开读取器从parquet构建。只是复用数据产物，不依赖旧训练或评测脚本执行。

## 命令

默认只显示计划，显式加`--execute`运行。以下在服务器项目根目录执行：

```bash
python scripts/evaluation/run_diagnostics.py --run-directory outputs/training/formal_v1/clip_standard/seed_42 --stage a --points m0 p001 p005 p020
python scripts/evaluation/run_diagnostics.py --run-directory outputs/training/formal_v1/clip_standard/seed_42 --stage b --points m0 p001 p005 p020
python scripts/evaluation/run_step1_diagnostic.py --run clip_standard --seed 42
```

`--probes lcs`可只选一个完整probe；`--tasks MSCOCO_t2i CIRR`可选已定义任务子集，报告会标记未完成的总体覆盖。`--batch-size`仅改变编码分批；默认32，`--memory-gib`默认3。实际GPU设备编号会显式规范化。

历史诊断队列入口（已完成并退役权重，不再启动）：

```bash
python scripts/evaluation/run_adaptive_diagnostic_queue.py --training-plan outputs/training_queues/formal_p020_seed42_20260921/plan.active.json
```

这套队列已经完成24项诊断，共225个子任务（每个A子任务是一个完整probe；每个B子任务是十二项Local）。2026-09-22补齐三项后，72份轨迹已校验并按用户指令删除，日志、原始向量和全部结果继续保留，不能再用已退役权重重跑。三个Full-3M FN-on未做该轮诊断。以下说明记录已执行的调度方式，不表示完整正式epoch调度已准备好。

训练期间只运行一个评测worker；训练全部完成并通过权重校验后，再评估2/3进程并行。用同一模型/阶段/probe的5个尚未完成任务分别进行2进程和3进程试跑，实际产出直接计入完成量。按原单进程中位耗时归一化比较吞吐量，3进程须比2进程高5%以上且显存峰值低于20000MiB，才选择3；不足5个可比任务时直接保守选择2（仅余一项时选1），不为测速重复完整评测。每进程仍限制PyTorch分配为3GiB，总显存超过22500MiB则回退并发并有限重试。为未完成训练保留预计写盘空间及10GiB余量。普通评测错误停止队列，不修改训练策略。

调度器独占原全局评测锁，只有带本协调进程身份的worker可以并行；每任务、M0/检查点向量、图片清单、派生probe及MMEB缓存分别加锁。底层DiagnosticRunner、数值计算、输入预处理、候选池、温度、结果身份保持不变；新调度及worker源码SHA另记入队列身份和执行日志。旧`run_diagnostics.py`入口继续维持单worker保护。

该队列可读取完成标记继续，不绕过来源和产物SHA检查。代码或协议改变时拒绝沿用旧队列身份。完成时间和实际资源消耗以真实运行日志为准。
