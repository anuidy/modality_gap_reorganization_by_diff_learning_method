# 正式A/B1/Local评测 / Formal A/B1/Local evaluation

**状态更新（2026-09-28）：训练已暂停，已完成实验和下一轮决策以[当前实验状态](../training/CURRENT_EXPERIMENT_STATUS.md)为准。下一轮恢复九分支并计划五轮，但新配置、81项矩阵及滚动保存尚未实施。下文中的旧轮数、FN取消、79项矩阵和“当前运行”均属于当时协议记录，不是新机器的自动执行计划。**

2026-09-22：B1代码已实现。本地与服务器使用同一实现；真实GPU运行证据见服务器`outputs/evaluation/formal_full_v1/queue/status.json`和任务日志。C、Global和B3不在本队列内。

## B1协议

| 数据 | 测试图片 | 原始描述 | 处理 |
|---|---:|---:|---|
| COCO Karpathy | 5000 | 25010 | 用户确认保留全部描述；10张图片各有6条，其余各5条，不截断第6条 |
| Flickr30K Karpathy | 1000 | 5000 | 仅测试集，每图5条；原始文字与标准划分逐条一致 |

COCO使用完整5000图片候选池，I→T对应完整25010描述池，不划成五个1K折。Flickr使用1000图片/5000描述。I→T的所有同图描述都属于正确答案，以排名最靠前的正确答案计算R@K；T→I的正确答案为该描述对应图片。mR为两个方向R@1/5/10六个比例的算术均值，结果存0–1比例。

文本使用Karpathy原始raw描述，模型各自使用原生tokenizer及长度约束。Flickr分包中的`alt_text`是另一套描述，禁止用于B1；`annotations/karpathy_test.json`里的五条原始描述已与`original_alt_text`精确核对。

编码输出raw向量，再L2归一化后计算float32余弦。复用当前GPU JPEG解码与tensor bicubic预处理，少数不支持的JPEG显式记录CPU回退。推理批次32；评分查询分块64，不缩小候选池，也不丢末尾样本。B1仅使用I/T，不构造IT。

分数相同时按原始候选顺序升序确定名次，保存并列数量；因此完全相同的坍塌向量不会全部被记为rank=1。此B1规则单独记录，既有A/Local的并列口径不修改。

每任务保存双向汇总、逐图片/描述排名、正确答案映射、数据/图像/权重/代码身份和完整性标记。不会修改训练权重或选择训练检查点。

## 数据准备

Flickr已经位于`data/raw/flickr30k/`，仅保留测试图片和描述，附下载包与校验报告。COCO仅从公共盘选择5000张测试图片，按原始2014文件名存入独立目录，图片字节不重新编码：

```bash
python scripts/data/prepare_b1_datasets.py --execute
```

该命令拒绝覆盖已有`data/raw/coco_karpathy/`。数据齐全后，B1读取器核验图片与标注的SHA，不修改原训练集、probe或M0。

## 单任务与队列

```bash
python scripts/evaluation/run_diagnostics.py --execute --allow-running \
  --config configs/evaluation/formal_ab.yaml \
  --run-directory outputs/training/formal_full_v1/clip_standard/seed_42 \
  --points p020 --stage b1 --tasks coco flickr30k --memory-gib 3

python scripts/evaluation/run_formal_queue.py \
  --training-plan outputs/training_queues/formal_full_seed42_20260922/plan.json
```

队列默认只显示计划，`--execute`才运行。已启动的队列不要重复启动；运行记录在`outputs/evaluation/formal_full_v1/queue/`。后台启动须独立会话、stdin=/dev/null、stdout/stderr写文件。

三个模型各9分支、seed42、五个点，加三个共享M0，共138个模型状态。每状态执行A-LCS、A-COCO、B-Local12、B1-COCO、B1-Flickr，共690子任务，其中B1为276项。M0每模型共享一次；C、Global、B3不纳入。

队列仅调度已发布的快照：权重与元数据都存在，加载前SHA和来源一致。显式`--allow-running`支持训练尚在进行的情况，不改写训练清单的running状态；旧命令默认仍只接受暂停/完成任务。p100等待final.json原子发布。

当前一个评测worker，最多与一个训练进程并行；PyTorch评测分配预算3GiB，启动要求整卡至少5120MiB实际空闲，整卡超过22500MiB时只停止评测进程并重试。非显存错误停止评测队列并保留日志，训练继续。磁盘保护为尚未产生的全部正式权重保留估计空间和12GiB余量。队列可按同一身份重启，完成产物按SHA校验后复用。

English: B1 evaluates the complete Karpathy test pools with all canonical captions (COCO 5,000/25,010; Flickr30K 1,000/5,000). Bidirectional cosine retrieval uses all image-associated captions as relevant answers, deterministic candidate-order tie breaking and the mean of six recalls. It reuses the established GPU encoding pipeline, preserves per-query results and provenance, and never updates training state. The detached single-worker queue adds B1 to A and MMEB Local across M0 and all five published checkpoints. C, Global and B3 remain deferred.
