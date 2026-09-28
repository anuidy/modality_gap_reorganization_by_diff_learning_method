# B3 / Global / C implementation

**状态更新（2026-09-28）：训练已暂停，已完成实验和下一轮决策以[当前实验状态](../training/CURRENT_EXPERIMENT_STATUS.md)为准。下一轮恢复九分支并计划五轮，但新配置、81项矩阵及滚动保存尚未实施。下文中的旧轮数、FN取消、79项矩阵和“当前运行”均属于当时协议记录，不是新机器的自动执行计划。**

2026-09-22：本轮仅实现代码与验证，不启动完整B3/Global，不将C加入当前运行队列。新代码在`src/evaluation/extended/`和`src/analysis/`；现有A/B1/Local源码身份与数值定义保持不变。当前运行的690项队列继续原计划。

## 已确认口径

- B3：官方OpenAI CLIP固定commit的1000类名和80个模板；每模板向量L2归一化，再逐类平均，最后再归一化。类索引按ImageNet leaf WNID排序，通过官方devkit把验证标注映射到对应索引。官方别名有重复，仍保留1000个类别索引，不擅自改名/合并。
- Global：官方M-BEIR完整测试候选库、16项任务，验证查询只用于校准，测试查询只用于评测。文本使用原始query_txt/txt，不追加新的指令前缀。原图像预处理、原生单模态编码与VISTA原生IT继续复用。
- Calibration：按“查询任务×候选模态”，先计算每条独立验证查询对该模态全部非相关候选的平均余弦，再对同任务查询等权平均。利用线性恒等式精确计算均值，不用近似候选采样。正确答案若不在测试候选库，对该验证查询的排除集合取实际在库的正确答案；测试查询缺少任何正确答案则拒绝评测。
- Oracle：根据测试标注，保留所有正确答案涉及的模态并集，不直接选择正确候选；同一模型、查询、余弦和原始候选顺序保持不变。它是借助答案模态的诊断，不能作为实际系统性能或单独的因果证据。
- Global三个视图raw/calibrated/oracle共用一份候选向量。raw指未经偏移校准的归一化余弦，不是raw向量点积。R@K采用官方UniIR实现的“前K中有任一正确答案即命中”；保存各任务、任务等权平均和查询加权平均，不混淆分母。并列按候选原顺序打破。
- C1：同模型/任务/点/视图内按训练seed配对，报告原值与差值均值、样本标准差、95%双侧Student-t区间。小seed数为参数近似；单seed不生成CI。VISTA Fixed-2M只作为seed42桥接对照，不虚构43/44的结果。
- C2：`Delta M = M_current - M0`，保留方向与分支数值排序，不除以M0、不用跨模型绝对值自动判断优劣。
- C3：仅输出保留model/branch/seed/point/数据视图/来源身份的链路数据，不做相关性、回归、显著性或因果推断。后续方法另定。

## 可执行入口

新入口默认只显示计划，`--execute`才运行。B3/Global完整GPU执行必须另外安排资源；GPU0上的正式A/B1队列仍存活时，新入口拒绝抢占同一卡。当前代码任务没有启动这些完整实验。

单卡逐项执行；双卡可对不同检查点分别运行`--device cuda:0`和`--device cuda:1`，每个进程独立编码/打分，不使用DDP。锁按阶段和权重SHA设置，同一检查点的同一阶段禁止重复并发，不同检查点的结果和向量目录各有独立身份。

### B3数据与评测

```bash
# 默认只检查公共盘devkit和文件，不提取图片。
python scripts/data/prepare_imagenet_b3.py
# 正式准备时才加--execute；只提取50000张验证图片，不读取/复制训练集。
python scripts/data/prepare_imagenet_b3.py --execute

python scripts/evaluation/run_extended.py --stage b3 \
  --run-directory outputs/training/formal_full_v1/clip_standard/seed_42 --point p100
```

ImageNet当前公共盘提供验证tar及devkit，但正式准备目录尚未生成。已准备数据的图片逐项SHA验证，推理不丢尾样本，逐图片保存标签与Top5预测。

### Global数据与评测

`mbeir_global_sources.json`锁定官方数据commit `023f7cc5425e15e3c93f6146f7295c28eae767dc` 的65个候选/查询/qrels文件。对应来源与字节数在同名provenance JSON中。本轮只获取文件身份，不下载大库。

```bash
python scripts/data/prepare_mbeir_global.py \
  --data-root /path/to/M-BEIR \
  --source-lock configs/evaluation/mbeir_global_sources.json
# 图片及所有锁定文件到齐后，再加--execute生成内容索引。

python scripts/evaluation/run_extended.py --stage global \
  --run-directory outputs/training/formal_full_v1/clip_standard/seed_42 --point p100
```

索引使用SQLite，逐项验证query与官方qrels一致、test相关候选存在、图片内容SHA、val/test查询ID和完整输入无重合；重合或缺失时明确拒绝，不自动改变实验划分。候选向量按8192行分片，float32归一化存储，SHA和输入顺序共同标识已完成分片；中断可复用已完成分片。它们属于B类大库缓存，不替代A类raw主产物。

B3结果和Global向量缓存还记录实际第三方模型源码、tokenizer（文本切分器）文件、VISTA文本骨干资源的SHA，以及相关运行库版本。更换这些依赖后生成新的缓存身份，不能沿用旧分片；当前A/B1队列的身份规则不改。

精确排名使用相同块边界的两遍扫描，不持久化查询×候选矩阵。结果保存逐查询排名、并列数、top1和最佳正确候选行号、候选数、校准来源和三个视图汇总。全量规模的吞吐、磁盘峰值与所有原始数据兼容性仍需真实Global实测；小型端到端测试不能替代这一点。

### C的范围与运行条件

```bash
# 先冻结预期全部A/B结果范围；这一步不运行C分析。
python scripts/analysis/analyze_formal.py --prepare-scope \
  --scope outputs/analysis/C_scope.json --seeds 42 43 44

# 默认仅核对覆盖，不计算C1/C2/C3。
python scripts/analysis/analyze_formal.py --scope outputs/analysis/C_scope.json
# 全部范围齐全后，才使用--execute；输出目录必须是新目录。
python scripts/analysis/analyze_formal.py --scope outputs/analysis/C_scope.json --execute
```

范围包括A两probe、B1两数据集、12项Local、B3和Global，及M0/五个检查点。按单seed生成范围可用于最终单seed描述性分析；多seed正式分析使用42/43/44，且执行前必须全部配对。当前A/B尚未结束，C不运行。C1另输出Global分支间Gap_raw、Gap_cal、带符号的Gap_cal−Gap_raw及绝对差距变化，避免把符号变化误称为差距缩小。

## 依赖、证据和限制

可选依赖为`requirements-evaluation.txt`中的SciPy 1.13.1，用于ImageNet MATLAB标注与t分布。服务器可将它安装至`_local/extended_eval_dependencies`，仅新入口按需加载；不覆盖训练环境的NumPy/PyTorch。

验证覆盖模板归一化顺序、重复官方类名、多模态多正确答案、并列分数、校准无泄漏、分块/完整矩阵一致性、分片中断恢复、qrels与图片身份、Global端到端小型流程、配对seed统计、M0为零以及C范围缺失拒绝。现阶段不宣称完整ImageNet、560万候选Global或正式C结果已经运行。

Sources: [official CLIP templates](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/notebooks/Prompt_Engineering_for_ImageNet.ipynb), [M-BEIR format](https://huggingface.co/datasets/TIGER-Lab/M-BEIR), [UniIR recall implementation](https://github.com/TIGER-AI-Lab/UniIR/blob/main/src/common/mbeir_retriever.py).

English: B3 and Global have separate executable entry points and never join the existing A/B1 queue automatically. C1/C2 code is implemented; C3 exports traceable linkage data only. All declared A/B artifacts must be complete and hash-verified before C execution. The full ImageNet/Global datasets and large-scale GPU runs remain future work; present tests verify computation and workflow, not full-benchmark completion.
