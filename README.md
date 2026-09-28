# Modality Gap / Representation Reorganization

[中文](#中文) · [English](#english)

## 中文

研究监督关系、候选组成和同实例竞争如何重组图片、文字及图文组合的表示，并影响评分、排序与下游任务。主要模型是CLIP、BEiT-3和VISTA，ALBEF保留为独立辅助对照。

### 当前状态

截至2026-09-24，本轮训练和评测已结束，目前暂停。CLIP完成seed=42的七个一轮分支；BEiT-3完成seed=42的六个三轮分支及410项A/B1/Local评测。BEiT-3 Standard中途改变过学习率调度，仅作参考，不能进入正式统计。VISTA已有先验及20%诊断结果。

下一轮讨论目标：双4090、500GB工作盘、三个模型各九个分支、seeds=[42,43,44]、每项五轮，共81项；FN-on重新纳入。按seed依次完成，拟滚动保留最近两份完整恢复文件，未完成规定评测的轨迹权重必须保留。

**五轮YAML、81项矩阵入口和上述滚动/转存队列尚未实施。** 当前CLI的 `--matrix` 仍是旧79项选择，不能当作新方案。Global/B3覆盖哪些检查点、C3干预和外部存储安排尚待确定。详见[当前实验状态](configs/training/CURRENT_EXPERIMENT_STATUS.md)。

### 存储范围

| 位置 | 保留内容 |
|---|---|
| GitHub | 代码、脚本、测试、配置、依赖规格、资源来源/SHA和知识说明 |
| 本地工作区 | 上述内容，另可按需保留probe、结果、图表和原始/最终模型；运行资源被Git忽略 |
| 实验机器及外部存储 | 训练和评测数据、全部需要的模型/检查点、向量、逐样本结果和日志 |

所有数据（含probe清单）、预训练及训练后权重、实验结果、凭据、交接文档、修改日志、第三方源码工作副本均不上传。忽略规则不删除磁盘文件；本次不改写Git历史。历史先验源码和操作记录继续留在已有归档位置。

### 新机器准备

clone取得完整源码；模型和数据仍需独立下载或从服务器资源包复制。以下安装命令用于**新的Linux环境**，不要在运行中的实验环境重新安装依赖。

```bash
git clone https://github.com/anuidy/modality_gap_reorganization_by_diff_learning_method.git
cd modality_gap_reorganization_by_diff_learning_method

conda create -n modality-gap python=3.12 -y
conda activate modality-gap
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -r requirements-evaluation.txt
```

实测环境为Python 3.12.3、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、NumPy 1.26.4。见[环境清单](configs/runtime/verified_autodl_4090_20260922.json)。清单和requirements不是完整传递依赖锁，新机器仍需检查。官方OpenAI CLIP由requirements按固定commit安装。

首次准备三个外部源码目录：

```bash
git clone https://github.com/FlagOpen/FlagEmbedding.git third_party/flag_embedding
git -C third_party/flag_embedding checkout --detach fd1a2bdf69488ffebe0327999d4400d8c8058a0b
git clone https://github.com/microsoft/unilm.git third_party/unilm
git -C third_party/unilm checkout --detach 833df7e7832e5064a281131ee64a481afa8e5b95
git clone https://github.com/salesforce/ALBEF.git third_party/albef
git -C third_party/albef checkout --detach b9727e43c3040491774d1b22cc27718aa7772fac
```

已有目录先核对commit和工作区，不覆盖本地改动。代码依赖固定版本，`third_party/`本身不进入本仓库Git。

### 独立准备资源

1. 按[资源来源](docs/RESOURCE_SOURCES.md)取得模型、tokenizer、图片和标注，按[校验值](configs/resources/checksums.sha256)核验。
2. 执行 `python scripts/data/restore_protocol_metadata.py` 恢复小型拆分规则和校验描述。
3. 从资源包复制两份冻结probe清单到 `data/splits/`，保留原始字节、路径字段和换行；SHA必须匹配[数据手册](configs/data/DATASET_RUNBOOK.md)。恢复脚本不包含实际样本，不下载清单或图片。
4. 放置LCS源标注、完整图片和COCO probe后，执行 `python scripts/data/prepare_formal_datasets.py` 生成并验证训练/验证清单。
5. B1和Local按对应运行手册准备，B3/Global按扩展说明另行准备。

AutoDL公共盘只是一种资源来源。新机器没有 `/root/autodl-pub` 时，COCO准备脚本接受 `--public-root`，ImageNet准备脚本接受 `--source`。保持数据版本及内容校验不变。

### 检查及查看计划

```bash
python -m pip check
python -m unittest discover -s tests -t . -p 'test_*.py' -q
python scripts/data/prepare_formal_datasets.py --validate-only --verify-images
```

完整单测需要依赖和固定外部源码，不需要训练集或模型权重。最后一条检查需要真实数据，只扫描图片引用，不代替全部图片解码检查。

以下命令只展示**已实现的历史三轮配置**，不会启动训练：

```bash
python scripts/training/run_independent.py --config configs/training/beit3_three_epoch.yaml --runs beit3_standard --seeds 42 --gpus 0
python scripts/training/run_independent.py --config configs/training/beit3_three_epoch.yaml --runs beit3_standard beit3_fixed_2m --seeds 42 --gpus 0 1
```

双卡每卡独立任务，WORLD_SIZE=1，无DDP、跨卡候选合并或梯度同步。双卡只有调度代码验证，尚未做新机器实测。五轮方案实现并核验后才生成新的执行命令；当前不安排自动开训。

### 项目导航

| 内容 | 入口 |
|---|---|
| 最新决策和未完成事项 | [当前实验状态](configs/training/CURRENT_EXPERIMENT_STATUS.md) |
| 分支和随机性定义 | [实验协议](configs/training/EXPERIMENT_PROTOCOL.md) |
| 已完成三轮协议 | [三轮说明](configs/training/BEIT3_THIRD_EPOCH_RUNBOOK.md) |
| 训练CLI及单双卡 | [训练手册](configs/training/TRAINING_RUNBOOK.md)、[执行模式](configs/training/EXECUTION_MODES.md) |
| 数据与模型 | [数据手册](configs/data/DATASET_RUNBOOK.md)、[资源来源](docs/RESOURCE_SOURCES.md) |
| A/B/C | [指标定义](configs/evaluation/EIGHT_METRIC_PROTOCOL.md)、[A/B诊断](configs/evaluation/AB_DIAGNOSTIC_RUNBOOK.md)、[B1](configs/evaluation/B1_FORMAL_RUNBOOK.md)、[扩展B/C](configs/evaluation/EXTENDED_BC_RUNBOOK.md)、[ABC设计](configs/evaluation/ABC_IMPLEMENTATION_DESIGN.md) |

`src/`保存共享实现，`scripts/`提供命令入口，`tests/`包含回归测试，`configs/`保存配置和协议，`docs/`保留知识参考。历史文档和配置用于解释已有实验，不能覆盖当前状态文件中的后续决策。

## English

This project studies how supervision, candidate composition and same-instance competition reorganize multimodal representations and affect scores, rankings and downstream tasks.

Training is paused. Seven one-epoch CLIP branches and six three-epoch BEiT-3 branches completed at seed 42; all 410 evaluation subjobs in the BEiT-3 campaign finished on 2026-09-24. Its Standard continuation is reference-only because its cosine horizon changed after two epochs. VISTA has pilot/p020 results.

The next proposed campaign uses two independent RTX 4090 workers, a 500 GB work disk, nine branches including FN-on, three seeds and five epochs: 81 runs. Five-epoch configs, the new matrix and rolling archival/retention are **not implemented yet**. The existing `--matrix` still selects the old 79-run scope. B3/Global checkpoint coverage, C3 interventions and external storage remain unresolved. See [current experiment status](configs/training/CURRENT_EXPERIMENT_STATUS.md).

GitHub contains source, scripts, tests, configuration and documentation only. Datasets (including probes), all weights/checkpoints, results, embeddings, logs, credentials, handoffs and change logs are excluded. Ignored resources remain on their existing disks; this does not rewrite Git history.

Follow the shell commands above to clone, create a fresh Python 3.12 environment, install the observed CUDA 12.8 PyTorch stack and check out three pinned upstream repositories. The [environment inventory](configs/runtime/verified_autodl_4090_20260922.json) is observational, not a transitive install lock. Do not reinstall dependencies during a running experiment.

A clone alone cannot train. Obtain resources separately using [resource sources](docs/RESOURCE_SOURCES.md), restore small descriptors with `python scripts/data/restore_protocol_metadata.py`, and copy the exact frozen probe manifests from the resource bundle, preserving bytes and line endings. Prepare/validate training manifests and downstream resources using the linked runbooks. COCO and ImageNet preparation accept `--public-root` and `--source`, so AutoDL's public mount is not required.

Run pip checks and unit tests before resource validation. Plan commands above do not train and use the existing three-epoch configuration, not the proposed five-epoch campaign. Two GPU slots run separate jobs without DDP or pooled candidates. Neither cloning nor setup automatically starts experiments.
