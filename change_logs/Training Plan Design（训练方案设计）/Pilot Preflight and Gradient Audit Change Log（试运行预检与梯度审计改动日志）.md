# Pilot Preflight and Gradient Audit Change Log（试运行预检与梯度审计改动日志）

## 改动类型

- 脚本：新增独立 preflight（运行前检查）与模块梯度审计，覆盖八组实验分支。
- 模型接口：仅增加 checkpoint（模型权重文件）加载诊断信息，不改变现有 forward/backward、损失或训练状态更新逻辑。
- 配置与文档：新增审计协议并更新训练手册；正式 `train_runs.yaml`、Probe（固定评测集）、M0 和实验矩阵均未修改。
- 数据：只使用固定 LCS Train 的一个真实批次，不下载数据，不读取 Probe 图片作为审计输入。

## 文件位置

- `scripts/training/audit_gradients.py`：命令行入口、独立输出目录、JSON 报告与退出码。
- `src/training/audit_config.py`：仅解析诊断所需参数，复用实验矩阵与数据身份校验，检查 M0 文件 SHA-256（内容校验值）。
- `src/training/audit_groups.py`：参数分组、按损失的预期梯度、共享参数别名、逐参数状态与分组判定。
- `src/training/gradient_audit.py`：加载单个训练批次、复用 backend（模型训练接口）、各关系/损失的独立反传、状态恢复、计时和显存报告。
- `src/training/backends.py`：增加 `checkpoint_load_report`；BEiT-3/ALBEF 保留缺失和多余权重名称，供审计严格拒绝不完整加载。现有正式训练加载行为未变。
- `tests/training/test_audit_config.py`：诊断参数、未冻结正式配置、数据角色与文件身份检查。
- `tests/training/test_gradient_audit.py`：梯度判定、路径覆盖、状态隔离、异常处理与运行器集成测试。
- `configs/training/GRADIENT_AUDIT_PROTOCOL.md`：范围、判定标准、结构差异、运行方法和验证边界。
- `configs/training/TRAINING_RUNBOOK.md`：加入独立审计入口，明确后续带 optimizer step（参数优化步骤）的 Pilot（小规模试运行）属于另一阶段。

## 实现逻辑

1. 调用者显式提供诊断 seed（随机种子）、batch size（实际批次大小）与 augmentation（图像预处理）；ALBEF 还需显式 alpha（动量软目标混合系数）。不构造虚假的正式学习率、训练步数或预热计划。
2. 从固定 Train manifest（样本清单）按现有确定性采样器取 epoch 0 首批，预处理一次。各独立项恢复同一初始模型 tensor state（参数及状态张量），清空梯度并重置相同 forward 随机种子。
3. Standard 检查 I↔T；Mixed 分别检查三个关系；Full ALBEF 对同一完整 forward 分别重算并检查总损失、ITC、ITM、MLM 的梯度。
4. 活跃组要求已有梯度全部有限且组内至少一个参数非零；非活跃组要求 `.grad=None`；动量参数额外要求冻结。未知参数、缺失必需组、非有限值不能通过。逐参数保存名称、别名、大小、梯度类别与 L2 norm（梯度长度）。
5. VISTA 图像路径会复用原生 joint（图文联合）方法，记录各方法调用次数；其固定 temperature（温度系数）不要求梯度。BEiT-3 注意力、FFN（前馈网络）和归一化的 A/B 参数分别列出。ALBEF 的共享 MLM decoder/词嵌入权重只计一次。
6. ALBEF 开始窗口时执行既有一次显式动量更新，结束时 abort（丢弃暂存特征），不入队。forward 只允许动量参数和温度发生既有显式变化，backward 不允许任何参数或 buffer（状态张量）变化。正常与异常结束均恢复初始 tensor state、模式与审计函数入口的随机数状态。
7. 不创建 optimizer、不执行其 step、不保存训练 checkpoint。输出位于独立 `outputs/pilot/gradient_audit/<run>/<UTC timestamp>/report.json`，含身份、相关源码 hash、环境、损失/梯度/状态、时间和峰值显存。

## 验证结果与边界

本地命令：

```powershell
& 'D:/conda/envs/modality-gap/python.exe' -m unittest discover -s tests -t . -p 'test_*.py' -q
```

结果：**79 项测试通过**，包括原有 55 项及新增 24 项。新增验证覆盖：断开梯度、旧梯度清理、全零/非有限值、未知/缺失模块、非活跃零梯度、未冻结动量参数、共享权重、forward/backward 非预期状态变化、异常恢复、ALBEF 逐损失检查与禁止入队、VISTA 原生路径计数、小型原生 torchscale BEiT-3 三关系反传、真实临时图片/清单与模拟 backend 集成、M0 不完整加载拒绝、CUDA 默认/显式设备编号解析。

命令行 `--help` 正常，已有文件差异通过 `git diff --check`。训练手册中原先直接进入参数更新 Pilot 的说明已调整为先独立审计，后续阶段另行确认；历史日志保留为历史记录。

本地 PyTorch 为 `2.9.1+cu128`，测试使用 CPU 小模型/模拟 backend；这不等价于仓库指定 `2.7.1` 依赖组合的 AutoDL 验证。未运行四个完整 M0 模型的 GPU 审计，未冻结诊断或正式超参数，未开始正式训练，未验证真实 BF16、A800 峰值显存或吞吐。

审计显存不含 optimizer 状态，时间不含模型状态复制/校验和梯度统计；Full ALBEF 的分损失检查仍包含完整 forward 图。gradient accumulation（梯度累积）、实际参数更新、checkpoint 写入、resume（恢复训练）一致性保留给下一阶段。

本次按用户确认进行本地 commit（提交）；不包含 push（推送）GitHub。
