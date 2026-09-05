# 正式训练运行手册

本目录现在包含正式 8-run 矩阵的可执行训练配置。代码不会把 COCO 5K 或 LCS 10K probe 当成训练集，也不会替研究者猜测尚未冻结的超参数。

## 代码位置

- `src/datasets/training_pairs.py`：训练 manifest、唯一 semantic instance 校验和确定性采样。
- `src/objectives/contrastive.py`：Standard 与 Count-Matched Mixed 的双向对比目标。
- `src/training/backends.py`：CLIP、VISTA、BEiT-3、ALBEF 的可训练后端。
- `src/training/config.py`：8-run 矩阵和控制变量校验。
- `src/training/engine.py`：单卡 A800 的 BF16、梯度累积、日志、断点与 checkpoint。
- `scripts/training/train.py`：单个 run 的命令入口。
- `scripts/training/audit_gradients.py`：独立真实批次梯度审计，不执行参数优化步骤。
- `scripts/training/run_all.sh`：配置完成后依次校验并运行 8 个分支。

## 训练数据 manifest

训练数据使用 JSON 或 JSONL。每条记录必须包含：

```json
{"sample_id":"pair-000001","semantic_id":"instance-000001","image":"subdir/000001.jpg","text":"caption"}
```

`image` 必须是相对 `image_root` 的路径。`sample_id` 和 `semantic_id` 必须全局唯一。这个限制保证每个 query 恰好只有一个 positive，其余 `N-1` 个 candidate 都是真 negative。完整示例见 `data/metadata/training_manifest_example.jsonl`。

## 先运行独立梯度审计

在填写正式训练参数前，先按 `configs/training/GRADIENT_AUDIT_PROTOCOL.md` 指定诊断 seed、batch size 与预处理设置，运行 `scripts/training/audit_gradients.py`。该入口核对 M0/数据身份并检查各关系、各损失的模块梯度；只做到 backward，不执行 optimizer step。ALBEF 还需显式诊断 alpha。

审计结果位于 `outputs/pilot/gradient_audit/`，不会写入正式训练目录。当前阶段先完成该审计；带参数更新的 Pilot 另行确认后执行。

## 正式训练前冻结配置

编辑 `configs/training/train_runs.yaml` 中的 `null` 字段：

- seed；
- train manifest 与 image root；
- optimizer、learning rate、weight decay；
- scheduler、warmup、minimum LR ratio；
- 每个模型的 micro-batch 与 gradient accumulation；
- max steps；trajectory progress 与 resume retention 已固定；当前 20% resume interval 在 Pilot 后重新确认；
- 每个模型的 augmentation。

顶层 `controls` 是四个模型共同的默认值。若不同模型需要不同优化设置，可在对应 `models.<model>` 下增加 `optimizer`、`scheduler` 或 `budget`；同一个模型的两个分支仍从同一个 model block 读取，因此不能在 run 层改变控制变量。

独立梯度审计通过后，下一阶段可另行确定带参数更新的 pilot 配置，跑通每个模型的两个分支各 2–10 个 optimizer steps，验证实际更新、优化器显存、checkpoint 写入和 resume，再冻结正式 budget。这个阶段不由梯度审计脚本启动。

## AutoDL 执行顺序

安装 CUDA 版 PyTorch 与依赖后，先完成上述独立梯度审计。以下命令属于配置已填写的训练阶段，逐项校验：

```bash
python scripts/training/train.py --run clip_standard --validate-only
python scripts/training/train.py --run vista_standard --validate-only
python scripts/training/train.py --run beit3_standard --validate-only
python scripts/training/train.py --run albef_itc_only --validate-only
```

运行一个分支：

```bash
python -u scripts/training/train.py --run clip_standard
```

中断后只能显式恢复，程序不会覆盖既有 run：

```bash
python -u scripts/training/train.py \
  --run clip_standard \
  --resume outputs/training/clip_standard/checkpoints/resume/step_00001000.pt
```

每个 run 的 `checkpoints/resume/latest.json` 会指向最新可恢复的完整 checkpoint；运行中只保留最新两份完整 resume state。训练进度 1%、5%、20%、50% 另存 model-only trajectory snapshot，100% 的 final full checkpoint 同时是最终轨迹点和评测输入。轨迹表示评测在 branch 完成后批量运行，不在训练中改变模型状态。

一个 branch 完成后，使用服务器 Pilot 已确认的评测 batch size 导出全部轨迹 raw embedding，并计算相邻变化：

```bash
python -u scripts/evaluation/evaluate_trajectory.py \
  --run clip_standard \
  --batch-size 128 \
  --device cuda
```

评测器固定处理 COCO 5K 与 LCS 10K，不接受改变 Probe 的命令行参数。正式定义见 `configs/evaluation/TRAJECTORY_EVALUATION_PROTOCOL.md`。

全部 8 个分支顺序执行：

```bash
bash scripts/training/run_all.sh
```

## 已编码的实验控制

- 每个 run 都重新从锁定 SHA-256 的 M0 checkpoint 初始化；只有显式 `--resume` 才恢复训练状态。
- 每个模型的两个分支共享数据、顺序、augmentation、optimizer、scheduler、batch、budget 和 seed。
- 数据 augmentation 与模型随机数按 batch/step 派生，避免一个分支的额外 forward 改变下一个 batch 的共同随机路径。
- relation 循环以 optimizer step 为单位；梯度累积的所有 micro-batch 使用同一 relation。
- Standard 与 Mixed 每个 micro-batch 都有 `2N` 个 positive supervision terms、每个 query 有 `N` 个 candidates 和 `N-1` 个 negatives。
- VISTA 的 `IT` 只调用原生 `encode_mm`；CLIP/BEiT-3 使用归一化 `I/T` 后的 `normalize(I + T)`。
- ALBEF ITC-only 保留原生 momentum encoders、queue、temperature 和 soft targets；Full 直接使用原生 ITC+ITM+MLM。
- 第一版正式运行目标是单张 A800；不会静默切换到多卡或改变 negative pool。

## 本地测试

不需要 checkpoint 或 GPU 的测试命令：

```bash
python -m unittest discover -s tests -t . -p 'test_*.py' -v
```

真实模型的 checkpoint load、单步 forward/backward、峰值显存和 BF16 数值稳定性必须在 AutoDL A800 pilot 中完成。
