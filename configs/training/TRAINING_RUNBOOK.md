# 正式训练运行手册

本目录现在包含正式 8-run 矩阵的可执行训练配置。代码不会把 COCO 5K 或 LCS 10K probe 当成训练集，也不会替研究者猜测尚未冻结的超参数。

## 代码位置

- `src/datasets/training_pairs.py`：训练 manifest、唯一 semantic instance 校验和确定性采样。
- `src/objectives/contrastive.py`：Standard 与 Count-Matched Mixed 的双向对比目标。
- `src/training/backends.py`：CLIP、VISTA、BEiT-3、ALBEF 的可训练后端。
- `src/training/config.py`：8-run 矩阵和控制变量校验。
- `src/training/engine.py`：单卡 A800 的 BF16、梯度累积、日志、断点与 checkpoint。
- `scripts/training/train.py`：单个 run 的命令入口。
- `scripts/training/run_all.sh`：配置完成后依次校验并运行 8 个分支。

## 训练数据 manifest

训练数据使用 JSON 或 JSONL。每条记录必须包含：

```json
{"sample_id":"pair-000001","semantic_id":"instance-000001","image":"subdir/000001.jpg","text":"caption"}
```

`image` 必须是相对 `image_root` 的路径。`sample_id` 和 `semantic_id` 必须全局唯一。这个限制保证每个 query 恰好只有一个 positive，其余 `N-1` 个 candidate 都是真 negative。完整示例见 `data/metadata/training_manifest_example.jsonl`。

## 先冻结配置

编辑 `configs/training/train_runs.yaml` 中的 `null` 字段：

- seed；
- train manifest 与 image root；
- optimizer、learning rate、weight decay；
- scheduler、warmup、minimum LR ratio；
- 每个模型的 micro-batch 与 gradient accumulation；
- max steps 与 checkpoint interval；
- 每个模型的 augmentation。

顶层 `controls` 是四个模型共同的默认值。若不同模型需要不同优化设置，可在对应 `models.<model>` 下增加 `optimizer`、`scheduler` 或 `budget`；同一个模型的两个分支仍从同一个 model block 读取，因此不能在 run 层改变控制变量。

推荐先只确定 pilot 配置，跑通每个模型的两个分支各 2–10 个 optimizer steps；pilot 验证 checkpoint load、显存、loss、relation audit 和 resume 后，再冻结正式 budget。

## AutoDL 执行顺序

安装 CUDA 版 PyTorch 与依赖后，先逐项校验：

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
  --resume outputs/training/clip_standard/checkpoints/step_00001000.pt
```

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
