# Formal Training Runtime Implementation Change Log（正式训练运行时实现改动日志）

## 目标

将已冻结的 8-run 实验矩阵落实为可在单张 AutoDL A800 上执行的训练代码，同时保留尚未冻结的数据路径和训练超参数，不在本地阶段擅自选择正式实验值。

## 新增内容

- `src/datasets/training_pairs.py`：paired training manifest、唯一 semantic instance 校验、确定性 epoch sampler。
- `src/objectives/contrastive.py`：Standard 与三组 Count-Matched Mixed 双向 relation。
- `src/training/config.py`：严格的 8-run 矩阵、共享控制与未冻结配置 fail-fast。
- `src/training/backends.py`：CLIP、VISTA、BEiT-3、ALBEF full-parameter training backend。
- `src/training/engine.py`：单卡 BF16、梯度累积、cosine/warmup、审计日志、checkpoint 和 resume。
- `scripts/training/train.py`、`scripts/training/run_all.sh`：单 run 与八 run 顺序入口。
- `configs/training/train_runs.yaml`、`TRAINING_RUNBOOK.md`：正式配置模板与服务器运行说明。
- `tests/datasets/`、`tests/objectives/`、`tests/training/`：不依赖真实 checkpoint/GPU 的单元与 dry-run 测试。

## 关键实现约束

- CLIP/VISTA/BEiT-3 的 Mixed relation 以 optimizer step 为周期：`I↔T → I↔IT → T↔IT`。
- 梯度累积期间不切换 relation；每个 micro-batch 保持一个 positive 和 `N-1` 个 negatives。
- VISTA 使用原生 `encode_mm` 产生 `IT`；CLIP/BEiT-3 不增加 projection。
- ALBEF ITC-only 保留原生 momentum encoder、queue、temperature 与 soft teacher target；Full 使用原生 `ITC+ITM+MLM`。
- ALBEF 官方单卡 gather 和 MLM mask 的现代 PyTorch/CUDA 兼容修正在项目侧运行时完成，不修改 ignored 的 third-party 源码。
- 所有 run 检查 M0 SHA-256；没有显式 `--resume` 时拒绝覆盖已有输出。
- augmentation 与模型 forward 使用由 batch/step 派生的随机种子，使同模型两个分支的共同路径可复现。

## 验证边界

CPU 单元测试和训练引擎 dry-run 已覆盖配置矩阵、relation 循环、count matching、数据顺序、梯度累积与 checkpoint。真实四模型的 BF16 forward/backward、峰值显存和 checkpoint resume 仍须在 AutoDL A800 pilot 中完成。
