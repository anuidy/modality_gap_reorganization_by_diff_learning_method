# 正式训练运行手册

**状态更新（2026-09-28）：训练已暂停，已完成实验和下一轮决策以[当前实验状态](CURRENT_EXPERIMENT_STATUS.md)为准。下一轮恢复九分支并计划五轮，但新配置、81项矩阵及滚动保存尚未实施。下文中的旧轮数、FN取消、79项矩阵和“当前运行”均属于当时协议记录，不是新机器的自动执行计划。**

当前训练协议见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)，硬件运行方式见 [单/双4090说明](EXECUTION_MODES.md)。代码已接入九分支、raw sum、同一步Mixed分组、FN屏蔽、独立随机流、验证尾批和四舍五入保存点。正式seed为42/43/44；默认单任务与gate使用42，VISTA Fixed-2M使用42。

## 原1轮完整单seed入口（历史计划）

使用`configs/training/formal_single_seed.yaml`：seed=42，完整15003步，输出根为`outputs/training/formal_full_v1/`。它与已实测配置的训练控制相同，只切换输出位置及单seed选择。CLIP/BEiT-3/VISTA各九分支分三组组织，ALBEF模板仅为配置结构兼容而保留，不在27项主计划内。

下面只打印一个分支的计划，不启动训练：

```bash
python scripts/training/run_independent.py --config configs/training/formal_single_seed.yaml --runs clip_standard --seeds 42 --gpus 0
```

CLIP九分支计划示例：

```bash
python scripts/training/run_independent.py --config configs/training/formal_single_seed.yaml --seeds 42 --gpus 0 --runs \
  clip_standard clip_fixed_2m clip_fixed_3m_fn_off clip_fixed_3m_fn_on \
  clip_mixed_2m clip_mixed_3m_fn_off clip_mixed_3m_fn_on clip_full_3m_fn_off clip_full_3m_fn_on
```

BEiT-3和VISTA使用同组分支名称及相应模型前缀。完整三组只读计划保存在`outputs/training_queues/formal_full_seed42_20260922/plan.json`。本次整理不执行`--execute`；全程评测调度与B/C缺项仍见核对报告，不把准备好的训练计划当成全链路已完成。

## 通用和历史计划入口

以下`SEED_A/B/C`分别为已确认的42/43/44。默认命令只展示计划，不启动训练、不创建任务输出。

```bash
SEED_A=42
SEED_B=43
SEED_C=44
# 单卡：三个gate任务依次运行。
python scripts/training/run_independent.py --gate --seeds "$SEED_A" --gpus 0
# 双卡：相同gate任务由两张卡独立领取。
python scripts/training/run_independent.py --gate --seeds "$SEED_A" --gpus 0 1
# 三个主模型79个任务的计划，ALBEF不在该矩阵中。
python scripts/training/run_independent.py --matrix --seeds "$SEED_A" "$SEED_B" "$SEED_C" --gpus 0
# 显式选择分支与种子。
python scripts/training/run_independent.py --runs clip_mixed_3m_fn_off beit3_standard --seeds "$SEED_A" --gpus 0
```

获得开训指令并完成目标4090的短程检查后，增加`--execute`执行。入口先验证所有选中配置、数据和权重身份，再开始训练。默认GPU槽为0，双卡需明确指定0 1；不自动探测后改变实验定义。`run_all.sh`仅转发同样参数。

## 停止在20%与模型快照

当前`save_resume_checkpoints: false`，只保存模型轨迹。`--gate`保持完整训练计划，在3001步保存20%模型快照并执行验证后停止。新启动队列拒绝非空输出。

```bash
python scripts/training/run_independent.py \
  --runs clip_standard beit3_standard vista_standard \
  --seeds "$SEED_A" --gpus 0 --stop-after-step 3001 --execute
```

上例展示诊断停止方式，不用于本次完整正式训练。已完成的诊断为24项，其权重已退役，旧目录仍保留非权重记录，因此不应重新执行旧诊断队列。完整任务从M0开始，不能从已保存的20%模型精确恢复；完整状态恢复仅作为兼容能力保留。见[检查点协议](CHECKPOINT_PROTOCOL.md)。

## 输出与并发

当前正式目录为`outputs/training/formal_full_v1/<模板run>/seed_42/`。历史诊断记录留在`outputs/training/formal_v1/`。manifest中的run_id包含seed，跨轮比较同时核对配置与权重SHA。每任务保留独立日志、轨迹和来源元数据；不创建恢复指针。数据和M0只读共享。

同一仓库的调度器使用GPU槽锁，单任务CLI使用输出锁。一任务失败后停止派发新任务，已运行同伴允许完成；中断时回收已启动子进程。外部程序占卡及异常退出遗留锁需核对进程后处理，程序不删除未知锁。

## 验证

训练/验证batch均36；验证固定丢8条，实际7992条/222批。关闭梯度裁剪仍计算并记录梯度范数，非有限损失或梯度拒绝更新。日志记录六方向损失、查询计数、分组、候选池、logit_scale与随机流版本。

本地CPU回归：

```powershell
& 'D:/conda/envs/modality-gap/python.exe' -m unittest discover -s tests -t . -p 'test_*.py' -q
```

CPU测试覆盖目标数学与小模型更新；24项已通过当前4090的20%实跑，三个Full GCL FN-on暂无本轮GPU实测。训练代码不自动降低batch、改为梯度累积或开启DDP。当前A0–A6和MMEB Local已实现；自动评测队列仍针对20%诊断，不可直接接管完整epoch。B1、B3、Global及C尚未全部落地。

A/B/C范围与未完成项见 [ABC_IMPLEMENTATION_DESIGN.md](../evaluation/ABC_IMPLEMENTATION_DESIGN.md)。实际已验证环境记录在`configs/runtime/verified_autodl_4090_20260922.json`；该文件为环境清单，不是完整安装锁，不应按旧requirements说明重装当前已跑通的环境。
