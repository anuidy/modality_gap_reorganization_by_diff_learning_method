> 先验阶段知识参考。当前训练参数和执行方式以[主README](../README.md)及正式协议为准。文中的交接/修改日志仅保留服务器；历史假设和建议不代表新正式协议。原始模长不等时，不可套用等模长的平方根恒等式。
>
> 文中先验专用脚本与旧配置路径指归档内原路径，已退出当前执行目录；归档位置及参考方法见[主README](../README.md)。

# CLIP 训练侧 `model.float()` 事实核验（问答形式）

> **用途**：单行代码 `self.model = model.float().to(device)` 的设计来源、必要性与实际作用的自包含核验记录，供外部读者（无仓库访问权限）理解 `modality-gap-reorganization` 项目里 CLIP 训练与评测两侧的权重精度差异。
> **核验日期**：2026-09-14（UTC）。
> **核验约束**：不修改任何代码/配置/checkpoint/指标产物；不启动训练；不重跑轨迹评测。Q0–Q7 只做事实核验、不含改动建议；**Q8 为用户 2026-09-14 的正式决议记录**（保持现状，方案 A）及其约束。
> **未落盘中间产物**：所有 dtype 与数值均来自临时进程内的真实前向/反向（真实 batch、真实目标函数、一步 AdamW）。
> **配套文档**：`docs/Implementation Fact Verification（训练与评测实现事实核验）.md`（14 条 run 的整体实现核验，本文件是其第 2 项在 CLIP 上的深挖）。

---

## Q0. 前提更正：是 `.float()` 让权重变成 fp32 的吗？

**不是。** 提问的前提（"训练侧额外调用了 `.float()`，导致模型的 446 个 tensor 全部变成 fp32"）与实测不符：

| 调用 | 参数 dtype 实测 |
|---|---|
| `clip.load(ck, device="cpu")` | **446 fp32** |
| `clip.load(ck, device="cuda")` | **291 fp16 + 155 fp32** |

官方 `clip.load()` 的结尾（`clip/clip.py:139-142`）：

```python
model = build_model(state_dict or model.state_dict()).to(device)
if str(device) == "cpu":
    model.float()          # ← 全 fp32 来自这里
return model, _transform(model.visual.input_resolution)
```

而训练 backend 从第一版起就是 `clip.load(str(checkpoint), device="cpu", jit=False)`。

**因此：446 个张量全为 fp32 是"用 CPU 加载"带来的（官方 CPU 分支），不是项目那行 `.float()` 造成的。该项目的那行 `.float()` 在 CPU 加载路径下是冗余 no-op**——实测见 Q4 的变体 B。

---

## Q1. `model.float()` 这行代码从哪来？是官方要求、第三方习惯，还是本项目自行加入？

**答：能定位到引入者与时间，但原因无法从仓库历史中确认。**

- 文件 `src/training/backends.py` 是在 commit **`2645a12`** 中**整体创建**的（diff 头为 `@@ -0,0 +1,599 @@`）。
- 该 commit：message = `Add formal training and dataset preparation pipeline`，时间 = **2026-08-31 23:12:41 +0800**，作者 = `Yangon 刘 <1160770364@qq.com>`（项目本人）。
- `git log --all -S "model.float()" -- src/training/backends.py` **只命中这一个 commit** → 该行自加入以来从未被修改（HEAD 在第 203 行，工作区第 224 行，内容逐字相同）。
- **从第一版起它就紧挨着 `device="cpu"`**，diff 中相邻两行为：
  ```python
  model, _ = clip.load(str(checkpoint), device="cpu", jit=False)
  self.model = model.float().to(device)
  ```
- **无任何说明**：commit message、邻近注释、`change_logs/**`、`configs/**` 中都没有解释（对 `\.float()|fp16|float16` 的全量检索只命中两处无关内容，讲的是目标函数里的 `F.normalize(x.float())`）。
- **不是从本仓库内的参考实现抄的**：仓库里没有 vendored 的 OpenAI-CLIP 训练代码；`convert_weights` 的命中都在其它第三方树（`third_party/unilm/beit2/vqkd_teacher/clip`、`third_party/unilm/kosmos-2/open_clip`、`third_party/flag_embedding/.../eva_clip`）。

> **historical reason: not recoverable**（可定位到引入者与时间，但仓库历史中没有记录动机）。

---

## Q2. OpenAI CLIP 官方实现是否要求训练前调用 `.float()`？

**答：官方加载路径不要求；官方安装包不含训练代码；仓库内确实有一处第三方参考实现这么做，且给出了明确理由——但该理由在本项目不成立。**

### （1）所安装的包是官方实现

`clip-1.0.dist-info/direct_url.json` = `https://github.com/openai/CLIP.git` @ commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`（版本 1.0）。

**它包含什么**：只有 `clip/__init__.py`、`clip/clip.py`、`clip/model.py`——**不含训练代码**（官方 repo 的 `src/training/main.py` 不在 wheel 内）。因此"官方训练路径是否要求"**无法从安装包判断**，本仓库也没有 vendored 的官方训练代码。

### （2）`clip.load()` 的完整行为

- 若文件是 TorchScript archive（本项目的 `ViT-L-14.pt` 就是），先 `torch.jit.load(..., map_location=device if jit else "cpu")`；`jit=False` → `map_location="cpu"`。
- 然后（`jit=False` 分支）：
  ```python
  model = build_model(state_dict or model.state_dict()).to(device)
  if str(device) == "cpu":
      model.float()
  ```
- `build_model`（`clip/model.py:399-438`）结尾必定执行：
  ```python
  convert_weights(model)        # Linear/Conv/MultiheadAttention + 名为 proj/text_projection 的属性 → fp16
  model.load_state_dict(state_dict)
  return model.eval()
  ```
- `convert_weights`（`model.py:375-396`）**不碰 LayerNorm 与 Embedding** → 得到 **291 fp16（Linear/Conv/MHA/proj）+ 155 fp32（LayerNorm/Embedding/`logit_scale` 等）**。实测：`conv1 = fp16`、`ln_pre = fp32`、`logit_scale = fp32`。

**结论**：CUDA 上加载得到 **fp16/fp32 混合**权重；CPU 上加载得到**全 fp32**。官方 `load()` 里唯一的 `.float()` 就是那个 **CPU 分支**（另一处在 `jit=True` 的 TorchScript 图改写路径 `clip.py:200`，与本项目无关）。

### （3）仓库内确实存在"训练前 `.float()`"的参考实现，但本项目未使用它

`third_party/unilm/kosmos-2/open_clip/src/open_clip/factory.py:80-85`：

```python
model = load_openai_model(model_name, device=device, jit=jit)
# See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
if precision == "amp" or precision == "fp32":
    model = model.float()
```

注释直接给出出处：**AMP 的 GradScaler 拒绝 unscale fp16 梯度**。

但：**本项目对 `open_clip` 零引用**（`src/`、`scripts/` 全库检索无命中），且**本项目不使用 GradScaler**——全库无 `GradScaler`，只有 `torch.autocast(bf16)` 或 `nullcontext`（`src/training/engine.py:85-92`）。**因此这条理由在本项目不成立。**

---

## Q3. `.float()` 在当前训练框架中的实际作用是什么？参数/梯度/优化器状态分别是什么 dtype？forward 里哪些算子在 bf16 下执行？

**答（全部为实测值，非 PyTorch 常识推断）**：

数据流：
```
fp16 checkpoint
→ clip.load(device="cpu")   → 已经全 fp32（官方 CPU 分支）
→ .float()                  → no-op
→ .to(cuda)                 → fp32 parameters
→ bf16 autocast forward     → 部分算子 bf16
→ backward                  → fp32 gradients
→ optimizer.step()          → fp32 optimizer state
```

实测（真实 batch、真实目标函数、一步 AdamW）：

| 对象 | dtype |
|---|---|
| parameters | **446 fp32** |
| gradients | **446 fp32** |
| AdamW `exp_avg` / `exp_avg_sq` | **fp32** |
| `logit_scale` | **fp32**，值 99.9988（clamp 上限 100） |

bf16 autocast 下各算子输入/输出 dtype（forward hook 实测）：

```
visual.conv1                             in=fp32  → out=bf16
visual.transformer.resblocks.0.attn      in=bf16  → out=(tuple)
visual.transformer.resblocks.0.mlp.c_fc  in=bf16  → out=bf16
visual.ln_pre                            in=bf16  → out=bf16
ln_final                                 in=fp32  → out=fp32
```

**结论**：
1. `.float()` 的含义是"**master parameters 为 fp32**"，**不等于**"模型完全用 fp32 计算"。
2. 矩阵乘/卷积类在 **bf16** 执行；LayerNorm 类在 **fp32** 执行（`ln_final` 上下游都是 fp32）。
3. **但这份 fp32 master 状态在本项目中并非 `.float()` 带来**（见 Q0/Q4）。

---

## Q4. 如果去掉 `.float()`，当前代码还能正常训练吗？

**答：分两种改法，结论完全不同。**

同一 batch（32 样本）、各做一次 forward + backward + `clip_grad_norm_(1.0)` + `AdamW.step()`；**未保存任何 checkpoint、未修改正式代码、未启动完整训练**：

| 变体 | 参数 dtype | 梯度 dtype | 优化器状态 | loss | 裁剪前 grad_norm |
|---|---|---|---|---|---|
| **A** 现状（`device="cpu"` + `.float()`） | 446 fp32 | 446 fp32 | fp32 | 0.433982 | 70.61 / 70.54（重复两次） |
| **B** 只删 `.float()`（仍 `device="cpu"`） | **446 fp32** | 446 fp32 | fp32 | **0.433982**（逐位相同） | 70.65 |
| **C** 真保留 fp16 权重（改 `device="cuda"`） | **291 fp16 + 155 fp32** | **跟随参数**：291 fp16 + 155 fp32 | **fp16 参数 → fp16 `exp_avg`**；fp32 参数 → fp32 | 0.434283（有限） | 46.76 |

**变体 B = 现状**：dtype 全同、loss 逐位相同；grad_norm 的 0.1% 差异与 A 自身的重复波动（70.61 vs 70.54）同量级。→ **只删 `.float()` 在当前代码下什么都不变。**

**变体 C（保留 fp16 权重）的逐项风险，全部实测**：

| 检查项 | 实测结果 |
|---|---|
| dtype mismatch / bf16 输入 × fp16 权重 | **不报错**：autocast 把两者都转到 bf16（`conv1` in=fp16 → out=bf16） |
| LayerNorm / softmax dtype | **仍走 fp32**（`ln_final` in/out 均 fp32），无问题 |
| 优化器对 fp16 参数的更新 | **能 step**，但 `exp_avg`/`exp_avg_sq` 跟随参数变为 **fp16**（二阶动量在 fp16 累积；默认 `eps=1e-8` 低于 fp16 最小次正规数 6e-8） |
| 梯度下溢 | **可测**：fp16 参数梯度中精确为 0 的元素占 **0.02%–0.05%**（conv1 0.04%、proj 0.02%、`in_proj_weight` 0.05%）；fp32 参数为 **0.0000%** |
| `logit_scale` dtype 与 clamp | **仍是 fp32**（`convert_weights` 不碰它），clamp 正常 |
| 梯度裁剪兼容性 | **兼容**，`clip_grad_norm_` 在混合 dtype 参数上正常返回（46.76）；裁剪前范数 > 1.0 时尺度被归一化 |

**一个容易误判的点**：OpenCLIP 注释所指的报错（`attempting to unscale FP16 gradients`）来自 **GradScaler**，而本项目没有 GradScaler。因此变体 C **不会报错，只会静默降低精度**。

---

## Q5. `.float()` 是不是为了配合 bf16 训练（FP32 master + BF16 autocast）？

**答：形式上是这套协议；但没有历史证据表明当初是这个目的；而且"必要性"这一半已被实测否掉。**

- **代码事实**：`.float()` 使 master weights 为 fp32，bf16 autocast 只影响算子执行精度 → 二者组合**在形式上就是** "FP32 master weights + BF16 autocast compute"。
- **无历史证据**：commit message 未提、无注释、无 change log。
- **不必要**：在本项目的实际调用方式（`clip.load(device="cpu")`）下，删掉它结果逐位相同（Q4 变体 B）。
- **若改为保留 fp16 权重**（变体 C），则确实进入另一套协议：**FP16 parameters + BF16 autocast**（优化器状态也随参数降到 fp16）。这是真实存在的技术差异，但**没有证据**说明作者当初是为了避免它才写这行。

> **likely technical rationale, not historically confirmed.**

---

## Q6. 训练完的 checkpoint 被保存成什么 dtype？评测时为什么又会变成 fp16？

**答：checkpoint 是 fp32；评测时被降成 fp16——这条链在 CLIP 上成立。**

| 步骤 | 实测 dtype |
|---|---|
| 训练时模型 | 446 fp32 |
| 保存的 trajectory checkpoint `model.*` | **446 fp32** |
| 评测适配器构造（`clip.load(M0, device="cuda")`） | **291 fp16 + 155 fp32** |
| `load_snapshot_into_adapter`（`src/evaluation/trajectory.py:225`，普通 `load_state_dict(strict=True)`） | 加载后**仍是 291 fp16 + 155 fp32** |
| 权重值变化 | conv1：checkpoint fp32 → 适配器内 fp16，**max\|Δ\| = 3.05e-5，相对 2.48e-4**（fp16 舍入量级） |

五个轨迹点 × 四条 CLIP run 全量检查：

```
clip_standard       p001/p005/p020/p050/p100 → 全部 {float32: 446}
clip_gcl2_fixed     p001/p005/p020/p050/p100 → 全部 {float32: 446}
clip_gcl2_rotating  p001/p005/p020/p050/p100 → 全部 {float32: 446}
clip_full_gcl       p001/p005/p020/p050/p100 → 全部 {float32: 446}
```

**机制不是"直接读 state_dict"，而是"先造 fp16 模型，再灌 fp32 值"**：

1. `create_m0_adapter` 用官方 `clip.load(M0, device="cuda")` 得到 **fp16/fp32 混合**权重的模型；
2. `load_snapshot_into_adapter` 用**普通 `load_state_dict(strict=True)`** 把 checkpoint 的 fp32 值 `copy_` 进这些参数 → PyTorch 按**目标 dtype** 转换 → **fp32 被降为 fp16**；
3. `strict=True` 只校验键名，**不校验 dtype**，所以不会报错。

### ⬛ 特别核验项：训练后的 fp32 checkpoint 为什么到了 evaluation path 又变成 fp16？

**成立，且仅限 CLIP。** 影响量化（同一 checkpoint、同一输入、同为 fp16 autocast，只改权重精度）：

| 分支 | rel | cos | max\|Δ\| |
|---|---|---|---|
| image | **0.000e+00** | 1.0000000 | **0.000e+00** |
| text | **8.42e-04** | 0.9999996 | 3.9e-03 |

**为什么只有 text 分支有差异**（精确机制，非推测）：CLIP 的 `encode_text` 第一行就把 token embedding 转成 `self.dtype`，而

```python
@property
def dtype(self):
    return self.visual.conv1.weight.dtype
```

适配器中 `conv1` 是 **fp16** → 文本张量在第一时间被降到 fp16 并在该精度下继续前向；全 fp32 模型中它保持 fp32，直到第一个算子才被 autocast 转换。视觉分支两者一致（输入都在算子处被转 fp16，转换点相同）。**即：权重精度差异通过"文本分支的 dtype 传播"显形，而不是通过算子级差异。**

**另外三个模型没有这个问题**：加载 p100 前后适配器 dtype 不变——`beit3` 494 fp32、`vista` 462 fp32、`albef` 837 fp32。所以"训练后 checkpoint 被重新以 fp16 评测"是 **CLIP 独有**，不是评测流程的普遍行为。

---

## Q7. 三层结论

### A. 已确认的历史原因

**historical reason: not recoverable.**
可定位到引入者与时间（commit `2645a12`，2026-08-31，作者本人，文件整体新建），但 commit message、注释、change log、设计文档中均无说明。仓库内唯一同款写法位于**未被本项目使用**的 OpenCLIP 中，其注释指向 AMP 不能 unscale fp16 梯度的报错。

### B. 已确认的技术作用

- 当前实际使参数、梯度、AdamW 状态全部为 **fp32**；bf16 autocast 只让矩阵乘/卷积类在 bf16 执行，LayerNorm 类在 fp32 执行，`logit_scale` 保持 fp32。
- **但这不是 `.float()` 的功劳**：在本项目的调用方式（`clip.load(device="cpu")`）下，删掉它**逐位无变化**（参数 dtype 相同，loss 0.433982 逐位相同）。真正决定全 fp32 的是官方 `load()` 的 CPU 分支。
- 若改为保留 fp16 权重（`device="cuda"`），则进入 **FP16 parameters + BF16 autocast** 协议：优化器状态随参数降为 fp16，梯度中 0.02–0.05% 元素下溢为 0；不报错，但精度静默下降。

### C. 是否属于官方要求

**officially used in the inspected reference path but not strictly required.**

- 官方 `clip.load()` 只在 **CPU 分支**调用 `.float()`（`clip/clip.py:141`）；CUDA 分支不调用，且 `build_model` 必定先 `convert_weights` 成混合 fp16。
- 官方安装包**不含训练代码**，无法从官方训练脚本得出该要求；仓库内也没有 vendored 的官方训练代码。
- 仓库内**确实**在训练前 `.float()` 的参考实现是 OpenCLIP（`third_party/unilm/kosmos-2/open_clip/src/open_clip/factory.py:84`），且给出明确理由（AMP / GradScaler），但**本项目不使用 GradScaler**，该理由不成立。

---

## Q8. 决议（用户 2026-09-14）：**保持现状（方案 A），不修改**

围绕"评测侧为什么是 fp16 / 有没有必要 / 能不能改成 fp32"的结论与决定，完整记录如下。

### Q8.1 为什么评测侧保留 fp16

**不是设计选择，是继承了官方 loader 的默认行为。**

- 评测侧 `src/model_adapters/clip_openai.py:43` 传 `device="cuda"`。官方 `clip.load()` 里 `build_model()` **必定**调用 `convert_weights`（Linear/Conv/MultiheadAttention/proj → fp16），而唯一的 `.float()` 位于 **CPU 分支**（`clip/clip.py:140-141`）→ CUDA 路径跳过它 → 291 fp16 + 155 fp32。
- 训练侧 `src/training/backends.py:223-224` 传 `device="cpu"` 再 `.to(cuda)` → 命中 CPU 分支 → 446 fp32。
- 即：**两侧精度都是被 `clip.load` 的 device 参数间接决定的，没有一处是显式的精度选择**（这也是训练侧那行 `.float()` 成为 no-op 的原因，见 Q0/Q4）。

### Q8.2 有必要吗——没有

- **契约不需要**：`EmbeddingAdapter` 抽象方法写明 "Return CPU **float32** raw embeddings"（`src/model_adapters/base.py:14-19`），各适配器出口即 `.detach().float().cpu()`。
- **下游不需要**：存盘即 float32；指标层全程 numpy，协方差/特征值还显式升到 float64（`_covariance` 的 `astype(np.float64)`）。
- **性能上也不必要**：评测实测峰值仅 2.42 GiB、单条 run 约 13 分钟，fp16 的速度收益不是瓶颈。
- 结论：fp16 是**性能默认值**，不是功能需求。

### Q8.3 能改成 fp32 吗——能，但是两件独立的事

| | 现状 | 改法 |
|---|---|---|
| 权重 dtype | 评测侧 291 fp16 + 155 fp32 | 与训练侧一致地加载（`clip.load(ck, device="cpu")` 再 `.to(device)`，或加载后 `.model.float()`） |
| compute 精度 | 适配器固定 `torch.autocast(cuda, fp16)`（`clip_openai.py:48-51`） | 改 `nullcontext`（纯 fp32）或 **bf16**（与训练一致） |

**只改权重不能消除全部差异**：生产设置下 CLIP 的相对差 1.65e-2 中，权重只占约 3.4e-3，其余来自 fp16-vs-bf16 autocast。BEiT-3 的 9.05e-3 **全部**来自 autocast（它权重本来就是 fp32）。

### Q8.4 决定保持现状的依据

**（1）导出产物内部自洽。** 用"存盘值往返 fp16 是否精确"做指纹检验（取前 2000 行）：

| 产物 | 可被 fp16 精确表示的比例 |
|---|---|
| CLIP M0（更早在 Windows 上导出） | **100.0000%** |
| CLIP 轨迹 p100（本机导出） | **100.0000%** |
| BEiT-3 M0 | **100.0000%**（来自适配器的 fp16 autocast，不是权重） |
| VISTA M0 | 0.0139%（最后一步不是 Linear） |

→ ① 所有 CLIP 导出向量带同一种 fp16 印记，**跨机器（Windows / Linux）、跨时间一致**；② 本项目主分析是**同模型 Δ**（M0 → 各分支），两侧同源，该印记在一阶上相消。

**（2）精度差远小于效应量。** 同输入相对差 3.4e-3（cos 0.999992），而待测效应为 centroid gap 相差 3 倍、effective rank 225→20、anisotropy 0.28→0.92 —— 相差 2–3 个数量级。

**（3）改动成本落在重导而非改码。** 只改代码不重导，会让新向量与既有产物差 ~1e-3 相对量级，破坏自洽；要恢复自洽必须**整批重导 CLIP**：M0 两个 probe + 4 条 run × 5 个轨迹点 × 2 个 probe（约 1 小时 GPU）并重算指标。BEiT-3 / VISTA / ALBEF 无需重导（分析限于同模型内）。

### Q8.5 决议与其约束

> **保持现状（方案 A）。** CLIP 评测侧继续使用官方 `clip.load(device="cuda")` 得到的 fp16/fp32 混合权重与 fp16 autocast；不改代码、不重导、不重算指标。**该精度状态作为已知条件写入所有分析报告**，不得被当作缺陷或误差来源反复重提。
>
> **后续约束**：若将来因为其他原因必须改（例如要与其他人的 fp32 结果对齐），必须**同时**重导 CLIP 的全部导出产物以维持内部自洽——不允许只改代码或只重导一部分。

---

## 附录 A：复现方式

全部结论可在临时进程内复现，不需要训练、不需要写盘：

| 结论 | 复现方式 |
|---|---|
| Q0 / Q2 的 dtype | `clip.load(ck, device="cpu")` 与 `clip.load(ck, device="cuda")`，统计 `state_dict()` 的 dtype |
| Q3 的算子 dtype | 在真实 `ClipTrainingBackend` 上注册 forward hook，打印目标模块的输入/输出 dtype |
| Q4 的三个变体 | 构造 `ClipTrainingBackend` → 按变体替换 `backend.model` → `prepare_batch` → `forward` → `backward` → `clip_grad_norm_` → `AdamW.step`（**不调用任何 checkpoint 保存路径**） |
| Q6 的链条 | `create_m0_adapter("clip", …)` → `load_completed_run_manifest` + `resolve_trajectory_snapshots` + `load_snapshot_into_adapter` → 统计 dtype 并比较权重值 |

## 附录 B：本次核验涉及的关键位置

```
clip/clip.py:94-145          官方 load()（CPU 分支的 .float() 在 :141）
clip/clip.py:200             jit=True 分支的 .float()（本项目不用）
clip/model.py:375-396        convert_weights（不碰 LayerNorm / Embedding）
clip/model.py:399-438        build_model（结尾必调 convert_weights）
clip/model.py 的 dtype property → self.visual.conv1.weight.dtype（Q6 的机制来源）
src/training/backends.py:223-224   clip.load(device="cpu") + model.float().to(device)
src/training/engine.py:85-92       _autocast（只有 bf16 / fp32，无 GradScaler）
src/evaluation/trajectory.py:194-226  load_snapshot_into_adapter（:225 普通 load_state_dict）
src/model_adapters/clip_openai.py:43  clip.load(..., device=cuda, jit=False)（fp16 混合权重）
third_party/unilm/kosmos-2/open_clip/src/open_clip/factory.py:80-85  参考实现的 .float() + 理由
```
