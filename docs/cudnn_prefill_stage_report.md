# prefill-only cuDNN attention 分支 —— 阶段报告（2026-09-26）

> 结论先行（细节与原始数据在下方；**未测量的东西一律不写成结论**）
>
> 1. **分支已实现并已按任务验证顺序跑完第 1–3 步**：新增显式策略 `eager` / `cudnn_prefill`
>    （默认 `eager`，历史行为不变），融合**只**允许在**显式标注的 `prefill` 阶段**发生。
>    阶段分流在 **CPU 43/43 全绿**，并在 GPU 上用**真实调用序列精确复核**
>    （`fused=52/52`、`ctx=1025` 短块回退 `52/52`、非 prefill 的 776 次 eager 被单独归类）。
> 2. **算子级显存结论成立**：8K 分块 prefill 的**第一块**（target `s=2048,T=8192`）
>    瞬时显存 **1984.0 MiB → 54.0 MiB**；draft 为 **2284.0 MiB → 51.5 MiB**；
>    profiler 按 kernel 名证明确实走到了 `aten::_scaled_dot_product_cudnn_attention`。
> 3. **正确性判据未通过（这是本阶段最重要的结论）**：算子级 12 行里 8 行 FAIL；
>    **模型级 10/10 行全部 FAIL**，`ctx=512` 时**生成序列真的变了**（首个分歧在 index 3）。
>    自对照（eager vs eager）在 5/5 组里 `abs = 0.0` ⇒ 差异**全部**来自融合路径，不是抖动。
> 4. **机制已定位**：eager 分支在 softmax **之前**把分数矩阵舍入成 bf16
>    （`model_self.py:357` 的 `matmul` 输入/输出都是 bf16），而 cuDNN 内部全程 fp32。
>    实测：eager 侧误差随分数幅度**线性增长**（1.3e-2 → 4.9e-1）且 `|eager−fused| ≈ |eager−fp32|`，
>    fused 侧恒定在 ~1e-2 ⇒ **差异来源是 eager 基线的 bf16 分数舍入**，融合侧在尖峰区准 **15–40×**。
>    同时排除了 GQA 假说（`enable_gqa` 与手工 `kv_repeat` 逐位相同，MHA 同样复现）。
> 5. **因此不能说"可以直接替换"**：预登记判据 FAIL、生成序列发生变化
>    ⇒ **注意力后端不是行为中性的**，换后端 = 换模型行为，任何跨后端 A/B 都自带这个混淆项。
> 6. **不要**把本文任何数字说成"8K 已解决"、"正式方案已切换"或"SpecDec 已加速"——**均未验证**。

---

## 1. 交付物

| 类别 | 路径 | 状态 |
| --- | --- | --- |
| 引擎策略分支（**唯一被改的既有文件**） | `specdec/engine/model_self.py`（+211 / −13） | **已实现，CPU 已验证；GPU 算子级已验证** |
| CPU 自测（阶段分流 / 掩码等价 / 短块策略 / 防静默失效） | `specdec/engine/attn_phase_selftest_cpu.py` | **新增，43/43 PASS** |
| GPU 验证阶梯（算子级 + 模型级） | `specdec/engine/g5_attn_cudnn_validate.py` | **新增**；算子级 + 模型级均已跑完 |
| 精度取证（差异来源定位） | `specdec/engine/g5_attn_precision_forensics.py` | **新增**；H1 成立、H2/H3/H4 被排除 |
| 原始日志 | `output/g5/attn/attn_phase_selftest_cpu_20260926T174715Z.log` | 保留 |
| 算子级机器可读结果 | `output/g5/attn/attn_cudnn_validate_20260926T175207Z.json` | 保留 |
| 模型级结果 | `output/g5/attn/attn_cudnn_validate_20260926T175638Z.json` + `acv_model_20260926T175637Z.log` | 保留 |
| 取证日志 | `output/g5/attn/forensics_20260926T180128Z.log` | 保留 |

**没有改动**：`spec_dec.py`、`spec_cache.py`（契约不变）、logits / chunk 语义 / `k` / 模型 / dtype /
分配器 / cache 布局；`model_self.py` 的 eager 分支**逐字保留**（只是被 `if path == "fused"` 分了出来）。

### 1.1 为什么必须动 `model_self.py`（它原本是冻结的）

这是本项目唯一一次"冻结文件被改"，理由是任务要求本身就是改它；改动方式是**纯增量**：
原 eager 代码块内容不变，只在其前面加了一个裁决点。改动清单：

| 位置 | 内容 |
| --- | --- |
| `model_self.py:101-109` | 策略常量：`ATTENTION_STRATEGIES=("eager","cudnn_prefill")`、`PREFILL_FUSED_MIN_S=2`、`SHORT_BLOCK_POLICY="eager_fallback"`、`FUSED_BACKEND_NAME="CUDNN_ATTENTION"`、默认 `ATTENTION_STRATEGY="eager"` |
| `:123-179` | `set_attention_strategy` / `attention_strategy` / `attention_phase`（contextmanager） / `current_attention_phase` / `reset_attention_stats` / `attention_stats` |
| `:181-194` | **`decide_attention_path(phase, s, strategy)`** —— 纯函数，唯一裁决点 |
| `:196-205` | `_kv_keep_mask(s,T)` = `~triu(ones, diagonal=(T-s)+1)`（bool keep 掩码；SDPA 的 `True` 表示**参与**，与 `masked_fill` 相反） |
| `:207-214` | `_sdpa(...)` —— **一层间接**，只为让测试能 spy 到真实 kwargs，不污染全局 `torch.nn.functional` |
| `:216-229` | `_fused_prefill_attention` —— 强制 `A.sdpa_kernel([A.SDPBackend.CUDNN_ATTENTION])`，传 `attn_mask=keep, dropout_p=0.0, scale=head_dim**-0.5, enable_gqa=(q.shape[1]!=k.shape[1])`，**无 try/except** |
| `:232-255` | `_record_attention_call` / `assert_strategy_used()` |
| `:265-...` | `chunked_prefill(model, ids, chunk)` —— 分块 prefill 的规范实现，内部把阶段标为 `prefill` |
| `SelfAttention.forward` | 算出 `T` → `decide_attention_path` → 记录统计 → 分叉（`fused` 调融合；否则原 eager 代码块） |

## 2. 设计要点（为什么这样而不是那样）

1. **不能用 `q_len > 1` 判定 prefill。** `spec_dec._verify` 的候选验证也是多 token
   （`s = kp ∈ [1..k]`，`spec_dec.py:37-42`），用长度判会把**验证**误判成 prefill。
   → 只能用调用方**显式标注**阶段。（实测：`verify` 的 `s_max=4`，仍必须全 eager。）
2. **`cache is None` 也不能判定 prefill。** 它只标识**第一块**：`chunked_prefill` 的第 2..n 块
   `cache is not None`，但仍然是 prefill。
3. **`s == 1` 必须回退。** cuDNN SDPA 不支持 `seq_len == 1`（台账 §4.17.x 实测）。
   策略预先写死为 `eager_fallback`，并**记录原因**（`fallbacks` 列表，非静默）。
4. **`assert_strategy_used()` 是对"忘了标注阶段"的报警器**：
   策略已切到 `cudnn_prefill` 却一次都没融合 ⇒ 直接抛错，而不是悄悄什么都没发生。
   **这个坑真实存在**：CPU 自测首跑就踩到了 —— `spec_dec._prefill` 是**直接**调 `model._run` 的
   （`spec_dec.py:17`），**不经过** `chunked_prefill`，所以不被标注就跑在 `other` 阶段 ⇒ `fused=0`。
   正确做法是**由调用方（harness）在这一层标注**，`spec_dec.py` **一行都不用改**。
5. **不吞异常**：融合路径的 OOM / "无可用 kernel" 一律向上抛。实测印证：
   在 **CPU** 上误标 `prefill`（策略为 `cudnn_prefill`）会立刻
   `RuntimeError: No viable backend for scaled_dot_product_attention was found`
   —— 没有静默回退成 math/eager。

## 3. 环境与版本（可复现性）

| 项 | 值 |
| --- | --- |
| git | `HEAD = 7d29776`（= `origin/main`，改动前工作树干净） |
| 平台 | Windows / RTX 4060 Laptop **8 GB**（sm_89）/ i7-12700H / 32 GB RAM |
| python / torch | 3.11.9 / **2.6.0+cu124**（cuda 12.4，**cudnn 90100**） |
| 环境变量 | `HF_HUB_OFFLINE=1`、`PYTHONUTF8=1`、`PYTHONIOENCODING=utf-8`；`PYTORCH_CUDA_ALLOC_CONF=None`（**未设置**） |
| torch 后端开关 | `flash_sdp_enabled=True`、`mem_efficient_sdp_enabled=True`、`math_sdp_enabled=True`、`cudnn_sdp_enabled=True` —— 注意：**开关是 True 不等于运行时可选中**，实测只有 CUDNN 真的可用（§4.17.x） |
| 环境资格（算子级当次） | `commit 可用 16.1 GiB ≥ 16`；`P-state qualified=True, sm=2460 MHz, mem=8000 MHz` |
| GPU 上的其它消费者（算子级当次） | **24 个** C+G 进程（含 Edge WebView / 网易云 / Teams / WeChat / 百度网盘 / 一个游戏安装器……）⇒ **所有显存数字都必须带这个前提读** |

## 4. 验证顺序第 1 步：CPU 自测（**43 PASS / 0 FAIL**）

命令：`.venv\Scripts\python.exe -m specdec.engine.attn_phase_selftest_cpu`
日志：`output/g5/attn/attn_phase_selftest_cpu_20260926T174715Z.log`（UTF-8 原文）

| 组 | 覆盖 | 结果 |
| --- | --- | --- |
| ① 掩码等价 | `_kv_keep_mask` vs eager `masked_fill` 逐元素全等：`(s,T) ∈ {(8,8),(5,32),(1,32),(1,8192),(2048,8192),(3,7)}`；`s==1` 全允许；方阵 == 标准因果 | 8/8 PASS |
| ② 分流裁决 | 策略 `eager` ⇒ 恒 eager；`cudnn + prefill + s≥2` ⇒ fused；`verify/decode/propose/repair/other` × `s∈{1,2,4,2048}` ⇒ **恒 eager**（含多 token）；`prefill + s==1` ⇒ eager 且**带原因** | 8/8 PASS |
| ③ 真实调用序列（tiny 模型 + stub 融合） | `fused_by_phase == {'prefill': 4}`（target+draft 各 2 层）；每个 fused 的 `phase=='prefill'`；送进 kernel 的 `s=20 ≥ 2`；`propose` 的 `s_max=1`、`verify` 的 `s_max=4`（**多 token 确实出现了**）；`propose/verify` 的 `fused=0` | 10/10 PASS |
| ③b 逐阶段确定性 | 候选验证长度 **kp=1/2/3/4** 四轮：`verify` 的 `s_max==kp` 且 `fused=0`；`repair`(s=1)/`decode`(s=1)/**未标注**(s=1) 全 eager；每轮全场 `fused=0` | 8/8 PASS |
| ④ 传给 kernel 的 kwargs | 无 `is_causal`（非方阵不等价，必须用 `attn_mask`）；bool `attn_mask` 形状 `(s,T)=(2048,8192)`；`enable_gqa=True`；k/v **保持 kv_heads=2 未展开**（q=12）；`scale=head_dim**-0.5` | 5/5 PASS |
| ⑤ 防静默失效 / 短块 / 默认 | 策略已切但零 fused ⇒ **抛错**；默认 `eager` 时融合函数**一次都没被调用**（`fused=0 eager=24`，历史行为不变）；`chunk=8`、9 token ⇒ 两块 `8+1`，末块回退且 `fallbacks` 记录含 `seq_len=1` 与 policy/形状 | 4/4 PASS |

**③ 的一条诚实附注**：这次 tiny 随机模型对 **两轮全接受**（`mismatch_rounds=0`），所以真实序列里
`repair` **未被触发**（`eager=0 fused=0`）——该行因此是**条件断言**（"若触发则必全 eager"）。
`repair` 的确定性覆盖由 **③b** 提供（直接调 `_step`）。这是"没观察到"与"不会发生"的区别，
不写成后者。

## 5. 验证顺序第 2 步：算子级 GPU 检查

工具：`specdec/engine/g5_attn_cudnn_validate.py`，`ACV_STAGE=ops`
结果：`output/g5/attn/attn_cudnn_validate_20260926T175207Z.json`，日志 `acv_ops_20260926T175205Z.log`

### 5.1 显存与路径（**预登记判据：`s ≥ 512` 的行要求 `fused < 200 MiB` 且 `fused < 0.25 × eager`**）

| 配置 | 行 | s | T | 路径 | eager 瞬时 (MiB) | **fused 瞬时 (MiB)** | 降幅 | 判据 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| target | block2048_T2048_first | 2048 | 2048 | fused | 496.0 | **18.0** | 27.6× | 通过 |
| target | **block2048_T8192_hist** | 2048 | 8192 | fused | **1984.0** | **54.0** | **36.7×** | 通过 |
| target | block512_T8192_hist | 512 | 8192 | fused | 532.0 | 13.5 | 39.4× | 通过 |
| target | block4_T8192_hist | 4 | 8192 | fused | 52.7 | 0.1 | — | 不适用（`s<512`） |
| target | block2_T2_first | 2 | 2 | fused | 0.0 | 0.0 | — | 不适用 |
| target | block1_T8192_hist | 1 | 8192 | **eager** | 48.9 | — | — | 路径符合策略 |
| draft | block2048_T2048_first | 2048 | 2048 | fused | 571.0 | 16.4 | 34.8× | 通过 |
| draft | **block2048_T8192_hist** | 2048 | 8192 | fused | **2284.0** | **51.5** | **44.3×** | 通过 |
| draft | block512_T8192_hist | 512 | 8192 | fused | 592.0 | 12.9 | 45.9× | 通过 |
| draft | block4_T8192_hist | 4 | 8192 | fused | 32.4 | 0.1 | — | 不适用 |
| draft | block2_T2_first | 2 | 2 | fused | 0.0 | 0.0 | — | 不适用 |
| draft | block1_T8192_hist | 1 | 8192 | **eager** | 29.1 | — | — | 路径符合策略 |

- **路径判据 12/12 符合预登记期望**（`s ≥ 2` ⇒ fused；`s == 1` ⇒ eager）。
- 这解释了 §4.17.x 里 `peak_reserved` 撞 8188 MiB 的一个主要成分：
  **8K prefill 第一块的一次 attention 就要 1984 MiB（target）/ 2284 MiB（draft）**，
  融合后降到 ~54 / ~52 MiB（约两个数量级）。
- **这不是端到端 8K 收益，也不是速度结论**（见 §7 免责）。

### 5.2 "真的用了 cuDNN kernel"的证据（不靠"没报错"推断）

profiler 抓到的 kernel：`['aten::_scaled_dot_product_cudnn_attention']` ⇒ `matched=True`。
即融合路径**确实是 cuDNN 融合注意力算子**，不是 math/eager 回退。

### 5.3 数值：**预登记判据未通过（8/12 行 FAIL），但失败方向可解释**

预登记判据（跑之前写死在文件顶部）：`abs_err = max|x_eager − x_fused| ≤ 1e-2` **且**
`rel_err = abs_err / max(1e-6, max|x_eager|) ≤ 1e-2`。

| 配置 | 行 | s | T | abs | rel | **abs / bf16 ULP** | eager vs fp32 | fused vs fp32 | 更近者 | 预登记判定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| target | block2048_T2048_first | 2048 | 2048 | 2.344e-02 | 7.937e-03 | 2.03 | 2.138e-02 | 7.919e-03 | fused | **FAIL**(abs) |
| target | block2048_T8192_hist | 2048 | 8192 | 1.465e-03 | 1.240e-02 | 3.17 | 1.432e-03 | 3.354e-04 | fused | **FAIL**(rel) |
| target | block512_T8192_hist | 512 | 8192 | 9.766e-04 | 1.031e-02 | 2.64 | 1.037e-03 | 3.151e-04 | fused | **FAIL**(rel) |
| target | block4_T8192_hist | 4 | 8192 | 8.545e-04 | 1.277e-02 | 3.27 | 8.373e-04 | 2.550e-04 | fused | **FAIL**(rel) |
| target | block2_T2_first | 2 | 2 | 1.562e-02 | 4.255e-03 | 1.09 | 1.200e-02 | 7.661e-03 | fused | **FAIL**(abs) |
| target | block1_T8192_hist | 1 | 8192 | — | — | — | — | — | — | PASS（未调 kernel，符合策略） |
| draft | block2048_T2048_first | 2048 | 2048 | 1.562e-02 | 5.747e-03 | 1.47 | 1.435e-02 | 6.516e-03 | fused | **FAIL**(abs) |
| draft | block2048_T8192_hist | 2048 | 8192 | 1.465e-03 | 1.172e-02 | 3.00 | 1.271e-03 | 3.776e-04 | fused | **FAIL**(rel) |
| draft | block512_T8192_hist | 512 | 8192 | 8.545e-04 | 8.373e-03 | 2.14 | 8.851e-04 | 2.983e-04 | fused | PASS |
| draft | block4_T8192_hist | 4 | 8192 | 4.883e-04 | 6.579e-03 | 1.68 | 3.822e-04 | 1.703e-04 | fused | PASS |
| draft | block2_T2_first | 2 | 2 | 1.562e-02 | 5.155e-03 | 1.32 | 1.013e-02 | 7.350e-03 | fused | **FAIL**(abs) |
| draft | block1_T8192_hist | 1 | 8192 | — | — | — | — | — | — | PASS（未调 kernel，符合策略） |

**怎么读这张表（重要，避免误读成"融合路径算错了"）**

1. **在本表的输入分布下，差异是 bf16 舍入量级**：10 个融合行的 `abs_err` 是
   **1.09–3.27 个 bf16 ULP**（ULP 按该行输出自身量级算，`2⁻⁸ × max|x|`）。
   ⚠️ **但这个"只有几 ULP"的结论只对 `randn` 输入成立**：随机正态 q/k 让注意力在几千个 key 上
   几乎均匀，误差互相抵消；**真实模型数据是尖峰注意力，误差不抵消，会放大到 12–60%**
   —— 机制与量化证据见 **§6.2**。本表**不能**用来支持"融合路径与 eager 等价"。
2. **融合侧不是较差的一侧**：以"同一套数学在 **fp32** 下重算"为参考，
   **10/10 行 `fused_err_vs_fp32 < eager_err_vs_fp32`**（差 1.4×–4.3×）。
   即 `|eager − fused|` 主要由 **eager 侧**的 bf16 累加/舍入贡献。
   fp32 参考对这次归因是**够用的真值**：bf16 舍入 2⁻⁸ 与 fp32 2⁻²⁴ 相差 ~600×；
   而且该参考**复用同一份掩码代码**，若掩码写错，融合侧会立刻显示出巨大偏离（没有出现）。
   （§6.2 用更极端的尖峰输入把这条差异放大了 15–40×，机制因此被直接定位。）
3. **两个判据朝相反方向失效**：`s=2,T=2` 这类小行 **只会** 失 `abs`
   （输出量级 ~3.7，2 ULP 就 1.6e-2 > 1e-2），而 `T=8192` 的行 **只会** 失 `rel`
   （输出量级 ~0.1，`abs` 只有 1.5e-3 ≪ 1e-2，但 `rel` = 1.2e-2）。
   ⇒ **一条"绝对 1e-2 且相对 1e-2"的判据，在 bf16 注意力上不可能同时满足所有输出尺度**。
   这是判据本身的性质，与分支实现无关；我**保留 FAIL 不改判据**，把这条写进结论。
4. 掩码语义的独立性证明在 **CPU ① 组**（逐元素全等），不依赖上面这些数值。

### 5.4 算子级失败/跳过清单

- **首跑失败（已修，属工具缺陷，不是引擎缺陷）**：`torch.profiler.profile(activities=[CUDA])` 在
  torch 2.6 上抛 `Legacy CUDA profiling requires use_cpu=True`；随后误加 `use_cpu=True` 形参又抛
  `TypeError`（该形参在 `torch.profiler.profile` 里**不存在**）。
  正解：`activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]`。修好后 `matched=True`。
  首跑的整体 `op_verdict=FAIL` 是因为**崩溃在 profiler**，已按设计把算子级结果先落盘保留。
- 其余无跳过；12 行全部跑成。

## 6. 验证顺序第 3 步：模型级（短上下文，串行）—— **预登记判据 FAIL（10/10 行）**

工具同上，`ACV_STAGE=model`；配置 `chunk=1024`、`max_new=16`、`k=4`、`reps=2`；
上下文 `128 / 512`（两种 kv 格式）+ `1025`（只 `base_bf16`，**专门制造 s=1 的末块**）；
同一段 wikitext 前缀（与其它 G5 工具同源）。
结果：`output/g5/attn/attn_cudnn_validate_20260926T175638Z.json`，日志 `acv_model_20260926T175637Z.log`。

> **draft 的 kv 格式固定为 `base_bf16`**：draft 的 `head_dim=64` 而 `GROUP_V=128`
> （`model_self.py:70,467` 的断言）⇒ `v_fp8` 在 draft 上**结构上不可能**
> （`AssertionError: head_dim 64 % GROUP_V != 0`，本工具首跑就撞上了）。
> 这与全项目口径一致（`g5_stability_probe.py:289` 显式写死 `draft_kv="base_bf16"`，其余工具用默认值）
> ⇒ 冻结设计里的 `v_fp8` **只压 target 的 V cache**。

### 6.1 结果表

`gen_eq` = 生成 id 序列是否完全相同；`argmax_eq` = prompt 各位置 logits 的 argmax 一致比例；
`fused/期望` = prefill 融合次数；`短块回退/期望` = **只统计 `s<2` 的短块策略触发**；
`全 eager` = 策略为 `cudnn_prefill` 时的全部非融合调用（含非 prefill 阶段的**正常** eager）。

| 类别 | kv | ctx | rep | gen_eq | logits abs | logits rel | argmax_eq | argmax 翻转 | 末位 abs（余量） | fused/期望 | 短块回退/期望 | 全 eager | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **对照** | base_bf16 | 128 | — | ✅ | **0.0** | 0.0 | 1.0000 | 0/128 | 0.0（0.562） | — | — | 0 | PASS |
| A/B | base_bf16 | 128 | 0/1 | ✅ | 3.617 | 1.247e-01 | 0.9297 | 9/128 | 0.766（0.562） | 52/52 | 0/0 | 1132 | **FAIL** |
| **对照** | v_fp8 | 128 | — | ✅ | **0.0** | 0.0 | 1.0000 | 0/128 | 0.0（0.562） | — | — | 0 | PASS |
| A/B | v_fp8 | 128 | 0/1 | ✅ | 4.594 | 1.584e-01 | 0.9609 | 5/128 | 0.766（0.562） | 52/52 | 0/0 | 1132 | **FAIL** |
| **对照** | base_bf16 | 512 | — | ✅ | **0.0** | 0.0 | 1.0000 | 0/512 | 0.0（5.125） | — | — | 0 | PASS |
| A/B | base_bf16 | 512 | 0/1 | **❌** | 17.34 | 6.006e-01 | 0.9531 | 24/512 | 1.38（5.125） | 52/52 | 0/0 | 1108 | **FAIL** |
| **对照** | v_fp8 | 512 | — | ✅ | **0.0** | 0.0 | 1.0000 | 0/512 | 0.0（5.250） | — | — | 0 | PASS |
| A/B | v_fp8 | 512 | 0/1 | ✅ | 17.06 | 5.858e-01 | 0.9492 | 26/512 | 1.61（5.250） | 52/52 | 0/0 | 1108 | **FAIL** |
| **对照** | base_bf16 | 1025 | — | ✅ | **0.0** | 0.0 | 1.0000 | 0/1025 | 0.0（0.750） | — | — | 0 | PASS |
| A/B | base_bf16 | 1025 | 0/1 | ✅ | 1.000 | 5.128e-02 | **1.0000** | **0/1025** | 1.00（0.750） | 52/52 | **52/52** | 828 | **FAIL** |

- 两模型加载后 `peak_allocated=3906.6 MiB`、`peak_reserved=4134.0 MiB`（**短上下文，不是 8K**）。
- 全部 10 行 A/B **都未通过预登记判据**（`abs ≤ 1e-2` 且 `rel ≤ 1e-2` 且 argmax 全一致）。
- **`ctx=512 / base_bf16` 生成序列真的变了**：首个分歧在 **index 3**（eager 出 `1602`，融合出 `883`）。
- **阶段分流在模型级精确成立**：`fused=52/52`（target 28 层 + draft 24 层 × 1 个完整块）；
  `ctx=1025` 的 `[1024,1]` 两块 ⇒ `fused=52/52` **且** 短块回退 `52/52`；
  `eager_reason_kinds = {'short_block_seq_len_lt_min': 52, 'phase_not_prefill': 776}`
  —— 两类 eager **被干净地分开了**（非 prefill 的 776 次是**正常**路径，不是回退）；
  每个 A/B 行的 `assert_strategy_used()` 都是 `ok`。
- **重复性**：`rep=0` 与 `rep=1` 的每个数字**逐字节相同**（0.0 方差）。

### 6.2 为什么差这么多 —— **机制已定位（这是本节最重要的结论）**

**先排除噪声**：自对照 `eager vs eager` 在全部 5 组里 `abs = 0.0`（**逐比特相同**）
⇒ 引擎是确定性的，**全部差异只能归因于融合路径**，不是运行间抖动。

**机制**：eager 分支的分数矩阵是 **bf16**——
`model_self.py:357` 的 `torch.matmul(q, k_rep.transpose(-2,-1))` 输入 bf16、**输出也被舍入成 bf16**，
然后才进 softmax；而 cuDNN 内部**全程 fp32** 累计分数。
bf16 的相对精度是 2⁻⁸ ≈ 0.0039，所以分数误差 ≈ 0.0039 × |score|，经 `exp()` 放大后
在**尖峰注意力**（真实模型数据）下直接改变注意力权重。

在 GPU 上直接量的证据（`s=T=128`、真实头配置、`q` 乘不同倍数改变尖峰程度）：

| q 倍数 | 分数 bf16 vs fp32 | **eager vs fp32** | **fused vs fp32** | eager vs fused |
| --- | --- | --- | --- | --- |
| ×1（平坦） | 2.090e-02 | 1.343e-02 | 9.040e-03 | 1.562e-02 |
| ×3 | 6.390e-02 | 6.864e-02 | 8.048e-03 | 6.836e-02 |
| ×10（尖峰） | 2.044e-01 | 2.119e-01 | 8.708e-03 | 2.109e-01 |
| ×30（尖峰） | 5.892e-01 | 4.918e-01 | 1.311e-02 | 4.922e-01 |

**读法**：
1. eager 侧误差**随分数幅度线性增长**（1.3e-2 → 4.9e-1），斜率与**分数矩阵的 bf16 舍入误差**一致；
2. **fused 侧误差与幅度无关，稳定在 ~0.8–1.3e-2**（内部 fp32）；
3. `|eager − fused| ≈ |eager − fp32|` ⇒ **差异的全部来源是 eager 路径自己的 bf16 分数舍入**，
   不是融合路径算错（融合侧在尖峰区比 eager 侧准 **15–40×**）。
4. 这也解释了 §5.3 与本节看似矛盾的现象：§5 的算子测试用 `randn` ⇒ 注意力**平坦**（8192 个 key 上几乎均匀），
   误差互相抵消 ⇒ 只有 2–3 ULP；模型里是**尖峰**注意力 ⇒ 误差不再抵消 ⇒ 放大到 12–60%。
5. 另外排除了 GQA 相关的怀疑：`enable_gqa=True` 与"手工 `kv_repeat` 展开成实体张量"结果**逐位相同**，
   MHA（12/12、14/14）同样出现该现象 ⇒ **与 GQA 广播无关**。

**四个假说的排除（专项取证工具 `g5_attn_precision_forensics.py`，
日志 `output/g5/attn/forensics_20260926T180128Z.log`）**

| 假说 | 检验方式（可观测对比） | 结果 | 结论 |
| --- | --- | --- | --- |
| H1 eager 在 softmax 前把分数舍入成 bf16 | eager 误差随分数幅度增长、fused 不增长，且 `\|eager−fused\| ≈ \|eager−fp32\|` | `\|ef\|/\|e32\|` = 1.16 / 0.95 / 1.01 / 1.01 | **成立** |
| H2 融合侧忽略掩码 | 构造"未来 key 的 value 极大"（`v=100` 只在最后 4 位）：融合应贴**无掩码**参照 | `\|eager掩码−eager无掩码\|=1.238e+01`（测试有效），`\|fused−eager掩码\|=3.125e-02` vs `\|fused−eager无掩码\|=1.238e+01` | **不成立**（掩码被正确遵守） |
| H3 GQA 广播错误 | `enable_gqa=True` vs 手工实体 `kv_repeat` | 差值 **0.000e+00**（逐位相同，两套头配置都是） | **不成立** |
| H4 缩放被重复施加 | 融合结果在同量纲 eager 变体里最贴哪一个 | 最贴"正确 scale 1/√d"（1.28e-1），远于 `scale=1.0`（2.31）与"多乘一次"（2.71） | **不成立** |

> **本工具自身写错过两处，均已修并重跑（如实记录）**：
> ① H2 首版在循环里先用 `v[i]=1.0` 把刚设的巨大值**覆盖**掉 ⇒ 两个参照物退化相同，
> 差 0.0 却被判成"H2 成立"；现在加了**退化前提检查**（参照物必须显著不同，否则报"测试无效"）；
> ② H4 首版把"fused vs fp32"塞进同一个 dict 再取 `min` ⇒ 量纲不同必然选中 fp32 那条，
> 得出"H4 成立"的错误结论；现在只在**同量纲的 eager 变体**之间比较。
> 两处若不修，会得出与 H1 直接矛盾的结论 —— 这正是"先看数据再下结论"要防的东西。

### 6.3 这条结果的正确含义（**不要过度解读**）

- **不能**说"融合路径算错了"：在尖峰区它比 eager 更接近 fp32。
- **也不能**说"这条分支可以直接替换"：预登记判据 FAIL，且**生成序列确实变了**
  ⇒ **注意力后端不是行为中性的**，换后端 = 换模型行为。任何"A/B 对比"若一侧换了后端，
  这个差异就是一个**混淆项**，必须显式声明。
- 顺带暴露一个**此前没人量过的事实**：项目自建的 eager attention 基线
  **在 softmax 之前把分数舍入到 bf16**，在尖峰注意力下其相对误差可达 **~50%**（对阵 fp32）。
  这不是本次改动引入的，是**既有基线**的性质；本次**没有**去改它（超出本任务范围，且基线是冻结的）。

### 6.4 模型级失败/跳过清单（工具缺陷，均已修，**不是引擎缺陷**）

| 现象 | 原因 | 处置 |
| --- | --- | --- |
| 第 1 次尝试：直接 `AssertionError: head_dim 64 % GROUP_V != 0` | 我把 `draft_kv=kv` 也设成了 `v_fp8`（**与全项目口径不符**） | 改为 `draft_kv="base_bf16"`，并把原因写进结果（`draft_kv_note`） |
| 第 2 次尝试：`RuntimeError: quantile() input tensor is too large` | `torch.quantile` 有 2²⁴ 元素上限，`ctx=128` 的 logits 就有 1.9e7 个元素 | 固定步长抽样并把**步长写进结果**（`diff_quantile_stride`） |
| 崩溃后 `model_level` 整段丢失 | 只在函数末尾赋值给 `rec` | 改为**逐条 append + 立即落盘**（同 variance probe 的 `stage=formal_done` 教训） |
| 环境拒绝加载（第 1 次模型级尝试） | `commit 可用 15.8 GiB < 安全线 16 GiB` | **不改门槛**，几分钟后重试即通过（16.09 GiB）；拒绝本身已记录为 `skipped` |

## 7. 免责与红线（**必须随数字一起引用**）

1. §5.1 的显存/§5.2 的 kernel 证据都是**算子级**的：**不等于**端到端 8K 收益，
   **不等于** SpecDec 被加速，**没有**任何端到端计时结论。
2. 按任务要求，本阶段**不做** 8K A/B、不做完整 12 格稳定性、不做 k sweep、不动 Pilot 正式数据。
3. §6.1/§6.2 的模型级数字是**短上下文（128/512/1025）**、`max_new=16`、单机单次串行；
   **不能**外推到 8K。`prefill_s` / `wall_s` 只是**诊断字段**（含插桩），禁止进入任何 speedup 表。
4. 本次 GPU 环境带有 **24 个其它 C+G 进程**，且 commit 余量在 **16 GiB 安全线上下浮动**
   （实测 16.2 → 15.8 → 16.1 GiB，约 4 分钟内），所有显存/内存结论都受此限制。
5. **不得**由本报告得出："8K 已解决"、"正式方案已切换为 cuDNN"、"SpecDec 已加速"。
   目前只支持：
   - "**prefill 阶段的 attention 瞬时显存可用融合路径降到约 1/37–1/44**"（算子级）；
   - "**阶段分流机制按设计工作**，短块回退与非 prefill 阶段被精确区分并可举证"；
   - "**换成该后端会改变模型输出**（模型级判据 FAIL、`ctx=512` 生成序列变化）"，
     且差异来源已定位为 **eager 基线的 bf16 分数舍入**，而非融合路径的语义错误。
6. §6.2 的 fp32 参考是**同一套 eager 数学在 fp32 下重算**，不是独立第三方实现；
   它足以在"bf16 量级差异"上做归因（2⁻⁸ vs 2⁻²⁴ 相差 ~600×），但**不能**用来断言"融合侧绝对正确"。
7. 掩码语义的正确性依据是 **CPU ① 组的逐元素全等**（与本次数值无关，不依赖上表任何数字）。

## 8. 复现命令

```bat
:: 1) CPU 自测（不需要 GPU）
cd /d D:\RealDSH\specdec
.venv\Scripts\python.exe -m specdec.engine.attn_phase_selftest_cpu

:: 2) 算子级（不加载权重，空卡上瞬时 ~2.3 GiB）
set "PYTHONUTF8=1" & set "PYTHONIOENCODING=utf-8" & set "HF_HUB_OFFLINE=1"
set "ACV_STAGE=ops"
.venv\Scripts\python.exe -m specdec.engine.g5_attn_cudnn_validate

:: 3) 模型级（短上下文，串行；需 commit ≥16 GiB 且 P-state 合格）
set "ACV_STAGE=model"
.venv\Scripts\python.exe -m specdec.engine.g5_attn_cudnn_validate

:: 4) 精度取证（不加载权重，定位 eager/融合差异来源）
.venv\Scripts\python.exe -m specdec.engine.g5_attn_precision_forensics
```

所有判据、期望路径、上下文/格式/重复次数都可用环境变量改，但**默认值就是预登记值**；
改默认值等于改判据，必须在报告里显式声明。

**工具自身的已知限制**：`g5_attn_cudnn_validate.py` 的算子级表里，`randn` 输入会让注意力**平坦**，
因而只能反映"平坦注意力"下的误差（§5.3 的 2–3 ULP）；**尖峰注意力**下的行为必须看 §6.2 与取证工具。
这是本阶段最大的方法论教训：**算子级随机输入测试不足以验证注意力实现**。
