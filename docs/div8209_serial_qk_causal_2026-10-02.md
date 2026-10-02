# 位置 8209：L0 QK 的**逐行 M=1 causal-prefix 因果干预**闭环（2026-10-02）

> **本轮唯一问题**：只把 **B/verify 路径 L0** 的 QK 改成"按每个 query 行分别以 `M=1` + **真实 causal
> key 前缀**计算、再拼回原 block"（其余算子完全不变）之后，**QK 第 0 行 / L0 输出 / 后续 27 层 /
> 最终 token** 是否恢复成 S/AR 路径？
>
> **不做**：不修 `q_proj`、不改 AV/o_proj/MLP、不做 profiler、不做 FP32、不切后端、不优化性能、
> 不永久启用诊断分支、**不跑正式 B6**、不启动 acceptance/pilot/final、不放宽 token 门槛、
> 不用 ULP/margin 豁免、不外推到其他位置/family/层/模型、不覆盖旧产物。**B6 状态未改变。**

## 0. 结论（一句话）

**QK 几何是真实分歧来源，但不是唯一来源**：

* **H1 通过（逐位）**：干预后 B1 的 QK 第 0 行与 S0 **逐位一致**（`n_diff=0`，sha16 都是
  `bed16041bdba3327`），且 QK 节点整张量、causal mask、AV、head merge 全部**逐位恢复**。
* **H2 未通过**：L0 输出**未恢复**——差异由 **111 降到 61**（`max|Δ|=0.00390625` 两者相同）。
  ⇒ 按用户给定的因果结论规则：**"QK shape sensitivity 是一个真实分歧来源，但 L0 内还有至少一个
  独立 shape-sensitive 边界"**。本轮定位到的是 **`o_proj`**（**首个"持续到 L0 出口"**的分歧算子：
  其输入 `heads_merged` **逐位相同**、输出有 2 个坐标不同）。
* **H3**：分歧在 **L0 内部**产生（第 0 层输入行逐位相同、输出行不同）；第 1–27 层的**输入已经不同**
  ⇒ 严格判据（输入相同且输出不同）在这些层**无法触发**，本轮**不能**判定它们是否为独立边界
  （也不允许用"传递"一词代替证据）；同时给出逐层 `n_diff` 轨迹（L1 处 61→1262 的**放大**必须如实报告）。
* **H4 = 类别 2**：最终 **token 恢复**（S0=**323**，B1=**323**；B0=659），但 **logits 未逐位一致**
  （`n_diff=125439`、`max|Δ|=0.1875`）⇒ 只能说 **"当前 token 选择恢复，但 B6 的严格数值路径仍未完全对齐"**。

## 1. 命令、commit、dirty 状态与产物

```powershell
cd D:\RealDSH\specdec
$env:PYTHONIOENCODING='utf-8'
# ① 纯逻辑 CPU 门（逐行 causal 前缀 / 拼接 / mask 覆盖 / 哨兵 / H1–H4 分类）
.\.venv\Scripts\python.exe -X utf8 -m specdec.engine.serial_causal_qk
# ② 工具 CPU 门（微型模型上走**真实** forward：关闭等价 / 只作用 L0 / 异常恢复 / 落盘）
$env:SQ_MODE='selftest'; .\.venv\Scripts\python.exe -X utf8 -m specdec.engine.serial_qk_l0_f13_p8209
Remove-Item Env:SQ_MODE
# ③ 飞行前检查（含本轮新增一节：配置/默认关闭/API/逐行口径/mask 覆盖/参考产物/H）
.\.venv\Scripts\python.exe -X utf8 scripts\div8209_preflight_cpu.py
# ④ 一次正式 GPU 运行（工具**先提交**，保证 worktree clean）：四臂 × 2 pass + 关闭对照
$env:SQ_TAG='div8209_sqk'; $env:SQ_LAYER='0'; $env:SQ_PRE='8208'; $env:SQ_TARGET='8209'
$env:SQ_K='4'; $env:SQ_FEED='20642'; $env:SQ_CANDS='20642,659,576,4531'
$env:SQ_REPEAT='2'; $env:SQ_TARGET_ROPE='cpu_fp32'; $env:SQ_POISON_CHECK='1'; $env:SQ_CAPTURE_OFF_CHECK='1'
.\.venv\Scripts\python.exe -X utf8 -m specdec.engine.serial_qk_l0_f13_p8209
# ⑤ 只读 CPU 汇总（从原始张量独立复算 H1–H4 + 逐算子链 + 28 层轨迹 + 控制）
.\.venv\Scripts\python.exe -X utf8 scripts\serialqk8209_verdict_cpu.py
```

* **commit / dirty**：正式 run 记录 `git_commit = 79fd63daf3ffcff5e16877fc647360825e6d04be`、
  **`dirty_worktree = false`**（干预代码与工具**先提交**再跑 GPU）；控制 18 记录 git 与产物路径。
* **产物（新 tag，不覆盖旧产物）**：
  * `output/g5/attn/div8209_sqk_20261002T184729Z.json` + `.pt`（`stage=complete`）
    —— `.pt` 含四臂的 **L0 探针原始张量（31 项）**、**28 层输入/输出决策行**、
    最终 RMSNorm 行、完整词表 logits 行、诊断分支的**逐行 raw/scaled 与布局**、哨兵结果、L0 轮前缓存；
  * CPU 汇总：`output/g5/attn/serialqk8209_verdict_cpu_20261002T185207Z.json`（16 项核对全 True）。
* **固定配置（逐项沿用有效实验）**：F13、ctx 8192、chunk 2048、k=4、`base_bf16`、**all-eager**
  （`eager_by_phase={decode:140, verify:140}`、`fused_by_phase={}`）、BF16、`qk_fp32=False`、
  `prob_fp32=False`、`allow_bf16_reduced_precision_reduction=True`（effective True）、
  `allow_tf32=False`（全程）、target CPU-FP32 `inv_freq`、S=`[20642]`、B=`[20642,659,576,4531]`、
  **只加载 target（从不加载 draft）**、不自由生成、不启用 cuDNN/SDPA/FlashAttention/协作者 PR、
  同一份 `H(8208)`（`divcap_ar_…185503Z.pt`，`0440d98c533ae3ed`，纯切片 8237→8208，
  与重建捕获 56/56 逐位相同）、权重 338/338 `bitmatch=True`、RoPE 作用域还原 `ok=True`。

## 2. 干预的数据流（逐行 M=1 + 真实 causal 前缀 → 拼回 block）

生产 eager QK（**未改**，`model_self.SelfAttention.forward` 原有那一行）：

```python
scores = torch.matmul(q, k_rep.transpose(-2, -1)) / (self.head_dim ** 0.5)
```

诊断分支（**默认关闭**）在**同一位置**把它替换成逐行结果，`p = T - s`：

| 步 | 代码 | 说明 |
| --- | --- | --- |
| 1 | `visible_i = p + i + 1` | 第 `i` 行的**真实** causal 前缀长度（B：8209/8210/8211/8212） |
| 2 | `qi = q[:, :, i:i+1, :]` | **M = 1**（真的以 M=1 执行；**不是**先算 M=4 再切） |
| 3 | `ki = k_rep[:, :, :visible_i, :]` | 真实 causal key 前缀（绝不读未来 key） |
| 4 | `qi = qi.contiguous(); ki = ki.contiguous()` | 只做**内存布局规范化**：让第 0 行的两输入与 S 路径收到的**同形状同步长** |
| 5 | `raw_i = torch.matmul(qi, ki.transpose(-2, -1))` | **同一个** `torch.matmul` 表达式（不用 einsum/循环点积/手写 kernel） |
| 6 | `scaled_i = raw_i / (head_dim ** 0.5)` | **同一个**标量除法 |
| 7 | `out[:, :, i, :visible_i] = scaled_i[:, :, 0, :]` | 逐位切片赋值（**不**做任何算术） |
| 8 | `out[:, :, i, visible_i:] = 0.0` | **有限**占位值；这些位置**恰好**是原有 causal mask 覆盖的位置 |

拼好的 `scores_serial_causal: (1,12,4,8212)` 之后**完全走生产原路**：
原有 causal mask（`triu(diagonal=p+1)`）→ **fp32** softmax → `attn.to(bf16)` → AV matmul →
head merge → `o_proj` → 残差 → post-attention RMSNorm → MLP → 后续 27 层 → 最终 RMSNorm → 词表投影。

**为什么第 0 行能与 S0 逐位一致（机制上）**：S 路径那次 matmul 收到的是
`q (1,12,1,128)`（stride `(1536,128,128,1)`）与 `k_rep (1,12,8209,128)`（连续）；
干预后第 0 行的 `ki` 被规范化成 `(1,12,8209,128)` **连续**（stride `(12609024,1050752,128,1)`），
`qi` 是 `(1,12,1,128)`；而上一轮 QK 2×2 的 **A 格**已经用**同一种构造**逐位复现了真实 S 的
`qk_scores_pre_mask` —— 本轮 H1 再次独立确认了这一点。

**干预的作用范围**（结构性 + 实测双重证据）：分支只对**显式注册的模块实例**生效
（`with serial_qk_for(target.layers[0].self_attn)`，注册期间强引用该模块以避免 `id` 复用；
`mods[id] is self` 身份核对）；其余层进入分支后**只执行一条 `seen.append(id)`** 记录语句，
随后走的仍是**原样的 block QK 表达式**。实测：`n_applied=1`（且就是 L0 的 `self_attn`）、
`n_seen=27`（= 其余层）、`seen ∪ applied` 恰为全部 28 层。

## 3. 四臂配置

| arm | 输入 | 相位 | QK | 说明 |
| --- | --- | --- | --- | --- |
| **S0** | `[20642]` | `decode`（s=1） | 生产原式 | 真实 S 基线（决策行 = 末位行） |
| **B0** | `[20642,659,576,4531]` | `verify`（s=4） | 生产原式 | 真实 B 基线（决策行 = 第 0 行） |
| **B1** | 与 B0 **完全相同** | `verify` | **只在 L0** 逐行 M=1 causal-prefix | 本轮干预 |
| **S1** | 与 S0 **完全相同** | `decode` | 在 S 路径开启同一分支（s=1） | 控制：必须与 S0 逐位一致 |

每臂都挂三层**只读**捕获：① `OP.capture`（L0 的 31 个探针点，含 QK raw 之外的全部边界）；
② `layer_chain_capture`（28 层的输入行/输出行）；③ `LB.record`（最终 RMSNorm 的输入/输出）。
四臂 × **2 个独立 pass**（控制 12），另加"干预关闭后再跑 S/B"的对照。

## 4. H1–H4 的逐位结果

| 判据 | 结果 | 关键数字 |
| --- | --- | --- |
| **H1** QK 第 0 行（12 头 × keys `0:8209`） | **通过** | 干预前 S0 vs B0：`n_diff=3`、`max|Δ|=4`、坐标 `(0,2367) (1,4981) (9,4011)`；**干预后 S0 vs B1：`exact_equal=True`、`n_diff=0`**、sha16 `bed16041bdba3327`/`bed16041bdba3327` |
| **H2** L0 输出 | **未恢复** | S0 vs B0 `n_diff=111` → S0 vs B1 **`n_diff=61`**（`max|Δ|` 两者都是 `0.00390625`）⇒ 减少 50；首个分歧算子 `q_proj`；**首个持续分歧算子 `o_proj`** |
| **H3** 后续层 | **未恢复**（且在 L1 被放大） | 第 0 层：输入行逐位相同、输出不同（分歧源自 L0 内部）；第 1–27 层：**输入已经不同** ⇒ 严格判据无法触发；轨迹 61 → L1 **1262** → 之后在 1294–1377 区间波动 |
| **H4** logits / token | **类别 2** | logits `n_diff=125439`、`max|Δ|=0.1875`、sha16 `1718244499d14b33`/`03568d220e2fdf2b`；**token S0=323、B1=323（一致）**；对照：B0=659（S0 vs B0 为**类别 3**） |

**H1 的附带验证**（必需控制 7/8）：B1 第 0 行 mask 后未来 3 列全 `-inf`（36 个）；B1 全张量被掩
元素 `72 = 12×(3+2+1)`，与 B0 **完全相同**；S0 为 0（s=1 不加掩码）；可见区全有限。
**哨兵控制**：把第 `i` 行之后（`>= p+i+1`）的 key 全部灌 **NaN** 后重算该行，**四行的 raw/scaled
全部逐位不变且有限** ⇒ 每行确实只读 `K[:p+i+1]`。

## 5. L0 内部全部关键边界（S0 vs B1 / S0 vs B0，同一套语义切片）

| 算子（数据流顺序） | 输入全同？ | 输出全同？ | `n_diff`(S0vsB1) | `n_diff`(S0vsB0) |
| --- | --- | --- | --- | --- |
| `input_rmsnorm` | ✔ | ✔ | 0 | 0 |
| **`q_proj`** | ✔ | ✘ | **2** | 2 |
| `q_reshape` | ✘ | ✘ | 2 | 2 |
| `rope_q` | ✘ | ✔ | **0**（差异被吸收） | 0 |
| `k_proj` / `k_reshape` / `rope_k` / `kv_concat_k` | ✔ | ✔ | 0 | 0 |
| `v_proj` / `v_reshape` / `kv_concat_v` | ✔ | ✔ | 0 | 0 |
| **`qk_matmul_scale`** | ✔ | ✔ | **0** | **3** |
| `causal_mask` | ✔ | ✔ | **0** | 3 |
| **`softmax`** | ✔ | ✘ | **38924** | 55343 |
| `attn_times_v` | ✘ | ✔ | **0**（差异被吸收） | 2 |
| `head_merge` | ✔ | ✔ | **0** | 2 |
| **`o_proj`** | ✔ | ✘ | **2** | 9 |
| `attention_residual_add` | ✘ | ✘ | 1 | 4 |
| `post_attn_rmsnorm` | ✘ | ✘ | 1 | 4 |
| `gate_proj` | ✘ | ✘ | 68 | 247 |
| `up_proj` | ✘ | ✘ | 86 | 318 |
| `silu_activation` | ✘ | ✘ | 64 | 232 |
| `gated_product` | ✘ | ✘ | 142 | 503 |
| `down_proj` | ✘ | ✘ | 125 | 203 |
| **`mlp_residual_add`（= `layer_output`）** | ✘ | ✘ | **61** | **111** |

**满足"所有输入逐位相同、输出首次不同"的全部算子 = `['q_proj', 'softmax', 'o_proj']`**：

* `q_proj`：**瞬时** —— 它的 2 个坐标差异在 `rope_q` 处**回到 0**（RoPE 这一步之内被吸收；
  这是**已证实**的观测，不是推断）；
* `softmax`：**瞬时（本例）** —— softmax 输出有 38924 个坐标不同，但 `attn_times_v` **逐位相同**；
  生产在该段之间有 `attn.to(q.dtype)`（bf16）⇒ 差异在这一步或 AV 归约中被吸收；
  **具体是被 bf16 回转还是被 AV 归约吸收，本轮没有分离（未证实）**；
* `o_proj`：**持续** —— 输入 `heads_merged` **逐位相同**、输出 2 个坐标不同，且**从它开始之后每个
  算子的输出都不同** ⇒ 它是本例 L0 里**首个持续到出口的 shape-sensitive 边界**（0.00390625 = 1 bf16 ULP）。

## 6. 28 层边界比较（S0 vs B1）

| 层 | 输入行全同？ | 输入 `n_diff` | 输出行全同？ | 输出 `n_diff` | 判定（按本轮判据） |
| --- | --- | --- | --- | --- | --- |
| **L0** | **✔** | 0 | ✘ | **61** | **分歧在 L0 内部产生**（L0 内首个持续边界 = `o_proj`） |
| L1 | ✘ | 61 | ✘ | **1262** | 输入已不同 ⇒ 严格判据**无法**触发（观察到**放大** 61→1262） |
| L2–L27 | ✘ | 1262…1377 | ✘ | 1294…1377 | 同上；`n_diff` 在约 1300–1377 之间波动 |

**必须如实说明的限制**：第 1–27 层的输入**已经**不同，因此"输入逐位相同、输出开始不同"这个判据
在这些层**结构上无法触发** —— 本轮**不能**判定 L1–L27 中是否存在新的独立 shape-sensitive 边界，
也**不允许**用"传递"一词掩盖 L1 处 61→1262 的放大。要判定它们需要**逐层**的同类干预（下一个最小步骤）。

## 7. 最终 logits、argmax 与 token

| 量 | S0 | B0 | B1 |
| --- | --- | --- | --- |
| argmax token | **323**（锚点 323 ✔） | **659**（锚点 659 ✔） | **323** |
| logits sha16 | `1718244499d14b33` | `6bec1cf6d3e0c632` | `03568d220e2fdf2b` |
| 与 S0 的 logits `n_diff` | — | 136478 | **125439** |
| 与 S0 的类别 | — | **3**（logits 与 token 都不同） | **2**（token 相同、logits 不同） |
| 最终 RMSNorm 输出行 `n_diff`（vs S0） | — | — | **1298**（`max|Δ|=1.0`） |
| L0 输出行 `n_diff`（vs S0） | — | 111 | **61** |

⇒ **token 选择恢复**，但**数值路径没有逐位对齐**。按用户规则，只能说
"当前 token 选择恢复，但 B6 的严格数值路径仍未完全对齐"，**不得**因此宣布 B6 通过。

## 8. 全部必需控制（18 项）

| # | 控制 | 结果 |
| --- | --- | --- |
| 1 | S0 逐位复现历史 S（最终行 / L0 输出 / QK / x0） | ✔ 全部 `exact`（sha16 `1718244499d14b33` / `ff7c4133bea45a00` / `bed16041bdba3327` / `87bdb3aa1faf5f3c`） |
| 2 | B0 逐位复现历史 B（同四项） | ✔ 全部 `exact`（`6bec1cf6d3e0c632` / `37ddf8a1ff092d86` / `488a3f2e1027e9a9` / `87bdb3aa1faf5f3c`） |
| 3 | B0 的 QK 3 坐标差异逐位复现 | ✔ `n_diff=3`、`max|Δ|=4`、坐标 `[[0,2367],[1,4981],[9,4011]]`（与上一轮 QK 2×2 结论**逐位一致**） |
| 4 | S1 ≡ S0 逐位 | ✔ payload 里全部 29 个可比节点张量 + 最终行 + 28 层全部输入/输出行 |
| 5 | B1 第 0 行 QK ≡ S0 | ✔ **H1**（`n_diff=0`，sha16 相同） |
| 6 | B1 每行用正确的 `p+i+1` | ✔ `visible=[8209,8210,8211,8212]`，`p=8208`、`T=8212` |
| 7 | B1 future 在 mask 后全 `-inf` | ✔ 第 0 行 36 个；全张量 72（= B0 的 72）；可见区全有限；S0 为 0 |
| 8 | B1 其余三行没有错误读取未来 keys | ✔ 哨兵（灌 NaN）后四行 raw/scaled **逐位不变且有限** |
| 9 | 干预只作用于 L0 | ✔ `n_applied=1` 且 = L0 的 `self_attn`；`n_seen=27` = 其余层；`seen ∪ applied` = 全部 28 层 |
| 10 | L1–L27 的 QK 仍走普通 block 路径 | ✔ 同上（结构性：只有注册模块能被替换）+ 实测记录 |
| 11 | 干预关闭时生产路径逐位不变 | ✔ S0/B0 与历史逐位一致；且**关闭上下文后再跑 S/B**，与 S0/B0 逐位一致；进入实验前后 `serial_qk_state().active=False` |
| 12 | 两次独立 B1 重复逐位一致 | ✔ 四臂的节点/最终行/诊断逐行 raw+scaled **全部逐位相同**（2 个 pass） |
| 13 | cache、权重、输入张量未被修改 | ✔ H 逐层 k/v 摘要前后相同；权重的 `data_ptr`/摘要稳定；Norm 权重稳定；L0 轮前缓存长度 8208 |
| 14 | hook/context 全部恢复 | ✔ 探针 `active=False`、诊断 `active=False`、28 层钩子与 `norm` 钩子全部为空 |
| 15 | 所有非 mask 张量有限 | ✔ 比较切片与 logits 全有限；唯一的非有限元素是 B 路径 mask 后的 `-inf`（设计行为） |
| 16 | `allow_tf32=False` | ✔ 前后都是 `False`（且 `allow_bf16_reduced_precision_reduction` 前后不变） |
| 17 | 从不加载 draft | ✔ `loaded_models=["target"]`，`draft_loaded=False` |
| 18 | commit/dirty/产物路径完整记录 | ✔ `79fd63d`、`dirty_worktree=false`、JSON+PT 路径写进产物 |

## 9. 已证实 / 支持性推断 / 未证实（严格分开）

**已证实（本 run 实测，逐位）**
1. 干预把 B1 的 **QK 第 0 行**恢复成与 S0 **逐位一致**（`n_diff=0`），并且 **QK 节点整张量、
   causal mask 后、AV、head merge 全部逐位恢复**（S0 vs B1 这些节点 `n_diff=0`）。
2. L0 输出差异由 **111 降到 61**（`max|Δ|` 不变，仍是 1 bf16 ULP）⇒ QK 几何**解释了一部分**、**不是全部**。
3. L0 内满足"所有输入逐位相同、输出不同"的算子恰为 **`q_proj`、`softmax`、`o_proj`**；
   其中 `q_proj` 的差异在 `rope_q` 处回到 0、`softmax` 的差异在 `attn_times_v` 处回到 0。
4. **`o_proj` 是首个持续分歧算子**：输入逐位相同、输出 2 个坐标不同，且其后每个算子都不同。
5. 第 0 层输入行逐位相同、输出行不同 ⇒ 分歧**产生于 L0 内部**；第 1–27 层输入**已经不同**。
6. 最终 **token 恢复（323 = 323）**，但 logits **未逐位一致**（`n_diff=125439`）。
7. 18 项控制全部通过；四臂两个 pass 逐位一致；S0/B0 与上一轮产物**逐位一致**。

**支持性推断（由上面的数支持，但不是本轮新测到的机制）**
* `q_proj`、`softmax` 的 shape 敏感性是**真实存在**的（输入逐位相同、输出不同），但它们的效应
  在本例被下游**吸收**；因此"L0 内还有独立边界"的正确表述应落在 **`o_proj`**（持续）上，
  而不是链上第一个（`q_proj`）。
* `o_proj` 与 `q_proj` 的形态一致（都是 bf16 GEMM，两侧只差 **M=1 vs M=4**）⇒ 与上一轮
  "M 维度决定 QK 几何敏感性"的结论**同源**（同样不得升级为"实现 bug"）。
* L1 处 61→1262 的放大与 L0 输出的 61 个坐标差异一致（更多坐标被推到 bf16 取舍边界），
  这与"差异在多层内被放大"一致，但**本轮不解释**其机制。

**未证实（本轮没有测、也不声称）**
1. `o_proj` 内部**具体**是哪种 kernel/tile/归约顺序造成这 2 个坐标的差异（**本轮不做 profiler**）。
2. `softmax` 的差异究竟是被 `attn.to(bf16)` 回转吸收，还是被 AV 归约吸收（本轮**没有分离**）。
3. L1–L27 中是否存在**新的独立** shape-sensitive 边界（判据在这些层**无法触发**）。
4. 323 与 659 谁在数学上"更正确"（**未**做 FP32），以及 token 恢复是否在别的输入/位置上同样成立。
5. 该结论能否外推到别的位置、长度、family、层或模型（**不外推**）。
6. B6 是否通过（**未跑 B6**；token 恢复 ≠ B6 通过）。

## 10. B6 状态

**未改变**：本轮**没有**跑正式 B6/acceptance/pilot/final，**没有**修改或放宽任何正确性门槛，
**没有**把诊断分支设为默认开启（它默认关闭、只对显式注册的模块实例生效，退出即恢复）。
B6（`SpecDec ids == target-only greedy ids` 且全有限、多轮确定）**仍未通过** ⇒
**acceptance / pilot / final 仍然禁止**。本轮的 "token 恢复" 只是**类别 2**，数值路径没有对齐。

## 11. 代码改动（原理 / 输入 → 输出 / 必要性）

| 改动 | 原理 | 输入 → 输出 | 为什么必要 |
| --- | --- | --- | --- |
| **新增** `specdec/engine/serial_causal_qk.py`（CPU 门 **18/0**） | 纯逻辑：逐行 `visible=p+i+1`、单行 M=1 matmul、拼回 `(1,H,s,T)`、有限占位、H1–H4 分类 | 张量 → 结构化比较/分类 | 干预的口径必须可独立 CPU 复算；分类判据不能靠 GPU 现场判断 |
| **改** `model_self.py`：新增默认 `None` 的 `_SERIAL_QK` 注册表 + `set_serial_qk/clear_serial_qk/serial_qk_state/serial_qk_calls/serial_qk_for` + 在 QK 生成处加一处**分支** | 关闭时只多两次身份判断（`_SERIAL_QK is None`），**不碰张量** ⇒ 生产表达式原样执行；开启时只替换被注册模块的 `scores` | `with serial_qk_for(L0.self_attn)` → 逐行 M=1 的 `scores` | 需要一个**默认关闭、只作用于指定实例、异常也恢复**的诊断分支；不复制整套 Attention/DecoderLayer |
| **新增** `specdec/engine/serial_qk_l0_f13_p8209.py`（CPU 门 **17/0**） | 四臂 × 2 pass 串行 + 三层只读捕获 + 哨兵控制 + 18 项控制 + 先落盘 | 模型 + `H(8208)` + 固定 token → JSON/PT | 本轮唯一实验；"先落盘再打印"沿用既有教训 |
| 改 `scripts/div8209_preflight_cpu.py`（+1 节） | 加载前核对配置/默认关闭/API/逐行口径/mask 覆盖/参考产物/H | 固定路径产物 → 断言 | 避免再白付一次模型加载 |
| **新增** `scripts/serialqk8209_verdict_cpu.py`（只读） | 从原始张量独立复算 H1–H4、逐算子链（首个 / 首个**持续**）、28 层轨迹、诊断构造自检、16 项核对 | `.json`+`.pt` → 汇总 JSON | 结论必须能从原始张量独立复算；**摘要错误不靠重跑 GPU 修** |
| **未改** | 生产 QK 表达式、AV/o_proj/MLP、dtype/精度开关、mask/RoPE/attention/cache/argmax、后端、proposal、门槛 | — | 本轮只做诊断性干预 |

## 12. 唯一最小下一步（**未执行**）

> 把**同一个**干预机制（逐行 M=1 + 真实 causal 前缀）从 QK **推广到 `o_proj` 这一个 GEMM**：
> 在 B 路径 L0 里同时对齐 QK 与 `o_proj` 的 M 几何（`o_proj` 的 4 行也逐行以 M=1 计算后拼回），
> 其余算子不变，看 **L0 输出是否恢复、token/logits 是否进一步对齐**。
>
> **判据（二值）**：若 L0 输出**逐位恢复** ⇒ 本例 L0 的 shape 敏感性由 QK 与 `o_proj` 两处构成；
> 若仍不同 ⇒ L0 内还有第三处独立边界，继续用同一条链（首个"输入同·输出不同"算子）定位。
>
> **纪律**：本轮**不执行**；不做 profiler、不切后端、不改生产默认路径、不做 FP32、
> **不跑 B6/acceptance/pilot/final**、不外推；新产物用**新 tag**，旧产物（含本轮产物）**一律保留**。

---

## 13. 后续轮次指针（**已完成，不在本文档范围内**）

本文档 §12 的"把干预推广到 `o_proj`"已在下一轮执行，并进一步推进到 **`gate_proj` 的定位**：
> `docs/div8209_serial_qk_oproj_2026-10-02.md`（报告 §9.58 / 台账 §4.73）——
> 在保留本轮的 serial-causal QK 之外，再把 L0 的 `self_attn.o_proj` 按行以 **M=1** 调用**同一个 `nn.Linear`**
> 后 `cat` 拼回（默认关闭、实例级、异常安全），五臂 **S0/B0/BQ/BQO/S1** 比较。
>
> 结果：**H2 通过** —— BQO 的 `heads_merged` 与 **`o_proj` 决策行都与 S0 逐位一致**，
> 本轮 BQ 的 **2 个 `o_proj` 差异被完全消除**，**attention 子块（QK/AV/head merge/o_proj/残差/post-attn RMSNorm）
> 全部逐位恢复**；**但 H3 仍未通过**：L0 输出差异 **111 → 61 → 21**（`max|Δ|` 恒 1 bf16 ULP）
> ⇒ **第三个独立边界 = 首个持续边界 `gate_proj`**（`up_proj` 同为其独立边界；两者共用同一输入）。
> **H5 = 类别 2**：token 恢复 323=323、logits 仍未对齐（`n_diff=129412`）⇒ **token 恢复 ≠ B6 通过**。
> 该轮**未**改生产默认路径、未做 profiler/FP32、未切后端、未跑 B6。**B6 状态仍未改变。**

再下一轮（**L0 闭环最后一轮**）把同一机制推广到 MLP 三个投影，并把 L0 完全闭合：
> `docs/div8209_serial_l0_closure_2026-10-02.md`（报告 §9.59 / 台账 §4.74）——
> 六臂 S0/B0/BQO/BQOGU/BQOGUD/S1；L0 输出差异轨迹 **111 → 21 → 5 → 0**，
> **BQOGUD 的 L0 输出与 S0 逐位一致**（sha16 `ff7c4133bea45a00`），后续**第一处**重新分歧层 = **L1**；
> `down_proj` 被**实测判定**为独立边界；最终 token 恢复但 logits 未对齐（类别 2，且 `n_diff` 非单调）。
> 该轮按约定**结束**了 L0 逐算子追踪，下一步转为**项目级决策**。**B6 状态仍未改变。**
