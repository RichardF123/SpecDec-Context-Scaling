# 位置 8209：L0 的 **QK + o_proj 联合对齐** —— attention 子块闭合，第三处边界 = **`gate_proj`**（2026-10-02）

> **本轮唯一目标**：在**保留**上一轮 L0 serial-causal QK 干预的前提下，把 L0 的 `self_attn.o_proj`
> 也按行以 **M=1** 调用**同一个 `nn.Linear`** 再拼回 `(1,4,1536)`，判定 `o_proj`、L0 输出、后续层与
> 最终 logits 能否进一步恢复。**只干预 L0 的 QK 与 L0 的 `o_proj`。**
>
> **不做**：不修 Q/K/V projection、softmax、AV、MLP；不改权重/mask/RoPE/cache/argmax/后端；
> 不做 profiler/FP32；不优化性能；**不跑 B6**、不启动 acceptance/pilot/final；不放宽 token 门槛；
> 不用 ULP/margin 豁免；不外推到其他位置/family/层/模型；不覆盖旧产物；不把诊断分支设为生产默认。
> **B6 状态未改变。**

## 0. 结论（一句话 + 关键数字）

**attention 子块已完全闭合，但 L0 仍未完全恢复；第三处独立边界 = `gate_proj`。**

| 量 | B0（无干预） | BQ（只 QK） | **BQO（QK + o_proj）** | 与 S0 的关系 |
| --- | --- | --- | --- | --- |
| QK 第 0 行（12 头 × keys 0:8209） | `n_diff=3`/`max 4` | **`n_diff=0`** | **`n_diff=0`** | **逐位一致** |
| `heads_merged` 决策行 | 2 | **0** | **0** | **逐位一致** |
| `o_proj` 决策行 | 9 | **2** | **0**（sha16 都是 `ba484a8bc131b224`） | **逐位一致** |
| `attention_residual` / `post_attention_rmsnorm` | 4 / 4 | 1 / 1 | **0 / 0** | **逐位一致** |
| **L0 输出** | **111** | **61** | **21** | 未恢复 |
| token | 659 | 323 | **323** | 一致 |
| logits `n_diff` | 136478 | 125439 | **129412** | 未对齐（类别 2） |

* **H2 通过**：BQO 的 `heads_merged` 与 `o_proj` 决策行都与 S0 **逐位一致**（`o_proj` 的 sha16 与 S0 相同）
  ⇒ serial per-row o_proj **成功闭环**；BQ 那 **2 个差异（坐标 `[153]`、`[1519]`）被完全消除**。
* **H3 未通过**：L0 输出仍差 **21/1536**（`max|Δ| = 0.00390625` = 1 bf16 ULP）
  ⇒ **L0 内存在第三个独立 shape-sensitive 边界**：**首个"持续到出口"的边界 = `gate_proj`**
  （满足"所有输入逐位相同、输出不同"的全部算子 = `['q_proj', 'softmax', 'gate_proj', 'up_proj']`；
  其中 `q_proj`、`softmax` 的差异仍被下游吸收）。
* **单独检查 MLP**：`attention_residual`、`post_attention_rmsnorm` 已与 S0 **逐位相同**，
  而 **`gate_proj`（输入逐位相同、输出 4 个坐标不同）与 `up_proj`（`6`）都是独立边界**
  ⇒ **MLP 的 BF16 Linear 也存在独立的 M=1/M=4 形状敏感性**（**不**合并成"MLP 不同"）。
* **H5 = 类别 2**：最终 **token 恢复（323 = 323）**，但 logits **未逐位一致**
  （`n_diff=129412`、`max|Δ|=0.1875`、`mean|Δ|=0.0320`、两边 top-2 margin 都是 `0.0`）
  ⇒ **只能说"token 结果恢复、数值路径仍未对齐"**，**不得**说完整恢复，**不得**宣布 B6 通过。

## 1. 命令、commit、dirty 状态与产物

```powershell
cd D:\RealDSH\specdec
$env:PYTHONIOENCODING='utf-8'
# ① 纯逻辑 CPU 门（逐行 Linear / 拼接顺序 / s=1 等价 / 链持久性 / H1–H5 / MLP 报告）
.\.venv\Scripts\python.exe -X utf8 -m specdec.engine.serial_align
# ② 工具 CPU 门（微型模型真实 forward：两个 context 嵌套 / 只作用指定模块 / 异常恢复 / 落盘）
$env:SLA_MODE='selftest'; .\.venv\Scripts\python.exe -X utf8 -m specdec.engine.serial_align_l0_f13_p8209
Remove-Item Env:SLA_MODE
# ③ 飞行前检查（含本轮新增一节：五臂定义/两个开关组合/默认关闭/API/上一轮 BQ 参考/H）
.\.venv\Scripts\python.exe -X utf8 scripts\div8209_preflight_cpu.py
# ④ 一次正式 GPU 运行（工具**先提交**，保证 worktree clean）：五臂 × 2 pass + 关闭对照
$env:SLA_TAG='div8209_sla'; $env:SLA_LAYER='0'; $env:SLA_PRE='8208'; $env:SLA_TARGET='8209'; $env:SLA_K='4'
$env:SLA_FEED='20642'; $env:SLA_CANDS='20642,659,576,4531'; $env:SLA_REPEAT='2'
$env:SLA_TARGET_ROPE='cpu_fp32'; $env:SLA_POISON_CHECK='1'; $env:SLA_CAPTURE_OFF_CHECK='1'
.\.venv\Scripts\python.exe -X utf8 -m specdec.engine.serial_align_l0_f13_p8209
# ⑤ 只读 CPU 汇总（从原始张量独立复算 H1–H5 + 逐算子链 + MLP 边界 + 28 层轨迹 + 控制）
.\.venv\Scripts\python.exe -X utf8 scripts\serialalign8209_verdict_cpu.py
```

* **commit / dirty**：正式 run 记录
  `git_commit = 39b93426e797b26029f628c8f6f277f92616f51c`（= "修掉 `_cmp_all_B` 分派缺陷 + 落盘提前"那次提交）、
  **`dirty_worktree = false`**；干预代码与工具**先提交**再跑 GPU。
* **产物（新 tag，不覆盖旧产物）**：
  * `output/g5/attn/div8209_sla_20261002T192540Z.json` + `.pt`（`stage=complete`）
    —— `.pt` 含五臂的 L0 探针原始张量（31 项）、28 层输入/输出决策行、最终 RMSNorm 输入/输出行、
    完整词表 logits 行、**两个诊断分支的逐行原始记录**、哨兵结果、L0 轮前缓存、`o_proj` 模块事实；
  * CPU 汇总：`output/g5/attn/serialalign8209_verdict_cpu_20261002T192933Z.json`（13 项核对全 True）；
  * **保留的失败 run（缺陷 ①）**：`output/g5/attn/div8209_sla_20261002T192404Z.json`
    （`stage=crashed`，**无 `.pt`**：10 次前向跑完后在比较阶段崩溃）。
* **固定配置（逐项沿用有效实验）**：F13、ctx 8192、chunk 2048、k=4、`base_bf16`、**all-eager**
  （`fused_by_phase={}`）、BF16、`qk_fp32=False`、`prob_fp32=False`、
  `allow_bf16_reduced_precision_reduction=True`（effective True）、`allow_tf32=False`（全程）、
  target CPU-FP32 `inv_freq`、S=`[20642]`、B=`[20642,659,576,4531]`、**只加载 target（从不加载 draft）**、
  不自由生成、不启用 cuDNN/SDPA/FlashAttention/协作者 PR、同一份 `H(8208)`
  （`divcap_ar_…185503Z.pt`，`0440d98c533ae3ed`，纯切片 8237→8208，与重建捕获 56/56 逐位相同）、
  权重 338/338 `bitmatch=True`、RoPE 作用域还原 `ok=True`。
* **`o_proj` 事实（从**实际模块**读取，不靠常量）**：`class_name=Linear`、`in_features=1536`、
  `out_features=1536`、**`has_bias=False`**、`weight_shape=[1536,1536]`、
  `weight_dtype=torch.bfloat16`、`weight_data_ptr` 在 run 前后**不变**。

## 2. serial o_proj 的准确数据流

生产原式（**未改**，`SelfAttention.forward`）：`op = self.o_proj(out)`，
S 路径 `out=(1,1,1536)`（等价 M=1）、B 路径 `out=(1,4,1536)`（等价 M=4）。

诊断分支（**默认关闭**）：

```python
rows = []
for i in range(out.shape[1]):
    oi = self.o_proj(out[:, i:i+1, :])     # 每行**实际**以 [1,1,1536] 调用**同一个** nn.Linear
    rows.append(oi)
op = torch.cat(rows, dim=1)                # 拼接只组合行，不做任何额外算术
```

| 要求 | 实测（控制编号） |
| --- | --- |
| 每行实际以 `[1,1,1536]` 调用 | ✔ 4 行全部 `[1,1,1536]`（控制 8；`per_row_input_shapes=[[1,1,1536]]×4`） |
| 同一模块实例 / 同一权重 / 同一 dtype | ✔ `module_id` 与 L0 的 `o_proj` 相同、`weight_data_ptr` 前后不变、输入 dtype 全 `bf16`、`has_bias=False`（控制 9） |
| 不允许先算 M=4 再切片 | ✔ 分支里**没有**任何整块调用（只有 4 次 `[1,1,1536]` 调用） |
| 不允许显式 matmul / FP32 / einsum / 手写 kernel | ✔ 只调用 `nn.Linear.forward`（与生产同一模块、同一调用形式） |
| 拼接只组合行 | ✔ `torch.cat(rows, dim=1)`；进入生产路径的张量与探针记录的 `o_proj` **逐位相同** |
| **第 0 行必须复现 S0 的 `o_proj` 输出** | ✔ **H2**：决策行逐位一致（sha16 都是 `ba484a8bc131b224`） |
| 其他三行仍保存并继续进入生产路径 | ✔ `(1,4,1536)` 拼回后照常进入残差 → post-attention RMSNorm → MLP（下游 4 个节点随之从 1 个差异降到 **0**） |

**为什么第 0 行能逐位一致**：S0 的 `out` 本身就是 `(1,1,1536)`；分支对第 0 行做的是
`out[:, 0:1, :]`（同一段内存的视图，dim1 长度为 1），`nn.Linear.forward` 内部对 >2 维输入做
`reshape(-1, in_features)` ⇒ 两条路径在**同一形状、同一 dtype、同一权重**下执行同一次调用。
实测确认（H2 + 控制 7）。

**作用范围**：只对**显式注册的模块实例**生效（`serial_linear_for(L0.self_attn.o_proj)`；`mods[id] is mod`
身份核对 + 注册期间强引用）。实测：`n_applied=1`（就是 L0 的 `o_proj`）、`n_seen=27`（= 其余 27 层的 `o_proj`，
它们进入分支后只执行一条计数语句、随后仍走**原样的一次调用**）、`seen ∪ applied` = 全部 28 层。

## 3. 五个 arm

| arm | 输入 | serial-causal QK | serial o_proj（逐行 M=1） | 本 run 实测 applied |
| --- | --- | --- | --- | --- |
| **S0** | `[20642]`（decode，s=1） | 关 | 关 | QK 0 / o_proj 0 |
| **B0** | `[20642,659,576,4531]`（verify，s=4） | 关 | 关 | 0 / 0 |
| **BQ** | 同 B0 | **开** | 关 | 1 / 0 |
| **BQO** | 同 B0 | **开** | **开** | 1 / 1 |
| **S1** | 同 S0 | **开** | **开** | 1 / 1 |

每臂都挂三层**只读**捕获（L0 的 31 个探针点 / 28 层输入-输出决策行 / 最终 RMSNorm 输入-输出），
各臂独立克隆同一份 `H(8208)` 的缓存，串行执行；**五臂 × 2 个独立 pass**（控制 12），
另加"两个分支都关闭后再跑 S/B"的对照（控制 13）与 QK 的"不读未来 key"哨兵（保留自上一轮）。

## 4. H1–H5 结果

| 判据 | 结果 | 关键数字 |
| --- | --- | --- |
| **H1** 旧结果复现 | **通过** | S0 ≡ 历史 S0（final/L0 输出/QK/x0 全 `exact`）；B0 ≡ 历史 B0（同四项全 `exact`）；**BQ ≡ 上一轮 QK-only 干预**（29 个节点 + 28 层行 + final + L0 输出全逐位）；**S1 ≡ S0**（节点 + 层行 + final） |
| **H2** o_proj 局部闭环 | **通过** | `heads_merged` 决策行逐位一致（`n_diff=0`）；`o_proj` 决策行**逐位一致**（`n_diff=0`，sha16 `ba484a8bc131b224`）；BQ 的 2 个差异（`[153]`、`[1519]`）**被完全消除** |
| **H3** L0 是否完全恢复 | **未恢复** | `111 → 61 → **21**`（`max|Δ|` 恒为 `0.00390625` = 1 bf16 ULP）；首个持续边界 = **`gate_proj`** |
| **H4** 后续层 | 未恢复 | 第 0 层输入逐位相同、输出不同（分歧源自 L0 内部）；第 1–27 层**输入已不同** ⇒ 判据无法触发（轨迹 21 → L1 **851** → … → 1273–1348 平台） |
| **H5** 最终结果 | **类别 2** | logits `n_diff=129412`、`max|Δ|=0.1875`、`mean|Δ|=0.0320`；**token 323 = 323**；top-2 margin S0=`0.0`、BQO=`0.0`；对照 B0：`n_diff=136478`、token 659（类别 3）；BQ：`n_diff=125439`、token 323 |

**关于 H5 的一个如实观察**：BQO 的 logits `n_diff`（129412）比 BQ 的（125439）**略大**，
即"多对齐一个算子"并没有单调地让 logits 更接近 S。这是**观测**，本轮**不解释**其机制（也不改变分类：
两者都是类别 2、token 都是 323）。

## 5. L0 全部算子边界比较（S0vsBQO | S0vsBQ | S0vsB0 的 `n_diff`）

| 算子（数据流顺序） | 输入全同？ | 输出全同？ | BQO | BQ | B0 |
| --- | --- | --- | --- | --- | --- |
| `input_rmsnorm` | ✔ | ✔ | 0 | 0 | 0 |
| **`q_proj`** | ✔ | ✘ | **2** | 2 | 2 |
| `q_reshape` | ✘ | ✘ | 2 | 2 | 2 |
| `rope_q` | ✘ | ✔ | **0**（吸收） | 0 | 0 |
| `k_proj`/`k_reshape`/`rope_k`/`kv_concat_k` | ✔ | ✔ | 0 | 0 | 0 |
| `v_proj`/`v_reshape`/`kv_concat_v` | ✔ | ✔ | 0 | 0 | 0 |
| **`qk_matmul_scale`** | ✔ | ✔ | **0** | **0** | **3** |
| `causal_mask` | ✔ | ✔ | **0** | 0 | 3 |
| **`softmax`** | ✔ | ✘ | **38924** | 38924 | 55343 |
| `attn_times_v` | ✘ | ✔ | **0**（吸收） | 0 | 2 |
| `head_merge` | ✔ | ✔ | **0** | 0 | 2 |
| **`o_proj`** | ✔ | ✔ | **0** | **2** | 9 |
| `attention_residual_add` | ✔ | ✔ | **0** | 1 | 4 |
| `post_attn_rmsnorm` | ✔ | ✔ | **0** | 1 | 4 |
| **`gate_proj`** | ✔ | ✘ | **4** | 68 | 247 |
| **`up_proj`** | ✔ | ✘ | **6** | 86 | 318 |
| `silu_activation` | ✘ | ✘ | 2 | 64 | 232 |
| `gated_product` | ✘ | ✘ | 8 | 142 | 503 |
| `down_proj` | ✘ | ✘ | 43 | 125 | 203 |
| **`mlp_residual_add`（= `layer_output`）** | ✘ | ✘ | **21** | **61** | **111** |

**满足"所有输入逐位相同、输出不同"的全部算子 = `['q_proj', 'softmax', 'gate_proj', 'up_proj']`**：

* `q_proj`：瞬时（差异在 `rope_q` 处回到 0）——**已证实**；
* `softmax`：瞬时（差异在 `attn_times_v` 处回到 0；生产在该段有 `attn.to(bf16)`）——
  **具体是被 bf16 回转还是被 AV 归约吸收，本轮仍未分离（未证实）**；
* **`gate_proj`**：**持续** —— 输入 `post_attention_rmsnorm` **逐位相同**、输出 4 个坐标不同，
  且**从它开始之后每个算子的输出都不同** ⇒ 本例 L0 中**第三个独立边界、也是当前首个持续边界**；
* **`up_proj`**：**也是独立边界**（同一输入逐位相同、输出 6 个坐标不同）——
  与 `gate_proj` 是**两个**并列的 BF16 Linear（**不**合并成"MLP 不同"）。

## 6. MLP 边界的独立检查（逐 projection，**不**合并）

| 算子 | 输入逐位相同？ | 输出逐位相同？ | BQO `n_diff` | 是否为独立边界 |
| --- | --- | --- | --- | --- |
| `gate_proj` | **✔**（`post_attention_rmsnorm`） | ✘ | **4** | **是（第一个）** |
| `up_proj` | **✔**（同一输入） | ✘ | **6** | **是** |
| `silu_activation` | ✘（输入 `mlp_gate_proj` 已不同） | ✘ | 2 | 否（判据无法触发） |
| `gated_product` | ✘（两个输入都已不同） | ✘ | 8 | 否 |
| `down_proj` | ✘ | ✘ | 43 | 否 |
| `mlp_residual_add` | ✘ | ✘ | 21 | 否 |

* 前提条件成立：`attention_residual` 与 `post_attention_rmsnorm` 已与 S0 **逐位相同**
  ⇒ 这两处边界**不是**上游污染，而是 **MLP 自身的 BF16 Linear 形状敏感性**（M=1 vs M=4）。
* 同时要如实说：`silu`/`gated`/`down`/`residual` 的输入**已经不同**，所以严格判据在它们身上
  无法触发；本轮**不能**判定它们是否也含独立来源（要判定需要继续逐算子干预）。

## 7. 28 层输入/输出轨迹（S0 vs BQO）

| 层 | 输入全同？ | 输入 `n_diff` | 输出全同？ | 输出 `n_diff` | 说明 |
| --- | --- | --- | --- | --- | --- |
| **L0** | **✔** | 0 | ✘ | **21** | 分歧在 L0 内部产生（attention 子块已闭合，剩余来自 `gate_proj`/`up_proj`） |
| L1 | ✘ | 21 | ✘ | **851** | 输入已不同 ⇒ 判据**无法触发**（观察到放大 21→851） |
| L2–L27 | ✘ | 851…1348 | ✘ | 1050…1348 | 在约 1050–1348 区间波动（B0 的对照平台在 1294–1377） |

**必须如实说明**：第 1–27 层的输入**已经**不同，因此"输入逐位相同、输出不同"这一判据在这些层
**结构上无法触发** —— 本轮**不能**判定 L1–L27 是否有新的独立边界，也**不允许**用"传递"一词掩盖
L1 处 21→851 的放大。逐层判定需要继续做同类逐算子干预。

## 8. 最终 logits / token 对照

| 量 | S0 | B0 | BQ | **BQO** |
| --- | --- | --- | --- | --- |
| argmax token | **323**（锚点 323 ✔） | 659（锚点 659 ✔） | 323 | **323** |
| logits `n_diff` (vs S0) | — | 136478 | 125439 | **129412** |
| `max|Δ|` | — | — | — | **0.1875** |
| `mean|Δ|` | — | — | — | **0.0320** |
| top-2 margin | **0.0** | — | — | **0.0** |
| 类别 | — | 3 | 2 | **2** |
| 最终 RMSNorm 输出行 `n_diff` | — | 1343 | — | **1318** |

⇒ **token 结果恢复**，但**数值路径没有逐位对齐**（类别 2）。按用户规则：**不得**说"完全恢复"，
**不得**据此宣布 B6 通过。

## 9. 20 项必需控制

| # | 控制 | 结果 |
| --- | --- | --- |
| 1 | S0 逐位复现历史 S（final/L0 输出/QK/x0） | ✔ 全 `exact` |
| 2 | B0 逐位复现历史 B（同四项） | ✔ 全 `exact` |
| 3 | BQ 逐位复现上一轮 QK-only 干预 | ✔ 29 节点 + 28 层行 + final + L0 输出全逐位；且 QK row0 与 S0 逐位一致 |
| 4 | S1 与 S0 逐位一致 | ✔ 节点 + 28 层行 + final（QK/o_proj 各 1 次调用，s=1） |
| 5 | BQO QK row0 与 S0 逐位一致 | ✔ `n_diff=0` |
| 6 | BQO `heads_merged` row0 与 S0 逐位一致 | ✔ `n_diff=0` |
| 7 | BQO `o_proj` row0 与 S0 逐位一致 | ✔ `n_diff=0`，sha16 同为 `ba484a8bc131b224` |
| 8 | serial o_proj 每行实际输入 shape `[1,1,1536]` | ✔ `[[1,1,1536]]×4`（1 次调用、4 行） |
| 9 | 同一模块实例 / 同一权重 / 同一 dtype | ✔ `module_id` 一致、`weight_data_ptr` 前后不变、输入全 bf16、`has_bias=False` |
| 10 | 干预只作用于 L0 o_proj | ✔ `n_applied=1` 且 = L0 的 `o_proj` |
| 11 | L1–L27 o_proj 仍走原路径 | ✔ `n_seen=27` = 其余层的 `o_proj`；`seen ∪ applied` = 全部 28 层 |
| 12 | 两次 BQO 重复逐位一致 | ✔ 五臂 × 2 pass 的节点/最终行/层行全部逐位相同 |
| 13 | 干预关闭后生产路径逐位不变 | ✔ S0/B0 逐位复现历史；关闭上下文后再跑 S/B 与 S0/B0 逐位一致；进入实验前后两分支 `active=False` |
| 14 | context / hook 全部恢复 | ✔ 探针关、两个分支关、28 层钩子与 `norm` 钩子全空 |
| 15 | cache、输入、权重未被修改 | ✔ H 逐层摘要前后相同；权重 data_ptr 与摘要稳定；Norm 与 o_proj 权重稳定；L0 轮前缓存长度 8208（五臂相同） |
| 16 | 所有张量有限，无 NaN/Inf | ✔ 比较切片与全部 logits 有限；唯一非有限元素是 B 路径 mask 后的 `-inf`（BQO 与 B0 各 72、S0 为 0，设计行为） |
| 17 | `allow_tf32=False` | ✔ 前后都是 `False` |
| 18 | 不加载 draft | ✔ `loaded_models=["target"]`、`draft_loaded=False` |
| 19 | commit / dirty / 产物路径完整记录 | ✔ `dirty_worktree=false`、`git.git_commit` 与 JSON/PT 路径写入产物 |
| 20 | 旧产物不覆盖 | ✔ 新 tag `div8209_sla`；上一轮 `div8209_sqk_*`、历史 `div8209_l0ops_*` 全部只读 |

## 10. 已证实 / 支持性推断 / 未证实（严格分开）

**已证实（本 run 实测，逐位）**
1. **serial per-row o_proj 成功闭环**：`heads_merged` 与 `o_proj` 决策行都与 S0 **逐位一致**；
   BQ 的 2 个差异（坐标 `[153]`、`[1519]`）被**完全消除**。
2. **attention 子块完全闭合**：QK、causal mask 后、AV、head merge、`o_proj`、`attention_residual`、
   `post_attention_rmsnorm` 全部 `n_diff=0`（BQ 时残差/post-norm 还各差 1）。
3. **L0 输出差异单调下降但不为零**：`B0=111 → BQ=61 → BQO=21`（`max|Δ|` 恒 1 bf16 ULP）。
4. **`gate_proj` 是当前首个持续边界**：输入 `post_attention_rmsnorm` 逐位相同、输出 4 个坐标不同，
   其后每个算子输出都不同；**`up_proj` 也是独立边界**（同输入、输出 6 个坐标不同）。
5. `q_proj` 的差异在 `rope_q` 处回到 0；`softmax` 的差异在 `attn_times_v` 处回到 0。
6. 第 0 层输入行逐位相同、输出行不同；第 1–27 层输入**已经不同**。
7. 最终 **token 恢复（323 = 323）**但 logits **未逐位一致**（`n_diff=129412`）。
8. 20 项控制全部通过；五臂 × 2 pass 逐位一致；S0/B0 与历史、BQ 与上一轮产物**逐位一致**。

**支持性推断（由上面的数支持，但不是本轮新测到的机制）**
* `gate_proj`/`up_proj`/`o_proj`/`q_proj` 都是 bf16 `nn.Linear`，两侧只差 **M=1 vs M=4**
  ⇒ 与上一轮"**M 维度**决定执行几何敏感性"的结论**同源**；**不**得升级为"实现缺陷"。
* L0 输出的 21 个坐标差异全部为 1 bf16 ULP、且 attention 子块已逐位恢复
  ⇒ 它们只能来自 MLP 段（`gate_proj`/`up_proj` 及其下游），但**具体传播路径**本轮只给出节点级 `n_diff`。
* BQO 的 logits `n_diff` 比 BQ 略大（129412 vs 125439）—— 这是**观测**，
  与"对齐更多算子不保证 logits 单调靠近"一致，**本轮不解释**。

**未证实（本轮没有测、也不声称）**
1. `gate_proj`/`up_proj` 内部具体是哪种 kernel/tile/归约顺序（**本轮不做 profiler**）。
2. `softmax` 的差异是被 `attn.to(bf16)` 回转吸收还是被 AV 归约吸收（仍未分离）。
3. L1–L27 是否有新的独立边界（判据在这些层**无法触发**）。
4. `silu`/`gated`/`down`/`mlp_residual` 是否也含独立来源（输入已不同 ⇒ 无法判定）。
5. 323 与 659 谁是数学真值（**未**做 FP32）；token 恢复是否在别的输入/位置同样成立。
6. B6 是否通过（**未跑 B6**；token 恢复 ≠ B6 通过）。
7. 对别的位置/长度/family/层/模型的外推（**不外推**）。

## 11. B6 状态

**未改变**：本轮**没有**跑正式 B6/acceptance/pilot/final，**没有**修改或放宽任何正确性门槛，
**没有**把两个诊断分支设为默认开启（它们默认关闭、只对显式注册的模块实例生效、退出即恢复，
并由控制 13 证明"关闭后生产路径逐位不变"）。B6
（`SpecDec ids == target-only greedy ids` 且全有限、多轮确定）**仍未通过** ⇒
**acceptance / pilot / final 仍然禁止**。本轮只是**类别 2**：token 恢复、数值路径未对齐。

## 12. 代码改动（原理 / 输入 → 输出 / 必要性）

| 改动 | 原理 | 输入 → 输出 | 为什么必要 |
| --- | --- | --- | --- |
| **新增** `specdec/engine/serial_align.py`（CPU 门 **19/0**） | 纯逻辑：`module_facts`（从实际模块读）、逐行 `M=1` 调用 + `cat`、链持久性（首个/首个持续）、H2–H5 分类、MLP 逐 projection 报告 | 模块 + 张量 → 结构化结果 | 干预口径与判据必须可独立 CPU 复算 |
| **改** `model_self.py`：新增默认 `None` 的 `_SERIAL_LINEAR` 注册表与 `set/clear/serial_linear_state/serial_linear_for/serial_linear_calls` + 在 `op = self.o_proj(out)` 处加一处**分支** | 关闭时只多一次身份判断（`_SERIAL_LINEAR is None`），**原样的生产调用**照常执行；开启时只替换被注册模块 | `with serial_linear_for(L0.o_proj)` → 逐行 M=1 的 `op` | 需要与 QK 分支同构的**默认关闭、实例级、异常安全**干预；不复制 Attention/DecoderLayer |
| **新增** `specdec/engine/serial_align_l0_f13_p8209.py`（CPU 门 **14/0**） | 五臂 × 2 pass 串行 + 三层只读捕获 + 两个分支的逐行原始记录 + QK 哨兵 + 20 项控制 + **捕获后立即落盘** | 模型 + `H(8208)` + 固定 token → JSON/PT | 本轮唯一实验；落盘提前是缺陷 ① 的直接教训 |
| 改 `scripts/div8209_preflight_cpu.py`（+1 节） | 加载前核对五臂定义/两个开关组合/默认关闭/API/上一轮 BQ 参考/H | 固定路径产物 → 断言 | 避免再白付一次模型加载 |
| **新增** `scripts/serialalign8209_verdict_cpu.py`（只读） | 从原始张量独立复算 H1–H5、逐算子链（首个/首个持续）、MLP 逐 projection 边界、28 层轨迹、serial o_proj 自检、13 项核对 | `.json`+`.pt` → 汇总 JSON | 结论必须能从原始张量独立复算；**摘要错误不靠重跑 GPU 修** |
| **未改** | Q/K/V projection、softmax、AV、MLP、权重、mask、RoPE、cache、argmax、后端、proposal、门槛 | — | 本轮只做诊断性干预 |

**缺陷记录（全部保留，不覆盖）**

| # | 现象 | 原因 | 影响 / 处理 |
| --- | --- | --- | --- |
| ① | 首跑 `stage=crashed`：`too many indices for tensor of dimension 3`，发生在 **10 次前向全部跑完之后**的比较阶段 | `_cmp_all_B` 里我用 **dict 字面量**做口径分派（`{"row": t[0,0], "heads": t[0,:,0,:], …}[kind]`）——**四个分支都会被求值**，对 3 维张量取 4 个索引直接 `IndexError` | 白付 1 次模型加载；该 run 的 JSON **保留**（`div8209_sla_20261002T192404Z.json`，**无 `.pt`**）。两处修正：① 改用 `if/elif` 只求值命中的分支（并写进 docstring 防复发）；② **把原始张量落盘提前到"捕获之后、任何比较之前"**（commit `39b9342`） |
| ② | 只读汇总首跑 `KeyError: 'oproj'` | 我把 `h2` 的键写成 `o_proj` 却在核对处取 `h2["oproj"]` | 只影响汇总脚本，已修（重跑通过，未消耗 GPU） |

## 13. 唯一最小下一步（**未执行**）

> 把**同一个**逐行 M=1 的 `nn.Linear` 干预，从 `QK`、`o_proj` 继续推广到 **L0 MLP 的两个输入投影**
> `gate_proj` 与 `up_proj`（它们**共用同一个输入** `post_attention_rmsnorm`，且都已满足
> "输入逐位相同、输出不同"）。目标是判定：
> ① 把这两个投影也按行以 M=1 调用后，**L0 输出是否完全恢复**；
> ② 若仍不恢复，沿同一条链继续给出**下一个**持续边界（预计落在 `down_proj`，但**不得提前下结论**）。
>
> **判据（二值）**：L0 输出**逐位一致** ⇒ 本例 L0 的持续形状敏感性由 `QK + o_proj + gate_proj + up_proj`
> 四处构成；否则继续按"输入逐位相同、输出不同、且持续到出口"的规则定位下一条边界。
>
> **纪律**：本轮**不执行**该实验；不做 profiler、不切后端、不改生产默认路径、不做 FP32、
> **不跑 B6/acceptance/pilot/final**、不外推；新产物用**新 tag**，旧产物（含本轮失败 run）**一律保留**。

---

## 14. 后续轮次指针（**已完成，本文档提出的 L0 闭环到此结束**）

本文档 §13 的"把同一机制推广到 `gate_proj` 与 `up_proj`"已在下一轮执行，并**一次做到 L0 全闭合**：
> `docs/div8209_serial_l0_closure_2026-10-02.md`（报告 §9.59 / 台账 §4.74）——
> 六臂 **S0/B0/BQO/BQOGU/BQOGUD/S1**，把 L0 MLP 的 `gate_proj`/`up_proj`/`down_proj` 也按行以 **M=1**
> 调用同一个 `nn.Linear`（多目标共用同一注册表，对象身份匹配）。
>
> 结果：**H2 通过**（gate/up/SiLU/gated 在 BQOGU 中逐位恢复）、**H3 实测判定 `down_proj` 为独立边界**
> （BQOGU 中其输入逐位相同、输出仍差 12）、**H4：L0 完全闭合** ——
> **差异轨迹 111 → 21 → 5 → 0**，BQOGUD 的 L0 输出与 S0 **逐位一致**（sha16 `ff7c4133bea45a00`）；
> **H5：后续第一处重新分歧层 = L1**（输入逐位相同、输出差 222；按约定**不进入**其内部）；
> **H6 = 类别 2**：token 恢复 323=323（但恢复来源是 S0 在 659/323 上的**并列取舍**，margin 0.0），
> logits 仍未对齐且 `n_diff` **非单调**（129412 → 122164 → 126298）。
> **B6 状态仍未改变。** 该轮按约定结束了 L0 逐算子追踪，下一步转为**项目级决策**（是否把"verify 路径
> GEMM 逐行 M=1"立项为工程选项，或按现有门槛推进）。
