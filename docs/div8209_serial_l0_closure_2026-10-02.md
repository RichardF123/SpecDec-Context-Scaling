# 位置 8209：L0 全部持续 GEMM 边界闭环完成 —— **L0 输出逐位恢复**，后续第一处分歧在 **L1**（2026-10-02）

> **本轮是 L0 逐算子闭环的最后一轮**。在保留 L0 `serial-causal QK` + `serial o_proj` 的基础上，
> 把 L0 MLP 的 `gate_proj` / `up_proj` / `down_proj` 也按行以 **M=1** 调用**同一个 `nn.Linear`**，
> 判定 MLP 中间状态、`down_proj` 边界、L0 输出、后续层与最终 logits/token。
>
> **不做**：不进入 L1–L27 的算子级诊断、不加更多逐层干预、不做 profiler/FP32、不切后端、不优化性能、
> 不把诊断开关设为默认、**不跑 B6**、不启动 acceptance/pilot/final、不放宽 token 门槛、不用 ULP/margin 豁免、
> 不外推到其他位置/family/模型、不覆盖旧产物。**B6 状态未改变。**

## 0. 结论

**L0 已完全闭合（逐位）**，后续第一处重新分歧在 **L1**；最终 token 恢复、logits 仍未对齐。

| 量 | B0 | BQO | BQOGU | **BQOGUD** |
| --- | --- | --- | --- | --- |
| L0 输出 `n_diff`（vs S0） | 111 | 21 | 5 | **0（逐位一致，sha16 `ff7c4133bea45a00` = 历史 S0）** |
| `gate_proj` / `up_proj` 决策行 | 247 / 318 | 4 / 6 | **0 / 0** | **0 / 0** |
| `gated_product` | 503 | 8 | **0** | **0** |
| `down_proj` 决策行 | 203 | 43 | 12 | **0** |
| 最终 logits `n_diff` | 136478 | 129412 | **122164** | 126298 |
| token（argmax） | 659（类别 3） | 323（类别 2） | 323（类别 2） | 323（类别 2） |

* **H2 通过**：在 BQOGU 中 `post_attention_rmsnorm`、`gate_proj`、`up_proj`、`SiLU`、`gated_product`
  决策行**全部逐位恢复**（BQO 时的 4/6/2/8 个差异被完全消除）⇒ 多目标 serial Linear 真实生效。
* **H3 通过**：在 BQOGU 中 `gated_product` 逐位相同、`down_proj` 输出仍有 12 个坐标不同
  ⇒ **`down_proj` 是新的独立 shape-sensitive 边界**（这是**实测判定**，不是预期）。
* **H4：L0 完全闭合** —— BQOGUD 的 L0 输出与 S0 **逐位一致**（`n_diff=0`），且其 **L1 输入与 S0 的 L1 输入
  逐位一致**。
* **H5**：L0 之后**第一处**「输入逐位相同、输出不同」的层 = **L1**（输出 `n_diff=222`）。
  按本轮约定：**只记录这一层，不进入其内部**。
* **H6 = 类别 2**（三个干预臂都是）：`token 323 = 323` 但 logits 未逐位一致；**且 `n_diff` 不单调**
  （BQO 129412 → BQOGU **122164**（更小）→ BQOGUD 126298（更大））。
* **重要细节（如实报告）**：S0 的 logits 在 `659` 与 `323` 上**并列**（`margin = 0.0`，`argmax` 取 323）；
  B0 的 margin 是 `0.125`（659 胜出）；三个干预臂的 margin 又回到 `0.0`。
  ⇒ 本例的"token 恢复"是**并列关系与 argmax 取舍恢复一致**，**不是** margin 变大 —— 不得写成"更稳"。

## 1. 命令、commit、dirty 状态与产物

```powershell
cd D:\RealDSH\specdec
$env:PYTHONIOENCODING='utf-8'
# ① 纯逻辑 CPU 门（逐行 Linear / 多目标 / MLP 三个投影不同维度 / 单模块回归 / 链持久性 / H2–H6）
.\.venv\Scripts\python.exe -X utf8 -m specdec.engine.serial_align
# ② 工具 CPU 门（微型模型真实 forward：六臂 / 多目标注册 / 身份隔离 / 嵌套 / 异常恢复 / 落盘）
$env:SML_MODE='selftest'; .\.venv\Scripts\python.exe -X utf8 -m specdec.engine.serial_mlp_l0_f13_p8209
Remove-Item Env:SML_MODE
# ③ 飞行前检查（新增一节：六臂定义/开关组合/四处生产分支/上一轮参考/H）
.\.venv\Scripts\python.exe -X utf8 scripts\div8209_preflight_cpu.py
# ④ 一次正式 GPU 运行（工具**先提交**，保证 worktree clean）：六臂 × 2 pass + 关闭对照
$env:SML_TAG='div8209_sml'; $env:SML_LAYER='0'; $env:SML_PRE='8208'; $env:SML_TARGET='8209'; $env:SML_K='4'
$env:SML_FEED='20642'; $env:SML_CANDS='20642,659,576,4531'; $env:SML_REPEAT='2'
$env:SML_TARGET_ROPE='cpu_fp32'; $env:SML_POISON_CHECK='1'; $env:SML_CAPTURE_OFF_CHECK='1'
.\.venv\Scripts\python.exe -X utf8 -m specdec.engine.serial_mlp_l0_f13_p8209
# ⑤ 只读 CPU 汇总（从原始张量独立复算 H1–H6 + 变化表 + MLP 边界 + 28 层 + 控制）
.\.venv\Scripts\python.exe -X utf8 scripts\serialmlp8209_verdict_cpu.py
```

* **commit / dirty**：正式 run 记录 `git_commit = c4772aeb3c5237a36c12bf4d6df984c9247811b1`、
  **`dirty_worktree = false`**（干预代码与工具**先提交**再跑 GPU）；控制 22 记录 git 与产物路径。
* **产物（新 tag，不覆盖旧产物）**：
  * `output/g5/attn/div8209_sml_20261002T201740Z.json` + `.pt`（`stage=complete`）
    —— `.pt` 含六臂的 L0 探针原始张量（31 项）、28 层输入/输出决策行、最终 RMSNorm 输入/输出行、
    完整词表 logits、**三个诊断分支的逐行原始记录**、哨兵结果、L0 轮前缓存、gate/up/down/o_proj 模块事实；
  * CPU 汇总：`output/g5/attn/serialmlp8209_verdict_cpu_20261002T202259Z.json`（**16 项核对全 True**）。
* **固定配置（逐项沿用有效实验）**：F13、ctx 8192、chunk 2048、k=4、`base_bf16`、**all-eager**、
  BF16、`qk_fp32=False`、`prob_fp32=False`、`allow_bf16_reduced_precision_reduction=True`、
  `allow_tf32=False`（全程）、target CPU-FP32 `inv_freq`、S=`[20642]`、B=`[20642,659,576,4531]`、
  只加载 target（**从不加载 draft**）、不自由生成、不启用 cuDNN/SDPA/FlashAttention/协作者 PR、
  同一份 `H(8208)`（`0440d98c533ae3ed`，纯切片 8237→8208，与重建捕获 56/56 逐位相同）、
  权重 338/338 `bitmatch=True`、RoPE 作用域还原 `ok=True`。
* **本轮 GPU 进程 1 次**（一次跑通，无失败 run）。

## 2. 六个 arm 的精确配置

| arm | 输入 | phase | serial QK | serial o_proj | serial MLP | 实测 applied（Linear） |
| --- | --- | --- | --- | --- | --- | --- |
| **S0** | `[20642]` | decode | 关 | 关 | 关 | 0 / 0 |
| **B0** | `[20642,659,576,4531]` | verify | 关 | 关 | 关 | 0 / 0 |
| **BQO** | 同 B0 | verify | 开 | 开 | 关 | 1 / 1 |
| **BQOGU** | 同 B0 | verify | 开 | 开 | **gate + up** | 1 / 3 |
| **BQOGUD** | 同 B0 | verify | 开 | 开 | **gate + up + down** | 1 / 4 |
| **S1** | 同 S0 | decode | 开 | 开 | **gate + up + down** | 1 / 4 |

六个 arm 各自**独立克隆**同一份 `H(8208)` 的 cache、**串行**执行；每臂 **2 个独立 pass**（控制 15）；
另加"全部开关关闭后再跑 S/B"的对照（控制 16）。

**干预作用域（控制 13/14）**：一次前向里共有 **112** 个 Linear 调用点（28 层 × {o_proj, gate, up, down}）。
BQOGUD 注册 L0 的 **4** 个实例 ⇒ `n_applied = 4`、`n_seen = 108`，且
`seen ∪ applied` **恰好覆盖全部 112 个调用点**（对象身份匹配，**不是**类名匹配）。
L1–L27 的对应模块进入分支后只执行一条计数语句，随后仍走**原样的一次调用**。

## 3. gate/up/down 的真实模块事实（从实际模块读取）

| 模块 | 类名 | in_features | out_features | bias | weight shape | weight dtype | 逐行输入形状（实测） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `gate_proj` | `Linear` | 1536 | **8960** | `False` | `[8960, 1536]` | `torch.bfloat16` | `[1,1,1536] × 4` |
| `up_proj` | `Linear` | 1536 | **8960** | `False` | `[8960, 1536]` | `torch.bfloat16` | `[1,1,1536] × 4` |
| `down_proj` | `Linear` | **8960** | 1536 | `False` | `[1536, 8960]` | `torch.bfloat16` | `[1,1,8960] × 4` |
| `o_proj`（对照） | `Linear` | 1536 | 1536 | `False` | `[1536, 1536]` | `torch.bfloat16` | `[1,1,1536] × 4` |

* 维度关系在 run 内被断言：`gate.out == up.out`、`down.in == gate.out`、`down.out == gate.in`（控制 10 系列）。
* 权重 `data_ptr`、模块 `module_id`、dtype 在 run 前后**不变**（控制 10/18）。
* `post_attention_rmsnorm` 输出形状 `[1,4,1536]`（B 臂）⇒ gate/up 的输入；`gated_product` 形状 `[1,4,8960]`
  ⇒ down 的输入。

**serial Linear 的数据流**（与规格一致）：
```python
rows = []
for i in range(x.shape[1]):
    yi = module(x[:, i:i+1, :])      # 每行**实际**以 [1,1,Din] 调用**原来的同一个 nn.Linear**
    rows.append(yi)
y = torch.cat(rows, dim=1)           # 拼接只组合行，不做任何额外算术
```
不先算 M=4 再切片、不改 matmul/einsum/FP32/kernel；dtype/权重/bias/模块实例不变。

## 4. H1–H6 结果

| 判据 | 结果 | 关键数字 |
| --- | --- | --- |
| **H1** 历史结果复现 | **通过** | S0 ≡ 历史 S0（final/L0 输出/QK/x0 全 `exact`）；B0 ≡ 历史 B0（同四项）；**BQO ≡ 上一轮**（29 节点 + 28 层行 + final + L0 输出全逐位）；**S1 ≡ S0**（节点 + 层行 + final；QK 1 次、Linear 4 次调用） |
| **H2** gate/up 局部闭环 | **通过** | `post_attention_rmsnorm` / `gate_proj` / `up_proj` / `SiLU` / `gated_product` 决策行**全部逐位一致**（BQO 对照分别差 0/4/6/2/8） |
| **H3** down_proj 独立边界 | **是** | 在 BQOGU 中：`gated_product` 逐位相同、`down_proj` `n_diff=12`、`max|Δ|=0.00195312` ⇒ **独立边界**（BQO 对照 43） |
| **H4** 完整 L0 是否恢复 | **完全恢复** | BQOGUD 的 L0 输出与 S0 `n_diff=0`、sha16 同为 `ff7c4133bea45a00`；**L1 输入也与 S0 逐位一致** |
| **H5** 后续第一处分歧 | **L1** | L0：输入 `0`/输出 `0`；**L1：输入 `0`、输出 `222`**；L2 起输入已不同 ⇒ 不再判独立（**本轮不进入 L1 内部**） |
| **H6** 最终结果 | **类别 2**（三臂一致） | 见 §7 |

## 5. L0 全部算子边界表（S0 视角的 `n_diff`）

| 算子 | 输入全同？ | B0 | BQO | BQOGU | **BQOGUD** |
| --- | --- | --- | --- | --- | --- |
| `input_rmsnorm` | ✔ | 0 | 0 | 0 | **0** |
| `q_proj` | ✔ | 2 | 2 | 2 | **2**（瞬时，见下） |
| `q_reshape` | ✘ | 2 | 2 | 2 | 2 |
| `rope_q` | ✘ | **0** | 0 | 0 | **0**（吸收） |
| `k_proj`/`k_reshape`/`rope_k`/`kv_concat_k` | ✔ | 0 | 0 | 0 | **0** |
| `v_proj`/`v_reshape`/`kv_concat_v` | ✔ | 0 | 0 | 0 | **0** |
| `qk_matmul_scale` | ✔ | **3** | **0** | 0 | **0** |
| `causal_mask` | ✔ | 3 | 0 | 0 | **0** |
| `softmax` | ✔ | 55343 | 38924 | 38924 | **38924**（瞬时，见下） |
| `attn_times_v` | ✘ | **2** | 0 | 0 | **0**（吸收） |
| `head_merge` | ✔ | 2 | 0 | 0 | **0** |
| `o_proj` | ✔ | **9** | **2** | 0 | **0** |
| `attention_residual_add` | ✔ | 4 | 1 | 0 | **0** |
| `post_attn_rmsnorm` | ✔ | 4 | 1 | 0 | **0** |
| `gate_proj` | ✔ | **247** | **4** | **0** | **0** |
| `up_proj` | ✔ | **318** | **6** | **0** | **0** |
| `silu_activation` | ✘(BQO 起 ✔) | 232 | 2 | 0 | **0** |
| `gated_product` | ✘(BQO 起 ✔) | 503 | 8 | **0** | **0** |
| `down_proj` | ✘(BQO 起 ✔) | 203 | 43 | **12** | **0** |
| **`mlp_residual_add`（= L0 输出）** | ✘(BQO 起 ✔) | **111** | **21** | **5** | **0** |

**在最终的 BQOGUD 里，只有两处「输入逐位相同、输出不同」的算子**仍是：
* `q_proj`（2 个坐标；在 `rope_q` 处回到 0 ⇒ **瞬时**，已被实测确认）；
* `softmax`（38924 个坐标；在 `attn_times_v` 处回到 0 ⇒ **瞬时**；生产在该段有 `attn.to(bf16)`，
  **具体是被 bf16 回转还是被 AV 归约吸收，本轮仍未分离**）。
从 `rope_q` 到 L0 输出，**每个算子的决策行都与 S0 逐位一致**。

## 6. B0 → BQO → BQOGU → BQOGUD 的差异变化（逐算子，S0 视角）

见 §5 的四列。要点：
* **QK 段**（`qk_matmul_scale`、`causal_mask`）：3 → 0 → 0 → 0（第一轮干预的贡献）；
* **attention 输出段**（`attn_times_v`、`head_merge`、`o_proj`、残差、post-norm）：2/2/9/4/4 → 0/0/0/0/0
  （第二轮的贡献；`o_proj` 在 BQO 时还差 2）；
* **MLP 段**（`gate_proj`、`up_proj`、`silu`、`gated`、`down`）：247/318/232/503/203 → 68…→ 本轮逐级清零；
  **`down_proj` 是唯一在 BQOGU 中仍有差异（12）的 MLP 算子** ⇒ 本轮第三/第四次干预的贡献；
* **L0 输出**：**111 → 21 → 5 → 0**。

## 7. 最终 logits / token / top-2 对照

| arm | 类别 | logits exact | `n_diff` | `max|Δ|` | `mean|Δ|` | argmax | top-2（token） | margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | — | — | — | — | — | **323** | `[659, 323]` | **0.0** |
| B0 | **3** | ✘ | 136478 | 0.25 | 0.0410212 | 659 | `[659, 323]` | 0.125 |
| BQO | **2** | ✘ | 129412 | 0.1875 | 0.0319965 | **323** | `[659, 323]` | 0.0 |
| BQOGU | **2** | ✘ | **122164** | 0.15625 | 0.0245593 | **323** | `[659, 323]` | 0.0 |
| BQOGUD | **2** | ✘ | 126298 | 0.1875 | 0.0300853 | **323** | `[659, 323]` | 0.0 |

* **不得假定单调**：`n_diff` 是 **129412 → 122164 → 126298**，即"对齐更多算子"**没有**让 logits 单调靠近 S0；
  这是**观测**，本轮**不解释**其机制。
* **token 恢复的来源必须说清**：S0 在 `659`/`323` 上**并列**（margin 0.0），`argmax` 取 323；B0 时 margin 0.125 ⇒ 659；
  三个干预臂的 margin 回到 0.0 ⇒ argmax 又取 323。**这是取舍关系恢复一致，不是"更稳"或"余量变大"**。
* 最终 RMSNorm：输出行 `S0 vs BQOGUD` `n_diff=1270`、其输入行 `1321`（仍未对齐）。

## 8. 24 项必需控制（全部通过）

| # | 控制 | 结果 |
| --- | --- | --- |
| 1 | S0 逐位复现历史 S | ✔ final/L0 输出/QK/x0 全 `exact` |
| 2 | B0 逐位复现历史 B | ✔ 同四项全 `exact` |
| 3 | BQO 逐位复现上一轮有效结果 | ✔ 29 节点 + 28 层行 + final + L0 输出全逐位 |
| 4 | S1 与 S0 逐位一致 | ✔ 节点 + 28 层行 + final |
| 5 | BQOGU/BQOGUD 的 QK row0 与 S0 一致 | ✔ 两臂都 `exact` |
| 6 | attention 子块决策行继续与 S0 一致 | ✔ 7 个节点（QK pre/post、AV、head merge、o_proj、残差、post-norm）在两臂全部 `exact` |
| 7 | `gate_proj` 每行实际 `[1,1,1536]` | ✔ `[[1,1,1536]]×4` |
| 8 | `up_proj` 每行实际 `[1,1,1536]` | ✔ `[[1,1,1536]]×4` |
| 9 | `down_proj` 每行实际 `[1,1,8960]` | ✔ `[[1,1,8960]]×4` |
| 10 | 三个 Linear 用原模块/原权重/原 bias/原 dtype | ✔ `module_id` 一致、`weight_data_ptr` 不变、`has_bias=False`、全 `bf16` |
| 11 | gate/up 输出在 BQOGU 中与 S0 决策行一致 | ✔ `exact`（含 SiLU、gated） |
| 12 | down_proj 输出在 BQOGUD 中与 S0 决策行一致 | ✔ `exact` |
| 13 | 干预只作用于 L0 指定实例 | ✔ `applied = {o_proj, gate, up, down}（L0）`、`n_applied=4` |
| 14 | L1–L27 对应模块保持普通路径 | ✔ `n_seen=108`、`seen ∪ applied` = 全部 112 个调用点 |
| 15 | 两次 BQOGUD 重复逐位一致 | ✔ 六臂 × 2 pass 的节点/最终行/层行全部逐位相同 |
| 16 | 所有开关关闭后生产路径逐位不变 | ✔ S0/B0 逐位复现历史；关闭后再跑 S/B 与 S0/B0 逐位一致 |
| 17 | context/hook 全部恢复 | ✔ 探针关、QK 关、Linear 关、28 层与 `norm` 钩子全空 |
| 18 | cache、输入、权重未被修改 | ✔ H 逐层摘要前后相同；权重 data_ptr 与摘要稳定；Norm 与 7 个投影权重稳定；L0 轮前缓存长度 8208（六臂相同） |
| 19 | 所有非 mask 张量有限 | ✔ 比较切片与六臂 logits 全有限；唯一非有限元素是 B 路径 mask 后的 `-inf`（BQOGUD 与 B0 各 72、S0 为 0） |
| 20 | `allow_tf32=False` | ✔ 前后都是 `False` |
| 21 | 不加载 draft | ✔ `loaded_models=["target"]`、`draft_loaded=False` |
| 22 | commit/dirty/产物路径完整 | ✔ `c4772ae…`、`dirty_worktree=false`、JSON+PT 路径写入产物 |
| 23 | 旧产物不覆盖 | ✔ 新 tag `div8209_sml`；`div8209_sla_*`/`div8209_sqk_*`/`div8209_l0ops_*` 只读 |
| 24 | 原始张量在任何摘要/打印前先落盘 | ✔ `.pt` 的 mtime **早于** JSON（`pt_written_first=True`）；落盘发生在六臂捕获 + 重复一致性之后、**任何比较之前** |

## 9. 已证实 / 支持性推断 / 未证实（严格分开）

**已证实（本 run 实测，逐位）**
1. **L0 完全闭合**：BQOGUD 的 L0 输出与 S0 决策行**逐位一致**（`n_diff=0`，sha16 `ff7c4133bea45a00`），
   且其 L1 输入与 S0 的 L1 输入**逐位一致**。
2. **gate/up/SiLU/gated 在 BQOGU 中逐位恢复**（BQO 的 4/6/2/8 个差异全部消除）。
3. **`down_proj` 是独立边界**：BQOGU 中其输入 `gated_product` 逐位相同、输出仍有 12 个坐标不同；
   在 BQOGUD 中该差异归零。
4. **L0 内的持久 GEMM 边界齐全**：`qk_matmul_scale`（第一轮）、`o_proj`（第二轮）、
   `gate_proj`/`up_proj`（本轮 H2）、`down_proj`（本轮 H3）。
5. **`q_proj` 与 `softmax` 的差异是瞬时的**（分别在 `rope_q`、`attn_times_v` 处回到 0）。
6. **后续第一处重新分歧层 = L1**（输入逐位相同、输出 `n_diff=222`）；L2 起输入已不同。
7. **最终 token 恢复（323 = 323）但 logits 未逐位一致**（类别 2；`n_diff` 122164–129412 之间**非单调**）。
8. 24 项控制全部通过；六臂 × 2 pass 逐位一致；S0/B0 与历史、BQO 与上一轮产物逐位一致。

**支持性推断（由上面的数支持，但不是本轮新测到的机制）**
* 四个持久边界都是 bf16 GEMM，两侧只差 **M=1 vs M=4**（QK 的 M、o_proj 的 M、gate/up/down 的 M）
  ⇒ 与"**M 维度**决定执行几何敏感性"的结论**同源**；不得升级为"实现缺陷"。
* L0 输出在四处几何对齐后逐位恢复 ⇒ 本例 L0 内的**全部**持久数值分歧都能由这四处解释
  （但这**不**意味着 L1 及以后只有同类机制，见"未证实"）。
* `softmax` 的差异被 `attn.to(bf16)`/AV 那一段吸收，**这与概率被舍入到 bf16 后差异消失相符**。

**未证实（本轮没有测、也不声称）**
1. **L1 内部**的具体边界（按本轮约定**不进入** L1–L27 算子级诊断）。
2. 各 GEMM 内部的具体 kernel/tile/归约顺序（**未做 profiler**）。
3. `softmax` 差异究竟被 bf16 回转还是被 AV 归约吸收（仍未分离）。
4. 323 与 659 谁是数学真值（**未**做 FP32）；S0 的并列（margin 0.0）在别的输入下是否同样出现。
5. B6 是否通过（**未跑 B6**；token 恢复 ≠ B6 通过）。
6. 对别的位置/长度/family/模型的外推（**不外推**）。

## 10. 代码改动（原理 / 输入 → 输出 / 必要性）

| 改动 | 原理 | 输入 → 输出 | 为什么必要 |
| --- | --- | --- | --- |
| **改** `model_self.py`：在 `MLP.forward` 的 `gate_proj`/`up_proj`/`down_proj` 三处加**同一**默认关闭分支 | 关闭时每处只多一次身份判断（`_SERIAL_LINEAR is None`）⇒ **原样的生产调用**执行；开启时只替换被注册实例 | `with serial_linear_for(L0.mlp.gate_proj, …)` → 逐行 M=1 的三个投影输出 | 需要把干预推广到 MLP；三处生产调用各只出现一次 |
| **改** `model_self.serial_linear_state`：`applied` 增加 `module_label` | 纯附加字段 | 状态 → 状态（多一个字段） | 让记录能区分 gate/up/down（同一 `nn.Linear` 类型） |
| **改** `serial_align.py`：新增多目标/不同 in-out 维度的 CPU 回归（gate/up `h→i`、down `i→h`、单模块路径逐位不变） | 纯测试增强 | — | 规格要求"保持旧的单模块行为逐位不变，并增加 CPU 回归" |
| **新增** `specdec/engine/serial_mlp_l0_f13_p8209.py`（CPU 门 **12/0**） | 六臂 × 2 pass + 三层只读捕获 + 三个分支的逐行原始记录 + **早落盘** + 24 项控制 | 模型 + `H(8208)` + 固定 token → JSON/PT | 本轮唯一实验 |
| **改** `scripts/div8209_preflight_cpu.py`（+1 节） | 加载前核对六臂/开关组合/四处生产分支/上一轮参考/H | 固定路径产物 → 断言 | 避免白付加载 |
| **新增** `scripts/serialmlp8209_verdict_cpu.py`（只读） | 从原始张量独立复算 H1–H6、逐算子变化表、MLP 边界、28 层轨迹、serial 自检与 16 项核对 | `.json`+`.pt` → 汇总 JSON | 结论必须能从原始张量独立复算 |
| **改** 上一轮工具 `serial_align_l0_f13_p8209.py` 的**两处自测断言** | 这两处原本写死"`n_seen == 1`"（当时每层只有 1 个被拦调用点）。本轮在 MLP 加了三处同类分支后，调用点变为**每层 4 个** ⇒ 原断言**过期**（表现为 2 个假失败）。改为**从模型结构推导**（`4 × n_layers`、`seen ∪ applied` = 全部调用点） | — | 这是**加强**而非放宽：今后再加分支只需改一处推导函数；**该工具的数值路径未改**，其旧产物仍成立 —— 本轮控制 3 已逐位复核 BQO ≡ 上一轮产物 |
| **未改** | Q/K/V projection、softmax、AV、SiLU、乘法、残差、mask、RoPE、cache、argmax、后端、门槛 | — | 本轮只做诊断性干预 |

**缺陷记录（全部保留；本轮 GPU 侧无失败 run）**

| # | 现象 | 原因 | 影响 / 处理 |
| --- | --- | --- | --- |
| ① | 工具 JSON **未写完整逐节点比较表**（只写了 H2/H3/H4 等派生字段） | 我在 `_run` 里算了 `n_s0b0/n_s0bqo/n_s0bqogu/n_s0bqogud` 却漏了 `rec["node_compare"] = …` | **未损失数据**：原始张量在任何比较之前已落盘，只读汇总从 `.pt` **独立复算**出全部逐算子表；汇总器改为与 JSON 中**确实存在**的字段（`H4.persistence_bqogud` 的 25 项、H2/H3 对照、trajectory）逐项核对，并**在文档中记录**该字段缺失 |
| ② | 三个**实现**缺陷在 CPU 门阶段被拦下（未消耗 GPU）：`_row` 定义在 `_run` 内部却被自测调用（`NameError`）；自测里把**原始**分支记录当 CPU 版记录取 `module_id`（`KeyError`）；落盘断言把 S 臂的 1 行期望写成 4 行 | 自测与工具记录结构不一致 | 全部在 GPU 之前修掉 ⇒ **零 GPU 代价**（这正是"CPU 门先过再加载模型"的价值） |
| ③ | 只读汇总首跑的 `node_json_match` 为 False | 我把 **BQOGU** 的闭环结果与 JSON 里的 **BQO 对照**相比（口径错配） | 已修（改为比 `h2["reference_BQO"]`），未消耗 GPU |

## 11. B6 状态

**未改变**：本轮**没有**跑正式 B6/acceptance/pilot/final，**没有**修改或放宽任何正确性门槛，
**没有**把三个诊断分支设为默认开启（它们默认关闭、只对显式注册的实例生效、退出即恢复，
并由控制 16 证明"关闭后生产路径逐位不变"）。B6
（`SpecDec ids == target-only greedy ids` 且全有限、多轮确定）**仍未通过** ⇒
**acceptance / pilot / final 仍然禁止**。
本轮结论是**局部（L0）逐算子闭环**：它把 L0 的持续数值分歧完整解释为四处 M 几何边界，
**但** L1 起仍有分歧、logits 仍未对齐。

## 12. 落盘顺序与产物完整性

* `_save_pt` 在**六臂捕获 + 重复一致性检查之后、任何比较/摘要/打印之前**执行
  （控制 24 用 mtime 佐证：`pt_written_first = True`）。
* `.pt` 含：六臂 × 31 个 L0 探针张量、六臂 × 28 层输入/输出决策行、六臂最终 RMSNorm 输入/输出行、
  六臂完整词表 logits、三个分支的逐行原始记录（QK 的 raw/scaled/布局；Linear 的逐行输入形状/numel/dtype）、
  哨兵结果、L0 轮前缓存、模块事实。

## 13. 唯一下一步（**项目级决策，不执行更多 L1 算子级追踪**）

> **本轮是 L0 逐算子闭环的终点。** 后续**不再**进入 L1–L27 的算子级诊断，也不再加逐层干预。
>
> **下一步是一次项目级决策**（本轮**不执行**）：在已知
> ① L0 的持续数值分歧 = 四处 **M 几何**边界（QK / o_proj / gate / up / down，全部为 bf16 GEMM 的
> `M=1 vs M=4` 执行几何差异）、
> ② 把 B/verify 路径的这些 GEMM 改成逐行 `M=1` 可以让 **本例 L0 输出逐位恢复**、
> ③ 但 token 恢复来自**并列关系的 argmax 取舍**、logits 仍未逐位一致、
> ④ L1 起仍存在同类或其它机制的分歧
> 的前提下，二选一并给出依据：
> **(A) 工程选项**：把"verify 路径的 GEMM 逐行 M=1"作为**可选运行模式**立项，先量化它的
> **性能代价**（逐行调用会把一次 M=4 的 GEMM 变成 4 次 M=1，延迟/吞吐代价必须实测）
> 与**正确性收益**（是否让 B6 的 token 一致率稳定提升），再决定是否纳入生产；
> **(B) 按现有门槛推进**：不改执行几何，直接以**未改动**的生产配置推进 B6/验收路线，
> 把本例的数值分歧记为"已定位、已知边界、不修"。
>
> **判据（二值）**：(A) 只有在"性能代价可接受 **且** 正确性收益可复现"时才立项；
> 否则走 (B)。两种选择都**不**放宽任何正确性门槛。
>
> **纪律**：本轮**不执行**该决策所需的任何实验；不做 profiler、不切后端、不改生产默认路径、
> 不做 FP32、**不跑 B6/acceptance/pilot/final**、不外推。
