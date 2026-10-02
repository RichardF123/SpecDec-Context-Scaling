# 位置 8209：L0 内部首个分歧算子边界 = **`q_proj`**（2026-10-02）

> **本轮唯一目标**：沿 L0（`layers[0]`，零基）的**真实生产数据流**做**只读**捕获，
> 找出**第一个**满足「该算子的**所有**输入逐位相同、而输出首次不同」的算子边界。
> **只定位算子**：不修复、不改生产实现/公式/dtype/mask/RoPE/attention/cache/linear/argmax/后端、
> 不做全模型 FP32、不启动 B6 或正式矩阵、不切换后端、不重跑大模型做正式捕获（硬失败不自动重试）、
> 新产物使用新 tag、旧产物**未覆盖**。**B6 状态未改变。**

## 0. 结论（一句话 + 关键数字）

**`first_divergence_operator = q_proj`**（L0 的 `self_attn.q_proj`，即 `q_linear`）：
它的输入（`input RMSNorm` 输出）**逐位相同**、权重是**同一张量**，而输出**有 2 / 1536 个坐标不同**，
`max|Δ| = 0.000244140625` = 该量级（|q|≈0.052）下**恰好 1 个 bf16 ULP**
（元素 1300：S `0.052001953125` vs B `0.05224609375`；`sha16 bad3c65395fe6d86` vs `99100a94e2b152c4`）

**同时满足条件的还有 `qk_matmul_scale`**（`q_rope` 与 `k_visible` 都逐位相同，scores 有 3 个坐标差
`max|Δ| = 4` = |s|≈844 处**恰好 1 个 bf16 ULP**：844 vs 848）—— 如实全部列出，
按生产顺序取第一个即 **`q_proj`**。

**按用户规定的措辞**：`q_proj` 是本案例中**最早观测到的 shape-sensitive 数值分歧边界**。
**不得**写成 PyTorch / cuBLAS / BF16 的实现缺陷 —— 要区分 kernel、归约顺序与实现错误，
还需要**独立控制**（本轮不做）。

其它关键实测（见 §3 完整表）：
* `k_proj` / `v_proj` 的输出**逐位相同**（与 q 形成对照，均为 N=256 的 GEMM）；
* 进入 RoPE 的 q **已经不逐位相同** ⇒ **RoPE 不是首次分歧边界**；且本案例里 **`q_rope` 的输出又变回逐位相同**
  （1 ULP 的 q 差异被 RoPE 的乘法＋bf16 舍入吸收）；
* 所有观测到的差异都只有 **1–2 个 bf16 ULP**（按其各自量级），**没有**粗差；
* L0 输出（`layer_output`）`n_diff=111`、`max|Δ|=0.00390625`，与上一轮全层捕获**逐位相同**。

## 1. 命令、commit、dirty 状态与产物

```powershell
cd D:\RealDSH\specdec
$env:PYTHONIOENCODING='utf-8'
# ① CPU 门（探针/切片/分类 + 工具机制）——**先过门，再加载模型**
.\.venv\Scripts\python.exe -X utf8 -m specdec.engine.op_probe
$env:L0_MODE='selftest'; .\.venv\Scripts\python.exe -X utf8 -m specdec.engine.l0_operator_f13_p8209
Remove-Item Env:L0_MODE
# ② 飞行前检查（纯 CPU：H/截断/历史/上一轮 L0 参考/链与探针表/锚点）
.\.venv\Scripts\python.exe -X utf8 scripts\div8209_preflight_cpu.py
# ③ 一次正式 GPU 捕获（诊断工具**先提交**，保证 worktree clean）
$env:L0_TAG='div8209_l0ops'; $env:L0_LAYER='0'; $env:L0_PRE='8208'; $env:L0_TARGET='8209'
$env:L0_K='4'; $env:L0_FEED='20642'; $env:L0_CANDS='20642,659,576,4531'
$env:L0_WATCH='323,659,20642,576,4531'; $env:L0_REPEAT='2'
$env:L0_TARGET_ROPE='cpu_fp32'; $env:L0_CAPTURE_OFF_CHECK='1'
.\.venv\Scripts\python.exe -X utf8 -m specdec.engine.l0_operator_f13_p8209
# ④ 只读 CPU 汇总（从原始张量独立复算 + 与 JSON/上一轮/第二次 run 逐项核对）
.\.venv\Scripts\python.exe -X utf8 scripts\l0ops8209_verdict_cpu.py
```

* **commit / dirty**：正式 run 记录 `git_commit = 175c660894ae787cd181b255f5ff7d5489970a47`、
  **`dirty_worktree = false`**（诊断工具先提交再运行）；本轮最终文档/汇总脚本见仓库 HEAD。
* **产物（新 tag，不覆盖旧产物）**：
  * `output/g5/attn/div8209_l0ops_20261002T172447Z.json` + `.pt`
    （`raw_S`/`raw_B` = 29 个节点的**完整原始 dtype 张量**、`slices` = 58 个语义切片、
    轮前缓存、x0/L0 输出/最终行、cos/sin 行；`attention_prob` 保持 **fp32**）
  * CPU 汇总：`output/g5/attn/l0ops8209_verdict_cpu_20261002T172731Z.json`
  * **保留的失败/中断 run**：`…T171831Z.json`（口径写错）、`…T171955Z.json`（打印格式写错）、
    `…T172115Z.json`（payload 口径写错）、`…T172301Z.{json,pt}`（控制 11 表达式写错；**数据完整**）
* **固定配置（逐项沿用全层实验）**：F13、ctx=8192、chunk=2048、k=4、`base_bf16`、**all-eager**、BF16、
  `qk_fp32=False`/`prob_fp32=False`、`allow_bf16_reduced_precision_reduction=True`（effective True）、
  `allow_tf32=False`、target CPU-FP32 `inv_freq`、S=`[20642]`（决策行=**最后一行**）、
  B=`[20642,659,576,4531]`（决策行=**第 0 行**）、同一份 `H(8208)`（`divcap_ar_…185503Z.pt`，
  sha256 `0440d98c533ae3ed`，**纯切片** 8237→8208，与重建捕获 `[0,8208)` **56/56**）、
  只加载 target、权重 338/338、`fused_by_phase={}`、`eager_by_phase={decode:140, verify:112}`、
  RoPE 作用域还原 `ok=True`（`n_calls=9`）。

## 2. L0 真实代码数据流 → 捕获点映射（以实际代码为准）

| # | 真实代码位置（`model_self.py`，探针行号） | 算子（语义） | 探针名 | 比较口径 |
| --- | --- | --- | --- | --- |
| 0 | `DecoderLayer.forward` L539 | L0 输入 | `x0` | `row` |
| 1 | L541（`input_layernorm`） | input RMSNorm | `input_rmsnorm` | `row` |
| 2 | L399（`self.q_proj`） | **q 投影（Linear/GEMM）** | `q_linear` | `row` |
| 3 | L401（`.view().transpose()`） | Q reshape（无算术） | `q_reshaped` | `heads` |
| 4 | L411（`apply_rope`） | RoPE(q) | `q_rope` | `heads` |
| 5 | L403 / L405 | k 投影 / K reshape | `k_linear` / `k_reshaped` | `row` / `heads` |
| 6 | L413 | RoPE(k) | `k_rope` | `heads` |
| 7 | L458（`torch.cat([cache["k"], k])`） | 历史 KV 拼接(K) | `k_visible` | `kv`（共同可见 0..8208） |
| 8 | L407 / L409 | v 投影 / V reshape | `v_linear` / `v_reshaped` | `row` / `heads` |
| 9 | L459 | 历史 KV 拼接(V) | `v_visible` | `kv` |
| 10 | L470 / L471（`kv_repeat`） | GQA 广播（只记 shape） | `k_repeated` / `v_repeated` | `shape_only` |
| 11 | L478（`torch.matmul(q, k_repᵀ)/√d`） | **QKᵀ × 缩放** | `qk_scores_pre_mask` | `scores`（B 第 0 行 × key 0..8208） |
| 12 | L487（`masked_fill(mask, -inf)`） | causal mask | `qk_scores_post_mask` | `scores` |
| 13 | L489（`torch.softmax(..., dtype=float32)`） | softmax | `attention_prob` | `scores`（**保持 fp32**） |
| 14 | L496（`torch.matmul(attn, v_rep)`） | 概率 × V | `attention_times_v` | `heads` |
| 15 | L499（`transpose/contiguous/view`） | head merge | `heads_merged` | `row` |
| 16 | L526（`self.o_proj`） | o 投影 | `o_proj` | `row` |
| 17 | L545（`x + atten_out`） | attention 残差加 | `attention_residual` | `row` |
| 18 | L547（`post_attention_layernorm`） | post-attention RMSNorm | `post_attention_rmsnorm` | `row` |
| 19 | L74/L76/L78/L80/L82/L84 | gate/up/silu/gated/down | `mlp_gate_proj`/`mlp_up_proj`/`mlp_activation`/`mlp_gated_product`/`mlp_down_proj` | `row` |
| 20 | L551（`xr + mo`） | MLP 残差加 = **L0 输出** | `layer_output` | `row` |
| — | 虚拟节点 | 轮前缓存 K/V | `cache_pre` | `kv` |

* `Attention.forward` 内的这些算子都是**函数式**（无对应 `nn.Module`）⇒ 用 `model_self.py` 的
  **默认 `None` 探针点** `_probe(name, t, who)` 捕获（见 §9）。
* 链定义（生产顺序，多输入算子要求**所有**输入逐位相同）见 `op_probe.CHAIN`（25 个算子）。

## 3. 每个中间边界的完整比较表（原始 dtype；仅统计时转 FP32）

| 节点 | S shape | 比较切片 | dtype | exact | n_diff | max\|Δ\| | 最大差位置 | S@max | B@max | S sha16 | B sha16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `x0` | (1,1,1536) | `[0,-1]` vs `[0,0]` | bf16 | **True** | 0 | 0 | — | — | — | `87bdb3aa1faf5f3c` | 同 |
| `input_rmsnorm` | (1,1,1536) | 同上 | bf16 | **True** | 0 | 0 | — | — | — | `f012125d65d98bf6` | 同 |
| **`q_linear`（q_proj）** | (1,1,1536) | 同上 | bf16 | **False** | **2** | **0.000244140625** | [1300] | 0.052001953125 | 0.05224609375 | `bad3c65395fe6d86` | `99100a94e2b152c4` |
| `q_reshaped` | (1,12,1,128) | `[0,:,-1,:]` vs `[0,:,0,:]` | bf16 | False | 2 | 0.000244140625 | [1300] | 同上 | 同上 | 同上 | 同上 |
| `q_rope` | (1,12,1,128) | 同上 | bf16 | **True** | **0** | 0 | — | 0.19140625 | 0.19140625 | `5d4f68964cd08c20` | 同 |
| `k_linear` | (1,1,256) | `[0,-1]` vs `[0,0]` | bf16 | **True** | 0 | 0 | — | 5.40625 | 5.40625 | `df19a40c3d13ffb7` | 同 |
| `k_reshaped` | (1,2,1,128) | `[0,:,-1,:]` vs `[0,:,0,:]` | bf16 | **True** | 0 | 0 | — | 5.40625 | 5.40625 | `df19a40c3d13ffb7` | 同 |
| `k_rope` | (1,2,1,128) | 同上 | bf16 | **True** | 0 | 0 | — | −1.570312 | −1.570312 | `945247e34fa96625` | 同 |
| `k_visible` | (1,2,8209,128) | 两侧 `[0,:,:8209,:]` | bf16 | **True** | 0 | 0 | — | 2.109375 | 2.109375 | `1b84a0f6c6fb41d3` | 同 |
| `v_linear` | (1,1,256) | `[0,-1]` vs `[0,0]` | bf16 | **True** | 0 | 0 | — | 0.5429688 | 0.5429688 | `63b946c4b70e1aed` | 同 |
| `v_reshaped` | (1,2,1,128) | 同上 | bf16 | **True** | 0 | 0 | — | 0.5429688 | 0.5429688 | `63b946c4b70e1aed` | 同 |
| `v_visible` | (1,2,8209,128) | 两侧 `[0,:,:8209,:]` | bf16 | **True** | 0 | 0 | — | −0.1054688 | −0.1054688 | `f10a19abe7ebf8e7` | 同 |
| **`qk_scores_pre_mask`** | (1,12,1,8209) | `[0,:,0,:8209]` 两侧 | bf16 | **False** | **3** | **4.0** | [13190] | 844.0 | 848.0 | `bed16041bdba3327` | `3d9af11d4276f1b3` |
| `qk_scores_post_mask` | 同上 | 同上 | bf16 | False | 3 | 4.0 | [13190] | 844.0 | 848.0 | 同上 | 同上 |
| `attention_prob` | 同上 | 同上 | **fp32** | False | 55343 | 5.28919e-06 | [2367] | 8.278566e-07 | 6.117046e-06 | `3f80172150eb3e19` | `7971855b362d09d1` |
| `attention_times_v` | (1,12,1,128) | `[0,:,-1,:]` vs `[0,:,0,:]` | bf16 | False | 2 | 6.10352e-05 | [35] | 0.01086426 | 0.01092529 | `d9d8deec28fb0fbc` | `0ec7210557c83cde` |
| `heads_merged` | (1,1,1536) | `[0,-1]` vs `[0,0]` | bf16 | False | 2 | 6.10352e-05 | [35] | 0.01086426 | 0.01092529 | 同上 | 同上 |
| `o_proj` | (1,1,1536) | 同上 | bf16 | False | 9 | 0.000244140625 | [185] | −0.06030273 | −0.06005859 | `ba484a8bc131b224` | `bf086dea9fdce8e7` |
| `attention_residual` | (1,1,1536) | 同上 | bf16 | False | 4 | 0.000244140625 | [185] | −0.06201172 | −0.06176758 | `574bb2d5e0e0e1d5` | `402c68b41f063ba6` |
| `post_attention_rmsnorm` | (1,1,1536) | 同上 | bf16 | False | 4 | 0.00048828125 | [185] | −0.06005859 | −0.05957031 | `229a0b0259c85255` | `67f86a1d78ccb46d` |
| `mlp_gate_proj` | (1,1,8960) | 同上 | bf16 | False | 247 | 0.0078125 | [7214] | −1.765625 | −1.757812 | `6158214241d80493` | `f74dc3068c3ec799` |
| `mlp_up_proj` | (1,1,8960) | 同上 | bf16 | False | 318 | 0.00390625 | [2943] | −0.5703125 | −0.5742188 | `1e17cd83b5c36b67` | `98c09e3337dd649c` |
| `mlp_activation` | (1,1,8960) | 同上 | bf16 | False | 232 | 0.00390625 | [366] | 0.3730469 | 0.3769531 | `7974ad4ae7940461` | `e0eaf4af73d8157a` |
| `mlp_gated_product` | (1,1,8960) | 同上 | bf16 | False | 503 | 0.001953125 | [6609] | −0.1464844 | −0.1445312 | `e6c476a80557ef21` | `d7c74ec9e3873a54` |
| `mlp_down_proj` | (1,1,1536) | 同上 | bf16 | False | 203 | 0.00390625 | [10] | −0.6015625 | −0.5976562 | `72bcc5543bbe3ae3` | `b785b3f7d2910c10` |
| `layer_output` | (1,1,1536) | 同上 | bf16 | False | **111** | **0.00390625** | [10] | −0.6171875 | −0.6132812 | `ff7c4133bea45a00` | `37ddf8a1ff092d86` |
| `cache_pre`（虚拟） | — | K 共同可见 0..8208 | bf16 | **True** | 0 | 0 | — | — | — | `64b561ae84fed808` | 同 |

**量级说明（避免夸大）**：所有差异按其各自量级都只有 **1–2 个 bf16 ULP**
（例：q 的 0.000244 = [2⁻⁵,2⁻⁴) 处 1 ULP；scores 的 4 = [512,1024) 处 1 ULP；
`mlp_activation` 的 0.00390625 = [0.25,0.5) 处 2 ULP），**没有**粗差；
其中 `attention_prob` 是 **fp32**（保留原 dtype，未强制 bf16），差异 ~5e-06。

## 4. 第一个「输入相同、输出不同」的算子

`op_probe.classify`（纯函数；"输入相同" = **所有**输入逐位相同；"输出不同" = 输出不逐位相同）：

| 算子 | 输入 | 所有输入相同？ | 输出相同？ | 判定 |
| --- | --- | --- | --- | --- |
| `input_rmsnorm` | `x0` | ✅ | ✅ | 否 |
| **`q_proj`** | `input_rmsnorm` | ✅ | ❌ | **✅ 首次分歧（生产顺序第一个）** |
| `q_reshape` / `rope_q` | `q_linear` / `q_reshaped` | ❌（输入已不同） | ❌ / ✅ | 否 |
| `k_proj` / `k_reshape` / `rope_k` / `kv_concat_k` | … | ✅ | ✅ | 否 |
| `v_proj` / `v_reshape` / `kv_concat_v` | … | ✅ | ✅ | 否 |
| **`qk_matmul_scale`** | `q_rope`, `k_visible` | ✅ | ❌ | **✅ 也满足（生产顺序第二个）** |
| `causal_mask` 及其后全部算子 | 含已不同的输出 | ❌ | ❌ | 否 |

⇒ **`first_divergence_operator = q_proj`**；
`all_operators_with_equal_inputs_and_different_output = ["q_proj", "qk_matmul_scale"]`
（如实全部报出，**不**只报一个）。

**措辞纪律**：只能说 `q_proj` 是"**最早观测到的 shape-sensitive 数值分歧边界**"——
因为两条路径在这里的差别**只有执行形状**（S: `(1,1,1536)` 输入、M=1；B: `(1,4,1536)` 输入、M=4），
输入与权重都逐位相同；**不能**据此判定它是 PyTorch / cuBLAS / BF16 的实现 bug
（那需要进一步区分 kernel 选择、归约顺序、实现错误；本轮**不做**）。

## 5. GQA 维度与对应切片说明

| 量 | 值 / 口径 |
| --- | --- |
| query heads | **12** |
| KV heads | **2** |
| head_dim | **128** |
| 重复倍数 `heads // kv_heads` | **6**（`kv_repeat(k_all, 6)`） |
| 映射 | 第 `h` 个 query head ↔ 第 `h // 6` 个 KV head（**连续**映射；CPU 门用真实 `kv_repeat` 逐头验证 12/12 一致） |
| 比较切片 | 只比较**原始** `kv` 形状 `(1, 2, T, 128)` 的共同可见范围 `[0:8209)`——**不**与广播后的 12 头维混比 |
| 广播后的形状 | 只在 `shape_only` 节点记录（S 应为 `(1,12,1,8209)`、B 应为 `(1,12,4,8212)`，由已记录的 `q`/`k_visible` 形状与 `kv_repeat` 规则可**推出**）—— ⚠️ **这两个 shape_only 节点的 layout 本轮没有落盘**（见 §9 缺陷 ⑤），报告中不作"实测值"引用 |

## 6. 必需控制（12 项，**全部通过**）

| # | 控制 | 结果 |
| --- | --- | --- |
| 1 | L0 输入 S/B 逐位相同 | ✔ `n_diff=0`，`sha16 87bdb3aa1faf5f3c`（两侧） |
| 2 | L0 输入 == 上一轮保存的 L0 输入 | ✔ 两侧都**逐位相同**（`87bdb3aa1faf5f3c`） |
| 3 | L0 输出复现：S/B `n_diff=111`、`max|Δ|=0.00390625`，且与上一轮原始行**逐位一致** | ✔（`ff7c4133bea45a00` / `37ddf8a1ff092d86`，两侧都与上一轮逐位相同） |
| 4 | S/B 使用相同 L0 权重 | ✔ 同一个模型实例的同一批张量（q/k/v/o/gate/up/down 的 `data_ptr` 全程不变） |
| 5 | L0 轮前 K/V cache 逐位一致（且 == H[0]） | ✔ `sha16 64b561ae84fed808`（S=B=H[0]） |
| 6 | 当前 token / position / cos/sin 对应行逐位一致 | ✔ cos `299971a65b0f0767`、sin `47fa5794d4acf8b2`（两侧相同）；缓存长度 8208→8209 / 8208→8212 |
| 7 | mask 可见范围正确、B 第 0 行看不见未来候选 | ✔ 被掩列从 `p+1=8209` 起，**36 = 12 头 × 3 列** 个 `-inf`；S（s=1）**0** 个；比较切片只取 0..8208 |
| 8 | 捕获开启/关闭时 L0 输出与最终 logits 逐位相同 | ✔ 三个子控制：探针关闭、**空探针**、关闭探针+层钩子，全部逐位相同 |
| 9 | 两次独立捕获逐位一致 | ✔ 全部 29 个节点 + 最终行（并且第二次**完整** run 的原始张量与本次**逐位相同**，见 §7） |
| 10 | hook / callback 全部移除 | ✔ 探针 `active=False`；28 层 pre/post + norm + embedding 钩子计数全为 0 |
| 11 | 输入、权重、原始历史 cache 未被诊断代码修改 | ✔ H 快照（逐层 k/v 摘要）不变、W/Norm/L0 七权重 `data_ptr` 不变、`x0` 行不变 |
| 12 | 所有中间张量有限、无 NaN/Inf | ✔ 29 个**比较切片**全部有限；唯一非有限元素 = B 路径 mask 后的 **36 个 `-inf`**（设计行为，已计数）；**无 NaN** |

其它一致性：与**已保存的局部对照行**（`div8209_cells_…064415Z.pt`）**逐位相同**；
与锚点一致（S=323、B=659）；位置口径（S 末位行、B `rows[0]`、槽位 1、可见 ≤8208）✔。

## 7. 两次重复与捕获关闭对照

| 对照 | 结果 |
| --- | --- |
| 同一 run 内两次独立捕获（各自独立克隆） | 全部 29 节点张量 + embedding/Norm/最终行 **逐位相同** |
| 探针**关闭**（无任何钩子） | 最终 logits 行与带捕获时**逐位相同** |
| **空探针**（安装 `lambda *a: None`） | 最终 logits 行与关闭时**逐位相同** ⇒ 探针**调用点**零副作用 |
| 探针关闭 + 仅 L0 层钩子 | L0 输出行与探针捕获的 `layer_output` **逐位相同** |
| **跨 run**（`…T172301Z` 与 `…T172447Z` 两次完整捕获） | 25 个节点 + `cache_pre` 的 sha16 **全部相同**、原始张量**逐位相同** |

## 8. 已证实 / 支持性推断 / 未证实（严格分开）

**已证实（本 run 实测）**
1. L0 输入（`x0`、`input_rmsnorm` 输出）S/B **逐位相同**；且与上一轮全层捕获**逐位相同**。
2. **`q_proj` 的输出不逐位相同**（2/1536 坐标、1 bf16 ULP），而其输入与权重都逐位相同
   ⇒ 它是本案例中**最早观测到的 shape-sensitive 数值分歧边界**（`first_divergence_operator`）。
3. `qk_matmul_scale` **也**满足同一条件（输入 `q_rope`/`k_visible` 逐位相同，scores 3 个坐标差 1 ULP）。
4. `k_proj`/`v_proj` 及其下游（`k_reshape`/`rope_k`/`kv_concat_k`/`v_*`）输出**逐位相同**。
5. 进入 RoPE 的 q 已经不同 ⇒ **RoPE 不是首次分歧边界**；且本例 `q_rope` 输出**又变回逐位相同**。
6. 全部差异仅 1–2 bf16 ULP（无粗差）；`attention_prob` 以 **fp32** 记录（差异 ~5e-06）。
7. L0 输出 `n_diff=111`、`max|Δ|=0.00390625`，与上一轮**逐位相同**；最终 logits 行也与既有产物逐位相同。
8. §6 的 12 项控制与 §7 的 5 项对照**全部通过**。

**支持性推断（由上面的数支持，但不是本轮新测到的机制）**
* 由于 `q_proj` 两侧的**输入**、**权重**与**精度设置**都相同，唯一差别是**执行形状**
  （M=1 vs M=4、N=1536、K=1536）⇒ 该 1 ULP 差异与**执行形状相关**（shape-sensitive），
  这与"执行路径相关的数值差异"一致，**不**需要也不允许升级为"实现 bug"。
* `q_proj`(N=1536) 与 `k_proj`/`v_proj`(N=256) 在**同样**的 M 差异下表现不同
  （前者有 1 ULP 差异、后者逐位相同）—— 这是**实测对照**，本轮**不**解释其原因。
* 差异在 L0 内部并不单调（`q_rope` 处一度回到逐位相同，随后在 QKᵀ 处再次出现），
  与"不许假定逐层/逐算子单调"一致。

**未证实（本轮没有测、也不声称）**
1. `q_proj` 内部**具体**是 kernel 选择、归约顺序还是别的实现细节造成这 1 ULP（**本轮只定位边界**）。
2. 323 / 659 谁是数学真值（**未**做 FP32 ⇒ 不做 `BF16_NEAR_TIE` 类判定）。
3. `qk_matmul_scale` 与 `q_proj` 的差异是否同源。
4. 该模式是否在其它位置/长度/历史/其它层同样出现（**不**外推）。
5. 8239 与 8209 是否同源。

## 9. 每项代码改动的原理 / 输入 / 输出 / 必要性

| 改动 | 原理 | 输入 → 输出 | 为什么必要 |
| --- | --- | --- | --- |
| **改** `model_self.py`：新增**默认 `None`** 的探针基础设施 `_ATTN_PROBE` / `set_attn_probe` / `clear_attn_probe` / `attn_probe_state` / `_probe(name, t, who)` | 关闭时每个探针点只做一次身份判断（不构造参数、不碰张量）⇒ 与改动前**逐位等价**；安装时回调只做只读克隆 | 张量 + 名字 + 模块 → 回调（或什么都不做） | Q/K/V 的 reshape、RoPE、KV 拼接、QKᵀ、mask、softmax、AV、head merge 都是**函数式算子**，普通 module hook 抓不到；**不**复制重写公式（那会改变 kernel/shape/舍入） |
| **改** `model_self.py`：在 31 个位置插入 `_probe(...)`，并做**两处表达式拆分**（MLP 的 silu 中间量、`o_proj` 调用） | 拆分保持**同一批算子、同一顺序、同一 dtype**（先 `gate_proj` 再 `silu`、再 `up_proj`、再乘、再 `down_proj`） | — | 只有把中间张量暴露出来才可能定位"首个分歧算子"；拆分的中性由"跨轮 L0 输出/最终 logits 逐位复现"实证 |
| **新增** `specdec/engine/op_probe.py` | 安装/移除探针（异常也移除）、只克隆**目标层**张量、前向结束后才搬 CPU、定义 5 种语义切片口径与 25 个算子的链、判定首个"输入相同输出不同"的算子 | 模型 + 层号 → 捕获记录（原始 dtype）/ 比较结果 / 分类结果 | 把"对应切片"与"首次分歧"做成**可 CPU 测试**的纯逻辑，避免整张量乱比与广播 |
| **新增** `specdec/engine/l0_operator_f13_p8209.py` | 两个捕获 pass + 关闭/空探针/层钩子对照 + 12 项控制 + 逐节点比较 + 首次分歧判定 + 早期落盘 | 捕获 `.pt` → JSON/PT（原始张量 + 全部比较字段 + 结论） | 本轮唯一实验；"先把数据落盘再打印/收尾"是三次白付加载的直接教训 |
| 改 `scripts/div8209_preflight_cpu.py`（+1 节） | 加载前核对 H/截断/历史/上一轮 L0 参考行（**并断言其为 1 维决策行**）/链与探针表/锚点 | 固定路径产物 → 断言 | 防止"口径写错"再次花掉一次模型加载 |
| **新增** `scripts/l0ops8209_verdict_cpu.py` | 从**原始张量**独立复算 29 个节点比较、`slices` 与 `raw_*` 重算一致、首次分歧判定，并与 JSON/上一轮/第二次 run 逐项核对 | 完整 run 的 `.json`+`.pt` → 汇总 JSON | 结论必须能从原始张量独立复算；**摘要错误不靠重跑 GPU 修** |
| **未改** | 生产公式、dtype 规则、mask/RoPE/attention/cache/linear/argmax、精度开关、后端、proposal、门槛 | — | 本轮只做诊断 |

**缺陷记录（全部保留，不覆盖）**
| # | 现象 | 原因 | 影响 / 处理 |
| --- | --- | --- | --- |
| ① | 首跑 `crashed`：上一轮行索引多一层 (`[0,0]`) | 上一轮链式捕获保存的是**(1536,) 一维决策行**，不是 `(1,s,hidden)` | 浪费 1 次加载；已修，并在**飞行前检查**里断言其维度 |
| ② | 次跑 `crashed`：`Unknown format code 's' for object of type 'bool'` | 表格打印把 bool 当字符串格式化 | 浪费 1 次加载；改为 `str(...)`，并**把数据落盘提前到打印之前** |
| ③ | 三跑 `crashed`：`raw_*` 的值被当成记录字典取 `["layout"]` | `_probe_payload` 返回的是**张量** | 浪费 1 次加载；改为**共用的 `_build_payload`**，并被 CPU 门覆盖 |
| ④ | 四跑 `checks_failed`：控制 11 假失败 | 我写成 `all(r == controls[...][r] ...)`（拿**键名**当数据比） | 该 run **数据完整**（含 PT）；已改为重新读取 `data_ptr` 再比较 |
| ⑤ | `k_repeated`/`v_repeated`（`shape_only` 节点）的 layout **未落盘** | payload 只落了非 shape_only 节点 | 报告改为"由已记录形状 + `kv_repeat` 规则**推出**"，**不**当实测值引用 |

## 10. B6 状态

**未改变**：本轮**没有**跑正式矩阵、**没有**改门槛/精度/算法/后端、**没有**放宽任何判据；
B6（`SpecDec ids == target-only greedy ids` 且全有限、多轮确定）**仍未通过** ⇒
**acceptance / pilot / final 仍然禁止**。本轮结论是**局部诊断**（定位到一个算子边界），
既不是修复，也不判定数学真值。

## 11. 唯一最小下一步（**未执行，不修复**）

> **对 `q_proj` 这一个边界做"形状敏感性"的独立控制实验**（只诊断，不改实现、不修 bug）：
> 在**同一份 `H(8208)`**、**同一输入**（L0 的 `input_rmsnorm` 决策行 / 或该行的重复）下，
> 只用**同一模块实例** `layers[0].self_attn.q_proj` 做**同一表达式** `x @ W.T`（`F.linear`/`Linear.forward`
> 原样调用），比较**不同 M**（例如 M=1 与 M=4，四行内容相同）的输出**第 0 行**是否逐位相同；
> 再对 `k_proj`/`v_proj`（N=256）做同样对照，确认"q 有差异、k/v 无差异"这一对照是否可独立重复。
>
> **判据（二值）**：若"同内容、不同 M"就能复现这 1 ULP 差异 ⇒ 该边界对**执行形状**敏感
> （仍**不**等于实现 bug，只是把"shape-sensitive"从描述变成受控实测）；若**不能**复现 ⇒
> 说明差异来自 L0 里更早/更细的上下文（例如 cache/连接布局），需再定位一步。
>
> **纪律**：本轮**不执行**该实验；**不**进入 kernel 级实验、**不**改实现、**不**做 FP32、
> **不**启动 B6/正式矩阵、**不**外推到别的位置/长度/历史。
