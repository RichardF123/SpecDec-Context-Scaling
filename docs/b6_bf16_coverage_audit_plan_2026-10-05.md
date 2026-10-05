# BF16 B6 正确性覆盖：只读审计 + 最小执行计划（2026-10-05）

> **本轮边界**：只做**只读审计、覆盖矩阵、文档修正**。**未启动 GPU**、**未编写任何新工具能力**、**未改动数值路径**。
> **Stage 2b 的授权（批 1）不延伸到"新增 B6 工具"或"新增 GPU 运行"** ⇒ 本文件把缺口与最小改动列清后 **停在待批准**。
> 正式实验代码（runner / 采集 / 统计分析 / 绘图 / final pipeline）由用户本人编写，本轮**未创建、未修改**。

## 0. 授权核对（先读仓库）

| 项 | 状态 |
| --- | --- |
| **已授权并已执行** | 批 1：仅 `F09/F03/F11/F04` × `ar/prod/serial` × 2 次 = **24 runs**（`tag=div8209_p2`，`tool=serial_oracle_gpu`）+ ≤4 次 GB6 诊断；结论仅到 `STAGE2B_PASS` |
| **已批准但未实施** | D-11(a)：保留 `:83` 的 prompts 类型覆盖要求，另列 **B6 类型覆盖轮** 待办（成本/准入见 §3） |
| **未批准事项** | ① 任何**新增 GPU 运行**（含 B6 类型覆盖轮）；② 运行器/harness 的**新能力**（字面 prompt 输入、额外字段记录）；③ 新增 CPU 检查器（分支覆盖断言）；④ `arm=cudnn_prefill` 的覆盖轮；⑤ `v_fp8` 相关一切（**本轮与下一阶段均不在范围**） |
| **不得推导** | `STAGE2B_PASS` ⇏ `B6_BF16_PASS` ⇏ `FULL_B6_PASS` ⇏ `BF16_FORMAL_READY` |

## 1. 冻结候选配置符合性（只读核对 24 个产物）

| 配置项 | 期望（冻结候选配置） | Stage 2b 产物中的记录 | 结论 |
| --- | --- | --- | --- |
| 计算精度 | BF16 | `input.dtypes.model = torch.bfloat16` | ✅ 已取证 |
| target KV | `base_bf16` | `dtypes.target_kv_payload = base_bf16` | ✅ |
| draft KV | `base_bf16` | `dtypes.draft_kv_payload = base_bf16` | ✅ |
| k / chunk / ctx / max_new | 4 / 2048 / 8192 / 64 | `smoke_config` 四项逐一相符 | ✅（24/24 run 无违规） |
| greedy argmax | 是 | 无采样相关字段，路径为 argmax | ⚠️ 未单列记录（依赖实现，非独立取证） |
| **attention = `cudnn_prefill`（仅 prefill 融合，其余 eager）** | `cudnn_prefill` | **`smoke_config.arm = "eager"`、`attention_stats.strategy = "eager"`、`calls.{eager=…, fused=0}`、`fused_by_phase={}`** | ❌ **不符合**：Stage 2b 的 24 run **全部**跑在 `arm=eager` 上，**未覆盖融合 prefill 路径**（详见 §2 覆盖矩阵"attention 后端"行） |
| diag preset = eager、`qk_fp32=False`、`prob_fp32=False` | False / False | **产物未记录**（`diag_precision = None`；该字段只有别的诊断工具会写） | ⚠️ **未取证**（需 §3 的 C-2） |
| EOS 按冻结规则 | 命中即停 | 24 run 全部 `stop_reason=max_new_reached` | ✅ 规则未被触发（≠ 未遵守） |
| AR = target-only | 是 | `mode=ar`，`kv_format.draft="not_loaded"` | ✅ |
| serial / production 与 AR 同一 target 配置 | 是 | 三模式 ctx/chunk/k/KV/dtypes 一致（`config_checks` 10/10） | ✅ |

**审计结论（配置）**：数值/精度/几何/输入一致性 **已取证**；**两项未取证**——① 融合 prefill 路径（本轮跑的是 `eager`）；② 诊断精度开关（未记录）。
在拿到这两项之前，不得声称"冻结候选配置下的正确性已覆盖"。

## 2. 覆盖矩阵（**以实际 trace 为证**；未触发即"未覆盖"，不得算 PASS）

### 2.1 提示类型（B6 `:83` 列出的形态）

| 类型 | 新口径证据（serial oracle vs AR） | 状态 | 复用脚本 | 最小必要改动 |
| --- | --- | --- | --- | --- |
| 自然语言 | Stage 2b：4 family × 24 run（8K WikiText），serial==AR、两次一致 | **PASS（限定范围）** | `serial_oracle_gpu.py` + `stage_check` + `stage2b_verdict` | 无 |
| 数学 | 无 | **未覆盖** | 同上 + 现有 B6 用例表 | **C-1** |
| 代码 | 无 | **未覆盖** | 同上 | **C-1** |
| 短输入 | 无（最短 ctx = 8192） | **未覆盖** | 同上 | **C-1** |
| 长输入 | 8K 上下文窗口已覆盖；旧 B6 的 `long` **句子**用例本身未跑 | **部分 / 按旧用例口径未覆盖** | 同上 | **C-1** |
| （补充）attention 后端 = 融合 prefill | 无（全部 `arm=eager`） | **未覆盖** | 同上 + `SMOKE_ARM` 开关 | **C-4**（配置决策 + 覆盖轮） |

### 2.2 分支形态（**只读 trace 统计；24 run / 388 轮**）

| 分支 | trace 证据 | 状态 | 复用脚本 | 最小必要改动 |
| --- | --- | --- | --- | --- |
| 全接受轮（`all_accept=True`，m==Kp） | **126 轮** | **PASS（分支级）** | `rounds` 字段 + 只读统计脚本 | **C-3**（把统计固化为断言） |
| 首项拒绝（m==0） | **88 轮** | **PASS（分支级）** | 同上 | C-3 |
| 中间拒绝（0<m<Kp） | **174 轮** | **PASS（分支级）** | 同上 | C-3 |
| 剩余预算裁剪（Kp<k=4） | **24 轮**（其中 **16 轮出现在最后一轮**） | **PASS（分支级）** | 同上 | C-3 |
| rollback / repair | 各 **262 轮** | **PASS（分支级）** | 同上 | C-3 |
| **EOS 停止** | **0 run**（真实 WikiText 64 token 内未命中 EOS） | **未覆盖（真实模型）**；逻辑级已有 CPU stub 证据（`g4_b5_eos_branch_tests.py`，4 case，**含"旧实现必抛 ValueError"的非空过对照**） | 该 CPU stub + 现有 `eos-prone` 用例 | **C-1 + C-5** |
| **输出长度非 k 整数倍** | 全部 `gen_len=64=16×4`；`len % 4 != 0` 的 run = **0** | **未覆盖** | 现有 B6 `len13/len5/len2` 用例（n=13/5/2） | **C-1** |
| 最终轮裁剪（末轮 Kp<k） | 末轮 Kp：F09=3、F03=1、F11=2、F04=1（`serial_trace_summary.kp_per_round`） | **PASS（分支级）** | 同上 | C-3 |

> **口径说明**：分支级 PASS 只说明"该分支**被真实模型执行过**且其轮记录自洽"；
> **不等于**该分支在新口径下通过了 exact-id 门——门只在 serial vs AR 的整串比较上判定（§2.3）。

### 2.3 门的判定（新口径）

| 项 | 证据 | 状态 |
| --- | --- | --- |
| serial == AR（完整 token IDs） | 4/4 family、8/8 serial run、`serial_failures=[]` | **PASS（限定范围）** |
| 两次独立重复一致 | 4/4 family（AR/prod/serial 各两次同 hash） | **PASS** |
| 有限性 / 结构 / 逐轮不变量 / 调用计数 | 18 项硬检查（含 D6）全通过 | **PASS** |
| production 作为独立观测对象 | F03 首分歧 59、F11 首分歧 49（两次一致）；F09/F04 `TOKEN_EQUAL` | **已记录**（发散轨迹；未自动豁免、未判控制逻辑失败） |
| 旧口径历史证据（**不可当新口径用**） | `B6-fp32` PASS 20/20、`B6-bf16` CHECK 19/20；工具 `g4_b6_correctness.py` = **production vs AR**（旧判据） | 仅作历史参照 |

## 3. 最小执行计划（**待批准；本轮不执行**）

### 3.1 需要的最小能力改动（每项：为什么 / 输入输出 / 证明什么 / 不能证明什么）

| # | 改动 | 为什么需要 | 输入 → 输出 | 证明什么 | **不能证明什么** |
| --- | --- | --- | --- | --- | --- |
| **C-1** | `serial_oracle_gpu.py`（+harness 取数入口）增加**字面 prompt 来源模式**：`SO_PROMPT`/固定用例表（如旧 B6 的 10 类），走 tokenizer 编码而非 manifest family | 数学/代码/短/长/EOS 与 `n` 非 k 倍数这些用例**必须**用固定输入；现有 runner 只能按 manifest family 取 8K 窗口 | prompt 文本 + n → 同结构产物（额外写 `input.source_kind="literal_prompt"`、prompt 的 `token_sha1`/`ids_hash`、`n`） | 该 run 确实使用了**声明的那段输入**（可复现） | 不证明任何 manifest family 的性质；不证明"自然语言全覆盖" |
| **C-2** | 运行器记录**诊断精度与后端实况**：`diag_precision()`（`qk_fp32/prob_fp32`）、`attention_stats.strategy`、`requested_backend`、`fused_by_phase` | §1 两项"未取证"直接影响"冻结配置符合性"的可辩护性 | 无 → 产物多出上述字段 | 该 run 的**精度开关与后端实况**（可判定 `arm` 是否按预期生效） | 不证明数值结果正确；不改任何计算 |
| **C-3** | 新增**只读 CPU 分支覆盖检查器**：按用例断言"预期分支确实被 trace 触发"（m==Kp / m==0 / 0<m<Kp / Kp<k / 末轮裁剪 / EOS 命中），未触发 ⇒ 记 **未覆盖** | 用户要求"自然语言 prompt 不保证触发某分支，必须以 trace 证明"；现在的 branch 结论是我用临时脚本统计的，需固化为可复跑检查 | 产物 JSON → 逐用例分支判定（PASS/未覆盖） | 分支**被真实执行过**（覆盖证据） | **不**证明正确性；不替代 exact-id 门 |
| **C-4** | 覆盖轮改用候选配置 `SO_ARM=cudnn_prefill`（仅 prefill 融合），并要求 `fused_by_phase["prefill"] > 0` | 冻结候选配置的后端是 `cudnn_prefill`；Stage 2b 跑的是 `eager` ⇒ 融合 prefill 路径**零覆盖** | 同 C-1 → 产物含 `fused_by_phase` | 融合 prefill 路径下的 serial==AR | **不**证明 `eager` 与 `cudnn_prefill` 等价（两者是不同配置，各自结论各自成立） |
| **C-5** | EOS 的真实触发：`eos-prone` 类用例能否在 n 内命中 EOS **不保证** ⇒ 需 C-3 判定；未命中则真实模型侧记**未覆盖**，逻辑级单独用 CPU stub（`g4_b5_eos_branch_tests.py`）报告 | 避免"看起来跑了 EOS 用例"被当成 EOS 分支已覆盖 | 固定 prompt + n → trace | 若命中：EOS 停止路径在真实模型上被覆盖 | 未命中时**不能**声称 EOS 已覆盖 |

### 3.2 运行清单、命令、次数、资源估计、产物

* **固定输入**：沿用旧 B6 的 10 类 case（`g4_b6_correctness.py:CASES`：`natural-1/2`、`math`、`code`、`short`、`long`、`len13`、`len5`、`len2`、`eos-prone`），**不改文本、不改 n**；仅把参照实现从"production vs AR"改为"**serial oracle vs AR**"（新口径），production 同输入同进程作为**独立观测对象**并行记录。
* **配置**：`ctx=8192`（短用例的实际 prompt 更短）、`chunk=2048`、`k=4`、`max_new=n`、`base_bf16`（target 与 draft 均 `base_bf16`）、greedy、`bf16_red_red=1`、`struct_check=1`；`arm` = **待定**（见下）；EOS 用冻结规则；**不改** k/精度/backend/阈值/提前停止。
* **运行次数（估算，待批准）**：
  * 方案 A（推荐，先做）：**10 用例 × 3 模式 × 2 次 = 60 runs**（`arm=eager`，与 Stage 2b 同配置）＋ ≤10 次诊断（仅有分歧时）。
  * 方案 B（若小组要求覆盖候选后端）：再 **60 runs**（`arm=cudnn_prefill`，含 `fused_by_phase["prefill"]>0` 断言）。
  * 逻辑级 EOS/裁剪 fixture：**CPU、毫秒级**，不占 GPU。
* **资源估计（估算，非承诺）**：短上下文远低于 8K 峰值；按 Stage 2b 实测单 run 33–96 s（含 ~20–30 s 进程内模型加载）上界推，
  方案 A ≈ **35–70 min GPU**，方案 A+B ≈ **70–140 min**；显存风险低（短 prompt），但要按 E-2 逐批过环境门。
* **命令（获批后；示意）**：`so_batch.ps1 -Tag <b6_tag> -Cases <case> -Modes ar,prod,serial -Reps 1,2` → 逐阶段 `stage_check` →
  （有分歧时）`correctness_protocol_margins --stage2b --runs-tag <b6_tag> --families <case>` → `stage2b_verdict`（新 tag，不复用 `div8209_p2`）。
* **预期产物**：`output/g5/attn/<b6_tag>_<case>_{ar,prod,serial}_rep{1,2}_<runid>.json/.pt`（60 套）＋（有分歧时）`…_margins_<utc>.json` ＋ 判决 JSON ＋ 逐 run 日志。
* **停止条件（沿用第 5 节口径，不自动重试）**：serial token mismatch · NaN/Inf · 结构失败 · 重复不一致 · OOM · 原生崩溃 · 环境门拒绝 ⇒ **保留产物并停止**；
  `TIE-*` 一律按 D-10 记 `TIE-C`（无双侧决策行证据时**不**判 `TIE-A`、**不**豁免）。

### 3.3 待批准项（**未获批准前不启动**）

1. **C-1**（字面 prompt 来源模式）与 **C-2**（诊断精度/后端字段）——属**运行器能力新增**；
2. **C-3**（CPU 分支覆盖检查器）——新增检查工具（只增不改既有检查）；
3. **C-4** 的**配置决策**：B6 覆盖轮是否必须包含 `arm=cudnn_prefill`（若包含，本轮 §1 的"后端未取证"才能消解）；
4. **新增 GPU 运行授权**：新 tag（如 `div8209_b6`）下的 **60（或 120）次** run + 诊断；
5. **D-11(b)** 若小组选择"修改/豁免 `:83` 覆盖要求"，属**实质协议变更**，须单独批准并列明放宽范围（不属本轮）。

## 4. 是否达到 `B6_BF16_PASS` —— **未达到**

* 已具备：**限定范围**的自然语言 8K 窗口上的 serial==AR 严格一致（`STAGE2B_PASS`）+ 若干**分支级**覆盖证据。
* 未具备（因此**不能**称 `B6_BF16_PASS`）：
  ① 数学 / 代码 / 短输入 / （旧用例口径的）长输入 **无新口径证据**；
  ② **EOS 停止**在真实模型上 **0 次触发**；
  ③ **输出长度非 k 整数倍** 的用例未跑（现全部 64=16×4；"预算<k 的末轮裁剪"机制已覆盖，但**不等价**于该用例）；
  ④ **融合 prefill（候选后端）路径零覆盖**（本轮全部 `arm=eager`）；
  ⑤ 诊断精度开关（`qk_fp32/prob_fp32=False`）**未被记录**。

## 5. 距 `BF16_FORMAL_READY` 还缺什么

| 类别 | 缺项 |
| --- | --- |
| **B6 证据** | `B6_BF16_PASS` 本身（§4 的五项）；`FULL_B6_PASS` 还需 **`B6_VFP8_PASS`**（下一阶段，不在本轮） |
| **协议批准** | **D-12**（"未过门禁止 acceptance/pilot/final"是**按 KV 分别放行**还是全局放行）——直接决定"BF16 侧能否先放行"；**D-5**（质量口径：是否两条轨迹都评、禁用"质量等同"表述）；**D-7**（判决词条）；D-9（是否再补 FP32） |
| **测量前置** | 正式测量的 speedup 声明资格 10 条（含两侧 `input_token_hash`/`target_kv`/`max_new_tokens` 一致、measured reps 完整、`measurement_status=OK`、无 OOM/NaN/ENV_INVALID 等）；诊断计时**不得**进入 speedup 结论 |
| **环境/工程** | 入口内存余量薄（Stage 2b 实测约 +1.4 GiB 于 18 GiB 门槛）；`cuda_free≈0` 的薄余量属候选配置工程风险（需在正式运行前再评估） |
| **未决解释规则** | 并列的双侧分类（P-1/P-2 未实施）⇒ 一切并列目前记 `TIE-C` |

## 6. 结论（严格限定）

* **本轮未运行任何 GPU 任务、未新增工具能力、未改动数值路径与历史产物。**
* **未达到 `B6_BF16_PASS`**；已达成的是 **限定范围 `STAGE2B_PASS`**（Stage 2b，见 `docs/stage2b_result_report_2026-10-05.md`）。
* **不得**由本报告推出：完整 B6 通过 · `FULL_B6_PASS` · BF16/FP8 正式实验放行 · "同输出无损加速"。
* **V-only FP8 独立准入**留待下一阶段，本轮**不扩大范围**。
* 顺带修正（**仅文档**）：`docs/stage2b_result_report_2026-10-05.md` §3 中 F03 的 `cache_structure_n_checks` 应为**两次重复均 40**（原写"40 → 56"错误；**56 属 F11**），并已补上勘误说明与 production 侧对照值（F09 52 / F03 38 / F11 54 / F04 48）。**未因此重跑任何数值实验。**
