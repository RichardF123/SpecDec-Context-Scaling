# 项目级候选报告：全层 **row-aligned verify** 候选模式（Stage 0–3，2026-10-03）

> **本轮不是正式 B6，不启动正式矩阵。** 目标是把已有的**诊断性逐行 M=1 机制**一次性应用到 target
> 的全部 28 层，形成一个**默认关闭**的"全层 row-aligned verify 候选模式"，并**分阶段**评估：
> ① 能否恢复已知失败位置/生成的 token equality；② 需要多少性能代价；③ 是否值得作为正式 B6 候选。
>
> **不做**：不进入 L1–L27 算子内部、不新增逐算子定位、不改 token acceptance 公式、不改 draft、
> 不做 FP32、不切后端、不启用 cuDNN/SDPA/FlashAttention、不把候选设为默认、不放宽 token equality、
> 不用 margin/ULP 豁免、不选择性删除失败案例、不把 Stage 3 称为正式 speedup 实验。

## 0. 决策（四选一）

# **`CORRECT_BUT_NO_SPEED_VALUE`**

* **正确性收益成立**：F13 已知失败案例上，**原始 SpecDec 与 target-only AR 不一致**
  （首个分歧 index **47**，与 2026-10-01 的历史记录**逐 token 相同**：AR `6e7522bf2521ad7b`
  / 原 Spec `e90dad2b3fcf1c65`）；而**全层 row-aligned 候选的生成 token IDs 与 AR 完全一致**
  （F13 与 F06 两个案例、每个 2 个独立进程，`gen_hash` 均 = AR 的 `6e7522bf2521ad7b`）。
* **但没有速度价值**：候选的 decode 时间是**原始 SpecDec 的 1.26×**（固定 block 的 target verify
  是 **1.69×**），而对 AR 的相对吞吐是 **0.176×**（原始 SpecDec 也只有 0.222×）
  ⇒ 在这个 8K / all-eager 配置下，SpecDec 本身远慢于 AR，候选又把它的 decode 再拖慢约 26%。
* 因此：**候选可作为"数值对齐"的工程选项，但不构成值得优先投入的 B6 候选**；
  只有 `VIABLE_B6_CANDIDATE` 才建议进入正式 B6，本轮**不是**该结论。

**正式 B6 状态：未改变**（未跑正式矩阵；acceptance/pilot/final 仍禁止）。

## 1. 命令、commit、dirty、环境与产物

```powershell
cd D:\RealDSH\specdec
$env:PYTHONIOENCODING='utf-8'
# Stage 0（纯 CPU + 微型模型；不加载大模型）
.\.venv\Scripts\python.exe -X utf8 -m specdec.engine.row_aligned
$env:RA_MODE='selftest'; .\.venv\Scripts\python.exe -X utf8 -m specdec.engine.row_aligned_stage1_f13_p8209
Remove-Item Env:RA_MODE
# 飞行前（含本轮新增一节）
.\.venv\Scripts\python.exe -X utf8 scripts\div8209_preflight_cpu.py
# Stage 1（一次 GPU：四臂 × 2 pass）
$env:RA_TAG='div8209_ball'; $env:RA_LAYER='0'; $env:RA_PRE='8208'; $env:RA_TARGET='8209'; $env:RA_K='4'
$env:RA_FEED='20642'; $env:RA_CANDS='20642,659,576,4531'; $env:RA_REPEAT='2'; $env:RA_TARGET_ROPE='cpu_fp32'
.\.venv\Scripts\python.exe -X utf8 -m specdec.engine.row_aligned_stage1_f13_p8209
# Stage 2（每个 (case, mode, rep) 一个独立进程；F13 主案例 + F06 对照）
$env:RA2_TAG='div8209_ra2'; $env:RA2_MAXNEW='64'
foreach ($c in 'F13','F06') { foreach ($m in 'ar','spec','aligned') { foreach ($r in 1,2) {
  $env:RA2_CASE=$c; $env:RA2_MODE=$m; $env:RA2_REP="$r"
  .\.venv\Scripts\python.exe -X utf8 -m specdec.engine.row_aligned_stage2 } } }
.\.venv\Scripts\python.exe -X utf8 -m specdec.engine.row_aligned_stage2 --aggregate
# Stage 3（同一 prompt：3 模式 × warm-up 1 + reps 3；再加固定 block micro-latency）
$env:RA2_TAG='div8209_ra3'; $env:RA2_CASE='F13'
foreach ($m in 'ar','spec','aligned') { foreach ($r in 1,2,3,4) {
  $env:RA2_MODE=$m; $env:RA2_REP="$r"
  .\.venv\Scripts\python.exe -X utf8 -m specdec.engine.row_aligned_stage2 } }
.\.venv\Scripts\python.exe -X utf8 -m specdec.engine.row_aligned_stage3_f13_p8209
.\.venv\Scripts\python.exe -X utf8 scripts\row_aligned_stage3_cpu.py
.\.venv\Scripts\python.exe -X utf8 scripts\row_aligned_decision_cpu.py
```

* **commit / dirty**：Stage 0 与工具先提交（`60e555e`），随后两次**只改自测/断言**的修复
  （`796886a`、`ccf4ba8`），Stage 2 运行器在 `0a8ae73`，Stage 3 工具与决策脚本在 `7d668a9`；
  **每次 GPU 运行的产物都记录当次 `git`（含 `dirty_worktree`）**，正式运行均为 `dirty_worktree=false`。
* **环境**：RTX 4060 Laptop 8 GB（峰值 reserved ≈ **10.7 GiB 未发生**，
  实测峰值 reserved **10656 MiB**、allocated **7792 MiB** ⇒ 无 OOM）；
  Stage 3 的 micro 测量记录 **P-state `qualified=True`（`sm=2430`）**；commit 空间 ≥ 16 GiB 门通过。
* **产物（全部新 tag；旧产物只读）**：
  * Stage 1：`output/g5/attn/div8209_ball_20261003T051846Z.{json,pt}`；
    **保留的失败 run**：`…T051651Z.json`（控制 9 误失败）、`…T051755Z.json`（控制 10 误失败）——**均为我的断言缺陷**，
    产物保留、不覆盖、不作为结论依据；
  * Stage 2：`div8209_ra2_{F13,F06}_{ar,spec,aligned}_rep{1,2}_*.{json,pt}`（12 个独立进程）
    + 汇总 `div8209_ra2_stage2_summary_20261003T053438Z.{json,pt}`；
  * Stage 3：`div8209_ra3_F13_{mode}_rep{1..4}_*.json`（12 次；rep1 = warm-up）+
    `div8209_ra3micro_20261003T054943Z.{json,pt}`（固定 block micro-latency）+
    汇总 `div8209_ra3_stage3_summary_20261003T055128Z.json`；
  * 决策：`row_aligned_decision_cpu_20261003T055154Z.json`。

## 2. 全层候选模式的准确作用范围

**只作用于 SpecDec 的 target block verification**，对 target 的**全部 28 层**（数量**从模型结构推导**）：

| 项目 | 内容 |
| --- | --- |
| **QK**（28 个 `self_attn`） | 每个 query 行分别：**M=1**、真实 causal key 前缀 `p+i+1`、**同一个** `torch.matmul`、**同一个** BF16 dtype、**同一个**缩放；然后拼回 block scores，继续走**原有** mask/softmax/AV |
| **Linear**（28 × 4 = 112 个） | `o_proj`、`gate_proj`、`up_proj`、`down_proj` 逐行以 **M=1** 调用**原来的同一个 `nn.Linear`**，再按 token 行 `cat(dim=1)` 拼回 |
| **不改变** | `q_proj`、`k_proj`、`v_proj`、RoPE、softmax、AV、RMSNorm、SiLU、gated product、residual、cache、argmax |
| **挂载点** | `spec_dec.set_verify_wrap(row_aligned_verify)` —— `spec_decode_greedy` 里**只**包住 `_verify`（那一次 `s=Kp` 的 `_run`）；AR/decode（`_step`）与 prefill **不经过**该挂载点 |
| **默认状态** | `_VERIFY_WRAP = None` 且两个注册表都是 `None` ⇒ 生产路径**一字不变**（控制 2 逐位复核） |
| **匹配方式** | 按**模块对象身份**（`id` + `is`）注册，**不**按类名/名称模糊匹配 |
| **draft** | **永不启用**：draft 的模块从不出现在注册表里；Stage 0 用第二个微型模型实例证明其 `applied = 0` |

**注册数量（运行时从结构推导并核对）**：`qk = 层数 = 28`、`linear = 4×层数 = 112`、`total = 140`。
Stage 1 实测 `QK applied = 28/28`、`Linear applied = 112/112`、`seen = 0`（注册集合完整覆盖全部被拦调用点）。

## 3. Stage 0–3：是否执行、是否通过

| 阶段 | 是否执行 | 结论 | 关键证据 |
| --- | --- | --- | --- |
| **Stage 0**（CPU + 微型模型门） | **执行** | **PASS 17/0** | 28 层微型模型 + 第二个"draft"微型模型：数量推导、身份覆盖、draft 零干预、AR s=1 开关前后**逐位一致**、block 每行 causal prefix、多 context 嵌套、正常/异常退出恢复、旧单层 L0 模式行为不变（`applied` 恰为 L0 的 5 个）、默认关闭、先落盘 |
| **Stage 1**（位置 8209 全层局部门） | **执行** | **1A PASS**；**1B = 类别 2** | S0/B0 逐位复现历史；**SALL ≡ S0**（层行 + final + 最终 Norm）；28 QK + 112 Linear 全部 applied；两次 pass 逐位一致；`S0 vs BALL`：token 一致（323=323）、logits 未对齐（`n_diff=118003`、`max|Δ|=0.125`） |
| **Stage 2**（已知失败案例 token equality 硬门） | **执行**（Stage 1 为类别 2 ⇒ 允许进入，并标注数值路径未对齐） | **硬门 PASS（12/12 检查全 True）** | F13：原始 Spec **失败**（首分歧 index 47）、**候选 == AR**；F06：两者都通过；每案例 2 个独立进程一致；全有限、无 `other_short`、结构不变量通过、无 OOM |
| **Stage 3**（性能代价观察） | **执行** | 完成（**pilot**：1 案例、warm-up 1 + reps 3） | 端到端三模式对比 + 固定 block micro-latency（见 §7） |

**阶段门严格执行**：Stage 1 的 1A 任一失败即停止；1B 为类别 3 时不得进入 Stage 2（本轮为类别 2，
按规格"正式 B6 是 token-ID 门"允许继续，并**明确标注数值路径未对齐**）；Stage 2 硬门不通过即停止、
不做逐层定位（本轮通过 ⇒ 进入 Stage 3）。

## 4. 位置 8209 的 logits / token / 逐层结果（Stage 1）

| 量 | 值 |
| --- | --- |
| 1B 类别 | **2**（token 一致、logits 不一致） |
| logits `S0 vs BALL` | `exact=False`、`n_diff=118003`、`max|Δ|=0.125` |
| token | S0 = **323**、BALL = **323**（margin 两者都是 `0.0`） |
| **L0–L1** | **逐位一致**（输入与输出行 `n_diff=0`） |
| **首个「输入同·输出不同」层** | **L2**（输入 `n_diff=0`、输出 `n_diff=126`） |
| L3–L27 | 输入已不同（126 → 1071 → … → 平台 1250–1340）；**按约定只记录编号与 n_diff，不进入层内部** |
| 候选计数 | QK applied 28/28、Linear applied 112/112、`seen=0`；每次 verify 调用的计数与 `target_block_calls` 一致 |
| 非有限值 / cache | 全部 logits 与 28 层决策行**有限**；四臂 cache 结构 = `{k,v}`、bf16、层内等长（S 臂 8209、B 臂 8212） |

> 与上一轮（只对齐 L0 的五处）对照：上一轮 L0 输出逐位恢复、**L1 即重新分歧（222）**；
> 本轮把候选扩到全层后，**L1 也逐位一致**，分歧推迟到 **L2**。这与"shape sensitivity 会在后续层
> 重复出现"一致 —— 本轮**不**进入 L2 内部（那是被明令禁止的逐算子考古）。

## 5. F13 / F06 完整 token equality（Stage 2）

配置逐项沿用既有记录：**F13/F06 manifest 窗口（8192 token，`token_sha1` 校验）**、ctx=8192、chunk=2048、
k=4、max_new=64、`base_bf16`、**arm=eager**、target RoPE=`cpu_fp32`、`SMOKE_BF16_RED_REDUCTION=1`、
`SMOKE_STRUCT_CHECK=1`；greedy；EOS 与停止条件、token 接受公式**均未改**；draft 走普通路径；
**只有 target verify 启用候选**；不共享 cache；不自动重试。

| 案例 | 模式 | rep1 `gen_hash` | rep2 `gen_hash` | 长度 | stop | 与 AR 一致 |
| --- | --- | --- | --- | --- | --- | --- |
| **F13**（已知失败） | AR（target-only） | `6e7522bf2521ad7b` | `6e7522bf2521ad7b` | 64 | `max_new_reached` | — |
| | **原始 SpecDec** | `e90dad2b3fcf1c65` | `e90dad2b3fcf1c65` | 64 | `max_new_reached` | **否**（首分歧 index **47**） |
| | **全层 row-aligned** | **`6e7522bf2521ad7b`** | **`6e7522bf2521ad7b`** | 64 | `max_new_reached` | **是（逐 token 相同）** |
| **F06**（对照） | AR | — | — | 64 | `max_new_reached` | — |
| | 原始 SpecDec | — | — | 64 | `max_new_reached` | 是（本来就不失败） |
| | **全层 row-aligned** | — | — | 64 | `max_new_reached` | **是** |

* **两次独立重复**：每个 (案例, 模式) 两个独立进程 ⇒ `ar`/`spec`/`aligned` 的两次 `gen_hash` 与
  完整 token IDs **逐 token 相同** ✓；Stage 2 硬门的 `reps_consistent` 三项全 True。
* **原始 SpecDec 的 F13 失败逐位复现历史**（`e90dad2b3fcf1c65`、首分歧 index 47）⇒ 说明我们用的
  规则/环境与历史记录可比，**没有**通过换配置制造通过。
* F06 的原始 SpecDec 本来就不失败 ⇒ 它在这里只作**对照**（证明候选不会把本来正确的案例弄错）。

## 6. 两次独立重复与不变量

| 项目 | 结果 |
| --- | --- |
| Stage 1：四臂 × 2 pass | 层行 / final row / cache 长度**逐位相同** |
| Stage 2：每 (case, mode) × 2 独立进程 | 完整 token IDs 与 `gen_hash` **一致** |
| Stage 3：每模式 warm-up 1 + reps 3 | F13 全模式 `gen_hash` 在 rep 间一致（AR `6e7522bf2521ad7b`、原 Spec `e90dad2b3fcf1c65`、候选 `6e7522bf2521ad7b`）；wall 时间 spread 见表 |
| 结构不变量 | `decision_coverage`（决策步覆盖）在两模型路径上都 `ok`；prefill logits 逐模型 finite；`gen_ids_in_vocab=True` |
| 停止原因 | 12 次 Stage 2 + 12 次 Stage 3 全部 `max_new_reached` ⇒ **无 `other_short`** |
| 显存 | 无 OOM；候选峰值 reserved 10656 MiB vs 原始 10648 MiB（**+8 MiB**） |

## 7. 性能与显存代价（Stage 3，**pilot**，不作 speedup 声明）

**端到端（F13，max_new=64；rep1 = warm-up 不计入；3 个计时 rep 的中位数）**

| 模式 | wall_s | prefill_s | decode_s | decode tok/s | e2e tok/s | 接受率 | rounds | target block calls | draft calls | peak alloc / reserved (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| target-only **AR** | 8.94 | 3.82 | 5.16 | **12.40** | 7.16 | — | 64（decode 步） | — | — | 6153 / 9668 |
| **原始 SpecDec** | 49.53 | 26.26 | 23.28 | **2.75** | 1.29 | 47.4% (46/97) | 25 | 25 | 97 | 7792 / 10648 |
| **全层 row-aligned** | 56.84 | 26.41 | 29.27 | **2.19** | 1.13 | 47.4% (46/97) | 25 | 25 | 97 | 7792 / 10656 |

**固定 block 的 target verify micro-latency**（同一份 `H(8208)`、同一批候选、同一实例；
预热 5 次后各测 20 次；OFF 取两个分块的中位数以抵消漂移；P-state `qualified=True`, `sm=2430`）

| 测量 | OFF 中位 | ON 中位 | 比值 |
| --- | --- | --- | --- |
| **block verify（s=4，即候选真正作用的那一次）** | **137.66 ms** | **232.62 ms** | **1.69×**（OFF 两块漂移仅 0.78 ms） |
| decode 单步（s=1，不该被改变） | 135.31 ms | 155.90 ms | 1.15×（且 ON/OFF **逐位一致** ✓） |

* **代价来源**：一次 `M=4` 的 block verify 被换成 **28 层 × (4 次 QK 逐行 matmul + 16 次逐行 Linear 调用)**
  ⇒ 1.69×；decode 单步虽只有 1 行，仍因逐行路径的副本/调用开销多 **15%**。
* **必须说清的口径**：原始 Spec 与候选在 F13 上的 **token 轨迹不同**（后者与 AR 一致）
  ⇒ 两者的 e2e 时间差（+7.3 s，1.15×）**不能**解释为纯 kernel 开销；纯开销看上面的固定 block 测量。
* **显存**：候选 **没有**明显增加峰值（+8 MiB reserved），因为逐行调用只是把同一批张量分次调用。

## 8. 原始 Spec / aligned Spec / AR 三方比较

| 维度 | target-only AR | 原始 SpecDec | 全层 row-aligned SpecDec |
| --- | --- | --- | --- |
| F13 token IDs vs AR | — | **不一致**（首分歧 47） | **一致**（逐 token 相同） |
| F06 token IDs vs AR | — | 一致 | 一致 |
| decode tok/s（F13） | **12.40** | 2.75 | 2.19 |
| 相对 AR 的吞吐 | 1.0 | 0.222× | **0.176×** |
| target block calls / rounds | — | 25 / 25 | 25 / 25 |
| 接受率 | — | 47.4% | 47.4% |
| 峰值 reserved | 9668 MiB | 10648 MiB | 10656 MiB |

**决策量（脚本现算，未预设人工阈值）**

```
correctness_gain = { 原始 Spec 是否失败: {F13: true, F06: false},
                     aligned 是否通过 token equality: {F13: true, F06: true} }
overhead         = { aligned_decode_over_original: 1.258,
                     aligned_e2e_wall_over_original: 1.148,
                     aligned_verify_micro_over_original: 1.690,
                     aligned_decode_step_micro_over_original: 1.152 }
net_value        = { aligned_spec_vs_ar: 0.176, original_spec_vs_ar: 0.222 }
⇒ CORRECT_BUT_NO_SPEED_VALUE
```

## 9. 已证实 / 支持性推断 / 未证实（严格分开）

**已证实（本轮实测）**
1. 全层候选的**作用范围可核**：28 个 QK + 112 个 Linear（数量从模型结构推导），一次 verify 里
   `applied = 28/112`、`seen = 0`；draft 零干预；AR/decode 的 s=1 与生产路径**逐位一致**。
2. **F13 已知失败被修复（token-ID 门）**：原始 SpecDec 与 AR 不一致（首分歧 47，逐位复现历史），
   全层候选的生成 token IDs 与 AR **完全一致**；F06 对照同样一致；每案例 2 个独立进程一致。
3. 位置 8209：候选把**L0–L1 变成逐位一致**，分歧推迟到 **L2**（输入逐位相同、输出差 126），
   但 `S0 vs BALL` 的 **logits 仍未逐位一致**（类别 2）。
4. 固定 block 的 target verify 代价 = **1.69×**；decode 单步 = 1.15×；端到端 decode = **1.26×**；
   候选相对 AR 的 decode 吞吐 = **0.176×**（原始 SpecDec 0.222×）。
5. 峰值显存 +8 MiB（reserved 10648 → 10656 MiB）；无 OOM；停止原因全部 `max_new_reached`。

**支持性推断（由上面的数支持，但不是本轮新测到的机制）**
* 候选之所以能让 F13 与 AR 的 token IDs 一致，是因为它把 verify 路径上**五处** bf16 GEMM 的执行几何
  从 `M=4` 搬到 `M=1`，从而与 AR 路径在**每一层**都走同一批几何 —— 与 L0 闭环轮的结论同源。
* L1 在本轮变一致、L2 才分歧，与"同样的 shape sensitivity 会在每一层重复出现"一致：
  早先只对齐 L0 时 L1 就分歧（222），全层对齐后 L1 也一致。
* 1.69× 的固定 block 代价与"4 次逐行调用替代 1 次块调用（含 4 次 QK、16 次 Linear）"相符。

**未证实（本轮没有测、也不声称）**
1. **L2 内部**是什么（按本轮约定**不进入** L1–L27 算子内部），以及它是否会随候选范围变化。
2. 各 GEMM 内部的具体 kernel/tile/归约顺序（**未做 profiler**）。
3. 候选在**其它位置/长度/family/模型**上是否同样通过 token equality（本轮只测 F13 与 F06 两个已登记案例）。
4. **正式 B6 是否通过**（未跑正式矩阵；本轮只是"候选评估"，且 Stage 3 不是正式性能实验）。
5. 本轮的速度数字能否外推为"任何配置下的代价"（Stage 3 是 pilot：1 案例、warm-up 1 + reps 3）。
6. 除 token equality 之外的数值路径对齐（logits 仍未逐位一致）。

## 10. 必需控制（20 项）与治理

| # | 控制 | 结果 |
| --- | --- | --- |
| 1 | 所有诊断开关默认关闭 | ✔ Stage 0 门 + Stage 1 进入前状态检查（探针/两个注册表/verify 包装都未激活） |
| 2 | 关闭后旧路径逐位不变 | ✔ Stage 1 的 S0/B0 逐位复现历史；关闭包装后再跑 S/B 与 S0/B0 逐位一致 |
| 3 | SALL ≡ S0 | ✔ 层行 + final row + 最终 Norm 逐位一致（Stage 0 与 Stage 1 双处） |
| 4 | 28 个 QK 模块覆盖完整 | ✔ `applied=28/28`，`expected_qk` 由层数推导 |
| 5 | 112 个 Linear 模块覆盖完整 | ✔ `applied=112/112`（每层 o/gate/up/down） |
| 6 | draft 零干预 | ✔ Stage 0 用第二个微型模型证明 `applied=0`；Stage 1/2/3 只注册 target 模块；AR 模式不加载 draft |
| 7 | target AR 不被候选改变 | ✔ AR 模式不安装包装；三模式 `ar` 的 `gen_hash` = 历史 `6e7522bf2521ad7b` |
| 8 | 只在 target verify 启用 | ✔ 挂载点在 `spec_dec._verify` 调用处（AR/prefill 不经过）；候选计数 == `target_block_calls` |
| 9 | mask 与 causal prefix 正确 | ✔ Stage 1 记录每层 QK 的 `visible = [8209,8210,8211,8212]`（`p+i+1`） |
| 10 | cache 长度/层数/dtype 正确 | ✔ 28 层、键 `{k,v}`、bf16、层内等长（S 8209 / B 8212）；Stage 2/3 的 `decision_coverage` ok |
| 11 | 权重、输入、原始历史 cache 未改 | ✔ H 逐层摘要前后相同；权重 `data_ptr`/摘要稳定；Norm 权重稳定 |
| 12 | 两次候选重复一致 | ✔ Stage 1 两 pass、Stage 2 每案例两进程、Stage 3 各 rep |
| 13 | context/hook 全部恢复 | ✔ 探针关、两注册表关、verify 包装未安装、层钩子与 `norm` 钩子为空 |
| 14 | 无 NaN/Inf | ✔ Stage 1 logits 与 28 层决策行全有限；Stage 2/3 prefill logits finite、`gen_ids_in_vocab` |
| 15 | `allow_tf32=False` | ✔ Stage 1 记录 `tf32_after=False` |
| 16 | commit/dirty/环境/产物完整 | ✔ 每条记录都写 `git`（含 dirty）、环境门、P-state、产物路径 |
| 17 | 新 tag、不覆盖旧结果 | ✔ `div8209_ball` / `div8209_ra2` / `div8209_ra3` / `div8209_ra3micro`；旧产物只读 |
| 18 | 原始张量/生成 IDs 先落盘 | ✔ Stage 1 在**任何比较前**写 `.pt`；Stage 2/3 每个进程先写生成 IDs 的 `.pt` 再写摘要 |
| 19 | 阶段硬门失败后不继续 | ✔ 本轮 1A 与 Stage 2 硬门都通过才继续；期间两次 Stage 1 因**我的断言缺陷**判失败时**未继续** |
| 20 | 不启动正式 B6/acceptance/pilot/final | ✔ 未跑正式矩阵；**B6 状态未改变** |

**缺陷记录（全部保留）**

| # | 现象 | 原因 | 处理 |
| --- | --- | --- | --- |
| ① | Stage 1 首跑 `checks_failed`（控制 9：causal prefix 为空） | 我在 `row_aligned_verify` 上下文**体内**读 `verify_log()`，而该日志在 `finally` 里才写入 | 改为退出后读取（commit `796886a`）；失败 run 保留 |
| ② | Stage 1 次跑 `checks_failed`（控制 10：cache 结构） | 我的断言要求四臂共用一个 cache 长度，实际上 S 臂 8209 / B 臂 8212 **本来就不同** | 改为"每臂内 28 层等长 + 等于该臂期望值"（commit `ccf4ba8`）；失败 run 保留 |
| ③ | Stage 0 首跑 2 个 FAIL | ①我的期望算错（微型 H 只有 5 key ⇒ p=5、`visible=6,7,8,9`）；②**真实设计缺陷**：同类 context 嵌套时内层退出会清空外层注册表 | ①改期望；②把 `serial_qk_for`/`serial_linear_for` 改成"进入保存、退出**恢复上一个**注册表"（嵌套语义），并回归全部旧门 |
| ④ | Stage 3 micro 首跑 `checks_failed` | 我把"候选 ON 与 OFF 的 block verify 必须逐位一致"写成硬门 —— 但**候选的目的就是改变这一次执行几何** | 改为：s=1 的 decode 单步等价设硬门；verify ON/OFF 的差异作**观测**记录（`n_diff=538674`、`max|Δ|=0.4375`） |
| ⑤ | Stage 2 汇总器首跑 `KeyError: 'smoke'` | 汇总记录里字段已扁平化，`.pt` 保存处仍按嵌套键取 | 改为 `r.get("gen")` |
| ⑥ | 我用 PowerShell `Set-Content` 重写 `scripts/div8209_preflight_cpu.py` 破坏了文件编码 | 未指定 UTF-8 | 从 git 恢复后用编辑器重做（教训：UTF-8 文本一律不用 `Set-Content` 重写） |
| ⑦ | 空集冒烟：决策/汇总脚本在没有数据时跑过一次 | 我在写完脚本后立刻试跑 | 该空集产物已删除（不是数据）；脚本在有数据后重跑 |

## 11. 正式 B6 状态

**未改变**：本轮**没有**跑正式 B6、acceptance、pilot、final，**没有**修改或放宽任何正确性门槛，
**没有**把候选设为默认生产模式（它默认关闭、只在显式安装包装时对 target verify 生效），
**没有**改 token 接受公式/停止条件/draft。B6（`SpecDec ids == target-only greedy ids` 且全有限、多轮确定）
**仍未通过** ⇒ **acceptance / pilot / final 仍然禁止**。
本轮是**候选模式评估**：它证明候选能让**已登记的 F13 失败案例**通过 token-ID 等价，
但代价为 decode 1.26×（固定 block verify 1.69×），相对 AR 的吞吐仅 0.176×。

## 12. 下一步（**项目级，不含任何 L1–L27 算子级诊断**）

**不再提出 L1–L27 逐算子定位。** 基于本轮四选一结果（`CORRECT_BUT_NO_SPEED_VALUE`），
项目级的选择是二选一（本轮**不执行**）：

* **(A) 把候选登记为"数值对齐选项"，但不作为默认**
  —— 保留 `set_verify_wrap` 与 `row_aligned_verify` 作为**默认关闭**的诊断/对齐工具，
  在需要"与 AR 逐 token 一致"的场合（如正确性复核）显式启用；
  同时**不**把它纳入生产解码路径（因为它把 decode 拖慢 1.26×、相对 AR 只有 0.176×）。
* **(B) 提升 SpecDec 本身的效率前提**（这是本轮真正的项目瓶颈）
  —— 当前 8K / all-eager 配置下 SpecDec 的 decode 吞吐只有 AR 的 **0.22×**，
  连"能加速"的前提都不成立；要把 SpecDec 变成有价值的方向，必须先解决
  **prefill/decode 的执行效率**（例如官方 attention 后端/量化 KV 等**已有设计冻结文档**里的方案），
  而不是继续在数值对齐上投入。

**判据（二值）**：(A) 只在"确实需要 token-ID 与 AR 严格一致"的用途下采用；
若目标是**加速**，则必须先达成 (B) 的前提（SpecDec 至少不能慢于 AR），再谈正确性对齐。

**纪律**：本轮不执行上述任何实验；不做 profiler、不切后端、不改生产默认路径、不做 FP32、
**不跑正式 B6/acceptance/pilot/final**、不外推到其它位置/family/模型。
