# BF16 B6 正确性覆盖：审计 + **修订后执行计划**（2026-10-05，rev.2）

> **本轮状态**：C-1/C-2/C-3 **工具最小改动已实现并通过 CPU 自测**（33/0）；
> **GPU 仍未运行**——本文件提交**确切清单**等待确认（"本条不授权未明确列出的 GPU 运行"）。
> **结论边界不变**：`STAGE2B_PASS`（限定范围）≠ `B6_BF16_PASS` ≠ `BF16_FORMAL_READY`；
> 不得由本报告推出完整 B6、FP8 门通过、正式实验放行或"同输出无损加速"。
> 正式实验代码（runner / 采集 / 统计 / 绘图 / final pipeline）由用户本人编写，本轮**未创建、未修改**。

## 1. 授权与边界（本轮）

| 项 | 状态 |
| --- | --- |
| **已获批准并已完成** | C-1（固定字面 prompt + 专项生成长度）、C-2（记录诊断精度/实际 backend/融合阶段/输入 hash）、C-3（从 trace 检查实际分支覆盖）**的最小测试工具改动 + CPU 自测** |
| **未获批准** | ① 修改数值路径 / 生产默认配置 / 验收门槛；② 正式实验代码；③ **任何未在 §4 明确列出的 GPU 运行** |
| **仍未实施（待批）** | §4 的 60 runs、8K 融合复核 6 runs、诊断 pass（若需要） |

## 2. 冻结候选配置符合性（修正后）

| 配置项 | 期望 | 证据 | 结论 |
| --- | --- | --- | --- |
| 计算精度 / target KV / draft KV | BF16 / `base_bf16` / `base_bf16` | `input.dtypes.model`、`target_kv_payload`、`draft_kv_payload` | ✅ 已取证（24/24） |
| k / chunk / ctx / max_new（常规真实模型用例） | 4 / 2048 / 8192 / **64** | `smoke_config` | ✅ |
| **attention = `cudnn_prefill`（仅 prefill 融合，其余 eager）** | `cudnn_prefill` | Stage 2b 的 24 run 是 **`arm=eager`**（`attention_stats.strategy="eager"`、`fused={}`） | ❌ **未覆盖融合 prefill 路径** ⇒ **本轮 GPU 清单改为优先跑候选 backend**（§4） |
| diag preset=eager、`qk_fp32=False`、`prob_fp32=False` | False/False | Stage 2b 产物**未记录**（`diag_precision=None`） | ⚠️ **上一轮未取证**；**本轮 C-2 已让运行器落盘**（`diag_precision_before_run` / `after_run` / `stable`）⇒ 新 run 将自动取证 |
| EOS 按冻结规则 / AR=target-only / 三模式同配置 | — | 24 run 全部 `max_new_reached`；`mode=ar` 且 `draft=not_loaded` | ✅（EOS **规则未被触发**，见 §3） |

## 3. 工具最小改动（C-1/C-2/C-3）— 逐项说明

| # | 改动（文件） | 为什么需要 | 输入 → 输出 | 证明什么 | **不能证明什么** |
| --- | --- | --- | --- | --- | --- |
| **C-1** | `family_input.resolve_input_plan()` 新增 `literal_prompt` 模式；`g5_8k_smoke_cudnn.py` 新增该分支（tokenizer 编码，`add_special_tokens` 显式）；`serial_oracle_gpu.py` 透传 `SO_PROMPT`/`SO_PROMPT_ID`，**不再设** `SMOKE_FAMILY` | 数学/代码/短/长/EOS/非 k 倍数用例必须用**固定输入**；原 runner 只能按 manifest family 取 8K 窗口 | prompt 文本 + `n` → 同结构产物（新增 `input.source_kind="literal_prompt"`、`prompt_id`、`prompt_sha1`、`input_tokens`） | 该 run **确实使用了声明的输入**（可复现），且**实测输入长度**如实记录 | 不证明任何 manifest family 的性质；不证明"自然语言全覆盖" |
| **C-2** | `serial_oracle_gpu.py` 落盘 `diag_precision_before_run` / `diag_precision_after_run` / `diag_precision_stable`、`backend_request`、`input_plan`；`rules` 字符串改为**按本次实参**生成 | 冻结候选配置含精度开关与后端；原产物**未记录** ⇒ 无法取证 | 无 → 产物新增字段 | 该 run 的**精度开关实际取值**（False/False）与"运行前后未变"；`attention_stats` 早已记录后端实况 | 不证明数值结果正确；**不改**任何计算 |
| **C-3** | 新增 `b6_branch_coverage_cpu.py`（只读 CPU 检查器）+ `b6_cases.py`（**用例单一来源**） | "自然语言 prompt 不保证触发某分支"⇒ 覆盖必须由 **trace 证明**；此前结论来自临时脚本 | 产物 JSON + 用例声明 → 逐用例判定：必需字段 / 配置 / 分支 `COVERED`·`UNCOVERED` / serial 门 / 重复一致 / production 分歧（只记录） | 分支**被真实执行过**；缺字段/未触发分支**不得 PASS** | **不**证明正确性；不替代 exact-id 门；`stage_check`/`stage2b_verdict` 是 **family 专用**（`F\d{2}`）⇒ 对字面用例**不适用**，本检查器即其下游校验 |

**新增 CPU 回归（`b6_branch_coverage_selftest_cpu.py`，33/0）**：G1 输入计划四态（legacy/manifest/literal/两者同设⇒显式报错）+ `prompt_sha1` 稳定性；G2 短输入 `ctx=5` 不被判违规、`input_tokens` 如实；G3 EOS 停止 + 非 k 整数倍 + 末轮裁剪 ⇒ `COVERED`；G4 未触发分支 ⇒ `UNCOVERED` 且 `CASE_INCOMPLETE`（**非 PASS**）；G5 缺字段 ⇒ `INCOMPLETE`、`k=3`/`qk_fp32=True`/非 prefill 融合 ⇒ `FAIL`；G6 serial≠AR ⇒ `FAIL`；G7 分歧字段**不串读**（含"误导字段被忽略"）；G8 production 只记录、无"无损"表述；G9 参数非法 ⇒ exit 3 且不写产物；G10 用例表与旧 `g4_b6_correctness.py:CASES` **逐字一致**（`ast` 比对）；G11 集合级分支要求未满足 ⇒ 非 PASS。
**既有 9 套自测全部不变**（guard 33/0 · stage_check 35/0 · verdict 92/0 · diag 25/0 · agg 21/0 · stage0 42/0 · env 24/0 · audit 19/0）⇒ **未指定参数时既有行为不变**。

**两处真实缺陷由自测抓到并修正**（都在新检查器里）：① 把"空容器"当缺字段（`fused_by_phase={}` 是合法值："没有任何阶段融合"）⇒ 改为只有 `None`/空串算缺（`gen` 例外：空列表不算证据）；② 字段**缺失**被误报成"运行前后不一致"⇒ 改为只在**显式 False** 时判配置漂移。真实 Stage 2b 产物上的只读冒烟确认：四个 family 的 `all_accept/first_reject/mid_reject/trim_last_round` 全部 `COVERED`，而 `diag_precision_*` 缺失被正确判为 **`CASE_INCOMPLETE`（证据不足）**、不再误报配置违规。

## 4. **修订后的 GPU 清单**（待确认；本条未授权执行）

### 4.1 三条纪律修正

1. **优先候选 backend**：先跑 `SO_ARM=cudnn_prefill`，**不**先跑 60 次 eager。
   既有 eager Stage 2b 的 24 runs **保留为辅助证据**：不重跑、不改写其适用范围。
2. **三模式互不共享**：AR / production / serial **各自独立进程、串行执行、从头 prefill、不共享 cache**
   （`so_batch.ps1` 一进程一 (case, mode, rep)）。**删除**上一版"production 同输入同进程"的表述。
3. **长度如实记录**：短 prompt 记录**实际 input_tokens**（不写 8192）；`len13/len5/len2` 作为**专项用例**并显式记录 `max_new`（13/5/2），**不改变**正式配置 `max_new=64`。

### 4.2 确切用例（单一来源 `specdec/engine/b6_cases.py`；文本与 `n` 与旧 B6 逐字一致）

| case_id | 类型 | 文本（摘要） | max_new | 声明分支（未触发即未覆盖） | 融合期望 |
| --- | --- | --- | --- | --- | --- |
| `natural-1` | 自然语言 | `The capital of France is` | 16 | — | 应融合 |
| `natural-2` | 自然语言 | `Photosynthesis is the process by which` | 16 | — | 应融合 |
| `math` | 数学 | `3 + 5 = 8. And 7 times 6 is` | 16 | — | 应融合 |
| `code` | 代码 | `def add(a, b): …` | 16 | — | 应融合 |
| `short` | 短输入 | `Hello` | 16 | — | **按分派规则**（不作硬性要求） |
| `long` | 长输入 | 长句（旧 B6 原文） | 16 | — | 应融合 |
| `len13` | 非 k 倍数 | `Q: what is the largest planet?` | **13** | `trim_last_round`、`len_not_multiple_of_k` | 应融合 |
| `len5` | 非 k 倍数 | `Once upon a time` | **5** | `trim_last_round`、`len_not_multiple_of_k` | 按分派规则 |
| `len2` | 预算 < k | `The answer is` | **2** | `trim_last_round`、`len_not_multiple_of_k` | 按分派规则 |
| `eos-prone` | EOS | `Repeat the word cat forever: cat cat …` | 16 | `eos_stop`（未命中 ⇒ **未覆盖**，不换样本） | 应融合 |

**集合级要求**：跨全部用例合计须覆盖 `all_accept` · `first_reject` · `mid_reject` · `rollback` · `trim_last_round`；类型须覆盖全部 8 类。

### 4.3 运行次数与预算（估算，非承诺）

| 行 | 内容 | 次数 | 备注 |
| --- | --- | --- | --- |
| **A（主清单）** | 10 用例 × 3 模式 × 2 次，`SO_ARM=cudnn_prefill`，新 tag `div8209_b6` | **60 runs** | ≈ 25–40 min（短 prompt，单 run ≈ 25–40 s 含进程与模型加载） |
| **B（8K 融合复核）** | manifest family `F09`（8K 窗口）× 3 模式 × 2 次，`SO_ARM=cudnn_prefill` | **6 runs** | ≈ 6–10 min；**需另行授权**（见 §4.6 P-B） |
| **C（诊断，可选）** | 仅当某 case 出现 production 分歧时 | **≤11 次** | 每次 ≈ 25 s；诊断计时**不进** speedup |
| **D（下游校验）** | `b6_branch_coverage_cpu`（CPU） | 1 次 | 毫秒级 |
| **未申请** | eager 对照 60 runs（arm 对比） | 0 | 仅在小组要求时另行提出 |

显存：短 prompt 峰值远低于 8K；行 B 沿用现有薄余量风险（`cuda_free≈0`），必须逐批过环境门。

### 4.4 命令（获批后；示意，逐 case 重复）

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
$env:SO_TAG='div8209_b6'; $env:SO_ARM='cudnn_prefill'; $env:SO_KV='base_bf16'
$env:SO_K='4'; $env:SO_CHUNK='2048'; $env:SO_CTX='8192'
# 逐 case（示例 natural-1；其余 9 个同法，文本取自 b6_cases.py）
$env:SO_PROMPT_ID='natural-1'; $env:SO_PROMPT='The capital of France is'; $env:SO_MAXNEW='16'
.\scripts\so_batch.ps1 -Tag div8209_b6 -Cases natural-1 -Modes ar     -Reps 1,2
.\scripts\so_batch.ps1 -Tag div8209_b6 -Cases natural-1 -Modes prod   -Reps 1,2
.\scripts\so_batch.ps1 -Tag div8209_b6 -Cases natural-1 -Modes serial -Reps 1,2
# 仅当该 case 的 production 与 AR 分歧时：
python -X utf8 -m specdec.engine.correctness_protocol_margins --stage2b `
  --runs-tag div8209_b6 --tag div8209_b6 --families natural-1
# 下游校验（CPU，只读）：
python -X utf8 -m specdec.engine.b6_branch_coverage_cpu --tag div8209_b6 --expect-backend CUDNN
```

### 4.5 停止条件（不自动重试、不换参、不换样本）

serial token mismatch · NaN/Inf · 结构失败 · 重复不一致 · OOM · 原生崩溃 · 环境门拒绝
⇒ **保留产物并停止推进**；环境门**不**旁路、**不**上调安全线；`TIE-*` 无双侧证据时一律 `TIE-C`（不判 `TIE-A`、不豁免）；
EOS 未命中 ⇒ 如实记"未覆盖"（**不**加跑、**不**换 prompt 直到命中）。

### 4.6 待批准项（**未获批准前不启动**）

| # | 待批准 | 内容 |
| --- | --- | --- |
| **P-A** | 行 A：**60 runs**（新 tag `div8209_b6`，字面用例，`arm=cudnn_prefill`）+ ≤10 诊断 | 新 tag 的新 GPU 运行授权 |
| **P-B** | 行 B：**6 runs** 8K 融合复核（manifest family `F09`，`arm=cudnn_prefill`） | 该运行要么扩展现有 D-3 授权（新 tag × 四 family），要么在 `div8209_p2` 下新增 arm ⇒ **两者都是新 GPU 运行**，须单独批准 |
| **P-C** | 是否要求**并列 margin 记账**（GB6）覆盖这些字面用例 | 若要求：每个分歧 case 追加 1 次诊断，并把 GB6 字段纳入校验（`stage2b_verdict` 不接受非 `F\d{2}` id ⇒ 需在 C-3 检查器里加 GB6 完整性项，属**新增能力**，需批准） |
| **P-D** | 是否要 eager 对照 | 本清单**不申请**；如需 arm 对比，另行提出 |

> **治理侧影响（已核对）**：字面用例的记录 `family=null`、`source_kind="literal_prompt"` ⇒
> 守卫的 `outside` / `smoke_other` / 候选消费三条检查**看不到**它们（不构成 manifest family 消费）
> ⇒ **无需修改守卫**；`preflight` 与既有工具亦不受影响（新模块加入后守卫仍 **33/0**）。

## 5. 是否达到 `B6_BF16_PASS` —— **仍未达到（且本轮未运行 GPU）**

已具备：限定范围 `STAGE2B_PASS` + 分支级覆盖证据（126 全接受 / 88 首项拒绝 / 174 中间拒绝 / 24 裁剪，均来自真实 trace）。
未具备：① 数学/代码/短/长（旧用例口径）**无新口径证据**；② **EOS 真实触发 0 次**；③ 非 k 倍数用例未跑；
④ **融合 prefill（候选 backend）路径零覆盖**；⑤ 诊断精度开关在新 run 之前**未取证**（C-2 已修工具，待新 run 落盘）。
逻辑级证据（与真实模型分开报告）：`g4_b5_eos_branch_tests.py` 用 **CPU stub** 覆盖 EOS 分支与末轮裁剪（4 case，含"旧实现必抛 ValueError"的非空过对照）。

## 6. 距 `BF16_FORMAL_READY` 仍缺

B6 证据（§5 五项）与 `FULL_B6_PASS`（还需 `B6_VFP8_PASS`，下一阶段）；
协议批准 **D-12**（按 KV 分别放行 / 全局放行）· **D-5**（质量口径）· **D-7** · **D-9**；
测量前置（speedup 声明资格 10 条、measured reps 完整、无 OOM/NaN/ENV_INVALID，诊断计时不入 speedup）；
工程（入口内存余量薄、`cuda_free≈0`）；未决解释规则（并列双侧分类 P-1/P-2 未实施）。

## 8. rev.3（2026-10-05）：验收器四漏洞修复 + 对计划的影响

**性质**：四个漏洞**全部在验收工具里**（判据/检查逻辑），**不涉及**生产 SpecDec 算法、数值路径或历史结果；
不得据此宣称算法错误。

| # | 漏洞（rev.2 检查器） | 最小复现（旧规则） | 修复 | 复现结果（新规则） |
| --- | --- | --- | --- | --- |
| 1 | **完整序列一致性**：只看重叠区间 `first_divergence` ⇒ `AR=[1,2,3,4]`、`serial=[1,2]` 判"无分歧" | `OLD_PASS` | 严格门改为**完整 token 列表相等**（长度不同也算不一致），报告共同前缀 / 两侧长度 / 长度差异位置 | **`CASE_FAIL`** |
| 2 | **硬检查只判存在**：`self_checks_all_ok`、`cache_structure_ok`、`prefill_logits_finite`、`round_invariants.all_ok` 全 False 仍 PASS | `OLD_PASS` | **逐项验值**（调用计数恒等式 / 有限性 / cache 结构 / 控制不变量 / 运行完成 / 杀开关复位 / 停止合法性）+ `gen_len`↔长度↔`max_new`↔`stop_reason`↔EOS 自洽性；缺失⇒`INCOMPLETE`、显式失败⇒`FAIL`、类型非法⇒非 PASS | **`CASE_FAIL`** |
| 3 | **AR 融合被误拒**：旧版对 AR 强制 `expect_prefill_fused=False` | `OLD_FAIL`（误拒） | AR 与 Spec 走**同一 target prefill 策略**；按分派证据（`strategy`/`requested_backend`/`fused_by_phase`/`s_max_by_phase`/`fallbacks`/`eager_reason_kinds`）判定；**禁止**非 prefill 融合；`expect=None` **≠** 不检查（须"真融合"或"有记录的回退理由"，否则证据不足）；声明应融合却跑 `arm=eager` ⇒ **未被检验**（`INCOMPLETE`）；`target/draft` 调用不可分 ⇒ **显式证据边界** | **`CASE_PASS`** |
| 4 | **跨模式/跨 rep 不匹配**：无输入与配置核对 | `OLD_PASS` | 同 (case,rep) 的 AR/prod/serial 必须同 input hash、同实际长度、同 target dtype/KV、同 backend 与 prefill 策略、同 `chunk`/`k`/`max_new`/EOS/冻结数值配置；同模式跨 rep 亦然；**AR 未加载 draft 属预期差异**；**AR 误加载 draft ⇒ 基线不纯 ⇒ FAIL** | **`CASE_FAIL`** |

**顺带修复（产物选择）**：`find_runs()` 不再静默择一 —— 同一 `(mode,rep)` 重复产物、非法 `rep`、
身份字段（tag/case/mode）与请求不一致，**全部显式报告并判非 PASS**。

**CPU 回归**：`b6_branch_coverage_selftest_cpu` **83/0**（保留原 33 项断言，新增 G12–G20：
严格前缀/同长度差异、9 类硬检查 × {非法取值/缺失/类型非法}、自洽性四项、AR 融合不误拒、
`expect=None` 两种情形、跨模式/跨 rep/AR-draft 六项、产物选择三项、完整夹具 PASS、production 只记录）。
全量套件 **374/0**；既有 9 套自测数值未变。

**对历史 Stage 2b 产物的只读核验（不改历史判决）**：四 family 全部 **`CASE_INCOMPLETE`**，
理由是**缺 `diag_precision_before_run/after_run/stable` 这一新增必需字段**（C-2 之前的产物本就没有）；
同时 `硬检查 0/84 非 ok`、`serial vs AR` 全部 `EQUAL`、production 为 F09/F04 `EQUAL`、F03/F11 `TOKEN_DIFF`。
⇒ 这是**证据不足**（旧产物缺字段），**不是**算法/数值问题；历史判决与产物一律未改。

**对 GPU 计划的影响**：**计划仍可执行** —— 检查器 CLI 未变，§4 的行 A/B 命令无需改动；
且新运行由 rev.2 运行器记录 `diag_precision_*`、`input_tokens`、`attention_stats`（含分派证据），
足以满足新的必需字段与匹配要求。**剩余阻塞项不变**：**P-A**（60 runs 新 tag 授权）、
**P-B**（6 runs 8K 融合复核：`F09`+候选臂，需扩展 run 级授权）、**P-C/P-D**（GB6 记账与 eager 对照是否要做）。
**本轮未运行任何 GPU 任务**，`div8209_b6` 产物数 = 0。

## 9. rev.4（2026-10-05）：诊断精度验收修复 + **候选后端 GPU 清单**（待确认，本轮不启动）

### 9.1 误放行漏洞：`diag_precision_*` 只判"存在"

**最小复现（修复前）**：把有效夹具的三个字段改成 `diag_precision_before_run="invalid"`、
`diag_precision_after_run="invalid"`、`diag_precision_stable="invalid"` ⇒ `evaluate_case` 返回 **`CASE_PASS`**。
**原因**：该字段不在 `REQUIRED_PATHS` 的必要判定里、不在类型表里，且 `config_violations` 只在
"是字典"时检查开关 ⇒ 三条路径全部绕过（"存在即通过"的同类错误）。

**修复**（`diag_precision_checks()`，逐条验值、**不用 `bool()` 强制转换**）：
1. `before_run`/`after_run` **必须是字典**（否则 `illegal` ⇒ 非 PASS）；
2. 两份字典都必须含 `qk_fp32`、`prob_fp32`（缺键/键为 `None` ⇒ `missing` ⇒ **证据不足**）；
3. 开关必须**严格为布尔 `False`**（`0`/`1`/`"False"`/`None`/`True` 一律非 PASS；其中显式非 False ⇒ `fail`）；
4. `diag_precision_stable` 必须**严格布尔 `True`**（字符串/整数/`None`/`False` 非 PASS）；
5. 检查器**自行比较前后两份记录**是否一致，**不只信任** `stable`；
6. 原因分类清楚区分：**证据不足（missing）** vs **类型非法（illegal）** vs **明确失败（fail）**；
   `stable=False` 与"前后不同"分别报出；
7. **不限制**记录只能有两个键：其它字段原样保留、不参与比较；**比较范围仅 `qk_fp32`/`prob_fp32`**（已写明）。

**修复后复现结果**：三字段全 `"invalid"` ⇒ **`CASE_INCOMPLETE`**（原因"诊断精度类型非法"）；
缺 `qk_fp32` 键 ⇒ `INCOMPLETE`（"诊断精度证据不足"，**不**升级为 FAIL）；
`qk_fp32=True` ⇒ `FAIL`；`qk_fp32=0` ⇒ `FAIL`；`stable=1` ⇒ `INCOMPLETE`（类型非法）；
前后不同但 `stable=True` ⇒ `FAIL`（由**自行核对**抓出）；合法一致 ⇒ `CASE_PASS`。

**CPU 回归**：`b6_branch_coverage_selftest_cpu` **115/0**（保留原 83 项 + 新增 G21a–G21o：
三字段字符串/列表/None、缺任一开关键、开关 `0`/`1`/`"False"`/`None`/`True`、`stable` 字符串/整数/None/False、
前后不同但 `stable=True`、合法一致仍 PASS、允许其它合法字段、比较范围已声明、
旧产物缺字段只记证据不足且序列一致性不受影响）。全量套件 **406/0**。

### 9.2 候选后端 GPU 清单（**仅列清单，本轮不启动**）

* **tag**：**`div8209_b6`**（新 tag；不复用 `div8209_p2`，避免与既有 24 runs 的 rep 编号冲突）
* **配置（行 A/B 同）**：`SO_ARM=cudnn_prefill`（**候选 backend，优先跑**）· `SO_KV=base_bf16`（target 与 draft 均
  `base_bf16`）· `SO_K=4` · `SO_CHUNK=2048` · `SO_CTX=8192` · greedy · `bf16_red_red=1` · `struct_check=1`；
  诊断精度开关 `qk_fp32=False`、`prob_fp32=False`（由运行器落盘取证）；EOS 按冻结规则
* **输入**：字面 prompt（`SO_PROMPT`/`SO_PROMPT_ID`，取自 `specdec/engine/b6_cases.py` 单一来源，
  文本与 `n` 与旧 `g4_b6_correctness.py:CASES` **逐字一致**）；生成长度 `SO_MAXNEW` 按用例
  （16/16/16/16/16/16/**13/5/2**/16）；**正式配置仍 `max_new=64`**（专项用例不改默认值）
* **进程纪律**：AR / production / serial **各自独立进程、串行执行、从头 prefill、不共享 cache**
  （`so_batch.ps1` 一进程一 `(case, mode, rep)`）；每 `(case, mode)` **2 次独立重复**
* **次数**：

| 行 | 内容 | 次数 |
| --- | --- | --- |
| **A** | 10 用例 × 3 模式 × 2 次（`div8209_b6`，`cudnn_prefill`） | **60 runs** |
| **B** | manifest family `F09`（8K）× 3 模式 × 2 次（`div8209_b6`，`cudnn_prefill`） | **6 runs** |
| **C** | 诊断：仅当某 case 的 production 与 AR 分歧（每 case ≤1） | **≤11 次** |
| **D** | 下游校验 `b6_branch_coverage_cpu`（CPU） | 1 次 |
| — | eager 对照 60 runs | **未申请** |

* **命令**：见 §4.4（逐 case 设 `SO_PROMPT*`/`SO_MAXNEW` 后三模式各跑 `-Reps 1,2`；
  分歧时跑 `correctness_protocol_margins --stage2b --runs-tag div8209_b6 --families <case>`；
  最后 `b6_branch_coverage_cpu --tag div8209_b6 --expect-backend CUDNN`）
* **停止条件**：serial 与 AR **完整序列**不相等（含长度不同）· 硬检查显式失败 · 必需字段/键缺失或类型非法 ·
  诊断精度开关非严格 `False` / `stable` 非严格 `True` / 前后不一致 · NaN/Inf · 结构失败 ·
  两次重复不一致 · OOM · 原生崩溃 · 环境门拒绝 ⇒ **保产物并停**；**不重试、不换参、不换样本**；
  EOS 未命中 ⇒ 如实记未覆盖（**不**追加运行）；环境门**不**旁路、**不**上调安全线；诊断计时**不进** speedup 结论
* **资源预算（估算，非承诺）**：行 A ≈ **25–40 min GPU**（短 prompt，单 run ≈ 进程+模型加载 20–30 s）
  · 行 B ≈ **6–10 min**（8K，显存余量薄，须逐批过环境门）· 行 C ≈ **≤5 min** ⇒ 合计 ≈ **36–56 min**
* **待批准**：**P-A**（行 A）· **P-B**（行 B，含把新 tag 纳入 run 级授权）· **P-C**（是否纳入 GB6 记账）· **P-D**（eager 对照）

## 10. 本轮**未执行**的部分（明确列出）

* **未运行任何 GPU 任务**（无非授权 GPU 使用；`div8209_b6` 尚无产物）；
* **未实现**：8K 融合复核的运行入口以外的任何新能力；`stage2b_verdict` 对字面用例的支持；并列双侧捕获 P-1/P-2；B6 覆盖轮之外的任何测试矩阵扩展；
* **未改动**：数值路径、生产默认配置（`arm` 默认仍 `eager`、`SMOKE_MAXNEW` 默认不变）、验收门槛、历史 FAIL/CHECK/PASS、proposal 与冻结判据文件；
* **未创建**：正式实验 runner / 采集 / 统计分析 / 绘图 / final pipeline（所有权属用户）。

## 11. rev.5（2026-10-05，**本轮已执行**）：编排落地 + 两处验收缺陷修复 + 66 次 run 的实际结果

> 本节只写"已经发生了什么"；**结果与验收结论见 `docs/b6_bf16_coverage_result_2026-10-05.md`**（不在此重复）。

* **已执行**：P-A 60 runs + P-B 6 runs = **66 runs**（新 tag `div8209_b6`，`SO_ARM=cudnn_prefill`），
  全部 `rc=0`；**失败即停**机制在第一次批次中真实触发过 1 次并正确停住（见下），修复后 `-Resume` 续跑完成；
  `F09` 8K 复核证明**融合 prefill 路径真的被执行**（`fused_by_phase={'prefill': 112/208}`、`fused_s_max=2048`）。
* **新增工具**（纯 CPU，见结果文档 §8）：`b6_orchestrate_cpu.py` + 两个自测（39/39、35/35）+
  `so_batch.ps1` 的最小改动（失败即停 / 分组验收 / 不覆盖不重跑 / 逐用例配置单一来源 / 进度清单 / 最终验收退出码）。
* **新 tag 的 run 级授权已登记**：守卫的授权范围由"单一 tag"扩为**授权 tag 集合**
  `{div8209_p2（批 1）, div8209_b6（B6 轮）}`；`tool` 与 `families` 口径**逐字未变**，其它 tag/tool 仍判越界；
  守卫 33/0、预检 `PREFLIGHT_OK`（rc=0）。
* **修了两处验收缺陷**（均为**分类/断言**问题，非模型或算法问题；历史产物与历史判决一律保留）：
  ① B6 验收器把 harness 的"非必需且未执行"检查误判为失败（rev.5 改为逐条核对 `required/executed/ok`）；
  ② `assert_strategy_used()` 与既有短块分派规则的矛盾（**仅改断言**，不动分派/阈值/精度/数值路径）。
  两处都补了 CPU 反例回归，并对**真实产物**做了改前/改后对照（历史 24 份裁定完全不变）。
* **仍未达成 `B6_BF16_PASS`**：唯一原因是 `eos-prone` 的 `eos_stop` **未被触发**（覆盖缺口，按预登记规则如实记录、
  不换样本）；9/10 用例 `CASE_PASS`、0 `FAIL`、集合级分支 5/5 与类型 8/8 均 `COVERED`。
* **与本轮改动无关的既有 CPU 失败**：`qk_scope_selftest_cpu` 33/3（干净 HEAD 上同样失败；属生产数值路径领域，未获授权、未触碰）。
