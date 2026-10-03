# 未见数据验证报告：分层正确性协议（24 runs，2026-10-03）

> 协议（**运行前冻结**）：`docs/correctness_protocol_post_shape_localization_2026-10-03.md`（`post_shape_localization_v1`）。
> 本报告是该协议下的**执行结果**。协议原文**未被修改**。

## 0. 最终决策

# **`ALGORITHMIC_GATE_FAIL`**

**四个预登记 family 中有 3 个（F02 / F08 / F10）的 diagnostic oracle 生成 token IDs 与 target-only AR 不一致**
⇒ 协议 §2.1 的硬门失败 ⇒ 按协议 §7：**保留产物、停止、不进入正式实验**。
正式 B6 与 factorial **未启动**，**B6 状态未改变**。

同时如实记录：oracle **并非无效** —— 在 F01 上它让 AR/Spec 的 token 完全一致（production 在 index 14 分歧），
在 F02/F10 上把首分歧从 7/8 推迟到 9/17，在 F08 上停在同一点 52。它只是**不足以**作为正确性保证。

## 1. 预登记的 4 个 family 与独立性证明

**选择规则（先于任何结果写下）**：从 manifest 的四个 stratum 各取 `family_id` 最小者，排除 **F13**（当前定位案例）
与 **F06**（已参与 row-aligned 对照）。

| family_id | stratum | source | window_token_start | `text_sha1` |
| --- | --- | --- | --- | --- |
| **F01** | `S4_mixed` | `wikitext-2-raw-v1/wiki.train.raw` | 0 | `36cdebc500b31686` |
| **F02** | `S2_high_markup` | 同上 | 8192 | `c9cde291f999d61e` |
| **F08** | `S1_low_entropy` | 同上 | 24576 | `aae3dcc645738ebe` |
| **F10** | `S3_prose` | 同上 | 8192 | `29cc721a3e0265d4` |

覆盖 **4 个 stratum**（要求 ≥3 ✓）；不含 F13/F06 ✓；输入由 `family_input.load_family_ids()` 按 manifest 重建
并做 **`token_sha1` 校验**（`verify_hash=True`；不匹配即拒绝、不降级）。

**独立性证明**（运行前）：
1. `python -X utf8 -m specdec.engine.verify_family_independence_cpu` ⇒ **PASS 25 / FAIL 0**；
   `family_use_authorization.outside_authorized = []`（授权范围之外没有任何 run 消费 manifest family）。
2. 该守卫的既有-run 扫描（**101** 条消费记录）里，**只有 F06 与 F13 被消费过**。
3. 对 `output/g5/**/*.json` 全量检索 `family_id: "F01|F02|F08|F10"` ⇒ **命中文件数 = 0**
   ⇒ 这 4 个 family **从未被任何 run（含诊断脚本）消费** ⇒ 属于**未见数据**。

## 2. 逐 family 聚合结果（24 runs：4 × 3 × 2）

配置：target 1.5B + draft 0.5B / ctx 8192 / chunk 2048 / k=4 / **max_new=64** / BF16 / `base_bf16` /
all-eager / target RoPE CPU-FP32 / greedy / 串行 / 不共享 cache / **每次独立进程**。

| family | AR 可重复 | production 可重复 | diagnostic 可重复 | diagnostic == AR | production == AR | first div(prod) | first div(diag) | 决策位置 margin | acceptance(prod/diag) | 结构检查 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **F01** | ✓ (2/2) | ✓ | ✓ | **✓（64/64）** | ✗ | **14** | **无** | 6.75（非近并列） | 0.441 / **0.704** | 全过 |
| **F02** | ✓ | ✓ | ✓ | **✗** | ✗ | 7 | **9** | 0.125 | 0.381 / 0.277 | 全过 |
| **F08** | ✓ | ✓ | ✓ | **✗** | ✗ | 52 | **52** | 0.125 | 0.643 / 0.643 | 全过 |
| **F10** | ✓ | ✓ | ✓ | **✗** | ✗ | 8 | **17** | **0.0（并列）** | 0.422 / 0.458 | 全过 |

* **AR 可重复**：四个 family 的 AR 两次独立进程 `gen_hash` 与完整 token IDs **逐 token 相同** ✓。
* **production 可重复**：两次独立进程一致 ✓（`reps_consistent=True`）。
* **diagnostic 可重复**：两次一致 ✓。
* **结构检查**（四 family × 三模式全部通过）：`finite`（prefill logits 逐模型有限）、`gen_ids_in_vocab`、
  `legal_stop`（全部 `max_new_reached`，**无 `other_short`**）、`length_ok`（`gen_len == 64`）、
  `round_invariants_ok`（逐轮 `tgt_len == drf_len == pre + (m+1 或 Kp)`、`emitted` 长度与 `m/Kp` 一致）、
  `decision_coverage`（决策步覆盖）✓、cache 长度/结构/dtype ✓、`diagnostic_only` 标记正确
  （ar/prod=false、diag=true）✓、**hook 作用域**：diag 模式的 `applied` 计数与 `target_block_calls` **一致**
  （`counts_match=True`）✓、无 OOM ✓。
* **P-state**：每次运行记录（harness 自检通过，未出现不合格即拒绝的情况）。
* **峰值显存**：记录在每次运行的 `mem_timeline_last`（diag 与 prod 同量级；无 OOM）。

### 2.1 分歧位置的 near-tie 证据（决策位置 margin）

| family | 决策位置 | AR top-1 / top-2 | margin | AR 选的 token | diag 选的 token |
| --- | --- | --- | --- | --- | --- |
| F01 | 63（无分歧，取末位） | 6013 (23.75) / 476 (17.0) | 6.75 | 6013 | 6013 |
| F02 | 9 | 9664 (23.5) / 2115 (23.38) | **0.125** | 9664 | 2115 |
| F08 | 52 | 419 (23.38) / 279 (23.25) | **0.125** | 419 | 279 |
| F10 | 17 | 862 (23.62) / 279 (23.62) | **0.0（完整并列）** | 279 | 862 |

⇒ 三处分歧都落在**近并列/并列**的 logits 上（margin 0.125 或 0.0）。这与"有限精度下形状相关的数值非不变性
在近并列处翻转 argmax"的机制一致；**但** F02/F08/F10 说明：**仅仅对齐那五类 GEMM 的执行几何不足以**
在所有未见数据上恢复 AR 的 token（F10 更是在**完整并列**处翻转）。

**F10 的并列细节（如实记录）**：该位置 AR 的 top-1/top-2 在 bf16 下**取值相同**（margin = 0.0），
`argmax` 取**最小 index** ⇒ AR 取 279；诊断路径在该位置选了 862 ⇒ 两者在该位置的判决行**不完全相同**
（差异来源不在本轮允许定位的范围内，**按纪律不做算子级追查**）。

## 3. 24 次运行的机器可读产物

* 每次运行：`output/g5/attn/div8209_proto_{F01,F02,F08,F10}_{ar,spec,aligned}_rep{1,2}_<UTC>.json`
  （含 provenance、`diagnostic_only`、逐轮表、统计量、峰值、P-state、stop reason、完整 token IDs）+ 同名 `.pt`（生成 IDs）
* 聚合：`div8209_proto_aggregate_20261003T120349Z.json` + **`.jsonl`**（每 (family, mode) 一行 + 末行决策）
* margins/profiler：`div8209_proto_margins_20261003T120234Z.json`
* **无覆盖**：所有产物使用新 tag `div8209_proto`；既有 JSON/JSONL 未被改写。

## 4. 可选 profiler 收尾（已执行，结果：不可用）

| 检查 | 结果 |
| --- | --- |
| 本机是否有 `nsys` | **无** |
| 本机是否有 `ncu` | **无** |
| `torch.profiler` CUDA activity 是否产出 kernel 事件 | **否**（`n_events=26`、`n_cuda_events=0`） |
| 结论 | **`PROFILER_UNAVAILABLE`**（**未安装任何软件、未重试、不阻塞本轮验证**） |

## 5. 逐步解释：为什么这样设计、输入输出、指标含义、结果如何推动、为什么必须分开

| 步 | 为什么这样设计 | 输入 → 输出 | 指标含义 | 结果如何推动下一步 |
| --- | --- | --- | --- | --- |
| **冻结协议（先文档后 GPU）** | 防止"看到结果再改判据"；预登记 family 防止挑样本 | 协议文本 → 冻结的 Gate A/B 与决策空间 | 判据在运行前固定 | 之后所有结果都只能落到三选一决策上，不允许解释成 PASS/FAIL/harmless |
| **独立性证明** | 未见数据必须**真的没见过** | guard（25/0）+ 101 条消费记录 + 全量检索 → 4 个 family 零消费 | "未见"= 没有任何 run 消费过该 family | 证明通过后才允许运行；否则记 CHECK |
| **AR 基线（每 family 2 次）** | 参照必须是**同一冻结配置**下的 target-only 贪婪生成 | manifest 窗口 → 64 token IDs | AR 可重复性 = 环境与实现是否确定 | 只有 AR 两次一致，后面的比较才有意义 |
| **production 模式（2 次）** | 后续性能实验只能用这个模式 | 原始 `M=4` block verify → 64 token IDs + 结构量 | token 分歧本身只记 **`NUMERICAL_DIVERGENCE_RECORDED`** | 提供"生产路径当前表现"的基线；**不**自动判好坏 |
| **diagnostic oracle（2 次）** | 只用它验证控制逻辑（accept/replace/rollback/cache） | 全层 row-aligned verify（仅 target verify）→ 64 token IDs | `diag == AR` 才说明"执行几何"是唯一剩余来源 | 本轮的 **FAIL** 正来自这一栏 ⇒ 结论是"oracle 不充分" |
| **margin/分歧位置提取** | 判断分歧是否发生在近并列处 | 按 AR 口径重建决策行 → top-1/top-2/margin | margin=0 ⇒ 完整并列；0.125 ⇒ 1 ULP 级 | 三处分歧 margin ≤ 0.125 ⇒ 与"近并列翻转"一致，但**不足以**解释全部 |
| **聚合与判决** | 判据必须是**独立复算**的，不靠单次运行的自述 | 24 份产物 → Gate A/B 与三选一决策 | Gate A = 控制逻辑正确性；Gate B = 生产路径可测量性 | Gate A FAIL ⇒ **停止**（不启动正式实验） |
| **为什么必须分开** | diagnostic 改的是**执行几何**，与生产模式是**两条不同的数值路径** | — | — | 混用会犯两类错：(a) 用 oracle 的正确性冒充生产正确性；(b) 用 oracle 更慢的时间冒充生产性能。本协议用模式名 + `diagnostic_only` 物理隔离，并把 oracle 明确排除在一切速度结论之外 |

## 6. 决策与建议

* **最终决策：`ALGORITHMIC_GATE_FAIL`**（协议 §7 的硬失败分支）。
* **不得**启动正式 factorial / B6；**B6 状态未改变**（acceptance/pilot/final 仍禁止）。
* **下一步（项目级，不是算子级）**（本轮**不执行**）：
  1. **承认 oracle 不充分**：全层 row-aligned 只对齐了五类 GEMM 的**执行几何**，
     而 F02/F08/F10 表明仍有**其他**近并列翻转来源（本轮按纪律**不**定位它）；
  2. 因此在"严格 token equality"与"SpecDec 加速"之间必须做**项目级取舍**：
     要么接受"生产模式会与 AR 在近并列处偶发分歧"（把它当**已知限制**记录，并在性能实验里如实报告分歧率），
     要么把整条 verify 路径换成**与 AR 同几何**的实现（已知代价 decode 1.26×、verify 1.69×，
     且在本配置下 SpecDec 本身只有 AR 的 0.22× ⇒ 负收益）；
  3. 若选前者，性能实验的**前置条件**是：把"与 AR 的分歧率"作为**一等指标**纳入协议（而不是当作失败），
     并在报告里始终标注 `NUMERICAL_DIVERGENCE_RECORDED`；
  4. 无论选哪条，**不得**把本轮结论写成 cuBLAS/PyTorch bug，也**不得**把数学等价写成逐位等价。

## 7. 治理：家族消费授权的**显式登记**（如实记录一次 guard FAIL）

**运行前**：`verify_family_independence_cpu` = **PASS 25 / FAIL 0**（`outside_authorized=[]`），
且 4 个 family 零消费 ⇒ 独立性成立、允许运行。

**运行后**：同一个守卫变成 **PASS 21 / FAIL 4** —— 这是**正确**的判定，它发现：

| 失败项 | 原因 |
| --- | --- |
| 预期之外的消费方 | 新增文件 `correctness_protocol_margins.py` 读取 manifest family 字段 |
| 消费方集合 ≠ opt-in 集合 | 同上（多了一个文件） |
| **授权范围之外有 run 消费 family** | 本轮 24 次 smoke 运行消费了 **F01**（及其它 3 个），而旧授权范围 = fam16 16 格 + F06/index25 定点取证 |
| smoke 工具的非 fam16 运行只允许 F06/F13 | 同上 |

**处置（按仓库纪律：新消费方必须"显式登记 + scoped 检查"，不得放宽既有检查）**：
1. 把 `correctness_protocol_margins.py` 加入 `OPT_IN_CONSUMERS`，注明**授权来源 = 本轮任务**
   （"从 manifest 选 4 个未参与定位的 family"），日期 `2026-10-03`；
2. 新增常量 `PROTOCOL_ROUND_FAMILIES = ("F01","F02","F08","F10")`，把协议验证轮的 smoke 运行
   纳入**受控授权集合**；
3. 新增**两条 scoped 检查**：
   * "协议验证轮的 smoke 运行**只**消费预登记的 4 个 family"；
   * "`correctness_protocol_margins` 只引用协议预登记的 4 个 family"；
4. **未放宽**任何既有检查；失败记录（21/4）已**落盘保留**：
   `output/g5/attn/_guard_after_protocol_runs_FAIL_21_4.log`；登记后守卫 = **PASS 27 / FAIL 0**
   （25 项原有 + 2 项新登记），另存 `_guard_after_registration.log`。

**这说明什么**：守卫按设计工作 —— 任何新的 family 消费都必须**显式登记并限定范围**。
本轮的登记**不是**对正确性判据的放宽（Guard A/B 与结构检查一字未改），只是把"本轮任务授权的消费"
写进可审计的白名单。

## 8. 已证明 / 尚未证明 / 已排除（本轮）

**已证明（24 runs 实测）**
1. 未见数据上，AR/production/diagnostic **各自的两次独立进程都可重复**（四 family、三模式全一致）。
2. **全部结构检查通过**：finite、无非法 token、合法停止（全部 `max_new_reached`、无 `other_short`）、
   长度 64、逐轮不变量、决策步覆盖、cache 结构、hook 作用域、`diagnostic_only` 标记正确、无 OOM。
3. **diagnostic oracle 在 4 个未见 family 中的 3 个上未能复现 AR 的 token IDs**（F02@9、F08@52、F10@17）
   ⇒ 协议 Gate A **FAIL**。
4. oracle 的部分有效性：F01 上 `diag == AR`（64/64）且 production 在 14 分歧；F02/F10 上把首分歧**推迟**
   （7→9、8→17）；F08 上停在同一点（52）。
5. 三处剩余分歧都发生在 **margin ≤ 0.125** 的位置（F10 为**完整并列 margin = 0.0**）。
6. `PROFILER_UNAVAILABLE`（本机无 `nsys`/`ncu`，且 `torch.profiler` 无 CUDA kernel 事件）。

**尚未证明**
1. 剩余分歧的**具体来源**（按本轮纪律**不**做 L2–L27 算子级定位）；只能说它**不在**已被对齐的五类 GEMM 的执行几何之内。
2. production 模式在**更大样本**上的分歧率（本轮只有 4 个 family、每组 64 token）。
3. 分歧是否会影响下游任务指标（本轮只做 token-ID 比较）。
4. `nsys`/`ncu` 若可用后能否给出 kernel 级证据（未安装、未测）。

**已排除（本轮实测）**
1. **环境/实现不确定性**：AR/prod/diag 的两次独立重复全部一致 ⇒ 不是随机性。
2. **结构性问题**：cache、长度、逐轮不变量、停止原因、非法 token、OOM —— 全部通过。
3. **hook 越界**：`diagnostic_only` 标记与 `applied` 计数证明 oracle 只作用于 target verify。
4. **样本挑选**：family 由**规则**预登记，且独立性证明显示它们从未被消费过；结果不好也**没有**换样本。
5. **"token 一致就等于数值对齐"**：F01 上 `diag == AR`（token 级），但上一轮已证明 logits 仍未逐位一致
   ⇒ 两者是不同命题（本报告继续沿用该口径）。
