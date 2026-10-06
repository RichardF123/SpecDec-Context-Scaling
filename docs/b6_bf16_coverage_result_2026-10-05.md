# B6（BF16 分支/类型覆盖）轮 —— 执行结果与验收结论（2026-10-05）

> 状态口径（每一条都标注）：**已存在**（本轮之前就有）/ **本次新增**（本轮实现）/ **已验证通过**（有产物证据）/
> **已实现未验证** / **未达成（不予声称）**。
> 本文件是**结果汇报**；判据与预登记清单在 `docs/b6_bf16_coverage_audit_plan_2026-10-05.md`。

## 0. 一句话结论

**66/66 次授权 run 全部完成**（`stage=complete`、`env_qualified=True`、`attention_assert=ok`、无 OOM/NaN/非有限项、
无重复槽位）；**10 个用例中 9 个 `CASE_PASS`、0 个 `FAIL`、1 个 `CASE_INCOMPLETE`**；
**serial 与 AR 在全部 10 用例 × 2 重复上严格完整序列相等**；集合级分支 5/5、类型 8/8 均 `COVERED`；
**`B6_BF16_PASS` 未达成、不予声称** —— 唯一原因是 `eos-prone` 的 `eos_stop` **未被触发**（覆盖缺口，
按预登记规则"未命中 ⇒ 未覆盖、不换样本"如实记录，不是失败）。

## 1. 授权与执行清单

| 项 | 计划 | 实际 | 状态 |
| --- | --- | --- | --- |
| P-A 行 A | 10 用例 × 3 模式 × 2 重复 = **60 runs**，tag `div8209_b6`，`SO_ARM=cudnn_prefill` | **60 runs**（54 + `short` 6），全部 `rc=0` | **已验证通过** |
| P-B 行 B | `F09`（8K manifest）× 3 模式 × 2 重复 = **6 runs**，**必须证明真的融合 prefill** | **6 runs**，`--require-fused-prefill` 门**全部通过** | **已验证通过** |
| 合计 | 66 runs | **66 runs**，槽位唯一（无重复、无非法 rep） | **已验证通过** |
| 未运行 | —— | **0**（无中断遗留槽位） | —— |

* 运行顺序（**操作性选择，非科学参数**）：行 A 的 60 次中，`short` 放在**最后**执行（原因见 §5），
  其余 9 个用例按用例表原顺序；这样在"已知会失败"的风险点上不浪费前面的 54 次 run。
* 环境门（每次 run 内 + 独立预检）：`ENVIRONMENT_OK`，commit 干净（`dirty=False`），入口余量 +4.8 GiB ≥ 18 GiB。

## 2. 逐用例结果（产物 = `output/g5/attn/div8209_b6_*`）

| 用例 | 类型 | 实际输入长度 | max_new | 停止原因 | 声明分支覆盖 | serial vs AR（2 reps） | production vs AR（2 reps，**只记录**） | 用例判定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `natural-1` | 自然语言 | 5 | 16 | `max_new_reached` | —— | **EQUAL** | EQUAL | `CASE_PASS` |
| `natural-2` | 自然语言 | 7 | 16 | `max_new_reached` | —— | **EQUAL** | EQUAL | `CASE_PASS` |
| `math` | 数学 | 15 | 16 | `max_new_reached` | —— | **EQUAL** | EQUAL | `CASE_PASS` |
| `code` | 代码 | 20 | 16 | `max_new_reached` | —— | **EQUAL** | EQUAL | `CASE_PASS` |
| `short` | 短输入 | **1** | 16 | `max_new_reached` | —— | **EQUAL** | EQUAL | `CASE_PASS`（融合路径**未覆盖**，见 §5） |
| `long` | 长输入 | 30 | 16 | `max_new_reached` | —— | **EQUAL** | **TOKEN_DIFF**（首分歧 idx=13，共同前缀 13，len 16 vs 16，两重复一致） | `CASE_PASS` |
| `len13` | 非 k 倍数 | 10 | **13** | `max_new_reached` | `trim_last_round` ✓ `len_not_multiple_of_k` ✓ | **EQUAL** | EQUAL | `CASE_PASS` |
| `len5` | 非 k 倍数 | 4 | **5** | `max_new_reached` | `trim_last_round` ✓ `len_not_multiple_of_k` ✓ | **EQUAL** | EQUAL | `CASE_PASS` |
| `len2` | 长度 < k | 3 | **2** | `max_new_reached` | `trim_last_round` ✓ `len_not_multiple_of_k` ✓ | **EQUAL** | EQUAL | `CASE_PASS` |
| `eos-prone` | EOS | 12 | 16 | `max_new_reached`（`eos_seen=False`） | `eos_stop` **UNCOVERED** | **EQUAL** | EQUAL | **`CASE_INCOMPLETE`** |

* 输入长度由 GPU 产物 `smoke.input.input_tokens` 实测，与**运行前**用同一 tokenizer 的 CPU 预测值**逐一相符**
  （预测产物：`output/g5/attn/_b6_prompt_tokenlen_cpu.json`）。
* `F09`（行 B，8K）：6 runs，`input_tokens=8192`，`s_max_by_phase['prefill']=2048`（chunk=2048 ⇒ 4 块），
  `fused_by_phase={'prefill': 112}`（AR：28 层 × 4 块）/ `{'prefill': 208}`（prod/serial：target 112 + draft 96），
  `fused_s_max_by_phase={'prefill': 2048}`，`attention_assert=ok` ⇒ **融合 prefill 路径确实被执行**（`CASE_PASS`，最终验收 `rc=0`）。
  —— 这正是 Stage 2b 的 24 次 run（全部 `arm=eager`、`fused={}`）**零覆盖**的那条路径。
* 集合级：分支 `{all_accept, first_reject, mid_reject, rollback, trim_last_round}` = **5/5 COVERED**；
  类型（8 类）= **8/8 COVERED**；`summary = {n_cases:10, n_pass:9, n_incomplete:1, n_fail:0, set_requirements_ok:True, all_pass:False}`。
* 产物：`div8209_b6_b6_branch_coverage_20261006T105953Z.json`（10 用例最终验收）、
  逐 case 编排判定 `div8209_b6_<case>_b6_orchestrate_*.json`、
  进度 `div8209_b6_b6_orchestrate_progress.json`、事件流 `div8209_b6_b6_orchestrate_events.jsonl`、
  批次日志 `div8209_b6_batch_*.log`（3 个：首次中断 / 续跑 / F09 / short）。

## 2b. production 与 AR 的分歧：**完整序列**记录（协议 v1：只记录，不判控制逻辑失败）

`long` 是唯一出现 production 分歧的用例（其余 9 个用例在 2 个重复上 production 与 AR **完全相等**，
`serial` 与 AR 在**全部 10 个用例 × 2 重复**上**完全相等**）：

| rep | 序列 | token |
| --- | --- | --- |
| 1 | **AR** | `[576, 2070, 702, 3884, 1657, 29903, 11, 714, 714, 279, 2070, 702, 1083, 3884, 1657, 96637]` |
| 1 | **serial** | 与 AR **逐 token 相同**（完整相等） |
| 1 | **production** | `[576, 2070, 702, 3884, 1657, 29903, 11, 714, 714, 279, 2070, 702, 1083, 1012, 65302, 553]` |
| 2 | **AR / serial** | 与 rep1 的 AR 逐 token 相同（跨重复完全一致） |
| 2 | **production** | 与 rep1 的 production 逐 token 相同（跨重复完全一致） |

* 分歧首个位置（**0-based**）= **13**；共同前缀长度 = **13**；两侧长度相同（16 vs 16）；两重复同一位置、同一替换。
* 分叉后的 token：AR/serial = `3884, 1657, 96637`；production = `1012, 65302, 553`。
* **口径**：这是**发散轨迹记录** —— 既不判"控制逻辑失败"，也不自动归为"无害浮点误差"，
  更**不得**据此声称"同输出无损加速"。其性质（并列 vs 真分歧）仍受 **P-1/P-2 双侧决策行捕获未实现**的证据边界限制
  ⇒ 只能记 **TIE-C（双侧证据不足）**，本轮不给出更强结论。

## 3. 版本与可比性（**不得**把跨版本 run 静默视为同一配置）

| commit | 含义 | run 数 |
| --- | --- | --- |
| `0b9c841` | 编排改动（失败即停 + 分组验收 + 新 tag 授权登记），**未含**两处验收缺陷修复 | **10**（`natural-1` 6 + `natural-2` ar/prod 4） |
| `429ff9a` | 两处验收缺陷修复（B6 验收器分类 + 策略断言与分派规则矛盾） | **56**（其余全部，含 `short` 6 与 `F09` 6） |

* 每份产物内 `git.git_commit` / `dirty_worktree=False` 逐 run 可查。
* 第 11 次 run（`natural-2/prod`）在 `0b9c841` 上被**误判为失败**并触发失败即停（§4.1）；该组的 4 份产物**保留未改**。
* 可比性论证（不是"假设相同"）：修复只改**断言的判定分支**，不改任何记录字段；对这批 run 而言，
  两版本下 `smoke.verdict.attention_assert` 均为 `"ok"`（66/66 实测），跨模式/跨重复比较所读的
  `input`/`smoke_config`/`kv_format` 等字段在两版本记录中完全一致，且验收器的跨模式/跨重复一致性检查全部通过。

## 4. 本轮发现并修复的两处验收缺陷（**都不是**模型/算法问题；修复前的历史产物一律保留）

### 4.1 B6 验收器把 harness 的"非必需且未执行"检查误判为失败（rev.5）

* **原来是什么**：`b6_branch_coverage_cpu.hard_checks()` 有一条 `harness_checks_uncovered_empty`：
  「`smoke.checks.uncovered` 非空 ⇒ `CASE_FAIL`」。
* **问题（真实数据触发）**：`smoke_checks.evaluate()` **只在** `meta["required"] is False` 且该检查未执行时才写
  `uncovered`（必需但缺失走 `failures` 且 `passed=False`；`smoke_checks.py:197-204`）⇒ `uncovered` 非空**本身不是失败**，
  而是"该分支本次没执行、因而不在必需集合里"的**如实记录**。
  首次触发：`natural-2` 生产模式 `mismatch_rounds=0 / rollbacks=0 / all_accept_rounds=4` ⇒ `repair` 按设计未执行
  ⇒ `uncovered=['finiteness:repair/draft','finiteness:repair/target']`，同一产物 `passed=True`、`failures=[]`、
  两条逐条记录均为 `required=False, executed=False`；对照 `natural-1`（`mismatch_rounds=6`）repair 必需且已执行、`uncovered=[]`。
  24 份历史产物 `mismatch_rounds≥8` ⇒ 该规则此前**从未**被真实数据触发。
* **改了什么**（**更细，不是放宽**）：① `checks.failures` 必须为空；② `uncovered` 每一条必须存在逐条明细且
  `required=False, executed=False`；③ 所有 `required=True` 的条目必须 `executed=True` 且 `ok=True`。
  另加**只读** `coverage_notes()`：如实记录"这次 run **没有**覆盖什么"。
* **改前/改后对照（同一批真实产物，只读）**：
  * `div8209_b6`：`natural-1` `CASE_PASS`→`CASE_PASS`（0 条非 ok 硬检查，两端一致）；
    `natural-2` **`CASE_FAIL`（2 条非 ok，全部是 `harness_checks_uncovered_empty`）→ `CASE_INCOMPLETE`**
    （0 条非 ok；此时仍 INCOMPLETE 的原因只是它当时**还缺 serial 两槽产物**；续跑完成后 = `CASE_PASS`）。
  * `div8209_p2` 历史 24 份：四个 family **两端完全一致**（均 `CASE_INCOMPLETE`、非 ok 硬检查均为 0）
    ⇒ **历史判决未被改动**。
* **CPU 回归**：`b6_branch_coverage_selftest_cpu` 新增 G22a–G22j（13 条：非必需未执行⇒不失败但记录覆盖缺口、
  缺逐条明细⇒INCOMPLETE、uncovered 含必需/已执行/无明细⇒FAIL、`failures` 非空⇒FAIL、
  必需条目未执行或 `ok=False`（即使 `passed=True`）⇒FAIL、缺字段⇒INCOMPLETE、真融合/短块回退的覆盖记录文案）；
  夹具同步升级为**与真实产物同构**（含 `checks.checks[]` 逐条明细），否则自测只会测到"夹具不像真的"。

### 4.2 `assert_strategy_used()` 与**既有分派规则**的矛盾（仅改断言）

* **原来是什么**：策略=`cudnn_prefill` 且 `calls["fused"]==0` ⇒ 抛"静默不生效"。
* **问题**：`decide_attention_path("prefill", 1)` 对 `s < PREFILL_FUSED_MIN_S(=2)` **按既有规则明确返回 `eager` 并给出理由**
  ⇒ 1-token prompt（用例 `short`，文本 `"Hello"`，实测 1 token）**结构上不可能**产生融合调用，却会被报成"静默不生效"；
  该 RuntimeError 被 `g5_8k_smoke_cudnn.py:696` 捕获后写入 `attention_assert`，而它是**必需检查**
  （`smoke_checks.py:216`）⇒ `stage=run_ended_checks_failed` ⇒ 用例必然不可能通过。
* **改了什么（严格限定，经用户批准）**：只改 `assert_strategy_used()` 的判定分支，
  **不动**分派函数、不动阈值常量、不动精度开关、不动任何模型数值路径（`git diff` 的改动行**全部**落在该函数内）。
  新判定把"**策略执行是否符合规则**"与"**融合路径是否真的被覆盖**"**分开**：
  ① 非 prefill 阶段出现融合 ⇒ 抛；② 存在满足融合条件的 prefill 调用（`s_max ≥ 2`）却零融合 ⇒ 抛；
  ③ 连 prefill 调用都没有 ⇒ 抛（保留原报警意图）；④ 全部 prefill 都短（`s_max < 2`）时，
  必须**逐条**有 `short_block_seq_len_lt_min` 记录、**理由计数 == prefill eager 调用数**、且**没有其它未解释的理由类别**
  ⇒ 判为**合法回退**（附 `strategy_check.verdict="legal_short_block_fallback"`、
  `fused_path_covered=False`、`note="融合路径未被覆盖"`），否则抛（不得凭"存在 short 理由"整体豁免）。
* **CPU 反例回归**：`attn_phase_selftest_cpu` 新增 ⑥a–⑥k（11 条）→ **54/0**；
  其中 ⑥g（存在未解释的理由类别）**在自测中抓到了我第一版实现的漏洞**并据此加严。
* **实测后果**：`short` 的 6 次 run 全部 `stage=complete`、`checks.passed=True`、`attention_assert=ok`、
  `fused=0` 且 `eager_reason_kinds={short_block_seq_len_lt_min: 28/52}`（AR/Prod+Serial）⇒
  验收器如实判 **`CASE_PASS` 但"融合路径未被覆盖"**（见 §5）。

## 5. 覆盖缺口（**已如实记录，不当作通过**）

| 缺口 | 事实 | 处置（预登记） |
| --- | --- | --- |
| `eos-prone` 的 `eos_stop` **未触发** | 6/6 run `stop=max_new_reached`、`eos_seen=False` | 按 §4 表："未命中 ⇒ **未覆盖**，不换样本" ⇒ 该用例 `CASE_INCOMPLETE`，**B6 因此不整体通过** |
| `short` 的**融合 prefill 路径未覆盖** | `in_tokens=1` ⇒ prefill `s=1 < 2` ⇒ 合法回退 eager，`fused=0` | 如实记录为"合法回退通过、**融合路径未覆盖**"；融合路径的覆盖由 **F09 8K 复核**单独证明（§2） |
| `repair` 有限性检查在某些 run 未执行 | `mismatch_rounds=0`（全接受）的 run 中 `repair` 未触发 | 非必需检查（`required=False`）⇒ 记为**覆盖记录**，不判失败（§4.1） |

## 6. 状态判定（六态互不蕴含）

| 状态 | 结论 | 依据 |
| --- | --- | --- |
| `STAGE2B_PASS` | **已存在**（上一轮） | `div8209_p2` 24/24、verdict `SERIAL_ORACLE_PASS`（历史结论未改） |
| `B6_BF16_PASS` | **未达成（不予声称）** | 9/10 `CASE_PASS`、0 `FAIL`、集合级 5/5 与 8/8 均满足，但 `eos-prone` 的 `eos_stop` 未覆盖 ⇒ 用例非 PASS ⇒ 不满足"全用例 PASS" |
| `B6_VFP8_PASS` | **未评估** | 本轮无 FP8 run（FP8 属另一 Gate A，D-6 待批） |
| `FULL_B6_PASS` | **未达成** | = `B6_BF16_PASS && B6_VFP8_PASS` ⇒ 前件即不成立 |
| `BF16_FORMAL_READY` | **未达成** | 需 `STAGE2B_PASS` + `B6_BF16_PASS` + D-12 的 per-KV 规则 + D-5 + 报告纪律 + 环境/预算准入；其中 D-5/D-12 仍是待批决策 |
| `FP8_FORMAL_READY` | **未达成** | 有自己的门；**不得**由 BF16 推断 |

## 7. 仍未决 / 待批项（本轮未做，不得视为已完成）

1. **EOS 覆盖**：要让 `eos_stop` 被真实触发，可选（a）协议上把 `eos_stop` 从 B6 必需分支中裁掉/降级，
   （b）授权新增一个 EOS 诱发用例或加长 `max_new`（**样本替换被明令禁止**）。需用户裁决（与 D-11 相邻）。
2. **协议决策**（沿用上一轮清单）：D-5 质量评分、D-6 FP8 独立 Gate A、D-7 判决词表、D-9 不加额外 FP32、
   D-11 与 `:83` prompt 类型覆盖、D-12 per-KV vs 全局。
3. **P-1/P-2**：决策行**双侧**捕获仍未实现 ⇒ 所有并列（tie）情形仍只能记 **TIE-C（双侧证据不足）**。
4. **既有 CPU 失败（与本轮改动无关，且**未**被修复）**：`qk_scope_selftest_cpu` = 33/3，
   原因 = `model_self.forward` 内 `DIAG_QK_FP32` 出现 **2** 个读取点（期望 1）。
   **证据**：在**干净 HEAD（`0ffa745`，不含本轮任何改动）**的独立 worktree 上同样 33/3；
   `model_self.py` 的 `mtime=2026-10-03`、未修改（`git status` 无该文件）⇒ 该失败**先于**本轮存在，
   属生产数值路径（`forward` / QK scope）领域，本轮**未获授权**且**未触碰**。
5. **诊断精度字段**：旧产物（含 `div8209_p2` 24 份）缺 `diag_precision_*` ⇒ 只会记"证据不足"、不改历史判决；
   新 66 份 run **全部**具备该证据（`qk_fp32=False`、`prob_fp32=False`、`stable=True`，由验收器逐条验值）。

## 8. 本轮交付的工具（**本次新增**、纯 CPU、无 GPU 依赖）

| 文件 | 作用 | 自测 |
| --- | --- | --- |
| `specdec/engine/b6_orchestrate_cpu.py` | 编排级判定（失败即停/分组验收/只记录项/进度清单）；取值规则全部**复用** `b6_branch_coverage_cpu` | 见下 |
| `specdec/engine/b6_orchestrate_selftest_cpu.py` | 模块级编排自测（O1–O22：合法完成/非 0 退出/硬检查失败/缺产物/重复产物/重复不一致/serial 不一致/仅 production 分歧/覆盖缺口/环境拒绝/臂漂移/要求融合/非 prefill 融合/缺字段/类型非法/诊断精度/进度/CLI 退出码/对偶断言/证据边界/覆盖记录） | **39/39** |
| `scripts/selftest_so_batch_orchestration.ps1` + `specdec/engine/b6_orchestrate_stub_runner_cpu.py` | `so_batch.ps1` 的**无 GPU** 端到端编排自测（假运行器 + `SELFTEST*` tag/独立目录双重保护，结构上不可能伪造真实取证） | **35/35** |
| `scripts/so_batch.ps1`（**最小改动**） | 失败即停、每组/case 验收、不覆盖不重跑（`-Resume`）、逐用例配置取自 `b6_cases.py`、进度清单、最终验收退出码、`-File` 参数错绑的显式拒绝 | 同上 |
| `docs/b6_f09_recheck_declared_2026-10-05.json` | 行 B 的用例声明（`expect_prefill_fused=true`、`max_new=64`） | 由 `--require-fused-prefill` 实证 |
