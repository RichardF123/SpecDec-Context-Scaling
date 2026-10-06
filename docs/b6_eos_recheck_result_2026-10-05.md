# 真实模型 EOS 补测 —— 执行结果（2026-10-05）

> 授权：本轮任务书 §一（报告 `docs/b6_eos_qk_closeout_2026-10-05.md` §5.A 的**严格单 prompt**方案）。
> **6 次真实模型 GPU 运行**（AR / production / serial × 2 独立重复），新 tag `div8209_eos`，**不并入** `div8209_b6`。
> 本轮**未**重跑既有 66 次、**未**新增 prompt、**未**自动重试、**未**追加 margin/FP32/FP8/k 扫描/性能测试；
> **未**改生产数值路径与验收门槛；**未**改写任何历史产物或判决（`eos-prone` 仍为 `CASE_INCOMPLETE`）。

## 0. 一句话结论

**6/6 run 的新生成序列都以真实 tokenizer 的 EOS（151645）收尾**（`stop_reason=eos`、`eos_seen=True`、`gen_len=10`），
三模式 × 两重复的 token IDs **完全相同**；`serial==AR`（逐 rep）与跨重复一致性均成立；验收器判
**`CASE_PASS` 且 `eos_stop` = `COVERED`**（集合级要求满足，`FINAL_CHECK_RC=0`）。
⇒ 真实模型 EOS 停止覆盖**已取得**（本轮新增证据，见 §4）；`eos-prone` 的历史 `INCOMPLETE` **保持不变**。
**`B6_BF16_PASS` 不自动宣布** —— 是否由本例补齐 B6 的 EOS 类型缺口属**协议裁决**（§7 的 D-13）。

## 1. 计划 / 完成 / 未完成

| 项 | 计划 | 实际 | 状态 |
| --- | --- | --- | --- |
| 运行数 | 3 模式 × 2 重复 = **6** | **6**（`rc=0` 全通过） | **已完成** |
| 未完成 | —— | **0** | —— |
| 自动重试 | 不允许 | **0 次** | —— |
| prompt | 1 条（预登记逐字） | 1 条（sha1 与预登记一致，见 §2） | —— |
| 中断 | —— | 无（`stopped=False stop_rc=0`） | —— |

## 2. 输入与配置核对（运行前 + 产物实测）

**运行前 CPU 编码记录复核**（`output/g5/attn/_eos_recheck_prompt_cpu.json`）：

| 判据 | 预登记值 | 复核结果 |
| --- | --- | --- |
| 实际输入 token 数 | 30 | **30** ✔ |
| 末 token 不是 EOS | 是（13048「Hi」） | 13048，非 151645 ✔ |
| tokenizer EOS id | 151645（= `EOS_ID_EXPECTED`） | 151645，`matches_expected=True` ✔ |
| 编码设置 | literal 模式、`add_special_tokens=True`、**不**套 chat template | 一致 ✔ |
| `prompt_sha1` | `c9d959169209c4af` | 一致 ✔ |
| prefill 预期分派 | `fused`（30 ≥ 2） | `fused` ✔ |

**产物实测（6/6 一致）**：`input.source_kind=literal_prompt`、`prompt_id=eos1`、
`prompt_sha1=c9d959169209c4af`（**与预登记逐字一致** ⇒ 运行时未重新设计 prompt）、
`input_tokens=30`、`ctx=30`（**未写成 8192**）、`ids_hash=b5e23c80d3ee32e8`。
**输入里的 3 处 `<|im_end|>` 不计入生成证据**：本文所有 EOS 结论只针对**新生成序列**。

**冻结配置实测（6/6 一致）**：`arm=cudnn_prefill`、`k=4`、`chunk=2048`、`max_new=24`、`kv=base_bf16`、
`target_rope=cpu_fp32`、`bf16_red_red=1`、`struct_check=True`、`dtypes.model=torch.bfloat16`、
`target_kv_payload=base_bf16`、`draft_kv_payload=base_bf16`、`cfg_mode=ar|spec` 正确；
`diag_precision_before/after={qk_fp32:False, prob_fp32:False}` 且 `stable=True`；
`row_aligned_wrap_active` 非真 ⇒ **未安装** serial-QK / row-aligned 等残留诊断干预。

## 3. 逐 run 证据（版本 / hash / 生成 / 停止）

| mode | rep | commit | dirty | stage | env_qualified | gen（新生成部分） | gen_len | stop_reason | eos_seen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ar | 1 | `ba41552` | False | complete | True | `[1052, 0, 2585, 646, 358, 7789, 498, 3351, 30, 151645]` | 10 | **eos** | True |
| ar | 2 | `ba41552` | False | complete | True | 同上（逐 token 相同） | 10 | **eos** | True |
| prod | 1 | `ba41552` | False | complete | True | 同上 | 10 | **eos** | True |
| prod | 2 | `ba41552` | False | complete | True | 同上 | 10 | **eos** | True |
| serial | 1 | `ba41552` | False | complete | True | 同上 | 10 | **eos** | True |
| serial | 2 | `ba41552` | False | complete | True | 同上 | 10 | **eos** | True |

* 6 份产物均在**同一 commit `ba41552`（`dirty=False`）**上产生 ⇒ 本组**无跨版本混跑**。
* `eos_id=151645`（`smoke.input.eos_config.eos_id`）；`gen.count(151645)=1` 且位于**末位**（index 9）。

## 4. 验收判据逐条（每 run 26 项，6/6 run 全过）

| # | 判据 | 结果 |
| --- | --- | --- |
| 1 | 6 次新生成序列均以 **151645** 结束、`stop_reason=eos` | ✔ 6/6 |
| 2 | EOS 首次出现在新生成序列**末位**、之后无输出、`gen_len ≤ 24` | ✔ 6/6（`gen_len=10`，`eos` 恰 1 次，位置=末位） |
| 3 | `serial` 与**匹配的 AR** 完整 token IDs 一致（两次重复分别比较） | ✔ rep1 EQUAL、rep2 EQUAL（验收器严格门） |
| 4 | 同模式跨重复：输入/配置/完整输出一致 | ✔ ar/prod/serial 三模式 rep1==rep2；`ids_hash`/`prompt_sha1`/配置字段全部一致 |
| 5 | finite、cache 结构、逐轮不变量、调用计数、停止自洽、诊断精度 | ✔ 6/6：`prefill_logits_finite=True`、`cache_structure_ok=True`（`n_checks>0`）、`round_invariants.all_ok=True`、`accounting.matches=True`、`checks.passed=True`（`failures=[]`、必需条目全部 executed&ok）、`diag_precision` 干净且 stable |
| 6 | 按各模式 cache 合同验收；Spec target/draft 同步、EOS 后无多余提交/后续轮次 | ✔ 见下（含 AR 与 Spec 的语义差异说明） |
| 7 | prefill 实际融合、非 prefill 不融合 | ✔ AR `fused_by_phase={'prefill':28}`、prod/serial `{'prefill':52}`（`s_max=30`）；**非 prefill 融合=无**；`eager_reason_kinds` 只有 `phase_not_prefill` |

**判据 6 的细节（含一处**语义差异**，如实说明）**

| 模式 | 末轮 / 最终 cache | 与冻结合同的关系 |
| --- | --- | --- |
| AR（target-only） | `target_cache_len=39`、`prompt_len=30`、`gen_len=10`、`decode_steps=9`；`round_invariants` 9 行全 `ok`（`cache_len == prompt_len + decode_runs`） | 符合 harness 的 AR 不变量 **及** 冻结条文 L42（"被输出/接受的 token 为 EOS → **输出后终止**（记录生成长度）"）：`g5_8k_smoke_cudnn.py:568-569` 在记录 EOS 后**立即 break、不把该 token 喂入 cache** ⇒ 39 = 30 + 9 |
| production / serial | 末轮 `Kp=2, m=2, pre=38 ⇒ tgt_len=drf_len=40`；`emitted=[30, 151645]`；`n_rounds=3 = len(rounds)`；`rollback=repair=False` | 符合 §5/§6 与 L43（"**EOS 候选本身照常写入 draft cache**"）：EOS 被接受 ⇒ `tgt_blk` 含 EOS ⇒ tgt=drf=40 = 30 + 10；**EOS 之后无后续轮次、无多余提交** |

⇒ **两模式各自符合各自的合同**，判据 3 比较的是 **token IDs**（完全一致）；
"AR 不消费 EOS token、Spec 提交 EOS token" 是**既有实现的既有语义**（AR 循环在 EOS 处先 break，从未被 EOS 停止触发过 ⇒
本组是**首次**出现 AR 的 EOS 停止记录），**不是**本轮引入，也**未**改动。是否需要在冻结文档中显式登记该差异见 §7 D-14。

**统计自洽**：AR `decode_steps=9 / rounds=10`；Spec 三模式均 `rounds=3, proposed=10, accepted=10,
target_block_calls=3, draft_calls=10, mismatch_rounds=0, all_accept_rounds=3, rollbacks=0`
⇒ 三轮**全部全接受**（`draft_calls == proposed`、`accepted == proposed`）。

## 5. production 与 AR 的分歧（只记录）

* 本组 **未出现分歧**：`production vs AR` 在 rep1、rep2 均为 **EQUAL**（首分歧无、共同前缀=全长 10、两侧长度 10/10）。
* 仍按协议口径**只作记录**：不判控制逻辑失败、不自动归因浮点误差、**不产生**任何加速/质量结论。
* 对照（历史，未改动）：`long` 用例 production 在 idx=13 与 AR 分歧（前轮已记录）。

## 6. 验收器与治理

* 最终验收（`b6_branch_coverage_cpu`，声明含 `set_requirements.branches=["eos_stop"]`）：
  **`[eos1] CASE_PASS`、分支 `{'eos_stop': 'COVERED'}`、集合级要求 `True`、`FINAL_CHECK_RC=0`**。
* 批次：`runs=6 nonzero=0 stopped=False stop_rc=0 final_check_rc=0`（失败即停编排未触发，因为无硬失败）。
* 覆盖记录（只读）：prefill 融合**已实际发生**；Spec 三模式的 `repair` 有限性检查因 `mismatch_rounds=0`
  未执行 ⇒ 记为"非必需且未执行"的覆盖记录（**不**判失败）。
* 治理回归：`verify_family_independence_cpu` **34/0**（含本轮新增的 scoped 检查：该 tag 的 run **未消费任何 manifest family**）；
  `stage2b_preflight_cpu` **rc=0（PREFLIGHT_OK）**，预检打印了 EOS 轮授权（`families=[]`，仅 literal）；
  全量 CPU 自测 **25/25 模块通过**。
* 声明文件：`docs/b6_eos_recheck_declared_2026-10-05.json`（prompt 由 CPU 编码记录**逐字派生**并自校验 sha1）。

## 7. 是否足以补齐既有 B6 的 EOS 缺口

**已取得的证据（本轮新增）**：真实模型 + 真实 tokenizer 的 **EOS 停止**，6/6 run、三模式两重复、
`stop_reason=eos`、EOS 恰一次且位于新生成序列末位、`serial==AR`（逐 rep 严格相等）、跨重复确定性、
全 finite、cache 合同、融合 prefill —— 这正是 `eos-prone` 6/6 `max_new_reached` 所**缺失**的那类证据。

**仍需你裁决的一点（不自动宣布）**：B6 的 EOS **prompt 类型**证据，是否可由**这一条新增的、单独登记的用例**
（tag `div8209_eos`、`max_new=24`、独立声明与验收）来承担？
* **建议（D-13）**：登记为 B6 EOS 类型的**补充用例证据**（因为它是在**运行前**预登记、有独立声明与验收、
  且不改写 `eos-prone` 的历史判决）；`eos-prone` 的 `CASE_INCOMPLETE` **保留不动**。
* 替代方案：把 `eos_stop` 登记为"控制逻辑由 CPU 矩阵覆盖 + 真实模型由本用例覆盖"的双层证据结构（同样需登记，不得静默）。
* 若采纳建议，`B6_BF16_PASS` 的**证据面**即齐备（10 个 B6 用例 9 PASS / 1 历史 INCOMPLETE + 本新增 EOS 用例 PASS 且 `eos_stop` COVERED）。
  **但"是否据此判 `B6_BF16_PASS`"仍属你的统一裁决** —— 本轮**不宣布** `B6_BF16_PASS`，更**不宣布** `BF16_FORMAL_READY`。

## 8. 正式准入仍缺的协议/测量项（集中列出，未扩大测试范围）

| 编号 | 项 | 状态 |
| --- | --- | --- |
| **D-13** | 新增 EOS 用例是否承担 B6 的 EOS 类型证据（§7 建议：登记为补充用例） | **待裁决** |
| **D-14** | AR 不消费 EOS token / Spec 提交 EOS token 的语义差异是否写入冻结文档（或要求对称化=生产改动） | **待裁决** |
| D-5 | 质量评分（rubric） | 待批（`BF16_FORMAL_READY` 需要） |
| D-6 | FP8 自有 Gate A | 待批（本轮无 FP8 运行） |
| D-7 | 判决词表统一 | 待批 |
| D-9 | 不再加额外 FP32 干预 | 待批 |
| D-11 | `design_freeze:83` 的 prompt 类型覆盖口径 | 待批 |
| D-12 | "B6 未过 ⇒ 禁止 acceptance/pilot/final" 是 per-KV 还是全局 | 待批 |
| P-1 / P-2 | 决策行**双侧**捕获（并列只能在证据不足下记 TIE-C） | 未实现 ⇒ 所有并列仍只能记 TIE-C |
| C | QK 两个诊断的**互斥**是否显式化（改生产文件加报错，或登记为规定行为） | 待裁决（见 `docs/b6_eos_qk_closeout_2026-10-05.md` §2.5） |
| E | 把两处**验收规则澄清**（B6 验收器 `uncovered` 语义、QK 读取点**角色**定义）登记进设计/冻结文档 | 可选（不改判据） |

**未做且不予声称**：`BF16_FORMAL_READY`、`FP8_FORMAL_READY`、任何加速/质量结论、任何对历史产物的改写。
