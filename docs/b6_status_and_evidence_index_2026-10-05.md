# B6 状态与证据索引（2026-10-05，基于 `95705c7`）

> 本文件**只整理状态与证据**：不新增 GPU 运行、不重跑 66 次 B6 或 6 次 EOS 补测、不改写任何历史产物或判决。
> 本轮四项批准（用户明确批准）见 §1；其后是六态现状、证据索引与边界。

## 1. 本轮四项批准（逐条登记，作为后续引用的权威口径）

| # | 批准内容 | 落地形式 |
| --- | --- | --- |
| **A1** | **接受 `div8209_eos` 作为 B6 EOS 类型的补充证据**：该用例运行前已登记、6/6 真实模型运行以真实 EOS 结束、`serial==AR`、重复一致与硬检查通过 | tag `div8209_eos`、case `eos1`、声明 `docs/b6_eos_recheck_declared_2026-10-05.json`、结果 `docs/b6_eos_recheck_result_2026-10-05.md`；**`eos-prone` 的 `CASE_INCOMPLETE` 保留**、历史产物不改写 |
| **A2** | **确认有限范围 `B6_BF16_PASS`**：汇总依据 = 既有类型/分支覆盖 + 66 次运行 + CPU EOS 控制逻辑矩阵 + 6 次真实 EOS 补测 | 状态口径见 §2 的**限定措辞**；**不声称**所有输入均正确、**不声称**生产批量验证与 AR 始终逐 token 一致、真实 EOS 补测**仅覆盖全接受停止**、其它 EOS 控制分支明确标为 **CPU fixture 证据** |
| **A3** | **登记 AR 与 Spec 的既有终止 cache 合同差异**：AR 输出 EOS 后不再消费它；Spec 可能已在验证中计算并提交 EOS。分别验收各自合同，**不**要求终止 cache 长度强行对称，**不**改数值路径 | 见 §3；依据 `g5_8k_smoke_cudnn.py:568-569`（AR 记录 EOS 后立即 break）与 `design_freeze_g4_specdec.md:43`（EOS 候选照常写入 draft cache） |
| **A4** | **正式准入采用 per-KV 规则**：BF16 与 V-only FP8 各自通过各自的门后分别放行；FP8 未完成**不得**写成已通过，但**不**因其未完成而阻塞满足其它条件的 BF16 主线 | 与既有 D-6/D-12 的推荐一致（见 `docs/bf16_formal_admission_2026-10-05.md` §4） |

## 2. 六态现状（互不蕴含）

| 状态 | 当前取值 | 依据 / 边界 |
| --- | --- | --- |
| `STAGE2B_PASS` | **成立**（2026-10-05） | `div8209_p2` 24/24 runs、`decision=SERIAL_ORACLE_PASS`（`div8209_p2_stage2b_verdict_20261005T083145Z.json`）；口径限于 **4 个新 family**（F09/F03/F11/F04）× 3 模式 × 2 次、8K 自然语言窗口、新口径（serial oracle exact-id）；5 个历史 family 只作**辅助证据** |
| **`B6_BF16_PASS`** | **成立（有限范围，2026-10-05 批准）** | 见下方限定措辞 |
| `B6_VFP8_PASS` | **未达成** | `v_fp8` 下**没有任何** Gate A/B 证据（现有 serial/EOS/字面覆盖全为 `base_bf16`）；A4 规定其必须**独立**验证 |
| `FULL_B6_PASS` | **未达成** | = `B6_BF16_PASS && B6_VFP8_PASS`；前件成立、后件未达成 |
| `BF16_FORMAL_READY` | **未达成** | 需 `STAGE2B_PASS` ✓ + `B6_BF16_PASS`（有限范围，已批准）+ A4/D-12 的 per-KV 放行规则 ✓ + D-5 质量口径（**部分待决**）+ 报告纪律 ✓ + 环境/预算准入（**待完成**，见准入表）；**不得**仅凭 `B6_BF16_PASS` 宣布 |
| `FP8_FORMAL_READY` | **未达成** | 需其**自己的**门（`B6_VFP8_PASS`）+ 环境/预算准入；**不得**由任何 BF16 状态推出 |

### 2.1 有限范围 `B6_BF16_PASS` 的**限定措辞**（引用时请照抄；不得扩写）

> **`B6_BF16_PASS`（有限范围，2026-10-05）**：在 `base_bf16`、`SO_ARM=cudnn_prefill`（仅 prefill 融合）、
> `k=4`、`chunk=2048`、贪心、冻结 EOS 规则下，
> ① 旧 `:83` 的**提示类型覆盖**（自然语言/数学/代码/短/长/全接受/首位拒绝/中途拒绝/EOS/非 4 倍数）
> 通过 10 个固定字面用例 × 3 模式 × 2 重复 = **60 次真实运行**（`div8209_b6`）实测：
> **9 用例 `CASE_PASS`、1 用例 `CASE_INCOMPLETE`（`eos-prone`：`eos_stop` 未触发）、0 `FAIL`**，
> 集合级分支 5/5、类型 8/8 `COVERED`，且**每个用例** `serial` 与 AR **完整 token IDs 相等**（20/20）；
> ② EOS 分支另由**独立登记的补充用例** `eos1`（`div8209_eos`，6 次真实运行）实测：6/6 以真实 EOS（151645）结束、
> `stop_reason=eos`、`serial==AR`、跨重复一致、硬检查全通过，`eos_stop` = `COVERED`；
> ③ 8K 融合 prefill 路径由 `F09` 6 次真实运行（`--require-fused-prefill` 门通过）证明；
> ④ CPU 侧另有 EOS 控制逻辑矩阵（10 分支 / 29 断言 / 变异体与非空过对照）覆盖**其它** EOS 控制分支。

**明确不声称**（边界，逐条）：
1. **不**声称所有输入都正确（证据是 10 个字面用例 + 1 个补充 EOS 用例 + 4 个 8K family 的新口径证据）；
2. **不**声称生产（batched verify）与 AR 始终逐 token 一致 —— 生产分歧**只记录**（例：`long` 在 idx=13 分歧，两重复一致）；
3. 真实 EOS 补测**只覆盖"全接受后停止"**这一条路径（`Kp=2, m=2`）；**其它** EOS 控制分支
   （拒绝后替换 token 为 EOS、EOS 位于预算边界、回滚/repair 后 EOS、额外 token 兑现为 EOS 等）
   **只有 CPU fixture 证据**（位置策略 stub，非真实模型、非数值证据）；
4. **不**覆盖 `v_fp8`；**不**覆盖 2K/4K 上下文；**不**覆盖 `max_new` 超过 24/64 的停止行为；
5. `eos-prone` 的历史判决 `CASE_INCOMPLETE` **保留不改**。

## 3. A3：AR 与 Spec 的终止 cache 合同差异（登记为既有行为）

| 模式 | EOS 停止后的 cache | 实现依据 | 说明 |
| --- | --- | --- | --- |
| **AR**（target-only） | `target_cache_len = prompt_len + 已喂入步数`（**不含** EOS 本身；本组 39 = 30 + 9） | `g5_8k_smoke_cudnn.py:552-576`：先 `gen_ids.append(nxt)`、记不变量、**遇 EOS 立即 `break`**，不再对 EOS 做 decode `_run` | 冻结合同 L42："被输出/接受的 token 为 EOS → **输出后终止**（记录生成长度）" |
| **production / serial** | `tgt_len == drf_len == pre + (m+1 if m<Kp else Kp)`（**含** EOS；本组 40 = 30 + 10） | `spec_dec.py:235-250` + L43："**EOS 候选本身照常写入 draft cache**" | 全接受 ⇒ `tgt_blk` 含 EOS；错配 ⇒ 在 `pre+m` 处裁剪后 `_step(r=EOS)` |

⇒ **分别验收各自合同**：AR 侧验 `cache_len == prompt_len + decode_runs`（逐行人核对）；Spec 侧验逐轮长度不变式与 EOS 提交；
**不要求**两侧终止 cache 长度对称；**未**改任何数值路径。该差异属**既有实现语义**（AR 循环从未被 EOS 停止触发过，本轮首次显现）。

## 4. 证据索引（可直接引用；第 4 列为判决，均未改写）

| # | 证据 | 结论 / 数值 | 产物（`output/g5/attn/` 等） | 判决 | 状态 |
| --- | --- | --- | --- | --- | --- |
| E1 | `B6-fp32`（CPU/float32，历史主 Gate） | 20/20 组合、216/348 token 逐 token 全等，两次复跑一致 | 见 `g5_measurement_protocol.md:268`、`plan_g4_next_steps.md:27` | PASS（历史） | 保留 |
| E2 | `B6-bf16`（生产 `M=4`，历史） | **CHECK 19/20**：唯一分歧 `natural-2@7`，归因 bf16 不可判定位置；用户裁决严格、不放宽 | 见 `plan_g4_next_steps.md:28`、`g4_b6_tie_diag.py` | CHECK（历史） | 保留 |
| E3 | serial oracle × 5 family（Gate A 历史辅助） | 5/5 与 AR 逐 token 相同（64/64）、两次独立进程一致 | `serial_oracle_stage1_f13_*`、`serial_oracle_stage2_*` | PASS（辅助证据） | 保留（缺新 D6 检查，**不得**与 E4 合并宣称 9-family） |
| E4 | **Stage 2b**（4 新 family，新口径） | 24/24 runs；`serial==AR` 全等；production 在 F03/F11 分歧（记录） | `div8209_p2_stage2b_verdict_20261005T083145Z.json` | **`SERIAL_ORACLE_PASS`** | 保留 |
| E5 | **B6 覆盖轮**（10 字面用例 × 3 模式 × 2 重复） | 60 runs、`rc=0` 全通过；9 `CASE_PASS` / 1 `CASE_INCOMPLETE` / 0 `FAIL`；集合级分支 5/5、类型 8/8；`serial==AR` 20/20；`long` 生产分歧 idx=13 | `div8209_b6_b6_branch_coverage_20261006T105953Z.json`、`div8209_b6_*` 60 份 | 9 PASS + 1 INCOMPLETE | 保留 |
| E6 | **8K 融合 prefill 复核**（F09） | 6 runs；`fused_by_phase={'prefill':112}`(AR)/`{208}`(prod,serial)、`fused_s_max=2048`、`input_tokens=8192` | `div8209_b6_F09_*_rep*.json` | CASE_PASS | 保留 |
| E7 | **CPU EOS 控制逻辑矩阵** | 10 分支 / 29 断言 0 失败；非空过对照 10/10；4 变异体 + 3 检查器自测被抓出 | 工具 `specdec/engine/g4_b5_eos_matrix_cpu.py`（+既有 `g4_b5_eos_branch_tests.py`） | PASS（**仅控制逻辑**） | 本轮新增 |
| E8 | **真实模型 EOS 补测**（`eos1`） | 6/6 run 以 151645 结束、`stop_reason=eos`、`gen_len=10`、`serial==AR`、跨重复一致、26/26 判据 | `div8209_eos_*` 6 份 + `div8209_eos_b6_branch_coverage_20261006T130231Z.json` | CASE_PASS、`eos_stop` COVERED | 本轮新增 |
| E9 | 治理守卫 | **34/0**（含"新 tag 只限 literal、不消费 family"的 scoped 检查） | `family_independence_cpu_check.json` | PASS | 本轮更新 |
| E10 | 环境门 | `ENVIRONMENT_OK`（harness 16 GiB 线通过、入口余量 ≥18 GiB、工作树 clean） | `serial_oracle_env_check_*.json` | PASS | 周期复跑 |
| E11 | 编排（失败即停 / 分组验收） | 模块级 39/39、脚本级 35/35；真实触发过 1 次失败即停并正确停住 | `b6_orchestrate_selftest_cpu`、`scripts/selftest_so_batch_orchestration.ps1` | PASS | 本轮新增 |
| E12 | CPU 自测总览 | **25/25 模块通过**（含 `qk_scope_selftest_cpu` 由 33/3 → 48/0） | 见 `docs/b6_eos_qk_closeout_2026-10-05.md` §2.3 | PASS | 本轮更新 |

## 5. 历史判决保留清单（**未改写**，再次确认）

* `eos-prone` = `CASE_INCOMPLETE`（`eos_stop` UNCOVERED）—— 本轮已只读复核；
* `B6-bf16` = CHECK 19/20、`natural-2@7` 记录与用户"严格不放宽"裁决 —— 保留；
* row-aligned Gate A = `ALGORITHMIC_GATE_FAIL`（4 family 中 1/4）—— 保留，不重新解释；
* `div8209_p2` 的 production 分歧（F03@59、F11@49 及 F09/F04 EQUAL）—— 保留；
* 全部历史 JSONL / 分片 / 汇总 —— 未删除、未覆盖、未重算（除既有 `g5_recanonicalize_run.py` 的"新版本另存"路径外无任何重建）。

## 6. 与正式实验的接口（一句话）

`B6_BF16_PASS`（有限范围）解决的是 `design_freeze_g4_specdec.md:83` 中 **`base_bf16` 条件**的提示类型覆盖与
控制逻辑/生产分歧的记账口径；它**不**解决 `v_fp8` 条件、**不**解决正式测量的 8K 可测性、也**不**构成
`BF16_FORMAL_READY`。正式准入的逐项状态见 `docs/bf16_formal_admission_2026-10-05.md`。
