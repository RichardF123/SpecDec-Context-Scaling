# 协议 v2 的证据清点与决定清单（2026-10-04）

* 配套文件：`docs/specdec_correctness_protocol_v2_2026-10-04.md`（v2 草案，**未冻结**）
* 本文件回答四件事：**①** 条款审计（谁要求逐 token 一致、谁只要求算法/结构正确）；
  **②** 已有证据 / 仍需验证 / 可能影响 proposal 的事项；**③** 下一轮最小验证矩阵与命令；
  **④** 面向导师的说明草稿 + 最终建议。
* **本轮不运行 GPU**、不启动 B6/pilot/正式 factorial、**不改写任何历史结论**。

---

## 1. 条款审计：既有承诺里"逐 token 一致"到底出现在哪

> 摘录方式：先由一次**只读**审计通读 15 份文档逐条摘录（未修改任何文件），
> 再对**最关键**的引文由我逐行**复核原文**（`design_freeze_g4` §16/83/96/103、
> `g5_measurement_protocol` §159-164/268/279-281、`production_correctness_decision` §11/19/54-60/103/113、
> `plan_g4_next_steps` §27-28、`proposal_freeze` §31）。下表引文均为 **verbatim**。

### 1.1 **要求** production SpecDec 与 AR 逐 token 一致的条款

| 文件:行 | 引文（verbatim） | 它 gate 什么 |
| --- | --- | --- |
| `design_freeze_g4_specdec.md:16` | `| 判定 | SpecDec ids == target-only greedy ids（exact equality；不只比较解码文本）|` | §1 冻结参数的总判定（B6 判据来源） |
| `design_freeze_g4_specdec.md:83` | `- **B6**：correctness gate——prompts：…；条件 base_bf16 与 v_fp8；判据 exact ids equality + 全 finite + 多次确定性一致。**B6 未过禁止 acceptance 测量/pilot/final。**` | **直接 gate** acceptance 测量 / pilot / final；且条件**含 `v_fp8`** |
| `design_freeze_g4_specdec.md:96` | `⇒ **B6 exact-token-id 门不可取消，且为 B6 保留风险**；不得写成"B6 已无风险"。` | **明文禁止取消** exact-id 门 |
| `g5_measurement_protocol.md:268` | `- **主 Gate**：FP32 `B6` 逐 token 精确相等（当前 PASS：20/20 组合，216/348）；` | G5 主 Gate（**FP32 口径**） |
| `g5_measurement_protocol.md:327` | `7. 正确性状态未因追求速度而降低 —— ✅（§8；FP32 B6 PASS 保持）` | Pilot 进入条件 |
| `plan_g4_next_steps.md:27` | `**B6-fp32: PASS** —— 10 case × 2 KV 档 = **20/20 组合逐 token id 全等**（CPU/float32，参照 = 同引擎 cached AR；总 216/348；…）。**= 实现正确性证据**。` | Phase 3 门；**"实现正确性证据"的原始定位** |
| `plan_g4_next_steps.md:28` | `**B6-bf16: CHECK 19/20** —— 唯一分歧 natural-2 @7，经 g4_b6_tie_diag.py 归因为 **bf16 数值不可判定位置**（signal 0.1491 ≤ noise 0.1491；fp32 真值 = spec 侧 token）。**用户裁决 A = 严格口径：不设例外、不放宽门**，该记录§4.5.5 原样保留。` | BF16 侧历史状态（**19/20**，不是全盘失败） |
| `proposal_freeze_2026-10-01.md:31` | `- 严格 AR/Spec 输出一致性按各 KV 表示分别验证，不放宽门槛；FP8 与 BF16 之间不要求逐 token 相同。` | **冻结提交稿**中的硬条件（按档验证、不放宽；且**跨档之间不要求**相同） |
| `specdec_frozen_submission_proposal_3members.md:43` | `speculative-decoding correctness validation including cache synchronization, verification alignment, and output equivalence` | **已提交 proposal** 中的分工承诺（只写 `output equivalence`，未展开为 id 级） |
| `production_correctness_decision_2026-09-29.md:19` | `| design_freeze_g4_specdec.md §1、§7 | greedy exact IDs、finite、确定性；B6 未过不进后续测量 | 是严格实现/生成约束，不能被诊断标签悄悄覆盖 |` | 审查结论：该约束**不可被诊断标签覆盖** |
| `correctness_protocol_post_shape_localization_2026-10-03.md:55` | `1. 完整 generated token IDs 与 **target-only AR** 一致（逐 token）；` | v1 Gate A 第 1 条（本修订**保留**的那条） |
| `verification_readiness_2026-09-29.md:254,267` | `若航线未获准，则按现行冻结判据处理（B6 exact-id 门）。` / `…**保持原样不放宽**` | 航线性开关缺席时的默认口径 |

**唯一的含糊处（明示）**：`proposal_freeze:31` 与 `specdec_frozen_submission_proposal_3members:43` 都**没有**逐字写"ids 相等"，
其含义依赖对既有判据（`design_freeze_g4` §1）的引用。**这一点必须由 D-1/D-8 澄清**。

### 1.2 **只要求算法/结构正确**、不要求与 AR token 相等的条款

| 文件:行 | 引文（verbatim） | 要点 |
| --- | --- | --- |
| `correctness_protocol_post_shape_localization_2026-10-03.md:69-70` | `四个 family 的 production 运行必须**全部**满足：finite；cache consistency；round invariants；**两次独立重复一致**；合法停止原因；生成长度符合协议（64）；无结构性失败。` | Gate B 主体（**无** token equality） |
| 同上 `:75` | `AR/Spec token 不同**本身**分类为 **NUMERICAL_DIVERGENCE_RECORDED**—— **不得**自动写成 PASS、FAIL、harmless 或 bug；` | 分歧**不是**失败分类 |
| 同上 `:31,:64` | `记录 AR/Spec 的 token 分歧（写入日志），但不得自动解释为算法错误或"无害误差"` / `**不要求** production mode 的 logits 与 AR 逐位一致。` | 生产模式允许分歧 |
| `design_freeze_g4_specdec.md:58,63,68,75` | `**不变式：cache 长度 ≡ 该模型已处理的"真实" token 数**；每轮结束时 target 长度 == draft 长度 == L + (m+1 if m<k' else k')` / `真实 token 永不被丢…rollback = 回到真实段` / `越界…抛 ValueError（不静默钳制）` / `try/finally 清理：异常路径 del 模型 + torch.cuda.empty_cache()` | cache/回滚/OOM 结构不变式 |
| `g5_measurement_protocol.md:159-164` | speedup 声明资格 10 条：`…3. 两侧 context_actual / target_kv / max_new_tokens / **input_token_hash** 一致；…8. 生成正确性状态已记录（tokens_finite=true）；…10. 行内未混入 profiling 数据。` | **关键**：资格里**不含** `ar_equal` ⇒ **"不同 token 也可测速度"在冻结测量协议中本来就成立** |
| `production_correctness_decision_2026-09-29.md:54` | `**保留的硬要求**：算法结构、缓存与索引语义、因果性、检查覆盖、finite、有效生成长度以及同配置重复性。已有 FP32 B6 证据保留；代码或语义变更后需回归。` | "路线 B"的硬要求清单 |
| 同上 `:56` | `**不再隐含承诺**：生产 BF16 轨迹与某条 AR 数值实现处处完全一致。对未达到 exact 的条件，明确报告共同前缀、分歧比例和限制，不改写旧 exact CHECK/FAIL。` | 与 v2 Gate B **同向**（该文状态：**待小组批准**） |
| `verification_readiness_2026-09-29.md:166,250` | `**硬失败** | OOM / 原生崩溃 / 护栏拒绝 / 环境不合格；权重逐位校验失败；**NaN 或 Inf**；必需检查缺失…` / `6. **同配置两次独立重复输出不同** ⇒ 标 **待查** 并在推进前停止；` | 硬停清单来源 |
| `correctness_protocol_validation_2026-10-03.md:155-156` | `**ACCEPTABLE_FOR_MEASUREMENT 的含义**：只表示"可以进行性能测量"，**不**表示 production 的 token 与 AR 完全相同。` | 生产层从来不是 exact-id 认证对象 |

### 1.3 质量（PPL/数据集/指标）与"两条不同轨迹"的比较

| 文件:行 | 引文（verbatim） | 要点 |
| --- | --- | --- |
| `specdec_frozen_submission_proposal_3members.md:27` | `Perplexity is used to assess quality degradation from FP8 KV storage, not speculative-decoding performance.` | PPL 的用途边界（**不是** spec-dec 性能指标） |
| `plan_v7_1_protocol_freeze.md:88` | `dataset=Wikitext-2 test；窗口=非重叠固定上下文窗口；每 condition ≥32 窗口；FP16 同窗口 reference；报告 absolute PPL + relative ΔPPL%；保留 δ 仅当正式 FP8 前预注册` | **冻结的**质量口径（数据集/窗口数/指标/δ 规则） |
| `plan_v7.md:65-66` | `**Quantization quality**：ΔPPL（主）…` / `PPL 是 quantization quality metric，不是 spec-dec effectiveness metric。` | 同上 |
| `production_correctness_decision_2026-09-29.md:103` | `标准AR PPL不能直接当作SpecDec生成质量。在需要声明任务质量保持时，还需相应独立任务评估，不能用PPL代替。` | **两条轨迹比较的核心限制** |
| 同上 `:58` | `NLL 很接近不能单独证明生成语义等价，也不能将近贴边分类等同于无损证明。` | 禁止用 near-tie 代替总体结论 |
| 同上 `:60` | `若输出不同，不得称为"完全相同输出下的无损加速"。吞吐需固定生成长度与停止策略并记录实际数目；不同轨迹可能改变接受率及工作量，必须视为解释限制。` | **GC1/GC4 的直接依据** |
| `proposal_new_acceptance_criteria_2026-09-26.md:57,63` | `行为：**必须**记录并**声明** argmax 一致率与生成 id 是否相同 | **不作通过条件**。` / `**风险**：明显弱于"逐位一致"；**不能**用来支持任何"两条路径等价"的表述。` | 一份**未生效**提议（可作先例，但**不可**当作已批准口径） |

### 1.4 计时：官方模式与"诊断路径不得计时"

| 文件:行 | 引文（verbatim） | 要点 |
| --- | --- | --- |
| `correctness_protocol_post_shape_localization_2026-10-03.md:26` | `| 用途 | 后续**正式性能实验**的唯一模式 |`（同行：verify 形状 `**原始 batched block verification（M=4）**`） | **明文指定 production 为官方计时模式** |
| 同上 `:16,:41,:139` | `⇒ **row-aligned 只能作为 correctness oracle，不得进入任何正式性能/吞吐/speedup 结论。**` / `| 性能 | **不进入**任何正式速度/吞吐/speedup 声明 |` / `diagnostic 模式的时间不具正式意义` | 诊断路径**排除**在计时之外 |
| `g5_measurement_protocol.md:279-281` | `正式计时：无任何插桩；` / `成本分解（g5_cost_breakdown）用外部 CUDA Event 包装…其数字 **不得写入正式主表**` | 计时纪律 |
| `production_correctness_decision_2026-09-29.md:113` | `所有新同步、重算或fallback成本进入最终计时。负加速可以保留为结果。` | 若改路径，其成本必须计入计时 |

### 1.5 修改这组条款会碰到"冻结/已提交"的哪些文件

| 分类 | 文件 |
| --- | --- |
| **冻结 / 已提交（改=改承诺）** | `proposal_freeze_2026-10-01.md:31`（提交稿）· `specdec_frozen_submission_proposal_3members.md:43`（已提交）· `g5_measurement_protocol.md`（标题即"冻结 v1"）· `plan_v7_1_protocol_freeze.md:104` · `plan_v7.md:48,66` |
| **状态含糊，需先澄清** | `design_freeze_g4_specdec.md`：自称 `DRAFT v0.3`，`:103` 批准栏 `- [ ] 我审阅并理解 §1–§9 —— 批准 v0.2 为 v1.0（冻结）` **未勾选**，却被 `handoff_to_analysis_agent.md:153` 与多份日报当作**权威/冻结判据**引用 ⇒ **D-8** |
| **内部工作文档（改动会改写已落盘结论，不得覆盖）** | `correctness_protocol_post_shape_localization_2026-10-03.md`（v1 预登记）· `correctness_protocol_validation_2026-10-03.md`（`ALGORITHMIC_GATE_FAIL`）· `verification_readiness_2026-09-29.md` · `handoff_to_analysis_agent.md` · `results_brief_for_analysis_2026-09-26.md` · `plan_g4_next_steps.md` · `daily_report_2026-09-15b_G4_B5B6_EOSfix.md` |
| **未生效提议（可援引为先例）** | `production_correctness_decision_2026-09-29.md`（**待小组批准**，与 v2 同向）· `proposal_new_acceptance_criteria_2026-09-26.md`（未生效） |

---

## 2. 已有证据 / 仍需验证 / 可能影响 proposal 的事项

### 2.1 已有证据（可直接引用，不必重跑）

| 证据 | 数值 | 位置 |
| --- | --- | --- |
| `B6-fp32`（CPU float32 exact-id，主 Gate） | **PASS 20/20 组合、216/348 token**，两次独立复跑一致 | `g5_measurement_protocol.md:268`、`plan_g4_next_steps.md:27` |
| `B6-bf16`（生产 `M=4`，GPU） | **CHECK 19/20**：唯一分歧 `natural-2 @7`，归因 bf16 数值不可判定位（signal ≤ noise），**用户裁决 A = 严格、不放宽**，记录保留 | `plan_g4_next_steps.md:28` |
| **serial oracle × 5 family**（Gate A 新仪器） | **5/5 与 AR 逐 token 相同（64/64）**，两次独立进程一致；每次运行 `s==1`、**0** 次 block 前向、记账恒等 | `serial_oracle_stage1_f13_*`、`serial_oracle_stage2_*` |
| **row-aligned 诊断（v1 的 Gate A 仪器）** | 4 family 中仅 **1/4** 与 AR 一致 ⇒ `ALGORITHMIC_GATE_FAIL`（**不改写**） | `correctness_protocol_validation_2026-10-03.md` |
| **production（BF16）分歧率（上一轮 4 family）** | F01@14、F02@7、F08@52、F10@8（F13@47 另有 Stage 1 记录）；两次重复一致 ⇒ **可重复分歧** | `serial_oracle_stage2_2026-10-03.md` §6 |
| 结构/有限性/确定性（4 family × 3 模式） | 全部通过；`decision_coverage` 覆盖完整 | `correctness_protocol_validation_2026-10-03.md:54-59` |
| 调用几何审计器与聚合器自测 | 审计器 **19/0**；聚合器 **18/0**（含字段名契约与 D6/D7 回归） | `serial_oracle_audit_selftest_*`、`serial_oracle_aggregate_selftest_*` |
| 治理守卫 | **PASS 29/0**（授权范围内 family 使用；`out_of_scope=[]`） | `family_independence_cpu_check.json` |

### 2.2 仍需验证（**尚未有证据**）

1. **Gate A 在 fp8 KV 下**：`design_freeze_g4_specdec.md:83` 明确 B6 的条件含 `v_fp8`，而现有 Gate A 证据**只有 BF16 base KV**（**D-6**）。
2. **Gate A 在更多 family / 更长生成**上：现有 5 个 8192 窗口 × 64 token；`max_new` 更长、`ctx` 2K/4K 未验。
3. **v2 的 GB6 记账端到端演练**：`correctness_protocol_margins.py` 已有字段能力，但**尚未在 tag `div8209_so` 的产物上跑过**（下一轮 M1 会覆盖）。
4. **serial 路径在 fp8 KV 下是否定义良好**（`_fmt`/KV 格式与逐 token 前向的兼容性）——需一次最小 smоke。
5. **差异率的总体规模**：目前"多少条件会分歧"只有 4–5 个 family 的证据；正式报告的分歧率需要更大的既有 family 样本（可复用 16 个 family 的 manifest）。
6. **两条轨迹的质量对比口径**是否要求独立任务评估（`production_correctness_decision:103` 提示 PPL 不足）。

### 2.3 可能影响 proposal 的事项（**只登记，不擅自改**）

| # | 事项 | 影响面 | 处置建议 |
| --- | --- | --- | --- |
| P-1 | 若批准"控制逻辑层 exact-id（serial oracle）+ 生产层记录型 Gate B"，则 `design_freeze_g4_specdec.md` 的 B6 判据需**重新分层表述**（**不取消**该门） | B6 判据文本、`g5_measurement_protocol.md` §8、Pilot 进入条件 | 需 D-1/D-8 决定；**不**自行改已提交稿 |
| P-2 | 研究问题与 6+2 矩阵**不受影响**（仍是 ctx 长度 × AR/Spec 的 decode speedup；FP8 V-only 8K 补充） | — | 无需改 proposal 范围 |
| P-3 | 但"主指标 decode speedup"在轨迹分歧时的**解释边界**需要写进报告（GC4/GC5） | 报告结构、与导师沟通口径 | 采纳 `production_correctness_decision:60` 的既有表述 |
| P-4 | fp8 条件可能**额外**需要 Gate A 运行（D-6） | 工作量 | 需决定 |
| P-5 | 质量评估若要求"两条轨迹都评"，则质量工作量翻倍 | 工作量/报告结构 | 需 D-5 决定 |
| P-6 | `design_freeze_g4_specdec.md` 的**批准状态不明** | 决定 D-1 是"改冻结判据"还是"批准草案" | D-8 |

---

## 3. 下一轮最小验证矩阵与命令

**目的**：把 v2 从"已验证的 5 个 family"变成"**事先登记、可复核**的协议"。**不**测性能、**不**启动正式实验。

* 预登记 family：**F09(S1) / F03(S2) / F11(S3) / F04(S4)** —— 每 stratum 取**未用过的最小 id**（规则同 v1，排除 F13/F06）。
* 矩阵：每 family **AR×2 → production×2 → serial×2**，组内顺序固定、阶段间停手核对。
* **总运行量 24 次**（4 family × 3 mode × 2 rep）；按本机实测约 **15–20 分钟**。
  缩减版（仅 Gate A）16 次 ≈ 10 分钟，但会跳过 GB6/GC3 演练，**不推荐**。
* 前置（缺一不可）：① **D-3 授权**（守卫当前会把 F09/F03/F11/F04 判为越界，这是设计）；
  ② 运行前后各跑一次守卫并 PASS；③ 环境门合格（harness 16 GiB 线 + 本轮入口门槛）。

```powershell
# 0) 前置：治理守卫 + 环境预检（不加载模型）
.\.venv\Scripts\python.exe -X utf8 -m specdec.engine.verify_family_independence_cpu
.\.venv\Scripts\python.exe -X utf8 -m specdec.engine.serial_oracle_env_check

# 1) 逐 family：AR×2 -> production×2 -> serial×2（每步之后先核对再继续）
foreach ($f in 'F09','F03','F11','F04') {
  & .\scripts\so_batch.ps1 -Tag div8209_p2 -Cases $f -Modes ar     -Reps 1,2
  .\.venv\Scripts\python.exe -X utf8 output\g5\attn\_so_stage_check.py $f ar
  & .\scripts\so_batch.ps1 -Tag div8209_p2 -Cases $f -Modes prod   -Reps 1,2
  .\.venv\Scripts\python.exe -X utf8 output\g5\attn\_so_stage_check.py $f prod
  & .\scripts\so_batch.ps1 -Tag div8209_p2 -Cases $f -Modes serial -Reps 1,2
  .\.venv\Scripts\python.exe -X utf8 output\g5\attn\_so_stage_check.py $f serial
}

# 2) 聚合（只读）+ 自测 + 守卫复查
.\.venv\Scripts\python.exe -X utf8 -m specdec.engine.serial_oracle_aggregate --scope final
.\.venv\Scripts\python.exe -X utf8 -m specdec.engine.serial_oracle_aggregate_selftest_cpu
.\.venv\Scripts\python.exe -X utf8 -m specdec.engine.verify_family_independence_cpu
```

> 新 tag `div8209_p2` **不覆盖**任何历史产物；family 输入由 `family_input` 重建并校验 token sha，校验失败即拒绝、不降级。
> 若 D-6 决定"fp8 也要 Gate A"，则再加一组：`-Tag div8209_p2fp8` + `SMOKE_KV=v_fp8`（**先**做一次 fp8 serial 最小 smoke 确认路径定义良好）。

---

## 4. 面向导师的简短说明草稿（可直接改用）

> 老师您好，关于投机解码的正确性验证，我们想把一个概念上的区分讲清楚，并请示一个口径问题。
>
> **现象**：在我们的实现里，**AR（逐 token 解码）** 与 **投机解码的"验证"步骤** 走的不是同一种 GPU 计算形状：
> AR 每步算 1 行，验证一次算 4 行（候选块）。在 **CPU/float32** 下，两者输出的 token id **完全一致**
> （历史记录：10 个用例 × 2 种 KV 档 = **20/20 组合、216/348 token 全等**）。但在 **GPU/BF16** 下，
> 同样的比较会出现**极少数位置**的分歧（历史记录：**19/20**，唯一分歧点经诊断落在 bf16 的
> "数值不可判定位"——信号强度与噪声同量级）。我们已在五类算子（QK、o_proj、gate/up/down_proj）上
> 复现该机制，结论是**执行形状相关的有限精度数值非不变性**（不是 cuBLAS/PyTorch 的 bug）。
>
> **我们做了什么来把它查清**：我们新增了一个**真正的"控制逻辑"探针**——把验证步骤换成与 AR **完全相同**的
> 逐 token 计算（其余接受/替换/回滚/缓存逻辑一字不改），并用调用几何审计**硬性证明**该探针下不存在任何
> 4 行前向。结果：在 5 个 prompt family 上，该探针与 AR 的输出**逐 token 完全相同（64/64）**，两次独立进程可重复。
>
> **因此我们想区分两件事**（并请示是否可以用这个口径写报告）：
> **(1) 控制逻辑是否正确**——用上述探针判定：正确（并有 FP32 exact-id 的独立证据互相印证）；
> **(2) BF16 块计算的逐 token 一致性**——这是**数值行为**问题，不是算法逻辑问题；生产路径仍会在个别位置与 AR 分歧，
> 我们会**如实记录**（首个分歧位置、两侧 token、决策 margin、共同前缀），**不称其为"无害"、也不称其为"错误"**。
>
> **这对结论的影响**：性能比较（decode speedup）**不受阻碍**——因为我们的测量协议对 speedup 的资格条件里
> **本来就不要求** AR 与投机解码的 token 相同（要求的是同一输入、同一 KV/长度配置、有限性、无插桩）。
> 但当两条路径产生不同文本时，接受率、轮数与工作量属于**各自轨迹**，我们会一并报告分歧位置与两侧统计，
> **不会**把它说成"完全相同输出下的无损加速"。
>
> **想请老师确认两点**：① 上述"控制逻辑正确 vs BF16 块计算一致性"的分层是否可以作为本课程报告的正确性口径
> （我们**不取消**逐 token 精确相等的门，只是把它用在**与 AR 同计算路径**的控制逻辑层）；
> ② 是否需要我们为 FP8 KV 档**单独**再做一遍控制逻辑层验证（因为提交稿要求"按各 KV 表示分别验证"）。

---

## 5. 最终建议

### **`NEEDS_DECISION`**

**理由**：协议 v2 在**技术层**已经就绪（Gate A 仪器已验证 5/5、审计器与聚合器自测全过、守卫 PASS、
下一轮矩阵与命令可直接执行），但**冻结它需要你/导师决定 9 项口径**（v2 §9 D-1…D-9），
其中 D-1 与 D-8 触及**已提交稿 / 状态含糊的冻结判据**，属于**承诺层面的决定**，我不擅自代决。

**冻结前真正需要决定的事（按优先级）**：

| 优先级 | 事项 | 一句话 |
| --- | --- | --- |
| **1** | **D-1 + D-8** | 是否批准"控制逻辑层 exact-id（serial oracle）+ 生产层记录型 Gate B"这一兑现方式；以及 `design_freeze_g4_specdec.md` 到底是不是已批准冻结（其批准栏未勾选） |
| **2** | **D-3** | 授权把 F09/F03/F11/F04 登记进守卫授权范围（否则下一轮**跑不了**，守卫会正确判越界） |
| **3** | **D-6** | fp8 条件是否也需要各自的 Gate A 证据（提交稿要求按 KV 分别验证） |
| **4** | **D-4** | 确认预登记分歧阈值（`index==0 ⇒ CHECK_NEEDS_REVIEW`） |
| **5** | **D-5** | 质量控制条清单与是否两条轨迹都评 |
| **6** | **D-2** | Gate A 是否扩到 9 family，还是把结论限定在 5 窗口 |
| **7** | **D-7 / D-9** | 新判决词条；是否补 FP32 口径 |

**若你确认 D-1/D-3/D-8，我可以立即把 v2 标为 FROZEN 并执行 §3 的 24 次最小矩阵**（约 15–20 分钟）；
**在此之前**，v2 保持 DRAFT，且**不**启动任何正式测量、B6、pilot 或 factorial。
