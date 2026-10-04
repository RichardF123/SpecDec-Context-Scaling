# Stage 2b 待批准决策单（2026-10-04）

> **本单是"请求批准"，不是批准结果。** 工具自测通过 = 工具证据，**不等于**协议批准。
> 提交本单时：**未运行任何 GPU 步骤**、未改任何冻结判据、未覆盖任何历史 FAIL/CHECK/PASS。
> 逐项推荐的完整论证见 `docs/protocol_decision_brief_2026-10-04.md`；本单只给"决策 + 影响 + 风险"。

## 0. 当前准入状态（本轮只读实测）

| 检查 | 实际命令 | 结果 |
| --- | --- | --- |
| 预检 | `python -X utf8 -m specdec.engine.stage2b_preflight_cpu` | **exit 3** → `PREFLIGHT_OK_TOOLS_BUT_FAMILIES_UNREGISTERED`；F09/F03/F11/F04 四项均 `会被判越界=True` |
| 治理守卫 | `python -X utf8 -m specdec.engine.verify_family_independence_cpu` | **31/0**；其中两条相关检查为 `Stage 2b 工具的源码里只有'已授权 ∪ 候选'的 family 字面量`、`Stage 2b 候选 family **尚未被任何 run 消费**（D-3 批准前不得跑）` |
| 环境门 | `python -X utf8 -m specdec.engine.serial_oracle_env_check` | **ENVIRONMENT_OK**；commit 可用 20.3 GiB ≥ 安全线 16 GiB；入口 free 20.31 GiB ≥ 18 GiB（margin +2.31） |
| 工具自测（**仅工具证据**） | 7 个 CPU 自测 | 合计 **254/0**（verdict 92 · diag 25 · agg 21 · guard 31 · audit 19 · env 24 · stage0 42） |
| 历史产物只读复核 | AR vs production 首分歧 | 输出下标 **47**（旧实现误取末位 63）；运行侧 GB6 字段全齐，仍缺 5 个诊断字段 |

**结论：GPU 阶段被"协议未批准"硬阻断；工具侧不再是阻断项。**

## 1. 逐项决策

### D-1 exact-token 门分层到 serial oracle —— **硬阻断**

| 项 | 内容 |
| --- | --- |
| **旧判据** | `design_freeze_g4_specdec.md:16`：判定 = `SpecDec ids == target-only greedy ids`（exact equality）；`:83`：**B6** correctness gate 条件写 `base_bf16` **与** `v_fp8`，判据 exact ids equality + 全 finite + 多次确定性一致，**B6 未过禁止 acceptance 测量/pilot/final**；`:96`：**exact-token-id 门不可取消** |
| **新判据（待批）** | Gate A = 在**与 AR 同路径**的 serial oracle（逐 token `M=1`）上要求完整 token IDs 逐 token 相等；**production（`M=4` 块验证）不设 exact-id 门**，其 token 分歧按"数值行为"**记录并分层解释** |
| **改变了什么** | 门槛数值没放松，但**被检验的对象变了**：从"生产路径是否与 AR 相同"→"**控制逻辑**在与 AR 同路径时是否与 AR 相同"。生产层由"门"变成"观察对象" |
| **风险** | ① 与 `:83` 字面语义（指向 production）冲突 ⇒ 若不同时解决 **D-8**，D-1 会被读成"改冻结判据"；② `:96` 明确保留风险，`:267` 记录的 **F06/index25 三方精确并列（margin 0.0）** 说明"逐 token 精确相等"在**数值并列**位置不具语义唯一性 ⇒ 新仪器也会遇到并列，必须预登记并列处置规则（见 D-4）；③ 声明边界：不同轨迹 ⇒ **不得**称"完全相同输出下的无损加速"（`production_correctness_decision:60`） |
| **建议** | **有条件批准**（三条件写进纪要）：① 明示"执行路径实质调整"；② 生产分歧一律作为"发散轨迹"记录，禁止无损表述；③ 并列处置规则**先预登记**再跑 |

### D-2 Gate A 覆盖范围：5 窗口 → **9 family** —— **硬阻断（决定是否允许用新 family 作证据）**

| 项 | 内容 |
| --- | --- |
| **旧判据** | Gate A 证据只来自已用 5 个 family（F13/F01/F02/F08/F10） |
| **新判据（待批）** | 扩到 9：新增 `F09/F03/F11/F04`（四 stratum 各一） |
| **改变了什么** | 结论从"5 窗口上的控制逻辑正确"扩为"9 窗口"，并由 preflight 复现其来源 |
| **风险** | 新增样本 ⇒ 更多 GPU 时间与更多可能失败点；但**不是事后挑选**：四个 family 取自 manifest 预登记，已核验 `token_sha1` 复现（`hash_verified=True`）、窗口两两不重叠、**32-gram 零交集**（含与已用 5 family 及调试输入）、现有产物**零消费** |
| **建议** | **批准**（五窗口不足以支撑系统级陈述） |

### D-3 登记四个新 family 进**run 级**授权 —— **硬阻断（不批准则跑不了）**

| 项 | 内容 |
| --- | --- |
| **旧判据** | 守卫的 run 级授权 = fam16 16 格 ∪ F06/index25 定点取证 ∪ 串行 oracle 轮（`F13` + `F01/F02/F08/F10`，tag `div8209_so`）。`STAGE2B_CANDIDATE_FAMILIES` **只让工具源码通过源级检查，不构成 run 级授权** |
| **新判据（待批）** | 新增 Stage 2b run 级授权 = **仅** `tag=div8209_p2` + `tool=serial_oracle_gpu` + `families=F09/F03/F11/F04` |
| **改变了什么** | 授权集合变宽 ⇒ 这四个 family 的 run 不再被判越界。代码上：新增 `STAGE2B_ROUND` 常量、并入授权集合，并把"候选尚未被消费"检查**替换**为 scoped 检查（只允许该 tag/tool 消费） |
| **风险** | 放宽授权会降低守卫对"越界使用"的敏感度；若只删检查不补 scoped 检查，会失去"这四个 family 被其它工具/tag 复用"的检出能力 |
| **建议** | **批准**，并要求：① 授权**绑定 tag+tool**；② scoped 检查随批准同批落地（只增检查）；③ 其它 12 family 仍然判越界 |

### D-4 分歧记录规则（删除 `index==0` 自动阈值）—— **硬阻断（预登记要求）**

| 项 | 内容 |
| --- | --- |
| **旧判据** | v2 草案初稿曾写：`首个分歧 index == 0 ⇒ CHECK_NEEDS_REVIEW`（自动阈值） |
| **新判据（待批）** | **DR1** 每处分歧完整记录（首位置、共同前缀 P、两侧 token、决策位置 top-1/top-2 与 margin、两轨迹接受率/轮数/target 调用数、两侧完整输出与实际长度 = GB6）；**DR2** 速度比一律标注"**分歧轨迹上的比较**"并同时报告 P 与两侧统计；**DR3** 全量保留、分层解释、**不按分歧位置筛样本**；**DR4（可选）** 若仍要数值阈值，只能在看到下一轮结果**之前**预登记并写明依据 |
| **改变了什么** | 取消自动准入判定，改为"全记录 + 预登记解释" |
| **风险** | ① 阈值删除使"是否准入"不再自动 ⇒ 需人工分层解释，必须**事先**写清必须报告哪些量；② 我的原稿曾以"`P < k` ⇒ 首个验证轮内分歧"为据（**该等价关系不成立**：`spec_dec.py:237` 的 rejection 轮只输出 `m+1` 个 token，共同前缀是各轮累加）⇒ 说明这类"看似显然"的判据必须逐条对代码核验 |
| **建议** | **采用 DR1–DR3**；**不采用** R4（无可辩护的依据，且诱导事后选择） |

### D-7 判决词条与"失败分类" —— **不阻断运行，阻断结论表述**

* **旧判据**：只有"算法正确/失败"一类，production 结构失败与算法失败会被混为一谈。
* **新判据（待批）**：保留 `NUMERICAL_DIVERGENCE_RECORDED`（分歧已记录）与 `PRODUCTION_RUNTIME_FAIL`（结构/运行时失败）等词条，把"分歧"与"失败"分开。
* **现状**：工具已实现 `SERIAL_ORACLE_PASS / SERIAL_ORACLE_FAIL / BASELINE_REGRESSION / CHECK_NEEDS_REVIEW` + `production_divergence_class`（`TOKEN_EQUAL` / `NUMERICAL_DIVERGENCE_RECORDED`），**但词条口径仍待小组确认**。
* **建议**：**批准保留**（可改名以降低沟通成本）。

### D-8 `design_freeze_g4_specdec.md` 的效力状态 —— **硬阻断（D-1 的落点问题）**

| 项 | 内容 |
| --- | --- |
| **事实** | 该文件自标 **DRAFT v0.3**，`:103` 批准栏**未勾选**，却被多处当"权威/冻结判据"引用 |
| **新判据（待批）** | 先裁决：它是**已冻结判据**（则 D-1 = **修改冻结判据**，须走正式变更）还是**待批准草案**（则 D-1 = **正式批准该草案**）。`proposal_freeze_2026-10-01.md:31` 已被新 `final` proposal 取代，亦须在纪要中明确"以 final 为准" |
| **风险** | 不裁决 ⇒ D-1 没有合法落点；两种解读下的批准动作与留痕要求不同 |
| **建议** | **与 D-1 同批裁决**，并在纪要写明结论 |

### D-5 质量口径 —— **不阻断 Stage 2b 运行，阻断"质量等同/无损"类声明**

* **旧判据**：WikiText-2 test、非重叠窗口、absolute PPL + relative ΔPPL；`production_correctness_decision:103` 已明确"标准 AR PPL 不能直接当作 SpecDec 生成质量"。
* **新判据（待批）**：沿用冻结质量口径 + 明确**不得**用于声称"分歧后生成质量等同"；是否**两条轨迹都评**由小组定。
* **建议**：**沿用 + 明确禁用范围**；"两条轨迹都评"请小组裁决（本阶段不受阻）。

### D-6 `v_fp8` 是否各自做 Gate A —— **阻断 FP8 档（不阻断 BF16 Stage 2b）**

* **旧判据**：`:83` 把条件写成 `base_bf16` **与** `v_fp8`。
* **新判据（待批）**：两个 KV 条件**各自**做 Gate A；现有 serial 证据**只有 BF16**。
* **建议**：**批准"各自验证"**；本阶段不跑 FP8，**禁止**把 BF16 结论外推到 FP8。

### D-9 是否再补 FP32 口径 —— **不阻断**

* 现状：`B6-fp32` 已 PASS（20/20 组合、216/348 token），与 BF16-serial 互为印证（`g5_measurement_protocol:268`）。**建议不重复**（除非代码/语义变更 ⇒ 需回归）。

## 2. 运行准入条件（**不是**协议批准，但缺一不可；需你同意执行顺序）

| # | 条件 | 说明 |
| --- | --- | --- |
| **E-1** | **诊断工具 GPU 分支未验证** | `gb6_diag_select` 的 CPU 配对/来源/配置核对已验证（25/0 + 真实产物 47），但 `correctness_protocol_margins` 的 GPU 重建分支（prefill + 逐 token + top-k/margin/cache）**从未在 GPU 上跑过** ⇒ 建议**第一个 GPU 动作**就是**单 family 端到端核验**（F09：AR×2 + prod×2 → 若有分歧则跑诊断），核对 6 项一致性，**失败即停** |
| **E-2** | 环境与运行纪律 | `env_check` 必须 `ENVIRONMENT_OK` 且入口 free ≥ 18 GiB；**一个 run 一个独立进程**（`scripts/so_batch.ps1 -Tag … -Cases … -Modes … -Reps …`）；**不自动重试、不挑选样本**；全部原始产物保留；禁止 `SMOKE_ALLOW_UNQUALIFIED`；serial 产物必须含 D6 清理后读数（否则 `stage_check` STOP） |
| **E-3** | 顺序与预算 | 4 family × 3 模式 × 2 rep = **24 次**；诊断 pass ≤ 4（仅在出现分歧的 family 上）；按 family 分块，每块结束即 `stage_check`，非 0 即停 |
| **E-4** | 诊断与计时分离 | 诊断产物**不进入**任何性能/加速比结论（工具记录里已写入 `timing_separation` 字段） |

## 3. 需要你明确回答的批准语句（可直接引用回复）

1. **D-1 + D-8**：是否确认"exact-id 门**重新分层**到 serial oracle"？并明确 `design_freeze_g4_specdec.md` 是**冻结判据**（⇒ D-1 为修改）还是**待批准草案**（⇒ D-1 为正式批准）？
2. **D-2 + D-3**：是否批准 Gate A 扩到 9 family，并授权把 `F09/F03/F11/F04` 登记进**绑定 `tag=div8209_p2` + `tool=serial_oracle_gpu`** 的 run 级授权（含 scoped 检查同批落地）？
3. **D-4**：是否接受"**删除自动阈值** + DR1–DR3（全量保留、分层解释、不按分歧位置筛样本）"？是否采用 DR4？
4. **D-5 / D-6 / D-7 / D-9**：D-5 是否两条轨迹都评；D-6 是否批准"FP8 各自做 Gate A"；D-7 是否保留词条；D-9 是否同意不重复 FP32。
5. **执行准入**：是否同意 E-1（**先做 F09 单 family 端到端诊断核验**，失败即停）+ E-2/E-3/E-4 的运行纪律？

## 4. 获批后的执行序列（**本轮全部未执行**）

| 步 | 目的 | 实际命令（获批后） | 退出码要求 | 产物路径 | 继续条件 |
| --- | --- | --- | --- | --- | --- |
| 3 | 复检准入 | `stage2b_preflight_cpu` | **必须 0** | 仅 stdout | 非 0 ⇒ 停止，回到 D-3 |
| 3 | F09 AR×2 | `so_batch.ps1 -Tag div8209_p2 -Cases F09 -Modes ar -Reps 1,2` + `stage_check … --modes ar` | 0 | `div8209_p2_F09_ar_rep{1,2}_*.json/.pt` | 非 0 ⇒ 停止 |
| 3 | F09 prod×2 | 同上 `-Modes prod` + `stage_check … --modes ar,prod` | 0 | 同上 prod | 非 0 ⇒ 停止 |
| 4 | F09 诊断（**仅当有分歧**） | `correctness_protocol_margins --stage2b --runs-tag div8209_p2 --families F09` | **0**，非 4 | `div8209_p2_margins_<utc>.json` | 4/5 ⇒ 停止；核对同 tag/输入/配置、首分歧、两侧 token、margin、cache 长度 |
| 3 | F09 serial×2 | `-Modes serial` + `stage_check … --modes ar,prod,serial` | 0 | 同上 serial | 非 0 ⇒ 停止 |
| 3–4 | F03/F11/F04 | 重复上述三块（诊断仅在分歧时） | 同上 | 24 JSON + 24 `.pt` | 同上 |
| 5 | 判决 | `stage2b_verdict --tag div8209_p2 --families F09,F03,F11,F04 --modes ar,prod,serial --reps 2 [--gb6 …]` | 0 | `div8209_p2_stage2b_verdict_<utc>.json` | ≠ PASS ⇒ 按失败分类报告，不放行 |

## 5. 尚未解决的风险（提交时状态）

1. **D-1 的文书风险**：`:83` 字面指向 production，与 D-8 未裁决叠加 ⇒ 存在"未授权修改判据"的表述风险。
2. **数值并列**：`verification_readiness:267` 的三方精确并列 + margin 0.0 说明 exact-id 语义在并列处不唯一；本轮只把 `tied_at_max` 记入 GB6，**处置规则本身仍待 D-4/小组明确**。
3. **诊断 GPU 分支未验证**（E-1）⇒ 若首个 family 核验失败，Stage 2b 的 GB6 记账无法补齐，判决将停在 `CHECK_NEEDS_REVIEW`（这是正确行为，不是放行）。
4. **历史 F13 的两套口径并存**：16 项口径的历史 PASS 保留（旧聚合器 `--scope final` 仍复现），18 项（含 D6）新口径给出 FAIL。Stage 2b 必须**重跑** serial，不得用历史产物充当新判据证据。
5. **单次运行的环境脆弱性**：commit 可用 20.3 GiB vs 16 GiB 安全线，margin 约 +4.3 GiB；若后台占用上升会被环境门正确拒绝（记录为"环境拒绝"，不计数值失败，但会造成必需重复缺失 ⇒ 判决 `CHECK_NEEDS_REVIEW`）。

---

*本文件为**新增文档**，不改动任何冻结判据、历史结论或原始产物。*
