# Stage 2b 待批准决策单（2026-10-04）

> **本单是"请求批准"，不是批准结果。** 工具自测通过 = 工具证据，**不等于**协议批准。
> **签署用的一页版**：`docs/stage2b_approval_record_2026-10-04.md`（逐项推荐 + 后果 + 谁阻断什么）。
> 提交本单时：**未运行任何 GPU 步骤**、未改任何冻结判据、未覆盖任何历史 FAIL/CHECK/PASS。
> 逐项推荐的完整论证见 `docs/protocol_decision_brief_2026-10-04.md`；本单只给"决策 + 影响 + 风险"。

## 0. 当前准入状态（本轮只读实测）

| 检查 | 实际命令 | 结果 |
| --- | --- | --- |
| 预检 | `python -X utf8 -m specdec.engine.stage2b_preflight_cpu` | **exit 3** → `PREFLIGHT_OK_TOOLS_BUT_FAMILIES_UNREGISTERED`；F09/F03/F11/F04 四项均 `会被判越界=True` |
| 治理守卫 | `python -X utf8 -m specdec.engine.verify_family_independence_cpu` | **31/0**；其中两条相关检查为 `Stage 2b 工具的源码里只有'已授权 ∪ 候选'的 family 字面量`、`Stage 2b 候选 family **尚未被任何 run 消费**（D-3 批准前不得跑）` |
| 环境门 | `python -X utf8 -m specdec.engine.serial_oracle_env_check` | **ENVIRONMENT_OK**；commit 可用 20.3 GiB ≥ 安全线 16 GiB；入口 free 20.31 GiB ≥ 18 GiB（margin +2.31） |
| 工具自测（**仅工具证据**） | 8 个 CPU 自测 | 合计 **289/0**（verdict 92 · diag 25 · **stage_check 35** · agg 21 · guard 31 · audit 19 · env 24 · stage0 42） |
| 编号覆盖反例（已修） | `stage_check_selftest_cpu` | 两个 `rep1` / `rep1+rep3` / 同一产物计入两次 / 缺 `rep2` / **非法 `rep`（`'bad'`、上标 `'²'`、Unicode 数字 `'٣'`/`'１'`）** ⇒ 均**不得 OK** 且**不崩溃**；同一批夹具下 `stage2b_verdict` 既有判决**不变**（V1–V6） |
| 历史产物只读复核 | AR vs production 首分歧 | 输出下标 **47**（旧实现误取末位 63）；运行侧 GB6 字段全齐，仍缺 5 个诊断字段 |

**结论：GPU 阶段被"协议未批准"硬阻断；工具侧不再是阻断项。**

## 1. 逐项决策

### D-1 exact-token 门分层到 serial oracle —— **硬阻断**

| 项 | 内容 |
| --- | --- |
| **旧判据** | `design_freeze_g4_specdec.md:16`：判定 = `SpecDec ids == target-only greedy ids`（exact equality）；`:83`：**B6** correctness gate 条件写 `base_bf16` **与** `v_fp8`，判据 exact ids equality + 全 finite + 多次确定性一致，**B6 未过禁止 acceptance 测量/pilot/final**；`:96`：**exact-token-id 门不可取消** |
| **新判据（待批）** | Gate A = 在**与 AR 同路径**的 serial oracle（逐 token `M=1`）上要求完整 token IDs 逐 token 相等；**production（`M=4` 块验证）不设 exact-id 门**，其 token 分歧按"数值行为"**记录并分层解释** |
| **改变了什么** | **⚠️ 这是替换旧 production 判据的实质协议变更**（不是"换一个更准的测量工具"，也**不是**"门没有取消"）：请按"批准对象"来理解 —— 应批准一份**修订协议**，逐条列明被替换的旧条款（见批准记录 §A2 的替换表）：**`:16` 与 `:83`（instrument/对象部分）被替换**、**`:83`（其余：prompts / finite / 确定性 / 每 KV 各自验证 / 未过门禁止 acceptance·pilot·final）保留**、**`:96` 保留约束力但适用对象改为 serial 门**、**production exact-id 门被移除**。**由此产生的表述纪律（必须写进纪要）**：<br>· serial oracle 通过 ⇒ **只**说明"控制逻辑在与 AR 同路径（逐 token `M=1`）时与 AR 逐 token 相等"；<br>· serial oracle 通过 **不等于** production（`M=4` 块验证）与 AR 输出一致；<br>· production 与 AR 的 token 差异**必须如实报告**为**发散轨迹（divergent trajectory）**上的结果；<br>· **禁止**"同输出无损加速 / 输出等价 / lossless speedup"类表述（`production_correctness_decision:60`；`g5_measurement_protocol:159-164` 的 speedup 声明资格 10 条**不含** `ar_equal`）；速度比必须标"**分歧轨迹上的比较**"并同时报告共同前缀 P、两侧接受统计与轮数（DR2） |
| **风险** | ① 与 `:83` 字面语义（指向 production）冲突 ⇒ 必须同时解决 **D-8**（否则 D-1 无合法落点）；② `:96` 明确保留风险，`:267` 记录的 **F06/index25 三方精确并列（margin 0.0）** 说明"逐 token 精确相等"在**数值并列**位置不具语义唯一性 ⇒ 新仪器也会遇到并列，**必须预登记并列处置规则（见 §1A，随本单提交）**；③ 声明边界见上（禁止无损表述）；④ **不得**把本次变更描述成"原样批准旧草案"或"只是换仪器" |
| **建议** | **有条件批准**（三条件写进纪要）：① 明示"**替换旧 production 判据的实质协议变更**"并附逐条替换表；② 生产分歧一律作为"发散轨迹"如实报告、禁止无损表述；③ **§1A 并列规则先行定稿**再跑。旧文件处置路径二选一（见批准记录 §C） |

### D-2 **运行范围与证据范围**（不再是"覆盖 5→9"）—— **硬阻断（决定可声称结论的边界）**

| 项 | 内容 |
| --- | --- |
| **旧判据** | Gate A 的历史证据来自已用 5 个 family（F13/F01/F02/F08/F10，**旧口径**：16 项硬检查，**无 D6**） |
| **新判据（待批）** | **保持现有运行范围**：本次只新跑 `F09/F03/F11/F04` ⇒ **4 family × 3 模式 × 2 次 = 24 runs**（`tag=div8209_p2`）+ **≤4** 次诊断 pass。结论一律写为"**4 个新口径验证 + 5 个历史辅助证据**" |
| **改变了什么** | ① 运行清单明确为 **24 runs**（不含旧 5 family 的串行重跑）；② **可声称结论**收窄：旧 5 family 的 serial 产物**缺新 D6 检查**，**不得**与新口径证据合并成"9 family 均通过同一新 Gate A" |
| **风险** | 若把旧 5 family 并入门内 ⇒ **等于暗中扩大范围**（旧产物无 D6、也没有新口径下的 serial 复跑）⇒ 结论不可辩护；另一方向的风险是样本偏少（仅 4 family），故必须**显式**说明是"4 新 + 5 历史辅助证据"，不得含糊成"9 个" |
| **建议** | **批准"保持 4 新 family 运行范围"**（四个 family 的 provenance 已核验：manifest 预登记、`token_sha1` 复现、窗口不重叠、32-gram 零交集、现有产物零消费）。**若要"统一 9-family Gate A"**：必须**另列**旧 5 family 按新口径重跑的成本与计划（serial ≥ 5×2 = 10 runs + 诊断）并**单独等待批准**，**不得**在本次签名里默认包含 |

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
| **建议** | **采用 DR1–DR3**；**DR4 明确不采用**（不设任何数值阈值；理由：无可辩护的依据、且诱导事后选择）。若小组仍要采用 DR4，属对推荐的**显式偏离**，必须在**看到结果之前**写明阈值依据与来源 |

### D-7 判决词条与"失败分类" —— **不阻断运行，阻断结论表述**

* **旧判据**：只有"算法正确/失败"一类，production 结构失败与算法失败会被混为一谈。
* **新判据（待批）**：保留 `NUMERICAL_DIVERGENCE_RECORDED`（分歧已记录）与 `PRODUCTION_RUNTIME_FAIL`（结构/运行时失败）等词条，把"分歧"与"失败"分开。
* **现状**：工具已实现 `SERIAL_ORACLE_PASS / SERIAL_ORACLE_FAIL / BASELINE_REGRESSION / CHECK_NEEDS_REVIEW` + `production_divergence_class`（`TOKEN_EQUAL` / `NUMERICAL_DIVERGENCE_RECORDED`），**但词条口径仍待小组确认**。
* **建议**：**批准保留**（可改名以降低沟通成本）。

### D-8 `design_freeze_g4_specdec.md` 的效力状态 —— **硬阻断（D-1 的落点问题）**

| 项 | 内容 |
| --- | --- |
| **事实** | 该文件自标 **DRAFT v0.3**，`:103` 批准栏**未勾选**，却被多处当"权威/冻结判据"引用 |
| **新判据（待批）** | 先裁决：它是**已冻结判据**（则 D-1 = **修改冻结判据**，须走正式变更、留痕、并说明"为什么不是取消门"）还是**待批准草案**（则 D-1 = **正式批准该草案**并补齐批准栏）。`proposal_freeze_2026-10-01.md:31` 已被新 `final` proposal 取代，亦须在纪要中明确"以 final 为准"。**两种解读下都不得"沉默地"改判据**：`:96` 的"门不可取消"必须保持字面有效——本方案**不取消**门，只更换其**兑现路径** |
| **风险** | 不裁决 ⇒ D-1 没有合法落点；两种解读下的批准动作与留痕要求不同；若被读成"草案直接当判据用"，会与 `verification_readiness:254`（"航线未获准则按现行冻结判据处理"）冲突 |
| **建议** | **与 D-1 同批裁决**，并在纪要写明结论 |

### D-5 质量口径 —— **不阻断 Stage 2b 运行，阻断"质量等同/无损"类声明**

* **旧判据**：WikiText-2 test、非重叠窗口、absolute PPL + relative ΔPPL；`production_correctness_decision:103` 已明确"标准 AR PPL 不能直接当作 SpecDec 生成质量"。
* **新判据（待批）**：沿用冻结质量口径 + 明确**不得**用于声称"分歧后生成质量等同"；是否**两条轨迹都评**由小组定。
* **建议**：**沿用 + 明确禁用范围**；"两条轨迹都评"请小组裁决（本阶段不受阻）。

### D-6 `v_fp8` 是否各自做 Gate A —— **阻断 FP8 档（不阻断 BF16 Stage 2b）**

* **旧判据**：`:83` 把条件写成 `base_bf16` **与** `v_fp8`。
* **新判据（待批）**：两个 KV 条件**各自**做 Gate A；现有 serial 证据**只有 BF16**。
* **建议**：**批准"各自验证"**；本阶段不跑 FP8，**禁止**把 BF16 结论外推到 FP8。

### D-11 `:83` 的 **prompts 类型覆盖**如何处置 —— **阻断"完整 B6 通过"状态，但不阻断 Stage 2b 本身**

| 项 | 内容 |
| --- | --- |
| **旧判据** | `design_freeze_g4_specdec.md:83`：B6 的 prompts 覆盖 = **自然语言/数学/代码/短/长/全接受/首位拒绝/中途拒绝/EOS/非 4 倍数长度**；条件 `base_bf16` **与** `v_fp8`；判据 = exact ids equality + 全 finite + 多次确定性一致 |
| **现状（事实，须分清口径）** | ① **旧口径**下确有覆盖：`g4_b6_correctness.py` 定义 10 类 case（`natural-1/2`、`math`、`code`、`short`、`long`、`len13`(非 4 倍数)、`len5`、`len2`(预算<k)、`eos-prone`），历史 `B6-fp32` PASS 20/20、`B6-bf16` CHECK 19/20；② **新口径（serial oracle exact-id）下这 10 类完全没有覆盖**，且**没有任何代码实现**"提示类型矩阵"；③ Stage 2b 的 4 个 8K family **只映射到"自然语言 + 长上下文"两项**（`rounds` 里自然出现全接受/拒绝轮，属**观测**，不是为这些类型**设计**的用例） |
| **结论（必须写进报告）** | **Stage 2b 不能宣称"完整 B6 已通过"** —— 它是**限定范围的新口径验证**（4 family × 8K 自然语言 × 24 runs） |
| **推荐（选项 a）** | **保持 `:83` 覆盖要求不变**，另列 **B6 类型覆盖轮**待办：<br>· 工具改动：serial-oracle 运行器现为 manifest/family 取数 ⇒ 需增"prompt 来源模式"（工具改动，需批准 + CPU 自测）；<br>· 成本**估算**（非承诺）：10 类 case × 3 模式 × 2 次 = **≥60 runs**（BF16）+ ≤10 次诊断；<br>· 准入：D-1…D-4 已批准、工具改动完成并自测通过、`preflight`=0、环境门通过 |
| **其它选项** | (b) **修改/豁免**该覆盖要求 ⇒ **实质协议变更**，须单独批准并列明放宽范围与理由（**不得**在文字里隐去）；(c) 暂缓 ⇒ "完整 B6"状态一律标**"未完成"**，且不得声称状态 ②③④ |
| **建议** | **选 (a)**：保持要求、另列待办；本轮**不**主张任何豁免 |

### D-12 `:83` 的"**B6 未过禁止 acceptance 测量/pilot/final**"的作用范围 —— **阻断/影响放行口径**

| 项 | 内容 |
| --- | --- |
| **旧判据** | `:83` 同句：**B6 未过禁止 acceptance 测量/pilot/final**（但**未写明**是每 KV 条件分别放行，还是两个 KV 全部通过后全局放行） |
| **选项 a（推荐）** | **按 KV 条件分别放行**：`base_bf16` 过 ⇒ 放行 BF16 侧；`v_fp8` 过 ⇒ 放行 FP8 侧。**性质 = 对 `:83` 的解释（clarification），不是修改**（旧文把门定义为"**条件** `base_bf16` 与 `v_fp8`"）。后果：不被另一 KV 阻塞；但每条结论必须标明 KV 条件，且遵守"**BF16 通过绝不外推为 FP8 通过**" |
| **选项 b** | **全局放行**：`base_bf16` **与** `v_fp8` 全部通过后才允许任何 acceptance/pilot/final。后果：BF16 结果在 FP8 门通过前**不能用于任何下游用途**（先封存）；FP8 成为 BF16 的前置。风险最低、进度最慢 |
| **建议** | **选 (a)**，并在纪要中**显式写明这是解释而非修改**；无论选哪种，报告必须区分 §A4 的四个状态 |

### D-9 是否再补 FP32 口径 —— **不阻断**

* 现状：`B6-fp32` 已 PASS（20/20 组合、216/348 token），与 BF16-serial 互为印证（`g5_measurement_protocol:268`）。**建议不重复**（除非代码/语义变更 ⇒ 需回归）。

### D-10 精确并列处置规则（= 原 §1A）—— **阻断 Stage 2b（须在跑前定稿）**

* 规则全文见 **§1A（T-0…T-6）**；**推荐：默认阻断、TIE-A 不豁免**；双侧分类在 P-1/P-2 落地前一律记 **TIE-C（证据不足）**。
* 与 D-11/D-12 的关系：并列分类只影响**解释**，**不改变** Gate A 判决；`:83` 的覆盖范围与放行范围问题**不会**因并列规则而消失。

## 1A. 精确并列处置规则（**待批准文本**；须在任何 Stage 2b GPU 结果出现**之前**定稿）

> **背景**：`verification_readiness:267` 记录过 F06/index25 位置 8217 的**三方精确并列**（margin `0.0`，靠 `argmax`
> 取最小下标选出 token），AR 臂为两方并列 —— 即冻结判据"逐 token 精确相等"在**数值并列**处不具语义唯一性。
> 本规则把"精确并列"变成一个**事先定义好的解释性分类**，**不引入任何新容差、不做任何自动豁免**。

**T-0 术语与适用范围**
* 本规则处理**精确并列（exact tie）**：判定行的最大值在同一 logits 张量里出现 ≥2 次，且这些最大值**浮点表示完全相同**（`(row == row.max()).sum() >= 2`）。`margin` 为**精确 `0.0`** —— **不是**"小于某个 eps"，**不设任何阈值**。
* 与 `g5_measurement_protocol:270-275` 的 `BF16_NEAR_TIE`（**近似**并列，需结构不变量 / off-by-one / 邻位更远 / `|fp32 margin| ≤ own_max_abs` 四条件）**是两件事**：本规则**不引入**那四个条件，**不得**用 near-tie 的 `NEEDS_INSPECTION` 代替本规则，**也不得**用"精确并列"去满足 near-tie。

**T-1 如何从 logits 判断并列**（对"做出该 token 决策的那一行"：block 臂 = production 块验证行；serial 臂 = 逐 token 单步行；AR 臂 = 其自身 decode 行）
1. 取该行**决策所用**的 dtype 与设备上的 logits 来判并列。**⚠️ 技术澄清（本轮修正）**：把**已有的 BF16 logits 单纯 cast 成 FP32 不会改变任何相等关系** —— BF16 的每个值都能被 FP32 **精确表示**，cast 是单射 ⇒ 并列集合（以及"是否并列"）在 cast 前后**完全相同**。因此 cast **既不能**用来判"是否真并列"，**也不能**用来当"FP32 对照"。
2. `m = row.max()`；`tied_at_max = int((row == m).sum())`；**并列 ⇔ `tied_at_max >= 2`**；
3. 记录 `top1`/`top2`（token id + 值）、`margin = v1 - v2`（并列时 `0.0`）、`argmax`；并列时框架 `argmax` 取**最小下标**，必须与该 run **实际选出的 token 一致** —— 不一致即"记录与实现不一致"，**判 STOP**（不得当作并列解释掉）；
4. **FP32 对照必须是"重新以 FP32 计算 logits"**（一次**独立的 FP32 前向**），**不是**把 BF16 结果 cast 上来。只有重算才可能**改变并列结构**（FP32 下原本相等的两个值可能分开，原本不等的两个值也可能变得相等）。该对照**仅作解释证据，不作为放行条件**，用于区分"真语义并列"与"BF16 舍入造成的并列"。

**T-2 serial 与 AR 在并列处 token 不同时如何分类**（逐位置独立分类，**不合并、不按位置筛样本**，与 DR3 一致）
* **TIE-A `EXACT_TIE_BOTH_TIED`（措辞已按本轮修正）**：两侧决策行在该位置**都**并列，且两个不同 token **各自都落在对应侧的并列最大值集合内** ⇒ 该结果**只能说"与并列敏感性相容"（tie-sensitivity-compatible）**，**不能**直接归因为"并列取值顺序"。理由：并列只说明**该位置对极微小的数值差异敏感**，它**不说明**两侧数值差异的**来源与大小**；"两侧都并列"与"差异由并列造成"是两件事。**要作因果归因，需要额外证据**（例如两侧决策行的逐元素差值、以及 FP32 重算对照），在缺这些证据时只能停在"相容"这一档措辞。
* **TIE-B `EXACT_TIE_ONE_SIDED`**：只有一侧并列，或另一侧 token **不在**其并列集合内（或该侧不并列且 `margin > 0`）⇒ **与并列敏感性不相容**（该差异**不能**用并列解释）。
* **TIE-C `EXACT_TIE_UNRESOLVED_EVIDENCE`**：缺判定行证据（缺任一侧的决策行 / dtype 未记录 / `argmax` 与实际 token 不符）⇒ **无法分类**（按"证据不足"处理，**不得**默认成 TIE-A，**也不得**用另一侧的证据代替）。

> ⚠️ **本轮新增的可得性限制（与 §1A-T-6 一致，必须与 T-2 一起读）**：当前工具**只能**给出 **AR 侧**的决策行重建
> ⇒ **TIE-A/TIE-B 双侧分类在现阶段不可判**，一律记 **TIE-C（证据不足）**。详见 T-6 的字段审计与最小补证方案。

**T-3 是否阻断 Gate A —— 我的建议：阻断（不自动豁免；本轮再次确认，严格门不变）**
* **默认阻断**：只要 serial 与 AR 在任一位置 token 不同，Gate A 判 `SERIAL_ORACLE_FAIL`（与 `design_freeze:96` 一致）。**并列分类（TIE-A/B/C）不改变判决**：它只是**附着在失败记录上的解释性分类**，"与并列敏感性相容"**不等于**"token 相等"，更**不等于**门通过。
* **明确禁止**：把 TIE-A 当作自动豁免、把"相容"读成"等价"、或据此声称 Gate A 通过。
* **若小组希望"TIE-A 不阻断"**：那属于**放宽冻结判据**（修改 exact-token 门），必须同时满足：① 在**看到任何 Stage 2b 结果之前**定稿；② 与 **D-8** 一起走正式变更流程并留痕；③ 豁免**仅限** T-0 定义的精确定义情形（两侧都并列且两个 token 都属各自并列集合），逐位置提供 T-4 全部证据；④ **不得**引入任何 margin/ULP 阈值。**我的建议是不采用该选项**：先按严格口径跑一轮、拿到并列频率实测数据后再由小组决定是否提出变更（届时任何阈值也只能对**下一轮**预登记）。
* 本规则同样适用于 production（`M=4`）与 AR 的比较（用于**解释**生产分歧），但生产层本就不设 exact-id 门。

**T-4 需要保留哪些证据**（逐位置）
1. 位置：输出内下标、绝对位置（`prompt_len + 下标`）、所属轮号（由该 run 的 `rounds` 推出）、共同前缀长度 `P`；
2. 两侧 token id + 该位置**行摘要**：`argmax`、`top1/top2`（id + 值）、`margin`、`tied_at_max`、**并列 token id 集合**（≤8 个全列；>8 个列总数 + 按 id 升序前 8 个）；
3. 产生该行的**臂**与 **dtype/layout**（block bf16 / serial bf16 / AR bf16 / FP32 对照）以及该行 cache 长度；
4. 该位置在**同配置两次独立重复**上是否同样出现（并列与 token 选择是否稳定）；
5. 两侧完整 token 序列与停止原因（= GB6 运行侧字段）以及两侧 accept/work 统计（AR 按 decode steps / target calls 记）；
6. 结论标签（TIE-A/TIE-B/TIE-C）+ 判据出处（哪一行、哪个 dtype、`tied_at_max` 值、并列集合成员关系逐侧列出）；
7. **描述性统计**：并列位置数 / 总决策位置数、**因并列而不同的位置数**（只统计，**不产格、不放行**）。

**T-5 预登记时点**
本规则随本决策单在**任何 Stage 2b GPU 运行之前**提交（以 commit 为证）。一经批准即冻结：**不得**在看到结果后修改阈值、增删分类或改变阻断结论。

**T-6 "规则所需证据 vs 当前工具实际产出"字段审计（本轮完成；结论：双侧分类暂不可判）**

> **⚠️ 四个状态互不推出（与批准记录 §A4 同步）**：**① Stage 2b（限定范围的新口径验证）通过 ⇏ ② 完整 B6 通过 ⇏ ③ BF16 正式实验可开始 ⇏ ④ FP8 正式实验可开始**。
> Stage 2b = 4 family × 8K **自然语言** × 24 runs，**不覆盖** `:83` 的 prompts 类型清单（数学/代码/短/EOS/非 4 倍数/…），也**不含** FP8 ⇒ **不得**宣称"完整 B6 已通过"，**不得**外推为 FP8。

> **⚠️ 本轮决定记录（2026-10-04，待小组确认）：不扩展并列诊断范围。**
> 现有 **AR 单侧重建足以完成既定的 GB6 记账**（`ar_top1`/`ar_top2`/`margin`/`tied_at_max` + `cache_len_after`），
> 但**不足以**判 **TIE-A/TIE-B**；在 P-1/P-2 获批并实现之前，凡**需要双侧证据**的分歧**一律记 TIE-C（证据不足）**。
> **严格 exact-id 门不变、不自动豁免**：并列分类永远不会把 `SERIAL_ORACLE_FAIL` 变成 PASS。
> 不为此改动任何生产推理路径，也不为此改运行器/harness（P-1/P-2 仍是"待批准"的可选项）。

> **核心技术事实**：`correctness_protocol_margins` **只重建 AR 口径的决策行** —— 它按 AR 的方式走
> （分块 prefill + 逐 token `M=1` 沿**该 run 的 token 序列**步进），在首分歧位置取 target 的最后一行 logits。
> 它**不能**给出 **serial 运行**或 **production（`M=4`）块验证**在该位置的**实际决策行**：
> 那两行的 cache 构造历史与 GEMM 形状都不同（本项目已记录的"形状敏感性"正是分歧机制本身）。
> **因此：不得用 AR 单侧 logits 推断另一侧是否并列，也不得把重建行当作任一侧的实际行。**

| T-4 所需证据 | 当前产出方 | 现状 |
| --- | --- | --- |
| 位置：输出内下标 / 绝对位置 / 轮号 / 共同前缀 `P` | AR 与 production 产物逐 token 比对 + `end_state.prompt_len` + `rounds` | ✅ **已有**（`stage2b_verdict` 的 GB6 运行侧；真实 F13：`47 / 8239 / 18 / P=47`） |
| 两侧 token id | 同上 | ✅ **已有** |
| **AR 侧**行摘要：`argmax` / `top1,top2` / `margin` / `tied_at_max` | `correctness_protocol_margins`（AR 口径**重建**） | ✅ **已有**（= "现有工具已能完成的 GB6 AR margin 记账"），但 `cache_len_after` 是**重建**出来的 cache 长度，**不是**该 run 的实际 cache 长度，需在记录里标注来源 |
| **并列 token id 集合**（成员，而非仅计数） | 无 | ❌ **缺**（现只记 `tied_at_max` 计数；要判 TIE-A 必须知道成员是哪些 token） |
| **对侧（serial / production）的实际决策行** | 无（诊断工具只有 AR 口径重建） | ❌ **缺** ⇒ **TIE-A/TIE-B 不可判** |
| 产生该行的臂 + dtype/layout（bf16/fp32、block/single-step） | 产物的 `kv_format` / `dtypes`；诊断工具未记"该行为 AR 口径重建"的显式臂标签 | ⚠️ **部分**（需在诊断记录里加 `arm`/`dtype` 显式字段） |
| FP32 **重算**对照行 | 无（现有 `--target-rope cpu_fp32` 是 RoPE 口径，非"该位置 logits 的 FP32 重算"） | ❌ **缺** |
| 两次重复上"并列是否同样出现" | 只有 token 层面的两次一致性 | ❌ **缺**（需要每侧每位置的行） |
| 两侧完整序列 / 停止原因 / work 统计 | GB6 运行侧字段 | ✅ **已有**（AR 按 decode steps / target calls 记，不填造接受率） |
| 标签 + 判据出处（TIE-A/B/C + 依据） | 无 | ❌ **缺**（本轮**不实现**：在无对侧证据时标签只能恒为 TIE-C） |
| 并列频率描述统计 | 需全位置的行 | ❌ **缺**（依赖下面的补证） |

**最小补证方案（**待批准、本轮不实施**；任何一项都不得改动生产推理路径的数值行为）**
* **P-1 serial 侧决策行（必需）**：在 serial oracle 运行器上加**可选、只读**的"决策行捕获"（如 `SO_CAPTURE_ROWS=1`），
  把该 run **每一步** target 已算出的 logits 行**按原 dtype 原样**存 side-car（≤64 步 × 词表；bf16 约 20 MB/run），
  并记录 `arm="serial"`、`dtype`、位置索引、cache 长度。**只读已算出的张量，不改变任何计算**；
  需以"开启/关闭捕获两次跑出相同 token"来证明无副作用（CPU 侧可先用 Stage 0 类自测覆盖开关逻辑）。
* **P-2 production 侧决策行（必需）**：在 harness 里同法捕获块验证行（每轮 `M=4` 的第 i 行，即判定 d_{i+1} 的那一行），
  记 `arm="production"`；同样只读、同样以确定性复跑验证。
* **P-3 诊断工具的自我标注（低成本）**：`correctness_protocol_margins` 的记录里显式写
  `arm="ar_reconstruction"`、`dtype`、`cache_len_after_source="reconstructed"`，避免被误读成 serial/production 的行；
  并增加 `tied_token_ids`（并列成员）。
* **P-4 FP32 重算对照（可选）**：同一位置用 FP32 **重新前向**（不是 cast 已有 bf16 结果）产出一行，作为"真语义并列 vs BF16 舍入"的解释证据。
* **在 P-1/P-2 落地之前**：T-2 的双侧分类**一律记 TIE-C（证据不足）**，只报告 **AR 侧** `tied_at_max` 作为**线索**；
  **不得**据此判 TIE-A、**不得**据此豁免 Gate A。
* **时序要求**：若小组希望 Stage 2b 的结果里带上双侧并列分类，则 **P-1/P-2 必须在跑 Stage 2b 之前**批准并实现
  （事后无法从已落盘产物补出对侧决策行）；若不带，则本阶段并列一律记 TIE-C，Stage 2b 仍可照常按严格门判决。

## 2. 运行准入条件（**不是**协议批准，但缺一不可；需你同意执行顺序）

| # | 条件 | 说明 |
| --- | --- | --- |
| **E-1** | **诊断工具 GPU 分支未验证** | `gb6_diag_select` 的 CPU 配对/来源/配置核对已验证（25/0 + 真实产物 47），但 `correctness_protocol_margins` 的 GPU 重建分支（prefill + 逐 token + top-k/margin/cache）**从未在 GPU 上跑过** ⇒ 建议**第一个 GPU 动作**就是**单 family 端到端核验**（F09：AR×2 + prod×2 → 若有分歧则跑诊断），核对 6 项一致性，**失败即停** |
| **E-2** | 环境与运行纪律 | `env_check` 必须 `ENVIRONMENT_OK` 且入口 free ≥ 18 GiB；**一个 run 一个独立进程**（`scripts/so_batch.ps1 -Tag … -Cases … -Modes … -Reps …`）；**不自动重试、不挑选样本**；全部原始产物保留；禁止 `SMOKE_ALLOW_UNQUALIFIED`；serial 产物必须含 D6 清理后读数（否则 `stage_check` STOP） |
| **E-3** | 顺序与预算 | 4 family × 3 模式 × 2 rep = **24 次**；诊断 pass ≤ 4（仅在出现分歧的 family 上）；按 family 分块，每块结束即 `stage_check`，非 0 即停 |
| **E-4** | 诊断与计时分离 | 诊断产物**不进入**任何性能/加速比结论（工具记录里已写入 `timing_separation` 字段） |

## 3. 需要你明确回答的批准语句（可直接引用回复）

1. **D-1 + D-8**：是否确认"exact-id 门**重新分层**到 serial oracle"（批准对象 = **修订协议 v1**，逐条替换表见批准记录 §A2：serial 门保留、**production exact-id 门被移除**）？旧文件处置路径选 **A（推荐：修订协议为批准对象、旧文件原样存档）** 还是 B（正式变更流程修订为 v0.4）？
2. **D-2 + D-3**：是否批准**保持现有运行范围**（`F09/F03/F11/F04` × 3 模式 × 2 次 = **24 runs**），并把结论写为"**4 个新口径验证 + 5 个历史辅助证据**"（**不**声称 9 family 同一新 Gate A）？并授权四者在**绑定 `tag=div8209_p2` + `tool=serial_oracle_gpu`** 下运行（scoped 检查同批落地）？若坚持"统一 9-family Gate A"，请**另附**旧 5 family 重跑计划与成本并单独批准。
3. **D-4**：是否接受"**删除自动阈值** + DR1–DR3（全量保留、分层解释、不按分歧位置筛样本）"？**DR4：推荐不采用**（不设数值阈值）；若采用，须在结果前写明依据。
4. **D-5 / D-6 / D-7 / D-9**：D-5 是否两条轨迹都评；D-6 是否批准"FP8 各自做 Gate A"；D-7 是否保留词条；D-9 是否同意不重复 FP32。
5. **D-10（= §1A 精确并列处置规则）**：是否批准"**默认阻断、TIE-A 不自动豁免**"？若希望"TIE-A 不阻断"，请明示这属于**放宽冻结判据**（须与 D-8 同批、且只能在看到结果前定稿）。
6. **D-11（`:83` 的 prompts 类型覆盖）**：选 **(a) 保持要求不变 + 另列"B6 类型覆盖轮"待办（推荐）** ／ (b) 修改或豁免（**实质协议变更**，须写明放宽范围与理由并单独批准） ／ (c) 暂缓（"完整 B6"一律标"未完成"）？请确认：**Stage 2b 只算"限定范围的新口径验证"，不得宣称完整 B6 通过**。
7. **D-12（"B6 未过禁止 acceptance/pilot/final"的作用范围）**：选 **(a) 按 KV 条件分别放行（推荐，作为对 `:83` 的解释；BF16 通过绝不外推为 FP8 通过）** ／ (b) 全局放行（两个 KV 全部通过后才允许）？
8. **执行准入**：是否同意 E-1（**先做 F09 单 family 端到端诊断核验**，失败即停）+ E-2/E-3/E-4 的运行纪律？
9. **并列双侧分类的补证（T-6 的 P-1/P-2）**：是否**在 Stage 2b 之前**批准并实现"决策行捕获"（serial 与 production 两侧）？
   若**不**批准，本阶段的并列**一律记 TIE-C（证据不足）**，只报 AR 侧 `tied_at_max` 作为线索 —— Stage 2b 仍可按严格门照常判决。

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
2. **数值并列**：`verification_readiness:267` 的三方精确并列 + margin 0.0 说明 exact-id 语义在并列处不唯一；本轮已把 `tied_at_max` 记入 GB6，并提交 **§1A 待批准规则**（默认阻断、TIE-A 不豁免）⇒ **规则未获批前不得开跑**（否则等于事后处置）。
3. **诊断 GPU 分支未验证**（E-1）⇒ 若首个 family 核验失败，Stage 2b 的 GB6 记账无法补齐，判决将停在 `CHECK_NEEDS_REVIEW`（这是正确行为，不是放行）。
4. **历史 F13 的两套口径并存**：16 项口径的历史 PASS 保留（旧聚合器 `--scope final` 仍复现），18 项（含 D6）新口径给出 FAIL。Stage 2b 必须**重跑** serial，不得用历史产物充当新判据证据。
5. **单次运行的环境脆弱性**：commit 可用 20.3 GiB vs 16 GiB 安全线，margin 约 +4.3 GiB；若后台占用上升会被环境门正确拒绝（记录为"环境拒绝"，不计数值失败，但会造成必需重复缺失 ⇒ 判决 `CHECK_NEEDS_REVIEW`）。
6. **编号覆盖已收紧，但仍依赖"产物即证据"**：`stage_check` 现在要求每个请求模式**恰好** `rep1..rep2` 各一份（两个 `rep1` / `rep1+rep3` / 同一产物计入两次 / 缺号 / **`rep` 非法值** ⇒ 一律 STOP，不返回 OK，且**不崩溃**）。这只是"记录层"的防线：**它不能证明 GPU 真的独立跑了两次**（同一 run 被复制成两个文件名，在记录层无法与"两次一致"区分）⇒ 核对仍依赖 `so_batch` 的逐 run 日志与 `run_id` 唯一性人工复核。
7. **并列双侧分类当前不可判**（T-6）：诊断工具只重建 **AR 口径**的决策行，缺 serial / production 的实际决策行 ⇒ TIE-A/TIE-B 记 TIE-C；若小组要双侧分类，**必须在 Stage 2b 之前**批准并实现 T-6 的 P-1/P-2（事后无法补）。
8. **覆盖范围混用风险（D-11/D-12）**：Stage 2b 只跑 4 个 8K **自然语言** family ⇒ **不得**宣称"完整 B6 已通过"（`:83` 的数学/代码/短/EOS/非 4 倍数等类型在新口径下**未覆盖**；历史 10 类 case 是**旧口径**证据）；且"未过门禁止 acceptance/pilot/final"的**作用范围**（按 KV 分别 / 全局）**旧文未写明** ⇒ 必须在纪要中显式选定，否则极易把 BF16 结论外推成 FP8。

## 6. 修改文件与测试清单（本轮）

**本轮（覆盖范围修正轮）修改**：`docs/stage2b_approval_record_2026-10-04.md`（§A2 增两行：`:83` prompts 覆盖的现状与缺口、`:83` 放行范围澄清条；新增 §A4 四状态互不推出、§A5 两处覆盖问题的推荐处置；(6) §A6 阻断清单与 §C 签署项增 D-11/D-12）·
本文件（新增 **D-11/D-12** 两节、**D-10** 正式编号、§1A-T-6 顶部四状态提示、§3 问题重排为 1–9、§5 新增风险 8）·
`docs/stage2b_runbook_2026-10-04.md`（§2 增"限定范围 + 四状态"声明、§5 待批准表增 D-10/D-11/D-12）。

**前几轮已改（沿用）**：`specdec/engine/stage_check.py` · `specdec/engine/stage2b_verdict.py` ·
`specdec/engine/stage_check_selftest_cpu.py`（35 项）· `docs/protocol_decision_brief_2026-10-04.md`（D-1/D-2/D-4 修订注记）。
**测试清单（最近一次实跑，全部 exit 0）**：

| 命令 | 结果 |
| --- | --- |
| `python -X utf8 -m specdec.engine.stage_check_selftest_cpu` | **35/0**（N1–N10 + V1–V6） |
| `python -X utf8 -m specdec.engine.stage2b_verdict_selftest_cpu` | **92/0**（既有判决不受影响） |
| `python -X utf8 -m specdec.engine.verify_family_independence_cpu` | **31/0** |
| `python -X utf8 -m specdec.engine.serial_oracle_aggregate_selftest_cpu` | **21/0** |
| `python -X utf8 -m specdec.engine.gb6_diag_select_selftest_cpu` | **25/0** |
| `python -X utf8 -m specdec.engine.stage_check --tag div8209_so --family F13 --modes ar,prod,serial --reps 2` | **STOP(1)**，仍为 6 项（编号覆盖未产生误报） |
| `python -X utf8 -m specdec.engine.stage_check --tag div8209_p2 --family F09 …` | **PENDING(2)** |
| `python -X utf8 -m specdec.engine.stage2b_preflight_cpu` | **exit 3**（F09/F03/F11/F04 未登记，待 D-3） |

**未执行**：任何 GPU 运行、任何 family 登记、任何冻结判据修改、任何历史结论覆盖、任何生产推理路径改动。

---

*本文件为**新增文档**，不改动任何冻结判据、历史结论或原始产物。*
*提交时状态：**Stage 2b GPU 未运行、协议未获批准**。*
