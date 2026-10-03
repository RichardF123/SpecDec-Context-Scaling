# SpecDec 正确性协议 **v2**（serial oracle 版）— 草案，**未冻结**

* 版本串：`layered_v2_serial_oracle`（2026-10-04）
* 状态：**DRAFT / 未冻结**。本文件**不改变**任何历史结论，也**不**替代任何已提交文件。
  按 §9 的建议，冻结前需要用户/导师就 6 项事项作出决定（判决：`NEEDS_DECISION`）。
* 取代关系：本版**取代** `docs/correctness_protocol_post_shape_localization_2026-10-03.md`
  （`post_shape_localization_v1`）中的 **Gate A 工具**（row-aligned → serial oracle）与**决策词表**；
  v1 的 Gate B 记录型条款被**保留并加固**。**v1 全文与其中所有 FAIL 记录原样保留、不改写。**
* 依据（只读引用，不重跑）：`docs/serial_target_oracle_design_2026-10-03.md`、
  `docs/serial_oracle_stage0_report_2026-10-03.md`、
  `docs/serial_oracle_stage1_f13_2026-10-03.md`、
  `docs/serial_oracle_stage2_2026-10-03.md`、
  `docs/correctness_protocol_validation_2026-10-03.md`（历史判决 `ALGORITHMIC_GATE_FAIL`，不改写）。

---

## 0. 为什么需要 v2（一段话）

v1 把"**控制逻辑是否正确的认证**"交给 **row-aligned** 诊断模式。但 row-aligned 并不是**真正的 oracle**：
它仍然在一次 `M=4` 的 block 前向里计算，只是把第 0 行取出来（其数值闭合与 L2 层差异均已记录），
因此它**同时**受"控制逻辑"和"块内执行形状的数值行为"两类因素影响 —— 结果就是它在 4 个未见 family 上
只有 1 个与 AR 一致，得出 `ALGORITHMIC_GATE_FAIL`，而**这个结论无法区分**"控制逻辑错"与"块形状数值差异"。

**serial target oracle** 把 target verification 换成**与 AR 完全相同的逐 token `M=1` 前向**
（`spec_dec._verify` 内部分派；accept/replacement/rollback/cache/EOS 逻辑**逐字复用**），
并在调用几何审计下**硬性证明** generation 阶段不存在任何 `s>1` 前向。于是它把两类因素**解耦**：
* 在 serial 下：控制逻辑 + 与 AR 相同的执行路径 ⇒ 若仍与 AR 不一致，就是**控制逻辑**的问题；
* 在 production 下：保留原始 batched verify ⇒ 差异属于**生产路径的数值行为**，单独记录与解释。

**这不降低任何门槛**：Gate A 的 token 一致性要求**不变**（仍然逐 token 完全一致），只是换成了一个
**真正**的 oracle；production 路径则从来就不是"必须与 AR 逐 token 相同"的认证对象，
v1 已明确 `ACCEPTABLE_FOR_MEASUREMENT` **不**表示 production token 与 AR 相同。

### 0.1 必须先分清**两个历史 B6**（本修订的最关键前提）

审计（见 `docs/specdec_protocol_v2_evidence_inventory_2026-10-04.md` §1）确认历史上存在**两个不同的 exact-id 门**：

| 门 | 条件 | 历史状态 | 说明 |
| --- | --- | --- | --- |
| **`B6-fp32`** | **CPU / float32**，参照 = 同引擎 cached AR | **PASS：20/20 组合逐 token 全等（216/348 token），两次独立复跑一致**（`g5_measurement_protocol.md:268`、`plan_g4_next_steps.md:27`、`handoff_to_analysis_agent.md:65`） | 这是 G5 的"**主 Gate**"；它证明的是**控制逻辑/索引/缓存语义**在无浮点执行形状干扰时**完全精确** |
| **BF16/GPU exact-id 门** | **BF16 + GPU**（`design_freeze_g4_specdec.md:83`：条件 `base_bf16` 与 `v_fp8`） | **`B6-bf16` = CHECK 19/20**：唯一分歧 `natural-2 @7`，经 `g4_b6_tie_diag.py` 归因为 **bf16 数值不可判定位置**（signal 0.1491 ≤ noise 0.1491；fp32 真值落在 spec 侧）；**用户裁决 A = 严格口径：不设例外、不放宽门**，该记录 §4.5.5 原样保留（`plan_g4_next_steps.md:27-28`） | 后续在更敏感的协议下，BF16 production 在 **F13@47 / F02@7 / F08@52 / F10@8** 上与 AR 分歧（`docs/serial_oracle_stage2_2026-10-03.md` §6），仍属同一类"执行形状相关的有限精度数值非不变性"（五类算子已复现） |

**因此"exact-id 门是否要取消"是一个伪命题**：该门**不取消**（`design_freeze_g4_specdec.md:96`：
"**B6 exact-token-id 门不可取消**"）。本修订把它**重新分层**：
* **控制逻辑层**（Gate A，用 serial oracle 这条**与 AR 同路径**的仪器）⇒ exact-id 要求**保留且已通过**（5/5 family）；
* **生产数值行为层**（Gate B）⇒ exact-id **从来不是**该项的判据（v1 §2.2 已如此规定），
  分歧**记录**并分类为 `NUMERICAL_DIVERGENCE_RECORDED`。
* 换言之：`B6-fp32` 的 PASS 与本轮 serial oracle 的 PASS **互相印证**（都在"无块形状干扰"的路径上得到 exact ids），
  而 BF16 production 的分歧属于**另一层**问题。

---

## 1. 与 v1 的**逐项对照表**

| # | v1 条款（`post_shape_localization_v1`） | v2 处置 | 说明 |
| --- | --- | --- | --- |
| 1 | §1.2 diagnostic oracle = **全层 row-aligned**（`set_verify_wrap(row_aligned_verify)`） | **替换** | v2 的 Gate A 仪器 = **serial oracle**（逐 token `M=1`）。row-aligned 保留为**历史诊断**，不再承担 Gate A |
| 2 | §1.1 production = 正式性能实验的**唯一**模式；row-aligned 不启用 | **保留（不变）** | 正式计时仍只用 production；serial 亦**不得**进入计时 |
| 3 | §1.2 诊断模式 `diagnostic_only=true`、互斥、作用域仅 target verify | **保留并加固** | v2 增加**调用几何硬审计**（`s==1`、0 次 `s>1`）+ **D6 复位证据** |
| 4 | §2.1 Gate A 第 1 条：完整 token IDs 与 AR 逐 token 一致 | **保留（要求不变，仪器更强）** | 见 §2 GA1 |
| 5 | §2.1 Gate A 第 2–7 条：accept/reject·replacement·rollback 不变量、cache 结构、finite、合法 token、无 `other_short`、两次独立进程一致 | **保留** | 见 §2 GA4–GA9；并增加"逐轮 trace 完整性" |
| 6 | §2.1 Gate A 第 8 条：diagnostic hook 只作用于 target verify（`applied == target_block_calls`） | **替换为更强的几何判据** | serial 下不存在"applied 计数"；改为**逐次调用的 `seq_len` 审计 + 记账恒等式**（GA2、GA3） |
| 7 | §2.1 末句"**不要求** production mode 的 logits 与 AR 逐位一致" | **保留** | 并在 Gate B/GB6 中**要求记录**分歧细节 |
| 8 | §2.1 末句：任一失败 = `ALGORITHMIC_GATE_FAIL`（硬失败、停） | **保留** | v2 词表见 §5；`ALGORITHMIC_GATE_FAIL` 历史判决**不改写** |
| 9 | §2.2 Gate B：finite / cache / round invariants / 两次重复一致 / 合法停止 / 长度 / 无结构失败 | **保留** | 见 §3 GB1–GB5 |
| 10 | §2.2 Gate B"记录但不自动豁免"：token 一致、common prefix、first divergence、两侧 token、top-1/top-2、margin、hash | **保留并加固为 GB6** | 明确**必须**用现有的 `correctness_protocol_margins.py` 口径提取，且不得标 harmless/error |
| 11 | §2.2 分类 `NUMERICAL_DIVERGENCE_RECORDED` | **保留（原样）** | 该词条继续有效，作为 Gate B 的分歧分类 |
| 12 | §3 预登记 4 个 family（F01/F02/F08/F10，每 stratum 取最小 id 并排除 F13/F06） | **保留为历史；v2 需**新增预登记（§4） | 5 个 family 的证据**不足以**作普遍结论（§4.1） |
| 13 | §4 固定配置（8192/2048/k4/max_new64/BF16/base_bf16/all-eager/CPU-FP32 RoPE/greedy/串行/独立进程） | **保留（不变）** | v2 不改任何配置 |
| 14 | §5 24 次矩阵（4 family × 3 mode × 2 rep） | **保留为 v1 历史矩阵**；v2 另给最小矩阵（§6） | 不重跑 production/row-aligned 已有产物 |
| 15 | §6 每次运行必须记录的字段 | **保留 + 新增** | 新增：调用几何审计、逐轮 trace、D6 复位读数、`target_gen_calls` 与记账恒等式 |
| 16 | §7 决策词表 `READY_FOR_FORMAL_MEASUREMENT` / `ALGORITHMIC_GATE_FAIL` / `CHECK_NEEDS_REVIEW` | **扩展** | v2 增加 `READY_FOR_FORMAL_MEASUREMENT_WITH_RECORDED_DIVERGENCE`（§5.2），其余保留 |
| 17 | §7 末段：只有 `READY_FOR_FORMAL_MEASUREMENT` 才允许**建议**启动正式实验；本轮不得自行启动 | **保留（不变）** | 本轮**不运行 GPU**、不启动 B6/pilot/factorial |
| 18 | §8 profiler 收尾（`PROFILER_UNAVAILABLE` 不阻塞；profiler 时间不进入正式结果） | **保留（不变）** | — |
| 19 | §9 实现纪律（最小修改、默认关闭、不覆盖历史 JSON、不放宽结构检查、不换样本、不用全模型 FP32、不缩短 `max_new`） | **保留（不变，全部继续有效）** | v2 遵守同一纪律 |
| 20 | §10 冻结签名 | **新版**（本文件 §8） | v1 签名与其 FAIL 记录保持原样 |
| 21 | —（v1 无） | **新增 Gate C** | 结果解释与质量（§4）：两条路径轨迹不同时，接受率/轮数/速度各自归属；质量沿用冻结数据与指标 |
| 22 | **上层冻结判据**：`design_freeze_g4_specdec.md:16/83/96`（`SpecDec ids == target-only greedy ids`、条件 `base_bf16` **与** `v_fp8`、**"B6 exact-token-id 门不可取消"**） | **不取消**，改为**分层兑现**（见 §0.1） | 控制逻辑层保留 exact-id（Gate A 已通过 5 family）；生产层以记录型 Gate B 承接。**该文件自身状态含糊**（自称 `DRAFT v0.3`、批准栏未勾选，却被多处以"权威判据"引用）⇒ 见 §9 D-8 |
| 23 | **上层冻结判据**：`g5_measurement_protocol.md:268/327`（主 Gate = FP32 B6；Pilot 进入条件）；`:159-164` **speedup 声明资格 10 条** | **保留（不变）**，并作为本协议的依据 | 关键事实：speedup 声明的 10 条资格里**不含** `ar_equal`（只要求输入 token hash 一致 + `tokens_finite` + 行内无 profiling 数据）⇒ **"不同 token 也能测速度"在冻结测量协议中本来就成立** |
| 24 | `production_correctness_decision_2026-09-29.md:54-60`（"保留的硬要求"清单 + "**不再隐含承诺**：生产 BF16 轨迹与某条 AR 数值实现处处完全一致" + "若输出不同，不得称为'完全相同输出下的无损加速'…不同轨迹可能改变接受率及工作量，必须视为解释限制"） | **采纳为 v2 的 Gate B / Gate C 依据** | 该文本身就是"路线 B"建议稿（状态：待小组批准）⇒ v2 是它的**形式化 + 仪器替换**，不是新发明 |
| 25 | `g5_measurement_protocol.md:270-275`（`BF16_NEAR_TIE` 分类器：4 条结构条件，不满足任一 ⇒ `NEEDS_INSPECTION`，不得豁免） | **保留并在 GB6 引用** | 分歧分类沿用**既有**分类器，不另造口径 |

---

## 2. Gate A：**控制逻辑正确性**（仪器 = serial target oracle）

**仪器定义**：`spec_dec.set_target_verify_mode("serial")`；不安装 row-aligned wrapper；
`SO_MODE=serial`（runner：`specdec/engine/serial_oracle_gpu.py`）；一次运行一个 family/配置/重复。

| 条款 | 要求 |
| --- | --- |
| **GA1** | **完整** generated token IDs 与**同配置 target-only AR** 逐 token 完全一致（不是前缀、不是 hash 相等即止；hash 与逐 token 两者都要核） |
| **GA2** | **oracle 有效性（硬审计）**：target 的**每一次** generation 前向 `seq_len == 1`；`s>1` 的 generation 前向**恰好 0 次**；分块 prefill（`s=2048`）必须被正确归为 prefill 而**不**算作 generation |
| **GA3** | **审计非空 + 记账恒等式**：审计捕获到的 target generation 调用数 `== stats.proposed + stats.rollbacks`；target/draft 必须按**实例身份**区分（不得按 head_dim/调用顺序猜） |
| **GA4** | **逐轮控制 trace 完整**：每轮的 `pre`、`Kp`、proposals（cands）、target predictions（`g`）、接受数 `m`、分支（全接受/mismatch）、`emitted`、replacement 或 bonus、target/draft cache before-after；serial 另记**每步** cache 长度与逐步 argmax |
| **GA5** | **cache 与轮不变量**：`tgt_len == drf_len == pre + (m+1 if m<Kp else Kp)`；逐层 cache 结构与 dtype 一致（`spec_cache.check_consistent`）；被拒候选**不残留**；`cache_len == prompt_len + len(gen)` |
| **GA6** | **finite**：prefill/verify/repair 的决策用 logits 全部有限；覆盖计数非零、`first_bad` 为空、逐阶段全 ok、harness 判定通过 |
| **GA7** | **合法 token 与停止原因**：`gen_ids_in_vocab`；stop ∈ {`max_new_reached`, `eos`}；**`other_short` 判失败**；生成长度符合冻结规则（本协议 `64`） |
| **GA8** | **无 OOM / 无结构失败 / 无异常**（rc=0，含 traceback 为空） |
| **GA9** | **独立重复**：**两个独立进程**的完整 token IDs（及 hash）一致 |
| **GA10** | **作用域**：`diagnostic_only == true`；row-aligned **未**启用；serial 只作用于 target verify（AR/prefill/draft 不受影响） |
| **GA11** | **杀开关证据（D6 回归）**：运行结束**清理完成后**必须**实际读取**并记录 verify 模式已恢复为 `block`；**不得**根据其它运行的旧字段事后推断。缺该读数 = 该项**未验证** |

**任一条失败 ⇒ 硬失败**：分类 **`ALGORITHMIC_GATE_FAIL`**，保留产物、**停止**、不进入 Gate B 之后的速度测量；
**不得**用 margin/ULP 自动豁免，**不得**转入逐层算子定位（本轮/下一轮均不做）。

### 2.1 Gate A 的**现有证据**（只读引用；不是普遍性证明）

| family | AR hash | serial == AR | 两次独立一致 | 几何（`s=1`/block 前向/记账） | 证据位置 |
| --- | --- | --- | --- | --- | --- |
| F13 | `6e7522bf2521ad7b` | ✅ 64/64 | ✅ | ✅ 115 / **0** / ✅ | `serial_oracle_stage1_f13_2026-10-03.md` |
| F01 | `5645f0d6943dcb23` | ✅ 64/64 | ✅ | ✅ 88 / **0** / ✅ | `serial_oracle_stage2_2026-10-03.md` |
| F02 | `3889082e6a0f4594` | ✅ 64/64 | ✅ | ✅ 141 / **0** / ✅ | 同上 |
| F08 | `4f4019d300256c2d` | ✅ 64/64 | ✅ | ✅ 88 / **0** / ✅ | 同上 |
| F10 | `79ca5c727a671e9d` | ✅ 64/64 | ✅ | ✅ 120 / **0** / ✅ | 同上 |

* 这是**五个 8192-token 窗口、各 64 token、BF16 base KV** 上的**实证支持**；
  **不是**"所有输入正确"的形式化证明，也**不是** fp8 KV 或其它长度上的证据（见 §4.3、§9 D-6）。
* 五个 family 的 stratum 覆盖：F08(S1_low_entropy)、F02+F13(S2_high_markup)、F10(S3_prose)、F01(S4_mixed)
  ⇒ **4 个 stratum 全覆盖**，但每个 stratum 只有 1–2 个窗口。

---

## 3. Gate B：**production 运行正确性**（仪器 = 原始 batched verify）

**仪器定义**：`SO_MODE=prod`（= 生产 `spec`），原始 `M=4` block verify，row-aligned 关闭。

| 条款 | 要求 |
| --- | --- |
| **GB1** | **确定性重复**：两个独立进程的完整 token IDs（及 hash）一致 |
| **GB2** | **finite**（同 GA6 口径） |
| **GB3** | **cache 与逐轮结构不变量**（同 GA5 口径），且不再出现"被拒候选残留" |
| **GB4** | **合法 token、合法停止原因、生成长度**（同 GA7 口径） |
| **GB5** | **无 OOM / 无结构失败**；几何上确认走的是**production batched** 路径（generation 里**出现** `s=Kp`）且 row-aligned **未**启用 |
| **GB6** | **分歧记账（强制执行，不自动解释）**：当 production 与 AR 的 token 不一致时，**必须**记录：首个分歧 index、完整共同前缀长度、两侧 token、决策位置（绝对位置 + 轮号）、该位置 top-1/top-2 取值与 margin、并列计数、该位置 cache 长度、以及两侧的完整生成轨迹；分类 **`NUMERICAL_DIVERGENCE_RECORDED`**。<br>**不得**自动写成 `PASS`/`FAIL`/`harmless`/`bug`；**不得**仅凭不一致就判定控制逻辑错误（Gate A 才是判据）；**不得**只挑 margin 最小的案例代表总体（见 GC2） |

**GB6 的字段已有现成实现**：`specdec/engine/correctness_protocol_margins.py`（产出
`div8209_proto_margins_20261003T120234Z.json` 形态的 per-family 记录：`position`、`ar_top1/top2`、
`margin`、`tied_at_max`、`ar_token/prod_token`、`cache_len_after`、family provenance + token sha 校验）。
**分类沿用既有分类器**：`g5_measurement_protocol.md:270-275` 的 `BF16_NEAR_TIE`（4 条结构条件；
不满足任一 ⇒ `NEEDS_INSPECTION`，**不得豁免**）；v2 **不**另造分类口径。

---

## 4. Gate C：**结果解释与质量**

| 条款 | 要求 |
| --- | --- |
| **GC1** | 两条路径（production 与 AR）**可能生成不同文本**：一旦分歧，**接受率、轮数、每轮工作量、速度**都属于**各自轨迹**；任何比较**必须同时报告**这些差异，不得把两者当作"同一计算的不同速度" |
| **GC2** | **质量评估沿用已冻结的数据集与指标**（按 proposal 与既有质量检查文档；本轮不新增、不替换指标）；**不得**用少量 near-tie 案例代替总体质量结论；margin 分布必须**整体**呈现，不得挑选 |
| **GC3** | 正式报告**同时**呈现：① production 路径的**分歧率**（多少条件分歧、首个分歧 index 分布）② 质量结果 ③ 速度结果与其轨迹归属 |
| **GC4** | **速度比口径**：分母 = **同 KV 表示的 target-only AR**（proposal 冻结口径，`g5_measurement_protocol.md:30-32`）；分子 = **production** SpecDec。若两者轨迹分歧，速度必须标注"在分歧轨迹上的比较"，并给出**共同前缀长度**与两侧接受统计；不得声称"same-work speedup"（依据 `production_correctness_decision_2026-09-29.md:60`："若输出不同，不得称为'完全相同输出下的无损加速'…必须视为解释限制"） |
| **GC5** | **serial / row-aligned 的耗时一律不进入任何速度/吞吐/speedup 结论**（v1 §1.2/§6 + `g5_measurement_protocol.md:279-281`"正式计时：无任何插桩；成本分解数字不得写入正式主表"）。本机实测 serial ≈ 53–72 s/次 vs AR ≈ 8.4 s/次，若误用会得到荒谬结论 |

---

## 5. 决策词表与启动条件

### 5.1 判决词表（Gate A/B 组合）

| 条件 | 判决 |
| --- | --- |
| Gate A 预登记 family **全部**通过（GA1–GA11）**且** Gate B（GB1–GB5）通过**且** GB6 分歧已记账 | `READY_FOR_FORMAL_MEASUREMENT`（production 与 AR **逐 token 相同**时）或 `READY_FOR_FORMAL_MEASUREMENT_WITH_RECORDED_DIVERGENCE`（存在已记账分歧时） |
| 任一 family 未通过 Gate A | `ALGORITHMIC_GATE_FAIL`（保留产物、停止；历史判决不改写） |
| 任一 family 未通过 Gate B（GB1–GB5） | `PRODUCTION_RUNTIME_FAIL`（保留产物、停止） |
| 环境不合格 / 字段缺失 / 审计空泛 / **D6 复位读数缺失** / family 超出已登记范围 / 工作树不 clean | `CHECK_NEEDS_REVIEW` |

### 5.2 逐问回答（本轮任务书要求）

**Q1：Gate A 已有五个 family 的证据是否足够，还是需要新增预登记样本？**
* **作为"这 5 个窗口上的证据"：足够**（5/5 通过，含 4 个 stratum 全覆盖、两次独立重复）。
* **作为协议的普遍性基础：不够**。协议 v2 要支持"控制逻辑正确"这一**系统级**陈述，样本必须**事先**登记。
  ⇒ 建议**新增预登记 4 个 family**（沿用 v1 的选择规则：每 stratum 取**未用过的最小 id**、
  排除 F13/F06）：**F09(S1)、F03(S2)、F11(S3)、F04(S4)**，使 Gate A 覆盖 **9 个 family**。
  这样每个 stratum 至少 2 个窗口，且包含**从未参与任何定位工作**的窗口。
* 若不新增：协议必须**明确**把结论限定为"这 5 个窗口"，并**不得**在 proposal 中声称一般正确性。

**Q2：Gate B 出现**可重复**的 token 分歧时，什么条件下允许继续测性能？**
* 前提：Gate A 在同一配置/ family 集上**已通过**（控制逻辑无问题），且 GB1–GB5 全部通过
  （两次重复一致、finite、结构不变量、合法停止、无 OOM）。
* **允许**继续测性能，但必须同时满足：① GB6 分歧已完整记账；② 共同前缀长度与两侧接受统计已报告；
  ③ 速度数按 GC4 标注为"分歧轨迹上的比较"。
* **分歧解释规则（2026-10-04 修订：删除原"`首个分歧 index == 0` 自动阻断"阈值）**。
  原阈值**已删除**：它没有实测依据（仓库内没有任何数据支持 index=0 与 ≥1 必须区别对待），
  且一旦看到结果就容易被围绕解释。改为**不按分歧位置筛选样本**的分层规则：
  * **DR1（记录完整性）**：production 与 AR 的**每一处**分歧都必须按 GB6 记账——首个分歧 index、
    完整共同前缀长度 P、两侧 token、决策位置（绝对位置 + 轮号）、该位置 top-1/top-2 与 margin、
    并列计数、该位置 cache 长度，以及**两侧完整输出与实际生成长度**、接受率、轮数、target 调用数；
  * **DR2（轨迹归属）**：速度比**一律**标注为"**分歧轨迹上的比较**"（GC4），并同时报告 P、两侧接受统计
    与轮数；**不得**声称"相同输出下的无损加速"；
  * **DR3（分层解释，不筛样本）**：**所有 family 的结果一律保留并报告**（含分歧与不分歧的），
    按"完全一致 / 早期分歧 / 晚期分歧"**分层叙述**；**不得**因分歧位置而删除、替换或降权任何 family 的
    结果，**不得**因分歧位置设定"是否准入/是否代表"的门槛；
  * **DR4（如需阈值）**：若小组仍希望有数值阈值，只能在**看到下一轮结果之前**预登记，并写明依据与来源。
    **本轮不预登记任何自动阈值。**
  * ⚠️ 上述 DR1–DR4 是**建议**，**待小组批准**（随 v2 一并批准或修改）；未经批准前不得据此对外表述。
  *（说明：v2 §9 的 **D-4** 相应改为"确认采用 DR1–DR4、不设自动阈值"。）*

**Q3：哪些情况必须硬停？**
1. Gate A 任一 family token 不一致（`ALGORITHMIC_GATE_FAIL`）；
2. 任一模式**两次独立重复不一致**（不确定性）；
3. 出现非有限值（logits/cache 任一）；
4. cache / 逐轮结构不变量失败（含被拒候选残留）；
5. 非法停止原因（`other_short`）或非法 token / 长度不符；
6. OOM / 崩溃 / traceback；
7. **审计空泛**（未捕获到 target 调用）或记账恒等式不成立（无法证明 `s==1`）；
8. Gate A 运行**缺 D6 复位读数**；
9. family 未在守卫中登记 / 守卫 FAIL；工作树不 clean；
10. 环境不合格（harness 16 GiB 线未过，或本轮入口门槛未达）——**不加载模型**即停。

**Q4：正式测量使用哪个模式，serial 数据如何与性能结果隔离？**
* **正式测量只用 production**（原始 batched verify）。AR 为 target-only 参照。
* serial（以及 row-aligned）**只**产出正确性证据：产物带 `diagnostic_only=true`；
  **其耗时不得进入任何速度结论**（GC5）；聚合/报告工具把它们与计时表**物理分离**
  （不同 tag、不同 mode 字段、不同判决词表）。
* 本协议**不**授权启动正式测量；只有 §5.1 的 `READY_FOR_FORMAL_MEASUREMENT*` 才允许**建议**启动。

**Q5：production 与 AR 不同轨迹时，速度比如何解释？**
* 分母固定为**同 KV、同窗口、同 `max_new` 的 target-only AR**（proposal 冻结口径）。
* 一旦分歧，二者是**两条不同的解码轨迹**：AR 的接受率/轮数/每轮产出都属于 AR 轨迹，
  production 的属于 production 轨迹。因此该比值应表述为
  "**production SpecDec 相对同配置 AR 的 wall-clock 比，在首个分歧 index = X、共同前缀长度 = P、
  production 接受率 = A_p、AR 步数 = S_a 的条件下测得**"，并同时给出 ③ 分歧率统计（GC3）。
* **不得**声称分歧之后仍是"相同计算更快"；**不得**用 serial 的耗时作为分子或分母参与正式 speedup。

---

## 6. 下一轮**最小验证矩阵**（本轮**不执行**）

目的：把 v2 从"已验证的 5 个 family"变成"**事先登记、可复核**的协议"。**不**测性能、**不**启动正式实验。

| 组 | family（**预登记**） | 模式 × 重复 | 次数 |
| --- | --- | --- | --- |
| **M1** | **F09 / F03 / F11 / F04**（每 stratum 一个未用过的最小 id） | AR×2 + production×2 + serial×2 | **24** |

* 每组内顺序（与 v2 一致，**阶段间停手核对**）：
  `AR×2 → 核对（两次一致 + 与既有 AR 产物/锚点对照）→ production×2 → 核对 → serial×2 → 核对`。
* 预计运行量：**24 次**；按本机实测（AR ≈ 8.4 s、production ≈ 43–48 s、serial ≈ 53–72 s）
  总计约 **15–20 分钟**（另加 4 次环境门开销）。
* 缩减版（若只验证 Gate A）：**16 次** = 4 family × (AR×2 + serial×2)，约 **10 分钟**，
  但会**跳过** Gate B/GB6 在新数据上的演练 —— 不推荐，因为 GB6/GC3 是 v2 新增条款。
* **前置（必须先做，缺一不可）**：① 用户在文档中确认新增 family 的**授权登记**
  （守卫的 `PROTOCOL_ROUND_FAMILIES` 当前只含 F01/F02/F08/F10；`F09/F03/F11/F04` 现在**越界**，
  守卫会 FAIL —— 这是**设计如此**，不得为通过而放宽检查）；② 运行前后各跑一次守卫并 PASS。

### 6.1 准确命令（复制即用；顺序执行，不并行）

```powershell
# 0) 前置：治理守卫 + 环境预检（不加载模型）
.\.venv\Scripts\python.exe -X utf8 -m specdec.engine.verify_family_independence_cpu
.\.venv\Scripts\python.exe -X utf8 -m specdec.engine.serial_oracle_env_check

# 1) 每个 family 依次：AR×2 -> production×2 -> serial×2（每步之后先核对再继续）
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

> ⚠️ `div8209_p2` 是**建议的新 tag**（不覆盖任何历史产物）；`F09/F03/F11/F04` 的登记需先获授权。
> 生成 ID / family token sha 一律由 `family_input` 重建并校验，校验失败即拒绝、不降级。

---

## 7. 已冻结判据的**实际改动清单**（透明化）

| 项 | v1（2026-10-03） | v2 草案 | 是否降低门槛 |
| --- | --- | --- | --- |
| Gate A oracle | row-aligned（`M=4` 内取第 0 行） | **serial（逐 token `M=1`，与 AR 同路径）** | **否**（oracle 更真；token 一致性要求不变） |
| Gate A 几何证据 | hook `applied == target_block_calls` | 逐次 `seq_len` 硬审计 + 记账恒等式 + 0 次 `s>1` | **否**（更强） |
| 杀开关 | 未要求 | GA11 要求**清理后实测** `block` | **否**（新增要求） |
| Gate B | 记录型（不自动豁免） | 保留 + GB6 强制记账字段与分类 | **否**（更明确） |
| Gate C | 无 | 新增（轨迹归属、质量口径、speedup 解释） | **否**（新增约束） |
| 判决词表 | 3 个 | 5 个（+`WITH_RECORDED_DIVERGENCE`、+`PRODUCTION_RUNTIME_FAIL`） | **否** |
| 历史结论 | `ALGORITHMIC_GATE_FAIL`（row-aligned 1/4） | **原样保留、不改写**、不重新解释 | — |

---

## 8. 冻结签名（机器可读快照，**待决定后才生效**）

* 协议版本：`layered_v2_serial_oracle`
* Gate A 仪器：`serial`（`spec_dec.set_target_verify_mode("serial")`；`_verify` 内部唯一分派点）
* Gate A 条款：`GA1..GA11`；Gate B：`GB1..GB6`；Gate C：`GC1..GC5`
* 已登记 family（Gate A 现有证据）：`["F13","F01","F02","F08","F10"]`
* 拟新增预登记 family：`["F09","F03","F11","F04"]`（**待授权**）
* 正式计时模式：`prod`（production）；`serial` / `row-aligned` **排除**
* 决策空间：`READY_FOR_FORMAL_MEASUREMENT` / `READY_FOR_FORMAL_MEASUREMENT_WITH_RECORDED_DIVERGENCE` /
  `ALGORITHMIC_GATE_FAIL` / `PRODUCTION_RUNTIME_FAIL` / `CHECK_NEEDS_REVIEW`
* 状态：**DRAFT（未冻结）** —— 见 §9。

---

## 9. 冻结前**真正需要决定**的事项（判决：`NEEDS_DECISION`）

| # | 事项 | 为什么必须由你/导师定（而不是我自行冻结） |
| --- | --- | --- |
| **D-1** | **批准"路线 B + serial oracle 作为 Gate A 仪器"**：即把 BF16 下 `SpecDec ids == AR ids` 的**兑现层次**从"生产 `M=4` 路径"改为"控制逻辑层（serial，与 AR 同路径）"，生产分歧改为**记录不设门** | `design_freeze_g4_specdec.md:16/83/96` 与 `proposal_freeze_2026-10-01.md:31` 是**上层冻结/已提交**判据；该文件明文"**B6 exact-token-id 门不可取消**"。本修订**不取消**该门（Gate A 保留 exact-id 并通过 5 family），但**改变了兑现该门所要求的执行路径** —— 这一解释**必须**由你/导师确认。注意：`docs/production_correctness_decision_2026-09-29.md` 已起草同方向建议（状态：**待小组批准**），故本项更准确地说是"**批准既有建议稿**" |
| **D-2** | Gate A 是否把样本扩到 **9 个 family**（新增 F09/F03/F11/F04），还是把结论**限定**在现有 5 个窗口 | 决定"控制逻辑正确"是**系统级陈述**还是**5 窗口的实证**；影响报告措辞与下一轮工作量 |
| **D-3** | 授权把新增 family 登记进**守卫**的授权范围（`PROTOCOL_ROUND_FAMILIES` 或新建具名集合） | 守卫当前会把 F09/F03/F11/F04 判为**越界**（这是设计）。登记是**授权行为**，须由你决定；只增检查、不放宽 |
| **D-4** | 确认采用 **DR1–DR4**（分歧**全量记录 + 分层解释**，**不设**任何"首个分歧 index"自动阈值） | **2026-10-04 修订**：原 `index == 0 ⇒ CHECK_NEEDS_REVIEW` 阈值已删除（无实测依据、诱导事后选择）。新的规则**不按分歧位置筛选样本**；如小组仍要阈值，只能在看到结果**之前**另行预登记 |
| **D-5** | 质量评估的**冻结数据集与指标**（Wikitext-2、非重叠窗口、`ΔPPL` 等既有口径）具体执行清单，以及是否要求**两条轨迹都评质量** | Gate C 引用"已冻结数据集与指标"，但执行清单需你确认；据 `production_correctness_decision_2026-09-29.md:103`，"标准 AR PPL 不能直接当作 SpecDec 生成质量"，故两条轨迹的质量口径需要明确 |
| **D-6** | **fp8 条件是否也要各自的 Gate A 证据** | `design_freeze_g4_specdec.md:83` 明文把 B6 的条件写成 **`base_bf16` 与 `v_fp8`**；`proposal_freeze:31` 要求"按各 KV 表示分别验证"。而现有 Gate A 证据**只有 BF16 base KV**。若要求分别验证 ⇒ 需要在 `v_fp8` 下**另跑** Gate A（并先确认 serial 路径在 fp8 KV 下定义良好）——**新工作量**，必须由你定 |
| **D-7** | 是否接受 v2 的**新判决词条**（`READY_FOR_FORMAL_MEASUREMENT_WITH_RECORDED_DIVERGENCE`、`PRODUCTION_RUNTIME_FAIL`）纳入正式口径 | 词表变更会影响与导师沟通的"通过/未通过"表述 |
| **D-8** | 澄清 `docs/design_freeze_g4_specdec.md` 的**状态**：它是"已批准冻结判据"还是"`DRAFT v0.3`（批准栏未勾选）" | 审计显示该文件自称 `DRAFT v0.3`、批准栏未勾选，却被 `handoff_to_analysis_agent.md:153` 与多份日报当作**权威/冻结判据**引用。这决定 D-1 是"改冻结判据"还是"正式批准一份草案" |
| **D-9** | 是否需要把 v2 的 Gate A 证据补到 **FP32** 口径（`B6-fp32` 已是 PASS 20/20），或明确"`B6-fp32` 与 BF16-serial 两条证据互为印证、不再重复" | 避免重复劳动，同时保证"exact-id 门"在两条路径上都有据可查 |

**在 D-1…D-7 决定之前，本文件保持 DRAFT**；我不会把它标成 FROZEN，也不会据此启动任何正式实验。
