# BF16 主实验正式准入盘点 + 实现规格（2026-10-05，基于 `95705c7`）

> 本文件**只整理协议、证据与交接**：不写正式 runner、不新增 GPU 运行、不重跑既有矩阵、不追加 FP32/margin/k 扫描/质量研究。
> 正式 runner、采集、统计分析、绘图、final pipeline 由用户本人编写。
> B6 侧的状态与证据索引见 `docs/b6_status_and_evidence_index_2026-10-05.md`。

## 1. BF16 主实验准入表

口径：**已满足**（有冻结条文或实测证据，可直接用）/ **待完成**（已定做法、缺执行）/ **待决策**（须你或小组定）。
"依据"列给出唯一来源；**本轮新增**的发现单独标注。

| # | 项目 | 状态 | 依据 / 说明 |
| --- | --- | --- | --- |
| 1 | **候选配置取值**：target `Qwen/Qwen2.5-1.5B-Instruct`、draft `Qwen/Qwen2.5-0.5B-Instruct`、BF16 compute、target KV `base_bf16`、draft KV 恒 BF16 未量化、k=4、chunk=2048、max_new=64、贪心、冻结 EOS 规则（`other_short` 判失败）、AR target-only、RoPE=CPU FP32、一条件一进程 | **已满足** | `proposal_freeze_2026-10-01.md`（FROZEN，PDF/TEX sha256 已记录）；`specdec_correctness_protocol_v2…§1` 第 13 行（保留不变）。**本轮格式核对**：与用例无关的部分（`arm`/`k`/`chunk`/`kv`/`dtypes`/`target_rope`/EOS id）在 66 次运行产物中**逐字段相符**；`max_new` 按**预先声明**取值（B6 覆盖轮 16/13/5/2 由 10 用例表声明，F09 复核 64，EOS 补测 24）——**非**运行时改动，正式矩阵一律 64 |
| 1b | **候选配置对应的"冻结 commit"** | **待完成**（小项） | 现状：每次运行自带 `git_commit`/`dirty_worktree`（可审计），但协议冻结签名里**没有**具名冻结 commit ⇒ 建议在签名中加一行 `frozen_config_commit=<sha>` |
| 1c | **最终 attention backend / 阶段策略** | **待决策**（D-16） | `proposal_freeze` 明文"最终 attention backend/阶段策略仍须通过正确性和运行验证"；候选 = `cudnn_prefill`（仅 prefill 融合、其余 eager），已有 **B6 级**证据（`div8209_b6` 66 runs + `F09` 融合复核 + `div8209_eos` 6 runs）。**本轮新增**：候选后端在 8K 双驻留下 `pt_peak_reserved=9114 MiB > 8187.5 MiB` 物理、`cuda_free=0` ⇒ **并未把 8K 拉回物理线内** |
| 1d | **若最终 backend ≠ eager ⇒ 重跑 40 窗口质量检查** | **待完成**（依赖 1c） | `proposal_candidate_rev2_changelog…§4-2`；工具与口径**已存在**（`nll_quality_*.json`，2026-09-26），只需按冻结配置重跑一次 |
| 2 | **六条件 C1–C6 = ctx{2K,4K,8K} × {AR,Spec}，target KV=base_bf16** | **已满足（设计 + 数据）** | `proposal_freeze`（主实验 6 条件）；`docs/prompt_families_manifest.json` 的 `prefixes` 含 **2048/4096/8192** ⇒ 三个 ctx 的数据齐全 |
| 2b | **8K 条件的"可测性"** | **待决策**（D-17） | **本轮新增的重要发现**：`docs/data_ledger.md` §4.20 实测同配置三样本 decode **10.04 / 11.18 / 16.22 s ⇒ 极差 61%**（≫20% 诊断线）、`peak_reserved` 10.5–10.7 GiB > 8 GiB 物理、`cuda_free=0`、同进程 warm0→rep0 退化 **+59%/+52%** ⇒ 按 `g5_measurement_protocol §7`，8K SpecDec 单元只能标 `CHECK`、`speedup_claim_allowed=false` ⇒ **C5/C6 可能拿不到可声明速度比**。ledger §4.20 建议：8K 降级为"受限条件"（只登记内存证据 + 计时不可复现），主战场转 2K/4K；或投 SDPA/FlashAttention 结构性手段（未验证） |
| 2c | **2K/4K 在正式配置下的计时稳定性** | **待完成**（有界） | 历史稳定性探针（12 条件 × 2 随机 block）显示 AR 侧 3.7% OK、8K SpecDec 28.9% CHECK（ledger §4.14/§4.20）；2K/4K 尚未在**正式配置**下核过 ⇒ 建议一次有界稳定性检查（见 §4 D-16） |
| 3 | **AR = target-only；Spec = 双模型** | **已满足** | `g5_measurement_protocol §3.1–§3.5`（AR-primary 为主基线且唯一可作分母；dual-resident 仅诊断、不入 factorial、`speedup_claim_allowed=false`；`results_schema.plan_process()` 强制进程隔离）；本轮产物 `kv_format.draft = not_loaded`(AR) / `fp16`(Spec) 实测一致 |
| 4 | **模型/tokenizer/dtype/backend/chunk/k/EOS/max_new 固定** | 部分（见 1c） | 模型/tokenizer（`family_input.TOKENIZER_ID`、`EOS_ID_EXPECTED=151645`）/dtype/**max_new=64**/chunk/k/EOS 规则 **已冻结且本轮实测一致**（`max_new` 在本轮覆盖轮/复核/补测中按**预先声明的用例预算**取值，见第 1 行说明）；**backend 待决策** |
| 5 | **prompt manifest、已消费 family、未使用 family** | **已满足（清单可核）** | manifest `v1-draft`：**16** family、4 stratum × 4、`prefixes` 支持 2048/4096/8192。**已消费 10**：F13（定位/历史）、F01/F02/F08/F10（协议轮）、F06（定点诊断）、F09/F03/F11/F04（Stage 2b；F09 另用于 8K 融合复核）。**未用 6**：**F05 / F07 / F12 / F14 / F15 / F16**（S4×2、S1×2、S3×1、S2×1） |
| 6 | **4-family pilot 与正式 family 的分离与可用数量** | **待决策**（D-15） | **本轮新增的矛盾**：proposal 冻结表述为"**4-family pilot + ≥16 formal families**"，而 manifest 共 **16** 个 family、其中 **10 个已被消费** ⇒ 若"≥16"指**未使用过**的 family，则**不可满足**（最多 6 个）。三种解读与推荐见 §4 D-15 |
| 7 | **warm-up、重复次数、随机化/分块、独立进程** | **已满足** | §3.5/§4/§5：1 次 warm-up（丢弃）+ **5** 次 measured reps；**≥2 个独立随机顺序 block**（seed 记档可重放）；一条件一全新子进程（父进程只编排；模型加载/tokenizer **不入**计时）；块前记录系统状态；条件间 cooldown 10 s 不计时。⚠️ 与 2b 的交互：8K 饱和条件下"同进程 1 warm-up + 5 reps"会被**状态累积退化**污染（ledger §4.20 实测 +59%），须按 §7b.3/§7c.3 分层判定并允许**整块**重跑（不得只删慢 rep） |
| 8 | **CUDA 计时边界 / 实际生成长度 / 环境门 / 失败处理** | **已满足** | §5（区间两端 `synchronize()`；区间内禁 `empty_cache`/`nvidia-smi`/文件 IO/文本 decode/任何诊断；cooldown 不计时）；实际生成长度必须如实报告；§7b（P-state 预检须**负载期**采样：SM≥2000 / mem≥7500 MHz、5 s、功耗仅辅助）+ harness 16 GiB 线 + 入口门槛；§7/§7b/§7c 状态分层 + `validate_canonical` 写盘前校验 + v2 §5.2 Q3 十类硬停 |
| 8b | **8K 跨顺序方差探针的登记** | **待完成**（小项，本轮新发现） | 探针**已跑过**（`output/g5/variance/p5v_*`，含两份 117 KB summary；ledger §4.18–§4.20 已定位"8K SpecDec 噪声量级最高 61%"），但 `g5_measurement_protocol §11` 第 3 项在文档里**仍标 ⬜** ⇒ 需把该结论**统一登记**进协议（并给出 2K/4K 的有界检查），避免"已测过却仍被当作前置未做" |
| 9 | **独立插桩与主计时分离** | **已满足** | §9：正式计时**无任何插桩**；成本分解用外部 CUDA Event 且数字**不得**写入正式主表；`P5_PROFILE=0` 量化插桩开销；`profiling` 带 `subset_phases`；v2 **GC5** 禁止 serial/row-aligned 耗时进入任何速度结论 |
| 10 | **输出序列 / 分歧 / 接受率 / 调用量 / 内存 / 配置记录** | **已满足（口径）+ 待完成（在你的 runner 中启用并验证）** | 口径：§2 运行身份字段；§6 三种显存口径（`peak_allocated` 主、`peak_reserved` 辅、`nvidia_smi_used` 物理参考）；§7c 映射/`availability` 纯函数/跨进程等价性由父进程填（缺对照写 `null`，**绝不写 False**）；§7c.5 十二条写盘前校验；GB6 分歧字段全清单（首分歧 index、共同前缀 P、两侧 token、决策位置 top-1/top-2 与 margin、并列计数、该位置 cache 长度、两侧完整输出与实际长度、接受率、轮数、target 调用数）。**能力在工具链中已具备**；按 `proposal_candidate_rev2…修订 3`，正式实验须**实际启用并逐项验证**这些字段 |
| 11 | **正式配置下的质量核验** | 部分（见 11a–11d） | **11a 已有**：FP8 V-only 质量（NLL **+0.000397**、relative PPL **+0.040%**，40 个非重叠 2048-token 窗口、chunk 512）与显存（KV 224.00→169.75 MiB，**1.32×**；allocated peak −0.06 GiB）——产物 `output/g5/attn/nll_quality_*.json`（2026-09-26）。**11b 待完成**：若 backend≠eager ⇒ 重跑该 40 窗口检查（依赖 1c）。**11c 待决策（D-5 剩余）**：是否**两条轨迹都评**、以及生成质量口径（PPL ≠ 生成质量，见 `production_correctness_decision:103`）。**11d 尚无**：正式配置（`cudnn_prefill`）下**任何**质量数据 |

### 1.1 不由"未完成的研究诊断"阻塞正式实验的项（逐条说明依据）

| 项 | 为何**不**作为 BF16 主实验的强制阻塞 | 依据 |
| --- | --- | --- |
| P-1/P-2 决策行**双侧**捕获 | 只影响**分歧性质的解释**（TIE 分 A/B/C），不影响 Gate A/B 的通过或速度测量资格；缺双侧证据时按既有规则记 **TIE-C（证据不足）**，不自动豁免、不自动归因 | 批准记录 §1A（"默认阻断"）；`docs/protocol_decision_brief…§4` |
| `qk_scope` 相关的诊断交互（QK FP32 × serial-QK） | 正式入口要求**相关诊断干预关闭**（`qk_fp32=prob_fp32=False`、未安装 serial-QK/row-aligned）；本轮 66+6 次运行实测均为干净状态 ⇒ 该交互在正式数据里**不可达** | 本轮任务书 §二；`docs/b6_eos_qk_closeout…§2.5/§8`；产物 `diag_precision_*` 实测 |
| 五类算子的形状敏感性定位（GEMM/attention 机制研究） | **机制解释**已完成到"特定同历史样本上的因果实验"；proposal rev2 已明确"**不**声称对所有 production 分歧做算子级定位" ⇒ 属**解释边界**，不是准入前提 | `proposal_candidate_rev2_changelog…修订 1` |
| `v_fp8` 的 Gate A / 质量 | 按 **A4（per-KV）**：FP8 未完成**不**阻塞 BF16 主线；但 FP8 档（A1/A2）**不得**在自身门通过前开跑 | 本轮批准 A4；批准记录 D-6 |
| k 扫描 / margin 诊断 / 额外 FP32 | 均属**诊断性**扩展，正式矩阵只需冻结的 `k=4`；FP32 口径已有历史 PASS（E1）且与 BF16-serial 互为印证 ⇒ 不重复 | 批准记录 D-9；`g5_measurement_protocol §1/§10` |

## 2. 正式实验代码**实现规格**（只描述输入/输出/流程/计时边界/验收；**不提供代码**）

### 2.1 输入（runner 的输入面）

| 类别 | 内容 |
| --- | --- |
| 条件 | `C1–C6 = ctx{2048,4096,8192} × mode{AR-primary(target-only), SpecDec(production)}`；可选 `A1/A2 = 8K v_fp8`（**须先过 FP8 自己的门**） |
| 数据 | `docs/prompt_families_manifest.json` 的 `families[*].prefixes["<ctx>"]`（token ids；**必须**用 manifest 的 `token_sha1` 在加载时校验，`hash_verified=True`） |
| 分块/重复 | 每条件 **≥2** 个随机顺序 block（`random.Random(seed).shuffle`，`seed = base + block`，记档）；每 (条件, family, block) 跑 **1 warm-up + 5 measured reps** |
| 身份字段 | `run_id`(UTC)、`git_commit`、`dirty_worktree`、`seed`、`block_index`、`condition_index`、`condition_order`、`condition_role` |
| 冻结配置 | 见准入表第 1 行；**backend/阶段策略** 从**单一配置源**读取（冻结时只改这一处） |
| 门 | 守卫（family 授权）→ 环境门（harness 16 GiB 线 + 入口门槛 + 工作树 clean）→ P-state 预检（负载期采样）→ 才允许进入计时 |

### 2.2 输出（每行 + 汇总）

* **逐 (条件, family, block, rep) 一行**，字段按 `g5_measurement_protocol §2/§6/§7c`：
  计时（`prefill_seconds`/`decode_seconds`/`total_seconds`）、吞吐、三种显存口径、
  SpecDec 统计（`accepted`/`proposed`/`rounds`/`draft_calls`/`target_block_calls`/`mismatch_rounds`/`all_accept_rounds`/`acceptance`=恒等式）、
  生成序列（`generated_token_ids` + hash）、**实际生成长度**、停止原因、
  跨进程等价性（`ar_equal`/`ar_equal_available`/`ar_equal_certified`/`equality_source`，**缺对照写 null**）、
  状态分层（`within_run_status`/`between_block_status`/`environment_status`/`measurement_status`）、
  `speedup_claim_allowed` + `speedup_claim_blockers`、`availability`（值的纯函数）。
* **GB6 分歧记录**（production vs AR 每一处）：首分歧 index、共同前缀 P、两侧 token、决策位置（绝对位置 + 轮号）、
  该位置 top-1/top-2 与 margin、并列计数、该位置 cache 长度、两侧完整输出、两侧接受率/轮数/target 调用数。
* **物理分离**：正确性/诊断（tag、`mode` 字段、判决词表）与计时表**不同产物、不同 tag**；serial/row-aligned 永不进入计时表。
* **不覆盖**：目标文件已存在且未显式 `append` ⇒ 直接报错；历史 JSONL 只读。

### 2.3 流程（每条件）

1. 前置门（守卫/环境/P-state）→ 记录系统状态（smi used/temp/power/clocks + CUDA free）。
2. **为该条件新起一个子进程**（AR target-only 与 SpecDec **绝不**同进程；AR 进程**从不加载** draft）。
3. 子进程内：加载模型 + tokenizer（**在计时区间之外**）→ 8K 必须用**分块 prefill**（`chunk=2048`；把 `spec_dec._prefill` 换成
   分块计时版本，**只调用一次** `spec_decode_greedy`，`decode = total − prefill`）→ `finally` 还原所有被替换的函数。
4. **warm-up 1 次并丢弃** → **measured reps 5 次**：每次计时区间 = `synchronize()` …（prefill+decode）… `synchronize()`；
   区间内**禁止** `empty_cache`/`nvidia-smi`/文件写入/文本 decode/诊断轮询。
5. 条件间 **cooldown 10 s**（不计时）；每 block 开始前记录系统状态；block 结束用**计时区间之外**的 monitor 快照做 P-state 审计。
6. 父进程：按 `(block_index, context_actual, target_kv, input_token_hash, max_new_tokens)` 配对 AR↔Spec 填 `ar_equal`；
   `speedup = median(T_AR) / median(T_Spec)`（同 ctx、同 KV、分子 SpecDec、分母 AR-primary），**每个配对 block 都合格**才允许声明。

### 2.4 计时边界（写死）

| 量 | 边界 |
| --- | --- |
| `total_seconds` | 从（模型加载完成、warm-up 结束后的）prefill 之前同步点，到生成结束同步点 |
| `prefill_seconds` | 包装后的分块 prefill 区间（`prefill_source ∈ {reported_median, reported_per_rep}`；**禁止**用 `total−decode` 偷算） |
| `decode_seconds` | `total − prefill`（同一 rep 内相减） |
| **不计入** | 模型/权重加载、tokenizer、warm-up、cooldown、系统查询、任何诊断/插桩 pass |
| 插桩 | 只在**独立 pass**（外部 CUDA Event），数字**不得**进正式主表；`P5_PROFILE=0` 用于量化开销 |

### 2.5 验收规则（runner 必须强制）

1. **每条件（GB1–GB5）**：finite（决策用 logits 全覆盖、`first_bad` 空）；cache/逐轮结构不变量；**两次重复一致**；
   合法停止（`max_new_reached`/`eos`；`other_short` = 失败）；无 OOM/结构失败/异常（rc=0、traceback 空）；长度符合冻结规则。
2. **GB6**：分歧**每一处**完整记账（字段见 §2.2）；**不**自动判 harmless、**不**自动归因浮点误差。
3. **声明资格（`g5_measurement_protocol §7b.4` 十条）**：分母 AR 且 `ar_residency=target_only`；`speedup_denominator_eligible`；
   两侧 `context_actual`/`target_kv`/`max_new_tokens`/`input_token_hash` 一致；reps 完整；两侧 `measurement_status=OK`；
   between-block 未超诊断线；无 OOM/NaN/ENV_INVALID；`tokens_finite=true`；非 dual-control；行内无 profiling 数据。
   任一不满足 ⇒ `speedup_claim_allowed=false` 并**打印具体原因**。
4. **硬停**（v2 §5.2 Q3）：控制逻辑 token 不一致（Gate A）/ 两次重复不一致 / 非有限值 / cache 或逐轮不变量失败 /
   非法停止或非法 token / OOM 或崩溃 / 审计空泛或记账恒等式不成立 / 缺 D6 复位读数 / family 越界或工作树不 clean / 环境不合格。
5. **写盘前自洽校验**（§7c.5 十二条）：任一条不过 ⇒ **拒绝写出**。

## 3. 结果解释纪律（照 proposal 的分层口径，逐条）

1. **serial（以及 row-aligned）只用于控制逻辑验收**，其耗时**不得**进入任何速度/吞吐/speedup 结论（GC5）。
2. **production 保持批量路径**，是正式计时的唯一模式。
3. **AR 与 Spec 的输出可能不同**：一旦分歧，性能比只能解释为"**两条不同轨迹的系统表现**"，
   **不得**称为"同输出无损加速"（`production_correctness_decision:60`）。
4. 报告**实际生成长度**、**首分歧 index**、**共同前缀 P**、**分歧比例**（多少条件分歧）及**各自工作量**（接受率、轮数、target 调用量）。
5. **不**声称分歧之后质量相同；**KV PPL 不能替代生成质量证据**。
6. 有分歧时速度必须标注"**在分歧轨迹上的比较**"并同时给出 P 与两侧接受统计（GC4）。
7. 不稳定单元按 §7 处理：原始数据保留、仍出现在完整表中、报 `median/min/max/MAD/IQR` 与 between-block 差异、
   标 `CHECK` 且 `speedup_claim_allowed=false`、可从显著性结论中排除但**必须注明**、**不得**从报告中消失。

## 4. 真正需要裁决的剩余事项（集中一次 + 推荐）

| # | 事项 | 推荐 | 理由（要点） |
| --- | --- | --- | --- |
| **D-15** | "4-family pilot + **≥16 formal families**"与 manifest 只有 16 个 family（其中 10 个已被消费）的矛盾 | **推荐解读 (iii)**：pilot 取 4 个；正式性能矩阵用**全体 16 个 family**（分层：其中 **6 个从未使用过**的 F05/F07/F12/F14/F15/F16 专用于**分歧率与质量**类结论的干净样本）；报告中逐 family 标注历史使用情况 | 性能与数据内容无关（速度不因"曾用于定位"而变化）；但分歧率/质量类结论必须能区分"干净样本"，否则会被质疑选择性 |
| **D-16** | 最终 attention backend / 阶段策略冻结值（候选 `cudnn_prefill`） | **推荐冻结 `cudnn_prefill`（仅 prefill 融合）**，并按 1d 重跑 40 窗口质量检查一次 | 已有 B6 级正确性证据（66 runs + F09 融合复核 + EOS 补测）；但依赖 1d 的质量重跑才能在正式配置下重新声明质量 |
| **D-17** | **8K 条件的处置**（实测不可测：极差 61%、`peak_reserved` 超物理、`cuda_free=0`、同进程退化 59%） | **推荐 ledger §4.20 路线 1+2**：8K 登记为"**受限条件**"（只报**内存**证据 + "计时不可复现"），主问题用 **2K→4K 的干净曲线 + 8K 的"超限边界"**回答；**不**发布 8K 速度倍数。若要救 8K，另立**预登记**的 SDPA/FlashAttention 后端研究（属新范围，须单独批准，且届时 1c/1d 结论作废需重估） | 依据是本机三样本实测 61% ≫ 20% 诊断线；这不是"未做诊断"，而是**已定位的物理限制**；强行发布 8K 速度会违反协议 §7 的声明纪律 |
| **D-18** | `g5_measurement_protocol §11` 第 3 项的**登记更正** + 2K/4K 稳定性有界检查 | **推荐**：把"8K 方差探针已跑、结论 = 该条件噪声量级最高 61%"登记进 §11（把 ⬜ 改为"已完成，结论为限制"），并**新增**一次 2K/4K 稳定性检查（范围：2K/4K × AR/Spec，各 ≥2 block；仅稳定性，不出速度结论；耗时按 §11 命令估计） | 避免"已完成的工作被当作前置未做"，同时补上 2K/4K 的可测性证据（它们将成为主战场） |
| **D-5（剩余）** | 质量口径：(a) 是否**两条轨迹都评**；(b) 生成质量是否需要独立口径（PPL 不足） | **推荐**：先做 (a) 的**最小版**（把 AR 与 Spec 的**实际输出**在同一 40 个非重叠窗口上各算一次 PPL/ΔPPL，明确标注"**不是**生成质量等价性证据"）；(b) 若要生成质量，另立**预登记**的最小方案（见 §5），不擅自实施 | 沿用冻结数据集与指标，避免新造口径；同时不越界声明质量等同 |
| **D-19（可选）** | 是否把 §2 的实现规格固化为"runner 契约文档"（供你实现时对照） | **推荐**：是（但由你决定形式）；本文件即为契约草案 | 便于你写代码时逐条对照验收规则；不涉及代码生成 |
| **D-20（可选）** | 报告模板（Gate C 的"分歧率 + 质量并列"） | **推荐**：在写 runner 的同时固定模板骨架（表头/字段/分层标签），避免事后选表 | 防止"事后选表"式解释（协议 §7 精神） |

**A4（per-KV 放行）已批准**，故原 D-6/D-12 的"BF16 是否被 FP8 阻塞"问题**已闭合**：FP8 未完成只阻塞 **A1/A2**，不阻塞 **C1–C6**。

## 5. 质量评估：如需新增范围的最小方案（**待你确认，不擅自实施**）

| 方案 | 最小内容 | 成本 | 能/不能声称 |
| --- | --- | --- | --- |
| **Q-min-1（推荐，零新指标）** | 用**既有**工具与口径（40 个非重叠 2048-token 窗口、absolute PPL + relative ΔPPL），对**冻结配置**下的 (a) 参考文本、(b) AR 输出、(c) Spec 输出各算一次 | 1 次质量运行（既有工具，属 1d 的同一批） | 能：冻结配置下 FP8/后端的质量影响；**不能**：生成质量等价性 |
| **Q-min-2（若要生成质量）** | 在 4-family pilot 的输出上，用**预登记**的固定评分表（例如人工/模型判分 rubric × N 条），AR 与 Spec **各评一遍**，并报告分歧样本的对照 | 需先定 rubric（= D-5）与评分流程；人工/模型评分工作量另计 | 能：有限样本上的生成质量对照；**不能**：总体质量结论（样本量不足，GC2） |
| **Q-min-3（不建议现在做）** | 任务级评测（下游任务准确率等） | 远大于 pilot 规模 | 超出 proposal 范围 ⇒ 不建议 |

> 三者**都**不得写成"分歧后质量相同"；KV PPL 不能替代生成质量证据（`production_correctness_decision:103`）。

## 6. 三问回答（逐条，明确）

### Q-a：**现在能否开始写正式实验代码？** —— **能**。

依据：① 测量协议（计时边界、schema、warm-up/reps/blocks、声明资格 10 条、写盘前校验）与 Gate A/B/C 条款**都已成文/冻结**；
② 唯一未定项（最终 backend/阶段策略）在实现上是**一个配置常量**，不改变 runner 结构 ——
只要把 backend 与 `ctx/chunk/k/max_new/kv/families/blocks/reps` 都从**单一配置源**读取，冻结时只改一处；
③ 写代码**不消耗 GPU**、不占用任何待决项，也不会造成"用结果反过来定协议"。
建议顺序：先做**可离线验证**的四件事（dry-run 计划输出、schema 校验、`validate_canonical` 自洽校验、声明门控），
再实现 GPU 路径 —— 这四件决定后面数据能不能用。

### Q-b：**现在能否跑 pilot？** —— **还不能；缺 1 项测量前置 + 2 项裁决**。

* 缺的测量前置：**8K 条件的处置结论**（D-17）与 **2K/4K 稳定性有界检查**（D-18）——
  ledger 已实测 8K 极差 61%，若 pilot 含 8K，pilot 自身就会产出不可声明的数字；
* 缺的裁决：**D-15**（pilot 的 4 个 family 具体是谁、正式集合如何取）与 **D-16/D-17**（backend 冻结 + 8K 处置）；
* **已满足**：环境门、P-state 门、schema、进程隔离、warm-up/reps/blocks、声明纪律、`B6_BF16_PASS`（有限范围，A2）。
⇒ 上述 3 项落定后，pilot 可在**一次有界运行**内开始（命令与规模见 `g5_measurement_protocol §11/§12`，耗时可按既有实测估算）。

### Q-c：**现在能否采集正式数据？** —— **不能**。

* 正式采集需要 `BF16_FORMAL_READY`；按批准记录 **§A3**，`*_FORMAL_READY` 属 **批 2**，**须另行批准**（当前**未批准**）；
* 即便批 2 之外的技术项也仍有缺口：8K 可测性（D-17）、最终 backend 冻结（D-16）+ 其质量重跑（1d）、
  质量口径剩余部分（D-5）、pilot 完成与稳定性前置（D-18）、**环境/预算准入**（运行预算与人时计划）；
* 因此：**不得**开始正式采集；也不得以 `B6_BF16_PASS`（有限范围）作为 `BF16_FORMAL_READY` 的依据。

## 7. 本轮**未做**的事（防止把计划写成完成）

* 未写任何 runner / 采集 / 统计 / 绘图 / final pipeline 代码（所有权属你）；
* 未新增任何 GPU 运行（未跑 pilot、未跑方差探针、未重跑 66 次 B6 或 6 次 EOS）；
* 未追加 FP32 / margin / k 扫描 / 质量研究 / 其它 GPU 诊断；
* 未修改任何门限、冻结判据、历史产物或历史判决；
* 未宣布 `BF16_FORMAL_READY` 或 `FP8_FORMAL_READY`。
