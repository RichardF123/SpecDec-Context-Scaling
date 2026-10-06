# BF16 主实验正式准入盘点 + 无代码实现规格（rev.2 修正版，2026-10-05，基于 `b3cd2cd`）

> **本轮只做只读核对与文档修正**：不运行 GPU、不改生产代码、不写正式 runner、不改 manifest、不消费新 family、不生成正式数据。
> 修正对象 = 本文件 rev.1；六处修正见 §0.2（含"原表述 → 修正后 → 依据"）。
> **交叉核对**：用户自己的阶段记录 `docs/stage_checkpoint_2026-10-06.md`（§仍然存在的限制）独立写出同一组六条口径，
> 与本文件 §0.2 逐条一致；该记录同时注明"正式准入草案仍需修订，不作为已批准的最终执行合同" ⇒ 本 rev.2 即该修订。
> B6 侧状态与证据索引见 `docs/b6_status_and_evidence_index_2026-10-05.md`。

## 0. 保留的已批准结论（不变）

1. **有限范围 `B6_BF16_PASS` 成立**（口径与五条不声称见 `docs/b6_status_and_evidence_index_2026-10-05.md` §2.1，引用时照抄，不得扩写）；
2. **原 `eos-prone` 的 `INCOMPLETE`、全部历史 FAIL/CHECK 与生产分歧原样保留**（不改写、不重算、不重新解释）；
3. **BF16 与 V-only FP8 按各自准入分别放行**（per-KV；FP8 未完成只阻塞 FP8 档 A1/A2，**不**阻塞 BF16 主线 C1–C6）；
4. **正式 runner、采集、统计与绘图由用户本人编写**（本文件只给无代码规格）。

### 0.1 六条件设计保留

`C1–C6 = ctx{2048, 4096, 8192} × mode{AR-primary(target-only), SpecDec(production)}`；补充消融 `A1/A2 = 8K V-only FP8 × {AR,Spec}`（须先过 FP8 自己的门）。
**本轮不砍 8K、不改条件数。**

### 0.2 本轮六处修正（原表述 → 修正后 → 依据）

| # | 原表述（rev.1） | 修正后 | 依据（本轮只读核对） |
| --- | --- | --- | --- |
| **1 8K 可测性** | 直接以"历史 61% 波动"推断**当前 `cudnn_prefill` 配置 8K 不可测**，建议提前把 8K 降级 | 历史 61% 来自**另一套配置与另一套计时方式**（见 §3.1 差异表）⇒ 只作为**风险证据**；**当前配置** 8K 的**正确性与运行证据已存在**，**正式计时稳定性待 pilot 验证**；**不**据此提前砍掉 8K | `output/g5/variance/p5v_*_20260926T074909Z_c2048_spec_0.json`（commit `7a70637`、`role=spec`、`kv=base_bf16`、`ctx=8192`、`k=4`、`max_new=64`、`chunk=2048`、`warm=1`、`reps=1`、`input_token_hash=bf6a34a82018fd39`）＋ `model_self.py:338 ATTENTION_STRATEGY="eager"`（该工具从不设策略）＋ 本轮 `div8209_b6_F09_*`（commit `ba41552`、`arm=cudnn_prefill`、manifest F09 窗口） |
| **2 显存口径** | "`peak_reserved` 超过 8 GB 即可证明物理超限" | **删除该推断**。三口径分开：`allocated`=PyTorch 活跃分配；`reserved`=分配器保留（**不等于**物理占用）；**设备驻留/共享显存/迁移需独立证据**。`cuda_free=0` 只是**压力信号**，不足以单独证明迁移机制或波动原因。原始数值保留、不受支持的归因删除 | `g5_measurement_protocol §6`（明文"`peak_reserved > 8 GiB` **不得**解释为使用了超过 8 GB 显存"）；本轮 8K 产物 `mem_timeline_last`：`pt_peak_allocated=7793.19`、`pt_peak_reserved=9114.0`、`cuda_total=8187.5`、**`smi_mem_used_MiB=7794.0`（驱动口径物理占用，未超 8188）**、`cuda_free=0`。⚠️ 与 `data_ledger.md §4.20`（"双模型+8K 超出 8 GB 物理显存这条可声明"）**存在冲突**，按 §6 与本节口径处理，历史文本不改写 |
| **3 数据划分** | 建议"正式性能矩阵用全体 16 个 family（**性能与数据内容无关**）" | **删除"性能与数据内容无关"**。prompt 内容会经**接受率、EOS、实际生成长度、调用量**影响性能 ⇒ 未见样本对**性能与分歧率**同样有意义。改为：**pilot 用 4 个已用于开发的 family**；**正式评估用 ≥16 个未用于调试、未参与 pilot 的 family**；沿用原构建规则、不按结果筛选 | 本轮库存核查（§4.3）：窗口池 **32**、**未用候选 22**（池内未选中 16 + manifest 内未消费 6）⇒ **数量可行**；分层/每 block 配额约束见 §4.3 |
| **4 质量口径** | 推荐"用 AR/Spec **自身生成文本**在同一 40 窗上算 PPL"作为质量影响/等同证据 | **删除该推荐**。**真实文本**上的 NLL/PPL 与**生成文本质量**分开：40 窗检查是**真实文本**上的 KV 表示质量回归（对象 = V-only FP8 KV vs base_bf16 KV，2026-09-08 于 eager 后端），属 **FP8 消融线**；**BF16 主线不因 FP8 质量未完成而阻塞**。自生成文本评分若保留，须写明**评分模型、用途、限制**，且为**可选研究，不新增主实验阻塞** | `docs/design_freeze_fp8_kv.md:48`、`docs/daily_report_2026-09-08b_G1G2G3_FP8KV.md:54`、ledger §3.5.3（40 窗 offset 40..79、2048-token、chunk 512、ΔNLL +0.000397、rel ΔPPL +0.040%）；`production_correctness_decision:103`（"AR PPL 不能直接当作 SpecDec 生成质量"） |
| **5 重复与 block** | "≥2 blocks × (1 warm-up + 5 measured reps)"（= 每 block 各 5 次） | 以**定稿 proposal** 为准：**每个 family、每个 condition 共 5 次 measured reps**，**分布**在 ≥2 个随机 block（**不是**每 block 各 5 次）。warm-up 的进程/条件单元与次数见 §6.3；与旧测量协议的**冲突显式登记**（§6.3），**不静默增加实验量** | `docs/proposal_candidate_rev2_changelog…§2`（保留项："1 warm-up + 5 reps / ≥2 blocks"）；`g5_measurement_protocol §3.5/§5`（"每条件 warm-up 1 次 + measured reps 5；至少 2 个独立 block"——**字面可读成每 block 5 次**） |
| **6 runner 规格** | `speedup = median(T_AR)/median(T_Spec)`（分子 SpecDec、分母 AR）；要求读取 manifest 的 token IDs | **`speedup = T_AR / T_Spec`**：**分子 = AR**、**分母 = Spec**、**>1 表示 Spec 更快**；并写明 **AR-primary 是比较基线**，**不把"基线"误写成公式分母**。数据侧：`manifest.prefixes` **只存长度与 hash**（无 token IDs）⇒ 必须**按 source/block/window/tokenizer 规则重建 token IDs 再校验 hash 与长度**，不得要求读取不存在的字段 | 本轮核对 `docs/prompt_families_manifest.json`：`prefixes["8192"] = {"n_tokens":8192,"token_sha1":"…"}`，**无 ids 字段**；重建机制 = `family_input.load_family_ids(..., verify_hash=True)`；⚠️ `g5_measurement_protocol §3.1` 自身"公式写 `T_AR/T_Spec`、括注却写'分子为 SpecDec'"= **文档内部矛盾**，已登记（§7 E-3） |

---

## 1. 已满足项（可直接进入用户写代码阶段）

| # | 项目 | 依据 |
| --- | --- | --- |
| 1 | **候选配置取值**：target `Qwen/Qwen2.5-1.5B-Instruct` / draft `Qwen/Qwen2.5-0.5B-Instruct`、BF16 compute、target KV `base_bf16`、draft KV 恒 BF16 未量化、k=4、chunk=2048、max_new=64、贪心、冻结 EOS 规则（`other_short`=失败）、AR target-only、RoPE=CPU FP32、一条件一进程 | `proposal_freeze_2026-10-01.md`（FROZEN，PDF/TEX sha256）；`specdec_correctness_protocol_v2 §1` 第 13 行。与用例无关的字段在本轮 66 次运行产物中逐字段相符；`max_new` 在本轮覆盖/复核/补测中按**预先声明**取值（16/13/5/2、64、24），非运行时改动 |
| 2 | **六条件设计与数据可用** | manifest `ctxs=[2048,4096,8192]`、每个 family 的 `prefixes` 含三段长度与 hash |
| 3 | **AR = target-only；Spec = 双模型** | `g5_measurement_protocol §3.1–§3.5`；`results_schema.plan_process()` 强制；本轮产物 `kv_format.draft=not_loaded`(AR)/`fp16`(Spec) |
| 4 | **manifest 与 family 消费清单可核** | 16 family（4 stratum × 4）；**已消费 10**：F13/F01/F02/F08/F10/F06/F09/F03/F11/F04；**未消费 6**：F05/F07/F12/F14/F15/F16（本轮只读复核） |
| 5 | **warm-up / reps / 随机 block / 独立进程的纪律**（**次数口径见 §6.3，以 proposal 的"共 5 次"为准**） | `g5_measurement_protocol §3.5/§4/§5` |
| 6 | **CUDA 计时边界 / 实际生成长度 / 环境门 / 失败处理** | §5（两端同步、区间内禁诊断/IO/`empty_cache`/`nvidia-smi`、cooldown 不计时）；§7b（P-state 负载期采样、阈值 2000/7500 MHz）；§7/§7b/§7c 分层与写盘前校验；v2 §5.2 Q3 十类硬停 |
| 7 | **插桩与主计时分离** | §9；v2 **GC5**（serial/row-aligned 耗时永不进入速度结论） |
| 8 | **记录口径（schema/字段/等价性/`availability`）** | §2、§6（三口径）、§7c（映射、父进程填 `ar_equal`、缺对照写 `null`、十二条写盘校验）、GB6 分歧字段清单 |
| 9 | **B6 侧**：有限范围 `B6_BF16_PASS` + `eos_stop` 真实覆盖 + 8K 融合 prefill 证据 | `docs/b6_status_and_evidence_index_2026-10-05.md`（E5/E6/E7/E8） |
| 10 | **治理与测试**：守卫 34/0、预检 rc=0、CPU 套件 25/25、编排自测 39/39 + 35/35 | 同上（E9/E11/E12） |

## 2. 用户写 runner 前需冻结的**配置与数据决策**

| # | 需冻结项 | 当前状态 | 建议取值（推荐） |
| --- | --- | --- | --- |
| F-1 | **attention backend / 阶段策略** | 候选 = `cudnn_prefill`（仅 prefill 融合、其余 eager），已有 B6 级正确性证据；**尚未列入冻结签名** | 冻结 `cudnn_prefill`；写入签名 `backend=cudnn_prefill(fused_prefill_only)` |
| F-2 | **配置冻结 commit** | 每次运行自带 `git_commit`/`dirty_worktree`，但签名里无具名 commit | 签名加 `frozen_config_commit=<sha>`（写代码时的起点） |
| F-3 | **pilot family（4 个）** | 未定 | **推荐 F09/F03/F11/F04**（Stage 2b 已用、stratum 各一：S1/S2/S3/S4、已有最丰富既有证据，含 F09 的 8K 融合复核）；备选 F01/F02/F08/F10（协议轮） |
| F-4 | **正式评估 family（≥16 个未用）** | 未定；本轮已核**数量可行**（22 个未用候选） | 从 §4.3 的 22 个候选中选 ≥16，**排除 pilot 的 4 个**；分层与每 block 配额按 §4.3 的约束说明处理 |
| F-5 | **reps/blocks 的最终口径** | proposal 说"共 5 次、分布于 ≥2 block"；measurement protocol 字面可读成"每 block 5 次" | 采用 proposal 口径：**总 5 次 measured**，≥2 个随机 block；建议 3+2 分布（§6.3） |
| F-6 | **质量口径** | 真实文本 40 窗（FP8 线）已有；生成质量无 | BF16 主线**不新增**质量门；质量只作为 FP8 线的独立待办（§5）；自生成评分列为**可选研究** |

## 3. pilot 的职责（不新增重复稳定性矩阵）

pilot **只**验证以下四类，且**在 pilot 之前不再另加"重复的稳定性矩阵"**：

1. **2K/4K/8K 的运行与资源**：三档能否按冻结配置跑通；显存三口径如实记录（`allocated`/`reserved`/`smi` 物理）；
2. **计时边界、重复与随机 block**：区间两端同步、区间内无诊断/IO、warm-up 与 measured reps 的分离、block 随机化与 seed 可重放；
3. **测量稳定性**：`within_run` 与 `between_block` 分层状态、是否触发 `CHECK`（20% 仅作诊断线）、P-state 资格；
4. **schema / 序列 / 失败 / 分歧记录**：字段齐全、`availability` 自洽、写盘前校验通过、GB6 分歧记账可用、硬停规则可用。

**8K 的处理纪律（保留六条件、不预判）**：pilot 若发现 8K 不稳定 ⇒ **保留全部结果**，随后按**预登记规则**在"修复（如结构性手段）"与"受限报告（只报内存证据 + 计时不可复现）"之间选择；
**不得**把历史风险当作新配置的失败预设，**不得**因历史波动提前砍掉 8K。
（历史 61% 波动与本轮候选配置的差异，见 §3.1；它只说明"该条件在旧配置/旧计时方式下曾高度不稳定"，**不是**当前配置的结论。）

### 3.1 历史 8K 波动 vs 当前候选配置：逐项差异（本轮只读核对）

| 维度 | 历史 61% 样本（2026-09-26，ledger §4.20） | 当前候选配置（本轮 8K 证据） |
| --- | --- | --- |
| commit | `7a70637` | `ba41552`（本轮 EOS 前为 `429ff9a`/`0b9c841`） |
| **backend / arm** | **eager**（`ATTENTION_STRATEGY="eager"`，无融合；该工具不设策略） | **`cudnn_prefill`**（仅 prefill 融合；AR/Prod/Serial 实测 `fused_by_phase={'prefill':28/52}`、`fused_s_max=2048`） |
| 输入 | legacy `wiki.test.raw` 前缀 8192（`input_token_hash=bf6a34a82018fd39`） | manifest **F09** 8K 窗口（`input_tokens=8192`，`source_kind=manifest`） |
| 数值配置 | ctx 8192 / chunk 2048 / k=4 / max_new 64 / base_bf16 / bf16 compute / rope cpu_fp32 | **相同**（逐字段相符） |
| **计时方式** | `g5_8k_spec_variance_probe`：单进程、`P5V_WARM=1`+`reps=1`、`_prefill` 包装计时、**无插桩**、无随机 block | `serial_oracle_gpu`：**诊断**运行（`diagnostic_only=true`、**含插桩**、产物自带"计时含插桩，禁止进入任何 speedup 表"）、每 (mode,rep) 一次、无 warm-up、无随机 block |
| 显存（原始值） | `peak_alloc≈7120.6 MiB`、`peak_resv≈10736 MiB`、`inact_split≈1371 MiB` | **6 次实测**：AR `alloc 6153.5 / resv 8134 / smi_used 7816`；Spec(prod/serial) `alloc 7793.2 / resv 9114 / smi_used 7785–7857`；`cuda_total 8187.5`；`cuda_free 0`（全部）。按 §0.2-2 口径解读：`resv` 是分配器保留**不是**物理占用；`smi_mem_used`（驱动口径设备总占用）**始终低于** `cuda_total`；`cuda_free=0` 仅表示**压力**（快照时无空闲）；`alloc/resv` 逐 rep **字节相同**，而 `smi_used` 有 ±0.5% 波动（外部消费者） |
| 结论 | 该条件在**旧配置**下跨顺序极差最高 61% | **正式计时稳定性未测**（现有 8K 数据是诊断/插桩/单次）⇒ **待 pilot 验证** |

### 3.2 pilot 通过后 → 正式采集的条件（清单）

pilot 满足以下各条后，才进入"可采集正式数据"的状态（**仍须 E-1～E-6 落定与批 2 批准**）：

1. **pilot 四类职责全部完成且记录齐全**（§3 的 1–4：2K/4K/8K 运行与资源、计时边界/重复/随机 block、测量稳定性、schema/序列/失败/分歧记录）；
2. **E-1/E-2/E-3 已落定**：backend 与 `frozen_config_commit` 冻结；pilot 与正式 family 集合确定（正式集 ⊆ 未使用候选、且与 pilot 不相交）；reps/blocks 与 speedup 方向的口径统一并登记；
3. **8K 处置已决定并写入报告模板**（E-5）：修复（另立预登记研究）或受限报告（只报内存证据 + 计时不可复现）；
4. **测量资格在 pilot 中未出现系统性问题**：P-state 与入口环境门未批量 `ENV_INVALID`；每 condition 的 measured reps 可按 §6.3 达到约定（总 5 次、跨 ≥2 block），无"整块重跑"反复；
5. **质量范围已确认**（E-4）：BF16 主线不新增质量门；FP8 线（40 窗真实文本）若因后端变更需重跑，则**已排期或明确记为未完成**且不写成已完成；
6. **报告模板就位**（Gate C）：分歧率 + 速度的轨迹归属 + 质量边界 + 不稳定单元的 §7 处理；产物命名/不覆盖规则可执行；
7. **批 2 批准**（`BF16_FORMAL_READY` 属批准记录 §A3 的"批准正式实验"，须另行批准）+ 运行预算与人时计划（环境/预算准入）。

## 4. 数据划分（pilot / 正式 / 辅助）

### 4.1 原则（修正后）

* **性能对内容敏感**：prompt 内容经**接受率**（决定每轮产出）、**EOS**（提前停止）、**实际生成长度**、**调用量**（target/draft 前向次数）直接影响计时 ⇒ 正式评估需要**未用于调试、未参与 pilot** 的样本，否则"代表性"不可声明。
* **pilot = 4 个已用于开发的 family**；**正式评估 = ≥16 个未用于调试、未参与 pilot 的 family**；**沿用原构建规则、不按结果筛选**。
* **旧 family 可作辅助分析**（如机制解释、历史对照），但**不得冒充独立未见样本**。

### 4.2 建议的 pilot IDs

**推荐 F09 / F03 / F11 / F04**（S1/S2/S3/S4 各一；均已用于 Stage 2b 开发，其中 F09 另有 8K 融合复核）。
备选：F01 / F02 / F08 / F10（协议轮，亦为 stratum 各一）。二者都满足"用于开发"的要求；**选定后即从正式池中排除**。

### 4.3 正式评估的可用窗口：**只读库存核查结果**

核查方法：用构造脚本同一规则（`scripts/build_prompt_families.py` 的 `load_block_texts`/`features`：4×300k 字符 block、8192-token 对齐窗口、同 tokenizer）**重建窗口池**，
并与 manifest 中 16 个 family 的 `prefixes["8192"].token_sha1` 逐一对账。

| 项 | 结果 |
| --- | --- |
| 语料 block | **4** 个，各 300000 字符（`wikitext-2-raw-v1/wiki.train.raw`） |
| 每 block token 数 / 对齐窗口数 | 67711/8、68462/8、70392/8、69864/8 |
| **窗口池总数** | **32** |
| 重建对账 | 16 个 manifest family 的 8192 `token_sha1` **全部一致（0 不一致）** ⇒ 核查所用规则与构造规则一致 |
| **未使用候选合计** | **22** = 池内未被 manifest 选中 **16** + manifest 内未被任何工作消费 **6**（F05/F07/F12/F14/F15/F16） |
| 未使用候选的每 block 分布 | 池内未选中：b0 **2**、b1 **5**、b2 **7**、b3 **2**；未消费 manifest family：b0 1（F05）、b1 1（F07）、b3 4（F12/F14/F15/F16） |
| **非重叠核查** | 同 block 按 `[i·8192,(i+1)·8192)` 对齐切分 ⇒ 窗口互不重叠；block 间为不相交字符区间；family 取自 `train.raw`、质量窗取自 `test.raw` ⇒ 与质量语料不重叠（builder 的 `non_overlap_argument`） |
| **分层约束** | 层标签由**全池 rank 规则**判定（S1=distinct 最低 4、S2=markup 最高 4、S3=prose 次 4、其余 S4）。实测极值段：S1 = F09 0.1852 / F14 0.1932 / F15 0.2002 / F08 0.2197；S2 = F13 0.0052 / F16 0.0045 / F03 0.0043 / F02 0.0042。**未使用候选的 distinct 比例集中在 0.226–0.275**（S1 极值段 0.185–0.220 大部分已被消费）⇒ 若沿用**原全池 rank 的既有标签**，未使用候选整体偏向 **S4**；但其 markup/prose 特征仍覆盖一定范围（未用候选中 markup 最高者 0.0039/0.0038/0.0035）⇒ **在正式集合内按同一 rank 规则重新分层（R-1）可得到均衡的 4×4**。既有 `caveat` 已声明"层标签宜视为排序而非独立文本体制" |
| 每 block 配额 | 原 `selection_rule` 每 stratum ≤4、每 block ≤4（不可行时放宽）。若要求**每 block ≤4** 且全部取未使用候选 ⇒ 最多 **15** ⇒ 要取满 **≥16** 需把每 block 上限放宽到 **5**（原规则**已允许**这种放宽，但需登记） |

**结论（数量可行、分层需修订）**：**≥16 个未用窗口在数量上可行（22 ≥ 16）**；真正的约束是**分层均衡**——
要在"未使用候选"中凑出 4 stratum × 4 的均衡集合，按原 rank 规则**不可行**（同 stratum 上限 4 会与"已消费极值"冲突）。

**最小修订建议（等待裁决；本轮未实施）**：
* **R-1（推荐）**：新增一份 **`formal_families` manifest 版本**（不改 `v1-draft` 原文件），用**同一 builder 规则**从 22 个未用候选中取 **16** 个，并把**每 block 上限放宽到 5**（登记为对 `selection_rule` 的一次显式修订）；
  层标签**在正式集合内按同一 rank 规则重新判定**，并在 manifest 里写明"层标签为集合内相对排序，不跨集合比较"（与既有 `caveat` 一致）。
* **R-2（保守）**：正式集合取 **15**（严格每 block ≤4），并在报告里显式声明"15 ≥ 16 不成立，故结论限于 15 个未用窗口 + 4 个 pilot 的对照"。
* **R-3（若必须 4×4 均衡）**：扩大语料 block 数（builder 的 `n_blocks=4` 是常量；增加 block 会改变 `corpus_rule`）⇒ 属**新数据构建**，须单独授权。

## 5. FP8 与研究支线的**独立**待办

| # | 待办 | 归属 | 是否阻塞 BF16 主线 |
| --- | --- | --- | --- |
| FP8-1 | `v_fp8` 下**自己的**控制逻辑门（现有 serial/EOS/B6 证据全为 `base_bf16`） | FP8 档（A1/A2） | **否**（per-KV 放行，已批准） |
| FP8-2 | 若后端冻结为 `cudnn_prefill`（≠ 历史 eager）⇒ **40 窗真实文本质量回归需按冻结配置重跑**（对象 = V-only FP8 KV vs base_bf16 KV；口径不变：40×2048、chunk 512、offset 40..79） | FP8 档 | **否**（BF16 主线不以 FP8 质量为门） |
| FP8-3 | FP8 档的显存/结构检查（S2–S4）与环境/预算准入 | FP8 档 | **否** |
| R-1 | **P-1/P-2 双侧决策行捕获**（并列现只能记 TIE-C） | 研究支线（可选） | **否**（只影响分歧解释精度） |
| R-2 | **自生成文本质量评分**（若要做） | 研究支线（可选） | **否**；若保留须写明**评分模型、用途、限制**（如：用固定外部评分器/人工 rubric 对 AR 与 Spec 各自输出打分；**用途**=有限样本对照；**限制**=不等于总体质量、不得声称分歧后质量等同） |
| R-3 | 五类算子形状敏感性的机制收尾（已定位到"特定同历史样本上的因果实验"） | 研究支线 | **否**（proposal rev2 已限定不声称对所有分歧定位） |

## 6. 无代码实现规格（修正版）

### 6.1 输入面

| 类别 | 内容 |
| --- | --- |
| 条件 | `C1–C6`（ctx{2048,4096,8192} × {AR-primary, SpecDec}）；`A1/A2` 须先过 FP8 自己的门 |
| 数据 | **从 manifest 的 `family_id`（source/block/window_index/window_token_start）+ 冻结 tokenizer 重建 token IDs**，并用 `prefixes["<ctx>"].token_sha1` 与 `n_tokens` **校验**（现有机制：`family_input.load_family_ids(..., verify_hash=True)`）。⚠️ `manifest.prefixes` **不含 token IDs**，**不得**要求读取该字段 |
| 分块/重复 | ≥2 个随机顺序 block（seed 记档）；**每 family × condition 共 5 次 measured reps**，分布在 ≥2 block（§6.3） |
| 身份字段 | `run_id`(UTC)、`git_commit`、`dirty_worktree`、`seed`、`block_index`、`condition_index`、`condition_order`、`condition_role` |
| 冻结配置 | target/draft 模型、BF16 compute、target KV `base_bf16`、draft KV BF16、k=4、chunk=2048、max_new=64、贪心、冻结 EOS 规则、AR target-only、rope cpu_fp32、**backend = 待冻结（推荐 `cudnn_prefill`）**；全部从**单一配置源**读取 |
| 门 | 守卫（family 授权）→ 环境门（harness 16 GiB 线、入口门槛、工作树 clean）→ P-state 预检（**负载期**采样）→ 才可计时 |

### 6.2 输出面

* **逐 (condition, family, block, rep) 一行**：`prefill_seconds`/`decode_seconds`/`total_seconds`、吞吐、
  **三显存口径**（`peak_allocated`（PyTorch 活跃分配，主）/`peak_reserved`（分配器保留，**不得**当作物理占用）/
  `nvidia_smi_used`（驱动物理参考，如 8K 实测 `smi_mem_used=7794 MiB`））、
  SpecDec 统计（`accepted`/`proposed`/`rounds`/`draft_calls`/`target_block_calls`/`mismatch_rounds`/`all_accept_rounds`/`acceptance=accepted/proposed`）、
  生成序列（ids + hash）、**实际生成长度**、停止原因、跨进程等价性（`ar_equal` 由**父进程**填；缺对照写 `null`，**绝不写 False**）、
  状态分层（`within_run_status`/`between_block_status`/`environment_status`/`measurement_status`）、
  `speedup_claim_allowed` + `speedup_claim_blockers`、`availability`（值的纯函数）。
* **GB6 分歧记录**：production vs AR 每一处 —— 首分歧 index、完整共同前缀长度 P、两侧 token、决策位置（绝对位置+轮号）、
  该位置 top-1/top-2 与 margin、并列计数、该位置 cache 长度、两侧完整输出、两侧接受率/轮数/target 调用数。
* **物理分离**：正确性/诊断产物与计时表分属不同 tag/文件；serial 与 row-aligned **永不**进入计时表。
* **不覆盖**：目标存在且未显式 `append` ⇒ 报错；历史 JSONL 只读。

### 6.3 重复次数、block 与 warm-up（**统一口径 + 冲突登记**）

* **以定稿 proposal 为准**：**每个 family、每个 condition 共 5 次 measured repetitions**，**分布**在**至少 2 个随机顺序 block** 中（**不是**"每 block 各 5 次"）。
* **推荐分布**：2 个 block ⇒ **3 + 2**；3 个 block ⇒ 2 + 2 + 1（若某 block 被环境门整块判 `ENV_INVALID`，按 §7b.2 跳过该 block 的正式测量，并在报告中标注该 condition 的 measured 次数下降）。
* **warm-up**：**每个条件单元各 1 次并丢弃**；"条件单元"= **(condition, family, block)** 的独立子进程（即同一 condition 在两个 block 各有一个新进程 ⇒ 共 2 次 warm-up）。warm-up **不计入** 5 次 measured，也**不进**计时。
* ⚠️ **与旧测量协议的冲突（显式登记，不静默改量）**：`g5_measurement_protocol §3.5/§5` 写"每条件 warm-up 1 次 + measured reps **5**；至少 2 个独立 block"，字面可读成"每 block 5 次"（= 10 次/条件）。**本文件按 proposal 的"共 5 次"执行**；建议把 §3.5/§5 的措辞统一为"**每 (condition, block) 各 1 次 warm-up；该 condition 的 measured reps 合计 5 次，跨 block 分布**"，并**登记**这次措辞统一（不改实验量）。
* 若小组另有决定（例如每 block 5 次），**必须显式批准并按新量登记**，不得由实现者默认。

### 6.4 计时边界

| 量 | 边界 |
| --- | --- |
| `total_seconds` | 从（权重加载完成、warm-up 之后）prefill 之前同步点 → 生成结束同步点 |
| `prefill_seconds` | 包装后的**分块** prefill 区间（8K 必须分块；`prefill_source ∈ {reported_median, reported_per_rep}`；**禁止**用 `total−decode` 偷算） |
| `decode_seconds` | 同一 rep 内 `total − prefill` |
| **不计入** | 权重/tokenizer 加载、warm-up、cooldown、系统查询、任何诊断/插桩 pass |
| 插桩 | 仅独立 pass（外部 CUDA Event）；**不入**正式主表；`P5_PROFILE=0` 量化插桩开销 |

### 6.5 speedup 口径（修正后，唯一写法）

* **`speedup = T_AR / T_Spec`**：**分子 = AR**、**分母 = Spec**；**`speedup > 1` 表示 SpecDec 更快**（AR 用时更长）。
* 只能**同 context、同 KV** 相比；分子/分母都取**全部 measured rep 合并后的中位数**（`median(t_decode_s)` 两侧相除；e2e 同理），与离线报表口径一致。
* **`AR-primary`（target-only）是"比较基线"**（分母侧的比较对象），**不得**把它写成"公式的分母"这类混淆表述：公式的分母是 **Spec**。
  ⚠️ `g5_measurement_protocol §3.1` 自身"公式 `T_AR/T_Spec`"与"括注：分子为 SpecDec、分母为 AR-primary"**互相矛盾**，本轮已登记（§7 E-3），实现以本文件为准。
* 一旦 production 与 AR 轨迹分歧：该比值必须标注"**在分歧轨迹上的比较**"，并同时给出 P（共同前缀）与两侧接受统计；**不得**声称"同输出无损加速"。

### 6.6 验收规则

1. **每条件（GB1–GB5）**：finite（决策用 logits 全覆盖、`first_bad` 空）；cache/逐轮结构不变量；**重复一致**（同 condition 的 measured reps 内容一致）；合法停止（`max_new_reached`/`eos`；`other_short`=失败）；无 OOM/结构失败/异常。
2. **GB6**：分歧逐处完整记账（不再自动判 harmless、不自动归因）。
3. **声明资格（`§7b.4` 十条）**：分母为 AR 且 `ar_residency=target_only`；`speedup_denominator_eligible`；两侧 `context_actual`/`target_kv`/`max_new_tokens`/`input_token_hash` 一致；reps 完整（= 该 condition 的 5 次 measured，或按 §6.3 的 block 跳过规则减少并标注）；两侧 `measurement_status=OK`；between-block 未超诊断线；无 OOM/NaN/ENV_INVALID；`tokens_finite=true`；非 dual-control；行内无 profiling 数据。
4. **硬停**（v2 §5.2 Q3）：控制逻辑 token 不一致 / 重复不一致 / 非有限值 / cache 或逐轮不变量失败 / 非法停止或非法 token / OOM 或崩溃 / 审计空泛或记账不成立 / 缺 D6 复位读数 / family 越界或工作树不 clean / 环境不合格。
5. **写盘前自洽校验**（`§7c.5` 十二条）：任一条不过 ⇒ **拒绝写出**。

## 7. 需要你裁决的事项（集中一次；不新增编号审批循环）

| 编号 | 事项 | 推荐 | 影响 |
| --- | --- | --- | --- |
| **E-1** | 正式 backend 冻结（F-1）与配置冻结 commit（F-2） | 冻结 `cudnn_prefill`（仅 prefill 融合）+ 记录 `frozen_config_commit` | 决定正式数据用哪个后端；写代码时可先按此默认实现 |
| **E-2** | pilot family 4 个（F-3）与正式 family 集合选取（F-4 / 数据划分 §4.3） | pilot = **F09/F03/F11/F04**；正式 = 从 22 个未用候选中取 **≥16**，采用 **R-1**（新增 `formal_families` manifest 版本 + 每 block 上限放宽到 5 + 集合内重新分层） | 决定"未见样本"声明是否成立；只读核查已证明**数量可行**、分层需修订 |
| **E-3** | 两份文档的内部矛盾统一：`g5_measurement_protocol §3.1`（speedup 分子/分母措辞）与 `§3.5/§5`（reps 是否每 block） | 按本文件 §6.3/§6.5 统一并**登记**（不改实验量，只改措辞） | 避免实现者按旧措辞多做 2 倍测量或算错方向 |
| **E-4** | 质量范围（F-6） | BF16 主线**不新增**质量门；40 窗检查归 **FP8 消融线**（后端变更后需按冻结配置重跑）；自生成评分列为**可选研究**（R-2） | 决定是否要新质量工作；**不**阻塞 BF16 主线 |
| **E-5** | 8K 处置（**不预判**）：pilot 若判 8K 不稳定，走"修复"还是"受限报告" | pilot 前不定；pilot 后按**预登记规则**二选一（修复 = 结构性手段的独立预登记研究；受限 = 只报内存证据 + 计时不可复现） | 影响 C5/C6 能否给出可声明速度比 |
| **E-6** | 是否采用 §4.3 的 R-2（15 个未用窗口）作为备选 | 若不愿新增 manifest 版本，则用 R-2 并显式声明"15 < 16" | 影响"≥16"字面要求的满足方式 |

## 8. 三问（修正后；不含宣布）

* **能否开始写正式代码：能。** 计时边界、schema、门控、声明资格、speedup 方向与数据重建规则都已明确（§6）；未定项（backend/pilot&正式 family）都是**配置与数据**参数，不影响 runner 结构；写代码不消耗 GPU、不占用待决项。建议先做**可离线验证**的四件事：dry-run 计划输出、schema 校验、`validate_canonical` 自洽校验、声明门控。
* **能否跑 pilot：建议先把 E-1/E-2/E-3 落定再跑**（否则 pilot 的 family 集合与 backend 可能与正式集合不一致，pilot 结论无法直接沿用）。**本轮不启动 pilot**；pilot 职责与"8K 不预判"纪律见 §3。
* **能否开始正式采集：不能。** 除 E-1～E-6 外，正式采集属批准记录中的**批 2**（须另行批准），且需 pilot 完成后按 pilot 结果决定 8K 处置与 family 集合。**不得**仅凭有限范围 `B6_BF16_PASS` 宣布 `BF16_FORMAL_READY`。

## 9. 本轮**未做**的事

* 未运行 GPU（未跑 pilot、未跑方差探针、未重跑 66 次 B6 或 6 次 EOS）；
* 未修改生产代码、门限、冻结判据、manifest 或任何历史产物/判决；
* 未编写正式 runner / 采集 / 统计 / 绘图 / final pipeline；
* 未宣布 `BF16_FORMAL_READY` 或 `FP8_FORMAL_READY`；未把计划写成完成。
