# 课程 proposal 候选提交版 — 修订记录（2026-10-04）

* 旧稿（**保留，未覆盖**）：`output/pdf/specdec_course_proposal_main6_ablation2_2026-10-01.tex` / `.pdf`
* 新稿（**候选提交版**）：`output/pdf/specdec_course_proposal_main6_ablation2_candidate_2026-10-04.tex` / `.pdf`
* **状态**：`docs/proposal_freeze_2026-10-01.md` 的 `FROZEN` 仅表示**此前的组内定稿**，
  **不代表已提交课程或导师**。本文件与候选版均**不**声称已提交。
* 依据（只读引用）：`docs/serial_oracle_stage1_f13_2026-10-03.md`、`docs/serial_oracle_stage2_2026-10-03.md`、
  `docs/correctness_protocol_validation_2026-10-03.md`、`docs/specdec_correctness_protocol_v2_2026-10-04.md`、
  `docs/specdec_protocol_v2_evidence_inventory_2026-10-04.md`（内含 B6-fp32 PASS 20/20 与 B6-bf16 CHECK 19/20 的原文引文）。

---

## 1. 逐项修订表（旧 → 新 → 依据）

| # | 位置 | 旧稿文本（要点） | 修订后 | 依据（实测证据 / 口径） |
| --- | --- | --- | --- | --- |
| **1** | Completed Preliminary Work → "Existing implementation and validation" 末句 | `However, BF16 AR/Spec output discrepancies remain under investigation; earlier component-level passes do not establish strict equivalence for the final production configuration.` | **删除该句**，另起一段 **"Correctness evidence, split into two layers"**：① serial verification oracle 的定义（与 AR 相同的逐 token 执行 + 逐字复用生产 accept/replacement/rollback/cache 逻辑 + 调用几何审计证明不存在多 token verify 前向）；② 在 **F13/F01/F02/F08/F10** 五个 family 上**完整 64/64 token 与 target-only AR 一致**，独立重复与结构检查通过；③ **只是这些输入上的控制逻辑证据，不是所有输入的形式化证明，且仅覆盖 `base_bf16`**；④ **两条历史记录原样保留**：row-aligned 局部干预在 4 个协议验证 family 中**3 个未通过**算法比较、早前 BF16 门记录**一处未解决分歧（19/20）**；⑤ production BF16 batched verify 在部分输入上**仍与 AR 生成不同 token**，已定位为**五类算子**上的形状相关有限精度行为（注意力 QK 与四个投影），**不是控制逻辑**；⑥ oracle 仅诊断用，**其耗时排除在一切性能结果之外**；⑦ **正式正确性门、pilot 与主矩阵并未因此通过** | Stage 1/2 报告（5 family PASS、几何与记账恒等式）；`correctness_protocol_validation_2026-10-03.md`（`ALGORITHMIC_GATE_FAIL`，**不改写**）；清点文件 §1.1（`B6-fp32` PASS 20/20、`B6-bf16` CHECK 19/20 原文引文） |
| **2** | Proposed Method 第 2 段 | `BF16 output discrepancies remain under investigation. Strict output-equivalence and regression gates apply separately within each KV representation; quantized outputs need not equal BF16 outputs. No numerical exemption is assumed.` | 改为**分层门槛**：*Control logic*（serial oracle 对比同 KV 档 target-only AR，要求**完整 token IDs 一致** + cache/逐轮不变量/finite/确定性/合法停止）；*Production numerical behaviour*（原始 batched verify，要求结构与重复性通过；与 AR 分歧时**记录**首位置/共同前缀/完整输出/接受与工作量，**不自动称无害、也不自动称算法错误**）；两档 KV **各自在同一档内**验证、**不要求跨档相同**；五 family 结果**只覆盖 `base_bf16`**，故 **FP8 仍需单独的控制逻辑验证**；`No numerical exemption is assumed.` 保留 | 协议 v2 §2/§3（Gate A/B）；本文件即"**提交前拟采用的新口径**"，非"旧门已通过" |
| **3** | Table 4 → Decode speedup (primary) | `... for matched inputs and outputs; $S>1$ means faster decoding.` | `... under matched inputs, KV representation, generation limit and timing boundary`；并增加：**若两路径生成不同 token，比值按"cross divergent trajectories"报告**，同时给出**分歧率、首分歧分布、实际长度、接受/轮/target 调用数**；`no lossless same-output claim is made`；**诊断（serial/row-aligned）运行绝不用来计算 speedup** | production 实测分歧（F13@47、F01@14、F02@7、F08@52、F10@8）；`production_correctness_decision_2026-09-29.md:60`（不得称为"完全相同输出下的无损加速"）；v1 §1.1/§6（诊断不计时） |
| **4** | Table 4 → Acceptance behavior | `Acceptance rate and accepted tokens per verification help explain target-call savings.` | 追加：**分歧后各自报告每条轨迹的工作量** | 同上（不同轨迹改变接受率与工作量） |
| **5** | Table 4 → Correctness and failures | `... finite outputs and cache invariants; OOM, crashes, and blocked conditions retained.` | 追加：**逐轮控制不变量与控制逻辑门状态**；**记录的 AR-vs-Spec 分歧（首位置与共同前缀）** | Gate A/GB6 的记录要求 |
| **6** | Table 4 → Component timings | 含 `no claim of independently isolated memory bandwidth.` | 内容保留；把"诊断运行不计 speedup"移到第 3 行（避免重复） | 版面压缩 |
| **7** | Table 1 → Evidence boundary | `... These findings do not establish benefits for a different backend or the formal Spec configuration.` | 追加：**这些数字只衡量"KV 表示改变后"的预测质量，单独**不能**证明 AR 与 production SpecDec 在轨迹分歧后生成文本质量相同**；旧 eager 结果不得外推到最终后端 | 任务书第 4 条；`production_correctness_decision:103`（PPL 不是 SpecDec 生成质量证明） |
| **8** | Measurement infrastructure 段 | `... independent-process runs and separate instrumentation passes are available.` | 追加：**每次运行记录逐轮控制 trace、调用几何审计与分歧记录** | Gate A/GB6 的能力前置 |
| **9** | Remaining work 段 | `Complete strict correctness and final-configuration regression, ...` | `Complete the control-logic correctness gate for the second KV representation and the final-configuration regression, ...` | `v_fp8` 同档验证缺口 |
| **10** | Timeline 第 8–9 周 | `Complete correctness diagnosis, unify entry-point configuration, and validate data provenance.` | `Finish the control-logic correctness gate for both KV representations, freeze the entry-point configuration, and validate data provenance.` | 同上（诊断已完成，剩下的是**按 KV 档**的门 + 配置冻结） |
| **11** | Timeline 第 10–11 周 | `... run the four-family pilot, then the six main conditions and gated 8K ablation.` | 追加 `recording output divergence for each pair` | 任务书第 3 条（分歧必须随速度一起报告） |
| **12** | Roles | Fang：`target/KV correctness and regression`；Deng：`strict output-equivalence validation` | Fang：`target/KV correctness and regression including the per-representation control-logic gate`；Deng：`control-logic output-equivalence validation through the serial oracle, production-divergence recordkeeping` | 与分层门槛一致；**三位成员与分工主体不变** |
| **13** | 版面 | `\newpage`×4、margin 1in、parskip 5pt、arraystretch 1.15、页眉两段 | 去掉 4 处强制分页、margin **0.9in**、parskip **3pt**、arraystretch **1.08**；页眉加 `candidate rev. 2026-10-04` | 内容净增后仍**保持 5 页**（旧稿 5 页）；几何审计：**0 越界词、0 Overfull** |

## 2. **明确未改动**的内容（研究范围与承诺）

* 主问题、主指标（decode speedup）、**6 主条件 + 2 补充条件**矩阵（C1–C6 / A1–A2）；
* 模型对（Qwen2.5-1.5B-Instruct / 0.5B-Instruct）、`k=4`、chunk 2048、最多 64 新 token、EOS 规则；
* 2K/4K/8K 嵌套前缀与 manifest token-sha 校验、4-family pilot + ≥16 formal families、
  1 warm-up + 5 measured reps / ≥2 randomized blocks、median + IQR 汇总；
* FP8 结论与 K-FP8 负结果、V-only FP8 的 40 窗口质量数字（仅补充边界说明）；
* **不恢复** 12 条件 factorial，**不加入**新模型/量化格式/CUDA 算子/参数搜索；
* 三位成员与分工主体、Responsible Use 与工具披露、参考文献。

## 3. 本稿**没有**声称的事（提交前必读）

1. **没有**说"旧 production exact-token 门已经通过"。本稿呈现的是**提交前拟采用的新口径**（分层门槛）；
   历史 `B6-bf16 CHECK 19/20` 与 row-aligned `ALGORITHMIC_GATE_FAIL` **保持原样、未被改写**。
2. **没有**把 serial oracle 的 PASS 写成 production B6 PASS：serial 是**诊断**，其结论只覆盖**控制逻辑**，
   且只覆盖 `base_bf16` 的五个 family。
3. **没有**声称所有输入正确（原文写有 `it is not a proof for all inputs`）。
4. **没有**声称"相同输出下的无损加速"（Table 4 与 Analysis 两处均写明禁止性表述）。
5. **没有**设置任何"首分歧 index=0 / ≥1"的准入阈值（任务书要求不得设置无证据支持的阈值）。
6. 页眉的 `candidate rev. 2026-10-04` 用于区分版本；**若导师要求正式提交，可在提交前移除**。

## 4. 提交前**仍未完成**的验证

| # | 未完成项 | 说明 |
| --- | --- | --- |
| 1 | **`v_fp8` 同档控制逻辑验证** | `design_freeze_g4_specdec.md:83` 把 B6 条件写成 `base_bf16` **与** `v_fp8`；现有 serial 证据**只有 BF16**。FP8 档需单独跑 Gate A（并先确认 serial 路径在 fp8 KV 下定义良好） |
| 2 | **最终测量配置冻结与回归** | attention backend / 阶段策略尚未冻结；若与记录的 eager 配置不同，需重跑 40 窗口质量检查 |
| 3 | **4-family pilot 未跑** | 计时稳定性、资源、插桩需 pilot 验证 |
| 4 | **主矩阵未跑** | C1–C6 与 A1–A2 均未开始；8K 补充对受同档正确性门约束 |
| 5 | **分歧率与质量并列呈现的模板** | Gate C 要求正式报告同时呈现分歧率与质量结果；具体图表模板尚未定型 |
| 6 | **更长生成 / 其它上下文长度的控制逻辑证据** | 现有 5 family 均为 8192 窗口 × 64 token |

## 5. 核查记录（本候选版）

| 检查 | 结果 |
| --- | --- |
| 编译 | `xelatex` 成功（exit 0）；`Overfull hbox/vbox = 0`、`Missing character = 0`、`LaTeX Warning = 0`、`undefined = 0`（1 处 Underfull hbox，纯排版拉伸） |
| 页数 | **5 页**（旧稿同为 5 页）；每页正文行数 44 / 51 / 47 / 50 / 41 |
| 版面几何 | 逐词 bbox 审计：**5 页全部 0 个词越出可打印区**（左右 0.9in 边距、下边界）；页眉与页码每页齐全 |
| 内容审计 | 必须出现项（`64/64`、`not a proof for all inputs`、`base_bf16`、`v_fp8`、`19/20`、`row-aligned`、`across divergent trajectories`、`never used to compute speedup`、`not passed by this evidence`、C1–C6/A1–A2、`k = 4`、参考文献 [1]–[5] 与作者名）**全部命中** |
| 旧句式清除 | `remain under investigation`、`matched inputs and outputs`、`establish strict equivalence` **均已不在文本中** |
| 矛盾扫描 | 全文中含 equal/identical/equivalence/lossless 的句子共 **4 句**，逐句人工确认**均为限定性/禁止性表述**，与"允许记录 production 分歧"**不冲突**：① `identical tokens across the two representations are not required ...`（限定：跨档不要求相同）；② `identical target prefill policy in AR and Spec`（指 **prefill 策略**相同，非输出相同）；③ `no lossless same-output claim is made`（**禁止**声称同输出无损加速）；④ `Deng Xiyue owns ... control-logic output-equivalence validation through the serial oracle, production-divergence recordkeeping`（把等价验证**限定在控制逻辑层**，并与分歧记账分列） |
| 视觉检查 | ⚠️ **未做图像级人工目视**：当前模型无图像输入能力，`read_image` 无法读取渲染出的 PNG。已改用**几何 bbox 审计 + 逐页文本提取 + 编译日志**三项程序化检查替代（结果如上）。PNG 已渲染保留（`output/pdf/_cand_preview-1..5.png`），可供人工目视 |

**产物**：`output/pdf/specdec_course_proposal_main6_ablation2_candidate_2026-10-04.tex`、
`output/pdf/specdec_course_proposal_main6_ablation2_candidate_2026-10-04.pdf`（5 页）。
旧稿 `.tex` / `.pdf`、旧 preview PNG、历史报告**均未覆盖**。
