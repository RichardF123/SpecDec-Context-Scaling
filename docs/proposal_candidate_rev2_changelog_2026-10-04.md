# proposal 候选稿 rev2 — 提交前四项小修（2026-10-04）

* **基础**：`output/pdf/specdec_course_proposal_main6_ablation2_candidate_2026-10-04.tex`（rev1，**未覆盖**）
* **本版**：`output/pdf/specdec_course_proposal_main6_ablation2_candidate_rev2_2026-10-04.tex` / `.pdf`（**5 页**）
* **旧稿**（2026-10-01 定稿）与 rev1 **均保留未动**；本轮**只改文档、编译并检查 PDF**，未运行 GPU、未改变 6+2 实验范围。
* **状态**：**未**标记为已提交。页眉保留 `candidate rev. 2026-10-04`（按你的指示，只有确认为最终提交版时才去掉）。

---

## 1. 四项修订对照

### 修订 1 —— 收紧正确性归因

| | 内容 |
| --- | --- |
| **修订前（rev1）** | `Production BF16 batched verification still generates different tokens from AR on some inputs, localised to shape-dependent finite-precision behaviour in five operator classes (attention QK and the four projections) rather than to the control logic.` —— **范围说过满**：读起来像"所有 production 分歧都已逐算子定位" |
| **修订后（rev2）** | 拆成两点、各自限定：<br>① `Production BF16 batched verification still differs from AR on some inputs, and two points must not be merged.`<br>② `Shape sensitivity in five operator classes (attention QK and the four projections) was confirmed by causal experiments on specific same-history samples; this identifies a mechanism but **does not account for every production divergence**, so **we claim no operator-level localisation for all of them**.`<br>③ `The five-family serial result shows that complete per-token target verification recovers the AR output on those inputs; **neither point establishes control logic for all inputs**.` |
| **依据** | 五类算子的形状敏感性来自**特定同历史样本**上的因果实验（`docs/gemm_shape_sensitivity_mechanism_2026-10-03.md`、`docs/div8209_serial_{qk_causal,oproj,l0_closure}_2026-10-02.md`）；五个 family 的 serial 结果见 `docs/serial_oracle_stage1_f13_*`、`docs/serial_oracle_stage2_*` |

### 修订 2 —— 明确 production 分歧的**准入语义**

| | 内容 |
| --- | --- |
| **修订前（rev1）** | Proposed Method 末句含糊的 `No numerical exemption is assumed.` |
| **修订后（rev2）** | `Admission rules are explicit: **non-finite values, cache or per-round structural failures, and non-reproducible repetitions are hard failures that stop the condition**; a token divergence between production and AR must be recorded and explained in full --- first position, common prefix, full outputs, acceptance and work volume --- **is never automatically called harmless, and does not support any claim of lossless speedup at identical output.**` |
| **全文一致性** | 与 Table 4 的 `no lossless same-output claim is made`、`never used to compute speedup`、Analysis 的 `without equivalent-output speedup claims` 三处**表述一致**（矛盾扫描见 §3） |
| **依据** | `docs/specdec_correctness_protocol_v2_2026-10-04.md` Gate A/B 的硬失败清单与 GB6 记账要求；`production_correctness_decision_2026-09-29.md:60` |

### 修订 3 —— 修正工具**覆盖范围**

| | 内容 |
| --- | --- |
| **修订前（rev1）** | `...per-round control traces, call-geometry audits and divergence records **are captured for every run**, and separate instrumentation passes are available.` —— 暗示**尚未执行**的正式实验已有这些记录 |
| **修订后（rev2）** | `...and **already provides the capability for** independent-process runs, per-round control traces, call-geometry audits and divergence records. **The formal experiment will enable and verify these fields under the frozen protocol**; formal timing still requires pilot validation.` |

### 修订 4 —— 改善**跨页阅读**

| 手段 | 说明 |
| --- | --- |
| 表格可跨页 | Table 3 / Table 4 由 `tabularx` 改为 `longtable`（可跨页拆分）⇒ 消除原先"表格整块推页"造成的空白（实测 p4 曾浪费 **20.6 行**、p3 曾浪费 **8.0 行**） |
| 禁止孤立残句 | 加入 `\widowpenalty=10000`、`\clubpenalty=10000`、`\interlinepenalty=10000` ⇒ **段落在页内完整结束** |
| 精简重复文字 | 合并"Existing implementation and validation"到 Preliminary Work 首段；合并 Table 2 的 Hardware/Compute 行、Table 4 的 latency/throughput 与 component/memory 行；合并 Timeline 的 12/13 周 |
| **结果** | **5 页**；**每页都从"段落或小节的开头"开始**（p2 `Correctness evidence...`、p3 `Remaining work...`、p4 `C1/C2...`、p5 `Analysis and feasibility...`）；**无跨页残句**；底部余量为 2.1 / 4.3 / 3.3 / 6.2 / 4.6 行 |
| **未使用** | **未**缩小字号（仍 11pt）、**未**缩小页边距（仍 0.9in，与 rev1 相同）、**未**压缩行距以外的任何"硬塞"手段 |

---

## 2. 明确**保留**的内容

* 主问题（固定模型对与 RTX 4060 Laptop 8GB 下上下文长度对贪心 SpecDec 相对 AR 解码时间比的影响）；
* **6 个 BF16 主条件**（C1–C6：2K/4K/8K × AR/Spec）+ **2 个 8K V-only FP8 消融**（A1/A2，K 保持 BF16、draft KV 保持 BF16）；
* 模型、`k=4`、chunk=2048、最多 64 token、EOS 规则、4-family pilot + ≥16 formal families、1 warm-up + 5 reps / ≥2 blocks；
* **三位成员与分工主体**（措辞微调：Deng 的 `strict output-equivalence validation` → `control-logic output-equivalence validation through the serial oracle` + `production-divergence recordkeeping`；Fang 增加 `per-representation control-logic gate`）；
* **全部实测数字**：NLL +0.000397、relative PPL +0.040%、224.00→169.75 MiB（1.32×）、allocated peak −0.06 GiB、40 非重叠 2048-token 窗口、chunk 512、140–175 person-hours、`k = 4`、2048、64；
* **真实状态**：`B6-bf16` **CHECK 19/20**（一条未解决分歧，用户裁决 A 严格口径）、row-aligned **算法比较 3/4 未通过**（`ALGORITHMIC_GATE_FAIL`）、serial oracle **五 family PASS（64/64）**、**`v_fp8` 同档正确性仍待验证**、正式门/pilot/主矩阵**未通过**。

---

## 3. 核查记录

| 检查 | 结果 |
| --- | --- |
| 编译 | `xelatex` exit 0；`Overfull hbox/vbox = 0`、`Missing character = 0`、`LaTeX Warning = 0`、`undefined = 0`、字体警告 = 0（1 处 Underfull hbox，纯拉伸） |
| 页数 | **5 页**（rev1 为 5 页） |
| 裁切/溢出 | 逐词 bbox 审计：5 页**全部 0 个词越出可打印区**；无表格溢出（0 Overfull vbox） |
| 跨页阅读 | 每页首行均为**段落/小节开头**，无孤立残句（实测见 §1 修订 4） |
| 引用 | `[1]`–`[5]`、`Leviathan`、`Merity`、`Qwen2.5 Technical Report` 均在位；`undefined = 0` |
| 四项修订落地 | 逐条字符串核对：修订 1（5 项）、修订 2（7 项）、修订 3（2 项）、保留项（16 项）**全部命中** |
| 矛盾扫描 | 含 equal/identical/equivalence/lossless/exempt 的 **5 句**逐句确认均为**限定性或禁止性**表述，与"允许记录 production 分歧"**不冲突** |
| 图像目视 | ⚠️ **未做图像级人工目视**：当前模型无图像输入能力（`read_image` 不可用）。已用 **几何 bbox + 逐页文本 + 编译日志** 三项程序化检查替代；逐页 PNG 已渲染保留（`_rev2_preview-1..5.png`）供人工确认 |

---

## 4. 提交前**仍未完成**的验证（不变）

1. **`v_fp8` 同档控制逻辑验证**（B6 条件含 `v_fp8`；现有 serial 证据只有 BF16 base KV）；
2. **最终测量配置冻结与回归**（attention backend/阶段策略；若与 eager 不同须重跑 40 窗口质量检查）；
3. **4-family pilot** 未跑；4. **主矩阵 C1–C6 / A1–A2** 未跑；
5. Gate C 的"分歧率 + 质量并列"报告模板未定型；
6. 现有控制逻辑证据仅覆盖 **8192 窗口 × 64 token**。
