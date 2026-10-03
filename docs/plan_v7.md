> **SUPERSEDED HISTORICAL PLAN — NOT THE CURRENT PROJECT SCOPE.**
> This v7 plan and its 12-condition design are retained as planning history.
> Current formal scope is the frozen **6 main + 2 supplement** design in
> [`proposal_freeze_2026-10-01.md`](proposal_freeze_2026-10-01.md); also consult
> [`README.md`](../README.md) and [`status_update_2026-10-02.md`](status_update_2026-10-02.md).

# SpecDec v7 —— 收敛式重构（研究方案总设计师 + 严格课程审稿人版）

> 原则：**让项目更容易回答一个清楚的研究问题，而不是让系统看起来更复杂。**
> 核心研究问题不变：context length × KV precision × speculative decoding 在 8GB 消费 GPU 上的关系；
> v7 把"系统叙事"收敛为"受控析因实验 + 经验归因 + 预设失败分析"。

---

## 1. v6/v6.1 最重要的 12 个问题（按严重程度排序）

| # | 严重度 | 问题 |
|---|---|---|
| 1 | **Critical** | **模型不一致**：方案写 Target=1.5B、Draft=0.5B，但 data ledger 的 correctness 全部是 **0.5B**；没有 1.5B target 的正确性门，0.5B 不能冒充 target 验证 |
| 2 | **Critical** | **8K 可行性假设已被实验推翻**：8K fp16 eager prefill OOM，但 v6 时间预算未把"长上下文分块 prefill"作为 blocker 计入 |
| 3 | **Major** | **研究叙事=“填满 18 格”**：混淆内部记账与成功定义；成功应=能否回答 factor 关系 |
| 4 | **Major** | **指标混用**：ppl 被当投机解码质量指标；性能用 best-of-N、单一 38-token prompt |
| 5 | **Major** | **归因伪精度**：S=G−K−V 声称可分离测 KV 读取成本；3 点插值求高精度 L* |
| 6 | **Major** | **优先级倒挂**：weight int8/int4、CUDA K1 占据正常执行路线；而实验/分析（45%）未保护 |
| 7 | **Major** | **失败分析未前置**：failure cases 在实验后才想；成功被定义为"必须 positive" |
| 8 | **Major** | **工时预算未对剩余工作重估**：把已做工作重复计费 + 用"挂机"掩盖工程复杂度 |
| 9 | **Minor** | 正式基准协议不达标：无 ≥5 reps/median/温度记录/条件交错 |
| 10 | **Minor** | repo 卫生：`model_self.py` 末尾 `__main__` 残留；缺 configs/results/scripts 规范目录 |
| 11 | **Minor** | L* 无不确定性说明；仅 3 个 context 点直接插值=伪精度 |
| 12 | **Minor** | 叙事风险："OURS 快于 transformers"易被质疑 fairness，应只作 baseline 可信度陈述 |

## 2. 修改原则

- **保留**：① 自研 eager engine 与 0.5B 开发期正确性（重新标记 development-stage）；② 分块 prefill（刚完成，转为 P1 正式项）；③ v6 的承诺纪律内核（只承诺 MUST、诚实负结果、可复现）；④ Draft=Qwen2.5-0.5B、Target=Qwen2.5-1.5B 的角色分工；⑤ ΔPPL 作为 quantization quality 指标。
- **修改**：研究叙事→3×2×2 析因；指标分族；正式基准协议；归因→经验归因框架；L*→自适应细化+区间报告；优先级重排；工时重估；课程提案剥离内部代号。
- **降级**：weight int8/int4 → optional supporting；CUDA K1 → 后置 gated 项（或删除）；"18 格"→ 内部 bookkeeping；部分 COULD（动态树/自训头/int2 KV）→ 默认不开。
- **删除**：S=G−K−V 的可分离测量主张；正式实验用 best-of-N；单一 prompt 支撑结论；0.5B 冒充 1.5B；无标注的 L* 单点；"FP8 一定更好"式承诺。

## 3. v7 完整方案（Master Plan）

### 3.1 Research Questions & Hypotheses
- **RQ1（主）**：在 8GB consumer GPU 上，context length 如何影响 speculative decoding 的 net speedup？FP8 KV 是否改变其 profitable 区间？
- **RQ2（归因子）**：观测到的 speedup 衰减主要由 draft acceptance 下降还是 verification overhead 上升导致？
- **H1**：net speedup 随 context 增长而下降（acceptance 与 verify 贡献待 RQ2 归因）。
- **H2**：FP8 KV 通过降低 KV 显存/带宽压力改变（延长或不延长）net speedup>1 的区间——方向不预设，positive/negative/mixed 均可作为成功结果。
- **H3（质量）**：FP8 KV 的 ΔPPL 保持在可接受阈值内（预设 δ，如实报）。

### 3.2 Models
- **Development / Draft model**：Qwen2.5-0.5B-Instruct —— engine 开发、早期正确性、speculative draft（**已完成的 0.5B correctness = development-stage validation**）。
- **Final Target model**：Qwen2.5-1.5B-Instruct —— 最终实验 target、最终 spec-dec 验证、最终报告性能。
- **最小 target correctness gate（P1 完成标准）**：
  1. OUR 1.5B full forward vs HF 1.5B：max|Δ| < 1e-3；
  2. OUR 1.5B cached decode vs OUR 1.5B full forward：max|Δ| < 1e-3 且 argmax 一致；
  3. greedy token output 等价（同一固定 prompt 集）。
- 禁止：把 0.5B 结果当作 1.5B target correctness。

### 3.3 System
- 自研 PyTorch eager engine（已实现 L1/L2/分块 prefill）；KV-cache；固定链 speculative decoding（draft=0.5B，同词表）；FP16/FP8 KV。
- 立场：实证型 LLM inference systems project；不声称新 spec-dec 方法；增量=8GB 消费 GPU + 自研 eager engine + context×KV×spec-dec 交互 + break-even/failure analysis + 低资源组合实证。

### 3.4 Experimental Design —— 3×2×2 受控析因
- **Factor A** Context Length：2K / 4K / 8K
- **Factor B** KV Cache Precision：FP16 / FP8
- **Factor C** Decoding Method：Standard AR / Speculative
- 每个 condition 使用**同一** prompt set、output length、generation settings、seed 策略、硬件、warmup 协议。
- 第二阶段（break-even refinement，可选）：当观察到 speedup crossing（如 4K>1、8K<1）才加 ≤2 个 adaptive probe（6K，必要时 5K/7K）。

### 3.5 Metrics（分族，禁止混用）
- **Performance**：tokens/s、ms/output token、net speculative speedup、peak VRAM（prefill/decode 分列，如适用）。
- **Speculative**：acceptance rate α、proposed/accepted tokens、accepted tokens per verification、draft latency、verification latency、total wall-clock。
- **Quantization quality**：ΔPPL（主）；small fixed task/QA 仅在有资源时作为 secondary sanity。
- **明确声明**：PPL 是 quantization quality metric，不是 spec-dec effectiveness metric。Greedy spec-dec 若实现正确，target output 理论保持 → output equivalence 是 correctness sanity，不用 ppl 论证"spec-dec 更好"。

### 3.6 Benchmark Protocol（正式）
- Warmup：若干次；Measured runs：优先 ≥5 reps，最低 3；汇总用 **median** + 至少一种 spread（std / IQR / min-max range）；禁止 best-of-N。
- 温度/功耗/thermal throttling：固定 power state（尽力）、条件执行顺序随机化/交错、记录 GPU 温度/频率或至少在 limitation 说明。
- 正式 prompt set：**16–32 个固定 prompts**，覆盖 factual QA / instruction following / summarization-continuation / simple reasoning / code-structured；固定 prompt IDs、tokenizer、生成长度、记录各 prompt token length；所有 condition 一一对应。单一 38-token prompt 仅用于 correctness/microbench。

### 3.7 Attribution Protocol（经验归因，防过度建模）
- 可测项：acceptance/accepted-token benefit、draft cost、verification cost、total net speedup（自研循环内分段计时）。
- KV bandwidth：仅 theoretical lower-bound estimate / supporting explanation，明确标注 estimate，不作为精确实测成分。
- 结论模板：*"We empirically attribute the degradation primarily to accepted-token benefit and verification overhead; KV bandwidth is treated only as a theoretical supporting estimate because it cannot be independently isolated in our eager implementation."*

### 3.8 Failure Analysis（前置定义，实验前确定）
- F1 acceptance 明显下降 → 随 context？随 prompt 类型？
- F2 acceptance 保持但 speedup 下降 → verification/draft cost 主导？
- F3 FP8 省显存但不提 speedup → 有价值的 negative；
- F4 FP8 质量退化 → ΔPPL、是否只出现在长 context；
- F5 8K feasibility 受实现限制 → 区分 algorithmic / implementation / hardware limitation。

### 3.9 Reproducibility
固定种子、config 哈希、append-only results.jsonl、env capture、16–32 prompt 集版本化、一条命令复跑核心结果、README+requirements+scripts/configs/results 目录规范（课程要求）。

### 3.10 Scope Ladder / Risk Management
P0（完成）环境/0.5B correctness/engine/分块 prefill（development）。P1 target 1.5B correctness + 8K feasibility 正式化。P2 FP8 KV + spec-dec + instrumentation。P3 正式 prompt 集 + factorial 运行 + 质量 sanity。P4 归因/failure/break-even/可视化。P5 复现/报告/演示。门：G1 1.5B 三检查过；G2 8K 分块可跑；G3 FP8 KV 与 spec-dec 正确性过；冻结 11/14。

## 4. 核心实验矩阵（12 conditions + 说明）

| # | A: Context | B: KV | C: Decode | 记录 |
|---|---|---|---|---|
| 1 | 2K | FP16 | AR | perf 基线 |
| 2 | 2K | FP16 | Spec | net speedup / acceptance |
| 3 | 2K | FP8 | AR | AR+FP8 对照 |
| 4 | 2K | FP8 | Spec | FP8×spec |
| 5 | 4K | FP16 | AR | |
| 6 | 4K | FP16 | Spec | |
| 7 | 4K | FP8 | AR | |
| 8 | 4K | FP8 | Spec | |
| 9 | 8K | FP16 | AR | feasibility（分块 prefill）|
| 10 | 8K | FP16 | Spec | |
| 11 | 8K | FP8 | AR | |
| 12 | 8K | FP8 | Spec | |

每格同 prompt 集/生成长度/seed/warmup；质量 sanity（ΔPPL）在代表性 condition 上跑；crossing 出现才做 adaptive probes。

## 5. Remaining-work 时间预算（不含已做）

**已做（sunk，不重复计费）**：环境；0.5B engine/L1/L2 correctness（0.000、缓存等价、生成对齐）；分块 prefill 开发与等价验证；方案/提案/文档；基线 sanity 与 8K OOM 探测。

**剩余（估算，保守偏上，不用"挂机不算人工"掩盖）**：
| 工作 | 估时(h) | 说明 |
|---|---|---|
| 1.5B target correctness gate（3 检查+加载配置）| 6–8 | G1 |
| 8K 分块 prefill 正式化 + 显存/时间验证（probe 重跑）| 6–10 | G2；清理进程驻留/expandable segments |
| FP8 KV（per-channel/组、读写路径、质量检查）| 12–16 | G3 前置 |
| Fixed-chain spec-dec（draft=0.5B、verify loop、token 计量）| 18–24 | G3 |
| Benchmark instrumentation（results schema、计时钩子、env capture）| 8–10 | 与 P2 并行 |
| 正式 prompt 集 16–32 + generation configs | 6–8 | P3 |
| Factorial 12 conditions × reps（工程+脚本/韧性；墙钟另计）| 14–18 | 可无人值守运行，但脚本/恢复按人工算 |
| ΔPPL 质量 sanity | 5–7 | |
| Attribution/figures/break-even/failure analysis | 22–28 | P4 保护项 |
| Reproducibility（clean rerun、configs、results、README）| 8–10 | P5 |
| Report（5–7 页）+ 演示 + 互评材料 | 28–34 | P5 保护项 |
| Buffer/考试周 | 8–12 | |
| **合计** | **≈140–175** | 视 1.5B/FP8 顺滑度 |

> 不再机械沿用 185–193h；已完成的 sunk 全部移出；新增 8K 工程与正式评测已计入；K1 与 weight int4 不在本表（见裁剪序）。

## 6. Cut Order（时间压力时明确砍序）
1. **第一个砍：CUDA K1**（直接删除；Creativity 已有自研 engine/量化/spec-dec 覆盖，45% 在实验分析）。
2. **第二个砍：weight int8/int4**（最多保留一个 int8 memory sanity；不进入主链）。
3. **第三个砍：FP8 的小 QA secondary / adaptive probes 若不稳定**（保 ΔPPL 主指标）。
4. **绝不能砍**：controlled evaluation、repetitions、failure analysis、reproducibility、report figures。
5. 若仍超时：砍 FP8 深度分析但保 2K/4K fp16 核心问答 + 诚实 negative（最小成功版本）。

## 7. Proposal Skeleton（未来 1–2 页，不带内部代号）
1. **Question**：主问题（context 长度如何影响 8GB GPU 上 spec-dec net speedup；FP8 KV 是否改变边界）+ 归因子问题（acceptance vs verification cost）。
2. **Motivation**：memory-constrained consumer GPU 上 spec-dec 的工程决策空白；已知方法在小卡/长上下文缺少受控实测。
3. **System/Method**：Qwen2.5-1.5B target + 0.5B draft；自研 PyTorch eager engine（含 KV cache）；FP16/FP8 KV；fixed-chain spec-dec。
4. **Experiment**：3 context × 2 KV × 2 decoding 受控析因，16–32 固定 prompts。
5. **Metrics**：speedup/latency/VRAM/acceptance/draft&verify cost/ΔPPL（分族说明）。
6. **Feasibility**：RTX 4060 8GB；scope cuts（无 K1/无 weight 主线）；已有 engine 与 0.5B correctness 进度；剩余预算 ~140–175h。

## 8. Professor/TA Pre-review（保守，不给伪精确概率）
按课程 rubric 给保守区间（依赖核心 factorial + 分析是否完整执行）：
| 组件 | 权重 | 保守区间 |
|---|---|---|
| Problem & Motivation | 10% | 7–9 |
| Technical Implementation | 20% | 15–18 |
| Experimental Design & Evaluation | 25% | 16–21 |
| Analysis & Insight | 20% | 13–18 |
| Report & Presentation | 15% | 11–14 |
| Creativity / Beyond Baseline | 10% | 7–9 |
| **总分（/100）** | | **≈69–89** |

典型可预期区间 **≈78–86**（若实验+分析执行充分），而非"A+ 概率 70%"式表述。低端=仅完成最小 factorial 且分析浅；高端=完整 factorial + FP8 扩展 + 归因 + 高质量 figures/failure。

## 9. Final Go / No-Go Verdict
- **结论：GO（方向值得继续），以"最小成功版本"为底线。**
- **最大风险**：1.5B target correctness 或 8K feasibility 若长期不过，会挤压 P3/P4（实验+分析）。
- **Pivot 条件**：若（a）1.5B correctness 三检查在多轮后仍不过，或（b）8K 即使分块也无法在预算内跑通且无可行替代 → 收敛最小版本（2K/4K × FP16 × AR/SpecDec，诚实 negative 分析）并向导师说明；仅在此类实现不可行时才 pivot，negative 结果本身不构成 pivot 理由。
- **最小成功版本**：RQ1 在 2K/4K（至少 FP16）得到受控可复现 net speedup 测量 + RQ2 经验归因 + 预设 failure analysis + 干净复现/图表（评分 ~73–80）。
- **高分版本**：完整 3×2×2（含 FP8 与 8K，用分块 prefill）+ break-even 区间 + attribution + F3/F4 类有价值的 negative + 5 张对图 + 5–7 页报告（评分 ~84–89）。

## 10. 禁止事项遵守自查
✓ 不新增模型/训练/DPO/LoRA/EAGLE tree；✓ K1 降级为 gated/删除；✓ weight int4 optional；✓ 不用单一 prompt 支撑结论；✓ 正式不用 best-of-N；✓ 不把 3 点插值写成高精度 L*（区间+不确定性）；✓ KV bandwidth 标 estimate；✓ ppl 定位正确；✓ 不预设 FP8 一定更好；✓ negative 不算失败；✓ 尊重 8K OOM 并纳入 blocker；✓ 不把 0.5B 冒充 1.5B；✓ 收敛而非扩充。
