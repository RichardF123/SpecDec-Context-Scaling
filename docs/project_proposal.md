> **SUPERSEDED HISTORICAL PROPOSAL DRAFT — NOT THE CURRENT FORMAL SCOPE.**
> This draft preserves an earlier optimization ladder and deliverable plan.
> Current formal scope is governed by the frozen **6 main + 2 supplement**
> design in [`proposal_freeze_2026-10-01.md`](proposal_freeze_2026-10-01.md);
> also consult [`README.md`](../README.md) and
> [`status_update_2026-10-02.md`](status_update_2026-10-02.md).

# SpecDec —— 课程项目方案（Proposal Draft）

> 用途：课程 10/7 提案初稿 + 组内/协作者对齐。权威技术细节见 `docs/plan_v6.md`；数据见 `docs/data_ledger.md`。

---

## 1. 基本信息
- **项目名**：SpecDec —— 从零推理引擎 × 投机解码（8GB 本地推理的实证优化）
- **方向**：(c) coding/reasoning（主线）+ (d) LLM training & efficiency（次线）
- **硬件**：RTX 4060 Laptop 8GB + i7-12700H + 32GB（免费 Colab 可选）
- **交付节点**：提案 10/7 → 演示 W12–13 → 结题 11/25

## 2. 一句话摘要
> 我们**从零手写**一个小型 LLM 推理引擎（Qwen2 架构），沿 **KV-cache → 量化 → KV 量化 → 投机解码** 逐级优化，用可复现的测量回答："8GB 小卡上投机解码随上下文增长何时失效？KV 量化能否延长其获利区间？"——并在引擎之上交付一个 **LLM 驱动的原型应用**。

## 3. 问题与动机
- 人人都想在 8GB 消费卡上跑能推理的小模型，但瓶颈是**生成慢**与**上下文一长显存爆**；优化手段（KV-cache/量化/投机解码）大多只有口号、缺少**在小卡上从零实现 + 逐级量化收益**的实测。
- 文献对投机解码的接受率衰减有报道（MagicDec、QuantSpec），但**在 8GB 消费卡 + ~1.5B GQA 模型上的成本结构与盈亏平衡点**缺少受控实测——我们补这一块。

## 4. 方案（做什么、已做到哪）
### 4.1 自建引擎（原创核心）
| 阶段 | 内容 | 状态 |
|---|---|---|
| L1 | 从零前向：RMSNorm/RoPE/GQA attention/SwiGLU/24 层/绑定打分 | ✅ 对齐门 **0.000** |
| L2 | KV-cache 解码 + 贪心生成 | ✅ 缓存等价（~2e-5）、**生成对齐 PASS**；基线 CPU 9.2 / GPU 10.6 tok/s |
| L4 | int8/int4 权重量化（显存/带宽） | 下一步 |
| L4b | fp8 KV 量化（KIVI 式，长上下文） | 待做 |
| L5 | 固定链投机解码 + 2K/4K/8K 曲线 | 待做 |
| 测量/归因 | 18 格表、Spec-Bench 风格协议、盈亏平衡上下文、results.jsonl | 待做 |

### 4.2 LLM 驱动原型（满足 "LLM-powered prototype"）
在引擎之上做一个薄应用层（候选/已倾向）：
- **代码 shape/数据流分析助手**：输入一段 NumPy/torch 代码 → 输出逐步数据流、变量形状、最终输出；**真跑代码验证形状** → 得到可客观评测的原型正确率。
- 演示时展示实时 tokens/s 与显存（把测量叙事带进应用）。
- 边界：单函数小片段（MVP）；小模型能力如实报告；放在 L4/L5 之后（W10–11 demo 期）。

## 5. 可交付清单（现在 + 计划）
**已有的**：
- 自写引擎代码 `engine/model_self.py`（L1+L2 全实现）
- 验收/调试工具：align_ours（0.000）、test_cache_equiv、align_generate（PASS）、bench_basic
- 性能基线数据（CPU/GPU prefill/decode）、数据台账 `docs/data_ledger.md`
- 文档：plan_v6、阶段报告、教学与 debug 复盘

**将产出的**：
- 报告（含测量表/图、归因分析、诚实负结果、与 EAGLE-2/KIVI/MagicDec/QuantSpec 对照）
- 可运行代码（一条命令复跑 L1–L5 + 外部对照）+ `results.jsonl`
- 18 格（2K/4K/8K × fp16/fp8 × 接受率/净加速/ppl）与盈亏平衡结论
- **原型应用**（shape/数据流分析助手 demo）+ 其正确率评测
- 演示（W12–13）、互评材料（rubric/自评/复跑说明）
- 可选研究延伸（小论文式总结；CUDA K1 加分轨视进度）

## 6. 技术栈
Python + PyTorch 2.6（cu124）；自写引擎（OURS）；HF 仅权重/tokenizer（LIB）；量化（int8/int4, fp8 KV）；固定链投机解码；评测 lm-evaluation-harness + 自建样例集；复现=种子+config 哈希+只追加 results.jsonl。

## 7. 里程碑（对齐 v6）
9 月：L1 已提前完成 → L2 完成（生成对齐 PASS）
10/7：提案（含裁剪梯/18 格/归因/原型声明）
10/7–11/14：L4 → L4b → L5 固定链 → 2K/4K/8K 曲线 → 外部基线/复现 → 冻结 11/14
11/15–25：只写不跑：报告/短文/演示/互评
W10–11：原型应用（shape/数据流分析助手）穿插 demo 期
旁路：CUDA K1（G0 冒烟决定，加分项）

## 8. 风险与诚实性
- 小模型（0.5B/1.5B）能力有限 → 原型与测量用可控任务，能力如实披露；
- 缓存/生成 vs 全量存在 ~2e-5 浮点噪声（kernel 顺序）→ 以 1e-3 容差 + 预测一致为门；
- 时间/考试周 → 只承诺纯 PyTorch MUST；11/14 冻结神圣；
- `hf.generate` 与其全量前向不一致 → 对齐裁判统一用"HF 全量逐步贪心"。

## 9. 对照文献（arXiv，提案引用前已核验）
EAGLE-2 (2406.16858)、KIVI (2402.02750)、MagicDec (2408.11049)、QuantSpec (2502.10424)。

## 10. 一句话价值主张
> 我们交的不只是"一个能跑的小模型"，而是**自写引擎 + 可复现的质量×成本实测 + 一个由引擎驱动的原型**——用数字回答"8GB 小卡上推理的钱该往哪花"。
