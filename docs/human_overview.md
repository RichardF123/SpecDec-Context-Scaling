> **SUPERSEDED HISTORICAL SNAPSHOT — NOT THE CURRENT PROJECT SCOPE.**
> This v3 overview preserves earlier planning ideas. Dynamic draft trees, a
> weight-quantization ladder, training a draft head, FlashAttention as a
> promised project stage, and int2/int4 KV as committed scope are historical
> ideas—not current commitments. Current formal scope is governed by
> [`proposal_freeze_2026-10-01.md`](proposal_freeze_2026-10-01.md); for current
> status, also consult [`README.md`](../README.md) and
> [`status_update_2026-10-02.md`](status_update_2026-10-02.md).

# SpecDec 推理引擎 —— 项目说明书（给你看，v3 版）

> 这是**用大白话**讲清"我们到底在做什么、为什么、怎么做、怎么衡量、最后得到什么"的文档。
> **本次更新（v3）**：采纳了两个顾问 agent 的结果——一个帮我们把方案升级得更创新（动态 draft 树 + KV 量化），另一个客观指出风险并定下"保底 vs 加分"纪律。看完这份，你就能确认这个方向是不是你要的。
> 技术细节：`specdec_final_plan_v3.md`（最终方案）/ `specdec_technical_proposal_v2.md`（改善版）/ `specdec_design_deliverable.md`（设计规格）。

---

## 0. 一句话（30 秒讲给任何人听）

> 我们在**一台 RTX 4060 8GB 的笔记本上，自己动手写一个小型 LLM 推理引擎**，然后沿优化阶梯逐级做 **KV-cache、Triton 加速、量化、KV 量化、投机解码（动态 draft 树）**，并用一张可复现的测量表证明：每一级快了多少、省了多少显存、代价是什么质量——最终回答一个问题：**投机解码在 8GB 小卡上随上下文变长到哪里失效，KV 量化能否延长它继续获利？**

---

## 1. 我们研究探索什么问题

**痛点**：人人都想在 8GB 消费卡上跑会推理的小模型，但两个瓶颈绕不开——**生成慢**、**上下文一长显存爆**。业界很多优化口号，但少有人在 8GB 小卡上给出一套**从零实现、逐级量化收益**的真实数据。

**研究问题**（可证伪）：

> 在 8GB 消费级 GPU 上，从零实现一个 LLM 推理引擎，沿"朴素 → KV-cache → Triton kernel → 权重量化 → KV 量化 → 投机解码"逐级优化，每级的**速度、内存、质量保持**是多少？其中**投机解码随上下文增长在哪里停止获利，KV 量化能否延长它继续获利的区间**？

**新增亮点（v3）**：我们不是"重写一批已知 kernel 报个数"，而是回答一个**文献没在 8GB 小卡上回答过的问题**——这是原创性的来源。

---

## 2. 为什么选它作为我们的 Project

| 维度 | 为什么合适 |
|---|---|
| 契合课程 | 覆盖 (c) 推理（投机解码、解码策略）+ (d) 训练与效率（推理优化、量化、训练 draft） |
| 真动手实现 | 前向、KV-cache、kernel、量化、投机解码全是**我们写**的，不是调库 |
| 原创可量化 | 交的是"逐级优化 vs 收益"实证表 + 一个可回答的研究问题，不是照教程 |
| 硬件可行 | 4060 8GB 正好是 1–2B 小模型量化推理的甜区，整套在你笔记本跑完 |
| 含真实训练 | 训练 EAGLE 式 draft 头（~0.2B、共享 KV、8GB 上 ≤90 GPU 分钟） |
| 可复现可互评 | 固定种子 + config 哈希 + 外部对照，一条命令复跑 |

---

## 3. 技术路线

一句话：**先造一个能自己跑的引擎，沿优化阶梯逐级加速，每步量收益；再用"动态 draft 树 + KV 量化"回答长上下文问题。**

### 3.1 优化阶梯（新增 L4b）

| 级 | 做什么 | 收益 |
|---|---|---|
| L1 | 朴素自回归（每步重算全序列） | 基线 |
| L2 | **KV-cache** 解码循环 | decode 快 ~4–10× |
| L3 | **Triton 融合 kernel**（attention/GEMV/RMSNorm） | 再快 ~20–60% |
| L4 | **权重量化**（int8/int4） | 显存降约一半 |
| **L4b** | **KV-cache 量化**（fp16→fp8→int4→int2，KIVI 式） | 长上下文可行 + KV 读取更快 |
| L5 | **投机解码**：小 draft 先猜，target 一次验证 | 净加速（接受率 ≥50% 时） |

### 3.2 两个"创新点"（v3 新增，冲高分关键）
1. **动态 draft 树 + 接受余量调度器**：不再固定"每次猜 4 个词"，而是让调度器根据 draft 头对自己猜测的"自信程度"动态决定这步多猜几个、猜多深，再用**树注意力一次验证**。这是 EAGLE-2/3、Medusa-2/3 的 SOTA 方向，且**无需额外训练**（复用同一组 draft 权重）。
2. **KV 量化阶梯**：量化 KV-cache 本身，直接解决"8GB 上下文短"的痛点，并和投机解码组合成统一叙事。

### 3.3 训练环节（辅，但真实）
自己蒸馏/微调 EAGLE 式 draft 头提高接受率（8GB 上 ≤90 GPU 分钟）。**注意（v3 纪律）**：训练头是"加分项"，不是保底——保底用的是"现成小模型 draft / n-gram"（免训练），所以就算训练环节翻车，投机解码那一级也能跑出数字。

---

## 4. 具体技术栈

| 层 | 用什么 |
|---|---|
| 目标模型 | Qwen2.5-1.5B / TinyLlama-1.1B（Apache-2.0） |
| 实现 | Python + PyTorch（前向/attention/KV-cache/采样/解码**我们写**） |
| 热路径 | Triton（flash attention、解码 GEMV、融合 RMSNorm/SwiGLU、量化 GEMM、树注意力、KV 反量化） |
| 量化 | int8/int4（自写路径；可选 FP8 需真机验证） |
| KV 量化 | KIVI/KVQuant 式（fp8/int4/int2，per-channel/per-token） |
| 投机解码 | 自写 draft+verify；固定链保底 + 动态树加分 |
| draft 训练 | QLoRA/PEFT + bitsandbytes，蒸馏 |
| 评测 | lm-eval-harness（ppl、tiny math）+ 自写速度基准（Spec-Bench 对齐） |
| 复现 | config 哈希 + 固定种子 + 环境快照 + 追加式 JSONL/CSV |

---

## 5. 量化标准

**速度**：decode tokens/s、TTFT、吞吐。**内存**：显存峰值、KV bytes/token。**投机**：接受率、净加速比、每步生成词数、树形状热图。**质量**：困惑度 / 自建小数学题集 / 生成一致性。

**两条硬规矩（保证公平与诚实）**：
1. **Triton go/no-go**：第 2 周末在真机跑对最小 attention kernel；不行就锁 eager，预算转投量化和投机解码。
2. **成功线下调**：投机解码"净加速 ≥1.0×（净正向）即达标"，1.6–2.2× 是 stretch；接受率 ≥50% 即达标。**没达标也如实报告**——诚实空结果是加分项。

---

## 6. 最后要得出什么结果

1. **一张逐级测量表**：L1→L5（含 L4b）每级的速度/TTFT/显存/接受率/质量；
2. **三个判断**：哪一级是 8GB 上最大瓶颈；动态树/训练 draft 是否真的更强；整条链质量保住多少；
3. **一个可核查结论**，例如：
> "在 8GB 卡上，我们的引擎解码比朴素快 ~6×；KV 量化让 8K 上下文从不可行变可行；投机解码在 ≤2K 时净加速 ~1.8×、接受率 ~0.66，到 8K 后跌到 ~1.1×——说明长上下文下投机解码接近失效，而 KV 量化把有效区间延长到 ~4K。"

**课程交付物**：报告（含测量表 + 诚实负结果）、可运行代码（一条命令复跑 L1–L5 + 外部对照）、互评材料、可选研究延伸（把"8GB 小卡上的投机解码"写成小论文）。

---

## 7. 保底 vs 加分（记住这张，就不会翻车）

- **保底（MUST，一定交付）**：L1→L4 + L4b(KV 量化至少 fp8) + L5 用现成 draft 的固定链。
- **加分（SHOULD，做好更好）**：动态 draft 树 + 接受余量调度器 + 训练 draft 头。
- **可选（COULD）**：FP8、flash-decoding 预填充、int2 KV、小论文。

**核心纪律：只对保底做承诺，加分项做砸不伤骨架。** 这直接回应了评估里"过度承诺是最大扣分点"的提醒。

---

## 8. 需要你确认的一件事

v3 把**投机解码从"固定猜 4 个词"升级成"动态 draft 树 + 调度器"**，并把研究问题定位成"**长上下文下投机解码何时失效、KV 量化能否延长**"。这是两个顾问 agent 认为最"创新、可落地、能冲高分"的升级。

请确认：**这个"动态树 + 长上下文"的新方向，是不是你想要的？** 如果"是"，我们就按 `specdec_ai_runbook.md`（已同步）开工；如果还想调整，告诉我哪里不对。
