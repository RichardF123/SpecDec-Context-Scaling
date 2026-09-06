# SpecDec 项目阶段总结报告（L1 + L2 完成）

> 覆盖：立项决策 → 规划（v3–v6）→ 环境 → L1 从零引擎（0.000 对齐）→ L2 KV-cache 与生成（对齐 PASS）→ 第一个性能基线。
> 另附：全程"值得回顾"的对话精华清单。

---

## 1. 项目是什么（30 秒版）

在 **RTX 4060 8GB 笔记本**上**从零实现**一个小型 LLM 推理引擎（Qwen2.5-0.5B/1.5B 架构），并沿优化阶梯做 **KV-cache → 量化 → 投机解码**，用可复现测量表回答："投机解码随上下文增长何时失效、KV 量化能否延长其获利区间"（方向 c+d；原创、hands-on 课程项目；10/7 提案 → 11/25 结题）。

## 2. 决策历程（值得回顾：我们怎么走到这一步的）

| 阶段 | 结论 | 关键理由 |
|---|---|---|
| 方向否定 RAG（ClimateCite） | 放弃 | 用户要"推理/训练效率 + 亲手实现" |
| 数据配方（ReasonLite）候选 | 放弃 | 偏抽象、非"实现一个模型" |
| **SpecDec（推理引擎 + 优化）** | **选定** | 满足：实现型 + 推理优化 + 8GB 可行 + 可测量 + 可冲分 |
| 多轮 agent 评审（A 改善 / B 评估 / 时间审计） | 采纳 | 砍 Triton 主线、CUDA 轨降为加分、MUST 保底纪律、M1 归因 |
| 导师 systems 关切 | CUDA/C++ 纳入 | 外科手术式 2 kernel + G0 冒烟 + 三级降级链（v6）|
| 时间审计 | 排序反转 | 纯 PyTorch MUST 先行（≈190h），CUDA = 尾巴加分 |

权威文档：`specdec_plan_v6.md`（当前唯一权威）/ `specdec_time_feasibility_report.md` / `specdec_plan_v5_evaluation.md` 等。

## 3. 环境搭建（踩坑全记录）

- Python 3.9（无 torch）→ 装 **3.11** + venv；conda 缺失 → 用 venv；
- torch：误装 CPU 版（204MB）→ 认出 **+cu124 必须 ≥2GB** → curl 断点续传 2.35GB → `setx TEMP/TMP → D:` 治 WinError 32；
- HF 缓存 C 盘满 → `HF_HOME=D:\hf-cache` + `HF_ENDPOINT=hf-mirror`；
- VS Code：Python 扩展未生效（受限模式/未重载）→ Trust + Reload；解释器选 `.venv`。

**教训**：CUDA 版 wheel 必带 `+cu124`；临时目录重定向；缓存指盘；扩展装完要 Reload。

## 4. L1 里程碑：从零引擎（0.000 对齐）

### 代码（用户亲手写，AI 只讲解/审阅/建测试）
| 部件 | 验证 |
|---|---|
| RMSNorm / Embedding(自写查表) / SwiGLU MLP | 手写公式对账（~1e-6 / 0.0）|
| RoPE（start_pos 前版本）| 保长度 |
| GQA 因果注意力 | 因果泄漏 0.0 |
| DecoderLayer + QwenLM(24 层, tied) | 冒烟 |
| **L1 对齐门** | **max|Δ|=0.000 PASS** |

### L1 调试实战（写成教学：`docs/debug_lesson_alignment.md`）
| 症状 | 根因 | 修法 |
|---|---|---|
| max|Δ|≈28 + unexpected bias×72 | q/k/v 缺 bias（三层未透传 attn_bias）| attn_bias 全链透传 |
| 降到 1~4 且 top1 对 | 逐层二分 → 概率对账 → inv_freq 前 4 个数不同 | **rope_theta 真实值 1e6 藏在 `rope_parameters`**，取数修正 |

**方法论沉淀**：量级定类别 → 二分（层→段→概率→频率表）→ 数值对账定罪 → 翻官方源码 → 别信默认值。

## 5. L2 里程碑：KV-cache + 生成（对齐 PASS）

| 任务 | 内容 | 验证 |
|---|---|---|
| T1 | RoPE `start_pos`（decode 绝对位置）| start_pos 切片等值 |
| T2 | SelfAttention 缓存（concat/mask 分支/返回 2 头档案）| 缓存==全量 ~2e-5、预测一致 |
| T3 | DecoderLayer/QwenLM `_run` 透传；forward 薄壳 | L1 回归仍 0.000 |
| generate | prefill + 循环（argmax/EOS/拼接/回喂 1 词）| — |
| **生成对齐门** | vs "HF 全量逐步贪心" | **4/4 整条序列一致 PASS** |

### L2 关键发现（写报告素材）
1. **`hf.generate` ≠ 模型全量前向**：官方 logits 第一 ' is'=21.75，`hf.generate` 却吐 ' was' → 裁判改为纯模型数学（HF full-forward 循环）。**教训：裁判要选"声称复刻的对象"。**
2. **缓存路径与全量差 ~2e-5 属 fp kernel 顺序噪声**（逐层 ~1e-7 且不放大 = 正常；业界容差 1e-3）。判据比"位级 0.0"更工程化。

### L2 性能基线（0.5B，初步数字；正式协议见 benchmark 阶段）
```
CPU fp32 : prefill 93 tok/s（38 tok 约 409ms）；decode 9.20 tok/s（约 109 ms/token）
GPU fp16 : prefill 398 tok/s；decode 10.58 tok/s（约 95 ms/token）
```
- decode 是 latency-bound（单 token 串行 + 24 层小 kernel + Python 开销），GPU 相对 CPU 提升有限——**这是朴素解码基线，正是后续量化/kernel/投机解码要对比的起点**；
- prefill 并行度高，GPU 优势明显（398 vs 93 tok/s）。

## 6. 概念收获（贯穿对话、值得回看的知识点）
1. **transformer 前向全程守恒 hidden**，只有两端换维（词表↔向量）；
2. **RoPE 用旋转不用加法**：R_mᵀR_n=R_{n−m} → 点积只剩相对位置；多频率=多把尺子防混叠；
3. 只旋转 **q/k**（位置属于"匹配"，不属于内容 v / MLP）；
4. **mask 用加 -inf 而非乘 0**（exp(-inf)=0 才精确排除）；必须在 softmax 前；
5. **GQA**：14 q 头共享 2 组 k/v → 缓存小 7 倍（L2/L4b 叙事的根基）；
6. 两个 norm = 两个子层各配"入口整理"，残差让每层只学增量、梯度抄近道；
7. **命名契约**：state_dict 键 = 属性路径，类名无所谓——对齐/加载全靠它；
8. **matmul 只算最后两维** → 头进"批量区"、S 进"数学区"（否则是"自嗨"）；
9. **KV-cache 的核心只是记忆化**（历史 K/V 因因果+固定权重永久有效）——思想 3 行，复杂在工程；
10. **调试指标词典**（见 debug 教学）：missing/unexpected → 量级 → 分层 → 概率 → 数值定罪；
11. **near-tie vs 真错**：margin < 容差的翻边是 fp 噪声，可接受。

## 7. 方法论约定（长期有效）
- AI = 讲解/审阅/测试工具；**引擎代码全部用户亲手写**（原创与学习）;
- OURS vs LIB 边界：数学/缓存/生成/kernel=OURS；权重加载/tokenizer/评测 harness=LIB（注明）;
- 只对 A 层承诺（v6 纪律）；诚实负结果；固定种子+config 哈希+只追加 results 的原则已建立。

## 8. 当前状态与进度

| 阶段 | 进度 |
|---|---|
| 规划/方案（v3–v6/评估/审计/导师） | ≈95%（10/7 提案文本待写）|
| 环境 | ≈100% |
| **L1 引擎 + 对齐** | **100%** |
| **L2 KV-cache + generate + 对齐** | **100%（含 CPU/GPU 计时基线）** |
| L4/L4b/L5/基准/M1/报告/CUDA 轨 | 0%（下一站 L4）|
| **整体** | **≈38%** |

## 9. 交付物地图（工作区）
- 规划：`specdec_plan_v6.md`、`specdec_design_v4.md`、`specdec_technical_proposal_v2.md`、`specdec_time_feasibility_report.md`、`specdec_evaluation_report.md`、`specdec_plan_v5*.md`
- 代码：`specdec/specdec/engine/model_self.py`（自写引擎）、`engine/align_ours.py`、`test_cache_equiv.py`、`align_generate.py`、`bench_basic.py`、debug 探针若干
- 教学/复盘：`docs/debug_lesson_alignment.md`、`docs/L1_code_teaching.md`、`docs/daily_report_*`

## 10. 下一步（建议顺序）
1. 修 GPU bench（最小探针 + nvidia-smi）→ 拿 GPU decode tokens/s；
2. commit（L2 PASS 存档）；
3. 修 `model_self.py` 末尾 __main__ 残留；
4. **L4 int8/int4 权重量化**（A 层 MUST）→ L4b fp8 KV → L5 固定链投机解码 → 2K/4K/8K 曲线（11/14 冻结）→ M1 归因 → 报告；
5. 旁路：L3/CUDA K1（G0 冒烟）只在主线绿后作为加分。

---

## 附：对话中值得回顾的"高光时刻"
- "KV-cache 怎么这么简单"——从害怕到看懂，记忆化的力量；
- "不是我自己啥也没写啊"——确立 AI 只讲解、代码自己写的边界（项目灵魂）；
- debug 全程 6 战（27→0.000）——"症状量级→二分→数值定罪"的实战演示；
- "hf.generate 和它自己都不一致"——找到正确裁判的那一刻；
- 每个"为什么"的讲解（RoPE 相对位置、SwiGLU 闸门、为什么两个 norm、为什么 (B,H,S,d)）——这些是你演示/报告的底气。
