# 对齐方法学（Alignment Methodology）—— 我们如何证明"引擎是对的"

> 适用：理解 L1 对齐(0.000)、缓存等价、生成对齐、分块等价、G1(1.5B) 背后的共同逻辑。
> 配套阅读：`engine_guide_for_team.md`（代码）、`debug_lesson_alignment.md`（一次实战）、`data_ledger.md`（数据）。

---

## 1. 什么是对齐，为什么它如此重要

**定义**：让我们的引擎（OURS）与一个可信的参考实现（HF transformers 的 eager 全量前向）在**同一输入、同一权重、同一数值环境**下输出一致——通过比较中间/最终张量来验证我们的实现数学正确。

**为什么重要（层层地基）**：
- 没有对齐，任何后续测量（FP8 KV、SpecDec、正式 3×2×2）都建立在"可能是错的引擎"上——错误会被放大且无法归因；
- 对齐把"实现的正确性"与"实验的科学性"解耦：先证明"车是对的"，再测"油耗"；
- 它也是课程"Technical Implementation (20%)"与"Experimental Design (25%)"的基础证据。

## 2. 核心思想：黄金参照差分测试

我们从不直接问"这一步对不对"（那太难定义），只问：
> **同一个输入，两边从哪个时刻开始不一样？**

参考实现（HF eager）是"黄金答案"。我们的工作 = 让两边的**结构、权重、输入、数值环境、运算顺序**逐一对齐，于是剩下唯一的差异源就是"我们的实现 vs 官方实现"，可比时才有意义。

## 3. 对齐体系：五层验证（各自的"证什么"）

| 层 | 测什么 | 判据（来自 ledger） | 在哪个模型 |
|---|---|---|---|
| **L1 前向对齐** | 整段前向 logits == HF | worst max|Δ| = **0.000**（fp32 CPU eager）| 0.5B（dev）|
| **缓存等价** | 缓存版解码 == 全量版 | ~2e-5 + argmax 一致 | 0.5B/随机小模型 |
| **生成对齐** | 整条 greedy 序列 == HF 全量贪心 | 逐 token 相等（near-tie 容差）| 0.5B（dev）|
| **分块等价** | 分块 prefill == 一次性 prefill | 行为等价（fp32 隔离 7.8e-4）| 0.5B/4K |
| **G1（1.5B）** | 前三件在 **target=1.5B** 重跑 | full<1e-3；cached<1e-3+argmax；生成相等 | **1.5B（target）** |

**原则**：0.5B 数据标为 development-stage validation；target(1.5B) 必须单独 G1 通过——不拿 0.5B 冒充 1.5B。

## 4. 五要素：什么必须一致才"可比"

1. **结构一致（命名契约）**：state_dict 键 = 属性路径。我们的 `embed_tokens/layers/norm/q_proj/gate_proj/…` 与 HF 键一一对应 → `load_state_dict` 后 missing/unexpected=[]。类名无所谓，键名说了算。
2. **同一权重**：从 HF checkpoint 原样拷贝（tie 时去掉 lm_head，用 embed 绑定打分）。
3. **同一输入**：同一个 tokenizer、同一段 ids——两边吃一模一样的 token 序列。
4. **同一数值环境**：fp32；`attn_implementation="eager"`（避免 HF 用 FlashAttention 换算法）；`eval()`（无 dropout）；CPU（确定性优先）。
5. **同一运算顺序**：这是"逐位 0.000"的前提；顺序不同则退到"~1e-5 级 + argmax 一致"。

## 5. 判据的量级学（怎么读差异）

| 量级 | 含义 | 处理 |
|---|---|---|
| 0.000 | 运算路径与官方逐位一致（最强）| 通过 |
| ~1e-5~1e-3 | 同数学、运算顺序/布局差异（fp kernel 顺序噪声）| 通过（配 argmax 或 margin）|
| ~0.1–几 | 结构/数值细节错（早期 rope_theta、bias 的教训）| 二分定位 |
| ~10–30 | 结构性缺失（缺 bias 等整体常数）| 查键名/参数 |
| NaN/巨大 | mask/softmax/维度错 | 查数学 |

**关键经验**：
- 位级(0.0)不是必须——不同 matmul 布局会引入 ~1e-5 噪声（如缓存 vs 全量）；判据用"容差 + 行为一致（argmax/生成序列）"；
- near-tie：某位置 top2 margin 极小（<0.01）时翻边是 fp 噪声，不是 bug；margin 大仍翻边才是真错。

## 6. 参考（裁判）选择原则：必须等于你声称要复刻的对象

- 我们声称复刻的是**模型前向数学** → 裁判 = HF **全量前向**的逐 token 贪心；
- **不要用 `hf.generate` 当裁判**：实测它与其自身全量前向不一致（官方 logits 第一是 ' is'，generate 却吐 ' was'）——它带不可见管线；
- 外部库（transformers/vLLM）的角色 = correctness reference / baseline sanity（证明我们的 eager 基线不病态慢），**不是主结论**（不写"我们打败了 transformers"）。

## 7. 踩坑实录（都是对齐教出来的）

1. **rope_theta**：Qwen2.5 = 1e6，且被 transformers 折叠进 `rope_parameters`——从顶层 config 读会取到默认 1e4 → 频率表全错（误差 1~4，top1 却对）。教训：**别信默认值，数值对账定罪**。
2. **attn_bias**：该 checkpoint attention 带 bias → 三层未透传 `attn_bias` 时 bias 全缺（误差 ~20–30，unexpected 键满屏）。教训：**missing/unexpected 是第一个探针**。
3. **fp16 缓存漂移 0.55 vs fp32 7.8e-4**：fp16 下不同矩阵布局的取整逐层累积会造成 K/V 漂移但 token 行为不变；用 **fp32 隔离实验**把"取整 vs 逻辑 bug"分开。
4. **near-tie 翻边**：生成序列在模型"犹豫"处（margin ~0.4–2）可能因 fp 噪声选不同词 → 用 margin 分类而不是直接 FAIL。
5. **命名契约即加载**：属性名不对 → missing/unexpected；class 名无所谓。

## 8. 对齐出问题时的定位流程（回溯 debug 教学）

```
1) 打印 missing/unexpected → 命名/参数问题
2) 看量级定类别（结构 vs 细节）
3) 逐层 hidden 对账 → 找"第一层分叉点"
4) 层内子模块对账（norm/attn/mlp/各 proj）
5) 语义边界量对账（attention 概率、频率表、cache）
6) 翻官方源码 + 读真实 config（别信默认值）
7) 修完跑全链路回归（0.000 / PASS），确认是"修复"而非"看起来好"
```

## 9. 最佳实践清单（给协作者/报告）

- [ ] 每改引擎/数值路径 → 跑 L1(0.000) + 缓存等价回归；
- [ ] 换 target 模型 → 重跑 G1（不沿用旧模型验证）；
- [ ] fp16/GPU 上的差异：先用 fp32 隔离判断"噪声 vs bug"；
- [ ] 结论一律"同模型/同种子/同评测/一条命令可复现"；
- [ ] 如实区分：implementation achievement vs research contribution；位级一致 vs 行为一致。

## 10. 一句话总结

> **对齐 = 让"结构、权重、输入、数值环境、运算顺序"逐一对齐后，只比较实现数学与官方数学；用"量级 → 二分 → 数值对账"定位差异，用"容差 + 行为一致"定通过——它是本项目的可信度地基，也是把"手搓引擎"变成"可解释的研究"的关键方法论。**

数据见 `data_ledger.md`；G1(1.5B) 通过后其三条记录将追加到 ledger §3.5。
