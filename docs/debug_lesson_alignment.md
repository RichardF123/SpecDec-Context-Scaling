# 调试教学：L1 对齐门通关记 —— 一次完整的分层定位实战

> 目标：让"从零手搓的引擎"和 transformers 的 Qwen2 输出一致（max|Δ logits| < 1e-3）。
> 本文用真实过程示范：**每次只看一个症状 → 用一个实验砍掉一半嫌疑 → 用证据说话**。

---

## 总纲：调试的六条心法

1. **有黄金参照，就用差异测试**：不问"对不对"（太难），只问"两边哪里开始不一样"（可机械化）。
2. **症状量级 = 第一层诊断**：误差大小本身告诉你"哪种错"。
3. **二分法**：只对比最终结果 = 只知道错了；逐层、逐段、逐值对比 = 把嫌疑区每次砍半。
4. **一个实验只回答一个问题**：探针的判据要能二选一。
5. **探针只读不改**：用 forward hook、output_attentions、手工复刻截中间量，不污染源码。
6. **源码是最终权威**：推理到细节时，翻开官方实现逐行对读。

---

## 第 0 阶段：症状出现前的纪律

写引擎每块时都配**独立自验**（RMSNorm 手写公式对账 diff≈1e-6、MLP 对账 0.0、RoPE 形状+保长度、attention 形状+因果泄漏=0.0）。
> 教训：**每块先自验、再组装**，组装后出问题就能排除"小零件没写对"。

---

## 第 1 战：max|Δ| = 27.87（结构性缺失）

**证据**
```
unexpected: ['layers.0.self_attn.q_proj.bias', ... ×72]
missing:   []
5 个 prompt 全部 top1=False，max|Δ|≈20~30
```

**推理链**
- 量级 20~30 = **结构性缺失**，不是细微误差（细微误差在 1e-3 量级）；
- `unexpected` = HF 有权重、你的模型没有 → **你的 q/k/v 投影没有 bias 参数**；
- bias 是"每个位置加的常数偏移"，全丢了 → 每个 logits 都差一截 → 20~30 合理。

**改动**
1. SelfAttention 加 `attn_bias` 参数，q/k/v 用 `bias=attn_bias`；
2. DecoderLayer / QwenLM 逐层透传 attn_bias；
3. 对账脚本从 HF 模型实测 `q_proj.bias is not None` 判定。

**结果**：max|Δ| 掉到 1.2~3.7，top1 **全对**。

---

## 第 2 战：max|Δ|≈1~4 但 top1 全对（细节差异 + 逐层放大）

**推理**
- top1 全对 → 整体结构对；
- 还有 1~4 且随层数增长 → 某个**小但系统性的差异**在 24 层里像复利一样放大；
- 下一步不问"差在哪一行"，先问"**第几层开始差**"。

**实验（align_debug）：逐层对比 hidden states**
```
embedding ?（没打印，假设 0）
L0  7.3e-2      ← 第一层就差了！
...
L23 1.5e+02
```
**结论**：不是"层数累积放大一个微小误差"——**第一层自己就有 bug**。病灶在层内。

---

## 第 3 战：病灶锁定到"注意力内部"

**实验（align_debug_layer0）：给第一层的每个子模块装探针（hook），逐段对账**
```
input_layernorm          0.000     ← norm 没错
q_proj / k_proj / v_proj  0.000     ← 投影、权重、bias 全对
self_attn (o_proj 后)    4.7e-03    ← 从这里开始有差异
post_attention_layernorm  0.316     ← 被小范数行放大
mlp                      7.2e-02
```
**结论**：q/k/v 投影之后、o_proj 输出之前的**注意力内部运算**有问题（RoPE / GQA 展开 / scores / mask / softmax 之一）。

---

## 第 4 战：在注意力内部再画分界线

**思路**：softmax 输出的"注意力概率"是天然分界线——之前错 → 概率错；之后错（加权 v / 拼接 / o_proj）→ 概率对但输出错。

**实验（align_debug_attn）①：手工复刻你的 forward 截出概率，对比 HF 的 `output_attentions=True`**
```
attn probs  max|Δ| = 3.9e-01     ← 概率就错了！
per-head: 0.232 0.000 0.018 ...  ← 分散、不成组
```
**推理**
- 概率错 → 问题在分界线之前（RoPE/scores/mask/展开）；
- per-head 分散（不按 7 个一组错）→ **不是 GQA 分组错位**。

**实验 ②：kv_repeat 对账**
```
kv_repeat == canonical(expand+reshape): True    ← GQA 展开无辜
```
**结论**：嫌疑收敛到 **RoPE**。

---

## 第 5 战：源码对读（最终权威）

翻开 venv 里 HF 的 Qwen2 源码 `modeling_qwen2.py`，逐段对照：
- 评分缩放 `scaling = head_dim**-0.5` → 和你一样 ✅
- repeat_kv 分组展开 → 和你一样 ✅
- fp32 softmax + mask 加法 → 语义等价 ✅
- RoPE 的 cos/sin、rotate_half、unsqueeze → 和你一样 ✅
- `attention_scaling = 1.0`（default rope）→ 不是它

**代码逐行都对得上，但结果不对** → 那问题不在公式，而在**喂给公式的数**（theta / 频率表 / 位置）。

---

## 第 6 战：对比频率表，一锤定音

**实验：读 HF rotary 模块里的 `inv_freq` 缓冲，和你的逐值对比**
```
HF  : [1.0, 0.649, 0.422, 0.274]
ours: [1.0, 0.750, 0.562, 0.422]
inv_freq max|Δ| = 1.5e-01
```
**推理**：每步衰减比 HF 慢 → 两个频率表用的是**不同的 theta**。

**查证**：读缓存的 config.json
```json
"rope_theta": 1000000.0
```
**真相**：Qwen2.5 的 rope_theta = **1e6**（不是 1e4）；且新版 transformers 把这个键**折叠进 `rope_parameters` 子字典**，从 `config.to_dict()` 顶层读不到 → 对账脚本取了默认 10000。

**修复（一行）**：从 `rope_parameters["rope_theta"]` 读，再传给引擎。

---

## 结局与复盘

**引擎一行没改；改的是"喂给引擎的超参读取"**。
- 你的 forward/RoPE/attention 全部正确；
- 真正错误 = 用 θ=1e6 训练的权重 + θ=1e4 的角度表旋转 → 位置信息全错。

### 通用收获

| 教训 | 对应动作 |
|---|---|
| 量级判断错误类别 | 27→结构缺件；1~4→细节放大；1e-3→近乎通过 |
| 二分到"组件/子步/数值" | 层→子模块→概率→频率表，每级一个探针 |
| 不要只信代码"看起来一样" | 最终用**数值对账**（inv_freq 前 4 个数）定罪 |
| 配置值要实测、别用默认 | rope_theta 藏在 rope_parameters 里，默认值会坑人 |
| 探针 = 只读传感器 | hook / output_attentions / 手工复刻，不污染源码 |

### 下次 debug 可以直接套用的清单
1. 打印 `missing / unexpected` —— 命名/参数不齐一眼现形；
2. 看量级定类别（结构 vs 细节）；
3. 逐层 hidden 对账，找"第一层分叉点"；
4. 层内子模块对账（norm/attn/mlp/各 proj）；
5. 有语义边界的中间量对账（概率、频率表）；
6. 翻官方源码对读 + 读真实 config（别信默认值）；
7. 修完重跑全链路，确认 PASS 而非"看起来好多了"。
