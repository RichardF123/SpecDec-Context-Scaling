# L1 引擎全程代码教学（基于你亲手写的 model_self.py，128 行）

> 这份文档把你磁盘上的 `specdec/engine/model_self.py` 从头教到尾：
> 每段代码 = 数学是什么 + 形状怎么走 + 为什么这么写 + 它被哪个测试验证。
> 读完你能"指着任何一行说出它在干嘛"。

---

## 0. 总览：文件地图 + 一条守恒律

```
imports
  RMSNORM         归一化（每个位置的 896 维拉到统一尺度）
  Embedding       入口：词编号 → 向量（自己存词表矩阵 + 索引）
  MLP             SwiGLU 门控（gate×up，扩到 4864 再收回）
  Build_Rope_cache / rotate_half / apply_rope   RoPE 位置编码
  kv_repeat       GQA：2 个 kv 头 → 14 个（分组展开）
  SelfAttention   注意力（q/k/v 投影→RoPE→展开→打分→mask→softmax→加权→拼接）
  DecoderLayer    一层 = norm→attention(残差) + norm→mlp(残差)
  QwenLM          整机 = embed + 24 层 + 最终 norm + 绑定打分
  __main__        冒烟测试
```

**守恒律**：Embedding 之后、lm_head 之前，所有模块进出最后一维都是 `hidden`（896）——attention/MLP 只在内部临时换维，进出保证还原。记住这条，看任何一行都不迷路。

---

## 1. imports（4 行）
```python
import torch          # 张量与运算
import torch.nn as nn # 神经网络模块库（nn.Module/nn.Linear/nn.Parameter）
import torch.nn.functional as F  # 纯函数算子（silu/softmax）
import math
```
`math` 目前没用上（√head_dim 你用了 `**0.5`）——用 `math.sqrt` 也可以，二选一，别留死代码（可删）。

## 2. RMSNORM（5–14 行）
```python
class RMSNORM(nn.Module):
    def __init__(self, hidden, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))   # 896 个"旋钮"，可训练
        self.eps = eps
    def forward(self, x):
        variance = torch.mean(x * x, dim=-1, keepdim=True)   # 沿最后一维求均方
        normed = 1 / torch.sqrt(variance + self.eps)          # 归一化系数
        return normed * x * self.weight                        # 乘原值再乘旋钮
```
- 数学：`y = x / √(mean(x²)+ε) * w`；
- `keepdim=True` 让系数保留 `(…, 1)` 维度，才能和 `(…, 896)` 广播相乘；
- **验证**：随机输入 vs 手写公式对账（~1e-6）+ 输出平方均值≈1。

## 3. Embedding（16–22 行）——你自己写的"真·查表"
```python
self.weight = nn.Parameter(torch.randn(vocab_size, hidden))  # 词表矩阵：每行一个词向量
def forward(self, token_ids):
    return self.weight[token_ids]                            # 索引=取行 → (B,S,hidden)
```
- `weight[token_ids]` 就是查表的全部原理（nn.Embedding 底层做的也是这张表）；
- 参数名 `weight` + 整机属性名 `embed_tokens` → 权重键恰好等于 HF 的 `embed_tokens.weight`（对齐能成功的原因之一）。

## 4. MLP（24–35 行）——SwiGLU
```python
gate = F.silu(self.gate_proj(x))   # 闸门：silu = x·sigmoid(x)
up   = self.up_proj(x)             # 内容
down = self.down_proj(gate * up)   # 开度×内容 → 压回 896
```
- 数学：`down(silu(xW_gᵀ) ⊙ xW_uᵀ)`；三个 Linear 各是独立参数；
- **验证**：与手写公式对账（0.0）。

## 5. RoPE 三件套（37–56 行）
```python
def Build_Rope_cache(head_dim, seq_len, theta, device, dtype):
    exponents = arange(0, head_dim, 2) / head_dim     # 偶数分母 2i/d → 32 个
    inv_freq  = 1.0 / (theta ** exponents)            # 32 把"不同转速的尺子"
    positions = arange(seq_len)                        # 0…S-1
    freqs = torch.outer(positions, inv_freq)           # (S,32)：每 token 每尺子的转角 m·θᵢ
    cos = cat([cos(freqs), cos(freqs)]).unsqueeze(0).unsqueeze(0)  # 复制成 64 宽 → (1,1,S,64)
    sin = 同理
def rotate_half(x): return cat([-x[..., h:], x[..., :h]])   # 逐对旋转的向量化写法
def apply_rope(x, cos, sin): return x * cos + rotate_half(x) * sin
```
- 配对 (i, i+h) 共享同一频率 → cos/sin 必须**复制成两半**；
- `(1,1,S,64)` 乘 `(B,H,S,64)`：S、64 对齐，B/H 自动广播；
- 旋转不改向量长度 → 自验"norm preserved"；**theta 必须从 config 读**（Qwen2.5=1e6，藏 rope_parameters 里——你踩过的坑）。

## 6. kv_repeat（58–62 行）——GQA 展开
```python
x.unsqueeze(2).expand(b, kv, n_rep, s, d).reshape(b, kv*n_rep, s, d)
```
- 插空维 → 假扩展（expand 不复制）→ 拍平 → 每个 kv 头**整组复制 7 份**；
- 分组顺序 = q 头 0–6 用 k₀、7–13 用 k₁（与 HF 相同）。

## 7. SelfAttention（64–90 行）——最大的一块，逐行
```python
q = q_proj(x).view(b,s,heads,head_dim).transpose(1,2)   # (B,H,S,D)：头进"批量区"
q = apply_rope(q, cos, sin);  k 同理                    # 位置贴到 q/k 上
k = kv_repeat(k, heads//kv_heads); v 同理               # 2→14 头
scores = q @ k.transpose(-2,-1) / √head_dim             # QKᵀ/√d → (B,H,S,S)
mask = triu(ones(s,s,bool), diagonal=1)                 # 上三角=True=未来
scores = scores.masked_fill(mask, -inf)                 # 封未来（必须在 softmax 前）
attn = softmax(scores, dim=-1)                          # 每行=注意预算，和=1
out = attn @ v                                          # 按预算取货
out = out.transpose(1,2).contiguous().view(b,s,heads*head_dim)  # 拼回 (B,S,896)
return self.o_proj(out)                                 # 融合多头
```
- **q/k/v 投影宽度不同**：q→896（14 头）、k/v→128（2 头）= GQA 的标志；
- **形状口诀**：H 进批量区、S 进数学区（否则 q·k 变成"自己头的自嗨"）；
- mask 用 `masked_fill(-inf)`：exp(-inf)=0，未来精确排除；
- 拼接三步：transpose 换轴 → contiguous 重排内存 → view 拍平头优先；
- **验证**：形状 (2,16,896) + 因果泄漏 0.0（改最后一个 token 前面不变）。

## 8. DecoderLayer（92–103 行）
```python
x = x + self.self_attn(self.input_layernorm(x), cos, sin)      # 开会，残差
x = x + self.mlp(self.post_attention_layernorm(x))             # 思考，残差
```
- pre-norm：每个子层前先归一化；两个 norm 是**两个独立实例**（各有自己的 weight）；
- 残差 `x + f(x)`：梯度抄近道 + 只学增量；顺序固定"先交流再思考"。

## 9. QwenLM 整机（105–120 行）
```python
self.embed_tokens = Embedding(vocab_size, hidden)          # 入口（名字对齐 HF）
self.layers = ModuleList([DecoderLayer(...) for _ in range(num_layers)])  # 24 层
self.norm = RMSNORM(hidden)                                 # 出口前总整理
def forward(self, token_ids):
    x = self.embed_tokens(token_ids)                         # (B,S,896)
    cos, sin = Build_Rope_cache(...)                         # 位置表只建一次
    for layer in self.layers: x = layer(x, cos, sin)         # 24 层，形状不变
    x = self.norm(x)
    return x @ self.embed_tokens.weight.T                    # 绑定打分 (B,S,vocab)
```
- cos/sin 建一次、传给所有层（同一位置每层转的角度一样）；
- tied：打分直接用词表矩阵转置（省一套 151936×896 参数，键名也不用 lm_head）。

## 10. __main__ 冒烟（121–128 行）——顺便指出一个习惯问题
```python
if __name__ == "__main__":
    torch.manual_seed(0)        # ← 只有这一行在 if 里（缩进只到这行）
model = QwenLM(...)             # ← 下面的代码在 if 外面（顶格）！
...
print("logits shape:", ...)
```
- **问题**：测试块只有第一行被 `if __name__ == "__main__"` 保护，其余都在模块顶层——所以**别人 import 你的文件时也会跑冒烟测试**（你在 align 脚本运行时看到 `logits shape/no NaN` 打印，就是这个原因）；
- **修法**：把 123–128 行整体缩进进 if 块。这属于代码卫生，不影响正确性，但会让 import 干净。

---

## 附 A：这套代码怎么被验证的（对齐 + 自验）

| 验证 | 脚本 | 看什么 | 过了吗 |
|---|---|---|---|
| 每个小件自验 | model_self.py 冒烟 | 形状 + 手写公式对账 | ✅ |
| 因果性 | 冒烟（改尾 token） | 泄漏 0.0 | ✅ |
| **官方对齐** | align_ours.py | missing/unexpected 空 + max|Δ| | ✅ **0.000** |

## 附 B：代码质量小建议（不阻塞，纯锦上添花）
1. 类之间空一行、统一命名（RMSNORM 大写 OK，但建议保持 PEP8 风格）；
2. 把冒烟整体收进 `if __name__`；
3. `import math` 不用就删；
4. 变量名 `postitions`（42 行）拼写可改 `positions`；
5. 将来把 `head_dim`、`rope_theta` 存进 config 对象统一管理，别散落传参。

## 附 C：这段代码=你项目的地基
- 它对 **L2（KV-cache）** 的改造点：SelfAttention 里加"缓存读写"、QwenLM 支持逐 token 输入 + 位置偏移；
- 它后面会被 L4 量化、L4b KV 量化、L5 投机解码"包起来"——但数学主体不变；
- 0.000 对齐 = 你对"引擎每一行都正确"的官方级证据。
