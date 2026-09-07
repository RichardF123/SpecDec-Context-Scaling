# 引擎代码全解析（给协作者看）

> 配合 `specdec/engine/model_self.py` 阅读：本项目自写的从零推理引擎（L1 前向 + L2 KV-cache/生成 + 分块 prefill）。
> 约定：OURS=我们写的（本文件全部）；LIB=借的（HF 仅加载权重/tokenizer、测试 harness）。
> 正确性证据：与 transformers 对齐 **0.000**；缓存等价/生成对齐/分块等价均 PASS。

---

## 0. 全局约定（看任何函数前先记两条）

1. **形状守恒律**：token 进来变成 `(B, S, hidden)` 后，中间所有模块进出最后一维都是 `hidden`（896）；只有开头（词表→向量）和结尾（向量→词表打分）换维。
2. **matmul 只算最后两维**：attention 要把"头"放进批量区、把"序列 S"放进数学区 → 形状 (B, H, S, d)，而不是 (B, S, H, d)。

**符号表**：B=批大小、S=序列长度、hidden/`hidden`=词向量维度(896)、H=`heads`=注意力头数(14)、kv=`kv_heads`=KV 头数(2)、d=`head_dim`=每头维度(64)、T=当前全部 key 数、V=`vocab_size`=词表(151936)。

---

## 1. RMSNORM —— 归一化

- **做什么/为什么**：每个位置的向量尺度不一，喂 attention/MLP 前拉回统一尺度（数值稳定）；乘可训练 `weight` 让模型自己调"每维缩放"。
- **输入**：`x (…, hidden)`；**输出**：同形状。
- **变量**：`variance`=(…,1) 沿最后维的平方均值；`normed`=1/√(variance+ε)；`self.weight (hidden,)` 可训练参数；`eps` 防除零（1e-6）。
- **公式**：`y = x / √(mean(x²,last)+eps) * weight`

## 2. Embedding —— 词编号→向量（自己写的查表）

- **做什么/为什么**：词编号是离散整数，不能直接算；查表取每词一行向量。
- **输入**：`token_ids (B, S)`，元素 ∈[0,V)；**输出**：`(B, S, hidden)`。
- **变量**：`self.weight (vocab_size, hidden)` —— 词表矩阵（可训练，参数名 weight 是为对齐 HF 键名）；`return self.weight[token_ids]` = 花式索引取行。

## 3. MLP（SwiGLU）—— 每个词独立"消化"

- **做什么/为什么**：attention 负责跨词交流，MLP 让每个词把收集到的信息加工一遍（非线性容量）；"先扩到 4864 想得开，再压回 896"。
- **输入**：`x (B, S, hidden)`；**输出**：`(B, S, hidden)`。
- **变量**：`gate_proj/up_proj: hidden→intermediate`；`down_proj: intermediate→hidden`（皆 bias=False）。
- **公式**：`down(silu(xW_gᵀ) ⊙ xW_uᵀ)` —— gate 分支过 SiLU 当"闸门"，up 分支给"内容"，逐元素相乘。

## 4. RoPE 三件套 —— 位置编码（旋转式）

- **做什么/为什么**：transformer 对词序麻木；把"绝对位置"以旋转角编码进 q/k，使 Q·K 只依赖相对距离 (m−n)。多频率=多把尺子，远近距离都能分辨。
- **Build_Rope_cache(head_dim, seq_len, theta, device, dtype, start_pos=0)**
  - **输入**：head_dim(64)、要建几个位置的表(seq_len)、转速基数 theta(1e6)、设备/精度、**start_pos**（这批 token 的绝对位置起点，decode/分块时=缓存长度）。
  - **输出**：`cos, sin (1,1,seq_len,head_dim)`。
  - **变量**：`exponents (32,)`=[0,2/64,…,62/64]（公式的 2i/d）；`inv_freq (32,)`=1/θ^exp（32 个转速）；`positions (seq_len,)`=start_pos..start_pos+seq_len−1；`freqs (seq_len,32)`=outer(位置,转速)=每个 token 每把尺子的转角；cos/sin 沿最后维**复制成 64**（rotate-half 配对共享频率）再 unsqueeze 两次成 (1,1,S,64)（广播用）。
- **rotate_half(x)**：`x (…,d)` → 把后半取负拼到前半 `(-x2, x1)` —— 逐对旋转的向量化写法。
- **apply_rope(x, cos, sin)**：`x*cos + rotate_half(x)*sin` = 对最后一维每个向量按对旋转；B/H/S 由广播自动处理。

## 5. kv_repeat —— GQA 展开

- **做什么/为什么**：只有 2 组 KV，却要匹配 14 个 Q 头；把 2 头按**整组复制 7 份**成 14 头（不是交错）。
- **输入**：`x (B, kv, S, hd)`、`n_rep=heads//kv_heads`；**输出**：`(B, kv*n_rep, S, hd)`。
- **实现**：unsqueeze 空维 → expand 假复制 → reshape 拍平（分组顺序：k₀×7, k₁×7）。

## 6. SelfAttention —— 注意力（核心）

- **做什么/为什么**：每个词用 q"找什么"去匹配所有历史 key"你是谁"，按相关度 softmax 分配预算，加权取 v"内容"——一次"环顾历史、各取所需"。GQA 省 7 倍缓存；因果只准看过去；多头 14 视角最后 o_proj 融合。
- **输入**：`x (B,S,hidden)`、`cos,sin`、`cache=None 或 dict{k,v (B,kv,T,hd)}`。
- **输出**：`(out (B,S,hidden), new_cache {k,v (B,kv,新的T,hd)})`。
- **内部变量/形状**：
  - `q/k/v`：投影+`view(b,s,头,head_dim)`+`transpose(1,2)` → q (B,H,S,d)、k/v (B,kv,S,d)（k/v 宽只有 2 头 → GQA 标志）；
  - rope 加在 q/k 上（位置只属于"匹配"）；`concat`：有 cache 时历史拼前面（序列维 dim=2），历史不重算；
  - `kv_repeat` 展开到 14 头（**必须在 concat 后**）；`scores=q@kᵀ/√d` → (B,H,s,T)；
  - **mask 三分支**（同一公式的不同参数化）：
    - cache=None（整段/首块）：`triu(ones(s,s),diag=1)`（经典上三角，p=0）；
    - s==1（decode 单词）：不 mask（没有未来）；
    - cache 且 s>1（分块）：`triu(ones(s,T),diag=p+1)`，p=T−s=块前 key 数 —— 行 i（绝对位置 p+i）封掉列>p+i，历史列全可见；
  - `softmax(dim=-1)` 归一成预算；`attn@v`→(B,H,s,d)；transpose/contiguous/view 拼回 (B,S,H·d)；`o_proj` 融合多头。
  - 缓存存 **2 头**（kv_repeat 之前）——14 头只是计算临时展开；建议 `.detach()`（推理不背图）。
- **为什么是 (B,H,S,d) 不是 (B,S,H,d)**：matmul 只算最后两维；头要当批量、S 要参与跨词点积，否则变成"同一词的头自嗨"。

## 7. DecoderLayer —— 一层 = 开会+思考（残差 + pre-norm）

- **输入**：`x (B,S,hidden)`、`cos,sin`、`cache=None`；**输出**：`(x (B,S,hidden), new_cache)`。
- **代码顺序**：`attn_out, nc = self_attn(self.input_layernorm(x), cos, sin, cache)` → `x = x + attn_out` → `x = x + mlp(self.post_attention_layernorm(x))` → return。
- **为什么两个 RMSNORM 实例**：attention 和 MLP 是两个会改变数值的子层，各配一个 norm（两个独立 weight 参数）；残差 `x+f(x)` 让梯度抄近道、每层只学增量。

## 8. QwenLM —— 整机

- **部件**：`embed_tokens`（入口，属性名对齐 HF）、`layers: 24×DecoderLayer`、`norm`（出口前总整理）、`head_dim/rope_theta` 超参。
- **`_run(x, cache_list=None, need_logits=True)`**（统一执行体，prefill/decode/分块都走它）
  - **输入**：已 embed 的 `x (B,S,hidden)`；`cache_list`：24 个 dict 或 None；`need_logits` 开关。
  - **逻辑/变量**：`start_pos` = `cache_list[0]["k"].shape[2]`（已处理 token 数，prefill=0）→ 建本批位置的 cos/sin → 逐层传/收 cache → `x=self.norm(x)` → 若 need_logits 则 `logits = x @ embed.weight.T (B,S,V)`，否则返回 None。
  - **输出**：`(logits 或 None, new_cache_list(24×dict))`。
- **`forward(token_ids)`**：`x=embed(ids)` → `_run(x, None)`（need_logits 默认 True）→ 返回 logits。保持 L1 语义，对齐脚本用它。
- **`prefill_chunked(token_ids, chunk=1024)`**：把长输入切块，逐块 `_run(embed(piece), cache, need_logits=False)` → 返回**完整 cache_list**。用途：8K 等长上下文建缓存不 OOM（scores 从 S² 降到 chunk×累计长），且与一次性 prefill 行为等价（已验证）。
- **`generate(prompt_ids, max_new_tokens, eos_id=None)`**：prefill 一次 → 循环：取 `logits[:,-1].argmax` 得新词 → EOS 判断 → 拼进 gen → 把新词（1 个）embed 后带 cache 再 `_run` → 返回 `(1, 提示词+生成)` 的整条序列。

## 9. 三种调用模式（输入→输出速查）

| 模式 | 怎么调 | 输入 | 输出 | 用途 |
|---|---|---|---|---|
| 整段 prefill | `_run(embed(ids), None)` | (1,S,896) | logits(1,S,V) + 24 层缓存 | 对齐/短上下文建缓存 |
| 分块 prefill | `prefill_chunked(ids, 1024)` | (1,S) | 24 层缓存（无 logits）| 8K 长上下文 |
| decode 单步 | `_run(embed(1词), cache)` | (1,1,896) | logits(1,1,V) + 更新的缓存 | 生成/测量 |
| 生成 | `generate(ids, N)` | (1,S) | (1, S+N) token 序列 | 完整文本 |

## 10. 测试与验收（每个测什么、为什么）

| 测试 | 验证 | 关键数字 |
|---|---|---|
| `align_ours` | 我们的全量前向 == transformers（fp32 eager）| **0.000** |
| `test_cache_equiv` | 缓存解码 == 全量（小模型）| ~1e-5、预测一致 |
| `align_cache_real` | 缓存==全量（真权重、各深度）| ~2e-5 |
| `align_generate` | 生成序列 == 官方数学（HF 全量贪心）| PASS 4/4 |
| `test_chunked_prefill` | 分块 == 一次性 prefill（fp16）| 行为等价（缓存差 0.55 为取整累积）|
| `test_chunked_fp32` | 同上但 fp32（隔离舍入）| worst 7.8e-4 → 证实逻辑正确 |
| `bench_basic` / `probe_gpu` | 性能/显存基线 | CPU 9.2、GPU 10.6–13.8 tok/s；8K 需分块 |

**为什么"缓存差 0.55 也算过"**：fp16 下两种矩阵布局的舍入逐层累积产生 K/V 漂移，但 fp32 对照只有 7.8e-4 且 decode token 行为完全一致 → 归因为取整而非逻辑错（报告局限里如实写）。

## 11. 关键"为什么"速记（评审/演示最常问）

1. 为什么从零写引擎：研究问题（KV/量化/投机解码的收益）需要引擎内部控制权与计时插桩；
2. 为什么用 eager + 与 transformers eager 对齐：公平可比、可解释；基线有库参考背书；
3. 为什么量化/分块/掩码这样设计：见上文各节——每个决定都对应一个"内存/正确性"约束；
4. 为什么处处留测试：每个结论都要"同模型同种子同评测 + 可复现"才有意义（课程 25% 评测 + 20% 分析）。

---

*数据/进度汇总见 `data_ledger.md` 与 `status_checkpoint.md`；方案见 `plan_v6.md`。*
