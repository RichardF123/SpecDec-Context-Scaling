# G3 —— FP8 KV 设计冻结（**FROZEN v1.0**，2026-09-08 批准）

> 状态：**FROZEN v1.0**（经用户审阅批准）。实现与验证须严格按本文件；任何改动需走文末变更记录（防 protocol drift）。
> 本文是 plan（意图/协议）；所有数字跑出来后进 `data_ledger.md`（只记实测）。

---

## 0. 术语与边界（Non-goals，冻结）

- **FP8-compressed KV cache**：只把**持久化的 K/V 缓存**以 FP8 存储；注意力本身仍是 fp16 eager matmul（非 native FP8 attention）。
- 只量化**已存进 cache 的 K/V**（post-RoPE 值，与现行缓存语义一致）；**不量化权重、不量化激活、不引入任何 fp8 kernel**。
- 引擎缓存结构维持 **cat-based**（每步 `torch.cat` 全量语义）——理由见 D6，不在此里程碑重设计成 ring buffer。
- 本设计只定义 AR（自回归）路径的 FP8 KV；spec-dec（G4）复用同一 KV 模块。

## 1. 冻结决策表（每项：决策 / 理由 / 被否方案）

### D1 格式：**E4M3**（torch dtype `float8_e4m3fn`），CUDA-only
| | E4M3 | E5M2 |
|---|---|---|
| 指数位/尾数位 | 4 / 3 | 5 / 2 |
| max finite | 448 | 57344 |
| 相对粒度（近 max 处）| ~2⁻⁴（细）| ~2⁻³（粗 1 倍）|
| 适用 | K/V 幅值本就在 fp16 小动态范围内 → 精度优先 | 需大动态范围（本场景不需要）|

**决策**：E4M3。**理由**：K/V 来自层激活，幅值由 fp16 决定，无大动态需求；E4M3 尾数多 1 位，量化误差更小。**门内复核结论**：E5M2 无优势，不采用（如后续 ΔPPL 异常大，先查缩放与实现，不经批准不换格式）。

### D2 缩放粒度：**per-kv-head × per-token 动态 scale，K/V 各自独立**
- 每个 (token, kv-head) 一个 K-scale、一个 V-scale；scale = 该 64 维向量 `max|v| / 448`（fp32 计算，0 向量时 scale=1 防除零）。
- **被否方案**：
  - per-tensor（整层一个 scale）：跨 token 动态范围大 → 误差随 ctx 累积，8K 下不可接受；
  - per-channel（每 d 维一个 scale）：误差最小但元数据 = 2 heads × 64 d × K/V = 128 scale/token/layer，过大；
  - K/V 共享 scale：K、V 幅值分布不同，共享会伤害较小者。
- **理由**：per-head-per-token 是 KV 量化文献的标准折中（如 DeepSeek-V2 路线）；本模型 kv_heads=2，元数据开销小。

**D2b（v1.1 临时修订，2026-09-08，以 V4 实测为准）**：per-kv-head × per-token 粒度实测
（bf16 base, 40 窗）relative ΔPPL ≈ **+283%** —— 过粗。改为 **per-token × per-kv-head ×
组内分组**：head_dim=64 先按 **G=2 组 × 32 维**（元数据 ×2）；若仍不足再 G=4 × 16 维（×4）。
- scale 形状 `(B, kv, T, G)`（G=2 或 4）；payload 形状不变 `(B,kv,T,64)` fp8；
- 量化/反量化只改 `quantize_kv_fp8`/`dequantize_kv_fp8` 内部（末维 view 成组），
  注意力/缓存读写路径零改动；scale 拼接仍在最后一维；
- 记账（G=2 / G=4，fp16 512 B 基准）：payload 256 B + scales 32 B → **288 B（1.78×）**；
  或 256+64 → **320 B（1.60×）**；
- 保留开关 `GROUP`（32 → 16），便于同一 harness 快速 A/B。
- **若 G=4 后 relative ΔPPL 仍大 → 停止细粒度方向，走 D 语义一致化或 A（negative/mixed）。**

### D3 scale 存储与生命周期
- scale 存 **fp32**（4 B），与 payload **同生命周期**（token 进缓存即算，读到缓存淘汰才消失——本引擎缓存随 run 生命周期存在）。
- 布局：`k_scale / v_scale` 形状 `(B, kv_heads, T)` fp32。
- **被否**：scale 存 fp16（总元数据省 8 B/token/layer，省 1.5%，正确性优先不值）；E8M0（超本次范围）。

### D4 量化/反量化公式（冻结，防实现漂移）
```
s      = max(|v_fp16|, dim=-1) / 448        # 每 (head, token)；max=0 → s=1
v_q    = (v_fp16 / s).to(torch.float8_e4m3fn)   # torch cast = round-to-nearest-even
v_hat  = v_q.to(torch.float16) * s              # 反量化点
```
- 无 clamp：`s = max/448` 保证 max 恰好映射到 ≤448，构造上不溢出（E4M3FN 无 inf）。
- 量化前 K/V 先转 fp32 求 max 再回 fp16 除法（避免 fp16 下 /s 溢出），实现细节以本公式为准。
- **引擎反量化输出 dtype（v1.0.2 澄清）**：引擎内反量化须产出 **fp16**（供 fp16 eager attention 同 dtype matmul）：`(q.to(f32) * s).to(f16)`（fp32 乘后单次舍入）。harness 参考实现保留 fp32 输出作为比较基准；两者仅差最后一步 fp16 舍入，属 fp16 attention 精度的固有部分。

### D5 反量化点与写点（注意力路径零改动）
- **读**：某步需要过去 p 个 token 的 K/V 时：`past16 = dequant(payload[:,:,:p])`（fp8→fp16 × scale）→ 与**本步新算的 fp16 K/V** `cat` → 交给**现有** eager attention（mask/softmax 等全部原样）。
- **写**：注意力算完之后，把**本步新增的 K/V（fp16）量化入库**；只量化一次（历史不反复重量化 → 无逐层累积重量化误差）。
- 每步开销诚实注记（D7 展开）：fp16 基线每步本就全量 `cat`（O(S) 复制）；fp8 用"整段 dequant（O(S)）+ 写点量化"替换之。

### D6 缓存布局（落到 `model_self.py` 真实结构）
现行：`cache_list[i] = {"k": (B,kv,T,d) fp16, "v": ...}`；`start_pos = cache_list[0]["k"].shape[2]`；attention 内 `torch.cat([cache["k"], k])` 后写回。
FP8 模式：payload 仍用键名 `"k"/"v"`、形状 `(B,kv,T,d)`，**dtype=float8_e4m3fn**；新增 `"k_scale"/"v_scale" (B,kv,T) fp32`；cache dict 增 `"fmt": "fp16"|"fp8"` 标记。
- `start_pos` 读 `shape[2]` **不变**（fp8 payload 同形）→ `_run` / `prefill_chunked` 无需改 RoPE 偏移逻辑。
- 模型级开关：kv 缓存格式由外部传入（config 键 `kv_format ∈ {fp16, fp8}`），**fp16 路径一行不动**（回归保 G1/G2）。
- **cat-based 保留理由**：① factorial 的 fp16/fp8 对比在**同一引擎结构**下才公平；② ring buffer 会改 fp16 基线本身 → 需重跑等价门，超 scope；③ 记录为 F5 型 limitation（实现层 inefficiency），报告"局限"里写。

### D7 每步开销预期（诚实注记，不预设结论）
- fp16 decode 每步已含 O(S) 全量 cat；fp8 decode 每步 = O(S) dequant + cat 小段。→ **decode-only 速度 fp8 vs fp16 预计 ≈1×**（被 factorial 证实或证伪）。
- FP8 在本引擎的主要可测效应 = **峰值显存与可行性**（持久 KV 减 1.88×）+ 写点量化开销；不是带宽/速度。
- 含义：RQ3 的"profitable range"将以**哪些 cell 装得下 + 峰值列 + 每步开销随 ctx 的斜率**来回答；若 fp16 spec-dec 某 cell 放不下而 fp8 放得下，那就是 FP8 改变可行区间的直接证据；若两边都放得下且速度 ≈1，那是 F7（FP8 只解 feasibility）类诚实结果——**合法、照报**。

### D8 CUDA-only 约束
- torch 的 float8 运算/转换仅 CUDA 可靠 → FP8 正确性/质量验证全部在 **CUDA fp16↔fp8** 上做行为等价（不做 CPU 逐位对齐）；实现与验证脚本都标注设备前提。

## 2. 内存记账（理论 bytes/token，×28 层）

| 项 | fp16 基线 | FP8-compressed |
|---|---|---|
| K payload | 2×64×2 = 256 B | 2×64×1 = 128 B |
| V payload | 2×64×2 = 256 B | 2×64×1 = 128 B |
| scales（fp32）| 0 | (2+2)×4 = 16 B |
| **合计 / token / layer** | **512 B** | **272 B**（1.88×）|
| ×28 层 / token | 14.0 KiB | 7.44 KiB |
| @2K / @4K / @8K | 28.0 / 56.0 / 112.0 MiB | 14.9 / 29.7 / 59.5 MiB |

- **理论值单独报**（bytes/token + 上述表）；实测峰值显存（torch alloc + nvidia-smi used）另行记（防 WDDM reserved 超订误读，见 ledger §4.5.1 教训）。
- 诚实注：≤8K 下 KV 相对模型权重（~3 GiB）很小 → FP8 的**绝对省量**在此长度有限（8K 省 ~52 MiB）；其作用在双模型 spec 配置与峰值列中体现。这正是 RQ3 要量化的，不在设计阶段美化。

## 3. ΔPPL 质量协议（落地 v7.1 §4.6）
- 数据集 Wikitext-2 **test**；**非重叠**固定长度窗口；窗口长 = 2K（与条件对齐）；**每 condition ≥32 窗**；
- 同窗口 fp16-KV reference：`ΔPPL = PPL(fp8) − PPL(fp16)`，报 **absolute + relative ΔPPL%**；
- **不设人工 PASS/FAIL δ**（freeze 规定）；ΔPPL 数字进台账与报告，归因（是否随 ctx 增大 / 只出现在特定层）在分析阶段做；
- 次要 sanity（非门）：固定 prompts 上 fp16-KV vs fp8-KV 贪心 decode，报 token 级分歧计数/比例——**分布外推到长生成的混沌分歧属预期**，不作门槛。

## 4. 引擎改动点清单（实现 spec —— 由你写代码，AI 只审）
1. `SelfAttention.forward`：fp8 模式下读路径 dequant + 写路径量化（D4/D5 公式），其余（mask 三分支、rope、kv_repeat）不动；
2. cache dict 结构（D6）与 `fmt` 传递；`start_pos` 不变；
3. `QwenLM` 加 kv 格式开关并透传层；`prefill_chunked`/`_run` 不加新分支（格式在 cache dict 里）；
4. 量化/反量化为**独立小函数**（如 `quantize_kv_fp8(fp16_tensor) -> (payload, scale)`、`dequant_kv_fp8(payload, scale) -> fp16`），便于 harness 单测与复用；
5. fp16 路径**零改动**（G1/G2 回归不受影响）。

**4.1 接口契约（v1.0.1 澄清，供实现与 harness 对齐；不改变任何 D* 决策）**
- 模型级开关：`QwenLM` 增加属性 `kv_format: str`（默认 `"fp16"`）+ 方法 `set_kv_format(fmt)`，递归设置到每个 `SelfAttention`（层内存 `self.kv_format`）；
- 首次建缓存（`cache is None`）时按 `self.kv_format` 决定写格式；`cache["fmt"]` 记录实际格式；`SelfAttention` 读路径以 `cache["fmt"]` 为准（防御混用）；
- harness 调用序列：`model = build_from_config(cfg)` → 载权重 → `model.set_kv_format("fp8")` → 正常 `_run`/`prefill_chunked`/`generate`（签名不变）；
- fp8 模式下 payload 键仍为 `"k"/"v"`（dtype float8_e4m3fn）、新增 `"k_scale"/"v_scale"` (B,kv,T) fp32 —— harness 通过 `cache_list[i]["k"].shape[2]` 等既有访问方式不变。

## 5. 验证 harness 计划（AI 写，你在 GPU 跑）
- **V1 往返精度单测**（随机 + 真实幅值分布 K/V）：quant→dequant 的 max|Δ| / 相对误差；无 NaN/Inf；max ≤ 448；
- **V2 fp8-vs-fp16 短窗行为**：同 prompt，fp8 cache vs fp16 cache decode 前 N tok：logits max|Δ| + token 分歧数（报数，不设门槛）；
- **V3 chunked fp8 消融**：同 G2-B 结构（chunk 2048 vs 512 建 fp8 缓存 → 行为一致 + 漂移量级）；
- **V4 ΔPPL**：§3 协议的落地脚本（窗口化取数 + 计 PPL + 差分表）。
- G3 门 = V1 过 + V2/V3 行为 PASS + V4 如实报数（无人工阈值）→ 设计冻结生效 + 台账登记。

## 6. 风险与诚实预期
- decode-only fp8≈fp16（D7）→ 合法 flat 结果，F7 路径；
- ΔPPL 若随 ctx 明显增大 → 记录层趋势并归因（per-token scale 已是最稳方案，若仍大则属 E4M3 固有，如实写 limitation）；
- 任何实现偏差先查代码再碰设计；改设计必走变更记录。

## 7. 批准栏
- [x] 我审阅并理解 D1–D8 与 §2–§5 —— **批准冻结为 v1.0**（2026-09-08）
- [ ] 修改意见：＿＿＿＿＿＿＿＿＿＿

---
*变更记录：v0.1（DRAFT，2026-09-08）：初稿。v1.0（FROZEN，2026-09-08）：用户批准，D1–D8 生效。v1.0.1（2026-09-08）：§4.1 澄清接口契约（`set_kv_format`），无决策变更。v1.0.2（2026-09-08）：D4 澄清引擎反量化输出须为 fp16（单次舍入），无决策变更。v1.1（2026-09-08，**临时**）：D2b scale 粒度加组内分组（G=2×32 起），待 V4 实测定稿；运行 dtype 因平台改为 bf16（CUDA fp16 对 Qwen NaN）。**v1.2（2026-09-08，正式）**：pilot 证明 V-fp8 任何粒度无损（ΔNLL≤+0.002）、K-fp8 任何粒度（128/64/32/16/1）均致 ΔNLL≥+2.4 → **正式方案改为非对称：K 保持 bf16（不量化），V 走 FP8（group=32）**；K-fp8 全粒度记录为 negative ablation；实现经 `set_fp8_matrices(k_on/v_on/k_group/v_group)` 支持（默认即正式路径），D 语义保留。*
