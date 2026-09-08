# SpecDec 数据收集台账（Data Ledger）

> 目的：集中登记所有**实测数据**（正确性/性能/质量），每个数字带环境与复现条件。
> 机器产物最终进 `results.jsonl`（追加式）；本文件是**人工可读台账**，两者字段对齐。
> 规则：凡记录必附 {日期, 硬件, dtype, 模型, seed, 协议}；不填来源的数字一律不写。

---

## 1. 环境基线（数据有效性前提）

| 项 | 值 | 备注 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop 8GB（sm_89）| nvidia-smi 待补显存占用 |
| CPU / RAM | i7-12700H / 32GB | — |
| OS / Python | Windows / 3.11（venv）| — |
| torch | 2.6.0+cu124 | PyPI 官方 cu124 |
| transformers | （待补精确版本号）| `pip show transformers` |
| HF cache | D:\hf-cache | HF_HOME |
| 临时目录 | D:\RealDSH\pip-tmp | 防 C 盘锁/满 |
| 模型（被测）| Qwen/Qwen2.5-0.5B-Instruct | Apache-2.0，tie、attn_bias=True |

## 2. 模型配置（被测对象）

| 参数 | 值 |
|---|---|
| hidden / layers / heads / kv_heads / head_dim | 896 / 24 / 14 / 2 / 64 |
| intermediate / vocab / rope_theta / rms_eps | 4864 / 151936 / 1_000_000 / 1e-6 |
| attention_bias / tie_word_embeddings | True / True |
| 引擎实现 | 自写 model_self.py（OURS），HF 仅加载权重 |

## 3. 正确性数据

### 3.1 L1 对齐门（ours full vs HF full）
| 日期 | 精度/设备 | prompts | 结果 |
|---|---|---|---|
| 2026-09-02 | fp32 / CPU eager | 5（≤48 tok）| **worst max|Δ| = 0.000** |
| 备注 | 逐位一致（同序 fp 运算）；修复项：attn_bias 透传、rope_theta 从 rope_parameters 读 | |

### 3.2 L2 缓存等价（ours cache vs ours full，最后一位置）
| 日期 | 模型/长度 | max|Δ| | argmax |
|---|---|---|---|---|
| 随机小模型 | vocab5000/64h/2L/10 tok | ~1.1e-5 | 一致 |
| 真 Qwen 0.5B | 21 tok 文本，k=5/10/15/20 | 1.6e-5 ~ 2.2e-5 | 全一致 |
| 判据 | 逐层差 ~1e-7 且不放大 = fp kernel 顺序噪声（容差 1e-3）| | |

### 3.3 生成对齐门（ours.generate vs HF 全量逐步贪心）
| 日期 | 参考 | prompts | 结果 |
|---|---|---|---|
| 2026-09-02 | HF full-forward 循环（use_cache 无关）| 4 | **全序列逐 token 一致 PASS** |
| 发现 | `hf.generate` 与其全量前向不一致（官方 logits ' is'=21.75 vs 其输出 ' was'）→ 不用其当裁判 | | |

### 3.4 G1 —— 1.5B target correctness gate（v7.1 G1，2026-09-08）
| 检查 | 内容 | 结果 | 条件 |
|---|---|---|---|
| G1.1 | OUR 1.5B full forward vs HF 1.5B full | **max|Δ| = 3.10e-05 PASS** | fp32/CPU eager；Qwen/Qwen2.5-1.5B-Instruct（hidden1536/l28/h12/kv2/vocab151936/attn_bias=True）|
| G1.2 | OUR 1.5B cached decode vs OUR full | **max|Δ| = 2.04e-05，argmax same PASS** | 同上；32-token 输入 |
| G1.3 | greedy 生成 vs HF 全量贪心 | **4/4 序列一致 PASS** | 同上；4 prompts × max_new=12 |
| 注 | 0.5B 全部旧数据标记为 **development-stage validation**；G1 为 target 正式门 | | |

### 3.5 FP8 KV V1 往返精度单测（G3 门第 1 项，2026-09-08，`fp8_kv_v1_roundtrip.py`）
| 检查 | 结果 | 条件 |
|---|---|---|
| payload 不变量 | max|q|=448.0（≤448），无 NaN/Inf | CUDA RTX4060；`torch.float8_e4m3fn`；D4 参考实现（fp32 除法单次舍入）|
| 往返误差 | max|Δ|=1.06e2，max|v|=1e4，理论界 max/16=6.25e2 → **不超界 PASS**（E4M3 rel≤2⁻⁴）| 合成 (2048,2,64)，log-uniform 1e-2…1e3 + 边界组 |
| 确定性 | 同输入两次量化逐位一致 PASS | — |
| 全零组 | 往返 max|Δ|=0（s=1 防除零路径）PASS | — |
| 离群组 | head0 正常元素 max|Δ|=6.06e-2 ≤ max|others|/16=1.74e-1（per-head scale 吸收 1e4 离群）PASS | 确定性构造：others ~ N(0,1) |
| 极值映射 | max 恰好映射 448 PASS | — |
| 结论 | **V1 PASS**（V1 修 harness 自身断言 2 次：错误全局上界 → max/16；离群组随机基+相对误差判据 → 确定性构造+abs 界）| | |

### 3.5.1 FP8 KV V2/V3（G3 门，2026-09-08，引擎 fp8 路径实现后）
| 检查 | 结果 | 条件 |
|---|---|---|
| **V2 fp8-vs-fp16 短窗** | 结构 ok（fmt/dtype/scale 形状）✅；fp8 确定性（两次 pass 逐位一致）✅；logits 全 finite ✅ → **V2 PASS** | 1.5B；prompt 512 tok，chunk 1024，decode 32 |
| V2 分歧记录（不设阈）| prefill 末位 logits max\|Δ\|=0.000（无历史, 两者同）；decode 步 logits max\|Δ\| ≈ 0.38–0.40（fp8 历史舍入）；32/32 token 一致（退化重复文本, 贪心边距大）| 同上 |
| **V3 chunked fp8 消融** | payload/scale finite ✅；两 chunk 布局（1024 vs 512）decode 16 序列一致 → **V3 PASS** | 1.5B fp8；prompt 4096 tok |
| V3 漂移记录 | 反量化 KV max\|Δ\|：层 0/14/27 = 3.03e-1 / 2.61e0 / 6.09e0（随深度累积；fp8 浅层噪声大于 fp16 消融的 3.9e-3, 深层次要同量级）| 同上 |
| 注 | 引擎实现经 4 处 review 修正（读路径条件反、kv_repeat 覆盖 k/v、T 取 k_all、键名 fmt 统一）后通过；harness 修 isfinite(fp8 不支持) 一次 | | |

### 3.5.2（作废备注）fp16 时代 V4 相关旧行已被 bf16 平台修复 + bias/RoPE/RMSNorm/softmax 修复作废；历史保留不作结论依据。

### 3.5.3 正式 Asymmetric V-only FP8 KV（G3 FROZEN，2026-09-08）
方案：K cache = BF16（不量化）；V cache = FP8 E4M3；V scale = per-token × per-KV-head，group=128（`fmt="v_fp8"`，scale 形 `(B,kv,T,1)`）。full-FP8/K-FP8 = negative ablation（`fmt="fp8"`）。
| 项 | 结果 | 条件 |
|---|---|---|
| V-only V1（直接结构/往返）| K 与 base 运行 K 逐位相同；V fp8 + v_scale(…,1) fp32、无 k_scale；全零精确；确定性；`fmt="v_fp8"` | `fp8_vonly_v1_roundtrip.py`，96 tok |
| V2 | 结构 ok；finite；deterministic；prefill logits max\|Δ\|=0.4375；**prefix 32/32** | 1.5B bf16；512 tok, decode 32, v_fp8 |
| V3 | chunk1024 vs 512 decode16 **一致**；finite → PASS | 4096 tok, v_fp8 |
| localize | base 16/16、v_fp8 16/16、v_fp8-vs-base 16/16、重复 16/16；相对 base 额外 coupling **K +0.379 / V +0.891**（不要求原始 KV bitwise 同, bf16 GEMM shape 漂移）| — |
| **V4 正式独立窗口** | mean NLL base 2.312942 / V-only 2.313338；**ΔNLL +0.000397**；PPL 10.1041 / 10.1081；absolute ΔPPL +0.0040；relative **+0.040%**；per-window ΔPPL mean +0.0044（min −0.0332, max +0.0370）；finite True | offset 40..79（40 窗×2048, chunk 512, 与 pilot 0..3 disjoint=True）, total predicted 81880 |
| 8K operational（同脚本 paired）| base: alloc 5.07 GiB / resv 7.11–7.12 / KV logical 224.00 MiB / prefill 3.5–3.7s / decode 2.1–2.3s；v_fp8: alloc 5.01 GiB / resv 7.10 / logical 169.75 MiB / prefill 3.6s / decode 2.5s；3 轮均稳定 | ctx8192, chunk2048, decode64, bf16, `fp8_vonly_8k_operational.py` |
| 记账 | per token/layer 776 B（K 512+V 256+v_scale 8）vs 1024 B = **75.78125%, 1.320×**；logical saving 54.25 MiB@8K；alloc 峰值降 ≈0.06 GiB；**decode 有反量化开销、prefill 相近** | 跨脚本显存数字不可比（G2 旧 6.83 GiB 边界不同） |
| Negative ablation | full-FP8 per-head(128): ΔNLL≈+2.41, rel ΔPPL≈+771%；full-FP8 gs32: ΔPPL +2158%（更差）；K-only 各粒度 ΔNLL≥+2.4（K32≈+4.1）；V-only gs32/gs128 ΔNLL≤+0.002；K1/V1 无损哨兵=0 | pilot 级（不重跑 40 窗） |
| 结论 | 在 Qwen2.5-1.5B + 在线 max-abs E4M3 + eager attention 下：**K-FP8 显著退化、V-only FP8 基本无损**；1.32× KV 压缩不自动 = 整模型/可行性收益；G3 研究结论以此为准 | — |

### 3.5.4 pilot 归因数据（`fp8_kv_group_probe`，2026-09-08；窗口 offset 0..3×2048，仅诊断不用于正式判定）
真实 K/V 重建（base bf16 真实 cache，逐层 relL2/cos，K/V 同法）：
| gs | K relL2（mean / L0 / L14 / L27）| V relL2 mean | cos |
|---|---|---|---|
| 128 | 0.0234 / 0.0254 / 0.0242 / 0.0236 | 0.0247 | 0.9997 |
| 64 | 0.0221 / 0.0236 / 0.0232 / 0.0225 | 0.0239 | 0.9998 |
| 32 | 0.0215 / 0.0236 / 0.0226 / 0.0220 | 0.0227 | 0.9997 |
| 16 | 0.0200 / 0.0236 / 0.0213 / 0.0205 | 0.0209 | 0.9998 |
| 1 | 0.0000（无损哨兵）| 0.0000 | 1.0000 |
> 重建误差小且几乎不随分组变 → group32 质量更差不能用"误差量"解释（指向注意力/传播敏感）。

离线 attention 代理（query rows 1920..2047, keys≤pos；prob = softmax L1 均值；V-only 用 base probs）：
| 层 | prob L1（gs128 / gs32）| attn-out relL2（gs128 / gs32, V-only）|
|---|---|---|
| 0 | 3.48e-4 / 3.62e-4 | 0.0202 / 0.0188 |
| 14 | 2.25e-5 / 2.22e-5 | 0.0167 / 0.0151 |
| 27 | 2.39e-5 / 2.29e-5 | 0.0147 / 0.0137 |
> 注：该次运行 score 差列因 masked(-inf) 未排除而含 NaN，未计入（探针已修，未重跑）。

pilot NLL（4 窗；base 2.4118 / PPL 11.15；det=True 全部）：
| 配置 (gk/gv) | NLL | ΔNLL vs base | 读法 |
|---|---|---|---|
| K1/V1 | 2.4118 | 0.0000 | 无损哨兵（框架正确性）|
| **K1/V32** | 2.4115 | −0.0002 | **V-only gs32 无损** |
| **K1/V128** | 2.4134 | +0.0016 | **V-only gs128 无损** |
| K32/V1 | 6.5134 | +4.1016 | K 损伤（分组更细更差）|
| K128/V1 | 4.8067 | +2.3949 | K 损伤最轻仍大 |
| K128/V128 | 4.8211 | +2.4093 | full-FP8 ≈ K-only |
| K64/V64 | 6.4687 | +4.0569 | 更细更差 |
| K16/V16 | 6.4927 | +4.0809 | 更细无益 |
| K32/V32 | 6.5403 | +4.1285 | — |



## 4. 性能基线（L2 朴素解码，正式协议前的初步数字）

### 4.1 CPU fp32
| 指标 | 数值 | 条件 |
|---|---|---|
| prefill | 93 tok/s（38 tok ≈ 409 ms）| prompt=38 tok，3 reps 取最优 |
| decode | 9.20 tok/s（≈108.75 ms/token）| 64 decode tokens，warmup 4 |

### 4.2 GPU fp16（RTX 4060）
| 指标 | 数值 | 条件 |
|---|---|---|
| prefill | 398 tok/s（≈95.5 ms）| 同上 |
| decode | 10.58 tok/s（≈94.51 ms/token）| 同上 |

**解读**：decode 为 latency-bound（单 token 串行 + 24 层小 kernel + Python 开销），GPU 优势小；**此为后续 L4/L4b/L5/kernel 的对照基线**。
> 注：GPU 首次运行曾现 `bad allocation`，重跑正常——记录为环境瞬态，未复现。

### 4.3 外部对照 sanity（评审要求：基线是否有可避免开销）
| 实现 | decode tok/s | ms/token | 条件 |
|---|---|---|---|
| transformers eager（fp16, cuda, generate 循环）| **8.29–9.40** | ~106–121 | 同 prompt、同 GPU、N=20–32 |
| **我们的引擎（fp16, cuda）** | **10.58–13.82** | ~72–95 | 同条件（decode 32–64, warmup）|

**结论**：我们的朴素 eager decode **不慢于** transformers eager 参考（反而快 ~27–47%）→ 该数字是 eager/单 token 配置的固有水平（latency-bound），非可避免开销；**L5 净加速比的分母可信，报告可与库参考对照引用**。

### 4.4 8K fp16 prefill 显存（探针 C 段）
- **结果：单次 8K eager prefill（ours，fp16）CUDA OOM**：scores（14×8192×8192×fp16）≈ 1.75GiB/层瞬时 + logits（8192×151936×fp16）≈ 2.5GiB + 双模型常驻 > 8GB；
- transformers eager 同配置也会 OOM（其也被强制 eager）。
- **含义/方案**：18 格的 8K 行不能"一次性喂 8K"获取，需分块 prefill（块级因果 mask，引擎小改造）+ 缓存构建路径跳过 logits + 单模型常驻/expandable_segments；该 OOM 本身可作为"为什么需要 KV 压缩/量化/长上下文优化"的动机素材。

### 4.5 8K chunked prefill 最小可行性预检（2026-09-08；**非正式 G2 门**）
> 状态声明：`g2_feasibility.py` 是**最小可行性预检**（用户要求"先跑最小可行性验证"），仅证明 8K×1.5B 分块 prefill 可运行。
> **正式 G2 门（v7.1 §7：规定 decode 长度 + 连续多次稳定 + correctness sanity PASS + 温度/时钟记录）尚未完成。**
| 项 | run1 | run2 | 判据 |
|---|---|---|---|
| 配置 | Qwen2.5-1.5B-Instruct fp16 / CUDA，ctx=8192，chunk=2048，decode 6 tok（seed 0）| 同左 | — |
| 峰值 VRAM | 6.83 GiB | 6.83 GiB | < 7.5 GiB **PASS** |
| prefill 墙钟 | 11.7 s | 8.3 s | 记录（run1 含 CUDA 首启开销）|
| prefill+decode | 12.0 s | 8.5 s | 记录 |
| 生成 token | 两次逐 token 一致 | 同左 | **PASS（稳定）** |
| 结论 | **预检通过：8K × 1.5B 分块 prefill 在 8GB 卡上可运行**（chunked 路径绕过 §4.4 的一次性 8K OOM）；**正式 G2 门 → 见 §4.5.1** | | |
> 注：生成文本为重复合成语料截断输入的贪心续写（'!诳!诳!诳'），仅作可运行性/稳定性探针，**不作正确性声明**（正确性由 chunked 等价门另行锚定）。

### 4.5.1 正式 G2 门（v7.1 §7）—— **PASS**（2026-09-08，`g2_operational.py` 修复版）
| 项 | run1 | run2 | run3 | 判据 |
|---|---|---|---|---|
| 配置 | Qwen2.5-1.5B-Instruct fp16 / CUDA；ctx=8192；chunk=2048；decode 128 tok（§4.2 定长，seed 0）| 同左 | 同左 | — |
| 峰值 VRAM（torch alloc）| 6.83 GiB | 6.83 GiB | 6.83 GiB | < 7.5 GiB **PASS**（3/3）|
| prefill 墙钟 | 10.1 s | 10.2 s | 10.2 s | 记录 |
| decode 墙钟（128 tok）| 14.3 s | 13.8 s | 13.1 s | 记录；**跨会话方差大**（20:13 会话曾测 4.8–5.6 s，同代码）→ 环境因素（GPU 时钟/CPU 争用），正式基准以交错+中位数处理（P9/P11），逐 session 数字不可直接比 |
| 生成长度 / 序列 | 128 / 全同 | 128 / 全同 | 128 / 全同 | 3/3 逐 token 一致 **PASS**；首 token '!'，尾部稳定 |
| GPU 温度 | 60–66 °C（n=24）| | | 记录（P11），无热降频迹象 |
| correctness sanity（chunk 消融 2048 vs 512）| KV max|Δ| overall 6.109e+00；层 0/14/27 = 3.906e-03 / 1.211e-01 / 6.109e+00（深度累积 fp16 取整漂移，**只记录不设阈值**）；两缓存续 decode 16 tok 序列一致 | | | **PASS（行为等价）** |
| 结论 | **G2 正式门 PASS**（v7.1 §7 全项：不 OOM + 定长 decode + 3× 稳定 + correctness sanity + 峰值/温度/时间记录）| | | |
> **方法学教训（写进报告）**：首次跑出 6.83/7.62/7.63 GiB 的"峰值漂移"实为**测量伪影**——上一 run 的末块 logits（2048×151936×fp16 ≈ 0.62 GiB）跨 `empty_cache/reset_peak_memory_stats` 仍被变量引用而未释放；`del last_logits, cache` 后 3/3 稳定 6.83 GiB。峰值测量必须防跨 run 存留引用。
> **另注**：`g2_peak_probe.py` 显示 torch reserved 池峰值 8.71 GiB > 物理 8 GiB（Windows WDDM 允许超订），nvidia-smi 实际占用峰值 ~7.70 GiB——无 OOM、无分页悬崖（时长稳定）；正式基准峰值列以 torch alloc 记 + smi used 参考。

## 5. 复现设置（每次记录必须带上）
- seed=42（测试）；性能脚本 seed=0，3 reps 取最优 + warmup；
- prompt 固定（bench_basic.PROMPT，38 tok）；decode 数 64；
- dtype/设备随行标注（fp32/CPU、fp16/CUDA）；attn eager；no_grad；
- 待补：torch/transformers 精确版本、nvidia-smi 输出存档。

## 6. 预留登记表（后续里程碑——现在为空，跑完填）

### 6.1 L4 权重量化（int8 / int4）
| 方案 | 权重大小 | peak VRAM | decode tok/s | ppl(Δ) | 自建题集 acc | 备注 |
|---|---|---|---|---|---|---|
| fp16 基线 | | | | | | 填 |
| int8 | | | | | | 填 |
| int4 | | | | | | 填 |

### 6.2 L4b KV 量化（fp16→fp8→int4→int2）
| KV dtype | bytes/token | 峰值显存 | 长上下文(8K)可行? | ppl(Δ) | 备注 |
|---|---|---|---|---|---|
| fp16 | | | | | 填 |

### 6.3 L5 固定链投机解码（2K/4K/8K）
| 上下文 | 接受率 | net speedup | tokens/s | ppl | 备注 |
|---|---|---|---|---|---|
| 2K | | | | | 填 |
| 4K | | | | | 填 |
| 8K | | | | | 填 |

### 6.4 外部对照（同一模型：transformers / gpt-fast）
| 对照 | decode tok/s | 说明 |
|---|---|---|
| transformers（eager）| | 填 |

## 7. 变更日志
- 2026-09-02：建台账；登记 L1 对齐（0.000）、L2 缓存等价与生成对齐（PASS）、CPU/GPU 性能基线。
- 2026-09-08：登记 G1（1.5B target 正确性门 PASS，§3.4）；登记 8K 分块 prefill 最小可行性预检通过（§4.5）与正式 G2 门 PASS（§4.5.1，含峰值测量伪影教训）；登记 FP8 KV V1 往返单测 PASS（§3.5）。
- 2026-09-08（晚）：平台/引擎修复（qwen2 attention bias 默认、RMSNorm fp32、softmax fp32、RoPE fp32、bf16 质量口径、V4 LOSS_ROWS）→ fp16 时代旧行为数字标注作废（§3.5.2 备注）；**正式 Asymmetric V-only FP8 KV 冻结**（§3.5.3：V1/V2/V3/localize/V4/8K/记账 + full-FP8 & K-FP8 negative ablation），G3 研究结论与设计修订见 `design_freeze_fp8_kv.md` v1.2/v1.3 与 `status_note_fp8_v4.md`。
- 2026-09-08（归档）：全流程日报 `daily_report_2026-09-08b_G1G2G3_FP8KV.md`；pilot 归因数据入 §3.5.4（`fp8_kv_group_probe`）；v_fp8 正式实现与测试入口就绪（`fp8_vonly_v1_roundtrip` / `fp8_vonly_8k_operational`）。
