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

### 4.5.2 B3 双模型 8GB 可行性（G4，2026-09-08，`g4_b3_dual_model_feasibility.py`）
条件：target=Qwen2.5-1.5B + draft=Qwen2.5-0.5B **同时常驻**，BF16 CUDA eager，batch=1；
target KV ∈ {base_bf16, v_fp8}，draft 恒 base BF16；ctx ∈ {512, 8192}（chunked 2048）；3 轮稳定，decode smoke 5 tok。
| 阶段/项 | 数值 | 备注 |
|---|---|---|
| 双模型权重常驻 | 加载后 free=2.00 GiB，smi used=5.76 GiB | target 后 free 0.81 GiB → draft 后 2.00 GiB |
| ctx512 base_bf16 | alloc 4.34（3/3 同），resv 4.58，无 OOM，finite，smoke 稳定 | PASS |
| ctx512 v_fp8 | alloc 4.336（3/3），resv 4.58，无 OOM，finite，smoke 稳定 | PASS（vs base −0.003 GiB 实测差）|
| ctx8192 base_bf16 | alloc 6.16（3/3 同），resv 8.92，无 OOM，finite，smoke 稳定 | PASS（resv>8 GiB 物理：WDDM 池超订, 未 OOM）|
| ctx8192 v_fp8 | alloc 6.16（3/3 同），resv 8.92，无 OOM，finite，smoke 稳定 | PASS（alloc 峰值与 base 相同——峰值被瞬态主导, 不宣称峰值收益；KV 逻辑差 54 MiB 见 §3.5.3）|
| prefill 墙钟（参考, 非性能）| 512 ~0.1s；8192 tgt 8.2–8.5s / dft 8.5s | 可行性参考 |
| decode smoke 5 tok | tgt ~0.2–0.8s；dft ~0.1–0.5s | 首 token tgt/dft 一致（512:' spans many de'; 8192:' of artificial'）|
| 结论 | **B3: PASS（4/4 条件）**——双模型 8K 同驻可运行；不做跨实验显存比较（≠G2 fp16 单模型、≠G3 单模型 paired）| — |

### 4.5.3 B4 cache 操作（G4，2026-09-08，`g4_b4_cache_tests.py` + 用户实现 `spec_cache.py`）
| 项 | 结果 | 条件 |
|---|---|---|
| base_bf16（0.5B）| len=32 → append 35；trim→L0+1 / L0 正确；`raises=3/3`（超长/负数/0）；幂等；原 cache 不变 → **全 True** | 真 prefill(chunk16)+append+trim |
| v_fp8（1.5B target）| len=32 → append 35；`v_scale-sync=True`；`raises=3/3`；`orig-unchanged=True` → **全 True** | 同上；`v_fp8` 为 target 专属（需 head_dim%128==0）|
| 一致性守卫 | 人为裁短某层 `v` → `check_consistent` 抛 RuntimeError ✓ | — |
| 结论 | **B4: PASS**（两模式，分进程运行以避免显存累积）| — |
> 契约要点（按实现校正）：`trim_cache` 对**所有张量**沿 dim=2 裁（`v_scale` 同步）、返回浅拷贝不改入参、`new_len<=0` 或超长抛 `ValueError`、`trim_cache(None,0) is None`。
> 环境经验：harness 的 `from_pretrained` 必须**本地优先**（`local_files_only=True`），否则 tokenizer 联网列模板会因网络抖动抛 `httpx.RemoteProtocolError`（曾被误判为"驱动静默杀进程"）。

### 4.5.4 B5 SpecDec 最小 greedy 主循环（G4，`g4_b5_helper_tests.py` + `g4_b5_smoke.py`，实现 `spec_dec.py`）
| 项 | 结果 | 条件 |
|---|---|---|
| 辅助函数单测 | **T1–T4 全 PASS** | T1 `_fmt`=True；T2 `_prefill`：shape `(1,5,151936)` finite、cache_len 5(exp 5)、**vs-forward max\|Δ\| = 0.000e+00**；T3 `_propose`：`cands == ref`（`[' Paris', '.', ' It', ' is']`）；T4 `_verify` 对齐：两条**独立** prefill 比较 `rows[j]` vs `serial[j]`（同一 row 索引），argmax-same `[True×4]`、block cache 9(exp 9)、same-context max\|Δ\| `[2.5e-01, 2.2e-01, 1.2e-01, 3.0e-01]` |
| 主循环 smoke | **PASS（真实投机发生）** | stats（**实测**）：rounds 8 / proposed 27 / accepted 11 / mismatch 5 / all_accept 3 / rollbacks 5 / **`target_block_calls` 8（= rounds）** / **`draft_calls` 27（= proposed）**；`fields-complete=True sane=True gen_len=16`；重复运行一致 |
| cache 不变量 | 每轮 `tgt_len == drf_len == pre + (m+1 if m<Kp else Kp)` 全成立 | 覆盖 m=0 / m=1 / m=4 与 Kp=1/2/4 |
| **修后 GPU 复验**（2026-09-15，EOS 修复 `f7097d4` 之后） | **PASS（两档，逐轮 trace 全 `ok`）** | 见下"逐轮 cache 账"；两档 trace 与 stats **完全相同**（V-FP8 未改变任何决策，与 §4.5.6 一致）|
| **结构性恒等式**（新登记） | `gen_len == accepted + mismatch_rounds`（16 = 11 + 5） | 每个错配轮恰好贡献 1 个 target token、每个全接受轮贡献 `kp`（= 已计入 accepted）⇒ 可作为 G5 落盘的自动校验式 |
| 结论 | **B5: PASS**（修后复验有效） | 改动仅 EOS 分支 |

> **逐轮 cache 账（修后实测，两档相同）**：
> `round0 Kp=4 m=1 pre=12→14` ｜ `round1 Kp=4 m=0 pre=14→15` ｜ `round2 Kp=4 m=0 pre=15→16` ｜
> `round3 Kp=4 m=1 pre=16→18` ｜ `round4 Kp=4 m=4 pre=18→22` ｜ `round5 Kp=4 m=4 pre=22→26` ｜
> `round6 Kp=2 m=0 pre=26→27` ｜ `round7 Kp=1 m=1 pre=27→28`
> —— 错配轮 `tgt=drf=pre+m+1`（0/1/2/3/6/7），全接受轮 `=pre+kp`（4/5）；`k_eff = min(4, 预算)` 的递减在 round6(Kp=2)/round7(Kp=1) 可见；给定 `max_new_tokens=16` 恰好产出 16。
> **vacuity 教训（已修）**：主循环曾把 logits 保持为 `(1,V)` 且 `_propose` 用 `lg[0,-1]` 取到标量 → argmax 恒 0 → 候选恒为 token 0 → 接受率 0/58 的"空过 PASS"。现统一 logits 合同为 `(1,1,V)`（`_step` 返回 `lg[:, -1:]`），并在 B6 加 vacuity guard（`accepted==0` 时 verdict 降为 CHECK）。
> 设计校正（v0.3）：错配分支**不得**复用 block 位置 `L+m+1` 的 logits（那里是**被拒 draft token** 的 K/V）；正确做法是 `trim_cache(tgt, pre+m)` 后对替换 token 做**一次单 token 前向**重算，两个模型都要做。
>
> **EOS 分支修复（2026-09-15，逐行评审发现）**：`_propose` 原把 `if t == eos_id: break` 放在**推进 cache 之前**，于是 draft 提议的最后一个候选是 EOS 时 draft cache 长度只有 `pre+kp-1`；当该轮被 target 全部确认（`m==kp`，即生成正常以 EOS 收尾）时，全接受分支的 `trim_cache(drf_cache, pre+kp)` 越界 → `ValueError`。**这是常规收尾路径，正式实验必踩。**
> - 修法（根治，非 clamp 补丁）：EOS 检查移到推进之后 ⇒ `_propose` 出口恒定 `len(cache)==pre+kp`（唯一不变量）；副产物 `draft_calls += kp` 与实际 draft 前向次数**精确相等**（此前 EOS 轮少算 1）。
> - 回归测试 `g4_b5_eos_branch_tests.py`（**纯逻辑 CPU stub，无模型/GPU/网络，毫秒级**，4 个 case）：修后全部 `gen` 符合预期、每轮 `tgt_len==drf_len==pre+(m+1 if m<kp else kp)`、`draft_calls==proposed`；**旧实现对照在 3 个 EOS 全接受 case 上如实复现 `ValueError: new_len N 越界 (current N-1)`**（非空过保证），EOS 被拒 case 两版行为一致。→ **B5-EOS-branch: PASS**。
> - 对已有登记的影响：改动**只落在"该轮出现 EOS 候选"时的 draft cache 长度**。B6 的 10 个 case 未崩溃 ⇒ 不存在"EOS 候选被全接受"的轮；"EOS 候选被拒"轮经上述对照确认两版逐 token 输出一致 ⇒ **B6-bf16 的 19/20 与 194/381 结论不变**。B5 原 PASS 记录基于修前版本，**建议在修后重跑 `g4_b5_helper_tests` + `g4_b5_smoke` 以保持登记有效**（改动仅 EOS 分支）。

### 4.5.5 B6 exact-token-id 正确性门（G4，`g4_b6_correctness.py`）→ **CHECK（19/20）**
条件：10 类 prompt × 2 KV 档（base_bf16 / v_fp8）= **20 组合**；判据 = `spec_decode_greedy(...)["gen"] == target-only greedy`（**同引擎 cached AR** 参照，不是 HF）逐 token id 全等，不比文本；另加 logits finite + 重复确定性。
| 项 | 结果 |
|---|---|
| base_bf16 档 | 9/10 逐 token 全等；唯一不等 = `natural-2`（"Photosynthesis is the process by which", n=16），首个分歧 @7 |
| v_fp8 档 | 10/10 逐 token 全等（同一 `natural-2` prompt 未翻面）|
| 合计 | **19/20 组合全等**；两档合计 accepted/proposed = **194/381**；logits 全 finite；重复运行一致 |
| 判据结论 | **B6: CHECK**（未达 PASS；按协议 §"不得改参照、不得放宽门"）|

**唯一分歧的归因**（`g4_b6_tie_diag.py`；同一 prefix = `ids + ref[:7]`，len=14）：

| 口径（同位置） | `' use'`(id 990) | `' convert'`(id 5508) | gap | argmax |
|---|---|---|---|---|
| bf16 full / cached / AR-history | 22.8750 | 22.8750 | **0.0000**（`tied_at_max=2`）| **三路径全 = `' use'`（= ref）** |
| fp32 full / cached / AR-history | 23.1752 | 23.3243 | **+0.1491** | **三路径全 = `' convert'`** |

- **signal vs noise（同位置、同一候选对）**：`signal = |gap_fp32| = 0.1491`；`noise = |gap_bf16 − gap_fp32| = 0.1491` ⇒ `|signal| ≤ noise` ⇒ 该位置在 bf16 下**数值不可判定**（与算法实现无关）。bf16 绝对偏差 0.300 / 0.449（1 ULP = 0.125 ⇒ 2.4–3.6 ULP），而真差距仅 1.2 ULP → **bf16 连 gap 符号都判反了**。
- 形状噪声实测（同位置整行 logits）：full-vs-cached max\|Δ\| = **0.250**；hist-vs-full = **0.281**（≈2–4 ULP）。
- `spec[7] = ' convert'` **等于 fp32 真值**；`ref[7] = ' use'` 是 bf16 精确并列下按最小 token id 破平局的结果 ⇒ 该位置**投机路径输出的才是数值上正确的 token**。
- **登记裁决（用户决定 = A 严格口径）**：**不设"平局例外"、不放宽 exact-id 门** ⇒ B6 记 **CHECK**，该位置作为"已知数值限制"写入报告，不转 PASS。
- 推论（须进报告 Method/Limitations）：投机解码的接受判据本质是 **argmax 相等性检验**，其可靠性上限 = bf16 下 gap 符号的可靠性 ⇒ 实测接受率含一层**不可消除的平局翻面噪声**（可作为 Analysis 的量化项：统计验证位置上 bf16 top-2 gap ≤ 实测形状噪声带的占比）。
- 后果（开放项，待用户决定）：协议"B6 未过不得提出主循环性能声明"仍生效；Part 4 测量是否降级为"工程观测（非已验证正确性）"需明确。
> 仪器/判读教训：**`topk` 的返回顺序 ≠ 决策路径的 `argmax`**（并列时前者不保证取最小 index）。本 case 首次输出里 `top1=' convert'` 是 `topk` 假象，真实 argmax 三路径全为 `' use'`。B6 的 `divergence_diag` 与 tie diag 均改为打印 argmax/tied 数。
> 设备契约：`spec_dec._verify` 曾写死 `device="cuda"`，已改为跟随 `cache[0]["k"].device`（CPU fp32 复算的前提）；fp32 阶段一律 CPU 单模型（8 GB 显存放不下 fp32 权重 6.2 GB + 临时拷贝）。

### 4.5.6 B6-fp32 实现正确性门（高精度档，预登记后实测）→ **PASS**
条件：同 §4.5.5 的 **10 个 case × 2 KV 档 = 20 组合**；**CPU / float32 / 双模型 eager / batch=1 / k=4 / greedy**（`g4_b6_fp32_correctness.py`）。判据与 §4.5.5 **逐字一致**（逐 token id 全等 vs 同引擎 cached AR；logits 全 finite；重复运行确定；vacuity guard；**case 与档位必须跑满**否则不得判 PASS）。判据在运行前入库（commit `3f7fe6f` 预登记 + `9aa6b00` 补漏），**脚本运行前后未做任何修改**；运行于 EOS 修复（`f7097d4`）**之后**。

| case | n | base_bf16 acc/prop | v_fp8 acc/prop | rounds | 耗时 b/v (s) |
|---|---|---|---|---|---|
| natural-1 | 16 | 11/21 | 11/21 | 6 | 24.3 / 26.9 |
| natural-2 | 16 | 16/16 | 16/16 | 4 | 16.9 / 20.6 |
| math | 16 | 9/31 | 9/31 | 9 | 34.5 / 36.1 |
| code | 16 | 14/19 | 14/19 | 5 | 22.3 / 24.6 |
| short | 16 | 12/24 | 12/24 | 6 | 12.1 / 25.8 |
| long | 16 | 12/25 | 12/25 | 7 | 13.6 / 26.8 |
| len13 | 13 | 11/15 | 11/15 | 5 | 9.3 / 19.1 |
| len5 | 5 | 5/5 | 5/5 | 2 | 4.0 / 8.3 |
| len2 | 2 | 2/2 | 2/2 | 1 | 2.2 / 4.8 |
| eos-prone | 16 | 16/16 | 16/16 | 4 | 8.9 / 20.0 |
| **合计** | — | **216/348** | 逐组合相同 | — | 单 case 2.2–36.1 |

- **20/20 组合 `equal=True`**（逐 token id）；`logits finite (all) = True`；`determinism[natural-1] = determinism[long] = True`（各 2 次，两档各测）；两档 `acc/prop` **逐 case 完全相同**（V-FP8 只改变数值、未改变任何决策 ⇒ 与 §3.5.3 的质量结论方向一致）。
- 修后一致性核对：修前小样（`B6F_LIMIT=2`）的 `natural-1 = 11/21`、`natural-2 = 16/16` 在全量运行中**逐字复现** ⇒ EOS 修复（仅 EOS 分支）未扰动其他路径，与设计预期一致。

**同 case 的 bf16 档 vs fp32 档对照（RQ2 用）**

| 口径 | 总 accepted/proposed | 接受率 |
|---|---|---|
| B6-bf16（§4.5.5） | 194/381 | 50.9% |
| B6-fp32（本节） | 216/348 | 62.1% |

- `natural-2`：bf16 首个分歧 @7（tie-collapse）；**fp32 下 16/16 全接受且逐 token 全等** —— 即该 prompt 上 draft 与 target 的决策其实**完全一致**，bf16 的翻面同时污染了"输出 token"和"接受判定"两件事。
- **解释边界（必须写明，不可越界归因）**：这 11.2 个百分点是**复合效应** —— bf16 下 draft 与 target 的 argmax 都会在近并列处翻面，既可能改变"提议了什么"，也可能改变"是否判为接受"，因此**不能**把差值单独归因于 tie-collapse。要归因需专门测量（统计验证位置上 bf16 top-2 gap ≤ 实测形状噪声带 0.25–0.281 的占比 = **tie 频率普查**），已列为 RQ2 的后续测量项。
- **接受率强烈依赖 prompt**：同一档内从 `math` 9/31（29.0%）到 `natural-2`/`eos-prone` 16/16（100%）⇒ **接受率不可用单条 prompt 代表**；这正是协议要求 ≥16 个 prompt families 的实证理由（`math` 类低接受率形态必须被覆盖）。

**门禁口径（防评审误读）**
- **B6 门 = 两档并列登记**：**B6-fp32 = 实现正确性证据（PASS）**；**B6-bf16 = 性能档行为（CHECK 19/20 + 归因）**，其失败与归因原样保留在 §4.5.5，**不作任何修改或豁免**。
- 本节**未放宽任何判据**：判据与 §4.5.5 逐字相同，差异仅在**被测量的配置**（dtype/设备）；"dtype 不是被测对象"已在本节头部（脚本 docstring）**运行前冻结**并入库。
- 结论适用范围：`spec_decode_greedy` 的**控制流 / cache 语义（prefill-trim-rollback 与长度不变量）/ logits 对齐 / 替换 token 单步重算 / EOS 与 budget<k 边界**在良定义配置下逐 token 正确；**bf16 档的 tie-collapse 作为已知数值限制披露**，不构成对性能结论的豁免。
- Phase 4 前提：本条 PASS 满足"实现正确性"门槛；测量仍须携带 §4.5.5 的数值限制说明。

### 4.6 Part 4 小规模测量（G4）— **工具链已修，8K 复验未运行 ⇒ 状态未定论**
> 门禁：Part 4 未定论期间**不启动 G5 正式落盘 / 4-family Pilot / Final**；不更新 speedup 主结论；
> 不改 k=4、模型、KV 方案与正式 factorial；B6-bf16 CHECK 19/20 的严格登记不变。

#### 4.6.1 两个 exploratory run（**受保护，不得覆盖/删除/修改**）
| 登记名 | 文件 | 条件 | 结果 | 已知缺陷 |
|---|---|---|---|---|
| `exploratory_run_1_fixed_order` | `output/part4/part4_dc0bd5d87389.jsonl` | 12（ctx 512/2048/8192 × 2 KV × 2 decode） | 12/12 条件、36/36 run 写入成功；**最后因 `mismatch_rounds_med` 缺失 KeyError** 终止（数据已落盘） | 固定条件顺序；`empty_cache` 在计时区内；标定误报"12 conditions" |
| `exploratory_run_2_reverse_order` | `output/part4/part4_ba0c985bcd5d.jsonl` | 4（仅 ctx=8192，`P4_REVERSE=1`） | 正常 exit 0 | 仅 4 条件；标定把 4 条件外推成"12 conditions / 3.0 GPU 小时"（无效）|

**完整性锚点（2026-09-15 记录；后续任何改动都会被 probe 的前后 sha256 校验报警）**

| 文件 | bytes | lines | sha256 |
|---|---|---|---|
| `part4_dc0bd5d87389.jsonl` | 41026 | 49 | `bfb85f509b6802393637ea9cea9d7f5275579deb457ed6456db9bb087664c53c` |
| `part4_ba0c985bcd5d.jsonl` | 14310 | 17 | `6ad22617a46be08328f980537eb77ed95fb7b38e5548166da3b910f335d4c7c3` |

#### 4.6.2 两次 exploratory 的关键读数与已确立判定（ctx=8192）
| 口径 | run1（base 先，含 empty_cache） | run2（v_fp8 先，无 empty_cache） |
|---|---|---|
| base_bf16 AR decode | 10.87 s → 5.89 tok/s | 6.44 s → 9.93 tok/s |
| v_fp8 AR decode | 2.91 s → 22.00 tok/s | 3.69 s → 17.35 tok/s |
| base SpecDec decode | 8.45 s | 13.64 s |
| v_fp8 SpecDec decode | 9.34 s | 15.60 s |
| base decode Spec/AR | **1.286×** | **0.472×** |
| v_fp8 decode Spec/AR | 0.311×（按 run1 原始行） | 0.236× |
| acceptance（两档） | 0.737 | 0.737（两档皆 `==AR:True`）|
| 接受长度分布（两档**完全相同**） | — | `m=0:3 1:12 2:12 3:0 4:33`（60 轮）|

**判定（事实）**
1. **1.286× 未复现**（run2 同格为 0.472×）⇒ 不是稳定加速，**不得写入速度主表**。
2. `v_fp8 > base`（AR）**两种顺序下方向一致**（3.74× → 1.75×），但**量级漂移 1.7×**，且
   `peak_resv` 8.05–10.47 GiB **越过 8 GiB 物理显存** ⇒ 归为 **allocator/WDDM/thermal-sensitive**；
   在正式 harness 采用随机化/交错 + 环境记录之前，**禁止进入正式速度主表**。
3. 只测 8192 的结果**不得当作 12 条件**：run2 的 3.0h/1.8h 外推作废。
4. **未验证**：v_fp8 优势的机制（字节节省 vs 驻留/paging 效应）**未被直接隔离**，只能作支持性推断。

#### 4.6.3 本轮工具链修正（CPU 自测 **42/42 PASS**）
| 项 | 内容 |
|---|---|
| 条件过滤 | `P4_KV` / `P4_MODE` / `P4_CTX` / `P4_ONLY=<ctx>,<kv>,<mode>` / `P4_REVERSE`；非法值立即报错 |
| 运行身份 | 输出名 `part4_<config_hash>_<RUN_ID>.jsonl`（RUN_ID 默认 UTC 时间戳）；**既有同名文件且非同 RUN_ID 组 → 直接报错退出，绝不以 `"w"` 静默覆盖**；hash 含 filter + condition_list |
| 环境监控 | 每 rep 记录 start/end：CUDA free/total、torch allocated/reserved、nvidia-smi used/temp/power/clocks/driver、allocator 配置；**取不到一律 `available=false` + `error`，不用 0 冒充** |
| 稳定性判据 | median/min/max/spread/**relative spread**；`rel_spread>20%` → 条件标 `CHECK`（仅用于识别测量不稳定，不作显著性）；另查 n_gen 一致 / token hash 一致 / stats 一致 / finite / SpecDec 是否逐 token 等于同档 AR（**跨进程用 `gen_sha1`**）|
| 标定修正 | 只基于**实际运行**条件集分类：`formal 12-condition`（仅 ctx={2048,4096,8192}）/ `exploratory 12-condition`（ctx={512,2048,8192}）/ `8K 4-condition measured cost` / `subset`；**subset 拒绝 per-family 外推**，如需外推必须显式写出假设公式并与实测分列 |
| 新增文件 | `specdec/engine/g4_part4_8k_isolated_probe.py`（两轮启动顺序 × 4 条件，每条件独立子进程，含受保护文件前后 sha256 校验）、`specdec/engine/g4_part4_selftest_cpu.py`（42 项 CPU 自测）|

#### 4.6.4 第二轮工具修复（正确性比较 / 统计 / 监控解析）
| 问题 | 修法 | 落点 |
|---|---|---|
| 单条件进程输出 `==AR=False` 无意义 | 单条件**没有** AR 对照 ⇒ 一律 `ar_equal=null, ar_equal_available=false`；判定只能由父进程跨子进程做；缺对照**绝不返回 False** | `run_once` 行字段 + `token_compare()` |
| 跨进程比较缺前提校验 | `token_compare()` 先校验 `ctx` / `max_new` / `eos` / **`input_token_hash`**；不等时给 `first_diff_index` + 两侧 token id + 长度；前提不全 → `certified=false` + 原因 | 同上 |
| JSONL 缺可比较的身份字段 | 每 run 行新增 `input_tokens`、`input_token_hash`、`max_new`、`eos_id`、`n_generated`、`generated_token_hash`、`generated_token_ids`（≤64 个体积可接受） | `run_once` |
| AR `rel_spread` 聚合出现 `nan%` | 新增 `aggregate_finite()`：只聚合有限值、记录剔除数、全缺失 → `available=false`（**不填 0/NaN**）；探针的 AR 聚合改用它 | `aggregate_finite` / probe §[4] |
| 缺跨**顺序**稳定性 | 新增 `order_stability()`：`order_ratio=min/max`、`relative_order_difference`、>20% → CHECK；与 within-run spread 分列报告 | `order_stability` / probe §[2] |
| 监控数值被当字符串丢弃 | `nvml_snapshot()` 把数值字段转 float，解析失败 → `None` + `unparsed_fields`（**不填 0**）；probe 侧对旧 run 的字符串做兼容强转 | `nvml_snapshot` / `probe.monitor` |
| 显存口径需审计 | 代码审计：`peak_alloc/peak_resv` 为 `max_memory_allocated()` / `max_memory_reserved()` **单次读数**，无相加、无跨阶段累计、不区分 target/draft；主指标 = `peak_alloc`，物理口径 = nvidia-smi used；新增"物理显存饱和"自动判定 | probe §[5] |
| 排序在 `kv=None` 时崩溃（自测抓到） | `compare_across_conditions` 排序键对每个分量做 `str()` | `g4_part4_pilot_measure` |
新增：`g4_part4_mismatch_diag.py`（三个历史不一致条件的确定性定位，**尚未运行**）；
CPU 自测扩展到 **62/62 PASS**（含 NaN 混合/全缺失/单值、`ar_equal` 空值语义、跨条件前提校验、
顺序稳定性、`[N/A]` 解析、旧版字符串兼容）。

#### 4.6.5 8K 独立进程复验结果（**已运行**，stamp `20260921T042125Z`，8 文件 / 40 run）
每条件独立进程、双模型常驻（AR 亦然）、warm-up=1、reps=5、max_new=64、chunk=2048。

| order | kv | mode | decode tok/s 中位数[min..max] | within-run rel% | decode s | prefill s | peak_alloc |
|---|---|---|---|---|---|---|---|
| A | base_bf16 | AR | 22.90 [21.58..24.62] | 13.3 | 2.79 | 7.31 | 6.06 |
| A | base_bf16 | SpecDec | 5.99 [5.93..6.06] | 2.1 | 10.69 | 31.06 | 6.95 |
| A | v_fp8 | AR | 22.06 [21.49..23.74] | 10.2 | 2.90 | 7.12 | 6.01 |
| A | v_fp8 | SpecDec | 5.69 [5.58..5.74] | 2.9 | 11.24 | 31.59 | 6.90 |
| B | base_bf16 | AR | 23.78 [23.50..24.28] | 3.3 | 2.69 | 7.37 | 6.06 |
| B | base_bf16 | SpecDec | 8.42 [8.26..8.59] | 3.9 | 7.60 | 27.61 | 6.95 |
| B | v_fp8 | AR | 20.36 [19.02..20.84] | 8.9 | 3.14 | 7.11 | 6.01 |
| B | v_fp8 | SpecDec | 5.72 [5.62..5.74] | 1.9 | 11.19 | 31.61 | 6.90 |

**跨顺序（between-order）**：`base_bf16 AR` 3.7% `ok`；**`base_bf16 SpecDec` 28.9% → CHECK**；
`v_fp8 AR` 7.7% `ok`；`v_fp8 SpecDec` 0.4% `ok`。
**AR within-run 聚合**：`available=true, n=4`，median 9.56%，max **13.25%**（≤20% 通过）。

**token 比较（跨子进程）**：四个 (order × kv) 配对全部 `ar_equal=true`，AR 与 SpecDec 均 64 token，
生成哈希一致 `610d0002bdfdcb37`；`certified=false`，原因：该 run 的 JSONL 未记录
`input_token_hash`（字段在本次修复后才加入）⇒ 判定**未认证**。

**性能判定**：`SpecDec/AR` decode = A base **0.261×**、A v_fp8 0.258×、B base 0.354×、B v_fp8 0.281×
（**全部未获利**）；`v_fp8/base` AR = A **0.963×**、B **0.856×**（**无速度优势**）；
历史 **1.286× 未复现**；`v_fp8` 仅使 `peak_alloc` 降约 0.05 GiB。

**环境现场（关键）**：`nvidia-smi used = 7811–7887 MiB / 8188 MiB（95.5–96.3%）`；
条件开始时 **`cuda_free` min = 0.00 GiB**；温度 64→77 °C（顺序 B 整体更热）；但
**SM clock 恒定 2520–2535 MHz、mem clock 8000 MHz（无降频）**，功耗 53.7–78.8 W，driver 555.97。
⇒ **8K 双模型运行点压在 8 GB 物理显存边界上（可用显存为 0），且温度/时钟不能解释跨顺序差异**；
`peak_resv` 8.05–10.47 GiB 是**分配器池**（含空闲块与虚拟预留），**不得**当作物理占用超 8 GB。
**归为 memory-state-sensitive**：8K 档耗时被显存状态主导，禁止进入正式速度主表。

#### 4.6.6 仍开放的问题（不得因 8K 全 `ar_equal=true` 而关闭）
`exploratory_run_1_fixed_order` 中下列条件的 SpecDec 与 AR **token 序列不等**，仍待定位：
`512 base_bf16`、`2048 base_bf16`、`2048 v_fp8`。
定位工具 `g4_part4_mismatch_diag.py` 已就位（输出 first_diff_index / 两侧 token / AR 与判定行
top-1·top-2·margin / round·m·分支 / block-vs-serial max\|Δ\| / FP32 控制；按 6 类证据门槛归因），
**当时尚未在 GPU 上运行** ⇒ 此处为历史未决状态；后续已由 §4.7.8 闭环。

#### 4.6.7 Part 4 状态
**当时状态（已由 §4.7.8 更新）为 CHECK**：8K 隔离复验已完成并**证伪"8K 稳定加速"**（1.286× 不可复现、SpecDec 全面慢于 AR、
v_fp8 无速度优势、`base SpecDec` 跨顺序 28.9% 不稳定、运行点显存饱和）；工具链与 62 项 CPU 自测
通过；三个历史 token 不一致条件的归因**未完成**。**不允许进入 G5 / Pilot / Final。**

### 4.7 target block logits 语义验证 —— **索引错误假设被证伪**（2026-09-21）
诊断曾对三个条件（`512/base_bf16`、`2048/base_bf16`、`2048/v_fp8`）报 `TARGET_BLOCK_LOGIT_INDEX`，
并推测实现用 `block_logits[i]` 判 candidate `i`（错位一格）。**代码审查 + 真实模型实测 + 哨兵测试
三方证据均不支持该假设。**

#### 4.7.1 代码审查（`spec_dec.py` 第 83 行）
```python
g = [int(x.argmax()) for x in [saved_next] + [rows[j] for j in range(kp - 1)]]
```
即 `candidate 0 ← saved_next`（prefix next logits）、`candidate i≥1 ← rows[i-1] = block_logits[i-1]`、
`rows[kp-1]` 不判候选而作为下一轮 prefix 行（本设计不产出 bonus）。**这正是规范要求的映射**，
实现中**不存在** `block_logits[i]` 判 candidate `i`。

#### 4.7.2 真实模型语义实测（`g4_block_logits_semantics`，fp32 CPU，prompt 8 tok + 4 候选）
| 对照 | max\|Δ\| | argmax 相同 |
|---|---|---|
| 同位置 `rows[j]` vs `serial[j]`（j=0..3） | **5.090e-05 / 5.722e-05 / 3.910e-05 / 4.041e-05** | **全 True** |
| 错位一格 `rows[j]` vs `serial[j-1]`（j=1..3） | **1.725e+01 / 1.408e+01 / 1.403e+01** | 全 False |
⇒ 语义确认为 `rows[j] = p(·|P, y0..y_j)`；**真错位一格的代价是 14–17 个 logit 单位**，
比诊断里观察到的 bf16 `block-vs-serial` 0.31 大 **约 50 倍**。

#### 4.7.3 哨兵索引测试（`g4_specdec_index_tests`，CPU，**35/35 PASS**）
判定行实测为 `[pre-1, pre, pre+1, pre+2]`（= prefix + rows[0..2]）；把同一份捕获数据按
"buggy 映射"（`block_logits[i]`）重算得 `[6,7,8,9]` ≠ 候选 `[5,6,7,8]` ⇒ **测试有区分力**
（错的实现会在第 0 个候选就被拒）；mismatch 出现在 i=0/1/2/3、全接受不产出 bonus、
EOS（候选/替换/尾块）与预算<K 边界全部通过。

#### 4.7.4 归因更正
诊断中三个条件的 `block-vs-serial max|Δ| = 0.3125 / 0.3086 / 0.5078` 属 **bf16 形状噪声**
（同位置噪声量级 0.25–0.51），而三处 **fp32 margin = 0.0454 / 0.0179 / 0.0345** 全部
**小于现场噪声** ⇒ **正确归因 = `BF16_NEAR_TIE`（信号 ≤ 噪声，该位置在 bf16 下不可判定）**。
诊断脚本的分类器本身有**优先级缺陷**（把"block-vs-serial argmax 不同且 Δ>0.30"排在近贴边判定之前），
已改为：① 结构性错位检查（判定行是否更贴近**相邻位置**而非自身位置，真错位 ≈14–17）
② `signal=|fp32 margin|` vs `noise=同位置 block-vs-serial max|Δ|`（**现场实测，不用人为常数**）。
同时修正 `rounds_from_events` 的事件重建 bug：**全接受轮不调用 `_step`**，下一轮 `saved_next`
取自本轮 `rows[kp-1]`，原先置 `None` 会产出假证据。

#### 4.7.5 处置
- `spec_dec.py` / `spec_cache.py` 的索引与 cache 语义**不做修改**（现实现正确；改动会造成真实回归）。
- §4.6 中"SpecDec 指标全部 INVALIDATED"的处置**暂缓**（`INVALIDATION_PENDING`）：失效前提
  （索引错位）已被代码审查 + 真实模型实测 + 哨兵测试三方证伪；**不基于被证伪的前提作废有效数据**。
  SpecDec 的计时数据仍因 §4.6.5 的**显存饱和 / 跨顺序 28.9% 不稳定**而不得进入正式速度主表
  —— 这是另一条独立且已确立的理由。
- 待用户在 GPU 上重跑 `python -m specdec.engine.g4_part4_mismatch_diag`（新分类器），
  若三个条件均给出 `BF16_NEAR_TIE` 且 `off_by_one=False`，则 Part 4 正确性可闭环。

#### 4.7.6 B6-fp32 端到端复跑（2026-09-21，CPU，**现行代码未改**）
命令 `python -m specdec.engine.g4_b6_fp32_correctness` → **PASS（tiers=2/2, cases=10/10, k=4,
exact token-id；总 accepted/proposed = 216/348）**，与 2026-09-15 首次结果**逐 case 完全相同**
（含 `natural-1 11/21`、`natural-2 16/16`、`math 9/31`、`len2 2/2`、`eos-prone 16/16`；
`determinism[natural-1]=determinism[long]=True`；`logits finite=True`）。
**意义**：这排除了"离散索引/接线错误"——错位一格属逻辑错误，**不会在高精度下消失**；
而 10 类 prompt × 2 KV 档、数百次验证决策在 fp32 下与 AR 逐 token 完全相同。
（首次后台尝试与语义验证并发运行，因两个 1.5B fp32 副本的瞬时内存峰值触发原生访问冲突
`exit=0xC0000005`；**单独重跑即通过**——记一条操作教训：不要并行跑两个重内存任务。）

#### 4.7.7 数据自洽性核查（为何"60 轮"与"100 轮"并存）
§4.6.2 表中的直方图 `m=0:3 1:12 2:12 3:0 4:33`（60 轮）来自 **run 2**（`REPS=3` ⇒ 3×20=60 轮）；
隔离探针为 `REPS=5` ⇒ 100 轮。二者**来自不同 run，均正确**（先前一度误以为是同一 run 的内部矛盾）。
隔离探针（5 rep）恢复直方图段后的实测（四格**完全一致**，跨 KV 档与跨启动顺序皆同）：
`m=0:5  1:20  2:20  3:0  4:55`，总轮数 100，`sum(m)=280`（每 rep 56），全接受轮占比 **55.0%**。
⇒ **决策层（接受/轮数/直方图）完全可复现**，只有**计时**不稳定（§4.6.5 的 28.9%）。

#### 4.7.8 GPU 归因复跑——**Part 4 正确性闭环**（2026-09-21）
新分类器在同一 GPU 环境下连续运行两次
`python -m specdec.engine.g4_part4_mismatch_diag`，两次结果完全一致：

| 条件 | first diff | 同位置 block-vs-serial max\|\Delta\| | 相邻位置 max\|\Delta\| | fp32 margin | 归因 |
|---|---:|---:|---:|---:|---|
| `512/base_bf16` | 3 | 0.3125 | 8.625 / 12.0625 | 0.0454 | `BF16_NEAR_TIE` |
| `2048/base_bf16` | 34 | 0.3086 | 11.4688 / 13.6562 | 0.0179 | `BF16_NEAR_TIE` |
| `2048/v_fp8` | 38 | 0.5078 | 5.5781 / 14.9688 | 0.0345 | `BF16_NEAR_TIE` |

三格均为 `off_by_one=False`、`g_recomputed_ok=True`、`emitted_len_ok=True`、
`invariant_violations=0`。同位置差异远小于相邻位置，且 fp32 真实决策间隔全部小于
bf16 形状噪声；因此三处差异均为近贴边 argmax 翻转，**不是索引、cache 或记账错误**。

**最新 Gate：**
- SpecDec logits 索引 / cache 结构正确性：**PASS**。
- FP32 端到端 token-id 正确性：**PASS**。
- BF16 严格 token equality：**CHECK（可解释的 near-tie 形状漂移）**。
- Part 4 正确性归因：**CLOSED**。
- Part 4 性能：**CHECK**（显存饱和 + `base SpecDec` 跨顺序 28.9% 计时不稳定）。
- 允许进入 G5 记录框架/计时稳定化开发：**YES**。
- 允许启动正式 Pilot / Final：**NO**，先冻结稳定计时协议。

### 4.8 G5 测量稳定化（2026-09-21）—— 工具与 schema 已冻结，**GPU 数据未运行**
> 协议见 `docs/g5_measurement_protocol.md`（冻结 v1）。本节只登记"已实现/已 CPU 验证"与"未运行"。

#### 4.8.1 G5 结果 schema（`specdec/results_schema.py`，`SCHEMA_VERSION = g5-results-1.0`）
- 46 个字段（run 身份 / 条件 / token 身份 / 计时 / 接受统计 / 显存 / 遥测 / 正确性 / 状态 / 可用性），
  `FIELDS` 为唯一定义源；
- **硬约束（`validate_row` 强制）**：值为 null 的字段必须在 `availability` 里显式 false；
  遥测字段不得为字符串或 NaN；`availability` 为元字段（不参与自身检查）；
- `write_g5_run()`：既有文件 + 非 append ⇒ **报错退出，绝不覆盖**；
- 兼容层 `read_legacy_part4()` / `build_g5_row()`：读旧 Part 4 行并映射
  （`t_decode_s→decode_seconds`、`gen_sha1→generated_token_hash`、`m_hist→accepted_length_histogram`、
  旧版字符串遥测强转 float、缺字段置 null），**只读、不改写原文件**（自测校验前后 sha256 相同）；
- BF16 近贴边判定 `classify_bf16_compare()`：**四条证据全部满足**才返回 `BF16_NEAR_TIE`
  （结构不变量通过 / `off_by_one=False` / 同位置比相邻更近 / `|fp32 margin| ≤ 现场噪声`），
  否则 `NEEDS_INSPECTION`（不自动豁免）。

#### 4.8.2 本轮新增工具（全部 **未在 GPU 上运行**）
| 文件 | 作用 | 关键纪律 |
|---|---|---|
| `specdec/engine/g5_driver.py` | 每条件独立子进程编排 + 系统快照 + 固定 seed 随机化 | 父进程只编排，模型加载不计时 |
| `specdec/engine/g5_stability_probe.py` | Task B：12 条件 × 2 个独立随机 block | warm-up 1 + 5 rep；cooldown 10s 不计时；`>20%` ⇒ `status=CHECK` 不入主表 |
| `specdec/engine/g5_cost_breakdown.py` | Task C：AR / SpecDec 阶段分解（CUDA Event 外部包装） | **不改冻结文件**；数字不得入正式主表；`kv_cat` 是 forward 子集、不可相加 |
| `specdec/engine/g5_k_sweep.py` | Task D：k ∈ {1,2,4,8} 诊断扫描 | 正式协议仍 k=4；只输出"修正建议"不自动改配置；不变更 bonus 变体 |
| `specdec/engine/g4_part4_pilot_measure.py` | 新增 `P4_K`（**默认仍为冻结的 4**）+ 行内记 `k` / `k_is_protocol` | 诊断专用开关 |

#### 4.8.3 CPU 验证（已运行）
- `python -m specdec.engine.g5_schema_selftest_cpu` → **PASS 43 / FAIL 0**：字段完整性、
  null↔availability 一致性、遥测字符串/NaN 拒绝、旧 Part 4 映射（含字符串遥测强转）、
  拒绝覆盖、append、读取不改写（sha256 前后一致）、MAD/IQR（样本不足 → None）、
  **三个真实历史条件 → `BF16_NEAR_TIE`** + 5 个反例（`off_by_one` / 不变量违反 / signal>noise /
  同位置不更近 / 缺 margin ⇒ 均不得豁免）、驱动层随机化可复现、快照类型、路径命名；
- 编译：`py_compile` 6 个文件全过（含 `spec_dec.py` / `spec_cache.py`，**均未修改**）。

#### 4.8.4 未运行（**不得填预期值**）
`g5_stability_probe` / `g5_cost_breakdown` / `g5_k_sweep` 的 GPU 数据：稳定性
（median / min / max / MAD / IQR、within-run、between-block）、成本分解（各阶段耗时与占比、闭合残差）、
k 扫描（acceptance、tok/round、rounds、block、draft、decode tok/s、peak）**全部未运行**。

#### 4.8.5 Gate 回答（本轮）
```
G5 schema complete:            YES（43 项 CPU 自测通过，含旧数据兼容层）
condition randomization:       YES（固定 seed + ≥2 block；工具就绪并 CPU 验证）
fresh-process isolation:       YES（驱动层强制；8K 隔离实验已按此执行）
timing variance characterized: NO （需 stability probe 的 GPU 数据）
cost breakdown:                NO （工具就绪，未运行）
k=4 retained:                  YES（正式协议；P4_K 仅供诊断）
correctness gate preserved:    YES（FP32 B6 PASS 保持；未动 spec_dec / spec_cache）
ready for formal Pilot:        NO
```

### 4.9 交接文档（2026-09-21）
`docs/handoff_to_analysis_agent.md`：面向接手**分析/实验设计**的 agent 的自包含交接报告，包含
Gate 总表、硬规则 10 条、代码与工具清单、**数据清单 + 受保护文件 sha256 锚点**、
"已确立事实 / 推断 / 未运行"三分类、开放阻塞项与最小解法、环境与 shell 陷阱、
以及下一步推荐动作与禁止动作。**本文档不引入新数据，只是既有台账的索引与解释。**

### 4.10 AR 基线口径修正（2026-09-21，v1.1）—— target-only 正式主基线 + dual-resident 诊断控制
**问题**：G5 协议原先规定"AR 条件也保持 target+draft 双模型常驻"。它适合控制显存状态，
但**不应作为唯一正式 AR 基线**：真实 AR 不需要 draft，强制双驻留会额外占显存，在 8K×8GB 下
可能推高显存压力、触发 WDDM 换页或改变分配器状态，从而**人为压低 AR、系统性偏向 SpecDec**。

**决策（已实现）**
| 口径 | 驻留 | 角色 | factorial | speedup 分母 |
|---|---|---|---|---|
| `AR-primary` | **仅 target**（draft 从不加载） | primary | ✅（12 之一） | ✅ |
| `AR-dual-resident-control` | target+draft（draft 仅常驻、不参与计算） | diagnostic_control | ❌ | ❌ |
| `SpecDec` | target+draft（算法必需） | primary | ✅ | ❌ |

- 正式 factorial 仍是 **12 cells**；额外 **6 cells** 为诊断控制，`P5_INCLUDE_DUAL_AR_CONTROL=1` 启用，
  汇总**单独分组**，**不得写成 18 条件 factorial**；
- `speedup = T_AR / T_Spec`（>1 = SpecDec 更快）；`residency_penalty = T_AR(target-only)/T_AR(dual)`，
  `penalty < 1`（`slowdown = 1/penalty > 1`）⇒ 双驻留本身拖慢 AR；
- **硬约束**：`AR + target_only` 要求进程**从未加载 draft** ⇒ 不得与 SpecDec 条件同进程
  （`results_schema.plan_process()` 强制，违反报错）。

**新增 schema 字段（`g5-results-1.1`，不兼容变更 → 版本升级）**：`resident_models` /
`ar_residency` / `condition_role` / `formal_factorial_member` / `speedup_denominator_eligible` /
`speedup_claim_allowed`；矛盾组合被 `validate_row` 拒绝（如 `AR target_only` 却声明含 draft、
`dual_control` 却标 `formal_factorial_member=true`、`SpecDec` 带非空 `ar_residency`、
`status=CHECK` 却允许加速声明）。
**旧数据**：`ar_residency` 等未知字段一律 `null` + `availability=false`（**不猜**）；
SpecDec 行因算法必需双模型 ⇒ 驻留可确定并填入；AR 行未知 ⇒ 全 null。

**历史数据措辞更正**：`output/part4/*` 的 AR 行是双驻留口径 ⇒ 记为
**`AR-dual-resident exploratory`**：**VALID EXPLORATORY DATA / INELIGIBLE FOR FORMAL SPEEDUP TABLE**
（它能证明显存饱和与顺序敏感，只是不符合正式主基线口径）。**不删除、不覆盖、不重写。**
同理 `output/cost_8192.txt`（本轮中止的半截 profiling）是按旧口径跑的，**作废但保留**。

**不稳定 cell 报告规则（v1.1 修正）**：原始数据永远保留 → 仍出现在**完整结果表**中 →
必须报 median/min/max/MAD/IQR 与 between-block 差异 → 标 `status=CHECK`、
`speedup_claim_allowed=false` → **不允许**从该 cell 做确定加速声明 → 可从稳定汇总/显著性结论中
排除（注明理由）→ **不得从报告中消失**。这样保持 12-cell factorial 完整，避免选择性报告。

**CPU 验证（已运行）**：`python -m specdec.engine.g5_schema_selftest_cpu` → **PASS 66 / FAIL 0**
（新增 23 项：驻留决策 3 项、非法组合拒绝 3 项、primary 12/control 6 分离 4 项、
speedup/penalty 方向 3 项、旧数据 null 语义 2 项、schema 矛盾拒绝 5 项、不稳定 cell 可报告但禁止声明 2 项、
以及"meta 覆盖 mode 后角色必须重算"这一被自测抓到的真 bug 的回归）。
`g4_part4_selftest_cpu` 仍 **62/62 PASS**（回归未破）。

**口径冒烟（已运行，非性能数据）**：`P4_MODE=AR P4_KV=base_bf16 P4_CTX=2048 P4_REPS=1 P4_WARM=0
P4_AR_RESIDENCY=target_only` → exit 0；日志出现 `[stage] draft 不加载（AR target_only 正式主基线口径）`、
`[mem] 驻留模型=['target'] alloc=2.88 GiB`；行内字段 `ar_residency=target_only`、
`resident_models=['target']`、`condition_role=primary`、`formal_factorial_member=true`、
`speedup_denominator_eligible=true`、`weights_resident_GiB=2.876`（对比双驻留基线 3.81 GiB
⇒ **draft 确实未加载**）。产物 `output/part4/part4_a34b1db83273_g5v11_ar_targetonly_smoke.jsonl`。
**注意**：该冒烟是 REPS=1/WARM=0，`peak_alloc`/`decode_tok_s` **不得**用于 residency 性能对比
（那需要 stability probe 的 warm-up + reps 数据）。
反向守卫亦已实测：无过滤（AR+SpecDec 同进程 + target_only）**立即报错退出、不加载任何模型**。

### 4.11 G5 stability probe 第 1 次运行（2026-09-21/22）—— **环境不合格（P-state）**，代码门禁已修复
**登记产物（**保留，不得覆盖/删除**）**
- `output/g5/g5_g5-results-1.1_5dc7b5eeacd1_g5stab_20260921T090033Z.jsonl`（24 行）
- `output/part4/part4_*_g5stab_b[01]_c*_pri_*.jsonl`（24 个子进程产物，每个 5 measured reps）

**已成功建立的部分**
- **AR target-only 正式主基线口径已生效**：`resident_models=['target']`、`weights_resident ≈ 2.88 GiB`
  （对比旧 dual-resident 3.81 GiB）；8K target-only AR `peak_allocated` = base 5.14 / v_fp8 5.09 GiB；
- 12 个 primary cell × 2 随机 block × (warm-up 1 + 5 reps) 全部跑完；`execution_status` 全 SUCCESS。

**结果：STABILITY FAIL / ENVIRONMENT NOT QUALIFIED**
| 现象 | 数值 |
|---|---|
| between-block 相对差 >20% 的 cell | **7/12**：2048 base AR 66.0%、2048 base Spec 66.9%、4096 base AR 69.9%、4096 v_fp8 Spec 68.5%、8192 base AR 67.4%、8192 v_fp8 AR 68.8%、8192 v_fp8 Spec 24.8% |
| within-run 相对离散 >220% 的 cell | **2 个**：4096 base SpecDec blk0 **222.1%**、8192 base AR blk0 **222.4%** |
| SM 时钟跨度 | **435–2535 MHz**（3.75×） |
| 低频 block（SM<2000MHz，9 个）吞吐中位 | **6.16 tok/s** |
| 高频 block（SM≥2000MHz，15 个）吞吐中位 | **17.67 tok/s（2.9×）** |
| 温度 | 慢运行 61–68 °C、快运行 69–81 °C ⇒ **非热降频** |
| 功耗 / 显存频率 | 慢：15.8–28 W、mem 6000 MHz；快：40–86 W、mem 8000 MHz |

**归因（截至本轮证据）**：GPU 在部分条件进入**低功耗 P-state**（供电/DVFS 层面），
与 Windows 电源模式 / NVIDIA Battery Boost·Whisper Mode / 厂商工具模式 / 供电状态有关；
**在获得新证据前不进一步断言是哪一个外部设置**。
修正此前表述：**8K 显存饱和仍是风险**，但本次 2K/4K/8K 的大幅跨 block 波动
**直接伴随 GPU 时钟与功耗档位变化**。

**clock-matched 子集（唯一可读的加速比）**：仅取 AR 与 SpecDec 双方 SM 均 ≥2400 MHz 的 block
（3 对）→ decode speedup 中位 **0.284**（0.200–0.684）、e2e 中位 0.152 ⇒ **全部 <1**。
**严格按 20% 诊断线**（不含 P-state 阈值）只有 `2048/v_fp8` 一对可入资格：
逐 block decode ≈ **0.533 / 0.538**、e2e ≈ **0.538 / 0.541**（10 rep 合并中位 decode **0.533**）——
但按**已实现的门禁**（SM ≥ 2000 MHz）它**也不合格**：该档 SpecDec 两 block 的
SM 中位为 1950 / 2070 MHz（< 满频 2520）⇒ 判 `ENV_INVALID`，**不得声明**。
**不为了得到合格结论而调低阈值**；若确要改阈值，必须在**下次运行之前**预登记。
（上述数字可用 `python scripts/g5_report_from_run.py 20260921T090033Z` 离线复算，section [3]/[8]。）

**本轮数据的定性**：**VALID EXPLORATORY / ENVIRONMENT-DIAGNOSTIC DATA —
INELIGIBLE FOR FORMAL SPEEDUP CLAIM**（它成功揭示了 P-state 切换，不是"无效数据"）。

**代码门禁（本轮修复，CPU 自测 92/92）**
- **状态分层**：`execution_status`(SUCCESS/ERROR/OOM) 与 `measurement_status`(OK/CHECK/ENV_INVALID/NOT_MEASURED)
  分离；**子进程 `spread_flag=CHECK` 必须把 measurement_status 降为 CHECK**（`exit=0` 不再写成 OK）；
  记录 `within_run_check_reason` / `within_run_rel_spread_pct`；
- **speedup 声明资格**：`speedup_claim()` 强制 10 条（target-only 分母、条件/hash 一致、reps 完整、
  两侧 measurement OK、between-block 合规、无 OOM/NaN/ENV_INVALID、正确性已记录、非诊断控制、
  无 profiling 混入）；不满足时输出 `NO CLAIM：<原因>`，**不再打印"可作确定声明"**；
- **P-state 预检** `specdec/engine/gpu_pstate_preflight.py`：负载期采样（空闲时钟**不参与判定**）、
  可配置阈值、`available=false` 不填 0、不合格即中止该 block；独立可复用；**已在 GPU 上实测**（见 §4.12）；
- **运行期审计**：每 cell 由 `monitor_start/end`（计时区间外）汇总
  `active_sm_clock_mhz / active_memory_clock_mhz / active_power_w / pstate_qualified / pstate_reason`；
- **平台门禁**：AC 供电 + 电源方案（**不使用空闲时钟**，空闲 315 MHz 属正常）；
- **离线复核** `scripts/g5_report_from_run.py`：对历史 run 复算 within/between/P-state 与
  NO CLAIM 原因（只读）。

**当前 Gate**（**已被 §4.14 的缩小版 run 取代**，此处保留作第一次运行的历史记录）
```
target-only AR baseline:        PASS（口径已生效且有实机证据；性能数据本身受环境污染）
within-run status propagation:  PASS（代码 + 测试）
speedup claim gating:           PASS（代码 + 测试；本轮 6/6 对全部 NO CLAIM）
GPU P-state preflight:          PASS（2026-09-22T03:58:05Z 实测 QUALIFIED；工具已修复 1 个真实缺陷）
STABILITY (run #1):             FAIL
ENVIRONMENT (run #1):           NOT QUALIFIED
ENVIRONMENT (当前时刻):          QUALIFIED（预检通过；是否在下一次长实验中保持合格仍需 block 级复检）
formal Pilot:                   HOLD（等缩小版复验）
```

### 4.12 GPU P-state 预检首次实机运行（2026-09-22T03:58:05Z）—— **发现并修复 1 个真实缺陷，复跑 QUALIFIED**

**产物**：`output/g5/pstate_preflight_pstate_20260922T035805Z.json`（2227 B，唯一命名，不覆盖）

**发现缺陷（工具级，非测量结果）**
- 第 1 次运行**崩溃**：`TypeError: 'Event' object is not callable`（`threading.py:1141 _wait_for_tstate_lock`）。
- **根因**：`Sampler.__init__` 把停止信号存在 `self._stop`，**覆盖了 `threading.Thread._stop` 方法**；
  `Thread.join()` 内部会调用 `self._stop()`，于是把 `Event` 当函数调用。
- **教训（已写入测试）**：自测此前只覆盖 `verdict_from_sampler`（纯函数），**没有跑真实入口的线程 start/stop/join**；
  这与"必须测工具自身入口"是同一条教训的第二次出现 ⇒ 新增 3 项 CPU 测试（⑯ 断言 `Sampler._stop` 未被覆盖、
  ⑰ 真实 start/sleep/stop 并断言采到样本、⑱ 快照不可用时计入 `failures` 而非造数）。
- **修复**：属性改名 `_stop_flag`（`specdec/engine/gpu_pstate_preflight.py`）。

**修复后实测（run_id `pstate_20260922T035805Z`）**
| 项 | 值 |
|---|---|
| 空闲快照（**不参与判定**） | SM 210 MHz / mem 405 MHz / 6.01 W（证明"用空闲时钟判资格"会误杀） |
| 负载期样本 | n=11，采样失败 0 次；负载 8717 次 2048² fp16 matmul / 5.63 s（≈1549 matmul/s） |
| SM clock | median **2355**，min 210（仅首个仍在爬频的样本），max 2460 MHz |
| mem clock | median **8000**，min 405，max 8000 MHz |
| 功耗 / 温度 | median 89.58 W / 82.0 °C |
| **verdict** | **QUALIFIED（可进入正式计时）**，退出码 0 |

**说明与边界**
- 该 JSON 是**环境证据，不是性能数据**；`matmul_per_s` 只用于证明负载真实发生，**不得**作为任何吞吐结论。
- 阈值（SM≥2000 / mem≥7500 MHz）是**本机经验值**，非通用 GPU 标准；**本轮未修改任何阈值**。
- 预检通过**只说明当下合格**；长实验中仍须依赖 stability probe 的 **block 级前置复检**（不合格则该 block 跳过），
  这已在 §4.11 的 run #1 中体现（当时的失败正是靠 cell 级审计发现的）。

**预检失败记录（`pstate_preflight_pstate_20260923T154630Z.json`，2026-09-23）**
```
verdict = ENV_INVALID   退出码 3
负载期 SM clock median 210 MHz（min 210 / max 2250）、mem clock median 405 MHz、power median 9.19 W
platform_state: on_ac=False（battery_status_code=1）、power_scheme=Performance、sm_clock_now=210 MHz
gate blocker: 当前在**电池供电**：dGPU 会被限频，计时无效。请插上电源后重跑
```
- **同一台机器、同一段 2048² matmul 负载**，当天早些时候（AC 供电）预检得到 SM median **2355–2400 MHz**、
  功耗 87–91 W；本次只有 **210 MHz / 9.19 W** ⇒ 差异的唯一共变量是**供电状态**，不是负载强度、不是温度（61 °C）。
- ⇒ **本次 8K 变体探针按纪律未启动**（不满足计时前置条件）。这也给 §4.11 run #1 的低频 block
  （SM 435–1300 MHz / mem 6000 MHz）提供了**一个高度一致的候选解释：当时可能也处于电池供电或供电切换中**。
  **这是待验证的候选解释，不是结论**——run #1 当时**没有**记录 `on_ac` 字段，无法回溯证实。
- 教训：`on_ac` 必须随每批数据落盘（现已在 `platform_state()` 中采集），否则事后无法判断环境。


### 4.13 分片命名缺陷修复：防止第 2 次 run **追加**进第 1 次的分片（2026-09-22，**在开跑前拦住**）

**发现（由"准备缩小版复验"触发，尚未造成任何数据损失）**
- 探针给子进程的 run_id 原为 `g5stab_b{b}_c{ci}_{tag}_{ctx}_{kv}_{mode}_{res}`，**不含本次 run 的 UTC 时间戳**；
- 而测量核心在 `P4_ONLY` 模式下 `append_ok=True`（单条件模式本就允许追加到同一 RUN_ID 组），
  于是**同一配置的第二次 run 会以同名分片文件 + 追加写入**落到第一次的文件里。

**后果（若不修）**
1. `output/part4/part4_*_g5stab_*` 这批**受保护**分片会被第 2 次 run 的数据改写（记录的 sha256 随之失效）；
2. 一个分片里混入**两个不同环境**的 rep（第 1 次低频、第 2 次满频），而报表按"该文件所有 run 行"取中位数
   ⇒ 会把两组环境悄悄平均掉，**得到看似稳定、实则混合的数字**；
3. 12 条件全量复验与第 1 次 run 的 cell 序号一旦重合，就会命中同一路径。

**修复（两处，均已落地）**
1. 子进程 run_id 加入本次 run 的 UTC 时间戳：
   `g5stab_{stamp}_b{b}_c{ci}_{tag}_{ctx}_{kv}_{mode}_{res}`（`stamp` 取自聚合 run_id）；
2. 开跑前**硬校验**：若目标分片 `part4_*_{cid}.jsonl` 已存在 ⇒ 直接 `SystemExit` 拒绝启动，
   把"误追加"变成"启动即失败"（防御纵深，即使时间戳逻辑将来被改坏也拦得住）。

**报表（`scripts/g5_report_from_run.py`）同步修复**
- 原行为：无参数 = **合并所有 g5stab 分片**，多次 run 会被混进同一张表；
- 新行为：无参数或 `latest` = **只取最新一次 run**（分片名内时间戳为等长 UTC，字典序最大即最新），
  给子串则按子串过滤，并在首行打印实际选中的 run 与可选清单。
- 回归：对第 1 次 run 离线复算通过，`[8]` 段对 **6/6 对全部输出 `NO CLAIM：<原因>`**。

**同时新增**：`scripts/run_g5_reduced.cmd`——把"预检 → 干跑 → 缩小版复验 → 离线报表"串成一条命令，
用带引号的 `set "VAR=value"` 形式，避免 cmd 里 `set X=a & set Y=b` 把 `&` 前的空格并进变量值，
从而**杜绝"条件名带空格 ⇒ 静默少跑条件"**（参见本节 4.11/自测 ⑲⑳）。

### 4.14 G5 stability probe 缩小版（run `g5stab_20260922T040625Z`）—— **P-state 已修复；canonical 记账已修正**

**登记产物（全部保留，不得覆盖/删除）**
| 文件 | 内容 | 备注 |
|---|---|---|
| `output/g5/g5_g5-results-1.1_0e4ed83b6fb3_g5stab_20260922T040625Z.jsonl`（58009 B） | **1.1 版 canonical（含 4 类错记）** | **保留作审计证据**；sha256 `10045ba8331afed26efc91a469d5b6cb28c50941a9300c1cb281b7f1a1cc01e9` |
| `output/g5/g5_g5-results-1.2_re44bd9ca4bb_g5stab_20260922T040625Z.jsonl`（61448 B） | **1.2 版 canonical（修正后重建）** | 由只读分片重建，**未重新测量** |
| `output/part4/part4_*_g5stab_20260922T040625Z_*.jsonl`（12 文件） | 6 条件 × 2 block 的原始分片 | **受保护**，本次只读 |

**配置**：ctx {2048, 8192} × KV base_bf16 × {AR target-only, SpecDec} + AR dual-residency 控制；
2 随机 block、warm-up 1 + 5 measured reps、k=4（协议冻结）、max_new 64、chunk 2048。

**① P-state 问题已解决（对比 §4.11）**
| 项 | §4.11 第 1 次 | 本次 |
|---|---|---|
| SM 时钟 | 435–2535 MHz | **≈2490–2520 MHz** |
| 显存时钟 | 6000 MHz（部分 block） | **8000 MHz** |
| `pstate_ok` | 部分 false | **全 true** |
| 温度 / 功耗 | 61–68 °C / 15.8–28 W | ≈82 °C / 68–70 W |

**② 2K base_bf16：稳定且**允许**声明**
| 量 | AR target-only | SpecDec |
|---|---|---|
| decode tok/s（blk0 / blk1） | 33.04 / 29.69 | 13.26 / 12.79 |
| between-block | **10.1%**（OK） | **3.5%**（OK） |

```
decode speedup = 0.406×   e2e speedup = 0.434×   speedup_claim_allowed = true
⇒ SpecDec decode 慢 2.46×、端到端慢 2.30×（合并全 rep 中位数口径）
```
**必须注明**：这是**缩小版** stability probe（2 长度 × 1 KV），**不是** 16-family 正式矩阵。

**③ 8K base_bf16：AR 稳定，SpecDec 不稳定 ⇒ 不得声明倍数**
| 量 | AR target-only | SpecDec |
|---|---|---|
| decode tok/s（blk0 / blk1） | 29.33 / 29.51 | 9.53 / 5.46 |
| between-block | **0.6%**（OK） | **42.7%（CHECK）** |
| prefill（blk0 / blk1） | 3.449 / 3.439 s | **31.958 / 32.000 s** |
| within-run rel | ≈2% | **1.9% / 2.0%（OK）** |

- 两 block 的 SM 2520 MHz / mem 8000 MHz / ≈82 °C / 68–70 W **完全相同**，
  prefill 也几乎相同（31.96 vs 32.00 s），**只有 decode 从 6.72 s 变成 11.72 s**；
- ⇒ **不是** 低功耗 P-state、**不是** 热降频、**不是** prefill 波动、**不是** token 决策随机性；
- 合并口径 decode ×0.236、e2e ×0.136 —— **该比值没通过稳定性 Gate，不得写成"8K 正式 speedup"**；
- 只能记录：*AR target-only 稳定；SpecDec 跨 block 差 42.7%；所有 clock-matched 观测均 <1；
  方向上未观察到加速；正式倍数不得声明*。

**④ target-only vs dual-resident 对照** ⚠️ **本条已被 §4.16 部分推翻，请以 §4.16 为准**
| 分组 | AR target-only | AR dual-control | 说明 |
|---|---|---|---|
| 2K | 29.7–33.0 tok/s | 30.4–32.0 tok/s | 差异很小 |
| 8K（**本表**，单会话 2 block） | 29.4 tok/s | 23.1 tok/s | 曾据此认为"仅多驻留 draft 就让 8K AR 慢 ≈21%" |

- **更正（2026-09-23）**：§4.16 用 **8 个独立进程**测 8K，`ar_dual` = 28.65–29.98 tok/s（中位 **29.81**）、
  与 `ar_target`（中位 29.66）**不可区分**，prefill 也同为 3.34 s（本表当时是 7.40/7.74 s）
  ⇒ **"21% residency 代价"不重现，降级为"单会话观察"**；
- 仍然成立且与本节其它部分无关的结论：**正式 AR 必须用 target-only 口径**（口径选择不依赖这个数字），
  且 dual-control **永远不作分母**（canonical `claim_allowed=false`）。

**⑤ canonical 记账修正（1.1 → 1.2，详见协议 §7c）**
| 缺陷 | 1.1 的错误表现 | 1.2 修正 |
|---|---|---|
| summary `*_med` 未映射 | `prefill_seconds`/`e2e_tok_per_s`/`acceptance`/`accepted`/`proposed`/`rounds`/`draft_calls`/`target_block_calls`/`mismatch_rounds`/`all_accept_rounds` **全为 null** 且 `availability=false` | 按显式映射表取 `*_med`，重算后**全部有值**且 `availability=true` |
| 子进程凭空判等价 | 驱动器"一条件一进程"使 `ref_gen` 为空，却拿 `gen_sha1([])` 去比 ⇒ `ar_equal=False` + `certified=True`（**8192 两侧 hash 明明都是 `610d0002bdfdcb37`**） | 子进程在**没有对照时不再判定**；父进程按 key 配对后填 `ar_equal`（8192 **True**；2048 False 且 `first_diff_index=34`，与已知 BF16 near-tie 一致） |
| 措辞跨层 | `within_run_check_reason=' | between-block CHECK'`，而 within-run rel 只有 1.9% | 三层状态分离：`within_run_status=OK` / `between_block_status=CHECK` / `environment_status=OK`；原因写进 `between_block_check_reason`；声明原因输出 `NO CLAIM：SpecDec between-block CHECK（42.7% > 20%）` |
| dual-control 误标资格 | 4 行 `speedup_claim_allowed=True` | 角色为 `diagnostic_control` 的行**永远** false（校验器兜底） |
| availability 陈旧 | 父进程补值后 availability 仍为 false ⇒ 自相矛盾行 | `availability` 改为**值的纯函数**，每次 finalize 重算；跨进程填写后**原地刷新** |

**⑥ 本轮 Gate**
```
P-state qualification:         PASS（两 block 预检均合格；全 cell pstate_ok=true）
summary mapping:               PASS（10 个统计字段全部落表 + available=true；151 项 CPU 自测）
cross-process equality:        PASS（8192 True / 2048 False，父进程填写且已认证）
claim blocker wording:         PASS（within-run 与 between-block 分离）
dual-control claim discipline: PASS（4/4 行 claim=false）
canonical validator:           PASS（写盘前强制；合规行必须通过、错行必须被抓）
8K SpecDec 波动原因:             **未定位** ⇒ 下一步 g5_8k_spec_variance_probe
complete 12-cell stability:    HOLD（8K 未解释前不跑）
cost breakdown:                HOLD（等环境与工具）
formal Pilot:                  HOLD
```

### 4.15 8K SpecDec 工具级 OOM 缺陷与修复（2026-09-22，**在交付给使用者之前拦住**）

**现象**：新写的 8K SpecDec 变体探针首次运行即崩 —— `spec` 角色的子进程**跑约 20 s 后退出且不产出任何文件**，
另外两个 AR 角色的子进程正常完成。

**根因（工具缺陷；不只是"显存不够"）**
- `spec_dec.spec_decode_greedy` 内部调用 `spec_dec._prefill(model, ids)`；
- 引擎默认的 `_prefill` 是 **一次前向吃完整个 prompt**（`model._run(embed_tokens(ids))`，**不分块**）；
- 8K 时该层 attention 的 `torch.softmax(scores, dim=-1, dtype=torch.float32)` **一次性申请 ≈3.00 GiB**
  （`scores` 约 `(1, 12, 8192, 8192)`），8 GB 卡上直接
  `torch.OutOfMemoryError: Tried to allocate 3.00 GiB ... 9.31 GiB is allocated by PyTorch`。

**为什么只有 spec 角色崩**：AR 角色走工具自己写的**分块** `chunked_prefill`；spec 角色的原实现
**在调用 `spec_decode_greedy` 之前先跑了一次 `chunked_prefill`** ——
① 那次 prefill **纯粹浪费**（引擎会**再** prefill 一次），② 它先把分配器池撑大，
③ 随后引擎的**非分块** prefill 才撞上那笔 3 GiB 单次申请。

**修复（与测量核心既有做法对齐）**
- 新增 `timed_chunked_prefill(acc, chunk)`：把它换成 `spec_dec._prefill`（**分块 + 单独计时**），
  再调用**一次** `spec_decode_greedy`，`decode = total − prefill`
  —— 与 `g4_part4_pilot_measure.spec_decode_timed` **完全同一模式**（该模式在缩小版 run 中已稳定产出
  prefill 31.96 s / decode 6.72–11.72 s）；
- 插桩 pass 同样换成 `ph["prefill_target"].wrap(chunked_prefill)`（否则插桩 pass 也会 OOM）；
- **同类缺陷一并修**：`g5_cost_breakdown.run_specdec_profiled` 原本也没包 `_prefill` ⇒ 它在 8K 首次运行
  **同样会 OOM**；现已同样处理（该工具此前从未运行过，属**运行前**拦下）。

**修复后实测验证（冒烟：1 个 spec 子进程、reps=1、warm=1、插桩 1 次；run_id `smoke2`）**
```
P-state qualified（SM 2295 / mem 8000 MHz）
warm0: decode=13.77s prefill=23.88s        ← warm-up 效应明显
rep0 : decode= 8.89s prefill=29.03s  dec_tok/s=7.20  n_gen=64
插桩：outer=37.89s 残差=25ms(0.1%)          ← 阶段闭合良好
exit=0
```
- prefill 29–31 s **与缩小版 run 的 31.96 s 一致** ⇒ 现在测到的确实是引擎真实的那次 prefill；
- 插桩残差 0.1% ⇒ 阶段划分可用。

**纪律说明（诚实登记）**
- 修复前首跑**被我方测试脚本超时中断**，残留 3 个 AR 子进程文件：
  `output/g5/variance/p5v_p5v_20260922T180840Z_{ar_target_0,ar_target_1,ar_dual_0}.json`；
- **保留不删**，但**必须视为中断的残缺 run，不得用于任何结论**（缺 `spec` 角色、无 summary）；
- `output/g5/variance/p5v_smoke2_spec_0.json` 是**单 rep 冒烟产物**，同样**不得当作实验数据**；
- 两类文件各带自己的 run_id，探针**只读与自身 run_id 匹配的文件** ⇒ 后续正式运行不会被污染。

### 4.16 2026-09-23 两次 8K 变体运行：**SpecDec 角色全部缺失**（工具缺陷已修）+ **"21% residency penalty"结论被推翻**

**运行与覆盖度**（配置均为 ctx=8192 / base_bf16 / k=4 / max_new=64 / chunk=2048 / warm=1 / reps=5 /
roles=[spec, ar_target, ar_dual] / procs=4）

| run_id | 子进程产物 | spec | ar_target | ar_dual | summary |
|---|---|---|---|---|---|
| `p5v_20260923T045620Z` | 2 | **1** | 1 | 0 | 无（中断） |
| `p5v_20260923T094741Z` | 8 | **0** | 4 | 4 | 有 |
| `p5v_20260923T100031Z` | 8 | **0** | 4 | 4 | 有 |

⇒ **后两次"跑完了"的运行里，被测对象（SpecDec）一个样本都没有**：4 个 spec 子进程全部崩溃且**不落盘**
（若是环境不合格会写 `skipped` 文件，实际连这个都没有 ⇒ 是真崩溃，不是预检跳过）。
父进程当时**照样打印报告 + 写 summary**，`Q1` 只写"进程数 <4，不做双档判定" ⇒ 属于**假成功**，已修（见下）。

**这两次运行里**有效**的数据：8K AR 对照（各 8 个独立进程，全部 P-state 合格 SM 2250–2370 MHz）**
| role | decode tok/s（8 个进程） | 中位 | max/min | prefill（8 个进程） | within-run rel |
|---|---|---|---|---|---|
| `ar_target`（target-only） | 28.25 / 29.82 / 29.66 / 29.74 / 28.99 / 29.47 / 29.84 / 29.66 | **29.66** | **1.056** | 3.34–3.45 s | 1.6–8.1% |
| `ar_dual`（双驻留控制） | 29.91 / 28.65 / 29.98 / 29.97 / 29.80 / 29.76 / 28.71 / 29.82 | **29.81** | **1.046** | 3.34–3.35 s | 1.2–6.5% |

⇒ **8K 的 AR（两种驻留口径）在 8 个独立进程上极稳**：跨进程极差 **≤5.6%**，远优于缩小版 run 的
SpecDec 42.7%；prefill 在 16 个样本里 **3.34–3.45 s（±1.6%）**。

**⚠️ 结论更正：§4.14 第 ④ 条的"8K 仅多驻留 draft 就让 AR 慢 ≈21%"不成立（不重现）**
| 来源 | 8K AR dual-control | prefill | target-only |
|---|---|---|---|
| §4.14 缩小版 run（1 个 session、2 个 block） | 22.89 / 23.23 tok/s | **7.40 / 7.74 s** | 29.33 / 29.51 tok/s |
| 本次（8 个独立进程） | **28.65–29.98 tok/s（中位 29.81）** | **3.34–3.35 s** | 28.25–29.84（中位 29.66） |

- 新版**8 个进程**一致给出 `residency_penalty = T_target/T_dual ≈ 0.995`（即**几乎无差**），
  而不是 0.79；两次运行连 **prefill** 都差 2.2×（7.4 s vs 3.34 s）；
- ⇒ 正确表述：**"8K residency 代价 ≈21%"降级为"单会话观察，未复现"**；
  在 8 个独立进程的证据下，**8K 下 dual-control 与 target-only 在 decode 上不可区分**；
- 同时说明：**8K 慢档不只出现在 SpecDec**——§4.14 那个 session 里连 AR dual-control 的 prefill 都慢了 2.2×，
  即"慢档"是**会话级/进程级**的状态，不是 SpecDec 算法特有。**成因仍未定位。**

**另一个新增未解异常：SpecDec 的 8K prefill 在不同会话间相差 2.4×**
| 来源 | SpecDec 8K prefill | decode |
|---|---|---|
| 缩小版 run（§4.14，2 block） | 31.958 / 32.000 s | 6.72 / 11.72 s |
| 本人冒烟（`smoke2`，1 进程 1 rep） | 29.03 s | 8.89 s |
| 本次成功样本（`045620Z` spec_0，5 reps） | **13.11–13.24 s**（极稳） | 10.58–12.27 s |
- 该样本的插桩 pass 给出 `prefill_target total=13107 ms calls=2 median=6553.6 ms`，
  **两次调用各 6.55 s** ⇒ "target 一次 + draft 一次"却**几乎等时长**，无法用模型大小解释；
- 正式 pass 与插桩 pass 的 prefill **互相一致**（13.2 vs 13.1 s）⇒ 不是计数错误，
  因此**必须按模型拆分**才能判定（工具已改，见下）；
- 参照：AR **dual-resident** 在 8K 的 prefill 只有 **3.34 s** ⇒ SpecDec 的 13.2 s 仍偏慢，
  与"双模型同驻 + KV 增长下的带宽/分配器压力"方向一致，但**在拆分数据出来前不得定论**。
- 决策层完全可复现（重要正面证据）：该样本 `rounds=20 / proposed=76 / accepted=56 / mismatch=8 /
  all_accept=12`，与缩小版 run 的 8K SpecDec **逐项相同**；`acceptance=0.7368` 一致。

**本轮修复的工具缺陷（都在交付前/交付中拦住）**
1. **子进程结果只在最末尾落盘** ⇒ 若插桩 pass OOM，**正式结果一起丢**（这正是 4 个 spec 子进程"零产物"的原因）。
   现在：正式 pass 结束先原子写出 `stage=formal_done`，插桩完成后原子升级为 `stage=complete`；
   插桩 pass 包 `try/except`，失败只记 `profiling_error`，**不影响正式结果**。
2. **父进程"缺整个角色"也照常出报告** ⇒ 现在 `coverage_report()` + `missing_required_roles()`：
   `spec` 或 `ar_target` 为 0 个样本时**拒绝打印报告、拒绝写 summary**，退出码 4，
   并把每个子进程的**退出码**写进 `p5v_<run_id>_failures.json`（崩溃细节仍在子进程 stderr）。
   已用真实数据回放验证：`094741Z` / `100031Z` 会被判 `missing_required=['spec']` 而拒绝出结果。
3. **prefill 未按模型拆分** ⇒ 新增 `prefill_target_s` / `prefill_draft_s`（正式 pass），
   插桩 pass 的 phase 也拆成 `prefill_target` / `prefill_draft`。
4. 样本不足时 `Q1` 必须显式写"**不能**据此对双档下结论"（不再只写"进程数<4"）。

**数据定性**
- `094741Z` / `100031Z`：**INCOMPLETE RUN**——其 16 个 AR 样本可作"8K AR 对照、n=8"的证据使用，
  但**这两轮没有回答任何关于 SpecDec 的问题**；summary 中的 `Q1/Q3` 均无效。
- `045620Z`：中断的 run，仅 1 个 spec 样本，**单样本不作为双档结论**，但可作为 prefill 异常的证据之一。
- 命名提醒：子进程文件名为 `p5v_p5v_<stamp>_<role>_<i>.json`（run_id 自带 `p5v_` 前缀 ⇒ 出现重复前缀），
  **纯属命名冗余、不影响解析**，为保持与既有产物连续**不改名**。

**根因已定位（2026-09-23，用户贴回完整 traceback）——8K SpecDec 撞物理显存墙**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 896.00 MiB.
GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free.
Of the allocated memory 6.07 GiB is allocated by PyTorch, and 1.08 GiB is reserved by PyTorch but unallocated.
  spec_dec.py:61   drf_lg, drf_cache = _prefill(draft, prompt_ids)      ← **draft 的 prefill**
    → 工具的分块替身 → model_self.py:169  torch.softmax(scores, fp32)   ← 需要 896 MiB
```
- 896 MiB = draft 最后一个 chunk 的 fp32 softmax：`14 heads × 2048(q) × 8192(kv) × 4 B = 940 MB` ✓ 数量级吻合；
  同理 target 的最后一个 chunk 需要 `12 × 2048 × 8192 × 4 = 806 MB`；
- 崩溃瞬间：PyTorch 池 **6.07 GiB allocated + 1.08 GiB reserved-unallocated = 7.15 GiB**，
  物理显存 **0 bytes free** ⇒ 请求**连续** 896 MiB 失败；
- **为什么 AR 角色不崩**：AR target-only 的 `peak_allocated` 只有 5.07 GiB、AR dual 5.99 GiB，
  而 SpecDec 8K 是 **6.95 GiB**（§4.14 实测）⇒ SpecDec 比 AR 多需要约 **1.9 GiB**，
  正好越过 8.00 GiB 物理上限。
- ⇒ **8K 双模型 SpecDec 在本机是"踩在物理显存线上"的条件**：能否跑完取决于
  ① 桌面/其它进程占用多少显存、② 分配器碎片（`inactive_split` 曾达 **1501 MiB**）。
- **实测旁证**：agent 事后检查时，这台机器上同时跑着**一个游戏**（`Gamble With Your Friends.exe`）
  与约 **20 个占 GPU 的应用**（Doubao / QQ / WeChat / Teams / CloudMusic / 多个 msedgewebview2 等），
  `nvidia-smi` 显示 **1440 MiB / 60% 利用率**、**28 个 C+G 进程挂着 GPU**；
  此时预检的"空闲快照"是 SM **2520 MHz / 53 W**（正常空闲应是 210 MHz / 5 W）
  ⇒ **"空闲快照"本身就是外部占用的探针**。
- 之前能跑通的三次（`045620Z` spec_0、agent 冒烟、缩小版 run 两 block）都发生在外部占用较低的时段。

**由此得到一个强有力的待验假设（⚠️ 仍是假设，不是结论）**
> 8K 下 SpecDec 的"快档/慢档"可能不是算法或 P-state 问题，而是**是否被挤出物理显存**：
> 一旦外部占用/碎片把可用显存压到临界线以下，KV 增长与软max 的**连续**申请就会触发
> WDDM 分页/重试（实测 `num_alloc_retries` 已出现过 1），于是 decode 与 prefill 同时变慢
> （§4.16 里 prefill 13.2 s → 32 s、decode 6.72 s → 11.72 s 正好是同一现象的两面）。
> **判据（必须先跑出数据）**：慢档进程的 `cuda_free_mib` / `smi_used_after_load` /
> `inactive_split_peak_mib` / `num_alloc_retries` 是否系统性高于快档。
> **在拿到该判据之前，不得把原因写成 WDDM paging。**

**为此新增的数据字段**（否则上述结论只能靠事后回忆）
- 每个子进程在**载入权重后立刻**记录 `smi_used_after_load`（物理占用）与
  `alloc_after_load.cuda_free_mib`；
- 新增 `gpu_consumers`：`nvidia-smi` 列出的 **GPU 上其它 C+G 进程数与非 python 进程名**；
- 新增 `alloc_conf`：把 `PYTORCH_CUDA_ALLOC_CONF`（若设置）随数据落盘，
  否则事后无法判断两批数据可比性。

**下一次运行的处置（按顺序，均需人执行）**
1. **关掉游戏与占 GPU 的应用**（当前 1440 MiB / 28 个进程）——**零协议影响**，先试这个；
2. 若仍 OOM：采用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   （直接针对 1.08 GiB "reserved but unallocated" 的碎片；**属分配器行为变更**，
   已随行记录 `alloc_conf`，采用时须在台账登记，且不得与未采用的批次混用）；
3. 若仍 OOM：把 `P5V_CHUNK` 降到 1024 作为**单独标注的**诊断臂（最大瞬时申请减半），
   **绝不与 chunk=2048 的行混用**。

**量化结论（2026-09-23 17:06/17:07/17:1x 三次直接诊断，决定性）**
1. **拟合测试脚本 `scripts\run_g5_fit_check.cmd`**（1 个 SpecDec 进程、warm1+rep1、无插桩，约 2 min）
   在**游戏已关闭、AC 供电、SM 2340 MHz** 的条件下**连续两次失败**：
   - 两次都是 **退出码 3221225477 = `0xC0000005`（原生访问冲突）**，日志停在
     `Loading weights: 100%|338/338` 之后、`[load] weights copied on CPU` 之前 ⇒
     崩溃发生在**权重装载阶段**，且**不是 Python 异常**（`try/except` 抓不到，`crash.txt` 因此为空）；
   - ⇒ 说明在 WDDM 超订下，**显存耗尽可以表现为原生崩溃而不是干净的 OOM**。
2. **直接跑该子进程**（不经脚本）则走到更远：两个模型**装载成功**，随后在同一处崩于
   `torch.OutOfMemoryError`。崩溃瞬间的**实测数字**（`crash.txt` 已落盘）：
```
alloc_conf='expandable_segments:True'      ← 配置**确实生效**（torch 读到了）
peak_allocated_mib = 6115.7      peak_reserved_mib = 6736.0
inactive_split_peak_mib = 945.06     num_alloc_retries = 2     num_ooms = 1
cuda_free_mib = 771.0            cuda_total_mib = 8187.5
请求：768 MiB   ← **在 771 MiB "空闲" 的情况下仍然失败**
```
3. **因此根因是"刚好越线 + 碎片"，不是"差很多"**：
   - PyTorch 池峰值需要 **≈7.4 GiB**（6736 reserved + 768 新请求）；
   - 而本机非 CUDA 占用约 **1.4 GiB**（27–28 个 C+G 进程 + 桌面 + 驱动），
     `8187.5 − 6736 = 1451 MiB` 正好被它吃掉；
   - 池内还有 **945 MiB `inactive_split`**（已释放但**无法复用**的碎片）⇒ 那笔 768 MiB **连续**申请
     既拿不到新物理页、也用不上碎片；`num_alloc_retries` 已重试 2 次后判 OOM。
   - **失败裕度只有 3 MiB**（771 请求 768）⇒ 任何几百 MiB 的外部占用变化都足以决定成败，
     这也解释了为什么同一份代码在不同时段 "有时成功、有时失败"。
4. **`expandable_segments:True` 在 Windows 上根本不被支持**（**2026-09-26 更正**）：
   当时写成"设置生效后失败数字逐字节相同"——**这个说法是错的**。§4.20 的日志里 PyTorch 明确警告：
   ```
   UserWarning: expandable_segments not supported on this platform
   (Triggered internally at .../c10/cuda/CUDAAllocatorConfig.h:28)
   ```
   ⇒ 该开关**从未生效**，所以"逐字节相同"不是"无效"的证据，而是"没执行"的证据。
   正确表述：**这条路在本平台不可用**（Windows 不支持），不要再尝试它。
5. 对照参考：缩小版 run 里 8K SpecDec 的 `peak_reserved_gib = 10.47`（**> 物理 8 GiB**）
   ⇒ 那两次成功是**靠 WDDM 分页挪到共享内存**才跑完的；这与"慢档"（decode 11.72 s、
   prefill 32 s）方向一致 —— **但"慢档 = 分页"仍是假设，判据见上。**

**⛔ 重大环境事故（2026-09-24 13:50:14）：整机黑屏硬挂，非预期关机**
事件日志（原样引用）：
```
2026/9/24 14:00:11  Microsoft-Windows-Kernel-Power  ID 41
  "The system has rebooted without cleanly shutting down first."
2026/9/24 14:00:19  EventLog  ID 6008
  "The previous system shutdown at 13:50:14 on 2026/9/24 was unexpected."
```
- **没有任何 BugCheck / 内存转储**（`C:\Windows\MEMORY.DMP` 与 `C:\Windows\Minidump\*.dmp` 均不存在）
  ⇒ 不是蓝屏，而是**整机挂死后被强制断电**（用户描述：黑屏、无法操作）；
- 无 WHEA 硬件错误、无 TDR(`4101`) 记录；NVMe 健康状态 Healthy/OK；
- 事故窗口（13:30–13:55）内 **本项目工具零产物**：最后一个产物是
  `p5v_p5v_20260924T040510Z_failures.json`（12:05:49），此前 1 小时 45 分内没有任何
  preflight / 子进程 / summary 写入 ⇒ **无法把这次硬挂归因到某一次具体运行**；
- 但**同一工作负载在 24 小时内已产生 3 类资源型失败**：① GPU VRAM OOM（768 MiB 请求 /
  771 MiB "空闲" / 945 MiB 碎片）② CPU commit 不足导致的**原生 `0xC0000005`**（载入模型阶段）
  ③ 本次整机硬挂。三者都出现在"8K 双模型 + 系统内存/显存余量很低"的条件下；
- 另有 **`ESENT 447` 在 13:09 报 `SRUDB.dat` B-Tree bad page link** —— 该症状常见于
  **此前已发生过非正常关机**之后，属"已经在损坏系统数据库"的信号，必须严肃对待；
- 仓库与数据在重启后**完好**：`HEAD=6cb215b`、受保护 JSONL 的 sha256 全部 MATCH、
  `output/part4` 47 个文件齐全。

**⇒ 安全结论（写死，后续不得违反）**
1. **commit 余量不足时禁止启动 8K 双模型工作负载**；已加入硬护栏 `commit_guard()`
   （默认要求 **≥16 GiB 可用 commit**，依据载入峰值 **≈9.1–9.6 GiB** + 余量）：
   父进程在启动任何子进程**之前**检查，子进程在载入模型**之前**再查一次，
   不足则**拒绝启动**（父进程退出码 5 / 子进程记 skipped），绝不"先跑了再说"；
2. **每次 GPU 运行前先看两行**：`nvidia-smi` 的 `memory.used`（外部显存占用）与探针打印的
   **commit 可用量**（重启后当前 ≈18 GiB，上限 36.5 GiB）；
3. 要真正抬高 commit 上限：**调大 pagefile**（当前 7667 MB ⇒ 上限仅 36.5–39 GiB）。
   这是**系统设置变更、零代码影响**，且只在**载入阶段**用得到（不在计时区内），不影响测量；
4. 每次硬挂后建议以管理员身份跑 `chkdsk C: /scan`（本次无管理员权限，未执行）。

**由此确定的两个缓解手段（供人决策）**
| 手段 | 效果 | 代价 / 性质 |
|---|---|---|
| A. 把外部占用压到最低（关浏览器/IM/音乐/网盘等，目标 `nvidia-smi` 空闲 ≤300 MiB） | 直接补上那 ~0.5–1.0 GiB 的缺口 | **零协议影响**，首选；但裕度只有 3 MiB，不保证 |
| B. `P5V_CHUNK=1024`（最大瞬时申请 768→约 384 MiB） | 把 3 MiB 的裕度变成约 390 MiB，**大概率直接解决** | **属配置变更**：必须整轮（含 AR 对照）统一用 1024，并登记为"chunk=1024 臂"，**其 prefill 数字不得与 chunk=2048 的缩小版 run 混用** |

**执行顺序（脚本已就绪）**：先 A → `scripts\run_g5_fit_check.cmd`（2 min）；
若仍失败 → 在脚本里放开 `P5V_CHUNK=1024` 再跑 fit check；通过后再跑
`scripts\run_g5_variance.cmd`（33 min）。

**本轮诊断新增/修复的工具项**
- `scripts\run_g5_fit_check.cmd`（新）：2 分钟"**到底装不装得下**"检查——1 个 SpecDec 进程、
  warm1+rep1、无插桩；失败时明确指向 `*_crash.txt` 与 `*_failures.json`；
- 崩溃文件现在同时包含**崩溃瞬间的分配器状态**与 `alloc_conf`；
- 子进程记录新增 `smi_used_after_load`（载入权重后立刻看物理占用）与
  `gpu_consumers`（GPU 上其它 C+G 进程数/名称）；
- 修复 **AR 插桩 pass 的负数闭合残差**（`outer` 原先只算 decode、却与含 `prefill_target` 的
  阶段和相减 ⇒ 残差 −3.4 s）；改为从 prefill 之前计时并改名 `outer_total_s`；
- `[V2]` 的"样本不足"警告与 `[V7]` 的空表现在都会明确打印；
- `missing_required_roles` 只在**被请求的**角色内判缺失，因此 `P5V_ROLES=spec` 的冒烟不会被自己的硬拦拒掉。

**第三次尝试（`p5v_20260923T155628Z`，2026-09-23 23:57 本地）—— 环境不合格，全部跳过**
- 9 个子进程（spec 0–2、ar_target 0–2、ar_dual 0–2）各写出一条
  `skipped = "P-state 不合格 -> 未启动计时（数据不产生）"` 记录，**没有任何计时数据**；
- 原因是**仍在电池供电**（`Win32_Battery.BatteryStatus = 1`），dGPU 锁在 SM 210 MHz / mem 405 MHz；
- **该 run 由 agent 的脚本超时中断**（本意只想跑 `--dry-run`，但漏设 `P5V_DRY_RUN=1` ⇒ 启动了真实运行；
  已记教训：跑本工具前必须先确认 `P5V_DRY_RUN`）。**无孤儿进程、无数据损失**；
- 意义：这批文件是"**环境不合格 ⇒ 一个样本都不产生**"的正面证据——`skipped` 与"崩溃不落盘"
  是两种可区分的失败模式（后者已由崩溃栈落盘修复覆盖）。按新逻辑，这种"全角色 0 样本"的 run
  **会被父进程拒绝出报告并退出码 4**。
- **定性：无测量数据，不得用于任何结论。**

### 4.17 attention 后端替换（FlashAttention / SDPA）的可行性论证（2026-09-24，**尚未上 GPU**）

**动机**：§4.16 的 OOM 就是手写 attention 里那一笔
`torch.softmax(scores, dim=-1, dtype=torch.float32)`（draft 896 MiB / target 806 MiB）。
FlashAttention / mem-efficient 后端**不物化** score 矩阵（分块 + 在线 softmax），
因此它**直接消除这笔申请**；而 PagedAttention 只改 KV 的**摆放方式**
（8K KV 逻辑仅 224 MiB，占 `peak_allocated` 的 3.1%），治不了这笔瞬时申请。

**现行实现（逐行确认，`model_self.py:156-170`）**
```python
k_rep = kv_repeat(k_all, heads // kv_heads)              # GQA 手工展开
scores = torch.matmul(q, k_rep.transpose(-2,-1)) / sqrt(head_dim)   # bf16
scores = scores.masked_fill(偏移因果掩码, -inf)          # cache 为 None 时 diagonal=1，否则 diagonal=T-s+1
attn   = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)   # ← 806–896 MiB
out    = torch.matmul(attn, v_rep)
```

**⚠️ 两个已实测的语义陷阱（必须在 amendment 里写死）**
1. **`is_causal=True` 不是我们的掩码**。PyTorch 的 `is_causal` 等价于
   `triu(ones(L,S), diagonal=1)`（**左上**对齐），而分块 prefill 的 query 是全长 KV 的
   **最后 s 个** token，需要 `diagonal=T-s+1`（**右下**对齐）。CPU fp64 实测：
   `s=8,T=8` → max|Δ| **2.145e-07**（等价）；`s=5,T=32` → **2.157**；`s=1,T=32` → **2.156**
   ⇒ **只有 `s == T` 时才能用 `is_causal`**。
2. **SDPA 的 bool mask 语义与 `masked_fill` 相反**：SDPA 中 **True = 参与**，
   所以要传 `keep = ~triu(..., diagonal=T-s+1)`。

**正确形式已被 CPU 证明等价**（同一组形状，fp64）：
```
keep-mask 形式  s=8  T=8   max|Δ| = 2.145e-07   ✅
keep-mask 形式  s=5  T=32  max|Δ| = 1.122e-07   ✅
keep-mask 形式  s=1  T=32  max|Δ| = 5.906e-08   ✅
```

**环境可行性（已核对，无需新依赖）**：`torch 2.6.0+cu124`、RTX 4060 = **sm_89**（Ada）；
`F.scaled_dot_product_attention` 支持 **`enable_gqa=True`**（GQA 2 个 KV 头无需手工展开）；
`FLASH_ATTENTION / EFFICIENT_ATTENTION / CUDNN_ATTENTION / MATH` 后端全部可用；
`torch.nn.attention.sdpa_kernel` 可用于**强制**指定后端，避免静默回退。

**工具**：`specdec/engine/g5_attn_backend_probe.py`（**不加载模型、不改引擎**）
- 先做 **CPU 语义自检**（上 GPU 之前先证明形式正确，否则直接退出码 3）；
- 再在真实形状上量**单笔瞬时显存**与**耗时**：
  `s=2048/T=8192/h=12,d=128`（target chunk）、`h=14,d=64`（draft chunk，OOM 崩点）、
  `s=5`（块验证）、`s=1`（decode）；`s=8192` 全长前向为 opt-in（手写路径约 3 GiB）；
- 每一格都与手写路径比 `max|Δ|` 并显式标注**是否等价**——不等价的"加速"一律作废；
- 安全性：只造 q/k/v，峰值约 1–2 GiB，**不建 8K KV、不载双模型**，
  远离触发 WDDM 超订 / commit 压力的那条路径；
- 产物 `output/g5/attn/attn_backend_<stamp>.json`（唯一命名）。
- **状态：已实现并通过 CPU 自检；GPU 微基准待由人在交互式终端执行。**

**若微基准显示收益足够，amendment 的范围与代价（供决策）**
- **只改 `SelfAttention.forward` 的 score→output 一段**；`spec_cache` 的 cache 契约、
  `trim_cache`、rollback、`_propose`/`_verify` 出口不变量、索引映射**全部不动**
  ⇒ 风险显著低于 PagedAttention（后者要动 cache 布局，正是本项目最脆弱的地方）；
- 但 `model_self.py` 是**冻结**文件 ⇒ 必须走协议 amendment + 你签字；
- 数值路径会变（kernel 内 fp32 累加 + 输出回 bf16 的舍入不同）⇒ 必须重跑
  **G1 / B4 / B5 / B6-fp32 / B6-bf16 / 35 哨兵 / BF16 tie 重基线**；
- 所有既有性能数字（含 2K 的 0.406×）将变成 **"pre-SDPA 时代"**，不得与之后的数据混表。

**4.17.x GPU 算子级微基准结果（2026-09-26，`attn_backend_20260926T091642Z.json`）—— 结论：值得做**

| 形状 | 变体 | 峰值瞬态 | 耗时 | vs 手写 max\|Δ\| | 等价 |
|---|---|---|---|---|---|
| **prefill chunk target**<br>12 头 / 2048q / 8192kv / d128 | manual | **1998.1 MiB** | 34.01 ms | — | — |
| | sdpa **默认（= MATH）** | 1932.1 MiB | 47.67 ms | 1.465e-03 | ✅ |
| | sdpa **FLASH** | **无可用 kernel** | — | — | ❌ |
| | sdpa **EFFICIENT** | **无可用 kernel** | — | — | ❌ |
| | **sdpa CUDNN** | **74.1 MiB** | **4.56 ms** | 1.465e-03 | ✅ |
| **prefill chunk draft**<br>14 头 / 2048q / 8192kv / d64 | manual | **2288.4 MiB** | 38.54 ms | — | — |
| | sdpa 默认 | 2150.9 MiB | 51.57 ms | 1.465e-03 | ✅ |
| | **sdpa CUDNN** | **59.4 MiB** | **4.12 ms** | 1.465e-03 | ✅ |
| **block verify**（5q） | manual | 52.9 MiB | 0.63 ms | — | — |
| | **sdpa CUDNN** | **0.2 MiB** | **0.39 ms** | 8.545e-04 | ✅ |
| **decode**（1q） | manual | 48.9 MiB | 0.50 ms | — | — |
| | sdpa CUDNN | **ERROR：cudnn 不支持 seq_len = 1** | — | — | ❌ |

**三个平台事实（torch 2.6.0+cu124 / Windows，均由 PyTorch 自己的警告给出）**
1. `This torch build was not compiled with flash attention` ⇒ **FLASH 后端不可用**；
2. `Memory Efficient attention has been runtime disabled` ⇒ **mem-efficient 后端被禁用**；
3. `both fused kernels require query, key and value to have the same num_heads`
   ⇒ **GQA（12q/2kv）不被 fused kernel 支持**；只有 **CUDNN** 能吃 `enable_gqa=True`。
4. 另：`cudnn SDPA does not support sequence length 1` ⇒ **decode（s=1）必须保留手写路径**。

**为什么手写路径要 2 GiB（一次要 4 个整尺寸张量）**
```
scores = q@kᵀ            (1,12,2048,8192) bf16 = 403 MiB
scores.masked_fill(...)  再一份                = 403 MiB
softmax(..., fp32)       fp32 升采样           = 806 MiB
.to(bf16)                再回写一份            = 403 MiB
───────────────────────────────────────── 峰值 ≈ 2015 MiB（实测 1998 MiB ✓）
```
CUDNN 融合核只用 **74 MiB** 工作区 ⇒ **省约 1.92 GiB/次调用**（draft 侧省 2.23 GiB）。

**对"踩线"的预期（这是关键）**
- 目标瞬时 768–896 MiB 只是**其中一份**；实际可省 **~1.9–2.2 GiB**（我上一轮按 0.9 GiB 估，**低估了**）；
- `peak_allocated` 7121 MiB → 预期 **~5.2 GiB**；`peak_reserved` 10736 → 有望**降到 8188 MiB 物理线以内**
  ⇒ **有机会真正"不踩线"**（不再走 WDDM 分页 ⇒ 时间也可能变得可复现）；
- prefill 时间：target 侧 attention 合计 ≈ 34 ms × 28 层 × 4 chunk ≈ 3.8 s → CUDNN ≈ 0.5 s
  （即 target prefill 7.34 s 里约 **3.3 s 是手写 attention**，可省）；
  draft 侧 ≈ 3.7 s → 0.4 s，但 draft prefill 实测 21–28 s，**其余部分仍待解释**。

**风险与代价（不变）**
- 只改 `SelfAttention.forward` 的 score→output 一段；**cache 契约 / trim / rollback / 索引映射全不动**；
- 需要**混合实现**：`s ≥ 2` 走 CUDNN-SDPA，`s == 1`（decode）保留手写（因 cudnn 不支持）；
- 数值：max|Δ| **1.465e-03**（bf16 输出）+ `attn_mask` 的 bool keep-mask（**不是** `is_causal`，见上）；
- 必须重跑 **B6-fp32 / B6-bf16 / 35 哨兵 / BF16 tie 重基线**；
  **G1（CPU fp32 对齐 transformers）不受影响**（CUDNN 只在 CUDA 路径启用）；
- 所有既有性能数字成为 **"pre-CUDNN 时代"**，不得混表。

### 4.18 chunk 分档扫描结果（2026-09-26）—— **8K SpecDec 跑通，且 chunk=2048 就够**；并首次按模型拆出 prefill

**环境前提（本轮同时修好）**
- commit：`c:\pagefile.sys` 固定 **8192 MB**（D 盘那条无效，已移除）⇒ 上限 **33.6 → 39.6 GB**，
  可用 commit 最坏 **9.3 → 20.5–22.7 GB**（载入需求 ≈9.1–9.6 GiB）；
- 显存外部占用 **476 MiB**（22 个 C+G 进程），P-state 合格（SM 2385 / mem 8000 MHz）。

**结果：`chunk=2048` → `FITS`（exit=0，113 s）⇒ 不需要改 chunk，与既有数据同口径**
| 量 | 实测 |
|---|---|
| prefill 中位 | **28.434 s** |
| ├ target 侧 | **7.340 s** |
| └ **draft 侧** | **21.094 s** ← 异常，见下 |
| decode 中位 | 10.044 s（**6.372 tok/s**；1 rep ⇒ `rel=0.0%` 不可用于稳定性判断） |
| e2e | 1.663 tok/s |
| `peak_allocated` | **7120.6 MiB**（6.95 GiB） |
| `peak_reserved` | **10736 MiB（10.48 GiB > 8 GiB 物理！）** ⇒ 靠 WDDM 换页硬撑 |
| `inactive_split` | 1371 MiB |
| `num_alloc_retries` | **0** |
| `cuda_free_mib` | **0.0**（物理已满） |
| `smi` 载入后 | 4705 MiB |

⚠️ **仍属"踩线"**：物理显存吃满（`cuda_free=0`），reserved 超物理 2.48 GiB。
本次能过是因为**外部占用低（476 MiB）**且 commit 充足；换到外部占用 ~1.5 GiB 的时段
（§4.16 记录的失败现场）就会 OOM ⇒ **8K SpecDec 的可运行性依赖外部占用，不是稳定可用的条件。**

**🔑 新发现（RQ2 的关键线索）：prefill 按模型拆开后，慢的是 draft 侧**
```
draft 21.094 s   vs   target 7.340 s     ← draft 只有 0.5B（target 的 1/3），却慢 2.9×
```
**交叉验证（两次独立测量吻合）**：
| 测量 | 8K target 侧 prefill |
|---|---|
| 本节 SpecDec 进程内（draft 常驻**且参与计算**） | **7.340 s** |
| §4.14 缩小版 run 的 **AR dual-control**（draft 仅常驻、**不**参与计算） | **7.40 / 7.74 s** |
| §4.14 缩小版 run 的 **AR target-only**（draft 完全不加载） | **3.45 / 3.44 s** |
| §4.16 8 进程独立测量的 AR target-only | **3.34–3.45 s** |

⇒ **"draft 常驻"本身就让 target 的 8K prefill 慢 2.16–2.30×**，且**与 draft 是否参与计算无关**
（dual-control 7.40 s 与 SpecDec 内的 7.34 s 几乎相同）。
**这修正了 §4.16 的更正**：那里用 8 个进程测得"8K residency 对 **decode** 无影响（penalty 0.995）"——
那只对 **decode** 成立；对 **prefill** 而言 residency 代价**真实且可复现（2.2×）**。
机制候选（**未隔离，不得当结论**）：prefill 的瞬时激活大（768–896 MiB/chunk）+ KV `cat` 复制，
一旦 draft 的 0.95 GiB 权重常驻，可用显存就被压到触发 WDDM 换页的水平；decode 的瞬时很小（s=1），故不受影响。
**待验证**：慢档进程的 `cuda_free` / `inactive_split` / `peak_reserved` 是否系统性更差。

**工具**：`specdec/engine/g5_chunk_fit_sweep.py`（新增，含 commit 护栏复查与
`FITS` / `REFUSED_COMMIT` / `OOM` / `NATIVE_CRASH` 分类）+ 一键脚本 `scripts\run_g5_chunk_sweep.cmd`；
产物 `output/g5/variance/p5v_chunksweep_20260926T073128Z_*`。

**下一步**：chunk 已确认无需改动 ⇒ 直接跑完整变体探针
（`scripts\run_g5_variance.cmd`，3 角色 × 4 进程 ≈33 min）→ 4 个 SpecDec 样本判双档 + 阶段归因。

### 4.19 同配置复跑（9 分钟内两次）揭示了"双档"的**判别性证据**（2026-09-26）

**两次运行：完全相同的配置**（ctx 8192 / base_bf16 / SpecDec / k=4 / max_new 64 / **chunk 2048** /
warm1 + rep1 / 同一份代码 / 同一台机器 / 外部 C+G 进程数都是 22）

| 量 | run `...T073128Z`（15:31） | run `...T074020Z`（15:40） | 变化 |
|---|---|---|---|
| decode | 10.04 s（6.37 tok/s） | 11.18 s（5.72 tok/s） | **+11.4%** |
| prefill 合计 | 28.43 s | 32.39 s | **+13.9%** |
| ├ **draft 侧** | 21.09 s | **24.98 s** | **+18.4%** |
| └ **target 侧** | 7.34 s | **7.41 s** | **+1.0%** |
| `peak_allocated` | 7120.6 MiB | 7120.6 MiB | **0.0%** |
| `peak_reserved` | 10736 MiB | 10736 MiB | **0.0%** |
| `inactive_split` | 1371 MiB | 1371 MiB | **0.0%** |
| `num_alloc_retries` | 0 | 0 | — |
| SM 时钟中位 | 2385 MHz | **2430 MHz（更高）** | ↑ |
| commit 可用 | 20.5 GiB | 19.6 GiB | −0.9 |

**三个排除（本轮最有价值的部分）**
1. **不是分配器/池大小造成的**：三项显存指标**逐字节相同**（0.0%）⇒ 慢的那次并没有"池更大"。
2. **不是时钟造成的**：慢的那次 SM 时钟**反而更高**（2385 → 2430 MHz）。
3. **不是整机/全局变慢**：若是全局争用，target 与 draft 应同步变慢；
   实际是 **target +1.0%、draft +18.4%** —— 变化**高度偏向 draft 侧**。

**机制（与既有证据一致，但仍属未隔离的机制）**
- `spec_decode_greedy` 的顺序是 **先 target prefill、后 draft prefill**；
  等 draft 开始 prefill 时，target 的权重 + KV 已占满显存（实测 `cuda_free = 0`）。
- ⇒ **后跑的一方（draft）承担显存压力**；它每次要申请的瞬时最大
  （chunk=2048 时 **896 MiB**，比 target 的 768 MiB 还大）⇒ 在满显存下付换页代价。
- 这解释了 **target 侧稳定（7.34–7.41 s）而 draft 侧波动（21.1–25.0 s）**。
- 也解释了 §4.18 的 residency 效应：draft 只是"常驻"（AR dual-control）就能让
  **target 的 prefill 从 3.34 s 变 7.40 s（2.2×）** —— 压力来自"多占的 0.95 GiB"。

**方法学后果（必须写进结论）** ⚠️ **已被 §4.20 修正**：本节的"11–18%"只基于 **2 个样本**；
§4.20 用 **3 个同配置样本**测得极差 **61%**（10.04 / 11.18 / 16.22 s）。
⇒ 正确的地板是"**最高 61%**"，因此 8K SpecDec 的**任何**速度结论都不可声明（详见 §4.20）。

**附**：B/C 组因**误用系统 python**（shell 未激活 venv ⇒
`ModuleNotFoundError: No module named 'torch'`）未运行。已提供一键脚本
`scripts\run_g5_alloc_ab.cmd`：**显式使用 `.venv\Scripts\python.exe`**，依次跑
`A 基线 / B expandable_segments:True / C max_split_size_mb:256`。

### 4.20 分配器 A/B/C 结果（2026-09-26）—— **两个旋钮都无效**；且**同配置噪声高达 61%**（修正 §4.19 的"11–18%"）

**五组 8K SpecDec 单进程测量**（均为 chunk=2048、warm1 + rep1、同一份代码、同一台机器）

| run | alloc_conf | decode_s | tok/s | prefill_s | ├ target | └ draft | `peak_alloc` | `peak_resv` | `inact_split` | retr | temp | power | sm_MHz | smi | C+G | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `T073128Z` | **无（A1）** | 10.04 | 6.37 | 28.43 | 7.34 | 21.09 | 7121 | **10736** | **1371** | 0 | 74 | 49.9 | 2385 | 7915 | 22 | 20.5 |
| `T074020Z` | **无（A2）** | 11.18 | 5.72 | 32.39 | 7.41 | 24.98 | 7121 | **10736** | **1371** | 0 | 78 | 48.8 | 2430 | 7907 | 22 | 19.6 |
| `T074909Z` | **无（A3）** | **16.22** | 3.95 | 36.11 | 7.46 | 28.65 | 7121 | **10736** | **1371** | 0 | 77 | 56.1 | 2430 | 7824 | 24 | 17.9 |
| `T075105Z` | `expandable_segments:True` | **7.79** | 8.22 | 26.66 | 7.16 | 19.50 | 7121 | **10736** | **1371** | 0 | 78 | 63.0 | 2392 | 7897 | 24 | 17.8 |
| `T075241Z` | `max_split_size_mb:256` | **3.38** | **18.94** | **56.62** | 9.14 | **47.48** | 7121 | **15188** | **678** | 0 | 78 | 74.3 | 2182 | 7881 | 24 | 17.6 |

**判定（按"目标 = `peak_resv` ≤ 8188 MiB 且 `inact_split` → 0"）**
1. **B（`expandable_segments:True`）不是"无效"，而是"从未生效"**：日志里 PyTorch 明确警告
   `expandable_segments not supported on this platform`（`CUDAAllocatorConfig.h:28`）
   ⇒ 该选项**在 Windows 上不被支持**。显存三项与 A **逐字节相同**（7121 / 10736 / 1371）
   正是"设置被忽略"的表现。它那次 decode 7.79 s 落在 **A 自己的噪声带内（10.04–16.22 s）**
   ⇒ **不可归因**。（同时更正 §4.16 第 4 条。）
2. **C（`max_split_size_mb:256`）把碎片砍半（1371 → 678）却把池撑大（10736 → 15188）**
   ⇒ 在"踩线"这件事上**离目标更远**（我们要 ≤8188，它到 15188）。其"decode 3.38 s 很快"伴随
   **prefill 56.62 s / draft 47.48 s（翻倍）** ⇒ 是**代价在两段之间转移**，总时间 60.0 s 反而不如 A3 的 52.3 s。
3. ⇒ **分配器这条路走死**：没有任何旋钮能把 `peak_resv` 压到物理线以内。

**C 的行为不是噪声，而是该配置的稳定属性**（同一进程内 warm0 → rep0 复现）：
```
A3: warm0 decode=10.23s prefill=23.70s  →  rep0 decode=16.22s prefill=36.11s   ← 进程内退化 +59%/+52%
B : warm0 decode=10.71s prefill=21.03s  →  rep0 decode= 7.79s prefill=26.66s
C : warm0 decode= 3.75s prefill=66.59s  →  rep0 decode= 3.38s prefill=56.62s   ← 两次都是"prefill 慢/decode 快"
```
- **C 的"慢 prefill、快 decode"在 warm0 与 rep0 上一致** ⇒ 是 `max_split_size_mb:256` 带来的**真实行为改变**
  （池更大 15188 但碎片更少 678 ⇒ 大瞬时申请更贵、小瞬时复用更好）；
- **A3 在同一个进程内从 warm0 到 rep0 退化 59%** ⇒ 支持"**状态累积型退化**"：
  同一进程内只多跑了一遍，就慢了一半以上。

**⚠️ 更重要的发现：同配置的噪声比 §4.19 登记的高得多**同配置（`alloc_conf=None`）三次：decode **10.04 / 11.18 / 16.22 s** ⇒ 极差 **61%**（§4.19 基于 2 个样本曾登记为 11–18%）。
且三次的 **`peak_resv` / `inact_split` 逐字节相同**（10736 / 1371）⇒ **计时差异与分配器状态无关**。
环境侧：`smi` 7824–7915 MiB（±1.2%）、C+G 22–24、commit 17.6–20.5 GiB ⇒ 外部显存**没有明显漂移**，
但 **power 从 49.9 W 单调升到 74.3 W**、decode 在 A 组内**单调变慢** ⇒ 支持"**状态累积型退化**"（未隔离）。

**⇒ 对结论的直接后果（按预登记 20% 规则）**
- **8K SpecDec 在这台机器上"能跑，但不能测"**：三次同配置极差 61% ≫ 20%
  ⇒ **8K 的任何速度倍数都不得声明**（这不是偶发，是该条件在 8 GB 卡上的真实属性）；
- 这也回过头解释了 §4.14 的 42.7% 跨 block 差异 —— 它就是**这个条件的噪声量级**；
- 8K 仍然**可以**提供**内存侧**证据：`peak_reserved` 10.48–10.72 GiB > 8 GiB 物理、`cuda_free = 0`、
  `peak_allocated` 稳定 7121 MiB（零方差）⇒ "**双模型 + 8K 超出 8 GB 物理显存**"这条**可声明**。

**建议路线（供决策）**
1. **8K 降级为"受限条件"**：只登记内存证据与"计时不可复现"这一结论，不发布速度倍数；
2. **主战场转向 2K / 4K**（4K 峰值显著低于 8K，很可能完全落在物理线内 ⇒ 可拿到干净数据）
   —— RQ1 依然能回答（2K → 4K 的干净曲线 + 8K 的"超限"边界）；
3. 若要**救 8K**，结构性手段只剩 **SDPA/FlashAttention**（消掉 768/896 MiB 的单笔瞬时，§4.17 已论证）；
   但**不保证**成功 —— 因为 `peak_resv` 超物理 2.5 GiB 的主因是**分配器池 + WDDM 分页**，
   而不是那笔瞬时。投产前应先做 §4.17 的算子级微基准。

### 4.21 结果汇报文档（2026-09-26）—— 给分析 Agent 的入口

新增 **`docs/results_brief_for_analysis_2026-09-26.md`**（**结果汇报 / 入口文档**）：
一页读完现状、**可声明的事实**（附台账锚点）、**已排除的假设及证据**（P-state / 热降频 / prefill /
token 随机性 / CPU commit / 分配器碎片 / 时钟 / 外部占用漂移）、**红线**、**决策点（D1 CUDNN amendment / D2 8K 降级）**、
数据与代码索引、Gate 总表。`docs/handoff_to_analysis_agent.md` 顶部已加指针。

同时在该简报中固定一条口径说明（避免同一实验出现两个数）：
**speedup 点估计 = 全部 measured rep 合并中位数**（协议 §7c.6）⇒ 2K = decode **0.406×** / e2e **0.434×**；
若改用"逐 block 中位数再平均"则为 **0.416× / 0.444×**（差 ~2.3%，**引用时必须注明口径**）。

### 4.22 prefill-only cuDNN 分支：实现 + 算子级显存成立，但**正确性判据未通过**，差异已定位到 eager 基线的 bf16 分数舍入（2026-09-26）

首次也是唯一一次修改冻结文件 `model_self.py`（+211/−13，纯增量：原 eager 代码块逐字保留，只在其前加一个裁决点）。
新增显式策略 `eager` / `cudnn_prefill`（**默认 `eager`，历史行为不变**），融合**只**允许在**调用方显式标注的
`prefill` 阶段**发生。`spec_dec.py` / `spec_cache.py` **一行未改**（契约不变）。

**为什么必须显式标注阶段（两条都实测过）**
- 不能用 `q_len > 1`：`spec_dec._verify` 的候选验证也是多 token（`s = kp ∈ [1..k]`）⇒ 会把**验证**误判成 prefill；
- 不能用 `cache is None`：它只标识**第一块**，`chunked_prefill` 的第 2..n 块仍是 prefill；
- **真实踩坑**：`spec_dec._prefill` 是**直接**调 `model._run` 的（`spec_dec.py:17`），不经过
  `chunked_prefill` ⇒ 不标注就跑在 `other` 阶段 ⇒ **静默不融合**（CPU 自测首跑 `fused=0`）。
  正确做法是由 harness 在这一层标注；`assert_strategy_used()` 是这件事的报警器。

**1. CPU 自测全绿（43/43）** — `specdec/engine/attn_phase_selftest_cpu.py`
（掩码等价逐元素 8/8、分流裁决 8/8、真实调用序列 10/10、kp=1/2/3/4 逐阶段确定性 8/8、
SDPA kwargs 5/5、防静默失效+短块+默认行为 4/4；日志 `output/g5/attn/attn_phase_selftest_cpu_20260926T174715Z.log`）。
`fused_by_phase={'prefill':4}`（target+draft 各 2 层）；`s=1` 末块回退且有原因；默认 `eager` 时融合函数**从未被调用**（`fused=0 eager=24`）。
附注：tiny 随机模型对这次**两轮全接受**（`mismatch_rounds=0`）⇒ 真实序列里 `repair` 未被触发，
该行是**条件断言**；`repair` 的确定性覆盖由 3b 直接调 `_step` 提供。

**2. 算子级显存（预登记判据通过）** — `g5_attn_cudnn_validate.py`，`ACV_STAGE=ops`，
结果 `output/g5/attn/attn_cudnn_validate_20260926T175207Z.json`：

| 行（s,T） | target eager→fused 瞬时 | draft eager→fused |
| --- | --- | --- |
| **`(2048, 8192)`（8K 分块第一块）** | **1984.0 → 54.0 MiB（36.7×）** | **2284.0 → 51.5 MiB（44.3×）** |
| `(2048, 2048)` | 496.0 → 18.0 MiB | 571.0 → 16.4 MiB |
| `(512, 8192)` | 532.0 → 13.5 MiB | 592.0 → 12.9 MiB |
| `(1, 8192)` | **按策略走 eager**（cuDNN 不支持 s=1） | 同 |

路径判据 12/12 符合预登记期望；profiler 抓到 kernel `aten::_scaled_dot_product_cudnn_attention`
⇒ **确实走了 cuDNN 融合算子，不是 math 回退**。

**3. 算子级数值：预登记判据（`abs≤1e-2` 且 `rel≤1e-2`）12 行里 8 行 FAIL，不放宽。**
但 fp32 诊断（**非判据**）显示：10 个融合行 `|eager−fused|` 为 **1.09–3.27 个 bf16 ULP**，
且 **10/10 行融合侧比 eager 侧更接近 fp32**。两个判据在**不同输出尺度上朝相反方向**失效
（大 T 失 `rel`、小 T 失 `abs`）⇒ 一条"绝对 1e-2 且相对 1e-2"的判据在 bf16 注意力上
不可能同时满足所有尺度。

**4. 模型级 A/B：10/10 行 FAIL（短上下文 128/512/1025，`max_new=16`，`k=4`，2 reps）**
结果 `output/g5/attn/attn_cudnn_validate_20260926T175638Z.json`。要点：
- 自对照 **eager vs eager 在 5/5 组里 `abs=0.0`（逐位相同）** ⇒ 引擎确定性，**差异全部来自融合路径**；
- logits 差 `abs` 3.6–17.3、`rel` 5%–60%，argmax 一致率 0.9297–1.0000（翻转 5–26 个位置）；
- **`ctx=512 / base_bf16` 生成序列真的变了**：首个分歧在 **index 3**（eager `1602` vs 融合 `883`）；
- **阶段分流在模型级精确成立**：`fused=52/52`（28+24 层 × 1 块）；`ctx=1025` 的 `[1024,1]` 两块
  ⇒ `fused=52/52` 且短块回退 `52/52`；`eager_reason_kinds={'short_block_seq_len_lt_min':52,
  'phase_not_prefill':776}` ⇒ 短块回退与非 prefill 的正常 eager **被干净分开**；`assert_strategy_used()` 全 `ok`；
- `rep=0` 与 `rep=1` 每个数字逐字节相同（0 方差）。
- 两模型驻留后 `peak_allocated=3906.6 MiB` / `peak_reserved=4134.0 MiB`（**短上下文，非 8K**）。

**5. 差异来源已定位（专项取证 `g5_attn_precision_forensics.py`，日志 `forensics_20260926T180128Z.log`）**

| 假说 | 结果 |
| --- | --- |
| **H1 eager 在 softmax 前把分数舍入成 bf16**（`model_self.py:357` 的 matmul 输入输出都是 bf16） | **成立**：eager 误差随分数幅度线性增长 1.34e-2 → 6.18e-1，fused 恒定 ~1e-2，且 `\|eager−fused\|/\|eager−fp32\|` = 1.16/0.95/1.01/1.01 |
| H2 融合侧忽略掩码 | 不成立（`\|fused−eager带掩码\|=3.1e-2` vs `\|fused−eager无掩码\|=1.24e+1`） |
| H3 GQA 广播错误 | 不成立（`enable_gqa` 与手工实体 `kv_repeat` 差值 **0.000e+00**） |
| H4 缩放被重复施加 | 不成立（最贴正确 scale 1/√d） |

⇒ **`|eager − fused| ≈ |eager − fp32|`：差异的全部来源是 eager 路径自己的 bf16 分数舍入。**
平坦注意力（`randn`、大 T）下误差互相抵消 ⇒ 只有 2–3 ULP；尖峰注意力（真实模型数据）下不抵消
⇒ 放大到 12–60%。**顺带暴露一个此前没人量过的事实：项目自建的 eager 注意力基线在 softmax 之前
把分数舍入到 bf16，在尖峰注意力下对阵 fp32 的相对误差可达 ~50%。这是既有基线的性质，
本次没有改动它（超出本任务范围，且基线冻结）。**

**6. 口径澄清（本条修正一个此前未写明的事实）**：冻结设计里的 `v_fp8` **只压 target 的 V cache**，
draft **必须**保持 `base_bf16` —— draft 的 `head_dim=64` 而 `GROUP_V=128`
（`model_self.py:70,467`）⇒ 在 draft 上设 `v_fp8` 会直接
`AssertionError: head_dim 64 % GROUP_V != 0`。全项目既有口径一致
（`g5_stability_probe.py:289` 显式写死 `draft_kv="base_bf16"`，其余工具用默认值）。

**7. 红线（不得由本条推出）**：**没有**任何端到端 8K 收益结论、**没有**任何速度结论
（算子级瞬时显存≠端到端；本阶段工具里的耗时字段含插桩，禁止进 speedup 表）。
本次环境有 **24 个其它 C+G 进程**，且 commit 余量在 16 GiB 安全线上下浮动（16.2→15.8→16.1 GiB）。
**不得**说"8K 已解决"、"正式方案已切换"、"SpecDec 已加速"。

**8. 但"可以替换了吗"的答案是「不」**：预登记判据 FAIL 且生成序列发生变化
⇒ **注意力后端不是行为中性的**，换后端 = 换模型行为；任何跨后端 A/B 都自带这个混淆项，必须显式声明。
阶段报告：`docs/cudnn_prefill_stage_report.md`。

### 4.23 真实模型张量上的逐项精度定位 + 独立质量检查（2026-09-26 第二轮）—— **提高 QK/score 精度使最大 logits 差下降约 92%（主导因素）；概率/AV 路径单独提高精度无明显效果；参考残差成因未测**

**方法（为什么可信）**：用 `forward_pre_hook` 抓每层真实的 `input_layernorm(x)/cos/sin/cache`，
在工具内**复算** q/k/v（**不改引擎**），再对**同一份** `q/k_all/v_all` 跑变体阶梯 ——
避免"比较已经沿不同路径传播的输入"。
**自检**：复算的 eager 必须与引擎真实输出**逐位相同** ⇒ 实测 28 层 × 2 用例 × 2 种 kv **全部 True**。
**回归核对**：给 `model_self.py` 加诊断开关前后各跑一次，`case_a/case_b` **28/28 层所有数值字段完全相同**
⇒ 默认 eager 路径逐位未变。**加载等价**：`lowmem_load` 逐张量 `torch.equal` vs HF state dict，
**338/338 一致**。**FP32 参考**：同一数学在 fp32 下重算，显式关闭 matmul TF32
（`matmul_allow_tf32=False`；`cudnn_allow_tf32=True`），**是受控参考、不是绝对真值**。

**1. 逐层（同一份真实张量，`|v − cudnn|`）**

| 用例 | 量 | eager | qk32(去R1) | prob32(去R2) | out32(去R2+R3) | fp32(地板) |
| --- | --- | --- | --- | --- | --- | --- |
| A 无历史 `s=T=256` | 中位 | 4.004e-02 | **3.125e-02** | 4.297e-02 | 3.970e-02 | 1.479e-02 |
| A | **最大** | **2.914** | **6.250e-02** | 2.914 | 2.917 | 4.507e-02 |
| B 有历史 `s=256,T=512` | 中位 | 4.883e-02 | **3.125e-02** | 4.492e-02 | 4.155e-02 | 1.561e-02 |
| B | **最大** | **3.023** | **6.250e-02** | 3.023 | 3.026 | 3.576e-02 |

逐项移除效应（中位）：A `R1=1.660e-02 / R2=0.000e+00 / R3=1.090e-03`；
B `R1=9.766e-03 / R2=0.000e+00 / R3=1.422e-03`。
可加性残差（中位）：A −6.464e-03、B −7.939e-03 ⇒ **三项不可加**（提高 QK 精度会改变 softmax 的输入）。
**关键结构**：`d(eager,cudnn)` 中位仅 4e-2 但最大 2.9，**28 层里只有 1 层**超中位 10 倍 ——
巨差点集中在**第 0 层**（两种用例都是）；第 0 层**提高 QK 精度**后**缩小 186.5×（case A）/ 193.5×（case B）**
到 1.562e-02。**残差与「FP32 参考 − cuDNN 残差」的关系必须分用例说**：
case A `qk32/残差 = 1.027`、case B `= 1.241`（**高于**残差）。
注意 `1.562e-02` **大于** `1.521e-02`（此前有一处把大小关系写反，已更正）
⇒ 可写「绝大部分由 QK/score 精度解释」，**不得**写「100% 收敛」或「降到残差以下」。

**2. 整模型精度阶梯（判决性，`ctx=512, chunk=256`，比较 logits）**

| 预设 | base_bf16 `\|Δ vs cudnn\|` | argmax 一致率 | v_fp8 | argmax 一致率 |
| --- | --- | --- | --- | --- |
| eager | **17.91** | 0.9336 | **17.34** | 0.9570 |
| **qk32（提高 QK 精度）** | **1.350** | **0.9961** | **1.434** | **0.9961** |
| prob32（提高概率精度） | 17.97 | 0.9453 | 17.38 | 0.9336 |
| fp32（QK+概率均高精度） | 1.709 | 0.9844 | 0.8125 | 0.9922 |

⇒ **报告口径**：**不写"解释比例 96%–100%"**（最大误差差值是距离、不可加）；只写"最大 logits 差下降约 92%"。
比值 qk32/(FP32参考−cuDNN残差) = 1.02（base）/ 0.96（v_fp8）仅作**参考比值**且不稳定。
**概率/AV 路径单独提高精度**：17.91→17.97（base）、17.34→17.38（v_fp8）⇒ **未明显缩小该最大误差**
（**不写**"概率精度贡献为零/完全不改变输出"——测的是到 cuDNN 的**最大距离**变化，**中位为 0 ≠ 输出相同**）。
⚠️ **「降到参考残差以下」只对 base_bf16 成立**：`qk32/残差` = 0.790（base）/ **1.764（v_fp8，仍高于残差）**。
**【交互 / 不稳定，本轮新发现】**：在**已提高 QK 精度**的基础上**再提高概率精度**（`qk32 → fp32`），
两种 kv 方向**相反** —— base 1.350→1.709（**变差**）、v_fp8 1.434→0.8125（**变好**）
⇒ ① 概率/AV 路径精度**不是**可分离主效应（单独无效果、交互里会动残差）；② 残差（0.8~1.7）在两种 kv 下**排序不稳定**，
处在「融合路径 vs fp32 近似参考」的不一致噪声量级 ⇒ **不得**据此推断 cuDNN 内部实现。
**限制**：整模型层面**无法**单独移除 R3 —— 下游 `o_proj` 是 bf16，fp32 输出喂不进去
（实测 `RuntimeError: expected mat1 and mat2 to have the same dtype`）；R3 只有逐层测量（≈1e-3）。

**3. 轨迹分叉**：首个 attn 输出不同的层 = **0**（该层 `d_x_ln=0` ⇒ 输入逐位相同），
`d_attn_out` 从 1.203 累积到第 27 层最大 16.125，56/56 层有差异；
logits `abs` 17.91/17.34、`rel` **0.6255/0.6086**、argmax 一致率 0.9336/0.9570、末位 margin 5.125。
**相关性旁证（非判据）**：`Spearman(max|分数|, d(eager,cudnn))` 仅 **+0.083 ~ +0.240**
⇒ 幅度**足以触发**该效应，但**不能预测**各层偏差大小（还取决于分数分布形状）。

**4. 独立质量检查（诊断规模，`W=256, N=128, L=384`；**不是**正式质量结论）**
一致性断言全通过：`N` 唯一、10 个预测器被评分 token 哈希相同、logits 全有限。

| 预测器 | mean NLL | PPL |
| --- | --- | --- |
| `hf_eager`（`_attn_implementation` 已核对） | **2.284016** | 9.8160 |
| `hf_sdpa`（已核对） | 2.240676 | 9.3997 |
| `ours_eager` (chunk 384) | **2.284016** | 9.8160 |
| `ours_cudnn_prefill` (384) | 2.246421 | 9.4538 |
| `ours_eager`+`v_fp8` (384) | 2.276779 | 9.7452 |
| `ours_cudnn_prefill`+`v_fp8` (384) | 2.245992 | 9.4498 |

成对逐 token：`ours_eager − hf_eager` = **0.000000（128/128 逐点完全相同，sd=0）**；
`ours_eager − ours_cudnn_prefill` = mean **+0.0376** / median **+0.0002** / sd 0.279 / 68 差 60 好；
`hf_eager − hf_sdpa` = mean **+0.0433** / median +0.0007 / sd 0.290 / 72:56
⇒ **换后端的 NLL 影响与 HF 自身 eager↔sdpa 同量级、分布同形**；
`v_fp8` 影响 −0.0072（sd 0.060）⇒ 本窗口**未检出**质量损失（样本小）。

**5. 因果掩码 / 泄漏（真·同长度对照）**：位置 255 读 logits，`A` 长 256 / `C` 长 384 原文 /
`B` 长 384 但**未来 token 换随机** ⇒ **`|B−C| = 0.000e+00（0.00 ULP）`＝逐位零泄漏**；
`|A−C| = 0.4375（4.98 ULP）` 只是长度导致的归约差；三者 argmax 相同。
（此前用"两种长度的 logits 差"当因果证据是**弱证据**，已在本轮升级为同长度对照。）

**6. 三分类结论**：**已证实** —— 提高 QK/score 精度使最大 logits 差下降约 92%（base 13.27×、V-only 12.10×）⇒ 主导因素、概率/AV 路径单独提高精度无明显效果、R3 很小但整模型不可测、
分叉自第 0 层、零泄漏、自建 eager 与 HF eager 逐 token NLL 相同、加载权重逐位一致、默认路径未变、
短块与非 prefill 的 eager 在模型级精确可数（`fused=28`/`56`）。
**仍是假说** —— 地板（fp32 vs cuDNN 1.5e-2~1.7）成因未测（**不得**写"cuDNN 全程 fp32"）；
cuDNN 是否也舍入概率（证据方向矛盾，未决）；R1 效应与分数幅度的定量关系（秩相关弱）；
v_fp8 略降 NLL（方向一致但样本仅 128 token）。
**未测试** —— 8K 端到端、完整稳定性矩阵、任何性能收益结论。

**7. 五类分开陈述**：算子误差（已量化）／生成差异（已确认存在）／质量损失（**未检出**，诊断规模）／
内存收益（算子级成立、端到端未测）／性能收益（**未测**）。

阶段报告：`docs/cudnn_prefill_stage_report.md` §9。原始结果：`output/g5/attn/rtf_*`、`nll_quality_*`。

**8. 主张 ↔ 原始结果审计（2026-09-26 补）**：新增
`specdec/engine/g5_attn_claims_audit.py` —— 把报告 §5/§6/§9 的每条可量化主张写成断言、直接读原始 JSON 复核，
**不产生新数据、不重新解释判据**（`python -m specdec.engine.g5_attn_claims_audit`，退出码 0 = 全一致）。
**当前：50/50 一致。** 它的价值在于**抓到了报告自身的两处过度概括**（而非数据错）：
①"去掉 R1 后降到地板以下"**只对 base_bf16 成立**，v_fp8 为地板的 **1.76 倍**
（`qk32/floor` = 0.790 / 1.764；逐层 case A 1.027、case B 1.241）⇒ 已按用例改写；
②并因此暴露出**一个此前未报告的交互效应**：在已去掉 R1 的基础上再去掉 R2，
两种 kv 方向**相反**（base 1.350→1.709 变差、v_fp8 1.434→0.8125 变好）
⇒ R2 非可分离主效应，且残差排序不稳定，属"融合路径 vs fp32 近似参考"的不一致噪声量级。
审计断言已相应改为"**相对 eager 缩小 ≥10×**"（报告真正主张的量），
"是否降到地板"降级为**记录的比值**而非通过条件。**审计不得被用来自动放宽任何科学判据。**

**8K smoke（条件性）状态**：工具 `g5_8k_smoke_cudnn.py` 已建并在 **ctx=512** 排练跑通
（双模型驻留 `pt_peak_reserved=4134.0 MiB`、`cuda_free=2947.0 MiB`、`smi_used=4859.0 MiB`；
`fused={'prefill':52}`、短块回退 0、权重校验 338+290 全过）。排练提前抓到一个真 bug
（`set_kv_format` 收引擎名而非实验名）。**8K 本体未运行**（该决定由任务方保留）
⇒ "8K 能否跑通 / 端到端峰值"仍属**未测试**。

**9. 新验收标准提议（2026-09-26，**未生效**）**：新增
`docs/proposal_new_acceptance_criteria_2026-09-26.md` —— 按任务要求"新验收标准若确有必要，
先提出理由并等待确认"。内容：① 说明旧判据为何失败（**两个界在不同输出尺度上朝相反方向失效**；
差异主导来源是 **eager 基线自身**的 bf16 分数舍入 ⇒ 旧判据实际在问"两条精度不同的实现能否逐位一致"）；
② 三个候选方案：**A 把判据改成"不劣 + 行为改变已声明"（推荐）**、B 先修基线为 fp32 分数再要求一致
（改动冻结基线，历史数据断层）、C 不设判据只做双报告（与现有 RQ 设计冲突）；
③ 与 D1 的关系与各自代价；④ **限制条款：本分支不承诺与 eager 逐位等价**。
**确认之前不改任何旧判据、不覆盖任何旧结果、旧 FAIL 保持 FAIL。**
同文件附录还**预登记**了 8K smoke 的成功定义 / 判定线（`peak_reserved ≤ 8188 MiB` 记为内存达标）/
解释纪律（首次跑通只代表"能跑"；计时禁入 speedup 表；`allocated` ≠ 全部物理占用；
`reserved` 超显存不单独作为分页证据；失败不重试放大）。

### 4.24 过度归因更正 + 受控 8K smoke（2026-09-27）—— **表述更正 8 条；smoke 因默认提交护栏未放行而未运行**

**纪律**：本轮**只更正表述**。所有原始 JSON / 日志**保持原样**，旧判据的 FAIL 记录**保持 FAIL**，
**不重写任何历史实验**。更正清单见 `docs/cudnn_prefill_stage_report.md` §9.12（8 条），摘要：

1. **"R1 解释掉 96%–100% 的可解释差异" → 撤回**。改为
   **"提高 QK/score 计算精度使最大 logits 差下降约 92%"**（base 17.91→1.35、V-only 17.34→1.434）
   ⇒ 支持其为主导因素。理由：**最大误差差值是"距离"，不能当作可加归因比例**；
   此前给出的 `1.02 / 0.96` 只是"距离相对 (eager − 参考残差)"的比值，且在两种 kv 下**不稳定**。
2. **"该干预隔离了一次 cast" → 撤回**。`DIAG_QK_FP32` 的实现是
   `matmul(q.float(), k.float()) / sqrt(d)` ⇒ 它**同时**改变 **QK 矩阵乘法精度**与
   **`1/√d` 缩放的运算精度**（默认在 bf16 分数上做除法）。**不得**写成"仅隔离了一次 cast"。
3. **"R2 贡献为零 / 完全不改变输出" → 撤回**。改为
   **"单独提高概率/AV 路径精度未明显缩小该最大误差"**。
   理由：测的是**到 cuDNN 的最大距离**的变化，**中位数为 0 ≠ 输出相同**。
4. **数值方向写反**：`1.562e-02` **大于**（不是小于）`1.521e-02` ⇒ 改为比值表述（case A 1.027、case B 1.241）。
5. **"不可约地板" → 降级为"FP32 参考与 cuDNN 之间的残差，成因尚未完全解释"**。
   它不是真值下界、也未被证明不可约。（**代码字段名 `irreducible_floor_fp32` 未改**，仅在报告措辞中降级。）
6. **128-token 质量结论 → 限定为"该（单）窗口未见平均 NLL 恶化"**；不称普遍无损、不称质量提升。
7. **`hf_sdpa` → 只表示"已配置为 sdpa"**：**未**核实 HF 内部实际选中的 kernel
   （flash/mem-efficient/cudnn/math）⇒ **不得**当作"已确认的 cuDNN 对照"，
   只作"另一条后端路径 / 后端差异量级参照"。NLL 工具的每条 HF 结果新增
   `attn_kernel_verified=False` + 说明字段。
8. **"R1/R2/R3 = 去掉某个舍入点"的因果口吻 → 改为精度干预口吻**
   （提高 QK/score 精度、提高概率精度、提高 AV 输出精度）。

**2. 受控 8K smoke：工具已按新的记录要求加固，但本轮**未运行**

工具 `g5_8k_smoke_cudnn.py` 本轮加固（**逐阶段落盘 + 新的记录项**）：
① 关闭所有诊断精度开关（`set_diag_preset("eager")`）并记录 `profiler_used=False`、`default_strategy_at_start`；
② 阶段状态：`after_target_load` / `after_draft_load`（= `loaded`）/ `target_prefill_done` /
   `draft_prefill_done` / `decode_done`，**每阶段都落盘**；
③ 计时边界 `torch.cuda.synchronize()`，prefill 计时在显式阶段标注内、逐模型分别记；
④ 记录 `kv_format`、dtype、输入 hash、`rounds` 逐轮明细与**冻结的逐轮不变量**
   （`tgt_len == drf_len == pre + (m+1 if m<Kp else Kp)`，逐轮 `all_ok`）；
⑤ 阶段分流确认：`fused_by_phase` 只含 `prefill`、非 prefill 的 fused 计数恒 0、
   且 `verify` 的 eager 计数 > 0（⇒ **没有错误扩展到验证阶段**）；
⑥ 内存时间线把 **PyTorch 记账与设备侧 `nvidia-smi` 占用分开**记录。

**为什么没跑**：**默认提交护栏未放行**（本轮**没有**下调阈值，符合任务要求）。
实测证据（`output/g5/attn/smoke_guard_poll_*.log`）：连续探测 **15 次**（间隔 40 s），
`commit 可用` 一直在 **11.57–11.89 GiB**，**始终 < 默认安全线 16 GiB**；
同期其它 GPU 消费者 24 个、P-state 合格。因此
**连 `ctx=512` 的排练也一起被拒绝**（同一护栏），只有不加载模型的路径能跑。
⇒ **8K 双模型 smoke 的"能否跑通 / 端到端峰值"仍是未测试**，本轮**不写任何结论**。

**3. 若要推进，需要显式决定（二选一，不由我自动选）**
- **(a) 释放 commit**：关闭占用较大的应用后重试；护栏不动。
- **(b) 明确授权文档化覆盖**：`P5V_MIN_COMMIT_GIB=12`。
  依据：护栏原文标注为"**安全护栏，不是科学门槛**"，其 16 GiB =
  实测加载需求（9.1–9.6，对应 `load_ours` 路径）+ 约 6.5 GiB 余量；
  而 smoke **双模型均走 `lowmem_load`**（先在 CPU 侧 `.to(bf16)` 再 `load_state_dict`，
  权重逐位校验），实际需求约 **5.7 GiB**（target）与 **1.4 GiB**（draft，串行）
  ⇒ 按"需求 + 6.5 余量"折算的等效线约 **12.2 GiB**，11.6 GiB 略低于该等效线但远高于实测崩溃点（9.5 GiB）。
  **该覆盖会写进结果 JSON（`commit_guard.line_gib` / `overridden`），不会被误读为正常护栏下的运行。**

**4. 本轮未做的其它事**：不启动正式性能矩阵、不做重复稳定性、不做参数搜索、不改 chunk/k/dtype/量化/cache/logits。

### 4.26 受控 8K 双模型 smoke：**两档均单次完成**（2026-09-27，分页文件激活后，**默认 16 GiB 护栏、无覆盖**）

**环境前提（本机自查，不沿用任何转述数字）**：`C:\pagefile.sys` 8192 MiB（在用 78 MiB）、
`D:\pagefile.sys` 16384 MiB ⇒ **commit 上限 39.6 → 55.64 GiB**，可用 **37.0 GiB**；
默认护栏 `P5V_MIN_COMMIT_GIB=16` 直接通过（结果 JSON 中 `overridden=False`）。
GPU：8188 MiB 总量 / 空闲 7513 MiB / 其它 C+G 进程 20 个；P-state 合格（sm 2385–2467 MHz）。
git `a7c8677` 干净。**未下调护栏、未杀进程、未改系统配置、未同时跑其它 GPU 实验。**
⚠️ pagefile 激活状态**不是**持久保证：任务方明确指出"启动时未自动激活的原因尚不明确，
重启后需重新确认，不能假设永久修复"。

**执行**：三次**独立进程、串行、各一次**（`scripts\run_g5_attn_forensics.cmd smoke`，**不传第 2 个参数**）

| # | 条件 | ctx | chunk/k/max_new | target KV | 结果 |
| --- | --- | --- | --- | --- | --- |
| 1 | 短排练 | 512 | 2048/4/4 | base_bf16 | 完成（exit 0） |
| 2 | 8K smoke A | 8192 | 2048/4/16 | base_bf16 | **完成（exit 0）** |
| 3 | 8K smoke B | 8192 | 2048/4/16 | **v_fp8**（draft 固定 base） | **完成（exit 0）** |

**结果（全部字段来自落盘 JSON；三轮均 `default_strategy_at_start=eager`、
`diag_precision={'qk_fp32': False, 'prob_fp32': False}`、`profiler_used=False`）**

| 量 | 排练 ctx=512 | 8K base_bf16 | 8K v_fp8 |
| --- | --- | --- | --- |
| 权重逐位校验 | 338+290 ✓ | 338+290 ✓ | 338+290 ✓ |
| 输入 hash | `49ea7dbe1395d898` | `bf6a34a82018fd39` | 同左（同一输入） |
| 生成数 / 在词表内 / logits 有限 | 4 / ✓ / ✓ | **16 / ✓ / ✓** | **16 / ✓ / ✓** |
| `fused_by_phase` | `{'prefill': 52}` | **`{'prefill': 208}`** | **`{'prefill': 208}`** |
| `eager_by_phase` | `{propose:96, verify:28, repair:52}` | `{propose:432, verify:140, repair:52}` | 同左 |
| `s_max_by_phase` | `{prefill:512, propose:1, verify:4, repair:1}` | `{prefill:2048, propose:1, **verify:4**, repair:1}` | 同左 |
| `dispatch_only_prefill` | True | **True** | **True** |
| 逐轮不变量 | all_ok（1 轮） | **all_ok（5 轮）** | **all_ok（5 轮）** |
| 最终 cache 长度 tgt/drf | 516 / 516 | **8208 / 8208** | **8208 / 8208** |
| peak allocated / reserved | 4594.3 / 4874.0 MiB | **6902.4 / 7296.0 MiB** | **6847.1 / 7178.0 MiB** |
| 物理线 8188 MiB | ✓ | **✓ 达标** | **✓ 达标** |

- 计数自洽：`fused=208=(28+24)×4 块`；`propose=432=18 次 draft 步×24 层`；`verify=140=5 轮×28 层`；
  `repair=52=28+24`（**仅 r1 一轮 `m<Kp`**）。
- **关键安全性质**：`verify` 的 `s_max=4`（多 token 候选验证）而 **fused 计数为 0**
  ⇒ **融合没有进入验证阶段**。
- 逐轮（两档相同）：`pre 8192→8196→8198→8202→8206`，`(Kp,m)` = `(4,4)(4,1)(4,4)(4,4)(2,2)`，
  `tgt_len==drf_len==pre+(m+1 if m<Kp else Kp)` 全 ✓。
- **两档生成的 16 个 token id 完全相同** —— 这是**事实观察**（单 prompt、16 token），
  **不**构成"v_fp8 与 base 等价"的结论。
- v_fp8 峰值 reserved 比 base **低 118 MiB**（7178 vs 7296）；两档 `after_decode` 时
  `cuda_free=0.0 MiB`、设备侧占用 7845/7727 MiB（**设备已满，但没有 OOM**）。

**⚠️ 口径纪律（必须随数字引用）**
1. **"smoke 通过"仅表示：该配置单次跑完、结构检查与有限性检查通过。**
   **不**代表 AR/SpecDec 严格等价、**不**代表质量无损、**不**代表重复稳定、**不**代表性能加速。
2. **阶段时间仅作诊断**（含插桩）：target/draft prefill 3.22/2.34 s（base）、3.09/2.34 s（v_fp8），
   总 prefill 5.55/5.42 s，wall 6.58/6.51 s。**禁止**据此与历史 8K 数字（§4.19 的 28–36 s）说成
   "加速 x 倍"——工具、插桩、并发占用、decode 长度均不同。
3. `allocated` ≠ 整卡物理占用；`reserved` 超显存**不单独**证明分页。
4. 本轮**不是**新质量验收、**不改变**旧 FAIL（旧 eager 对齐门仍 FAIL）、**未采用**任何未经批准的新判据。
5. `lowmem_load` 是相对共享 `load_ours` 的**受控替代**（只改"何时转 bf16"，终点权重与 HF state dict
   **逐位相同** 338+290），把加载需求从 ~9.1–9.6 GiB 降到 ~5.7 GiB。**未改** chunk/k/dtype/量化/cache/logits/allocator。

**过程记录（如实）**：首次 512 排练（`smoke512_20260927T122151Z.log`）暴露**我自己工具的一处缺口**——
smoke 只标注了 `_prefill`，使 `propose/verify/repair` 落进 `'other'` 相位
（`仅prefill融合=False`、`eager_by_phase={'other':176}`）。**引擎无问题**
（`fused_by_phase={'prefill':52}` 仍正确），是标注不全导致分相位计数无法举证。
已修（四个入口全部标注，与其它工具一致）并**重跑排练**确认（`smoke512_20260927T122317Z.log`，
`仅prefill融合=True`）后才推进 8K。`output/g5/attn/` 下三次运行的 JSON 与日志全部保留
（含此前被护栏拒绝与失败的历史记录，未删除）。

### 4.27 smoke 工具检查缺口修复 + 8K 重复可运行性（3×base + 3×V-only，2026-09-27）

**1. 修复的检查缺口（工具层，未改生产算法）**

| # | 原问题 | 修复 | 原理/作用 |
| --- | --- | --- | --- |
| 1 | `logits_all_finite` **只**检查两模型 prefill 返回的 logits，名称夸大覆盖范围 | 新运行的字段改为显式命名 `prefill_logits_finite`，并在 `verdict` 里写明旧字段口径（**旧 JSON 不改写**） | 口径与名称一致，避免"看起来全查了" |
| 2 | 有限性检查不在决策点 | 新增 `smoke_checks.CheckRegistry.check_finite()`，在 **propose 入口（紧接 argmax）、propose 返回、verify 返回的候选行（紧接 argmax）、repair 返回、prefill 返回** 五处检查；**非有限立即抛 `FinitenessError` 并停止** | 任务要求"检查必须发生在 argmax/接受判断之前"；且覆盖"实际用于决策的 logits" |
| 3 | "未执行的分支"可能被空过 | `evaluate()` 区分 `required/executed/ok`，并输出 `uncovered`（该跑没跑的**可选**分支）与 `not_applicable`（**结构上不存在**的组合，显式列出而不是默认为已覆盖） | "没有执行的分支标为未覆盖，不能空过" |
| 4 | 执行结束即 `stage=complete`、exit 0 | 拆成 `stage=run_ended` 与 `checks.passed`；**只有 passed=True 才 exit 0**，否则 **exit 5**；`attention_assert` 失败/必需检查缺失/异常/生成长度异常一律判失败 | 区分"执行结束"与"验证通过" |
| 5 | 少生成可能被当成完成 | 记录 `stop_reason ∈ {max_new_reached, eos, other_short}`，并校验 `gen_len==max_new`（或 EOS 且末位为 eos）；`other_short` **判失败** | "不能少生成却当固定长度完成" |
| 6 | 可能由 `reserved ≤ 物理容量` 推断整卡安全 | 字段改名 `pt_reserved_within_physical_line`，并加 `not_a_whole_card_safety_verdict=True`；同时记录设备侧 `smi_mem_used` 与各阶段 `cuda_free` | 预登记口径：`allocated ≠ 整卡占用`，不能据此推断整卡安全 |
| 7 | runner 只认 0/2/3 | 新增 `exit 5 → CHECKS FAILED` 分支，且**上层编排遇非零即停** | "上层 runner 必须传播失败退出码，并停止后续测试" |

**2. 轻量回归（不加载大模型）**
- 纯逻辑回归 `smoke_checks_selftest_cpu.py`：**27 PASS / 0 FAIL**，覆盖
  ① 正常通过 ② 非有限值（NaN/Inf 在决策前被拦、记录含 NaN 计数与相位）③ 分流失败
  ④ cache 不变量失败 ⑤ **缺失检查**（含"空过"语义：未执行的可选分支不算失败但必须列入 `uncovered`；
  mismatch>0 时 repair 转为必需并判失败）；另测生成长度/停止原因与其余必需项。
- **注入故障回归**（`SMOKE_SELFTEST=<scenario>`，**不加载模型**，走的是与真实运行同一套判定+落盘+退出码）：

| scenario | runner 退出码 | tool 退出码 | 结果 |
| --- | --- | --- | --- |
| `normal` | **0** | 0 | passed=True |
| `nonfinite` | **1** | **5** | 失败项含 `run_no_exception`；`first_bad.role=rows` |
| `dispatchfail` | **1** | **5** | 失败项 `dispatch_only_prefill` + `attention_assert` |
| `invariantfail` | **1** | **5** | 失败项 `cache_invariants` |
| `missing` | **1** | **5** | 失败项 `finiteness:propose/draft` + `finiteness:verify/target` |

⇒ 五种情形的 JSON **全部落盘**、错误**全部被拦住**、runner **不再判成功**（符合"错误会被拦住、结果能落盘、
runner 不会继续推进"）。审计工具已把该回归纳入（**54/54**）。

**3. 重复可运行性：6 次独立进程、串行、预固定交错顺序、默认护栏**

编排 `scripts/run_g5_8k_repeat.ps1`（每步一个全新 `cmd`/python 进程；**每次重查环境**；
**不给 runner 传第 2 个参数**⇒ 默认 16 GiB 护栏）。顺序：`base1 → vfps1 → base2 → vfps2 → base3 → vfps3`。
配置：target 1.5B + draft 0.5B，BF16，`ctx=8192 / chunk=2048 / k=4 / max_new=16`，
target KV = base_bf16 或 v_fp8，draft 恒 base_bf16，策略 `cudnn_prefill`（仅 prefill 融合）。

| step | KV | stage | gen/stop | passed | 不变量 | 有限性检查 | peak allocated | peak reserved | cuda_free | 退出码 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base1 | base_bf16 | run_ended | 16 / max_new_reached | True | ✓ 5 轮 | 19 次 | 6902.4 | **7296.0** | 0.0 | 0 |
| vfps1 | v_fp8 | run_ended | 16 / max_new_reached | True | ✓ 5 轮 | 19 次 | 6847.1 | **7178.0** | 0.0 | 0 |
| base2 | base_bf16 | run_ended | 16 / max_new_reached | True | ✓ 5 轮 | 19 次 | 6902.4 | **7296.0** | 0.0 | 0 |
| vfps2 | v_fp8 | run_ended | 16 / max_new_reached | True | ✓ 5 轮 | 19 次 | 6847.1 | **7178.0** | 0.0 | 0 |
| base3 | base_bf16 | run_ended | 16 / max_new_reached | True | ✓ 5 轮 | 19 次 | 6902.4 | **7296.0** | 0.0 | 0 |
| vfps3 | v_fp8 | run_ended | 16 / max_new_reached | True | ✓ 5 轮 | 19 次 | 6847.1 | **7178.0** | 0.0 | 0 |

- **6/6 完成且 `passed=True`**；`required=13`、`failures=[]`、`uncovered=[]`、
  `not_applicable=['finiteness:propose/target','finiteness:verify/draft']`（结构上不存在）。
- **零方差项**：`peak_allocated`（base 6902.3798828125 ×3、v_fp8 6847.1298828125 ×3）、
  `peak_reserved`（7296.0 ×3 / 7178.0 ×3）、`fused={'prefill':208}`、
  `eager={'propose':432,'verify':140,'repair':52}`、`s_max_by_phase`、最终 cache 长度 8208、
  输入 hash `bf6a34a82018fd39`、**生成的 16 个 token id 六次完全相同**。
- 设备侧占用（`smi_mem_used`）有 5–6 MiB 抖动：base 7845/7845/7840、v_fp8 7729/7723/7724 ⇒
  **PyTorch 记账稳定，但设备侧受其它 20 个消费者影响**；6 次 `cuda_free` 均为 **0.0 MiB**。
- 环境（每次重查）：C/D 两个分页文件均活跃（8192/16384 MiB），
  commit 上限 **55.64 GiB**、可用 **35.69–36.71 GiB**、`P5V_MIN_COMMIT_GIB=16.0`、`overridden=False`（6/6）；
  GPU 空闲 7534–7538 MiB；P-state 全部合格（sm 2220–2325 MHz）。
- 阶段耗时**仅诊断**：prefill 5.64–6.36 s、wall 6.69–7.72 s（插桩含在内）。
  **不得**据此声称性能稳定性、**不得**进 speedup 表；**父子阶段耗时不可相加**（prefill 是 wall 的子集）；
  **分页文件扩容不等于 GPU 显存增加**（显存压力未消失：`cuda_free=0`）。

**4. 本轮的证与未证**
- **已证**：该配置在**默认护栏**下**重复可运行**（6/6）、结构检查与内存记账**零方差**、
  有限性检查**覆盖全部 6 个决策点组合**且未发现非有限值、融合**只**在 prefill（`verify` 的 `s_max=4` 而 fused=0）、
  逐轮 cache 不变量全通过、检查逻辑的失败路径在注入故障下**确实被拦住并落盘**。
- **未证**：AR/SpecDec 严格等价（**两档生成 id 相同不等于 SpecDec 与 AR 等价**）、质量无损、
  性能稳定性/加速、更长生成长度（64 token）与更大质量窗口的行为。
  旧 eager 对齐门**仍然 FAIL**，历史记录未改，未采用任何待批准的新验收判据。

**5. 原始结果**：`output/g5/attn/smoke8k_cudnn_prefill_20260927T1246{12,10,54,36,16,56}Z.json`
（依次 base1/vfps1/base2/vfps2/base3/vfps3）与各同名 `.log`；
汇总 `repeat6_summary.json`；编排主表 `run_g5_8k_repeat_*.master.tsv`；
控制台 `run_g5_8k_repeat_console.log`；注入故障回归的 5 个 JSON 亦全部保留（未删除失败样本）。

### 4.28 draft 中间步有限性检查 + 8K/64-token 同后端 AR/SpecDec 对照（2026-09-27）

**1. 补齐 `_propose` 中间步检查（新模块 `smoke_hooks.py`，不改生产语义）**

`spec_dec._propose` 的循环是 `t=argmax(lg)` → `lg,cache=draft._run(...)` → 下一轮再用 `lg`，
因此循环内每次 `draft._run` 的返回值**会在下一次 argmax 处被使用**；只在入口/出口检查覆盖不到它。
做法：把"阶段标注 + 决策点检查"抽成 `DecisionHooks`，并在 `QwenLM._run`（或测试用 stub 的 `_run`）上
装一个**普通函数** hook，在 `phase=='propose'` 时对**每次 draft 前向的返回**做检查
（role=`mid_lg`，记录 `round` 与**候选步编号**）。**不复制投机算法、不插入任何计算**。
另加**覆盖核对** `verify_decision_coverage()`：`mid_lg 次数 == 解码器报告的 draft 前向次数`
且 `input_lg 次数 == 轮次数`，不符即判"未完整覆盖"（"阶段出现过一次检查" ≠ 完整覆盖）。

**2. 轻量集成测试（不加载大模型）** `smoke_midstep_selftest_cpu.py`：**28 PASS / 0 FAIL**
覆盖：正常多步、**中间步 NaN**、Inf、入口非有限、**k_eff 缩短**、**EOS 提前结束**、步号/轮次记账、
以及"**入口/出口检查不能替代内部检查**"的对照（不装 hook 时同一脚本 4 次前向全跑、NaN 静默进入 argmax；
装了 hook 则第 2 次前向返回即停、只发生 2 次前向）。hook 卸载按**对象身份**核对，且测试期间不碰生产类。
其余回归：`smoke_checks_selftest_cpu` 27/27、`attn_phase_selftest_cpu` 43/43、注入故障 5 情形（normal→0，四类→exit 5 / runner 1）。

**3. 两个**我自己**的 harness 缺陷（如实记录，均已修，失败样本保留）**

| # | 现象 | 根因 | 修法 |
| --- | --- | --- | --- |
| a | 8K AR 首跑**中途 OOM**：`pt_reserved` 5722 → 22924 MiB（`smoke8k_cudnn_prefill_20260927T134045Z.json` 保留） | 我的 AR 解码循环**漏了 `torch.inference_mode()`**，而引擎自己的 AR 单步 `spec_dec._step`（`spec_dec.py:45-51`）与分块 prefill 都在 inference_mode 内 ⇒ 同一前向在模式外执行，显存行为完全不同 | 补齐 inference_mode（**恢复**生产语义，不是放宽条件）；并新增 OOM/crash 的 `traceback` 记录与**逐步内存采样**（首跑缺这两项，无法定位） |
| b | AR 与 SpecDec "**在 index 0 就分歧**" | 我的 AR 循环**先 `_run` 再记录**，于是 **prefill 决策出的第一个 token 从未被记录**，整条序列错位一位（引擎 `generate` 与 8K 探针的 AR 都是**先记录再喂**） | 改为先记录再喂；修正后 ctx=512 的 AR 序列开头与 SpecDec 一致（`[438,330,264,883]`）。**这曾表现为"分歧"，实为工具记账缺陷，不是引擎/cuDNN 分歧** |

**4. 四个单次条件（独立进程、串行、各一次、默认 16 GiB 护栏、每次重查环境）**

| 条件 | mode | target KV | 完成 | gen | stop | peak alloc/reserved | cuda_free | 覆盖核对 | 退出码 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_ar | AR | base_bf16 | ✓ | 64 | max_new_reached | 5261.7 / **5722.0** | 1343.0 | `decode_lg 63 vs 63` ✓ | 0 |
| base_spec | SpecDec | base_bf16 | ✓ | 64 | max_new_reached | 6906.4 / **7332.0** | 0.0 | `mid_lg 78 vs 78; input_lg 20 vs 20` ✓ | 0 |
| vfps_ar | AR | v_fp8 | ✓ | 64 | max_new_reached | 5203.4 / **5588.0** | 1477.0 | `decode_lg 63 vs 63` ✓ | 0 |
| vfps_spec | SpecDec | v_fp8 | ✓ | 64 | max_new_reached | 6847.6 / **7178.0** | 0.0 | `mid_lg 75 vs 75; input_lg 19 vs 19` ✓ | 0 |

- **AR 是 target-only**（`resident_models=['target']`，权重校验 338），**不是** dual-resident 冒充；
  SpecDec 为 `['target','draft']`（338+290）。
- 阶段分流：AR `fused={'prefill':112}`（28×4 块）+ `eager={'decode':1764=63×28}`，
  `s_max={'prefill':2048,'decode':1}`；SpecDec `fused={'prefill':208}`，
  `eager={'propose':1872=78×24, 'verify':560=20×28, 'repair':468=9×52}`（v_fp8 为 1800/532/312）
  ⇒ **只有 prefill 融合，`verify` 的 `s_max=4` 而 fused=0**。
- SpecDec 统计自洽：base `rounds=20, proposed=78, accepted=55, mismatch=9` ⇒ 55+9=64 ✓；
  v_fp8 `rounds=19, proposed=75, accepted=58, mismatch=6` ⇒ 58+6=64 ✓。
- 逐轮 cache 不变量 `all_ok`（AR：`cache_len==prompt+decode_runs`；SpecDec：`tgt_len==drf_len==pre+(m+1 if m<Kp else Kp)`）。
- 耗时**仅诊断**：AR prefill 3.26/3.79 s、wall 5.64/6.70 s；SpecDec prefill 5.51/6.40 s、wall 9.17/10.14 s。
  **不算 speedup、父子阶段不可相加**。

**5. 对照结论：共同前缀 31/64，随后在同一个决策点分歧（记录，不立刻归因）**

| KV | 完整序列相同 | 共同前缀 | 首个分歧（index 31） |
| --- | --- | --- | --- |
| base_bf16 | 否 | **31 / 64** | AR=`1083` vs SpecDec=`5880` |
| v_fp8 | 否 | **31 / 64** | AR=`5880` vs SpecDec=`1083` |

- 前 31 个 token 在**两条解码路径 × 两种 KV** 下**完全一致** ⇒ 两条路径在同一起点、同一上下文下高度一致；
- 分歧发生在**同一个 index（31）**，且两种 KV **恰好交换**了这两个候选（1083 ↔ 5880）；
- **本轮到此为止**（任务要求：仅 token 分歧时保存配对结果并停止自动推进）。
  **不**判定为逻辑错误，**也****不**豁免为"BF16 漂移"——两者都需要证据。
- 下一步（**尚未执行**）：在**完全相同的共同前缀**（prompt + 前 31 个 token）下，
  比较该决策点的 **target 决策行**：top1/top2 与 margin；**不**比较已分叉的后续上下文来归因，
  **不**自动放宽 exact-id 门。

**6. 边界**：旧 eager 对齐 FAIL 保持原样；未采用任何待批准验收判据；
本轮**未**做重复长跑、参数搜索或正式矩阵；耗时字段不构成性能稳定性或加速结论。

**7. 原始结果**：`output/g5/attn/smoke8k_cudnn_prefill_2026{0927T135218Z,0927T135254Z,0927T135336Z,0927T135413Z}.json`
（base_ar/base_spec/vfps_ar/vfps_spec）与同名 `.log`；主表 `run_g5_8k_repeat_20260927T135208Z.master.tsv`；
汇总 `arspec4_summary_v2.json`；**失败的 AR 首跑** `...T134045Z.json`、
**错位样本**（首版 AR 四条件）`...T13{4512,4547,4630,4706}Z.json` 与 `arspec4_summary.json` **全部保留**。

### 4.29 8K 同后端 AR/SpecDec 第 32 个 token 分歧：**定点定位完成**（2026-09-27，base_bf16）

**1. 先钉死位置/索引语义（CPU 哨兵，不加载大模型）**：新增
`spec_verify_index_selftest_cpu.py`，**16 PASS / 0 FAIL**，用 tiny 真引擎 + 真 `spec_dec._verify` 证明：
- `saved_next` = 位置 **P−1** 的 logits ⇒ 回答 **cands[0]**；
- `rows[j]` = 位置 **P+j** 的 logits ⇒ 回答 **cands[j+1]**（即"`rows[j-1]` 对应第 j 个候选"），
  且它与"cache 长度 P+j 上单步输入 cands[j] 的 logits"是**同一位置**（fp32 CPU 下 max|Δ|=7.6e-6，仅形状不同）；
- block 内因果性：**同长度**替换**未来**候选时 `rows[0..j-1]` **逐位不变**（不是用不同长度差值判断）；
- `_verify` **会就地改写传入的 cache 字典**（长度 10→14）⇒ 任何"两条分支共用快照"的设计**必须** `clone_cache()`；
  且逐层 dict 浅拷贝**足够**（引擎只重绑键、不就地改写张量）——原 cache 未被污染。

**2. 定点复现（单进程、只加载 target、默认 16 GiB 护栏、`overridden=False`）**
工具 `g5_8k_divergence_diag.py`，`run_id=20260927T233506Z`，结果
`output/g5/attn/divdiag_base_bf16_20260927T233506Z.json`（日志 `divdiag_base_20260927T233504Z.log`）。
- 输入 = **原始 wikitext 前缀（8192 token，hash `bf6a34a82018fd39`）+ 已确认一致的前 30 个输出 token**，
  **不重新选 prompt**；四份来源 JSON 的前 30 token 一致 = **True**；
- 复现检查：prefill 位置 8191 的决策 = **15**（应 15）；推进到 `cache_len=8222` 后，
  位置 **8221** 的决策 = **572** ⇒ **与四份实测的 g30 完全一致（reproduced=True）**。

**3. 【A. 同一 cache、不同计算形状】结果（决策位置 = 8222，1-based 第 32 个 token 的决策点）**

| 路径 | 预测位置 | cache 来源 / 长度 | top1 | top2 | **margin** | 1083 分数 | 5880 分数 | 5230 分数 | 与另一行 max\|Δ\| |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ① 单步 `cands[0]`（**s=1**，AR 形状） | **8222** | 同一快照，len=8222 | 5880 | 1083 | **0.0000（恰好并列）** | **20.625** | **20.625** | 20.375 | **0.15625 = 2.50 bf16 ULP** |
| ② block `[572,5230,304,279]` → `rows[0]`（**s=4**，SpecDec 形状） | **8222** | 同一快照，len=8222（block 后 8226） | 5880 | 1083 | **0.1250** | 20.5 | 20.625 | 20.25 | 同上 |

- 两行**有限**；`argmax` **不同**（`argmax_same=False`）：① 中 1083 与 5880 **恰好并列**于 20.625，
  `argmax` 取**较小下标** ⇒ **1083**（正是 AR 实测输出）；② 中 1083 低 **1 个 ULP**（20.5）⇒ **5880**
  （正是 SpecDec 实测输出）。最大差 0.15625 = **2.50 bf16 ULP**（|x|≈20 时 1 ULP = 0.0625）。
- **快照隔离核对**：两条分支各自 clone 后，快照指纹（逐层长度 + 末 8 位 k/v 和）**前后一致**
  （`snapshot_unchanged=True`）⇒ **没有互相污染**。

**4. 精度敏感性（**只**提高 attention 的 QK/概率精度，**不是**整个模型 FP32 真值）**

| 变体 | ①(s=1) top1 | ②(s=4) top1 | ① 的 1083 / 5880 |
| --- | --- | --- | --- |
| `qk32`（QK 用 fp32） | 5880 | 5880 | 20.375 / 20.5 |
| `fp32_attn`（QK+概率 fp32） | 5880 | 5880 | 20.5 / 20.625 |

⇒ 提高 attention 精度后**并列消失**，两条形状**都**给 5880。即：**分歧不是由 attention 精度不足造成，
而是由两种解码形状在同一位置上约 1 ULP 的舍入差异 + 「恰好并列时 `argmax` 取较小下标」共同决定。**

**5. 【B. 相同形状、不同 cache 历史】本轮**未做**（按任务分阶段：先完成 base 的定点诊断）。
限制与策略已写明：`_verify` 会就地改写 cache ⇒ 需 `clone_cache()`；单个 8K cache 副本约 236 MiB，
两份副本 + 目标模型（~3154 MiB reserved）在 8 GiB 内可行；若同时要 draft 驻留（SpecDec 历史），
按任务建议**分进程重放或转存 CPU**，不同时在 GPU 上加载多个模型。

**6. 分类（**不强迫归入某一类**）**
- **实现错误已证实？否。** 位置/索引/mask/cache 长度语义由 CPU 哨兵 16/16 钉死；复现与实测一致；
  快照无污染；两种形状的**最大值都是 5880**（只是 `argmax` 在并列时取了下标较小者）。
- **数值路径差异有证据？是。** 同位置、同前缀、同一快照下两种形状差 **2.5 ULP**，且决策恰好在该差异上翻转；
  提高 attention 精度后并列消失、两侧一致。
- **"BF16 漂移已被唯一证明"？未主张。** 缺少：(i) fp64 真值以判定并列两侧谁更接近真值；
  (ii) 非注意力部分（MLP/RMSNorm/lm_head）在两形状下是否逐位一致；(iii) 并列在真实数据上的发生频率。
- **未复现？** base_bf16 **已复现**；v_fp8 **未运行**（分阶段等待决定）。
- **仍未解释？** 为何 s=1 形状恰好落在**精确并列**（值恰为 20.625）；以及"精确并列时取较小下标"
  是否是项目想要的语义（并列本身没有唯一正确答案）。

**7. 工具修复（本轮）**
- `scripts/run_g5_8k_repeat.ps1`：① 产物**按明确 `RUN_ID` 关联**（从步日志解析 `RUN_ID=`，据此构造
  文件名并校验存在），**不再猜"目录里最新的 JSON"**；② **每个同 KV 的 AR/Spec 配对结束立即比较生成序列**，
  记录 `pair_correctness` 与共同前缀/首个分歧；**分歧时记 CHECK 并停止后续条件**；
  ③ smoke 工具新增打印 `RUN_ID=<id>`。
  （在新片段上对既有 4 份 JSON 做过离线核对：base 对 `DIFF prefix=31 first_div=1083|5880`、
  v_fp8 对 `DIFF prefix=31 first_div=5880|1083`，与手算一致。）
- `g5_8k_divergence_diag.py`：修正位置标签（实测行位于 **8222**，非 8221；由 `cache_len_before=8222`
  与 `block_cache_len_after=8226` 可推得）。

**8. 边界**：未做性能重复、未启动正式矩阵、未改 dtype/k/chunk/量化/后端/判据；旧 eager 对齐 FAIL 保持原样；
未采用任何待批准验收标准；耗时未进 speedup 表。

### 4.30 8K 首个分歧的证据链补齐：字段/ULP 更正 + **真实两条路径的真实 cache 历史** 2×2（2026-09-27/28）

**1. 字段与 ULP 更正（新模块 `diag_fields.py`，纯函数；`diag_fields_selftest_cpu.py` 35 PASS / 0 FAIL）**
- **分开记录**：`argmax_id`（`torch.argmax` 真正返回的下标 = 解码实际选择）、`max_score`（数值本身）、
  `n_max_tied`、`tied_max_tokens`（**完整**并列集合）、`topk`（仅参考列表，**不得**称其顺序为解码 top1）。
  自测已覆盖：并列时 `argmax_id`（较小下标）可与 `topk[0].token` 不同。
- **ULP 更正**：bf16 有 7 位尾数 ⇒ [16,32) 区间相邻间距 = 2^(4−7) = **0.125**（**不是 0.0625**）；
  自测断言 `bf16_ulp(20.625)=0.125`、`bf16_ulp(1.0)=2^-8`、`(20.625−20.5)/ulp = 1.0`。
- **主口径改为原始绝对误差**；ULP **逐元素**计算，且**仅在 |x|≥1 上统计**：首跑用"整行单一幅度 ULP"
  归一化得到 `ulp_err_max=1.35e6`（被**近零元素**毒化），已改为原始 `abs_err_max` 为主 +
  受限 ULP 统计 + 明确 `normalization_caveat`。**⇒ 上一轮报告中"0.15625 = 2.50 ULP"的说法作废**：
  正确写法是**原始绝对误差 0.15625**，在 [16,32) 区间相当于 1.25 个相邻间距，且**不同幅度元素的间距不同**。
- 自测还覆盖**位置编号**三套口径：输入位置 8222 / 所预测 token 位置 8223 / 生成序列 index **31**
  （`PREFIX_TOKENS=30` **不是**分歧下标，一致性检查对此会报 False）。
- **`divergence_index` 由 30 更正为 31**；`reproduced` 更名为 **`prefix_checkpoint_match`**
  （它只核对"产出 572 的那个决策"，**不是**分歧点复现）。

**2. 重新构建 2×2（`g5_8k_divergence_2x2.py`）：用**真实**两条路径的**真实** cache 历史**
- `DIAG_MODE=ar`：真实 AR 路径（prefill + 单步推进到 cache_len=8222）→ 捕获该处真实 cache；
- `DIAG_MODE=spec`：跑**真实** `spec_decode_greedy`，用 `DecisionHooks.on_verify_pre` 在
  `_verify` **就地修改之前**快照 `pre==8222` 的 target cache，并记录**实测** `rows[0]` 与本轮记录；
  拿到即中止（不做多余的 31..63），随后 `del draft` 再跑反事实，避免顶满显存；
- `DIAG_MODE=compare`（纯 CPU）：读两份 `.pt` 转存做**逐张量 `torch.equal` + 最大绝对差**。
- 生产精度：`set_diag_preset("eager")` 且记录；逐决策有限性检查保留；每次重查环境；
  **默认 16 GiB 护栏、`overridden=False`**；分进程重放（不在 GPU 上同时驻留两套双模型副本）。

**2×2 结果（原始分数；位置均为输入 8222 / 预测 token 8223）**

| KV | cache 历史 | 计算形状 | argmax_id | max_score | n_max_tied | margin | 1083 | 5880 | 5230 | 是否实测 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_bf16 | **AR** | single(s=1) | **1083** | 20.625 | **2（并列）** | 0.0000 | **20.625** | **20.625** | 20.375 | ✅ AR 实测 g31 |
| base_bf16 | AR | block(s=4) | 5880 | 20.625 | 1 | 0.1250 | 20.5 | 20.625 | 20.25 | 反事实 |
| base_bf16 | **Spec** | single(s=1) | 5880 | 20.625 | 1 | 0.1250 | 20.5 | 20.625 | 20.25 | 反事实 |
| base_bf16 | **Spec** | block(s=4) | **5880** | 20.625 | 1 | 0.1250 | 20.5 | 20.625 | 20.375 | ✅ Spec 实测 rows[0] |
| v_fp8 | **AR** | single(s=1) | **5880** | 20.625 | **2（并列）** | 0.0000 | 20.5 | **20.625** | 20.375 | ✅ AR 实测 g31 |
| v_fp8 | AR | block(s=4) | 5880 | 20.625 | 1 | 0.1250 | 20.5 | 20.625 | 20.25 | 反事实 |
| v_fp8 | **Spec** | single(s=1) | 5880 | 20.625 | 1 | 0.1250 | 20.375 | 20.625 | 20.25 | 反事实 |
| v_fp8 | **Spec** | block(s=4) | **1083** | 20.5 | **3（三并列）** | 0.0000 | **20.5** | **20.5** | 20.25 | ✅ Spec 实测 rows[0] |

- **完整并列集合（`tied_max_tokens`，必须逐格核对，不能只看计数）**：并列只出现在**两个**格子，
  且**并列的双方并不相同**：
  - base_bf16 **AR×single**：`max_score=20.625`，**`tied_max_tokens=[1083, 5880]`** ⇒ `argmax` 取较小下标
    **1083**；即该格的并列**正是** 1083/5880 之争。
  - v_fp8 **Spec×block**：`max_score=20.5`，**`tied_max_tokens=[1083, 5880, 22909]`（三并列）**
    ⇒ `argmax` 取较小下标 **1083**。
  - ⚠️ **更正一处此前的表述**：v_fp8 **AR×single** 也有并列，但其并列是 **`[5880, 22909]`（均 20.625）**，
    而 **1083 在该格是 20.5（低 0.125）** ⇒ 该格的 `argmax=5880` **与 1083/5880 之争无关**。
    因此**不能**把 v_fp8 的 AR×single 说成 base 的"镜像"：两档的并列**不是**同一对候选。
- **两档的实测输出都被 2×2 里的对应格子精确复现**（base：AR×single=1083 / Spec×block=5880；
  v_fp8：AR×single=5880 / Spec×block=1083）。
- **形状效应（同 cache、不同形状）**：base 的 **AR 历史**上成立（single 并列 → 1083；block 分离 → 5880，
  `abs_err_max=0.15625`）；v_fp8 的 **Spec 历史**上成立（single → 5880；block 三并列 → 1083，
  `abs_err_max=0.1875`）。
- **cache 历史效应（同形状、不同 cache）**：**存在**。两档的两份真实 cache 在**同一 token 前缀**下
  **都不是逐位相同**：base 比较 56 个张量 → `all_bitwise_equal=False`，**max|Δ| = 0.6259765625**；
  v_fp8 → `max|Δ| = **1.0625**`（**fp8 载荷先反量化再比**；直接比 payload 得到的 32.0 无物理含义，已作废）。
  同形状下两历史给出的行也不同（例：base 的 block 行里 5230 = 20.25 vs 20.375）。
- **交互**：并列**只出现在特定格子**（base: AR×single；v_fp8: Spec×block）⇒ **不能**把两效应
  拆成可加百分比。
- **未解释**：为何恰好在那些格子上落成**精确并列**；以及 `argmax` 并列时取较小下标的语义是否为项目所需。

**3. 快照与因果性核对**
- 快照隔离改为**逐张量 `torch.equal`**（参照副本 vs 跑完两种形状后的快照，56 个张量全部相等 ⇒
  `snapshot_unchanged_all_layers=True`）；**不再**用"末 8 位置求和"这类近似指纹宣称逐位不变。
- 浅 dict 拷贝的**前提**已由 CPU 哨兵确认（引擎只重绑键、不就地改写张量），并在结果里写明该限制；
  `_verify` 本身**会**就地改写传入的 cache（哨兵 ⑤ 实测长度 10→14）。
- 同长度替换未来候选时早期行**逐位不变**（CPU 哨兵 ④ 实测 0.0；同长度，非不同长度差值）。
- 归因只用**同一前缀**的两份 cache；**未**使用已分叉的后续上下文。

**4. 分类（不强迫归入某一类）**
- **实现错误已证实？否。** 位置/索引/mask/cache 长度语义由哨兵钉死；两档实测输出都能被 2×2 精确复现；
  快照未被污染。
- **数值路径差异有证据？是，且是两类**：① 计算形状（s=1 vs s=4）在同 cache 上产生可复现的分数差
  （base 0.15625、v_fp8 0.1875）；② 同一 token 前缀下**两条真实路径的 cache 本身不逐位相同**
  （base max|Δ|=0.626、v_fp8 反量化后 1.0625）。
- **"BF16 漂移唯一归因"/"仅并列规则问题"？均不主张。** 缺少 fp64 真值判定并列两侧谁更接近真值、
  缺少非注意力部分（MLP/RMSNorm/lm_head）在两历史下的逐位一致性、缺少并列与缓存差的**因果分离实验**
  （例如把 AR 的 cache 喂给 Spec 形状的路径、反之）。
- **未复现？无**：两档的**实测输出**都在 2×2 中复现（但复现的是"该位置为何这样选"，不是"差异已消除"）。
- **报告更正**：撤回"提高精度后一致 ⇒ 不是 attention 精度不足所致"，改为
  **"该位置对 attention 精度敏感；本次干预消除了决策差异，泛化未验证"**；
  ULP 口径与 `divergence_index` 已按上文更正；**旧 JSON 不覆盖**，更正写入报告与本节。

**5. 原始结果**：`output/g5/attn/div2x2_20260927T235747Z_base_bf16_{ar,spec}_*.json` 与
同名 `.cache.pt` / `.rows.pt`、`..._compare_*.json`；v_fp8 对应 `div2x2v_20260927T235925Z_*`；
日志同目录 `div2x2_*.log`。旧一轮定点诊断 `divdiag_base_bf16_20260927T233506Z.json`
**保留不动**（其字段口径已被本轮更正）。

### 4.31 修正 2×2 诊断的实验配置与快照校验后重跑（base_bf16 + v_fp8，2026-09-28）

**本节更正 §4.30 的两处证据缺陷（§4.30 原文保留、不覆盖）**

1. **生成预算改变了被诊断的计算形状**：§4.30 的 `run_spec` 用 `max_new=32`（原实验 **64**）。
   目标轮前已生成 30 token ⇒ `k_eff = min(4, 32−30) = **2**`，而对照重放用**硬编码的 4 候选**
   ⇒ **实际执行的是 Kp=2 的块**。证据就在原 JSON 里（未改动）：
   v_fp8 `spec_actual_rows0.argmax_id=**5880**`、`cells.block.argmax_id=1083`、
   `spec_actual_vs_replayed_block.argmax_same=**false**`。
   ⇒ **§4.30 把"Kp=4 重放行"标成"实测 `rows[0]`"是错的**；「两档实测输出都被精确复现」的说法**作废**。
   （数值 1083 本身没错，它对应的是**原 64-token 实验**真正执行的 Kp=4。）
   另：§4.30 称"两档的并列**不是**同一对候选"仍然成立（base AR×single 并列 `[1083,5880]`；
   v_fp8 AR×single 并列 `[5880,22909]`；v_fp8 Spec×block 三并列 `[1083,5880,22909]`）。
2. **快照校验是空证据**：§4.30 的参照是**浅 dict 拷贝**（张量共享）⇒
   `torch.equal(snap_ref[i]['k'], snap[i]['k'])` 比的是**同一对象**、**恒为真**，
   且只覆盖 `k`/`v`、漏掉 `v_scale`/`fmt`/层数/长度。
   ⇒ `snapshot_unchanged_all_layers=True` **不构成证据**（这属于**证据不足**，
   **不代表**已经发现缓存污染）。
3. 另更正一处口径：§4.30 的 v_fp8 历史差 `1.0625` 用的是 **bf16** 反量化；
   改用 **float32** 后为 **1.0089292526245117**（float32 下 `q*scale` 精确可比，是更可靠的口径）。

**新增工具与轻量测试（全部 exit 0）**
- `diag_cache_cmp.py`：`deep_clone`（逐张量独立副本）、`shallow_clone`（保留旧行为供对照）、
  `compare_caches`（全层全键 + `fmt` + 层数 + 长度；`dequant_dtype` 默认 float32）、
  `integrity_check`、`metadata`（**全层**）。CPU 自测 **28 PASS / 0 FAIL**，其中专门证明：
  ★ 浅拷贝参照**检不出**原地修改（all_equal 恒 True）；deep_clone 参照**能**检出原地修改（max\|Δ\|=7.0）、
  键重绑、**只改 `v_scale`**、`fmt`/层数/长度/dtype/shape 变化；
  ★ e4m3 在 ~288 处相邻间距为 32 ⇒ "载荷 +1"是**无效突变**（会被舍入回原值，第一版自测就踩了这个坑）。
- `g5_8k_divergence_2x2.py` **v2**：`max_new=64`/`k=4` + 捕获 hook 停止（`stop_policy` 记录）；
  `capture_verdict` 逐字段断言；重放用**捕获到的真实候选**；`real` / `single` / `block` 与
  更短候选的 **`is_real_executed_row=False`** 额外对照；完整 logits 落盘；
  AR 侧改**自由生成**并逐步交叉核对。CPU 自测 **22 PASS / 0 FAIL**（含"Kp=2 ⇒ all_match=False"、
  "`argmax_same=True` 但 `elementwise_equal=False`"、"并列 `margin_top1_top2=0` 而
  `gap_to_next_distinct` 不为 0"）。
- `diag_fields.compare_rows` 新增 `elementwise_equal`（**不得**以 argmax 相同代替逐元素相同）。

**本次定点诊断结果（tag `div2x2v2b_20260928T004000Z`）**

| KV | max_new/k | 捕获 round/pre/Kp | 候选 | `g` | all_match | run_id |
| --- | --- | --- | --- | --- | --- | --- |
| base_bf16 | **64/4** | 8 / 8222 / **4** | [572,5230,304,279] | [572,**5880**,304,279] | **True** | spec 20260928T003353Z |
| v_fp8 | **64/4** | 8 / 8222 / **4** | [572,5230,304,279] | [572,**1083**,304,279] | **True** | spec 20260928T003459Z |

- **实测行 vs Kp=4 重放行**：两档都 `elementwise_equal=True`、`max|Δ|=0.0`、`argmax_same=True`
  ⇒ 重放正确，且 `on_verify_pre` 的快照**未被 `_verify` 的就地改写污染**。
- **2×2（输入位置 8222 / 预测 token 位置 8223 / 生成 index 31）**
  - base：AR×single = **1083**（`tied=2 [1083,5880]`，`margin_top1_top2=0`）= 实测 AR g31 ✓；
    Spec×block = **5880** = 实测 Spec g31 ✓；AR×block = 5880；Spec×single = 5880。
  - v_fp8：AR×single = **5880**（`tied=2 [5880,22909]`，margin 0）= 实测 AR g31 ✓；
    Spec×block = **1083**（**三并列** `[1083,5880,22909]`，margin 0）= 实测 Spec g31 ✓；
    AR×block = 5880；Spec×single = 5880。
- **形状效应（逐元素）**：4 种组合 `elementwise_equal` **全为 False**（max|Δ| 0.15625 / 0.21875 /
  0.2109375 / 0.1875），但只有 base×AR 与 v_fp8×Spec **翻转 argmax**；
  另两格 argmax 相同而行不同 ⇒「argmax 相同」不能当作"形状无影响"。
- **cache 历史效应**：base 56 个张量**无一逐位相同**（max|Δ|=**0.6259765625**；k 0.626 / v 0.281）；
  v_fp8 84 个里**仅 1 个**逐位相同（max|Δ|=**1.0089292526245117**；k 0.5 / v 1.009 / v_scale 4.19e-4）。
  层数/长度/dtype/shape 全一致（`structural=[]`）⇒ 差异是数值层面的，不是结构错配。
- **快照完整性（有效口径）**：四个 run 全部 `all_equal=True`、`max|Δ|=0.0`、
  `shapes_are_independent_copies=True`、`master_vs_workcopy_shared_storage=False`
  ⇒ single/block **互不污染**得到可失败的检验。
- **AR 轨迹**：自由生成（`teacher_forced=False`），30 步与来源轨迹**逐步一致**、`n_steps_disagreeing=0`。
- **额外对照（更短候选）**：base×AR 的 Kp=2/3 与 Kp=4 的 `rows[0]` **逐元素相同**（max|Δ|=0），
  而 base×Spec、v_fp8×AR、v_fp8×Spec 则不同（0.0625–0.25）。
  ⇒ **"形状效应"不是简单的 s=1 vs s=4**；机制未定位，列为**未解释项**。这些格 `is_real_executed_row=False`。

**三档分类（严格）**
- **已确认**：捕获条件与原实验逐字段一致且实测 Kp=4；实测行与重放逐元素相等；形状效应存在；
  同前缀两条路径的 cache 不逐位相同；AR 是真实轨迹；快照完整性在有效口径下通过。
- **支持但未证实**：1083 恰好出现在"有顶部并列"的两格且 `argmax` 取较小下标（缺 fp64 真值、缺并列频率普查）；
  "两条路径 cache 差来自各自历史"（缺**交叉喂 cache 的因果分离实验**，故形状/历史不能拆成可加贡献）。
- **尚未排除**：为何某些格恰好精确并列；为何块长度敏感性在 base×AR 下为 0；非注意力部分
  （MLP/RMSNorm/lm_head）在两种形状/两段历史下是否逐位一致（**未验证**）；
  是否所有后端/路径都不做就地张量改写（本轮只核对该诊断路径）。

**运行纪律**：每步独立进程、串行、**无重试循环**；`overridden=False`（可用 ≈33.5–33.7 GiB ≥ 16 GiB）；
P-state `qualified=True`；生产精度（`qk_fp32=False, prob_fp32=False`）；**不做性能测量**。
**保留的失败记录（不覆盖）**：tag `div2x2v2_20260928T001500Z` 的
`base_bf16_spec_20260928T003222Z.json`（`stage="crashed"`，
`TypeError: dict() got multiple values for keyword argument 'note'`，工具自身缺陷；
该 run 的 `capture_check.all_match=True`，即捕获成功但**未生成成功结论**）；
同 tag 另两个产物一并作废，避免混用代码状态。

### 4.32 QK FP32 最小精度干预测试：控制组 vs 干预组（8 次完整 64-token 生成，2026-09-28）

**目的**：仅把 **target** 的 QK 打分改 fp32（`DIAG_QK_FP32`）能否改善 8K 下 AR 与 SpecDec 的输出不一致。

**先检查（CPU，未上 GPU）：开关实际改变什么**
- **影响计算的读取点只有 1 处**，在 `QwenLM.forward` 的 **eager 分支**（`model_self.py:411`）；
  `diag_precision()` 里那处只是状态读取器。AST 扫描 + 自测断言。
- 改变：`matmul(q,kᵀ)` 与 `1/sqrt(head_dim)` 缩放改 fp32 ⇒ softmax 的**输入**不再是 bf16 舍入后的 scores。
- **不变**：softmax 本身（本来就 float32）、`attn→bf16` 回转、AV matmul、`o_proj`/MLP/RMSNorm/lm_head/
  权重/缓存 dtype、并列规则、验收阈值。**输出 dtype 不变（仍 bf16）**。
- **融合 prefill 绕过该开关**：`_fused_prefill_attention` 源码里没有该开关名；`decide_attention_path`
  仅当 `phase=="prefill"` 且 `s>=2` 才返回 `fused`。⇒ 生效阶段 = target 的 **decode/verify/repair**。
  **不是全链路升精度**；未为启用开关而换 backend（仍 `cudnn_prefill`）。
- 开关原是**模块级全局**、无 per-model 作用域。新增 `diag_qk_scope.QKScopeHook`（包 `QwenLM._run`，
  按模型实例逐次开关、异常也复位），**未改 `model_self.py`**；`qk_scope_selftest_cpu.py` **36 PASS / 0 FAIL**。

**运行**：ctx=8192、chunk=2048、**max_new=64**、**k=4**、greedy；输入 hash `bf6a34a82018fd39`；
target KV ∈ {base_bf16, v_fp8}、draft KV=`base_bf16`；tag `qk32c_20260928T025500Z`；
每步独立进程、串行、无重试循环；`overridden=False`；**无性能测量**。

**控制组原样复现**：4 条序列与原 64-token 失败实验的 4 份原始 run **逐 token 完全相同**，
峰值显存也相同 ⇒ 复现无问题，可归因于干预。

**结果表（控制组 vs 干预组）**

| KV | 组 | AR↔Spec 共同前缀 | 首个分歧 | AR token | Spec token | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| base_bf16 | control | **31** | 31 | **1083** | **5880** | 复现已知分歧 |
| base_bf16 | qk32 | **52** | 52 | **279** | **64871** | **分歧推迟（+21），部分改善≠PASS** |
| v_fp8 | control | **31** | 31 | **5880** | **1083** | 复现已知分歧 |
| v_fp8 | qk32 | **58** | 58 | **16** | **15** | **分歧推迟（+27），部分改善≠PASS** |

**作用域落地（任务硬要求）**：干预组 target 开关**全开**（`calls_target == flag_on_calls`：
67/67、33/33、67/67、31/31），draft **全关**（`calls_other == flag_off_calls`：0/93、0/86）；
控制组全关。峰值 alloc/reserved 与历史逐位相同，未抬高 reserved。

**决策行（同共同前缀、两侧 input_position 相同）**
- 先核对 `_verify` 位置语义 `argmax(rows[j-1]) == g[j]`：4 份 spec run 共 **229 项 / 0 不一致**。
- **必须区分"该位置存在一行"与"该行真的决定了输出"**：某轮被否决(`m<Kp`)后其后续行**被丢弃**；
  首版分析误用了这种行（把 base idx52 的 Spec 行报成 `verify_rows[14][2]` argmax=1084），
  已改为按 `rounds[].emitted` 定位槽位，槽位 0 的行由 `saved_next`（上一轮 `m==Kp ? rows[Kp-1] : repair 行`）**重建**。
  - base/control idx31：AR 单步@8222 argmax **1083** 并列`[1083,5880]` margin=0 ↔ Spec `rows[0]`@8222 argmax **5880** margin **0.125**
  - base/qk32 idx52：AR 单步@8243 argmax **279** 并列`[279,64871]` margin=0 ↔ Spec saved_next(repair)@8243 argmax **64871** margin **0.125**
  - v_fp8/control idx31：AR@8222 argmax **5880** 并列`[5880,22909]` margin=0 ↔ Spec `rows[0]`@8222 argmax **1083** 三并列 margin=0
  - v_fp8/qk32 idx58：AR@8249 argmax **16** margin **0.125** ↔ Spec saved_next@8249 argmax **15** 并列`[15,16]` margin=0
- 干预后 **index 31 不再分歧**（两档 AR=Spec=5880）；**残余分歧与 idx31 同形态**：
  一侧顶部**精确并列**、另一侧不并列。

**干预改变了参考生成（不可含糊）**：base AR 首个差异位置 **31**（1083→5880）、Spec **34**（23→16）；
v_fp8 AR **34**（23→16）、Spec **31**（1083→5880）。
⇒ **"新精度下 index 31 对齐"不得写成"原配置原样通过"**；QK FP32 产出的是**另一条**轨迹。
无 fp64 真值 ⇒ **不能**说新轨迹更正确，只能说两路径数值在 fp32 QK 下**更久落在同一侧**。

**结论**：两档均为 `DIVERGENCE_POSTPONED_PARTIAL_NOT_PASS` —— **部分改善，不是 PASS**。
按判读规则本轮**停止**：未扩展全模型 FP32、未换 backend、未堆叠其它精度开关、未改并列规则或验收协议。
**正确性阻塞未解除**：分歧仍在；参考生成被改变；融合 prefill 仍 bf16（链路仍混合精度）；
干预会改变生产输出。

**保留的失败记录（不覆盖）**：tag `qk32_20260928T010000Z`
`control_base_bf16_spec_20260928T023107Z.json`（`_verify_spy` 形参顺序错 ⇒ `TypeError`）；
tag `qk32b_20260928T024500Z` `control_base_bf16_ar_20260928T023204Z.json`
（签名自检误看已被包装的函数 ⇒ `AssertionError`）。二者均为**工具自身缺陷**，各自以新 tag 完整重跑。

### 4.33 (a) `prefill_logits_finite` AR 误报修正；(b) base_bf16 残余分歧的缓存历史单变量探针（2026-09-28）

**(a) 汇总缺陷：AR 恒报 False**
- 旧表达式 `all(reg.calls.get(("prefill", m), {}).get("ok", False) for m in ("target","draft"))`：
  **AR 不加载 draft** ⇒ 该检查不存在 ⇒ 取到 `False` ⇒ **AR 恒报 `prefill_logits_finite=False`**；
  而同一记录 `checks.passed=True`、`not_applicable` 已列出 `finiteness:prefill/draft` —— 自相矛盾。
  **未驻留 = not_applicable，不是失败。**
- 修法：`smoke_checks.prefill_finiteness` / `prefill_finiteness_live`，只对**实际执行过** prefill 的
  模型求合取；未驻留排除在外；**一个都没执行 ⇒ False**。两个工具均改用。
  回归自测 `prefill_finite_selftest_cpu.py` **15 PASS / 0 FAIL**。
- **修订分析**（原 JSON 不覆盖）：`g5_8k_prefill_finite_revise.py` →
  `qk32c_20260928T025500Z_prefill_finite_revision_20260928T151333Z.json`：
  8 份里 **4 份 AR 由 False→True**（executed=['target']、na=['draft']），
  4 份 spec 不变（executed=['draft','target']）；修正后 8/8 与 `checks.passed` 一致。

**(b) 探针：只改变缓存历史，是否足以复现不同 argmax？**
- **只**定位 base_bf16 干预组（tag `qk32c_20260928T025500Z`/`qk32`）的新分歧：
  生成 index **52**、输入位置 **8243**（AR=**279** / Spec=**64871**）。
  不泛化 2×2、不扩展 v_fp8、不改 backend、max_new 仍 64、不改算法或验收标准。
- **控制量全部相等**（`all_controls_equal=true`）：同一单 token 前向入口 `spec_dec._step`
  （其函数体 `model._run(model.embed_tokens([[token]]), cache, need_logits=True)`
  与 AR 解码单步**是同一表达式**）；当前 token 均 **323**；位置均 **8243**；
  送入前 cache 长度均 **8243**；预测位置 8244 / index 52；两侧均 target-only QK FP32。
  共同前缀由捕获断言核对（AR 自由轨迹前 52 token 一致；SpecDec 捕获轮 round/Kp/候选/g/emitted
  与参考全字段一致，本轮自身 emitted 前 52 个也一致）。
- **先证明重放匹配真实行**：AR 侧重放 vs 真实 decode 行 `elementwise_equal=True, max|Δ|=0.0`；
  SpecDec 侧重放 vs 真实 repair 行同样 `True / 0.0`；两侧快照完整性 `all_equal=True, max|Δ|=0`
  （56 张量，独立副本）。
- **结果**

| 量 | AR 缓存 | SpecDec 缓存 |
| --- | --- | --- |
| 重放行 argmax | **279** | **64871** |
| max_score | 20.75 | 20.875 |
| 顶部并列集合 | **`[279, 64871]`（2 并列）** | `[64871]` |
| margin_top1_top2 | **0.0** | **0.125** |

  两份缓存 `all_equal=False`、**max|Δ|=0.375**（bf16 原始载荷，56 张量）；
  两行 `elementwise_equal=False`、`argmax_same=False`、max|Δ|=0.25。
- **判定 `VERDICT = HISTORY_ALONE_SUFFICES`**：同位置/同 token/同精度/同入口下，
  **只换缓存历史就足以复现不同 argmax**。机制与 index 31 同类（一侧顶部**精确并列**、另一侧不并列；
  此处并列在 AR 侧：279 与 64871 精确相等 ⇒ 取较小下标 279）。**差别**在于本位置两侧形状**完全相同**
  （都是 s=1 单步）⇒ 残余分歧**不需要**用形状效应解释。
- **边界**：只回答 base_bf16 / index 52 / 这一组缓存；**不**外推到 v_fp8 index 58，
  **不**声称"缓存历史是唯一成因"，**不**给出并列发生频率；"修 tie-break 或提高缓存精度能否消除分歧"**未测**。

**本轮自身缺陷（保留记录，不覆盖）**：① `reshist_20260928T152000Z_..._ar_20260928T151621Z.json`
（`dict(**compare_rows(...), note=...)` 撞键 ⇒ `TypeError`）；②
`reshistb_20260928T153000Z_base_bf16_spec_*.json`（`_run_spec` 重复安装 `DecisionHooks`，
`_HOOK_REGISTRY` 以类为键被覆盖 ⇒ `_run_hook ↔ _wrapper` **无限递归** `RecursionError`）。
③ 误标修正：`repair_cache_len_before` 原先在 `_step` **之后**读取（`_step` 就地追加 token）⇒ 记成 8244，
已改为调用前读取并加 `repair_cache_len_after`；本轮结论依赖 `trimmed_to=8243` 与交叉核对，不依赖该字段。

**原始结果**：`output/g5/attn/reshistc_20260928T154500Z_base_bf16_{ar,spec,compare}_*.json` + `.cache.pt` / `.rows.pt`。

### 4.34 有限性汇总契约 v2 + 历史记录离线复核 + 生产正确性方案审查（2026-09-28）

**(1) 契约 v2：修掉两处汇总缺陷（CPU 先复现再修）**
- **B1 漏检**：Spec 下 `draft` **必需但未执行**时 `all_executed_ok` 仍为 **True**
  —— v1 只对"**执行过**"的模型求合取，必需没跑的模型被静默排除。
- **B2 伪装**：`prefill_finiteness_live` 把 registry 里**没有记录**的模型直接标成 `not_applicable`
  —— **从缺失推断 N/A**，把"漏执行"降级成"不适用"。
- **契约 v2**：必需集合按**模式显式**给出且由 `expected_map` 派生（与 `evaluate` 同一份判据，不建第二套）：
  **AR={target}，Spec={target,draft}**。必需模型必须**有记录 + executed=True + ok is True**；
  **只有显式声明**不适用的才可排除；必需集合为空 ⇒ 不通过；`all_executed_ok` 仅字面含义并**明文标注不得当判据**，
  放行一律读 **`all_required_ok`**（`gate_field`）；新增 `contract_note()` 写入结果 JSON。
- **CPU 回归 `prefill_finite_selftest_cpu.py`：37 PASS / 0 FAIL**，覆盖 9 类反例（见报告 §9.21.1）。

**(2) 已有记录离线复核（不重跑 GPU、原 JSON 不覆盖）**
- 工具 `g5_8k_prefill_finite_revise.py`（`REVISE_ALL=1`），产物
  `prefill_finite_revision_v2_20260929T041952Z.json`，每项带 `basis`。
- 候选 38 份 → **可判定 27**（ar 13 / spec 14）；**11 份因未标注 `mode` 跳过**（不做判定，避免假阳性）。
- **变化 18**；旧值违反"整体通过 ⇒ prefill 必须通过" **12**；修正后违反 **0**；
  检出**真实必需缺口 1**（`smoke8k_cudnn_prefill_20260927T131837Z`：spec、draft 未执行、该 run 本身 `passed=False`）；
  5 份更老 schema 根本没这个字段。迁移：`False→True` 13、`None→True` 4、`None→False` 1、`True→True` 9。
- ⚠️ **历史数据受影响**：不止上一轮 tag —— **原始 8K 源 run 也波及**：
  `smoke8k_cudnn_prefill_20260927T135218Z`（base **AR**）与 `…135336Z`（v-fp8 **AR**）的
  `run.prefill_logits_finite` 由 **False 更正为 True**；两个 spec run（`…135254Z`/`…135413Z`）本来就正确。
  **原始 JSON 未改动**，更正只在修订分析文件。
- **口径澄清**：不能用 `new == checks.passed` 判一致 —— `passed` 是**整体**门，`all_required_ok` 只管 prefill；
  "prefill 有限但别的检查失败"正常。唯一有意义的单向不变式：**整体通过 ⇒ prefill 必须通过**；
  按此口径 4 条原先被判"不一致"的记录全部得到解释，violation 归零。

**(3) 收窄后的研究结论**
> **在 base_bf16 / index 52 的受控单步重放中，真实缓存历史的数值差异足以改变该位置的 argmax。**
- 限定：局部归因，**不**外推 v_fp8/其它位置；**不**写"缓存被损坏"；**不**写"缓存历史是唯一原因"；
  **不**写"所有分歧机制相同"（index 52 与 31 只有**形态**一致）；
  **不**把"不需要当前这一步形状不同"读成"形状效应无关"（历史差异仍可能来自此前不同形状的写入）；
  **不**写"修好 tie-break"（`argmax` 规则未被证明错误，两条路径的**分数与并列集合本身就不同**，
  统一规则不保证恢复一致）；**不**把局部复现当完整 64-token PASS；
  **不**把提高缓存存储 dtype 等同于重新计算更准确的缓存。

**(4) 生产正确性决策表（只做方案审查，本轮不实现）** —— 详见报告 §9.21.4，摘要：
- 三类正确性：**算法结构**（FP32 B6 exact 20/20、216/348；位置语义 229/0；哨兵 16/16；重放 max|Δ|=0；快照完整）
  与 **FP32 实现**（FP32 下 SpecDec==AR 精确）证据较强，但**门未全部跑**（参照是 fp32 受控近似参考，非独立实现）；
  **生产 BF16 一致性门未通过**，且本轮证据**不足以**判 `BF16_NEAR_TIE`（缺 `prev/next_max_abs` 与
  **`fp32_margin`**）⇒ 按 `results_schema.classify_bf16_compare` 现行契约只能是 **`NEEDS_INSPECTION`**。
- 方案清单 S1（QK fp32 变默认）／S2（小 margin 单步重算）／S3（LM head 精度）／S4（缓存存储 dtype）／S5（改协议），
  每项给出"改变什么计算 / 参考输出与 cache 语义 / 性能显存 / 最小检验 / **停止条件**"。
  ⚠️ **S2 的充分性已被本轮证据否定**：index 52 的真实 AR 决策行本身就是单步，
  与 SpecDec 的单步行 argmax 不同（279 vs 64871）⇒ **缓存历史已有差异时，单步重算不保证恢复 AR**。
  ⚠️ **S4 存储 dtype ≠ 重新计算**：无法消除此前已在 bf16 下写入 cache 的舍入误差。
  ⚠️ **S5 属"需团队批准的协议变更"**，不自动生效、不抹去原门失败；
  `proposal_new_acceptance_criteria_2026-09-26.md` 仍**未生效**，且它针对 attention backend，**不**直接适用于 exact-id 门。
- **唯一推荐的下一步候选（不开启新 GPU 测试）**：补齐 `classify_bf16_compare` 缺失输入
  （FP32 下的 `fp32_margin` + 相邻位置 `prev/next_max_abs`），把该分歧按**现有契约**正式分类。
  理由：不新增精度开关、不改协议、不跑正式矩阵；它是 S1–S5 任何路线的共同前置；分类给出后即**收敛**，
  不再需要继续复现同一分歧。

**(5) 与并行复盘文档的关系**：`docs/retrospective_2026-09-29_correctness_cache_history.md`
（**并行分析 agent 撰写，记录时未纳入版本控制**）的数值与结论与本节**一致**
（20.750/20.750 vs 20.750/20.875、并列集合、margin 0 vs 0.125、cache max|Δ|=0.375、
行 max|Δ|=0.25、tag `reshistc_20260928T154500Z`、commit `ec42713`），
其 §5 的五个"直觉修法为何不成立"与本节 (4) 的 S1–S5 逐条对应，其 §6 记录的正是本轮修的两处缺陷。
两文无冲突；该文件**非本轮产物、未由本轮提交**。

**(6) 撤回本节 (4) 的推荐下一步**：`docs/production_correctness_decision_2026-09-29.md`
（**建议稿，待批准；未纳入版本控制**）指出 `results_schema.classify_bf16_compare` 对 `own_max_abs` 的
**文档定义是"同位置 block-vs-serial"**，而本轮是**不同缓存历史下的 single-vs-single** ⇒
把 0.25 填入 `own_max_abs` 属**误用字段语义**，且分类标签**不是**任何路线的普遍共同前置。
⇒ **不再推荐"补输入做分类"作为下一步**；本轮不为此启动新实验。该建议书另提出两条路线
（A 保持严格 exact-token 并审查共同数值路径；B 分层正确性与有限精度行为报告，其建议先评估），
**本轮未批准、未采纳、未实施**，也**未**修改协议/生产代码/历史数据。详见报告 §9.21.7。

**(7) 纪律**：本轮**未**修改模型算法、默认 backend/精度、验收门槛或内存护栏；**未**开启新 GPU 测试；
**未**宣布正式实验获准启动；旧数据一律保留未覆盖。

### 4.35 验证准备清单（阶段 1 之前，待批准；**未启动 GPU**，2026-09-29）

**产物**：`docs/verification_readiness_2026-09-29.md`（清单本体）+
`specdec/engine/verify_family_independence_cpu.py`（只读 CPU 自测）。基线 commit `5c2a1e4`。
**未**启动 GPU、**未**改工具/manifest/默认 backend/精度/验收门槛/内存护栏。

**(1) 固定候选配置（不改默认实现）**：`cudnn_prefill`（仅 prefill 融合，其余 eager）、bf16 计算、
`set_diag_preset("eager")`（**QK32 关闭**；现有记录键名为 `diag_precision_baseline`，
另配 `precision_scope_effective`/`qk_scope_report` 佐证未开干预）、
target KV ∈ {base_bf16(引擎名 fp16), v_fp8}、draft 恒 base_bf16(fp16)、
ctx 8192 / chunk 2048 / k 4 / max_new **64** / greedy、
EOS = `tokenizer.eos_token_id` = **151645**（命中即停并记实际长度；未命中且未达 64 ⇒ `other_short` ⇒ 失败）、
默认 16 GiB 护栏（不上调）、P-state 环境门。以上 ctx/chunk/k/max_new/eos 与 `qk32c` 记录逐项核对一致。

**(2) 输入：两个未参与此前调试的 family**（manifest `v1-draft`，取自 `wiki.train.raw`；
此前全部诊断都用 `wiki.test.raw` 前 8192 token，hash `bf6a34a82018fd39`）
- **F06**（`S3_prose`，block 0，win 5，`window_token_start=40960`，8192 前缀 hash `71e630c6cd702180`）
- **F13**（`S2_high_markup`，block 3，win 3，`window_token_start=24576`，8192 前缀 hash `9a278731d744a859`）
- ⚠️ **与建议书的偏差**：建议书举例"自然文本与代码"，但本 manifest **无代码档**（strata 仅 S1/S2/S3/S4）
  ⇒ 改取**该清单自身设计轴（预登记接受率 rank 2 vs 4）差异最大**且**跨 block** 的两档。
- **CPU 自测 `verify_family_independence_cpu.py`：PASS 12 / FAIL 0**，含
  **16 family × (2048/4096/8192 + text) = 64/64 hash 全部复现**、调试输入 hash **再次复现 `bf6a34a82018fd39`**、
  **16/16 family 与调试输入零 32-gram 交集**、候选彼此零交集且不同 block、
  **除 builder 外无 runner 读取过 manifest**（⇒ 使用历史为空）。产物 `output/g5/attn/family_independence_cpu_check.json`。
  注：扫描消费方**必须区分大小写** —— 不区分时 `F1[0-6]` 会命中 `bf16`，首版扫描因此误报一串消费方。

**(3) 16 次清单**：2 family × 2 KV × {AR, Spec} × 2 次独立进程重复。AR 为 target-only；
同 family/KV 的 AR 与 Spec **target 侧配置逐项相同**；重复必须独立进程、串行、从头 prefill（不共享 cache 历史）。

**(4) 脚本记录能力审计的缺项（本轮只列方案，不实施）**
- **G1**：工具**无法输入 manifest family** —— `P5V.load_ids()` 把 `test.raw` 前缀**写死**
  ⇒ **第 3 节清单当前不可执行**。最小修：加只读 family 输入解析（`SMOKE_FAMILY`/`SMOKE_MANIFEST`），
  **未指定时行为逐位不变**。
- **G2**：`run_g5_8k_repeat.ps1` **一遇 AR/Spec 分歧就 `exit 1`** ⇒ 16 次会在第 2 步中断。
  最小修：加 `REPEAT_ON_DIVERGENCE=record`（分歧记录并继续，仅硬失败停），默认仍 `stop`。
- **G3**：无**重复一致性**字段。最小修：复用**已有纯函数** `g5_8k_qk32_analysis.first_divergence`
  比较 r1/r2 的 `run.gen`，写 `repeat_consistency`；两次不同 ⇒ 标**待查**并停止推进。
- **G4**：**未做跨层 cache 结构检查**。最小修：prefill 后与 decode 后各调一次引擎自带的
  `spec_cache.check_consistent`，记 `cache_structure_ok`。
- **G5**：缺显式 `failure_class` 与 `family_id/window_token_start/token_sha1` 溯源字段。
- ✅ 已具备：完整 token（`run.gen`/`gen_len`/`stop_reason`）、finite（`checks`+`decision_coverage`+`non_finite`）、
  显存（`mem_timeline` 8 时点含 `pt_peak_allocated/reserved` + 设备侧）、
  失败原因（`stage`/`traceback`/`oom`/`verdict`）、结构不变量（`round_invariants`）。
  ⚠️ 首个分歧只在 master TSV，run JSON 内没有，且受 G2 限制。

**(5) 三类结果分类**：**硬失败**（OOM/崩溃/护栏拒绝/环境不合格、NaN/Inf、必需检查缺失、
结构不变量失败、`other_short`、重复不一致）⇒ 落盘+停止；**数值差异待评估**（结构与有限性全过但 AR/Spec token 不同）
⇒ 记录并继续，**不自动归因浮点误差、不自动豁免**；**性能观察** ⇒ 仅观察记录，**禁止**进入 speedup 声明。
重申：**不得**把"不同缓存历史的 single-vs-single"差异填进 `classify_bf16_compare.own_max_abs`
（该字段文档定义是**同位置 block-vs-serial**）；`QK FP32 ≠ 全模型 FP32`；bf16 缓存转 fp32 **不等于**从头重算。

**(6) 资源估算（实测值，非新测量）**：peak alloc/reserved = base AR **5261.651/5722.0**、
base Spec **6906.38/7332.0**、v_fp8 AR **5203.401/5588.0**、v_fp8 Spec **6847.63/7178.0** MiB
（全部 < 8188 MiB；Spec 余量约 856 MiB ⇒ 不得并行）；commit 近期可用 ≈29–34 GiB；
串行 16 次 ≈ **11–14 分钟**；每份 JSON 106–179 KB。

**(7)(8)** 停止条件 7 条与待确认事项 7 项详见清单 §8/§9；**清单本身尚不可执行**（依赖 G1/G2 获批）。

### 4.36 最小工具实现：G1–G5 + 五类状态分开记录（**未启动 GPU**，2026-09-29）

**产物**：`specdec/engine/family_input.py`（G1）、`status_fields.py`（五类状态）、
`family16_summary.py`（G3/汇总）、`verify_readiness_selftest_cpu.py`（47/0）、
并小改 `g5_8k_smoke_cudnn.py`（G1/G4/G5，均**显式启用**）与 `scripts/run_g5_8k_repeat.ps1`
（`families16` 计划、G2、G3、无 BOM 的索引 JSON）。

**(1) G1–G5：全部显式启用，默认行为不变**

| 编号 | 开关 | 未设时的默认 |
| --- | --- | --- |
| G1 manifest 输入 | `SMOKE_FAMILY=F06/F13`（+`SMOKE_MANIFEST`） | **mode=legacy**，走原写死 `test.raw` 前缀；hash 不符**拒绝运行**（`FamilyInputError`） |
| G2 分歧后继续 | `REPEAT_ON_DIVERGENCE=record` | **stop**（原行为：分歧即 `exit 1`） |
| G3 重复一致性 | `families16` 计划内自动（r1 vs r2，按 (family,kv,mode) 配对） | 不做比较 |
| G4 跨层 cache 结构检查 | `REPEAT_STRUCT_CHECK=1` → `SMOKE_STRUCT_CHECK=1` | **不检查**；`cache_structure_ok=None` ⇒ 状态 `not_checked`（**不得当通过**） |
| G5 溯源/失败分类 | 随 G1 写 `input.family`；`failure_class` 恒写出 | 仅追加字段 |

默认保持已验证：`resolve_input_plan()=legacy`、`STRUCT_CHECK=False`、`ON_DIVERGENCE=stop`。

**(2) 五类状态分开记录**（`status_fields.py`，写进每个 run 与汇总）
`execution_status`（`complete` / `env_refused` / `crashed` / `algorithm_failed` / `incomplete` / `missing`）、
`structural_status`（`ok` / `fail` / `partial` / **`not_checked`**）、`repeatability_status`、
`ar_spec_equality`、`formal_gate_status`。
- **环境拒绝与算法失败分开分类**，但 `stops_advance()` 对**两者都为 True**（都停止本轮推进）。
- `formal_gate_status` **结构上只可能是 `not_established` / `fail`**：B6 要求 10 类 family × 两档 KV ×
  exact + determinism + 结构，而本工具每次只跑单 family/单模式 ⇒ **永不 pass**。
- 汇总器**没有**任何名为 `passed`/`success` 的总体字段；缺格 ⇒ `diagnostic_completeness != complete`
  且 `complete_success_claimable=False`；退出码 **0 = 诊断齐全自洽（仍非正式通过）/ 3 = 不齐全**。

**(3) CPU 自测**：`verify_readiness_selftest_cpu.py` **47 PASS / 0 FAIL**，覆盖要求的 8 类：
默认路径不变；manifest/hash 不匹配**拒绝**（含正对照与 `verify_hash=False` 时**如实**记 `hash_verified=False`）；
必需检查缺失**拒绝**；AR/Spec 分歧在 `record` 下**可记录继续**（`stop` 下停止）；
结构失败/重复不一致/异常短输出**停止**；**缺失运行不得生成"完整成功"汇总**（缺 1 格即 `incomplete`，
补齐后也只是 `complete_diagnostics_only`，`formal_gate_status` 仍 `not_established`）。
自测还抓到并修掉两处**自身缺陷**：`structural_status` 把 `all_ok=None`（未记录）误判成 `fail`；
hash 篡改测试因传 `tok=None` 先抛 `TypeError` ⇒ 实际没测到 hash 校验（改为真实 tokenizer + 正对照）。

**(4) 文档更正（逐条对应本轮要求）**
- **选样"接受率序"来源已核实**：`expected_acceptance_rank` 是 builder 里**硬编码的先验**
  （`rank={S1:1,S3:2,S4:3,S2:4}`，层由 **`distinct_token_ratio` 分位**切出），
  manifest 自述 `preregistered_hypothesis … （pilot 证伪）` ⇒ **不是实测接受率**；F06/F13 的差距是先验序差距。
- **不再把"无消费者扫描"说成"从未使用"的证明**：改为
  "**当前代码库中没有预期之外的消费方**（唯一 opt-in 消费方是 `g5_8k_smoke_cudnn.py`，默认关闭）"
  + **经验证据**："现有 run 记录中**没有任何一条**用过 manifest family"（直接看产出，不依赖 import）。
- **不再用 `reserved` 推算物理显存余量**：只保留事实（四格 peak_reserved 5722.0/7332.0/5588.0/7178.0 MiB，
  均 ≤ 8188 MiB，`pt_reserved_within_physical_line`）与结论"不得并行"；
  理由改为"Spec 档本身已到 GB 级高位且同刻设备侧占用可观测"，**不是**由 `reserved` 算出的"余量"。
- **显式固定** `add_special_tokens=True`（写入代码而非依赖库默认）、EOS（`eos_token_id`，期望 **151645**，
  运行时核对）、tokenizer id；**`revision` 未钉**并**明确记为未钉**（不假装已钉）。
- 另记一处**manifest 与 zip 的字符串不一致**：manifest 写 `…-raw-v1/wiki.train.raw`，
  zip 真实条目是 `…-raw/wiki.train.raw`（builder 用的是后者）⇒ 按 basename 解析并在记录里写明差异。

**(5) 本轮未做**：未启动 GPU；**未**启用 QK32；**未**改引擎精度、默认 backend、护栏或验收门槛；
**未**保存全 logits；现有 FAIL/CHECK 与正式正确性门槛全部保留；两份外部文档**原样入库**（保留待批准状态）。

### 4.37 16 次独立 GPU 诊断结果（tag `fam16_20260929T060000Z`，commit `bfc75ff`，2026-09-29）

**授权范围**：仅本轮诊断；不改正式正确性门槛、不启动全量正式实验。

**配置**：target 1.5B / draft 0.5B、BF16、`cudnn_prefill`（仅 prefill 融合）、
target KV ∈{base_bf16, v_fp8}、draft 恒 base_bf16、ctx 8192 / chunk 2048 / k 4 / max_new 64、
greedy、EOS 151645、AR target-only；`diag_precision={qk_fp32:false, prob_fp32:false}` **16/16 确认关闭**；
输入 = manifest family（`source_kind=manifest` 16/16），hash 核验 F06 `71e630c6cd702180`、F13 `9a278731d744a859`；
P-state `qualified=True` 16/16；commit 可用 29.97–30.39 GiB ≥ 16 GiB，**未降门槛**。
串行、独立进程、每次重新 prefill 不共享 cache、无重试、未换输入、未缩短输出、未挑样本。

**完成度**：**16/16** 完成，`missing_cells=0`、`extra_cells=0`、拒绝/失败 **0**；
长度 16/16 = 64，`stop_reason` 16/16 = `max_new_reached`；finite 16/16 通过；
结构检查 16/16 `ok`（AR 每次 2 项；SpecDec 每次 44–54 项）；
`diagnostic_completeness=complete_diagnostics_only`、`claimable=True`、退出码 0；
`formal_gate_status=**not_established**`。

**八组重复：8/8 identical**（64/64，逐 token 相同；峰值显存亦逐位相同）⇒ 无重复性异常。

**八组 AR/Spec 配对：8/8 different**（r1 与 r2 给出的前缀与分歧 token **完全相同** ⇒ 确定性）

| family | KV | 共同前缀 | 首个分歧 idx | AR | SpecDec |
| --- | --- | --- | --- | --- | --- |
| F06 | base_bf16 | **25** | 25 | **715** | **576** |
| F06 | v_fp8 | **25** | 25 | **576** | **4710** |
| F13 | base_bf16 | **47** | 47 | **13282** | **25806** |
| F13 | v_fp8 | **23** | 23 | **28693** | **25179** |

⇒ 分歧在**两条全新 family × 两档 KV**上稳定复现，**不是原调试输入的特例**；
**不自动归因浮点误差、不自动豁免**（现有证据按冻结分类器只能停在 `NEEDS_INSPECTION`）。
另注：AR 侧同样受 KV 档影响（F06 的 AR 在 base/v_fp8 下 token 分别为 715 / 576）⇒
"AR↔SpecDec 不一致"并非唯一差异来源。

**显存（两种口径，不得混用）**：peak_allocated = AR 5261.7/5203.5、Spec 6906.4/6847.7 MiB；
peak_reserved = AR 5722.0/5588.0、Spec **7332.0/7180.0** MiB；
**设备侧同刻**：AR `cuda_free` 1343/1477 MiB、smi used 6302/6163 MiB；
Spec **`cuda_free`=0.0 MiB、smi used 7848/7757 MiB（总 8188）** ⇒ 设备侧确实到顶。
**不得由 `reserved` 推物理余量**；本轮**不做**加速/减速声明。

**五类状态**：运行 16/16 complete；结构 16/16 ok；可重复性 8/8 identical；
输出一致性 8/8 different；正式验收 **not_established**。
**诊断完成 ≠ 正式 correctness PASS。**

**本轮修正的一处自身缺陷**：16 次全 PASS 但汇总报 `execution_status=incomplete ×16` ——
`g5_8k_smoke_cudnn.py` 的终止 stage 是 `run_ended`，而 `status_fields` 首版只认 `"complete"`。
已改为显式终止取值表 + 非终止 stage 记 `incomplete`，并加 CPU 回归；
`verify_readiness_selftest_cpu` 现 **54 PASS / 0 FAIL**，汇总修正为 `claimable=True`。
**该缺陷只在真跑后暴露（工具 vs 实跑的词汇表边界）。**

**进入正式实验仍缺**：① 正确性口径批准及必要验证；② 最终配置回归；③ 小规模 12 条件测量 pilot。
**本轮未自动执行**其中任何一项。

### 4.38 F06/base_bf16 首次分歧（index 25）因果定位（2026-09-29）

**版本矛盾先解决**：16 份记录 `commit=bfc75ff…`、`dirty_worktree=True`（09:00:27Z–09:11:06Z）。
`git_meta` 以 `git status --porcelain` 非空判 dirty ⇒ 未跟踪/未提交都计入。
**可证明**：被执行的 `g5_8k_smoke_cudnn.py` 在 `bfc75ff`↔最终 commit 间 **diff 为空**；
`bfc75ff→c409a95` 的 6 个改动里引擎侧**只有 `status_fields.py`**，其调用点**仅 2 处且都在生成结束之后**。
**不可由产物证明**：记录只有布尔值 ⇒ 无法反推具体脏文件；按操作记录为**文档**更正
（`verification_readiness_…md`），**标注为依据操作记录、非产物可证**。**无证据显示运行时差异涉及执行代码**，
但因记录粒度所限**非完全可证**。

**精确复现**：index 25、AR=**715**、Spec=**576**、25-token 共同前缀一致；**max_new=64 语义完整保留**，
仅用观察钩子暂停。

**决策来源（关键）**：SpecDec **round 11 的 block 行 `rows[1]`**（`pre=8215, Kp=4, m=2`，
`cands=[27685,659,715,4882]`，`g=[27685,659,576,576]`），**输入位置 8216 → 预测 8217 = index 25**；
**不是** `saved_next`。AR 侧为 **s=1 单步**（cache 8216，喂 **659 @8216**）。
块重放与真实行 `elementwise_equal=True, max|Δ|=0`；快照完整性 56 张量全等。

**位置/索引/mask/cache 无错误证据**：AR `s=1,T=8217`（**不应用 mask**，可见 keys ≤8216）；
SpecDec `s=4,T=8219,triu(diag=8216)`（row1 可见 ≤8216）；逐层 cache 长度一致。

**同 token 同位置 logits**：715 与 4710 **逐位相同**（17.125）；**576 相差正好 0.125**（AR 17.0 / Spec 17.125）；
4882 差 0.0625。AR 顶部并列 `{715,4710}` ⇒ argmax **715**；SpecDec 三并列 `{576,715,4710}` ⇒ argmax **576**。

**判决实验（同一份 cache、同一预测位置）**：SpecDec 自己的 cache(8215) 上，
块(s=4) → argmax **576**、三并列、margin 0；单步(s=1) → argmax **4710**、无并列、margin 0.125；
715/576 均 +0.125，4710 不变 ⇒ `elementwise_equal=False, max|Δ|=0.1875`。
**判定 `SAME_HISTORY_STILL_DIFFERS__SUPPORTS_CURRENT_COMPUTE_PATH`** ⇒
**仅"当前这一步的计算形状不同"就足以解释该翻转**，不需要依赖两侧 cache 历史不同。

**已证明**：① 精确复现；② 决策来自 block 行 `rows[1]`、位置自洽；③ 无位置/mask/结构错误证据；
④ 实质差异仅 576 的 0.125；⑤ 同 cache 上形状差异单独即可翻转选词。
**未证明/未排除**：① **未做逐层定位**（0.125 首现于哪个算子未知）；② 两侧 cache 历史**确实也不同**
（共同前缀 8215 上 `all_equal=False, max|Δ|=0.25`，28 个 k 张量仅 1 个逐位相同）⇒ **可能共同作用**，
本轮只证明"形状单独够用"；③ **未**称其为 BF16 固有、**未**归因代码 bug、**未**提出修复；
④ **不**把 bf16 cache 转 fp32 当高精度真值；**不**填入 block-vs-serial 语义的
`classify_bf16_compare.own_max_abs`。

**推荐唯一下一步**：同 cache/同位置/同 token 下，对 s=4 块与 s=1 单步做**逐层张量对照**，
定位 0.125 首现算子（RMSNorm / Q/K/V / RoPE / attention 归约 / AV / o_proj / MLP / 残差 / lm_head）。
若首现于**注意力归约** ⇒ 直接证实"当前计算路径的形状敏感性"；若首现于**非注意力算子** ⇒ 存在更上游输入差异，
需回到 mask/位置或 cache 写入路径。**本轮不做修复、不做精度改动、不启动完整回归、不做性能结论。**

**产物**：`f06i25b_20260929T130000Z_F06_{ar,spec}_*.json/.cache.pt/.rows.pt`、`…_analysis_cpu.json`、
`…_samehistory_*.json/.pt`；工具 `g5_f06_index25_forensics.py`、`analyze_f06_index25_cpu.py`、
`replay_f06_i25_samehistory.py`。

### 4.39 F06/index25 缩到具体模块：首个差异在 L0 Linear（2026-09-29）

**事实复核**（读原始 JSON + 重放脚本，非报告）：AR=715/Spec=576、前 25 token 相同、
决策来自 round 11 的 **block 行 rows[1]**（输入 8216 → 预测 8217，非 saved_next）、
块重放==真实行（max|Δ|=0）、同 cache(8215) 块→576 / 串行→4710 / max|Δ|=0.1875、
重放脚本**不跑生成**（因此不缩短 max_new、块长=真实 4、候选=真实）。

**插桩保真（§三）**：无插桩块行 vs 真实行、插桩块行 vs 真实行、插桩 vs 无插桩 **三者均逐元素相同（max|Δ|=0）**，
并复现 576/4710。（首版有个 **zip 错位**缺陷：串行臂累积 56 条 softmax 捕获 ⇒ 与块的 28 条错位到第一步，
打印还把 non-comparable 误印成"相同"；已修并把两类分开计数。）

**逐层（对齐行 8216→8217；attention 只在共同有效 key [0,8216] 上比）**
- **逐位相同**：`embed_tokens`、`L0.input_layernorm`、**`L0.k_proj`**、**`L0.v_proj`**、**L0 attention 分数**。
- **首个非零差异**：**`L0.q_proj` 6.10352e-05**（输入逐元素相同、absmax 52.25）；同层 **`L0.o_proj` 4.883e-04**（输入逐元素相同）
  ⇒ **同输入同权重下 Linear 输出已不同** ⇒ 差异来自**该 GEMM 对 M(4 vs 1) 的归约**，**不是**上游带来。
- `L0.q_proj` 的 6e-05 未改变 L0 分数（RoPE→bf16 回转吸收）；`L0.o_proj` 的 4.9e-04 **进入残差**，
  是**首个向下游传播**的差异。分数从 **L1** 起出现差异（27/28 层），L1 分数 max|Δ|=**1.0**。

**放大链**：| `embed_tokens` | 0 | — | / | `L0.input_layernorm` | 0 | — | / | `L0.k_proj` | 0 | — | / | `L0.v_proj` | 0 | — | / | `L0.q_proj` | 6.10352e-05 | — | / | `L0.scores` | 0 | — | / | `L0.probs` | 1.19209e-07 | 0.00195× | / | `L0.o_proj` | 0.000488281 | 4.1e+03× | / | `L0.post_attention_layernorm` | 0.000976562 | 2× | / | `L1.input_layernorm` | 0.015625 | 16× | / | `L1.q_proj` | 0.03125 | 2× | / | `L1.scores` | 1 | 32× | / | `L1.probs` | 0.0136769 | 0.0137× | / | `L1.o_proj` | 0.00976562 | 0.714× | / | `final_norm` | 0.5 | 51.2× | / | `final_logits` | 0.1875 | 0.375× |

**局部同输入隔离重放（§六）**：`q_proj` 不同(6.1e-05)、`o_proj` 不同(4.9e-04)、
**`k_proj`/`v_proj` 逐位相同** ⇒ 支持"1536 宽 GEMM 的 M 敏感"，排除"所有 Linear 同类差异"。

**结论边界**：**无逻辑错误证据**（位置/索引/mask/cache 结构均正常）；观察到的是**数值归约路径差异**，
**不**归因代码 bug，**不**称已修复，**不**外推为所有 BF16 分歧的解释。
**因果链部分闭合**：已证"同 cache 仅形状不同即可翻转"且形状敏感点在 L0 Linear；
**尚未**用中间值替换重放证明"L0 的 6e-05/4.9e-04 经 28 层放大后改变了 576 的 0.125"。
**AR=715 的历史贡献仍未解释**（本轮未比较 AR cache 8216 vs Spec cache 8215）。

**推荐唯一下一步**：在诊断分支做**明确范围的中间值替换重放**——把块路径 `L0.q_proj`/`L0.o_proj` 的对齐行输出
替换为串行路径同位置值，看 576 是否回到 4710。**停止条件**：仍为 576 ⇒ L0 Linear 非充分环节，转向逐层替换定位充分环节；
变为 4710 ⇒ L0 Linear 作用被证实，再检验是否仅此一处足够。

### 4.40 F06/index25 四臂中间值替换诊断：L0 q/o 差异**不足以**翻转选词（2026-09-29）

**口径修正（只改解释，保留原始数据）**：① 删除"probs→o_proj 放大 ≈4100 倍"这类**倍率/因果**说法
（不同张量的 max|Δ| 之比不能当误差传播倍率）；② "Q 差异被 RoPE 舍入吸收"降级为**未验证假设**
（scores 相同不能唯一定位差异消失处）；③ "K/V 相同 ⇒ 输出宽度决定差异"降级（内核/归约顺序未取证）；
④ "首个差异"限定为**已观察的对齐位置与观察点**。

**参考与机制验证（全部通过）**：无干预块重放 == 已保存块行（max|Δ|=0）、argmax=576；
串行参考 argmax=4710；**无操作 hook == 臂 A（max|Δ|=0，调用 2 次）**；
替换**机制层**只作用于 `[0, 对齐行]`（hook 每臂对目标模块恰好触发 1 次：B/C 各 1、D 共 2）。
⚠️ **修正**：**最终**输出的非目标行在 C/D 下**会变**（实测 B:True / C:False / D:False）——
被改写行会被后续层注意力看到，属**预期**；首版把"最终非目标行不变"当成范围判据是**错的**。
另：三臂中 **576/715/4710 三个 token 的分数完全未变（均 17.125）**，C/D 与 A 的 0.1875 差异落在其它词表位置。
固定状态：round-11 前 cache（8215）、候选 `[27685,659,715,4882]`、对齐行 `rows[1]`（输入 8216→预测 8217）、
BF16/base_bf16/原 backend、QK32/prob32 关闭、**不跑生成**、每臂独立 cache 副本。

**四臂**
| 臂 | 替换 | argmax | 并列 | 576/715/4710 | vs A 相等 | vs A max|Δ| | vs 串行 max|Δ| |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_original_block | — | 0 | **576** | 3[576, 715, 4710] | 0.0 | 17.125 / 17.125 / 17.125 | True | 0 | 0.1875 |
| B_q_proj_only | q_proj | 1 | **576** | 3[576, 715, 4710] | 0.0 | 17.125 / 17.125 / 17.125 | True | 0 | 0.1875 |
| C_o_proj_only | o_proj | 1 | **576** | 3[576, 715, 4710] | 0.0 | 17.125 / 17.125 / 17.125 | False | 0.1875 | 0.25 |
| D_q_and_o | q_proj, o_proj | 2 | **576** | 3[576, 715, 4710] | 0.0 | 17.125 / 17.125 / 17.125 | False | 0.1875 | 0.25 |

- **B（仅 q_proj）与 A 逐元素完全相同（max|Δ|=0）** ⇒ 该差异对最终对齐行**无可测影响**。
- **C（仅 o_proj）/ D（q+o）改变 logits（vs A 0.1875）但仍选 576**；**C 与 D 最终行逐元素相同（True）**
  ⇒ 与"B 无影响"一致；⚠️ **D 不是两个独立因素之和**（o_proj 在 q_proj 下游）。
- 三臂都**未**让块路径等于串行路径（vs 串行 0.1875/0.25/0.25，C/D 甚至更远）。

**结论强度（按证据支持到哪里）**：**第 0 层差异对最终选词的因果证据弱且不充分** ——
形状差异在 L0 Linear 处确实存在；但仅替换 L0 o_proj、乃至同时替换 q+o **都不足以**把 576 变为 4710。
**未解释**：AR=715 的历史贡献（仍未比较 AR cache 8216 vs Spec cache 8215）；让 576 由 17.0 升到 17.125 的具体环节；
且**单行替换不覆盖前一位置(8215)新写入的 KV 与其它层的独立差异** ⇒ C/D 未选 4710 **不等于**其余层无贡献。
**不声称**已解释 AR=715、不外推所有 BF16 分歧、不称已修复或正式通过。

**推荐唯一下一步**：**逐层递进的前缀替换**——从 L0 起把 L1..L27 的输入残差流替换为串行路径对应值，
找出使 argmax 由 576 变 4710 的**最小层号**。**停止条件**：替换到最后一层仍不为 4710 ⇒ 差异不止来自残差流，
转查 KV 写入/mask/位置；某层替换后变 4710 ⇒ 该层为充分环节下界，再检验仅该层是否足够。
**不再**扩大为精度干预/修复/回归。

**产物**：`f06i25b_20260929T130000Z_substitution_*.json` + `…_substitution_*.pt`；
工具 `specdec/engine/substitution_f06_i25.py`。

### 4.41 F06/index25 输出端隔离：选词由**隐藏向量来源**决定，非 LM head 形状（2026-09-29）

**口径清理（保留历史）**：全文清除"误差放大 ≈4100 倍 / 4.1e3×"等**倍率因果**表述，改为同一对齐行、同一口径的
**数值序列**；"Q 差异被 RoPE 舍入吸收"降级为**未验证假设**；"K/V 相同 ⇒ 输出宽度决定差异"降级
（内核/归约顺序未取证）；"首个差异"限定为**已观察的对齐位置与观察点**。

**固定与复现**：round-11 前 cache(8215)、候选 `[27685,659,715,4882]`、对齐行 index=1（输入 8216→预测 8217）、
BF16/base_bf16/原 backend、QK32/prob32 关闭、**不跑生成**、每臂独立副本。
块 argmax=**576** 且与已保存真实块行 `elementwise_equal=True, max|Δ|=0`；串行 argmax=**4710**。
捕获 `H_B`=(1,4,1536) bf16（stride [6144,1536,1]，contiguous）、`H_S`=(1,1,1536) bf16，**保留原始 dtype**。

**机制校验**：A 逐元素复现原块 logits（0）、D 逐元素复现串行 logits（0）；C 仅替换行 1（其余行相等）；
权重未改；**布局控制 view vs contiguous 逐元素相同（0）** ⇒ 形状差异是 **M 维度**效应而非布局效应；
A/B、C/D 输入按构造同一。
⚠️ **修掉自身缺陷**：首版 `H_B[0,1:2]`（(1,1536)）用 `[0,0]` 取到**标量**，与行相比广播 ⇒ 报出**虚假 max|Δ|=91**；
已改为 `[0]` 并从已保存张量复核。**四格当时用的是 `[0]`，未受影响。**

**四格**
| 格 | 向量 | 形状 | argmax | 并列 | margin | 576/715/4710 |
| --- | --- | --- | --- | --- | --- | --- |
| A | H_B 行1 | 块(4行) | **576** | 3[576,715,4710] | 0.0 | 17.125/17.125/17.125 |
| B | H_B 行1 | 单行 | **576** | 3 并列 | 0.0 | 17.125/17.125/17.125 |
| C | H_S | 块 | **4710** | 1[4710] | 0.125 | 17.0/17.0/17.125 |
| D | H_S | 单行 | **4710** | 1[4710] | 0.125 | 17.0/17.0/17.125 |

**配对**：A vs B（同向量、变形状）max|Δ|=**0.0625**、**argmax 不变**；C vs D =**0.03125**、**不变**；
A vs C（同形状、变向量）=**0.1875**、**argmax 576→4710**；B vs D =**0.1875**、**翻转**；
布局控制 =**0**。`H_B` 行1 vs `H_S` max|Δ|=**0.5**（同类量级 ~90）。

**结论（按证据支持到哪里）**：在本例固定输入上，**决定选词的是进入 LM head 的隐藏向量不同**（两种形状下都翻转）；
**LM head 形状效应确实存在（logits 差 0.03125–0.0625）但不足以改变选词**（argmax 不变）⇒
按规则只证明"数值差异"，**不**宣称解释选词翻转。两效应同时存在，**不可拆成可相加贡献**。

**未解释**：① `H_B` 行1 与 `H_S` 的 0.5 差异**从何而来未定位**（本轮只隔离 LM head 边界，
**不能**归因最终 RMSNorm，可能来自更早环节）；② 原始 **AR=715 的历史影响仍在范围外、未解释**。

**推荐唯一下一步**：把**进入最终 RMSNorm 之前的残差流**（最后一层输出）在对齐行上两臂比较**一次**。
已不同 ⇒ 差异不来自最终 RMSNorm、来自更早；逐元素相同而 H 不同 ⇒ 最终 RMSNorm 在同输入下有形状敏感，
可在该算子直接做同输入隔离。**停止条件**：一次比较无法区分 ⇒ **不自动**扩展为 27 层扫描，需另行授权。

**产物**：`f06i25b_20260929T130000Z_lmhead_*.json` + `…_lmhead_*.pt`；工具 `specdec/engine/lmhead_shape_f06_i25.py`。

### 4.42 最终 RMSNorm 边界核验：差异在 Norm **上游侧**，Norm 形状无贡献（F06/index25，2026-09-29）

**先修诊断校验**：新增 `diag_checks.py`（CPU 自测 **12/0**），修 6 项：① 逐元素比较**先断言 shape**（禁广播，
修掉"标量索引 → 虚假 max|Δ|=91"）；② finite **实算** `torch.isfinite` 且覆盖完整张量（自测用**末尾元素** NaN）；
③ "未改动"必须与**运行前独立快照**比较；④ 缺观察点抛错并停止归因；⑤ `.contiguous()` 如实报告是否真复制
（`layout_control`）；⑥ 任一失败 ⇒ `checks_failed`+落盘+退出 5。
**新断言当场拦下我自己的两次标量索引错误**（`nB[0,0]`、`nD[0]`）⇒ 实验中止而非给出假结论，失败记录保留。
**可事后核验**：上轮 A/B、C/D 输入同一性确为 True（用保存张量另存补充审计）；
**不能事后核验**：上轮 `H_B_unmodified` 等字段当时用**标量比较**得出错误值，旧 JSON 不覆盖，**证据不足**（标注）。

**捕获与复现**：`R_B`/`H_B`=(1,4,1536) bf16、`R_S`/`H_S`=(1,1,1536) bf16；hook **只挂 `tgt.norm`**
并在 hook 内断言 `module is tgt.norm`（非 Decoder 内 Norm）；无插桩 vs 有插桩最终 logits **逐元素相同（0）**；
块 **576**、串行 **4710** 精确复现；两臂对齐行同为**输入 8216 → 预测 8217**。

**四格（调用现有 `tgt.norm`，不重写公式；四格对齐行统一成 (1,1,1536) 过同一份 LM head）**
| 格 | 输入 | argmax | 并列 | margin | 576/715/4710 |
| --- | --- | --- | --- | --- | --- |
| A | R_B（块形状） | **576** | 3[576,715,4710] | 0.0 | 17.125/17.125/17.125 |
| B | R_B 对齐行（单行） | **576** | 3 并列 | 0.0 | 17.125/17.125/17.125 |
| C | R_B(行1:=R_S)（块） | **4710** | 1[4710] | 0.125 | 17.0/17.0/17.125 |
| D | R_S（单行） | **4710** | 1[4710] | 0.125 | 17.0/17.0/17.125 |

**配对**：A vs B（同输入、变 Norm 形状）=**逐元素相同（0）**；C vs D=**相同（0）**；
A vs C（同形状、变输入来源）=**0.5 且选词翻转**；B vs D=**0.5 且翻转**。
机制：A **逐元素复现** `H_B` 行1（0）、D **逐元素复现** `H_S`（0）；C 仅替换行1；权重未改；
`layout_control.control_valid=**False**`（`.contiguous()` 未复制）⇒ **本次不构成独立布局控制**，如实记录。

**边界结论**：**`R_B` 行1 vs `R_S` 进入最终 Norm 之前已不同（max|Δ|=2.5）** ⇒ **上游已贡献差异**；
而**同输入下 Norm 的块形状与单行形状输出逐位相同** ⇒ **本例最终 Norm 调用形状无贡献**，调查重点应**向上游移动**。
`R` 差 2.5 与 `H` 差 0.5 是不同张量的 max 绝对差，**不得**作倍率结论。

**限制**：布局控制无效（如实记录，不能声称已排除布局影响）；仅此一个 cache/位置/输入，不外推；
**未解释** 2.5 的差异由哪层哪个算子产生；AR=715 历史影响仍**未解释**；未做逐层扫描/精度干预/修复/回归。

**推荐唯一下一步**：在同一对齐行比较**最后一层（L27）的输出残差**两臂是否已不同。
已不同 ⇒ 差异产生于更早层，需另行授权分层定位；逐元素相同 ⇒ 差异由最后一层内部算子产生，
可在该层做**同输入隔离**（本轮已验证可用）。**停止条件**：一次比较无法区分 ⇒ **不自动**扫描 27 层。

**产物**：`f06i25b_20260929T130000Z_normboundary_*.json` + `…_normboundary_*.pt`；
工具 `norm_boundary_f06_i25.py`、`diag_checks.py`；三次被 shape 断言拦下的失败记录**保留未覆盖**。

### 4.43 L27 内部定位：差异**已传入 L27**（F06/index25，2026-09-29）

**先纠正上一轮推断（读 `model_self.py` 得代码事实）**：`for layer in layers: x = layer(x)` 之后**紧接着**
`x = self.norm(x)`，中间**无任何计算** ⇒ **L27 输出 == 最终 Norm 输入** ⇒ 上轮 `R` 差 **2.5 就是 L27 输出差**，
不需要再测；且**不能**由"L27 输出不同"推出"差异一定来自更早层"（L27 自身也可能新增差异）。
若 L27 输出相同而 Norm 输入不同，才应先查观察点/快照 —— 本轮已核验两者**逐元素相同**。原记录未覆盖。

**已有产物审计**：逐层记录里 L27 有 9 个观察点但**只有统计量、无原始张量**，且 `layerwise_*.pt` **不存在**
⇒ 历史逐层张量**无法事后复原**（标注**证据不足**）；本轮只**补采缺项**（层输入残差、o_proj 输出、
attention 残差、MLP 输出、层输出 + KV/位置/mask），只读 hook，未重写 forward。

**机制校验（全过）**：插桩 vs 无插桩最终 logits **max|Δ|=0**；**L27 输出 == 最终 Norm 输入**
（块/串行均 `True`，同 run 内两 hook 直接比对）；复现 **576/4710**；全部经 `cmp_tensor`（**先断言 shape**、
实算 isfinite、独立快照）——本轮又两次拦下**我自己的**索引/元组取法错误（失败记录保留）。

**L27 逐点（对齐行，块 vs 串行）** —— 顺序按 `DecoderLayer.forward` 实际代码
| `layer_in` | [1536] | False | **1** | 0.033117 | [609] | 154 | 155 |
| `input_layernorm` | [1536] | False | **0.125** | 0.00584727 | [865] | 11.3125 | 11.1875 |
| `q_proj` | [1536] | False | **0.0625** | 0.00639744 | [441] | 9.5625 | 9.625 |
| `k_proj` | [256] | False | **0.0820312** | 0.0129474 | [119] | -0.726562 | -0.808594 |
| `v_proj` | [256] | False | **0.0996094** | 0.0194 | [218] | 0.394531 | 0.494141 |
| `atten_out` | [1536] | False | **0.25** | 0.00814429 | [609] | 30.5 | 30.75 |
| `post_attention_layernorm` | [1536] | False | **0.25** | 0.00767073 | [553] | 17.5 | 17.25 |
| `mlp` | [1536] | False | **0.5** | 0.0211104 | [609] | -78 | -77.5 |
| `attn_residual(derived)` | [1536] | False | **1.25** | 0.034168 | [609] | 184.5 | 185.75 |
| `layer_out` | [1536] | False | **2.5** | 0.0432567 | [609] | 106 | 108.5 |

**首个不同点 = `layer_in`（L27 输入残差）max|Δ|=1.0** ⇒ **差异已经传入 L27**，不是在其中新产生。
（`attn_residual` 为派生量，已标注。）**没有一个点逐位相同**，但 o_proj/MLP/残差处的差异是在**不同输入**下
得到的，按规则**不能**据此归因这些模块本身。**不同张量 max|Δ| 不构成放大倍率**，不给倍率结论。

**KV/位置对齐**：共同 cache **8215**；块 T=8219/s=4、新增位置 8215–8218，位置 8215 token=**27685**
（作为**块输入**计算）；串行 step1 T=8216、step2 T=8217/s=1，位置 8215 token=**27685**（由**第一步**写入）；
对齐行有效可见 keys ≤ **8216**（两臂相同）；mask 分别 `triu(diag=8216)` 与 `s==1` 不应用。
⇒ token 相同但 **K/V 由不同前向调用产生** ⇒ **不能**把"隐藏向量相同"当"attention 输入相同"。

**局部对照未执行并给出理由**：所有 L27 观察点（含**第一个点 `layer_in`**）输入**都已不同**
⇒ 无法在 L27 内构造"固定相同输入"的同输入实验；要合法隔离需**残差流替换或从更早层取相同输入**，
**超出本轮范围**（按 §六停止并提交方案），**本轮不展开**。

**已证明**：R 差 2.5 就是 L27 输出差；**差异已传入 L27**；机制校验全过；KV/位置/mask 已记录。
**未证明**：差异由哪层/哪个算子**首次**产生（未向前追溯）；L27 **是否新增**差异（不能归因）；
AR=715 历史影响（范围外）。**不**宣称 L27 是唯一根因或生产修复点。

**推荐唯一下一步**：把同一套逐点对照**向上推进一层（L26）只做一次**：若 `layer_in` 已不同 ⇒ 更早，
继续上溯需另行授权；若 `layer_in` **相同**而 `layer_out` 不同 ⇒ **L26 内部产生差异**，可在该层做同输入隔离。
**停止条件**：某层两处同时不同且无法区分 ⇒ **不自动**继续上溯，需授权；本轮结论不因下一步改变。

**产物**：`f06i25b_20260929T130000Z_l27_*.json` + `…_l27_*.pt`；工具 `l27_forensics_f06_i25.py`。

### 4.44 L27 MLP：输入来源效应与调用形状效应分离（F06/index25，2026-09-30）

**两处口径纠正（原记录保留）**：① §9.28.6 说"L27 内部无法构造同输入实验"**是错的** ——
**自然两臂输入不同不代表无法构造同输入实验**：固定一份**已捕获输入**、用**块/单行两种形状**调用**同一模块**
即可合法隔离**调用形状**效应（本轮已在 L27 MLP 上实现）。② 上轮 `attn_residual(derived)`
= `layer_in.float()+atten_out.float()` 是 **FP32 派生和，非实际残差**，其 1.25 的口径已作废。

**复用/补采**：`.pt` 里 L27 的 MLP 输入/输出**只有对齐行**，不足以做四格（A/C 需完整 (1,4,1536)）
⇒ 补采**完整** `U_B`/`U_S`、MLP 输出、**真实 attention 残差**（`post_attention_layernorm` 的**输入**，bf16 原值）。
只读 hook、全程原始 bf16。

**机制校验（全过）**：插桩 vs 无插桩最终 logits `max|Δ|=0`；复现 **576/4710**；`U_B=(1,4,1536)`、`U_S=(1,1,1536)`；
**A 逐元素复现捕获块 MLP 行（0）**、**D 逐元素复现捕获串行 MLP（0）**；C 仅替换行 1；源未被原地修改；
`layout_control.control_valid=False` ⇒ **不**宣称已验证布局排除。

**真实 attention 残差（bf16）**：块行1 vs 串行 `equal=False`、**max|Δ|=2**、
mean 0.0347104、idx [609]、184 vs 186。

**MLP 四格配对**
| **A vs B** 同输入(U_B 对齐行)、变形状 | False | **0.015625** | 2.39785e-05 | [971] | 2.54688 | 2.5625 |
| **C vs D** 同输入(U_S)、变形状 | False | **0.015625** | 0.000413676 | [108] | -2.92188 | -2.90625 |
| **A vs C** 同形状(块)、变输入来源 | False | **0.5** | 0.0211199 | [609] | -78 | -77.5 |
| **B vs D** 同形状(单行)、变输入来源 | False | **0.5** | 0.0211286 | [609] | -78 | -77.5 |

**结论**：**两种影响已分开** —— 调用形状效应**可测但小**（同输入变形状 = **0.015625**）；
输入来源效应**更大**（同形状变来源 = **0.5**）。即 **MLP 收到不同输入带来的差异远大于 MLP 形状本身**，
但**不同张量的 max|Δ| 之比不是"放大倍率"**，不给倍率、不预设可加性。
**不证明** MLP 形状效应导致 576/4710 翻转（未做最终选词替换干预）；
**attention 完整输入未隔离**（KV **来源**已记录，**K/V 张量未做数值核验** ⇒ 来源记录 ≠ 数值核验）；
差异**首次产生处**仍未定位；AR=715 仍在范围外。

**推荐唯一下一步**：把同一方法用于 **L27 attention 分支**，但**必须同时固定 Q/K/V（含位置 8215/8216 新增 KV
与有效可见范围）与 mask**；若无法固定**完整** attention 输入 ⇒ **停止并提交方案**，只报告差异。

**产物**：`f06i25b_20260929T130000Z_mlpshape_*.json` + `…_mlpshape_*.pt`；工具 `mlp_shape_f06_i25.py`。

### 4.45 严格三维 MLP 四格 + 有界尾部重组：形状差**传到 logits 但没碰到判决坐标**（F06/index25，2026-09-30）

**设置**：同 §4.44（F06, `base_bf16`, index 25, 对齐行 1 = 预测位置 **8217**）。
**新工具**：`mlp_tail_f06_i25.py`（修正版四格 + 四条件尾部）、`tail_shape_control_f06_i25.py`（调用形状控制）。

**工具修正（§二）**：B 格改用**严格三维** `U_B[:,1:2,:]`（旧版是 `(1,1536)` 二维）；
每格记录**实际** shape/stride/dtype；必需检查全部改为 `require`；CPU 回归 `diag_checks`
**12 → 18 PASS / 0 FAIL**（新增二维误切片必须被抓）；旧记录一律不覆盖。

**复用审计（§三）**：`U_B_bf16/U_S_bf16` **直接复用**；attention 残差虽以 fp32 保存，但
**fp32→bf16 往返逐元素精确** ⇒ 可作**真实残差**复用。
⚠️ **语义纠正**：`l27` 的 `R_tap_*` 是**最终 Norm 输入（= L27 输出）**，**不是** attention 残差；
误用会导致 max|Δ|=**78**（首跑 `checks_failed`，该失败记录保留）。

**四格（严格三维，全部 require 通过）**：A `(1,4,1536)`、B `(1,1,1536)`、C `(1,4,1536)`、D `(1,1,1536)`；
A/D **逐位复现**真实 MLP 输出；`残差 + MLP == L27 输出` 恒等式两臂**逐元素成立**（max|Δ|=0）。

| 配对 | max\|Δ\| | mean\|Δ\| |
|---|---|---|
| A vs B（同输入变形状） | **0.015625** | 2.4e-05 |
| C vs D（同输入变形状） | **0.015625** | 4.1e-04 |
| A vs C（同形状变来源） | **0.5** | 0.0211 |
| B vs D（同形状变来源） | **0.5** | 0.0211 |

**尾部四条件**（`T = R + M` 生产 bf16 加法 → `tgt.norm(T)` → **同一** LM head；均 `(1,1,1536)`）：

| 条件 | argmax | 最高分并列 | margin | 576/715/4710 |
|---|---|---|---|---|
| T_A 块残差+A | **576** | **3 [576,715,4710]** | **0.0** | 17.125/17.125/17.125 |
| T_B 块残差+B | **576** | **3** | **0.0** | 17.125/17.125/17.125 |
| T_C 串行残差+C | **4710** | 1 [4710] | **0.125** | 17.0/17.0/**17.125** |
| T_D 串行残差+D | **4710** | 1 [4710] | **0.125** | 17.0/17.0/**17.125** |

`T_A`/`T_D` **逐元素等于真实 L27 块/串行输出**（max|Δ|=0）⇒ 尾部路径不是自造。
形状差**存活**（T_A/T_B 差 2 个坐标 max 0.015625；T_C/T_D 差 72 个坐标 max 0.03125），
**传播**到最终 logits（8312 / 36097 个坐标不同，max 0.0625），
但**判决三坐标处处逐元素相同（0.0）** ⇒ **不改变选词**。

**调用形状控制**：**生产块路径**（norm 作用于 `(1,4,1536)`）**逐元素复现真实块行（0）**；
单行路径（`(1,1,1536)` 或 `(1,1536)`，二者逐元素相等）与生产差 **max 0.0625**（非判决坐标）；
**`tgt.norm` 输出对形状不敏感（0）**，敏感的是 **LM head 的 M 维**（与 §4.41 原记录的 0.0625 一致）。

**真实行证据（磁盘既有产物，未新跑）**：位置 8217 上
AR 行 = 17.0/**17.125**/**17.125**（并列 `{715,4710}`，margin 0）→ 715；
块行 = **17.125/17.125/17.125**（并列 `{576,715,4710}`，margin 0）→ 576；
串行行 = 17.0/17.0/**17.125**（margin 0.125 = 该量级 1 个 bf16 ULP）→ 4710。
**AR 行 vs 块行：max|Δ| = 0.1875 = 1.5 ULP**，无任何坐标差 > 0.25；判决坐标上仅 576 处差 **0.125**。
⇒ index 25 的"AR=715 vs Spec=576"是 **1 ULP 改变并列结构、并列结构（最小下标规则）改变选词**。

**不证明**：更早层的首次产生点；attention 完整输入的数值等价性；AR/Spec 的 K/V 数值一致性；
该机制是否解释窗口内**其它**差异点；任何修复或正式正确性通过。
**推荐唯一下一步**：只用既有 `rows.pt` 逐位置统计"差异是否落在并列/1 ULP 量级"，
以决定**并列语义**是否需要单列讨论；出现任一"明显 >1–2 ULP 且不涉及并列"的差异即停止并单独提交。

### 4.46 产物覆盖审计 + K/V 数值核验（CPU）+ attention 方案（F06/index25，2026-09-30，**未跑 GPU**）

**范围**：只读产物/审代码/写方案/CPU 工具；方案全文 `docs/l27_attention_plan_2026-09-30.md`。

**A. `rows.pt` 不覆盖 64 个生成位置**（本轮核实）：Spec `rows.pt` 仅 `round11_rows`/`round12_rows` = **8 行**；
AR `rows.pt` 仅 `real`/`replay` = **2 行且同一位置**；Spec JSON `rounds` 只有 12 条**元数据**、无 logits。
⇒ 「64 位置逐项比较」在本产物集上**不可执行**，本轮**未**执行。**定位点行 ≠ 全程 logits**。

**B. K/V 数值核验完成（CPU，无需 GPU）**：两臂 cache 均为 **RoPE 后、GQA 前**原始 bf16 `(1,2,L,128)`。

| 对象 | 比较 | 相等 | max\|Δ\| | 不同元素 |
|---|---|---|---|---|
| **L27 共同前缀**（位置 0..8214，2,103,040 个数） | AR vs Spec `snap_pre` **K** | **False** | 0.09375 | **4540**（0.22%） |
| 同上 | **V** | **False** | 0.203125 | **5270**（0.25%） |
| **L27 位置 8215**（token 27685，不同 forward） | **K** | **False** | 0.078125 | **189/256** |
| 同上 | **V** | **False** | 0.140625 | **237/256** |
| **L0 共同前缀**（单点对照） | **K** | **True** | **0** | **0** |
| **L0 位置 8215**（单点对照） | **K** | **False** | 0.00390625 | 2/256 |

⇒ ① L0 共同前缀 K **逐位相同**；② **L27 共同前缀 K/V 已不同** ⇒ 「隐藏向量不同 ⇒ attention 输入不同」
**已是实测**，台账中「K/V 张量未做数值核验」一项**关闭**；③ L0 位置 8215（token 与**可见 key 集合**均相同，
差别只能来自调用配置：块 4 行/`s=4`/`T=8219` vs 单步 1 行/`s=1`/`T=8216`）已有 2/256 个 K 元素不同
⇒ 是「调用形状效应」的实测点，但**不能**说它只来自线性层 M 维（L0 内同时有 attention 配置差异）。
**范围**：只核验 L27 + L0 对照，其余 26 层**未核验**（非逐层扫描）。

**C. attention 数据流关键事实（按代码）**：`cudnn_prefill` 只在 `phase=='prefill' 且 s>=2` 走 fused；
**verify(s=4) 与 repair(s=1) 都走 eager** ⇒ 块与单步**内核路径相同**，本例差异不是"融合 vs 非融合"。
mask 语义：`cache=None → triu(1)`；**`s==1 → 不施加 mask`**；否则 `triu(diag=T-s+1)`；
`scores` 为 **bf16**，softmax 以 **fp32** 计算后默认回转 bf16。

**D. 缺失张量**：RoPE 后 Q、attention 核心输出（post-AV/pre-`o_proj`）、scores/概率、
完整 `input_layernorm` 输出。⇒ 最小补采 **< 0.5 MB**，无需生成实验，不保存逐层 attention 矩阵。
**已有**：K/V 全 cache；`U_B`/MLP 输出/L27 输出完整 `(1,4,1536)`。

**E. 下一轮唯一有界实验（待批准，未执行）**：排除「**attention 调用形状（4 行 vs 1 行）+ K/V 张量长度**
造成 8217 选词差异」这一解释。分组 G0a/G0b 重放门 → G1 只改行数（固定 Q/K/V、T、可见性 ≤8216）
→ G2 只改长度 → G3 只改 K/V 来源 → G4 核心与 `o_proj` 分开记账。
**变量纪律**：行数与长度**禁止同时变**后再称纯形状效应；单查询固定长度时**必须显式施加等价 mask**
（生产 `s==1` 分支**不施加** mask，这是刻意的受控偏离，必须记录）。
**不承诺** attention 为唯一根因；**不新增** ULP 阈值；**不**回溯改门槛。

### 4.47 L27 attention 最小对照实验 **G0/G1/G2 已执行**（F06/index25，2026-09-30）

执行记录全文 `docs/l27_attn_g012_run_2026-09-30.md`；来源 hash：源 cache `afc66d8a270c9343`、
源 rows `dc8a866af8ae2c9b`、源 l27 `90bb2eb361bf43d6`；commit `07f8d34…`（dirty，含本轮新工具）。
**未做** G3、正式矩阵、完整生成、逐层扫描；生产文件**零改动**；QK32/prob32 关闭；`weights_bitmatch=True`(338)。

**阶段 A 补采 + G0a 生产重放门（通过）**：一次完整 forward（`snap_pre`(8215) deep clone → `spec_dec._verify`，
phase=verify），只读捕获 L27 `q_rope (1,12,4,128)`（`apply_rope` 只读包装 + L27 作用域标志，finally 还原）、
核心输出（= `o_proj` 输入）、`o_proj` 输出、`ln` 输出。检查：重放最终 logits **逐元素 == 真实块行**
（argmax 576 / 并列 3 / margin 0.0）；`k_all/v_all` **逐元素 ==** `round11_blk` cache；
**历史 cache 前 8215 项未被覆盖**；块形状 `o_proj` 行 == 已存 `atten_out`（**仅重放校验**）；
对齐行可见 key **0..8216（8217 个）**、看不到 8217/8218；施加 mask 后有效位置 scores 全有限、
被 mask 位置**恰为 `-inf`**（`-inf` 为预期值，不误报非有限）。

**G0b 功能式重放门（通过 ⇒ 允许归因）**：功能式**四行**核心输出 **逐元素 ==** 生产捕获核心输出。
CPU 自测 `attn_l27_g012_selftest_cpu` **PASS 33 / FAIL 0**。

**G1（只改查询行数 4→1；同 K/V、T=8219、可见性 0..8216、mask 与块行1逐元素相同）**：
核心输出 **2/1536** 个元素不同（**max 4.88281e-04**，mean 3.18e-07，idx 610：0.076171875 vs 0.07666015625）；
scores **3/98628**（max 7.8125e-03，**未逐一定位**）；概率 **1/98628**（1.19209e-07，head3/key4571 可见）；
统一单行 `o_proj` 后 **23/1536**（max 7.8125e-03）。**概率 top-3 与行和（0.999534）完全不变**。
⇒ 「查询行数」是**真实存在但极小**的数值变量。

**G2（只改被 mask 的 K/V 长度 T=8219→8217；纯切片，有效范围仍 0..8216）**：
核心输出 / scores / 概率 / 统一单行 `o_proj` **全部逐元素相同（0，0 个不同元素）**
⇒ 在本例固定输入下**未观察到**「物理长度」效应；mask 与 kernel 行为符合预期。

**边界**：排除的是「多带的、被 mask 掉的未来 K/V 长度」；**未**排除「查询行数」；输出相同 ≠ 所有输入/kernel 正常；
输出不同 ≠ bug、≠ 证明导致 576/4710 翻转；**不**解释 AR=715（未碰 AR cache）；不作性能声明、不放宽门槛、
不标正式 correctness 通过。**限制**：scores 的 3 处差异未逐一定位（未存 G1 scores 行，且**未**为此重跑）；
未采峰值显存；只测 L27 单层单位置。

**唯一下一步（建议，未执行）**：以「**只替换对齐行、其余 3 行不变、后续子层仍用块形状**」做一次选词因果验证
（走生产 `post_attention_layernorm → mlp → 残差 → 最终 norm → LM head`，比较判决三坐标与并列结构），
并**同时报告**「单行 vs 块形状」混淆（MLP 形状效应 0.015625）。**停止条件**：只做一次；不做 G3；不逐层倒查。

### 4.48 L27 attention 行数差异的**末端传播验证**：A/B 选词不变（F06/index25，2026-09-30）

报告全文 `docs/l27_attn_tail_ab_2026-09-30.md`；commit `b8279ca`（本轮新增文件 ⇒ dirty，**不虚报 clean**）。
**未做** G3、精度扫描、下一层诊断、正式矩阵、完整生成；**未改**生产实现/精度/backend/manifest/门槛。

**唯一变量**：L27 对齐行的 **attention core 数值**（A = G0 四行调用的对齐行；B = G1 单行调用）。
其余全部相同：残差=**同一份 L27 层输入**（`l27.block_layer_in`）、权重同一份、全 bf16、形状全 `(1,1,1536)`、
运算顺序（core→o_proj→**BF16 残差加**→post_attention_layernorm→MLP→**BF16 残差加**→final norm→LM head）相同、
**无新精度开关**、**未**用 FP32 求和替代 bf16 加法。**未新增捕获**（core 与参考 `o_proj` 复用上轮 `.pt`）。

**控制（全部真判定）**：① 逐键 `require` + fp32→bf16 往返精确性；② **残差来源恒等式**
`bf16_add(layer_in, attn_out) == 上轮保存的 attention 之后残差`（逐元素）⇒ 排除「重复加 attention」；
③ **跨运行复现**：本轮单行 `o_proj` == 上轮保存同形状结果（A/B 两条，逐元素）；④ **重复控制**：A 连跑两次
logits 逐元素相同；⑤ 输入与权重未被原地修改；⑥ `weights_bitmatch=True`(338)。

**逐阶段 A vs B**（bf16，`(1,1,1536)`）：`o_proj` **23/1536**（max **7.8125e-03**）→ **BF16 残差加法后 1/1536**
（max 不变）→ post_attention_layernorm 后 max 1.95312e-03 → MLP 输出 max **3.125e-02** →
**MLP 残差加法后 77/1536**（max 3.125e-02）→ 最终 RMSNorm **66/1536**（max 3.125e-02）→
**最终 logits max 6.25e-02、mean 1.18942e-03、34023/151936 个坐标不同**。
⇒ 差异**保留并在绝对量级上放大**（BF16 残差加法**未**吸收掉它）。

**最终**：A 与 B **都是 argmax 576**，最高分 17.125，margin **0.0**，并列集合均为 **3 [576,715,4710]**，
**判决三坐标（576/715/4710）逐位相同（17.125==17.125）** ⇒ **A/B 选同一 token**。
**独立参考（不混入因果结论）**：A（单行下游）vs 真实四行 block 行 max|Δ|=0.0625、8306/151936 不同，
真实块行同为 576 + 3 方并列；参考线（真实 L27 输出→单行 Norm→head）同为 576 + 3 方并列。

**结论**：**该位置、该固定下游条件下，L27 query 行数导致的 core 差异不足以改变最终选择** ⇒
**收束这条局部传播诊断**。**不**宣称所有 attention 差异均无关；**不**解释 AR=715（未碰 AR cache）；
**不**定位差异首次产生处；**不**外推到其它位置/输入；**正式正确性门槛仍未被通过**（状态不变）。

**保留的失败记录**：`…_attntailab_20260930T175218Z.json`（把 `R_tap_block` 的 `(1,4,1536)` 当成 `(1536,)`，
形状断言当场拦住，**未覆盖**）。





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
- 2026-09-08（会话续，A4）：**V-only V1 实测 PASS**（fp8_vonly_v1_roundtrip [1]–[7] 全 True，含 current==stored 断言）→ **G3: PASS / FROZEN**（正式 = Asymmetric V-only FP8 KV；negative = K-FP8/full-FP8）。G4 尚未开始。
- 2026-09-08（日终）：阶段 B 仅文档：`design_freeze_g4_specdec.md` v0.1 DRAFT（k=4/greedy/exact-equivalence/cache 不变式/B3 前置门）；B3–B6 未实现；日报 `daily_report_2026-09-08b_G1G2G3_FP8KV.md` §七 收尾归档。
- 2026-09-08（协议修订 + G4 设计审查）：`plan_v7_1_protocol_freeze.md` 增 **Amendment v7.2（§0）**：Compute=BF16；KV 因子={base_bf16, v_fp8}；质量参考=uncompressed BF16；Δ 相对 base_bf16；AR/SpecDec 同档比较；full/K-FP8 仅 negative ablation 不入 factorial；正式=12 conditions（历史"18 格"=旧 bookkeeping）；带宽归因改"反量化开销>省内存潜在收益、未独立隔离带宽、不作更强归因"。`design_freeze_g4_specdec.md` → **v0.2**（取消 bonus；logits 对齐规则 [saved, b_0..b_{k'-2}] 判 d1..d_k'，b_{k'-1} 不判；输入/template 一致；trim/rollback 精确语义；越界抛错；OOM finally；base/v_fp8 共用逻辑）。新增 **G4-0 harness**（`g4_0_draft_correctness.py`，0.5B bf16 门槛，编译通过，**尚未 GPU 运行**）。
- 2026-09-08（G4-0 归因，已实测）：**CONDITIONAL PASS — BF16 shape drift characterized**。OURS full==HF full（max|Δ|=0）、OURS cached==HF cached（=0）；HF cached/full 同处分歧；所有路径重复确定；FP32 四路径一致 → 非 ENGINE BUG/NONDETERMINISM。block-vs-serial：被测 argmax 一致、cache 终长一致、无 off-by-one；block/serial cache 数值差 max|Δ|≈2.0 → **B6 exact-id 门保留为风险**。失败样本（cached=' for' vs full=' since', 生成第 8 步）保留。新增诊断 harness `g4_0_drift_diag.py`、`g4_block_vs_serial.py`（已实现，GPU 运行结果已用于归因）；G4-0 全量生成 exact-id 等价仍由 B6 把关（未做）。B3 harness `g4_b3_dual_model_feasibility.py` 就绪待跑。
- 2026-09-08（B3 实测）：**B3: PASS（4/4 条件）**——1.5B+0.5B 双模型同驻 BF16，ctx512/8192 × target-KV base_bf16/v_fp8 全部 prefill+decode smoke 无 OOM、finite、3 轮稳定；详见 §4.5.2。v_fp8 与 base 的 alloc 峰值在双模型 8K 下相同（6.16 GiB），不宣称峰值收益。
- **日期更正 + 2026-09-15 登记**：§4.5.2（B3）与 §4.5.3（B4）实际运行于 **2026-09-15**（早前标签沿用会话内 09-08，特此更正，数据不变）。同日：协议 Amendment v7.2 写入 `plan_v7_1_protocol_freeze.md`；G4 设计文档升 v0.2（取消 bonus / logits 对齐规则 / cache 回滚语义）；G4-0 归因登记为 **CONDITIONAL PASS（BF16 shape drift）**；`spec_cache.py` 由用户实现并**通过 B4**；`spec_dec.py` 已建文件但**尚未实现**（B5/B6 harness 已就绪：`g4_b5_smoke.py` / `g4_b6_correctness.py`）。日报见 `daily_report_2026-09-15.md`。
- 2026-09-15（B5/B6 实测）：**B5: PASS**（T1–T4 单测 + 真实投机 smoke；cache 不变量全成立；修掉 `(1,V)`→`(1,1,V)` 造成的 0/58 空过 PASS）；**B6: CHECK（19/20）**——base_bf16 档 9/10、v_fp8 档 10/10，唯一分歧 `natural-2` @7 经 `g4_b6_tie_diag.py` 归因为**bf16 数值不可判定位置**（signal 0.1491 ≤ noise 0.1491；fp32 真值 = spec 侧 token）；**用户裁决 A = 严格口径：不设例外、不放宽门，B6 记 CHECK**。同日修正 `spec_dec._verify` 的 `device="cuda"` 硬编码；详见 §4.5.4 / §4.5.5。
- 2026-09-15（EOS 分支修复 + B6-fp32 预登记）：逐行评审 `spec_dec.py` 发现并修复 **EOS 全接受路径的 `trim_cache` 越界**（`_propose` 的 EOS break 位置 → draft cache 长度 `pre+kp-1`），改为"EOS 检查在推进之后"使不变量统一为 `pre+kp`；新增**纯逻辑回归测试** `g4_b5_eos_branch_tests.py`（CPU stub，含旧实现对照复现 ValueError，B5-EOS-branch: PASS，详见 §4.5.4）。同时新增并**预登记** `g4_b6_fp32_correctness.py`（B6-fp32：CPU/float32 下的实现正确性门；判据在运行前入库，仅当 case×两 KV 档全部逐 token 相等才 PASS），结果待跑。
- 2026-09-15（B6-fp32 实测）：**B6-fp32: PASS** —— 10 case × 2 KV 档 = **20/20 组合逐 token id 全等**（参照 = 同引擎 cached AR），finite 全真、确定性通过、非空过（总 **216/348**，62.1%），两档逐 case 数字相同；运行于 EOS 修复之后，且修前小样数字逐字复现。**B6 门按两档并列登记：fp32 档 = 实现正确性证据；bf16 档 = 性能档行为 CHECK 19/20 + tie-collapse 归因（§4.5.5 原样保留）**。新增 RQ2 观察：bf16 档接受率 50.9% vs fp32 档 62.1%，且**接受率强依赖 prompt**（29.0%–100%）；差值属复合效应，单独归因需 tie 频率普查。详见 §4.5.6。
- 2026-09-15（B5 修后 GPU 复验）：`g4_b5_helper_tests` + `g4_b5_smoke` 在 EOS 修复（`f7097d4`）**之后**重跑 → **两档全 PASS**，逐轮 cache 账全 `ok`；`target_block_calls 8`、`draft_calls 27` 由实测确认为推导值；新登记结构性恒等式 `gen_len == accepted + mismatch_rounds`。B5 的 PASS 记录现已对应当前代码（§4.5.4）。
- 2026-09-30（§4.45，F06/index25 工具修正 + 尾部重组）：修正 `mlp_shape` 的 B 格为**严格三维**并重做四格（结论与旧版一致）；新增**四条件尾部**与**调用形状控制**；`diag_checks` CPU 回归 **12→18 PASS / 0 FAIL**。关键新事实：MLP 形状差**存活、传播到 logits，但不改变选词**（判决三坐标逐元素相同）；**生产块路径逐元素复现真实块行**，单行调用路径差 **0.0625**（`tgt.norm` 输出不敏感，敏感的是 **LM head 的 M 维**）；位置 8217 上 AR 行与块行**逐元素最大差 0.1875 = 1.5 bf16 ULP**，三臂分别为并列 `{715,4710}` / `{576,715,4710}` / 单点+0.125 ⇒ index 25 的 715↔576 是**1 ULP 改变并列结构**所致。口径纠正：`l27.R_tap_*` 是**最终 Norm 输入**而非 attention 残差；我此前把 §9.26 的 LM head 形状效应说成"0"有误（记录为 **0.0625**，记录本身正确）。**正式门状态不变（`not_established`）**，本节只作诊断。
- 2026-09-30（§4.46，**纯 CPU**：产物审计 + K/V 数值核验 + attention 方案）：核实 **`rows.pt` 不覆盖 64 个生成位置**（Spec 仅 8 行、AR 仅 2 行且同一位置；`rounds` 只有元数据）⇒ 明令**未**执行「64 位置逐项比较」。**关闭「K/V 未做数值核验」这一旧缺口**：两臂 cache 均为 RoPE 后、GQA 前 bf16，CPU 逐元素比出——**L27 共同前缀**（位置 0..8214）K 差 4540/2103040（max 0.09375）、V 差 5270/2103040（max 0.203125）；**L27 位置 8215** K 差 189/256、V 差 237/256；而 **L0 共同前缀 K 逐位相同（0 差异）**、**L0 位置 8215 有 2/256 个 K 不同**（token 与**可见 key 集合**均相同 ⇒ 只能来自调用配置差异）。代码核对确认 `cudnn_prefill` 只在 `prefill 且 s≥2` 融合，**verify/repair 都走 eager** ⇒ 块与单步**内核路径相同**。下一轮唯一有界实验（待批准）＝排除「**attention 调用形状 + K/V 张量长度**造成 8217 选词差异」；变量纪律：行数与长度**不得同时变**，单查询固定长度**必须显式施加等价 mask**（生产 `s==1` 不施加）。**未跑 GPU、未改生产实现、未新增阈值、门槛不变**。
- 2026-09-30（§4.47，**G0/G1/G2 已执行**，有界 GPU 重放）：方案按 7 条修订（G0 必须完整四行、Q 布局核验为 `[b,heads,s,hd]`、对齐行可见 **8217 个 key（0..8216）**、删除模糊 ULP 停止规则、补采/局部重放资源分开、修正「L0 K/V 一致」与「K/V 差异不得归因本层 attention」、G3 暂缓）。**G0a 通过**（重放最终 logits 逐元素==真实块行；`k_all/v_all`==既有 blk cache；历史 cache 未被覆盖；块形状 `o_proj` 行==已存 `atten_out`；mask 与 `-inf` 语义分开核验）。**G0b 通过**（功能式四行==生产核心，逐元素）⇒ 允许归因。**G1（只改查询行数 4→1）**：核心 **2/1536**（max 4.88e-4）、scores 3/98628、概率 1/98628（1.19e-7）、统一单行 `o_proj` 23/1536（max 7.81e-3）；**top-3 概率与行和不变** ⇒ 效应**存在但极小**。**G2（只改被 mask 的 K/V 长度 T=8219→8217）**：核心/scores/概率/`o_proj` **全部逐元素相同（0）** ⇒ 该解释**被排除**。CPU 自测 **33 PASS / 0 FAIL**；两条失败记录保留未覆盖。**门槛不变、无性能声明、不解释 AR=715**。
- 2026-09-30（§4.48，**末端传播验证已执行**）：只替换 L27 对齐行 attention core（A=四行 vs B=单行），下游形状统一 `(1,1,1536)`、残差取 **L27 层输入**（并用恒等式 `layer_in+attn_out==` 上轮保存的 attention 之后残差 硬校验，杜绝重复加 attention）、BF16 加法、生产模块、无新精度开关。控制全过（跨运行复现单行 `o_proj`、A 重复跑逐元素一致、输入/权重未改、权重 bitmatch 338）。逐阶段差异**保留并放大**：`o_proj` 23/1536（max 7.8e-3）→ 残差加法后 1/1536 → MLP 后 max 0.03125 → 最终 logits **max 0.0625、34023/151936**；但**判决三坐标逐位相同**，A/B **同为 argmax 576**、margin 0.0、并列 3 [576,715,4710]。⇒ **该位置、该固定下游条件下，行数导致的 core 差异不足以改变最终选择**，这条局部传播诊断**收束**。**不**宣称所有 attention 差异无关、**不**解释 AR=715、**不**定位首次产生处、**不**外推；**正式正确性门槛状态不变**。CPU 自测 **22 PASS / 0 FAIL**；失败记录保留未覆盖。
