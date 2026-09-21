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
**尚未在 GPU 上运行** ⇒ 归因**未定**。

#### 4.6.7 Part 4 状态
**CHECK**：8K 隔离复验已完成并**证伪"8K 稳定加速"**（1.286× 不可复现、SpecDec 全面慢于 AR、
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
