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
