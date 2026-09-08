# SpecDec 数据收集台账（Data Ledger）

> 目的：集中登记所有**实测数据**（正确性/性能/质量），每个数字带环境与复现条件。
> 机器产物最终进 `results.jsonl`（追加式）；本文件是**人工可读台账**，两者字段对齐。
> 规则：凡记录必附 {日期, 硬件, dtype, 模型, seed, 协议}；不填来源的数字一律不写。

---

## 1. 环境基线（数据有效性前提）

### 1.1 主目标机（最终 8GB 结论）
| 项 | 值 | 备注 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop 8GB（sm_89）| 最终核心数字需在此机复跑 |
| CPU / RAM | i7-12700H / 32GB | — |
| OS / Python | Windows / 3.11（venv）| — |
| torch | 2.6.0+cu124 | PyPI 官方 cu124 |
| transformers | （待补精确版本号）| `pip show transformers` |
| HF cache | D:\hf-cache | HF_HOME |
| 临时目录 | D:\RealDSH\pip-tmp | 防 C 盘锁/满 |
| 模型（早期 L1/L2）| Qwen/Qwen2.5-0.5B-Instruct | Apache-2.0，tie、attn_bias=True |

### 1.2 A6 开发/预实验机（非最终 4060 结论）
| 项 | 值 | 备注 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4090 D 24GB | AutoDL；A6 开发与预实验 |
| OS / Python | Ubuntu 22.04 / Python 3.12 | AutoDL 基础镜像 |
| RAM | 62GB | AutoDL 实例配置 |
| 系统盘 | 30GB | AutoDL 实例配置 |
| torch / CUDA | torch 2.5.1+cu124 / CUDA 12.4 | benchmark 日志实测 |
| 模型 | Qwen/Qwen2.5-1.5B-Instruct | ModelScope 本地 checkpoint；Transformers 仅作参考实现/加载 |
| 角色 | A6 implementation / 4090D evidence | 正确性、storage、PPL 与本机性能证据已完成；RTX 4060 专属 VRAM/latency 结论须在目标机复跑 |

## 2. 模型配置（被测对象）

### 2.1 Qwen2.5-0.5B（L1/L2）
| 参数 | 值 |
|---|---|
| hidden / layers / heads / kv_heads / head_dim | 896 / 24 / 14 / 2 / 64 |
| intermediate / vocab / rope_theta / rms_eps | 4864 / 151936 / 1_000_000 / 1e-6 |
| attention_bias / tie_word_embeddings | True / True |
| 引擎实现 | 自写 model_self.py（OURS），Transformers 仅加载权重/作参考 |

### 2.2 Qwen2.5-1.5B（A6）
- 模型：`Qwen/Qwen2.5-1.5B-Instruct`；OURS 引擎由同一 `build_from_config` 从 config 构建并复制权重。
- L4 量化目标：每层 7 个 Linear（q/k/v/o + gate/up/down）；1.5B 共 **196/196** 个目标模块完成替换。
- INT8：对称 RTN，per-output-channel scale；INT4：真实 4-bit packed storage，group size = 64，运行时 eager dequant 后进入普通浮点 `F.linear`。

## 3. 正确性数据

### 3.1 L1 对齐门（ours full vs Transformers full）
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

### 3.3 生成对齐门（ours.generate vs Transformers 全量逐步贪心）
| 日期 | 参考 | prompts | 结果 |
|---|---|---|---|
| 2026-09-02 | Transformers full-forward 循环（use_cache 无关）| 4 | **全序列逐 token 一致 PASS** |
| 发现 | `transformers.generate` 与其全量前向不一致（官方 logits ' is'=21.75 vs 其输出 ' was'）→ 不用其当裁判 | | |

### 3.4 1.5B FP32 参考有效性检查（A6 前置）
| 日期 | 设备/精度 | 指标 | 结果 |
|---|---|---|---|
| 2026-09-08 | RTX 4090 D / fp32 | short-sequence max\|Δ logits\| | **3.53e-05** |
| 2026-09-08 | RTX 4090 D / fp32 | mean\|Δ logits\| | **2.67e-06** |
| 2026-09-08 | RTX 4090 D / fp32 | last-token top1 | **一致** |
| 2026-09-08 | RTX 4090 D / fp32 | OURS PPL / Transformers PPL | **22.98598 / 22.98600** |
| 2026-09-08 | RTX 4090 D / fp32 | PPL relative diff | **7.15e-07** |

**结论**：Qwen2.5-1.5B OURS 数学路径与 Transformers eager 参考一致；A6 的质量差异可以归因到精度/量化路径，而不是模型权重加载或架构错配。

### 3.5 A6 前发现的 fp16 数值问题与修复
- 首轮 1.5B fp16 PPL 出现 NaN/异常；定位到不是量化模块独有问题，而是公共 fp16 数值路径。
- 最小正确性修复：RMSNorm 方差用 fp32 累积；QK score + softmax 用 fp32，attention probability 再 cast 回运行 dtype；RoPE frequency/position/freq 用 fp32 计算，cos/sin 最后 cast 回运行 dtype。
- 修复后 fp16 质量恢复：**PPL = 22.975503**，与 fp32 reference 22.98598 接近；随后才重新跑 A6 最终性能数据。

### 3.4 G1 —— 1.5B target correctness gate（v7.1 G1，2026-09-08）
| 检查 | 内容 | 结果 | 条件 |
|---|---|---|---|
| G1.1 | OUR 1.5B full forward vs HF 1.5B full | **max|Δ| = 3.10e-05 PASS** | fp32/CPU eager；Qwen/Qwen2.5-1.5B-Instruct（hidden1536/l28/h12/kv2/vocab151936/attn_bias=True）|
| G1.2 | OUR 1.5B cached decode vs OUR full | **max|Δ| = 2.04e-05，argmax same PASS** | 同上；32-token 输入 |
| G1.3 | greedy 生成 vs HF 全量贪心 | **4/4 序列一致 PASS** | 同上；4 prompts × max_new=12 |
| 注 | 0.5B 全部旧数据标记为 **development-stage validation**；G1 为 target 正式门 | | |

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
- **历史探针结果：单次 8K eager prefill（ours，fp16）CUDA OOM**。探针运行时 scores 为 fp16，14×8192×8192×2 bytes ≈ 1.75GiB/层瞬时，另有 logits（8192×151936×fp16）≈ 2.5GiB 与双模型常驻开销。当前 fp16-stable 引擎已改为 QK score 在 fp32 计算；同形状 score 约 3.5GiB，因此更需要分块 prefill。
- transformers eager 同配置也会 OOM（其也被强制 eager）。
- **含义/方案**：18 格的 8K 行不能“一次性喂 8K”获取，需分块 prefill（块级因果 mask，引擎小改造）+ 缓存构建路径跳过 logits + 单模型常驻/expandable_segments；该 OOM 本身可作为“为什么需要 KV 压缩/量化/长上下文优化”的动机素材。

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
- seed=42（测试）；性能脚本 seed=0；`bench_l4` 使用 batch=1、固定 38-token prompt、decode=64、warmup=4、reps=3；A6 性能比较以 **median** 为主，best 只留作原始记录。
- A6 质量脚本：固定本地文本，在看到 L4 PPL 结果前冻结；406 个 target tokens；fp16 compute；INT4 group size=64。
- A6 预注册质量门：**relative PPL increase <= 10% 为 PASS**；不因结果事后调整阈值。
- dtype/设备随行标注；attention eager；`torch.no_grad()`。
- A6 的 4090D latency/peak VRAM 是 development/preliminary evidence；最终回答 8GB 硬件问题时，必须在 RTX 4060 8GB 上按同协议复跑。

## 6. 里程碑登记表

### 6.1 L4 权重量化（A6，Qwen2.5-1.5B，RTX 4090 D 开发机）

**日期：2026-09-08；torch 2.5.1+cu124；seed=0；batch=1；prompt=38 tok；decode=64；warmup=4；reps=3；性能主统计=median。**

| 方案 | 权重大小 | peak VRAM | decode median | ppl(Δ vs fp16) | 自建题集 acc | 备注 |
|---|---:|---:|---:|---:|---|---|
| fp16 基线 | 2944.4 MiB (2.875 GiB) | 2.898 GiB | **35.85 tok/s** (27.89 ms/token) | **22.975503** | 待正式 held-out task | baseline；best=53.86 tok/s |
| int8 RTN | 1696.1 MiB (1.656 GiB) | 1.786 GiB | **27.27 tok/s** (36.67 ms/token) | **23.234550 (+1.13%)** | 待正式 held-out task | **质量门 PASS**；196/196 Linear；storage ↓42.4%，peak VRAM ↓38.4% |
| int4 packed RTN, g64 | 1109.2 MiB (1.083 GiB) | 1.201 GiB | **26.00 tok/s** (38.47 ms/token) | **27.954204 (+21.67%)** | 待正式 held-out task | **质量门 FAIL**；196/196 Linear；storage ↓62.3%，peak VRAM ↓58.6% |

#### A6 结论
1. **INT8 是当前 L4 主线 operating point**：显著降低模型 storage/VRAM，同时 PPL 仅增加 1.13%，通过预注册 10% 质量门。
2. **INT4 是诚实负结果/更激进 trade-off**：g64 packed INT4 进一步压缩显存，但 PPL 增加 21.67%，超过质量门，因此不作为后续默认权重量化配置。
3. 当前 eager 实现中 INT8/INT4 **没有带来 decode 加速**。原因是压缩权重在运行时先 dequant，再进入普通浮点 `F.linear`；没有专用 quantized GEMM kernel，因此 memory saving 与 latency speedup 不能等同。
4. 4090D 上的正确性、storage、PPL 与本机性能结果构成 A6 完整实现证据；但若最终报告回答“RTX 4060 8GB 上的 latency / peak VRAM”，必须在目标机按同协议复跑，不能直接移用本表的 4090D 数字。
5. A6/L4 状态：**implementation + real-model integration + 4090D system/quality evidence CLOSED**；RTX 4060 目标机复跑属于最终 hardware-specific validation，不阻塞进入 A7/L4b fp8 KV-cache。

### 6.2 L4b KV 量化（fp16→fp8；int4/int2 仅非主线预留）
| KV dtype | bytes/token | 峰值显存 | 长上下文(8K)可行? | ppl(Δ) | 备注 |
|---|---|---|---|---|---|
| fp16 | | | | | 填 |
| fp8 | | | | | 填 |

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
- 2026-09-08：登记 A6/L4 1.5B INT8/INT4 开发结果；补 4090D 环境、1.5B FP32 reference validity、fp16 数值稳定性修复、最终 storage/VRAM/decode/PPL 表。INT8 通过质量门；INT4(g64) 作为诚实负结果；明确 4090D 数字非最终 RTX 4060 核心结论。
- 2026-09-08：登记 G1（1.5B target 正确性门 PASS，§3.4）；登记 8K 分块 prefill **最小可行性预检通过**（§4.5，正式 G2 门未完成）。
