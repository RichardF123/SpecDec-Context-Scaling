# SpecDec — 从零实现的小型 LLM 推理引擎（课程项目）

> **当前 proposal 定稿（2026-10-01）**：[版本与范围记录](docs/proposal_freeze_2026-10-01.md)。主实验 6 条件 + 8K V-only FP8 补充 2 条件；保留已有 FP8 成果。以下旧日期的 12 条件描述是历史计划，不代表当前提交范围。定稿不等于正确性通过，执行协议与工具仍需配套同步。

> 从零实现 Qwen2 系推理引擎 + KV-cache 解码 + 优化阶梯（KV 压缩 → 投机解码），
> 在 8GB 笔记本 GPU 上做可复现测量。
> 权威文档：`docs/plan_v7_1_protocol_freeze.md`（实验协议冻结 + Amendment v7.2）、`docs/data_ledger.md`（数据台账）、
> `docs/design_freeze_fp8_kv.md`（FP8 KV 设计冻结 v1.3）、`docs/design_freeze_g4_specdec.md`（G4 协议 v0.3）、
> `docs/plan_g4_next_steps.md`（G4 阶段状态）。

## 当前状态（2026-10-02 更新）

**引擎、V-only FP8 KV、greedy SpecDec 和测量工具已有实现；当前阻塞项是生产 BF16 路径的 AR/Spec 严格生成一致性。正式实验尚未开始，不声明 SpecDec 已加速。**

当前提交范围为 **6 个主条件（2K/4K/8K × AR/Spec，base BF16）+ 2 个补充条件（8K V-only FP8 × AR/Spec）**，不是历史 12 条件矩阵。定稿范围不等于执行协议、后端和正确性已全部验收。

| 最新进展 | 已验证结果 | 结论边界 |
|---|---|---|
| RoPE 来源定位与 C1 对齐 | 诊断中统一 target 的 CPU FP32 inv_freq 来源后，所测 8K prefill 的 56/56 KV 张量和末位置 logits 与 HF 逐位一致 | 局部参考对齐完成；不是全流程正确性通过，也未改变生产默认 |
| 真实 F13 AR/Spec 复现 | all-eager、BF16、8K、k=4、64 tokens；仅索引 47 不同：AR=6702，Spec=6652；本轮捕获复现上一轮序列 | 在没有 cuDNN 的配置下仍出现分歧；不能直接归咎于 cuDNN |
| 同前缀四格定位 | AR 历史下 single/block 都选 6702；Spec 历史下两者都选 6652；真实路径重放逐元素吻合，格内重复一致 | 缓存历史影响本位置选择；当前形状仍改变 logits，但未改变这四格中的选择 |
| 缓存与决策检查 | prefill 段逐位相同；生成缓存最早在位置 8192 出现差异；本决策点的行映射、替换规则和尾部泄漏检查未发现问题 | 最早产生差异的具体算子、其他家族及全流程严格一致性尚未确认 |
| 已有 FP8 成果 | V-only 独立 40 窗 relative ΔPPL +0.040%；target 8K 逻辑 KV 约 1.32× 压缩 | 保留为补充研究；不等于整机显存同比下降或解码加速 |

**下一步（尚未执行）**：做一次默认关闭的串行历史重建诊断，覆盖全接受轮和拒绝轮；从轮前缓存重放已提交 token，并同步刷新 saved_next。优先检查同前缀历史是否逐位一致、原位置 8239 的选择是否恢复，再观察完整生成。该实验用于验证因果，不是免费性能优化，也不保证其他位置不会产生新的形状敏感分歧。正式矩阵仍等待严格正确性及测量准备验收，不放宽门槛。

详见：[本次状态与结论边界](docs/status_update_2026-10-02.md)、[RoPE 对齐](docs/c1_rope_source_alignment_2026-10-01.md)、[真实生成对照](docs/ar_spec_aligned_rope_2026-10-01.md)、[第 47 位四格定位](docs/divasc_index47_localization_2026-10-01.md)。定位报告的原下一步建议需按本次状态说明修正，尚未据此执行新实验。


## KV optimization engineering closeout（2026-10-04）

This supplementary engineering track is closed and does **not** redefine the
formal 6+2 course experiment. The separate production BF16 AR/SpecDec
long-context strict-consistency issue remains unresolved.

- Paged KV storage and the final direct PagedAttention path for
  `q_len=1..4` are implemented and semantically validated in the tested
  configurations.
- The direct route covers the asymmetric V-FP8 target and SpecDec
  propose/verify/repair. Real target+draft validation completed through 8K
  context on the recorded RTX 4060 8GB setup for batch 1, `k=4`, and 16
  generated tokens.
- Direct page-aware access removes full-history materialization temporary
  allocation, with savings increasing at 4K/8K; the current custom kernel was
  slower than the materialized CUDA reference in the final RTX 4060
  microbenchmark.
- H2O-inspired committed-history retention is integrated. At the nominal 50%
  budget, resident/seen approaches 0.5, but the current reference
  score-capture and compaction implementation adds substantial decode
  overhead. This is not a speed claim or an intrinsic lower bound for H2O.

See the [authoritative final closeout](docs/kv_optimization_final_closeout_2026-10-04.md),
[4K/8K PagedAttention evidence](results/paged_attention_4k8k_2026-10-04/),
and [H2O latency evidence](results/h2o_latency_2026-10-04/).

Optimization execution addendum（2026-10-04）
After the earlier KV closeout, three additional GPU experiment stages were
completed and archived:
- Phase12 optimized H2O execution with page-native compaction and kernel-
  integrated score capture. It preserves the tested reference output/state and
  materially reduces the old H2O implementation overhead, but optimized H2O
  remains slower than standard direct SpecDec in this harness.
- Phase13 adds an opt-in custom flash_paged_cuda route: a page-native,
  GQA-aware tiled online-softmax CUDA kernel for q_len 1..4, including the
  tested asymmetric V-FP8 target path. This is Flash-style custom CUDA, not
  the official FlashAttention package.
- A separate official FlashAttention-2 BF16 backend was tested with
  preallocated contiguous KV via flash_attn_with_kvcache. It is the strongest
  BF16 latency reference in this extension, but it is intentionally not paged
  and does not claim V-FP8/H2O support.
See
[`docs/optimization_execution_addendum_2026-10-04.md`](docs/optimization_execution_addendum_2026-10-04.md)
and [`results/optimization_results_index_2026-10-04.md`](results/optimization_results_index_2026-10-04.md).
The formal frozen 6+2 course experiment and the separate production BF16
AR/SpecDec strict-consistency issue are unchanged.

## 历史状态快照（2026-09-27）

以下保留当时记录，后续状态以上方 2026-10-02 更新为准；其中“下一步”及 12 条件范围均为历史计划。

**已完成引擎、V-only FP8 KV 和投机解码实现；当前重点是 8K 测量稳定性与 prefill-only cuDNN 验证，尚未完成正式 12 条件实验矩阵。** 以下新结果来自截至 2026-09-26 的已归档实验，不代表后续测试已经通过。

| 阶段 | 当前结果 | 结论边界 |
|---|---|---|
| G1 / G3 | target 正确性通过；V-only FP8 设计冻结 | 正式计算口径为 BF16；V-only 不等于 K/V 全 FP8 |
| G4 | B6 FP32：20/20 组合逐 token 一致；BF16：19/20，CHECK | FP32 实现证据不替代 BF16 exact-id 门；216/348 是 accepted/proposed，不是正确率 |
| G5 测量系统 | canonical v1.2、阶段计时、独立进程探针、P-state 预检及失败记录已实现；已有 151 项 CPU 自测记录 | 测量工具完成不代表正式实验完成 |
| 8K eager 双模型 | 历史可行性测试跑通过；后续出现 OOM/崩溃和计时不稳定 | “能跑一次”不等于可进行稳定的正式性能测量 |
| prefill-only cuDNN | 可选分支已接入，默认仍为 eager；CPU 43/43，GPU 路径复核符合预期 | 只融合显式 prefill；decode/propose/verify/repair 保持 eager |
| cuDNN 对齐验证 | 算子级 8/12 行未过门；模型级 5 配置 × 2 次重复均未过门，ctx512/base 生成发生分歧 | 未通过预登记 eager 对齐判据；不直接等同于融合实现错误，也不改写为 PASS |

### 关键实测数据

| 数据 | 结果 | 测量范围 |
|---|---|---|
| V-only FP8 质量 | 独立 40 窗：ΔNLL +0.000397，relative ΔPPL +0.040% | 已冻结 eager 路径；不外推到新后端 |
| target 8K 逻辑 KV | 224.00 → 169.75 MiB，约 1.32× 压缩 | 仅 target KV，不是双模型总显存 |
| 2K base_bf16 探索性结果 | decode speedup 0.406×；e2e 0.434×，未加速 | 缩小版 stability probe；合并 measured reps 取中位数后相除，不是 ≥16-family 正式矩阵 |
| 8K eager SpecDec 波动 | 同配置三次 decode：10.04 / 11.18 / 16.22 s，极差约 61%（相对最小值） | 不支持正式速度倍数声明 |
| target attention 临时显存 | s=2048、T=2048：496 → 18 MiB；s=2048、T=8192：1984 → 54 MiB | eager → cuDNN，单算子测试，不是整模型峰值 |
| draft attention 临时显存 | s=2048、T=8192：2284 → 51.5 MiB | 同上；profiler 确认 cuDNN attention 路径 |

`s` 是当前 query 块长度，`T` 是含历史缓存的总 KV 长度；`s=2048,T=8192` 不是 prefill 第一块。`allocated` 是 PyTorch 跟踪的张量分配，`reserved` 是分配器保留池，两者都不能直接等同于整卡物理占用；单凭 reserved 超过 8 GiB 不能证明发生分页。

**数值归因仍需补证**：合成 Q/K/V 对照支持 BF16 中间舍入是重要原因，但尚不能断言真实模型全部差异只来自 score 舍入，或 cuDNN 内部“全程 FP32”。保留原 FAIL，区分数值差异、生成差异与质量损失。

**下一步顺序**：真实模型同一份 Q/K/V 的精度定位 → 固定短窗口独立 NLL/PPL 对照 → 在证据支持后进行独立进程完整 8K 双模型 smoke → 重复稳定性与新后端 correctness 验证。新旧后端数据分开记录，正式 AR/SpecDec 使用一致的后端策略；当前不启动正式矩阵，不声明 8K 已解决或 SpecDec 已加速。

来源：[数据台账](docs/data_ledger.md)（§3.5.3、§4.14、§4.20、§4.22）、[G5 测量协议](docs/g5_measurement_protocol.md)、[cuDNN 阶段报告](docs/cudnn_prefill_stage_report.md)。阶段报告中的唯一根因表述仍需上述真实张量实验补证，不能据此认定已完成验证。

## 已完成阶段与历史验收

以下为对应时间和配置下的历史记录，不代表当前 8K 正式测量或新后端已经通过。

- ✅ **引擎正确性基线**：L1 从零前向对齐 0.000（fp32 CPU，vs transformers）；L2 KV-cache + 贪心生成对齐 PASS；
- ✅ **G1 — target 1.5B correctness**（fp32 CPU）：full-forward `3.1e-05`、cached-vs-full `2.0e-05`、argmax same、生成序列一致 → **PASS**；
- ✅ **BF16 GPU 对齐（质量口径）**：OURS vs HF 512-token `max|Δ|=0`、argmax 100%、NLL/PPL 相等。本机 CUDA FP16 对该 Qwen 路径会产生 NaN，因此**正式质量计算一律 BF16**；
- ✅ **G2 — 8K × 1.5B operational**（chunked prefill、3 runs 稳定、无 OOM）→ **PASS**；
- ✅ **G3 — FP8 KV FROZEN**：正式方案 = **Asymmetric V-only FP8 KV**（K cache BF16 不量化 + V cache FP8 E4M3，per-token×per-head scale，group=128）。独立 40 窗（不与 pilot 重叠）结果：`ΔNLL +0.000397`、`relative ΔPPL +0.040%`；V2 生成前缀 32/32、V3 chunk 行为一致；8K KV cache 224.00 → 169.75 MiB（**1.32×**）。**诚实边界**：eager decode 无加速且有 V 反量化开销；整机峰值仅降 ~0.06 GiB；1.32× 仅指 KV cache 逻辑压缩；K-FP8 / full-FP8 为 negative ablation（ΔNLL ≥ +2.4，**限定 Qwen2.5-1.5B + 在线 max-abs E4M3 + eager attention，不外推**）；
- ✅ **G4-0 — draft 正确性**：**CONDITIONAL PASS（BF16 shape drift 已表征）**——OURS full==HF full、OURS cached==HF cached；HF 自身 full/cached 亦同处分歧；FP32 恢复一致 ⇒ 非引擎 bug、非不确定性。block-vs-serial 被测 argmax 一致、cache 终长一致、**无 off-by-one**；但 cache 数值差 max\|Δ\|≈2.0 → exact-id 门保留为风险；
- ✅ **G4-B3 — 双模型 8GB 可行性**：**PASS 4/4**（1.5B+0.5B 同驻 bf16；ctx512 alloc 4.34/4.336 GiB；ctx8192 alloc 6.16 GiB、resv 8.92 GiB；无 OOM、finite、3 轮稳定）；
- ✅ **G4-B4 — cache 操作 API**：**PASS**（用户实现 `spec_cache.py`：`cache_length` / `trim_cache` / `check_consistent`；base 与 v_fp8 两档全 True）；
- ✅ **G4-B5 — greedy SpecDec 主循环**：**PASS**（用户实现 `spec_dec.py`；T1–T4 单测 + 真实投机 smoke：rounds 8 / proposed 27 / accepted 11 / mismatch 5 / all_accept 3；每轮不变量 `tgt_len==drf_len==pre+(m+1 or Kp)` 全成立；`target_block_calls` 8、`draft_calls` 27 实测）。**EOS 修复后已 GPU 复验**（两档逐轮 trace 全 `ok`，B5 记录对应当前代码）；另登记结构性恒等式 `gen_len == accepted + mismatch_rounds`；
- ✅ **G4-B6 — correctness gate（两档并列登记）**：
  - **B6-fp32 = 实现正确性证据 → PASS**：10 case × 2 KV 档 = **20/20 组合逐 token id 全等**（CPU/float32，参照 = 同引擎 cached AR；总 216/348；finite、确定性、非空过、case 与档位跑满）。判据在运行前入库；"dtype 不是被测对象"已明确；
  - **B6-bf16 = 性能档行为 → CHECK 19/20**：唯一分歧 `natural-2@7` 经实测归因为 **bf16 数值不可判定位置**（候选对精确并列 22.8750；signal 0.1491 ≤ noise 0.1491），fp32 真值 = spec 侧 token。**严格口径：不设例外、不放宽 exact-id 门**，该记录原样保留并作为已知数值限制披露；
- ✅ **EOS 路径 bug 修复**：`_propose` 的 EOS 检查位置导致 draft cache 少 1，全接受分支 `trim_cache` 越界（生成以 EOS 正常收尾时必踩）→ 已修 + 纯逻辑回归测试（含旧实现对照复现 ValueError）；
- **后续正式实验**：先完成上方数值与稳定性验证，再推进 ≥16 prompt families（2K⊂4K⊂8K）的 pilot/final 与 break-even 分析；G5 instrumentation 和 Part 4 探索测量已有实现与结果，不再列为未启动；
- 测量纪律：只记录实测（数据见 `docs/data_ledger.md`）；跨脚本显存数字不作比较；进度与决策变更走 amendment/日志而非改写已冻结方案。

## 项目简介

**在做什么**：从零手写 Qwen2 架构推理引擎（RMSNorm / RoPE / GQA attention /
SwiGLU / KV-cache 解码 / 贪心生成），目前沿 **KV-cache → V-only FP8 KV
压缩 → 投机解码** 进行实现与验证，用可复现测量表回答研究问题：
> 投机解码在 8GB 小卡上随上下文变长到哪里失效？KV 量化能否延长它的获利区间？

**意义**：课程要求"原创 + hands-on LLM system"——我们交付自写引擎 + 质量×成本
测量表（decode tokens/s / 显存 / 接受率 / ppl），而不是跑教程报一个数字；结果
可作为小模型部署的决策表；亲手实现 transformer 与推理引擎也正是 AI Infra
方向的核心技能。

**技术栈**：Python + PyTorch 2.6（cu124）；自写引擎 `specdec/engine/model_self.py`（OURS）；
HuggingFace 用于权重加载、tokenizer 与参考实现（LIB）；自写缓存、V-only FP8 量化及 greedy 投机循环。
新增的可选 prefill 融合路径通过 PyTorch 调用 cuDNN attention（LIB）。
非 GEMM 算子（RMSNorm、RoPE、SwiGLU、causal GQA）另有 CUDA / Triton 实现，用法和一条不加载权重的样例在 [`specdec/kernels/README.md`](specdec/kernels/README.md)。
int8/int4 权重量化仍不在当前推进范围；测量遵循 G5 协议并落盘保存真实数据。

**为什么这么做**：调 API/库只能看整机表现，研究问题（缓存、量化、投机解码的收益）
需要引擎内部控制权（缓存布局、解码循环、计时插桩）——这是自建引擎的必要性而非炫技；
"从零"配上对齐基线（0.000）才让每个结论可复现、可归因；同时遵守风险分层纪律
（只承诺纯 PyTorch MUST，高方差项为加分轨）。

## 快速开始

```powershell
# 1) 建环境（方式 A：conda / 方式 B：venv）
conda create -n specdec python=3.11 -y ; conda activate specdec
# 或：
py -3.11 -m venv .venv ; .\.venv\Scripts\Activate.ps1

# 2) 安装依赖（必须装 CUDA 版 —— CPU wheel 会让 cuda=False，静默失败）
pip install -U pip
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-local.txt
pip install -e .

# 3) 验证机器（第 0 步门）
python -m specdec.cli env-check
```
预期输出：`cuda available: True`、`NVIDIA GeForce RTX 4060 Laptop GPU`、显存约 8GiB。

> 若 HuggingFace 下载慢/失败，先执行：`set HF_HOME=D:\hf-cache` 和
> `set HF_ENDPOINT=https://hf-mirror.com`，再跑下面的验收命令。
> Windows 说明：官方 `triton` 没有 Windows 版——kernel 轨（L3）在 WSL2 里做，
> 或按 go/no-go 规则保持 eager（详见 `docs/plan_v6.md`）。

## 验收命令（仓库根目录 + venv 已激活）

```powershell
python -m specdec.engine.align_ours          # L1 门 —— 预期 0.000，PASS
python -m specdec.engine.test_cache_equiv    # L2 缓存==全量 —— 预期 < 1e-3，PASS
python -m specdec.engine.align_generate      # L2 生成对齐 —— 预期 PASS
python -m specdec.engine.bench_basic         # 性能基线（CPU，有 GPU 则含 GPU）
```

G4 门（投机解码；GPU 脚本请在交互式终端串行运行）：

```powershell
python -m specdec.engine.g4_b3_dual_model_feasibility   # B3 双模型 8GB 可行性 —— PASS 4/4
python -m specdec.engine.g4_b4_cache_tests              # B4 cache API —— PASS（base + v_fp8）
python -m specdec.engine.g4_b5_helper_tests             # B5 辅助函数 T1–T4 —— PASS
python -m specdec.engine.g4_b5_smoke                    # B5 主循环 smoke —— PASS
python -m specdec.engine.g4_b5_eos_branch_tests         # EOS 分支回归（纯逻辑 CPU，秒级）—— PASS
python -m specdec.engine.g4_b6_correctness              # B6-bf16 exact-id 门 —— CHECK 19/20（已归因）
python -m specdec.engine.g4_b6_fp32_correctness         # B6-fp32 实现正确性门（CPU/float32）—— PASS 20/20
python -m specdec.engine.g4_b6_tie_diag                 # 分歧位置的 signal-vs-noise 归因
```

`g4_b6_fp32_correctness` 支持 `set B6F_LIMIT=N` 跑小样（仅验证环境/耗时；**小样永远不会打印 PASS**）。设 `set HF_HUB_OFFLINE=1` 可避免 tokenizer 联网抖动。

## 讲解文档（投机解码）

- 📄 **`docs/pdf/specdec_speculative_decoding_explained.pdf`**（19 页）— **投机解码：原理、代码与实现细节**。八节：①原理（为什么 decode 慢、凭什么省、为什么 greedy 下输出不变）②一次生成的全景数据流（含真实轮次演算与 8 轮真实 trace）③逐行讲 `spec_dec.py`（每个输入/输出张量的形状/dtype/含义，每个 PyTorch 操作在算什么）④引擎侧 cache 与 FP8 量化算子逐行 ⑤逐行讲 `spec_cache.py` 与契约 ⑥正确性证据（两档门 + "差一个 token"案例的完整归因 + 三个真实 bug）⑦常见误解 Q&A ⑧速查表（形状/不变量/统计字段/命令）。
- 源文件 `docs/specdec_explained.md`（可 diff、可评审）；重新生成：`python scripts/generate_specdec_explained_pdf.py`（需要 `reportlab`，见 `requirements-local.txt`）。

## 文档

- [合作者 PR #2 / #3 评审与隔离研究归档](docs/collaborator_pr_review_2026-10-02.md) — 保留 KV 压缩、two-pass attention 和性能归因成果；未整包合并旧引擎，不改变现有默认或正式验收。
- `docs/plan_v7_1_protocol_freeze.md` — 实验协议冻结（v7.1 + §0 **Amendment v7.2**：Compute=BF16、KV 因子、12 conditions）
- `docs/plan_v7.md` — v7 方案（方向、承诺、裁剪序）；`docs/plan_v6.md` — v6/v6.1 历史方案
- `docs/data_ledger.md` — 实测数据台账（§3.4 G1、§3.5.x FP8 KV、§4.5.1 G2、§4.5.2 B3、§4.5.3 B4、§4.5.4 B5、§4.5.5 B6-bf16、§4.5.6 B6-fp32）
- `docs/status_note_fp8_v4.md` — FP8 KV：根因、修复、正式判定、G4 前置清单
- `docs/design_freeze_fp8_kv.md` — FP8 KV 设计冻结（v1.0→v1.3）
- `docs/design_freeze_g4_specdec.md` — G4 投机解码协议（v0.3：logits 对齐、回滚语义、`_propose` 出口不变量、EOS 规则）
- `docs/design_g4_cache_api.md` — cache 接口契约；`docs/plan_g4_next_steps.md` — G4 七阶段规划与状态
- `docs/specdec_explained.md` + `docs/pdf/specdec_speculative_decoding_explained.pdf` — 投机解码原理/代码/实现细节讲解（19 页）
- `docs/daily_report_2026-09-15b_G4_B5B6_EOSfix.md` — 本日日报（B5/B6/EOS 修复）
- `docs/daily_report_2026-09-15.md`、`docs/daily_report_2026-09-08b_G1G2G3_FP8KV.md` — 阶段归档日报
- `docs/stage_report_l1_l2.md`、`docs/L1_code_teaching.md`、`docs/debug_lesson_alignment.md` — 阶段与教学文档

## 包结构

```
specdec/
  cli.py          CLI：env-check、run
  config.py       RunConfig + config_hash / run_id（复现哈希链）
  results.py      只追加 results.jsonl 台账
  engine/         model_self.py（OURS 自写）+ 对齐/测试/基准工具
  kernels/        CUDA / Triton：RMSNorm、RoPE、SwiGLU、causal GQA（见 kernels/README.md）
  quant/          Phase 4：int8/int4（OURS）
  kv_quant/       Phase 4b：KV 量化（OURS）
  spec_dec/       Phase 5：投机解码（OURS）
  bench/          Phase 7：基准测试
```

诚实性规则：`results.jsonl` 只追加真实测量数据，绝不写假行。引擎、缓存与投机循环由团队
编写，底层张量运算依赖 PyTorch；权重加载、tokenizer、参考实现及可选 cuDNN 融合 attention
使用第三方库。历史 eager 与新后端的实验口径必须分别记录，不混表、不把诊断数据当正式结论。
