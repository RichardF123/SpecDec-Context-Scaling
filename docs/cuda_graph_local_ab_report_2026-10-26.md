# CUDA Graph 局部 A/B 报告：定位 SpecDec 剩余执行开销（2026-10-26）

> `experiment_class = operator_mechanism_diagnostic`；`formal_result_allowed=false`；
> `quality_claim_allowed=false`；`full_specdec_speedup_claim_allowed=false`。
> 局部算子/固定 shape 的 eager vs CUDA Graph A/B，不是完整 CUDA Graph SpecDec、不是生产实现、
> 不是 EAGLE 实验，也不得把局部结果称为整机加速。
> 基线 `codex/relaxed-specdec` @ `732f1fe`；未改正式 runner / strict / relaxed / bonus oracle 与历史数据。

## 0. Graph-safety 审计（`docs/cuda_graph_local_ab_design_audit_2026-10-26.md`）

逐项结论：
1. **CPU 同步**：`_run` 的 decode/propose/verify 路径无 `.item()/.tolist()`；Python 分支只依赖 capture 时的值
   （`start_pos = cache_len`、`decide_attention_path(phase,s)`）⇒ 固定 shape 下可 capture。
2. **capture 期分配**：`torch.cat`（KV append）、`Build_Rope_cache`、`torch.triu` mask 均是新分配，但走 graph
   私有 pool，capture 允许（本机实测通过）。
3. **地址固定**：静态 token / 初始 cache 由本实验预分配；capture 返回的输出张量地址在 pool 内稳定。
4. **KV 写入方式**：`cache["k"] = torch.cat(...)` 是"原地改 dict、新张量"——Python 赋值只在 capture 时执行一次，
   replay 只重放 kernel；本实验以"捕获返回的 tensor"为读出口，并用 input-update 测试防"重放旧输入"。
5. **RoPE/mask/position**：全部在 capture 时固化 ⇒ **一个 graph 只对一个 (history length, s, position) 有效**；
   本实验对每个 combo 单独 capture（Stage G0/G1 范围）。
6. **Attention backend**：decode/verify 走 eager（matmul+softmax），cuDNN 只用于 prefill ⇒ graph-safe。
7. **Windows WDDM + torch 2.6/cu124**：可 capture；本项目既有 16 步 chain capture 记录为证。
8. **重放旧 cache 风险**：通过 input-update 与 determinism 测试排除（见 gate）。
9. **固定历史长度可安全重放**：可以；**跨历史长度不可复用**（已在审计中记录为集成阻塞点）。
10. **显存**：graph pool 实测增量中位 −70 MiB（多数情况反而释放了临时缓存），最大 +24 MiB；
    peak_reserved 指标在本机 WDDM 下不可靠（见局限）。

**实测发现的额外阻塞点（已修复，不绕过）**：`target_rope_scope`（CPU-FP32 inv_freq）在每次 `_run` 内做
`inv_cpu.to(device)`（阻塞 H2D），capture 期间报 `operation not permitted when stream is capturing`。
修复方式：capture 前用同一公式（`build_with_cpu_invfreq`，同一 inv_freq）预生成 `(seq_len, start_pos)` 的 cos/sin
表，capture 期间把 `Build_Rope_cache` 临时替换为"返回预生成表"的闭包（数值同路径、去 H2D）；
draft 走原 GPU 实现无需此补丁。

## 1. 覆盖度与实现

- 组件：A1 draft s=1；A2 4 步 candidate chain；A3 bonus catch-up + 4 步 chain；B1 target M=4；B2 target M=5。
- 组合：families `F09/F04` × contexts `2048/4096/8192` × model（draft 或 target）→ **12 个 combo / 30 个组件实例**，
  每实例 5 trials（每 trial A1=60、A2=20、A3=16、B1=30、B2=24 次重复）+ 10 次 warm-up，CUDA Event 计时。
- 每个进程**只加载一个模型**（draft 或 target），且 **先测 eager、再 capture**：
  早期两模型版本（`output/cuda_graph_local/run`、`run2`）出现 8K 显存压力和"先 capture 再测 eager"导致 eager
  膨胀 2–3×，其证据保留；最终数据用 `output/cuda_graph_local/run3`（eager-first + one-model）。
- 输入 token 全部来自真实 bonus oracle 轨迹（`output/bonus_accept_oracle/run/*_bonus/result.json` 的 control）；
  cache 来自真实 chunked prefill（chunk=2048）。
- 代码：`experiments/cuda_graph_local/{local_ops,fixtures,draft_graph,verify_graph,gate_gpu,benchmark,analyze,rope_tables}.py`。

## 2. 正确性 gate

**CUDA_GRAPH_LOCAL_GATE: PASS**（30/30 组件实例）：
- logits **bitwise 全等 30/30**；argmax 全等；shape 全等；
- cache **bitwise 全等 30/30**（逐层 K/V、dtype、长度、finite、`check_consistent`）；
- replay determinism：同输入两次 replay 一致；
- **input-update**：换最后一个真实 token 后 replay 输出改变，且与 eager 在新 token 上 bitwise 一致
  （证明 graph 未重放旧输入）；
- M=4/M=5 位置语义：verify 张量形状 `(1,M,V)` 与 eager 一致；draft chain 逐 step 输入 token 与 eager 相同；
- 无 argmax 变化、无 cache 错位；无 gate 失败。

## 3. 结果（每组件中位，eager / graph kernel-only / graph+static copy，ms；两 family 中位）

### Draft proposal（0.5B，历史 = ctx）

| ctx | A1 s=1 | A2 4-step | A3 bonus+4-step | A1 加速 | A2 加速 | A3 加速 |
| --- | --- | --- | --- | --- | --- | --- |
| 2K | 43.1 / 7.6 / 7.6 | 177.3 / 29.7 / 29.7 | 223.7 / 37.1 / 37.1 | 5.7× | 5.97× | **6.03×** |
| 4K | 42.3 / 8.8 / 8.8 | 171.4 / 35.0 / 35.0 | 215.7 / 43.9 / 44.0 | 4.8× | 4.9× | 4.91× |
| 8K | 42.6 / 11.6 / 11.6 | 171.3 / 46.5 / 46.5 | 215.8 / 58.2 / 58.2 | 3.7× | 3.69× | 3.71× |

- 单步 draft 在 **eager 42–43 ms** vs **graph 7.6–11.6 ms** ⇒ 每步约 **30–35 ms 是 host/launch 开销**
  （GPU 实际计算 ~7.6–11.6 ms），**draft s=1 是 launch-bound**。
- 4/5-step chain 的加速与单步近似（5–6× @2K），说明 chain 没有引入额外 host 开销；
  A3 比 A2 多 1 步 catch-up，额外成本 ≈ 单步成本（图内 +7.3 ms，eager +46 ms）。

### Target verification（1.5B，历史 = ctx）

| ctx | B1 M=4 | B2 M=5 | B1 加速 | B2 加速 | M=5 相对 M=4（graph） |
| --- | --- | --- | --- | --- | --- |
| 2K | 54.6 / 20.7 / 20.7 | 55.1 / 21.2 / 21.2 | 2.64× | 2.60× | +2.4% |
| 4K | 55.9 / 24.3 / 24.3 | 56.2 / 25.2 / 25.2 | 2.30× | 2.24× | +3.7% |
| 8K | 50.7 / 33.6 / 33.7 | 51.1 / 37.5 / 37.5 | 1.51× | 1.36× | +11.6% |

- verify 也明显受 launch 影响（2.2–2.6× @2K/4K），但 8K 时 GPU 计算（长历史 attention）占比上升 ⇒ 只有 1.4–1.5×。
- M=5 相对 M=4 在 graph 下仅 +2–12%，eager 下 +1–2%：bonus 契约的 verify 额外成本很小。

### kernel-only vs effective（含 static copy / cache copy-in）

- static token copy：0.01–0.5 ms/次；**graph+copy 与 kernel-only 的差异 <2%**。
- 真实动态 cache → 静态 buffer 的 copy-in（clone，作为集成成本单独测量）：target 0.79/1.16/2.15 ms，
  draft 0.67/0.75/1.02 ms（2K/4K/8K）。即使每轮都做一次，也仅占 round 时间（~200 ms）的 1–2%。
  ⇒ **不存在"kernel-only 快、加入 copy 后收益消失"的情形**。

### 资源

- capture 时间：中位 **355 ms**（273–1101 ms；chain 图更久；A1 ~0.3 s、A3 ~1.0 s）。
- 首次 replay：中位 **29.9 ms**（含 graph 初始化）；稳态 replay 即上表 graph 时间。
- graph pool 增量：中位 **−70 MiB**、最大 **+24 MiB**（多数情况把临时 tensor 计入池后反而低于 eager 峰值）。
- 显存结论的限定：**单个固定 shape Graph 在本次进程中成功 capture**；**完整 13 轮 Graph 的总显存尚未测量**。
- 局限：`peak_reserved` 在本机 WDDM 下报出 >8 GiB 的异常值，**不作为显存结论**；以 pool 增量与审计§10 估算为准。

## 4. Amdahl 投影（归一化口径）

真实 per-round 时间取自 bonus oracle 实测（decode/13：200.2/196.5/196.2 ms），微基准只提供
"draft/verify 占比"与 graph 加速比（避免把 8K 微基准的绝对 eager 值直接外推）：

| ctx | draft 占比 | draft 加速 | verify 加速 | 观察 speedup | **optimistic** | **realistic** | **conservative** | 跨 1 所需降低 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2K | 0.80 | 6.03× | 2.60× | 0.912 | 4.36 | **4.36** | 4.20 | 8.8% |
| 4K | 0.79 | 4.91× | 2.23× | 0.926 | 3.64 | **3.64** | 3.55 | 7.4% |
| 8K | 0.81 | 3.71× | 1.35× | 0.874 | 2.44 | **2.43** | 2.34 | 12.6% |

- 投影全部 **>1**，且 conservative 仍为 2.3–4.2：这是 **optimistic fixed-shape local upper bound**——
  若局部 graph 能以稳态 replay 的方式覆盖每轮 draft chain + verify，bonus-aware 结构有空间跨过 AR；
  但它**不是** realistic 完整路径预测。
- **必须同时报告的反向证据**：微基准 eager 的 round 合计（A3+B2 ≈ 267–279 ms）比真实 bonus round（~196–200 ms）
  **更慢**（residual 为负）。说明生产循环里 host launch 与 GPU 有部分重叠；因此上面的 2.4–4.4× 是**上界**，
  真实集成收益会低于该投影。结合"eager 每步 42–55 ms、graph 7–12/21–38 ms"的事实，固定 shape 局部
  draft/verify 存在显著 launch 开销；**该收益能否在动态完整循环中兑现尚未验证**。
- **集成阻塞（关键）**：bonus 循环每轮历史 T 都不同 ⇒ 每个 T 单独 capture；13 轮 × capture 0.3–1.1 s
  ≈ 4–14 s ≫ decode 2.5 s ⇒ **"每轮重 capture"不可行**。要兑现 projection，必须改成
  **静态容量缓冲（预分配 KV + padding 注意力 + 预生成 RoPE 表）**，或按固定 shape 分段重捕获（不现实）。
  静态容量路径会改变 attention 的归约形状，必须先用本轮 gate（fixture token 级 bitwise）验证，未验证前不得宣称。

## 5. 结果分类

**Outcome A-local（end-to-end integration unverified）**：
- gate 通过；kernel-only 与 effective 几乎相同（copy 可忽略）；
- realistic 投影 2.43–4.36× >1，conservative 2.34–4.20 也 >1；
- 显存增量可忽略（graph pool 中位 −70 MiB）；
- 结论：**固定 shape 的局部 draft/verify 存在显著 launch 开销**（draft 单步 42 ms 中约 34 ms 可被 graph 消除），
  值得做局部 graph 集成；但**该收益能否在真实双模型、动态 13 轮 cache 推进的完整循环中兑现尚未验证**，
  需单独做端到端 2K 可行性门。

## 6. 报告必须回答的 10 个问题

1. **当前 cache 是否 graph-safe？** 固定历史长度下 **是**（30/30 组件 bitwise 通过，含 cache bitwise）；
   但 CPU-FP32 inv_freq 的 H2D 必须先预生成 RoPE 表；跨历史长度不可复用 ⇒ 完整动态循环目前**不安全**。
2. **Draft s=1 是否 launch-bound？** **是**：eager 42–43 ms、graph 7.6–11.6 ms ⇒ 约 30–35 ms/步是 host 开销。
3. **Draft 4/5-step chain 减少多少？** 4-step 4.8–6.0×；5-step（bonus+4）3.7–6.0×（2K 6.03×、4K 4.91×、8K 3.71×）。
4. **Target M=4/M=5 graph 减少多少？** 2K 2.6×、4K 2.2–2.3×、8K 1.4–1.5×；M=5 只比 M=4 贵 2–12%。
5. **kernel-only 与 effective 差多少？** <2%（static copy 0.01–0.5 ms；cache copy-in 0.7–2.2 ms/轮，约占 round 1–2%）。
6. **Graph 额外显存？** pool 中位 −70 MiB，最大 +24 MiB（peak_reserved 指标异常，不作为结论）。
7. **哪个组件最值得集成？** **Draft chain**（绝对时间最大：2K 每轮 224 → 37 ms；verify 仅 55 → 21 ms）。
8. **Amdahl 能否跨 AR？** 稳态投影能（2.4–4.4×，保守 2.3–4.2×），但这是上界；生产 per-round 已部分重叠 host，
   实际收益低于投影；且必须解决动态 T 的静态化才能兑现。
9. **属于哪类？** **Outcome A**（局部 graph 有效且投影 >1；值得集成，但其前提是 cache 重构）。
10. **下一步？** **重构 cache（静态容量 + padding + 预生成 RoPE）再做完整 graph 集成**，不是直接"每轮 capture"，
    也不是立刻转 EAGLE；EAGLE 是 padding 破坏 token 一致性时的备选。

## 7. 局限

- 局部算子/固定 shape 口径；Amdahl 是**上界式**投影（未含 capture 摊销与动态 T 的额外成本）。
- 早期两模型版本（`run`、`run2`）被 8K 显存压力与 capture-first 顺序污染，已保留证据并弃用；
  最终数据为 one-model + eager-first 的 `run3`。
- 仅 2 个开发 family × 3 contexts；未跑正式 family；无 GPU trace（可选步骤，Windows 稳定性不确定，未做）。
- peak_reserved 指标异常需后续单独核查；本报告不使用它做显存结论。
- 没有修改任何 acceptance 规则/bonus 契约；strict/relaxed/bonus oracle 的历史结果未被覆盖。

## 8. 产物

- 设计审计：`docs/cuda_graph_local_ab_design_audit_2026-10-26.md`
- 代码：`experiments/cuda_graph_local/{local_ops,fixtures,draft_graph,verify_graph,gate_gpu,benchmark,analyze,rope_tables}.py`
- 原始数据：`output/cuda_graph_local/run3/{<fam>_<ctx>_<model>/result.json,*.log}`（12 combo、30 组件）
- 聚合：`output/cuda_graph_local/run3/analysis/{components.csv,amdahl.csv,cuda_graph_local_summary.json}`
  （副本 `docs/measurements/cuda_graph_local_{components,amdahl,summary}_2026-10-26.*`）
- 图：`docs/figures/cuda_graph_local/local_ab.{png,pdf,svg}`（A3/B2 各自 eager→graph kernel→graph+copy、
  Amdahl 三档投影 vs y=1、各组件 kernel/effective 加速比）
- 复现：`gate_gpu` 由 `benchmark` 内联执行；`python -m experiments.cuda_graph_local.benchmark --out ...`
  → `python -m experiments.cuda_graph_local.analyze --run ...`


---

## 9. 结论修订（end-to-end 2K 可行性门之前，2026-10-27）

本报告在端到端 2K 兑现门（`docs/cuda_graph_e2e_2k_report_2026-10-27.md`）之前先做措辞修订，
原始数据/CSV/图**未改动**。修订要点：

1. **Outcome 限定**：本节结论为 **Outcome A-local / end-to-end integration unverified**。
2. **2.4–4.4× 的性质**：它是 **optimistic fixed-shape local upper bound**，**不得**称为 realistic 完整路径预测。
3. **负 residual**：局部组件时间相加（A3+B2 ≈ 267–279 ms/round）**大于**真实 bonus round（≈196–200 ms），
   raw residual < 0 ⇒ 局部时间**不可直接按 Amdahl 相加**推断整机收益。
4. **瓶颈表述**：从"完整 SpecDec 主要是 launch-bound"改为：**固定 shape 的局部 draft/verify 存在显著 launch
   开销；该收益能否在动态完整循环中兑现尚未验证**。
5. **显存表述**：单个固定 shape Graph 本次成功 capture；**完整 13 轮 Graph 的总显存尚未测量**。
6. **保留上下文发现**：拆掉 launch 后的真实时间随上下文增长（Draft A3 37.1→44.0→58.2 ms；Verify M5
   21.2→25.2→37.5 ms @2K/4K/8K），Graph 相对收益随上下文增大而下降；解释为短上下文主要受 launch 影响、
   长上下文真实 attention/KV 成本占比提高。该解释只适用于已测量的局部组件，不自动外推整机。
