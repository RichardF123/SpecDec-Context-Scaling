# 2K 端到端 CUDA Graph 兑现门报告（2026-10-27）

> `experiment_class = end_to_end_graph_feasibility_diagnostic`；`formal_result_allowed=false`；
> `quality_claim_allowed=false`；`production_claim_allowed=false`；`scope = F09/2K/bonus-all-accept/64-token`。
> 只回答：固定 shape 局部 graph 的收益能否在真实双模型、真实 13 轮 cache 推进、完整 64-token 循环中兑现，
> 并使 2K 稳定超过 AR。**不扩展 4K/8K，不跑正式 family，不接 EAGLE。**
> 基线 `codex/relaxed-specdec` @ `f2ddb22`。

## 0. 先修的措辞（已落实到 `docs/cuda_graph_local_ab_report_2026-10-26.md` §9）

1. 原 **Outcome A** 改为 **Outcome A-local（end-to-end integration unverified）**；
2. 2.4–4.4× 标为 **optimistic fixed-shape local upper bound**，不是 realistic 完整路径预测；
3. 明确：局部组件时间相加（A3+B2 ≈267–279 ms/round）**大于**真实 bonus round（≈196–200 ms），**raw residual < 0**，
   局部时间不可直接按 Amdahl 相加；
4. "完整 SpecDec 主要是 launch-bound"改为"固定 shape 的局部 draft/verify 存在显著 launch 开销；
   该收益能否在动态完整循环中兑现尚未验证"；
5. 显存结论改为：单个固定 shape Graph 本次成功 capture；**完整 13 轮 Graph 的总显存尚未测量**；
6. 保留上下文发现（Graph 后 Draft A3 37.1→44.0→58.2 ms、Verify M5 21.2→25.2→37.5 ms，收益随上下文下降）。
原始数据/CSV/图未改动。

## 1. 设计审计与路线选择

见 `docs/cuda_graph_e2e_2k_design_audit_2026-10-27.md`（13 问）。结论：**走路线 A（13 个固定位置 Graph）**：
bonus-aware 的 13 轮各自 (history, s) 不同（round0: draft 4 步/target s=4；round1..12: draft 5 步/target s=5），
当前 graph 只能绑定固定 (cache 地址、history、position、RoPE 表、mask/shape)，因此需要 **13 个 draft 链 graph +
13 个 verify graph = 26 个 graph** 一次 capture、全部存活到测量结束。
路线 B（静态容量 + padding 的 KV 原型）需要改写 `model_self` 的 attention/mask 才能按 valid length 归约，
超出本轮 scope；按任务要求**不扩大 scope**，先按路线 A 实测。

## 2. 实现（本轮新增）

`experiments/cuda_graph_e2e/`：
- `design.py`：13 轮表（draft 起点 [2048,2052,…,2107]、steps 4/5、verify s 4/5、gen 位置、容量 2116）、
  amortized / break-even 公式；`check_schedule()` 自检（13 轮、52 candidates、12 bonus、64 token）。
- `graph_state.py`：**单组大容量静态 KV buffer**（target/draft 各 2116 tokens，bf16）+ 静态 token/gen/cand/bonus 缓冲；
  `write_slice` 把每步 cat 输出在图内 `copy_` 回 buffer 的 [L:L+s] 段。
- `capture_rounds.py`：为每轮 capture draft 链与 verify graph（图内 GPU argmax→embedding、图内 slice copy_、
  verify 的 bonus=argmax(最后一行) 写入静态 bonus buffer；不保留 lg/cache 引用以省显存）；
  固定 13 轮顺序 replay（Python 只做调用，token/cache 全由 GPU 产生）；verify 输入 [pending bonus]+candidates 用
  GPU `copy_` 拼装（图外，单独计时）。
- `gate_gpu.py`：与 eager bonus oracle 的逐项对齐门（64 tokens、13 轮、逐轮 candidate/bonus、cache bitwise、
  determinism、finite、verify 输入长度）。
- `benchmark_2k.py`：同批三模式（AR / eager bonus / graph）子进程；graph 模式含显存账本、capture 计时、
  first-request、reset 成本、assemble 成本、5 次 measured replay。
- `test_cpu.py`：**E2E_CPU: PASS 5 / 5**（轮次表、gen 位置、摊销、break-even）。
- 复用：`rope_tables.py`（target CPU-FP32 inv_freq 的 H2D 在 capture 期间非法 ⇒ 预生成 13 组 (s,start) RoPE 表）。

## 3. 结果：路线 A 被显存阻断（Outcome C）

**capture 逐步显存账本**（`output/cuda_graph_e2e/mem_probe/probe.json`，同一进程先加载 target+draft 并 prefill）：

| 阶段 | reserved (GiB) | 增量 | 物理可用 |
| --- | --- | --- | --- |
| prefill + big buffers 后 | 4.93 | — | 1.97 GiB |
| chain0 | 5.75 | +1.61 | 1.09 |
| verify0 | 6.61 | +0.88 | 0.24 |
| chain1 | 8.59 | +2.00 | 0.00 |
| verify1 | 9.46 | +0.89 | 0.00 |
| chain2 | 11.42 | +1.99 | 0.00 |
| verify2 | 12.29 | +0.89 | 0.00 |
| chain3 | 14.27 | +2.00 | 0.00 |
| verify3 | 15.14 | +0.89 | 0.00 |
| chain4 | **OOM** | — | — |

- **capture 8/26 个 graph 后 OOM**（第 5 个 round 的 chain）；
- 每个 **chain graph 保留 ≈1.6–2.0 GiB**、每个 **verify graph ≈0.88–0.89 GiB**；
  外推 26 个 graph ≈ **13×2.0 + 13×0.89 ≈ 37.6 GiB** 保留内存，超过 8 GiB 物理显存 4 倍以上；
- 已尝试：单 pool 共享（更早失败）、每 graph 独立 pool（上表）、不保留每 graph 的 lg/cache 输出引用、
  `expandable_segments:True`、一个进程只加载所需两模型 —— 均无法让 26 个 graph 共存；
- 因此**端到端 graph 路径无法在 8 GB 内运行完整 13 轮**，gate/replay/性能测量都无法进行。
  按任务约定：**Outcome C（memory-blocked）**，不提高显存护栏、不改模型路径。

**Gate 状态**：`E2E_GRAPH_2K_GATE: BLOCKED(memory)` —— 不是数值/正确性失败（未进入任何 replay，
不存在 token/cache 分歧）；是 capture 阶段的显存不足。失败证据保留在
`output/cuda_graph_e2e/run/graph.log`、`run/graph/failure.json`、`mem_probe/probe.json`。

## 4. 同批基线（AR / eager bonus，5 次 measured）

| 模式 | capture (s) | decode median (s) | min | max | speedup vs AR | 输出长度 |
| --- | --- | --- | --- | --- | --- | --- |
| AR（target-only） | — | **2.407** | 2.381 | 2.533 | 1.000 | 64 |
| eager bonus all-accept | — | **2.691** | 2.667 | 2.727 | **0.894** | 64 |
| graph replay | — | 被阻断 | — | — | — | — |

eager bonus 计数与 oracle 契约一致：rounds=13、candidate=52、bonus=12、draft calls=64、target block calls=13、
rollback/repair/mismatch=0、verify 首轮 4 / 后续 5。AR 与 eager 的输出长度均为 64（无 EOS）。
（注：本批 AR 2.407 s、eager 2.691 s，比上一批 2.374/2.603 s 略慢，属环境/时钟波动；speedup 0.894 与上一批 0.912 同向。）

**摊销（请求数 N）**：Graph 无法运行，因此**不计算/不报告虚假摊销收益**；first-request 与 break-even 请求数
在 Graph 可运行前均无定义。capture 成本已知下限：即使按局部 A/B 的 capture 时间（0.27–1.1 s/graph），
26 个 graph 的 capture 也达 ~7–29 s；而显存才是硬阻断。

## 5. 显存账本（同口径）

- PyTorch allocated/reserved、`mem_get_info`（物理 free/total）已按阶段记录在
  `output/cuda_graph_e2e/run/graph.log` 与 `docs/measurements/cuda_graph_e2e_2k_memory_probe_2026-10-27.csv`；
- 关键事实：prefill 后 reserved 4.93 GiB、phys_free 1.97 GiB；**每个 graph 的池化分配不会跨 graph 释放**，
  因此 graph 数量与显存线性相关（≈2.0/0.89 GiB per graph）；
- 本机 `allocated 14.65 GiB` 的报错文本在 8 GB 卡上不可能成立，属 WDDM/overcommit 口径异常；
  判定依据是**物理 free 归零 + OOM**，不是 reserved 数值。

## 6. Outcome

**Outcome C：13 轮 Graph 无法在 8 GB 内共存（memory-blocked）**。
- 这是路线 A 的结构性限制：**"每轮 history 不同 ⇒ 每轮一个固定 shape graph" 的代价是 26 个 graph × 各自保留的
  graph pool，≈38 GiB**，远超 8 GB；
- 固定 shape 局部 graph 的收益（draft 3.7–6×、verify 1.4–2.6×）是真实的，但**无法在完整动态 cache 推进的
  13 轮循环中兑现**；这不是数值问题，而是"graph 只能绑定固定 shape"与"cache 每轮增长"冲突的直接后果；
- 路线 B（静态容量 + padding + 单套 graph）理论上可把 graph 数降到 2 个（draft/verify 各一），但需要改写
  attention/mask 以按 valid length 归约——**超出本轮 scope**，且必须另行做 token 级等价验证。

## 7. 对"系统效率 vs EAGLE"的结论

- Phase O（同步优化）≈4%；relaxed/bonus（提高 acceptance）→ 0.87–0.93 仍 <1；
- 局部固定 shape graph：draft 3.7–6×、verify 1.4–2.6×（bitwise 正确），但**端到端 13 轮需要 26 个 graph、
  ≈38 GiB 保留显存，8 GB 内不可共存** ⇒ 该路线在真实动态 cache 循环中**不可兑现**（Outcome C）；
- 因此：在当前 qwen2.5-1.5B/0.5B + 独立 draft + 动态 cache + 8 GB 约束下，**launch 优化不足以翻盘**；
  剩余差距是动态状态管理（每轮不同 history 的固定 shape 图）与真实 KV/attention 成本共同造成。
  按任务约定：**停止 Graph 路线，转向 EAGLE 可行性审计**。

## 8. 下一步唯一推荐动作

**转入 EAGLE 类 feature-level drafting 可行性审计**（单模型多步/轻量 head，消除每 token 一次 0.5B draft 前向的
结构成本）。若仍希望在 graph 方向推进，唯一允许的后续是路线 B 的"静态容量 + padding"原型（单套 2 个 graph），
但它需要模型 attention/mask 改写与 token 级等价门，属于新 scope，需另行批准。

## 9. 产物

- 修订后的局部 Graph 报告：`docs/cuda_graph_local_ab_report_2026-10-26.md`（§9 结论修订）
- 2K 端到端设计审计：`docs/cuda_graph_e2e_2k_design_audit_2026-10-27.md`
- 代码：`experiments/cuda_graph_e2e/{design,graph_state,capture_rounds,gate_gpu,benchmark_2k,analyze,test_cpu}.py`
- 数据：`output/cuda_graph_e2e/run/{ar,eager,graph,failure evidence}`、`output/cuda_graph_e2e/mem_probe/probe.json`
- 聚合：`docs/measurements/cuda_graph_e2e_2k_{core_table,memory_probe}_2026-10-27.csv`、
  `cuda_graph_e2e_2k_summary_2026-10-27.json`、`cuda_graph_e2e_2k_mem_probe_raw_2026-10-27.json`
- 图：`docs/figures/cuda_graph_e2e/e2e_2k.{png,pdf,svg}`（a 逐 graph capture 的 reserved 与 8 GiB 线；
  b AR / eager / graph(blocked) 的 decode 时间）
- 复现：`python -m experiments.cuda_graph_e2e.test_cpu`；`benchmark_2k --mode ar|eager`；
  `--mode graph`（在 8 GB 上会在 capture 第 9 个 graph 处 OOM，属预期 Outcome C）；
  `python -m experiments.cuda_graph_e2e.analyze --run output/cuda_graph_e2e/run`

## 10. 局限

- 只测 F09/2K/64-token；不覆盖 4K/8K/正式 family。
- Graph 路径未进入 replay，因此没有 graph 的 64-token 输出/token 对齐/性能数据（被显存阻断）；
  gate 状态为 BLOCKED(memory)，不得当作 PASS 或数值 FAIL。
- 本机 WDDM 的 allocated/reserved 报数异常，判定用物理 free 归零 + OOM。
- eager bonus 与 AR 为同批测量；graph 无法运行，speedup 表中 graph 行标注"blocked"。
