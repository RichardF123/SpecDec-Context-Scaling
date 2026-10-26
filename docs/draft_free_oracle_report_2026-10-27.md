# Draft-Free Perfect-Accept Bonus Oracle 报告（2026-10-27）

> `experiment_class = mechanism_diagnostic`；`deployable = false`；`quality_claim_allowed = false`；
> `formal_result_allowed = false`。本实验是**反事实 lower-bound/oracle**：不可部署、不作生成质量声明、
> 不进入正式 confirmatory 区间、**不是新的 SpecDec 方法**。
> 基线 `codex/relaxed-specdec` @ `e79b125`；不修改正式 runner / 正式结果 / perfect-accept / CUDA Graph 产物。

## 0. 结论先行

**Outcome A**：删除 draft decode 后，保留 target block verification + bonus emission + cache/控制流的
Draft-Free oracle 在 **2K/4K/8K 全部远超 AR**（S 中位 **4.46 / 4.24 / 4.21**），
而同批 perfect+bonus 仍为 0.92/0.87/0.86。
⇒ **在当前配置（1.5B target + 0.5B 独立 draft + k=4 + 动态 cache + eager decode）下，
"每 token 一次 0.5B draft forward" 是无法 break-even 的决定性因素**；
target verification / bonus / cache / 控制路径本身（零成本候选时）约为 AR 的 1/4.2 延迟。

## 1. 新代码改了什么

新增（独立诊断模块，不改正式 runner / 模型默认路径）：
- `experiments/mechanism/draft_free_oracle.py`
  - `candidate_layout(frozen_ids, k, max_new)`：按 bonus-aware 规则切块（每轮 4 candidate + 1 bonus，末轮无 bonus）。
  - `freeze_candidates(target, prompt)` / `freeze_via_ar(target, tokenizer, ids, job)`：计时外生成冻结候选
    （target AR greedy，64 token，EOS-aware；记录 sha1）。
  - `DraftCallGuard(draft)`：计时区间内包装 `draft._run`，一旦被调用即计数并抛错（硬失败）。
  - `decode_draft_free(...)`：target 真实 verify（block，s=4/5）+ KV 更新 + bonus emission + round accounting +
    stop 控制；**不调用 draft**；bonus 按 forced-accept 契约取冻结序列下一 token（同时记录 target argmax
    与冻结 token 的不一致数）。
- `experiments/mechanism/run_draft_free_oracle.py`：子进程 runner（每 (family,ctx,mode) 独立进程、串行、
  warm-up 1 + measured 3；环境门沿用 16 GiB；环境拒绝等待自然恢复后重试；8K OOM 如实记录不缩短配置）。
- `experiments/mechanism/analyze.py`：core/family 表、ctx 汇总、AR/bonus/draft-free 对比图、Outcome 判定。
- `experiments/mechanism/test_cpu.py`：**DRAFT_FREE_CPU: PASS 4 / 4**（切块、64-token 预算、位置连续、
  draft 调用护栏反例）。

实现中发现的真实 bug（已修复并重测）：`decode_draft_free` 的 verify 循环最初未包在
`torch.inference_mode()` 中，8K 时 autograd 图保留导致 OOM；修复后 8K 全部通过（`draft_forward_calls=0`）。
失败尝试按纪律停止、保留日志后重测，未修改上下文/k/max_new。

## 2. 函数输入/输出/原理

| 函数 | 输入 | 输出 | 原理 |
| --- | --- | --- | --- |
| `candidate_layout` | frozen ids（64）、k=4、max_new=64 | 13 个 `{round, cands[4], expected_bonus}` | round r 的 candidate = frozen[5r:5r+4]，bonus 期望 = frozen[5r+4]；末轮 60..63 无 bonus |
| `freeze_candidates` / `freeze_via_ar` | target（或正式 AR job）、prompt | `{gen[64], sha1, source}` | 计时外跑 target greedy AR，得到同一输入的 64-token 冻结序列 |
| `DraftCallGuard` | draft | 计数器 + 异常 | 计时区间内把 `draft._run` 换成"计数并抛错"的包装；任何调用即硬失败，退出时恢复 |
| `decode_draft_free` | target、prompt、frozen ids | gen / stats / control / 计时 / 最终 cache | 每轮：输入 `[pending_bonus?] + cands`，target 一次 block forward（真实 KV/cache 更新），bonus 取冻结序列下一 token（forced），round 记账；不调用 draft |

## 3. 候选 token 如何生成并冻结

1. 在**计时区间外**运行正式 AR runner（`W.run_ar_once`，同 prefill（chunk 2048）/greedy/EOS 口径），
   得到 64-token 序列；对 8K 直接读取**同批 AR 结果文件**（`frozen_source = ar_result_file`）以避免额外的
   64-step decode 内存抖动；
2. 记录 token 列表与 sha1（例如 F09/2K smoke：`849ff462…`）；
3. 按 §2 规则切成 13 轮 candidates；所有切块/CPU 张量准备在计时前完成，计时区间内不做文件读取、
   tokenizer 调用或 CPU→GPU 传输。

## 4. 如何证明计时区间内没有 draft forward

- 计时区间内 `DraftCallGuard` 包装 `draft._run`：若被调用会**计数并抛 RuntimeError**；每个 measured rep
  记录 `draft_forward_calls`，实际全部为 **0**（12/12 条件）；
- 结构门把 `draft_forward_calls == 0` 作为硬条件，非零即中止且不输出性能结论；
- draft 保持 resident 且做了真实 prefill（双模型环境保留，`draft_resident=True`，peak allocated 3.82 GiB）；
- 代码路径上 `decode_draft_free` 只调用 `target._run`。

## 5. 保留了哪些 SpecDec 工作

- target block verification（每轮一次，s=4 首轮 / s=5 后续，真实 block 前向）；
- target KV 更新与 cache 推进（`_run` 的 cat/覆盖语义，末轮 cache = ctx+64）；
- bonus emission（bonus 作为输出提交，下一轮作为 pending bonus 的首 token 进入 verify 输入）；
- round accounting（rounds / candidate / bonus / block calls / verify input 长度）；
- stop 控制与输出记录（stop_reason、generated ids、长度）；
- finite 检查与 CUDA 同步边界（与既有机制实验一致）；
- 删除的只有：draft decode forward、draft candidate generation、draft catch-up forward。

## 6. 各 family/context 原始结果（同批 AR / bonus / draft-free；3 reps 中位）

| family | ctx | AR (s) | perfect+bonus (s) | **draft-free (s)** | S=AR/df | S=AR/bonus | rounds | tgt blocks | draft calls | len |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F09 | 2048 | 2.462 | 2.591 | **0.553** | 4.45 | 0.95 | 13 | 13 | 0 | 64 |
| F09 | 4096 | 2.440 | 2.848 | **0.541** | 4.51 | 0.86 | 13 | 13 | 0 | 64 |
| F09 | 8192 | 2.289 | 2.864 | **0.580** | 3.95 | 0.80 | 13 | 13 | 0 | 64 |
| F03 | 2048 | 2.675 | 2.554 | **0.540** | 4.95 | 1.05¹ | 13 | 13 | 0 | 64 |
| F03 | 4096 | 2.288 | 2.585 | **0.613** | 3.73 | 0.89 | 13 | 13 | 0 | 64 |
| F03 | 8192 | 2.283 | 2.518 | **0.535** | 4.26 | 0.91 | 13 | 13 | 0 | 64 |
| F11 | 2048 | 2.366 | 2.704 | **0.558** | 4.24 | 0.88 | 13 | 13 | 0 | 64 |
| F11 | 4096 | 2.258 | 2.585 | **0.550** | 4.11 | 0.87 | 13 | 13 | 0 | 64 |
| F11 | 8192 | 2.237 | 2.772 | **0.539** | 4.15 | 0.81 | 13 | 13 | 0 | 64 |
| F04 | 2048 | 2.491 | 2.799 | **0.558** | 4.47 | 0.89 | 13 | 13 | 0 | 64 |
| F04 | 4096 | 2.297 | 2.690 | **0.526** | 4.37 | 0.85 | 13 | 13 | 0 | 64 |
| F04 | 8192 | 2.436 | 2.580 | **0.540** | 4.51 | 0.94 | 13 | 13 | 0 | 64 |

¹ 单格（F03/2K）bonus S=1.05，属开发集单格噪声；同批 ctx 中位仍为 0.92/0.87/0.86。

**ctx 汇总（family 中位）**：

| ctx | AR 中位 | bonus 中位 | draft-free 中位 | **S_draftfree**（min–max） | S_bonus | df 相对 bonus 加速 | 结构门 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2K | 2.476 | 2.648 | **0.555** | **4.46**（4.24–4.95） | 0.920 | 4.79× | all ok |
| 4K | 2.293 | 2.638 | **0.545** | **4.24**（3.73–4.51） | 0.865 | 4.91× | all ok |
| 8K | 2.286 | 2.676 | **0.540** | **4.21**（3.95–4.51） | 0.857 | 4.86× | all ok |

- 全部 12/12 条件结构门通过：rounds=13、candidate=52、bonus=12、target blocks=13、verify 输入首轮 4/后续 5、
  gen 长度 64、cache 末长 = ctx+64 且 `check_consistent`、有限性、同配置 3 次 rep 的 token 完全一致；
- `draft_forward_calls = 0`（12/12）；draft 保持 resident，peak allocated 3.82 GiB；
- 8K：F11 有 2 轮 target argmax 与冻结 bonus 不一致（近并列、bf16 内核形状差异），按 forced-accept
  契约仍发出冻结 token，记录为 `bonus_argmax_vs_frozen_mismatch=2`（不影响 tokens/cache 结构）。

## 7. Outcome

**Outcome A（Draft-free oracle 稳定满足 S>1，全部 ctx）**：
- 删除 draft decode 后，target verify + bonus + cache/控制路径在 2K/4K/8K 的 S 中位为 4.46/4.24/4.21，
  且每 family 的 S 都 >3.7；
- 说明"候选 token 零成本"时，当前 target 侧执行结构**有约 4 倍于 AR 的余量**；
- 而真实的 perfect+bonus 仍为 0.86–0.92 ⇒ 差距几乎全部来自 draft 执行成本。

## 8. 对"SpecDec 为什么慢"的归因增加了什么证据

此前链条：strict 0.40–0.60 → relaxed 0.68–0.71 → +bonus 0.86–0.93（仍 <1）。
本轮补上最后一段：**把 draft forward 置零后，同一 target 侧结构变成 S≈4.2–4.5**。
因此慢的**不是** verification/cache/控制/bonus 语义，而是：
1. 每个提交 token 需要一次 0.5B draft 前向（64 次/64 token）；
2. 该 draft 前向与 target 前向在 batch=1、长历史 KV 下成本相近（memory-bound），
   导致每轮 4 draft + 1 verify 的结构成本 ≈ 4.6 个 target 步；
3. 即使接受率 100%、轮数降到 13，也无法弥补这部分固定开销。

## 9. 仍然不能得出的结论

- 不能把 Draft-Free oracle 称为可部署方法/加速方案（它没有 drafter，输出由冻结序列强制决定，
  不是真实 SpecDec/AR 分布）；
- 不能把 4.2–4.5× 当作"真实 SpecDec 可以达到"的收益：它衡量的是"零成本 drafter"的上界；
- 不能外推到 4K/8K 之外的配置、或正式 16-family 的确认性结论（本轮为开发集诊断，3 reps 不当作 3 个样本，
  family 为分析单位）；
- 不能说明"target verification 成本可忽略"：它只是相对 draft 成本小；本实验未把 target 侧拆到
  attention/KV/控制粒度；
- strict 参照（Stage B，跨批次 ~4.69/4.66/5.59 s）只作参照，不与本批直接算精确比值。

## 10. 下一步

- **不需要**再为"draft 是否瓶颈"做算子级 profiler——本轮已用因果删除给出结论；
- 若继续，唯一有价值的方向是**降低 draft 执行成本**：
  - EAGLE 类 feature-level drafting（单模型多步/轻量 head，消除每 token 一次 0.5B draft 前向）；
  - 或 route B 静态容量/padding 的 graph 集成（单套 graph，需模型 attention/mask 改写 + token 级等价门）；
- 不建议继续在 acceptance/bonus 协议上投入；也不建议对 target verify 做算子级 profiler（S≈4.2 说明其不是主瓶颈）。

## 11. 产物与复现

- 代码：`experiments/mechanism/{draft_free_oracle,run_draft_free_oracle,analyze,test_cpu}.py`
- 原始数据：`output/draft_free_oracle/full/<family>_<ctx>_<mode>/{result.json,log}`（36 条件 × 3 reps）、
  `output/draft_free_oracle/smoke2/`（最终代码的 F09/2K smoke）
- 聚合：`output/draft_free_oracle/full/analysis/{core_table.csv,family_table.csv,draft_free_summary.json}`
  （副本 `docs/measurements/draft_free_oracle_{core_table,family_table}_2026-10-27.csv`、`draft_free_oracle_summary_2026-10-27.json`）
- 图：`docs/figures/draft_free_oracle/draft_free.{png,pdf,svg}`
  （a 三路径 decode 时间；b S=AR/path 的 ctx 趋势 + family 散点 + y=1 线）
- 复现：`python -m experiments.mechanism.test_cpu`；
  `python -m experiments.mechanism.run_draft_free_oracle --stage smoke`；
  `--stage full`；
  `python -m experiments.mechanism.analyze --run output/draft_free_oracle/full`

## 12. 局限

- 只测 4 个开发 family × 3 ctx；3 reps 只做同配置重复性检查，不作统计推断；
- draft-free 的 bonus 采用 forced-accept 契约（冻结 token），8K 有 2 处 target argmax 近并列不一致已记录；
- 本批 AR/bonus/draft-free 为同批测量；strict 仅为跨批次参照；
- 未做 GPU profiler、未做 4K/8K 之外的扩展、未启动正式实验。
