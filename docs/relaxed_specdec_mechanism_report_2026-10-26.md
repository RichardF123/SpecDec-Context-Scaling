# Margin-Aware Relaxed SpecDec 机制实验报告（2026-10-26）

> **性质声明**：本实验是**机制诊断（development finding / exploratory result）**，不是原正式主实验，
> 也不是"新算法"提案。方法属于**已有 relaxed / lossy speculative decoding 范畴**（放宽 greedy top-1 验证，
> 以 target logit gap 阈值接受 draft token）。strict baseline 与正式协议、正式 runner、旧 JSON/CSV 全部保持不变；
> 结果只允许作为候选点的开发集证据，**不得升级为正式结论**，除非后续在冻结正式 family 上独立验证。

---

## 1. 研究问题

在 Phase O（同步 8→2 次/轮、H2D=0、decode 仅改善 3.8–4.5%，仍未超过 AR）与 acceptance diagnostics
（strict acceptance 中位 0.579/0.599/0.429；176 个首次 mismatch 中 target rank≤5 占 83.5%、gap≤1.0 占 57.4%）
之后，用 **relaxed verification 做因果干预**，判断当前 SpecDec 不加速主要来自：

1. 严格 greedy top-1 验证导致接受率不足；
2. 双模型、eager kernel launch、proposal/verification 等系统开销过高；
3. 或两者共同作用。

## 2. 方法（严格按预注册的接受规则）

对每轮的位置 j（**longest accepted prefix** 语义）：

```
candidate = candidates[j]
top1      = argmax(target_logits[j])
gap       = target_logits[j, top1] - target_logits[j, candidate]
accept    = (candidate == top1)                      若 delta == 0（strict baseline）
            或 (candidate == top1) or (gap <= delta) 若 delta > 0（relaxed）
```

- 首次拒绝即停止：输出 target_top1，回滚（trim 到 pre+m）并 repair；**后续候选不得继续接受**（其 logits 基于
  未确认的 draft prefix）。
- `delta=0` **直接委托 Phase O strict executor**（`execution_optimization.fast_spec_dec`），保证 ids/stats/cache/stop
  逐位一致；relaxed 路径复用同一套 proposal/verify/repair 张量与函数，只替换 acceptance 判定。
- 观察者效应控制：
  - **timed pass** 只计算决策必需的 `top1_logit / candidate_logit / gap`（向量化 GPU 归约，无 top-k/rank/sort、
    无逐 token `.item()`；每轮 2 次显式同步，与 Phase O 相同预算）；
  - **diagnostic pass** 单独运行（finite 检查、cache 校验），从不进入 speedup；rank/top-k 类分析不进入 timed loop。
- 冻结配置：Qwen2.5-1.5B target / 0.5B draft、BF16、`base_bf16` KV、`cudnn_prefill`（仅 prefill 融合）、
  k=4、chunk=2048、max_new=64、greedy EOS-aware、contexts 2048/4096/8192、开发 family F09/F03/F11/F04。

实现与测试：`experiments/relaxed_specdec/{relaxed_spec,run_dev,analyze,gate_gpu,test_relaxed_cpu}.py`；
`specdec/engine/spec_dec.py` 的 strict 路径**未修改**；正式 runner 与既有产物未改动。

## 3. 前置门（全部通过后才允许计时）

1. **CPU 单元测试**：`test_relaxed_cpu.py` **PASS 8 / FAIL 0**（acceptance 前缀语义、m=0/m=kp、首个拒绝发出 top1、
   后续候选不接受、delta=0 委托一致性、tiny-delta 等价、relaxed 计数/regret 自洽、cache 不变量、EOS 截断、
   每轮同步 ≤2 且 h2d≤1）。
2. **GPU correctness gate**（`gate_gpu.py`，12 个开发条件）：
   - `delta=0` 与 Phase O strict executor **12/12 一致**（ids/stats/cache/stop/gen_len）；
   - `delta=0.5` 全部 cache 不变量通过、逐轮 `tgt_len == drf_len == pre + emitted`、末轮 = ctx+gen、
     `check_consistent` 通过、有限性检查通过、attention 策略断言通过；
   - `STRICT_EQUAL_CASES: 12 / 12`，`ALL_OK`。
3. **Stage A**（F09/2K：delta 0/0.3/0.5/1.0 + AR）：cache/finite/schema 全过；acceptance 随 delta 非下降
   （0.658→0.692→0.763→0.859）；每条件 3 次 measured 输出一致；boundary clock 全部合格。**允许进入 Stage B。**
4. **Stage B**（4 families × 3 contexts × 5 delta = 60 relaxed + 12 AR，独立进程、串行、warm-up 1 + measured 3）：
   全部完成；216 条 measured 记录（72 条件 × 3）；relaxed 条件 finite/cache/attention 全部通过；
   boundary clock 全部合格；生成长度全部 64、stop 全部 `max_new_reached`；无 failure、无拒绝（环境门曾在外层
   因 commit < 16 GiB 拒绝一次，已保留证据并等待自然恢复后继续，门槛未改）。

## 4. 结果（Stage B；decode 中位，speedup = median_AR / median_relaxed）

### 4.1 逐 context × delta（4 family 中位）

| ctx | delta | acceptance | rounds | mismatch(rollback) | speedup | decode 相对 strict 降低 | mean regret/接受 token | max regret | 与 strict 共同前缀 | token 全等 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2048 | 0.0 | 0.579 | 23 | 13.0 | 0.534 | — | 0 | 0 | 64 | 4/4 |
| 2048 | 0.1 | 0.616 | 22 | 12.0 | 0.556 | +4% | 0.0000 | 0.00 | 0 | 0/4 |
| 2048 | 0.3 | 0.651 | 21 | 10.5 | 0.588 | +6% | 0.0045 | 0.19 | 0 | 0/4 |
| 2048 | 0.5 | 0.767 | 19 | 6.5 | 0.670 | +21% | 0.0228 | 0.50 | 0 | 0/4 |
| 2048 | 1.0 | 0.811 | 18 | 6.0 | **0.714** | +25% | 0.0671 | 0.94 | 0 | 0/4 |
| 4096 | 0.0 | 0.599 | 22 | 13.5 | 0.509 | — | 0 | 0 | 64 | 4/4 |
| 4096 | 0.1 | 0.640 | 21 | 11.0 | 0.558 | +9% | 0.0000 | 0.00 | 0.5 | 0/4 |
| 4096 | 0.3 | 0.680 | 20.5 | 10.0 | 0.582 | +13% | 0.0036 | 0.13 | 0.5 | 0/4 |
| 4096 | 0.5 | 0.694 | 20.5 | 8.5 | 0.567 | +12% | 0.0102 | 0.25 | 0.5 | 0/4 |
| 4096 | 1.0 | 0.849 | 18 | 4.0 | **0.680** | +26% | 0.0722 | 1.00 | 0.5 | 0/4 |
| 8192 | 0.0 | 0.429 | 26 | 19.5 | 0.401 | — | 0 | 0 | 64 | 4/4 |
| 8192 | 0.1 | 0.490 | 24.5 | 17.5 | 0.413 | +2% | 0.0000 | 0.00 | 0 | 0/4 |
| 8192 | 0.3 | 0.564 | 23.5 | 13.5 | 0.497 | +17% | 0.0053 | 0.13 | 0 | 0/4 |
| 8192 | 0.5 | 0.655 | 21 | 11.0 | 0.556 | +24% | 0.0165 | 0.44 | 0 | 0/4 |
| 8192 | 1.0 | 0.745 | 19.5 | 7.0 | **0.560** | +29% | 0.0656 | 1.00 | 0 | 0/4 |

（4K 的 d0.5 相对 d0.3 略慢（0.567 vs 0.582）是单格噪声，趋势在 d0.1→0.3→1.0 上单调。）

### 4.2 逐 family 关键点（delta 0 → 1.0）

| family | ctx | speedup d0 | speedup d1.0 | acceptance d0 → d1.0 | 共同前缀 @d1.0 | mean regret @d1.0 |
| --- | --- | --- | --- | --- | --- | --- |
| F09 | 2048 | 0.605 | **0.804（全矩阵最佳）** | 0.658 → 0.859 | 0 | 0.049 |
| F03 | 2048 | 0.481 | 0.552 | 0.516 → 0.619 | 1 | 0.094 |
| F11 | 2048 | 0.575 | 0.732 | 0.596 → 0.792 | 0 | 0.081 |
| F04 | 2048 | 0.493 | 0.696 | 0.562 → 0.831 | 0 | 0.053 |
| F03 | 4096 | 0.518 | 0.787 | 0.621 → 0.924 | 1 | 0.037 |
| F09 | 4096 | 0.483 | 0.735 | 0.562 → 0.912 | 1 | 0.054 |
| F04 | 4096 | 0.511 | 0.597 | 0.593 → 0.675 | 0 | 0.103 |
| F11 | 4096 | 0.507 | 0.624 | 0.605 → 0.787 | 0 | 0.090 |
| F03 | 8192 | 0.636 | 0.684 | 0.776 → 0.857 | 0 | 0.071 |
| F11 | 8192 | 0.383 | 0.559 | 0.398 → 0.737 | 1 | 0.060 |
| F04 | 8192 | 0.418 | 0.560 | 0.460 → 0.753 | 0 | 0.047 |
| F09 | 8192 | 0.331 | 0.436 | 0.309 → 0.538 | 0 | 0.120 |

## 5. 四个研究问题的回答

**RQ1（放宽接受后的 acceptance / rounds / rollback 改善）**
明显改善，且随 delta 单调（除单格噪声）：acceptance 中位 2K 0.579→0.811（+0.23）、4K 0.599→0.849（+0.25）、
8K 0.429→0.745（+0.32）；verification rounds 2K 23→18、4K 22→18、8K 26→19.5；rollback（mismatch rounds）
2K 13→6、4K 13.5→4、8K 19.5→7。只有 7–10.5 个/轮的 accepted token 来自 relaxed-only（其余仍与 strict top-1 相同）。

**RQ2（接受率改善能否转化为真实 wall-clock，2K 能否接近/超过 AR）**
能转化，但**不足以越过 break-even**：d1.0 时 decode 相对 strict 降低 **25.5% / 25.7% / 28.5%**（2K/4K/8K），
但 speedup（AR/Spec）只到 **0.714 / 0.680 / 0.560**；全矩阵 60 个 relaxed 条件**没有任何一格 > 1.0**
（最高为 F09 2K d1.0 的 **0.804**，仍比 AR 慢约 20%）。因此 2K "接近但未达到" AR。
外推：2K 要在 d1.0 上达到 speedup=1.0，需再减少约 40% decode，即把 rounds 从 18 降到 ~13 以下，
而 k=4 下即使 acceptance→1.0 也只能把每轮长度推向 k+1=5 token（64 token 需 ~13 轮）——恰在边界，
但实际 acceptance 0.81 且每轮还要 target verify+repair 固定成本，因此**系统开销仍是限制项**。

**RQ3（轨迹/regret/长度/质量代价）**
- **完全改变输出轨迹**：任何 delta>0 的 12/12 family-context 都不再与 strict token 全等；共同前缀 0–1 个 token
  （2K 全为 0；4K 部分为 1；8K 全为 0）。这是 relaxed 定义直接导致的（接受非 top1 token 会改变后续上下文）。
- **长度无变化**：216 次 measured 全部 64 token，stop 全部 `max_new_reached`，length_delta=0；
  所以本实验没有"输出变短导致更快"的伪加速路径。
- **regret 可控但非零**：relaxed-only token 的 target-logit regret 每 run 合计中位 d0.3≈0.2–0.3、d0.5≈0.6–1.25、
  d1.0≈3.8–4.3（即每接受 token 平均 0.004–0.12 logit 单位）；单 token 最大 regret 在 d1.0 达到 **1.0**
  （接受 token 的 target 概率可低至 top1 × exp(-1) ≈ 36.8%）。
- 结合已有 acceptance diagnostics：首次 mismatch 中 83.5% 的 draft token 在 target rank≤5、57.4% gap≤1.0，
  本实验的 d1.0 acceptance 提升（8K 0.43→0.75）与这一分布一致；`delta=0.3/0.5` 是"质量代价明显更小"的档位
  （总 regret 中位 ≤1.25、单 token 最大 ≤0.5），但速度提升也更有限（speedup ≤0.67）。

**RQ4（上下文长度的影响）**
- 低上下文（2K/4K）strict acceptance 更高（0.58/0.60），relaxed 后 rounds 降得少（23→18），但 baseline 已接近
  break-even（F09 2K 0.60 的 strict speedup 本身已高于其他格），d1.0 达 0.71–0.80。
- 8K strict acceptance 最低（0.429），接受率**相对提升最大**（+0.32），rounds 降得最多（26→19.5），
  但 speedup 仍最低（0.56）：说明 8K 的固定成本（更长 prefill 与更长历史 attention）把收益吃掉更多；
  质量风险与 2K/4K 同量级（mean regret 0.066 vs 0.067/0.072；共同前缀 0；长度不变）。
- 结论：上下文越长，**接受率瓶颈越重、但系统成本瓶颈也越重**，relaxed 只能部分抵消后者。

## 6. 判定（按任务 §十）

**结果 B：接受率提高但仍不加速（strict verification/alignment 不足是真实瓶颈，但系统执行成本占主导，
即使人为改善接受率也不足以跨过 break-even）。**

- 无任何 (family,ctx,delta) 达到 speedup>1；2K 中位最高 0.714，最佳单格 0.804。
- 接受率/rounds/rollback 的改善确实转化为 25–29% 的 decode 时间下降——这是**真实的机制证据**：
  若未来能把系统固定成本压低（或换更快执行路径），acceptance 改善将直接变成加速。
- 但当前配置下（k=4、eager decode、双模型、cudnn_prefill-only）不足以跨越 AR。

**不适用 A/C/D**：A（加速）需要 speedup>1；C（加速但质量明显恶化）也要求先有加速；D（轻微 delta 即改善）
虽有 d0.3 的低 regret 档，但 speedup 仍 <1，不能作为"候选加速点"。

## 7. 是否值得进入独立验证阶段？

- **不建议**把任何 relaxed 档位升级为正式候选：speedup 全部 <1，且输出轨迹全部偏离 strict（不是 lossless）。
- **建议的后续（需批准、不在本轮）**：若后续把系统固定成本（Phase O 未解决的 launch 数/双模型）压低到
  ~×1.3 以上，可用本轮 fixtures 与 run_dev 复测 d0.3/d0.5（低 regret 档）看是否跨 1.0；
  在冻结正式 family 上独立验证也只应在那个前提下做。
- 本轮结论只可表述为：**acceptance 是重要但不是唯一瓶颈；在当前系统实现下，relaxed verification 换来
  25–29% 的 decode 加速，但离 AR 还差 20–44%**。

## 8. 局限

- 4 个开发 family × 3 次 measured；mechanism diagnostic，不作显著性或正式推断。
- 8K 的 speedup 中位（0.56）低于 2K/4K：与历史正式基线（0.52/0.49/0.52）顺序一致，8K 仍有更高的方差；
  单格（F03 8K strict 0.636）说明时钟/环境噪声不可忽略。
- 只测了 k=4、64 token；未测更长生成或更大 k（冻结配置）。
- 长度全为 64 说明本批次的长短效应未被触发（所有 family 都在 64 内未遇 EOS）。
- delta=0.1 的轨迹已完全偏离（共同前缀 0），说明"极小 delta"也不能视为近似无损——这是 relaxed 规则本身的性质，
  不是实现 bug（gate 的 12/12 严格等价只保证 delta=0）。

## 9. 产物与可复现

- 代码：`experiments/relaxed_specdec/{relaxed_spec,run_dev,analyze,gate_gpu,test_relaxed_cpu}.py`（本次新增；
  未改 `specdec/engine/*` 默认路径、未改正式 runner）。
- 数据：`output/relaxed_specdec/gate/`（correctness gate）、`output/relaxed_specdec/stage_a/`（pilot）、
  `output/relaxed_specdec/stage_b/`（72 条件 × 3 reps；`runs.jsonl`、逐条件 `result.json`、`analysis/`）。
- 摘要 CSV/JSON：`docs/measurements/relaxed_stageb_{condition_summary,speed_quality_frontier,cases}_2026-10-26.csv`、
  `docs/measurements/relaxed_stageb_summary_2026-10-26.json`。
- 图：`docs/figures/relaxed_specdec/relaxed_specdec_frontier.{png,pdf,svg}`（a 接受率、b rounds、c speedup、
  d mean regret、e 共同前缀、f speed-quality Pareto）。
- 复现：`python -m experiments.relaxed_specdec.gate_gpu --out ...` →
  `... run_dev --stage a` → `... run_dev --stage b` → `... analyze --run output/relaxed_specdec/stage_b`。
