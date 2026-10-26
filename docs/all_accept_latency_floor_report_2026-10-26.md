# Perfect-Accept Latency Floor 机制实验报告（2026-10-26）

> **性质声明**：本实验是 **mechanism diagnostic / latency lower-bound oracle**，不是正式主实验、
> 不是可部署的 relaxed SpecDec、不是质量可接受的生成方法、不是新算法、**不是 lossless SpecDec**，
> 也不能进入论文主结果表作为生产配置。
>
> **结果**：**Outcome A** —— 即使 acceptance = 100%，当前 1.5B target / 0.5B draft / k=4 / eager 执行结构
> 的 decode speedup 仍只有 **0.85–0.86（全部 <1）**。这意味着当前执行结构存在**结构性延迟下界**：
> 继续优化 acceptance 不可能让这组模型对跨过 AR break-even。

---

## 1. 本轮做了什么

在 `codex/relaxed-specdec`（基线 `ecfef27`）上新增一个**明确命名的诊断模式** `all_accept_oracle`：

- 每轮 draft **仍然真实**提出最多 k=4 个 candidate（proposal 全走）；
- target **仍然真实**执行一次 block verification（rows 全走）；
- 对所有有效 candidate **无条件接受**（m = kp），走现有 `m == kp` 分支：
  `emitted = candidates`，target cache = block cache，draft cache trim 到 `pre+kp`，
  `saved_next = rows[kp-1]`，`drf_lg = prop_lg`；
- EOS/max_new 截断、cache 更新、长度契约全部复用现有 SpecDec 契约；
- **不**跳过 proposal、**不**跳过 verification、**不**伪造/离线估算时间、**不**删除生产路径的 cache 操作。

实现文件（新增，最少代码、全部复用 Phase O 的 `fast_spec_dec` 原语）：
`experiments/relaxed_specdec/all_accept_oracle.py`、
`run_all_accept.py`（24 条件计时）、`gate_all_accept.py`（Gate 2）、
`analyze_all_accept.py`（核心表/逐 family 表/图/判定）、`test_all_accept_cpu.py`（Gate 1）。

输出 schema 明确写入：`experiment_class="mechanism_diagnostic"`、`accept_mode="all_accept_oracle"`、
`quality_claim_allowed=false`、`formal_result_allowed=false`；默认 strict 与已有 relaxed 模式逐位不变
（CPU 测试 `test_default_strict_and_relaxed_unchanged` 验证 delta=0 仍等于 strict executor）。

## 2. 前置门

### Gate 1（CPU，`ALL_ACCEPT_CPU: PASS 7 / 7`）

覆盖：所有 candidate 被接受；m==kp；bonus token 契约（当前 `m==kp` 只发出 candidates，
`bonus_token_count=0`）；candidate 中 EOS（proposal 截断 + 本轮停止）；EOS 作为最后被接受 token 时停止；
max_new 截断（最后一轮 k_eff<k）；target/draft cache 逐轮一致；最终 cache = context + generated length；
EOS 后不再读取 candidate；默认 strict/relaxed 行为不变。

### Gate 2（GPU，12 条件一次性非计时检查，`ALL_OK`）

| 检查 | 结果 |
| --- | --- |
| finite（propose/verify 每次 `_run` 后 isfinite） | 12/12 通过 |
| cache：逐轮 `tgt_len=drf_len=pre+emitted`，末轮 = ctx+gen，`check_consistent` | 12/12 通过 |
| attention 策略断言 | 12/12 通过 |
| 每轮 `emitted == candidates`（所有非 EOS 候选被接受） | 12/12 |
| rollback / mismatch / repair | 全部 0 |
| acceptance_rate | 1.000 |
| gen_len / stop | 全部 64 / `max_new_reached` |
| rounds | 全部 **16** |

**轮回数与理论下界**：当前契约下 `m==kp` 只发出 kp 个 candidate（**不额外发 bonus token**），
因此 64 token / k=4 的理论最少轮数 = **ceil(64/4) = 16**；任务文本里的 13 轮只适用于"每轮发 5 个 token"的
契约。实测全部 16 轮，与契约一致，不存在实现错误；没有 EOS、没有长度截断污染。

## 3. 计时实验（24 条件：12 AR + 12 all-accept，同批同环境重新跑 AR）

- 4 families × 3 contexts × {AR（target-only）, all_accept_oracle}；每条件独立进程、串行、
  warm-up 1 + measured 3；环境门沿用（曾因 commit <16 GiB 被拒，等待自然恢复后继续，未放宽门槛）；
- 72 条 measured 记录：boundary clock 全部合格；oracle 条件 finite/cache/attention 全部通过；
  生成长度全部 64；stop 全部 `max_new_reached`；outputs_consistent 全部通过；
- 质量声明：`quality_claim_allowed=false`（all-accept 会接受 target 明显不偏好的 token，生成文本无质量意义）。

### 核心表（decode 中位；AR 与本批 oracle 同批）

| ctx | AR decode | strict speedup¹ | δ=1 速度¹ | all-accept decode | **all-accept speedup** | acceptance | rounds | rollback |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2048 | 2.528 s | 0.534 | 0.714 | 2.912 s | **0.860**（0.793–0.878） | 1.000 | 16 | 0 |
| 4096 | 2.407 s | 0.509 | 0.680 | 2.829 s | **0.858**（0.835–0.887） | 1.000 | 16 | 0 |
| 8192 | 2.312 s | 0.401 | 0.560 | 2.726 s | **0.849**（0.827–0.868） | 1.000 | 16 | 0 |

¹ strict / δ=1.0 来自 relaxed Stage B 批次（跨批次参照，见 §6「可比性」；不用于同批 speedup 结论）。

### 逐 family 表（本轮同批 AR vs oracle）

| family | ctx | AR decode | all-accept decode | speedup | rounds | output len | stop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F09 | 2048 | 2.579 | 3.250 | 0.793 | 16 | 64 | max_new_reached |
| F09 | 4096 | 2.385 | 2.855 | 0.835 | 16 | 64 | max_new_reached |
| F09 | 8192 | 2.316 | 2.740 | 0.845 | 16 | 64 | max_new_reached |
| F03 | 2048 | 2.477 | 2.840 | 0.872 | 16 | 64 | max_new_reached |
| F03 | 4096 | 2.396 | 2.802 | 0.855 | 16 | 64 | max_new_reached |
| F03 | 8192 | 2.307 | 2.791 | 0.827 | 16 | 64 | max_new_reached |
| F11 | 2048 | 2.443 | 2.880 | 0.848 | 16 | 64 | max_new_reached |
| F11 | 4096 | 2.418 | 2.812 | 0.860 | 16 | 64 | max_new_reached |
| F11 | 8192 | 2.354 | 2.712 | 0.868 | 16 | 64 | max_new_reached |
| F04 | 2048 | 2.583 | 2.944 | 0.878 | 16 | 64 | max_new_reached |
| F04 | 4096 | 2.524 | 2.846 | 0.887 | 16 | 64 | max_new_reached |
| F04 | 8192 | 2.300 | 2.697 | 0.853 | 16 | 64 | max_new_reached |

**最佳单格 0.887（F04 4K），最差 0.793（F09 2K）；没有任何一格 ≥1。**

## 4. 机制图

`docs/figures/all_accept_oracle/latency_floor.{png,pdf,svg}`：

- **Panel A**：3 个 ctx 上 strict(δ=0) → δ=0.5 → δ=1.0 → all-accept 的 acceptance（0.43–0.60 → 0.81–0.85 → 1.0）；
- **Panel B**：对应 verification rounds（26/22/23 → 18/18/16 → 16/16/16；all-accept 恒为 16 =
  ceil(64/4)，虚线标注理论下界）；
- **Panel C**：对应 `T_AR/T_Spec`，含 y=1 break-even 线（strict 0.40–0.53 → δ1 0.56–0.71 → all-accept 0.85–0.86）；
- **Panel D**：把 normalized decode time（Spec/AR）拆成 strict、Phase O、δ=1.0、all-accept 四组，
  显示"每次干预带来的增益"：Phase O 的同步优化只把 strict 从 ~1/0.51 拉低到 ~1/0.53，
  接受率干预（δ1）再到 ~1/0.68，而 all-accept 的上限仍停在 ~1.18（即 speedup 0.85）。

（Panel D 中 strict/Phase O/δ1 为跨批次参照；all-accept 与 AR 为本批同环境。见 §6。）

## 5. 结果支持什么 / 不支持什么

**支持**：
1. **Outcome A：结构性下界**。在 2K/4K/8K 上，all-accept（100% acceptance、0 rollback、0 repair、16 轮）的
   speedup 中位为 0.860/0.858/0.849，**全部 <1**。即"接受率完全不是限制"的极限情况下，这组模型 + k=4 +
   当前 eager 执行结构仍比 AR 慢 ~15%。
2. 接受率干预确实有效但收益递减：strict→all-accept 把 decode 从 4.7–5.6 s 降到 2.7–2.9 s（约 −38%），
   但离 AR（2.3–2.5 s）还差 13–16%。
3. 结构解释（用实测反推）：16 轮 × (4 draft 步 + 1 verify 步) = 64 draft 步 + 16 verify 步；
   AR = 64 target 步。令 target 步成本 c、verify 步成本 v、draft 步成本 d：
   `speedup = 64c / (64d + 16v) = 0.85–0.86` ⇒ **每轮固定成本 4d+v ≈ 4.6c**。
   即使 v≈c，也有 **d ≈ 0.9c**——在 batch=1、长历史 KV 的 memory-bound decode 下，0.5B draft 的一步
   几乎和 1.5B target 的一步一样贵。这就是 k=4 独立 draft 路径的结构性下界来源。
4. 该下界与本 GPU/本 backend/k=4/这组模型对绑定；不能说成"所有 SpecDec 都无法加速"。

**不支持**：
1. **不支持**把 all-accept 写成"可部署配置"、"质量可接受的生成方法"或"正式加速结果"；
   all-accept 会接受 target 不偏好的 token，生成文本没有质量意义（`quality_claim_allowed=false`）。
2. **不支持**"接受率是唯一瓶颈"的结论——相反，它显示接受率提升到 100% 仍不够。
3. **不支持**"long-history attention 让 decode 变慢"的措辞；只能说 decode/verification 调用本身随历史变长
   更贵（prefill 时间单独记录、不混入 decode；本实验 speedup 全部基于 decode 时间）。
4. **不支持**用 "reserved > 8 GiB" 推断显存超限；本实验峰值 reserved 未超物理显存。

## 6. 可比性说明

- 本批 **AR 与 all-accept 同批、同环境、同输入**重新运行，speedup 只使用这两者；
- strict / δ=0.5 / δ=1.0 的 speedup 来自 relaxed Stage B 批次（独立日期、同代码/配置/口径），
  仅用于机制图与"每一步干预"的分解；Phase O 的 optimized 时间来自 2026-10-25 批次；
- 这些跨批次数值不进入同批 speedup 结论；若需正式口径，应在同一批次内重跑 strict/δ1（本轮按任务
  最小运行量约束未重跑）。

## 7. 对论文主故事的影响

- 这是"经典独立 draft 模型 SpecDec 在当前小模型对 + k=4 + eager 执行下不具备加速空间"的**结构性证据**：
  即使 acceptance=1，结构下界仍 ~0.85。
- 因此论文中应当明确：acceptance 干预（relaxed/delta）只能把速度从 0.53 拉近到 0.85 附近，
  **无法越过 1.0**；"接受率不足"与"执行结构成本"共同作用，且后者是当前配置下的硬约束。
- 论文不能把 all-accept 当作方法；只能说它给出了该配置的 latency lower-bound oracle。

## 8. 下一步唯一推荐动作

**转向执行/模型结构层面的改造**，而不是继续优化 acceptance：

1. 首选：**EAGLE 类 feature-level drafting**（用 target 的 hidden features 做 draft、单模型多步/树式提议），
   把"每轮 4 个独立 draft 步 + 1 verify 步"的固定成本降下来——本实验表明独立 0.5B draft 的每步成本
   已接近 target（draft≈0.9c），这是 k=4 路径无法翻盘的根因；
2. 次选：更强的**执行融合**（CUDA Graph 静态化/手工融合 kernel），把每步 launch 成本压下去；
   但 Phase O 已显示纯 host 侧同步优化只有 4% 量级，需要的是算子级合并；
3. 不建议继续在 acceptance 上做优化（本实验已给出上界）。

## 9. 产物清单

- 代码（新增，均在本 worktree 的 `experiments/relaxed_specdec/`）：
  `all_accept_oracle.py`、`run_all_accept.py`、`gate_all_accept.py`、`analyze_all_accept.py`、
  `test_all_accept_cpu.py`。
- 原始数据：`output/all_accept_oracle/gate/`（Gate 2）、`output/all_accept_oracle/run/`
  （24 条件 × 3 reps 的 `runs.jsonl`、逐条件 `result.json`、`analysis/`）。
- 聚合：`output/all_accept_oracle/run/analysis/{core_table.csv, family_table.csv, all_accept_summary.json}`
  （并复制到 `docs/measurements/all_accept_{core_table,family_table}_2026-10-26.csv`、
  `docs/measurements/all_accept_summary_2026-10-26.json`、`docs/measurements/all_accept_gate_2026-10-26.json`）。
- 图：`docs/figures/all_accept_oracle/latency_floor.{png,pdf,svg}`。
- 复现：`python -m experiments.relaxed_specdec.gate_all_accept --out ...` →
  `python -m experiments.relaxed_specdec.run_all_accept --out output/all_accept_oracle/run` →
  `python -m experiments.relaxed_specdec.analyze_all_accept --run output/all_accept_oracle/run`。

## 10. 局限

- 4 个开发 family × 3 contexts × 3 measured；不是正式统计结论；仍只覆盖本 GPU（RTX 4060 Laptop）、
  bf16、k=4、batch=1、eager decode、`cudnn_prefill` 仅 prefill 融合。
- all-accept 输出与 strict 轨迹不同（本就不同），长度均为 64、stop 全部 `max_new_reached`，因此本批
  不存在 EOS/长度污染（Outcome D 未触发）。
- 跨批次参照（strict/δ1/Phase O）只用于机制图分解，不作为同批 speedup 结论。
- 环境门沿用 16 GiB；期间出现过一次自然内存不足（已保留证据并等待恢复），未放宽门槛。
