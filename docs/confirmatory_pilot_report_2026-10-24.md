# 确认测量 Pilot 报告（2026-10-24）

> 范围：SpecDec 独立确认实验的 **pilot**（4 个开发族 × 3 上下文 × 2 模式 × 2 交错 block）。
> 预注册：`docs/confirmatory_measurement_preregistration_2026-10-24.md`（本报告只按该文档 §10 裁决，阈值未改动）。
> 本轮**没有**启动完整 16-family 正式确认实验；**没有**覆盖、删除或改写任何旧正式批次/旧 CHECK/Route C 证据。
> 标签约定：**已验证通过** / **已实现未验证** / **已存在** / **设计完成未实现**。

## 2026-10-25 只读审计勘误

Pilot 的原始数据、裁决和数值不变。后续源码审计发现，本 pilot 使用的调度器是在**单任务**层面
交错 block，而不是本报告第 1 节所写的“完整 pair 连续执行”：实际序列为 block0 的 pair 第一个
任务、block1 的另一个 pair 第一个任务、再回到 block0 的第二个任务。因此同一 AR/Spec pair 相距
一个插入任务，仍然接近但并非严格相邻。

本批次使用的未提交源码与预注册文档已按原字节保存到
`output/confirmatory_pilot/20261024T061624Z_0b0792bb/source_snapshot/`，并附 SHA256 manifest。
完整确认实验的后续 runner 已改为**完整相邻 pair**作为交错单位，并按 context 分层保持 AB/BA
各 8/8；该修订不回填或重解释本 pilot。正式执行协议另见
`docs/confirmatory_full_protocol_2026-10-25.md`。

---

## 1. 本轮改了什么，各自解决哪个旧问题

旧正式批次的四个已知问题（来自 `docs/formal_measurement_and_operator_closeout_2026-10-24.md`）：

| 旧问题 | 本轮改动 | 原理 |
| --- | --- | --- |
| (a) Route C 让旧 8K 无法参与 speedup 分析 | 预注册 8K 规则：六项检查全 OK ⇒ 允许 family-level `T_AR/T_Spec`；否则只保留绝对量并标 restricted；不回填旧批次 | 把“能不能用比例”绑定在**本批次**证据上，而不是历史路线条件 |
| (b) 4K 的 `0/16 all-OK` 被误当成“4K 不可测” | 新增 pilot 的六项检查复算器，把 within/between/boundary 分开计数；4K 先用 4 个开发族做流程裁决 | 区分“数据错”与“调度导致的块间漂移”，先验证流程再扩样本 |
| (c) 旧 4K 的 block-era 混淆（AR block1 13/16 更慢，中位 +28.4%） | 新调度：**pair 为最小单位**（同一 (family,ctx) 的 AR+SpecDec 连续执行、各自独立进程），block0/block1 在时间线上**逐对交错**，不再“先跑完 block0 再跑 block1” | block 身份不再与运行时段绑定，块间差异只反映短时环境波动 |
| (d) 旧块顺序固定 | 固定种子决定：一半 pair 用 AB、一半用 BA；block/ctx 交错 | 用顺序平衡把“先跑 vs 后跑”的系统效应摊平并可检验 |
| (e) 旧 raw/CHECK/证据不可动 | 新 pilot 只写 `output/confirmatory_pilot/`；旧文件零改动 | — |

实现（**已验证通过**，均为最小增量，未复制模型实现）：
- `experiments/confirmatory/plan.py`：`Task`（含 `confirmatory_batch_id/pair_id/pair_order/pair_position/measurement_block/schedule_seed/planned_order`）+ `build_plan` + `validate_plan`。
- `experiments/confirmatory/worker.py`：复用 `experiments/formal/worker.py` 的全部函数（`load_input/load_target/load_draft/configure_target/run_arithmetic…`），只在任务级追加调度元数据；`experiment_stage="development"`。
- `experiments/confirmatory/run.py`：dry-run 与 `--execute`；入口 18 GiB 环境门槛；每任务独立进程 + `launch.json/exit.json` + `verify_job_results`。
- `experiments/confirmatory/analyze_pilot.py`：CPU 离线分析（六项检查、block shift、AB/BA、AR/Spec 输出一致性、裁决）。
- 测试：`test_confirmatory_cpu.py` **PASS 23/0**；`test_analyze_pilot_cpu.py` **PASS 5/0**（见 §3）。

冻结项（未改动，**已验证通过**）：模型 `Qwen2.5-1.5B/0.5B`、BF16、`base_bf16` KV、`cudnn_prefill` 仅 prefill、eager decode/propose/verify/repair、`chunk=2048`、`k=4`、`max_new=64`、greedy、CPU-FP32 inv_freq、`model_self.py` 数学路径。

## 2. 实际跑的任务

- 批次：`output/confirmatory_pilot/20261024T061624Z_0b0792bb/`
- 规模：**48 tasks / 120 measured**（4 家族 × 3 ctx × 2 模式 × 2 block；每 block 3+2 次 measured，=5/条件）；另有 48 个任务级 warm-up（不计入）。
- 家族：`F09 / F03 / F11 / F04`（开发集；**未使用 FF01–FF16**）。
- 顺序平衡：AB=6 对、BA=6 对（12 对）；block 交错 transitions=47，max_same_block_run=1（两 block 严格交替）。
- 环境：入口 `commit_free≈20.5 GiB ≥ 18 GiB`，GPU P-state 预检通过；无 OOM、无 `failure.json`、无拒绝（`output/confirmatory_pilot/` 下无 refusals 目录）。
- 时间：batch 06:16:24Z → 07:04:51Z，span **48.5 min**；每任务 wall 中位 **48.5 s**（38–75 s，p95 69 s）。
- 48/48 `development_job_verified`；hash、`same_mode_diagnostic_tokens_equal`、Spec 对账（accepted+mismatch=gen、rounds=block_calls、draft_calls=proposed）全部通过（分析器硬校验，违反即中止）。

## 3. CPU 测试 vs GPU 数据

- **CPU（不触 GPU、无测量数据）**：`test_confirmatory_cpu.py`（同种子同计划、换种子任务集不变顺序变、无重复/缺漏、pair 完整、AB/BA 差 ≤1、block 交错条件、reps 3+2=120、输入 hash、dry-run 48 行且无 CUDA 分配、plan 独占写、worker dry-run 与计划一致、策略仍为 eager 等）23 项；`test_analyze_pilot_cpu.py`（合成批：READY 分支、缺任务 ENV_BLOCKED、boundary 不完整 NEEDS_FIX、4K 大漂移 NEEDS_FIX、hash 不一致拒绝）5 项。
- **dry-run（CPU）**：`run --execute` 之前先打印完整 48 任务执行序列；与预注册一致性由 `validate_plan` 强制。
- **GPU 实测数据**：仅本批次 48 tasks / 120 measured 的 records（`worker_results/*.json`）；以下所有表格均来自它们。分析器中**没有**任何合成数值进入输出 CSV（合成批只存在于测试临时目录）。

## 4. 结果（48 tasks / 120 measured）

### 4.1 六项检查（20% 诊断阈值，沿用旧审计定义）

| ctx | pair 数 | all-six-OK | AR/Spec 输出一致 | SpecDec 更慢（ratio<1） |
| --- | --- | --- | --- | --- |
| 2048 | 4 | **4/4** | 3/4 | 4/4 |
| 4096 | 4 | **4/4** | 1/4 | 4/4 |
| 8192 | 4 | **4/4** | 0/4 | 4/4 |

- within（单任务 3/2 次 measured 的 throughput 极差/中位）：全部 OK，最大 11.6%（阈值 20%）。
- between（同 (family,ctx,mode) 两 block 的 throughput 中位差/max）：全部 OK，最大 10.0%（F09 4K AR）。
- boundary：120 条 measured 的 `runtime_environment_audit.evidence_status` 均为 COMPLETE、`boundary_clock_qualified=True`。

### 4.2 family-level 计时（5 次 measured 合并；decode 为秒）

| family | ctx | AR decode 中位(min–max) | SpecDec decode 中位(min–max) | within | between | 8K 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| F09 | 2048 | 2.43 (2.30–2.58) | 4.24 (4.20–4.28) | OK | OK | — |
| F09 | 4096 | 2.42 (2.24–2.55) | 4.82 (4.77–4.87) | OK | OK | — |
| F09 | 8192 | 2.21 (2.20–2.24) | 7.11 (6.93–7.50) | OK | OK | 可报 |
| F03 | 2048 | 2.42 (2.36–2.51) | 5.16 (5.12–5.25) | OK | OK | — |
| F03 | 4096 | 2.33 (2.27–2.39) | 4.62 (4.41–4.81) | OK | OK | — |
| F03 | 8192 | 2.36 (2.25–2.46) | 3.50 (3.44–3.59) | OK | OK | 可报 |
| F11 | 2048 | 2.34 (2.28–2.38) | 4.73 (4.61–4.88) | OK | OK | — |
| F11 | 4096 | 2.29 (2.26–2.33) | 4.65 (4.48–4.77) | OK | OK | — |
| F11 | 8192 | 2.45 (2.22–2.53) | 5.94 (5.91–5.97) | OK | OK | 可报 |
| F04 | 2048 | 2.42 (2.33–2.54) | 4.83 (4.80–4.90) | OK | OK | — |
| F04 | 4096 | 2.28 (2.24–2.35) | 4.81 (4.64–5.02) | OK | OK | — |
| F04 | 8192 | 2.22 (2.21–2.23) | 5.68 (5.51–5.94) | OK | OK | 可报 |

（完整 MAD/IQR、prefill、total、峰值见 `pilot_family_timing.csv`；峰值 reserved 最高 6.58 GiB，8K 无异常。）

### 4.3 block shift（block1 vs block0 decode 中位 %）

| ctx | AR：中位 %（block1 更慢族数） | SpecDec：中位 %（更慢族数） |
| --- | --- | --- |
| 2048 | +5.36%（4/4） | −0.19%（2/4） |
| 4096 | **+3.15%（3/4）** | **+2.55%（2/4）** |
| 8192 | +3.45%（4/4） | +1.12%（2/4） |

### 4.4 AB/BA 顺序对比（pair 内先执行 vs 后执行；每格 n=2–6）

| ctx | mode | 先执行中位 (s) | 后执行中位 (s) | 先 vs 后 % |
| --- | --- | --- | --- | --- |
| 2048 | AR | 2.343 | 2.422 | −3.3% |
| 2048 | SpecDec | 4.834 | 4.728 | +2.3% |
| 4096 | AR | 2.292 | 2.419 | −5.2% |
| 4096 | SpecDec | 4.816 | 4.660 | +3.4% |
| 8192 | AR | 2.220 | 2.418 | −8.2% |
| 8192 | SpecDec | 4.737 | 6.387 | **−25.8%** |

8192/SpecDec 的“先执行更快”是单格现象（n_first=4, n_second=4）；按预注册“同方向 >20% 的格子 ≥3 才算系统性”未触发，但它记录在案，完整批次用更多 pair 复核。

### 4.5 AR/SpecDec 输出一致性

| family | ctx | 一致? | 首个分歧位置（64 个生成 token 内） |
| --- | --- | --- | --- |
| F09 | 2048 | 是 | — |
| F09 | 4096 | 否 | 26 |
| F09 | 8192 | 否 | 21 |
| F03 | 2048 | 否 | 1 |
| F03 | 4096 | 否 | 1 |
| F03 | 8192 | 否 | 33 |
| F11 | 2048 | 是 | — |
| F11 | 4096 | 否 | 56 |
| F11 | 8192 | 否 | 4 |
| F04 | 2048 | 是 | — |
| F04 | 4096 | 是 | — |
| F04 | 8192 | 否 | 35 |

预注册已规定“AR/Spec 输出分歧不是硬失败，记录后继续”。本轮观察到 8K 全部 4/4 分歧、4K 1/4 一致、2K 3/4 一致；**同一目标族的同模式内 3/2 次 measured 全部 token 一致**（`same_mode_diagnostic_tokens_equal` 全 True），说明分歧出现在两模式之间而非测量不稳定。可能来源：生产 verify 路径为 block 验证（k+1 token 并行前向）与 AR 逐 token 串行前向的 bf16 数值差异（项目早期已验证 bf16 近并列 token 翻转现象），以及 mismatch 后 repair 路径的再计算；**本轮未进一步归因**，完整批次继续记录该指标，若同样普遍出现将作为独立问题分析，不改变计时裁决。

### 4.6 SpecDec 行为与开销

- acceptance（accepted/proposed）：0.31–0.78，8K 最低（F09 0.31、F11 0.41），2K–4K 0.47–0.66。
- rounds：19–31；mismatch rounds：5–26；`target_block_calls=rounds`、`repair=mismatch_rounds`（对账通过）。
- 12/12 pair `T_AR/T_Spec` 中位 0.40–0.67，即 **SpecDec decode 慢于 AR**（与项目此前结论一致：k=4 的 verify 开销占主导）。pilot 不作 speedup 结论。

## 5. 与旧正式批次对照（只做描述，不作显著性检验）

| 指标 | 旧正式批次（16 family） | 本 pilot（4 family） | 结论 |
| --- | --- | --- | --- |
| 4K all-six-OK | 0/16 | **4/4** | 流程上 4K 可测（样本小，需完整批次确认） |
| 4K AR block1 更慢 | 13/16，中位 +28.4% | 3/4，中位 **+3.2%** | 块间漂移显著减小 |
| 4K SpecDec block1 更慢 | 10/16，中位 +8.1% | 2/4，中位 **+2.6%** | 同上 |
| 2K all-six-OK | 3/16 | 4/4 | 同左 |
| 8K all-six-OK | 5/16 | 4/4 | 同左 |
| 8K 可报比例 | 仅 Route C 条件满足时 | 4/4 六项 OK ⇒ 允许 `T_AR/T_Spec` | 见 §8 |
| 8K AR block1 更慢 | 7/16，中位 −1.9% | 4/4，中位 +3.5% | 8K 方向不同（小样本，完整批次复核） |

旧批次数字来自 `output/paper_evidence/2026-10-24_v1/block_shift_summary.csv` 与 `timing_check_counts.csv`（旧文件未改动）。

## 6. 交付物

- 预注册：`docs/confirmatory_measurement_preregistration_2026-10-24.md`（**已存在**，本轮未改阈值）。
- 计划/runner/分析器：`experiments/confirmatory/{plan,worker,run,analyze_pilot}.py`（**本次新增**）；测试 `experiments/confirmatory/test_confirmatory_cpu.py`、`test_analyze_pilot_cpu.py`（**本次新增，已验证通过**）。
- 原始产物：`output/confirmatory_pilot/20261024T061624Z_0b0792bb/`（`plan.json`、`batch.json`、`tasks/NN_*/{task_meta,launch,worker.log,exit}.json`、`worker_results/*.json`、`summary.json`）。
- 分析产物（`<batch>/analysis/`，**本次新增**）：
  - `pilot_tasks.csv`（48 行任务记录，含判序/块/起止时间/计时/spread/边界）
  - `pilot_family_timing.csv`（24 行 family-level 计时与检查状态）
  - `pilot_block_shifts.csv`（24 行块差）
  - `pilot_order_ab_ba.csv`（6 行 AB/BA 对比）
  - `pilot_pair_summary.csv`（12 行 pair：ratio、输出一致性、acceptance、峰值、8K 状态）
  - `pilot_context_checks.csv`（3 行 ctx 汇总）
  - `summary.json`（含裁决）
  - `pilot_stability.{svg,png,pdf}`（稳定性格：左=family decode 中位与散点；右=block shift 箱线 + 逐点，±20% 参考线）

## 7. 完整确认实验估计

- 规模：16 family × 3 ctx × 2 模式 × 2 block = **192 tasks / 480 measured**（3+2 每 block），48 对，AB=24/BA=24；`validate_plan` 预览已跑通（transitions=191，max run=1）。
- 时间：pilot 每任务 median 48.5 s；192 × 48.5 s ≈ 155 min，加 191 次 ×10 s 冷却 ≈ 32 min ⇒ **约 3.1–3.5 h GPU wall**（输入加载与 FF 家族验证另计少量）。
- 磁盘：pilot 48 tasks ≈ 8 MB（含日志/记录）；192 tasks ⇒ **约 30–35 MB**（records + logs + analysis），可忽略。
- 前提：内存入口 18 GiB 门（本 pilot 入口 20.5 GiB）、无环境拒绝；中途 OOM/拒绝按停止条件处理。

## 8. 裁决

**`READY_FOR_CONFIRMATORY_FULL`**（按预注册 §10 逐条核对）：

- 48/48 tasks、120/120 measured，0 failure、0 refusal：`development_batch_verified`。
- boundary 证据 120/120 COMPLETE 且 qualified；输入 hash 120/120 与 manifest 一致；同模式诊断等于计时输出；Spec 对账通过。
- 4K all-six-OK = 4/4 ≥ 1；2K = 4/4 ≥ 1；8K = 4/4 ≥ 2。
- AB/BA 系统性检验：同方向 >20% 的格子最多 1 个（8192/SpecDec = −25.8%，n=4/4），低于“≥3 格”的系统性门槛 ⇒ 不算系统性顺序偏差。
- 8K：四对六项全 OK ⇒ 按预注册允许 family-level `T_AR/T_Spec`；**不得**用 pilot 结果回填/解锁旧批次 8K。
- 未用 pilot 筛选/剔除正式 family。

## 9. 局限（必须随报告保留）

1. 4 个开发族不是 16 个正式族；pilot 只回答“流程是否可进入完整实验”，不构成吞吐结论。
2. 每条件仅 5 次 measured（3+2），无正式统计检验；20% 是诊断阈值，不是显著性门槛。
3. AR/Spec 输出分歧在 pilot 中普遍（8K 4/4），按预注册不算拒收；其归因（block 验证 bf16 数值差 vs repair 语义）未在本轮完成，完整批次需继续记录并单独分析。
4. 8192/SpecDec 存在单格 −25.8% 的顺序差（n=4/4）：pilot 未达到“系统性”门槛，但完整批次必须复核；若完整批次出现同方向 ≥3 格 >20%，按 §10 转为 `NEEDS_MEASUREMENT_FIX`。
5. 4K 漂移改善结论基于 3/4 与 2/4 计数与小样本中位数，显著性未检验。
6. 旧正式批次（192 tasks / 480 measured）未重跑、未重标；Route C/CHECK 证据保持原样。
7. 未做 GPU 时钟/P-state 之外的电源与后台程序控制；interleaving 不能消除所有时段效应，只能平衡。
