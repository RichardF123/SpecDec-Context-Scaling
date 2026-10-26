# Bonus-Aware Perfect-Accept Oracle 报告（2026-10-26）

> **性质**：mechanism diagnostic / latency lower-bound oracle（bonus 契约版）。
> **不是**正式主实验、**不是**可部署算法、**不是** lossless SpecDec、**没有**质量声明资格
> （`quality_claim_allowed=false`、`formal_result_allowed=false`）。
> 基线：`codex/relaxed-specdec` @ `e9b6603`；严格路径、正式 runner、Phase O / relaxed / 无 bonus 产物全部未改动。

## 0. 设计审计结论（先于 GPU，见 `docs/bonus_accept_design_audit_2026-10-26.md`）

- **无 bonus 契约**：每轮 draft 提出 kp≤4 个 candidate，target 做一次 block verify，`m==kp` 时只提交
  `emitted = candidates`；最后一行 `rows[-1]`（预测 c_kp 之后 token 的 logits）只作为下一轮的 `saved_next` 保留，
  **不提交** ⇒ 64 token 需 16 轮，bonus=0。
- **bonus 来源**：`bonus = argmax(rows[-1])`——即无 bonus 契约里被丢弃的"最后一行 argmax"，本来就是预测值，
  不是任何前向的输入。
- **pending-bonus carry-forward（计时路径）**：发出 bonus 后 target cache 不含它（长度 = ctx+committed−1）；
  下一轮 draft 先做 1 次 catch-up `_step(bonus)`，target 的 block 输入改为 `[pending_bonus] + candidates`
  （首轮 L=kp，之后 L=kp+1）；`row[0]` 消费 pending bonus、`row[-1]` 产生新 bonus；**target 只做 block call**，
  timed path 的额外 target 单步 = 0。另一条 `catchup`（显式"bonus 立即由两侧 _step 消费"）仅作正确性参照。
- **轮数 13 的原因**：每轮最多提交 kp+1=5；12×5=60，尾轮剩余 4 ⇒ 只发 4 个 candidate、不发 bonus；
  共 13 轮、candidate 52、bonus 12、合计 64。
- **cache 不变量**：`target_len + pending == draft_len + pending == ctx + committed − pending`
  （pending 为该轮 bonus 是否尚未被两侧消费；发出 bonus 后 pending=1、两侧都比 committed 少 1，下一轮补入）。

## 1. 做了什么

实现 `experiments/relaxed_specdec/bonus_accept_oracle.py`（`mode="carry"` 计时路径 / `mode="catchup"` 正确性参照）、
`test_bonus_accept_cpu.py`（Gate 1）、`gate_bonus_accept.py`（Gate 2）、`run_bonus_accept.py`（同批 36 条件）、
`analyze_bonus_accept.py`（核心表/工作量分解/4 面板图/判定）。全部新增，不覆盖任何旧文件与旧结果。

## 2. Gate 结果

**Gate 1（CPU）**：`BONUS_ACCEPT_CPU: PASS 12 / 12`。覆盖首轮无 pending、candidates+bonus 提交、
draft catch-up、target block 以 pending 开头、pending 不重复输出、row 位置映射、13 轮、
尾轮不发 bonus、candidate EOS、bonus EOS、max_new 截断、逐轮 target/draft cache 长度、
最终 cache 与 committed 一致、catchup≡carry、默认 strict/relaxed/无 bonus 行为不变。

**Gate 2（GPU，12 条件非计时）**：
- 结构 12/12 通过：`rounds=13`、candidate=52、bonus=12、total=64、`stop=max_new_reached`、
  acceptance=1、rollback/repair/mismatch=0、finite/cache/attention 全过、最终 cache 两侧均 = ctx+64、
  timed path 的 `target_extra_single_step_calls=0`。
- **token 级对照：11/12 与 catchup 完全一致**；唯一例外 **F04/2048 第 2 轮 bonus argmax**：
  carry 侧该行 top1/top2 同为 **18.75（gap=0.0，精确并列）**，catchup 侧 gap=**0.125（1 个 bf16 ulp）**；
  候选 token 全部一致、cache/计数/不变量一致。
- 根因（已隔离实测）：carry 的 verify 是 s=5 单块，catchup 是 s=1 step + s=4 块，**kernel 形状不同**，
  在 0-gap 的并列 argmax 上 bf16 舍入相差 1 ulp → 选到不同 token。不是 cache 错位、不是位置映射错误。
- 处理：经批准**按"语义等价 12/12 + 该 1 例记为 bf16 tie 伪差异"继续计时**；本报告如实保留
  `BITWISE_EQUAL_CASES: 11/12`、`SEMANTIC_EQUAL_CASES: 11/12` 与分歧细节。

## 3. 运行覆盖度

- 同批同环境：**12 AR + 12 no-bonus + 12 bonus = 36 条件**，独立进程、串行、warm-up 1 + measured 3
  ⇒ **108 条 measured**；
- 全部 boundary clock 合格；oracle 条件 finite/cache/attention 全过；输出长度全部 64、stop 全
  `max_new_reached`（无 EOS）；无 failure/拒绝（环境门在等待后满足，未放宽）。

## 4. 核心表（decode 中位；同批）

| ctx | AR decode | no-bonus decode | no-bonus speedup | no-bonus rounds | bonus decode | **bonus speedup** | bonus rounds | bonus 数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2048 | 2.374 s | 2.754 s | 0.870 | 16 | 2.603 s | **0.912** | 13 | 12 |
| 4096 | 2.365 s | 2.780 s | 0.845 | 16 | 2.555 s | **0.931** | 13 | 12 |
| 8192 | 2.230 s | 2.674 s | 0.830 | 16 | 2.551 s | **0.874** | 13 | 12 |

- 轮数 16 → 13（round reduction = 3）；wall-time reduction（no-bonus/bonus）= **1.058 / 1.088 / 1.048**；
- 每个节省轮的成本变化 = (bonus − no-bonus)/3 = **−0.051 / −0.075 / −0.041 s**（负值 = bonus 更快，不是"每省一轮多付开销"）。

逐 family（12 格中 9 格 <1，**3 格略 >1：F09/2048 1.011、F11/4096 1.029、F04/8192 1.021**，均在单格噪声量级）：

| family | ctx | AR | no-bonus | bonus | AR/bonus | rounds | cand/bonus | draft calls | target blocks | len |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F09 | 2048 | 2.597 | 2.764 | 2.568 | **1.011** | 16→13 | 52/12 | 64 | 16→13 | 64 |
| F09 | 4096 | 2.312 | 2.842 | 2.522 | 0.917 | 16→13 | 52/12 | 64 | 16→13 | 64 |
| F09 | 8192 | 2.188 | 2.638 | 2.633 | 0.831 | 16→13 | 52/12 | 64 | 16→13 | 64 |
| F03 | 2048 | 2.339 | 2.744 | 2.574 | 0.909 | 16→13 | 52/12 | 64 | 16→13 | 64 |
| F03 | 4096 | 2.398 | 2.881 | 2.537 | 0.945 | 16→13 | 52/12 | 64 | 16→13 | 64 |
| F03 | 8192 | 2.214 | 2.663 | 2.565 | 0.863 | 16→13 | 52/12 | 64 | 16→13 | 64 |
| F11 | 2048 | 2.342 | 2.672 | 2.654 | 0.882 | 16→13 | 52/12 | 64 | 16→13 | 64 |
| F11 | 4096 | 2.647 | 2.677 | 2.574 | **1.029** | 16→13 | 52/12 | 64 | 16→13 | 64 |
| F11 | 8192 | 2.246 | 3.071 | 2.536 | 0.885 | 16→13 | 52/12 | 64 | 16→13 | 64 |
| F04 | 2048 | 2.406 | 2.788 | 2.631 | 0.914 | 16→13 | 52/12 | 64 | 16→13 | 64 |
| F04 | 4096 | 2.333 | 2.718 | 2.613 | 0.893 | 16→13 | 52/12 | 64 | 16→13 | 64 |
| F04 | 8192 | 2.522 | 2.684 | 2.471 | **1.021** | 16→13 | 52/12 | 64 | 16→13 | 64 |

### 工作量分解（每条件中位）

| ctx | 模式 | rounds | draft candidate calls | draft bonus catch-up | draft 总调用 | target block calls | target 输入 token | target 额外单步 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2K/4K/8K | no-bonus | 16 | 64 | 0 | **64** | 16 | 64 | 0 |
| 2K/4K/8K | bonus | 13 | 52 | 12 | **64** | 13 | 64 | **0** |

**关键结构事实**：bonus 把 target block calls 从 16 降到 13（每次输入 s=4→5），但 **draft 总调用数保持 64**
（每轮多 1 次 bonus catch-up、少 1 个 candidate），所以 bonus 只节省 target 侧约 3 个 block 的开销；
这就是"轮数 16→13 但 speedup 只从 ~0.85 提到 ~0.87–0.93"的原因（见 §6 与 §7 的成本归因纪律）。

## 5. 图

`docs/figures/bonus_accept_oracle/bonus_latency_floor.{png,pdf,svg}`：
Panel A rounds（16→13）、Panel B 三模式 decode 时间、Panel C speedup（AR/no-bonus、AR/bonus，含 y=1 线）、
Panel D 工作量分解（draft candidate calls / draft bonus catch-up / target block calls / target block 输入 token 数）。
图中未混入跨批次 strict/relaxed 数据。

## 6. Outcome 判定

**Outcome A**：bonus-aware 在 2K/4K/8K 的 speedup 中位 0.912/0.931/0.874，**全部 <1**。
即：即使采用 k+1 提交并达到 100% acceptance，当前 1.5B/0.5B + k=4 + eager 执行结构**仍不能稳定超过 AR**。
（12 个 family-cell 中有 3 格 1.01–1.03，处于噪声量级、不足以构成"多个 context 稳定超过 1"的 C/D 结论；
无 0.95–1.0 的全 context 情形，故不是 B。）

## 7. 对旧结论的影响 / 必须的 erratum

旧结论"**即使 100% acceptance 也无法加速**"需要限定契约。证据：
- 无 bonus 契约（每轮 k 个 token）：all-accept speedup 0.849–0.860；
- **bonus 契约（每轮 k+1 个 token）**：0.874–0.931（同批 no-bonus 0.830–0.870），最接近 1 的格 1.029；
- 结论方向不变（仍无稳定加速），但因果表述必须区分两种契约，且 bonus 收益来源是 **target block 数减少**，
  不是 draft 工作减少。

需要修订的位置（建议，不修改历史文件）：
1. `docs/all_accept_latency_floor_report_2026-10-26.md`：把"即使 acceptance=100% 也无法超过 AR"
   改为"**在无 bonus-token 的当前契约下**，即使 acceptance=100% 也无法超过 AR；k+1 bonus 契约下
   speedup 提升到 0.87–0.93，仍无稳定 break-even"。
2. 项目 README / 中文阶段报告 / 论文故事草稿：凡引用"100% acceptance 仍加速不了"的地方，
   加上"（无 bonus 契约；bonus 契约实测 0.87–0.93）"的限定。
3. 保留两者区别：无 bonus 契约 = 每轮 k token（16 轮）；bonus 契约 = 每轮 k+1 token（13 轮、target block 13）。

## 8. 成本归因纪律

本轮报告只报告可测的量：
- `4d + v` 联合成本（不单方面声称 d≈0.9c；'v≈c' 未被独立测量）；
- no-bonus 与 bonus 的 target block 调用数（16→13）与 draft 调用数（64→64，含 12 次 catch-up）；
- 实测 decode 时间与阶段占比（诊断运行用 CUDA Event，含插桩同步开销，不进入 speedup）；
- 若需估 d/v，必须再做"只跑 proposal 链"与"只跑 verify 块"的独立计时，本轮未做。

## 9. 下一步唯一推荐动作

**转向 EAGLE 类 feature-level drafting（单模型多步/轻量 head 提议）**，其次才是算子级融合。
理由（基于本轮同批数据）：bonus 契约已把 target 侧调用压到 13 次 block、draft 侧仍固定 64 次单步前向；
即使接受率 100%，剩余差距主要来自"每提交一个 token 都必须经过一次 draft 前向"这一结构成本。
k+1 提交只能省 target 块、省不了 draft 步，因此**继续在 acceptance/bonus 契约上加码的空间已很小**；
要跨过 1.0 需要把 draft 侧 64 次前向降下来（feature drafting/融合执行），而不是继续调 acceptance 或 k。

## 10. 产物

- 代码：`experiments/relaxed_specdec/{bonus_accept_oracle,test_bonus_accept_cpu,gate_bonus_accept,run_bonus_accept,analyze_bonus_accept}.py`
  （新增；旧 `all_accept_oracle.py` / `run_all_accept.py` / `analyze_all_accept.py` 未改动）。
- 设计审计：`docs/bonus_accept_design_audit_2026-10-26.md`。
- 原始数据：`output/bonus_accept_oracle/gate/gate.json`、`output/bonus_accept_oracle/run/`（36 条件 × 3 reps 的
  `runs.jsonl`、逐条件 `result.json`、`analysis/`）。
- 聚合：`output/bonus_accept_oracle/run/analysis/{core_table.csv,family_table.csv,workload.csv,bonus_accept_summary.json}`
  （副本在 `docs/measurements/bonus_accept_*_2026-10-26.*`）。
- 图：`docs/figures/bonus_accept_oracle/bonus_latency_floor.{png,pdf,svg}`。
- 复现：`gate_bonus_accept --out ...` → `run_bonus_accept --out ...` → `analyze_bonus_accept --run ...`。

## 11. 局限

- 4 开发 family × 3 ctx × 3 reps；机制诊断，不作正式统计推断；仅覆盖本 GPU（RTX 4060 Laptop）、bf16、
  k=4、batch=1、eager decode、`cudnn_prefill` 仅 prefill 融合。
- bonus-aware 输出无质量意义；与 AR 长度同为 64（本轮无 EOS、无 length_mismatch），但"不同"不等于"质量下降"。
- catchup ≡ carry 的 bitwise 只在 11/12 成立；唯一例外是 0-gap 并列 argmax 上 1 个 bf16 ulp 的 kernel 形状差异，
  已如实记录（Gate 2 的 BITWISE/SEMANTIC 计数），不构成 cache 错位。
- 跨批次（Phase O / relaxed / 无 bonus 旧实验）的数据未混入本轮 speedup；如引用需单独标注批次。
