# Bonus-Aware Perfect-Accept Oracle 设计审计（2026-10-26，实现前）

> 性质：mechanism diagnostic / latency lower-bound oracle（bonus 契约版）；不是正式实验、不是可部署算法、
> 不是 lossless SpecDec、没有质量声明资格。基线：`codex/relaxed-specdec` @ `e9b6603`。
> 本审计先于任何 GPU 计时运行；确认后再实现与运行。

## 1. 现有（无 bonus）契约是什么

当前 all-accept oracle（`all_accept_oracle.py`）每轮：

```
draft 提出 c1..ckp；target 对 [c1..ckp] 做一次 block verify；
m == kp 分支：emitted = candidates；tgt_cache = 本轮 block cache；
draft cache trim 到 pre+kp；saved_next = rows[kp-1]（**未被提交**）；
drf_lg = prop_lg。
```

因此每轮只提交 `kp ≤ k` 个 token，而 `saved_next = rows[kp-1]`（target 对"最后一个 candidate 之后"的
下一 token 预测）只被保留为下一轮的第一个判定行，**没有作为输出提交**。64 token / k=4 ⇒ 16 轮。
证据：`docs/measurements/all_accept_core_table_2026-10-26.csv`（rounds=16，bonus count=0，speedup 0.849–0.860）。

## 2. Bonus token 从哪一行来

一次 verify 的输入是 `[pending_bonus?] + [c1..ckp]`（长度为 L，见 §5）。target 前向返回
`rows = lg[0, -L:]`，其中：

- `rows[i]` = 消费到块内第 i 个输入 token 之后、位置 `pre+i` 的 logits；
- `row[0]`（若含 pending bonus）= 消费 pending bonus 后、预测 c1 的 logits；
- `row[L-1]`（= 最后一个 candidate 之后）= 预测"c_kp 之后第一个 token"的 logits。

**bonus = argmax(rows[L-1])**。对首轮（无 pending）`L=kp`，bonus = argmax(rows[kp-1])，与
无 bonus 契约里的 `saved_next` 完全相同——即 bonus token 本来就是该行的 argmax，无 bonus 契约只是没有发出它。

## 3. Bonus 发出后 target cache 为何尚未包含 bonus

target 本轮 block 只消费到 `c_kp`（pending 若有，也已在本轮块首进入 cache）。bonus 是从
`rows[L-1]` 的 argmax 得到的**预测**，并不是该次前向的输入，因此 target cache 长度 = ctx + committed − 1
（committed 是已提交 token 数，含本轮 bonus）。这是一个**明确的 pending 状态**，不是错位。

## 4. Draft cache 为何也必须消费 bonus

bonus 被提交后，下一轮 draft 的候选必须基于"history + c + bonus"这条正确前缀生成；否则 draft 会基于
错误历史提出候选（token 语义错误）。因此下一轮开始时 draft 必须先做一次 **catch-up 单步**（消费 bonus），
再正常 propose k 个 candidate。该调用单独计数（`draft_bonus_catchup_calls`），不与 candidate 调用混在一起。

## 5. 下一轮 target 如何在正确历史上验证新 candidate

target cache 比 draft/committed 少一个 bonus。下一轮 verify 输入：

```
[pending_bonus, c'_1, ..., c'_kp]        （L = kp + 1）
```

一次 block forward 后：

- `row[0]`：消费 pending_bonus 后的 logits（预测 c'_1）——同时让 target cache 纳入 bonus；
- `row[j]`：消费 pending_bonus + c'_1..c'_j 后的 logits（预测 c'_{j+1}）；
- `row[L-1]`：消费全部 candidate 后 → 下一个 bonus 的 argmax；
- target cache 最终 = ctx + committed_new − (新 bonus 是否发出)。

即：**target 每轮仍保持一次 block call**；首轮 L=kp，后续轮 L=kp+1；没有额外的 target 单步。

## 6. 为什么不能只在 gen 里追加 bonus 而不更新 cache

gen 是"已提交输出"，cache 是"模型已消费的历史"。若只追加 gen 不更新 cache：下一轮 target 的 verify 会在
**缺少 bonus** 的历史上计算 logits（位置错 1、rope 位置错 1、KV 内容少 1 个 token），cache 与 committed 永久错位，
后续所有 logits 都是错的；draft 同理。因此 bonus 必须在下一轮被两侧 cache 消费（或在本轮显式追赶）。

## 7. 为什么不用"每轮额外调用一次 target _step(bonus)"作为计时主路径

这样虽然正确（cache 立即对齐、下一轮 verify 只含 candidates），但它**额外增加一次 target 单步调用**：
all-accept 每轮本来只有 1 次 target block（s≈kp），多一次单步 ≈ 多 25% target 侧调用，会系统性抬高
bonus-aware 的耗时，无法测量"bonus 减少轮数"的潜在收益。因此：

- **correctness reference（explicit catch-up）**：全部接受 → bonus = argmax(rows[-1]) → emit candidates+bonus
  → target `_step(bonus)` → draft `_step(bonus)` → 两侧 cache 对齐；仅用于正确性对照，**不进计时**；
- **timed path（pending-bonus carry-forward）**：上游 carry-forward，target 不额外调用单步；
  1 次 block call/轮（首轮 s=kp，之后 s=kp+1）。

## 8. Pending-bonus carry-forward 的不变量（写入测试与文档）

记 `committed` = 已提交（gen）token 数，`pending ∈ {0,1}` = 上一轮 bonus 尚未被两侧消费：

| 时刻 | 不变量 |
| --- | --- |
| 轮首（有 pending） | draft_len = ctx + committed − 1；target_len = ctx + committed − 1 |
| 轮首（无 pending） | draft_len = target_len = ctx + committed |
| 轮末（发出 bonus） | draft_len = target_len = ctx + committed − 1（pending = 1） |
| 轮末（未发 bonus） | draft_len = target_len = ctx + committed（pending = 0） |
| 全过程 | `target_len + pending == draft_len + pending == ctx + committed − pending`；同步执行时两侧始终同长 |

每轮记录：verify_input_length（L）、pending_bonus_included、candidate_start_row（=pending）、
next_bonus_row（=L−1）、candidate_count、bonus_count、emitted_count、remaining_before/remaining_after、
target/draft cache 长度。

## 9. Token 预算与轮数（max_new=64, k=4）

- 每轮最多提交 kp + 1（首轮 5），直到剩余预算不足：
  60 = 12×5，之后剩余 4 ⇒ 第 13 轮 candidate 4 个、**不再发 bonus**；总轮数 **13**；
- 预期累计：candidate tokens = 12×4 + 4 = **52**，bonus tokens = **12**，合计 **64**；
- 最后一轮 `remaining ≤ k` 时只生成 remaining 个 candidate；**不得**先发 bonus 超过 64 再裁剪；
- 若 EOS 出现（candidate 或 bonus），按 §10 提前停止并记录实际长度；与 AR 长度不同则单独标记 length_mismatch。

## 10. EOS 契约

1. candidate 中出现 EOS：proposal 在 EOS 处截断（kp = 首个 EOS 位置 +1），接受后 `emitted[-1]==EOS` → 停止，
   **不发 bonus**（即使预算允许）；
2. bonus 本身是 EOS：发出 bonus（它已是 EOS）→ 停止，不再开始下一轮；此时 pending=1，目标 cache 比 committed 少 1，
   作为设计状态记录（不视为错位）；
3. 最后一轮预算不足：不生成 bonus；
4. 两种模式下，target/draft 最终 cache 与"实际已提交 token + pending 状态"必须一致（§8 不变量）。

## 11. 实现与门禁计划

新增（不改旧文件）：`bonus_accept_oracle.py`（carry / catchup 两种路径）、`test_bonus_accept_cpu.py`（Gate 1）、
`gate_bonus_accept.py`（Gate 2：catchup vs carry 逐 token 一致 + 不变量 + finite/cache/attention）、
`run_bonus_accept.py`（同批 36 条件：AR / no-bonus / bonus）、`analyze_bonus_accept.py`。

GPU 前的门：
- Gate 1 CPU：15 条清单（首轮无 pending、emit candidates+bonus、draft catch-up、target 块首含 pending、
  bonus 不重复输出、row 映射、13 轮、尾轮不发 bonus、candidate EOS、bonus EOS、max_new 截断、
  target/draft cache 长度、最终 cache、catchup==carry 一致、旧模式不回归）；
- Gate 2 GPU：4 families × 3 ctx 非计时，`catchup` 与 `carry` 的 token/rounds/bonus/stop 全等，
  rounds=13、bonus=12、total=64、acceptance=1、rollback/repair/mismatch=0、finite/cache/attention 全过。

**本审计确认后才运行 GPU 计时。**
