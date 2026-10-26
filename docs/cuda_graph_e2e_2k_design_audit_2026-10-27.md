# 2K 端到端 CUDA Graph 兑现门：设计审计（2026-10-27，实现前）

> 性质：`experiment_class = end_to_end_graph_feasibility_diagnostic`；`formal_result_allowed=false`；
> `quality_claim_allowed=false`；`production_claim_allowed=false`；`scope = F09/2K/bonus-all-accept/64-token`。
> 只回答：固定 shape 局部 graph 的收益能否在真实双模型、真实 13 轮 cache 推进、完整 64-token bonus-aware
> 循环中兑现并稳定超过 AR。**不扩展 4K/8K，不跑正式 family，不接 EAGLE。**
> 基线 `codex/relaxed-specdec` @ `f2ddb22`；不改正式 runner / strict / relaxed / bonus oracle 历史产物。

## 1. bonus-aware 的 13 轮，各轮 history 不同

F09/2K、k=4、max_new=64、carry-forward 契约（`bonus_accept_oracle`，已实测）：
- round r 的 draft 链：round0 = 4 步（无 catch-up，从 prefill logits 起步）；round r≥1 = catch-up(1) + 4 步；
- round r 的 target verify：round0 = s=4（无 pending）；round r≥1 = s=5（pending bonus 作首 token）；
- 轮起点的 draft/target cache 长度：

| round r | draft 起点 | draft 链长 | target 起点 | verify s | 提交 |
| --- | --- | --- | --- | --- | --- |
| 0 | 2048 | 4 | 2048 | 4 | 4c+bonus |
| 1..11 | 2048+5r−1 | 5 | 2048+5r−1 | 5 | 4c+bonus |
| 12 | 2048+59 | 5 | 2048+59 | 5 | 4c（无 bonus） |

⇒ 每轮 (history, s) 都不同，共 13 个 draft 链 shape + 13 个 verify shape。

## 2. 当前 graph 只能绑定固定 (cache 地址, history, position, RoPE 表, mask/shape)

`_run` 的 RoPE 由 `start_pos=cache_len`（Python int）固化、mask 由 `(s,T)` 固化、KV append 的 cat 输出地址在
capture 时固定。因此**每个 (history, s) 必须单独 capture**；一个 graph 不能跨轮复用。

## 3. 为什么不能每轮现场 capture

一次完整 64-token 生成的 decode 只需 ~2.6 s；实测单个 graph capture 0.27–1.1 s（局部 A/B）。
13 轮 × 26 个 graph 若放在请求内 capture ⇒ **≥ 7–28 s**，远大于被优化对象。且 capture 期间会破坏
steady-state 计时语义。⇒ capture 必须在计时前一次性完成，measured repetition 只 replay。

## 4. 为什么不能把参考 token 写死

把预先记录的 64 个 token 填进 graph 输入只是"重放静态轨迹"的微基准，不是端到端兑现：
它跳过了"本轮输出 → 下一轮输入"的真实数据流，也绕过了 token 决策必须由 GPU 产生的约束。
（本轮基线 eager 也会用真实 argmax 产生 token；graph 路径必须同样由 GPU argmax 产生。）

## 5. 为什么不能 CPU 读取 argmax 再回填

CPU 读 argmax 需要 `.item()`/同步，会把 host 逻辑放回热路径、破坏"launch 消除"的测量目标，并且
新增同步会把 graph 收益掩盖。token 决策必须在图内完成（`argmax → embedding` 的 GPU 链路），
跨轮只允许 GPU `copy_` 搬运。

## 6. 13 个 graph 能否共享 graph pool

PyTorch 允许同一 pool 被多个 graph 共享，但要求这些 graph 的生命周期一致；本轮**全部 26 个 graph 同时存活
到测量结束**，因此理论上可共享。实测风险：上一轮 CUDA Graph A/B 尝试"共享 pool + 逐个释放"时触发
`it->second->use_count > 0` 内部断言。本轮采用**共享 pool 且不提前释放**的策略试一次；若断言/显存失败，
退回独立 pool，并用显存账本判定是否 Outcome C。

## 7. 两模型 graph 如何交替 replay

每轮顺序：`chain_r.replay()` → `verify_r.replay()`。两者绑定各自的 big buffer 与静态输入；交替 replay
没有地址冲突（不同 graph、不同 buffer 段）。CPU 只做 Python 调用（固定 13 轮顺序），不做 token 决策。

## 8. 上一轮输出如何成为下一轮真实输入

在 GPU 上完成三类 handoff（都不经过 CPU）：
- candidate token：chain 图内把每步 argmax 写入 `cand_buf`（静态张量，图内 `copy_`）；
- bonus token：verify 图内把最后一行 argmax 写入 `bonus_buf`（图内 `copy_`）；
- cache：`_run` 的 cat 输出在图内 `copy_` 回**大容量静态 KV buffer** 的对应切片；下一轮图的输入切片的
  指针落在同一 buffer 上 ⇒ 下一轮 replay 直接读到新内容。
verify 的输入 token（[pending bonus] + 4 candidates）用 `verify_tok.copy_(bonus_buf/cand_buf)` 在 GPU 上拼装
（图外、记录为 static-copy 时间）。

## 9. Target/draft cache 如何真实推进

预分配 big KV buffer（容量 2048+64+k=2116 tokens/层）：
- 预填 prefill 的真实 cache（`copy_` 进 buffer 的 [0:2048]）；
- 每轮图内的 `_run` 读 buffer 切片 `[0:L]`，cat 产生新张量后在图内 `copy_` 到 `[L:L+s]`；
- 下一轮图读 `[0:L+s]`。
全程只有 GPU kernel（含 copy_），无 CPU 回填。

## 10. Graph 总显存如何计算

- big KV buffer：target 每 token ≈28.7 KB（28 层×2×2 heads×128 dim×2B）×2116 ≈ 61 MB；
  draft 每 token ≈12.3 KB ×2116 ≈ 26 MB；静态输入/输出（token、logits 输出、candidates、gen buffer）几 MB；
- graph pool：每个 graph 的峰值临时量（kv_repeat 物化 + scores/softmax）≈ 每层 ~32–65 MB（target verify）
  与 ~4–9 MB/层（draft step），但层间可复用；估算 26 个 graph 合计 ~1.5–3 GB；
- 总账本 ≈ target weights 3.05 + draft weights 1.0 + buffer 0.09 + pools 1.5–3 + 预填/参考 cache 0.3 ≈ 6–7.5 GB。
若 capture 后 `mem_get_info` 物理余量不足（<200 MiB）或 capture OOM：**Outcome C（memory-blocked）**，
不提高护栏、不改显存约束。

## 11. Capture 是否可跨 measured repetition 复用

可以：capture 在计时前完成一次，之后 5 次 measured repetition 只 replay。capture 时间单独报告
（first-request latency = capture + 首次 64-token replay）。

## 12. Capture 是否可跨 prompt 复用（本轮不要求实现）

不可直接复用：prefill cache 内容不同，graph 输入 buffer 的**内容**可以 `copy_` 替换，因此同一组 13 轮 graph
在**同一 ctx/同一口令模板**下可复用；跨不同 ctx（历史长度不同）不能复用（shape 固化）。
本轮只验证单 prompt 的复用（fixed F09/2K），不实现跨 prompt 调度。

## 13. 失败时的停止条件

- eager 与 graph 的 64 token / 每轮 candidate+bonus / 13 轮 / cache 长度或数值出现 **token 不一致** ⇒ 停止（Outcome E）；
- 出现 NaN/Inf、argmax margin 变化、cache 错位 ⇒ 停止；
- capture OOM / 13 轮 graph 无法共存 ⇒ Outcome C（memory-blocked），不提高护栏；
- 只能用预录 token 才能跑通 ⇒ Outcome F（微基准，不输出系统结论）；
- 以上任一发生：保留全部失败产物，不静默重试、不改参数。

## 实现结论：选路线 A

采用"**13 个链 graph + 13 个 verify graph + 单组大容量静态 KV buffer**"的路线 A：
- 26 个 graph 一次 capture，全部存活到测量结束，共享 pool（失败则独立 pool，按显存账本判定）；
- 每轮 replay 顺序固定，token/cache/bonus 全部由 GPU 产生并 GPU 搬运；
- 参考 eager（`bonus_accept_oracle` carry）作逐 token/逐轮对齐基准；
- 不实现路线 B（静态容量 + padding 的 model_self 重构）——本轮 A 已覆盖固定 2K/64-token 场景；
  若 A 因显存/断言失败，先按 Outcome C 报告，不在本轮扩大 scope 到 B 的模型重写。
