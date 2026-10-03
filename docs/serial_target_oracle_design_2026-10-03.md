# 串行 Target Verification Oracle 设计（2026-10-03，代码之前冻结）

> **唯一问题**：如果 target verification **完全**使用与 AR 相同的逐 token `M=1` `_step` 路径，
> SpecDec 的接受/拒绝/replacement/rollback/cache 管理能否生成与 target-only AR **完全相同**的 token IDs？
>
> **本文件在改代码之前写下**，用于：① 记录**现有代码的真实语义**（不凭经验猜）；② 冻结串行 verifier 的算法与 contract；
> ③ 冻结 Stage 0 的 13 个状态机用例与验收判据。
>
> **纪律**：只诊断、不进入任何性能实验；不改 acceptance/replacement/rollback 规则；不启用 row-aligned hook；
> 不做全模型 FP32；不继续 L2–L27 算子定位；不启动正式 B6/pilot/factorial。

## 1. 代码语义审计（以现有实现为准）

### 1.1 主循环 `spec_dec.spec_decode_greedy`

```python
tgt_lg, tgt_cache = _prefill(target, prompt_ids)     # 一次 _run(s=len(prompt))
drf_lg, drf_cache = _prefill(draft,  prompt_ids)
saved_next = tgt_lg[:, -1:]                           # ← 位置 pre-1 的行，预测位置 pre
gen = []
while len(gen) < max_new_tokens:
    k_eff = min(k, max_new_tokens - len(gen))          # ← 预算不足时 Kp 会变小
    pre   = spec_cache.cache_length(tgt_cache)         # ← 本轮开始时 target 已含的 token 数
    cands, drf_cache, prop_lg = _propose(draft, drf_cache, drf_lg, k_eff, eos_id)
    kp = len(cands)                                    # ← 可能 < k_eff（draft 提前出 EOS）
    rows, tgt_blk = _verify(target, tgt_cache, cands)  # ★ 本轮的**唯一** s>1 target 前向（block 模式）
    g = [argmax(saved_next)] + [argmax(rows[j]) for j in range(kp-1)]
    m = 最长前缀使得 g[i] == cands[i]（i < kp）
    if m < kp:                                         # ---- mismatch
        r = g[m]; emitted = cands[:m] + [r]
        tgt_cache = trim_cache(tgt_blk, pre + m)       # 丢弃被拒候选的 KV
        saved_next, tgt_cache = _step(target, r, tgt_cache)
        drf_cache = trim_cache(drf_cache, pre + m)
        drf_lg, drf_cache = _step(draft, r, drf_cache)
    else:                                              # ---- full accept
        emitted = cands; tgt_cache = tgt_blk
        drf_cache = trim_cache(drf_cache, pre + kp); drf_lg = prop_lg
        saved_next = rows[kp-1]                        # 位置 pre+kp-1 的行 → 预测 pre+kp
    gen += emitted
    if emitted and emitted[-1] == eos_id: break
```

* `_verify(target, cache, cands)` = 一次 `_run(embed([cands]), cache)`，返回 `(lg[0,-kp:], cache_blk)`；
  `cache_blk` 含**全部 Kp 个候选**的 KV。
* `_step(model, token, cache)` = 一次 `_run(embed([[token]]), cache)`（**`s=1`**），返回 `(lg[:, -1:], cache)`。

### 1.2 Invariant 表（**必须**在 Stage 0 逐一验证）

| 问题 | 现有实现的真实语义 |
| --- | --- |
| **target cache 长度对应哪些已输入 token** | `= prompt_len + 已提交(emitted)的 token 数`。round 开始 `pre = prompt_len + len(gen)`；mismatch 后 `pre+m+1`（含 replacement `r`）；full-accept 后 `pre+kp`。 |
| **`saved_next` 对应哪个位置的 logits** | round 开始时：位置 `pre-1` 的行（预测位置 `pre`）。mismatch 分支后：`_step(target, r)` 返回的行 = 位置 `pre+m` 的行（预测 `pre+m+1`）。full-accept 后：`rows[kp-1]` = 位置 `pre+kp-1` 的行（预测 `pre+kp`）。 |
| **emitted token 是否已写入 cache** | **是**。accepted 的 `cands[:m]` 由 `tgt_blk` 携带；replacement `r` 由本轮 `_step` 写入；full-accept 的 `cands` 全部在 `tgt_blk` 里。 |
| **mismatch candidate 是否曾写入 target cache** | `cands[m..kp-1]`（含被拒的 `cands[m]`）**曾**随 `tgt_blk` 写入，但被 `trim_cache(tgt_blk, pre+m)` **切掉**；`trim_cache` 返回**新列表**、不改原对象。`cands[m]` 被**丢弃**，写入的是 `r = g[m]`。 |
| **replacement token 何时写入 cache** | **本轮**：`_step(target, r, tgt_cache)` 与 `_step(draft, r, drf_cache)`。 |
| **full-accept 的额外 target token 如何处理** | `rows[kp-1]`（预测位置 `pre+kp` 的行）**本轮不 emit、不写 cache**，只作为下一轮的 `saved_next` ⇒ 下一轮的 `g[0]`；若下一轮 `cands[0]` 命中即被接受 ⇒ 该 token 实际在**下一轮** emit。 |
| **EOS** | `_propose` 遇到 EOS 立即 `break` ⇒ `kp < k_eff` 可能；主循环 emit 后若 `emitted[-1] == eos_id` 即停止。 |
| **截断** | `k_eff = min(k, max_new - len(gen))` ⇒ 最后一轮 `Kp` 可以 `< k`；`g` 只有 `kp` 项。 |
| **draft cache 与本轮的关系** | propose 阶段每提出一个候选就 append 一个（`s=1`）⇒ `cache_length(drf_cache) == pre + kp`；mismatch 时 `trim(pre+m)` 再 `_step(r)` ⇒ `pre+m+1`；full-accept 时 `trim(pre+kp)` ⇒ 与 `pre+kp` 相同（保留）。 |

**AR 的对应物**：`target.generate`（或 8K harness 的 AR 路径）每步 `_run(s=1)`、`argmax` 后**先记录再喂入**，
cache 长度 = `prompt_len + 已生成 token 数`，`saved_next` 与 emitted 一一对应。**串行 oracle 的目标就是让 verify 走同一条路。**

## 2. 串行 verifier 的算法与 contract

### 2.1 算法（逐 token，`s=1`）

```python
def _verify_serial(target, cache, cands):
    rows, c = [], cache
    for y in cands:                      # y1..yKp
        lg, c = _step(target, y, c)      # ★ s=1，与 AR 完全相同
        rows.append(lg[:, -1])           # 位置 pre+j 的行
    return stack(rows), c                # (rows, cache_blk) —— **与 _verify 同 contract**
```

* **不改主循环**：accept / replacement / rollback / emitted / EOS / 截断**逐字复用**现有实现；
* `rows[j]` 的语义与 block 版**一致**（位置 `pre+j` 的行），只有**数值路径**变成 M=1 ⇒ 这正是要验证的变量；
* `cache_blk` 与 block 版**语义一致**（append 了 kp 个候选），因此 mismatch 分支的 `trim(pre+m)` 仍能**精确**丢掉被拒候选 ✓；
* 不产生任何 `s>1` 的 target generation 前向 ✓；不安装 row-aligned hook ✓。

### 2.2 为什么 row-aligned 不是真 oracle

row-aligned 只是把**五类 GEMM** 的执行几何从 `M=4` 搬到逐行 `M=1`，**其余一切照旧**：
block verify 仍然是一次 `s=Kp` 的 `_run`（同一个 mask/softmax/AV/残差/MLP/cache 追加路径），
而 AR 走的是 `s=1` 的 `_step`。上一轮 4 个 family 里只有 1 个通过，说明**只对齐那五处不够**。
串行 oracle 换掉的是**整个 verifier**：target 在 generation 阶段的每一次前向都与 AR 同形（`s=1`）、
同一 dtype/cache 格式/RoPE/backend/argmax/`saved_next` 口径 ⇒ 它才是"控制逻辑 vs 执行路径"的**真隔离器**。

### 2.3 模式切换（默认关闭）

```python
spec_dec.set_target_verify_mode("block")    # 默认；生产路径逐位不变
spec_dec.set_target_verify_mode("serial")   # 仅诊断
spec_dec.target_verify_mode()
```
* 默认 `"block"` ⇒ 主循环那一行**原样**执行（不进入任何新分支）；
* `"serial"` 与 row-aligned（`_VERIFY_WRAP`）**互斥**：同时开启直接报错（禁止"内部先算 M=4 再切"）；
* 所有诊断输出带 `diagnostic_only=true`、`target_verify_mode="serial"`、`all_target_calls_m1=true`。

### 2.4 调用几何审计（硬失败）

在 generation 期间包装 `target._run`，逐次记录 `s = x.shape[1]`：
* **允许**：恰好一次 prefill 调用（`s == prompt_len`）；
* **generation 阶段**：每一次调用必须 `s == 1`，否则**立即抛错**（不得降级、不得重试）；
* draft 的调用单独记录（不受该断言约束）；
* 记录 `target_block_calls`（`s>1` 的 generation 调用数）必须为 **0**。

## 3. Stage 0：纯 CPU 状态机测试（13 例）

**方法**：用**脚本化 stub**（可控 logits）替换 target/draft 的计算，但**复用真实**的
`spec_dec.spec_decode_greedy` / `_step` / `_verify_serial` / `spec_cache` ⇒ 测的是**控制逻辑**本身。
stub 的 cache 用真实的 `[{k,v}]*L` 结构（token 维长度精确跟踪），因此 cache 长度/回滚可被逐轮核对。

| # | 用例 | 期望 |
| --- | --- | --- |
| 1 | `m=0`：第一个候选就被拒 | emitted = `[g0]`（replacement）；`m=0` |
| 2 | `m=1` | emitted = `[y1, r]`；`m=1` |
| 3 | `m=Kp-1` | emitted = `y1..y(Kp-1)` + `r` |
| 4 | `m=Kp`：全接受并产生 bonus | emitted = `y1..yKp`；`saved_next` = `rows[Kp-1]`；bonus 在**下一轮** emit |
| 5 | EOS 出现在 replacement | 本轮结束，`stop=eos` |
| 6 | EOS 出现在 accepted candidate | `Kp < k_eff`（propose 提前 break）且本轮结束 |
| 7 | `max_new` 剩余 < `K+1` | `Kp = k_eff < K`，emitted 长度受预算约束 |
| 8 | 最后一轮 `Kp < K` | 同上；`g` 只有 `Kp` 项 |
| 9 | mismatch 后被拒候选**不残留** | target cache 长度 = `pre+m+1`（只有 accepted + replacement） |
| 10 | draft cache 回滚到正确位置 | `= pre+m+1`；full-accept 时 `= pre+Kp` |
| 11 | target cache 与 emitted history 一致 | `cache_len == prompt_len + len(gen)` |
| 12 | 异常退出后 verifier 模式恢复 | 上下文异常 ⇒ `target_verify_mode() == "block"` |
| 13 | serial 关闭后生产路径逐位不变 | block 模式的 `_verify` 调用与历史一致（用 stub 的调用记录核对） |

每个用例核对：emitted tokens、`m`、target cache 长度、draft cache 长度、`saved_next` 所属位置、
replacement、round invariants（`tgt_len == drf_len == pre + (m+1 | Kp)`）、以及 **target generation 调用全部 `M=1`**。
**Stage 0 任一失败 ⇒ 不得运行 GPU。**

## 4. Stage 1 / Stage 2（GPU）

* **Stage 1（F13 定点）**：3 模式（AR / production / **serial oracle**）× 2 独立进程 = 6 runs。
  预期：production 复现历史 index 47 分歧；serial oracle 与 AR **64/64 完全一致**；两次一致；
  结构检查全过；**target generation 调用全部 M=1**。若 serial oracle ≠ AR ⇒ `SERIAL_ORACLE_FAIL`，
  保留首个分歧轮的完整 trace，**停止**、不做算子级定位。
* **Stage 2（protocol validation families）**：F01/F02/F08/F10 × {AR ×2, serial ×2} = **16 runs**
  （**不重跑** production 与 row-aligned，直接引用上一轮结果）。
* 配置与上一轮完全相同：ctx 8192 / chunk 2048 / k=4 / max_new=64 / BF16 / `base_bf16` / all-eager /
  target RoPE CPU-FP32 / greedy / 串行 / 不共享 cache / 不自动重试 / 每次独立进程。

## 5. 交叉检查（每轮）

1. **Round-boundary history check**：每轮开始记录 emitted prefix hash、target/draft cache 长度、
   `saved_next` 位置、`argmax(saved_next)`；serial oracle 的 token history 必须与独立 AR replay 的同前缀一致
   （cache 张量若因历史不同而有 BF16 差异，至少要求：token history 一致、cache 长度一致、每层结构一致、
   下一 token 预测与 AR `_step` 一致）。
2. **Target-call geometry audit**：所有非 prefill target 调用 `s == 1`；**禁止** block `_run(s>1)`、
   禁止 row-aligned wrapper、禁止"先算 M=4 再切第 0 行"。
3. **Control-logic trace**：每轮记录 `pre / Kp / proposals / g / m / branch / emitted /
   target cache before-after / draft cache before-after / replacement 或 bonus`。

## 6. 验收与解释纪律

* `SERIAL_ORACLE_PASS`：5 个 family（F13 + F01/F02/F08/F10）serial oracle 与 AR 的完整 token IDs **全部一致**；
  两模式两进程一致；全部 target generation 调用 M=1；轮不变量/cache/finite/合法停止全过；无 OOM；
  模式不作用于 production/draft/prefill；关闭后生产路径逐位不变。
* `SERIAL_ORACLE_FAIL`：任一 family token 不一致 / cache 或轮不变量失败 / 出现 M>1 /
  被拒候选残留 / replacement·bonus 提交错误 / 两次重复不一致。
* `CHECK_NEEDS_REVIEW`：字段缺失 / AR 自身不重复 / 环境不合格 / 无法证明全部 M=1 / 接口无法无侵入替换。
* **若 PASS**，只允许结论：**在 target verification 使用与 AR 相同的逐 token 执行路径时，
  现有 SpecDec 控制逻辑在这 5 个 family 上生成与 AR 相同的 token IDs；production block verification 的
  剩余差异可归入有限精度执行路径差异**。不得外推为所有输入的形式化证明。
* **若 FAIL**：说明问题不只在块 GEMM 形状，verifier contract / cache 提交 / 控制逻辑仍有未解决问题。
* **不得**用 margin/ULP 豁免；**不得**自行启动正式 B6/pilot/factorial。
