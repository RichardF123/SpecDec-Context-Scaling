# G4 —— Fixed-chain Greedy SpecDec 设计冻结（DRAFT v0.3，2026-09-15）

> 状态：**DRAFT v0.3**（v0.2 基础上修正"替换位需单步重算"的回滚语义，见 §3.1）。
> 范围：只做 **correctness**（exact greedy equivalence）；不运行正式矩阵、不宣称加速、不做
> kernel/sampling/beam/INT 量化/改 k/新模型。批准后 v1.0；批准前不实现主循环。

## 1. 冻结参数（第一版）
| 项 | 值 |
|---|---|
| target | Qwen/Qwen2.5-1.5B-Instruct（engine OURS，BF16 GPU 计算）|
| draft | Qwen/Qwen2.5-0.5B-Instruct（engine OURS）|
| proposal length | **k = 4（固定）**；**第一版无 bonus token**（全接受时本轮产出恰好 k 个，下一轮重新提议）|
| decoding | greedy（argmax；无采样接受规则）|
| batch | 1 |
| KV 条件 | `base_bf16`（K/V BF16）与 `v_fp8`（K BF16 + V FP8 E4M3 gs128）——同一份 SpecDec 逻辑，KV 条件仅作参数 |
| 判定 | SpecDec ids == target-only greedy ids（exact equality；不只比较解码文本）|

## 2. 输入一致性与词表（前置事实，测试须验证）
- 0.5B draft 与 1.5B target 使用**同一个 tokenizer**：词表/特殊 token/raw-text 编码完全一致（两者都是 Qwen2.5，151936 词表）；
- chat template / BOS / EOS / raw prompt 处理在两条生成路径（target-only 与 SpecDec）中**逐位一致**：以同一 tokenizer 默认策略编码同一 prompt，不额外加/去 BOS；
- 因此 draft 的提议 token、target 的验证 token 都来自同一 id 空间，接受比较是**同 id 比较**。

## 3. correctness 语义与 logits 对齐（v0.2 修正后；off-by-one 必须按此实现与测试）
设本轮开始时 prefix 已处理 L 个真实 token。

1. **进入本轮前保存** target 最后位置的 logits：`target_next_logits`（预测 prefix 之后下一个 token，用于验证 d1）。
2. draft 自回归提出至多 k=4 个 token：`d=[d1..dk']`（k'=min(k, budget)；draft cache 相应推进；遇 EOS 停提议）。
   **不变量（实现层，2026-09-15 修订）**：提议结束时 draft cache 长度**恒为 `L+k'`**——EOS
   检查必须在"推进 cache"**之后**：若在之前 break，EOS 候选的 K/V 不写入，长度变成 `L+k'-1`，
   则 m=k'（全接受，即生成正常以 EOS 收尾）时 §5 的 `trim_cache(drf, L+k')` 会越界抛
   `ValueError`。同时 `draft_calls += k'` 才与实际 draft 前向次数精确相等。
3. target 对 `prefix + d` 做**一次 block forward**，得到 block logits 行 `b_0..b_{k'-1}`——**第 i 行预测"输入 d_i 之后"的下一 token**。
4. **验证对齐（防 off-by-one）**：验证列表 = `[saved_target_next_logits, b_0, ..., b_{k'-2}]`：
   - 第 j 个 logits（j=1..k'）判定 d_j → saved 判 d1，b_0 判 d2，…，b_{k'-2} 判 d_{k'}；
   - 各 logits 的 argmax 构成 target-greedy 序列 `g1..gk'`。
   - **红线：b_{k'-1} 不参与判定**（它预测 d_{k'} 之后，判 d_{k'} 会错位）。
5. **接受/替换**：m = g 与 d 连续相等前缀长（0≤m≤k'）；
   - m<k'：接受 d1..dm；替换 token `r = g_{m+1}`（target token，本轮真实产出 m+1 个）；
     **替换位必须由 target 单步重算**（见 §4 修正）：block 里位置 m+1 的 K/V 是"被拒 draft token"的，
     不能当作 r 的 K/V。
   - m=k'（全接受）：**第一版无 bonus**，本轮输出 k' 个（=d，已验证 g==d），结束本轮，下一轮重新提议。
6. **EOS**：被输出/接受的 token 为 EOS → 输出后终止（记录生成长度）；draft 提议遇 EOS 停提议（k'<k）。
   EOS 候选本身照常写入 draft cache（不得破坏第 2 步的不变量）；若 m<k'（EOS 候选被拒），
   走 §5 的错配分支，两版实现行为一致（已由 `g4_b5_eos_branch_tests.py` 对照验证）。
7. **budget**：剩余 `max_new_tokens` 小于 k → k' = 剩余数。
8. `b_{k'-1}` 仅当 m=k' 时才有意义（"继续下一个 token 的分布"）；第一版不消费它（无 bonus），下一轮用新提议覆盖。

### 3.1 替换位单步修正（v0.3，2026-09-15；修正 v0.2 的指导错误）
- **问题**：block forward 的输入是 `[d1..dk']`，故 target cache 位置 `L+j` 存的是 `d_j`（draft 猜的 token）的 K/V。
  若 `m<k'`，位置 `L+m+1` 存的是**被拒的 draft token** 的 K/V —— 直接保留到 `L+m+1` 会让这一格内容错配，
  且 `b_m`（输入是被拒 token）预测的是"被拒 token 之后"，不能当下一轮的 `saved_next`。
- **正确做法**（m<k'）：target `trim` 到 `L+m` → **单步前向吃替换 token r** → 该步同时(a)写入 r 的正确 K/V、
  (b) 把长度推进到 `L+m+1`、(c) 产出"r 之后"的 logits 作为下一轮 `saved_next`。
  draft 对称：`trim` 到 `L+m` → 单步吃 r（上下文与真实前缀一致，下一轮可直接提议）。
- **每轮代价**：有拒绝轮 = 1 次 block + 1 次 target 单步 + k' 步 draft + 1 步 draft；全接受轮 = 1 次 block + k' 步 draft。

## 4. cache 不变式与回滚（修正后）
- **不变式：cache 长度 ≡ 该模型已处理的"真实" token 数**；每轮结束时 **target 长度 == draft 长度 ==**
  `L + (m+1 if m<k' else k')`。
- target：`m<k'` → `trim(tgt_blk, L+m)` + 单步吃 r（→ `L+m+1`）；`m==k'` → 保留 block 结果（`L+k'`，全是真实 token）。
- draft：`m<k'` → `trim(drf, L+m)` + 单步吃 r（→ `L+m+1`）；`m==k'` → `trim(drf, L+k')`（= block 提议后长度，无操作）。
- 裁剪作用于该 cache **全部 token 张量、同一 dim=2**：base_bf16 → `k`,`v`；v_fp8 → `k`,`v`,`v_scale`；任何时刻必须相等，否则判 bug。
- 真实 token 永不被丢（长度单调不减）；rollback = 回到真实段。
- `m=0`：target 裁到 `L` 后单步吃 r → `L+1`；draft 同样 `L` → `L+1`。

## 5. cache API（独立模块 `specdec/engine/spec_cache.py`，不并入 model_self）
- `cache_length(cache)` → `cache[0]["k"].shape[2]`（两 KV 通用）。
- `trim_cache(cache, new_len)`：每层每张量 `t[:, :, :new_len]`（v_scale 同 dim=2）；越界（new_len<0 或 >当前）抛 ValueError（不静默钳制，测试断言抛错）；None → None。
- 独立测试（B4）覆盖：append 长度、trim 到 L、trim 1/多 token、trim 到 0（越界）、mismatch 第 1..4 位、base 与 v_fp8 都能 trim、v_scale 同步、token 维不一致即 fail。

## 6. greedy SpecDec 主循环（独立模块 `specdec/engine/spec_dec.py`）
每轮按 §3.1–§3.8；记录（behavioral）：proposed、accepted、target block calls、draft calls、mismatch 轮、全接受轮、rollback 次数。复用 OURS engine 双实例（target/draft）与 §5 cache API；不做无关重构。

## 7. 资源/OOM 纪律
- 双模型实例创建后 `try/finally` 清理：异常路径 `del` 模型 + `torch.cuda.empty_cache()`；
- 实现前先过 **B3 dual-model 8GB 可行性**（实测，不作理论断言）；放不下 → 停 G4 实现并报 blocker，不加权重量化改协议。

## 8. 验证顺序与门槛
- **G4-0**：0.5B draft correctness（full vs HF bf16 / cached vs full / greedy 生成一致 / finite / 权重 strict）——失败先定位，不进双模型。
- **B3**：双模型 8GB 可行性（base 与 v_fp8、短+8K、decode smoke、峰值、3 轮稳定）。
- **B4**：cache 操作测试（§5）。
- **B5**：最小 greedy SpecDec（§6）。
- **B6**：correctness gate——prompts：自然语言/数学/代码/短/长/全接受/首位拒绝/中途拒绝/EOS/非 4 倍数长度；条件 base_bf16 与 v_fp8；判据 exact ids equality + 全 finite + 多次确定性一致。**B6 未过禁止 acceptance 测量/pilot/final。**
- （通过后的小规模预实验属 Part 4，先不启动。）

### 8.1 G4-0 登记（2026-09-08，由 g4_0_drift_diag / g4_block_vs_serial 实测归因）
**G4-0: CONDITIONAL PASS — BF16 shape drift characterized**
- ENGINE BUG: no（OURS full == HF full、OURS cached == HF cached，max|Δ|=0 on tested cases）
- NONDETERMINISM: no（所有路径重复运行确定一致）
- 0.5B OURS full vs HF full: exact on tested cases；cached vs HF cached: exact on tested cases
- BF16 cached vs full-recompute: 在 top-1/top-2 接近处可能改变 argmax（失败样本: "...history of artificial
  intelligence..." 生成第 8 token cached=' for' / full=' since'）
- FP32 control: cached/full equality restored
- Target block-vs-serial probe: 被测验证位置 argmax 全一致、cache 终长一致、未发现 off-by-one；
  **但 block-built 与 serial-built cache 数值不完全相同（max|Δ| 最高 ~2.0）** → 多轮仍可能因 BF16
  cache 数值漂移翻 token ⇒ **B6 exact-token-id 门不可取消，且为 B6 保留风险**；不得写成"B6 已无风险"。
- 原始失败数据保留于 ledger（不删除）。

## 9. 范围排除（本轮禁止）
正式 12-condition 矩阵 / "SpecDec 加速"宣称 / CUDA kernel / sampling/beam / 动态 k 或改 k / INT 权重量化 / K-FP8 / 跨机器数据混合 / bonus-token 状态（第一版）。

## 10. 批准栏
- [ ] 我审阅并理解 §1–§9 —— 批准 v0.2 为 v1.0（冻结）
- [ ] 修改意见：＿＿＿＿＿＿

---
*变更记录：v0.1（DRAFT，2026-09-08）：初稿（含 bonus）。v0.2（DRAFT，2026-09-08）：Part-2 审查修正——取消 bonus；明确 logits 对齐/off-by-one（saved target_next_logits 判 d1；验证列表 [saved, b_0..b_{k'-2}]；b_{k'-1} 不判定）；输入/template 一致性；cache trim/rollback 精确语义；越界抛错；OOM finally 清理；base 与 v_fp8 共用同一 SpecDec 逻辑。**v0.3（2026-09-15）：修正替换位回滚语义**——`m<k'` 时 target 必须 `trim` 到 `L+m` 再**单步重算替换 token r**（block 位置 L+m+1 的 K/V 属被拒 draft token，不能复用）；draft 对称裁到 `L+m` 后单步吃 r；每轮结束 **target 长度 == draft 长度 == L+(m+1 或 k')**。此修正替代 v0.2 中"保留到 L+m+1、saved_next=b_m"的写法。*
