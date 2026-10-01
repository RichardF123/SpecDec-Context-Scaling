# 统一执行几何（UCG）最小原型设计（2026-10-01）

> 目标优先级仍是**严格 AR/Spec 输出一致性**，**不放宽门槛**。
> 本文是 GPU 执行**之前**的设计交付；原型只放在**独立诊断入口**，不改生产默认 backend/精度/proposal/门槛。

## 0. 候选一句话

**把 target 的每一次前向都统一成"4 行块形状"**：
- AR 每步喂 `[t, pad, pad, pad]`，只取 **row 0**，随后 `trim_cache(blk, pre+1)`；
- Spec 的 **verify 本来就是 s=4**（不变）；**repair 改成 `[r, pad, pad, pad]` + trim**；
- **尾轮**（`k_eff < k`）把候选 **pad 到 4** 个，取前 `k_eff` 行的 logits，trim 到实际提交数。

⇒ 两臂对**相同 token 历史**的每一次 target 前向，`s`、`T`、mask 对角线、LM head 的 M 维**全部相同**。

## 1. 要统一哪些操作 / 哪些仍可能因形状不同而变化

| 环节 | 现状（会分歧） | UCG 后 |
| --- | --- | --- |
| target 单步（AR decode / Spec repair） | `s=1`，`T=pre+1`，**不施加 mask** | `s=4`，`T=pre+4`，mask `triu(diag=pre+1)` |
| target 分块（Spec verify） | `s=Kp`（可能 <4），`T=pre+Kp` | `s=4`（尾轮 pad），`T=pre+4` |
| QK/softmax/AV 的形状 | 随 `s`、`T` 变化 ⇒ **bf16 归约顺序不同** | **固定** |
| LM head 的 M 维 | `M=1`（单步）vs `M=Kp`（分块） | **固定 `M=4`** |
| KV 提交 | 单步提交 1；verify 提交实际候选 | **不变**（pad 行的 KV **一律 trim 丢弃**） |
| **draft 模型**的调用形状 | `s=1`（propose） | **不统一**（draft 只产生候选；其数值不进入 target 的 logits） |
| prefill | 两臂**都已**是 chunk=2048 分块（同一实现） | 不变 |
| padding 的 RoPE 位置 | — | pad 行位于 `pre+1..pre+3`，row 0 看不见（mask），其 KV 不提交 |

**仍然"可能因形状不同而变化"的**：draft 的 propose（不影响 target 数值，但影响"候选是否被接受"）；
以及**同一形状下不同 kernel 启发式**（若 cuBLAS 依据对齐/步长选不同 kernel，仍可能有差 —— 这正是要实测的）。

## 2. 如何从 prefill 之后就开始维护历史（不只修最后一个位置）

**从 prefill 之后的第一步起**：AR 的**每一步**都用 s=4（而不是只在最后一步），因此
**所有被提交位置的 K/V 都由"同一几何"的调用产生** ⇒ 历史不再分叉（这是与"只修一个位置"的本质区别）。
`trim_cache` 保证 pad 行的 K/V 永不进入有效 cache，有效长度 = `pre + 提交数`。

## 3. 接受 / 拒绝 / rollback / repair / EOS / 尾轮不足 k

生产循环 `spec_dec.spec_decode_greedy` **不改**，只在诊断模式下包两层：

* `_verify_padded(model, cache, cands)`：把 `cands` pad 到 4 → 调用**真** `_verify` → 只回传前 `len(cands)` 行
  + 未 trim 的 `blk`（生产随后按 `pre+m` trim，语义不变）；
* `_step_padded(model, token, cache)`：pad 成 4 行 → 同样的块调用 → 取 **row 0** 作为 `(1,1,V)` logits →
  `trim_cache` 到 `pre+1` ⇒ 与生产的 `_step` 返回值形状/语义一致。

于是：**接受**（m=kp，全接受）、**拒绝/rollback**（m<kp，`trim_cache(blk, pre+m)` + repair）、
**repair**（走 `_step_padded`）、**EOS**（沿用冻结规则：EOS 检查在推进之后）、
**尾轮 `k_eff<k`**（pad 到 4，只取前 `k_eff` 行）全部由**生产逻辑**处理，原型不重写状态机。

## 4. padding 语义（mask / 不提交 / position / 有效长度）

* **哪些位置被 mask**：pad 行对应位置 `pre+1..pre+3`；块 mask 为 `triu(diag=(T-s)+1)=triu(diag=pre+1)`，
  即 row 0 只能看见 `key ≤ pre` ⇒ **row 0 看不到任何 pad 行**（公式推导；本轮将用**实测**补充验证）。
* **哪些 KV 不得提交**：pad 行的 K/V **一律不提交**（`trim_cache(blk, pre + 提交数)`）。
* **position/RoPE**：pad 行使用 `pre+1..pre+3` 的位置编码（由 `_run` 的 `start_pos` 决定）；
  由于 row 0 不读它们，**对 row 0 的结果无影响** —— 这一点必须**实测**（见 §7 的 padding 不变性测试）。
* **有效 cache 长度**：始终等于 `prompt + 已提交 token 数`，运行中逐步校验。

## 5. 是否改变权重 / 精度 / 算法 / 实际工作量

| 项 | 变化 |
| --- | --- |
| 权重 | **不变**（同一模型、同一 bf16 权重） |
| 精度 | **不变**（BF16；`qk_fp32=prob_fp32=False` 不动） |
| 算法语义 | **不变**（同因果 mask、同 argmax 规则、同接受/rollback 规则、同 EOS 规则） |
| **实际工作量** | **AR 侧显著增加**：每 token 的 target 前向由 `s=1` 变成 `s=4`（约 **4×** target 计算），
  但**只提交 1 个位置**；Spec 侧仅 repair 由 `s=1`→`s=4`（repair 占比小），verify 不变 |
| 显存 | **不增加**：`s=4` 与现有 verify 同级；pad 行随 trim 释放 |

⇒ **这是"正确性控制"配置，不是加速修复**。若每个 token 都以块形状串行复核，
**只能标为 correctness control**；后续速度对比必须使用**公平的新 AR 基线**（新 AR 比旧 AR 慢）。

## 6. 如何避免"提前知道答案 / 运行中回填"

1. **不读取任何保存的答案**：原型运行中只使用模型自身产生的 token；pad 内容为**与答案无关的固定 id**
   （如 0），且 pad 行不参与任何决策；
2. **不修改 tie-breaking**：仍用 `argmax`（最小下标），不选候选、不做人工替换；
3. **不只修已知分歧 token**：从 prefill 之后**每一步**都使用统一几何（§2），不做位置特例；
4. **不隐藏串行复核成本**：AR 的 4× 代价如实记录，且**不计入**任何 speedup。

## 7. 8GB 资源预算与可行性

* 复用**已有 PyTorch 实现**：`spec_dec._verify` / `_step`、`spec_cache.trim_cache`、`QwenLM._run`；
  **不手写 CUDA、不从零写 attention**；
* 显存与现有 Spec verify 同级（模型 2.87 GiB + draft ≈1 GiB + cache）；AR 侧每步 4 行 ⇒ 峰值与 Spec verify 相同；
* **可行性结论：可行**（前提是 §8 的实测通过）。

## 8. 目标语义（必须二选一，不得含糊）

**本候选同时修改 AR** ⇒ 只能宣布 **目标 B**：

> **B：AR 与 Spec 同时改用统一几何后彼此一致。**

* **不**宣称复现**旧 AR** 的逐 token 序列（旧 AR 是 `s=1` 几何，新 AR 是 `s=4` 几何）；
* **旧 AR 与新 AR 的差异必须单独记录**（同一输入、同一轨迹、逐位置比较）；
* 正式采用需**另行批准、重新验证**，并使用**公平的新 AR 基线**；
* 旧的严格门槛（AR/Spec 逐 token 一致）**不因本候选而放宽**；本轮的"通过"只针对 B 目标下的
  两个开发样本（F13、F06），**不**宣称全局正确性。

## 9. 必须先过的 CPU/短输入测试（GPU 之前）

1. **padding 不变性**：同一 cache + 同一 row-0 token，**不同 pad 内容**（0 / 随机 / 重复）下
   row 0 的 logits **必须逐位相同**（用 `model_self` 的微型模型在 CPU 上实测，含 mask/softmax/AV 全链路）；
2. **trim 正确性**：padded 调用后 `trim_cache` 的长度与已提交位置的 K/V 必须与"非 padded 单步"一致；
3. **尾轮/接受/拒绝/rollback/EOS** 的纯逻辑计划表（`k_eff<4` 时 pad 到 4、只取前 `k_eff` 行、提交数正确）；
4. **有限性 / 位置 / 长度**：运行后逐层 finite、cache 长度 = prompt + 已提交、输入与快照未被原地修改。

## 10. 提前声明的否定条件（不硬跑）

* 若 **padding 不变性**在实测中**不成立**（pad 内容影响 row 0）⇒ 候选**被否定**，停在设计报告；
* 若同一几何下 AR 与 Spec 仍**逐位不同** ⇒ 候选**被否定**（说明还有别的形状/路径差异），停在定位记录；
* 若 OOM / 非有限 / 结构失败 / 复现门失败 ⇒ 落盘停止，**不**换参数追求通过。

## 11. 不能保证的事项（写入结论）

* 本候选**只**覆盖"同一模型对、同一几何"下的一致性；**不**证明实现无 bug；
* **不**证明在其它 k、其它 chunk、其它 KV 档（FP8）、其它上下文长度下同样成立；
* **不做**正式速度声明；**不**启动 FP8 或正式矩阵；
* 即使 F13 与 F06 都通过，也**只**支持"该几何下这两个样本一致"，随后应提出**独立回归需求**。

---

## 12. 实测结果（2026-10-01）：**候选在 F13 上被否定**（目标 B 未达成）

### 12.1 先过的 CPU/短输入测试（**26 PASS / 0 FAIL**，`unified_geometry_selftest_cpu`）

* **padding 不变性（关键前置）逐位成立**：同一 cache + 同一 row-0 token，四种不同 pad 内容
  （`zeros / ones / repeat / rand`）下 row 0 的 logits **逐位相同** ⇒ pad 内容不可能影响决策、也不可能泄题；
* **trim 正确性**：padded 调用 + trim 后，**提交位置的 K/V 与"非 padded 单步"逐位相同**（pad 行的 KV 未进入有效 cache）；
* 计划表：AR 步 / repair / verify / 尾轮 `(s, T-pre, mask 对角线)` **完全一致**（`(4,4,1)`）；
  尾轮 `k_eff<4` pad 到 4 只取前 `k_eff` 行；接受/拒绝/rollback（m=4/1/0）判定正确；
* 有限性 / 长度 / 快照未被修改。

### 12.2 F13 GPU 实测（`UG_MODE=ar|spec`，BF16 / base_bf16 / chunk2048 / k4 / max_new64 / 原 EOS / 全 eager）

| 运行 | 结果 |
| --- | --- |
| UCG-AR `ucg_20261001_F13_ar_20261001T114131Z` | 64 token、`max_new_reached`、几何一致 `s=4, T-pre=4`（63 个计划）、`fused={}` |
| UCG-Spec `ucg_20261001_F13_spec_20261001T114629Z` | 64 token、`max_new_reached`、几何一致 `4×4`（59 个计划）、`fused={}` |

**判定**：
* **UCG-AR vs UCG-Spec：不一致** —— 共同前缀 **47**，**首次分歧 index 47（预测位置 8239）：AR=6702、Spec=6652**
  ⇒ **目标 B 未达成，候选被否定**；
* UCG-Spec vs **旧生产 Spec**：`identical=True`（本轮的 pad 化 repair 未改变任何决策）；
* UCG-AR vs **旧生产 AR**：`identical=False`，首次差异 index **47**
  ⇒ 新几何**确实改变了 AR 自己的轨迹**（因此只能宣称目标 B，**不**能宣称复现旧 AR）。

### 12.3 机制：统一 `s`（行数）**不足以**统一决策行的 `T`

AR(UCG) 每步 pad 成 4 行、`pre = P-1` ⇒ 决策行的 `T = (P-1)+4 = P+3`。
Spec 的 verify 在 `pre` 处喂 `Kp` 行（pad 到 4），**块内第 j 行**预测 `P = pre+j+1`，其 `T = pre+4 = P+3-j`：
只有 `j=0` 时 `T = P+3` 与 AR 相同；`j≥1` 时 `T` 更小 ⇒ **决策行的 T 依"该行在块内的行号"而不同**，
而"哪一行决定某个位置"取决于接受模式。

⇒ **需要统一的不只是行数 `s`，还有决定某个位置的那一行的 `T`（总 key 长度）**。
要做到 `T_ar(P) = T_spec(P)`，Spec 的 verify 必须把"预测 P 的那一行"放在 `pre = P-1` 且为**行 0**，
即**一次只验一个候选** —— 那等于放弃块验证（退化为逐 token 串行）。

**因此本轮结论（证据限定于本实现、本配置、F13 样本）**：
* 候选被否定；**块验证与"逐位等于单步 AR"在该引擎的表格下结构上不相容**；
* 这**不**证明实现有 bug，也**不**证明其它几何（如手写统一 kernel、固定 T 的 padding 方案）不可能
  —— 但**本轮不新增候选**（按任务的停止规则）。

### 12.4 缺项与不能保证的事项（如实登记）

* **Spec 侧的逐轮 `pre/Kp/m` 与 `plans` 未落盘**（本轮 runner 未传 `log=` 回调、未保存 `plans`）
  ⇒ "分歧位置在 Spec 侧的块内行号 j" **无法从本轮产物反算**；机制说明（§12.3）是**依据
  `_verify` 索引语义的结构性分析**，不是本轮实测的行号。**未**为补该记录重跑（按停止规则）；
* UCG 的 AR 侧 `target` 计算量约 **4×**（正确性控制，**不是**加速方案）；本轮**未**做任何 speedup 声明；
* 只测了 **F13**；**F06 未跑**（按"先 F13，出现分歧即停"的规则）；
* 未验证其它 k、chunk、KV 档、上下文长度；未改生产默认；proposal 与验收门槛未动。

