# SpecDec 日报 2026-09-15b（G4 阶段 B 后半段：B5 通过 / B6 两档建门 / EOS 修复）

> 前情：同日 `daily_report_2026-09-15.md` 已完成"协议 v7.2 → G4-0 归因 → B3 → B4"，当时 `spec_dec.py` 为 0 行、B5 未实现。
> 本日续篇：**`spec_dec.py` 落地 → B5 PASS → B6 两档（bf16 CHECK / fp32 PASS）→ 逐行评审发现并修复 EOS 越界 bug**。

## 一、今日一句话

**B5 主循环 PASS（T1–T4 + 真实投机 smoke，并修掉曾导致 0/58"空过 PASS"的 logits 形状 bug）；B6 建立为两档并列的门——`B6-bf16 = CHECK 19/20`（唯一分歧经实测归因为 bf16 数值不可判定位置：signal 0.1491 ≤ noise 0.1491）、`B6-fp32 = PASS 20/20`（判据运行前预登记入库）；逐行评审 `spec_dec.py` 抓到并修复 EOS 全接受路径的 `trim_cache` 越界（常规收尾路径，正式实验必踩），配纯逻辑回归测试含旧实现对照。**

## 二、今日完成（按门）

| 项 | 结果 | 证据/落点 |
|---|---|---|
| **B5 辅助函数单测** | **PASS** | `g4_b5_helper_tests.py` T1 `_fmt` / T2 `_prefill` / T3 `_propose` / T4 `_verify` 对齐；T4 用两条**独立** prefill 比较 `rows[j]` vs `serial[j]`，max\|Δ\| 0.12–0.30 |
| **B5 主循环 smoke** | **PASS（真实投机）** | rounds 8 / proposed 27 / accepted 11 / mismatch 5 / all_accept 3；每轮不变量 `tgt_len == drf_len == pre + (m+1 if m<Kp else Kp)` 全成立（覆盖 m=0/1/4、Kp=1/2/4）；重复运行一致。`target_block_calls` 与 rounds 恒等（每轮 1 次块前向）⇒ 8；`draft_calls` 与 proposed 恒等（每候选 1 次前向）⇒ 27 |
| **B5 修后 GPU 复验** | **PASS（两档）** | EOS 修复（`f7097d4`）之后重跑 `g4_b5_helper_tests` + `g4_b5_smoke`：两档逐轮 trace 全 `ok`，stats 与修前一致；`target_block_calls` **8**、`draft_calls` **27** 由推导变**实测**；新登记结构性恒等式 **`gen_len == accepted + mismatch_rounds`**（16 = 11 + 5）⇒ B5 记录已对应当前代码 |
| **B6-bf16** | **CHECK 19/20** | `g4_b6_correctness.py`：base_bf16 9/10、v_fp8 10/10；总 accepted/proposed = **194/381**；唯一分歧 `natural-2` @7。**用户裁决 A = 严格口径：不设平局例外、不放宽 exact-id 门、不替换参照**（→ §4.5.5） |
| **tie-collapse 归因** | 完成 | `g4_b6_tie_diag.py`（同 prefix=ids+ref[:7]，len=14）：bf16 三路径（full / cached / AR-history）argmax **全 = `' use'`**，两候选**精确并列** 22.8750（`tied_at_max=2`）；fp32 三路径 **全 = `' convert'`**，gap **+0.1491**。`signal(0.1491) ≤ noise(0.1491)` ⇒ 该位置在 bf16 下**数值不可判定**；`spec[7]` 恰等于 fp32 真值。形状噪声实测：full-vs-cached 0.250、hist-vs-full 0.281（≈2–4 ULP，1 ULP=0.125）|
| **B6-fp32 预登记** | 完成 | `g4_b6_fp32_correctness.py`：判据在**运行前**入库（`3f7fe6f` 预登记 + `9aa6b00` 补漏：小样跑不得判 PASS）；明确"被测对象是实现，dtype 不是被测对象" |
| **B6-fp32 实测** | **PASS** | 10 case × 2 KV 档 = **20/20 组合逐 token id 全等**；总 **216/348**（62.1%）；finite 全真、确定性通过、非空过、case 与档位跑满；单 case 2.2–36.1 s（CPU）。运行于 EOS 修复**之后**，且修前小样数字**逐字复现**（natural-1 11/21、natural-2 16/16）|
| **EOS 越界 bug** | **已修复** | `_propose` 原把 `if t==eos_id: break` 放在推进 cache **之前** ⇒ draft 提议末位为 EOS 时 draft cache 长度 `pre+kp-1` ⇒ 该轮被 target 全接受时（生成正常以 EOS 收尾）全接受分支 `trim_cache(drf, pre+kp)` 越界抛 `ValueError`。修法：EOS 检查移到推进之后 ⇒ 出口恒定 `pre+kp`；副产物 `draft_calls` 记账变精确 |
| **EOS 回归测试** | **PASS** | `g4_b5_eos_branch_tests.py`（**纯逻辑 CPU stub，无模型/GPU/网络，毫秒级**，4 case）；修后 gen/不变量/记账全 OK；**旧实现对照在 3 条 EOS 全接受 case 上如实复现 `ValueError: new_len N 越界 (current N-1)`**（非空过保证），EOS 被拒 case 两版一致 |
| **设备契约修复** | 完成 | `spec_dec._verify` 的 `device="cuda"` → 跟随 `cache[0]["k"].device`（CPU fp32 复跑的前提）|

## 三、关键数据（详见 `data_ledger.md` §4.5.4 / §4.5.5 / §4.5.6）

**B6-fp32（实现正确性证据）**

| case | n | base_bf16 acc/prop | v_fp8 acc/prop | rounds |
|---|---|---|---|---|
| natural-1 | 16 | 11/21 | 11/21 | 6 |
| natural-2 | 16 | 16/16 | 16/16 | 4 |
| math | 16 | 9/31 | 9/31 | 9 |
| code | 16 | 14/19 | 14/19 | 5 |
| short | 16 | 12/24 | 12/24 | 6 |
| long | 16 | 12/25 | 12/25 | 7 |
| len13 | 13 | 11/15 | 11/15 | 5 |
| len5 | 5 | 5/5 | 5/5 | 2 |
| len2 | 2 | 2/2 | 2/2 | 1 |
| eos-prone | 16 | 16/16 | 16/16 | 4 |
| **合计** | — | **216/348** | 逐 case 相同 | — |

**同 case 的 bf16 档 vs fp32 档对照（RQ2 用）**

| 口径 | 总 accepted/proposed | 接受率 |
|---|---|---|
| B6-bf16 | 194/381 | **50.9%** |
| B6-fp32 | 216/348 | **62.1%** |

- `natural-2`：bf16 分歧 @7（tie-collapse）；**fp32 16/16 全接受且逐 token 全等** ⇒ 该 prompt 上 draft 与 target 决策其实**完全一致**；bf16 的翻面同时污染"输出 token"与"接受判定"两件事。
- **边界（不可越界归因）**：11.2 个百分点是**复合效应**（bf16 下两模型 argmax 都会翻面，既影响"提议什么"也影响"是否接受"），**不能**单独归因于 tie-collapse；需 tie 频率普查。
- **接受率强依赖 prompt**：同档内 29.0%（math）～100%（natural-2 / eos-prone）⇒ 单条 prompt 不可代表，multi-family 是必需。
- **V-FP8 未翻转任何决策**：两档逐 case acc/prop 完全相同，与 §3.5.3 的质量结论（V-only ΔNLL +0.0004）方向一致。

## 四、今日新增/修改文件

**代码**
- 新增（AI/测试）：`g4_b5_helper_tests.py`、`g4_b6_tie_diag.py`、`g4_b6_fp32_correctness.py`、`g4_b5_eos_branch_tests.py`
- 修改：`spec_dec.py`（用户核心代码；AI 落两处修复：`_verify` 设备契约、`_propose` EOS 顺序）、`g4_b5_smoke.py`、`g4_b6_correctness.py`（`divergence_diag` 增补 argmax/tied）

**文档**
- 修改：`data_ledger.md`（**新增 §4.5.4 B5 / §4.5.5 B6-bf16 / §4.5.6 B6-fp32** + 变更日志）、`design_freeze_g4_specdec.md`（§3.2 加"`_propose` 出口长度不变量"、§3.6 EOS 写入规则）、`plan_g4_next_steps.md`（Phase 2 ✅ / Phase 3 门已建立 / Phase 4 解锁）、`README.md`（进度快照更新）

## 五、今日排查与经验（方法学，值得进报告）

1. **`topk` 的返回顺序 ≠ 决策路径的 `argmax`**：并列时 `topk` 不保证取最小 index。首轮输出里 `top1=' convert'` 是 `topk` 假象，真实 argmax 三路径全为 `' use'`。**判读一律以 argmax 为准**（B6 的 `divergence_diag` 与 tie diag 均已改）。
2. **门不能在"噪声 ≥ 信号"处判定**：`natural-2@7` 的真实 gap 0.1491（1.2 ULP）小于 bf16 自身形状噪声 0.25–0.281（2–4 ULP）。此时"逐 token 全等"既非实现的错、也非可修复的判据缺陷 ⇒ **不放宽判据**，而是把**同一份实现**放到良定义配置（CPU/float32，误差 ~1e-7）中测量，且**判据运行前入库**。
3. **信号/噪声判据不需要人为阈值**：`signal=|gap_fp32|`、`noise=|gap_bf16−gap_fp32|`，同位置、同候选对比较；`signal ≤ noise` 即不可判定。
4. **fp32 阶段必须放 CPU**：8 GB 卡放不下 1.5B fp32 权重（5.8 GiB）+ 工作区 + 桌面占用。**CPU 在正确性门里是"标尺"而非被测系统**（性能与质量口径仍是 GPU/BF16）。
5. **逐行评审核心代码能抓到测试抓不到的路径**：EOS 全接受是**常规收尾路径**，B5/B6 的 10 个 case 全未覆盖（旧代码在该路径上是直接崩溃而非静默错）。用**位置策略 stub 模型**可在 CPU 上毫秒级复现整条分支（无需权重/GPU/网络），并带"旧实现必须复现崩溃"的对照防止测试空过。
6. **记账精度**：`draft_calls` 在 EOS 轮原先少算 1 次实际前向（记录 `kp`、实际 `kp−1`）⇒ 修复后二者精确相等，对 RQ2 的 overhead 归因是实质修正。
7. **GBK 控制台不能输出 `✓/✗`**（`UnicodeEncodeError`）⇒ harness 输出一律 ASCII 标记。

## 六、风险与限制（更新）

1. **bf16 tie-collapse**：既翻转输出 token，也翻转接受判定；bf16 vs fp32 接受率差 11.2 个百分点属**复合效应**，单独归因需 tie 频率普查（已列为 RQ2 后续测量项）。
2. **接受率强依赖 prompt**（29.0%–100%）⇒ 必须 ≥16 families 覆盖不同形态（`math` 类低接受率形态必须在内）。
3. **8K 双模型长 decode 峰值未实测**（B3 只跑 5 token smoke）。
4. **final 规模未标定**：≥16 families × 12 conditions × ≥5 reps ≈ ≥960 次运行，估 8–16 h 纯 GPU 时间；**须据 Part 4 实测单次成本在开跑前定死**。

## 七、下一步（顺序）

1. ~~B5 在 EOS 修复后的 GPU 复验~~ **✅ 已完成**（本日，两档 PASS，见 §二）；
2. **≥16 prompt families（2K⊂4K⊂8K）+ 版本化 manifest**（AI 起草、用户定稿）；
3. **G5 instrumentation**（`results.jsonl` schema / config hash / resume）；
4. **Part 4 测量 harness（含 AR 基线同档）+ 单次成本标定** → 据此定 final 规模；
5. 并行：**10/7 提案补姓名与分工声明（硬 deadline）**；tie 频率普查。

## 八、提交状态

本日提交（均含在 `main`）：`647eed1`（core: spec_dec B5 PASS）、`e448ab3`（B6 CHECK 与归因登记）、`3f7fe6f`（B6-fp32 预登记）、`9aa6b00`（预登记判据补漏）、`f7097d4`（EOS 修复 + 回归测试）、`56c87de`（B6-fp32 PASS 登记）、`5ea72a5`（本日报 + README 刷新）、`9f0c...`（B5 修后复验登记）。工作树干净，`main == origin/main`。
