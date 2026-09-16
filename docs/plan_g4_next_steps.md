# G4 下一步执行规划（2026-09-08 制定，待执行）

> 依据：`design_freeze_g4_specdec.md` v0.2、`plan_v7_1_protocol_freeze.md` §0 Amendment v7.2、`docs/design_g4_cache_api.md`、`data_ledger.md` §4.5.2/§3.5.3。
> 分工原则：**核心代码（引擎/缓存/主循环）由用户实现；AI 提供框架指导、规格、评审与全部测试 harness。**
> 门禁原则：一步一测、未过不进下一步；不发明协议外阈值；不宣称未实测的性能/加速。

## Phase 0 — 入库保护（10 分钟，AI 执行）
- commit + push：Amendment v7.2、G4 doc v0.2、G4-0/B3/B4 harness、diag 脚本、cache API 指导、ledger/daily 更新。
- 通过标准：`main` 与 `origin/main` 同步；工作树干净。

## Phase 1 — B4：KV cache 操作（✅ 完成 2026-09-08）
- **状态**：`spec_cache.py`（用户实现）通过 `g4_b4_cache_tests.py` 全部断言 → **B4: PASS**
  （base_bf16 用 0.5B、v_fp8 用 1.5B；分进程运行避免显存累积）。
- 契约校正：`trim_cache` 对**所有张量**沿 dim=2 裁（`v_scale` 同步）、浅拷贝不改入参、
  `new_len<=0` 或超长抛 `ValueError`、`trim_cache(None,0) is None`；`v_fp8` 为 target 专属（`head_dim%128==0`）。
- 环境经验：harness 的 `from_pretrained` 必须**本地优先**（`local_files_only=True`），
  否则 tokenizer 联网列模板会因网络抖动抛 `httpx.RemoteProtocolError`（曾被误判为"驱动静默杀进程"）。

## Phase 2 — B5：最小 greedy SpecDec 主循环（✅ 完成）
- **状态**：`spec_dec.py` 已实现，`g4_b5_helper_tests.py` T1–T4 全 PASS；`g4_b5_smoke.py` PASS（真实投机：rounds 8 / proposed 27 / accepted 11 / mismatch 5 / all_accept 3；cache 不变量 `tgt_len==drf_len==pre+(m+1 or Kp)` 全成立；重复运行一致）→ **B5: PASS**（详见 ledger §4.5.4）。
- 修正记录：vacuity bug（logits 合同 `(1,V)`→`(1,1,V)`，曾致接受率 0/58 的"空过 PASS"）；错配分支改为 `trim(pre+m)` + 替换 token **单步重算**（两个模型都做），废弃"复用 block 位置 L+m+1"的旧指导。
- **风险**：off-by-one（用 `b_{k-1}` 误判第 k 个 proposal）；draft 漏裁替换位之前的位置；全接受写成 k+1（bonus 已取消）。

## Phase 3 — B6：correctness gate（⚠️ CHECK 19/20，用户裁决 A = 严格口径）
- **状态**：`g4_b6_correctness.py` 实测 **19/20** —— base_bf16 档 9/10、v_fp8 档 10/10；唯一分歧 `natural-2` @7。
- **归因（已实测，`g4_b6_tie_diag.py`）**：该位置 bf16 下候选对**精确并列**（22.8750 / 22.8750，gap 0）三路径 argmax 一致取 `' use'`；fp32 三路径一致给出真值 `' convert'`（gap **+0.1491**）。`signal(0.1491) ≤ noise(0.1491)` ⇒ **bf16 数值不可判定位置**；`spec[7]=' convert'` 恰为 fp32 真值。
- **裁决（用户，2026-09-15）**：**A = 严格** —— 不设"平局例外"、不放宽 exact-id 门、不替换参照 ⇒ **B6 记 CHECK，不转 PASS**；该位置作为已知数值限制写入报告，并在 Method/Limitations 说明"接受判据 = argmax 相等性检验，其可靠性上限 = bf16 gap 符号可靠性"。
- **后果（开放项）**：协议"B6 未过禁止测 acceptance/pilot/final"与"不得提出主循环性能声明"仍生效。Part 4 若启动，只能作为**工程观测**（标注"未通过正确性门"），不得作为已验证结论 —— 需用户明确同意后再做。
- **保留风险**：BF16 block-built 与 serial-built cache 数值差（实测 max\|Δ\| 0.250–0.281 ≈2–4 ULP）> 部分位置的真实 gap → 多轮累积仍可能翻 token。

## Phase 4 — Part 4：小规模测量（**被 Phase 3 门禁阻断**）
- **目标**：少量 prompt 的预实验（非正式矩阵）：acceptance=accepted/proposed、每轮接受长度分布、target block calls、draft token calls、decode-only 与 e2e 延迟、tokens/s、peak alloc/reserved、base 与 v_fp8 差异。
- **纪律**：含 CUDA sync、warm-up、≥3 次重复、报中位数与离散；**不只报最好一次**；不宣称加速（先出数）。
- **交付**：`g4_part4_pilot_measure.py`（AI）。
- **估算**：1–2h 编写 + 30 分钟运行。**依赖**：Phase 3。

## Phase 5 — 并行非技术/基建（可与其他阶段并行）
- **10/7 课程提案**：定 2–3 人团队 + 贡献声明；提案 1 页稿按 v7.2 口径更新（BF16 compute、KV 因子 base/v_fp8、12 conditions、K-FP8 negative）。
- **Prompt families**：设计 ≥16 个 nested-prefix families（2K⊂4K⊂8K），manifest 版本化；先出设计草案（AI 起草，用户定稿）。
- **G5 instrumentation**：`results.jsonl` schema、config hash、resume/recovery 设计（AI 出规格，用户实现）。
- **Freeze Checklist** 逐项核对（≥16 families、spec k、schema、clean rerun 等）。

## Phase 6 — Pilot（B6 + G5 + families 就绪后）
- 4 families × 12 conditions × 3 reps；只查 OOM/instrumentation 正确性/cache 与 spec 正确性/时间方差/thermal/schema/resume。
- pilot 数字不进最终主表（除非协议未变且注明）。

## Phase 7 — Final 12-condition + 分析 + 报告
- ≥16 families × 12 conditions × ≥5 reps（3 ctx × 2 KV × 2 decode）；质量 sanity（ΔNLL/ΔPPL 相对 base_bf16）；
- 归因（acceptance benefit vs verification overhead；KV 带宽仅理论估计）、失败分析 F1–F7、break-even 区间；
- 4–5 张核心图；5–7 pp 报告 + 可复现（一条命令重跑核心结果）+ 演示（可选）。

## 立即动作（本次会话）
1. Phase 0 入库（AI，等你一句"提交"）；
2. 你开始写 `spec_cache.py`（照 `docs/design_g4_cache_api.md`），写完贴 AI 审 → 跑 B4 测试。
