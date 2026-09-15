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

## Phase 2 — B5：最小 greedy SpecDec 主循环（进行中：等你实现 `spec_dec.py`）
- 测试已就绪：`g4_b5_smoke.py`（结构不变量+确定性）、`g4_b6_correctness.py`（exact-token-id 门）。
- **目标**：fixed k=4、无 bonus、greedy；严格按 v0.2 §3 的 logits 对齐与 §4 的 cache 回滚语义。
- **交付**：`specdec/engine/spec_dec.py`（用户）；AI 提供规格 + 评审 + 状态字段约定。
- **通过标准（实现自检）**：m<k 与 m=k 两分支 cache 长度分别为 `pre+m+1` / `pre+k`；`check_consistent` 不报错；EOS/budget<k 符合协议。
- **风险**：off-by-one（用 `b_{k-1}` 误判第 k 个 proposal）；draft 漏裁替换位之前的位置；全接受写成 k+1（bonus 已取消）。

## Phase 3 — B6：correctness gate（AI 写测试 + 双方判定）
- **目标**：SpecDec ids == target-only greedy ids（exact；不只比文本），base_bf16 与 v_fp8 两档；全 finite、重复确定。
- **覆盖 prompts**：自然语言 / 数学 / 代码 / 短 / 长 / 全接受 / 首位拒绝 / 中途拒绝 / EOS / 生成长度非 4 倍数。
- **交付**：`g4_b6_correctness.py`（AI）。
- **通过标准**：全部 case 逐 token 全等 + finite + ≥2 次运行一致。**未过禁止测 acceptance/pilot/final。**
- **估算**：测试编写 1–2h；运行 10–20 分钟。**依赖**：Phase 2。
- **已知风险（必须保留）**：BF16 block-built 与 serial-built cache 数值差 ≈2.0 → 多轮累积可能翻 token；若出现不一致，先做 margin 诊断，区分 near-tie 与真实 bug，**不得改 baseline 或放宽门**。

## Phase 4 — Part 4：小规模测量（B6 通过后）
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
