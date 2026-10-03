> **SUPERSEDED HISTORICAL SNAPSHOT — NOT THE CURRENT PROJECT SCOPE.**
> This checkpoint preserves an early project-state snapshot; its progress,
> matrix, commitments, and next steps are not current requirements. Current
> formal scope is governed by
> [`proposal_freeze_2026-10-01.md`](proposal_freeze_2026-10-01.md); for current
> status, also consult [`README.md`](../README.md) and
> [`status_update_2026-10-02.md`](status_update_2026-10-02.md).

# SpecDec 状态快照（Checkpoint）

> 用途：阶段性"收工记录"——汇总当前进度、已定决策、实测数据、未决项与下一步。
> 关联：数据台账 `data_ledger.md`（数字+条件）；方案 `plan_v6.md`（v6.1 修订）；阶段报告 `stage_report_l1_l2.md`。

## 1. 进度（整体 ≈ 40%）
| 阶段 | 进度 | 备注 |
|---|---|---|
| 规划/方案（v6.1 已吸收评审 #3–#9）| ≈95% | 10/7 提案待交；课程提案 1 页稿已备 |
| 环境 | ≈100% | torch cu124、HF 缓存、VS Code |
| L1 从零引擎 | 100% | 对齐 0.000 |
| L2 KV-cache + 生成 | 100% | 生成对齐 PASS；GPU 基线 sanity 完成 |
| L4 int8/int4 / L4b fp8 KV / L5 投机解码 | 0% | 下一站 L4 |
| 18 格 / M1 归因 / 报告 | 0% | 待 L4–L5 数据 |

## 2. 已锁定决策
- 不做 demo/原型为主线；shape/数据流分析助手仅作"可选演示项"（课程：demo 不必须）；
- 核心承诺收敛：**18 格（或 15/18）+ M1 归因 + 诚实分析 + 报告**；短文/演示/K1 均可选；
- draft 定案：**Qwen2.5-0.5B**（与 1.5B 同 151936 词表、免训练）；self-spec 仅 COULD；
- K1（CUDA）= 第一优先砍项，不进任何承诺；
- 指标定位：ppl=量化质量指标；投机解码质量=接受率+净加速（greedy verify 无损）；
- 增量定位：8GB 从零 eager 引擎上对已知方法的组合实证与扩展（非新方法）；
- M1 归因重心 = 草稿收益 vs 验证开销（可测）；KV 带宽仅理论下限估计列，失真写进局限；
- 出数窗口可提前挂机应对考试周；11/14 冻结神圣。

## 3. 实测数据（详见 data_ledger.md）
| 项 | 值 | 条件 |
|---|---|---|
| L1 对齐（ours vs HF full）| **0.000** | fp32/CPU eager，5 prompts |
| L2 缓存等价 | ~2e-5，预测一致 | 真权重各深度 |
| 生成对齐 | **PASS（4/4 序列一致）** | vs HF 全量逐步贪心 |
| CPU fp32 decode | 9.20 tok/s（~109 ms/tok）| 38-tok prompt, 64 decode |
| GPU fp16 decode（ours）| 10.58–13.82 tok/s | 38-tok prompt, 32–64 decode |
| GPU fp16 decode（**transformers eager 参考**）| **8.29–9.40 tok/s** | 同条件 sanity |
| 结论 | 基线无"可避免开销"（快 ~27–47%），L5 分母可信 | — |
| **8K fp16 单次 prefill** | **OOM（CUDA）**：scores 1.75GiB + logits 2.5GiB + 双模型 >8GB | 需分块 prefill + 跳过 prefill logits（详见 data_ledger §4.4）|
| **分块 prefill** | ✅ 已实现并验证：fp32 下与一次性 prefill 一致（worst 7.8e-4、逐层平滑）；fp16 缓存有取整漂移（量级随层累积）但 token 行为一致（首 token + 6–12 步 decode 全同）| 8K/18 格硬前提已清 |

## 4. 未决/待办
- [ ] 修 `model_self.py` 末尾 `__main__` 残留（评审跑直接运行会 NameError）
- [ ] 10/7 前：定 2–3 人团队与分工（贡献声明影响个人评分）
- [x] ~~8K 测量前置工程~~（分块 prefill 已完成并验证）
- [ ] L4 int8/int4 权重量化（下一主线）

## 5. 下一步（建议顺序）
1. 8K 探针（半天）→ 确定 18 格或 15/18
2. 修 __main__ 残留 + 整理 repo（configs/results/ 目录按课程要求）
3. **L4 量化**（A 层 MUST）→ L4b → L5 → 18 格 → M1 → 报告
4. 10/7 交课程提案（1 页稿已备，需填队名与分工）

## 6. 本轮结论一句话
> 方案 v6.1 已吸收评审、进度 ≈40%（领先 v6 时间表），引擎正确性有 0.000 与生成对齐背书，decode 基线经库参考验证可信——**下一个兑现点是 L4 量化**。
