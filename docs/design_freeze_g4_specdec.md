# G4 —— Fixed-chain Greedy SpecDec 设计冻结（DRAFT v0.1，2026-09-08，会话续）

> 状态：**DRAFT，待实现前逐项批准**。范围：只做 **correctness**（exact greedy equivalence）；
> 不运行 2K/4K/8K 正式矩阵、不宣称加速、不做 CUDA kernel/sampling/beam/INT 量化/改 k。
> 本文件是 plan（意图/协议）；实测数据进 `data_ledger.md`。

## 1. 冻结参数（第一版）
| 项 | 值 |
|---|---|
| target | Qwen/Qwen2.5-1.5B-Instruct（engine OURS，bf16 GPU 计算）|
| draft | Qwen/Qwen2.5-0.5B-Instruct（engine OURS，同词表 151936）|
| proposal length | **k = 4**（by design，同 v7.1 冻结，本轮不变）|
| decoding | greedy（temperature 0；argmax；无采样接受规则）|
| batch | 1 |
| KV 条件 | base/bf16（K/V bf16）与 v_fp8（K bf16 + V FP8 E4M3 gs128，G3 正式方案）|
| 判定 | **SpecDec 输出 token ids 与 target-only greedy ids 完全一致（exact equality）** |

## 2. correctness 语义（单轮数据流）
1. draft 从当前 prefix（其 cache 长度 = 已处理真实 token 数）自回归贪心提议至多 k 个 token；
   - 遇 EOS 停提议；剩余预算 < k 时按实际剩余数；
2. target 对 prefix+proposals 一次性前向 → 每位置 argmax = target-greedy；
3. 逐位比较：接受**连续相等前缀**；首个不一致处取 target token；
4. 全部相等：draft 同轮再提议 1 个（bonus 位置），target 验证 k+1 位；全同→接受 k+1
   （含 bonus），否则接受 k 个 + bonus 位 target token；此后才进入下一轮新提议；
5. EOS：若 target 确认位置含 EOS → 停生成（生成长度按冻结协议记录）；draft 提议内 EOS
   不自动终止整轮，仍交给 target 验证（同 v7.1 §4.3 语义，实现前再与 target-only 对拍确认）。

## 3. cache 不变式（本协议核心）
- **cache 长度 ≡ 该模型已处理的"真实" token 数**（真实 = 已被 target 确认并加入输出的 token）。
- 每轮结束时：
  - target cache 长度 = 旧长 + (接受数 m) + (m<k ? 1 替换位 : bonus 位确认 ? 1 : 1)
    —— 即 target 只保留"已验证为真实"的位置；mismatch 之后的假 draft 位置一律裁剪。
  - draft cache 长度 = 旧长 + m（只含被接受的 draft token；替换/bonus token 由 target
    产出，draft 未处理 → 下一轮 draft 第一步先消化该 token 再提议）。
- 裁剪必须**沿 token 维 dim=2 一致**作用于该 cache 的全部张量：
  - base：`k`、`v`（bf16）
  - v_fp8：`k`（bf16）、`v`（fp8 payload）、`v_scale`（fp32）
- 原因：注意力每步用 cache 做完整历史 K/V；若 cache 多含"从未确认的 token"，后续
  softmax 会向它们分配概率 → 破坏与 target-only 的等价性（不止是性能问题，是正确性）。
- 本轮不做 cache 克隆做"分支生成"；全接受路径用 bonus 规则直接推进，无分叉需回溯。

## 4. cache API（实现目标；独立模块 `specdec/engine/spec_cache.py` 或等价，不入 model_self）
- `cache_length(cache) -> int`：读 `cache[0]["k"].shape[2]`（base/v_fp8 通用）。
- `trim_cache(cache, new_len) -> cache`：对每层、每张量 `t = t[:, :, :new_len]`
  （v_scale 同样 dim=2）；`new_len < 0` 或 `> 当前` 抛错/钳制（越界保护）；
  `None` cache 返回 None。
- 独立测试覆盖：None/空、正常 trim、trim 到当前长度、v_fp8 的 v_scale 同步、
  越界保护。

## 5. greedy SpecDec 主循环（独立模块，如 `specdec/engine/spec_dec.py`）
每轮按 §2 走 1–5；记录并报告（behavioral，不作 correctness 判定）：
proposed tokens、accepted draft tokens、acceptance rate、target verification rounds、
rollback count（=mismatch/裁剪发生次数）、all-accepted rounds、mismatch rounds。

## 6. Correctness gate 测试（B6）
- prompts：普通英文 / 数学 / 代码 / 短 prompt / 含 EOS 边界 / `max_new_tokens < k` /
  多轮 mismatch 型文本；
- 条件：base KV 与 v_fp8 KV 各跑；
- 判据：`SpecDec ids == target-only greedy ids` **exact equality**；同时报告 §5 行为指标。
- 说明：acceptance rate 只解释行为与（未来）速度，不参与 correctness。

## 7. 前置门（实现前先做，未通过则停止并报 blocker）
**B3 dual-model 8GB 最小可行性**（同机 RTX 4060 Laptop 8GB）：
- target(1.5B) 与 draft(0.5B) 能否同时加载（bf16）；
- 短 prompt 各自前向 OK；
- base KV 与 v_fp8 KV 各自 decode 无 OOM；记录 peak allocated/reserved；
- 若双模型放不下 → 停 G4 实现、报实际数字 blocker；**不**私自引入权重量化改协议。

## 8. 范围排除（本轮禁止）
3×2×2 正式性能矩阵 / "SpecDec 加速"宣称 / CUDA kernel / sampling/beam / 修改 k /
INT8/INT4 权重量化 / 跨机器(4090D vs 4060)数据混合 / 概率式接受。

## 9. 批准栏
- [ ] 我审阅并理解 §1–§8 —— 批准为 v1.0（冻结）
- [ ] 修改意见：＿＿＿＿＿＿＿＿

---
*变更记录：v0.1（DRAFT，2026-09-08）：初稿（B1 讲解 + B2 冻结）。*
