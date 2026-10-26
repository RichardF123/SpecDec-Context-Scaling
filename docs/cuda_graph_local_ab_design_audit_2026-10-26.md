# CUDA Graph 局部 A/B 设计与 graph-safety 审计（2026-10-26，实现前）

> 性质：`experiment_class = operator_mechanism_diagnostic`；`formal_result_allowed=false`；
> `quality_claim_allowed=false`；`full_specdec_speedup_claim_allowed=false`。
> 这是局部算子/执行路径诊断，不是完整 CUDA Graph SpecDec、不是生产实现、不是 EAGLE 实验。
> 基线 `codex/relaxed-specdec` @ `732f1fe`；不改正式 runner / strict / relaxed / bonus oracle 结果与历史数据。

## 1. capture 期间的 CPU 同步

| 检查 | 结论 |
| --- | --- |
| `_run` 内 `.item()` / `.tolist()` | **无**（decode/propose/verify/repair 路径均无 host 标量读取；token 在 relaxed/bonus 优化后走 GPU 张量） |
| Python 数据依赖分支 | 有，但只依赖 **capture 时** 的 Python 值：`start_pos = cache[0]["k"].shape[2]`、`decide_attention_path(phase, s)`、`s==1` 分支 —— 这些在 capture 时被固化，replay 不重跑 Python |
| 动态 shape 判断 | `_run` 的 s 是 Python int（capture 时固定）；`kv_repeat` 的 n_rep 固定 |
| 显式 `torch.cuda.synchronize` | capture 路径内无（只有外层计时边界使用） |

结论：**固定 shape/固定历史长度下，`_run` 本身没有 CPU 同步**，满足 capture 基本要求；但因此 graph
只能用于 capture 时的那个 `(history length, s, positions)`。

## 2. capture 期间的新分配

| 位置 | 类型 | capture 兼容性 |
| --- | --- | --- |
| `torch.cat([cache["k"], k])`（每层 K、V） | 新 tensor | PyTorch CUDA Graph 允许 capture 期分配：使用 graph 私有 memory pool，replay 复用固定地址 |
| `Build_Rope_cache`（`arange/outer/cos/sin/cat`） | 新 tensor（1×1×s×d） | 同上；位置值由 Python int 固化 |
| `_kv_keep_mask`（verify s>1 的 `(s,T)` mask） | 新 tensor | 同上；mask 值/形状固化 |
| `torch.tensor([[token]])` 类 | **无**（Phase O 起 token 走 GPU 张量） | — |
| 静态输入 / 输出缓冲 | 需自建 | 预分配 token、初始 cache 快照，replay 前 `copy_` |

结论：允许 capture（分配走 pool），但每层的 cat 输出地址在 capture 时固定；**replay 不会自动把 Python
侧 cache dict 更新**（dict 赋值只在 capture 时执行一次），因此本实验的读出口径是"捕获时持有的输出 tensor"。

## 3. 输入/输出地址固定性

- 输入：静态 token 张量（`torch.empty(1,s,long)`，`copy_` 更新）与初始 cache（每层 K/V 的
  **clone**，长度 T，capture 前准备好）。
- 输出：capture 返回的 logits 与新 cache 张量（graph pool 内固定地址，replay 后被新值覆盖）。
- 本实验只做**单算子/固定 chain**，每次 replay 的初始 cache 内容不变（graph 只读不写输入），
  因此 replay 幂等；若需要"同一 graph 换历史"，必须重建 graph（见 §7 阻塞点）。

## 4. KV cache 原地写入还是新对象

当前实现：`SelfAttention.forward` 在 `fmt=="fp16"` 分支执行 `cache["k"] = torch.cat(...)` ——
**返回新 tensor、dict 原地更新**。capture 时 Python 执行一次、记录 cat kernel；replay 只重放 kernel，
不再执行 dict 赋值。因此本实验必须：
1. capture 前 clone 一份初始 cache 并保存引用，避免被 capture 期间的 dict 更新覆盖；
2. 对比时读取 capture 返回的"最终 cache tensor"（dict 里指向的 graph-pool 张量），而不是重新读 dict 的旧引用；
3. 先做 replay 的 input-update 测试（改 token 后输出必须改变），排除"重放旧输入"。

## 5. RoPE / mask / position 预分配

- RoPE：由 `Build_Rope_cache(head_dim, seq_len, theta, start_pos)` 在每次 `_run` 内创建；`start_pos` 是
  Python int ⇒ **每个 graph 位置固化**。本实验对每个 `(family, ctx, component)` 单独 capture（固定 T、固定 s），
  这是 Stage G0/G1 的合法范围。
- mask：verify s>1 时 `torch.triu(torch.ones(s,T))` 固化在 graph 内；s=1 decode 无 mask。
- position IDs：与 RoPE 相同（位置 = 历史长度 + 块内偏移，捕获时固化）。

## 6. Attention backend 在 decode/verify 是否 graph-safe

正式策略 `cudnn_prefill` 只在 prefill 阶段融合；decode/propose/verify/repair 全部走 eager
（matmul + softmax + kv_repeat），无 cuDNN 调用，因此无 cuDNN graph 兼容问题。cuBLAS 在 capture 前需要
warm-up（本实验做 ≥3 次 warm-up，并在 capture 前同步）。

## 7. Windows WDDM + PyTorch 2.6/cu124 是否允许 capture

允许。**本项目已在同一环境成功捕获过 16 步 growing-history chain**
（`docs/measurements/cudagraph_chain_corrected_2026-10-24.csv`：target 605.8→273.9 ms、draft 524.2→118.9 ms），
说明本机 Windows + cu124 + torch 2.6 的 capture/replay 可用。

## 8. capture 后 replay 是否会错误复用旧 cache

风险存在（graph 的输入指针指向 capture 时的张量）。缓解：
- replay 前只通过 `copy_` 更新静态输入（token、必要时 cache）；
- gate 里做 **input-update 测试**：换 token 后 replay 的输出必须变化，且与 eager 在新 token 上一致；
- 做 **多次 replay deterministic 测试**（同输入两次 replay bitwise 相同）。

## 9. 固定历史长度下能否安全重放

**能**（本实验范围）：固定 T、固定 s、固定 token 长度。不能跨历史长度复用同一 graph；
**把 graph 接进 bonus-aware 动态循环需要每轮按新 T 重 capture 或做静态容量 + padding** —— 后者会改变
attention 归约顺序，必须先做 token 级等价验证，本实验不做、仅记录为集成阻塞点。

## 10. 显存约束（8 GB）

粗估（8K 条件）：target 权重 ≈3.0 GiB；eager/prefill cache ≈0.23 GiB；M=4/M=5 graph 各持一份输出 cache
≈0.23 GiB；chain graph 5 步逐级增长 cache ≈1.2 GiB；静态输入 cache 副本 1 份 ≈0.23 GiB。
合计 ≈5 GiB，加 graph pool 与 fragment 仍在 8 GB 内，但必须实测 `peak_reserved` 与 pool delta；
若超过则把 chain 拆成多次 capture 或降低 ctx。

## 11. 本实验的实现口径（先于代码确定）

- 组件：A1 draft s=1；A2 4 步 candidate chain；A3 bonus catch-up + 4 步 candidate chain；
  B1 target M=4 verify；B2 target M=5 verify。
- 历史：T = 2K/4K/8K（每 combo 单独 capture）；token 取自 **bonus oracle 真实轨迹**（读现有
  `output/bonus_accept_oracle/run/<fam>_<ctx>_bonus/result.json` 的 control），不是随机 tensor。
- eager 对照：同一初始 cache 内容、同一 token；每次迭代传"浅 dict 副本"（cat 只改副本 dict，不动初始张量）。
- graph 对照：静态缓存 clone + 静态 token；replay 前 `copy_` 更新 token；**两次口径**：
  * kernel-only：只计 replay；
  * replay-ready effective：replay + 静态输入 `copy_`（+ 若需要恢复初始 cache，则计 restore 成本；
    本设计 replay 不改输入 cache，restore=0，但会单独测"真实动态 cache → 静态 buffer 的 copy 成本"并报告）。
- 计时：CUDA Events，warm-up≥10，5 trials，每 trial N 次 replay（按组件调 N，使单 trial≈0.5–1 s）；
  capture 时间、首次 replay、稳态 replay、static copy、pool delta、peak allocated/reserved 单独记录。
- 正确性门：logits/cache 优先 bitwise；非 bitwise 记录 max|diff| 与 argmax；**argmax 变化即 FAIL**；
  cache dtype/层数/长度/finite/`check_consistent`；replay deterministic；input-update；M=4/M=5 语义；
  draft chain 逐 step token 来自真实轨迹且与 eager 输入一致。

**阻塞点（记录，不绕过）**：动态历史长度需要每轮重 capture（或静态容量+padding 的等价性验证）；
本实验只测固定历史局部算子，不得把 kernel-only 收益当作端到端加速。
