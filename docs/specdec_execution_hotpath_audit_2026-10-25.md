# SpecDec 执行热路径审计（Phase O0，2026-10-25）

> 纯只读审计：**未修改任何代码**。本文区分【代码事实】与【待验证假设】。
> 参考版本：`main` c07fe55 工作区 + 冻结分支 `codex/confirmatory-full-freeze`(4c0e99b)。
> 生产配置：target=`Qwen/Qwen2.5-1.5B-Instruct`、draft=`Qwen2.5-0.5B-Instruct`、BF16、
> `base_bf16` KV（引擎内 `kv_format="fp16"`）、`cudnn_prefill`（仅 prefill 融合）、k=4、chunk=2048、
> max_new=64、greedy EOS-aware。

---

## 1. 一次 SpecDec round 的调用链（代码事实）

`specdec/engine/spec_dec.py : spec_decode_greedy()`（行 189–263）：

```
_prefill(target) ×1           行 140–143 → chunked prefill（chunk=2048，仅末块取 logits）
_prefill(draft)  ×1           （同上；两次 prefill 在 decode 之外）
while len(gen) < 64:                                    行 204
  k_eff = min(k=4, 64-len(gen))                         行 205
  pre  = cache_length(tgt_cache)                        行 206  ← 读 shape[2]（int，无同步）
  _propose(draft, drf_cache, drf_lg, k_eff, eos)        行 146–157
      for _ in range(k_eff):                            ← 最多 4 次 draft._run(s=1)
          t = int(lg.reshape(-1,V)[-1].argmax())       行 150  ← **同步点**
          cands.append(t)
          lg,cache = draft._run(embed(torch.tensor([[t]])), cache)   行 152–154 ← H2D + s=1 前向
          if t == eos_id: break                        行 155
  _verify(target, tgt_cache, cands)                     行 160–177（block 模式）
      lg, cache_blk = target._run(embed(torch.tensor([cands])), cache) 行 172–175 ← H2D + s=kp 前向
      rows = lg[0, -kp:]                                行 177
  g = [int(x.argmax()) for x in [saved_next] + rows[:-1]]  行 230 ← **kp 次同步**
  m = 首个 g[j] != cands[j] 的 j（Python 逐 token）        行 231–233
  if m < kp:  (mismatch)
      r = g[m]; emitted = cands[:m]+[r]
      tgt_cache = trim_cache(tgt_blk, pre+m)            行 238（视图切片，无拷贝）
      saved_next, tgt_cache = _step(target, r, tgt_cache)  行 239 ← H2D + s=1 前向
      drf_cache = trim_cache(drf_cache, pre+m)          行 240
      drf_lg, drf_cache = _step(draft, r, drf_cache)    行 241 ← H2D + s=1 前向
  else:       (all-accept)
      emitted = cands; tgt_cache = tgt_blk
      drf_cache = trim_cache(drf_cache, pre+kp)         行 247
      saved_next = rows[kp-1]; drf_lg = prop_lg         行 248–249
  gen += emitted; 记录 log（诊断时）；emitted[-1]==eos ⇒ break  行 252–261
```

每轮计数（kp = 本轮候选数 1..4）：draft `_run` **kp 次**（s=1）；target `_run` **1 次**（s=kp）；
mismatch 时 target/draft `_step` 各 1 次（s=1）；`trim_cache` 视图 1–2 次；`argmax` **kp（propose）+ kp（g）** 次。
各阶段在 `experiments/formal/worker.py : spec_generation_scope()`（行 274–297）被标注为
`propose / verify / repair`，`scripts/profile_specdec_cost_2k.py : phase_scope()` 用 CUDA Event 做阶段计时
（draft proposal ≈ 59–63%、repair ≈ 17–23% 的来源，见 `docs/measurements/cost_profile_2k_2026-10-23.csv`）。

## 2. 逐项检查

| # | 检查项 | 结论（代码事实） | 位置 |
| --- | --- | --- | --- |
| 1 | proposal 循环 Python 循环次数 | **每轮 kp 次（≤4）**；`k_eff<4` 只在最后 1 轮出现 | spec_dec.py:149 |
| 2 | 每个 draft token 触发的 `_run` 次数 | **1 次**（s=1）；loop 内第 j 步即 1 次 target? 否：draft 自身 1 次 | spec_dec.py:152 |
| 3 | `.item()/.cpu()/.tolist()` | `_propose` 中 `int(...argmax())` ×kp；主循环 `int(x.argmax())` ×kp；`cache_length` 用 shape（无同步）；**无 .cpu()/.tolist()** | spec_dec.py:150,230 |
| 4 | 隐式 GPU→CPU 同步 | 上述 `int()` 即隐式 `.item()` 同步：**每轮约 2·kp 次**（kp=4 ⇒ 8 次）；另有 `bool` 比较（none） | 同上 |
| 5 | 重复创建 position IDs/mask/索引 | 每个 `_run` 重建 RoPE：`torch.arange`+`outer`+`cos/sin`+`cat`（`Build_Rope_cache`，model_self.py:272–284）。mask：仅 s>1 时建（verify 的 (kp,kp) 极小块，model_self.py:915–921）；`kv_repeat` 每层物化 2 次（行 897–898）。`torch.tensor([cands])`/`torch.tensor([[t]])` 每轮 3–5 次 H2D（`_propose` 1/步、`_verify` 1、`_step` 1–2） | model_self.py:272,864–882,897,915–920；spec_dec.py:152,173–174,183–184,240–241 |
| 6 | KV 是否 `torch.cat` 扩展 | **是**：`fmt=='fp16'` 时每层每步 `torch.cat([cache["k"],k])`、`V` 同理（28 层 target / 24 层 draft ⇒ 每步 56/48 次全长重分配） | model_self.py:863–868 |
| 7 | rollback 是否重新分配/复制完整 cache | `trim_cache` 只切片（视图、无拷贝）；但 trim 后下一次 `_step/_verify` 的 `cat` 会**重新分配并复制全长历史** | spec_cache.py:10–30；model_self.py:864–867 |
| 8 | verify/repair 是否创建动态形状临时张量 | verify 的 mask 形状 (s,T) 其中 T 随历史增长（s=kp 固定）；`cat` 形状随 T；`torch.tensor([cands])` 形状固定 (1,kp)；repair `_step` 的 `torch.tensor([[t]])` 固定 (1,1) | 同上 |
| 9 | 每轮 `torch.cuda.synchronize()` | **计时路径内无显式 synchronize**（只在 `run_spec_once` 的计时边界外，worker.py:382/390；`spec_prefill_scope` 行 250/259）。`int()` 是隐式同步 | spec_dec.py:150,230 |
| 10 | attention fallback/统计/日志是否进入 timed boundary | **是**：`_record_attention_call`（每层每调用一次，dict 更新 + 字符串 reason 判断）与 `attention_phase` 上下文都在 decode 计时内；`_probe` 为 no-op 但仍每层每 op 调用一次；`log=...` 仅诊断路径 | model_self.py:598–632,822–965；spec_dec.py:255–259 |
| 11 | acceptance/first-mismatch 由 Python 逐 token 判断 | **是**：`while m<kp and g[m]==cands[m]`（Python int 比较） | spec_dec.py:231–233 |
| 12 | 哪些形状在 k=4 下可静态化 | 候选 (1,kp)、verify 输入 (1,kp)、repair 输入 (1,1)、mask (kp,kp)、出/入 token id 全部可静态化；**KV cache 长度 T 不可静态化**（每轮增长，长度依赖 accepted 长度与 repair，属控制流数据依赖）；RoPE 表可按位置预计算但位置 start_pos 依赖 T（不可在 graph 内静态化，见 §5） | 分析 |

## 3. 潜在同步点（每轮，kp=4，mismatch 轮）

| 同步点 | 次数 | 证据类型 |
| --- | --- | --- |
| `int(argmax)` in `_propose` | kp（≤4） | 代码事实（.item 语义） |
| `int(x.argmax())` for `g` | kp（≤4） | 代码事实 |
| `torch.tensor([cands])` H2D | 1 | 代码事实（host→device 拷贝，不阻塞 GPU） |
| `torch.tensor([[t]])` H2D（propose 每步 + repair） | kp + 1..2 | 代码事实 |
| `torch.cuda.synchronize` | 0（计时路径内） | 代码事实 |

⇒ **每轮 8–10 次隐式/显式 host 同步**（全轮约 23 轮 ≈ 200+ 次同步）。GPU 覆盖仅在 proposal span 的 16%
（`docs/draft_propose_nsys_report_2026-10-23.md`，F09 2K，nsys `--cuda-graph-trace=node` 统计），
说明 **host 供不上 GPU**：84% 时间为 launch/同步间隙。

## 4. 潜在分配点

| 分配 | 规模 | 频次 |
| --- | --- | --- |
| `torch.cat` KV（每层 K、V） | O(T)，2·28·T·128·2B ≈ 1–7 MB/层/步 | 每步；每轮 propose 4 + verify 1 + repair 1–2 ⇒ 6–7 次/轮/层 |
| `kv_repeat` 物化（12 heads） | 12·T·128·2B | 每层每步 2 次 |
| `attention` 中间（scores/softmax/attn） | (1,12,s,T) | 每层每步 |
| RoPE 表（arange/outer/cos/sin/cat） | (1,1,1,d) + (1,d/2) | 每次 `_run` |
| `torch.tensor(...)` token 张量 | 1–kp long | 每轮 3–5 次 |
| `trim_cache` | 无（视图） | 1–2 次/轮 |

估算：F09 2K `draft_step_ms≈37.7ms`、`target_repair_step_ms≈42.1ms`（cost ledger），
每 round ≈ 4×37.7 + verify(≈1 step×s=5) + 2×~40 ≈ 230 ms 量级（与 4.94s/23rounds≈215ms 一致）。

## 5. CUDA Graph 可行性（代码事实 + 假设）

- 既有隔离测量：fixed-2K history 下 16 步链 capture/replay，target 605.8→273.9 ms、draft 524.2→118.9 ms
  （`docs/measurements/cudagraph_chain_corrected_2026-10-24.csv`，**每步 token 在 graph 外填充**，否则固定 token 陷阱）。
- 生产接入的两个硬阻塞（**假设，待 O2.4 验证**）：
  1. **位置依赖**：`_run` 的 RoPE 由 `start_pos = cache["k"].shape[2]` 决定（model_self.py:1003–1006），
     每轮 T 不同 ⇒ 同一 graph 无法跨轮复用；按 T 逐值 capture 数量不可控（64 gen × 23 轮）。
  2. **cache 内容依赖**：static buffer 必须在 replay 前从动态 cache `copy_` 全量历史（O(T)/轮），
     回写追加段；收益需大于 copy 成本，且 graph 池需覆盖 propose(k步)/verify/repair 各静态形状。
- 结论：**先把同步与 Python 逐 token 判断去掉**（不改变数值），再评估 graph 的净收益；
  若 graph 无法安全接入生产 cache，记录 blocker，不做破坏正确性的接入。

## 6. 优化候选与优先级（供 O2 使用）

| 优先级 | 候选 | 预期作用 | 风险 |
| --- | --- | --- | --- |
| P0 | propose 内 token 走 GPU（argmax 张量直接作为 embedding 索引，去掉 `.item()`；EOS 检查延后到轮末一次同步） | 去掉 kp 次/轮同步 + kp 次 H2D | 低（失败即回退） |
| P0 | acceptance 向量化：`g=argmax(cat(saved_next, rows))`、`m=leading-equal` 在 GPU 求，一次 `.item()` | 去掉 kp 次/轮同步 + Python 逐 token | 低 |
| P1 | repair 用 GPU token 直接 embed（去 H2D）；控制张量预分配（(1,k),(1,1),(kp,kp) mask 复用） | 去 1–2 次 H2D、少量分配 | 低 |
| P2 | KV append 改造（预分配 buffer + `index_copy_`/切片赋值），数学等价 | 去 O(T) cat 重分配 | **中高**（需逐位等价验证；不改数值） |
| P3 | CUDA Graph 子路径（固定形状 propose 链/verify/repair） | 最大（隔离 2.2×），但受 §5 阻塞 | 高（动态 cache） |

## 7. 代码事实 vs 待验证假设

**代码事实**：§1–§4 的行号/计数/调用结构；每轮 8–10 次隐式同步；无显式 synchronize；KV 用 `torch.cat`；
acceptance 为 Python 逐 token；RoPE 每 `_run` 重建；graph 隔离实验的数字。

**待验证假设**（O2 中以实测判定）：
- 去掉 `.item()`/H2D 后 proposal 阶段时间的实际下降幅度（sync 数 ≠ 时间，GPU 空闲间隙可能由 launch 数主导）；
- KV `cat` 重分配对总时间的占比（vs 每层 op 的 launch 开销）；
- graph 接入的生产净收益（copy 成本 vs launch 节省）。
