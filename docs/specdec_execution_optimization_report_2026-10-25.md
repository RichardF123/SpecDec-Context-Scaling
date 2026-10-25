# Phase O：SpecDec 执行路径优化报告（2026-10-25）

> 目标：在**不改变模型/输入/k/max_new/接受-回滚语义**的前提下，降低 classic draft-model SpecDec 的
> proposal / verification / repair 执行开销，并测量对其 decode ratio 的影响。
> **本报告不启动正式 16-family 矩阵**；只做开发集（F09/F03/F11/F04 × 2K/4K/8K）的优化与验证。
> 结论先行：正确性全过；**性能未达到推进门槛** ⇒ `PHASE_O_VALID_BUT_NO_MATERIAL_GAIN`（理由见 §8）。

冻结配置（全程未改）：target `Qwen/Qwen2.5-1.5B-Instruct`、draft `Qwen2.5-0.5B-Instruct`、BF16、
`base_bf16` KV、`cudnn_prefill`（仅 prefill 融合）、k=4、chunk=2048、max_new=64、greedy EOS-aware、
contexts 2048/4096/8192、AR=target-only。默认执行路径仍是 frozen/reference；优化只在
`SPECDEC_EXECUTION_MODE=optimized`（或显式 flags）时启用。

---

## 1. 优化前的数据流（O0 审计摘要）

见 `docs/specdec_execution_hotpath_audit_2026-10-25.md`。一轮 round：

```
_propose(draft)   ← kp≤4 次 draft._run(s=1)；每步 int(argmax) → H2D token → embed
_verify(target)   ← 1 次 target._run(s=kp)；候选 list → torch.tensor H2D
acceptance        ← Python 逐 token：g = [int(argmax) ...] ×kp；m 逐 token 比较
rollback          ← trim_cache 视图切片（无拷贝）；随后 _step 的 torch.cat 重现复制全长历史
repair            ← target/draft 各 1 次 _step(s=1)，token 经 torch.tensor([[t]]) H2D
```

代码事实（审计 §2–§4）：每轮 **≈8–10 次 host 同步**（kp×proposal int + kp×acceptance int + H2D），
KV 每层每步 `torch.cat` 全长重分配，RoPE 每 `_run` 重建，attention 统计与阶段判定进入 timed 区。
F09 2K Nsight proposal span **GPU 覆盖仅 16%、gaps 84%**（host 供不上 GPU）。

## 2. 实施的改动（代码位置 / 原理 / 输入输出 / 预期作用）

旧问题 → 本轮改动总览：

| 旧开销 | 改动 | 文件 |
| --- | --- | --- |
| proposal 每 token `int(argmax)` 同步 + H2D token | 全 GPU 链：argmax 0 维张量直接作 embedding 索引；候选缓冲 `copy_` 写入；EOS 检查延后到轮末一次 | `experiments/execution_optimization/fast_spec_dec.py : _propose_gpu` |
| acceptance Python 逐 token（kp 次 int 同步） | 向量化：`g=argmax(cat(saved_next, rows[:-1]))`，`m=(g==cands).cumprod().sum()` 全 GPU，轮末一次 `.tolist()` 取回全部控制值 | 同上 `spec_decode_greedy_optimized` |
| repair token H2D | `_step_gpu`：token 直接取 `g_t[m]`（GPU 0 维）→ `embed(ids.view(1,1))` | 同上 `_step_gpu` |
| 每轮多个临时 token 张量 | 候选缓冲按 k 预分配 + `copy_`；控制值一次性回传 | 同上 |
| （正确性对齐）初始单次 s=ctx prefill 在 4K/8K 与分块路径数值不同 | prefill 按 `chunk=2048` 逐块 `_run`，与 `test_chunked_prefill` 逐字同构（这是**必须**的，不是优化） | 同上 `_prefill_chunked` |

为什么可能降低延迟：GPU 空闲 84% 的 proposal span 中，每个 `.item()`/H2D 都让 CPU 等待完整 GPU 往返；
把 kp 次往返压缩为轮末 1 次，并把逐 token 的 Python 控制流换成 2 个极小的 GPU 归约，理论上减少
~6 次同步/轮与 3–5 次 H2D/轮。

如何证明语义不变（设计层面）：
- proposal 的 token 序列、draft `_run` 次数、EOS 截断规则（首个 EOS 位置 +1）、verify 的 s 与候选集合、
  acceptance/rollback/repair 的判定值、emitted token、trim 长度、stats 计数全部与 reference 一致；
- verify 仍是 `target._run` 的 block 前向（s=kp），repair 仍是 s=1 的 `_run`，无自研算子；
- 唯一行为差异：当 EOS 出现在候选块内部时，优化版会"多跑"到 k_eff 步再按 kp 截断（多算的 draft 步在
  输出语义上不可见；`exposed` 分支以 eos 结尾必然 break，extra 状态不被下一轮读取）。此差异在 CPU 测试中
  有专门的 EOS 用例覆盖，并在 GPU probe 的 12 个真实用例中零差异。

未做/被 blocker 挡住的项：
- **O2.4 CUDA Graph**：隔离测量显示 16-step target 605.8→273.9 ms、draft 524.2→118.9 ms，但生产接入有
  两个硬阻塞：① RoPE `start_pos` 由 cache 长度决定（model_self.py:1003–1006），每轮 T 不同，同一 graph
  不能跨轮复用；② 图内 attention 的 T 是静态形状，需 padding + `-inf` mask，而 bf16 归约顺序变化会破坏
  token 逐位一致（本项目的硬性判据）。按任务要求**记录 blocker，不强行破坏正确性**。
- KV `cat` 预分配重写：未实施（`trim` 视图 + 布局/连续性会改变 matmul 输入 stride，存在数值/性能双风险；
  实测 KV 复制在每轮耗时中占比很小，见 §9）。

## 3. 消除了哪些同步/分配（GPU probe 实测，非性能计时）

`probe_gpu.py` 对 12 个 fixture 用例各跑 reference 与 optimized，统计 host 同步类操作
（`.item()/.tolist()/.cpu()/int(tensor)`）：

| 指标 | reference | optimized |
| --- | --- | --- |
| 同步类操作 / round | **7.71–8.00** | **2.00** |
| H2D token 拷贝 / round | 3–5（代码事实：propose kp + verify 1 + repair ≤2） | **0** |
| 控制轨迹差异（Kp/m/pre/cands/g/emitted/tgt_len/drf_len） | — | **0（12/12）** |

## 4. 正确性证据

1) `py_compile`：`fast_spec_dec / fixtures / probe_gpu / bench_dev / analyze_bench / config` 全部 rc=0。
2) **CPU 单元测试** `test_fast_spec_cpu.py`：**PASS 7 / FAIL 0**。轻量因果 stub 覆盖 proposal 正常链、
   `k_eff<4` 尾轮、m=0 / 0<m<kp / m=kp、EOS 出现在候选块内（截断到 kp=3）、开关全关的 reference 等价路径；
   断言 reference 每轮 ≥6 次同步、optimized ≤2 次；optimized `ledger.h2d == 0`。
3) **reference fixtures**（O1）：4 families × 3 ctx = **12/12**，含 input sha1、AR/SpecDec ids、逐轮控制轨迹、
   stats、attention 统计、逐轮 cache 长度不变量（诊断 hooks 在每步 argmax 前检查有限性）。
4) **GPU correctness probe**：**12/12 OK**（`all_ok: true`）：
   - optimized ids / stats == reference == fixture（三向）；trace diffs = 0；
   - AR 与 fixture AR 一致（12/12）；reference 在本进程也复现 fixture（12/12，说明测量可重复）；
   - 每轮 `tgt_len = drf_len = pre + len(emitted)`；末轮 = ctx + gen；`spec_cache.check_consistent` 通过；
   - 有限性：`finite_check=True` 下 propose/verify/repair 每次 `_run` 后检查 `isfinite` 全过；
   - attention 策略断言 `assert_strategy_used()` 通过（decode/propose/verify/repair 全 eager、prefill 融合）。
5) **开发集基准交叉核对**：bench 的 `ref`、`opt`、`ar` 输出与 fixtures 逐 token 一致（12/12 每组），
   说明跨进程、跨软件阶段的确定性；每个条件的 3 次 measured 与阶段账本 run 输出一致（bench 自检强制）。
6) boundary clock：36 条件全部 `boundary_clock_qualified = true`（0 次不合格）；生成长度全部 64，
   stop 全为 `max_new_reached`；峰值 reserved ≤ 5.13 GiB。

## 5. 逐 family 数据（decode 秒，3 次 measured 中位；AR 为同批独立进程）

| family | ctx | AR | reference | optimized | R_ref=T_AR/T_ref | R_opt=T_AR/T_opt | gain(T_ref/T_opt) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F09 | 2048 | 2.41 | 4.40 | 4.31 | 0.549 | 0.560 | +2.0% |
| F09 | 4096 | 2.34 | 4.79 | 4.86 | 0.489 | 0.482 | **−1.5%** |
| F09 | 8192 | 2.41 | 6.97 | 6.81 | 0.346 | 0.354 | +2.5% |
| F03 | 2048 | 2.59 | 5.10 | 4.93 | 0.507 | 0.525 | +3.5% |
| F03 | 4096 | 2.32 | 4.56 | 4.31 | 0.509 | 0.539 | +6.0% |
| F03 | 8192 | 2.21 | 3.73 | 3.51 | 0.594 | 0.630 | +6.1% |
| F11 | 2048 | 2.45 | 5.00 | 4.80 | 0.490 | 0.510 | +4.1% |
| F11 | 4096 | 2.43 | 4.73 | 4.63 | 0.513 | 0.524 | +2.2% |
| F11 | 8192 | 2.41 | 6.21 | 5.90 | 0.387 | 0.408 | +5.2% |
| F04 | 2048 | 2.53 | 5.08 | 4.86 | 0.499 | 0.521 | +4.5% |
| F04 | 4096 | 2.34 | 5.04 | 4.69 | 0.465 | 0.499 | +7.4% |
| F04 | 8192 | 2.31 | 5.79 | 5.58 | 0.398 | 0.413 | +3.8% |

（F09 4K 的 −1.5% 是单格噪声级波动；11/12 为正，中位 +4.0%。）

## 6. 三个 context 的 ratio（中位）

| ctx | R_ref | R_opt | gain 中位 | decode 中位降低 | proposal 阶段中位降低 |
| --- | --- | --- | --- | --- | --- |
| 2048 | 0.503 | 0.523 | +3.79% | 3.65% | 1.37% |
| 4096 | 0.499 | 0.511 | +4.11% | 3.91% | 4.30% |
| 8192 | 0.393 | 0.410 | +4.51% | 4.31% | 3.64% |

## 7. 优化前后 stage breakdown（逐 ctx 平均 share；诊断运行，CUDA Event 账本）

| ctx | 角色 | prefill | propose | verify | repair | residual |
| --- | --- | --- | --- | --- | --- | --- |
| 2048 | ref | 7.8% | 55.5% | 18.0% | 18.2% | 0.5% |
| 2048 | opt | 8.0% | 55.0% | 18.1% | 18.3% | 0.6% |
| 4096 | ref | 14.8% | 51.7% | 16.5% | 16.5% | 0.5% |
| 4096 | opt | 15.1% | 51.1% | 17.0% | 16.2% | 0.5% |
| 8192 | ref | 25.7% | 44.7% | 13.7% | 15.4% | 0.5% |
| 8192 | opt | 26.5% | 44.5% | 13.7% | 14.8% | 0.5% |

**proposal 仍是最大项**（44–55%），但**优化没有显著压缩它**——这正是与"同步数减少 4×"一起出现的关键反证：
同步/回传不是主导成本，主导成本是 `_run` 内每层每步的大量 CUDA launch（GPU 空闲 84%），只有
graph 级合并/编译能消除，而 graph 接入被 §2 的 blocker 挡住。

## 8. 是否达到推进门槛（预注册口径）

| 门槛 | 结果 |
| --- | --- |
| 4/4 families 输出与 reference SpecDec 一致 | ✅（12/12 条件） |
| 12/12 family-context 控制轨迹一致 | ✅（fixture + probe trace_diffs = 0） |
| cache invariants & finite & attention strategy | ✅ |
| proposal 阶段中位时间降低 ≥30% | ❌ **3.6%（三 ctx 中位）** |
| 且满足 R>1 / 2K R≥0.80 / decode 中位降低 ≥30% 之一 | ❌（2K R 0.523；decode 降低 3.8%；无 R>1） |

⇒ **`PHASE_O_VALID_BUT_NO_MATERIAL_GAIN`**（正确性通过、性能未达门槛）。按任务要求：**停止**，不启动
正式矩阵，也不把"优化版"用于正式 16-family 采集。

## 9. 未解决的瓶颈

1. **每层每步的 launch/host 开销**：proposal 的 84% GPU 空闲（nsys）说明 GPU 在等 CPU；本轮把同步从
   ~8/轮降到 2/轮，只换来约 4% 总 decode 时间（≈6 ms/轮 × 23 轮）。余下的 launch 数（每步 ~数百个
   CUDA op）无法通过"少几次同步"消除。
2. **CUDA Graph 阻塞**：动态 T（position 依赖）+ 动态 cache 内容；单 graph 需按 T capture；static buffer
   需每轮 O(T) copy-in/out；若用 padding 固定形状，`-inf` 掩码下 bf16 归约顺序变化会破坏 token 逐位一致。
   不做破坏正确性的接入。
3. **KV cat 重分配**：估算每轮复制量在 2K≈0.35 GB、8K≈1.4 GB（bf16 K/V × 6-7 步 × 28 层），
   按 250 GB/s 约 1.4–5.6 ms/轮，占总 decode 的 <3%；用预分配 buffer 替换会改变 matmul 输入 stride
   （可能引入内部拷贝），不是高性价比方向。
4. 环境噪声：bench 期间多次 commit 门槛拒绝（14.3–14.6 GiB），已保留证据并等待自然恢复后重跑；
   未改门槛、未改参数。

## 10. 下一步建议

- 若要在**不改变语义**的前提下拿到显著收益，唯一现实路径是 graph 级或编译级合并，且必须解决
  **动态长度位置编码 + 动态 cache** 的静态化问题（例如：把 `start_pos` 变成 device 标量 + 预生成 RoPE 表，
  把 attention 改成对固定容量的 masked 计算），并以本轮的 **fixture + 三向竞态检查**作回归门；
  在 token 逐位一致通过前不得进入正式矩阵。
- 若保持"冻结语义"，本轮结论就是该方向的负结果：**同步/回传优化的上限 ≈ 4%**，不足以改变
  SpecDec 慢于 AR 的结论（2K/4K/8K 的 R 仍 ≈0.52/0.51/0.41）。
- 建议把本轮产物作为后续 graph 尝试的**正确性 oracle 与回归套件**（fixtures 12 例、probe 三向检查、
  bench 的 ref/opt/ar 交叉核对脚本可直接复用）。

## 11. 产物清单

- 代码（新增，均在 `experiments/execution_optimization/`）：
  `__init__.py`、`config.py`（显式开关 `SPECDEC_EXECUTION_MODE` 与三个 OPT flag）、
  `fast_spec_dec.py`（优化执行器）、`fixtures.py`（reference fixture 构建）、`probe_gpu.py`（三向正确性 probe）、
  `bench_dev.py`（开发集基准，36 独立进程，含内存等待/断点续跑）、`analyze_bench.py`（汇总/图/裁决）、
  `test_fast_spec_cpu.py`（CPU 单元测试 7/0）。
- 数据：`output/specdec_execution_optimization/20261025T133313Z_1da85b1c/{fixtures,manifest.json,probe/}`、
  `output/specdec_execution_optimization/20261025T142610Z_99ac0e75/{<36 条件目录>,analysis/,manifest.json}`。
- 论文口径 CSV：`docs/measurements/ot_execution_{conditions,ratios,stages,context_summary}_2026-10-25.csv`。
- 图：`docs/figures/specdec_execution_optimization/ot_execution_gain.{svg,png,pdf}`（四面板：a decode 延迟分布、
  b 三 ctx 的 R_ref/R_opt、c reference vs optimized 的阶段占比、d gain 随 ctx 趋势）。
- 审计与报告：`docs/specdec_execution_hotpath_audit_2026-10-25.md`、本文件。

## 12. 局限

- 4 个开发 family、每条件 3 次 measured；不是正式 16-family 统计结论，不用于筛选 family。
- 阶段账本来每条件额外 1 次诊断运行（CUDA Event 会引入同步），只用于 share 分析，不进入主 ratio。
- 单格噪声可见（F09 4K −1.5%），说明 4% 量级的增益在单条件上可能被时钟/环境波动淹没；ctx 中位是更稳的读法。
- fixtures/probe 子进程在 Windows 上出现"工件齐全但进程退出码为 1"的现象（torch/CUDA teardown 侧），
  已核对 JSON/CSV 完整、bench 父进程 exit 0；该异常与结果无关，未规避。
- 未做 graph 实现（blocker 已在 §2/§9 记录）；未改任何冻结配置；未运行正式 16-family 矩阵。
