# G5 测量协议（冻结 v1，2026-09-21）

> 目的：把 Part 4 暴露出来的三个测量问题（**显存饱和导致的顺序效应**、**决策可复现但计时不可复现**、
> **正确性口径混用**）变成可执行的固定协议，使正式 Pilot 的数字可审计、可复现、不可事后美化。
> 本协议**不改算法**：`k=4`、target/draft 模型、BF16 计算口径、V-only FP8 设计全部保持冻结。

## 1. 允许与禁止

| 允许 | 禁止 |
|---|---|
| 开发 G5 harness / schema / 稳定性探针 / 成本分解 / k 扫描（诊断） | 在稳定化完成前跑 ≥16 families × 12 conditions 的正式矩阵 |
| 读旧 Part 4 JSONL（**只读**） | 修改/删除/覆盖任何历史 JSONL |
| 用新 run_id 产出新 JSONL | 修改已正确的 logits 索引；改 frozen V-only FP8 设计；引入 K-FP8 / 权重量化 / FlashAttention |
| 诊断性 k 扫描（k≠4） | 因扫描结果私自改 proposal；偷偷换成 K+1（bonus）算法 |

## 2. 运行身份（防覆盖）

- 文件名 `g5_<schema_version>_<config_hash>_<RUN_ID>.jsonl`，`RUN_ID` 默认 UTC 时间戳；
- `write_g5_run()` 在目标已存在且未显式 `append` 时**直接报错**，绝不覆盖；
- 每条行自带 `schema_version` / `git_commit` / `dirty_worktree` / `seed` / `block_index` /
  `condition_index` / `condition_order`；
- 缺字段一律 `null` + `availability[field]=false`；**遥测字段绝不用 0 或字符串冒充**（schema 校验强制）。

## 3. 每条件独立进程

- 一个条件 = 一个全新 Python 子进程（`g5_driver.spawn_condition`）；父进程只编排；
- **AR 条件同样保持 target+draft 双模型常驻**，显存驻留条件与 SpecDec 一致；
- 模型加载 / tokenizer / 文件写入**不进** prefill/decode 计时（它们在子进程的计时区间之外）。

理由：8 GB 卡上 8K 双模型时分配器池会越过物理显存，池状态可改变同一格耗时（实测 6.44s ↔ 10.87s）；
只有每格从干净进程状态开始，格与格之间才可比。

## 4. 随机化与 block

- 每个 block 内的条件顺序由 `random.Random(seed).shuffle` 决定，**seed 记录在案**（可重放）；
- 至少 **2 个独立随机顺序 block**（默认 2，seed = `P5_SEED + block`）；
- 每个 block 开始前记录系统状态（nvidia-smi used/temp/power/clocks + CUDA free）。

## 5. 计时边界（写死在代码里）

- 计时区间由 CUDA 同步夹住：`torch.cuda.synchronize()` 在区间两端各一次（子进程内）；
- 计时区间内**禁止**：`empty_cache()`、`nvidia-smi`、文件写入、文本 decode、除同步外的诊断；
- 条件之间 cooldown（默认 10 s）**不进计时**；
- 每个条件 warm-up 1 次并丢弃，measured reps = 5。

**CUDA 同步的意义**：`perf_counter` 只测 CPU 侧排队时间，GPU kernel 是异步的；只有
`synchronize()` 之后的墙钟才包含真实 GPU 执行时间。因此"decode_seconds"=
（prefill 同步点之后）到（生成结束同步点）之间的墙钟。

## 6. 显存三口径（不得混用）

| 字段 | 来源 | 含义 |
|---|---|---|
| `peak_allocated_gib`（**主指标**） | `torch.cuda.max_memory_allocated()` | PyTorch 实际张量分配峰值 |
| `peak_reserved_gib`（辅助） | `torch.cuda.max_memory_reserved()` | 分配器池峰值（含空闲块/虚拟预留，**≠物理占用**） |
| `nvidia_smi_used_mib`（物理参考） | `nvidia-smi memory.used` | 驱动视角物理占用 |

`peak_reserved > 8 GiB` **不得**解释为"使用了超过 8 GB 显存"。8K 双模型实测
`used ≈ 7811–7887 / 8188 MiB`、`cuda_free ≈ 0.00 GiB` ⇒ 该运行点按**显存饱和**标注。

## 7. 稳定性判据与预登记处理

- 报 `median / min / max / MAD / IQR` + within-run rep spread + between-block（顺序）差异；
- `20%` 只作**"需要调查"的诊断线**，**不是**研究协议规定的性能 PASS 门槛；
- 预登记处理：`between-block 相对差 > 20%` ⇒ 该条件 `status=CHECK`，
  **不得进入正式主表**；数据保留、不删除、不重跑挑选。

## 8. 正确性口径

- **主 Gate**：FP32 `B6` 逐 token 精确相等（当前 PASS：20/20 组合，216/348）；
- BF16 出现差异时必须记录 `first_diff_index` + `top1/top2 margin` + 是否已知 near-tie；
- 只有**同时满足**下列四条才允许标 `BF16_NEAR_TIE`（由 `results_schema.classify_bf16_compare` 强制）：
  1. 结构不变量通过（`invariant_violations == 0`）
  2. `off_by_one == False`（判定行不比相邻位置更远）
  3. 同位置明显比相邻位置更近（`own_max_abs < min(prev, next)`）
  4. `|fp32 margin| <= own_max_abs`（现场 bf16 形状噪声）
- 不满足任一条 ⇒ `NEEDS_INSPECTION`，**不得豁免**。

## 9. Profiling 与正式计时分离

- 正式计时：无任何插桩；
- 成本分解（`g5_cost_breakdown`）用外部 CUDA Event 包装（**不改冻结文件**），其数字
  **不得写入正式主表**；`P5_PROFILE=0` 可跑同一条件但无包装，用于量化插桩自身开销；
- `kv_cat` 是 forward 阶段的**子集**，不可与 forward 阶段相加。

## 10. 本轮的 exploratory 工具（数字不入正式主表）

| 工具 | 作用 | 状态 |
|---|---|---|
| `g5_stability_probe` | 12 条件 × 2 随机 block 的稳定性探针 | 已实现，**未运行** |
| `g5_cost_breakdown` | AR/SpecDec 阶段成本分解 | 已实现，**未运行** |
| `g5_k_sweep` | k ∈ {1,2,4,8} 诊断扫描 | 已实现，**未运行** |

## 11. 进入正式 Pilot 的最低条件

1. 所有运行可审计且不覆盖旧数据 —— ✅（schema + run_id + 拒绝覆盖，43 项 CPU 自测）
2. AR/SpecDec 驻留与计时边界一致 —— ✅（协议 §3/§5 写死）
3. 跨顺序差异已量化 —— ⬜ **需跑 stability probe**（工具已就绪）
4. 8K 显存饱和被明确标注 —— ✅（§6 + 探针自动判定）
5. 性能不稳定时有预先规定的处理 —— ✅（§7）
6. profiling 与正式 timing 完全分离 —— ✅（§9）
7. 正确性状态未因追求速度而降低 —— ✅（§8；FP32 B6 PASS 保持）

## 12. 命令清单

```powershell
python -m specdec.engine.g5_schema_selftest_cpu     # CPU：schema/驱动自测（43 项）
python -m specdec.engine.g5_stability_probe         # GPU：12 条件 × 2 随机 block（Task B）
python -m specdec.engine.g5_cost_breakdown          # GPU：阶段成本分解（Task C，profiling）
python -m specdec.engine.g5_k_sweep                 # GPU：k 扫描（Task D，诊断）
```
