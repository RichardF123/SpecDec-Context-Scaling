> **SUPERSEDED SCOPE NOTICE — HISTORICAL MEASUREMENT PROTOCOL.**
> Its measurement discipline remains useful historical context, but its
> 12-condition matrix is not the current formal course scope. Current scope is
> the frozen **6 main + 2 supplement** design in
> [`proposal_freeze_2026-10-01.md`](proposal_freeze_2026-10-01.md); also consult
> [`README.md`](../README.md) and [`status_update_2026-10-02.md`](status_update_2026-10-02.md).

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

## 3. 每条件独立进程 + **AR 基线口径（v1.1 修正）**

### 3.1 三种资源口径（不得混用）

| 口径 | 驻留模型 | 角色 | 是否入 3×2×2 factorial | 能否当 speedup 分母 |
|---|---|---|---|---|
| **`AR-primary`**（正式主基线） | **仅 target** | primary | ✅ | ✅ |
| `AR-dual-resident-control`（诊断控制） | target + draft（draft 仅常驻，**不参与** AR 的 prefill/decode） | diagnostic_control | ❌ | ❌ |
| `SpecDec` | target + draft（算法自身必需） | primary | ✅ | ❌ |

正式 factorial 仍是 **12 cells**：`ctx{2K,4K,8K} × KV{base_bf16,v_fp8} × mode{AR-primary, SpecDec}`。
额外的 **6 cells**（`ctx × KV × AR-dual-resident-control`）是**诊断控制，不计入 factorial**，
启用方式 `P5_INCLUDE_DUAL_AR_CONTROL=1`；汇总中必须单独分组，**不得写成"18 条件 factorial"**。

**speedup 方向统一**：`speedup = T_AR / T_Spec`（>1 = SpecDec 更快，<1 = 更慢），
且只能同 context、同 KV、分子为 SpecDec、分母为 **AR-primary**。
**residency penalty**：`penalty = T_AR(target-only) / T_AR(dual)`；`penalty < 1`（等价
`slowdown = T_dual/T_target > 1`）⇒ **双驻留本身**就拖慢了 AR。

### 3.2 为什么正式 AR 是 target-only

- 真实 AR 推理**不需要 draft**：`AR total = target prefill + target decode`；
- 每种方法应承担**自身真实所需资源**；让 AR 背上它根本不用的 draft，会把"资源需求差异"
  错误地记成"方法速度差异"；
- 尤其在 8K×8GB 上，额外驻留会推高显存占用、触发 WDDM 换页或改变分配器状态，
  **人为压低 AR**，从而系统性地偏向 SpecDec —— 这会让 RQ1/RQ2 的结论失去意义。

"相同硬件条件"≠"强制相同模型驻留"：可比的**硬件与计时边界**应相同（同 GPU、同 dtype、
同 KV 格式、同 prompt、同引擎原语、同 CUDA 同步边界），但**资源必须各自真实**。

### 3.3 为什么仍保留 dual-resident 控制组

它的用途是**隔离**：回答"draft 仅常驻、不参与任何计算时，AR 是否已经变慢"，
从而把"额外模型驻留/显存压力"从 AR 的性能里分离出来，并解释 8K 的 WDDM/显存饱和效应。
它**不进入**正式 factorial，也**不作为**正式 speedup 分母。

### 3.4 历史 Part 4 双模型常驻 AR 数据

历史 `output/part4/*` 的 AR 行是在**双模型常驻**下测的 ⇒ 记为
**`AR-dual-resident exploratory`**：**VALID EXPLORATORY DATA /
INELIGIBLE FOR FORMAL SPEEDUP TABLE**（它可以证明显存饱和与顺序敏感，只是不符合正式主基线口径）。
**不删除、不覆盖、不重写**；其 `ar_residency` 在 G5 schema 中写 `null`（不猜）。

### 3.5 进程隔离（硬约束）

- 一个条件 = 一个全新子进程；父进程只编排；模型加载/tokenizer **不进** prefill/decode 计时；
- **`AR + target_only` 要求该进程从未加载 draft** ⇒ 不能与 SpecDec 条件同进程
  （`results_schema.plan_process()` 强制；违反直接报错）；
- 每 block 条件顺序由固定 seed 随机化（seed 记档可重放），至少 2 个独立 block；
- 每条件 warm-up 1 次 + measured reps 5；条件间 cooldown 不进计时；
- 每个 block 开始前记录系统状态。

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

## 7. 稳定性判据与预登记处理（v1.1 修正）

- 报 `median / min / max / MAD / IQR` + within-run rep spread + between-block（顺序）差异；
- `20%` 只作**"需要调查"的诊断线**，**不是**研究协议规定的性能 PASS 门槛；
- **不稳定 cell 的完整处理规则**（预先规定，避免选择性报告）：
  1. **原始数据永远保留**（不删除、不重跑挑选、不覆盖）；
  2. 该 cell **仍出现在完整结果表中**（12-cell factorial 必须完整展示）；
  3. 必须报告 `median / min / max / MAD / IQR` 与 between-block 差异；
  4. 标记 `status=CHECK`、`speedup_claim_allowed=false`；
  5. **不允许**从该 cell 做确定的加速声明；
  6. 可从**稳定 speedup 汇总 / 显著性结论**中排除（并注明排除理由）；
  7. **不得**让该 cell 从报告中消失（也不得用"剔除后平均"冒充整体结论）。

## 7b. 状态分层与 GPU P-state 门禁（v1.1 第二轮，2026-09-22）

### 7b.1 **`exit=0` 不等于测量有效**

| 字段 | 取值 | 含义 |
|---|---|---|
| `execution_status` | SUCCESS / ERROR / OOM | **程序**是否跑完 |
| `measurement_status` | OK / CHECK / ENV_INVALID / NOT_MEASURED | **测量**是否有效 |
| `status`（兼容） | 执行失败时=执行状态，否则=`measurement_status` | —— |

子进程的 `spread_flag='CHECK:relative-spread>20%'` **必须**把 `measurement_status` 降为 CHECK，
并把 `within_run_check_reason` / `within_run_rel_spread_pct` 记进行里。
`returncode==0` 只是"程序成功完成"，**不能**作为 `status=OK` 的依据。

### 7b.2 GPU P-state 预检（独立模块 `gpu_pstate_preflight`）

- **必须在负载期采样**：空闲时 SM≈210–315 MHz / mem≈405 MHz 属正常节能状态，
  **不得**用空闲快照判资格（否则会把合格机器误判为不合格，反之也不能用空闲时钟高证明合格）；
- 预检运行在 measured reps **之前**，**不进**正式计时：短固定低显存 CUDA 负载
  （2048² fp16 matmul ≈ 8 MiB）+ 后台线程采样 `nvidia-smi`；
- 记录 `samples / median·min·max(SM·mem 时钟、功耗、温度) / duration / 阈值 / verdict`；
  采样不可用 ⇒ `available=false`（**不填 0 冒充**）；
- **阈值可配置**（默认是本机经验值，**不是通用 GPU 标准**）：
  `P5_MIN_ACTIVE_SM_CLOCK_MHZ=2000`、`P5_MIN_ACTIVE_MEM_CLOCK_MHZ=7500`、
  `P5_PSTATE_PREFLIGHT_SECONDS=5`；功耗只作辅助证据，**不作硬 Gate**；
- 预检不合格 ⇒ 该 block 标 `env_qualified=false`，`measurement_status=ENV_INVALID`，
  `speedup_claim_allowed=false`，**并跳过该 block 的正式测量**（避免继续产出整批无效时间数据）；
  已产生的低功耗数据**保留**（它是定位环境问题的关键证据）。

### 7b.3 运行期 P-state 审计（每 cell）

用每个 measured rep 的 `monitor_start/end` 快照（都在计时区间之外、且 GPU 处于负载）
汇总 `active_sm_clock_mhz / active_memory_clock_mhz / active_power_w / pstate_qualified /
pstate_reason`；**不在正式计时区间内轮询 `nvidia-smi`**。
若某 block 内任一 primary cell 因 P-state 失败 ⇒ 该 block 建议整体重跑，
**不允许只删慢 rep 保留快 rep**。

### 7b.4 正式 speedup 的声明资格（10 条全满足才允许 `speedup_claim_allowed=true`）

1. 分母是 `AR` 且 `ar_residency=target_only`；2. `speedup_denominator_eligible=true`；
3. 两侧 `context_actual` / `target_kv` / `max_new_tokens` / **`input_token_hash`** 一致；
4. 两侧 measured reps 完整；5. 两侧 `measurement_status=OK`；
6. 两侧 between-block 未超诊断线；7. 无 OOM / NaN / ENV_INVALID；
8. 生成正确性状态已记录（`tokens_finite=true`）；9. 不是 dual-resident 诊断控制；
10. 行内未混入 profiling 数据。

任意一项不满足 ⇒ `speedup_claim_allowed=false`，且**必须打印具体原因**
（`NO CLAIM：AR between-block CHECK` 等），**不得**输出"可作确定声明"。

---

## 7c. Canonical 记账规则（v1.2，2026-09-22）—— 修正 4 类错记

背景：1.1 版的 canonical 出现 4 类错记（summary `*_med` 全为 null；跨进程等价性写成
**凭空否定**的 `ar_equal=False`；within-run / between-block 的措辞混用；dual-control 误标
`claim_allowed=True`）。以下规则把记账口径**写死**，并由 `validate_canonical()` 在**写盘前**强制。

### 7c.1 字段映射必须按 `kind` 走**显式映射表**
- `kind=summary` → `SUMMARY_FIELD_MAP`（**只认 `*_med`**）：
  `t_prefill_s_med→prefill_seconds`、`t_decode_s_med→decode_seconds`、`t_total_s_med→total_seconds`、
  `decode_tok_s_med→decode_tok_per_s`、`e2e_tok_s_med→e2e_tok_per_s`、
  `peak_alloc_GiB_med→peak_allocated_gib`、`peak_resv_GiB_med→peak_reserved_gib`、
  `accepted_med→accepted`、`proposed_med→proposed`、`rounds_med→rounds`、
  `draft_calls_med→draft_calls`、`target_block_calls_med→target_block_calls`、
  `mismatch_rounds_med→mismatch_rounds`、`all_accept_rounds_med→all_accept_rounds`、
  `accept_rate_med→acceptance_reported`；
- `kind=run` → `RUN_FIELD_MAP`（裸字段，去掉 `_med`）；
- **禁止**用"删掉 `_med`"这类泛化字符串替换（`accept_rate_med` 与 `acceptance` 语义不同）；
- **禁止**用 `total_seconds - decode_seconds` 偷算 prefill（中位数之差 ≠ 差的中位数；
  8K 实测两者相差 ~0.03 s）。`prefill_seconds` 的来源记在 `prefill_source`：
  `reported_median` / `reported_per_rep`（**不产生 `derived`**）；
- `acceptance` 是**恒等式** `accepted / proposed`（`proposed>0` 时），
  子进程汇报的比率另存 `acceptance_reported`；非有限值（NaN）一律视为"没有值"；
- AR 行里 SpecDec 专属统计量（`acceptance`/`accepted`/`proposed`/`rounds`/`draft_calls`/
  `target_block_calls`/`mismatch_rounds`/`all_accept_rounds`/`accepted_length_histogram`）
  **语义上不存在 ⇒ 一律 null**（不得把子进程汇报的 0 当成测量值）；
- `availability` 是**值的纯函数**：值非 null ⇒ true，null ⇒ false；由 `finalize_row()` 每次重算，
  因此"先 finalize 后填值"不会再留下 `值有 but availability=false` 的自相矛盾行。

### 7c.2 跨进程等价性由**父进程**填写
子进程按"一条件一进程"启动，AR 与 SpecDec **从不共存** ⇒ 进程内**没有**可比对象。
父进程按 key `(block_index, context_actual, target_kv, input_token_hash, max_new_tokens)`
配对，且**只**比较 `AR target_only primary` ↔ `SpecDec primary`（**dual-control 绝不作为正确性参照**）：

| 情况 | `ar_equal` | `ar_equal_available` | `ar_equal_certified` |
|---|---|---|---|
| 前提齐全（ctx/KV/input_hash/max_new/EOS/n_generated/ids 或 hash） | ids 逐位比较结果 | true | true |
| 有对照但前提缺失 | **null** | false | false |
| 无对照（key 配不上） | **null** | false | false |
| AR 参照行自身 | null | false | false（`equality_source=reference_row`） |

**缺对照绝不写 `False`**（"没比过" ≠ "不相等"）。判定优先用 `generated_token_ids`
逐位比较；只有缺 ids 时才降级为 hash 比较并在 `equality_reason` 里注明。
`equality_source`：`parent_cross_process` / `reference_row`。

### 7c.3 状态**分层**，措辞不得跨层
| 字段 | 含义 |
|---|---|
| `within_run_status` | **同一进程内** reps 之间的稳定性（来源：子进程 `spread_flag`） |
| `between_block_status` | 两个随机顺序 block 之间的一致性（来源：父进程 two-block 相对差） |
| `environment_status` | P-state / 供电 / 环境资格 |
| `measurement_status` | 三层的**最坏值**（ENV_INVALID > CHECK > OK） |

- `within_run_check_reason` **只**保存 within-run 原因；
- between-block 的原因**另存** `between_block_check_reason`；
- **禁止**把 between-block 导致的 CHECK 写成"within-run CHECK"，也**禁止**拼接
  `within_run_check_reason + " | between-block CHECK"`；
- 正确措辞示例：`NO CLAIM：SpecDec between-block CHECK（42.7% > 20%）`。

### 7c.4 dual-control 的声明纪律
`condition_role=diagnostic_control` 的行必须同时满足
`formal_factorial_member=false`、`speedup_denominator_eligible=false`、
`speedup_claim_allowed=false` —— **即使** execution SUCCESS、within-run OK、between-block OK、
P-state 合格也**永远**没有声明资格。`compute_speedup()` 既已拒绝其作分母，
canonical 行自身也必须一致（`validate_row` / `validate_canonical` 双重拦截）。

### 7c.5 写盘前的 canonical 自洽校验（`validate_canonical`）
1. 已测得的行必须含 `prefill_seconds`/`decode_seconds`/`total_seconds`；
2. SpecDec 已测得必须含 `acceptance`/`accepted`/`proposed`/`rounds`；
3. `acceptance == accepted/proposed`（容差 1e-9）；且 `accepted + mismatch_rounds == n_generated`；
4. `generated_token_hash` 必须等于按 `generated_token_ids` 重算的哈希；
5. `ar_equal=true` ⇒ 两侧 `generated_token_hash` 相同；
6. `ar_equal=false` ⇒ 两侧不同；
7. `ar_equal_certified=true` ⇒ 认证前提字段齐全；
8. dual-control 不得 `speedup_claim_allowed=true`；
9. `measurement_status=CHECK` 必须有结构化原因；
10. `within_run_status=OK` 时不得带 `within_run_check_reason`；between-block CHECK 时
    `speedup_claim_blockers` 不得出现 "within-run" 字样；
11. 字段有值时 `availability` 不得为 false；
12. 字段为空时 `availability` 必须为 false。

任一条不通过 ⇒ **拒绝写出**（`ValueError`），宁可写不出来，也不写一份自相矛盾的 JSONL。

### 7c.6 聚合口径（写死，避免同一实验两个数）
- **canonical 行**：每 `(cell, block)` 一行，逐 block 判定资格；
- **speedup 点估计**：**全部 measured rep 合并取中位数**再相除
  （`median(t_decode_s)` 两侧相除；e2e 同理），与离线报表 `scripts/g5_report_from_run.py`
  的 `[8]` 段完全一致；
- **声明资格**：`speedup_claim_over_blocks()` 要求**每个配对 block 都合格**，
  原因合并去重后打印。

### 7c.7 已发布 canonical 的修订方式
**绝不原地改写**已写出的 canonical（那是审计证据）。修正映射后一律用
`scripts/g5_recanonicalize_run.py --run-id <stamp>` 从**只读的 part4 分片**重建一份
**新 schema 版本**的新文件；`notes` 里写明"由哪些分片重建、未重新测量"。

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
- `kv_cat` 是 forward 阶段的**子集**，不可与 forward 阶段相加；
- `g5_8k_spec_variance_probe` **先**跑无插桩的正式 pass，**再单独**跑插桩 pass；
  两套数字分别存在 `formal` / `profiling` 两个互不覆盖的字段里，且 `profiling` 里
  明确带 `subset_phases`（当前仅 `kv_cat`）以保证闭合求和时不重复计入；
- **计时区间内禁止**：轮询 `nvidia-smi`、`empty_cache`、文件 IO、`print` 之外的任何系统查询
  （本探针的所有系统/显存查询都在计时区间**之外**）。

## 10. 本轮的 exploratory 工具（数字不入正式主表）

| 工具 | 作用 | 状态 |
|---|---|---|
| `g5_stability_probe` | 12 条件 × 2 随机 block 的稳定性探针 | 已运行（第 1 次 ENV_INVALID；缩小版 2K/8K 见 ledger §4.14） |
| `g5_cost_breakdown` | AR/SpecDec 阶段成本分解 | 已实现，**未运行**（等环境 Gate） |
| `g5_k_sweep` | k ∈ {1,2,4,8} 诊断扫描 | 已实现，**未运行**（等环境 Gate） |
| `g5_8k_spec_variance_probe` | 8K SpecDec **跨进程速度双档**专项定位（正式 pass 与插桩 pass 分离） | 已实现，**未运行** |

> `g5_8k_spec_variance_probe` 的**预登记**双档检出规则：把各进程的 SpecDec decode 中位数排序，
> 若存在切分点使**两侧各 ≥2 个进程**且 `min(高档)/max(低档) ≥ P5V_BIMODAL_RATIO`（默认 1.5），
> 报告"检出双档"并给出切分，否则"单档"。该阈值是**可配置的诊断阈值，不是统计显著性判据**，
> 也不作为任何 Gate；改阈值必须**事先**改并在此登记。

### 10b. SpecDec 工具的硬性实现约束（2026-09-22 实测踩中，已修）
`spec_dec.spec_decode_greedy` **内部会自己 prefill**，而引擎默认的 `spec_dec._prefill` 是
**一次前向吃完整个 prompt**。8K 时那一层 fp32 softmax **一次性申请 ≈3.00 GiB** ⇒ 8 GB 卡上必然
`torch.OutOfMemoryError`。因此任何在 8K 上驱动 SpecDec 的工具**必须**：

1. **不要**在调用 `spec_decode_greedy` 之前自己再跑一次 `chunked_prefill`
   （浪费显存，且让 prefill/decode 的划分变成假的 —— 引擎会**再** prefill 一次）；
2. **必须**把 `spec_dec._prefill` 换成**分块 + 计时**版本（本仓库两处已如此：
   `g4_part4_pilot_measure.spec_decode_timed`、`g5_8k_spec_variance_probe.timed_chunked_prefill`），
   调用**一次** `spec_decode_greedy`，然后 `decode = total − prefill`；
3. 插桩 pass 同样要包 `_prefill`（否则插桩 pass 自己会 OOM）；
4. 用 `finally` 还原 `_prefill` / `_propose` / `_verify` / `_step` / `trim_cache` / `torch.cat`。

对照检查：修复后 8K SpecDec prefill 实测 **29–32 s**、decode **6.7–11.7 s**，
与缩小版 run 的 31.96 s / 6.72–11.72 s 一致；插桩闭合残差 **0.1%**。

## 11. 进入正式 Pilot 的最低条件

1. 所有运行可审计且不覆盖旧数据 —— ✅（schema + run_id + 拒绝覆盖；`g5_schema_selftest_cpu` 151 项 / `g4_part4_selftest_cpu` 68 项全 PASS）
2. AR/SpecDec 驻留与计时边界一致 —— ✅（§3：AR-primary = target-only；边界写死）
3. 跨顺序差异已量化 —— ⬜ **8K SpecDec 仍 42.7%（缩小版实测）** ⇒ 先跑 `g5_8k_spec_variance_probe`
4. 8K 显存饱和被明确标注 —— ✅（§6 + 探针自动判定）
5. 性能不稳定时有预先规定的处理 —— ✅（§7 / §7b / §7c）
6. profiling 与正式 timing 完全分离 —— ✅（§9）
7. 正确性状态未因追求速度而降低 —— ✅（§8；FP32 B6 PASS 保持）
8. canonical 记账自洽（映射/等价性/分层/声明纪律）—— ✅（§7c；写盘前强制校验）

## 12. 命令清单

```powershell
python -m specdec.engine.g5_schema_selftest_cpu     # CPU：schema/驱动/记账自测（151 项）
python -m specdec.engine.g4_part4_selftest_cpu      # CPU：测量核心自测（68 项）
python -m specdec.engine.gpu_pstate_preflight       # GPU：环境门禁（退出码 0 才可计时）
python -m specdec.engine.g5_stability_probe         # GPU：12 primary（+ 可选 6 control）× 2 随机 block
set P5_INCLUDE_DUAL_AR_CONTROL=1                    # 可选：加上 6 个 dual-resident 诊断控制
python -m specdec.engine.g5_8k_spec_variance_probe --dry-run   # 先看计划（不跑 GPU 条件）
python -m specdec.engine.g5_8k_spec_variance_probe  # GPU：8K SpecDec 双档专项定位（约 60–90 min）
python -m specdec.engine.g5_cost_breakdown          # GPU：阶段归因（AR 用 target-only；dual 可选）
python -m specdec.engine.g5_k_sweep                 # GPU：k 扫描（AR 分母 = target-only，每 ctx 只测一次）
python scripts/g5_report_from_run.py latest         # 离线：报表 + [8] 声明门控（只读）
python scripts/g5_recanonicalize_run.py --run-id <stamp>   # 离线：按新口径重建 canonical（只读分片）
```

> AR-primary 单独测法（诊断/对照，非必须）：
> `set P4_MODE=AR` + `set P4_AR_RESIDENCY=target_only` → `python -m specdec.engine.g4_part4_pilot_measure`
> （进程只加载 target；若同时要求 SpecDec 条件，必须改为逐条件独立进程。）
