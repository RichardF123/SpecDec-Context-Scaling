# 正式实验 runner 实现规格（无代码 · 2026-10-06）

> 面向用户本人实现的 runner。**只描述输入/编排/计时/字段/失败与验收**，不提供任何代码。
> 上游文档：`docs/config_freeze_2026-10-06.md`（冻结值与 baseline）、`docs/g5_measurement_protocol.md`（§2/§5/§6/§7b/§7c）、
> `docs/bf16_formal_admission_2026-10-05.md`（准入与数据决策）。

## 1. 条件与数据

| 项 | 规定 |
| --- | --- |
| 条件 | `C1–C6 = ctx{2048,4096,8192} × mode{AR-primary(target-only), SpecDec(production)}`；补充消融 `A1/A2 = 8K v_fp8`（**须先过 FP8 自己的门**，本轮不允许开跑） |
| pilot 数据 | **F09 / F03 / F11 / F04**（`docs/prompt_families_manifest.json`，`v1-draft`） |
| 正式数据 | **`FF01–FF16`**（`docs/prompt_families_formal_manifest.json`，`formal-v1`）；与 pilot **身份级不相交** |
| 输入重建 | 按 `source=wikitext-2-raw-v1/wiki.train.raw` + `block` + `window_token_start` 取该 block 文本 → 冻结 tokenizer 编码 → 切 `[window_token_start, +8192)`；再核对 `prefixes[ctx].n_tokens` 与 `token_sha1`（**manifest 不存 token ids，禁止读该字段**）；沿用 `family_input.load_family_ids(..., verify_hash=True)` 的等价做法 |
| 嵌套前缀 | 2K ⊂ 4K ⊂ 8K 为**同一窗口**的前缀（同一 token 序列的前 N 个）⇒ 三档是同一个续写任务 |
| 块结构 | 每 block（含缺口）独立 300000 字符；窗口按 8192 token 对齐、不跨 block |

## 2. 编排与重复（3+2）

```
for block in (0, 1):                      # ≥2 个随机顺序 block；seed 记档可重放
    order = shuffle(conditions × families, seed=base_seed + block)   # 顺序随机化并记录
    for (condition, family) in order:
        # 该单元 = 一个全新子进程（父进程只编排）
        spawn_process(condition, family, block, measured_reps = 3 if block == 0 else 2)
```
* **warm-up**：每个 `(family, condition, block)` 进程内 **1 次并丢弃**（不进计时、不计入 measured）；
* **measured**：`(family, condition)` 合计 **5 次**，**3+2** 分布在两个 block（block 0 三次、block 1 两次）；
* **缺次数**：任何原因导致某 `(family, condition)` 少于 5 次 ⇒ 报告中标"**未完成 n/5**"，**不得**宣称完整；
* **条件间 cooldown** 默认 10 s，不进计时；每 block 开始前记录系统状态（`nvidia-smi` used/temp/power/clocks + CUDA free）；
* **进程隔离**：AR-primary 进程**从不加载 draft**；SpecDec 进程双模型；两者**绝不**同进程。

## 3. 运行前置门（顺序执行，缺一不可）

1. 治理守卫（family 授权；当前 `verify_family_independence_cpu` = 34/0）；
2. 环境门（harness 16 GiB 安全线 + 入口门槛 free ≥18 GiB + 工作树 **clean**）；
3. P-state 预检（**负载期**采样，SM ≥2000 MHz、mem ≥7500 MHz、≥5 s；不合格 ⇒ 该 block `ENV_INVALID` 并跳过正式测量）；
4. 正式集合校验（`scripts/verify_formal_families_cpu.py`，当前 18/0）；
5. 记录 `run_id`(UTC)/`git_commit`/`dirty_worktree`/`seed`/`block_index`/`condition_index`/`condition_order`/`condition_role`。

## 4. 计时边界（写死）

| 量 | 边界 |
| --- | --- |
| `total_seconds` | 权重加载完成、warm-up 之后：**prefill 之前同步点** → **生成结束同步点**（两端各一次 `torch.cuda.synchronize()`） |
| `prefill_seconds` | 包装后的**分块** prefill 区间（8K 必须分块，`chunk=2048`）；`prefill_source ∈ {reported_median, reported_per_rep}`；**禁止**用 `total − decode` 偷算 |
| `decode_seconds` | 同一 rep 内 `total − prefill` |
| **不计入** | 权重/tokenizer 加载、warm-up、cooldown、系统查询、任何诊断或插桩 pass |
| 区间内**禁止** | `empty_cache()`、`nvidia-smi`、文件写入、文本 decode、除同步外的任何诊断轮询 |
| 插桩 | 只在**独立 pass**（外部 CUDA Event）；数字**不得**进正式主表；`P5_PROFILE=0` 用于量化插桩开销 |

## 5. 结果字段（每行 = 一个 `(condition, family, block, rep)`）

* 计时/吞吐：`prefill_seconds`、`decode_seconds`、`total_seconds`、`decode_tok_per_s`、`e2e_tok_per_s`、`prefill_source`；
* 显存三口径：`peak_allocated_gib`（主）、`peak_reserved_gib`（辅，**不得**读成物理占用）、`nvidia_smi_used_mib`（物理参考）；
* 生成与停止：`generated_token_ids`、`generated_token_hash`、**实际生成长度**、`stop_reason ∈ {max_new_reached, eos}`（`other_short` = 失败）、`eos_id`、`tokens_finite`；
* SpecDec 统计（**AR 行一律 `null`**）：`accepted`、`proposed`、`rounds`、`draft_calls`、`target_block_calls`、`mismatch_rounds`、`all_accept_rounds`、`acceptance = accepted/proposed`；
* 跨进程等价性（**由父进程**填）：`ar_equal`、`ar_equal_available`、`ar_equal_certified`、`equality_source`；**缺对照写 `null`，绝不写 `False`**；
* 状态分层：`within_run_status`、`between_block_status`、`environment_status`、`measurement_status`（三层最坏值）；
* 声明门控：`speedup_claim_allowed` + `speedup_claim_blockers`（必须打印具体原因）；
* `availability`：**值的纯函数**（有值 ⇒ true；`null` ⇒ false），每次 finalize 时重算；
* **GB6 分歧记录**（production vs AR 的**每一处**）：首分歧 index、完整共同前缀长度 P、两侧 token、决策位置（绝对位置 + 轮号）、
  该位置 top-1/top-2 与 margin、并列计数、该位置 cache 长度、两侧完整输出、两侧接受率/轮数/target 调用数。

**聚合**：`speedup = T_AR / T_Spec`（分子 **AR**、分母 **Spec**；**>1 ⇒ Spec 更快**）；两侧取**全部 measured rep 合并后的中位数**再相除；
`AR-primary` 是**比较基线**，**不是**公式分母。分歧时该比值标注"**在分歧轨迹上的比较**"，并同时给出 P 与两侧接受统计。

## 6. 失败记录与恢复（不挑样本、不自动重复）

1. **保留失败记录**：环境拒绝/OOM/崩溃/非有限值等一律落盘，标 `ENV_INVALID` 或相应状态；**不删除、不覆盖**；
2. **禁止挑样本**：不得只删慢 rep 或按结果挑 block；块内任一 primary cell 环境失败 ⇒ **整块**重跑；
3. **禁止自动重复直到合格**：每次重跑**先写 `rerun_reason`**（环境原因/机器状态/明确缺失槽位），并在报告中可见；
4. **恢复只补缺失槽位**：中断后只补计划中确实缺失的 `(family, condition, block, rep)`；**已存在 ⇒ 报错不重跑**；
   补齐后仍须满足"5 次、3+2"分布，否则标"未完成 n/5"；
5. **硬停**（v2 §5.2 Q3）：控制逻辑 token 不一致 / 重复不一致 / 非有限值 / cache 或逐轮不变量失败 / 非法停止或非法 token /
   OOM 或崩溃 / 审计空泛或记账恒等式不成立 / 缺 D6 复位读数 / family 越界或工作树不 clean / 环境不合格。

## 7. 验收与声明资格

1. **每条件（GB1–GB5）**：finite（决策用 logits 全覆盖、`first_bad` 空）；cache/逐轮结构不变量；重复一致；
   合法停止；无 OOM/结构失败/异常；长度符合冻结规则；
2. **GB6**：分歧逐处记账（不自动 harmless、不自动归因）；
3. **声明资格**（`§7b.4` 十条，缺一即 `speedup_claim_allowed=false` 并打印原因）；
4. **写盘前自洽校验**（`§7c.5` 十二条）：任一条不过 ⇒ **拒绝写出**；
5. **报告纪律**：不稳定单元保留在完整表中并标 `CHECK`；serial/row-aligned 耗时**永不**进入速度结论；
   不得声称"同输出无损加速"；不得用 KV PPL 或自生成 PPL 声称"分歧后质量等同"。

## 8. 明确**不**要做的（本轮边界）

* 不要用历史（2026-09 `eager`）8K 波动**预判**当前配置失败；**不要**在 pilot 前砍掉 C5/C6；
* 不要按 pilot 的接受率/速度/分歧/质量结果**筛选**正式 family（正式集合已在 pilot 前按固定规则选定并校验）；
* 不要修改 `v1-draft`、不要复用已消费窗口、不要扩大条件数；
* 不要开跑 FP8 档（A1/A2）、k 扫描、生成质量或数值机制研究（均独立待办）。
