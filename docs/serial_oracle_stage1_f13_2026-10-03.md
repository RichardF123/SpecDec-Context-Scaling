# Serial Target Verification Oracle — Stage 1（F13）报告（2026-10-03）

> 设计（代码之前冻结）：`docs/serial_target_oracle_design_2026-10-03.md`
> Stage 0 报告：`docs/serial_oracle_stage0_report_2026-10-03.md`
> 本轮任务书：重跑 F13 Stage 1（三模式 × 2 独立进程），入口门槛 commit ≥ 18 GiB。

## 0. 结论速览

| 项 | 结果 |
| --- | --- |
| **本轮判决** | **`ENVIRONMENT_BLOCKED`** |
| 原因 | 开跑前环境门实测 available commit = **14.441 GiB**，低于 harness 安全线 16 GiB（更低于本轮 18 GiB 入口门槛） |
| 是否加载模型 | **否**（工作树 clean、未设置 `SMOKE_ALLOW_UNQUALIFIED`、未降低任何安全线） |
| 是否执行 Stage 1 | **否**（0/6 次运行） |
| 是否执行 Stage 2 | **否**，且**不允许建议**执行（只有 `SERIAL_ORACLE_STAGE1_PASS` 才可建议） |
| 关于 serial oracle 的结论 | **无**（真实模型上一条数值证据都没有） |
| 这不是 | 不是 `SERIAL_ORACLE_STAGE1_FAIL`（任务书明确禁止把环境拒绝归为此类）；也不是 `BASELINE_REGRESSION`（AR/production 锚点**未验证**，无从判回归） |

## 1. 本轮实际执行了什么（时间线，UTC）

| 时间 | 动作 | 结果 |
| --- | --- | --- |
| 15:11:59Z | Stage 1 聚合器首次运行（针对已存在的记录做口径演练） | `CHECK_NEEDS_REVIEW`（0 个可用运行） |
| 15:12:35Z | **一次性开跑前环境检查**（`serial_oracle_env_check.py`，不加载模型） | **`ENVIRONMENT_BLOCKED`**，退出码 1 |
| 15:13:49Z | Stage 1 聚合器最终运行（本轮记录状态） | `CHECK_NEEDS_REVIEW`（0 个可用运行） |

**在环境门之后没有执行任何生成、没有加载任何模型、没有再重试。**
环境检查内部唯一的 GPU 动作是 harness 自带的 **P-state 负载预检**（`P5V.PF.qualify()`，约 6 秒矩阵乘）；
除此之外本轮未消耗 GPU 计算。

### 1.1 环境门拒绝的历史尝试（全部在**加载模型之前**被拒，作为记录保留）

| 时间（UTC） | 方式 | 读数（available commit） | 是否产生 runner JSON |
| --- | --- | --- | --- |
| 14:00:38–14:01:28Z | 批运行 6 次（ar/prod/serial × rep1,2） | 15.2 GiB（6/6 全拒） | **否**（当时 runner 后处理 bug，已修并记录为缺陷 D5） |
| 14:06:53Z | 单次门试探（AR） | 14.9 GiB | 是 |
| 14:38:15Z | 单次门试探（AR） | 15.5 GiB | 是 |
| 14:44:22Z | 单次门试探（AR） | 14.9 GiB | 是 |
| 15:12:35Z（本轮） | **一次性环境检查** | **14.441 GiB** | 是（环境检查产物） |

⇒ 累计 **10 次**环境门拒绝，**0 次**模型加载成功。

## 2. 环境门结果（本轮，逐项）

产物：`output/g5/attn/serial_oracle_env_check_20261003T151235Z.json`

| 项 | 实测值 |
| --- | --- |
| 工作树 | commit `8f477743608bc4550f8f071296d0b747a4fab8dd`，`dirty=False`，**clean** ✅ |
| commit available（harness 口径） | **14.441 GiB**（limit 55.64 GiB，used 41.20 GiB） |
| harness 安全线（**未改**） | 16.0 GiB ⇒ `commit_guard` 判 **失败**：`commit 可用 14.4 GiB < 安全线 16 GiB` |
| 本轮入口门槛 | 18.0 GiB ⇒ margin **−3.559 GiB**，不通过 |
| 物理内存 | 31.64 GiB 总 / 7.98 GiB 可用 |
| P-state | `qualified=True`，`sm_median=2370 MHz`，`mem_median=8000 MiB`，`reason=ok` ✅ |
| GPU | RTX 4060 Laptop，driver 555.97，8188 MiB 总 / 500 MiB 已用 / 10% 利用率 / 61 °C |
| GPU 计算进程 | nvidia-smi 列出 **21** 个；`P5V.gpu_consumers().count = 22` |
| 其中 python/实验进程 | **0 个**（逐条核对：ToDesk、BaiduNetdisk、CloudMusic、DSH Desktop、ChatGPT/Codex ×3、EdgeWebView ×3、Teams、WeChat 插件、FastLink、Doubao、Outlook、Windows shell 组件） |
| 版本 | torch 2.6.0+cu124 · CUDA 12.4 · cuDNN 90100 · transformers 5.16.1 · Python 3.11.9 |
| `SMOKE_ALLOW_UNQUALIFIED` | 未设置（`allow_unqualified_env_set=false`） |

**判决阶梯命中分支**（写死于模块内，未事后调整）：`free < 16 GiB` ⇒ `ENVIRONMENT_BLOCKED`。

### 2.1 判定器的**开放缺陷**（本轮如实报出，**未**在事后修改判据）

`gpu_residue.clean` 的判据被写成 `count == 0`，而这台机器**任何时刻**都有约 21 个桌面程序持有
GPU 上下文（历史成功的 24 轮协议与 F13 基线运行时同样如此）。因此该子判据在本机**永远为 false**。

* **本轮影响**：无 —— 判决由 commit 分支先行决定（14.441 < 16 < 18）。
* **本轮处置**：**不**修改判据（避免"看到输出后调判据"），只在报告里记录；正确的口径应为
  "列表中不存在本仓库 venv 的 python/实验进程"（本轮该口径**通过**：0 个 python 进程）+
  GPU 显存余量阈值。
* **下一轮前置项**：该子判据必须先由你来定口径，否则环境检查在本机不可能通过。

## 3. AR 历史锚点：**未验证**

运行前已登记（写死在 `serial_oracle_env_check.py` 并落盘，时间戳早于任何运行，**运行后未修改**）：

| 字段 | 预登记期望 |
| --- | --- |
| `ar.expected_gen_hash` | `6e7522bf2521ad7b` |
| `ar.expected_len` | 64 |
| `ar.expected_stop` | `max_new_reached` |

**本轮实测：无**（未运行 AR 的任何一次）。因此：
* 不能声称"新 runner 未改变 AR 路径"——该命题只能由
  `ar_gen_hash == 6e7522bf2521ad7b` 的**实测**来证明，本轮没有这个数据；
* 同理**不能**声称发生回归（`BASELINE_REGRESSION` 需要"跑出来但复现失败"，本轮是"没跑"）。

## 4. production（block verify）历史锚点：**未验证**

| 字段 | 预登记期望 |
| --- | --- |
| `prod.expected_gen_hash` | `e90dad2b3fcf1c65` |
| `prod.expected_len` | 64 |
| `prod.expected_stop` | `max_new_reached` |
| `prod.expected_first_divergence_vs_ar` | `47` |

**本轮实测：无**（未运行 production 的任何一次）。

## 5. Serial oracle 调用几何审计：**未测量**

审计器本身已就绪并**已在 CPU 上自测通过**（`serial_oracle_audit_selftest_cpu.py`，PASS 19/0）：
阶段感知（prefill 分块不算 generation）、按实例身份分组（target/draft 用 `is` 判定，
绝不按 head_dim/调用顺序猜）、非 prefill 阶段 `s>1` 立即抛错、反空泛（零调用/零 generation 调用
一律**不**判通过）、层数与参数量交叉核对。

但**真实运行中的几何**（需求项 1–16）本轮**一条都没有测**：

| 需求项 | 本轮状态 |
| --- | --- |
| 1 `diagnostic_only=true` | 未测量 |
| 2 `target_verify_mode="serial"` | 未测量（模式**未安装**，因为未加载模型） |
| 3 row-aligned 未启用 | 未测量（运行时状态；代码层面本轮确认未安装 wrapper） |
| 4 target generation 前向全 `seq_len==1` | 未测量 |
| 5 无 block（`s>1`）generation 前向 = 0 | 未测量 |
| 6 调用审计非空集合 | 未测量 |
| 7 `target_calls == proposed + rollbacks` | 未测量 |
| 8 draft/target 按实例身份区分 | 未测量 |
| 9 分块 prefill 未被误判为 generation | 未测量 |
| 10 每轮 trace 完整 | 未测量 |
| 11 round invariants | 未测量 |
| 12 cache 结构/长度/dtype | 未测量 |
| 13 logits/cache finite | 未测量 |
| 14 gen IDs 合法 | 未测量 |
| 15 stop reason 合法 | 未测量 |
| 16 无 OOM | 未测量 |
| 17 两次独立 serial 一致 | 未测量 |

> **口径澄清（已写进聚合器与代码注释）**：生产 `specdec_stats.target_block_calls` 的历史含义是
> "**每轮一次 verify 调用**"的计数器，**不是**"`s>1` 的 block 前向数"。serial 模式下该计数器=轮数。
> 因此需求项 5 的实质（不存在 block 前向）由审计的**逐阶段 seq_len 明细**数出的
> `target_n_gen_calls_with_s_gt_1` 给出（必须为 0）。两者都如实记录，不做"改名冒充"。

## 6. Serial 与 AR 的 token equality：**未测量**

核心命题（"若 target verification 完全走与 AR 相同的逐 token `M=1` 路径，SpecDec 控制逻辑是否生成与
target-only AR 完全相同的 token IDs"）**本轮没有得到任何真实模型证据**。
不存在 `first divergence index` / round / proposals / `g` / `m` / branch / cache before-after
可记录 —— 因为没有任何一次生成。

## 7. 已证明 / 未证明（严格区分）

| 已证明（CPU 层，本轮之前已完成并提交） | 未证明（需要 GPU 真实运行） |
| --- | --- |
| serial verifier 与 block verifier 的 **contract 一致**（`(rows, cache_blk)`，`rows.shape==(kp,V)`），控制逻辑逐轮等价（Stage 0 用例 8） | serial 在真实模型上是否生成与 AR **完全相同**的 64 个 token IDs |
| accept / replacement / rollback / EOS（replacement 与 candidate 两条路径）/ 预算截断 / cache 提交与回滚 的状态机正确性（Stage 0 用例 1–7、9–11） | 真实运行中 target generation 调用是否**全部** `M=1`（需求项 4） |
| 与 `smoke_hooks` 的插桩兼容性：`_verify` 是唯一分派点，轮次记账/有限性检查不会被绕过（用例 16） | production block 路径在本 runner 下是否复现历史 `gen_hash` 与首分歧 index 47 |
| serial 与 row-aligned **互斥**；block 模式行为不变（用例 13、15） | 真实运行中的 cache 结构/长度/dtype、轮不变量、有限性、合法停止（需求项 11–16） |
| 审计器**可用且非空泛**（19/0，含"必须能失败"的负对照） | 两次独立 serial 运行的可重复性（需求项 17） |
| 治理守卫在新增 2 条 scoped 检查后 **PASS 29/0** | 任何速度/加速比结论（本轮范围之外，且**不**作为准入判据） |

## 8. 分类纪律（为什么**不能**写成 FAIL 或回归）

* 任务书 §1 的阶梯把"`<16 GiB` ⇒ 记录 `ENVIRONMENT_BLOCKED`、停止、不加载模型"写死；
  §7 明确"不得把环境拒绝分类为 `SERIAL_ORACLE_FAIL`"。本轮严格执行。
* `SERIAL_ORACLE_STAGE1_FAIL` 的成立条件（token 不一致 / 不变量失败 / 出现 `s>1` / 两次不一致）
  都需要**已运行的 serial 数据**；本轮没有运行，因此该判决在逻辑上不可达。
* `BASELINE_REGRESSION` 需要"AR 或 production 跑出来但复现锚点失败"；本轮二者都没跑，
  所以同样不可达（"没跑"≠"跑失败"）。
* 聚合器（只读，`--scope stage1`）对同一批事实给出的分类是 `CHECK_NEEDS_REVIEW`
  （它的词表里没有环境类），并**逐条列出**"环境不合格（未加载模型）"与"缺 AR/prod/serial 运行"。
  两者不矛盾：**本轮判决以任务书 §1 的 `ENVIRONMENT_BLOCKED` 为准**，
  聚合器的 `CHECK_NEEDS_REVIEW` 是同一事实在其词表下的映射。

## 9. 下一轮前置条件（要重新跑 Stage 1，必须先满足）

1. **释出 commit**：需要 `free commit ≥ 18 GiB`（当前 14.44 GiB，缺口 3.56 GiB；harness 的 16 GiB 线也需越过）。
   当前提交量前列：`node` 3.02 · `DSH Desktop` 2.55（我所在环境，别关）· `BaiduNetdiskUnite` 1.44 ·
   `ChatGPT`×3 ≈ 2.55 · `Doubao` 0.66 · `msedge` 0.66 · `QQMusic` 0.60 · `mysqld` 0.57 GiB。
   关掉 BaiduNetdisk + ChatGPT 三进程 + Doubao + QQMusic + mysqld 约 5.8 GiB，可把 free commit 推到 ~20 GiB。
2. **确定 `gpu_residue` 判据口径**（见 §2.1）：否则环境检查在本机不可能通过。
3. 环境检查通过后，严格按任务书 §6 的顺序执行 6 次运行（AR×2 → 检查锚点 → Production×2 →
   检查锚点 → Serial×2 → 独立聚合），**任何一步失败即停，不自动重试**。
4. 入口门槛与所有检查**不得**因结果而调整；`SMOKE_ALLOW_UNQUALIFIED` 继续**不得**使用。

## 10. 本轮产物清单

| 类型 | 路径 | 状态 |
| --- | --- | --- |
| 环境检查 JSON | `output/g5/attn/serial_oracle_env_check_20261003T151235Z.json` | ✅ 存在（本轮新建） |
| Stage 1 聚合 JSON | `output/g5/attn/div8209_so_stage1_aggregate_20261003T151349Z.json` | ✅ 存在（0 可用运行） |
| Stage 1 聚合 JSONL | `output/g5/attn/div8209_so_stage1_aggregate_20261003T151349Z.jsonl` | ✅ 存在（3 条环境不合格记录） |
| Stage 1 聚合 JSON（口径演练） | `output/g5/attn/div8209_so_stage1_aggregate_20261003T151159Z.json` | ✅ 保留（未覆盖） |
| 6 次独立运行的 JSON/日志 | — | ❌ **不存在**（未运行，按 §1 阶梯停止） |
| 环境门拒绝记录 | `div8209_so_F13_ar_rep1_*.json`（3 份）+ `div8209_so_F13_{ar,prod,serial}_rep{1,2}.log`、`div8209_so_batch_*.log` | ✅ 保留 |
| harness 环境不合格记录 | `output/g5/attn/smoke8k_eager_{20261003T140058Z,20261003T140108Z,20261003T140118Z,20261003T140128Z,...}.json` | ✅ 保留 |
| 本报告 | `docs/serial_oracle_stage1_f13_2026-10-03.md` | ✅ 本轮新建 |

## 11. 下一步（需你决定，我不会自行重试）

* 若你释出 commit 并确定 §2.1 的判据口径 ⇒ 我可以按任务书 §6 的固定顺序重跑 F13 Stage 1（6 次，约 15–20 分钟）。
* 在拿到 `SERIAL_ORACLE_STAGE1_PASS` 之前，**不得**建议或执行 F01/F02/F08/F10 Stage 2，
  更不得启动 B6 / pilot / factorial。
* 本轮**未**修改 serial oracle 算法、未修改任何阈值、未覆盖任何历史产物。
