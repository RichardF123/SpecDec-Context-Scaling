# 冻结：SpecDec **分层正确性协议**（形状定位之后，2026-10-03）

> **本文件在运行任何 GPU 之前冻结。** 它把既有的正确性定位结论转成**可执行、可审计**的两层门槛，
> 并**预登记**未见数据验证的 4 个 family。
> **不得**继续对 L2–L27 做逐算子考古；**不得**重新排查 cache / mask / accept-reject / rollback / RoPE / logit index；
> **不**启动正式 factorial；**不**做全模型 FP32；**不**缩短 `max_new`；**不**因某个 family 失败而换样本。

## 0. 前提（已确认事实，本协议不回退、不重证）

1. AR 走逐 token `M=1`；SpecDec 的 target block verification 走 `M=4`。
2. 在**共同历史、相同 token、相同可见前缀**下，CUDA BF16 GEMM 会因**执行形状**产生逐位差异。
3. 该差异已在**五类算子**上复现：QK matmul、`o_proj`、`gate_proj`、`up_proj`、`down_proj`。
4. 内容、stride、view/copy、其他三行内容**均已排除**。
5. 全层 row-aligned 模式能在已知 F13 案例上恢复 AR 的 token IDs，但：logits 仍未逐位闭合、L2 再次出现差异、
   block verify 慢 **1.69×**、decode 慢 **1.26×**、相对 AR 吞吐仅 **0.176×**。
6. ⇒ **row-aligned 只能作为 correctness oracle，不得进入任何正式性能/吞吐/speedup 结论。**
7. 当前结论只能称"**有限精度下的执行形状相关数值非不变性**"；**不得**称为 cuBLAS/PyTorch bug。
8. 正式 B6 与 factorial **尚未获准启动**。

## 1. 两种模式（本协议的核心）

### 1.1 **Production mode**（`mode=prod`，= 现有 `spec` 模式）

| 项目 | 规定 |
| --- | --- |
| 用途 | 后续**正式性能实验**的唯一模式 |
| verify 形状 | **原始 batched block verification（`M=4`）** |
| dtype | BF16 |
| row-aligned | **不启用**（`set_verify_wrap(None)`，注册表为 `None`） |
| 规则 | **不**改 acceptance / replacement / rollback / cache / greedy |
| 允许 | 记录 AR/Spec 的 token 分歧（**写入日志**），但**不得**自动解释为算法错误或"无害误差" |
| 标记 | `diagnostic_only=false` |

### 1.2 **Diagnostic oracle mode**（`mode=diag`，= 现有 `aligned` 模式）

| 项目 | 规定 |
| --- | --- |
| 用途 | **只**用于验证 SpecDec 控制逻辑（accept/reject、replacement、rollback、cache 管理）是否正确 |
| 实现 | `spec_dec.set_verify_wrap(row_aligned.row_aligned_verify)` —— 全层 row-aligned verify |
| 作用域 | **只**在 target 的 block verification 生效；**AR、prefill、draft、普通 decode 不受影响** |
| 性能 | **不进入**任何正式速度/吞吐/speedup 声明 |
| 标记 | 输出**必须**带 `diagnostic_only=true` |

**为什么必须分开**：诊断模式改变的是**执行几何**（`M=4` → 逐行 `M=1`），它让 verify 与 AR 走**同一批几何**，
因此可以把"控制逻辑是否正确"与"生产模式下的数值行为"解耦。若把诊断模式的正确性或时间混入生产结论，
就会出现两类错误：(a) 用 oracle 的正确性冒充生产路径的正确性；(b) 用 oracle 的（更慢的）时间冒充生产性能。
本协议用 `diagnostic_only` 标记 + 互斥的模式名把这两条路**物理隔离**。

## 2. 两层门槛（Gate A / Gate B）

### 2.1 **A. Algorithmic correctness gate**（用 diagnostic oracle）

四个 family 的 diagnostic 运行必须**全部**满足：

1. 完整 generated token IDs 与 **target-only AR** 一致（逐 token）；
2. accept/reject、replacement、rollback 不变量全部通过（逐轮：`tgt_len/drf_len` 与 `m`/`Kp`/`pre` 的关系）；
3. 每层 cache 长度、K/V 结构、dtype 一致（`{k,v}`、bf16、层内等长）；
4. 所有 logits、cache 与中间检查**有限**；
5. 无非法 token（`gen_ids_in_vocab`）；
6. 无 `other_short`；
7. 同配置**两个独立进程**结果一致（AR/prod/diag 各两次）；
8. diagnostic hook **只**作用于 target verify（`applied` 计数 == `target_block_calls`；draft/AR/prefill 零干预）。

**不要求** production mode 的 logits 与 AR 逐位一致。
**任一项失败 = 硬失败**，分类 `ALGORITHMIC_GATE_FAIL`，保留产物、停止，不进入正式实验。

### 2.2 **B. Production numerical-behavior gate**（用 production mode）

四个 family 的 production 运行必须**全部**满足：
finite；cache consistency；round invariants；**两次独立重复一致**；合法停止原因；生成长度符合协议（`64`）；
无结构性失败。**并记录但不自动豁免**：
是否与 AR token 一致、common prefix、first divergence index、两侧 token、决策位置 top-1/top-2 与 margin、
output hash（若工具支持再记 FP32 CPU control）。

AR/Spec token 不同**本身**分类为 **`NUMERICAL_DIVERGENCE_RECORDED`**——
**不得**自动写成 `PASS`、`FAIL`、`harmless` 或 `bug`；最终解释必须结合 **algorithmic oracle** 与结构检查。

## 3. 未见数据验证：**预登记**的 4 个 family

**选择规则（在看到任何结果之前写下，且不因结果改变）**：
从 `docs/prompt_families_manifest.json` 的四个 stratum 中，**各取 `family_id` 最小的一个**，
并排除 **F13**（当前定位的案例）与 **F06**（已参与 row-aligned 对照与 Stage 2 主对照）。

| # | family_id | stratum | source | window_token_start | `text_sha1` | `prefixes[8192].token_sha1` |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **F01** | `S4_mixed` | `wikitext-2-raw-v1/wiki.train.raw` | 0 | `36cdebc500b31686` | 见 manifest（运行时由 `family_input` 重建并校验） |
| 2 | **F02** | `S2_high_markup` | 同上 | 8192 | `c9cde291f999d61e` | 同上 |
| 3 | **F08** | `S1_low_entropy` | 同上 | 24576 | `aae3dcc645738ebe` | 同上 |
| 4 | **F10** | `S3_prose` | 同上 | 8192 | `29cc721a3e0265d4` | 同上 |

覆盖 **4 个 stratum**（要求 ≥3 ✓）；**均不含 F13/F06** ✓；输入取自 manifest 已登记窗口，
token 由 `family_input.load_family_ids()` 重建并做 `token_sha1` 校验（校验失败即拒绝、不降级）。

**独立性证明（运行前，来自现有工具）**：

* `python -X utf8 -m specdec.engine.verify_family_independence_cpu` ⇒ **PASS 25 / FAIL 0**
  （`outside_authorized = []`：授权范围之外没有任何 run 消费 manifest family）；
* 该守卫的既有-run 扫描（101 条消费记录）里，**只有 F06 与 F13 被消费过**；
* 对 `output/g5/**/*.json` 做全量文本检索：包含 `F01|F02|F08|F10` 的产物文件数 = **0**
  ⇒ 这 4 个 family **从未被任何 run（含诊断脚本）消费**。

## 4. 固定配置（逐项沿用冻结值）

| 项目 | 值 |
| --- | --- |
| model | target `Qwen/Qwen2.5-1.5B-Instruct`（1.5B） + draft `Qwen/Qwen2.5-0.5B-Instruct`（0.5B） |
| context / chunk | **8192 / 2048** |
| k / max_new | **4 / 64** |
| dtype / target KV | **BF16 / `base_bf16`** |
| attention | **all-eager**（`fused_by_phase = {}`） |
| target RoPE | 冻结的 **CPU-FP32 `inv_freq`**（`inv_freq_cpu_sha = 138c99b109d7affb`） |
| 解码 | greedy argmax；EOS 与停止规则沿用冻结协议（`max_new_reached` / `eos`；**`other_short` 判失败**） |
| 运行 | **串行**、不共享 cache、不自动重试、**每次独立进程** |

## 5. 实验矩阵（24 次）

对**每个** family 运行：

| 模式 | 次数 | 说明 |
| --- | --- | --- |
| target-only AR | **2** | 不加载 draft（target-only 语义） |
| production SpecDec（`prod`） | **2** | 原始 `M=4` block verify，row-aligned 关闭 |
| row-aligned diagnostic（`diag`） | **2** | 全层 row-aligned，仅 target verify；`diagnostic_only=true` |

**4 families × 3 modes × 2 independent repetitions = 24 runs**，**串行**、互不并行。

## 6. 每次运行必须记录

**通用**：commit、dirty worktree、run ID、family provenance（stratum/source/window/token sha）、
完整生成 token IDs、output hash、生成长度、stop reason、finite、cache consistency、round invariants、
peak allocated/reserved、traceback/OOM、P-state 是否合格。

**SpecDec 专属**：rounds、draft calls/tokens、target block calls、accepted tokens、acceptance rate、
replacement rounds、rollback/cache-length 检查、row-aligned hook applied 数量、`diagnostic_only`。

**与 AR 比较**：token equality、common prefix length、first divergence index、AR token、Spec token、
决策位置 top-1/top-2、margin、比较分类（`NUMERICAL_DIVERGENCE_RECORDED` 或 `TOKEN_EQUAL`）。

**时间**：可以记录，但本轮**不得**形成 speedup 声明（diagnostic 模式的时间不具正式意义）。

## 7. 聚合判据与最终决策

| 条件 | 分类 |
| --- | --- |
| 四 family 全部满足 A 的全部 8 条 | **Algorithmic gate PASS** |
| 任一 family 不满足 A | **`ALGORITHMIC_GATE_FAIL`**（保留产物、停止） |
| 四 family 全部满足 B（且 A 同时 PASS） | **`ACCEPTABLE_FOR_MEASUREMENT`** |
| 出现以下任一：diagnostic ≠ AR、production 重复不一致、margin/分歧信息无法提取、family 独立性无法证明、hook 作用到 draft/AR/prefill/普通 decode、任一必需字段缺失 | **CHECK** |

**最终只允许三种决策之一**：
* **`READY_FOR_FORMAL_MEASUREMENT`** ⇔ Algorithmic gate PASS **且** production behavior ACCEPTABLE_FOR_MEASUREMENT；
* **`ALGORITHMIC_GATE_FAIL`** ⇔ 任一 family 的 Gate A 失败；
* **`CHECK_NEEDS_REVIEW`** ⇔ 其余情况（含任何 CHECK 条件、字段缺失、独立性问题）。

**`ACCEPTABLE_FOR_MEASUREMENT` 的含义**：只表示"可以进行性能测量"，
**不**表示 production 的 token 与 AR 完全相同。**只有** `READY_FOR_FORMAL_MEASUREMENT`
才允许建议下一轮启动正式性能实验；**本轮不得自行启动**。

## 8. 可选的 profiler 收尾（不阻塞）

先只检查本机是否已有 `nsys` / `ncu`（**不安装**新软件）。
若存在：只捕获 L0 QK 与 `gate_proj` 的 `M=1/M=4` 最小复现，**最多各一次**，记录 kernel 名与 shape；
**profiler 时间不得进入正式性能结果**。
若不存在、或仍无 kernel 事件：记录 **`PROFILER_UNAVAILABLE`**，**不重试**、不阻塞本轮验证。
（已知：本机 `torch.profiler` 在 CUDA activity 下 `n_cuda_events = 0`，CUPTI 活动静默失效。）

## 9. 实现纪律

* 优先复用现有 runner / manifest parser / row-aligned hook / 比较函数；只做支持 24-run validation 所需的**最小修改**；
* 新模式**默认关闭**；不修改历史产物与历史结论；**不覆盖**旧 JSON/JSONL；
* **不**放宽现有结构检查；**不**使用全模型 FP32；**不**缩短 `max_new`；**不**因失败换样本；
* **不**继续定位 L2 内部算子；**不**启动正式 factorial；
* 所有产物使用**新 tag**，并保留全部失败 run。

## 10. 冻结签名（本协议的机器可读快照）

* 协议版本：`post_shape_localization_v1`（2026-10-03）
* 预登记 family：`["F01","F02","F08","F10"]`（stratum：`S4_mixed, S2_high_markup, S1_low_entropy, S3_prose`）
* 模式名：`ar`（target-only 参照） / `prod`（production） / `diag`（diagnostic oracle）
* 门槛：`A_algorithmic`（8 条） + `B_production_behavior`（7 条）
* 决策空间：`READY_FOR_FORMAL_MEASUREMENT` / `ALGORITHMIC_GATE_FAIL` / `CHECK_NEEDS_REVIEW`
