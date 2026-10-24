# SpecDec 独立确认测量 —— 预注册（Pilot）

日期：2026-10-24
状态：**预注册（pilot 运行之前）**。本文件只登记协议与判据；pilot 结果与裁决见
`docs/confirmatory_pilot_report_2026-10-24.md`。旧正式批次（192 tasks / 480 measured）的原始记录、
CHECK 标签、Route C 与报告全部保留，不覆盖、不重标。

## 1. 目的

第一轮正式采集完整（192 进程 / 480 measured / 16 formal family / 2K/4K/8K × AR/SpecDec /
每 family-condition 5 次 measured = block0 3 + block1 2），但按 block 分期执行 ⇒ block 身份与运行时期
相关（4K AR：block1 在 13/16 family 更慢、中位 +28.4%；SpecDec 10/16、+8.1%）。本 pilot 验证一个
**block 时间交错 + AB/BA 配对平衡**的独立测量流程能否稳定测量 2K/4K/8K，并给出是否可进入完整确认
实验的裁决。**不启动完整 16-family 确认实验、不改旧数据。**

## 2. 冻结配置（不得改变）

| 项目 | 固定值 |
|---|---|
| Target / Draft | Qwen2.5-1.5B-Instruct / Qwen2.5-0.5B-Instruct |
| Compute dtype | BF16 |
| Target KV / Draft KV | base_bf16（引擎字符串 `fp16`）/ base_bf16 |
| Attention | `cudnn_prefill`（只融合 prefill） |
| Decode/propose/verify/repair | eager |
| Context / Chunk | 2048 / 4096 / 8192 × chunk 2048 |
| SpecDec k / max_new | 4 / 64 |
| Decode | greedy argmax |
| Target RoPE | 冻结 CPU-FP32 inv_freq |
| Families | pilot 用开发族 F09/F03/F11/F04（**不使用 FF 正式族调参**）；正式 family 集合不动 |

不加入 CUDA Graph、SDPA、FP8、不同 k/chunk 或其他优化。

## 3. 调度原则（本 pilot 的核心变化）

统计单位仍是 family。对每个 `(family, context)` 建立 AR/Spec 配对单元：

- 两种模式仍跑在**独立进程**（不共享模型/cache/allocator 状态）；
- 一半配对 **AB**（AR→SpecDec），一半 **BA**（SpecDec→AR），由固定 seed 决定，数量差 ≤ 1；
- 每个 measurement block 内，family/context 配对顺序**独立随机化**（seed + block）；
- 两个 block 的执行序列**时间交错**（block0 与 block1 任务交替进入全局序列），不允许"先跑完
  block0 再统一跑 block1"；
- 每 family-condition 仍 5 次 measured：block0 = 3、block1 = 2；每进程 1 次 discarded warm-up。

本 pilot：4 family × 3 context × 2 mode × 2 block = **48 个任务 / 120 measured**。

## 4. 8K 的预注册处理

- 8K 不再自动因旧 Route C 被隐藏；
- 若 confirmatory batch 满足完整性、环境门、有限值与测量稳定性要求，可计算 family-level
  `T_AR / T_Spec`；
- 若未满足预注册门槛，保留绝对 latency / memory / acceptance / workload 数据并标为
  **restricted**；
- **不回填、不解禁第一轮正式批次的 8K ratio。**

## 5. 最小实现（复用正式代码）

- 新增 `experiments/confirmatory/{plan,worker,run}.py`：plan builder（AB/BA、block 交错、seed
  随机化、校验）、worker 包装（复用 `experiments/formal/worker.py` 的模型计算与生成逻辑，只附加
  元数据字段）、批次编排（入口 18 GiB、串行、10 s 冷却、失败即停、独占目录）。
- 每条 measured record / diagnostic / failure 记录：`confirmatory_batch_id`、`pair_id`、
  `pair_order`（AB/BA）、`pair_position`、`measurement_block`、`schedule_seed`、`planned_order`、
  `actual_start_utc`、`worker_started_utc`、family/context/mode。
- 不改 `experiments/formal/*`、`model_self.py` 数学路径、权重、BF16、KV、RoPE、attention、接受规则。

## 6. 验收（CPU/dry-run，先于 GPU）

1. 同 seed 计划完全相同；2. 不同 seed 不改变任务集合、只改顺序；3. 无重复/缺失任务；
4. 每 (family,ctx,block) 恰含 AR 与 SpecDec；5. AB/BA 差 ≤ 1（同配对顺序一致）；
6. 两 block 真交错（max 连排 ≤ 2、首尾都含两 block、transitions 足够）；
7. measured 总数正确（48 任务 / 120 measured）；8. 输入 hash 与 pilot manifest 一致；
9. dry-run 不加载模型、不触碰 GPU、不建输出目录；10. 计划 JSON 读回并通过校验；
11. worker dry-run 与计划逐字段一致、篡改 meta 被拒；12. 默认生产路径不变。

## 7. Pilot 验收与分析

按 family/context/mode 汇总 5 次 measured 的 decode median/min/max/MAD/IQR、within-process spread、
两 block 中位差、boundary qualification、AR/Spec 输出是否相同与 first divergence、Spec 的
acceptance/rounds/proposed/accepted/mismatch/target verify/repair calls、peak allocated/reserved、
全部失败与环境拒绝。重点比较旧流程与新流程：4K AR block1 vs block0 中位偏移、4K all-six-check
OK 数量、8K 是否有可报告稳定 pair、AB/BA 是否有系统性顺序差、SpecDec 是否仍明显慢于 AR。
**不用 pilot 结果筛选正式 family。**

## 8. 停止条件

OOM、原生崩溃、环境门拒绝、NaN/Inf、输入 hash 不符、权重校验失败、cache/round invariant 失败、
任务重复/遗漏/顺序记录缺失、同模式重复生成不一致、8K 资源异常、或发现 runner 修改了模型数学/
生成逻辑 ⇒ 立即停止后续 GPU 任务并保留证据；不自动重试、不换参数。AR/Spec 输出不同不是硬失败，
记录 divergence 后继续。

## 9. 交付

1. 本预注册文档；
2. confirmatory plan/worker/run 代码；
3. CPU/dry-run 自测结果；
4. pilot 原始产物目录；
5. `docs/confirmatory_pilot_report_2026-10-24.md`；
6. CSV：pilot task records / family-level timing / block shifts / AB-BA order comparison；
7. pilot 稳定性图（SVG/PNG/PDF）；
8. 裁决：`READY_FOR_CONFIRMATORY_FULL` / `NEEDS_MEASUREMENT_FIX` / `ENVIRONMENT_BLOCKED`。

## 10. 裁决判据（pilot 的决策辅助，非科学阈值；六项检查沿用旧正式审计的 20% 诊断阈值）

沿用旧审计定义：`within` = 单任务内 throughput 极差/中位 >20% ⇒ CHECK；`between` = 该
(family,ctx,mode) 两 block 的 throughput 中位差/max >20% ⇒ CHECK；`boundary` = 所有 rep 的
boundary clock qualified 且证据 COMPLETE。all-six-OK = AR 与 SpecDec 各三项同时 OK。

- **ENVIRONMENT_BLOCKED**：pilot 期间出现环境门拒绝 / OOM / 崩溃 / hash 或权重校验失败 / 任务
  中断，导致无法覆盖 48 任务或 120 measured（或 8K 资源异常）。保留证据，停止。
- **NEEDS_MEASUREMENT_FIX**：48/48 任务完成且 120 measured，但满足任一：
  ① 4K all-six-OK 的 family 数 = 0/4；或 ② 存在 ≥3 个 (ctx×mode) 单元出现统一方向的 AB/BA
  顺序差 >20%；或 ③ 任一 measured 记录的 boundary 证据非 COMPLETE；或 ④ 任务重复/遗漏/顺序
  记录缺失。
- **READY_FOR_CONFIRMATORY_FULL**：48/48 任务完成、120 measured、无失败/拒绝、边界证据全
  COMPLETE，并且 4K all-six-OK ≥1/4、2K ≥1/4、8K 稳定 pair ≥2/4（8K 稳定 = 六项检查全 OK），
  且 AB/BA 顺序差未出现"≥3 个单元统一方向 >20%"。
- 参考旧批（16 formal family）：2K 3/16、4K 0/16、8K 5/16 all-six-OK；4K AR block1 中位 +28.4%
  （13/16 更慢）、SpecDec +8.1%（10/16）。
