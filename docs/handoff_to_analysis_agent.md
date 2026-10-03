> **SUPERSEDED HISTORICAL SNAPSHOT — NOT THE CURRENT PROJECT SCOPE.**
> This 2026-09-21 handoff preserves its period-specific evidence and plans.
> Its 12-condition factorial has been superseded by the frozen **6 main + 2
> supplement** design in [`proposal_freeze_2026-10-01.md`](proposal_freeze_2026-10-01.md).
> For current status, also consult [`README.md`](../README.md) and
> [`status_update_2026-10-02.md`](status_update_2026-10-02.md).

# SpecDec — 分析 Agent 交接报告（2026-09-21）

> 读者：接手**分析与实验设计**的 agent。本文档自包含：项目目标、冻结口径、**已确立事实 / 推断 /
> 未验证**三分类、数据与代码清单、开放阻塞项、环境约束、下一步推荐动作。
> **硬规则：本文档中标注"未运行"的东西一律不得填预期值；任何数字必须能追溯到 `docs/data_ledger.md`
> 的条目或 `output/` 下的原始文件。**
>
> **⚡ 2026-09-26 起请先读 `docs/results_brief_for_analysis_2026-09-26.md`（结果汇报 / 入口文档，一页读完现状）**：
> 它含可声明的事实、**已排除的假设及证据**、红线、决策点与数据索引。本文档作为更完整的历史与协议上下文。

---

## 0. 一句话现状

自研 Qwen2 推理引擎（含 Asymmetric V-only FP8 KV）与贪心投机解码实现**正确性已闭环**
（FP32 逐 token 等价 20/20；索引语义经哨兵 + 真实模型 + 端到端三方验证）。
**性能结论已从"负 + 不稳定"收敛为"可分的两部分"**：
- **2K base_bf16 已可声明**：decode **0.406×**、e2e **0.434×**（`CLAIM ALLOWED`，缩小版口径）；
- **8K base_bf16 只能记录方向**：AR target-only 极稳（0.6%），但 **SpecDec 跨 block 差 42.7%**
  ⇒ 在原因定位前**不得声明 8K 的正式倍数**。
G5 协议与 schema 已升级到 **`g5-results-1.2`**（状态分层 + P-state 门禁 + 跨进程等价性 +
canonical 自洽校验），CPU 自测 **151/151 + 68/68**。
**下一步唯一动作：跑 `g5_8k_spec_variance_probe` 定位 8K 双档；正式 Pilot 仍被禁止。**

---

## 1. 项目与目标

**课题**：从零实现小型 LLM 推理引擎 + 优化阶梯，在 **8 GB 笔记本 GPU** 上做可复现测量。
**研究问题（已冻结）**
- **RQ1**：上下文长度如何影响投机解码的加速比（8 GB 约束下）。
- **RQ2**：加速变化主要来自"接受收益"还是"草案/验证开销"。
- **RQ3**：FP8 压缩 KV（V-only）是否改变投机解码仍然获利的上下文区间。

**课程**：DSAI 4207 "LLMs in Practice"；Rubric：Problem&Motivation 10% / Technical
Implementation 20% / Experimental Design & Evaluation 25% / Analysis & Insight 20% /
Report & Presentation 15% / Creativity 10%。截止：提案 **10/7**、演示 W12–13（**11/18、11/25**）、
终稿 **11/25**。团队 3 人（提案含姓名学号与分工声明，已入库）。

**正式配置（冻结，不得改）**
| 项 | 值 |
|---|---|
| target / draft | Qwen2.5-**1.5B**-Instruct / Qwen2.5-**0.5B**-Instruct |
| 计算 dtype | **BF16**（本机 CUDA fp16 对该 Qwen 路径产生 NaN，已弃用） |
| KV 条件 | `base_bf16`（K/V 均 BF16）、`v_fp8`（K BF16 + V FP8 E4M3，per-token×per-head，gs128） |
| 解码 | AR（cached greedy）与 SpecDec（**greedy、k=4、不发射 bonus**） |
| 正式 factorial | ctx {2K,4K,8K} × KV {base_bf16, v_fp8} × mode {AR, SpecDec} = **12 条件** |
| **AR 基线口径（v1.1）** | **正式 AR = `AR-primary` = target-only**（只加载 target，draft 从不加载）；它是正式 speedup 分母。`AR-dual-resident-control`（双模型常驻、draft 仅驻留）只是**诊断控制**，**不入 factorial、不作分母** |
| speedup 方向 | `speedup = T_AR / T_Spec`（>1 = SpecDec 更快）；`residency_penalty = T_AR(target-only)/T_AR(dual)`，`<1` ⇒ 双驻留本身拖慢 AR |
| prompt | 真实 wikitext 文本的**嵌套前缀**（2K⊂4K⊂8K），不插合成填充；family 用 `train.raw`，质量窗用 `test.raw` |

---

## 2. Gate 状态总表

| Gate | 状态 | 证据位置 |
|---|---|---|
| G1 target 1.5B 正确性 | **PASS** | ledger §3.4（fp32 full 3.099e-05 / cached 2.038e-05；bf16 512-token max\|Δ\|=0、argmax 100%） |
| G2 8K operational | **PASS** | ledger §4.5.1（分块 prefill、3 轮稳定、无 OOM） |
| G3 V-only FP8 KV | **PASS / FROZEN** | ledger §3.5.3（ΔNLL +0.000397、ΔPPL +0.040%、1.320× 逻辑压缩；K-FP8/full-FP8 negative） |
| G4-0 draft 正确性 | CONDITIONAL PASS | BF16 shape drift 已表征（非引擎 bug） |
| B3 双模型 8 GB 可驻留 | **PASS 4/4** | ledger §4.5.2 |
| B4 cache API | **PASS** | ledger §4.5.3 |
| B5 主循环 | **PASS** | ledger §4.5.4（不变量 + EOS 修复后 GPU 复验） |
| **B6-fp32（主正确性 Gate）** | **PASS** | ledger §4.5.6 / §4.7.6（**20/20 组合逐 token 相等，216/348**，两次独立复跑一致） |
| B6-bf16 | **CHECK 19/20** | ledger §4.5.5（唯一分歧 = 已量化 tie-collapse：signal 0.1491 ≤ noise 0.1491） |
| 索引语义 / cache 不变量 | **PASS** | ledger §4.6.5、§4.7（哨兵 35/35 + 真实模型 fp32 语义 + B6-fp32） |
| Part 4 正确性归因 | **CLOSED** | 三条件 = `BF16_NEAR_TIE`，`off_by_one=False` |
| **AR 正式基线（target-only）** | **口径已生效，第 1 次数据 ENV_INVALID** | §1 表；历史 AR 数据是 `AR-dual-resident exploratory`，**不能当正式分母**；新探针已实测 target-only 可运行（`resident_models=['target']`，`weights_resident≈2.88 GiB`） |
| Part 4 性能数据 | **VALID EXPLORATORY DATA / INELIGIBLE FOR FORMAL SPEEDUP TABLE** | 两个原因：① AR 是双驻留口径，不符合正式主基线；② 8K 显存饱和 + between-block 28.9% 不稳定。**数据有效（它证明显存饱和与顺序敏感），只是不符合正式主基线口径** |
| G5 schema / 协议 | **已冻结（v1.2）** | `docs/g5_measurement_protocol.md` §7b/§7c；`results_schema.py`（**151 项** CPU 自测 PASS；part4 68 项） |
| GPU P-state 预检 | **PASS（2026-09-22 两次实测 QUALIFIED）** | `specdec/engine/gpu_pstate_preflight.py`；ledger §4.12；阈值 `SM≥2000 MHz`、`mem≥7500 MHz`、负载期采样 5 s（idle 仅记录不判定）。**预检通过只代表当下合格，长实验中仍靠 block 级复检** |
| G5 稳定性探针（第 1 次） | **ENV_INVALID（7/12 条件 between-block >20%）** | ledger §4.11；`output/g5/g5_g5-results-1.1_5dc7b5eeacd1_g5stab_20260921T090033Z.jsonl` + 24 个 part4 分片 |
| **G5 缩小版（`g5stab_20260922T040625Z`）** | **P-state PASS；2K `CLAIM ALLOWED`；8K SpecDec `between-block CHECK`** | ledger §4.14；1.2 canonical `g5_g5-results-1.2_re44bd9ca4bb_g5stab_20260922T040625Z.jsonl` |
| canonical 记账（映射/等价性/分层/声明纪律） | **PASS（写盘前强制校验）** | ledger §4.14⑤；协议 §7c |
| **8K SpecDec 速度双档原因** | **未定位（唯一未闭环项）** | 下一步 `g5_8k_spec_variance_probe` |
| G5 成本分解 / k 扫描 | **未运行** | ledger §4.8.4（**必须在环境 Gate 与 8K 定位之后**） |
| 正式 Pilot / Final | **禁止** | 见 §9 |

---

## 3. 必须遵守的纪律（硬规则）

1. **不得编造数字**。缺失写 `null` + `availability[field]=false`；遥测字段禁止 0 / 字符串 / NaN 冒充
   （`results_schema.validate_row` 强制）。
2. **不得覆盖历史数据**。G5 文件名含 schema 版本 + config hash + UTC run_id；`write_g5_run` 默认拒绝覆盖。
3. **不得基于被证伪的前提作废数据**。"SpecDec 指标全部 INVALIDATED"目前是
   `INVALIDATION_PENDING`（索引错位假设已三方证伪）；计时数据被排除是**另一条**独立理由（显存饱和 + 顺序不稳）。
4. **不得改**：`spec_dec.py` / `spec_cache.py`（无新的、独立且可复现的失败证据时）、frozen V-only FP8 设计、
   正式模型/dtype、`k=4`、factorial。不得引入 K-FP8 / 权重量化 / FlashAttention。
5. **BF16 差异不得自动豁免**。标 `BF16_NEAR_TIE` 需**四条证据同时成立**（见 §13.4）。
6. **profiling 与正式 timing 分离**：成本分解数字**不得**进正式主表。
7. **不稳定的预登记处理**：`between-block 相对差 > 20%` ⇒ 该条件 `status=CHECK`，**不入主表**，数据保留。
8. **`20%` 只是"需要调查"的诊断线**，不是研究协议规定的性能 PASS 门槛。
9. **只在拿到阶段数据后才归因**：当前"为什么慢"只有算术反推，**未做阶段级归因**。
10. **GPU 脚本必须由人运行**：本 agent 的 shell 跑 GPU 脚本会挂起（见 §10）。

---

## 4. 代码与工具清单

### 4.1 核心引擎（**冻结 / 只读**）
| 文件 | 内容 |
|---|---|
| `specdec/engine/model_self.py`（319 行） | 自研 Qwen2：RMSNorm(fp32 内部) / RoPE(fp32 生成 cos·sin) / GQA / SwiGLU / tied embedding；KV cache 四形态（`fp16`=base、`v_fp8`、`fp8` 负消融、None）；`set_kv_format` / `set_fp8_groups` / `set_fp8_matrices` |
| `specdec/engine/spec_dec.py`（116 行） | `_prefill` / `_propose` / `_verify` / `_step` / `spec_decode_greedy`；**候选判定映射**：`g = [saved_next] + rows[0..kp-2]`（candidate 0 ← prefix logits；candidate i≥1 ← `block_logits[i-1]`；`rows[kp-1]` 只作下一轮 prefix，不产出 bonus） |
| `specdec/engine/spec_cache.py`（65 行） | `cache_length` / `trim_cache`（浅拷贝、沿 dim=2 裁所有张量、`new_len<=0` 或越界抛错）/ `check_consistent` |
| `specdec/engine/align_ours.py` | `build_from_config`（qwen2 默认 `attn_bias=True`） |

### 4.2 正确性 / 门禁 harness
`g4_0_draft_correctness.py`、`g4_0_drift_diag.py`、`g4_block_vs_serial.py`、`g4_b3_dual_model_feasibility.py`、
`g4_b4_cache_tests.py`、`g4_b5_helper_tests.py`、`g4_b5_smoke.py`、`g4_b5_eos_branch_tests.py`、
`g4_b6_correctness.py`、`g4_b6_fp32_correctness.py`、`g4_b6_tie_diag.py`、
`g4_specdec_index_tests.py`（**35/35** 哨兵索引测试）、`g4_block_logits_semantics.py`（**PASS**，真实模型 fp32 语义验证）

### 4.3 Part 4 / G5 测量工具
| 文件 | 作用 | 状态 |
|---|---|---|
| `g4_part4_pilot_measure.py` | 测量核心：12 条件、AR 同档基线、分块 prefill、warm-up/rep、`P4_KV/P4_MODE/P4_CTX/P4_ONLY/P4_K/P4_REVERSE` 过滤、run_id 防覆盖、每 rep 环境监控、`--aggregate` 离线报表 | 已 GPU 验证 |
| `g4_part4_8k_isolated_probe.py` | 8K 四条件 × 两启动顺序，每条件独立子进程；离线报表（within-run / between-order / token 比较 / AR 聚合 / 显存审计 / 环境 / 判定） | 已 GPU 运行 |
| `g4_part4_mismatch_diag.py` | 历史分歧条件定位：first_diff、AR 与判定行 top1/top2/margin、round/m/分支、**own/prev/next 三行对比**、FP32 控制、6 类归因 | 已 GPU 运行（结论见 §6） |
| `specdec/results_schema.py` | **G5 单一 schema**（`g5-results-1.1`，60+ 字段：新增 `resident_models` / `ar_residency` / `condition_role` / `formal_factorial_member` / `speedup_denominator_eligible` / `speedup_claim_allowed` / `execution_status` / `measurement_status` / `within_run_*` / `between_block_*` / `pstate_*` / `env_qualified` / `speedup_claim_blockers`）+ 校验 + 拒绝覆盖 + 旧 Part 4 只读兼容 + MAD/IQR + `derive_statuses` / `speedup_claim` | **89 项 CPU 自测 PASS** |
| `g5_driver.py` | 每条件独立子进程编排、系统快照、固定 seed 随机化、run 元数据、**`clock_stats` / `pstate_verdict` / `pstate_audit` / `platform_state` / `platform_gate`** | CPU 验证 |
| `gpu_pstate_preflight.py` | **环境预检**：单线程采样器记录负载期（2048² fp16 matmul ≈8 MiB）与 idle 期 SM/mem 时钟与功耗，出 verdict；退出码 0 合格 / 3 ENV_INVALID / 2 不可用；写 `output/g5/pstate_preflight_<run_id>.json` | **已 GPU 实测 2 次**：首次暴露 `Sampler._stop` 覆盖 `Thread._stop` 的真实缺陷，修复后 **QUALIFIED**（ledger §4.12） |
| `g5_stability_probe.py` | Task B：12 条件 × 2 随机 block（warm-up 1 + 5 rep、cooldown 不计时）；**每 block 前置 P-state 预检（不合格则跳过该 block）**、每 cell 事后 P-state 审计、状态分层传播、between-block 在全部 block 结束后计算、G5 行在最后统一落盘 | **已运行 1 次（ENV_INVALID）** |
| `g5_cost_breakdown.py` | Task C：SpecDec/AR 阶段分解（外部 CUDA Event，不改冻结文件）；AR 主基线先以 target-only 跑完 | **未运行（等环境 Gate）** |
| `g5_k_sweep.py` | Task D：k∈{1,2,4,8} 诊断扫描（只出建议）；每 ctx 一个 target-only AR 分母 | **未运行（等环境 Gate）** |
| `g5_8k_spec_variance_probe.py` | **8K SpecDec 跨进程速度双档专项定位**：role×procs 个独立进程（spec / ar_target / ar_dual），每进程先跑**无插桩正式 pass**、再跑**单独插桩 pass**；阶段= draft 提议/块验证/修复步×2/cache 裁剪/kv_cat/残差；含 allocator 段状态、`inactive_split`、allocation retries、`mem_get_info`、**尽力**采集 WDDM/共享显存计数器 | **已实现并干跑，未运行** |
| `g5_schema_selftest_cpu.py` / `g4_part4_selftest_cpu.py` | CPU 自测 **151/151** / **68/68** | 已运行 |
| `scripts/g5_report_from_run.py` | **只读离线报表**：within-block / between-block / speedup / 遥测 / 时钟-吞吐对照 / 时钟匹配 speedup / **[8] 声明门控（打印 `NO CLAIM：<原因>`）**；**无参数或 `latest` = 只取最新一次 run**（不混多次 run） | 已运行（`[8]`：2K CLAIM ALLOWED / 8K NO CLAIM） |
| `scripts/g5_recanonicalize_run.py` | **只读分片重建 canonical**（修正映射后重新记账；**不重新测量**；不覆盖旧文件） | 已运行（生成 1.2 canonical） |
| `scripts/run_g5_reduced.cmd` | **一键缩小版复验**：预检 → 干跑 → 复验 → 离线报表；用 `set "VAR=value"` 避免尾随空格污染条件名 | **已干跑验证** |

---

## 5. 数据清单（`output/` 被 gitignore，但**必须原样保留**）

| 文件 | 内容 | 状态 / 锚点 |
|---|---|---|
| `output/part4/part4_dc0bd5d87389.jsonl` | **exploratory_run_1_fixed_order**：12 条件、36 run（固定顺序，含 `empty_cache` 于计时区） | **受保护**；41026 B / 49 行；sha256 `bfb85f509b6802393637ea9cea9d7f5275579deb457ed6456db9bb087664c53c` |
| `output/part4/part4_ba0c985bcd5d.jsonl` | **exploratory_run_2_reverse_order**：4 条件（仅 8192，`P4_REVERSE=1`，REPS=3） | **受保护**；14310 B / 17 行；sha256 `6ad22617a46be08328f980537eb77ed95fb7b38e5548166da3b910f335d4c7c3` |
| `output/part4/part4_*_8k[AB]_20260921T042125Z.jsonl`（8 文件） | 8K 隔离复验：4 条件 × 2 顺序 × 5 rep = 40 run + 8 summary | 已用（§6） |
| `output/part4/part4_*_8kA/B_*.jsonl`（命名规则） | `part4_<config_hash>_<RUN_ID>.jsonl`，RUN_ID 含 UTC | 新 run 不得与旧同名 |
| `output/g5/g5_g5-results-1.1_5dc7b5eeacd1_g5stab_20260921T090033Z.jsonl` | **第 1 次稳定性探针**：12 primary 条件 × 2 随机 block 的 G5 行（97557 B） | **受保护**；`ENV_INVALID`，仅作环境诊断证据（§6.16–6.20） |
| `output/part4/part4_*_g5stab_b[01]_c*_pri_*.jsonl`（24 文件） | 上述探针的每条件原始分片（每 cell 独立子进程，含 token 哈希与 P-state 字段） | **受保护**；离线报表 `scripts/g5_report_from_run.py` 的输入 |
| `output/g5/pstate_preflight_pstate_20260922T035805Z.json` | **环境预检结果**（2227 B）：负载期 SM median 2355 / mem 8000 MHz、89.58 W、82 °C、verdict **QUALIFIED** | **受保护**；**是环境证据，不是性能数据**（§6.22） |
| `output/g5/g5_g5-results-1.1_0e4ed83b6fb3_g5stab_20260922T040625Z.jsonl`（58009 B） | 缩小版 run 的 **1.1 版 canonical（含 4 类错记）** | **受保护 / 保留作审计证据**；sha256 `10045ba8331afe...` |
| `output/g5/g5_g5-results-1.2_re44bd9ca4bb_g5stab_20260922T040625Z.jsonl`（61448 B） | 同一批测量的 **1.2 版 canonical（修正后重建）** | 由只读分片重建；**未重新测量** |
| `output/part4/part4_*_g5stab_20260922T040625Z_*.jsonl`（12 文件） | 缩小版 run 的原始分片（6 条件 × 2 block） | **受保护** |
| `docs/data_ledger.md` | **唯一数据台账**：§3.4 G1、§3.5.x FP8 KV、§4.5.x G2/B3/B4/B5/B6、§4.6 Part 4、§4.7 索引语义与 B6 复跑、§4.8 G5 | —— |
| `docs/g5_measurement_protocol.md` | G5 测量协议冻结 v1 | —— |
| `docs/plan_v7_1_protocol_freeze.md` | 实验协议 v7.1 + **Amendment v7.2**（Compute=BF16、KV 因子、12 conditions） | 权威 |
| `docs/design_freeze_fp8_kv.md` / `design_freeze_g4_specdec.md` / `design_g4_cache_api.md` | FP8 KV v1.3 / G4 v0.3（logits 对齐、回滚语义、`_propose` 出口不变量、EOS 规则）/ cache 契约 | 权威 |
| `docs/prompt_families_manifest.json` + `_design.md` | 16 个 nested-prefix family（4 层 × 4），预登记接受率排序假设 | **草案，待定稿** |
| `docs/pdf/specdec_speculative_decoding_explained.pdf`（19 页）+ `docs/specdec_explained.md` | 投机解码原理/代码/实现细节讲解 | 已完成 |
| `docs/daily_report_2026-09-15*.md`、`2026-09-08b_*.md` | 日报 | 已完成 |

---

## 6. 已确立的实测事实（可引用，均可追溯）

**正确性**
1. FP32 B6：**10 case × 2 KV = 20/20 逐 token 相等**（总 216/348），两次独立复跑逐 case 一致；
   `len2/len5/len13`（预算<k、非 K 倍数）、`eos-prone`、`code/math/short/long` 全过。
2. BF16 B6：**19/20**；唯一分歧（`natural-2@7`）bf16 两候选**精确并列** 22.8750、
   三路径 argmax 一致；fp32 三路径一致给 `' convert'`，gap **+0.1491**；同位置形状噪声 0.250–0.281。
3. 索引语义（真实模型 fp32）：`rows[j]` vs 同位置 serial **max|Δ| 3.9e-05–5.7e-05（argmax 全同）**；
   vs 相邻位置 **14.03–17.25（argmax 全不同）** ⇒ 现实现映射正确，**不存在错位一格**。
4. 哨兵索引测试 35/35：判定行实测 `[pre-1, pre, pre+1, pre+2]`；按"错位一格"映射重算得
   `[6,7,8,9]` ≠ 候选 `[5,6,7,8]` ⇒ 测试有区分力。
5. 三个历史 BF16 分歧条件（512/base、2048/base、2048/v_fp8）：
   同位置 max|Δ| 0.3125 / 0.3086 / 0.5078；相邻 8.6–15.0；**fp32 margin 0.0454 / 0.0179 / 0.0345**
   ⇒ 判据 `signal ≤ 现场噪声` ⇒ **`BF16_NEAR_TIE`**，且 `off_by_one=False`、不变量 0 违反。

**性能（8K，每条件独立进程、双模型常驻、warm-up 1 + 5 rep）**
6. decode tok/s 中位数：A 顺序 base AR **22.90** / base Spec **5.99** / v_fp8 AR **22.06** / v_fp8 Spec **5.69**；
   B 顺序 **23.78 / 8.42 / 20.36 / 5.72**。
7. **within-block 极稳**（MAD 0.02–0.40，相对 0.3–1.7%）；**between-block 不稳**：
   base Spec **28.9%（CHECK）**、base AR 3.7% ok、v_fp8 AR 7.7% ok、v_fp8 Spec 0.4% ok。
8. **决策层完全可复现**：acceptance 四格全 **0.737（56/76）**；接受长度直方图四格全同
   `m=0:5 1:20 2:20 3:0 4:55`（100 轮，sum(m)=280，全接受轮 55%）；rounds 20、block 20、draft 76、mismatch 8。
9. **token 等价**：8K 四格 AR/SpecDec 生成 64 token、哈希全同 `610d0002bdfdcb37`（`ar_equal=true`；
   `certified=false` 仅因该 run 未记录 `input_token_hash`）。
10. `peak_allocated`：AR 6.01–6.06 GiB、SpecDec 6.90–6.95 GiB（**V-FP8 仅降 ~0.05 GiB**）。
11. 运行现场：`nvidia-smi used 7811–7887 / 8188 MiB（95.5–96.3%）`、条件开始时 `cuda_free 0.00 GiB`、
    温度 64→77 °C、**SM 2520–2535 MHz / mem 8000 MHz（无降频）**、功耗 53.7–78.8 W、driver 555.97。
12. 反推单价（算术）：AR **44.3 ms/target 前向**（2.79 s ÷ 63）；SpecDec **534.5 ms/轮**（10.69 s ÷ 20）
    = **12.1 个 AR 步**，而每轮只产 **2.8 token**；四格介于 **8.9–12.2 个 AR 步/轮**。
13. 历史 `1.286×`（8K base Spec/AR）**未复现**（fresh process 全 <1：0.261 / 0.354 / 0.258 / 0.281）。

**其它**
14. G3：V-only FP8 `ΔNLL +0.000397`、`ΔPPL +0.040%`、8K 逻辑 KV 224.00→169.75 MiB（**1.320×**，776 vs 1024 B/token/layer）。
15. G1/G2：见 §2。B3：ctx8192 双模型 alloc 6.16 GiB / resv 8.92 GiB、无 OOM。

**G5 稳定性探针第 1 次（12 primary × 2 block，`20260921T090033Z`；`ENV_INVALID`）**
16. **7/12 条件 between-block >20%**：2048 base AR **66.0%**、2048 base Spec **66.9%**、4096 base AR **69.9%**、
    4096 v_fp8 Spec **68.5%**、8192 base AR **67.4%**、8192 v_fp8 AR **68.8%**、8192 v_fp8 Spec **24.8%**。
17. **2 个 within-run 离群**：4096 base Spec blk0 相对离散 **222.1%**、8192 base AR blk0 **222.4%**
    ⇒ 子进程 `spread_flag='CHECK:relative-spread>20%'` **向上传播**为 `measurement_status=CHECK` 并阻断声明。
18. **P-state 是这批数据的主导变量**：SM 时钟跨度 **435–2535 MHz**；低时钟 block（9 个）**6.16 tok/s**
    vs 高时钟 block（15 个）**17.67 tok/s**（**2.87×**）。
19. **不是热降频 / 不是功耗墙**：慢 run 功耗仅 **15.8–28 W**、温度 **61–68 °C**。
20. **target-only AR 口径实测可跑**：`resident_models=['target']`、`weights_resident≈2.88 GiB`
    （对比双驻留 3.85 GiB）；8K `peak_allocated` base 5.14 / v_fp8 5.09 GiB ⇒ 少驻留一个模型**并未**带来显存压力。
21. **时钟匹配后的 speedup（两侧均 ≥2400 MHz，3 对）**：decode 中位 **0.284**（0.200–0.684）、
    e2e 中位 **0.152**，**全部 <1**。仅按 20% 规则时 2048/v_fp8 会给出
    逐 block decode **0.533 / 0.538**（10 rep 合并中位 **0.533**）、e2e 0.538 / 0.541，
    但按已实现的 **SM≥2000 MHz** 门槛该 cell 判 `ENV_INVALID`（SpecDec 侧 1950/2070 MHz）⇒ **不得声明**。
22. **环境预检实测（2026-09-22T03:58:05Z）**：idle SM 210 / mem 405 MHz（**不得用于判定**）；
    加载 5.63 s 后负载期 SM **median 2355**、mem **8000** MHz、**89.58 W**、**82 °C** ⇒ **QUALIFIED**，退出码 0。
    **这是环境证据，不是性能数据**（`matmul_per_s` 不得引用为吞吐）。

**G5 缩小版 run（`g5stab_20260922T040625Z`；2 长度 × 1 KV × 2 模式 + dual 控制 × 2 block）**
23. **P-state 已修复**：正式条件运行期间 SM **2490–2520 MHz**、mem **8000 MHz**、`pstate_ok=true`；
    第 1 次的 480–1300 MHz / 6000 MHz **未复现**；温度 ≈82 °C、功耗 68–70 W。
24. **2K base_bf16（可声明）**：
    AR target-only decode **33.04 / 29.69** tok/s（between-block **10.1%**）；
    SpecDec **13.26 / 12.79** tok/s（**3.5%**）⇒
    `decode speedup 0.406×`、`e2e speedup 0.434×`、**`speedup_claim_allowed=true`**
    （SpecDec decode 慢 **2.46×**、端到端慢 **2.30×**，合并全 rep 中位数口径）。
    **这是缩小版 stability probe，不是 16-family 正式矩阵。**
25. **8K base_bf16（只说方向，不声明倍数）**：AR target-only **29.33 / 29.51** tok/s（**0.6%**，极稳）；
    SpecDec **9.53 / 5.46** tok/s（**42.7% ⇒ CHECK**）。合并口径 decode ×0.236、e2e ×0.136，
    **该比值未通过稳定性 Gate，禁止写成"8K 正式 speedup"**。
26. **8K 波动的排除项（实测）**：两 block 的 SM 2520 MHz / mem 8000 MHz / ≈82 °C / 68–70 W
    **完全相同**；prefill **31.958 vs 32.000 s**（几乎相同）；SpecDec within-run **1.9% / 2.0%（OK）**；
    ⇒ **不是**低功耗 P-state、**不是**热降频、**不是** prefill 波动、**不是** token 随机性。
    只有 decode **6.72 s → 11.72 s**。**原因尚未定位**。
27. **residency 对照**：2K 时 dual-control 与 target-only 接近（30.4–32.0 vs 29.7–33.0 tok/s）。
    ⚠️ **8K 的"仅多驻留 draft 就让 AR 慢 ≈21%"已在 §6.29 被推翻**（不重现）。
    **与数字无关的口径结论仍然成立**：正式 AR 必须 target-only，dual-control **永不作分母**
    （canonical `claim_allowed=false`）。
28. **canonical 记账已修正（1.1 → 1.2）**：见 §2 表与协议 §7c。1.1 的 4 类错记中，
    最严重的是**跨进程等价性凭空判否**——8192 两侧 `generated_token_hash` 都是
    `610d0002bdfdcb37`，1.1 却写 `ar_equal=False, certified=True`；2.0/1.2 修正后为
    **8192 `ar_equal=True`**、**2048 `ar_equal=False`（`first_diff_index=34`，与已知 BF16 near-tie 一致）**。

**8K 变体探针的尝试（2026-09-23，ledger §4.16）**
29. **两次运行都缺了整个 SpecDec 角色**（`094741Z` / `100031Z`：spec **0/4**，AR 却 4/4 正常）；
    旧版工具**照样打印报告 + 写 summary** ⇒ 假成功，已修：现在
    `spec` 或 `ar_target` 为 0 样本时**拒绝出报告**（退出码 4）并写 `p5v_<rid>_failures.json`。
    这两轮**没有回答任何关于 SpecDec 的问题**；其 `Q1/Q3` 无效。
30. **8K AR 对照极稳（n=8/口径）**：`ar_target` 28.25–29.84（中位 **29.66**）、
    `ar_dual` 28.65–29.98（中位 **29.81**），跨进程极差 ≤**5.6%**；prefill 16 个样本全在 **3.34–3.45 s**。
    ⇒ 这是目前对"8K AR 稳定"最强的证据（比缩小版的 2 个 block 强得多）。
31. **更正 §6.27**：8 个独立进程给出 `residency_penalty ≈ 0.995`（几乎无差），
    **不是 0.79**；缩小版那次连 dual 的 prefill 都是 7.4 s（vs 本次 3.34 s）⇒
    "21%"降级为**单会话观察**。同时说明：**8K 慢档不是 SpecDec 特有**（当时连 AR dual 都慢了）。

---

## 7. 推断（方向可信，机制未隔离）

- 8K 的计时波动**与显存状态一致**（饱和 + 无降频），但**未直接隔离 paging**。
- SpecDec 慢的**主要来源在"每轮固定开销"**（算术支持），但**阶段级归因未测**。
- 每轮 ~280 ms 的"未解释缺口"存在（把 draft 前向按 1/3 权重粗估 + 块验证按 1 步估后仍差约 4 倍）
  ⇒ 候选解释：8K 下 draft 并不便宜、块验证 attention 成本、饱和下的分配/换页、`torch.cat` 逐步增长 KV 的复制。
  **在拿到 §8 的阶段数据前不得择一断言。**

---

## 8. 未验证 / 未运行（**明确清单，不得填预期值**）

1. `g5_stability_probe` 全量 12 条件：第 1 次 ENV_INVALID；缩小版（2 长度 × 1 KV）已通过 P-state，
   但**全量 12 条件仍未在合格环境下跑过**；2048 base_bf16 已有可声明数据。
2. `g5_cost_breakdown`：阶段单价（draft 提议 / 块验证 / 修复步 / cache 裁剪 / KV cat / control）与闭合残差。
3. `g5_k_sweep`：k∈{1,2,4,8} 的 acceptance/tok-per-round/rounds/decode/peak。
4. 生成口径的质量对照（base vs v_fp8 的输出一致率 / NLL）——目前质量证据是**离线 teacher-forcing**（G3）。
5. 16 个 prompt family 的正式运行（manifest 是**草案**，4 项待定稿）。
6. G5 的 ≥16 families × 12 conditions × ≥5 reps 正式矩阵。
7. `input_token_hash` 认证：8K 隔离 run 早于该字段加入 ⇒ `certified=false`（新 run 自动带上）。
8. **8K SpecDec 速度双档的原因：仍未闭环**。2026-09-23 的两次尝试（`094741Z` / `100031Z`）
   **SpecDec 角色 0 个样本**（子进程崩溃、旧版工具不落盘），而这两轮的 AR 对照反而给出了两条新事实：
   - **8K AR 在 8 个独立进程上极稳**（两种口径跨进程极差 ≤5.6%，prefill 3.34–3.45 s）；
   - **"8K 仅多驻留 draft 就让 AR 慢 21%"不重现**（8 进程下 dual 中位 29.81 vs target 29.66 ⇒ penalty≈0.995）
     ⇒ §6.27 已更正，该结论降级为"单会话观察"；
   - 新增未解异常：**SpecDec 的 8K prefill 在不同会话间差 2.4×**（13.2 s vs 29.0 s vs 31.96 s），
     而唯一成功样本的"target/draft 两次 prefill 各 6.55 s"无法用模型规模解释 ⇒
     工具已改为**按模型拆分 prefill**（`prefill_target_s` / `prefill_draft_s`），待重跑判定。
   在拿到阶段级证据前，**不得**把原因写成 WDDM paging / allocator / cache 增长中的任一项。
9. V-FP8 在缩小版里**未测**（`P5_KV=base_bf16`），RQ3 目前仍无新证据。

---

## 9. 开放问题与阻塞（按优先级）

| # | 阻塞项 | 最小解法 | 命令（**GPU，需人运行**） |
|---|---|---|---|
| **0** | **8K SpecDec 在同频率下出现两个稳定速度档（42.7%）**，原因未定位 | 跑专项探针得到阶段级证据 | `python -m specdec.engine.g5_8k_spec_variance_probe`（先加 `--dry-run` 看计划） |
| 1 | ~~GPU 掉进低功耗 P-state~~ | **已解决**：预检工具 + block 级复检；缩小版运行已实测满频 | `python -m specdec.engine.gpu_pstate_preflight`（当前 QUALIFIED） |
| 2 | ~~正式 AR 主基线无合格数据~~ | **2K 已有 `CLAIM ALLOWED`**；8K 待双档定位后重跑 | 见 §12 |
| 3 | V-FP8 在缩小版里未测（RQ3 无新证据） | 环境稳定后把 `P5_KV` 扩到 `base_bf16,v_fp8` | 见 §12 |
| 4 | "为什么慢"缺阶段归因 | 阶段成本分解（**必须在 8K 定位与环境 Gate 之后**） | `python -m specdec.engine.g5_cost_breakdown` |
| 5 | 8K 显存饱和的**协议级处置**未决 | 人工决策（是否把 8K 降级为受限条件） | —— |
| 6 | k 的收益曲线未知 | k 扫描（**同样需环境 Gate 先通过**） | `python -m specdec.engine.g5_k_sweep` |
| 7 | family manifest 待定稿 | 人工定稿 | —— |

---

## 10. 环境与运行约束

| 项 | 值 / 规则 |
|---|---|
| GPU | RTX 4060 Laptop **8 GB**（WDDM 允许 reserved 超订，`peak_resv` 可 >8 GiB） |
| CPU / RAM | i7-12700H / 32 GB |
| 软件 | Python 3.11 venv（`.venv`）、`torch 2.6.0+cu124`、`transformers 5.16.1`、Windows；HF cache `D:\hf-cache`；`reportlab 5.0.1` + `pypdf` 仅用于生成 PDF |
| GPU 运行 | 建议**由人**在交互式终端串行执行（本项目内的计时实验以人工执行为准）；agent shell **能**调起 CUDA（已实测），但 GPU 重跑归属人 |
| 计时前置条件 | **必须**通过 `gpu_pstate_preflight`（SM≥2000、mem≥7500 MHz，负载期采样）；探针会在每个 block 前自动复检，不合格则该 block 跳过 |
| 每次运行前 | `set HF_HUB_OFFLINE=1`（harness 已本地优先，但可避免网络抖动） |
| shell 陷阱 | 用户 shell 在 cmd 与 PowerShell 间切换：`cd /d` 只在 cmd 有效；`$env:X=` 只在 PowerShell 有效；**PowerShell 内联 Python 里的双引号会被吞掉**（用单引号）；**绝不要用 PowerShell 文本 cmdlet 改源码**（会毁掉 UTF-8，已发生两次） |
| 内存陷阱 | **不要并行跑两个 1.5B fp32 CPU 任务**（两个 ~12 GB 瞬时峰值 → 原生访问冲突 `0xC0000005`） |
| 输入设备 | 8K 双模型时显存已饱和（free 0），不要同时开浏览器等占显存程序 |

---

## 11. Git 状态

`main == origin/main`（`HEAD = 88a196f`）；最近提交（新→旧）：
```
88a196f  G5 v1.2 canonical accounting: summary mapping, cross-process equality, layered status, claim discipline
a93af68  Fix 3 pre-run hazards found while preparing the reduced re-run (no data was harmed)
5141369  docs: handoff -- G5 v1.1 status, P-state preflight result (ledger 4.12), corrected blocker list and next-step order
f713332  G5 v1.1: status layering + P-state preflight (and a real Sampler crash fix), claim gating
1f34daa  fix: stability probe crashed unpacking build_dual_controls (3-tuples); add probe-level cell tests and a --dry-run preflight
3235201  G5: freeze the measurement protocol, schema, stability probe, cost breakdown, k sweep
d2cad31  docs+probe: B6-fp32 re-run, histogram self-consistency, restored histogram section
067df03  correctness: refute the block-logits index hypothesis (3 independent evidence lines)
aa2078f  part4: correctness-comparison fields, cross-process token compare, no-NaN aggregation,
         between-order stability, monitoring parse fix, memory audit, self-test 62/62
df199f0  part4: per-condition process isolation, run identity that cannot overwrite,
         env monitoring, subset-aware cost calibration, CPU self-test 42/42
8ceef40  part4: per-condition process isolation + JSONL-only aggregation
3054f4f  docs/families: ≥16 nested-prefix prompt families（草案）
6827448  part4: remove empty_cache from the timed region; order-control switch; spread column
c57d107  part4: fix duplicate keyword in the summary JSONL row
d4eb389  G4 Part 4: measurement harness（AR 同档基线 + 标定）
b2ec86e  docs: 19-page PDF explaining speculative decoding + md source + generator
000cc9b  docs: fix a placeholder commit hash in the 2026-09-15b report
11a913c  G4 B5 post-fix GPU re-verification PASS（两档）
5ea72a5  docs: 2026-09-15b daily report + README refreshed to the G4 state
```
以 `git log --oneline` 为准。**受保护历史 JSONL 在 `output/` 下（gitignore），其 sha256 见 §5。**

---

## 12. 给分析 Agent 的下一步（推荐顺序）

**A. 先拿数据（人运行 GPU，串行）**
0. `python -m specdec.engine.gpu_pstate_preflight`（<1 min）→ 退出码 0 才继续；
   非 0 先插电/切高性能电源方案/关掉占显存程序，**不要在未通过时开跑计时实验**；
1. **`python -m specdec.engine.g5_8k_spec_variance_probe --dry-run`** 看计划（不跑 GPU 条件），
   然后 `python -m specdec.engine.g5_8k_spec_variance_probe`
   （**约 60–90 min**；12 个独立进程 = 3 角色 × 4；先缩小可设 `P5V_PROCS=3 P5V_REPS=3` ≈ 30 min）
   → 回答 8K 双档是否存在、集中在哪个阶段、与哪个状态量同步；
   **这是当前唯一优先级最高的实验**（8K 不闭环就不该恢复完整矩阵）；
2. 8K 结论到手后，再决定 8K 是否进正式 factorial；若结论是"实现开销主导"，
   必须先走协议 amendment + 重跑 B5/B6/B6-fp32，才能改实现；
3. 全量 `python -m specdec.engine.g5_stability_probe`（12 条件 × 2 block，25–40 min；
   先加 `--dry-run`）→ 量化 within-run 与 between-block；
4. `python -m specdec.engine.g5_cost_breakdown`（5–15 min）→ 阶段归因，解释 12 AR 步/轮的构成；
5. `python -m specdec.engine.g5_k_sweep`（20–40 min）→ k 曲线（若某 k 明显更好，**只提交修正建议**）。

**A'. 离线工具（不需要 GPU，只读）**
- `python scripts/g5_report_from_run.py latest` → 报表 + `[8]` 声明门控（`NO CLAIM：<原因>`）；
- `python scripts/g5_recanonicalize_run.py --run-id <stamp> [--dry-run]` → 用修正后的映射**重建** canonical
  （只读分片、不重新测量、不覆盖旧文件）。

**B. 拿到数据后分析 Agent 应产出**
- 稳定性判定：哪些条件可入正式主表、哪些必须标 `CHECK`、哪些是 `ENV_INVALID`；
  **规则是分层的**（§7c.3：within_run / between_block / environment 三层 + 最坏值），
  `exit=0` **不等于**测量有效；声明门控见 `docs/g5_measurement_protocol.md` §7b/§7c，**不得为了"能出数"而放宽**；
- 8K 双档的阶段归因（**只按探针实测证据归因**，不得先验地写 WDDM paging）；
- 成本方程：把 `SpecDec/AR < 1` 分解为 draft 提议 / 块验证 / 修复 / cache / control 的贡献比；
- 8K 饱和的可执行处置建议（含"是否把 8K 从正式 factorial 降级为受限条件"的权衡）；
- 若结论为"实现开销主导"⇒ 提出**最小改动**的实现优化提案（例如 KV 增长方式、chunk/并发），
  但**必须走协议 amendment + 重跑 B5/B6/B6-fp32** 才能生效。

**C. 禁止**
- **在 8K 波动未解释前启动完整 12-cell stability**；**在 canonical 记账未修正前启动正式 Pilot**（记账已修正 ⇒ 本条已解除）；
- **在环境 Gate（`gpu_pstate_preflight`）未通过时开跑 cost breakdown / k sweep / Pilot**；
- **在看到结果后调整 P-state 阈值**（`SM≥2000`、`mem≥7500`）、20% 稳定性规则或双档检出阈值（`P5V_BIMODAL_RATIO`）——任何改动必须**事先**登记；
- **删除慢 run / 只保留 8K 快档**；
- 把 dual-control 当正式 AR 分母；把 profiling 数字当正式吞吐；
- 在 §9.1–9.2 数据到手前启动 ≥16 families × 12 conditions；
- 修改 `spec_dec.py` / `spec_cache.py` / V-only FP8 设计 / 正式 BF16·模型·k=4（除新的、独立且可复现的失败证据）。

---

## 13. 关键定义（避免歧义）

1. **acceptance** = `accepted / proposed`；**替换 token 不计入分子**（它是 target 自己产出的）。
   恒等式：`gen_len = accepted + mismatch_rounds`（本设计无 bonus）。
2. **cache 不变量**：`len(tgt_cache) == len(drf_cache) == prompt + len(gen)`；每轮末
   `tgt_len == drf_len == pre + (m+1 if m<kp else kp)`；`_propose` 出口恒为 `pre + len(cands)`。
3. **显存三口径**：`peak_allocated_gib`（PyTorch 实际张量分配，**主指标**）/
   `peak_reserved_gib`（分配器池，含空闲块与虚拟预留，**≠物理占用**）/ `nvidia_smi_used_mib`（物理，辅助）。
4. **`BF16_NEAR_TIE` 四条证据（全部满足才允许）**：结构化不变量通过；`off_by_one=False`；
   同位置明显比相邻位置更近；`|fp32 margin| ≤` 现场 bf16 形状噪声。不满足 ⇒ `NEEDS_INSPECTION`。
5. **exploratory vs formal**：`g5_stability_probe` / `g5_cost_breakdown` / `g5_k_sweep` /
   `g5_8k_spec_variance_probe` / 两次 Part 4 exploratory run 均为 exploratory，**不入正式主表**；
   正式 = ≥16 families × 12 conditions × ≥5 reps。
6. **canonical（canonical JSONL）** = `output/g5/g5_<schema>_<hash>_<run_id>.jsonl`，
   每 `(条件, block)` 一行，是**唯一的正式结果表**；`status` 是兼容别名，
   判定一律看 `execution_status` / `measurement_status` / 三层状态字段。
7. **三层测量状态**：`within_run_status`（同进程 reps）/ `between_block_status`（两随机 block）/
   `environment_status`（P-state）；`measurement_status` = 三者最坏值。
   **原因字段分层存放**，不得混用（§7c.3）。
8. **跨进程等价性**：`ar_equal` 由**父进程**按 key `(block, ctx, KV, input_token_hash, max_new)`
   比较 `AR target_only primary` ↔ `SpecDec primary` 得出；`ar_equal_available` = 是否有对照；
   `ar_equal_certified` = 认证前提是否齐全。**缺对照写 null，绝不写 false**（§7c.2）。
9. **speedup 点估计口径**：**全部 measured rep 合并中位数**再相除；
   声明资格要求**每个配对 block 都合格**（§7c.6）。
10. **`residency_penalty`** = `T_AR_target_only / T_AR_dual`；`<1` ⇒ 仅多驻留一个模型就拖慢了 AR。
    8K 实测 ≈ **0.79**（dual 比 target-only 慢 ≈21%）。
11. **重新记账 vs 重新测量**：`scripts/g5_recanonicalize_run.py` 只重算映射/分类，
    **不使用任何新测量**；旧 canonical 一律保留。
