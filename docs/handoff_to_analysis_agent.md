# SpecDec — 分析 Agent 交接报告（2026-09-21）

> 读者：接手**分析与实验设计**的 agent。本文档自包含：项目目标、冻结口径、**已确立事实 / 推断 /
> 未验证**三分类、数据与代码清单、开放阻塞项、环境约束、下一步推荐动作。
> **硬规则：本文档中标注"未运行"的东西一律不得填预期值；任何数字必须能追溯到 `docs/data_ledger.md`
> 的条目或 `output/` 下的原始文件。**

---

## 0. 一句话现状

自研 Qwen2 推理引擎（含 Asymmetric V-only FP8 KV）与贪心投机解码实现**正确性已闭环**
（FP32 逐 token 等价 20/20；索引语义经哨兵 + 真实模型 + 端到端三方验证）；
**但 8K 上的性能结论仍是"负 + 不稳定"**：SpecDec 比 AR 慢 3–4 倍、V-FP8 无可复现速度优势、
计时受显存饱和与启动顺序支配。G5 测量协议与 schema 已冻结并通过 43 项 CPU 自测，
**稳定性探针 / 成本分解 / k 扫描三个 GPU 实验尚未运行** ⇒ **正式 Pilot 仍被禁止**。

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
| Part 4 性能数据 | **INVALID（不得入正式主表）** | 原因：8K 显存饱和 + between-block 28.9% 不稳定（**与索引无关**） |
| G5 schema / 协议 | **已冻结** | `docs/g5_measurement_protocol.md`；`results_schema.py`（43 项 CPU 自测） |
| G5 稳定性 / 成本分解 / k 扫描 | **未运行** | ledger §4.8.4 |
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
| `specdec/results_schema.py` | **G5 单一 schema**（`g5-results-1.0`，46 字段）+ 校验 + 拒绝覆盖 + 旧 Part 4 只读兼容 + MAD/IQR + `classify_bf16_compare` | 43 项 CPU 自测 PASS |
| `g5_driver.py` | 每条件独立子进程编排、系统快照、固定 seed 随机化、run 元数据 | CPU 验证 |
| `g5_stability_probe.py` | Task B：12 条件 × 2 随机 block（warm-up 1 + 5 rep、cooldown 不计时） | **未运行** |
| `g5_cost_breakdown.py` | Task C：SpecDec/AR 阶段分解（外部 CUDA Event，不改冻结文件） | **未运行** |
| `g5_k_sweep.py` | Task D：k∈{1,2,4,8} 诊断扫描（只出建议） | **未运行** |
| `g5_schema_selftest_cpu.py` / `g4_part4_selftest_cpu.py` | CPU 自测 **43/43** / **62/62** | 已运行 |

---

## 5. 数据清单（`output/` 被 gitignore，但**必须原样保留**）

| 文件 | 内容 | 状态 / 锚点 |
|---|---|---|
| `output/part4/part4_dc0bd5d87389.jsonl` | **exploratory_run_1_fixed_order**：12 条件、36 run（固定顺序，含 `empty_cache` 于计时区） | **受保护**；41026 B / 49 行；sha256 `bfb85f509b6802393637ea9cea9d7f5275579deb457ed6456db9bb087664c53c` |
| `output/part4/part4_ba0c985bcd5d.jsonl` | **exploratory_run_2_reverse_order**：4 条件（仅 8192，`P4_REVERSE=1`，REPS=3） | **受保护**；14310 B / 17 行；sha256 `6ad22617a46be08328f980537eb77ed95fb7b38e5548166da3b910f335d4c7c3` |
| `output/part4/part4_*_8k[AB]_20260921T042125Z.jsonl`（8 文件） | 8K 隔离复验：4 条件 × 2 顺序 × 5 rep = 40 run + 8 summary | 已用（§6） |
| `output/part4/part4_*_8kA/B_*.jsonl`（命名规则） | `part4_<config_hash>_<RUN_ID>.jsonl`，RUN_ID 含 UTC | 新 run 不得与旧同名 |
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

---

## 7. 推断（方向可信，机制未隔离）

- 8K 的计时波动**与显存状态一致**（饱和 + 无降频），但**未直接隔离 paging**。
- SpecDec 慢的**主要来源在"每轮固定开销"**（算术支持），但**阶段级归因未测**。
- 每轮 ~280 ms 的"未解释缺口"存在（把 draft 前向按 1/3 权重粗估 + 块验证按 1 步估后仍差约 4 倍）
  ⇒ 候选解释：8K 下 draft 并不便宜、块验证 attention 成本、饱和下的分配/换页、`torch.cat` 逐步增长 KV 的复制。
  **在拿到 §8 的阶段数据前不得择一断言。**

---

## 8. 未验证 / 未运行（**明确清单，不得填预期值**）

1. `g5_stability_probe`：12 条件 × 2 随机 block 的 GPU 数据（2048/4096 完全未测；8K 仅有两顺序各 5 rep）。
2. `g5_cost_breakdown`：阶段单价（draft 提议 / 块验证 / 修复步 / cache 裁剪 / KV cat / control）与闭合残差。
3. `g5_k_sweep`：k∈{1,2,4,8} 的 acceptance/tok-per-round/rounds/decode/peak。
4. 生成口径的质量对照（base vs v_fp8 的输出一致率 / NLL）——目前质量证据是**离线 teacher-forcing**（G3）。
5. 16 个 prompt family 的正式运行（manifest 是**草案**，4 项待定稿）。
6. G5 的 ≥16 families × 12 conditions × ≥5 reps 正式矩阵。
7. `input_token_hash` 认证：8K 隔离 run 早于该字段加入 ⇒ `certified=false`（新 run 自动带上）。

---

## 9. 开放问题与阻塞（按优先级）

| # | 阻塞项 | 最小解法 | 命令（**GPU，需人运行**） |
|---|---|---|---|
| 1 | 跨顺序不稳定未量化到 12 条件 | 随机化 2 block 探针 | `python -m specdec.engine.g5_stability_probe` |
| 2 | "为什么慢"缺阶段归因 | 阶段成本分解 | `python -m specdec.engine.g5_cost_breakdown` |
| 3 | 8K 显存饱和的**协议级处置**未决（标注 / 降并发 / 缩小 max_new） | 人工决策 | —— |
| 4 | k 的收益曲线未知 | k 扫描（诊断） | `python -m specdec.engine.g5_k_sweep` |
| 5 | family manifest 待定稿（N、分层规则、是否用 train.raw、是否加密中间长度） | 人工定稿 | —— |
| 6 | 验证器 / 报告 | 等 §9.1–9.2 数据 | —— |

---

## 10. 环境与运行约束

| 项 | 值 / 规则 |
|---|---|
| GPU | RTX 4060 Laptop **8 GB**（WDDM 允许 reserved 超订，`peak_resv` 可 >8 GiB） |
| CPU / RAM | i7-12700H / 32 GB |
| 软件 | Python 3.11 venv（`.venv`）、`torch 2.6.0+cu124`、`transformers 5.16.1`、Windows；HF cache `D:\hf-cache`；`reportlab 5.0.1` + `pypdf` 仅用于生成 PDF |
| GPU 运行 | **必须由人**在交互式终端串行执行；agent shell 跑 GPU 脚本会挂起 |
| 每次运行前 | `set HF_HUB_OFFLINE=1`（harness 已本地优先，但可避免网络抖动） |
| shell 陷阱 | 用户 shell 在 cmd 与 PowerShell 间切换：`cd /d` 只在 cmd 有效；`$env:X=` 只在 PowerShell 有效；**PowerShell 内联 Python 里的双引号会被吞掉**（用单引号）；**绝不要用 PowerShell 文本 cmdlet 改源码**（会毁掉 UTF-8，已发生两次） |
| 内存陷阱 | **不要并行跑两个 1.5B fp32 CPU 任务**（两个 ~12 GB 瞬时峰值 → 原生访问冲突 `0xC0000005`） |
| 输入设备 | 8K 双模型时显存已饱和（free 0），不要同时开浏览器等占显存程序 |

---

## 11. Git 状态

`main == origin/main`；最近提交（新→旧）：
```
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
1. `python -m specdec.engine.g5_stability_probe`（25–40 min）→ 量化 12 条件的 within-run 与 between-block
2. `python -m specdec.engine.g5_cost_breakdown`（5–15 min）→ 阶段归因，解释 12 AR 步/轮的构成
3. `python -m specdec.engine.g5_k_sweep`（20–40 min）→ k 曲线（若某 k 明显更好，**只提交修正建议**）

**B. 拿到数据后分析 Agent 应产出**
- 稳定性判定：哪些条件可入正式主表（`between-block ≤20%`）、哪些必须标 `CHECK`；
- 成本方程：把 `SpecDec/AR < 1` 分解为 draft 提议 / 块验证 / 修复 / cache / control 的贡献比；
- 8K 饱和的可执行处置建议（含"是否把 8K 从正式 factorial 降级为受限条件"的权衡）；
- 若结论为"实现开销主导"⇒ 提出**最小改动**的实现优化提案（例如 KV 增长方式、chunk/并发），
  但**必须走协议 amendment + 重跑 B5/B6/B6-fp32** 才能生效。

**C. 禁止**
- 在 §9.1–9.2 数据到手前启动 ≥16 families × 12 conditions；
- 为获得加速而改 prompt / 计时边界 / 删除慢 run / 换 dtype / 改 AR 驻留条件；
- 修改 `spec_dec.py` / `spec_cache.py`（除新的、独立且可复现的失败证据）；
- 把 exploratory 数据混入正式主表；把 profiling 数字当正式吞吐。

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
5. **exploratory vs formal**：`g5_stability_probe` / `g5_cost_breakdown` / `g5_k_sweep` / 两次 Part 4 exploratory
   run 均为 exploratory，**不入正式主表**；正式 = ≥16 families × 12 conditions × ≥5 reps。
