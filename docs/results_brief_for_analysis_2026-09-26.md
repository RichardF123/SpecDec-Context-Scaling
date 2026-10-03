> **SUPERSEDED HISTORICAL SNAPSHOT — NOT THE CURRENT PROJECT SCOPE.**
> This file preserves the project state and evidence as reported on 2026-09-26.
> Current formal scope is governed by
> [`proposal_freeze_2026-10-01.md`](proposal_freeze_2026-10-01.md); for current
> status, also consult [`README.md`](../README.md) and
> [`status_update_2026-10-02.md`](status_update_2026-10-02.md).

# SpecDec — 结果汇报（给分析 Agent）｜2026-09-26

> **读者**：接手**分析与实验设计**的 agent。
> **本文档是入口**：一页读完现状、可声明的事实、已被排除的假设、红线、待办与决策点。
> **细节一律以 `docs/data_ledger.md` 为准**；本文每个数字都带台账锚点（如 `§4.20`）。
> **本文档自包含**，但**不替代** `docs/handoff_to_analysis_agent.md`（更完整的历史与协议上下文）。
>
> **硬规则（沿用）**
> 1. 本文档中标"未验证"的东西**一律不得填预期值**；任何数字必须能追溯到台账条目或 `output/` 原始文件。
> 2. **不得**为了得到好看的结论而调整阈值、删除慢数据、或把 profiling 数据当吞吐。
> 3. 三分类纪律：**已确立（有实测证据）/ 推断（方向可信、机制未隔离）/ 未验证（未跑或未复现）**。

---

## 0. 一句话现状

引擎与投机解码的**正确性已完全闭环**（G1 / B5 / **B6-fp32 20/20** / 索引语义 / FP8 质量）；
**性能侧只剩一个问题：8K + 双模型 SpecDec 在 8 GB 卡上"能跑，但计时不可复现"**。
已用证据**排除**了四个外部假设（GPU 降频、热降频、CPU commit、分配器碎片），
并把原因收敛到**显存压力**：`peak_allocated` 稳定 7121 MiB，而 `peak_reserved` 达 **10.48–10.72 GiB > 8 GiB 物理**、`cuda_free = 0`。
**下一步的关键决策**：是否批准一个**只改 attention 一段**的 amendment（实测可省 **1.9–2.2 GiB** 瞬时显存，见 §4.17.x）。

---

## 1. 可声明的事实（按类别）

### 1.1 正确性（全部有实测，可写进报告）

| 结论 | 证据 | 锚点 |
|---|---|---|
| 引擎前向与 transformers 对齐 | fp32 CPU：full `3.099e-05` / cached `2.038e-05`，argmax 全同 | G1，§3.4 |
| **投机解码与 AR 输出逐 token 完全相同** | **20/20 组合、216/348 token 全等**，两次独立复跑一致 | **B6-fp32**，§4.5.6 / §4.7.6 |
| bf16 下 19/20（唯一分歧已归因） | 该点 bf16 两候选**精确并列**、fp32 margin `0.1491 ≤ 噪声 0.1491` | B6-bf16，§4.5.5 |
| 判定行索引语义正确（不存在错位一格） | 哨兵 **35/35** + 真实模型 fp32：同位置 `3.9e-05–5.7e-05` vs 相邻位置 `14.03–17.25` | §4.6.5、§4.7 |
| 三个历史分歧条件 = `BF16_NEAR_TIE` | fp32 margin `0.0454/0.0179/0.0345` ≤ 现场噪声，且 `off_by_one=False` | §4.6.5 |
| V-only FP8 KV 质量几乎无损 | `ΔNLL +0.000397`、`ΔPPL +0.040%`；K-FP8 / full-FP8 为负结果 | G3，§3.5.3 |

### 1.2 内存与规模（新，且**零方差**，适合作为硬结论）

| 结论 | 数值 | 锚点 |
|---|---|---|
| 8K 双模型 SpecDec **超出 8 GB 物理显存** | `peak_allocated` **7120.6 MiB（三次测量逐字节相同）**；`peak_reserved` **10736 → 15188 MiB**；`cuda_free = 0`；`smi` 7824–7915 MiB | §4.18、§4.20 |
| 8K 逻辑 KV **不是**瓶颈 | base_bf16 **224 MiB**、v_fp8 169.75 MiB ⇒ 仅占 `peak_allocated` 的 **~3%** | §3.5.3、§4.14 |
| 权重常驻占大头 | target+draft 合计 **3.82 GiB**（占 `peak_allocated` 55%） | §4.18 |
| 手写 attention 的单次瞬时是最大单项 | target **1998 MiB** / draft **2288 MiB**（一次要 4 个整尺寸张量），其中 fp32 softmax 仅 806/896 MiB | §4.17.x |

### 1.3 性能（**注意口径**）

| 条件 | 数值 | 可否声明 |
|---|---|---|
| **2K base_bf16**（缩小版复验） | AR target-only decode **33.04 / 29.69** tok/s（between-block **10.1%**）；SpecDec **13.26 / 12.79**（**3.5%**）；**decode speedup 0.406×、e2e 0.434×** | ✅ **CLAIM ALLOWED** —— 但**必须注明**是"**缩小版 stability probe（2 长度 × 1 KV），不是 16-family 正式矩阵**" |
| **8K base_bf16** | AR target-only **29.33 / 29.51**（**0.6%**，极稳）；SpecDec **9.53 / 5.46**（**42.7% ⇒ CHECK**）；合并 decode 0.236× / e2e 0.136× | ❌ **不得声明倍数**。只能写：*AR 稳定；SpecDec 跨 block 差 42.7%；方向未观察到加速* |
| **8K SpecDec 同配置三次** | decode **10.04 / 11.18 / 16.22 s**（极差 **61%**）；而 `peak_resv`/`inact_split` **逐字节相同** | ❌ **不得声明任何速度**（61% ≫ 预登记 20% 门限） |

> **口径必须写清（否则同一实验会出现两个数）**：协议 §7c.6 规定 speedup 点估计 = **全部 measured rep 合并取中位数**再相除。
> 按此：**2K = decode 0.406× / e2e 0.434×**；若改用"逐 block 中位数再平均"会得到 decode **0.416×** / e2e **0.444×**。
> 两者差 ~2.3%，**不是错误**，但**引用时必须注明口径**；报告里统一用协议口径。

### 1.4 阶段构成（8K SpecDec，单进程）

| 阶段 | 数值 | 占比 |
|---|---|---|
| prefill 合计 | **28.43 / 32.39 / 36.11 s** | **~74%** |
| ├ target 侧 | 7.34 / 7.41 / 7.46 s | 稳定 |
| └ **draft 侧** | **21.09 / 24.98 / 28.65 s** | **波动最大** |
| decode | 10.04 / 11.18 / 16.22 s | 波动大 |

**决策层完全可复现**（重要正面证据）：`rounds=20`、`proposed=76`、`accepted=56`、`mismatch=8`、
`all_accept=12`、`acceptance=0.7368` —— 与缩小版 run 的 8K SpecDec **逐项相同**（§4.14、§4.18）。

### 1.5 residency 效应（**只对 prefill 成立**）

| 测量 | 8K target 侧 prefill |
|---|---|
| draft **不加载**（AR target-only） | **3.34–3.45 s**（8 个独立进程 + 2 个 block，共 16 样本） |
| draft **仅常驻不参与计算**（AR dual-control） | **7.40 / 7.74 s** |
| draft **参与计算**（SpecDec 进程内） | **7.34 / 7.41 / 7.46 s** |

⇒ **"draft 只要常驻"就让 target 的 8K prefill 慢 2.16–2.30×**，且**与它是否参与计算无关**（两次独立会话复现，§4.18、§4.19）。
**但 decode 不受影响**：8 进程测得 `residency_penalty ≈ 0.995`（§4.16）。
⚠️ **更正历史**：早期登记的"8K residency 让 AR 慢 ≈21%"是**单会话观察，未复现**（§4.14 已标注）。

---

## 2. 已被排除的假设（"分析 & 洞察"分的重要素材）

| 假设 | 排除证据 | 锚点 |
|---|---|---|
| **GPU 降频（P-state）** | 第一次运行 SM 时钟 **435–2535 MHz** 乱跳（低频 block 吞吐仅 6.16 tok/s vs 高频 17.67）；引入预检 + block 级复检后消除 | §4.11、§4.12 |
| **热降频 / 功耗墙** | 慢 run 仅 **15.8–28 W**、**61–68 °C**；快 run 40–86 W、69–81 °C | §4.11 |
| **prefill 全局不稳** | AR target-only 的 8K prefill 在 **16 个独立样本**里稳定在 **3.34–3.45 s**（±1.6%） | §4.16 |
| **token 决策随机性** | 决策层指标（rounds/accepted/proposed/mismatch/all_accept）逐项相同 | §4.18 |
| **CPU commit 不足**（导致原生 `0xC0000005` / 整机硬挂） | 载入需 **~9.1–9.6 GiB** commit；失败现场仅剩 **9.5 GiB**；把 pagefile 固定为 8 GB 后（上限 33.6→**39.6 GB**，可用 9.3→**20.5+ GiB**）**8K 可正常跑通** | §4.16、§4.18 |
| **分配器碎片**（`inactive_split` 1371 MiB） | ① `expandable_segments` **在 Windows 上不被支持**（PyTorch 明确警告，**从未生效**）；② `max_split_size_mb:256` 把碎片砍半（1371→678）却把池**撑到 15188 MiB**，总时间反而更差 | §4.20 |
| **时钟导致时间差** | 慢的那次 SM 时钟**反而更高**（2385 → 2430 MHz） | §4.19 |
| **外部显存占用漂移** | 五次运行 `smi` 全程 **7824–7915 MiB**（±1.2%） | §4.20 |

**尚未排除 / 未隔离的机制**（**不得写成结论**）：
- **WDDM 分页**是否为"慢档"的量级来源（`peak_reserved` 超物理 2.5 GiB、`cuda_free=0` 是同向证据，但未直接测分页量）；
- 为什么 **draft 侧**的波动（+18.4%）远大于 target 侧（+1.0%）——当前候选解释是"它后跑、此时显存已满"，**未隔离**；
- 8K SpecDec 的**状态累积型退化**（同一进程内 warm0 → rep0 慢 59%）**未隔离**。

---

## 3. 红线（**不得**做的事）

1. **不得声明 8K 的任何速度倍数**（同配置极差 61% ≫ 20%）。
2. **不得**把 `dual-control` 当正式 AR 分母；**不得**把 profiling 数字当吞吐。
3. **不得**动这些阈值：P-state `SM≥2000 / mem≥7500 MHz`、稳定性 `20%`、双档检出 `P5V_BIMODAL_RATIO=1.5`、commit 护栏 `16 GiB`。
   任何改动必须**事先**登记为协议 amendment。
4. **不得**删除慢数据（`output/part4/`、`output/g5/` 下的产物全部受保护，含 sha256 清单见 §5）。
5. **不得**把 exploratory 数据混入正式主表。
6. **不得**在未解释 8K 波动前启动完整 12-cell stability / 正式 Pilot。

---

## 4. 待办与决策点

### 4.1 需要人拍板的决策

| # | 决策 | 依据 | 预期收益 |
|---|---|---|---|
| **D1** | **是否批准 CUDNN attention amendment**（只改 `SelfAttention.forward` 的 score→output 一段） | 算子级微基准：瞬时 **1998/2288 MiB → 74/59 MiB**，耗时 **34.0/38.5 ms → 4.56/4.12 ms**，数值 max\|Δ\|=1.465e-03 | `peak_allocated` 7121 → **~5.2 GiB** ⇒ `peak_reserved` **有望降到 8188 MiB 以内** ⇒ 8K 可能从"不可测"变成"可测" |
| **D2** | 若不做 D1：**8K 是否降级为"受限条件"**（只登记内存证据 + "计时不可复现"），主战场转 **2K/4K** | §4.20 | 零成本；RQ1 仍有干净曲线（2K→4K）+ 8K 边界结论 |
| **D3** | 8K 的 **v_fp8** 是否纳入（RQ3 目前无新证据） | §4.14（缩小版只测了 base_bf16） | —— |

**D1 的低风险验证路径**（建议）：改 attention（混合实现：`s≥2` 走 CUDNN，`s==1` 保留手写）→
**只跑一次 8K SpecDec 冒烟** → **只看 `peak_reserved` 有没有 ≤ 8188 MiB** → 降下去才走正确性阶梯；否则回滚（成本 ~1 小时）。

> **2026-09-26 更新（D1 的前两步已完成，结论有变，引用前必读）**：
> 混合实现（策略 `eager`/`cudnn_prefill`，默认 `eager`；**只有显式标注的 `prefill` 阶段**允许融合）
> 已实现并跑完"算子级 + 短上下文模型级"，详见 **`docs/cudnn_prefill_stage_report.md`** 与台账 **§4.22**。
> 要点：**算子级显存结论成立**（8K 分块第一块 target **1984.0 → 54.0 MiB** / draft **2284.0 → 51.5 MiB**，
> kernel 名证实 `_scaled_dot_product_cudnn_attention`），**但正确性判据未通过**：
> 模型级 A/B **10/10 行 FAIL**，`ctx=512/base_bf16` 的**生成序列真的变了**（首位分歧 index 3）。
> 差异**已定位**为 **eager 基线在 softmax 前把分数舍入成 bf16**（尖峰注意力下对阵 fp32 相对误差可达 ~50%），
> 融合侧在尖峰区反而准 15–40×；掩码/GQA/缩放三个假说均被排除。
> ⇒ **D1 仍不能按"行为中性的替换"推进**：换后端 = 换模型行为，8K 的 eager↔融合对比**自带混淆项**，
> 必须先决定"以哪一侧为基线"，再谈收益。**8K 冒烟尚未执行**（本阶段刻意不做）。

### 4.2 未闭环清单（按优先级）

| # | 未闭环项 | 状态 |
|---|---|---|
| 1 | **8K SpecDec 计时不可复现** | 已定位到"显存压力"层级，机制未隔离 |
| 2 | **8K v_fp8 / 4K v_fp8** | **完全未测** ⇒ RQ3 目前**没有**新证据 |
| 3 | **4K base_bf16** | 未测（但峰值显著低于 8K，**很可能干净可测**） |
| 4 | `g5_cost_breakdown`（阶段单价与闭合残差） | 工具就绪，**未运行** |
| 5 | `g5_k_sweep`（k∈{1,2,4,8}） | 工具就绪，**未运行** |
| 6 | 16 个 prompt family manifest | **仍是草案**（N、分层规则、是否用 `train.raw`、中间长度 4 项待定稿） |
| 7 | ≥16 families × 12 conditions × ≥5 reps 正式矩阵 | 未启动 |
| 8 | 生成口径的质量对照（base vs v_fp8 输出一致率 / NLL） | 现有质量证据是**离线 teacher-forcing**（G3） |
| 9 | tie-frequency census、F1–F7 失败分类、4–5 张图、5–7 页报告、一键复现 | 未开始 |
| 10 | `README.md` 停在 2026-09-15 快照；`model_self.py` 有 `__main__` 残渣 | 待清理 |

---

## 5. 数据与代码索引

### 5.1 受保护数据（**不得改动，sha256 已登记**）

| 文件 | 内容 | sha256（前缀） |
|---|---|---|
| `output/part4/part4_dc0bd5d87389.jsonl` | exploratory run 1（12 条件固定顺序） | `bfb85f509b680239…` |
| `output/part4/part4_ba0c985bcd5d.jsonl` | exploratory run 2（4 条件 reverse） | `6ad22617a46be083…` |
| `output/g5/g5_g5-results-1.1_5dc7b5eeacd1_g5stab_20260921T090033Z.jsonl` | stability run #1（ENV_INVALID） | `cae4060a6939bf11…` |
| `output/g5/g5_g5-results-1.1_0e4ed83b6fb3_g5stab_20260922T040625Z.jsonl` | 缩小版 run 的 **1.1 canonical（含 4 类错记，作审计证据保留）** | `10045ba8331afe…` |
| `output/g5/g5_g5-results-1.2_re44bd9ca4bb_g5stab_20260922T040625Z.jsonl` | **同一批测量的 1.2 canonical（修正后重建，未重新测量）** | —— |
| `output/g5/variance/p5v_*.json` | 8K 变体探针系列（含 chunk sweep / alloc A-B-C / 崩溃栈） | —— |
| `output/g5/attn/attn_backend_20260926T091642Z.json` | attention 后端微基准 | —— |

### 5.2 关键工具

| 文件 | 作用 |
|---|---|
| `docs/data_ledger.md` | **唯一数据台账**（§3.x 正确性、§4.5.x 门禁、§4.11–4.20 本轮全部排查） |
| `docs/g5_measurement_protocol.md` | 协议冻结 v1 + **§7c canonical 记账规则** + §9b/§10b 实现约束 |
| `specdec/results_schema.py` | G5 schema **v1.2**（状态分层 / 跨进程等价性 / 12 条写盘前自洽校验） |
| `scripts/g5_report_from_run.py` | 只读离线报表 + `[8]` 声明门控（`NO CLAIM：<原因>`） |
| `scripts/g5_recanonicalize_run.py` | 按新口径**重建** canonical（只读分片，不重新测量） |
| `scripts/run_g5_chunk_sweep.cmd` / `run_g5_variance.cmd` / `run_g5_alloc_ab.cmd` | 一键运行（**内部固定用 `.venv\Scripts\python.exe`**） |
| `specdec/engine/g5_attn_backend_probe.py` | attention 后端算子级微基准（含 **CPU 语义自检**） |

### 5.3 运行须知（**已踩过的坑**）

| 坑 | 规则 |
|---|---|
| venv | shell 里**未必激活** venv ⇒ 一律用 `.venv\Scripts\python.exe` 或 `.cmd` 脚本（裸 `python` 会 `ModuleNotFoundError: torch`） |
| commit 护栏 | 可用 commit **< 16 GiB 时工具拒绝启动**（历史上曾因 commit 耗尽导致**整机黑屏硬挂**） |
| P-state | 跑前先 `python -m specdec.engine.gpu_pstate_preflight`；**电池供电时必然不合格**（应插电） |
| 环境判断 | 探针的"空闲快照"SM 时钟就是外部占用的探针（真空闲 ~210 MHz，被占用则上千 MHz） |
| 阈值 | **不得**事后调整；`expandable_segments` 在 Windows **不可用**（别再试） |
| chunk | 8K 用 **chunk=2048 可以跑通**，无需改动；`chunk` 只影响 4K/8K 的 prefill 口径，**2K 结果不受影响** |

---

## 6. Gate 状态总表

| Gate | 状态 |
|---|---|
| G1 target 正确性 | **PASS** |
| G2 8K operational | **PASS** |
| G3 V-only FP8 KV | **PASS / FROZEN** |
| B3 / B4 / B5 | **PASS** |
| **B6-fp32（主正确性 Gate）** | **PASS（20/20，216/348）** |
| B6-bf16 | **CHECK 19/20**（已归因 = tie-collapse） |
| 索引语义 / cache 不变量 | **PASS** |
| canonical 记账（v1.2） | **PASS**（写盘前强制校验） |
| **2K base_bf16 速度声明** | **CLAIM ALLOWED（缩小版口径）** |
| **8K 速度声明** | **NO CLAIM**（同配置极差 61%） |
| 8K 内存超限结论 | **可声明**（`peak_reserved` > 物理、`cuda_free=0`，零方差） |
| 4K / 8K 的 v_fp8 | **未测** |
| cost breakdown / k sweep | **未运行** |
| 正式 Pilot / Final | **HOLD** |

---

## 7. 怎么读这份文档 / 下一步怎么走

1. **想知道"能声明什么"** → 读 §1.3 + §3（红线）。
2. **想写"分析 & 洞察"** → 用 §2 的排除表（每个假设都配了实测证据）+ §1.5 的 residency 发现。
3. **想继续推进实验** → 先看 §4.1 的 **D1/D2 决策**；D1 的低风险路径是"改 attention → 一次冒烟 → 只看 `peak_reserved`"。
4. **想核对任何数字** → 一律回 `docs/data_ledger.md` 对应小节。

> **最诚实的一句话总结**：我们没能给出 8K 的加速比 —— **因为在 8 GB 笔记本 GPU 上，
> 长上下文 + 双模型投机解码是被显存限制的，计时在物理上就不可复现**。
> 我们把这个"给不出数字"的原因量化到了 MiB 级（`peak_allocated` 7121 vs `peak_reserved` 10736 vs 物理 8188），
> 并逐条排除了四个外部解释。**这本身就是 RQ1 的答案**，也是本次实验设计最有价值的部分。
