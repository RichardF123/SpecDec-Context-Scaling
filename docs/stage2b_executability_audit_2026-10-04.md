# Stage 2b 可执行性审计：新 family / 逐阶段停止条件 / 命令核对 / FP8 缺口（2026-10-04）

> **本轮只做只读审计 + CPU 自测。**未运行 GPU、未批准 v2、未改冻结判据、未声明 `READY_FOR_FORMAL_MEASUREMENT`。
> 复现脚本（本轮新增，位于被忽略的 `output/` 下）：
> `output/g5/attn/_newfam_audit.py`（family 核对）、`output/g5/attn/_cmd_audit.py`（命令核对）。
> 两份脚本都**不加载模型、不使用 GPU**（`_newfam_audit` 需要 tokenizer，CPU）。
> 配套决策清单：`docs/protocol_decision_brief_2026-10-04.md`。

---

## 1. 新增 family（F09 / F03 / F11 / F04）核对结果 —— **可用，但必须先登记**

### 1.1 来源与 hash（实测，`verify_hash=True`，失败即抛）

| family | stratum | source | window_token_start | `text_sha1` | `prefixes[8192].token_sha1` | 复现结果 |
| --- | --- | --- | --- | --- | --- | --- |
| **F09** | `S1_low_entropy` | `wikitext-2-raw-v1/wiki.train.raw` | 49152 | `5b1c4923589c840f` | `e5dbe83616ea348a` | **hash_verified=True** ✅ |
| **F03** | `S2_high_markup` | 同上 | 16384 | `a620dd21a483471d` | `92755adeb5715d47` | **True** ✅ |
| **F11** | `S3_prose` | 同上 | 8192 | `c78f391798929e79` | `2c14a59130440e9a` | **True** ✅ |
| **F04** | `S4_mixed` | 同上 | 24576 | `499074063a95dda9` | `e9004b83814fe321` | **True** ✅ |

* 四个 family **各占一个 stratum**（与 D-2 的"每 stratum 取未用过的最小 id"规则一致）；
  加上已用的 F08(S1)/F02+F13(S2)/F10(S3)/F01(S4) ⇒ **9 个 family 仍覆盖全部 4 个 stratum**。
* 重跑命令：`.\.venv\Scripts\python.exe -X utf8 output\g5\attn\_newfam_audit.py`

### 1.2 独立性（**按仓库自己的判据 n=32**，不是我自己另定的长度）

| 检查 | 结果 |
| --- | --- |
| 窗口区间两两重叠 token 数 | 四者**全部为 0**（窗口分别是 16384–24576 / 24576–32768 / 49152–57344 / 8192–16384） |
| 新 family **两两之间** 32-gram 交集 | 全部 **0** |
| 新 family 与**已用 5 family**（F13/F01/F02/F08/F10）32-gram 交集 | 全部 **0** |
| 新 family 与**调试输入**（wiki.test 前缀）32-gram 交集 | 全部 **0**（即仓库守卫的独立性判据 `SHINGLE=32` 满足）✅ |
| 现有产物消费情况 | `output/g5/**/*.json` 中含这四个 family 字面量的文件数 = **0** ⇒ 从未被任何 run 消费 ✅ |

> ⚠️ **如实补充**：在更短的 n-gram 上存在少量**样板文本**碰撞（8-gram：F09∩F04=14、F09∩F03=4、F03∩F04=2、
> F04∩F13=13、F03∩F02=8…）。这些是 WikiText 的 markup/样板片段，**不构成窗口或语义重叠**；
> 仓库的独立性口径是 **32-gram**，在此口径下**全为 0**。报告时应按 32-gram 口径陈述，不要声称"任意长度零重叠"。
> 另外：守卫目前只对**调试输入 vs 16 family** 以及 **F06-vs-F13** 做 32-gram 检查，
> **未覆盖**"新 family 两两之间"⇒ 本审计第 1.2 节补上了这一缺口（建议把该检查固化进守卫，见 §5）。

### 1.3 守卫登记需求（**已实测：四个都会判越界**）

| 常量（实测值） | 值 |
| --- | --- |
| 源码局部常量 `REGISTERED_DIAG_FAMILIES` | `("F06", "F13")` |
| `PROTOCOL_ROUND_FAMILIES` | `("F01","F02","F08","F10")` |
| `SERIAL_ORACLE_ROUND.families` | `("F13","F01","F02","F08","F10")` |
| `SHINGLE` | `32` |

| family | 定点诊断 | 协议轮 | 本轮授权集合 | 结论 |
| --- | --- | --- | --- | --- |
| F09 / F03 / F11 / F04 | False | False | False | **会被判越界**；记录级检查会报警 ⇒ **不登记就跑不了**（守卫会正确地 FAIL） |

⇒ **D-3 是硬前置**：必须在守卫里**显式登记**（建议新增一个具名集合，如
`STAGE2B_FAMILIES = ("F09","F03","F11","F04")`，并把 `SERIAL_ORACLE_ROUND.families` 扩为 9 个），
**只增检查、不放宽任何既有检查**。

---

## 2. 逐阶段停止条件（AR×2 → production×2 → serial×2）

**Stage −1（开跑前，缺一不可）**：守卫 PASS（含 D-3 登记）· 环境检查 `ENVIRONMENT_OK`
（harness 16 GiB 线 **通过** 且 ≥ 18 GiB 入口门槛）· 工作树 clean · 无本项目残留实验进程。
任一不满足 ⇒ **不加载模型、停止**（记录读数）。

| 阶段 | 通过条件 | **停止条件（硬）** |
| --- | --- | --- |
| **AR ×2** | 两次 rc=0；`checks.passed=True`；长度=64；stop ∈ {`max_new_reached`,`eos`}；finite；cache 结构检查通过；**两次完整 token IDs 一致**；`gen_hash` 与 harness 自报 `end_state.gen_hash` 一致 | 两次不一致（⇒ 记 `BASELINE_REGRESSION`，保留产物、停止）· 非有限值 · `other_short` · 结构失败 · OOM · 必需字段缺失 |
| **AR 与参考核对** | 对**有上一轮 AR 产物的 family**（F01/F02/F08/F10）逐位核对 tokens/hash/len/stop；对**新 family**（F09/F03/F11/F04）本轮**没有历史参考** ⇒ 只能核对"两次自一致 + manifest token_sha 来源可信" | 与上一轮 AR 不一致（⇒ `BASELINE_REGRESSION`，停止） |
| **production ×2** | 两次 rc=0；`checks.passed`；几何上**出现** `s=Kp` 的 batched 前向（确认走的不是 serial）；两次完整 token IDs 一致；**GB6 分歧记账**已产出（首位置/共同前缀/两侧 token/margin/接受与工作量） | 两次不一致（⇒ 不确定性，停止）· 结构/有限性/停止原因失败 · **`margin`/分歧信息无法提取** · OOM |
| **serial ×2** | 18 项硬检查全过（含 `s==1`、**0** 次 `s>1`、审计非空、记账 `target_calls == proposed+rollbacks`、row-aligned 关闭、**D6 清理后实测 `block`**）· 两次完整 token IDs 一致 · **完整 64/64 与同 family AR 相同** | 与 AR **任何** token 分歧（⇒ `ALGORITHMIC_GATE_FAIL` 类；**不得**用 margin/ULP 豁免，**不得**转算子级定位）· 审计空泛 · D6 缺失 · 两次不一致 |

**全局硬停**（任一出现即停并保留产物）：结构失败 / 非有限值 / 重复不一致 / 非法停止 / OOM /
必需字段缺失 / 审计无法证明 `s==1` / Gate A 运行缺 D6 读数 / family 未登记（守卫 FAIL）/ 工作树不 clean / 环境不合格。

---

## 3. 命令与脚本的可执行性核对（**不假设文档命令有效**）

### 3.1 存在性

| 文档引用的路径 | 存在 | 备注 |
| --- | --- | --- |
| `scripts/so_batch.ps1` | ✅ | 已跟踪；含 `-Tag/-Cases/-Modes/-Reps`，设置 `SO_TAG`，调用 `specdec.engine.serial_oracle_gpu` |
| `specdec/engine/serial_oracle_gpu.py` | ✅ | 可**导入**（不执行生成）；默认 `MODE='serial'`、`CASE='F13'`、`EXPECTED_INSTANCES={'ar':1,'prod':2,'serial':2}` |
| `specdec/engine/serial_oracle_aggregate.py` | ✅ | 见 §3.2 的两处**阻断** |
| `specdec/engine/serial_oracle_aggregate_selftest_cpu.py` | ✅ | 实测 **PASS 18/0** |
| `specdec/engine/serial_oracle_env_check.py` | ✅ | 实测可运行（最近一次 `ENVIRONMENT_OK`） |
| `specdec/engine/verify_family_independence_cpu.py` | ✅ | 实测 **PASS 29/0** |
| `output/g5/attn/_so_stage_check.py`（v2 §6 逐阶段核对用） | ✅ | **未跟踪**（`output/` 被 `.gitignore` 忽略）⇒ 脆弱：`output/` 一旦清理，文档命令即失效 |

### 3.2 两处**阻断级**不一致（实测证据）

| # | 问题 | 实测 | 影响 |
| --- | --- | --- | --- |
| **B-1** | 聚合器 **tag 硬编码** | `AG.TAG = 'div8209_so'`；`_load_runs()` → **25** 条，而 `_load_runs('div8209_p2')` → **0** 条 | v2 §6 建议用新 tag `div8209_p2` 跑，但聚合器**读不到任何新产物** ⇒ 会给出"看不见新运行"的判决 |
| **B-2** | 聚合器 **family 硬编码** | `STAGE2_FAMILIES = ('F13','F01','F02','F08','F10')`；F09/F03/F11/F04 **不在其中** | 即便读到了新产物，聚合器**会忽略**这四个 family ⇒ **静默通过风险**（最危险的一类缺陷） |
| **B-3** | 逐阶段核对器 **tag 硬编码** | `_so_stage_check.py` 的 `TAG = "div8209_so"`（硬编码）且在 `output/`（未跟踪） | 换 tag 后逐阶段核对**找不到文件**；建议把它移进 `scripts/` 并参数化 tag/family |

**⇒ 结论：v2 §6 的命令序列现在**不能**照原样跑通。** 需要先做三项最小修改（都不改判据、不碰历史）：
① 聚合器与核对器的 **tag 参数化**（CLI 或环境变量）；② 聚合器的 **family 集合参数化**（或改为从产物自动发现 family）；
③ 核对器移入 `scripts/` 并参数化。修改后**必须**用 `serial_oracle_aggregate_selftest_cpu`（含字段名契约检查）+ 守卫复跑验证。

---

## 4. V-only FP8 的 Gate A 缺口（**不得**用 BF16 的五 family PASS 外推）

### 4.1 serial 模式在 `v_fp8` 下的语义（代码事实，`model_self.py`）

| 事实 | 位置 | 含义 |
| --- | --- | --- |
| `v_fp8` ⇒ `k_on=False, v_on=True, gv=GROUP_V(128)` | `model_self.py:602-603` | **K 保持 bf16，仅 V 量化**；V 的 group = 整个 head（128） |
| `quantize_kv_fp8`：`amax` **只在最后一维的组内**取（`xg.abs().amax(dim=-1)`），scale 形状 `(B,H,T,g)` | `:304-314` | scale 是**逐 token、逐组**的 ⇒ **不存在跨 token 耦合** ⇒ **`s=1`（serial）与 `s=Kp`（block）写出的每 token q/scale 相同** |
| 当前 chunk 的 V 先量化→再反量化后参与 attention（输出 dtype=bf16） | `:616-617` | 序列模式下每个 token 独立走同一条"量化→反量化"路径 ⇒ **serial 在该档下语义良好**（前提：scale 逐 token，已确认） |
| 缓存分支由 `"v_scale" in cache` 决定（fp8 payload vs bf16 全长） | `:628-641` | 存在**两种 KV 布局**；`v_scale` 分支只读历史、**不回写**新块 ⇒ **必须逐 run 记录并断言实际布局**，否则"FP8 存储"的语义含糊 |

⇒ **结论**：serial 在 `v_fp8` 下**语义良好、可以跑**；但**数值路径与 BF16 不同**（V 被 e4m3 舍入），
因此 **Gate A 必须在 fp8 档重跑**；BF16 的 5-family PASS **不构成** fp8 的证据。

### 4.2 该档需要**额外**的结构检查（现有门不够）

现有 `spec_cache.check_consistent` 只做：每层内所有 `dim>=3` 的 tensor 键**token 维（`shape[2]`）等长**、
且跨层等长（`spec_cache.py`）。它**不**检查 dtype、**不**检查 scale 键是否存在、**不**检查 scale 取值。

| # | 需要的检查 | 现状 |
| --- | --- | --- |
| S1 | token 维一致性（含 scale 张量）| ✅ 已有 `check_consistent` |
| S2 | **布局断言**：逐层键集合 + dtype 记录，并断言"有 `v_scale` ⟺ V 为 fp8 payload" | ❌ **需新增** |
| S3 | **dtype 断言**：scale 为 float32、形状 `(B,H,T,g)`；`v` 为 `float8_e4m3fn`；`k` 保持 bf16 | ❌ **需新增** |
| S4 | **scale 完整性**：全部有限且 > 0（代码对 0 置 1，需确认无异常） | ❌ **需新增** |
| S5 | round invariants（`tgt_len==drf_len==pre+(m+1|Kp)`）、finite、合法停止、确定性 | ✅ 已有（沿用 BF16 口径） |

### 4.3 最小测试矩阵（FP8 档，**待批准后才跑**）

| 组 | 内容 | 次数 |
| --- | --- | --- |
| **FP8-A（最小）** | **F13** × {AR×2, production×2, serial×2}，`kv=v_fp8` | **6** |
| **FP8-B（建议）** | 追加 **F10**（历史 margin=0.0 的 tie 家族，数值最敏感）× 同上 | +6 |
| **FP8-0（前置 smoke）** | 1 次 fp8 KV 结构 smoke：确认布局（S2/S3）与 S4 可记录 | **1** |

* 选择 F13 的理由：已知的 BF16 分歧首例（index 47）；F10 的理由：历史 margin=0.0，最能暴露 fp8 舍入影响。
* **不**把 BF16 五 family PASS 外推；**不**在 fp8 档声明任何 PASS 直到该档 Gate A 实跑完成。

---

## 5. 按依赖排序的执行清单

| # | 事项 | 状态 | 依赖 / 备注 |
| --- | --- | --- | --- |
| 1 | Stage 0 状态机测试（serial oracle）| **已完成** | PASS 42/0（`serial_oracle_stage0_*`） |
| 2 | 调用几何审计器自测 | **已完成** | PASS 19/0 |
| 3 | Stage 1（F13: AR×2/prod×2/serial×2）| **已完成** | `SERIAL_ORACLE_STAGE1_PASS`；AR/prod 历史锚点复现 |
| 4 | Stage 2（F01/F02/F08/F10: AR×2/serial×2）| **已完成** | `SERIAL_ORACLE_PASS`；16/16 运行 |
| 5 | 聚合器自测（字段名契约 + D6/D7 回归）| **已完成** | PASS 18/0 |
| 6 | 治理守卫 | **已完成** | PASS 29/0（记录级检查非空：22 次模型运行、`out_of_scope=[]`） |
| 7 | 协议 v2 草案 + 条款审计 + 本审计 | **已完成** | 本轮交付 |
| 8 | **D-1 / D-8**：确认"exact-id 门重新分层到 serial oracle"；澄清 `design_freeze_g4` 状态 | **待批准** | 阻断后续全部；见 brief §5 |
| 9 | **D-3**：登记 F09/F03/F11/F04 进守卫授权范围 | **待批准** | **阻断 Stage 2b**（已实测会被判越界） |
| 10 | **D-4**：删除 `index==0` 自动阈值，采用 R1–R3 解释规则 | **待批准** | 阻断判决口径定稿 |
| 11 | **D-2 / D-5 / D-6 / D-7 / D-9** | **待批准** | 见 brief §3 |
| 12 | **B-1/B-2/B-3**：聚合器与核对器的 tag/family 参数化 + 核对器移入 `scripts/` | **阻断** | 不修则 v2 §6 命令跑不通（§3.2 实测） |
| 13 | Stage 2b（F09/F03/F11/F04：AR×2 + production×2 + serial×2）| **待运行** | 依赖 8、9、12；预计 **24 次**运行 |
| 14 | FP8-0 / FP8-A（必要时 FP8-B）| **待运行** | 依赖 8、9、12 + D-6；预计 **7–13 次**运行 |
| 15 | v2 冻结（标记 FROZEN）| **待批准** | 依赖 8–11；本轮**不**自行冻结 |
| 16 | 4-family pilot / 正式测量（C1–C6、A1–A2）| **阻断** | 依赖 15；**未获准启动** |

**历史记录保持**：本轮**未**覆盖任何产物、**未**改动任何判决；
`ALGORITHMIC_GATE_FAIL`（row-aligned 1/4）、`B6-bf16 CHECK 19/20`、`SERIAL_ORACLE_STAGE1_PASS`、
`SERIAL_ORACLE_PASS` 等既有结论**原样保留**。本轮新增文件仅：
`docs/protocol_decision_brief_2026-10-04.md`、本文件、以及 `output/g5/attn/_newfam_audit.py`、`_cmd_audit.py`（只读审计脚本）。
