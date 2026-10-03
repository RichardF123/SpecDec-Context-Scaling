# Serial Target Verification Oracle — Stage 2 报告（四个 protocol validation family，2026-10-03）

> 设计（代码之前冻结）：`docs/serial_target_oracle_design_2026-10-03.md`
> Stage 0：`docs/serial_oracle_stage0_report_2026-10-03.md`
> Stage 1（F13）：`docs/serial_oracle_stage1_f13_2026-10-03.md`
> 上一轮协议验证：`docs/correctness_protocol_validation_2026-10-03.md`

## 0. 结论速览

| 项 | 结果 |
| --- | --- |
| **本轮判决** | **`SERIAL_ORACLE_PASS`** |
| 覆盖 | F13（Stage 1 已判 PASS）+ 本轮四个 **protocol validation family**：F01 / F02 / F08 / F10 |
| 运行 | 本轮 **16/16** 次运行完成（每 family：AR ×2 → 核对 → serial ×2），每次独立进程、串行执行 |
| 核心结果 | **五个 family 的 serial oracle 全部与同 family 的 target-only AR 逐 token 相同（64/64，无首分歧）**，且两次独立运行一致 |
| 几何 | 每次 serial 运行的 target generation 前向**全部 `s=1`**、**block（`s>1`）前向 = 0**、调用数与 `proposed + rollbacks` 恒等 |
| 环境 | 开跑前 `ENVIRONMENT_OK`（available commit 23.75 GiB）；**未**使用 `SMOKE_ALLOW_UNQUALIFIED`、**未**改门槛、**未**结束用户进程、**未**自动重试 |
| 术语 | F01/F02/F08/F10 上一轮已被使用 ⇒ 统一称 **protocol validation family**，**不**称"未见数据" |
| 下一步 | 仅**建议**按需规划后续；本轮**未**启动 B6 / pilot / 正式 factorial |

## 1. 运行前（治理守卫 + 环境预检）

| 项 | 结果 |
| --- | --- |
| 工作树 | clean，commit **`1f407cf892f34f2378631676f753a13dae6f4f74`**（运行前） |
| 治理守卫 | `verify_family_independence_cpu` ⇒ **PASS 29 / FAIL 0**（授权范围内 family 使用；无越界） |
| 环境检查 | `output/g5/attn/serial_oracle_env_check_20261003T163951Z.json` ⇒ **`ENVIRONMENT_OK`** |
| available commit | **23.75 GiB**（harness 16 GiB 线 **通过**；本轮 18 GiB 入口门槛 margin **+5.75 GiB**） |
| P-state | `qualified=True`，`sm_median=2400 MHz`，`mem_median=8000 MiB` |
| GPU 残留 | 19 个 GPU 进程中 **本项目残留 0**（本项目 python/实验进程 = 0）；显存 7563 MiB 可用 |
| `SMOKE_ALLOW_UNQUALIFIED` | 未设置 |

## 2. 实验矩阵与执行顺序（严格逐 family 交错核对）

对每个 family **先 AR ×2 并核对**（两次一致 + 与**上一轮该 family 的 AR 产物**逐位比较），**再** serial ×2：

| family | AR×2 产物 | serial×2 产物 |
| --- | --- | --- |
| F01 | `div8209_so_F01_ar_rep{1,2}_20261003T1640*.json` | `div8209_so_F01_serial_rep{1,2}_20261003T164*.json` |
| F02 | `div8209_so_F02_ar_rep{1,2}_20261003T1645*.json` | `div8209_so_F02_serial_rep{1,2}_20261003T1646*.json` |
| F08 | `div8209_so_F08_ar_rep{1,2}_20261003T1650*.json` | `div8209_so_F08_serial_rep{1,2}_20261003T165*.json` |
| F10 | `div8209_so_F10_ar_rep{1,2}_20261003T1654*.json` | `div8209_so_F10_serial_rep{1,2}_20261003T165*.json` |

固定配置（逐项与冻结设计一致，**未改**）：ctx 8192 · chunk 2048 · k 4 · max_new 64 · BF16 ·
target KV `base_bf16` · all-eager · target RoPE CPU-FP32 `inv_freq` · greedy argmax ·
`SMOKE_BF16_RED_REDUCTION=1` · `SMOKE_STRUCT_CHECK=1` · family 输入由 manifest 重建并校验 token hash ·
**不重跑** production / row-aligned（引用上一轮产物作背景对照）。

## 3. 每 family 结果

### 3.1 AR 基线（本轮 ×2 与上一轮产物逐位对照）

| family | 本轮 AR hash（两次） | 长度 | stop | 两次一致 | 与上一轮 tokens / hash / len / stop | 自报 hash 与 harness 一致 |
| --- | --- | --- | --- | --- | --- | --- |
| F01 | `5645f0d6943dcb23` | 64 | `max_new_reached` | ✅ | **True / True / True / True** | ✅ |
| F02 | `3889082e6a0f4594` | 64 | `max_new_reached` | ✅ | **True / True / True / True** | ✅ |
| F08 | `4f4019d300256c2d` | 64 | `max_new_reached` | ✅ | **True / True / True / True** | ✅ |
| F10 | `79ca5c727a671e9d` | 64 | `max_new_reached` | ✅ | **True / True / True / True** | ✅ |

⇒ 四个 family 的**基线完全复现**（含完整 token IDs），AR 几何均为 `s=1`×63、记账 63/63。
**未出现** `BASELINE_REGRESSION` 条件。

### 3.2 serial oracle（每 family ×2）

| family | serial hash（两次） | == AR | 首分歧 | target 生成调用 | 全 `s=1` | block 前向 | 记账 | 轮数 / trace | 两次一致 | D6 复位 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F01 | `5645f0d6943dcb23` | ✅ | **无** | 88 | ✅ | **0** | ✅ | 21 / 21 | ✅ | ✅ `block` |
| F02 | `3889082e6a0f4594` | ✅ | **无** | 141 | ✅ | **0** | ✅ | 30 / 30 | ✅ | ✅ `block` |
| F08 | `4f4019d300256c2d` | ✅ | **无** | 88 | ✅ | **0** | ✅ | 20 / 20 | ✅ | ✅ `block` |
| F10 | `79ca5c727a671e9d` | ✅ | **无** | 120 | ✅ | **0** | ✅ | 26 / 26 | ✅ | ✅ `block` |
| （F13，Stage 1） | `6e7522bf2521ad7b` | ✅ | 无 | 115 | ✅ | 0 | ✅ | 25 / 25 | ✅ | na（见 §5） |

每次 serial 运行的 17 项硬检查 + 第 18 项（D6）**逐项通过**：
`diagnostic_only=true` · 运行期模式 `serial` · row-aligned **关闭** · `generation_seq_lens=[1]` ·
`n_block_generation_forwards=0` · 审计非空（按实例身份识别 target/draft）· 记账
`target_calls == proposed + rollbacks` · 分块 prefill（4×2048）未被误判 · 逐轮 trace 字段完整 ·
round invariants（独立重算 `tgt_len==drf_len==pre+(m+1|Kp)`）· cache 结构/长度/dtype ·
有限性（`checks.finiteness_coverage.n_checked > 0`、`first_bad=null`、逐阶段 ok、`uncovered=[]`）·
gen IDs 合法 · stop 与 AR 相同 · 无 OOM（峰值 reserved 10648 MiB）。

运行观测（**含插桩，仅供运行参考，不作为加速比结论**）：AR ≈ 8.4 s / 次；serial ≈ 53–72 s / 次。

## 4. 判决及其依据

判决词表与判据（本轮任务书）：

| 判决 | 本轮是否命中 | 依据 |
| --- | --- | --- |
| **`SERIAL_ORACLE_PASS`** | **✅ 命中** | F13 + F01/F02/F08/F10 全部与 AR 逐 token 一致；重复性一致；所有硬检查（含 D6）通过；`needs_review=[]`、`baseline_failures=[]`、`serial_failures=[]` |
| `SERIAL_ORACLE_FAIL` | ✗ | 无 token 分歧、无控制不变量失败 |
| `BASELINE_REGRESSION` | ✗ | 四个 family 的 AR 两次一致，且与上一轮 AR 产物 tokens/hash/len/stop 全同 |
| `CHECK_NEEDS_REVIEW` | ✗ | 环境合格；审计非空；D6 复位证据齐备（本轮 16 次运行实测）；必需字段完整 |

机器可读判决：`output/g5/attn/div8209_so_stage2_aggregate_20261003T170149Z.json`
（`decision=SERIAL_ORACLE_PASS`，`next_round_allowed=true`；同目录 `.jsonl` 为逐 run 明细）。

## 5. D6 回归（本轮**实做**，不是从 F13 旧产物推断）

`serial_oracle_gpu.py` 曾在**复位之前**读 `verify_mode_after_run`（D6）。修正后：运行结束时先记
`verify_mode_at_run_end`，**清理完成后**再实际读取 `verify_mode_after_run`（应为 `block`），
并由硬检查 18 判定。

| 运行 | `verify_mode_before` | `verify_mode_at_run_end` | **清理后实测** | 硬检查 18 |
| --- | --- | --- | --- | --- |
| 本轮 8 次 serial 运行（F01/F02/F08/F10 ×2） | `serial` | `serial` | **`block`** | **True** |
| F13 的 2 次 serial 运行（Stage 1，**D6 修正之前**） | `serial` | 键不存在 | `serial`（旧的复位前读数） | **na**（写明理由，不记为 False、也不静默跳过） |

⇒ D6 的复位证据**由本轮 16 次运行的实测记录提供**；F13 的旧产物按"不适用 + 理由"处理，
不据此旧字段推断。

## 6. 与此前 production / row-aligned 结果的关系（只读引用上一轮产物）

| family | 本轮 AR（= serial） | 上一轮 production（block）首分歧 | 上一轮 row-aligned 诊断首分歧 | 本轮 serial 首分歧 |
| --- | --- | --- | --- | --- |
| F01 | `5645f0d6943dcb23` | 14（hash `67388e02c2b6350c`） | 无（hash 与 AR 相同） | **无** |
| F02 | `3889082e6a0f4594` | 7（hash `edf0ec5509650dfc`） | 9（hash `10bf5e0b9d71e999`） | **无** |
| F08 | `4f4019d300256c2d` | 52（hash `47494ac4e1656e3f`） | 52（同 hash） | **无** |
| F10 | `79ca5c727a671e9d` | 8（hash `d4b5c6a137456cd5`） | 17（hash `34ba931f608bae3a`） | **无** |
| F13 | `6e7522bf2521ad7b` | 47（hash `e90dad2b3fcf1c65`） | —（row-aligned 轮未跑 F13） | 无 |

上一轮协议验证的判决为 `ALGORITHMIC_GATE_FAIL`（Gate A 要求 row-aligned 与 AR 逐 token 一致，
仅 F01 通过；F02/F08/F10 在 9/52/17 处分歧）。

**关系（本轮能说与不能说）**：
* **能说**：本轮把 target verification 换成**与 AR 完全相同的逐 token `M=1` 执行路径**后，
  五个 family 都逐 token 复现 AR；而上一轮"在 block 内保留形状、只取第 0 行"的 row-aligned 诊断
  在 3/4 个 family 上仍产生分歧 ⇒ **既有控制逻辑不是分歧来源**，分歧与"块内多行执行路径"相关。
* **不能说**：本轮**不**证明差异的具体成因（不定位到算子/舍入层），也**不**推翻上一轮的 row-aligned
  结论（该结论描述的是它自身的执行路径）。"有限精度执行路径差异"仍是**解释**，不是本轮数据能证明的命题。
* 本轮**未**重跑 production/row-aligned，故其数字一律标注为"上一轮产物"。

## 7. 本轮工具缺陷（D9）与同族回归测试

| # | 缺陷 | 表现 | 修正 |
| --- | --- | --- | --- |
| **D9** | `_row()` 未暴露 `verify_mode_at_run_end` / `verify_mode_after_run`，而硬检查 18 读这两个名字 | 全部 8 次 serial 的 D6 检查读出 `None` ⇒ 假阴性 ⇒ 首次聚合误判 **`SERIAL_ORACLE_FAIL`** | `_row` 按原始字段名暴露两个字段；硬检查 18 读回真实值 |

**中间判决保留**：`div8209_so_stage2_aggregate_20261003T165902Z.json`（`SERIAL_ORACLE_FAIL`，由 D9 造成）
**未覆盖**；修正后对**同一批 16 次运行产物**重新判定得 `SERIAL_ORACLE_PASS`
（`..._20261003T170149Z.json`）。**未重跑任何实验**、未改任何门槛、未放行任何 margin/ULP 例外。

同一类字段映射缺陷本轮/上一轮已发生三次（D7 有限性计数、D9 D6 字段、以及 Stage 1 的 D6 记录顺序），
因此新增 **`serial_oracle_aggregate_selftest_cpu.py`**：一条**字段名契约**检查
（硬检查源码里引用的每个 `r.<字段>` 都必须存在于 `_row` 输出），外加 D6/D7 的显式回归与
"F13 的 D6 必须为 na 且带理由"断言。首次运行 **14/1**（测试自身选错产物 ⇒ 已修正并保留产物
`serial_oracle_aggregate_selftest_20261003T170051Z.json`），修正后 **PASS 18/0**
（`..._20261003T170140Z.json`）。

## 8. 已证明 / 未证明与适用范围

**已证明（五个 family 的实证）**
1. 在冻结配置下（ctx 8192/chunk 2048/k 4/max_new 64/BF16/`base_bf16`/all-eager/target CPU-FP32 RoPE/greedy），
   当 target verification 完全走与 AR 相同的逐 token `M=1` 路径时，**现有 SpecDec 控制逻辑在
   F13/F01/F02/F08/F10 五个 family 上生成与 target-only AR 完全相同的 64 个 token IDs**。
2. 伴随**调用几何硬证据**：每次运行 target 的 generation 前向**全部 `s=1`**、**0** 次 `s>1` block 前向、
   分块 prefill 未被误判；调用数与 `proposed + rollbacks` 恒等（F01 88、F02 141、F08 88、F10 120、F13 115）。
3. 每个 family 的**基线可复现**：AR 两次一致，且与上一轮该 family 的 AR 产物逐位（tokens/hash/len/stop）相同。
4. 控制逻辑（accept/replacement/rollback/EOS/截断/cache 提交与回滚）在真实模型上逐轮不变量全过；
   cache 结构、有限性、合法 token、停止原因全部通过；两次独立运行可重复。

**未证明 / 不得外推**
* **不是**所有输入上的形式化证明，也**不是** 8192-token 语料全集的结论：仅 **5 个 family**（各 1 个 8192 窗口、64 token）。
* **不**解释 production block verification 差异的**成因**，**未**做算子级定位（本轮明确不做）。
* **不**给出速度结论：serial 每轮 `kp` 次串行前向（≈53–72 s/次）远慢于 AR（≈8.4 s/次），
  计时含插桩，**不得**作为加速比。
* Stage 2 PASS **不**等于 B6 通过，**不**授权 pilot / 正式 factorial。

## 9. 产物清单

| 类型 | 路径 |
| --- | --- |
| 环境检查 | `output/g5/attn/serial_oracle_env_check_20261003T163951Z.json` |
| 16 次运行原始产物 | `output/g5/attn/div8209_so_F0{1,2,8}_*/…`、`…F10_*`（16 个 `.json` + 16 个 `.pt`） |
| 运行日志 | `output/g5/attn/div8209_so_F01_{ar,serial}_rep{1,2}.log` 等 8 份 per-run 日志 + `div8209_so_batch_*.log` |
| Stage 2 聚合 | `div8209_so_stage2_aggregate_20261003T170149Z.json` / `.jsonl` |
| 中间判决（保留） | `div8209_so_stage2_aggregate_20261003T165902Z.json`（D9 造成的 FAIL） |
| 聚合器自测 | `serial_oracle_aggregate_selftest_20261003T170140Z.json`（PASS 18/0）；`..._20261003T170051Z.json`（14/1，保留） |
| 治理守卫 | `output/g5/attn/family_independence_cpu_check.json` ⇒ PASS 29/0（记录级：22 次模型运行、`out_of_scope=[]`） |
| 本报告 | `docs/serial_oracle_stage2_2026-10-03.md` |

## 10. 下一步（仅建议，本轮不执行）

* Stage 2 的 PASS 只构成**五个 family 的实证支持**。后续若要把范围扩到更多 family 或更长生成，
  需要**新的预登记**（family 集合、窗口、max_new 与判据在运行前写死），并按同样纪律执行。
* B6 / pilot / 正式 factorial：本轮**未启动**，也不在本报告的授权范围内。
