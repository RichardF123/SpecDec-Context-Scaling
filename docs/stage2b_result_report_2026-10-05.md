# Stage 2b 结果报告（归档与交接，2026-10-05）

> **结论（唯一可声称的状态）**：**限定范围 Stage 2b 通过 —— `STAGE2B_PASS`**。
> 即：**4 个新 family（`F09/F03/F11/F04`）× 3 模式 × 2 次 = 24 runs**，在 **8K 自然语言窗口**上，
> 按**新口径**（serial oracle `M=1` 逐 token exact-id）通过。
> **不得**据此推出：`B6_BF16_PASS` / `FULL_B6_PASS` / `BF16_FORMAL_READY` / `FP8_FORMAL_READY`，
> 也**不得**作"同输出无损加速"类表述。FP8 **完全未运行**。
>
> 本报告为**归档文档**：只读复核既有产物，**未重跑 GPU**、未修改任何历史产物、未扩展测试矩阵。

## 1. 运行覆盖（24/24）

| 项 | 结果 |
| --- | --- |
| 三元组覆盖 | 期望 **24**，实得 **24**（4 family × {`ar`,`prod`,`serial`} × rep{1,2}） |
| 环境拒绝 | **0**（`n_env_blocked=0`；全部 `env_qualified=True`、`model_run_happened=True`） |
| 重试 / 样本挑选 | **无**（一 run 一独立进程；硬失败即停的规则未被触发） |
| 运行窗口（UTC） | `20261005T075114Z` → `20261005T082221Z` |
| 每 run 状态 | `stage=complete`、`stop_reason=max_new_reached`、`gen_len=64`、`checks_passed`、逐轮不变量 ok、`self_checks_all_ok` |

## 2. AR 基线（新口径下的"新 family 无输出锚点"路径）

| family | AR 独立 hash（两次一致） | 基线判据结果 |
| --- | --- | --- |
| F09 | `33c5d4de2abb9205` | `ar_baseline_ok=True` |
| F03 | `872c31c494a76b26` | `ar_baseline_ok=True` |
| F11 | `fbe9f524fd3d16ac` | `ar_baseline_ok=True` |
| F04 | `155cf457a10ebad9` | `ar_baseline_ok=True` |

判据内容：两次独立重复逐 token 一致 · 独立计算 hash / `end_state.gen_hash` / 实际长度 / `end_state.gen_len` /
`smoke.stop_reason` / `end_state.stop_reason` 两两自洽 · manifest 输入 `token_sha1` 匹配。
**`baseline_failures=[]`**（旧 family 的登记锚点规则未被套用到这四个新 family 上）。

## 3. serial exact equality（Gate A 的实际判据）

| family | serial 与同 family AR | 两次 serial 一致 | serial 侧 D6 读数 | serial 侧 cache 结构检查数（**两次重复各自**） |
| --- | --- | --- | --- | --- |
| F09 | **完整 token IDs 相同** | 是 | `verify_at_end=serial` → `after_cleanup=block` | `cache_structure_n_checks = 52`（rep1 与 rep2 **均为 52**） |
| F03 | **相同** | 是 | 同上 | `cache_structure_n_checks = 40`（rep1 与 rep2 **均为 40**） |
| F11 | **相同** | 是 | 同上 | `cache_structure_n_checks = 56`（rep1 与 rep2 **均为 56**） |
| F04 | **相同** | 是 | 同上 | `cache_structure_n_checks = 48`（rep1 与 rep2 **均为 48**） |

> **勘误（2026-10-05，仅文档修正，未重跑任何数值实验）**：本表 F03 行原先写作"`ok / 40 → 56`"，**错误**。
> 经只读复核：**F03 的两次 serial 重复均为 40**；**56 属于 F11**（两次均为 56）。AR 侧该字段恒为 2（AR 无投机轮）。
> 各 family 的 production 侧值供对照：F09 `52`、F03 `38`、F11 `54`、F04 `48`（两次重复均相同）。

serial 侧另通过 18 项硬检查（含 D6）、逐轮字段完整、trace 与轮记录一致；
**`serial_failures=[]`**、`serial_equals_ar=True`、`serial_reps_identical=True`。

## 4. production 与 AR 的分歧（**发散轨迹**，非门）

| family | production 分类 | **production 对 AR 的真实首分歧**（JSON 分模式字段） | 两次重复 | 两侧 token（该下标） |
| --- | --- | --- | --- | --- |
| F09 | `TOKEN_EQUAL` | 无（逐 token 相同） | 一致 | — |
| **F03** | `NUMERICAL_DIVERGENCE_RECORDED` | **输出内下标 59**（共同前缀 59） | 一致（59/59） | AR `11577` vs prod `23195` |
| **F11** | `NUMERICAL_DIVERGENCE_RECORDED` | **输出内下标 49**（共同前缀 49） | 一致（49/49） | AR `304` vs prod `572` |
| F04 | `TOKEN_EQUAL` | 无 | 一致 | — |

production 结构检查（两次一致 · `checks_passed` · cache 结构 · 逐轮不变量 · 有限性 · token 在词表 ·
停止合法 · **存在 `s>1` 的 batched 几何** · 非 OOM）四个 family 全部通过。
**production 的分歧只作发散轨迹记录，不判失败；不得表述为"同输出无损加速"。**

## 5. GB6 字段完整性（仅分歧 family 需要诊断）

| family | `divergent` | 运行侧缺 | 诊断侧缺 | `missing` |
| --- | --- | --- | --- | --- |
| F03 | True | `[]` | `[]` | `[]` |
| F11 | True | `[]` | `[]` | `[]` |
| F09 / F04 | False（无分歧 ⇒ 无需记账） | — | — | `[]` |

运行侧字段（实测值）：F03 `首分歧下标=59`、`绝对位置=8251`、`轮号=17`（来源 `prod`）、`ar_token=11577`、`prod_token=23195`；
F11 `49 / 8241 / 20 / 304 / 572`。工作量：AR `decode_steps=63`、`target_gen_calls=63`、
`acceptance="not_applicable（AR 无投机）"`（**未填造 AR 接受率**）；F03 prod `rounds=19/proposed=73/accepted=56/rollbacks=8`、
F11 prod `27/107/43/21`。
诊断侧字段（`correctness_protocol_margins`，**AR 口径重建行**）：F03 `top1=23195, top2=11577, margin=0.0, tied_at_max=2, cache_len_after=8251`；
F11 `572 / 304 / 0.0 / 2 / 8241`。

**⚠️ 并列证据的边界（不可越界解读）**：两处的 `margin=0.0`、`tied_at_max=2` 是 **AR 单侧重建行**的并列，
**只能**作为"该位置对微小数值差异敏感"的**线索**；**不得**归因为 production 分歧的最终成因，
**不得**标 `TIE-A`（双侧决策行证据缺失）。按 D-10，本阶段一律记 **`TIE-C`（证据不足）**，且**不豁免** exact-id 门。

## 6. 退出码一览（本次执行实测）

| 步骤 | 命令 | 退出码 |
| --- | --- | --- |
| 预检（运行前 / 运行后 / 本轮只读复核） | `stage2b_preflight_cpu` | **0 / 0 / 0** |
| 治理守卫 | `verify_family_independence_cpu` | **0**（**33/0**） |
| 环境门（运行前） | `serial_oracle_env_check` | **0**（`ENVIRONMENT_OK`） |
| 24 runs（12 次 `so_batch.ps1` 调用） | `so_batch.ps1 -Tag div8209_p2 -Cases <F> -Modes <m> -Reps 1,2` | 每次 **0**（每次批末 `nonzero=0`） |
| 逐阶段核对（每 family 三档） | `stage_check --tag div8209_p2 --family <F> --modes … --reps 2` | **0**（12 次全 0；本轮只读复核再跑 4 次仍全 **0**） |
| GB6 诊断（3 次：F03 / F11 / F03+F11 合并） | `correctness_protocol_margins --stage2b --runs-tag div8209_p2 --tag div8209_p2 --families …` | **0 / 0 / 0**（非 4、非 5） |
| 只读判决 | `stage2b_verdict --tag div8209_p2 --families F09,F03,F11,F04 --modes ar,prod,serial --reps 2 --gb6 <合并诊断>` | **0**，`decision=SERIAL_ORACLE_PASS`，`needs_review=[]` |

## 7. 原始产物路径（相对仓库根）

* **运行记录（24）**：`output/g5/attn/div8209_p2_F{09,03,11,04}_{ar,prod,serial}_rep{1,2}_<runid>.json`
  （例：`div8209_p2_F09_ar_rep1_20261005T075114Z.json` … `div8209_p2_F04_serial_rep2_20261005T082221Z.json`）
* **每 run 的 `.pt`**：同名前缀的 `…_<runid>.pt`（24 份）
* **harness smoke（24，被上述 run 经 `smoke_artifact` 链接）**：`output/g5/attn/smoke8k_eager_<runid>.json`
* **GB6 诊断（3）**：`div8209_p2_margins_20261005T080631Z.json`（F03）、`…081637Z.json`（F11）、`…082420Z.json`（F03+F11，判决所用）
* **判决（2）**：`div8209_p2_stage2b_verdict_20261005T083145Z.json`（+ `.jsonl`）
* **逐 run 日志**：`div8209_p2_<family>_<mode>_rep<rep>.log`、`div8209_p2_batch_<stamp>.log`

## 8. 两个审计注意事项（**必须随本报告一起签收**）

1. **治理修正的时间顺序**：`fe2b250`（"按 `smoke_artifact` 链接逐份授权 harness smoke"的 scoped 修正）
   提交于 **GPU runs 之后**（提交时间 `2026-10-05T16:32:50+08:00`；run 窗口为 `15:51–16:23`，判决产物 `16:31:45`）。
   **最终治理检查通过（守卫 33/0、preflight 0），但不得写成"这项修正从首个 run 起就已生效"。**
   事实顺序为：`089b28a`（D-3 登记，运行前，使 preflight 3→0）→ 24 runs → 诊断 → 只读判决 → `fe2b250`（运行后补齐配套授权）。
   *同时说明*：24 runs 期间未被治理检查"当场放行"的原因是每次 run 还会写一份 harness smoke 产物；
   在 `fe2b250` 之前，守卫/预检无法接受这些链接产物（当时代码尚未按链接授权）。这不改变任何 run 的数值结果，
   但**签收时应按上述顺序理解**。
2. **判决 CLI 屏幕摘要的正确读法**：CLI 打印的 `首分歧=[None,None]` 指 **serial 对 AR**（`None` = 逐 token 相同，故为通过信号），
   **不是** production 对 AR。**production 的真实首分歧在 JSON 的分模式字段**：
   * `per_family["F03"].production_divergence.first_divergence = [59, 59]`（共同前缀 `[59, 59]`）
   * `per_family["F11"].production_divergence.first_divergence = [49, 49]`（共同前缀 `[49, 49]`）
   * `per_family[<F>].production_divergence_class ∈ {TOKEN_EQUAL, NUMERICAL_DIVERGENCE_RECORDED}`
   **归档一律以 JSON 分模式字段为准，不以屏幕摘要为准。**

## 9. 结论边界与交接

* **可声称**：`STAGE2B_PASS`（限定范围：4 个新 family / 8K 自然语言 / 24 runs / 新口径）。
  **不可声称**：完整 B6 通过 · `B6_BF16_PASS` · `FULL_B6_PASS` · BF16/FP8 正式实验放行 · 同输出无损加速。
* **代码所有权**：正式实验 runner、factorial 采集、性能统计与分析、绘图、最终流水线**由用户亲自编写**；
  本次**未创建、未修改**这些代码，也**未**安排其它 agent 编写。**后续 B6 类型覆盖工具停留在待办**（D-11(a)），未经另行授权不实现。
* **仍待小组决定（批 2）**：D-11（B6 提示类型覆盖如何处置）· D-12（"未过门"按 KV 分别放行 / 全局放行）·
  D-5（质量口径）· D-6（FP8 各自 Gate A）· D-7（词条）· D-9（是否再补 FP32）· 双侧行捕获 P-1/P-2 是否实施。
* **已知风险**：① 覆盖范围仅自然语言 8K 窗口，**不能**外推为完整 B6；② 两处分歧落在**精确并列**且只有 AR 侧证据（记 `TIE-C`）；
  ③ 诊断的 `cache_len_after` 是 AR 口径**重建**长度，不是被诊断 run 的实际长度；④ 入口内存余量薄（约高出 18 GiB 门槛 1.4 GiB），
  后续运行仍须先过环境门；⑤ 本轮治理工具改动（`089b28a` / `fe2b250`）属 D-3 授权与配套 scoped 检查，按"逐份链接"限定，未按 family 泛指。
