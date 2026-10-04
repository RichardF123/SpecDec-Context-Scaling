# Stage 2b 运行手册（2026-10-04，**GB6 记账轮**修订）

> **本轮结论（两问两答，先给答案再给依据）**
>
> **1. 工具是否已能正确判决 Stage 2b？—— 本轮的三个判决漏洞已修好并**在真实产物上复现验证**；
> 但"能正确判决"目前只覆盖**产物侧**（读记录、配对、分类、拒绝）。真正的 GPU 阶段仍不能开跑**
> （见第 2 条）。本轮**没有**任何 GPU 运行，因此诊断工具的 GPU 分支属**已实现未验证**。
>
> **2. 即使工具就绪，哪些协议批准仍阻止 GPU 阶段？** —— `D-1`（exact-id 门重新分层到 serial oracle）、
> `D-3`（登记 `F09/F03/F11/F04` 进 run 级授权）、`D-4`（采用 DR1–DR4）、`D-5`、`D-6`（FP8 Gate A）
> 均**待批准**；preflight 因 D-3 仍返回 `3`。**工具就绪 ≠ 获准开跑。**
>
> 判定依据（本轮实测，逐条可复现）：见 §0 与 §6 —— **三个反例的修复前/修复后实际输出**，
> 而不是"自测总数 PASS"。

---

## 0. 本轮修的三个漏洞（修复前实测输出 → 根因 → 修复 → 修复后实测输出）

### CF1：算出了 `gb6["missing"]` 却从不据此阻断

| | |
| --- | --- |
| **修复前实测**（合成一套齐全产物，只抹掉 `end_state.prompt_len`） | `gb6.missing = ['absolute_position']`、`gb6.divergent = True`，判决 = **`SERIAL_ORACLE_PASS`** |
| **根因** | `evaluate()` 只检查了**诊断侧**缺失（`from_diagnostic_missing`），**运行侧**的 `gb6["missing"]` 算出来后被丢弃；另外 `v in (None, {})` 这种判法把"字典存在但关键值全为 `None`"当成字段齐全 |
| **修复位置** | `stage2b_verdict.py`：① `_gb6_present()`（None/空串/空容器/**全 None 的字典或序列**都算缺失，而 `0`/`False` 算存在）；② `GB6_WORK_REQUIRED`（`ar_work` 必须有 `decode_steps`/`target_gen_calls`；`prod_work` 必须有 `rounds`/`proposed`/`accepted`，**逐子字段**点名）；③ `evaluate()` 中 `if gb6["missing"]:` ⇒ `CHECK_NEEDS_REVIEW` |
| **修复后实测** | 判决 = **`CHECK_NEEDS_REVIEW`**，理由 `F09: GB6 必需字段**实际为空** ['absolute_position']（运行侧缺 ['absolute_position']；诊断侧缺 []）⇒ 不得判 PASS`；真实 F13 的运行侧现在**全齐**（`from_runs_missing=[]`） |
| **顺带修正的语义** | AR 的工作量**不再塞进"接受率"模板**：`ar_work = {decode_steps: 63, target_gen_calls: 63, actual_len: 64, stop_reason: 'max_new_reached', acceptance: 'not_applicable（AR 无投机 ⇒ 无 proposed/accepted，不填造接受率）'}`（真实 F13 实测） |
| **无分歧时不误伤** | 旧规则若直接改会在"token 相等"时把全部字段判缺；现在 `divergent=False ⇒ missing=[]` 且写 `not_applicable`（用例 CF1d：无分歧 ⇒ 无需诊断即可 PASS） |

### CF2：AR 的两处自报状态从不互相核对

| | |
| --- | --- |
| **修复前实测**（新 family AR：`smoke.stop_reason=max_new_reached`，`end_state.stop_reason=eos`） | `ar_checks.stop_legal_and_consistent = true`、`ar_baseline_ok = True`，判决 = **`SERIAL_ORACLE_PASS`** |
| **根因** | `_ar_baseline_new` 只对 `smoke.stop_reason` 判"合法且两次一致"，**从不与 `end_state.stop_reason` 对照**；`end_state.gen_len/gen_hash` 与独立计算值的缺失也只会变成 `None != hash` 式硬失败，**分不出"缺字段"还是"相互矛盾"** |
| **修复位置** | `_ar_baseline_new` 重写为**逐条**核对：生成 IDs **独立算出的** `gen_hash`、`end_state.gen_hash`、实际长度、`end_state.gen_len`、`smoke.stop_reason`、`end_state.stop_reason`（每 rep 一份 `per_row` 明细），并把问题**分类**：<br>· `missing_fields`（缺字段）⇒ `CHECK_NEEDS_REVIEW`<br>· `self_contradiction`（字段都在但矛盾，含预登记长度不符/停止非法）⇒ `CHECK_NEEDS_REVIEW`（**不**声称基线回归：矛盾说明记录坏了）<br>· `between_reps`（两次重复之间不一致）⇒ `BASELINE_REGRESSION`（确定性失败）<br>旧 family（F13/F01/F02/F08/F10）**保留原判据不变**，只是把分类显式化 |
| **修复后实测** | 矛盾 ⇒ **`CHECK_NEEDS_REVIEW`**，理由 `AR 产物**自报状态相互矛盾** {'rep1': {'stop_smoke_vs_end_state': {'smoke': 'max_new_reached', 'end_state': 'eos'}}, …}`；缺 `end_state.stop_reason` ⇒ 归为 `AR 基线**证据缺失**`；两次不一致 ⇒ 仍 `BASELINE_REGRESSION`（CF2d）；两者同时出现 ⇒ 矛盾优先（CF2e） |
| **真实数据现状（诚实说明）** | div8209_so/F13 的 6 份有效产物 `smoke.stop_reason` 与 `end_state.stop_reason` **全部一致** ⇒ CF2 是**潜在**漏洞（只在记录被写坏/工具写入口径不一致时才暴露），本轮靠合成用例覆盖 |

### CF3：诊断工具按旧模式名取产物，找不到 divergent 侧就退化为末 token

| | |
| --- | --- |
| **修复前实测**（真实 `div8209_so` / F13，照抄旧选择逻辑） | `{'ar_len': 64, 'aligned_len': 0, 'spec_len': 0, 'cp': 64, 'pos': 63}` ⇒ 位置 = **末 token 63**，而真实首分歧 = **47** |
| **根因** | `correctness_protocol_margins.py` 用 `_gen("ar") / _gen("aligned") / _gen("spec")`；Stage 2b 的模式名是 `ar/prod/serial` ⇒ `diag_gen` 为空 ⇒ `pos = max(0, len(ar_gen)-1)` |
| **修复位置** | ① **新增纯 CPU 模块** `gb6_diag_select.py`：`select_divergent_pair()` 负责配对 + 来源/配置核对 + 首分歧计算，**缺 divergent 侧即抛 `MISSING_ARTIFACT`**，绝不退化；② `correctness_protocol_margins.py`：模式名参数化（`--ar-mode/--div-mode/--stage2b`）、改用被选中的**那两份文件**、载入输入长度与产物 `prompt_len` 交叉核对、`position_origin` 自证来源、退出码 **4** = 有 family 配对被拒；③ 消费端 `stage2b_verdict`：诊断必须带 `GB6_DIAG_PROVENANCE`（`family`/`runs_tag`/`position_origin`/`position`/`prompt_len`）且 `position_origin` 必须含 `first_divergence`，否则**拒绝** |
| **修复后实测（有 prod）** | `status=DIVERGENT 首分歧=47 position=47 round_index=18 prompt_len=8192 ar_token=6702 prod_token=6652`，来源串写明用的是**哪两份文件**；配置核对 10 项全 `true` |
| **修复后实测（无 prod）** | 真实 `div8209_so`+`F01`（只有 AR/serial）⇒ `MISSING_ARTIFACT`：`tag='div8209_so' family='F01' mode='prod' 在 attn 下没有任何产物（**不允许**用其它模式/其它 tag/family 代跑，也不允许退化为末 token）` |
| **修复后实测（消费端）** | `position_origin='legacy_last_index（…非首分歧）'` 的记录 ⇒ 判决 `CHECK_NEEDS_REVIEW`，理由点名 `position_origin … first_divergence`；缺 `position_origin` 的**旧 margins 产物**同理被拒（CF3/CF3b） |
| **顺带修掉的一个**过度严格**（真实数据抓到） | 首版把 `kv_format` **整体**比对 ⇒ 真实配对被误拒：AR `{'target':'fp16','draft':'not_loaded'}` vs prod `{'target':'fp16','draft':'fp16'}` —— **AR 本来就不加载 draft**。现只比对 **target 侧**（`kv_format.target`、`dtypes.model`、`dtypes.target_kv_payload`），并把这条豁免写进记录的 `config_notes` |
| **观察（不在本轮改动范围）** | 真实产物里 target KV 标签写作 `"fp16"`，而 `dtypes.target_kv_payload` 是 `base_bf16`。这是既有标注惯例，**本轮只做"两侧相等"核对，不改任何数值路径**；如需澄清属另一轮事项 |

> **本轮另外发现并顺手修的**：`end_state.n_decode_steps` 未在 `SV._row` 暴露（AR 工作量要按它记录）
> ⇒ 已按产物真实字段名暴露；聚合器字段名契约自测（s12）因此回到 **21/0**。
> 过程说明：我一度用 PowerShell `Set-Content` 回写过一个 UTF-8 源文件（历史上曾因此产生 mojibake），
> 随即做了字节级校验（合法 UTF-8、替换字符 0、`py_compile` 通过）并**剥掉多余 BOM**；
> 之后所有源码改动一律走编辑工具。

## 1. 工具清单（全部受版本控制）

| 工具 | 用途 | 退出码 |
| --- | --- | --- |
| `python -X utf8 -m specdec.engine.stage2b_preflight_cpu` | 开跑前只读预检 | `0` 可跑 · `3` 工具就绪但 family **未登记** · `1` 硬问题 |
| `python -X utf8 -m specdec.engine.stage_check --tag … --family …` | 逐阶段硬门（AR/production/serial） | `0` 通过 · `1` STOP · `2` PENDING · `3` 参数非法 |
| `python -X utf8 -m specdec.engine.stage2b_verdict --tag … --families … [--gb6 …]` | Stage 2b 判决 | `0` PASS · `1` 其它判决 · `2` 参数非法（不落盘） |
| `python -X utf8 -m specdec.engine.correctness_protocol_margins --stage2b …` | **GB6 诊断 pass**（位置 + top-1/top-2/margin/tie/cache） | `0` 完成 · **`4` 有 family 配对被拒** · `5` 检查失败 · `2` 崩溃 |
| `python -X utf8 -m specdec.engine.gb6_diag_select_selftest_cpu` | 诊断**配对/来源核对**自测（纯 CPU） | `0` PASS（25 项） |
| `python -X utf8 -m specdec.engine.stage2b_verdict_selftest_cpu` | 判决工具自测（合成产物） | `0` PASS（92 项） |
| `python -X utf8 -m specdec.engine.serial_oracle_aggregate_selftest_cpu` | 字段名契约自测 | `0` PASS（21 项） |
| `python -X utf8 -m specdec.engine.verify_family_independence_cpu` | 治理守卫 | `0` PASS（31 项） |

## 2. 各阶段硬门（`stage_check` 的真实口径）

* **AR**：两次在场且逐 token 一致 · 自报 hash/长度/停止自洽 · 停止合法。
  **新 family 另有逐条核对**（见 CF2 表）：独立 hash / `end_state.gen_hash` / 实际长度 /
  `end_state.gen_len` / `smoke.stop_reason` / `end_state.stop_reason`，**缺字段与相互矛盾分类阻断**。
  **旧 family**（F13/F01/F02/F08/F10）仍用**登记输出锚点 + 上一轮产物**的原判据。
* **production**：两次一致 · `checks_passed` · cache 结构 · 逐轮不变量 · 有限性 · token 在词表 ·
  停止合法 · `s>1` batched 几何 · 非 OOM。**与 AR 的分歧只作 `[待办·不阻断]`**（不判失败、不阻断 serial）。
* **serial**：18 项硬检查（含 D6）· 逐轮字段 · trace 一致 · `stage=="complete"` · `self_checks_all_ok` ·
  与同 family AR **逐 token 相等** · 两次一致。

## 3. GB6 记账：字段来源（谁提供、怎么核）

| 字段 | 来源 | 备注 |
| --- | --- | --- |
| `first_divergence_index` / `common_prefix_len` | **运行侧**（AR vs prod 产物逐 token 比） | 输出序列内下标 |
| `ar_token` / `prod_token` | 运行侧；诊断侧若也记则**必须一致**（不一致 ⇒ 拒） | CF3c |
| `ar_len` / `prod_len` | 运行侧 | |
| `absolute_position` | 运行侧 = `end_state.prompt_len + 下标` | 缺 `prompt_len` ⇒ 该字段判缺 ⇒ 阻断（CF1） |
| `round_index` | 运行侧：**divergent（production）侧**的 `smoke.rounds` 推出 | AR 产物没有 `rounds`（真实实测） |
| `ar_work` | 运行侧：`decode_steps`（`end_state.n_decode_steps`）+ `target_gen_calls`；`acceptance=not_applicable` | **不填造 AR 接受率** |
| `prod_work` | 运行侧：`specdec_stats` 的 `rounds/proposed/accepted/…` | 真实 F13：`25/97/46`，rollbacks 18 |
| `ar_top1` / `ar_top2` / `margin` / `tied_at_max` / `cache_len_after` | **独立诊断 pass**（GPU，AR 口径重建决策行） | 现有产物**推不出**；缺 ⇒ 不 PASS |
| `position_origin` / `position` / `prompt_len` / `family` / `runs_tag` | 诊断**来源/配置**字段 | 必须自证"同 tag、同 family 的 AR vs production 首分歧" |

**核对规则（缺一即 `CHECK_NEEDS_REVIEW`）**：family 一致 · tag 一致（**禁止**跨 tag 迁移旧 margin）·
来源字段齐全 · `position_origin` 含 `first_divergence` · `position` 与产物推出的首分歧下标一致 ·
`prompt_len` 与产物一致 · 两侧 `token` 一致。**诊断与正式计时分离**：诊断产物不进入任何性能结论。

## 4. 复制即用命令

### 4.0 预检（只读；现在就可执行）
```powershell
python -X utf8 -m specdec.engine.stage2b_preflight_cpu                     # 期望：exit 3（待 D-3）
python -X utf8 -m specdec.engine.verify_family_independence_cpu            # 期望：31/0
python -X utf8 -m specdec.engine.serial_oracle_env_check                   # 期望：ENVIRONMENT_OK
```

### 4.1 逐 family：AR×2 → production×2 → serial×2（**GPU；需 D-1/D-3**）
> **编号覆盖（2026-10-04 收紧）**：`stage_check` 要求每个请求模式**恰好** `rep1..rep2` 各一份；
> 两个 `rep1` / `rep1+rep3` / 同一产物计入两次 / 缺号 / `rep` 非法值 ⇒ 一律 **STOP**（不再只看份数），
> 且非法值不会再让工具崩溃。它保证的是"**记录层**编号正确"，**不能**证明 GPU 真的独立跑了两次。

```powershell
$tag='div8209_p2'; $reps=2; $py='.\.venv\Scripts\python.exe'
foreach ($f in 'F09','F03','F11','F04') {
  & .\scripts\so_batch.ps1 -Tag $tag -Cases $f -Modes ar -Reps 1,2
  & $py -X utf8 -m specdec.engine.stage_check --tag $tag --family $f --modes ar --reps $reps
  if ($LASTEXITCODE -ne 0) { throw "$f AR 未通过（rc=$LASTEXITCODE）" }
  & .\scripts\so_batch.ps1 -Tag $tag -Cases $f -Modes prod -Reps 1,2
  & $py -X utf8 -m specdec.engine.stage_check --tag $tag --family $f --modes ar,prod --reps $reps
  if ($LASTEXITCODE -ne 0) { throw "$f production 未通过（rc=$LASTEXITCODE）" }
  & .\scripts\so_batch.ps1 -Tag $tag -Cases $f -Modes serial -Reps 1,2
  & $py -X utf8 -m specdec.engine.stage_check --tag $tag --family $f --modes ar,prod,serial --reps $reps
  if ($LASTEXITCODE -ne 0) { throw "$f serial 未通过（rc=$LASTEXITCODE）" }
}
```
**停止条件**：任一阶段 rc ≠ 0 立即停（`stage_check` 会点名是哪一项硬门/哪个字段）。
production 与 AR 的分歧只打印 `[待办·不阻断]`，**不**阻断 serial。

### 4.2 GB6 诊断 pass（**仅当某 family 的 production 与 AR 分歧**；GPU）
```powershell
$env:PM_FAMILIES='F09,F03,F11,F04'      # 支持 env 参数化
$env:PM_RUNS_TAG='div8209_p2'           # ★ 必须等于 Stage 2b 的 tag（否则判决工具拒收）
python -X utf8 -m specdec.engine.correctness_protocol_margins --stage2b --runs-tag div8209_p2 --tag div8209_p2 --families F09,F03,F11,F04
```
* **预期产物**：`output/g5/attn/div8209_p2_margins_<utc>.json`（顶层 `runs_tag`，`families` 下每 family 一条，
  含 `position_origin`/`position`/`prompt_len`/`pair`/`config_checks`）。
* **停止条件**：
  * 退出码 **4**（有 family 配对被拒，如缺 `prod` 产物 / tag 不符 / 配置不符）⇒ **停下核对**，
    不得手工填数、不得从旧 tag 迁移 margin；
  * 某 family 状态为 `TOKEN_EQUAL` ⇒ 该 family **不需要**诊断（无分歧），如实记录即可；
  * 退出码 5 ⇒ 检查失败（载入输入长度与 `prompt_len` 不符等），同样停。
* **⚠️ 覆盖范围限制（2026-10-04 字段审计结论）**：本工具**只重建 AR 口径**的决策行 ⇒ 它的
  `ar_top1/ar_top2/margin/tied_at_max` 与 `cache_len_after` **属于 AR 侧**（`cache_len_after` 还是**重建**出来的
  cache 长度，不是该 run 的实际长度）。**它不能给出 serial / production 的实际决策行** ⇒
  **不得**用 AR 单侧 logits 推断另一侧是否并列；精确并列的双侧分类在补证（决策单 §1A-T-6 的 P-1/P-2）落地前
  **一律记 TIE-C（证据不足）**。诊断产物**不进入**任何计时/加速比结论。

### 4.3 最终判决（只读）
```powershell
python -X utf8 -m specdec.engine.stage2b_verdict --tag div8209_p2 `
  --families F09,F03,F11,F04 --modes ar,prod,serial --reps 2 `
  --gb6 output/g5/attn/div8209_p2_margins_<utc>.json
```
**停止条件**：判决 ≠ `SERIAL_ORACLE_PASS`；或 exit 2（参数非法，此时不会写任何产物）。
`SERIAL_ORACLE_FAIL` = 真实硬门失败；`BASELINE_REGRESSION` = AR 基线可评估且不成立；
`CHECK_NEEDS_REVIEW` = 证据不足/记录矛盾/记账不全（含本文 §0 的三个漏洞）。

### 4.4 FP8 档（**待 D-6**）
```powershell
$env:SO_KV='v_fp8'
& .\scripts\so_batch.ps1 -Tag div8209_p2fp8 -Cases F13 -Modes ar -Reps 1     # 结构 smoke
```
**禁止**把 BF16 五-family 的 serial PASS 外推为 FP8 PASS。

## 5. 仍待批准 / 阻断（工具就绪 ≠ 获准开跑）

| # | 事项 | 状态 |
| --- | --- | --- |
| **D-1** | exact-id 门**重新分层**到 serial oracle | **待批准**（阻断全部后续） |
| **D-3** | 登记 `F09/F03/F11/F04` 进**run 级**授权 | **待批准**（preflight 返回 3） |
| **D-4** | 采用 DR1–DR4（删除 `index==0` 自动阈值；全量记录、分层解释） | **待批准** |
| D-2 / D-5 / D-6 / D-7 / D-8 / D-9 | 见 brief §3 | **待批准** |
| E-1（上轮发现） | 诊断工具模式名不匹配 ⇒ 已**参数化 + 拒绝式退化**；但 GPU 分支**未验证**（无 GPU 运行） | **代码已改，待 GPU 轮验证** |
| — | Stage 2b 24 次运行（+ 可能的诊断 pass） | **待运行**（依赖 D-1/D-3） |
| — | FP8-0 / FP8-A | **待运行**（依赖 D-6） |
| — | pilot / 正式矩阵 | **阻断** |

## 6. 自测与真实数据证据（本轮实测，全部 exit 0）

| 自测 | 结果 |
| --- | --- |
| `stage2b_verdict_selftest_cpu` | **92/0**（上一轮 64 → 本轮 +28：CF1/CF1b–d/CF2/CF2b–e/CF3/CF3b–d，CE4f 改为"缺 prompt_len ⇒ 拒"两条） |
| `gb6_diag_select_selftest_cpu`（新） | **25/0**（T1 正常配对、T2 **缺 prod 拒绝**、T3–T5 来源、T6–T12 配置、T13 无分歧不记位置、T14 legacy 标注） |
| `serial_oracle_aggregate_selftest_cpu` | **21/0** |
| `verify_family_independence_cpu` | **31/0**（新登记 `gb6_diag_select.py` + 其自测；run 级授权**未**放宽） |
| `serial_oracle_audit_selftest_cpu` / `env_check` / `stage0` | **19/0** · **24/0** · **42/0** |
| 真实数据复核（只读） | 新配对逻辑在 `div8209_so`/F13 上给出**首分歧 47**（旧逻辑 63）；缺 prod 时 `MISSING_ARTIFACT`；真实 F13 运行侧 GB6 字段**全齐**，仅缺 5 个诊断字段 |

## 7. 本轮修改文件

**新增**：`specdec/engine/gb6_diag_select.py`（纯 CPU 配对/来源核对）·
`specdec/engine/gb6_diag_select_selftest_cpu.py`（25 项自测）。
**修改**：`specdec/engine/stage2b_verdict.py`（CF1/CF2/CF3 消费端修复 + `_ar_work` + `_gb6_present`
+ 诊断来源/一致性核对 + `end_state_n_decode_steps` 暴露）·
`specdec/engine/correctness_protocol_margins.py`（模式名参数化 / `--stage2b` / 拒绝式退化 / 退出码 4）·
`specdec/engine/stage2b_verdict_selftest_cpu.py`（+31 项）·
`specdec/engine/verify_family_independence_cpu.py`（登记新消费方）· 本文件。

**未修改（按边界要求）**：`model_self.py` / `spec_dec.py` 的任何数值路径 · proposal ·
历史 FAIL/CHECK/PASS 结论与既有原始产物 · 冻结判据文件。**未运行 GPU，未批准协议，未外推 FP8。**
