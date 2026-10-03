# Stage 2b 运行手册（2026-10-04，工具加固轮修订）

> **本轮判决（一句话，两条问题分别回答）**
>
> 1. **工具能否放行 Stage 2b？—— 不能。** 两个独立原因：
>    * **治理（阻断）**：`D-1`（exact-id 门重新分层）与 `D-3`（登记 `F09/F03/F11/F04`）**仍未批准**；
>      preflight 因此返回 `3`，任何 run 现在开跑都会被守卫正确判越界。
>    * **可执行性（阻断，本轮新发现 E-1）**：唯一能产出 GB6 诊断字段的生产者
>      `specdec/engine/correctness_protocol_margins.py` 仍按**旧模式名** `aligned`/`spec` 取产物
>      （`:136`），而 Stage 2b 的模式名是 `prod`/`serial` ⇒ 取不到 divergent 侧产物时
>      `position` 会**静默退化**为"最后一个 token"（`:142`）。本轮的 GB6 位置交叉核对会把这种
>      记录**判为不一致并拒绝**（安全侧），但在 E-1 修好之前，一旦某 family 出现分歧，
>      判决只能停在 `CHECK_NEEDS_REVIEW`——**这是正确行为，不是放行**。
> 2. **协议是否仍阻断 GPU 运行？—— 是。** `D-1/D-3/D-4/D-5/D-6` 均待批准；
>    **本轮未运行任何 GPU 任务**，未批准协议，未启动 pilot/正式实验，未外推 FP8 结论。
>
> 本轮做了两件事：**(a) 修掉判决工具的 5 处虚假 PASS / 阶段放行漏洞并各自留下回归自测；
> (b) 修掉自查中又发现的 3 处同类漏洞。** 上一版本手册顶部那句"工具已具备可执行性"
> 是**未经验证的结论，已删除**（当时 `stage2b_verdict` 的 GB6 记账分支因"读到空 gen"而**恒不触发**，
> 见 §0 的 C-1）。
>
> 复现证据（全部纯 CPU、只读产物）：`stage2b_verdict_selftest_cpu` **64/0** ·
> `serial_oracle_aggregate_selftest_cpu` **21/0** · `verify_family_independence_cpu` **31/0** ·
> `serial_oracle_audit_selftest_cpu` **19/0** · `serial_oracle_env_check_selftest_cpu` **24/0** ·
> `serial_oracle_stage0` **42/0**（合计 **201 项，0 失败**）·
> `stage2b_preflight_cpu` → `PREFLIGHT_OK_TOOLS_BUT_FAMILIES_UNREGISTERED (exit 3)`。

---

## 0. 本轮修掉的漏洞（每一项：反例 → 根因 → 修复位置 → 复现与修复后结果）

| # | 反例 / 症状 | 根因 | 修复位置 | 修复后复现结果 |
| --- | --- | --- | --- | --- |
| **CE1** | `--tag definitely_missing_stage2b_tag --families , …` 曾返回 **exit 0 + `SERIAL_ORACLE_PASS`** | family 集合为空 ⇒ `期望三元组 = 0` ⇒ "覆盖性检查"空转 ⇒ 无任何 needs_review ⇒ 直落 PASS 分支 | `stage2b_verdict.validate_arguments()`（tag 非空、family 非空/无重复/形如 `F\d{2}`、模式**精确**等于 `ar,prod,serial`、reps==预登记值 2）；`_cli()` 在**评估之前**校验并以 **exit 2** 返回、**不写任何产物**；`evaluate()` 内部**再校验一次**（双保险，防绕过 CLI） | exit **2**，打印 `[非法] family 集合为空（不允许：会导致『期望三元组=0 ⇒ 空泛 PASS』）`；落盘目录文件数 **0**（实测临时目录为空） |
| **CE2** | 只有 `F13 AR×2` + `--modes ar` 曾判 PASS（prod/serial 各 0 次） | 期望三元组只按传入的模式集合展开 ⇒ "没跑的模式"从不进入核对 | 同 `validate_arguments()`（模式必须精确三元素）+ `evaluate()` 中**独立于三元组的逐模式存在性检查** | `--modes ar` ⇒ exit **2**「模式必须精确包含 `['ar','prod','serial']`」；`--modes ar,prod,serial` 但目录内只有 AR ⇒ `CHECK_NEEDS_REVIEW`「prod/serial 模式没有任何可用产物 ⇒ 该阶段未执行」 |
| **CE3** | `stage_check --tag div8209_so --family F13 …` 曾返回 **OK(exit 0)**，而该 F13 的 serial 产物是 `complete_with_selfcheck_failures` 且缺 D6 清理后读数 | 首版核对器对 serial 只做了"存在性 + token 等价"，**没有真正跑它所声称的 18 项硬检查**（`require_d6=True` 从未生效） | `stage_check._gate_serial()`（18 项硬检查 + `_rounds` 完整性 + trace 一致性 + `stage=="complete"` + `self_checks_all_ok` + 与同 family AR 的**逐 token 相等** + 两次重复一致） | exit **1 / STOP（7 项失败/缺字段）**，含 `18_d6_mode_restored_after_cleanup`（清理后实测 `serial`、运行结束时 `None`）、`stage=complete_with_selfcheck_failures`、`self_checks_all_ok != True`。**历史 F13 的聚合 JSON 未被改写**（§2 末尾说明） |
| **CE4** | production 与 AR 分歧时，生产分歧记录**缺 v2 GB6 必需字段**却仍可 PASS | 分歧只写"首分歧下标"，未区分"产物可直接取得"与"需独立诊断 pass"的字段；也没有"缺字段 ⇒ 不得 PASS"的通道 | `stage2b_verdict.GB6_FROM_RUNS` / `GB6_FROM_DIAGNOSTIC` 两分；`_gb6_record()`；`evaluate()` 的 `gb6_needs_review` 通道 + `--gb6` 链接 | 无诊断 ⇒ `CHECK_NEEDS_REVIEW`「GB6 必需字段未齐全（需独立诊断 pass）：`['ar_top1','ar_top2','cache_len_after','margin','tied_at_max']` ⇒ 不得判 PASS」。真实 F13 的分歧现在**如实**记录为 `first_divergence_index=47`、`ar_token=6702`、`prod_token=6652`、`absolute_position=8239`、`round_index=18`（取自 prod 侧） |
| **CE5** | 每条"环境拒绝"产物都被塞进 `needs_review`，即使有效重复齐全 | 环境拒绝与"证据缺失"混为一类 | `evaluate()` 把环境拒绝单列为 `env_blocked_records` / `n_env_blocked`（`env_blocked_policy` 字段写明规则）；仅当其**造成必需重复缺失**时由覆盖性检查阻断 | 实测：tag 内 3 条 F13 AR 环境拒绝 ⇒ 单独列出、`needs_review` 中**没有**环境噪声；有效重复齐全时不影响判决；造成缺重复时仍 `CHECK_NEEDS_REVIEW`（用例 K） |
| **C-1** | （自查发现）**GB6 记账分支恒不触发** —— 我加的 CE4 回归第一次跑就抓到 | `_gb6_record` 直接读 runner 顶层 `rec["gen"]`，而真实产物**没有顶层 `gen`**（`_row` 从 `smoke.gen` 取）⇒ `gen=None` ⇒ 判不出分歧 ⇒ `divergent=False` ⇒ 分歧从不记账却走 PASS 路径 | 新增 `_gb6_run_view()`：`gen`/`rounds`/`specdec_stats` 一律走与聚合器唯一访问器 `_row` **同一路径**（`smoke.*`）；`prompt_len` 取 `smoke.end_state.prompt_len` | 分歧被真正识别；CE4/CE4a–CE4l 全部通过 |
| **C-2** | （自查发现）**诊断记录的 tag 核对形同虚设** | 真实诊断产物（`div8209_proto_margins_*.json`）把 `runs_tag` 放在**文件顶层**，family 条目里没有 ⇒ 首版把 `diag_tag=None` 当成"无需核对"而放行 ⇒ 旧配置 margin 可被静默采用 | `load_gb6_diagnostics()` 把顶层 tag **下推**到每条 family 记录；`_gb6_record()` 改为**没有可核对 tag 即拒绝**（不容忍"默认同配置"） | CE4h（无 tag ⇒ 拒绝 + 理由"没有可核对的 tag"）、CE4i（文件式顶层 tag ⇒ 下推后 PASS）、CE4j（`runs_tag=div8209_proto` ⇒ 拒绝且理由含"禁止跨配置迁移"）全部通过 |
| **C-3** | （自查发现）**位置交叉核对坐标系自相矛盾** | 生产者 `correctness_protocol_margins.py:142` 记的 `position` 是**输出序列内下标**（`pos=cp`；历史 F01: `position=14, prompt_len=8192`），首版却拿它与 `prompt_len + d`（绝对位置）比较 ⇒ 既可能假拒绝、又给了"绝对位置"一个没有生产者支撑的口径 | `_gb6_record()` 统一按**输出下标**与 `common_prefix_len` 比较（`position_frame="output_index"`）；另加 `prompt_len` 一致性核对（不同输入配置 ⇒ 拒绝） | CE4d/CE4g（位置不符 ⇒ 不 PASS）、CE4f（无 `prompt_len` 变体仍 PASS）、CE4k（`prompt_len` 不符 ⇒ 不 PASS）全部通过 |
| **C-4** | （自查发现）**决策轮号恒缺**：真实 F13 上 `round_index = null` | AR 模式产物**没有** `smoke.rounds`（AR 无投机轮），而首版只从 AR 侧推导 ⇒ GB6 必需字段"决策轮号"永远缺失 | `_gb6_record()` 改以 **divergent（production）侧的 rounds** 推出决策轮号，并同时留下 `round_index_prod` / `round_index_ar` / `round_index_source` | 真实 F13：`round_index=18`（`source=prod`，`round_index_ar=None`）；CE4l（人为删掉 AR 的 `rounds`）通过 |
| **C-5** | （自查发现）`--outdir` 对**落盘位置**无效 | `_cli()` 写产物时写死 `OUTDIR`，读却用 `--outdir` ⇒ "读 A 目录、写到默认目录" | `_cli()` 改为写 `Path(a.outdir)` | 与 `evaluate(outdir=...)` 读取口径一致 |

**判决优先级（本轮明确，任何一条都不会放行 PASS）**：
① 参数非法 / 覆盖性不全 / AR 基线无法评估（缺字段）⇒ `CHECK_NEEDS_REVIEW`；
② AR 基线**可评估且不成立** ⇒ `BASELINE_REGRESSION`；③ serial 硬检查 / token 等价 / production 结构失败 ⇒ `SERIAL_ORACLE_FAIL`；
④ **GB6 分歧记账不全** ⇒ `CHECK_NEEDS_REVIEW`；⑤ 全通过 ⇒ `SERIAL_ORACLE_PASS`。
> ④ 故意排在 ②③ 之后：不得因"记账没做"把**真实的基线回归 / serial 失败**降级为"证据不足"（自测 F 抓到过这种降级）。

## 1. 工具清单（全部受版本控制）

| 工具 | 用途 | 退出码 |
| --- | --- | --- |
| `python -X utf8 -m specdec.engine.stage2b_preflight_cpu` | 开跑前只读预检：family 来源/hash/独立性/守卫登记/工具就绪 | `0` 可跑 · `3` 工具就绪但 family **未登记** · `1` 硬问题 |
| `python -X utf8 -m specdec.engine.stage_check --tag … --family … --modes … --reps …` | **逐阶段硬门核对**（AR → production → serial；真的跑 18 项 serial 硬检查） | `0` 通过 · `1` STOP（失败/缺字段）· `2` 尚有阶段未跑（PENDING）· **`3` 参数非法** |
| `python -X utf8 -m specdec.engine.stage2b_verdict --tag … --families … --modes … --reps … [--gb6 …]` | **Stage 2b 判决**（参数/覆盖/基线/分歧记账四道门） | `0` PASS · `1` 其它判决 · **`2` 参数非法（不落盘）** |
| `python -X utf8 -m specdec.engine.stage2b_verdict_selftest_cpu` | 判决工具自测（合成产物在临时目录） | `0` PASS（64 项） |
| `python -X utf8 -m specdec.engine.serial_oracle_aggregate_selftest_cpu` | 聚合器字段名契约自测（含新工具） | `0` PASS（21 项） |
| `python -X utf8 -m specdec.engine.verify_family_independence_cpu` | 治理守卫（family 授权范围 + 工具源级扫描） | `0` PASS（31 项） |
| `python -X utf8 -m specdec.engine.serial_oracle_env_check` | 环境门（16 GiB 线 + 18 GiB 入口门槛） | `0` ENVIRONMENT_OK |
| `.\scripts\so_batch.ps1 -Tag … -Cases … -Modes … -Reps …` | 顺序批量运行（**GPU**；一个进程 = 一个三元组） | 逐 run 的 rc 记入日志 |

**已删除的本机临时脚本**：`output/g5/attn/_so_stage_check.py`（由受版本控制的 `specdec/engine/stage_check.py` 取代）。

## 2. 各阶段实际执行的硬门（`stage_check` 的真实口径，不是声称）

* **AR 阶段**：两次重复在场 · 两次逐 token 一致 · 自报 hash/长度与 `end_state` 一致 · 停止原因合法且一致 · 新 family
  必须另有 manifest 输入 `token_sha1` 与实际输入一致（**不得**套用旧 family 的登记输出锚点）。
* **production 阶段**：两次在场且逐 token 一致 · `checks_passed` · cache 结构 ok 且检查数 > 0 · 逐轮不变量 ok ·
  有限性 · token 在词表内 · 停止合法 · **存在 `s>1` 的 batched 几何** · 非 OOM ·
  **与 AR 的分歧只作"待办提示"（`[待办·不阻断]`），既不判失败也不阻断 serial 阶段** ——
  历史五个 family 的 production 都与 AR 分歧，那是被记录的现象；serial 的逐 token 等价才是核心证据，
  不能因为分歧就不跑 serial。分歧的**记账**要求由最终判决工具执行（缺 GB6 诊断 ⇒ 不放行）。
* **serial 阶段**：**18 项**硬检查（`_serial_hard_checks2(require_d6=True)`，含 D6"清理后模式已复位"）·
  `_rounds` 字段完整 · trace 与轮记录一致 · `stage == "complete"` · `self_checks_all_ok == True` ·
  与**同 family 的 AR** 逐 token 完全相等 · 两次 serial 一致。

> **历史记录与"更严门"的关系（必须说清，且不得改写历史）**：
> `div8209_so` 的 F13 在**当时 16 项**硬检查口径下的 Stage-2 `SERIAL_ORACLE_PASS` **仍然是历史事实**，
> 本轮**没有**改写它（`output/g5/attn/div8209_so_stage2_aggregate_20261003T170149Z.json` 保持原样；
> 旧的 16 项口径自测 `--scope final` 现在仍复现 `SERIAL_ORACLE_PASS`，见该工具新时间戳产物）。
> 但新增的第 18 项（D6）在 F13 上**不成立** ⇒ 新工具对同一批产物给出 `SERIAL_ORACLE_FAIL`。
> 两者**不是矛盾**：前者是"当时判据下的历史结论"，后者是"更严判据下的当前结论"；
> 正因如此，Stage 2b 的 serial 必须**重新跑**（历史产物不能充当新判据的证据）。

## 3. 复制即用命令（按顺序；每条都标注前置 / 预期产物 / 停止条件）

### 3.0 预检（只读；**现在就应执行**）
```powershell
python -X utf8 -m specdec.engine.stage2b_preflight_cpu
python -X utf8 -m specdec.engine.verify_family_independence_cpu
python -X utf8 -m specdec.engine.serial_oracle_env_check
```
* **前置**：工作树 clean。
* **预期产物**：仅 stdout（不落盘）。
* **停止条件**：preflight 返回 `1` 或 `3`；守卫 `FAIL`；环境非 `ENVIRONMENT_OK`。
  当前：preflight = `3`（**待 D-3**）、守卫 = **31/0**、环境门最近一次 `ENVIRONMENT_OK`。

### 3.1 每个 family 的"AR×2 → production×2 → serial×2"（**GPU 阶段；本轮未执行**）
```powershell
$tag = 'div8209_p2'; $reps = 2; $py = '.\.venv\Scripts\python.exe'
foreach ($f in 'F09','F03','F11','F04') {
  & .\scripts\so_batch.ps1 -Tag $tag -Cases $f -Modes ar -Reps 1,2
  & $py -X utf8 -m specdec.engine.stage_check --tag $tag --family $f --modes ar --reps $reps
  if ($LASTEXITCODE -ne 0) { throw "$f AR 阶段未通过（rc=$LASTEXITCODE），停止" }

  & .\scripts\so_batch.ps1 -Tag $tag -Cases $f -Modes prod -Reps 1,2
  & $py -X utf8 -m specdec.engine.stage_check --tag $tag --family $f --modes ar,prod --reps $reps
  if ($LASTEXITCODE -ne 0) { throw "$f production 阶段未通过（rc=$LASTEXITCODE），停止" }

  & .\scripts\so_batch.ps1 -Tag $tag -Cases $f -Modes serial -Reps 1,2
  & $py -X utf8 -m specdec.engine.stage_check --tag $tag --family $f --modes ar,prod,serial --reps $reps
  if ($LASTEXITCODE -ne 0) { throw "$f serial 阶段未通过（rc=$LASTEXITCODE），停止" }
}
```
* **前置**：3.0 全部通过（**尤其 D-3 已批准 ⇒ preflight 返回 0**）。
* **预期产物**：每三元组 `{tag}_{family}_{mode}_rep{rep}_{runid}.json` + `.pt`；
  每次 `so_batch` 一份 `{tag}_{family}_{mode}_rep{rep}.log` 与一份 `{tag}_batch_*.log`；
  共 **4 family × 3 mode × 2 rep = 24 个 JSON**（+24 个 `.pt`）。
* **停止条件**：`stage_check` 任一阶段 rc ≠ 0（**逐阶段硬停**，不得"先跑完再一起看"）。

### 3.2 GB6 独立诊断 pass（**仅当某 family 的 production 与 AR 分歧时需要**）
```powershell
# ⚠️ 前置：E-1 必须先修（见文首）—— 现在这个工具按旧模式名 aligned/spec 取产物，
#    对 Stage 2b 的 prod/serial 取不到 divergent 侧序列 ⇒ position 会静默退化成最后一个 token。
$env:PM_FAMILIES = 'F09,F03,F11,F04'      # 支持 env 参数化（默认 F01,F02,F08,F10）
$env:PM_RUNS_TAG = 'div8209_p2'            # ★ 必须等于 Stage 2b 的 tag，否则判决工具会拒收
python -X utf8 -m specdec.engine.correctness_protocol_margins
```
* **预期产物**：`output/g5/attn/div8209_p2_margins_<utc>.json`（顶层含 `runs_tag`，`families` 下每 family 一条）。
* **该产物必须提供的 GB6 诊断字段**：`ar_top1` · `ar_top2` · `margin` · `tied_at_max` · `cache_len_after`
  **外加**坐标核对用的 `position`（**输出序列内下标**口径）与 `prompt_len`。
* **链接方式**（判决时显式给出，工具不会自动找）：
  ```powershell
  python -X utf8 -m specdec.engine.stage2b_verdict --tag div8209_p2 `
    --families F09,F03,F11,F04 --modes ar,prod,serial --reps 2 `
    --gb6 output/g5/attn/div8209_p2_margins_<utc>.json
  ```
* **判决工具的核对规则（缺一即 `CHECK_NEEDS_REVIEW`）**：family 必须一致 · tag 必须一致（**禁止**用
  `div8209_proto` 等旧配置的 margin 迁移）· `position` 必须与产物推出的首分歧下标一致 ·
  `prompt_len` 必须与产物一致。
* **停止条件**：任一必需诊断字段缺失 / family 或 tag 不符 / 位置或 `prompt_len` 不符 ⇒ **停在
  `CHECK_NEEDS_REVIEW`**，不得手填、不得估算、不得从上一轮产物迁移 margin。

### 3.3 最终判决（只读；24 次跑完 + 需要的诊断 pass 之后）
```powershell
python -X utf8 -m specdec.engine.stage2b_verdict `
  --tag div8209_p2 --families F09,F03,F11,F04 --modes ar,prod,serial --reps 2 `
  --gb6 output/g5/attn/div8209_p2_margins_<utc>.json
```
* **前置**：3.1 四个 family 三阶段全过；（若分歧）3.2 的诊断产物齐全。
* **预期产物**：`output/g5/attn/div8209_p2_stage2b_verdict_<utc>.json` + `.jsonl`（新时间戳，不覆盖历史；
  落盘位置跟随 `--outdir`）。
* **停止条件**：判决 ≠ `SERIAL_ORACLE_PASS`；或 CLI 以 **exit 2** 返回（参数非法，此时**不会有**产物）。

### 3.4 FP8 档（**待 D-6；另起 tag，且先做 1 次结构 smoke**）
```powershell
$env:SO_KV = 'v_fp8'
& .\scripts\so_batch.ps1 -Tag div8209_p2fp8 -Cases F13 -Modes ar -Reps 1     # smoke：确认布局/结构字段可记录
# 若 smoke 通过且 D-6 批准：F13 × {ar,prod,serial} × 2 = 6 次
```
* **前置**：D-6 批准 + `v_fp8` 结构 smoke 通过（需新增 S2/S3/S4 结构检查，见审计 §4.2）。
* **禁止**：把 BF16 五-family 的 serial PASS 外推为 FP8 PASS。

## 4. 仍待批准 / 待运行 / 阻断

| # | 事项 | 状态 |
| --- | --- | --- |
| **E-1** | `correctness_protocol_margins.py` 的模式名仍为 `aligned`/`spec`（`:136`），Stage 2b 为 `prod`/`serial` ⇒ 分歧位置静默退化（`:142`）；需参数化 divergent 模式名并加一条 CPU 可测的取产物自测 | **本轮新发现，未修**（属 GPU 侧工具的语义改动，需与 3.2 同轮做并验证）；**阻断 GB6 记账** |
| D-1 | exact-id 门**重新分层**到 serial oracle（执行路径的实质调整） | **待批准**（阻断全部后续） |
| D-3 | 登记 `F09/F03/F11/F04` 进守卫授权范围 | **待批准**（阻断 Stage 2b；preflight 现返回 `3`） |
| D-4 | 采用 DR1–DR4（**删除** `index==0` 自动阈值；全量记录 + 分层解释，不按分歧位置筛样本） | **待批准**（已写入 v2 §5.2 与 brief §4，标注"待批准"） |
| D-2 / D-5 / D-6 / D-7 / D-8 / D-9 | 见 brief §3 | **待批准** |
| — | Stage 2b 的 24 次运行（+ 可能的诊断 pass） | **待运行**（依赖 D-1/D-3 与 E-1） |
| — | FP8-0 / FP8-A（7–13 次） | **待运行**（依赖 D-6） |
| — | pilot / 正式矩阵（C1–C6、A1–A2） | **阻断** |
| — | 守卫的**源级扫描**扩到新工具 | **已完成**：`stage2b_verdict.py` / `stage2b_preflight_cpu.py` / `stage2b_verdict_selftest_cpu.py` 已登记为 opt-in 消费方；新增 `STAGE2B_CANDIDATE_FAMILIES`（候选集合，显式注明"**不构成 run 级授权**"）与两条新检查 ⇒ 守卫 **31/0**。`stage_check.py` **故意不登记**（检测器扫不到它，而"消费方集合 == OPT_IN_CONSUMERS"是等式检查）。**run 级授权未放宽**：新检查要求"候选 family 尚未被任何 run 消费" |

## 5. 本轮修改文件清单与自测汇总

**本轮改动（全部为工具/文档/CPU 自测；未触碰数值路径）**：
`specdec/engine/stage2b_verdict.py`（§0 的 CE1/CE2/CE4/CE5 与 C-1…C-5 修复）·
`specdec/engine/stage2b_verdict_selftest_cpu.py`（27 → **64** 项：新增 CE1、CE2/CE2b、CE3、CE4a–CE4l、CE5）·
`specdec/engine/stage_check.py`（真实执行各阶段硬门；serial 走 18 项含 D6；STOP/PENDING/OK 三态）·
`docs/stage2b_runbook_2026-10-04.md`（本文件：删除未经验证的"可执行性"结论，补 GB6 诊断 pass 的运行/链接/停止条件）。

**上一轮已新增（本次沿用）**：`stage2b_preflight_cpu.py` ·
`specdec/engine/verify_family_independence_cpu.py`（守卫登记与候选集合）·
`specdec/engine/serial_oracle_aggregate.py`（legacy 反静默：越界 family ⇒ NEEDS_REVIEW）·
`serial_oracle_aggregate_selftest_cpu.py`（字段名契约扫描扩到新工具 ⇒ 21/0）·
`docs/specdec_correctness_protocol_v2_2026-10-04.md`（DR1–DR4）·
`docs/protocol_decision_brief_2026-10-04.md`（删 R3）·
`docs/stage2b_executability_audit_2026-10-04.md`（§3 追加修复说明）。

**自测汇总（本轮实测，全部 exit 0）**：
`stage2b_verdict_selftest_cpu` **64/0** · `serial_oracle_aggregate_selftest_cpu` **21/0** ·
`verify_family_independence_cpu` **31/0** · `serial_oracle_audit_selftest_cpu` **19/0** ·
`serial_oracle_env_check_selftest_cpu` **24/0** · `serial_oracle_stage0` **42/0** ⇒ 合计 **201/0**。

**未修改**（按边界要求）：`model_self.py` / `spec_dec.py` 的任何数值路径 · proposal（`output/pdf/*.tex`）·
历史 FAIL/CHECK/PASS 结论与既有原始产物 · 冻结判据文件（`design_freeze_g4_specdec.md`、`g5_measurement_protocol.md`）。
