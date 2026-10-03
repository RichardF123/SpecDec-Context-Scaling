# Stage 2b 运行手册（2026-10-04）：命令、前置、预期产物、停止条件

> **本轮判决（一句话）**：
> **Stage 2b 工具已具备可执行性**（tag/family/mode/reps 全部显式、缺一即 `CHECK_NEEDS_REVIEW`、
> CPU 自测 27/0 证明"不可能因漏读/错读数据而虚假 PASS"）；
> **但 GPU 阶段仍被协议批准阻断**——`D-1`（exact-id 门重新分层）与 `D-3`（登记 F09/F03/F11/F04）
> 尚未批准，`D-5/D-6` 亦未定；代码就绪 ≠ 获准开跑。**本轮未运行任何 GPU 任务。**
>
> 复现证据（全部纯 CPU、只读）：`serial_oracle_aggregate_selftest_cpu` **21/0**、
> `stage2b_verdict_selftest_cpu` **27/0**、`stage2b_preflight_cpu` → **PREFLIGHT_OK_TOOLS_BUT_FAMILIES_UNREGISTERED (exit 3)**、
> `verify_family_independence_cpu` **29/0**。

---

## 1. 工具清单（全部受版本控制；不再依赖 `output/` 里的临时脚本）

| 工具 | 用途 | 退出码 |
| --- | --- | --- |
| `python -X utf8 -m specdec.engine.stage2b_preflight_cpu` | 开跑前只读预检：family 来源/hash/独立性/守卫登记/工具就绪 | `0` 可跑 · `3` 工具就绪但 family **未登记** · `1` 硬问题 |
| `python -X utf8 -m specdec.engine.stage_check --tag … --family … --modes … --reps …` | **逐阶段核对**（AR → production → serial），参数化 | `0` 通过 · `2` 尚有阶段未跑（PENDING）· `1` 不一致/缺字段 |
| `python -X utf8 -m specdec.engine.stage2b_verdict --tag … --families … --modes … --reps …` | **Stage 2b 判决**（显式集合；缺一即 NEEDS_REVIEW） | `0` PASS · `1` 其它判决 |
| `python -X utf8 -m specdec.engine.stage2b_verdict_selftest_cpu` | 判决工具自测（合成产物在临时目录） | `0` PASS |
| `python -X utf8 -m specdec.engine.serial_oracle_aggregate_selftest_cpu` | 聚合器字段名契约自测（含新工具） | `0` PASS |
| `python -X utf8 -m specdec.engine.verify_family_independence_cpu` | 治理守卫（family 授权范围） | `0` PASS |
| `python -X utf8 -m specdec.engine.serial_oracle_env_check` | 环境门（16 GiB 线 + 18 GiB 入口门槛） | `0` ENVIRONMENT_OK |
| `.\scripts\so_batch.ps1 -Tag … -Cases … -Modes … -Reps …` | 顺序批量运行（**GPU**；一个进程 = 一个三元组） | 逐 run 的 rc 记入日志 |

**已删除的本机临时脚本**：`output/g5/attn/_so_stage_check.py`（其功能已由受版本控制的
`specdec/engine/stage_check.py` 取代）；`output/g5/attn/_newfam_audit.py`、`_cmd_audit.py` 的复核能力
已分别并入 `stage2b_preflight_cpu.py` 与本手册第 4 节。

## 2. 三项阻断的修复与证据

| # | 原阻断（审计 §3.2） | 修复 | 证据 |
| --- | --- | --- | --- |
| **B-1** | 聚合器 tag 硬编码 ⇒ `_load_runs('div8209_p2')` 读到 **0** 条 | 新增 `stage2b_verdict`：`--tag` **必填**，且只读该 tag（`load_runs` 不回退）；`--no-write` 可只打印 | 自测 **A**：错误 tag ⇒ 读到 0 条、报缺 24 个三元组、判决 `CHECK_NEEDS_REVIEW` |
| **B-2** | family 集合硬编码 ⇒ 新 family 被**静默忽略** | `stage2b_verdict` 的 `--families/--modes/--reps` 全部必填；期望三元组逐一点名核对；**legacy** 路径增加"越界 family ⇒ NEEDS_REVIEW"反静默提示 | 自测 **H**（完整数据 24/24 ⇒ 四 family 全部纳入判决、PASS）· **B/C/D/J**（缺 family、缺重复、重复编号冲突、越界 ⇒ 均 `CHECK_NEEDS_REVIEW`） |
| **B-3** | 逐阶段核对器硬编码 tag 且在未跟踪的 `output/` | 新增受版本控制的 `specdec/engine/stage_check.py`（`--tag/--family/--modes/--reps` 全部显式） | 真实数据：`stage_check --tag div8209_so --family F13 --modes ar,prod,serial --reps 2` ⇒ **OK(exit 0)**；`--tag div8209_p2 --family F09` ⇒ **PENDING(exit 2)** |

**关键安全性质（自测直接证明）**：错误 tag / 缺 family / 缺重复 / 重复编号冲突 / 环境拒绝造成缺重复
⇒ **一律 `CHECK_NEEDS_REVIEW`，绝不 `SERIAL_ORACLE_PASS`**；
**新 family 不得套用旧 family 的登记输出 hash**（自测 F：F13 锚点不符 ⇒ `BASELINE_REGRESSION`；
自测 E：新 family AR 两次不一致 ⇒ `BASELINE_REGRESSION`，且**不**被误记为 serial 失败）；
**production 与 AR 分歧但结构正常 ⇒ 不判失败**（自测 G：仍 PASS，并记为 `NUMERICAL_DIVERGENCE_RECORDED`，首分歧位置在案）。

## 3. 复制即用命令（按顺序；每条都标注前置 / 预期产物 / 停止条件）

### 3.0 预检（只读；**现在就应执行**）
```powershell
python -X utf8 -m specdec.engine.stage2b_preflight_cpu
python -X utf8 -m specdec.engine.verify_family_independence_cpu
python -X utf8 -m specdec.engine.serial_oracle_env_check
```
* **前置**：工作树 clean。
* **预期产物**：仅 stdout（不落盘）。
* **停止条件**：preflight 返回 `1`（硬问题）或 `3`（family 未登记）；守卫 `FAIL`；环境非 `ENVIRONMENT_OK`。
  当前状态：preflight = `3`（**待 D-3**）、守卫 = 29/0、环境门最近一次 `ENVIRONMENT_OK`。

### 3.1 每个 family 的"AR×2 → production×2 → serial×2"（**GPU 阶段；本轮不执行**）
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
* **前置**：3.0 全部通过（**特别是 D-3 已批准 ⇒ preflight 返回 0**）。
* **预期产物**：每个三元组 `{tag}_{family}_{mode}_rep{rep}_{runid}.json` + `.pt`；
  每次 `so_batch` 一份 `{tag}_{family}_{mode}_rep{rep}.log` 与一份 `{tag}_batch_*.log`。
  共 **4 family × 3 mode × 2 rep = 24 个 JSON**（+24 个 `.pt`）。
* **停止条件（逐阶段，硬）**：
  * **AR**：两次不一致（⇒ `BASELINE_REGRESSION`）· 非有限值 · `other_short` · 结构失败 · OOM · 字段缺失；
  * **AR 基线（新 family）**：两次自一致 **且** 自报 hash/长度/停止一致 **且** manifest 输入 `token_sha1` 一致；
  * **production**：两次不一致（不确定性）· 结构/有限性/停止失败 · 缺 `s>1` 的 batched 几何 · margin/分歧无法提取；
  * **serial**：任一 token 与 AR 不同（`ALGORITHMIC_GATE_FAIL` 类）· 18 项硬检查任一失败 · 审计空泛 ·
    **D6 清理后读数缺失** · 两次不一致。**不得**用 margin/ULP 豁免、**不得**转算子级定位。

### 3.2 最终判决（只读；24 次跑完后）
```powershell
python -X utf8 -m specdec.engine.stage2b_verdict `
  --tag div8209_p2 --families F09,F03,F11,F04 --modes ar,prod,serial --reps 2
```
* **前置**：3.1 全部 family 三个阶段都通过。
* **预期产物**：`output/g5/attn/div8209_p2_stage2b_verdict_<utc>.json` + `.jsonl`（**新时间戳，不覆盖历史**）。
* **停止条件**：判决 ≠ `SERIAL_ORACLE_PASS`（`CHECK_NEEDS_REVIEW` / `BASELINE_REGRESSION` / `SERIAL_ORACLE_FAIL`
  各自对应待查、基线、硬门三类原因，逐条打印）。

### 3.3 FP8 档（**待 D-6；另起 tag，且先做 1 次结构 smoke**）
```powershell
$env:SO_KV = 'v_fp8'
& .\scripts\so_batch.ps1 -Tag div8209_p2fp8 -Cases F13 -Modes ar -Reps 1     # smoke：先确认布局/结构字段可记录
# 之后（若 smoke 通过且 D-6 批准）：F13 × {ar,prod,serial} × 2 = 6 次
```
* **前置**：D-6 批准 + `v_fp8` 结构 smoke 通过（需新增 S2/S3/S4 结构检查，见审计 §4.2）。
* **禁止**：把 BF16 五-family 的 serial PASS 外推为 FP8 PASS。

## 4. 仍待批准 / 待运行 / 阻断

| # | 事项 | 状态 |
| --- | --- | --- |
| D-1 | exact-id 门**重新分层**到 serial oracle（是执行路径的实质调整，非"不降低门槛"） | **待批准**（阻断全部后续） |
| D-3 | 登记 `F09/F03/F11/F04` 进守卫授权范围 | **待批准**（阻断 Stage 2b；preflight 现返回 3） |
| D-4 | 采用 DR1–DR4（**删除** `index==0` 自动阈值；全量记录 + 分层解释，不按分歧位置筛样本） | **待批准**（已写入 v2 §5.2 与 brief §4，标注"待批准"） |
| D-2 / D-5 / D-6 / D-7 / D-8 / D-9 | 见 brief §3 | **待批准** |
| — | Stage 2b 的 24 次运行 | **待运行**（依赖 D-1/D-3） |
| — | FP8-0/FP8-A（7–13 次） | **待运行**（依赖 D-6） |
| — | pilot / 正式矩阵（C1–C6、A1–A2） | **阻断** |
| — | 守卫的**源级扫描**扩到新工具 | **已完成**：`stage2b_verdict.py` / `stage2b_preflight_cpu.py` / `stage2b_verdict_selftest_cpu.py` 已登记为 opt-in 消费方（它们确实读 manifest 字段）；**新增** `STAGE2B_CANDIDATE_FAMILIES`（候选集合，显式注明"**不构成 run 级授权**"）与两条新检查 ⇒ 守卫 **31/0**（原 29/0）。`stage_check.py` **故意不登记**：检测器扫不到它，而"消费方集合 == OPT_IN_CONSUMERS"是等式检查。<br>**run 级授权未放宽**：新检查明确要求"候选 family 尚未被任何 run 消费"，任何 run 使用它们仍判越界 |
| D-3 | 登记候选 family 进**run 级**授权范围（用于真正开跑） | **仍待批准**（阻断 Stage 2b；preflight 现返回 3） |

## 5. 本轮修改文件清单

**新增（受版本控制）**：`specdec/engine/stage2b_verdict.py`（判决工具）·
`specdec/engine/stage_check.py`（逐阶段核对器）· `specdec/engine/stage2b_preflight_cpu.py`（预检）·
`specdec/engine/stage2b_verdict_selftest_cpu.py`（自测）· 本文件。

**修改**：`specdec/engine/serial_oracle_aggregate.py`（legacy 路径**反静默**：越界 family ⇒ NEEDS_REVIEW）·
`specdec/engine/serial_oracle_aggregate_selftest_cpu.py`（字段名契约扫描扩到新工具，+3 项 ⇒ 21/0）·
`specdec/engine/verify_family_independence_cpu.py`（登记 3 个新消费方 + 新增候选集合与 2 条检查 ⇒ **31/0**）·
`docs/specdec_correctness_protocol_v2_2026-10-04.md`（§5.2 分歧规则改为 DR1–DR4，**删除** `index==0` 阈值；§9 D-4 同步）·
`docs/protocol_decision_brief_2026-10-04.md`（§4 **删除** R3 并写明不成立的理由）·
`docs/stage2b_executability_audit_2026-10-04.md`（§3 追加"三项阻断已修复"的更新说明，其余保留为当时记录）。

**删除**：`output/g5/attn/_so_stage_check.py`（被 `specdec/engine/stage_check.py` 取代的本机临时脚本）。

**未修改**（按边界要求）：`model_self.py` / `spec_dec.py` 的任何数值路径 · proposal（`output/pdf/*.tex`）·
历史 FAIL/CHECK/PASS 结论 · 任何既有原始产物 · 冻结判据文件（`design_freeze_g4_specdec.md`、`g5_measurement_protocol.md`）。
