# 验证准备清单（阶段 1 之前的待批准材料）— 2026-09-29

> **状态：准备清单，未获批准，不授权运行。** 本文只固定候选配置、选定输入、列出 16 次运行清单、
> 审计现有脚本的记录能力并给出最小修改方案，以及分类/停止/待确认事项。
> **本轮未启动 GPU、未修改任何工具、未修改正式正确性门槛、未开展正式矩阵。**
> 分层验收（关键决策建议书的路线 B）**仍待小组批准**。

## 0. 本文的证据来源（全部已核）

| 项 | 值 | 出处 |
| --- | --- | --- |
| 基线 commit | `5c2a1e4` | `git log` |
| family manifest | `docs/prompt_families_manifest.json`，`v1-draft`，16 families | 本文件 |
| manifest 可复现性 | **16 family × (2048/4096/8192 前缀 + text) = 64/64 hash 全部复现** | `family_independence_cpu_check.json` |
| 调试输入 | `wiki.test.raw` 前 600k 字符的前 8192 token，hash **`bf6a34a82018fd39`**（本轮**已再次复现**） | 同上 |
| 独立性 | 16/16 family 与调试输入 **零 32-gram 交集**；候选取自不同 block 且彼此零交集；除 builder 外**无任何 runner 读过 manifest** | 同上 |

CPU 自测：`python -m specdec.engine.verify_family_independence_cpu` → **PASS 12 / FAIL 0**
（工具为 `specdec/engine/verify_family_independence_cpu.py`，只读 zip/manifest，不加载模型、不碰 CUDA）。

## 1. 固定候选配置（**本轮不改变任何默认实现**）

| 维度 | 固定值 | 说明 |
| --- | --- | --- |
| commit | `5c2a1e4` | 运行时另记 `git.dirty_worktree`；运行前须为 clean |
| attention backend | `cudnn_prefill`（**仅 prefill 走融合 cuDNN**，其余阶段 eager） | 由 `M.set_attention_strategy("cudnn_prefill")` 显式设定；`assert_strategy_used()` 必须 ok；`fused_by_phase` **只允许含 `prefill`** |
| 计算 dtype | **bf16**（权重 bf16；无 fp32 权重、无全模型 fp32） | 与历史 G1–G5 一致 |
| 诊断精度开关 | **全部关闭**：`set_diag_preset("eager")` = `{qk_fp32: False, prob_fp32: False}`（记录键名在现有工具里是 `diag_precision_baseline`，另配 `precision_scope_effective` 与 `qk_scope_report` 佐证"未开干预"） | **不启用 QK32**；不新增/不堆叠精度开关 |
| target KV | `base_bf16` 与 `v_fp8` 两档 | 引擎名分别是 `fp16` / `v_fp8`（`spec_dec.FMT`）；v_fp8 = K bf16 + V fp8 e4m3 group=128 |
| draft KV | 恒 `base_bf16`（引擎名 `fp16`） | draft head_dim=64 < GROUP_V=128 ⇒ v_fp8 对其**结构上不可用** |
| ctx / chunk | **8192** / **2048** | 禁改（不改 chunk 以规避历史不可比） |
| k | **4** | 固定链 |
| 生成长度 | **max_new = 64** | 不得为提前停止而缩短 |
| 解码 | **greedy**（`argmax`），无采样、无温度 | |
| EOS 规则 | `eos_id = tokenizer.eos_token_id`（Qwen2.5 → 151645）；命中即停并记 `stop_reason="eos"` **与实际长度** | 未命中且未达 64 ⇒ `stop_reason="other_short"` ⇒ **判失败** |
| 内存护栏 | 沿用默认（commit ≥ 16 GiB 线），**不提高** | 记录 `commit_guard.overridden` |
| 环境门 | P-state `qualified=True` | 不合格则 `stage=refused` 且不加载 |

## 2. 输入：两个未参与此前调试的 8K family

**选择依据**：manifest 取自 `wiki.train.raw`，而此前**全部**诊断（含 8K 分歧、2×2、QK32 干预、
缓存历史探针）都用 `wiki.test.raw` 的同一段前缀 ⇒ **split 级独立**。

⚠️ **口径更正（不要把扫描说成"从未使用"的证明）**：我做的扫描只能说明"**在当前代码库里，
除 builder 之外没有 runner 读取这个 manifest**"（区分大小写的文件内容扫描，覆盖 `specdec/engine/*` 与 `scripts/*`）。
这**不是**"这 16 个 family 从未被任何人使用过"的证明：扫描不覆盖仓库外的手工运行、不覆盖按 token 内容
（而非按 manifest 字段）取用同一段语料的做法，也不覆盖历史提交之外的环境。因此正确表述是：
**没有发现会消费它们的现有 runner ∨ 选样只用到 manifest 自身字段**，而不是"保证从未使用"。

⚠️ **与决策建议书的偏差（需确认）**：建议书举例"自然文本与代码"，但本 manifest 的 stratum 只有
`S1_low_entropy / S3_prose / S4_mixed / S2_high_markup`，**没有代码档**。故改取
**该 manifest 自身设计轴（见下方"rank 来源核实"）上差异最大**的两档，且**分属不同 block**：

| | Family A | Family B |
| --- | --- | --- |
| family_id | **F06** | **F13** |
| stratum | `S3_prose`（先验 rank 2） | `S2_high_markup`（先验 rank 4） |
| source | `wikitext-2-raw-v1/wiki.train.raw` | 同左 |
| block / window_index | **block 0** / win 5 | **block 3** / win 3 |
| 截取位置 `window_token_start` | **40960**（= 5 × 8192） | **24576**（= 3 × 8192） |
| 窗口长度 | 8192 token | 8192 token |
| 8192 前缀 token hash | **`71e630c6cd702180`** | **`9a278731d744a859`** |
| 2048 / 4096 前缀 hash | `dde4b84095b7cb2d` / `0f1567932074438c` | `f846a2d2c2e59aa9` / `66e1c61d836d57a7` |
| text_sha1 | `e896ef193068dc2b` | `941bbcc6e4284397` |
| 独立性 | 与调试输入零 32-gram 交集；与 F13 零交集 | 同左 |

**取用规则（必须写进运行记录）**：`ids = tokenize(block_text)[window_token_start : window_token_start+8192]`，
`block_text = raw_train[i*300000 : (i+1)*300000]`，tokenizer `Qwen/Qwen2.5-1.5B-Instruct`，
**显式固定**（不再依赖库默认漂移，见 §2b）；**ctx=8192 即取窗口全长的前 8192 token**。

### 2b. rank 来源核实 + tokenizer/EOS 显式固定

**`expected_acceptance_rank` 的来源（已核实到 builder 源码）**：它不是实测接受率，而是一条**先验**：
1. `scripts/build_prompt_families.py` 按 **`distinct_token_ratio`** 分位把候选切成
   `S1_low_entropy / S3_prose / S4_mixed / S2_high_markup` 四层（`assign(by_distinct, ...)`）；
2. builder 里**硬编码** `rank = {S1:1, S3:2, S4:3, S2:4}`，并写成 manifest 的 `expected_acceptance_rank`；
3. manifest 自己写着 `preregistered_hypothesis: "接受率排序 S1 > S3 > S4 > S2（pilot 证伪）"`。
⇒ 所以 F06(2) 与 F13(4) 的差距是**先验排序上的差距**，**不是**"F06 的接受率实测高于 F13"。
若 pilot 证伪该假设，按 G3 的 pilot-driven amendment 流程重新分层。

**显式固定的输入与 EOS 配置**（避免"默认值变了而无人察觉"）：

| 项 | 值 | 说明 |
| --- | --- | --- |
| tokenizer id | `Qwen/Qwen2.5-1.5B-Instruct` | 与 manifest 的 `tokenizer` 字段一致（已断言） |
| `add_special_tokens` | **显式 `True`** | builder 用库默认；这里把它写进代码（`family_input.TOKENIZER_ADD_SPECIAL_TOKENS`），并作为 hash 校验的一部分 |
| tokenizer `revision` | **未钉（`None`）** | 本地缓存未记录 commit hash ⇒ **明确记为"revision 未钉"**，不假装已钉；若日后要求可复现性，应钉 revision 并重跑 hash 校验 |
| EOS | `tokenizer.eos_token_id`，期望 **151645** | 运行时核对 `matches_expected`；EOS 规则见 §1 |
| 语料条目名 | manifest 写 `…-raw-v1/wiki.train.raw`，zip 真实条目是 `…-raw/wiki.train.raw` | **字符串不一致但同一语料**；按 basename 解析并**在记录里写明差异**（不计入 hash 校验） |

## 3. 16 次运行清单

`2 families × 2 KV × {AR, Spec} × 2 次独立进程重复 = 16`

| # | family | KV | mode | 重复 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 1–2 | F06 | base_bf16 | AR | r1, r2 | AR 为 **target-only**（不加载 draft） |
| 3–4 | F06 | base_bf16 | Spec | r1, r2 | 配对条件：同 family/KV 的 target 配置与 AR 完全一致 |
| 5–6 | F06 | v_fp8 | AR | r1, r2 | |
| 7–8 | F06 | v_fp8 | Spec | r1, r2 | |
| 9–10 | F13 | base_bf16 | AR | r1, r2 | |
| 11–12 | F13 | base_bf16 | Spec | r1, r2 | |
| 13–14 | F13 | v_fp8 | AR | r1, r2 | |
| 15–16 | F13 | v_fp8 | Spec | r1, r2 | |

**配对纪律**：AR 与 Spec 的 target 侧配置（backend/dtype/KV/ctx/chunk/k/max_new/EOS/精度开关）
必须逐项相同；差异只允许是"AR 单步解码" vs "SpecDec 块验证 + 修复"。重复运行必须**独立进程**、
串行、同配置、**不共享 cache 历史**（每跑从头 prefill）。

## 4. 现有脚本记录能力审计（含缺项与最小修改方案）

审计对象：`g5_8k_smoke_cudnn.py`（单次运行的记录主体）与 `scripts/run_g5_8k_repeat.ps1`（多步编排 + 配对比较）。

| 需求 | 现状 | 结论 |
| --- | --- | --- |
| 完整生成 token | ✅ `run.gen`、`gen_len`、`stop_reason`、`gen_ids_in_vocab` | 够 |
| finite | ✅ `checks`（逐阶段逐模型）+ `decision_coverage` + `non_finite` 事件 + `stage_*` 逐阶段落盘 | 够 |
| 显存 | ✅ `mem_timeline`：8 个时点，每点含 `pt_allocated_mib` / `pt_peak_allocated_mib` / `pt_peak_reserved_mib` / `cuda_free_mib` + 设备侧占用 + `gpu_consumers_before` | 够 |
| 失败原因 | ✅ `stage` / `traceback` / `oom` / `non_finite` / `alloc_on_oom` / `verdict` | 基本够；建议补一个显式 `failure_class`（见 G5） |
| 首个分歧（AR vs Spec） | ⚠️ **run JSON 里没有**；只有 `run_g5_8k_repeat.ps1` 在 master TSV 里算 `pair_correctness` / `common_prefix` / `first_divergence` | **部分**：**且该 runner 一旦分歧就 `exit 1` 停止**（G2） |
| 重复一致性 | ❌ 无任何字段或比较 | **缺**（G3） |
| 缓存结构检查 | ⚠️ 只用 `cache_length` 做 AR 不变量与 `round_invariants`（`tgt_len == drf_len == pre + (m+1 或 Kp)`）；**未调用** `spec_cache.check_consistent`（跨层 token 维一致） | **缺**（G4） |
| 输入身份 | ⚠️ 只有 `input.ids_hash`（且**只能**用写死的 `test.raw` 那一段） | **缺**（G1/G5） |

### 缺项 → **已实现**（全部显式启用；默认行为保持不变）

> 状态：**已实现并已 CPU 自测**（本轮）。**未启动 GPU**，因此"实测可跑通"仍未验证。

| 编号 | 状态 | 开关 | 默认行为 |
| --- | --- | --- | --- |
| **G1** manifest 输入 | **已实现** | `SMOKE_FAMILY=F06/F13`（+ `SMOKE_MANIFEST` 可选） | **未设 ⇒ mode=legacy**，走原写死的 `test.raw` 前缀，逐字段不变；hash 不符**拒绝运行** |
| **G2** 分歧后继续 | **已实现** | `REPEAT_ON_DIVERGENCE=record` | **未设 ⇒ stop**（原行为：分歧即 `exit 1`）；`record` 时记录并继续 |
| **G3** 重复一致性 | **已实现** | 跟随 `families16` 计划自动比较 | 非该计划时不做比较（原行为） |
| **G4** 跨层 cache 结构检查 | **已实现** | `REPEAT_STRUCT_CHECK=1` → 传 `SMOKE_STRUCT_CHECK=1` | **未设 ⇒ 不检查**，`cache_structure_ok=None` ⇒ 状态为 `not_checked`（**不得当通过**） |
| **G5** 溯源与失败分类 | **已实现** | 随 G1 自动写 `input.family`；`failure_class` 始终写出 | 追加字段，不改计算 |
| **新** 五类状态分开 | **已实现** | 始终写出，**无开关** | 只追加记录字段，不改计算；无任何"总体 PASS"字段 |
| **新** 缺失运行汇总 | **已实现** | `SUMMARY_INDEX=<index.json> python -m specdec.engine.family16_summary` | 只读；缺格 ⇒ `diagnostic_completeness != complete` 且退出码 3 |

**实现位置**
- `specdec/engine/family_input.py`（G1 + 16 格清单单一定义 + EOS/tokenizer 显式配置）
- `specdec/engine/status_fields.py`（**五类状态分开**；`formal_gate_status` 结构上只能是 `not_established`/`fail`）
- `specdec/engine/family16_summary.py`（G3/汇总；**不产生** `passed`/`success` 字段；退出码 0/3 语义见下）
- `g5_8k_smoke_cudnn.py`（G1 接输入、G4 结构检查、G5/状态字段；均显式启用）
- `scripts/run_g5_8k_repeat.ps1`（`families16` 计划、G2、G3、索引 JSON 输出）

**退出码语义（汇总器）**：`0` = 16 格齐全且结构与重复自洽（**仍不是正式正确性通过**）；
`3` = 报告已写出但诊断不齐全/不可宣告。**刻意不用 0 表示"正式通过"**。

### 原始缺项与最小修改方案（**历史记录，已被上表取代；保留以便追溯**）

| 编号 | 缺项（**当时状态**） | 最小修改（**当时计划；现已按上表实施**） |
| --- | --- | --- |
| **G1** | （**当时**）工具**无法输入 manifest family**：`P5V.load_ids()` 把 `test.raw` 前缀写死 ⇒ 清单当时不可执行。**现已实现**（`SMOKE_FAMILY`，默认关闭） | 新增**只读**的输入解析：`SMOKE_FAMILY=F06`（+ `SMOKE_MANIFEST`）时，按第 2 节的取用规则重建 ids，并在 `input` 里写 `source/block/window_token_start/token_sha1/ids_hash`；未设该变量时**行为与现在逐位一致**（回归保护） |
| **G2** | 编排器**遇到 AR/Spec 分歧就停**，会让 16 次清单在第 2 步中断 | 加一个显式开关（如 `REPEAT_ON_DIVERGENCE=record`）：分歧时**记录并继续**，仅在**硬失败**时停（硬失败定义见 §5）。默认仍为 `stop`，避免改变历史语义 |
| **G3** | 无重复一致性字段 | 复用**已有的纯函数** `g5_8k_qk32_analysis.first_divergence(a,b)`（已单测）比较 r1/r2 的 `run.gen`，写入 `repeat_consistency = {identical, first_diff_index, a_token, b_token}`；两次不同 ⇒ 标 **待查** 并停止推进（按建议书 §6 阶段 1） |
| **G4** | 未做跨层 cache 结构检查 | 在 prefill 之后与 decode 结束各调一次 `spec_cache.check_consistent(cache)`（引擎自带、纯读取），把异常／`None` 结果记入 `cache_structure_ok`；不新增自造口径 |
| **G5** | 输入来源与失败分类不够显式 | 在 `run` 里加 `failure_class ∈ {none, env, load, finite, structure, coverage, length, oom, crash}`，由已有字段派生；`input` 里加 `family_id` / `window_token_start` / `token_sha1`（G1 一并完成） |

> G1 的回归保护很重要：新分支必须做到"未指定 family 时，输出与改动前逐字段一致"，
> 才能避免把"支持新输入"变成"悄悄改了历史口径"。

## 5. 结果分类（三类，互不替代）

| 类别 | 判据（全部可机检） | 处置 |
| --- | --- | --- |
| **硬失败** | OOM / 原生崩溃 / 护栏拒绝 / 环境不合格；权重逐位校验失败；**NaN 或 Inf**；必需检查缺失（契约 v2 `all_required_ok=False`）；结构不变量失败（跨层 cache 长度不一致、`round_invariants.all_ok=False`）；`stop_reason="other_short"`；同配置重复**不一致** | **落盘 + 停止**；不重试、不换参数、不挑成功样本 |
| **数值差异待评估** | 结构 / 有限性 / 覆盖 / 长度全部通过，但同 family 同 KV 的 AR 与 Spec 的 token 序列不同 | **记录并继续**（须在获准路线下）：记 `common_prefix`、`first_divergence` 位置与两侧 token。**不自动归因为浮点误差**、**不自动豁免**、**不自动标"可生产使用"** |
| **性能观察** | `pt_*` 峰值、设备侧占用、各阶段计时 | **仅作观察记录**；含插桩的计时**禁止**进入任何 speedup 声明；不同轨迹会改变接受率与工作量，属解释限制 |

**特别纪律（对应决策建议书 §7）**：
- **不得**把本轮"不同缓存历史下的 single-vs-single"差异直接填入 `classify_bf16_compare.own_max_abs`
  —— 该字段的文档定义是"同位置 **block-vs-serial**"；填错语义正是"缺证据不得豁免"要防的事。
- 因此**未达标**的分歧一律停在"**待评估**"，并**逐项列出缺哪些输入**，不追求一个标签而启动昂贵实验。
- `QK FP32 ≠ 全模型 FP32`；把已有 BF16 缓存转 FP32 **不等于**从共同前缀重算高精度历史 ⇒
  两者都**不得**含糊标为"fp32 真值"。

## 6. 运行命令（**预登记；获准后才执行**）

单次运行（与现有工具同一入口，仅增加 family 输入 —— 依赖 G1）：

单次（如需手工跑某一格；`SMOKE_FAMILY` 一旦设置即走 manifest，hash 不符直接拒绝）：

```bat
REM F06 / base_bf16 / AR base（生成 64，启用结构检查）
cmd /c "cd /d D:\RealDSH\specdec && set \"SMOKE_MODE=ar\" && set \"SMOKE_KV=base_bf16\" && ^
 set \"SMOKE_MAXNEW=64\" && set \"SMOKE_FAMILY=F06\" && set \"SMOKE_STRUCT_CHECK=1\" && ^
 set \"REPEAT_ON_DIVERGENCE=record\" && scripts\run_g5_attn_forensics.cmd smoke"
```

编排（16 次，串行、独立进程；依赖 G2）：

**准确命令（16 次；`REPEAT_MAXNEW=64` 是该计划的强制前置，缺省直接 `exit 2` 拒绝）**：

```powershell
$env:REPEAT_PLAN          = 'families16'
$env:REPEAT_MAXNEW        = '64'      # 强制：不设就拒绝运行（避免误用默认 16 把实验做短）
$env:REPEAT_ON_DIVERGENCE = 'record'  # AR/Spec 分歧记录并继续（默认 stop = 原行为）
$env:REPEAT_STRUCT_CHECK  = '1'       # G4 跨层 cache 结构检查
$env:REPEAT_TAG           = 'fam16_' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_g5_8k_repeat.ps1
```

**先做零成本预演（不跑 GPU，只采样环境并写出索引）**：

```powershell
$env:REPEAT_DRYRUN='1'; $env:REPEAT_PLAN='families16'; $env:REPEAT_MAXNEW='64'; $env:REPEAT_TAG='dryrun_fam16'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_g5_8k_repeat.ps1
```

**跑完（或中途停止）后汇总**（缺格/重复不一致会如实反映，退出码 3）：

```powershell
$env:SUMMARY_INDEX='output\g5\attn\fam16_<UTC>.index.json'
python -m specdec.engine.family16_summary
```

**纪律**：不并行加载模型；不开自动重试循环；不改 chunk/k/dtype/backend；不提高内存护栏。

## 7. 预计资源开销（基于已完成运行的**实测**值，不是新测量）

| KV / mode | peak allocated | peak reserved | 备注 |
| --- | --- | --- | --- |
| base_bf16 / AR | **5261.651 MiB** | **5722.0 MiB** | 单项已实测（`qk32c_*` control/base_bf16/ar） |
| base_bf16 / Spec | **6906.38 MiB** | **7332.0 MiB** | 双模型驻留 |
| v_fp8 / AR | **5203.401 MiB** | **5588.0 MiB** | |
| v_fp8 / Spec | **6847.63 MiB** | **7178.0 MiB** | |

- 记录事实：四格 `pt_peak_reserved` 分别 5722.0 / 7332.0 / 5588.0 / 7178.0 MiB，均低于 8188 MiB 的
  物理容量（`pt_reserved_within_physical_line`）。
- ⚠️ **口径更正**：**不得用 `reserved` 反推"物理显存还剩多少"**。`reserved` 是 PyTorch 分配器的
  保留量，`reserved ≤ 物理容量` 只是**记账事实**，既不等于整卡占用，也不能推出安全余量
  （当时还有 20+ 个 C+G 消费者；同刻的 `cuda_free` / `smi_mem_used` 才是设备侧观测量）。
  ⇒ 结论只保留："不得并行、不得叠加额外占显存的动作"，理由是 **Spec 档本身已到 GB 级高位**
  且同刻设备侧占用可观测，**不是**由 `reserved` 算出的"余量"。
- commit 可用量近期实测 ≈ **29–34 GiB** ≥ 16 GiB 线；`overridden` 应为 `False`。
- **墙钟**：串行 16 次 × 约 **40–50 s**（含加载）≈ **11–14 分钟**（无重试前提下）。依据：`qk32c` 的
  run_id 时间戳间隔（base AR `023252` → base Spec `023322` → v_fp8 AR `023404` → v_fp8 Spec `023439` …）。
- **磁盘**：每份 run JSON 约 **106–179 KB**；现有 `.rows.pt`（只存决策行摘要，**不是**全 logits）
  约 **4–23 KB**。若本阶段改为转存**完整 logits**（AR 每步一行 ≈ 151936 × 4 B），
  则每份 AR 约 **39 MB**、每份 Spec 约 **数 MB** —— 这是 §9 第 7 项待确认的资源影响。

## 8. 停止条件

**立即停止并落盘（不重试、不挑样本）**：
1. OOM / 原生崩溃 / 被护栏拒绝 / 环境门不合格；
2. 出现 **NaN 或 Inf**；
3. 必需检查缺失（`all_required_ok=False`）或决策覆盖核对失败；
4. 结构不变量失败（跨层 cache 长度不一致、`round_invariants.all_ok=False`）；
5. `stop_reason="other_short"` 或生成长度 ≠ 64 且非 EOS；
6. **同配置两次独立重复输出不同** ⇒ 标 **待查** 并在推进前停止；
7. 权重逐位校验不一致。

**不**作为停止条件（但必须记录）：AR 与 Spec 的 token 差异本身 —— 在获准的分层报告路线下它是**观测对象**，
不是错误；但**也不**因此豁免（见 §5）。若航线未获准，则按现行冻结判据处理（B6 exact-id 门）。

## 9. 待确认事项（**需小组/团队裁决，本文不代为决定**）

| # | 待确认 | 不确认时的状态 |
| --- | --- | --- |
| 1 | 是否批准评估**路线 B**（分层正确性与有限精度行为报告），并接受"不再无条件声称生产 BF16 逐 token 复现指定 AR"的声明限制 | 保持现行冻结判据；**不放行**；本轮不跑 |
| 2 | 是否授权**阶段 1**这 16 次独立 smoke（在 manifest 与运行规则审阅之后） | **不跑 GPU** |
| 3 | 是否接受 §2 的**family 选择**（manifest 无代码档 ⇒ 改取**先验 rank** 差异最大且跨 block 的 F06/F13） | 输入未定 |
| 4 | 是否允许为支持 manifest family 而修改 runner（G1/G2）——这是**工具改动**，需先批准 | 现有工具**无法**按第 3 节清单运行 |
| 5 | 阶段 1 的重复一致性若不一致，按"**硬失败**"还是按"**待查并停止推进**"记录 | 建议后者（与建议书 §6 一致）；需确认 |
| 6 | `max_new=64` 与 EOS 政策在本阶段是否即为最终口径 | 若正式实验用更长输出，本最小测试**不替代**对应长度回归 |
| 7 | 是否要求把全 logits 转存纳入本阶段（影响磁盘与后续可复核性） | 默认**不转存**，只存 token 序列与决策行摘要 |

## 10. 本轮**未**做的事（边界声明）

- （**写于工具实现之前**）**未**启动任何 GPU 运行；**未**跑 16 次清单（当时清单尚不可执行，见 G1/G2）。
  ⚠️ **该句已过期**：G1–G5 已于 commit `bfc75ff` 实现并通过 CPU 自测，清单**已可执行**；
  本条保留为历史记录，不再代表当前状态。
- **未**修改任何工具代码、manifest、默认 backend/精度、验收门槛或内存护栏；
- **未**采用、**未**批准决策建议书的任何路线；协议变更事项**未**生效；
- **未**把任何旧 FAIL / CHECK 改写，也**未**新增判据；
- 唯一新增产物是本清单与一个**只读** CPU 自测脚本。

**关联**：`docs/production_correctness_decision_2026-09-29.md`（建议稿，待批准）；
`docs/cudnn_prefill_stage_report.md` §9.19–§9.21；`docs/data_ledger.md` §4.32–§4.34；
`docs/prompt_families_manifest.json`（`v1-draft`）。
