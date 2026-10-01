# 勘误：`xb_probe` 的 HF 缓存克隆缺陷、无效归因的撤回与工具修正（2026-10-01）

> **本轮范围（严格）**：只改**诊断工具**、跑 **CPU 自测**、审阅**既有产物**、写**记录**，并给出**一份** C1 最小定位方案。
> **未做**：未重跑 8K、未重跑阶段 B、未做逐层/逐算子扫描、未做任何新精度实验、未跑正式矩阵。
> **未改**：生产模型、精度默认、后端、`proposal_freeze_2026-10-01.md`（保持冻结）、严格 AR/Spec 一致门。
> **B6 门未过 ⇒ 仍禁止 acceptance 测量 / pilot / final；正式正确性状态不变（未通过）。**

状态图例：**已验证通过** / **已实现未验证** / **设计完成未实现** / **本次新增** / **已撤回** / **判 INVALID**。

---

## 1. 缺陷本体（**已验证通过**）

### 1.1 代码事实（原来的实现）

`xb_probe.py` 的 HF 分支里有一个**手写克隆**：新建 `DynamicLayer` 后**只填 `keys`/`values`**
（本轮已删除该实现，改为共享克隆；错误版本保留为 `cache_clone.clone_cache_old_xb` 供回归测试）。

本机 `transformers 5.16.1` 的 `DynamicLayer.__dict__`
= `['device', 'dtype', 'is_initialized', 'keys', 'values']`，其中
**`is_initialized` 是普通布尔属性**（不是 property，也不会由 `keys` 反推）：

* 新实例 `is_initialized=False` ⇒ `get_seq_length()` **返回 0**；
* 第一次 `update()` 会**重新初始化该层并清空**已填入的 K/V。

### 1.2 影响机制（CPU 实测，`specdec.engine.cache_clone` 自测第 3、6 组）

| 检查（微型 HF，CPU） | 结果 |
| --- | --- |
| 源缓存长度 6 / `is_initialized=True` | 实测 |
| **旧克隆**后 `get_seq_length()` | **0**（应为 6） |
| **旧克隆**后 `is_initialized` | **False** |
| 旧克隆 + 追加 1 个 token 后长度 | **1**（应为 7） |
| 旧克隆 + 追加后前 6 位 K 是否保留 | **未保留** |
| 旧克隆的单步 logits vs 正确克隆 | **不同**（argmax 11 vs 30） |
| 旧克隆的单步 logits vs **完全没有历史**的单步 logits | **逐位相同** |

⇒ 那次运行的两臂**不是从 8240-token 历史出发**，而是"历史被清空"的前向。

### 1.3 修复（**本次新增**，`specdec/engine/cache_clone.py`）

统一改用 `copy.deepcopy`（保留完整对象状态，含 `is_initialized`/`device`/`dtype` 与各层元数据）：

* `clone_cache_full(cache)`（HF `DynamicCache`）、`clone_list_cache_full(cache)`（自研 `list[dict]`）；
* `require_clone_invariants(...)`（HF，8 项不变量）与 `require_list_clone_invariants(...)`（自研，7 项）；
* 错误版本保留为 `clone_cache_old_xb` / `clone_list_cache_old_xb`，**只用于回归测试**（证明新测试抓得住错法，不只测正例）。

`xb_probe.py` 现在把克隆不变量**写进产物**（`arm_clone_check`），并在 `XB_STAGE=ab` 时对 HF 侧用一次性探针
副本做"克隆 → 追加 → 长度 L+1、历史保留、源与兄弟副本不受影响"的现场断言。

---

## 2. 影响面：哪些结论失效、哪些不受影响

### 2.1 **判 INVALID**（已撤回，不得引用）

| # | 内容 | 原因 |
| --- | --- | --- |
| I1 | `xb` 阶段 B **跨实现**对照：OURS-Serial vs HF-Serial `max\|Δ\|` 12.54–16.49、OURS-Block vs HF-Block 12.58–16.48，j=1/j=2 argmax 不同 | HF 侧两臂**历史被清空**（缺陷 1.1/1.2） |
| I2 | 由 I1 派生的任何表述（"跨实现差异远大于同实现内差异"、"HF 与自研在 pre 之后分道扬镳"等） | 同上 |
| I3 | **归因"prefill 分块日程敏感性"**（"上一轮 HF 探针 2096 末块 ⇒ 日程会改变 HF 自身下游结果"） | **双重不成立**：① 旧 HF 探针的日程与本协议 S\* **相同**（源码 `for st in range(0, seq_len, CHUNK)` ⇒ 末块 **48**，五个 chunk），"2096 末块"是**无依据的说法**；② 两次 HF 行差异的真实来源是**克隆 bug**（见 §6） |
| I4 | 把 12–16 的下游差量当作**模型数值误差** | 同上 |

### 2.2 仍然 **VALID**（克隆发生**之前**的 prefill 检查点）

| 检查点 | 位级 | max\|Δ\| | mean\|Δ\| | 不同元素 | argmax OURS/HF | margin OURS/HF | K/V 位级 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **C1**（prompt 4×2048 分块之后，len 8192） | 否 | **0.1875** | 0.0416068 | 139637 | 220 / 220 | 2.25 / 2.125 | **1/56** |
| **C2**（+48 token 一个 chunk，len 8240） | 否 | **0.210938** | 0.0330954 | 130414 | 220 / 220 | 5.75 / 5.875 | **1/56** |

C1/C2 在克隆**之前**产生（prefill 阶段不使用任何克隆），因此不受该缺陷影响。

### 2.3 **不受影响**的工具与产物

| 对象 | 为什么不受影响 | 证据 |
| --- | --- | --- |
| `hf_serial_block_probe.py` 及其历史产物 | 它的 `clone_cache` 在 `type(c)()` 之后**额外拷贝了 `is_initialized`/`is_sliding`/`layer_idx`**，实测克隆 6-token 历史后长度 6 → 追加 1 个 token → 7、前 6 位 K 保留 | `cache_clone` 自测；本轮已把它切换为共享 deepcopy（**行为不变**），并重跑其短输入自测 8/0 |
| 引擎侧（自研 `list[dict]`）克隆 | 那次运行的层 dict **只含 `k`/`v` 两个张量键**，旧写法 `[{k: v.clone()}]` 在张量键上**是正确的**（新自测有正向回归证明）；错只在 HF 侧 | `cache_clone` 自测第 5 组 |
| `spec_verify_index_selftest_cpu.clone_cache` 的 `[dict(d)]` 浅拷贝 | 它由该自测自己的哨兵项验证过"引擎从不就地改写旧张量、只把新张量重新绑定到键上" | 该自测源码 |
| C1/C2 快照、F13 四格、UCG、`hfsb` 行级产物、`f13i50_*` | 均不经过 xb 的 HF 克隆 | — |

### 2.4 一处**过程错误**（如实记录）

本轮第一次重跑比较工具时，旧版 `xb_compare_cpu.py` 用**固定文件名**
`xb_20261001_compare_<ours>_<hf>.json`，因此**覆盖**了上一轮那份派生比较记录（`output/` 被 `.gitignore` 忽略，
无法从 git 恢复）。该文件内容**本就已判 INVALID**（I1），且两侧原始产物
（`xb_20261001_{ours,hf}_*.json/.pt`）完好，仍可一键重算。
**已修**：比较工具的输出名现在带**写入时间戳**并对已存在文件加序号，杜绝再次覆盖。

---

## 3. 撤回与更正（逐条：原话 → 状态 → 更正）

| 原表述（出处） | 状态 | 更正后 |
| --- | --- | --- |
| "本协议 C2 与上一轮 HF 探针的 **2096 末块**日程不同"（报告 §9.42.2、ledger §4.57） | **已撤回** | 旧探针日程与本协议**相同**：4×2048 + 末块 **48** |
| "**日程敏感性**：prefill 分块日程会改变 HF 自身下游结果"（报告 §9.42.4、ledger §4.57） | **已撤回** | 两次 HF 运行的日程相同、且两侧**初始历史的第 0 层 K 逐位相同**（§6）；行差异来自克隆 bug |
| "差异最早出现在 C1"（报告 §9.42.3） | **部分更正（措辞）** | 只能说 **"C1 是第一个已检查且发现差异的检查点"**；C1 之前没有任何已检查的检查点被比较过，"首次产生于 C1"不可用 |
| "跨实现阶段 B 存在 12–16 的差异"（报告 §9.42.3、ledger §4.57） | **判 INVALID** | HF 侧两臂空历史；不得引用，不得当作数值误差 |
| 微型自测"两边**都没有** qkv bias（config `attention_bias=False`）"（旧 `xb_probe._selftest`） | **已更正** | 见 §4.2：v5 **硬编码** `bias=True`；生产 checkpoint **本身有** 84 个 q/k/v bias 张量；自测已改为**映射** bias 并核查实际模块 |

---

## 4. 本轮工具改动（文件 / 函数 / 输入输出）

### 4.1 `specdec/engine/cache_clone.py`（**本次新增**）

* 输入：HF `DynamicCache` 或自研 `list[dict]`；输出：独立副本 + 不变量断言结果（dict）。
* 新增函数：`clone_cache_full`、`clone_list_cache_full`、`clone_cache_old_xb`、`clone_list_cache_old_xb`、
  `require_clone_invariants`、`require_list_clone_invariants`、`CloneError`、`require`。
* 自测：`python -m specdec.engine.cache_clone`（**PASS 24 / FAIL 0**），含**机制组**（坏克隆 == 无历史）
  与**两类错法**回归（HF 侧只填 keys/values；引擎侧浅拷贝/丢键）。

### 4.2 `specdec/engine/xb_probe.py`（修改）

* **克隆**：两侧统一 `clone_cache_full` / `clone_list_cache_full`；产物新增 `arm_clone_check`。
* **阶段闸门（`XB_STAGE`，默认 `c1`）**：`c1` 只跑 C1 落盘即停；`c2` 到 C2；`ab` 才跑两臂。
  产物记 `stage_gate.executed`，比较工具据此**拒绝**把"未执行"当"相同"。闸门**不自动**推进（单侧运行无法知道另一侧 C1 是否逐位相同）。
* **自测重写**（`XB_SELFTEST=1`，**PASS 11 / FAIL 0**），分三组：
  * **(a) 权重映射**：`seed` 在建模**之前**；**以实际模块为准**探测 bias（`cfg.attention_bias` 不作为依据）；
    `attn_bias_force=...` 后 `load_state_dict(strict=True)`；断言"引擎参数名集合 == 映射后键集合"、"逐张量位级一致 26/26"、
    "丢弃键 ⊆ {`lm_head.weight`}"、"bias 被**映射**而不是删除"；并注明**范围限制**（生产权重映射由 `lowmem_load` 的 338/338 单独证明）。
  * **(b) 数值对齐**：只**记录** `max|Δ|`/argmax（微型 bf16 CPU 实测 `max|Δ|=0`），**不设 PASS 阈值**。
  * **(c) 缓存结构与克隆**：布局一致；HF 侧 8 项不变量；两臂不共享存储；**旧克隆被回归抓出**；引擎侧不变量。
* **两条实测事实**（本轮核对，写入模块 docstring）：
  * `transformers 5.16.1` 的 `Qwen2Attention.__init__`（`modeling_qwen2.py:189`）**硬编码** `nn.Linear(..., bias=True)`，
    建模代码**不读** config 的 `attention_bias` ⇒ 从 config 建的 Qwen2 一律带 q/k/v bias；
  * 生产 checkpoint（`model.safetensors` 头）共 **338** 个张量，其中 **84** 个是 `layers.*.{q,k,v}_proj.bias`（28 层×3），**无** `lm_head.weight`
    ⇒ 338 = 254 + 84 ⇒ 引擎侧 `attn_bias=True` 才是与生产一致的配置。
  * 因此旧自测"删掉 bias 再断言两边都没 bias"不只是**自证**，还把微型模型做成了**与生产结构不同**的样子。

### 4.3 `specdec/engine/hf_serial_block_probe.py`（修改）

`clone_cache(c)` 改为 `return clone_cache_full(c)`（deepcopy），docstring 明确记录其**历史行为本来就正确**、
历史产物**不受**该 bug 影响。其余逻辑未动。

### 4.4 `specdec/engine/xb_compare_cpu.py`（修改）

* **阶段感知**：只比较两侧 `stage_gate.executed` 的**交集**；未执行记 `executed=False` 并打印"未执行"，
  **不参与**最早差异判定；旧产物（无 `stage_gate`）按存在键推断范围并标 `legacy_artifact`。
* **阶段 B 前置条件**：两侧都必须带 `arm_clone_check`，否则**拒绝**比较阶段 B。
* **协议断言**：`schedule`/`ctx`/`chunk`/`pre`/`tokens`/`family`/`precision` 逐项相等，否则报错退出。
* **措辞**：`earliest_difference_stage` 使用"C1 是**第一个已检查且发现差异的检查点**"。
* **不覆盖**：输出名带写入时间戳（见 §2.4）；`pred` 不再硬编码 `8264-24+j`。

### 4.5 `specdec/engine/xb_hfsb_row_recompute_cpu.py`（**本次新增**，只读）

用**既有产物**重算"同协议跨实现行对照"，三重前置校验（协议 / 历史位级 / 精度设置），任一不过即 `rc=6` 拒绝出结论。
详见 §6。

### 4.6 清理

删除临时脚本 `output/g5/attn/_scratch_cachecheck.py`（已验证不存在）。

---

## 5. CPU 回归证据（本轮全部为 CPU）

| 模块 | 结果 | 关键断言 |
| --- | --- | --- |
| `specdec.engine.cache_clone` | **PASS 24 / FAIL 0** | 8 项 HF 不变量 + 7 项引擎不变量；旧实现被抓出；坏克隆 == 无历史；`output_attentions` 中性 |
| `specdec.engine.xb_probe`（`XB_SELFTEST=1`） | **PASS 11 / FAIL 0** | (a) 映射 26/26 位级、(b) 数值只记录、(c) 克隆与旧 bug 回归 |
| `specdec.engine.hf_serial_block_probe`（`HFSB_MODE=selftest`） | **PASS 8 / FAIL 0** | 切换到共享克隆后行为不变 |
| `specdec.engine.verify_family_independence_cpu` | **PASS 21 / FAIL 0** | 治理守卫（未放宽任何检查） |
| `specdec.engine.dispatch_judge_selftest_cpu` | **PASS 22 / FAIL 0** | 后端调度判定 |
| `specdec.engine.sb_align_selftest_cpu` | **PASS 38 / FAIL 0** | 溯源对齐 |
| `specdec.engine.decision_rows_selftest_cpu` | **PASS 28 / FAIL 0** | 决策行对齐 |
| `specdec.engine.unified_geometry_selftest_cpu` | **PASS 26 / FAIL 0** | UCG 原型（仍为否定结论，未重启） |
| `specdec.engine.xb_hfsb_row_recompute_cpu` | `rc=0` | 见 §6 |
| `specdec.engine.xb_compare_cpu` | `rc=0` | 阶段感知；阶段 B 被拒（缺 `arm_clone_check`） |

**源码级核对（不是实测）**：`hf_serial_block_probe.install_mask_recorder` 的包装器**只记录并原样转发**参数
（`return fn(module, query, key, value, attention_mask, *a, **k)`）；HF 的调用点
（`modeling_qwen2.py:220`）用 **5 个位置参数**（`self, query, key, value, attention_mask`）+ 关键字传参
⇒ 包装器里 `attention_mask is None and a` 的回退分支**不会触发**。

---

## 6. 重算（**新证据**）：用既有产物做同协议跨实现**行级**对照

* 工具：`xb_hfsb_row_recompute_cpu.py`（只读，纯 CPU，**无任何 GPU 运行**）；
* 产物：`output/g5/attn/xb_hfsb_row_recompute_20261001T132915Z_20261001T125516Z_20261001T140243Z.json`
* 使用的既有产物：
  * OURS：`xb_20261001_ours_20261001T132915Z.{json,pt}`（引擎侧克隆**正确**）
  * HF 全历史：`hfsb_20261001_F13_20261001T125516Z.json`（`hf_serial_block_probe`，其克隆**正确**）
  * HF 空历史（症状对照）：`xb_20261001_hf_20261001T132947Z.{json,pt}`（坏克隆）
* **三重前置校验（全部通过）**：
  1. 协议一致：`ctx=8192`、`chunk=2048`、`pre=8240`、`tokens=[220,16,16,1154]`；
  2. **历史位级一致**：`hfsb.initial_cache.L0_k.sha = 45402a5ed3fc2e5c` **==** `xb-hf.C2 L0_k.sha = 45402a5ed3fc2e5c`
     ⇒ 两支独立 HF 运行对同一 8240-token 历史算出的第 0 层 K **逐位相同**（同实现同权重下，这实际上排除了"输入序列/位置不同"）；
  3. 精度一致：只接受有效 `allow_bf16_reduced_precision_reduction == True` 的 hfsb 运行
     （`125546Z` 那次是显式 `False`，**被排除**；`122539Z`/`122902Z` 未记录该字段，也**被排除**）。

| j | 预测位置 | OURS-Serial | OURS-Block | HF 全历史-Serial | HF 全历史-Block | HF 空历史-Serial | HF 空历史-Block | argmax 同(S/B) | 行位级同(S/B) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 8241 | 16 | 16 | 16 | 16 | 16 | 16 | 是 / 是 | 否 / 否 |
| 1 | **8242** | **20** | **20** | **20** | **20** | 13 | 13 | 是 / 是 | 否 / 否 |
| 2 | 8243 | 11 | 11 | 11 | 11 | 9754 | 9754 | 是 / 是 | 否 / 否 |
| 3 | 8244 | 220 | 220 | 220 | 220 | 220 | 220 | 是 / 是 | 否 / 否 |

**读法（严格性上限必须一起引用）**：

* **四行 argmax 全同**（serial 与 block 都是），**行级逐位相等的行数 0/4**（logits 仍逐位不同，与 C1/C2 的 ~0.19/0.21 一致）；
* HF **空历史**两臂 = `[16,13,9754,220]` ⇒ 这就是**坏克隆的症状**，也正是被错误归因给"日程敏感性"的那组数字；
* ★ **它不是**"同一份代码的同路径对照"：两侧来自**不同工具**、不同 API 调用约定；HF 侧来自 `hfsb`，
  它额外传了 `output_attentions=True`（微型 CPU 检查显示**数值中性**，见 §5）并装有原样转发的 mask 记录器；
* 它**不**判定哪一方"正确"，**不**改变任何门槛，**不**用于正式验收，**不**处理 F13 index50 的 AR/Spec 分歧本身。

### 6.1 附带观察（**不是**结论）

* 在 F13 的决策点 **8242**：HF 全历史（含 `hfsb` 与引擎 Spec 路径一致）给 **20**；而记录中引擎 **AR 模式**曾给 **17**，
  且该处 margin 为 **0.0**（17 与 20 并列在 22.0）。这把"执行方式 × 历史"的相互作用（既有结论）与
  既有四格诊断**并行放置**了，但**本轮不做归因**：两个实验的历史不是同一份，跨实验比较**不成立**。

---

## 7. 既有产物上的 C1 差异结构（**只读重算**，用于给定位方案定起点）

来源：`output/g5/attn/xb_20261001_compare_20261001T132915Z_20261001T132947Z_20261001T135722Z.json`
（`C1_kv.rows`，逐层 K/V 的位级与差量摘要；两张 C1 快照已在 `*.pt` 中）。

| 层 | K 是否逐位相等 | K 不同元素 / 总数 | K max\|Δ\| | V 是否逐位相等 | V 不同元素 | V max\|Δ\| |
| --- | --- | --- | --- | --- | --- | --- |
| **L0** | 否 | **60 / 2097152（0.003%）** | **0.015625** ~~（1 ULP）~~ 🔴 | **是** | **0** | **0** |
| L1 | 否 | 18806（0.90%） | 0.25 | 否 | 34174（1.63%） | 0.015625 |
| L2 | 否 | 705275（33.63%） | 0.820312 | 否 | 1000373（47.70%） | 0.212402 |

> 🔴 **【本节 §7 的三条推断已被勘误之二（§11）更正，2026-10-01 同日】**：① `input RMSNorm` 在 K/V 投影**之前**，
> **不能**按执行顺序排除；② "L0 V 逐位相等"只直接证明**该输出**相等，**不**证明输入 hidden 或 `k_proj` 相等；
> ③ "1 ULP"**错误**：逐元素核对 bf16 间距后，这 60 个差异是 **1/2/4/8/16/64 个 bf16 间距**，不是 1 个；
> ④ 由差异元素数量反推扰动大小、"排除 TF32"、"'fp32 最后一位'"三条**全部撤回**。详见 **§11**（含实测清单）。

* 56 个张量中**唯一**逐位相等的是 **L0 的 V**。
* ~~**可推的结论（有前提）**：层内 K/V 都在**注意力之前**算完，
  ⇒ **① score 缩放、② mask 施加、LM head、RMSNorm 都不可能产生 L0 的 K/V 差异**（它们作用位置在 K/V 之后或之外）；
  ⇒ L0 的差异**种子**在 K 的路径上：`k_proj` → **RoPE**；而 L0 V 逐位相等说明**输入 hidden、权重与投影这一段
  在该形状上是可以逐位复现的**（**前提/未验证**：`k_proj` 与 `v_proj` 是不同权重矩阵，理论上仍可能选到不同 kernel；
  V 的逐位相等只**直接**证明了 `v_proj` 一侧）。~~
  > 🔴 **【已更正 · 见 §11】** 正确写法：层内数据流是 **Embedding → input RMSNorm → K 投影 → RoPE → K cache**。
  > K/V 都在注意力**之前**算完 ⇒ 只能排除**作用于 K/V 之后**的环节（①score 缩放、②mask、LM head）；
  > **不能**排除 `input RMSNorm`（它在投影之前）。"L0 V 逐位相等"只直接证明 `v_proj` 的输出相等，
  > **不**能据此说输入 hidden 或 `k_proj` 也逐位相等。因此 L0 的差异可能出现在
  > **RMSNorm / K 投影 / RoPE** 任一环节，必须由**实测的节点对照**区分，不能靠顺序推理。
* ~~**数值量级**：L0 K 只有 **60/2M** 个元素差 1 ULP。用"bf16 相对 ULP ≈ 2^-9，翻转概率 ≈ ε/2^-9"做量级估计，
  60/2M（每张 cos/sin 表项影响 2 维×2 头 = 4 个元素）⇒ 表值相对扰动 ≈ **10⁻⁸ 量级（≈ 0.2 个 fp32 ULP）**
  ⇒ **排除 TF32 级差异**（那会造成成千上万个元素翻转），并把种子限制在 **fp32 的最后一位**。
  ⚠️ 这是**量级估计**，不是测量，不作为判据。~~
  > 🔴 **【整条撤回 · 见 §11】** 由"差异元素数量"反推扰动幅度、据此"排除 TF32"、并把种子"限定在 fp32 最后一位"
  > **都是没有依据的推断**。实测（§11）：`|Δ|/bf16 间距` ∈ {1, 2, 4, 8, 16, 64}，绝对值 3.05e-05 – 0.015625，
  > 与"最后一个 fp32 位"的量级**不符**。根因未定，不得再用计数反推幅度。

---

## 8. C1 最小定位方案（**设计完成未实现**；本轮**不执行**）

**目标**：只回答"L0 K 那 60 个 1-ULP 差异是否来自 RoPE"。**不**回答 F13 的 AR/Spec 分歧，**不**做逐算子扫描。

**边界与纪律**：不改生产默认/门槛；不发明容差；"有差异 ≠ 实现错误"；任一步结论不成立就**停**，不自动扩展。

### 第 1 步（**CPU-only，0 次 GPU 运行**）：RoPE 表逐位对照

* 数据：本地源码两侧实现 —— 自研 `model_self.Build_Rope_cache(head_dim=128, seq_len=8192, theta=1e6, dtype=bf16, start_pos=0)`
  （以及 `start_pos=2048` 的后续分块）与 HF `Qwen2RotaryEmbedding`（`Qwen2Config` 同参数，`position_ids=arange(...)`）。
* 比较：cast 成 bf16 前后的 cos/sin **逐位**比较；报"不同的表项数"与首批不同的 `(position, dim)` 列表。
* 判据：**逐位相同** ⇒ 排除 RoPE 表；**不同** ⇒ 得到一张 `(position, dim)` 预测集。

### 第 2 步（**CPU-only，0 次 GPU 运行**）：位置集合比对（决定性）

* 从既有 C1 快照取出 L0 的 K（两侧），列出那 60 个不同元素的 `(token 位置, 维度)`；
* 与第 1 步的预测集比对：**60 个是否落在预测集内**（超集关系），以及 `dim` 是否成对出现（RoPE 每次旋转影响成对的两维×2 个 kv 头）。
* 判据：
  * **是** ⇒ RoPE 表被确认为 L0 差异来源（仍**不**等于"哪一方对"）；
  * **否** ⇒ 停下来记录**否定结果**，进入第 3 步（不扩大扫描）。

### 第 3 步（**只有第 2 步为"否"时才做**）：单层最小注入对照

* 运行上限：**每个实现各 1 次**前向，**只到 C1**（不喂 C2、不做两臂）；只捕获**第 0 层**的
  `hidden_in / input_layernorm 输出 / q,k,v（RoPE 前与后）/ cos,sin 实际取值`，只取**最后一列位置**与**存在差异的那 60 个位置**。
* 停止条件（任一满足即停，记录缺口，不追下去）：① 定位到具体环节；② 差异在 RoPE 前就存在（则种子在 `k_proj`，
  本轮**不**继续向 kernel 层扩展）；③ 两次运行本身不一致（复现性问题，另案处理）。

**本方案**不含**：逐层逐算子扫描、精度旋钮扫描、其它 family/长度、UCG 重启、任何门槛或容差调整。

> 🔴 **【本方案的两处判据已被勘误之二（§11）收紧，2026-10-01 同日】**：
> ① 第 1 步"逐位相同 ⇒ 排除 RoPE 表"**过强**：CPU 上两侧 cos/sin 表逐位相同**不能**排除 GPU 路径差异
>   （表的生成、cast、以及旋转运算都可能在不同设备/内核上有不同舍入）；CPU 对照只能给"表本身是否相同"。
> ② 第 2 步"落在预测集内 ⇒ RoPE 表被确认为 L0 差异来源"**过强**：位置集合重合只**支持**假设，
>   **不构成因果**。真正的判据改为 **§四/§六 的节点对照 + 复现门 + 条件式替换**（见 `docs/l0_k_localization_2026-10-01.md`）。
> 另：本方案"只取存在差异的那 60 个位置"**违反**"不得缩小矩阵形状"的要求 —— 必须按**原 chunk 形状**（2048）计算后再抽取记录。

---

## 9. 数据与产物清单

| 类型 | 路径 |
| --- | --- |
| C1/C2 原始（VALID） | `output/g5/attn/xb_20261001_ours_20261001T132915Z.{json,pt}`、`xb_20261001_hf_20261001T132947Z.{json,pt}` |
| 阶段感知比较（新） | `output/g5/attn/xb_20261001_compare_20261001T132915Z_20261001T132947Z_20261001T135722Z.json` |
| 行级重算（新，**无 GPU**） | `output/g5/attn/xb_hfsb_row_recompute_20261001T132915Z_20261001T125516Z_20261001T140243Z.json` |
| 被覆盖的旧比较记录 | `output/g5/attn/xb_20261001_compare_20261001T132915Z_20261001T132947Z.json`（**已判 INVALID**，内容被本轮重跑覆盖；两侧原始产物完好，可重算） |
| HF 全历史行（不受 bug 影响） | `output/g5/attn/hfsb_20261001_F13_20261001T125516Z.json`（另有 `122539Z`/`122902Z`/`125546Z`） |
| 其它判 INVALID 的派生记录 | 报告 §9.42.3 阶段 B 表、ledger §4.57 阶段 B 表（**原地保留 + 撤回标记**） |

---

## 10. 治理与未改项

* `verify_family_independence_cpu` **21 PASS / 0 FAIL**（**未**放宽任何检查；未新增 family；F13 仍**不**作为独立验收输入）；
* 生产模型、精度默认（`bf16_reduced_precision_reduction` 未改）、后端、`proposal_freeze_2026-10-01.md`（冻结 6 主 + 2 补充）、
  严格 AR/Spec 一致门（**B6**）**均未改动**；
* **B6 未过 ⇒ 仍禁止 acceptance 测量 / pilot / final**；**正式正确性状态不变（未通过）**；
* 未重启 UCG；未做任何性能/加速比声明（所有计时仅诊断）。

---

## 11. 勘误之二：更正 §7/§8 中的六条过度推断（2026-10-01 同日）

读实际代码与实测数据后逐条更正。**原文本一律原地保留并加 🔴 标记**，本节只做更正说明。

### 11.1 数据流（读源码所得，不是推断）

| 实现 | 第 0 层 K 的实际数据流 |
| --- | --- |
| 自研 | `Embedding.weight[ids]` → `layers[0].input_layernorm`（`RMSNORM`，fp32 内部、回 cast 到 bf16）→ `self_attn.k_proj`（`nn.Linear(1536→256, bias=True)`，fp16/bf16 权重）→ `view(1,2048,2,128).transpose(1,2)` → `apply_rope(k, cos, sin)` |
| HF 5.16.1 | `model.embed_tokens(ids)` → `layers[0].input_layernorm`（`Qwen2RMSNorm`）→ `self_attn.k_proj` → `view(1,2048,-1,128).transpose(1,2)` → `apply_rotary_pos_emb(q,k,cos,sin)` |

出处：`model_self.py:12-17, 349-355, 461-473`；`modeling_qwen2.py:280, 206-211, 384-385`。

### 11.2 六条更正

| # | 原表述（§7/§8） | 更正 | 依据 |
| --- | --- | --- | --- |
| 1 | "①score 缩放、②mask、LM head、**RMSNorm** 都不可能产生 L0 的 K/V 差异（作用位置在 K/V 之后或之外）" | **错**。`input_layernorm` 在 K/V 投影**之前**，**不能**按执行顺序排除。只能排除**作用于 K/V 之后**的环节：①score 缩放、②mask、LM head、`o_proj/MLP`、最终 `norm` | 源码数据流（§11.1） |
| 2 | "L0 V 逐位相等说明**输入 hidden、权重与投影这一段**都可以逐位复现" | **过强**。"V 逐位相等"只**直接**证明 `v_proj` 的**输出**相等；不能推出输入 hidden 或 `k_proj` 也逐位相等 | 逻辑更正 |
| 3 | 第 1 步"CPU 上 cos/sin 表逐位相同 ⇒ **排除** RoPE 表" | **过强**。CPU 表对照只能说明"表本身是否相同"，**不能**排除 GPU 路径差异（表的生成设备、cast、旋转内核都可能不同） | 逻辑更正 |
| 4 | 第 2 步"60 个落在预测集内 ⇒ RoPE 表被**确认**为 L0 差异来源" | **过强**。位置集合重合只**支持**假设，**不构成因果**。判据必须是节点对照 + 复现门 + 条件式替换 | 逻辑更正 |
| 5 | "由 60/2M 反推表值扰动 ≈10⁻⁸（≈0.2 个 fp32 ULP）⇒ 排除 TF32、"种子限制在 fp32 最后一位"" | **整条撤回**（无依据的推断） | 实测见 §11.3：`|Δ|/间距` = 1…64，与"最后一位"不符 |
| 6 | "`K max|Δ|=0.015625`**（1 ULP）**" | **错**。必须按 dtype 实测间距：该元素处 bf16 间距 = 0.0078125 ⇒ 是 **2 个间距**；全 60 个差异的 `|Δ|/间距` ∈ **{1,2,4,8,16,64}**。**今后只报绝对差**，除非逐元素核对了间距 | `l0_k_diff_cpu.py` 实测 |

### 11.3 实测（零 GPU，既有快照，`l0_k_diff_cpu.py`）

产物：`output/g5/attn/l0k_diff_20261001T132915Z_20261001T132947Z_20261001T143031Z.json`
（固定路径读取，未覆盖任何历史产物；输入序列由 `seq_sha16=679ad27f88c66887` 复核一致）。

* L0 K：形状 `(1,2,8192,128)`、两侧均**精确 bf16 值**、`nonfinite=0`；**60/2097152** 个元素不同，
  `max|Δ|=0.015625`、`mean|Δ|=1.82306e-07`；
* L0 V：**逐位相同**（`n_diff=0`、`max|Δ|=0`、`nonfinite=0`）；
* 60 个差异的 `|Δ|` ∈ [3.05e-05, 0.015625]；所在 bf16 间距 ∈ {3.05e-05 … 0.015625}；
  **`|Δ|/间距` ∈ {1,2,4,8,16,64}**（不是"1 ULP"）；
* 分布：chunk {0:4, 1:9, 2:21, 3:26}；`kv_head` {0:32, 1:28}；**只涉及 2 个 head_dim：18 与 82**；
  37 个不同 token 位置（其中 17 个位置有多个维度差）；**4 个 chunk 全覆盖**。
* 结构含义（**假设，未验证**）：18 与 82 正是 RoPE 的**同一旋转对**（`d` 与 `d+64`，配对下标 18），
  ⇒ 若差异来自 RoPE 表，则应集中在这一个频率下标上。**这只支持假设，不是因果**（§11.2 第 4 条）。
