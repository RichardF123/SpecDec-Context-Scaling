# target RoPE 对齐后的真实 F13 AR/Spec 完整生成一致性（2026-10-01）

> **本轮只回答**："采用该诊断配置（target 的 `inv_freq` 改为 CPU-FP32）后，真实 AR 与 Spec **从头生成的完整序列**是否一致？"
> 不采用固定旧 `pre/Kp/m` 的重放代替真实生成；**不预设**原 index50 分歧一定消失，也不因分歧"更晚"而判通过。
> **B6 未过 ⇒ 仍禁止 acceptance/pilot/final；正式正确性状态不变（未通过）。**
> 前置：`3f64540`（C1 与 HF 逐位一致）。

## 0. 结论（一句话）与状态分类

**AR 与 Spec 的完整生成不一致**：共同前缀 **47**，首次分歧 **index 47（绝对位置 8239）**，
两侧 token 为 **AR=6702 / Spec=6652**，64 个 token 中差异 **1** 个（长度与停止原因相同）。
按预先声明的口径 ⇒ **保存并停止本轮**：不跑重复组、不跑 F06、不新增取证、不做精度组合、不做逐层扫描。

| 类别 | 内容 |
| --- | --- |
| **观察（实测）** | ① 两次运行的 target 初始状态**三方一致**（AR↔Spec 56/56 张量、↔HF C1 快照 56/56、↔已对齐诊断运行 56/56；logits 同约定 sha 相同）；② 干预调用记录：target 全部 `cpu_fp32`、draft 全部 `original`、恢复核查通过；③ 全阶段 eager、无融合；④ 判据/不变量/有限性检查全过；⑤ 序列比较：首组不一致（index 47）；⑥ 同协议既有运行分 4 组（基线、基线 RED=0、本轮 AR 组、基线 Spec 组） |
| **支持的假设** | 位置 8239/8242 一带存在**易翻转决策**：多种互相独立的数值改动（RED 开关、RoPE 频率来源、UCG 几何）都把该处的 token 在 {6652, 6702} 之间搬动。**仅支持，不是因果** |
| **已验证的局部因果** | 无（本轮**没有**做任何因果定位；按边界不做取证） |
| **未知（不得当作结论）** | ① 该分歧是"逻辑错误"还是"形状相关舍入在易翻转处必然翻转"——**未分开**；② 两处决策的 margin/并列情况**未测**；③ 频率来源对齐**不能**解释该分歧（见 §7 基线对照）；④ 换长度/backend/其它 family 时的行为未知 |

---

## 1. 边界、固定引用与核验

* **唯一干预**：只把 **target** 的 RoPE 频率来源换成 CPU-FP32（HF 公式）。draft 保持原实现；
  权重、BF16、`reduction=True`、QK32/PROB32、attention backend、mask、投影、旋转实现、argmax、
  接受/拒绝/回滚/repair 逻辑**全部未改**；**未**改生产默认、proposal、验收门槛；**未**跑 FP8/性能/正式矩阵；**未**安装依赖。
* **协议**：F13、ctx=8192、chunk=2048、k=4、max_new=64、`base_bf16`（双模型 KV 均未量化）、全 eager、
  `allow_bf16_reduced_precision_reduction=True`（显式）、QK32/PROB32 关闭；AR=target-only；Spec=target+draft。
* **固定引用与核验**：

| 引用 | 路径 | 核验 |
| --- | --- | --- |
| HF C1 参考 | `output/g5/attn/xb_20261001_hf_20261001T132947Z.json/.pt` | 56/56 层摘要与新运行逐位一致 |
| 已对齐诊断运行 | `output/g5/attn/c1rope_run_20261001T152125Z.json` + `c1rope_new_dump_….pt` | 其 K/V 与 logits 已在上一轮逐位等于 HF |
| 本轮 AR/Spec | `smoke8k_eager_20261001T165332Z.json` / `…T165431Z.json` | 固定文件名，不"自动选最新" |
| 比较产物 | `output/g5/attn/arspec_consistency_20261001T165332Z_20261001T165431Z_20261001T170447Z.json` | 只读派生，未覆盖任何历史文件 |

* 输入来源：manifest F13（`ids_hash=9a278731d744a859`，与更早的 F13 产物一致）；两次运行 `ids_hash` 相同。
* 权重：target **338/338** 逐张量位级一致；Spec 运行另有 draft **290** 逐张量位级一致。

---

## 2. 最小诊断改动 + 作用域自测（**先过这一关**）

**新增** `specdec/engine/target_rope_scope.py`（实例级作用域）+ 生产入口的一个**默认关闭**开关
（`SMOKE_TARGET_ROPE=cpu_fp32`）。

**为什么不能直接 patch 模块函数**：`QwenLM._run` 调用的是**模块级** `Build_Rope_cache`，而 Spec 流程
**同时驻留 target(1.5B, head_dim=128) 与 draft(0.5B, head_dim=64)**；模块级替换会**连 draft 一起改**，
而本轮只允许改 target，且**禁止**按 `head_dim` 或调用顺序猜身份。

**做法（三层）**：① 包装**类级** `QwenLM._run`（普通函数，保证实例绑定正确），进入时把当前实例压栈；
② 包装模块级 `Build_Rope_cache`，用 `当前实例 is target` 判定用哪张表，**实例未知则抛 `RopeScopeError`**
（绝不静默用错表）；③ `restore()` 用 `is` 核查两个模块属性已还原、当前实例已清空。
`inv_freq` 在安装时按模型配置算**一次**（CPU-FP32），之后只搬运；安装顺序在 `DecisionHooks.install()` **之前**。

**CPU 自测（`python -m specdec.engine.target_rope_scope`，PASS 14 / FAIL 0）**：

* target 的 4 次调用（prefill/decode/verify/repair，形状 6/1/4/1，`start_pos` 0/6/7/11）**全部** `source=cpu_fp32`；
* **draft 的调用全部 `source=original`**，且 draft 的 logits 与 cache 在安装前后**逐位相同**；
* 直接调用模块级 `Build_Rope_cache`（无当前实例）⇒ **抛错**（不许静默用错表）；
* forward 中途注入异常后仍能恢复：`restore()` 的 `is` 核查全 True、恢复后 target 前向正常、重复 `restore()` 幂等；
* ⚠️ **CPU 上两条路径的 `pow` 结果相同** ⇒ 本自测证明的是**作用域与恢复**；数值层差异由上一轮的 GPU 表验证门证明
  （本轮两次运行记录的 `inv_freq_cpu_sha` 均为 `138c99b109d7affb`，与已验证的 HF 表一致）。

---

## 3. 真实生成实验（第一组）

每次独立进程、串行执行、从空缓存重新 prefill（四个 chunk）、不读旧缓存、不回填 token、不固定候选/接受长度/回滚日程。

| 运行 | 模式 | 驻留 | gen | 停止原因 | 内部判定 | 干预调用计数 | 恢复 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **A** `smoke8k_eager_…T165332Z` | `ar`（target-only） | target | 64 | `max_new_reached` | `passed=True` | `target\|cpu_fp32`: **67** | `ok=true` |
| **B** `smoke8k_eager_…T165431Z` | `spec` | target+draft | 64 | `max_new_reached` | `passed=True` | `target\|cpu_fp32`: **47**，`other\|original`: **119** | `ok=true` |

* 各阶段实际 backend：`strategy=eager`，**fused_by_phase={}**；
  eager：AR `{prefill:112, decode:1764}`、Spec `{prefill:208, propose:2328, verify:700, repair:936}`；
  `dispatch_only_prefill=True`（两臂）。
* 有限性检查：AR 64 次（覆盖 prefill/decode）；Spec 210 次（覆盖 prefill/propose/verify/repair）。
* 结构/不变量：Spec 25 轮，`tgt_len == drf_len == pre + (m+1 if m<Kp else Kp)` **全过**；AR 逐步
  `cache_len == prompt_len + 已生成数` **全过**；`cache_structure_ok=True`（跨层 token 维一致）。
* 结束状态（派生字段）：AR `target_cache_len=8255`（8192 + 63 次 decode；第 64 个 token 按生产记账顺序
  只记录、不再喂入）；Spec 末轮 `tgt_len=drf_len=8256`（8192 + 64）。两者 `stop_reason=max_new_reached`、
  `eos_seen=false`。
  ⇒ **观察到一处记账差异**：两条路径的最终 cache 长度相差 1（8255 vs 8256），原因是两个循环的终止方式不同
  （AR 在记录第 64 个 token 后 `break`；Spec 把最后一轮 `emitted` 全部写入 cache）。
  **这属于工具/协议层面的记账差异，不影响已生成 token 序列的比较**，如实记录、**不**当作通过或失败依据。
* SpecDec 统计：25 轮、proposed 97、accepted 46、`target_block_calls=25`、`draft_calls=97`、
  `mismatch_rounds=18`、`all_accept_rounds=7`、`rollbacks=18`。
* 逐轮记录（Spec）含 `pre / Kp / m / rollback / repair / correction / all_accept / cands / g / emitted`；
  `rollback=repair=(m<Kp)`、`correction=g[m]` 由 `spec_dec` 的分支与 `m` 直接派生（口径写在产物里）。
* 资源：AR `pt_peak_alloc=3777 MiB / reserved 9668 MiB`；Spec `5419 / 10648 MiB`；
  **仅诊断，不作任何 speedup 或整卡安全性声明**（WDDM 允许 reserved 超物理容量；同机另有其它消费者）。

---

## 4. 初始状态核验（**先过门，再谈生成差异**）

**比较强度（写明）**：K/V 用 `sb_align.stable_sha` 的**逐张量 sha16（摘要级）**，三方交叉；
logits 在 **AR 与 Spec 之间**是**同一工具、同一约定**的整块 `(1,2048,V)` sha ⇒ **逐位一致**；
与 `c1rope`/HF 参考之间改用**与归约顺序无关**的量（`argmax` / `max` / `margin`）对位 —— **不是**逐位证明。

| 比较 | 结果 |
| --- | --- |
| AR prefill vs Spec prefill（target，28 层 × k/v = 56） | **56/56 逐位相同** |
| AR prefill vs **HF C1 快照** | **56/56 逐位相同** |
| AR prefill vs **已对齐诊断运行（c1rope）** | **56/56 逐位相同** |
| 末位 logits：AR vs Spec（同约定 sha） | **相同**（`577953898c586fcf`） |
| 末位 logits：AR vs c1rope 末位行 | argmax **220/220**、margin **2.125/2.125**、max **20.625/20.625** ⇒ 一致 |
| cache 长度 / 层数 | 8192 / 28（两侧） |

⇒ **初始状态核验通过**：两次运行从**逐位相同**的 target 状态出发，且与 HF C1 参考一致
⇒ 后续的生成差异**不能**归因于"初始状态不同"或"干预未忠实执行"。

* ⚠️ 比较口径更正（保留记录）：首版把 **不同约定**的 logits sha 直接比较，得出"未通过"的**错误**结论；
  另把 smoke 里**在 CUDA 上**求出的 `sum` 与本地**在 CPU 上**重算的 `sum` 判等（差 1 个 fp32 ULP，
  归约顺序所致）⇒ 已剔除 `sum` 判等并写明理由。

---

## 5. 序列比较（完整 token IDs，不是文本）

| 项 | AR | Spec |
| --- | --- | --- |
| 长度 | 64 | 64 |
| 停止原因 | `max_new_reached` | `max_new_reached` |
| 内部判定 | `passed=True` | `passed=True` |

* **token IDs 完全一致 = False**；**common prefix = 47**；**差异个数 = 1**。
* **首次分歧**：`index=47`（绝对位置 **8239**，由位置 **8238** 的 logits 决定），
  **AR=6702 / Spec=6652**。
* 上下文：AR `[… 572, 5880, 389, **6702**, 220, 16 …]`；Spec `[… 572, 5880, 389, **6652**, 220, 16 …]`
  ⇒ 两侧在 index 48 起**重新同步**（故差异总数只有 1）。

---

## 6. 首次分歧对应的**真实轮次**（Spec 侧）

按各轮 `emitted` 累积映射：该 token 属于 **第 18 轮**。

| 字段 | 值 |
| --- | --- |
| 轮次 | **18**（`gen[45..47]`） |
| `pre` | **8237** |
| `Kp` | 4 |
| `m` | **2** |
| `cands` | `[5880, 389, 5470, 220]` |
| `g`（target 各行的 argmax） | `[5880, 389, 6652, 220]` |
| `emitted` | `[5880, 389, 6652]` |
| rollback / repair | **True / True**（`m<Kp` ⇒ 回滚 + 由 target 修正） |
| correction | **6652** |
| `tgt_len`（轮末） | 8240 |
| 分歧 token 的角色 | 轮内偏移 **2 == m** ⇒ **修正 token**（draft 候选 `5470` 被拒，target 自行给出 `6652`） |

AR 侧对应解码步：`step=47`、`decode_runs=47`、`cache_len=8239`、`expected=8239`、`ok=True`
（即 AR 在**同一位置 8239** 单步预测出 **6702**）。

⇒ 事实陈述：**同一位置 8239**，Spec 通过"4 行块验证"得到的修正 token 是 6652，AR 通过"单步"得到 6702。
**本轮不进一步取证**（不做 logits/margin 捕获、不做逐层扫描）——按边界，这是下一轮的事。

---

## 7. 同协议既有运行交叉核对（**只读**，固定清单，不重跑）

清单口径：ctx=8192 / chunk=2048 / k=4 / max_new=64 / F13 / base_bf16 / **eager**。
（2026-09-29 的 `smoke8k_cudnn_prefill_*` 已排除：`arm=cudnn_prefill`，且存在已记录的"整条序列错位一位"记账缺陷。）

**按完整序列分组（4 组）**：

| 组 | t47 | t50 | 成员 |
| --- | --- | --- | --- |
| 1 | 6652 | **17** | `AR-A1`、`AR-A2`、`AR-A3(RED=1)`（lineage A 的 AR） |
| 2 | **279** | 24 | `AR-A4(RED=0)` |
| 3 | **6702** | 20 | **`AR-B1`（本轮对齐后的 AR）**、`SP-A4(RED=0)`、`UCG-AR` |
| 4 | 6652 | **20** | `SP-A1`、`SP-A2`、`SP-A3(RED=1)`、**`SP-B1`（本轮对齐后的 Spec）**、`UCG-SP3` |

**关键比较**：

| 比较 | 结果 |
| --- | --- |
| **基线（lineage A）AR vs Spec** | **不一致**：首次分歧 **index 50（位置 8242）**，AR=**17** / Spec=**20**，差异总数 **14** |
| **本轮（lineage B）AR vs Spec** | **不一致**：首次分歧 **index 47（位置 8239）**，AR=**6702** / Spec=**6652**，差异总数 **1** |
| AR：lineage A vs B（**只换 target 频率来源**） | 不一致：首次分歧 index 47（6652 → 6702），差异总数 15 |
| Spec：lineage A vs B（同一改动） | **完全一致（64/64）** |
| AR：RED=1 vs 默认（同谱系重复性） | **完全一致（64/64）** |
| Spec：两次独立运行（同谱系重复性） | **完全一致（64/64）** |

**可以下的结论（严格）**：

1. **频率来源对齐没有消除 AR/Spec 分歧**：基线（未对齐）同样不一致（位置 8242、17 vs 20、14 个差异），
   本轮只是把**首次分歧位置前移**到 8239 并且差异数降到 1。**两者都没有通过**（B6 要求精确相等）。
2. **对齐把 AR 序列搬进了一个"本来就存在"的序列类**：本轮 AR 的 64 个 token 与
   `SP-A4(RED=0)`（lineage A 的 Spec + RED=0）以及 `UCG-AR`（当时被否定的统一执行几何原型）**逐 token 完全相同**；
   本轮 Spec 则与基线 Spec **逐 token 完全相同** ⇒ 这次干预**没有**产生"新"的数值状态，只是在不同数值改动之间
   重新组合了这些类。
3. ⇒ 位置 8239 与 8242 一带的决策**对多种互相独立的数值改动都敏感**（RED 开关、RoPE 频率来源、UCG 几何）。**这是观察，不是归因**。
4. **旁证（跨协议）**：基线运行的首次分歧落在**绝对位置 8242**，该处的局部 token 历史（8240=220、8241=16）
   与两侧 token（AR=17 / Spec=20）都与更早的 F13/index50 定点捕获（`pre=8240` 协议、另一支工具）**一致**
   ⇒ 两个不同协议、两种 RoPE 谱系都复现了同一个翻转点。⚠️ 两协议的 token 历史并非同一份 ⇒ 只作**旁证**。
5. ❌ **不**把"差异从 14 降到 1"读成"改善了"：那是**一次样本**上的计数，机制未知，且**仍然不是 0**；
   本轮不做门槛调整、不做容差、不因分歧"更晚/更少"而判通过。

---

## 8. 判定（按预先声明的口径）

> **不一致（首组）**：AR 与 Spec 的完整生成在 index 47（绝对位置 8239）首次分歧，
> 两侧 token 为 AR=6702 / Spec=6652，差异总数 1（长度与停止原因相同）。
> 按口径：**保存并停止本轮** —— 不跑重复组、不跑 F06、不新增取证、不做精度组合、不做逐层扫描。

**不豁免声明**：不把「小 margin」「更晚分歧」「较小 logits 差异」当作豁免；分歧就是分歧。
初始状态已通过 ⇒ 分歧**不能**归因于初始状态不同或干预未忠实执行。

---

## 9. 尚未覆盖的验收范围

1. **未做重复组**：本轮按"首组不一致"的规则**停止**，因此没有第二次 AR、第二次 Spec、也没有"两组都一致"的判定
   ⇒ **不构成通过**。（本轮的重复性证据只有§7 里 **lineage A** 的两对独立运行逐位相同。）
2. **未跑 F06**（规则要求 F13 全过才做），**未新增任何独立验收 family**。
3. **未覆盖**：其它 family、其它 ctx/chunk/k/max_new、`cudnn_prefill` 或其它 backend（本轮全 eager）、
   KV 量化档（FP8/V-only）、draft 侧的频率来源（**保持不变**）、CPU 侧生成。
4. **未做定位**：分歧的性质（逻辑错误 vs 形状相关舍入在易翻转处翻转）**未分开**；相关决策点的 margin/并列**未测**。
5. **未改生产**：`Build_Rope_cache` 仍是原实现（在 device 上算 `inv_freq`）；**未**改 proposal/门槛/B6。
6. **历史结果按旧配置保留**：新旧缓存与参考轨迹不得混用；本轮产物与历史产物**并列保存**，未覆盖任何文件。

---

## 10. 失败与缺陷记录（保留，不覆盖）

| # | 阶段 | 现象 | 影响 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | 作用域自测 | 包装函数形参名写成 `theta_`，而 `_run` 用 `theta=` **关键字**调用 | **自测第 3 组直接抓到**，**未消耗任何 GPU 运行** | 形参名与 `Build_Rope_cache` 完全一致；自测 14/0 |
| 2 | 比较器 | 语法错误（中文引号/未闭合字符串/无效 f-string） | 解析期失败，无产出 | 逐条修正后 `py_compile` 通过 |
| 3 | 比较器（**口径**） | 把不同约定的 logits `sha` 直接比较（smoke 哈希整块 `(1,2048,V)`，c1rope 哈希末位行）⇒ 误判"**初始状态未通过**" | **错误结论**（幸好与 K/V 56/56 矛盾而暴露） | 改为同约定比较 + 与参考用 `argmax/max/margin` 对位，并在产物里写明`比较强度` |
| 4 | 比较器（**口径**） | 把 smoke 在 **CUDA** 上求的 `sum` 与本地在 **CPU** 重算的 `sum` 判等 ⇒ 差 1 个 fp32 ULP | 同样是口径问题 | **剔除** `sum` 判等并写明理由（归约顺序与设备相关） |

---

## 11. 唯一推荐的下一步

> **对两个翻转点做一次有界的"决策点取证"**（**1 组**诊断运行，仅诊断、不改任何默认）：
> 在同一份历史上，捕获 **位置 8239（本轮首次分歧）与 8242（基线首次分歧）** 在
> **s=1 单步**与 **s=4 块验证**两种执行方式下的 target 行、候选、`g`、top-2 margin 与并列集合。
>
> * **为什么是它**：本轮只能给出"分歧在哪里、属于哪一轮、角色是什么"；下一步唯一的岔路口是
>   **"逻辑错误"还是"形状相关舍入在易翻转处必然翻转"**。这两者的**修法完全不同**：
>   前者要查 `spec_dec`/`spec_cache` 的记账（行对应、KV 追加与回滚），
>   后者要决定"是否让验证路径用与单步相同的算术计算决策行"（这会与 SpecDec 的加速目标冲突，是个取舍）。
> * **判据（二值）**：若两处决策行在**明显非并列**（margin 远大于实测 bf16 间距）处仍给出不同 argmax
>   ⇒ 指向逻辑/记账问题；若差异只出现在**近并列**处 ⇒ 指向结构性数值敏感，需要就 B6 的口径做决定。
> * **纪律**：**不**用 margin 当豁免、**不**改门槛；上限 1 组运行；不跑重复组/F06/正式矩阵/FP8；
>   历史产物绝不覆盖；**需单独授权**（本轮边界明确禁止"新增取证"）。
> * **另外（不是行动，是决定）**：生产是否采纳该频率来源改动，建议**等上面这组结果之后再定** ——
>   本轮已证明它**既非必要也非充分**（不消除 AR/Spec 分歧，且会把所有历史自研数值重新基线化）。

---

## 12. 治理

* `verify_family_independence_cpu` **25 PASS / 0 FAIL**：新登记只读消费方 `ar_spec_consistency_cpu.py`
  （**范围仅限该工具 + F13**，并新增对应范围检查）；**未**放宽任何既有检查；**未**新增 family；
  F13 仍**不**作为独立验收输入。
* 生产默认 `Build_Rope_cache`、权重、BF16、reduction、backend、mask、投影、旋转实现、argmax、
  冻结 proposal 与 B6 门**均未改**；`SMOKE_TARGET_ROPE` 是**默认关闭**的诊断开关。
* 未跑 FP8/性能/正式矩阵；未安装依赖；所有新产物带时间戳，**未覆盖**任何历史文件；
  失败记录与日志（`_trs_selftest.log`、`_arspec_*.log`、`_asc*.log`）保留。
