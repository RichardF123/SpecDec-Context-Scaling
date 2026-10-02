# 位置 8209 的全层边界：最早分歧产生层 = **L0**（2026-10-02）

> **本轮唯一目标**：在**同一次真实 S/B 前向**里捕获全部 28 层的**决策行**，找出
> 「该层**输入行** S/B 逐位相同、而**输出行**不同」的**第一层** —— 即当前测试轨迹中**最早观测到的分歧产生层**。
> **只定位到层**：不做某层四格、不进入 attention/MLP/RMSNorm 内部、不改生产实现/精度/门槛、
> 不做 FP32、不自由生成、不加载 draft、不启用 cuDNN/SDPA/协作者 PR；GPU 串行；硬失败落盘停止（无自动重试）。
> **B6 状态不变（仍未通过）**；本轮的定位**不构成任何豁免**；旧产物**未覆盖**。

## 0. 结论（一句话 + 三个边界）

**`FIRST_DIVERGENCE_LAYER = L0`（零基）**：
L0 的**输入行**（= 位置 8208 的 embedding 行，两条路径**逐位相同**，`sha16 = 87bdb3aa1faf5f3c`）
而 L0 的**输出行**已经不同（`n_diff = 111`、`max|Δ| = 0.00390625` = 该量级 1 个 bf16 ULP）
⇒ **差异是在 L0 内部被产生的**；此后 L1..L27 的输入差异**逐位等于**上一层的输出差异（层间链自洽）
⇒ L1..L27 是在**传递**这个已产生的差异。

| 三个边界（**严格分开**） | 值 |
| --- | --- |
| `earliest_observed_input_difference`（最早的层，其**输入行** S/B 不逐位相同） | **L1** |
| `earliest_observed_output_difference`（最早的层，其**输出行** S/B 不逐位相同） | **L0** |
| `first_same_input_different_output`（第一层**输入逐位相同、输出不同**） | **L0** ← 本轮要找的那一层 |
| `l0_input_already_differs`（L0 输入是否已不同 ⇒ 分歧是否在 DecoderLayer 之前） | **False**（⇒ 分歧**不**在 DecoderLayer 之前） |

**注意**：`earliest_observed_input_difference = 1 ≠ 0` —— 因为 **L1 的输入 = L0 的输出**，
所以"输入差异最早出现在 L1"只是 L0 已产生差异的**推论**；**不能**把 L1 称为首次来源。

## 1. 命令、commit、dirty 与产物

```powershell
cd D:\RealDSH\specdec
$env:PYTHONIOENCODING='utf-8'
# ① CPU 门（捕获/链式原语 + 工具机制 + 三边界分类）
.\.venv\Scripts\python.exe -X utf8 -m specdec.engine.lmhead_boundary
$env:LC_MODE='selftest'; .\.venv\Scripts\python.exe -X utf8 -m specdec.engine.layer_chain_f13_p8209
Remove-Item Env:LC_MODE
# ② 飞行前检查（纯 CPU：H/截断/历史/已保存行/L27 跨工具参考/锚点）
.\.venv\Scripts\python.exe -X utf8 scripts\div8209_preflight_cpu.py
# ③ 一次有界 GPU 全层捕获
$env:LC_TAG='div8209_chain'; $env:LC_PRE='8208'; $env:LC_TARGET='8209'; $env:LC_K='4'
$env:LC_FEED='20642'; $env:LC_CANDS='20642,659,576,4531'
$env:LC_WATCH='323,659,20642,576,4531'; $env:LC_REPEAT='2'
$env:LC_TARGET_ROPE='cpu_fp32'; $env:LC_CAPTURE_OFF_CHECK='1'
.\.venv\Scripts\python.exe -X utf8 -m specdec.engine.layer_chain_f13_p8209
# ④ 只读 CPU 汇总（从原始 bf16 行独立复算 + 与 JSON/上一轮逐项核对）
.\.venv\Scripts\python.exe -X utf8 scripts\layerchain8209_verdict_cpu.py
```

* **commit / dirty**：run 记录 `git_commit = 60840e179155cddd7967e0c7937ce1c4c31f1290`、`dirty_worktree = true`
  （本轮改动当时未提交；正式提交见仓库 HEAD）。
* **产物**：
  * `output/g5/attn/div8209_chain_20261002T164558Z.json` + `.pt`
    （`.pt` 里每条路径每层 `input`/`output` 决策行均为**原始 bf16**；另含 pass2 全量行、embedding/Norm/最终行）
  * CPU 汇总：`output/g5/attn/layerchain8209_verdict_cpu_20261002T164819Z.json`
* **固定配置（逐项沿用 L27 有效实验）**：F13、ctx=8192、chunk=2048、k=4、`base_bf16`、**all-eager**、BF16、
  `qk_fp32=False`/`prob_fp32=False`、`allow_bf16_reduced_precision_reduction` 默认 True → 显式 True（effective True）、
  `allow_tf32=False`、target CPU-FP32 `inv_freq`、S=`[20642]`（决策行=末位行）、B=`[20642,659,576,4531]`（决策行=`rows[0]`）、
  同一份 `H(8208)`（`divcap_ar_…185503Z.pt`，sha256 `0440d98c533ae3ed`，**纯切片** 8237→8208，与重建捕获 `[0,8208)` **56/56**）、
  只加载 target、权重逐张量 `torch.equal` **338/338**、`fused_by_phase={}`、`eager_by_phase={decode:84, verify:84}`、
  `s_max_by_phase={decode:1, verify:4}`、RoPE 作用域还原 `ok=True`（`n_calls=6`）。

## 2. 全部 28 层输入/输出差异表（原始 bf16 张量，比较时才转 FP32）

| 层 | input n_diff | input max\|Δ\| | output n_diff | output max\|Δ\| | output mean\|Δ\| | in_eq | out_eq |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **L0** | **0** | **0** | **111** | **0.00390625** | 8.194e-05 | **True** | **False** |
| L1 | 111 | 0.00390625 | 1234 | 0.03125 | 0.00382252 | False | False |
| L2 | 1234 | 0.03125 | 1258 | 0.03125 | 0.00496674 | False | False |
| L3 | 1258 | 0.03125 | 1306 | 0.046875 | 0.00879002 | False | False |
| L4 | 1306 | 0.046875 | 1363 | 0.0478515625 | 0.01175745 | False | False |
| L5 | 1363 | 0.0478515625 | 1345 | 0.078125 | 0.01400733 | False | False |
| L6 | 1345 | 0.078125 | 1343 | 0.0625 | 0.01337433 | False | False |
| L7 | 1343 | 0.0625 | 1360 | 0.09375 | 0.01381499 | False | False |
| L8 | 1360 | 0.09375 | 1339 | 0.1015625 | 0.01331097 | False | False |
| L9 | 1339 | 0.1015625 | 1360 | 0.125 | 0.01469549 | False | False |
| L10 | 1360 | 0.125 | 1365 | 0.25 | 0.01483123 | False | False |
| L11 | 1365 | 0.25 | 1358 | 0.125 | 0.01361454 | False | False |
| L12 | 1358 | 0.125 | 1353 | 0.109375 | 0.01326463 | False | False |
| L13 | 1353 | 0.109375 | 1328 | 0.125 | 0.01326386 | False | False |
| L14 | 1328 | 0.125 | 1352 | 0.125 | 0.01421515 | False | False |
| L15 | 1352 | 0.125 | 1338 | 0.25 | 0.01322166 | False | False |
| L16 | 1338 | 0.25 | 1339 | 0.1875 | 0.01295482 | False | False |
| L17 | 1339 | 0.1875 | 1322 | 0.5 | 0.01252792 | False | False |
| L18 | 1322 | 0.5 | 1323 | 0.375 | 0.01290472 | False | False |
| L19 | 1323 | 0.375 | 1318 | 0.75 | 0.01544547 | False | False |
| L20 | 1318 | 0.75 | 1318 | 0.75 | 0.01700544 | False | False |
| L21 | 1318 | 0.75 | 1331 | 0.75 | 0.01825523 | False | False |
| L22 | 1331 | 0.75 | 1333 | 0.75 | 0.02133735 | False | False |
| L23 | 1333 | 0.75 | 1344 | 0.75 | 0.02586921 | False | False |
| L24 | 1344 | 0.75 | 1330 | 0.5 | 0.02907364 | False | False |
| L25 | 1330 | 0.5 | 1340 | 0.5 | 0.03356711 | False | False |
| L26 | 1340 | 0.5 | 1357 | 0.5 | 0.03861038 | False | False |
| L27 | 1357 | 0.5 | 1386 | 2.0 | 0.05218061 | False | False |

* **只有 L0 的输入行逐位相同**（其余 27 层的输入都不同）；**没有任何一层的输出逐位相同**。
* **差异不随层数单调增长**：`n_diff` 从 111 升到 1365 后回落（1343 / 1328 / 1318…），
  `max|Δ|` 也从 0.75 回落到 0.5；`mean|Δ|` 总体上升但中间有回落。
  ⇒ **不假定单调**；本轮 28 层**全部直接测量**，**不**用二分排除任何层；后续层即使重新逐位相同也会**如实保留**（本轮没有出现）。
* 每层 `cos`/`sin` 决策行 S/B **逐位相同**（28/28 层都相同）—— RoPE 不是差异来源。
* 关键 sha16：L0 输入 `87bdb3aa1faf5f3c`（S=B，且等于 embedding 行）；
  L0 输出 S `ff7c4133bea45a00` vs B `37ddf8a1ff092d86`；
  L27 输出 S `30b997c9d821c221` vs B `b80cd206b26020b5`。

## 3. 逐位控制（全部通过）

| # | 控制 | 结果 |
| --- | --- | --- |
| 1 | S/B **embedding 决策行**逐位相同 | ✔（`sha16 87bdb3aa1faf5f3c`；ids S=`[20642]`、B=`[20642,659,576,4531]`，第 0 行是同一 token/位置） |
| 2 | **L0 输入行 == embedding 行**（两条路径各自核） | ✔ 逐位 |
| 3 | **layer[i].输出行 == layer[i+1].输入行**（S、B 各自核，27 条边 × 2） | ✔ 全部逐位（首条不自洽的边：无） |
| 4 | **L27 输出行 == 最终 Norm 输入行**（两条路径各自核） | ✔ 逐位 |
| 5 | **捕获开启/关闭**的真实 S/B logits 行逐位相同 | ✔（不挂任何钩子时逐位相同） |
| 6 | **hook 全部移除**（28 层 pre/post + norm + embedding） | ✔ |
| 7 | **权重未变**（绑定权重 `data_ptr`/抽样摘要、Norm 权重 `data_ptr`） | ✔ |
| 8 | **历史 cache / 输入张量未被修改** | ✔ `H` 快照逐层 k/v 摘要前后一致、长度 8208 未变；每层缓存按设计**只追加**（S 8208→8209、B 8208→8212，引擎正常行为） |
| 9 | **重复两次捕获**（各自独立克隆，全部层行 + embedding + Norm 行） | ✔ 逐位相同 |
| 10 | 与**已保存的局部对照行**（`div8209_cells_…064415Z`）比较 | ✔ S/B 两行都**逐位相同** |
| 11 | 与**运行记录锚点**（argmax 级）比较 | ✔ S=323、B=659 |
| 12 | **跨工具核对**：本轮 L27 的输入/输出行 vs 上一轮 L27 run 的 `aS/aB4/zS/zB4` 第 0 行 | ✔ 四个都**逐位相同** |
| 13 | 最终行（本轮真实引擎行）：最终 Norm 输出 S vs B `n_diff=1343 / max 1.0`；**最终 logits** S vs B `n_diff=136478 / max 0.25`、argmax 323 vs 659 | ✔ 与早先四格 run 的 `same_history_S_vs_B_AR`（136478 / 0.25 / mean 0.0410212）一致 |

**CPU 汇总独立复算**：`scripts/layerchain8209_verdict_cpu.py`（**不加载模型**）从 `.pt` 的原始 bf16 行重算
28 层比较 + 三个边界 + 边界链 + 两次捕获一致性，并与 JSON/上一轮参考逐项核对：
`json_per_layer_match 28/28 全真`、`json_boundaries_match True`、L27 四行与上一轮逐位相同、
最终行与已保存的局部对照行逐位相同 ⇒ 汇总可独立复算，**没有**为修摘要重跑 GPU。

## 4. 已证实 / 支持性推断 / 未证实（严格分开）

**已证实（本轮实测）**
1. **L0 输入行 S/B 逐位相同**（= embedding 行，`sha16 87bdb3aa1faf5f3c`）；**L0 输出行不同**
   （`n_diff=111`、`max|Δ|=0.00390625`、`mean|Δ|=8.194e-05`）⇒ **`FIRST_DIVERGENCE_LAYER = L0`**。
2. 三个边界分别为：最早输入差异层 **L1**、最早输出差异层 **L0**、第一层同输入异输出 **L0**；
   `l0_input_already_differs = False`（分歧**不**在 DecoderLayer 之前）。
3. 层间边界链**逐位自洽**（27 条边 × 2 条路径），且 L0 输入 = embedding 行、L27 输出 = 最终 Norm 输入。
4. 每层 `cos`/`sin` 决策行 S/B 逐位相同（28/28）。
5. 全部逐位控制通过（§3），包括捕获开关等价、hook 移除、权重/cache/输入未变、两次捕获一致、
   跨工具（L27）与跨 run（已保存对照行/锚点）一致。
6. 差异**不随层数单调增长**（`n_diff` 与 `max|Δ|` 都有回落）。

**支持性推断（由上面的数支持，但不是本轮新测到的机制）**
* 由于 L0 的**输入与 cache 基线**在两路径下都逐位相同（H 相同、embedding 相同、RoPE 相同），
  而 L0 的输出已经不同 ⇒ 该差异是 **L0 在"单步 `s=1` decode" vs "四行 `s=4` verify"两种执行形状下**
  产生的最小可测差异（首个差异量级 = 1 个 bf16 ULP：`0.00390625`，出现在 111 个坐标上）。
  **本轮只到"层"这一级**：L0 **内部**是哪个算子最先产生它，**未测**。
* L1..L27 的输入差异逐位等于上一层的输出差异 ⇒ 后续层在**传递**该差异；
  这与前几轮"L27 主要传递上游差异、其自身形状效应不足以翻转"的结论**一致但不重复**：
  本轮给出的是"差异**首次产生**在 L0"，前几轮给出的是"在 L27 这一格它只是被传递"。
* 位置 8209 的最终选择对 ≤1 ULP 级差异敏感（`margin=0.125`；多格出现**精确并列**），
  这是**贴边**事实的延续，不是新机制。

**未证实（本轮没有测、也不声称）**
1. **L0 内部**第一个产生差异的算子/子模块（attention？q/k/v/o 投影？input_layernorm？MLP？）——**本轮不进入层内**。
2. **323 / 659 谁是数学真值**（**未**做 FP32 ⇒ 不做 `BF16_NEAR_TIE` 类判定）。
3. 差异的来源分解（bf16 舍入 vs fp32 累加顺序 / kernel 选择 / reduction 切分）——**不**升级为"必然只是 BF16 舍入"。
4. 该差异是否在**其他位置/其他历史**上同样"首次产生于 L0"（**不**外推）。
5. 8239（上一轮的现场）与 8209 是否同源。

## 5. 每项代码改动的原理 / 输入 / 输出 / 必要性

| 改动 | 原理 | 输入 → 输出 | 为什么必要 |
| --- | --- | --- | --- |
| **扩充** `lmhead_boundary.py`（复用上一轮模块；加两个原语） | ① `layer_chain_capture(model, indices, row_index)`：对每层挂**只读** pre/post 钩子，只记录**决策行** `a_row`/`z_row`（`(hidden,)`、原始 dtype、**GPU 上 `detach().clone()`**）与元数据（`s`/phase/cache 长度/cos-row/sin-row），**不在钩子里 `.cpu()`**；② `to_cpu_rows(rec)`：前向结束后统一搬 CPU（28 层 × 2 × 1536 × 2B ≈ 172 KB/路径） | 模型 + 行号 → 每层 `{pre:{a_row,…}, post:{z_row,…}}` → 搬 CPU 后的记录 | 全层边界必须有**入口**状态；在钩子里 `.cpu()` 会每层同步、可能改变执行轨迹 ⇒ 明文禁止 |
| **新增** `layer_chain_f13_p8209.py` | 同一份 H 上跑 S/B（复用 `spec_dec._step`/`_verify`，与已保存四格同调用形状），一次前向挂 embedding + 28 层链 + norm 钩子；**两次捕获**（各自独立克隆）+ **一次关闭捕获**；每层比较 + 边界链控制 + 跨工具核对 + 三个边界分类（纯函数 `_classify`） | 捕获 `.pt` + 运行记录 → JSON/PT（每层原始 bf16 行 + 全部比较字段 + 三个边界） | 本轮唯一实验；把"最早在**哪一层**产生"做成**逐位**判定，并把"三个边界"严格分开 |
| 改 `scripts/div8209_preflight_cpu.py`（+1 节） | 加载模型前核对 H/截断/历史一致/已保存行/**L27 跨工具参考**/锚点 | 固定路径产物 → 断言（失败即退出） | 前几轮各**白付过一次模型加载**（键名/口径/过滤器类缺陷） |
| **新增** `scripts/layerchain8209_verdict_cpu.py` | 从**原始 bf16 行**独立复算 28 层比较、边界链、两次捕获一致性、三个边界；与 JSON/上一轮参考逐项核对；输出层级表格 | 完整 run 的 `.json`+`.pt` → 汇总 JSON | 摘要必须能从原始张量独立复算；**摘要错误不通过重跑 GPU 修复** |
| 未改 | 生产 `spec_dec`/`spec_cache`/`model_self`、层与 Norm 公式、proposal、门槛、精度开关、mask/RoPE/attention/cache/argmax | — | 本轮只做诊断 |

## 6. B6 状态

**未改变**：本轮**没有**跑正式矩阵、**没有**改门槛/精度/算法、**没有**放宽任何判据；
B6（`SpecDec ids == target-only greedy ids` 且全有限、多轮确定）**仍未通过** ⇒
**acceptance / pilot / final 仍然禁止**。本轮结论是**局部诊断**，不是修复，也不是"数学真值"的判定。

## 7. 唯一最小下一步（**未执行，需单独授权**）

> **对 `FIRST_DIVERGENCE_LAYER = L0` 做与 L27 完全相同的"输入来源 × 执行形状"四格**：
> 用**同一模块实例** `layers[0]`、**每格独立克隆的轮前缓存**、同一份 `cos/sin`（第 0 行逐位核对），
> 四格 = {输入 `aS`(1,1,1536) / `aB`(1,4,1536) 第 0 行} × {单行 `s=1` decode / 四行 `s=4` verify}，
> 决策行统一成 `(1,1,1536)` → **同一最终 Norm** → **同一绑定权重**完整词表投影；
> 控制：`L0(aS,单行) ≡ 捕获 zS`、`L0(aB4,四行) 第 0 行 ≡ 捕获 zB4`（**逐位**）。
>
> **判据（二值）**：① 固定 L0 形状、换输入是否翻转（支持**输入来源效应**）；
> ② 固定输入、换 L0 形状是否翻转（支持 **L0 执行路径效应**）；两者都成立 ⇒ 如实记**交互**。
> **本轮不执行该实验**；**不**进入 L0 层内算子扫描；**不**改计算路径；**不**做 FP32；
> **不**假定逐层单调、**不**用二分排除未测量的层。

---

## 12. 追加：下一步已执行（2026-10-02 同日，L0 内部算子边界）

> 本节由**下一轮**追加，**不改上面任何原文**。

* 上面 §7 的"对 L0 做四格"被**用户修正**为：不再重复"输入来源"轴（L0 入口处 `aS == aB[row0]`，该轴已塌缩），
  而是**直接进入 L0 内部**沿真实数据流定位首个分歧算子。
* **结果**：`first_divergence_operator = **q_proj**`（`layers[0].self_attn.q_proj`）——
  输入 `input_rmsnorm` 输出**逐位相同**（`sha16 f012125d65d98bf6`）、权重同一张量，
  而输出 **2/1536** 坐标不同、`max|Δ|=0.000244140625`（该量级 1 bf16 ULP）。
  **也**满足条件的还有 `qk_matmul_scale`（如实全部列出，取生产顺序第一个）。
* **对照**：`k_proj`/`v_proj` 输出**逐位相同**；进入 RoPE 的 q 已不同 ⇒ **RoPE 不是边界**，
  且本例 `q_rope` 输出**又变回逐位相同**（差异被舍入吸收）。
* **与本文件的交叉核对**：本文件的 `rows_S_raw[0]["input"]`（`87bdb3aa1faf5f3c`）与
  `rows_S_raw[0]["output"]`（`ff7c4133bea45a00`）在下一轮被**逐位复现**（B 侧 `37ddf8a1ff092d86` 同样复现），
  L0 输出 S/B `n_diff=111`、`max|Δ|=0.00390625` 不变。
* 全文 `docs/div8209_l0_operator_boundary_2026-10-02.md`（报告 §9.55 / 台账 §4.70）。
