# CUDA BF16 形状敏感性的底层机制确认与修复候选评估（2026-10-03）

> **本轮不是正式性能实验，不启动正式 factorial/B6。** 目标只有两个：
> **Q1** 确认 `M=1` 与 `M=4` 在五类 GEMM 上是否走了不同的 CUDA/cuBLAS 算法/kernel/tile/归约；
> **Q2** 判断是否存在**兼顾正确性与性能**的修复。
> **不**重新检查已排除的 cache / mask / accept-reject / rollback / logit index / RoPE；
> **不**修改或重写历史 FAIL/CHECK；**不**宣布"cuBLAS bug"或"PyTorch bug"。

## 0. 结论摘要

* **Q1（机制）**：**形状是原因**（内容不是、stride 不是），差异来自库内**形状相关的执行/归约选择**；
  但**本环境抓不到 CUDA kernel 级证据**：`torch.profiler` 在 CUDA activity 下**没有任何 kernel 事件**
  （chrome trace 里只有 `cpu_op`、legacy API 无设备时间、`cupti64_2024.1.0.dll` 存在但 CUPTI 活动静默失效）
  ⇒ **无法给出 kernel 名 / tile / split-K 的直接证据**。这在任务验收里正是 **CHECK** 条件。
  我们用**严格控制实验**补上了这条链：同一 PyTorch 算子、同一 dtype、同一权重、
  把第 0 行内容**原样填充**到 `M=4` 仍然不同；改其他三行内容不影响第 0 行；stride 变体无影响；
  `M=1` 内部 view 与 contiguous copy 逐位一致。⇒ **形状本身**驱动差异。
* **Q2（修复）**：**只有"逐行执行"能修**（五类算子逐行 ⇒ L0 决策行 token 从 659 回到 **323**），
  而**精度/确定性旋钮全部无效**、**已有 attention 后端在本路径上根本没生效**。
  逐行执行的代价（上一轮**非 profiler** 测量）是 **decode 1.26×、固定 block verify 1.69×**，
  而 SpecDec 本身只有 AR 的 **0.22×** 吞吐 ⇒ **严格的 token equality 需要付出可能抵消（本就没有的）
  性能收益的代价** —— 这正是任务要求如实报告的重要**负结果**。
* **验收分类**：**CHECK**（不是 PASS-A：缺少 kernel 级证据；不是 PASS-B：同类持续边界在 L0/L1/L14/L27
  **都**仍然存在，且代价已测量并明显不利）。
* **B6 状态未改变**；不启动正式 factorial。

## 1. 实验问题与假设（以及为什么这样设计）

| 问题 | 假设 | 怎么判 | 为什么这样设计 |
| --- | --- | --- | --- |
| Q1-a：`M=1`/`M=4` 是否走不同 **算法/kernel**？ | H1：库内按 M 选择不同 kernel/tile/归约 | profiler 抓 kernel 名；若抓不到，用**严格控制**分离 `M`、`layout`、`content` | 只有把三个因素分开，"形状敏感"才不是一句空话 |
| Q1-b：差异是**形状**还是**内容/布局**？ | H2：只有 `M` 起作用 | 同内容填满 `M=4`、改其他行、stride 变体、view vs copy | 这是可判定的最小对照；也是 PASS-A 允许的替代证据路径 |
| Q2：有没有**便宜**的修复？ | H3：旋钮/后端可能消掉差异 | 旋钮 OFF/ON + 已有后端观测 | 若某个开关就能修，工程上最优；若不能，必须诚实否定 |
| 扩展性：是否**全层同类**？ | H4：同一机制在每层重复 | 在 L0/L1/L14/L27 用**该层真实张量**重跑五算子最小矩阵 | 不能只在 L0 上成立就外推；也不能进入逐算子考古（本轮明令禁止） |

**判据的二值化**：`n_diff == 0`（第 0 行逐位一致）为"闭合"；`> 0` 为"存在形状相关非不变性"。
**不**使用 margin/ULP 豁免；**不**把数学等价写成逐位等价。

## 2. 冻结配置（Step 0）

| 项目 | 值 |
| --- | --- |
| commit / 工作树 | `4ab1e63`（工具提交后运行；工作树状态由每条产物内的 `git` 字段记录） |
| torch / CUDA / cuDNN | **2.6.0+cu124 / 12.4 / 90100**（`torch.version.cublas` 未暴露） |
| GPU / 驱动能力 | **NVIDIA GeForce RTX 4060 Laptop GPU，sm_89（capability [8,9]），8.00 GiB** |
| 精度开关 | `allow_tf32=False`；`allow_bf16_reduced_precision_reduction=True`；`cudnn.allow_tf32=True`（未用 cuDNN 路径）；`deterministic=False`；`preferred_blas_library=Cublas`；`CUBLAS_WORKSPACE_CONFIG` 未设 |
| 模型 | `Qwen/Qwen2.5-1.5B-Instruct`（**只加载 target**，权重 338/338 逐张量位级一致） |
| dtype / 位置 / history | BF16；位置 **8209**；`H(8208)` = `divcap_ar_20261001T185503Z.pt`（sha256 前缀 `0440d98c533ae3ed`，**纯切片** 8237→8208，28 层） |
| token / 形状 | 单步 token `20642`；块候选 `[20642,659,576,4531]`；比较形状 **M=1 与 M=4** |
| 后端 | all-eager（`fused_by_phase={}`、`eager_by_phase={decode:28, verify:28}`） |
| RoPE | target CPU-FP32 `inv_freq`（作用域安装/还原 `ok`） |

**复现结果（Step 0，必须通过才继续；已通过）**

| 路径 | argmax | 323 的 logit | 659 的 logit | 关系 |
| --- | --- | --- | --- | --- |
| 单步 `M=1`（S） | **323** | **23.75** | **23.75** | **并列**（`argmax` 取 323） |
| 四行块 `M=4`（B） | **659** | **23.625** | **23.75** | 659 > 323 |

⇒ 与任务书给定的事实**逐项一致**（含"单步两 token 均 23.75"与"块路径 323=23.625"）。

## 3. 五类算子的真实形状（Step 0/1，来自真实捕获张量）

| 算子 | 协议 | 输入（B 臂，s=4） | 权重 | M=1 变体 | M=4 变体 |
| --- | --- | --- | --- | --- | --- |
| **QK matmul** | `torch.matmul(q, k_repᵀ)/√128` | `q=(1,12,4,128)`、`k_rep=(1,12,8212,128)` A 连续 | 无（K 来自 GQA 展开） | `q[:,:,0:1,:]` contiguous + `k_rep[:,:,:]**` | 整块 |
| **`o_proj`** | `F.linear` | `x=(1,4,1536)`（`heads_merged`） | `W=(1536,1536)` bf16，无 bias | `x[:,0:1,:]` | `x` |
| **`gate_proj`** | `F.linear` | `x=(1,4,1536)`（`post_attention_rmsnorm`） | `W=(8960,1536)` | `x[:,0:1,:]` | `x` |
| **`up_proj`** | `F.linear` | 同上（**同一输入**） | `W=(8960,1536)` | `x[:,0:1,:]` | `x` |
| **`down_proj`** | `F.linear` | `x=(1,4,8960)`（`mlp_gated_product`） | `W=(1536,8960)` | `x[:,0:1,:]` | `x` |

* 三个 Linear 都是 `nn.Linear(bias=False)`、权重 bf16、`data_ptr` 在全程不变（同一张量）。
* **PyTorch 层算子相同**（代码事实）：Linear 都走 `F.linear`（`aten::linear` → BLAS）；QK 都走
  `torch.matmul`（4 维 ⇒ `aten::bmm` → BLAS）。⇒ **PyTorch 侧没有"换算法"**，
  所以差异只能来自**库内按形状选择的 kernel/tile/归约顺序**。

## 4. profiler kernel/算法对照表（Step 1）——**本环境无法给出 kernel 证据**

**五算子最小矩阵（L0；`M=4` 的**第 0 行** vs 独立跑 `M=1`；重复 5 次）**

| 算子 | `n_diff` | `max|Δ|` | 第 0 行逐位一致 | M=1（同 N） vs M=1（前缀） | M=1 view vs contiguous |
| --- | --- | --- | --- | --- | --- |
| QK matmul | **3**（共 12×8209） | **4.0** | **否** | 3（N 不同） | 0（逐位一致） |
| `o_proj` | **2**（共 1536） | 3.8147e-06 | **否** | 2 | 0 |
| `gate_proj` | **6**（共 8960） | 0.00390625 | **否** | 6 | 0 |
| `up_proj` | **11** | 0.00195312 | **否** | 11 | 0 |
| `down_proj` | **6** | 0.00195312 | **否** | 6 | 0 |

**kernel 抓取结果：失败（环境限制）**——证据：

| 尝试 | 结果 |
| --- | --- |
| `torch.profiler.profile(activities=[CUDA])` | torch 2.6 直接报错：`Legacy CUDA profiling requires use_cpu=True` |
| `activities=[CPU, CUDA]` + `record_shapes=True` | **正常运行但 `n_cuda_events = 0`**；`key_averages()` 里**没有任何 kernel 行**（只有 `aten::*`，且设备时间列为 0） |
| `export_chrome_trace()` | trace 内 `cats = {cpu_op: 26, None: 7, Trace: 1}` ⇒ **`kernel` 类事件 0 条** |
| legacy `torch.autograd.profiler.profile(use_cuda=True)` | 所有行的 `cuda_time_total = 0.0` |
| CUPTI 库 | `torch/lib/cupti64_2024.1.0.dll` **存在**，但活动静默失效 |

⇒ **无法**给出 `M=1`/`M=4` 的 kernel 名称、tile 形状、split-K 策略、workspace 大小或 Tensor Core 路径的
**直接**证据。按任务验收，这是 **CHECK**（"profiler 无法确认底层路径"）。
**`alloc_delta_mib`（显存增量）**在两类变体上都在同一量级（记录在 JSON），但因 kernel 事件缺失，
**不作为** workspace 结论。

## 5. layout / content 控制实验（Step 2）——把原因钉在 **M** 上

| 控制（L0，五算子） | 结果 | 说明 |
| --- | --- | --- |
| **同一段数据填满 `M=4`**（第 0 行复制 4 次，内容与 M=1 完全一致） | QK **3**、`o_proj` **2**、`gate` **6**、`up` **11**、`down` **6** —— **仍然不同** | ⇒ **不是"其他行内容"造成的**；形状本身起作用 |
| 四行完全相同 vs 真实 `M=4` 第 0 行 | 与上面同值（3/2/6/11/6 中的"同内容"列） | 同上，双重确认 |
| **只改其他三行内容**（第 0 行不动） | 第 0 行 vs 真实 `M=4` 第 0 行 = **0**（五个算子全部） | ⇒ **其他行内容不影响第 0 行**（排除串扰） |
| **stride 变体**（同一数据放进更大 buffer 再切片，非连续） | vs 真实 `M=4` 第 0 行 = **0**（QK 的 `k` 与 Linear 的 `x`） | ⇒ **stride/layout 不是原因** |
| `M=1` 第 0 行 **view vs contiguous copy** | **0**（逐位一致） | ⇒ `M=1` 内部布局无关（也解释了为什么"逐行执行"是可复现的干预） |
| 等价性对照：`M=4` 第 0 行 vs **contiguous copy 后**独立 `M=1` | 与"view 版 M=1"同值（3/2/6/11/6） | ⇒ 用哪个 `M=1` 变体都一样，差异稳定 |

**旋钮控制（Step 3 候选 ④）**：在**最小复现**上（`gate_proj` 与 QK）

| 设置 | `gate_proj` `n_diff` | QK `n_diff` | 说明 |
| --- | --- | --- | --- |
| baseline | 6 | 3 | —— |
| `allow_bf16_reduced_precision_reduction=False` | 6 | 3 | **无效** |
| `preferred_blas_library="cublaslt"` / `"ck"` | 6 | 3 | **无效**；且状态快照显示 `prefer_blas` 仍为 `Cublas` ⇒ **该 setter 可能需重启进程才生效**（如实标注） |
| `torch.use_deterministic_algorithms(True, warn_only=True)` | 6 | 3 | **无效** |

⇒ **没有任何既有开关能消除该形状非不变性**。（这也与"这不是 bug 开关能绕过的实现错误，而是有限精度下
形状相关的数值非不变性"一致。）

## 6. 修复候选对照表（Step 3）

| # | 候选 | L0 是否逐位闭合 | 决策行 token | 代价（**非 profiler** 测量） | 是否改变算法语义 | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| ① | **原始块 `M=4`** | ✗（L0 输出 `n_diff=111`） | **659** | 1.0×（基准） | 否 | 与 AR 不一致 ⇒ 现状 |
| ② | **五类形状敏感算子逐行 `M=1`** | **✓（111→0）** | **323** ✓ | decode **1.26×**、固定 block verify **1.69×**（上一轮） | 只改执行几何（同一算子/权重/dtype） | **能修正确性，但付出 1.26× decode** |
| ③ | **全部 Linear（含 q/k/v）+ QK 逐行** | ✓ | **323** ✓ | ≥ ②（逐行调用更多；未单独计时） | 同上 | **在 L0 上无额外收益** |
| ④ | 精度/确定性旋钮 | ✗（6/3 不变） | 659 | ~0 | 无 | **不可用** |
| ⑤ | 其它已有 attention 后端 | 不适用 | 659 | — | 若生效会整体改变数值路径 | **在本路径上没生效**（见下） |

**候选 ⑤ 的观测（只作观测，不当正确性证明）**：

| 后端 | S argmax | B argmax | token 一致 | L0 输出 `n_diff` | `fused_by_phase` | `eager_by_phase` |
| --- | --- | --- | --- | --- | --- | --- |
| `eager` | 323 | 659 | ✗ | 111 | `{}` | `{decode:28, verify:28}` |
| `cudnn_prefill` | 323 | 659 | ✗ | 111 | `{}` | `{decode:28, verify:28}` |
| `sdpa` | — | — | — | — | 该策略名**不存在**（仓库只支持 `('eager','cudnn_prefill')`） | — |

⇒ 仓库里唯一的替代后端 `cudnn_prefill` **只作用于 prefill 阶段**（本路径 decode/verify 全部走 eager，
`fused_by_phase={}`），因此**它在这条路径上根本不参与**，也就**不可能**成为形状敏感性的修复候选。

**关于"token 恢复"的口径**：候选 ② 的 L0 输出是**逐位闭合**（111→0）而不是"仅 token 恢复"；
但在**整条生成链**上（上一轮）它是 `token recovered, bitwise not closed`：F13 的 token IDs 与 AR 逐 token 相同，
但 logits 仍未逐位一致（类别 2）。

## 7. 跨层验证（Step 4）

在 **L0 / L1 / L14 / L27** 上做两件事：① B0（无干预）vs BALL（全层候选）的层输入/输出与"首个持续边界"；
② **在该层用真实捕获张量**重跑五类算子的 `M=1` vs `M=4` 最小矩阵。

| 层 | 输入 `n_diff`（B0vsBALL） | 输出 `n_diff` | 首个持续边界 | 该层最小 `M1vsM4`（QK / o_proj / gate / up / down） |
| --- | --- | --- | --- | --- |
| **L0** | **0** | **111** | `['qk_matmul_scale']` | **3 / 2 / 6 / 11 / 6** |
| **L1** | 111（已不同） | 1234 | `[]`（判据无法触发） | **6 / 2 / 5 / 9 / 6** |
| **L14** | 1351 | 1383 | `[]` | **2 / 3 / 5 / 5 / 6** |
| **L27** | 1327 | 1320 | `[]` | **12 / 1 / 6 / 6 / 14** |

* **五类算子在四层上都表现同类形状非不变性**（每个格都 `> 0`）⇒ **"全层同类机制"的受限结论成立**
  （限制：只测了这四层；只测了这五类算子；不外推到别的位置/family/模型）。
* **L1/L14/L27 无法做"首个持续边界"归因**：它们的 B0-vs-BALL **输入已经不同**（111/1351/1327），
  所以"输入逐位相同、输出不同"这一判据**结构上无法触发** —— 这是诚实限制，**不**用"传递"一词掩盖。
* L0 的边界是 `qk_matmul_scale`（生产顺序上第一个满足条件的算子）；上一轮已证明 L0 的**持续**边界集合
  是五类算子（QK/o_proj/gate/up/down）。

## 8. 已证明 / 尚未证明 / 已排除

**已证明（本 run 实测，可复算）**
1. Step 0 精确复现：单步 323（323=659=**23.75** 并列）、四行块 659（323=**23.625**、659=23.75）。
2. **五类算子各自都能在最小复现里重现形状差异**：QK 3、`o_proj` 2、`gate` 6、`up` 11、`down` 6
   （`M=4` 第 0 行 vs 独立 `M=1`，真实捕获张量）。
3. 差异由 **M** 驱动：同内容填满 `M=4` 仍不同；改其他行内容不影响第 0 行（0）；stride 变体 0；
   `M=1` 内部 view vs copy 0。
4. **旋钮无效**：bf16 reduced-reduction、`preferred_blas_library`、deterministic 全都不改变差异。
5. **干预有效**：五类算子逐行 ⇒ L0 决策行 token **659 → 323**（上一轮进一步证明 L0 输出 111→0、
   全层候选 ⇒ F13 token IDs 与 AR 逐 token 相同）。
6. **同层五类算子在 L0/L1/L14/L27 都表现同类形状非不变性**。
7. `cudnn_prefill` 在 decode/verify 路径上**未参与**（`fused_by_phase={}`）。

**尚未证明（本轮没有测到 / 测不到）**
1. **具体 kernel / tile / split-K / workspace / Tensor Core 路径**——**本环境 profiler 拿不到 CUDA kernel 事件**
   （CHECK 条件）。因此"是否换了算法"只有**库内行为**层面的推断，没有直接证据。
2. `preferred_blas_library` 的切换**是否真的生效**（状态快照显示仍为 `Cublas`，可能需要重启进程；
   本轮没有做跨进程验证）。
3. 差异在 `M=2`/`M=3`/`M=8` 等其它 M 上的分布（未测）。
4. 除 token equality 之外的整链数值对齐（logits 仍未逐位一致）。
5. 在别的位置 / family / 模型 / 更长上下文上的同结论（**不外推**）。

**已排除（本轮的控制实验直接排除）**
1. **其他行内容**：改其他三行、或让四行同内容，都不改变第 0 行的差异模式 ⇒ 不是串扰。
2. **stride/layout**：把同一数据放在非连续视图里结果不变（0）⇒ 不是布局。
3. **`M=1` 内部布局**（view vs contiguous copy）：逐位一致。
4. **cache / mask / accept-reject / rollback / logit index / RoPE**：本轮**没有重新检查**（任务要求），
   且既往轮次已排除；本轮所有比较都在**同一份 `H(8208)`**、**同一批候选**、**同一可见前缀**下进行，
   没有使用不同权重/mask/可见范围的对照。
5. **精度旋钮**：bf16 reduced-reduction / deterministic 都不影响 ⇒ 不是这些开关的问题。
6. **已有后端**：`cudnn_prefill` 不参与该路径；`sdpa` 在本仓库不存在。

## 9. 对 B6 与正式实验启动条件的建议

* **B6 状态未改变**：本轮**没有**跑正式 B6/acceptance/pilot/final，**没有**修改或放宽任何门槛。
* **不建议**在当前状态下启动正式 factorial/B6，理由（都是本轮的实测结论）：
  1. 严格 token equality **只能**通过"逐行 M=1 执行五类 GEMM"实现（且要**全部 28 层**），
     代价是 **decode 1.26×、固定 block verify 1.69×**；
  2. 而在同一配置下 SpecDec 本身只有 AR 的 **0.22×** 吞吐 ⇒ 这条修复会**进一步放大负收益**；
  3. 且整链 logits 仍**未逐位一致**（类别 2）⇒ 即使跑 B6，得到的也只是"token-ID 门通过"，
     而不是"数值路径对齐"。
* **若要推进**，建议的**前置条件**（本轮不执行）：先解决 SpecDec 的执行效率前提（使其在 8K/all-eager 下
  至少不慢于 AR），再讨论是否引入逐行几何作为**默认关闭的数值对齐选项**。
* **措辞纪律**：本轮结论只能说"**有限精度下形状相关的数值非不变性**"，
  **不得**写成 cuBLAS bug / PyTorch bug；也**不得**把数学等价写成逐位等价。

## 10. 每一步的原理、为什么这样设计、输入输出、结果如何推动下一步

| 步 | 原理（为什么这样设计） | 输入 → 输出 | 结果如何推动下一步 |
| --- | --- | --- | --- |
| **Step 0 冻结+复现** | 只有在"共同历史、相同位置、相同可见前缀"下复现出 323/659 与两个 logit，后续算子级实验才有意义 | `H(8208)` + token/候选 → S/B 决策行与 top-2 | 复现通过（含**并列**这一关键细节）⇒ 允许做算子级最小复现；若失败本应停止 |
| **Step 1 最小复现+profiler** | 把"层内 111 个差异"降到"单个算子的 2–11 个差异"，才能确定机制 | 真实张量 → 五算子 × {M=4, M=1, M=1-copy} 的 `n_diff`/`max\|Δ\|` + profiler | 五算子都能复现 ⇒ 进入"形状 vs 内容/布局"分离；profiler **无 kernel 事件** ⇒ 立即转向**严格控制**路径（任务允许的替代证据），并把"缺 kernel 证据"如实记为 CHECK |
| **Step 2 控制实验** | 只有把 `M`、`layout`、`content` 三者分开，"形状敏感"才是可判定的命题 | 同内容填满 M=4 / 改其他行 / stride 变体 / view vs copy → 五算子 `n_diff` | 全部指向 **M 本身** ⇒ 后续只考虑"改执行几何"这一类修复，不再猜内容/布局 |
| **Step 3 修复候选** | 工程上先找"便宜"的修法（旋钮/后端），再评估"贵"的修法（逐行） | 旋钮 6 种设置、后端 2 种 → `n_diff` 与 token | 旋钮全无效、后端不参与 ⇒ 只剩"逐行"；逐行代价已知 1.26×/1.69× ⇒ 结论是**负结果**，直接决定第 9 节的建议 |
| **Step 4 跨层** | 避免"只在 L0 成立就外推"；同时**不**进入逐算子考古（任务禁止） | L0/L1/L14/L27 的真实张量 → 层输入/输出差异、边界、五算子最小矩阵 | 四层都同类 ⇒ 得到**受限的**"全层同类机制"结论；同时暴露"L1+ 无法归因"这一结构性限制，为下一步（项目级决策）提供边界 |

## 11. 产物（机器可读）与缺陷记录

**产物**（新 tag，不覆盖旧产物）：
* `output/g5/attn/div8209_gemm_20261003T105817Z.json` + `.pt`（Step 0–4 全量；`stage=complete`）
* `output/g5/attn/gemm_shape_summary_20261003T110456Z.json` + **`.jsonl`**（每个 (层,算子,变体)/旋钮一行）
* `output/g5/attn/div8209_backend_20261003T110242Z.json`（后端观测：eager / cudnn_prefill）
* **保留的失败 run**：`div8209_gemm_20261003T105300Z.json`（序列化，**未加载模型**）、
  `…T105334Z.json`（profiler 活动）、`…T105423Z.json`（Step 4 探针槽）、`…T105706Z.json`（kernel 抽取/打印）、
  `div8209_backend_20261003T110204Z.json`（`sdpa` 策略名不存在）

**缺陷（全部保留，均为我的实现/断言缺陷）**

| # | 现象 | 原因 | 处理 |
| --- | --- | --- | --- |
| ① | 首跑崩在**加载模型之前**：`_BlasBackend is not JSON serializable` | 把 `preferred_blas_library()` 的枚举直接塞进 JSON | `str()` 化（零 GPU 代价） |
| ② | 第二跑：`Legacy CUDA profiling requires use_cpu=True` | torch 2.6 只给 CUDA activity | 同时开 CPU activity（浪费 1 次加载） |
| ③ | 第三跑：Step 4 `KeyError: 'input_rmsnorm'` | `OP.capture` 是**单探针全局槽**，同时 enter 四层只有最后一个生效 | 改为**逐层各跑一次前向**（浪费 1 次加载） |
| ④ | 第四跑抓到的是 `aten::mm`/`aten::bmm`（ATen 名） | `key_averages()` 的 `self_cuda_time_total` 是**含子项总和** | 改抓 device 事件（浪费 1 次加载） |
| ⑤ | 第五跑：kernel 列表为空 + 打印 `KeyError: 'kernel'` | 设备过滤静默失败 + 打印占位 `{}` | 双路径抽取 + 安全打印（浪费 1 次加载） |
| ⑥ | 第六跑：`cuda_kernels` 仍为空 | **环境事实**：CUPTI 活动在本机静默失效（chrome trace 无 `kernel` 事件） | 如实记录为**环境限制**，不再重试（这条不是我的缺陷） |
| ⑦ | 汇总器 `torch.load` 失败 | torch 2.6 默认 `weights_only=True`，产物里含 `TorchVersion` | 显式 `weights_only=False`（产物来源可信） |
| ⑧ | 后端探针首跑：`未知 attention 策略 'sdpa'` | 仓库只支持 `('eager','cudnn_prefill')` | 改用合法值重跑（浪费 1 次加载） |

**GPU 进程**：本工具 6 次（其中 5 次为我的缺陷，1 次为环境限制）+ 后端探针 2 次（1 次策略名缺陷）
= **8 次**；每次都在产物里记录了 `git`、门与版本。
