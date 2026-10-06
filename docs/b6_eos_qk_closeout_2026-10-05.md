# BF16 B6 收尾：EOS 覆盖 + QK scope 历史失败归因（2026-10-05）

> 状态口径：**已存在**（本轮之前就有）/ **本次新增**（本轮实现）/ **已验证通过**（有产物证据）/ **未达成（不予声称）**。
> 本轮**未运行任何 GPU 任务**、**未改任何生产数值路径**、**未重跑已完成的 66 次 B6 run**、**未改任何历史产物或判决**。
> 结果基线：`docs/b6_bf16_coverage_result_2026-10-05.md`（66/66 run、9 PASS / 1 INCOMPLETE / 0 FAIL）。

## 0. 一句话结论

1. **EOS 控制逻辑**：CPU 矩阵 **10/10 分支通过（29/0 断言）**，其中 6 条是此前未覆盖的分支（含"拒绝后 target 替换 token = EOS"）；
   但 **CPU 证据 ≠ 真实模型证据**：冻结判据要求的是**真实模型上的 exact ids equality + 全 finite + 多次确定性一致**
   ⇒ **`B6_BF16_PASS` 仍缺 EOS 的真实模型覆盖**（`eos-prone` 6/6 都停在 `max_new_reached`）。
2. **QK scope 33/3**：三个失败断言**全部是静态计数假设过时**（该假设写在"serial-QK 逐行干预"轮次**之前**），
   **不是**生产计算范围或开关隔离被破坏；已按**角色分类**最小更新（保留并加强语义约束），`qk_scope_selftest_cpu` **48/0**。
   另外发现一个值得裁决的**隐式行为**：两个 QK 诊断互斥只在注释里，没有显式报错（见 §3.4）。

## 1. 交付 1：EOS 覆盖矩阵（CPU；正反例；证据边界）

### 1.1 复用的既有工具（避免重复造工具）
| 工具 | 状态 | 作用 |
| --- | --- | --- |
| `specdec/engine/g4_b5_eos_branch_tests.py` | **已存在**（2026-09 修 EOS cache 越界时写） | 4 条 EOS 分支 + "同一组 case 在**旧实现**下必须抛 ValueError"的非空过保证；`StubModel` 位置策略 stub |
| `specdec/engine/g4_b5_eos_matrix_cpu.py` | **本次新增** | **复用**上者的 `StubModel` / `_propose_old` / `invariants`，补 6 条分支并做**矩阵化**断言（含分支见证、cache 合同、变异体） |

### 1.2 矩阵（10 条；每条都断言**最终结果**，不是"没有异常"）
| id | 被测实现分支（`spec_dec.py` 行） | fixture（prompt_len=2, k=4, EOS=3, vocab=32） | 期望 gen | 期望 stop_reason | 期望轮数 | 见证断言 | 结果 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `A_all_accept_eos_last` | 全接受、末候选=EOS（155-156 / 244-250 / 260-261） | draft=target={1:5,2:EOS}, N=8 | `[5,EOS]` | `eos` | 1 | 全接受 + 末候选=EOS + 末输出=EOS | ✅ |
| `B_all_accept_eos_kp3` | 同上、kp=3 | {1:5,2:7,3:EOS} | `[5,7,EOS]` | `eos` | 1 | 同上 | ✅ |
| `C_budget1_first_eos` | 预算=1 且首个候选=EOS（205 / 260） | draft=target={1:EOS}, N=1 | `[EOS]` | `eos` | 1 | 全接受（预算与生成同时用尽 ⇒ EOS 优先） | ✅ |
| `D_midround_eos_rejected` | EOS 候选**被拒**（235-243 §5） | draft={1:5,2:EOS}, target={1:5,2:6}, N=4 | `[5,6,0,0]` | `max_new_reached` | 2 | r0 `m=1<kp` 且末候选=EOS | ✅ |
| **`E_reject_replacement_is_eos`** | **★拒绝后 target 替换 token = EOS**（236-237 / 260） | draft={1:5,2:6}, target={1:5,2:EOS}, N=8 | `[5,EOS]` | `eos` | 1 | r0 `m=1<kp` 且末输出=EOS | ✅ |
| **`F_eos_at_budget_boundary`** | **★EOS 恰落在预算边界**（204-205 / 260-261） | {1:5,2:7,3:EOS}, N=**3** | `[5,7,EOS]` | `eos` | 1 | 全接受 + 末输出=EOS；`len(gen)==max_new` | ✅ |
| **`G_budget_cut_before_eos`** | 预算在 EOS 之前截断（205；**对照**） | {1:5,2:7,3:EOS}, N=**2** | `[5,7]` | `max_new_reached` | 1 | r0 `Kp=2`（EOS 候选被 `k_eff` 截掉） | ✅ |
| **`H_rollback_then_eos_next_round`** | **★回滚/repair 后下一轮以 EOS 收尾**（235-243 / 238-241 / 260） | draft={1:5,2:6,3:EOS}, target={1:5,2:9,3:EOS}, N=8 | `[5,9,EOS]` | `eos` | 2 | r0 `m=1<kp`（回滚已执行，`rollbacks≥1`）+ r1 全接受 EOS | ✅ |
| **`I_carried_bonus_is_eos`** | **★全接受轮的"额外 token"（`saved_next=rows[kp-1]`）在下一轮以 EOS 兑现**（248） | {1:5,2:7,3:8,4:0,5:EOS}, N=6 | `[5,7,8,0,EOS]` | `eos` | 2 | r0 全接受（`Kp=4`，当轮**不**输出额外 token）+ r1 `g[0]=EOS` 且输出 `[EOS]` | ✅ |
| **`J_first_candidate_eos_rejected`** | **★首个候选=EOS 但被拒（`m=0`）**（155-156 / 238/240 trim） | draft={1:EOS,2:EOS}, target={1:4,2:4}, N=4 | `[4,4,0,0]` | `max_new_reached` | 3 | r0 `m=0`、末候选=EOS、输出 `[4]` | ✅ |

**需求 3 的核对结论（不为不存在的路径新增生产逻辑）**：实现里**没有**独立的"全接受额外 token 路径" ——
`spec_dec.py:248` 的 `saved_next = rows[kp-1]` 只把额外那一行 logits **留到下一轮**，当轮不输出；
其等价可观测行为由 `I_carried_bonus_is_eos`（额外 token 在下一轮以 EOS 兑现）覆盖。
另：`_propose` 遇 EOS 即停（155-156）⇒ **候选里 EOS 之后不可能有 token**；本矩阵对每条 case 都断言了这一点。

### 1.3 断言项（每条 case 全过才算通过）
① `gen` 与期望**逐 token 相等**；② EOS **恰好 1 次**且在末位（不该出现 EOS 的 case 断言其**不出现**）；
③ `stop_reason` 按 **harness 规则**（`g5_8k_smoke_cudnn.py:713-718`，**EOS 优先于预算**）与期望一致，
且 `len(gen) ≤ max_new`、`EOS 位置 == gen_len-1` 自洽；④ **EOS 之后不再生成**（轮数固定、无后续轮次）；
⑤ 每轮长度不变式 `tgt_len == drf_len == pre + (m+1 if m<Kp else Kp)`（复用 B5 `invariants`）；
⑥ **末轮 cache 合同**：EOS 必须按合同提交（全接受 ⇒ `pre+Kp`；错配 ⇒ `pre+m+1`），未提交即失败；
⑦ 统计自洽：`draft_calls == proposed`、`rounds == len(trace)`、`rollbacks == mismatch_rounds`、`accepted ≤ proposed`；
⑧ **分支见证**（防止"数字碰巧对上"）。

### 1.4 负例（能抓出"错误停止 / 错误 cache 提交"）
| 负例 | 做法 | 期望 | 结果 |
| --- | --- | --- | --- |
| 变异体等价性自证 | `_loop_variant(mutate=None)` vs 真实 `spec_decode_greedy` | 10/10 case 的 `gen`/`stats`/轮数**全同** | ✅（等价成立，变异体 = "已证等价的副本 + 单点改动"） |
| 缺陷① EOS 后不终止 | 副本里去掉 EOS `break` | 被**同一个检查器**抓出 | ✅ |
| 缺陷② EOS 之后多带 token | 输出序列尾部多加一个 token | 被抓出 | ✅ |
| 缺陷③ cache 被裁短 | `trim_cache` 少裁一个（EOS 未按合同提交） | 被抓出（长度合同/不变式） | ✅ |
| 缺陷④ 候选在 EOS 之后还有 token | 检查器自测喂入篡改 trace | 被抓出 | ✅ |
| 检查器自测 | EOS 后补 token / 丢掉 EOS / cache 长度不符 | 三种都被报错 | ✅ |
| **非空过保证**（每条 case 各自成立） | `old_fails`（旧 `_propose` 必抛 ValueError）5 条；`old_same`（新旧一致）3 条；`mut_no_terminate`（关闭 EOS 终止后必须被抓出）1 条 | 10/10 通过 | ✅ |

### 1.5 证据边界（必须如实声明）
* stub 用**位置策略**替换 target/draft 的 logits ⇒ 证明的是**控制逻辑与 cache 账**，
  **不是**真实模型行为、**不是**数值/质量证据；stub 的 `tok_at(pos)` 与输入 token 值无关 ⇒ 对数值传播**不适用**。
* 因此：**CPU EOS 通过 ≠ 真实模型 EOS 通过**；本轮 **未**新增任何 GPU 运行，**未**改动 `eos-prone` 的 6 份产物或判决。

### 1.6 现行冻结门要求的是**哪一种**证据
`docs/design_freeze_g4_specdec.md:83`（B6 = correctness gate）：
> prompts：自然语言/数学/代码/短/长/全接受/首位拒绝/中途拒绝/**EOS**/非 4 倍数长度；条件 base_bf16 与 v_fp8；
> 判据 **exact ids equality + 全 finite + 多次确定性一致**。**B6 未过禁止 acceptance 测量/pilot/final。**

⇒ EOS 这一项需要的是：**真实模型 + 真实 tokenizer** 的 run，其**生成序列真的以 tokenizer 的 EOS id 收尾**，
并满足（我轮协议修订后的口径）serial 与 AR 的**完整序列相等**、全 finite、跨重复确定性一致。
CPU stub 矩阵**不能**替代它（它只覆盖控制逻辑合同）；也**不得**用"把普通 token 临时当 EOS"冒充。

## 2. 交付 2：QK scope 三个失败的逐项归因

复跑 `qk_scope_selftest_cpu` 的原始输出（修复前）：**PASS 33 / FAIL 3**

| # | 失败断言 | 实测 | 归因 |
| --- | --- | --- | --- |
| 1 | `模块里 DIAG_QK_FP32 的读取点共 2 处（含状态读取器）` | 3 处：`L359 diag_precision`、`L748 forward`、`L751 forward` | **静态计数假设过时**：公式写死在"只有 1 个精度读取点 + 1 个报告器"的时代；guard 是后来"serial-QK 逐行干预"轮次加入的 |
| 2 | `★ 影响计算的读取点只有 1 处（排除 diag_precision 这类读取器）` | 2 处（L748、L751） | 同上：第二处是**路径选择 guard**，**不**改任何精度表达式 |
| 3 | `该唯一读取点位于 attention 的 eager 分支（forward 内）` | 前提（唯一性）不成立 ⇒ 断言连带失败 | 同上；其**意图**（精度干预只在 eager 分支）仍然成立（见下） |

### 2.1 逐个读取点解释（分支 / 影响 / 默认可达性 / 与 `_SERIAL_QK` 的交互 / 合同）

| 读取点 | 所属分支 | 对 QK 计算的影响 | 默认（`qk_fp32=False`）可达性 | 与 `_SERIAL_QK` 的交互 | 是否破坏"QK FP32 只改变指定计算范围" |
| --- | --- | --- | --- | --- | --- |
| **`model_self.py:359`（`diag_precision()`）** | 状态报告器 | **无**（只读出来报告） | 可达（被调用即读） | 无 | 否（不参与计算） |
| **`model_self.py:748`（`forward` eager 分支内）** | `_sq = (_serial_qk_branch(...) if (_SERIAL_QK is not None and not DIAG_QK_FP32) else None)` | **不改精度**；只决定本轮 QK 由**哪条实现**提供：逐行 M=1 causal-prefix（诊断）或原 block QK | **运行期不求值**（`and` 短路：默认 `_SERIAL_QK is None` ⇒ 第二个操作数不评估） | 与 `_SERIAL_QK` **联合判定**：注册了 serial-QK 时，`qk_fp32=True` 会**关掉**逐行路径（动态实测：applied 0 vs ≥1） | **否**：它是"同一 QK 步的两条实现之间选路"，既不扩大精度干预到别的算子/阶段，也不影响融合 prefill（`_fused_prefill_attention` 源码无该名字） |
| **`model_self.py:751`（`elif DIAG_QK_FP32:`）** | R1 精度分支 | **改精度**：`matmul(q.float(), k_float^T)/sqrt(head_dim)` ⇒ softmax 输入不再是 bf16 舍入后的 scores | 可达（打开开关即走） | 无 | 否：`unchanged_ops` 逐项仍有断言（softmax 本身/AV matmul/o_proj/MLP/RMSNorm/lm_head/权重/缓存 dtype） |

**结论（不是模型错误）**：仅凭"读取点数量"从来不足以判定范围问题；按角色拆开后可见：**precision 读取点仍恰好 1 处、
且在 `forward` 的 eager 分支**；新增的那一处是**默认短路**的路径选择 guard。
⇒ 属**静态测试假设过时**，非生产计算范围/开关隔离被破坏。

### 2.2 最小更新（保留并加强语义约束；未删除任何范围检查）
| 文件 | 改动 | 说明 |
| --- | --- | --- |
| `specdec/engine/diag_qk_scope.py` | 新增 `read_site_roles()`（AST 按角色分类：`accessor` / `precision` / `guard` / `unclassified`），`static_scope_proof()` 增加角色字段 | `n_qk_compute_read_sites` **含义不变**（仍是非报告器的读取点）；新增 `n_qk_precision_read_sites` / `n_qk_guard_read_sites` / `n_qk_unclassified_read_sites` / `guards_inert_by_default`；`qk_read_only_in_eager_branch` 改为基于 precision 角色（**该字段公式原先已随 guard 失效**，见 §2.4） |
| `specdec/engine/qk_scope_selftest_cpu.py` | §1 改为角色化断言（**unclassified 必须为 0**、precision 恰好 1 且必须是 `if/elif` 测试本身、guard 必须**默认惰性**、非报告器读取点 = precision + guard、accessor 只允许在报告器里、融合 prefill 仍绕过）；§1b 分类器**合成源码回归**（`if S:`⇒precision；`(_C is not None and not S)`⇒惰性 guard；写反顺序⇒guard 但**不**惰性；`S+1`⇒unclassified）；§8 新增**动态**交互测试 | 未改成"无意义的计数上限"：分类器对**任何未归类**读取点直接判失败，可捕获未来静默新增 |
| **未修改** | `specdec/engine/model_self.py`、`spec_dec.py`、`spec_cache.py`、`g4_b5_eos_branch_tests.py`、所有 B6 产物、所有门限 | 生产/数值路径**零改动** |

### 2.3 回归结果
* `qk_scope_selftest_cpu`：**33/3 → 48/0**（**已验证通过**）。
* 分类器输出：`L359/diag_precision=accessor | L748/forward=guard(默认短路) | L751/forward=precision`。
* 全量 CPU 自测：**25/25 模块通过（0 失败）**——本轮之前唯一失败的模块即 `qk_scope_selftest_cpu`。
* `verify_family_independence_cpu` **33/0**；`stage2b_preflight_cpu` **rc=0（PREFLIGHT_OK）**。
* 既有 EOS 工具回归：`g4_b5_eos_branch_tests` **PASS**（未改动）；`g4_b5_helper_tests` **PASS**。

### 2.4 历史影响（如实说明）
* 含 `attention_scope_proof` 的历史产物 **16 份**（`qk32*_20260928*`）全部记录
  `n_qk_read_sites=2 / n_qk_compute_read_sites=1 / qk_read_only_in_eager_branch=true` —— 那时 **guard 尚不存在**，
  因此历史记录**没有被误写**，本轮也**未**改动它们。
* `g5_8k_qk32_intervention.py` / `g5_8k_residual_history_probe.py` 会把 `static_scope_proof()` 写进新产物：
  本轮**只增字段**，旧字段含义未变 ⇒ 新产物与旧产物**并列可读**；`qk_read_only_in_eager_branch` 的含义变化已在
  `diag_qk_scope.py` 的 `role_note` 中写明。

### 2.5 值得裁决的**隐式行为**（本轮只报告，**未**擅自改生产文件）
两个 QK 诊断（`DIAG_QK_FP32` 与 `set_serial_qk` 的逐行 M=1 路径）**互斥**这一点目前**只写在注释里**，
没有任何断言/异常：若同时打开，会**静默**退化为 block QK 路径（可查 `serial_qk_state()["n_applied"] == 0`，但不报错）。
对照：`serial`-verify 与 row-aligned wrapper 的冲突会显式抛 `VerifyModeError`（`spec_dec.py:61-62`）。
⇒ 建议二选一（均需授权，其一涉生产文件）：
（i）在 guard 处/`set_serial_qk` 里显式报错（生产文件改动，需配套回归）；
（ii）在冻结文档中把该互斥登记为**规定行为**，并把 §8 的动态测试作为其证据（本轮已实现该测试）。

## 3. 交付 3：本轮修改 / 未修改文件与回归

**本次新增**
* `specdec/engine/g4_b5_eos_matrix_cpu.py` —— EOS 覆盖矩阵（10 case / 29 断言 / 4 变异体 / 3 检查器自测）。
* `docs/b6_eos_qk_closeout_2026-10-05.md` —— 本文件。

**本次修改（均为 CPU 测试/诊断辅助，非数值路径）**
* `specdec/engine/diag_qk_scope.py` —— 读取点**角色分类** + 角色字段（`QKScopeHook` 行为未改）。
* `specdec/engine/qk_scope_selftest_cpu.py` —— §1 角色化、§1b 分类器回归、§8 动态交互测试。

**明确未修改**
`model_self.py`（生产数值路径）、`spec_dec.py`、`spec_cache.py`、`b6_branch_coverage_cpu.py` 及所有门限、
`g4_b5_eos_branch_tests.py`（复用）、全部 66 份 B6 run 产物与其判决、全部历史 FAIL/CHECK/PASS。
正式实验 runner/采集/统计/绘图仍由用户本人编写。

## 4. 交付 4：`B6_BF16_PASS` 还缺什么

| 缺口 | 事实 | 现状 |
| --- | --- | --- |
| **真实模型的 EOS 停止覆盖** | `eos-prone` 6/6 run `stop=max_new_reached`、`eos_seen=False` ⇒ `eos_stop` 分支 **UNCOVERED** ⇒ 该用例 `CASE_INCOMPLETE` | **仍缺**；本轮 CPU 矩阵**不**改变这一事实 |
| 其它 9 个用例 | 全部 `CASE_PASS`；`serial==AR` 20/20；集合级分支 5/5、类型 8/8 `COVERED`；0 `FAIL` | 已满足 |
| 协议决策 | D-5 质量评分 / D-6 FP8 独立 Gate A / D-7 判决词表 / D-9 不加额外 FP32 / D-11 `:83` 类型覆盖 / D-12 per-KV vs 全局；P-1/P-2 双侧决策行捕获 | 仍未决（与 `BF16_FORMAL_READY` 相关，不影响 B6 分支矩阵本身） |

⇒ 只差**一项**：EOS 的真实模型证据（或一份**显式**的协议豁免登记，见 §5.B）。

## 5. 交付 5：所需的最小授权 / 协议裁决（集中一次）

**A. EOS 真实模型补测（预登记方案，本轮只提方案、不跑 GPU）**

* **目的**：让**真实模型 + 真实 tokenizer** 的生成序列**真的以 tokenizer EOS id 收尾**，从而覆盖 `eos_stop`。
* **固定输入（逐字，单一来源 = 一份新的声明文件）**：
  `"<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>\n<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\nHi"`
  （**纯文本 prompt**，仍走现行 literal 模式：`add_special_tokens=True`、**不**套用 chat template、不额外加/去 BOS；
  `<|im_end|>` 作为**词表内特殊 token 文本**被 tokenizer 编码为 151645。记录 `prompt_sha1` 供复核。）
  * **CPU 编码事实（已实测、只读；`output/g5/attn/_eos_recheck_prompt_cpu.json`）**：
    `n_tokens=30`；`151645` 出现 **3 次且各为单个 token** ⇒ 特殊 token 文本被正确识别；
    prompt **末 token 不是** EOS（末 token=13048「Hi」）；`eos_config.eos_id=151645` 与 `EOS_ID_EXPECTED` 一致；
    `decide_attention_path("prefill", 30)` = **`fused`**（⇒ 该 run 会顺带覆盖**字面 prompt 上的融合 prefill**）；
    `prompt_sha1=c9d959169209c4af`。
  * ⚠️ 该 CPU 预检**只**证明"这条 prompt 可被正确编码"，**不**预测模型是否真的输出 EOS —— 那必须由真实 run 证明。
* **固定配置**：`SO_ARM=cudnn_prefill`、`SO_KV=base_bf16`、`chunk=2048`、`k=4`、`ctx=8192`（prompt 远短于 ctx）、
  **`max_new=24`（预先声明，非自适应）**、贪心、冻结 EOS 规则、`struct_check=1`；三模式 = 独立进程、从头 prefill、不共享 cache。
* **次数**：`ar/prod/serial × rep1/2` = **6 runs**，**新 tag**（建议 `div8209_eos`；不并入 `div8209_b6`，避免把跨版本矩阵混为一个）。
* **停止条件（预登记验收判据，全部满足才算覆盖成功）**：
  1. 6/6 run 的 `gen[-1] == eos_id(151645)`、`stop_reason == "eos"`、`eos_seen == True`、`gen_len == index(EOS)+1`；
  2. `serial` 与 AR **完整序列相等**（2 reps 各一次）；`production vs AR` 差异**只记录**（不判控制逻辑失败、不称无损）；
  3. 同模式跨重复**完全一致**（determinism）；
  4. 全部硬检查通过（finiteness / accounting / cache 结构 / 逐轮不变量 / 诊断精度 / `attention_assert=ok`）；
  5. EOS 候选按冻结合同提交（`tgt_len == drf_len`，EOS 之后无候选、无后续轮次）；
  6. 该 prompt 长 > 2 token ⇒ 顺带证明**融合 prefill 路径**（`fused_by_phase['prefill'] > 0`）。
* **未命中规则（关键）**：任一 run 未出 EOS ⇒ **本轮判定"EOS 真实模型覆盖仍缺失"并停止**；
  **不得**换 prompt、**不得**加长预算、**不得**把普通 token 当 EOS、**不得**重跑已完成的 66 次 run。
* **资源估计**（依据已完成的 60 次短 prompt run 实测）：单 run ≈ **35–40 s**（含模型加载），
  峰值 `pt_reserved` ≈ **4.2 GiB**、双驻留（spec）时 `cuda_free` ≈ **2.9 GiB**；
  **6 runs ≈ 4–5 min GPU** + ≤1 min 编排；入口余量门槛不变（实测 +4.8 GiB ≥ 18 GiB）⇒ **低风险**（非 8K 类）。
* **可选变体（需你明确选择，二选一）**：
  (a) **严格单 prompt**（推荐）：只跑上面这一条；未命中即报告缺失。
  (b) **≤2 条预登记 prompt**：第 2 条也在运行前逐字登记（并说明理由），规则 = "P1 未命中才跑 P2；P2 之后无论如何停止"。
  —— 这不是"反复换 prompt 直到命中"，但确实放宽了"只跑一条"的严格性，因此**必须由你批准**。
* **另一条更可靠但改变输入约定的选项**：使用 tokenizer 的 **chat template** 编码（`apply_chat_template`）。
  它会让 assistant 轮自然以 `<|im_end|>` 结束，命中率高得多；但它引入**新的输入编码模式**（与现行 10 个 B6 用例的
  纯文本编码不同），属**协议层面的输入约定变更** ⇒ 需单独裁决，且必须作为一个**明确标注**的失败/成功判据来源。

**B. 协议裁决：`eos_stop` 是"必须真实模型覆盖"还是"登记为显式豁免"**
* 维持现状（推荐）：`eos_stop` 必须有真实模型证据 ⇒ 需要 A。
* 或者：在冻结文档中**显式登记豁免**（理由 = 控制逻辑合同已由 CPU 矩阵覆盖，真实模型未覆盖），
  并同时登记"B6 在 EOS 类型上以豁免通过"。**不得**静默豁免、**不得**把 CPU 结果写成真实模型证据。

**C. QK 互斥是否显式化**（§2.5）：(i) 授权改生产文件加显式报错 + 回归，或 (ii) 登记为规定行为（本轮已补动态测试）。

**D. 既有待批协议项**：D-5 / D-6 / D-7 / D-9 / D-11 / D-12（沿用清单，未变化）。

**E. 可选文档登记**：把本轮两处**验收规则澄清**（B6 验收器 `uncovered` 的语义、QK 读取点的**角色**定义）
写进冻结/设计文档，避免今后再被"静态计数"误判（纯文档，不改判据）。
