# AR/Spec 配置审计 + 初始状态 / 后续计算路径的最小直接对照（F06/index25，2026-09-30）

> 本轮目标：用**直接对照**区分三件事 ——（1）AR/Spec 实际配置不一致；（2）prefill 初始状态不一致；
> （3）从相同初始状态出发，单步与分块执行产生差异。
> **未**预设 cuDNN 是原因，也**未**预设差异必然是正常浮点误差。
> **未**改生产算法/默认精度/attention backend/manifest/验收门槛；**未**跑正式矩阵；**未**做速度声明。

---

## 0. 结论速览

| 假设 | 判定 | 依据 |
| --- | --- | --- |
| **H1 AR/Spec 配置不一致** | **路径相关字段未发现不一致** | 两边都是**分块 prefill**（chunk=2048）、都在 `phase='prefill'` 走**融合 cuDNN**、`base_bf16`、精度开关全关（代码 + 产物双重证明，见 §1） |
| **H2 prefill 初始状态不一致** | **同一 backend 内：确定性且两次建立逐元素一致**；**不同 backend 之间：初始状态确实不同** | 同 backend 两次独立建立：logits 逐位相同、28 层 K/V 摘要全同；cuDNN vs eager：**54/56** 层 K/V 的 sum 不同（§3.4） |
| **H3 相同初始状态下的 S/B 差异** | **成立，且足以复现 index 25 的分歧** | 从**同一份**初始状态、喂**同一串** token：S/B 的 logits **从第一个 verify 位置（预测位置 8193）起就不同**，token 级**首次分歧正好在 8217**（S=715，B=576），与真实 AR/Spec **一致**（§3.3） |

**一句话**：在**同一条 cuDNN 生产配置**下，只要初始状态相同、喂的 token 相同，
**「单步 vs 四行分块」这一条差异就足以复现 AR=715 / Spec=576**；
而不换 backend 的 prefill 配置审计**没有发现不一致**。
换 backend（eager prefill）会改变初始状态并**改变后续轨迹**，但它**未复现**生产参考值 ⇒ 按预登记规则**不做因果结论**（§3.4）。

---

## 1. 配置审计（分开列：代码已确认 / 原始产物已证明 / 尚未验证）

### 1.1 路径相关字段对照

| 字段 | AR 路径 | Spec 路径 | 状态 |
| --- | --- | --- | --- |
| 权重来源 | `lowmem_load.load_ours_bf16(TARGET)` | 同（target 相同权重；draft 另加载） | **代码已确认** |
| 权重相等性 | `weights_bitmatch=True`（逐张量 338 个） | 同 | **产物已证明**（16 次运行记录） |
| compute dtype | bf16（KV `base_bf16`） | 同 | **代码 + 产物已证明** |
| prefill 实现 | `P5V.timed_chunked_prefill(acc, CHUNK, tgt)`（`g5_8k_smoke_cudnn.py:272`） | **同一个实现**（经 `DecisionHooks.install()` 替换 `spec_dec._prefill`，`smoke_hooks.py:155`） | **代码已确认** |
| prefill chunk | 2048（`CHUNK`） | 2048 | **代码 + 产物已证明** |
| prefill phase | `attention_phase("prefill")` | 同（`smoke_hooks.py:101`） | **代码已确认** |
| prefill 后端（**实际执行**） | `fused_by_phase={'prefill': 112}` = 28 层 × **4** chunk | `fused_by_phase={'prefill': 208}` = target 28×4 + draft 24×4 | **产物已证明**（smoke JSON 的 `attention_stats`） |
| prefill `s_max` | **2048** | **2048** | **产物已证明**（若为单次 8192 前向，这里会是 8192） |
| decode 路径 | `phase='decode'`，`s=1`，eager | `phase='verify'`（s=4）/`'repair'`（s=1），eager | **代码 + 产物已证明**（`eager_by_phase`） |
| qk_fp32 / prob_fp32 | False / False | 同 | **产物已证明**（`diag_precision`） |
| RoPE | `_run` 内 `start_pos = cache[0]["k"].shape[2]`；θ / head_dim 来自配置 | 同 | **代码已确认** |
| causal mask | `s==1`：**不施加**；`s>1`：`triu(diag=T-s+1)`；`cache is None`：`triu(1)` | 同 | **代码已确认** |
| KV 格式 | `fp16`（= base_bf16 语义，K/V 均 bf16 存储） | 同 | **代码已确认** |
| cache 深拷贝/截断/回滚 | AR 无截断（只追加） | `spec_cache.trim_cache(blk, pre+m)` + `_step(r)` 回滚 | **代码已确认** |
| 位置计数 | `cache_length(cache)` | 同 | **代码已确认** |
| prefill 末位 logits 的使用 | `nxt = argmax(lg[..., -1, :])`（预测位置 = prompt_len） | `saved_next = tgt_lg[:, -1:]` | **代码已确认，语义一致** |

### 1.2 ★ 需要报告的配置风险（不是本次 F06 对照的不一致，但会造成别处不一致）

`spec_dec._prefill` 的**默认实现**是「一次前向吃下整个 prompt」（`spec_dec.py:14-18`，`cache=None`）
⇒ 在不同 `phase` 下会落到 **eager**（`_ATTENTION_PHASE` 默认是 `"other"`，`model_self.py:107`），
8K 时该路径需要 (1,12,8192,8192) 的 scores + fp32 softmax（约 3 GiB）⇒ **OOM 风险**。

- **走 `DecisionHooks` 的入口**（`g5_8k_smoke_cudnn.py`、`g5_f06_index25_forensics.py`、
  `g5_8k_qk32_intervention.py`、`g5_8k_residual_history_probe.py`、`g5_8k_divergence_2x2.py`）
  ⇒ 被替换为**分块**实现，与 AR 一致 ✓（**本次 F06 对照属于这一类**）；
- **不走 hooks、直接调用 `spec_dec.spec_decode_greedy` 的入口**：`g4_b6_correctness.py`、
  `g4_b6_fp32_correctness.py`、`g4_b6_tie_diag.py`、`g4_b5_smoke.py`、`g4_part4_mismatch_diag.py`、
  `g4_specdec_index_tests.py`、`g5_attn_cudnn_validate.py`、`attn_phase_selftest_cpu.py`
  ⇒ 其 **prefill 默认是单次 eager**，与 AR 的分块融合路径**不同**；短上下文下这通常不影响 token，
  但**不能**与 8K 的分块融合路径当作同一配置比较。
- **尚未验证**：这些入口在 8K 下的实际行为（是否会 OOM / 是否产生不同 token）。

### 1.3 尚未验证（如实列出）

1. `positions`/RoPE 在两臂之间的**逐元素**等价性（未单独测 RoPE 表）；
2. KV cache 的**截断/回滚**在真实 Spec 运行中对 K/V 数值的逐元素影响（本轮只记录了长度与边界）；
3. draft 侧 prefill/decode 的路径（本轮按任务要求**不加载 draft**，只吃记录的 `cands`）；
4. 8K 下 eager **单次** prefill 是否真的 OOM（本轮只测了 eager **分块**，它**没有** OOM）。

---

## 2. 复用的旧产物与新增运行

| 用途 | 来源 | 是否新跑 |
| --- | --- | --- |
| F06 输入（8192 token） | `family_input.load_family_ids`（hash 见记录） | 复用（不新造输入） |
| 真实 Spec 轮次边界/候选/repair | `…_F06_spec_20260929T093904Z.json` 的 `rounds`（12 轮，`pre`/`Kp`/`m`/`cands`/`g`/`emitted`） | 复用 |
| 真实 AR 的 `gen_head` / 决策位置 | `…_F06_ar_20260929T093951Z.json` | 复用 |
| 配置与后端证据 | `smoke8k_cudnn_prefill_*.json` 的 `attention_stats` | 复用 |
| **新增运行 1** | `…_prefillsb_20260930T181032Z.json/.pt`（arm=cudnn_prefill，**complete**） | **新跑** |
| **新增运行 2** | `…_prefillsb_20260930T181158Z.json`（arm=eager，**checks_failed**：S 臂未复现 715） | **新跑** |

**未把不同版本/精度/输入拼成同一组实验**：两次新运行使用同一脚本、同一 commit 基线、
同一 F06 输入、同一精度开关，唯一差别是 `PREFSB_ARM`。

---

## 3. 对照执行情况与数据

### 3.1 2×2 的实际执行状态

| | 后续串行 S | 后续分块 B |
| --- | --- | --- |
| **eager prefill** | **E-S 已执行 → 复现门失败（记录）** | **E-B 未执行**（预登记停止规则） |
| **cuDNN prefill** | **C-S 已执行 ✓** | **C-B 已执行 ✓** |

- C 轴两臂（S/B）**两个复现门全部通过**，因此允许做同位置因果对照；
- E 轴：初始状态本身可复现（两次建立逐元素一致），但 **S 臂在输入位置 8216 给出 576，而生产 AR 给 715**
  ⇒ 按 §三「若不能复现，停止因果分析」**停止**，E-B **未执行**（如实记录，不补跑、不换参数）。

### 3.2 规范初始状态（C 轴）

| 项 | 值 |
| --- | --- |
| cache 长度 | **8192**（= prompt 长度，28 层 × `(1,2,8192,128)` bf16，finite） |
| prefill 末位 argmax（= 首 token） | **659**（与记录一致） |
| 二次**独立**建立 | **logits 逐位相同**；28 层 K/V 的摘要 **56/56 全同** |
| S/B 起点 | 同一份 P0 的**两次深拷贝**，逐层数据指针不同（不共享可变存储）✓ |

### 3.3 C-S vs C-B（同一初始状态、同一串 token）

- 两臂喂入 token 序列完全相同：`emitted[:25]` == 真实 AR 的 `gen_head`（逐元素，老师强制）；
- **共同可比较的预测位置：26 个**（8192 来自 prefill 末位，8193..8217 来自后续调用）。

| 预测位置 | S argmax | B argmax | 同 argmax | max\|Δ\| | mean\|Δ\| | 不同坐标数 |
| --- | --- | --- | --- | --- | --- | --- |
| 8192 | 659 | 659 | ✓ | **0** | 0 | 0 |
| **8193** | 330 | 330 | ✓ | **0.1875** | 0.0310 | 133438 |
| 8194 | 576 | 576 | ✓ | 0.1875 | 0.0412 | 140123 |
| 8197 | 1083 | 1083 | ✓ | 0.15625 | 0.0244 | 127144 |
| 8202 | 594 | 594 | ✓ | 0.25 | 0.0320 | 130513 |
| 8205 | 879 | 879 | ✓ | 0.1875 | 0.0395 | 136645 |
| 8216 | 659 | 659 | ✓ | 0.289062 | 0.0833 | 146639 |
| **8217** | **715** | **576** | **✗** | 0.1875 | 0.0294 | 132519 |

| 指标 | 位置 |
| --- | --- |
| **首次非零 logits 差异** | **8193**（第一个 verify 行 vs 第一个串行步） |
| **首次 argmax 分歧** | **8217**（S=715，B=576） |

- 逐位相同的只有 8192（两臂共用同一份 prefill logits）；**从 8193 起每一个位置的 logits 都不同**
  （12 万～14.6 万个坐标不同，max\|Δ\| 0.125～0.289），**但 argmax 一直相同**，直到 8217 才分歧。
- 8217 处的并列结构：**S：{715,4710} 并列（margin 0.0）**；**B：{576,715,4710} 三方并列（margin 0.0）**
  ⇒ 两臂都由 `argmax` 取最小下标决定，差异来自**并列结构不同**（576 在 S 臂比最高分低 0.125，
  在 B 臂恰好等于最高分）。这与先前 §9.30.8 的真实行证据一致。
- **中间 cache 的状态**：两臂在相同预测位置时 cache 长度相同，但**数值历史已不同**
  （S 逐 token 追加、B 按 verify/repair 边界追加与截断）⇒ 自 8193 起**不可再称「当前算子输入完全相同」**，
  只能说「**相同 token 前缀，但数值历史不同**」。

### 3.4 E 轴（eager prefill）与 E/C 初始状态差异

| 项 | 结果 |
| --- | --- |
| eager **分块** prefill @8K 是否可运行 | **可运行，未 OOM**（修正旧说法：历史上「8K eager 必 OOM」对应的是**单次** 8192 前向，不是分块） |
| E 轴初始状态确定性 | 两次独立建立：logits 逐位相同、28 层摘要全同 ✓ |
| **E/C 初始状态差异** | 用**跨进程可比**的 float sum：**54/56** 个层-K/V 的 sum 不同（例：`L10_k` -126660.42 vs -134893.02；`L12_v` 50135.36 vs 57122.41）⇒ **backend 改变了初始 KV** |
| E-S 轨迹 vs C-S 轨迹（同老师强制 token） | 25 个预测位置中 **2 处**不同：**8202**（C=594，E=**364**）、**8217**（C=715，E=**576**） |
| E 臂 8217 的并列结构 | 单点最高 **margin 0.25**（不是并列） |
| 按预登记规则的处置 | **S 臂未复现生产参考值 ⇒ 停止，不做因果分析，E-B 未执行** |

> ⚠️ **这是复现缺口，不是根因结论**。可如实记录的观察是：
> 在**本位置**上，prefill backend 的改变**不是结果中性的**（初始 KV 与后续轨迹都变了）。
> **不能**据此判 cuDNN 实现有错，也**不能**据此说 eager 会让 AR 与 Spec 一致
> （E 臂轨迹在 8202 就已偏离，两者不是同一 token 历史下的可比值）。
> **`sha` 字段是进程内 `hash()`，不可跨进程比较**——这是本工具的一个真实缺陷，已在 §5 列为待修项。

---

## 4. 结论（按任务 §六 的规则）

| 规则 | 本轮命中情况 |
| --- | --- |
| 两种 backend 下 S/B 均分歧 ⇒ cuDNN 不是该类分歧的必要条件 | **未命中**（E 轴未通过复现门，E-B 未执行，不能下这个结论） |
| 仅 cuDNN 起点下分歧 ⇒ 支持初始状态与后续路径存在交互 | **部分命中**：C 轴 S/B 分歧已确认；但 E 轴**未复现生产现象**，故只能说「**复现不足**」，**不判 cuDNN 实现错误** |
| 两种 backend 下 S/B 均一致但真实 Spec 分歧 ⇒ 检查调用边界与状态维护 | **未命中** |
| 发现配置不一致 ⇒ 报告字段与执行位置 | **命中（风险项）**：`spec_dec._prefill` 默认单次 eager 与 AR 的分块融合不同（§1.2，列出 8 个入口） |
| 对照无法复现原始现象 ⇒ 结论为复现不足，不给根因判断 | **E 轴按此处置** |

**本轮支持 / 削弱的假设**
- **支持 H3**：在**同一** cuDNN 配置、**同一**初始状态、**同一** token 序列下，
  「单步 vs 四行分块」足以复现 index 25 的 AR/Spec 分歧（715 vs 576），
  且**首次非零 logits 差异出现在第一个 verify 位置（8193）**、**token 级首次分歧在 8217**。
- **削弱 H1**：路径相关配置字段逐项对照**未发现不一致**（除 §1.2 的入口依赖风险）。
- **H2 细分**：同一 backend 内初始状态**完全确定且可复现**；**跨 backend 则不同**（54/56 层 sum 不同）。

---

## 5. 下一步：一个优先级最高、可验证的修复候选

**候选（优先做、成本低、可判定）**：把 prefill 路径**统一到唯一实现**，并把它写进生产入口契约 ——
即让 `spec_dec._prefill` 的**默认实现本身就是分块 + 显式 `phase='prefill'`**
（而不是依赖 `DecisionHooks` 替换），并给 `spec_decode_greedy` 增加一个
**入口自检**：若在 `cudnn_prefill` 策略下检测到 prefill 阶段出现 `eager` 调用或 `s_max['prefill'] > CHUNK`，
就**报错停止**，而不是静默地走单次 eager。

**理由**：本轮已证明 (a) 同 backend 内初始状态可复现，(b) 不同 prefill 路径会给出**不同初始 KV**
（54/56 层 sum 不同）并**改变后续轨迹**（2/25 位置），(c) 目前是否有入口会走默认单次 eager 只靠**调用方是否安装 hooks**，
没有任何自检把关。统一 + 自检把「实验条件是否可比」从**约定**变成**可判定的运行时不变量**，
且**不改任何数值算法、不放宽任何门槛**。

**可验证判据（预登记）**：① 全部调用 `spec_decode_greedy` 的入口在 8K 下 `s_max['prefill'] == CHUNK` 且
`fused_by_phase == ['prefill']`；② 同一 prompt 的初始 KV 在两次独立运行间逐元素相同；
③ 修复**不改变**任何既有 PASS/FAIL 记录的语义（旧记录保留，重跑标记为新配置）。

**本轮的即时小修（已完成）**：产物里的内容 hash 改用**稳定**算法（`sha256` on bytes，跨进程可比），
不再用进程内 `hash()`；并**用同一脚本重跑了一次 C 轴**（`…_prefillsb_20260930T181447Z.json`，**complete**），
结果与首跑**逐位一致**（首次非零 logits 差异 @8193、首次 argmax 分歧 @8217、两个复现门都通过、
判定表数值完全相同）⇒ 该小修**未改变任何结论**，且此后跨臂比较初始 KV 可直接比 sha，而不是只比 sum。
首跑记录 `…_prefillsb_20260930T181032Z.json` 保留未覆盖。

---

## 6. 交付与纪律

- 本轮**唯一新增的 GPU 运行**：2 次（C 轴 complete；E 轴 checks_failed，均保留原始 JSON/PT，未覆盖）；
- 串行执行、无并行 GPU 作业、无自动重试；commit 护栏未上调；8K eager **分块**实测**未 OOM**（如实记录）；
- 只保存必要快照（初始状态摘要 + 每位置 logits 行 + L27 K/V），未保存全层 attention 矩阵；
- **不做**速度声明；**不改**生产实现/精度/KV/门槛；**正式正确性状态保持不变（未通过）**；
- 本轮结束后停止：不自动追加逐层扫描、精度组合扫描、手写 CUDA 算子，也不放宽正确性标准。

### 已完成的文档修正
- token **576 的文本是 `' The'`**（不是换行符）；`715=' \n'`、`4710=' \n\n'`。
  本报告与 `docs/l27_attn_tail_ab_2026-09-30.md` 的口径已修正；
  **不得把 topk 的并列展示顺序当作 argmax 规则**（本例 topk 首位落在 4710，而 argmax 是 576）；
  工具已改为**按 argmax id 单独查表**取文本，并把该注意事项写进产物字段与打印。

---

## 7. ★ 治理守卫被正确触发（**新出现的 3 个 FAIL，必须如实登记**）

本轮新增的诊断工具消费了 manifest family，触发了既有治理守卫：

```
verify_family_independence_cpu:  16 PASS / 0 FAIL  →  13 PASS / 3 FAIL
[FAIL] ★ 没有**预期之外**的消费方（唯一 opt-in 消费方是 g5_8k_smoke_cudnn.py，默认关闭）
[FAIL] 消费方集合 = 预期的 opt-in 集合
[FAIL] ★ **授权范围之外**没有任何 run 消费 manifest family（授权范围 = fam16 16 格 + F06/index25 定点取证）
```

**原因（不是引擎或正确性问题）**：
- `verify_family_independence_cpu.py:86-87` 的 `OPT_IN_CONSUMERS = {g5_8k_smoke_cudnn.py, g5_f06_index25_forensics.py}`
  与 `:248` 的 `authorize_tools = ("g5_f06_index25_forensics",)` 是**显式的授权名单**；
  本轮新工具 `prefill_sb_replay_f06.py` 消费 `family_input.load_family_ids` 并在产物里记录 `family`，
  但它**不在名单里**，其 run 记录的 `tool` 也不是已授权前缀 ⇒ 三条检查同时命中。

**处置（本轮的决定）**：**不**修改守卫、**不**自行把工具加入授权名单。
理由：该守卫的作用正是「任何新的 manifest 消费方都必须被显式授权」；在看到失败后自行放宽，
等同于把治理门槛按结果调整。**这就意味着本套件在注册完成前保持 13/3。**

**注册所需的最小改动（待批准，未执行）**：把 `prefill_sb_replay_f06.py` 加入
`OPT_IN_CONSUMERS`；并把 `authorize_tools` 扩展为同时接受
`("g5_f06_index25_forensics", "prefill_sb_replay_f06")`（本工具属于**同一** F06/index25 定点取证范围，
未引入任何新 family）。**未**授权前，本轮的 C/E 运行记录只作为**诊断产物**保留，不作为治理意义上的「授权消费」。

