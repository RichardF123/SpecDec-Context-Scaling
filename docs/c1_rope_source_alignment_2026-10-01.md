# 仅对齐 RoPE 频率来源：C1 跨实现差异是否消失？（2026-10-01）

> **本轮只回答一个问题**："把 `inv_freq` 的来源换成 CPU-FP32（本机 HF 的实际公式）之后，
> 自研**完整 prefill** 的 C1 缓存与 logits 是否与 HF 一致？"
> **不**宣称 CPU 更准确；**不**预设能解决 AR/Spec 分歧。
> **B6 未过 ⇒ 仍禁止 acceptance 测量/pilot/final；正式正确性状态不变（未通过）。**
> 前置：b0db582（L0 K 定位，`docs/l0_k_localization_2026-10-01.md`）。

## 0. 结论（严格限定）

**是。** 仅把 `inv_freq` 的计算搬到 **CPU-FP32**（HF 的公式）、其余逐行沿用自研原实现之后：

| 比较 | K/V（28 层 ×(k,v) = 56 张量） | C1 末位 logits（151,936 维） |
| --- | --- | --- |
| **新自研 vs HF** | **56/56 逐位相同**，不同元素 **0**，max\|Δ\|=**0** | **逐位相同**，不同元素 **0/151936**，max\|Δ\|=**0** |
| 旧自研 vs HF | 1/56 逐位相同，不同元素 **83,453,336**，max\|Δ\|=1.10938 | 不同元素 **139,637**，max\|Δ\|=**0.1875**，mean 0.0416068 |
| L0 K（此前的 60 个差异） | 旧 vs HF = **60** ⇒ 新 vs HF = **0**（**已消除**） | — |

按预先声明的口径：**"仅对齐频率来源，在该样本与该配置下足以消除 C1 跨实现差异。"**

**限定范围（必须与结论一起引用）**：该结论只在 **F13 / pre=8240 序列的前 8192 个 token / ctx=8192 / chunk=2048 /
eager / bf16 / reduction=True / QK32·PROB32 关闭** 这一组条件下成立；**不**推及其他样本、长度、chunk 日程、backend 或精度开关。

**不宣布（本轮明确未证明）**：AR/Spec 严格一致性已修复；所有输入或 backend 已通过；正式实验可以启动；
"此前所有差异只有这一个原因"。

**必须一起说明的副作用**：对齐之后，自研 prefill 与**自研自己的历史捕获**（例如 F13 的 AR/Spec 捕获与
b0db582 之前的全部自研数值）**不再逐位可比**——`新自研 vs 旧自研` 有 **83,453,336** 个 K/V 元素与
139,637 个 logits 维度不同（与 `旧 vs HF` 完全同量）。

---

## 1. 边界与固定引用（先核验，再动手）

| 项 | 值 / 路径 | 核验结果 |
| --- | --- | --- |
| 旧自研 C1 | `output/g5/attn/xb_20261001_ours_20261001T132915Z.json/.pt` | 固定路径；`seq_sha16=679ad27f88c66887` |
| HF C1 | `output/g5/attn/xb_20261001_hf_20261001T132947Z.json/.pt` | 固定路径；同 hash |
| HF 侧重放记录（config + 实际 `inv_freq`） | `output/g5/attn/l0_replay_hf_20261001T143734Z.json` | `theta=1000000.0`、`head_dim=128`（config.json **无** `head_dim` 键，由 hidden/heads 推出） |
| HF 实际使用的 cos/sin | `output/g5/attn/l0_replay_hf_dump_20261001T143734Z.pt` | 该重放曾与其 C1 快照**逐位复现** ⇒ 就是 HF 实际表 |
| 自研侧实际 cos/sin | `output/g5/attn/l0_replay_ours_dump_20261001T143510Z.pt` | 同上（用于还原核查） |
| 输入 | `family_input.load_family_ids(tok,"F13",ctx=8192)` + 捕获的前 48 token | 重算 `seq_sha16` **逐位一致** |
| 权重 | `lowmem_load.load_ours_bf16`（HF state_dict 去 `model.`、tied 去 `lm_head`、strict 载入） | **338/338 逐张量位级一致**，`mismatch=0` |
| dtype / KV | bf16 / 未量化（`set_kv_format("fp16")`=base_bf16） | 记录在产物 |
| 日程 | `[(0,2048),(2048,4096),(4096,6144),(6144,8192)]`，空缓存起步 | 与旧 C1 完全一致 |
| 精度开关 | `allow_bf16_reduced_precision_reduction=True`、`allow_tf32=False`（实测） | 与旧 C1 一致；未改 |

**未**安装/升级任何依赖；**未**使用 F13 之外的 family；**未**覆盖任何历史文件（所有新产物带时间戳）。

---

## 2. 唯一干预（接入点、原理、恢复方式）

**接入点**：`model_self.QwenLM._run` 里的一句（`model_self.py:489`）

```python
cos, sin = Build_Rope_cache(self.head_dim, seq_len=x.size(1), theta=self.rope_theta,
                           device=x.device, dtype=x.dtype, start_pos=start_pos)
```

全仓库只有这一处在 prefill 路径上调用它（另外两处 `align_debug_attn.py` / `fp8_kv_group_probe.py`
是各自独立的诊断，本工具**不动**它们）。

**原理**：诊断入口用一个**逐行复刻**的函数替换模块属性 `model_self.Build_Rope_cache`，
其中**只有第一行不同**：

```python
# 原实现（自研）：在 GPU 上算         | 诊断版：在 CPU-FP32 上按 HF 公式算，再 .to(device) 只做搬运
exponents = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
inv_freq  = 1.0 / (theta ** exponents)   |  inv_freq = 1.0/(theta ** (torch.arange(0,head_dim,2,dtype=torch.float)/head_dim))
                                         |  inv_freq = inv_freq.to(device=device, dtype=torch.float32)
postitions = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=calc_dtype)   # 原样
freqs = torch.outer(postitions, inv_freq)                                                    # 原样
cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], -1).unsqueeze(0).unsqueeze(0)          # 原样
sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], -1).unsqueeze(0).unsqueeze(0)          # 原样
return cos.to(dtype), sin.to(dtype)                                                          # 原样（BF16 转换）
```

⇒ 位置计算、`outer`、`cos/sin`、BF16 转换、`apply_rope` 全部沿用自研原实现；差异源**只有** `inv_freq` 的计算设备。

**Q/K 同时生效**：`_run` 每次只算**一份** cos/sin 并传给每一层；`SelfAttention.forward`
（`model_self.py:354-355`）用**同一个** cos/sin 对 q 与 k 各调一次 `apply_rope` ⇒ 不是"只改 K"。

**不改**：权重、BF16、`allow_bf16_reduced_precision_reduction`、attention backend、mask、
投影、旋转实现（`apply_rope`/`rotate_half`）、argmax；**不回填**任何参考缓存/hidden/logits。

**恢复方式与无残留核查**：
1. 只替换**模块属性**，进程内 `try/finally` 还原；
2. `M.Build_Rope_cache is orig` ⇒ `restored=True`；
3. `restore_check`：用**还原后的原函数**重新生成 chunk-0 的表，与 b0db582 落盘的**自研侧实际表**逐位比较 ⇒
   **True/True**（cos 与 sin 都逐位相同）——同时证明"还原成功"与"相对基线只改了那一处"；
4. 本诊断不写任何模块级/磁盘全局状态（产物只有结果 JSON/PT），进程退出即完全消失。

---

## 3. 先过表验证门（**不加载模型**）

产物 `output/g5/attn/c1rope_gate_20261001T151856Z.json`（`stage="complete"`，`gate_pass=true`）。

| 检查 | 结果 |
| --- | --- |
| CPU-FP32 `inv_freq`（HF 公式） vs **已记录的 HF** `inv_freq` | **逐位相同**（sha16 均为 `138c99b109d7affb`） |
| GPU 算的 `inv_freq` vs 已记录的 HF | **不同**（sha `d0260aaaaed04542`），差异下标 **[18, 50]** |
| 每个 chunk 的 `cos` vs **HF 实际表** | **逐位相同**（4/4） |
| 每个 chunk 的 `sin` vs **HF 实际表** | **逐位相同**（4/4） |
| 每个 chunk 的 `cos/sin` vs **自研旧表** | 不同 6/8、24/14、42/28、42/38（与 b0db582 的记录一致，作为对照） |
| dtype / shape / 位置覆盖 | `torch.bfloat16`、`(1,1,2048,128)`、位置 `[st, en)` 与 HF dump 逐位一致 |
| NaN / Inf | 0（cos 与 sin 各 4 个 chunk） |

**布局归一化按语义进行**：自研表是 `(1,1,seq,dim)`、HF 表是 `(1,seq,dim)`；比较前统一 `reshape(-1, dim)`
到 `(seq,dim)`，并把 `layout` 与"数值是否相同"**分开记录**，不把形状差异当成数值差异。

任一不一致即落盘停止、不继续完整 prefill、不追加其它修改（实际**未触发**）。

---

## 4. 一次 C1 诊断运行

产物：`output/g5/attn/c1rope_run_20261001T152125Z.json` + `c1rope_new_dump_20261001T152125Z.pt`。

* **配置**：ctx=8192、chunk=2048、四个完整 chunk、空缓存起步、**到长度 8192 立即停止**；
  不加载 draft、不跑 C2、不做 decode；`set_kv_format("fp16")`、`set_diag_preset("eager")`、
  `set_attention_strategy("eager")`、`reduction=True`；QK32/PROB32 = False/False（记录在产物）。
* **干预调用记录**：`Build_Rope_cache` 被调用 **4 次**，参数 `(head_dim=128, seq_len=2048, theta=1e6,
  device=cuda, dtype=bfloat16, start_pos=0/2048/4096/6144)`；`restored=True`；`restore_check` True/True。
* **结构检查**：cache 长度 **8192**、层数 **28**；每层 `k/v` 形状 `(1,2,8192,128)`、bf16、
  长度 8192、**`isfinite` 全 True**；存的 K/V 是**精确 bf16 值**（断言通过）。
* **结果**：末位 logits `argmax=220`、top-2 margin **2.125**、并列最大 1 个
  （对照：旧自研 margin **2.25**、HF margin **2.125**）。
* **后端/阶段记录**：`attention_stats()` 与 `attention_phase("prefill")` 一并落盘。
* **资源**：峰值分配/保留显存与剩余显存记录在产物中，**仅诊断，不作性能声明**。
* 全部现有护栏（commit guard）通过；无 OOM、无非有限、无结构错误 ⇒ 未触发任何停止条件。

---

## 5. 三方比较（**只读**）

产物 `output/g5/attn/c1rope_compare_20261001T152219Z.json`。三份输入的 `seq_sha16` 均相同。

### 5.1 K/V：全部 56 个张量

| 比较 | 逐位相同张量 | 不同元素合计 | max\|Δ\| |
| --- | --- | --- | --- |
| 旧自研 vs HF | **1/56**（唯一逐位相等的是 **L0 的 V**） | **83,453,336** | 1.10938 |
| **新自研 vs HF** | **56/56** | **0** | **0** |
| 新自研 vs 旧自研 | **1/56**（同样是 L0 的 V） | **83,453,336** | 1.10938 |

**逐层传播（旧自研 vs HF，K+V 合计不同元素数，每层上限 4,194,304）**：

| 层 | 0 | 1 | 2 | 3 | 4 | 7 | 10 | 15 | 20 | 25 | 27 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 不同元素 | **60** | 52,980 | 1,705,648 | 2,652,907 | 2,972,088 | 3,078,842 | 3,325,929 | 3,352,623 | 3,376,772 | 3,376,997 | 3,374,171 |

即：L0 的 **60** 个元素差异在 3 层内扩散到该层的多数元素；对齐频率来源后**全部归零**（56/56 逐位相同）。

### 5.2 C1 末位 logits（151,936 维）

| 比较 | 逐位相同 | 不同元素 | max\|Δ\| | mean\|Δ\| | argmax | margin A/B | 并列最大 A/B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 旧自研 vs HF | 否 | 139,637 | **0.1875** | 0.0416068 | 220 / 220 | **2.25 / 2.125** | 1 / 1 |
| **新自研 vs HF** | **是** | **0** | **0** | **0** | 220 / 220 | **2.125 / 2.125** | 1 / 1 |
| 新自研 vs 旧自研 | 否 | 139,637 | 0.1875 | 0.0416068 | 220 / 220 | 2.125 / 2.25 | 1 / 1 |

### 5.3 L0 K：此前的 60 个差异

`旧自研 vs HF = 60` ⇒ `新自研 vs HF = 0`；`新自研 vs 旧自研 = 60` ⇒ **60 个差异已消除**。

### 5.4 内部一致性交叉校验

`新自研 vs 旧自研` 与 `旧自研 vs HF` 的 **max|Δ| 与计数完全相同**（K/V 1.10938 / 83,453,336；
logits 0.1875 / 139,637），符合"新自研 == HF"这一结论的预期（否则说明还有第三种数值状态）。

### 5.5 与既有 digest 字段的独立核对

两次既有产物里的逐层 K sha（`kv_digest.c1_sha`，旧代码路径算的）与新运行的 `layers[i].k_sha` 对比：

| 层 | 旧自研 C1 | HF C1 | **新运行（对齐后）** |
| --- | --- | --- | --- |
| L0 K | `b90a990b291ff464` | `88a6a976e83e1566` | **`88a6a976e83e1566`（== HF）** |
| L1 K | `bf49ac6cde1816ac` | `6411ad870148d169` | **`6411ad870148d169`（== HF）** |
| L2 K | `8f3c993ce3947f5d` | `9f412739bf95ae11` | **`9f412739bf95ae11`（== HF）** |

⇒ 这不依赖本轮新增的比较代码，是同一条结论的**独立证据**。

---

## 6. 结果解释（按预先声明的口径）

* 因为**全部缓存与 logits 都逐位相同**，结论写成：
  **"仅对齐频率来源，在该样本与该配置下足以消除 C1 跨实现差异。"**
* **不是**"部分相同"的情形，因此没有"把 max|Δ| 下降称为完全修复"的风险；本轮也不需要分开报告剩余差异。
* L0 差异**已消除**，因此不触发"停止并核查干预是否忠实执行"的分支（该核查仍然做了：`restore_check` True/True、
  调用记录 4 次、参数与日程一致）。
* **即使全通过也不宣布**：AR/Spec 严格一致性已修复 / 所有输入或 backend 已通过 / 正式实验可以启动 /
  此前所有差异只有这一个原因。

---

## 7. 未验证事项（明确列出）

1. **其他条件未覆盖**：其它 family、其它上下文长度、其它 chunk 日程、`sdpa`/`flash` 等 backend、
   QK32/PROB32 打开、KV 量化档、CPU 侧生成 —— 全部**未测**。
2. **C2（len 8240）与两臂（serial/block）未测**：本轮按边界只到 8192，没有喂那 48 个 token、没有做单步/块调用。
3. **AR/Spec 内部一致性未测**：F13 在 8242 处的 AR 17 vs Spec 20 是否受影响**完全未知**；
   而且对齐后自研数值与**历史捕获不再逐位可比**，任何复测都必须与历史产物**并列保存**、不得混比。
4. **生产默认未改**：`Build_Rope_cache` 仍是 GPU 计算 `inv_freq`。是否改为 CPU-FP32 是**产品决策**，
   需要单独授权；它会把本仓库所有历史自研数值重新基线化。
5. **"哪一方更准"未判定**：两侧都是 1-ULP 级的 `pow`/除法舍入差别；本轮结论只是"与 HF 参考逐位一致"，
   **不**等于"自研此前算错"，也**不**等于"CPU 一定更准"。
6. **未做历史结论审计**：没有回看此前的诊断/四格/UCG 结论是否受该差异影响（本轮不扩大范围）。

---

## 8. 失败记录（全部保留，不覆盖）

| # | 阶段 | 现象 | 影响 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | gate 首次 | 读旧 xb HF 记录里的 `hj["config"]`（该记录**没有** config 字段）⇒ `KeyError: 'config'` | **未加载模型**，无任何计算 | 保留日志；改从 b0db582 的 HF 重放记录取 config/精度，并做三方 hash 交叉核对 |
| 2 | run 首次 | 包装函数参数名写成 `theta_`，而 `_run` 用 `theta=` **关键字**调用 ⇒ `unexpected keyword argument 'theta'` | 崩溃**在任何层前向之前**（0 层计算、无缓存建立）；产物 `c1rope_run_20261001T151926Z.json` 保留 | 参数名与 `Build_Rope_cache` 完全一致 |
| 3 | run 第二次 | **prefill 已算完**，在保存路径写了 `o["c1_kv"]`（应为 `out`）⇒ `NameError` | 完整 prefill 执行了但**数值未落盘**；产物 `c1rope_run_20261001T152019Z.json` 保留 | 修 typo 后执行；**第三次**执行的产物 `c1rope_run_20261001T152125Z.json` 才是交付件。**如实记录**：本轮共执行了 2 次完整 prefill（第二次无产出），干预完全相同 |
| 4 | compare 首次 | 读 `rj["run_id"]`（本工具记录用 `run_utc`）⇒ `KeyError: 'run_id'` | 纯读取失败 | 改 `rj.get("run_id") or rj.get("run_utc")` |

另：`compare` 现在会**拒绝**把非 `stage=="complete"` 或缺 `dump_pt` 的运行当作比较对象。

---

## 9. 唯一推荐的下一步

> **在已消费的 F13 上，做一次有界的 AR/Spec 自洽性复测（对齐频率来源之后）**：
> 用**同一个**诊断开关（只改 `inv_freq` 来源）跑一次既有的 F13 定点协议（`pre=8240`、`Kp=4`、`m=2`，
> target-only，沿用既有捕获/对照工具），只回答一件事：
> **"prefill 与 HF 逐位一致之后，引擎内部的 AR 与 Spec 在 8242 处的分歧（17 vs 20）是否仍然存在？"**
>
> * 为什么是它：本轮把"引擎 ↔ HF 参考"这条线彻底对齐了，剩下的**唯一未回答**的主线问题是引擎**内部**
>   AR/Spec 自洽性；它恰好也是 B6 严格门所在的那条线。
> * 二值可判：**分歧仍在** ⇒ 说明该分歧是引擎内部问题、与 HF 参考差分无关；**分歧消失** ⇒ 说明此前观测到的
>   分歧由 RoPE 表差异引发，此前的四格结论需要重新解释。
> * 上限与边界：**1 组**诊断运行；不跑正式矩阵、不做验收测量、不做性能测试、不跑 FP8；
>   历史捕获**绝不覆盖**（新旧数值不可混比，需并列保存）；**需单独授权**（本轮边界明确禁止生成/C2/SpecDec）；
>   仍未过 B6 ⇒ 不进入 acceptance/pilot/final。

---

## 10. 治理与未改项

* `verify_family_independence_cpu` **24 PASS / 0 FAIL**：新登记消费方 `c1_rope_source_diag.py`
  （**范围仅限该工具 + F13**，并新增对应范围检查）；**未**放宽任何既有检查；**未**新增 family。
* **生产默认 `Build_Rope_cache` 未改**（仍是 GPU 计算 `inv_freq`）；权重、BF16、归约精度开关、
  attention backend、mask、投影、旋转实现、argmax **均未改**；
  `proposal_freeze_2026-10-01.md`（冻结 6 主 + 2 补充）与严格 AR/Spec 一致门（B6）**均未改**。
* **未**做生成、C2、SpecDec、FP8、性能测试、正式矩阵；未安装/升级依赖；未使用其它 family。
* 所有新产物带时间戳、**未覆盖**任何历史文件；失败记录与日志（`_c1r_*.log`）保留。
