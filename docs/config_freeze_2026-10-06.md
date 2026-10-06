# 候选配置冻结记录（2026-10-06）

> 本记录把**正式实验的候选配置**固定下来，供用户编写 runner 时对照。
> **配置冻结 ≠ `BF16_FORMAL_READY`**：它只表示"正式运行要用这一套配置"，**不**表示 pilot/正式采集已获准，
> 也**不**表示任何正确性/性能结论。
> 相关文档：`docs/bf16_formal_admission_2026-10-05.md`（准入表）、`docs/formal_runner_spec_2026-10-06.md`（runner 规格）、
> `docs/g5_measurement_protocol.md`（计时与记账口径，已按本轮授权修正四处措辞）。

## 0. 引擎基准 commit

| 项 | 值 |
| --- | --- |
| **`engine_baseline_commit`** | **`d296cd852b6e58eaeea77948d165753e25487547`**（`d296cd8`） |
| 该 commit 包含 | 正式集合构建/校验工具、`docs/prompt_families_formal_manifest.json`、守卫登记、测量协议四处措辞修正 |
| 本记录自身 | **只新增文档**（不改代码、不改配置、不改数据规则）⇒ 不改变 baseline |
| 正式运行的记录要求 | **逐 run** 记录 `git_commit` 与 `dirty_worktree`；必须为 `dirty_worktree=False`；报告中必须写明实际 commit |
| commit 与 baseline 不一致时 | **只允许**"仅新增文档"的提交；任何触及**配置 / 数值路径 / 数据规则 / 门限**的提交 ⇒ **必须重新冻结**并登记新 baseline（附重新自测结果） |

## 1. 冻结值（逐项）

| 项 | 冻结值 | 来源 / 依据 |
| --- | --- | --- |
| target 模型 | `Qwen/Qwen2.5-1.5B-Instruct` | `proposal_freeze_2026-10-01.md`（FROZEN） |
| draft 模型 | `Qwen/Qwen2.5-0.5B-Instruct` | 同上 |
| compute dtype | **BF16** | 同上 |
| target KV | `base_bf16` | 同上（BF16 主条件 C1–C6） |
| draft KV | `base_bf16`（恒 BF16 未量化） | 同上 |
| **backend / 阶段策略** | **`cudnn_prefill`**：**仅 prefill 融合**（`s ≥ PREFILL_FUSED_MIN_S = 2`），**其余阶段 eager** | 本轮授权固定；既有证据：`div8209_b6` 66 次运行（`fused_by_phase={'prefill':28/52}`、`fused_s_max=2048`）+ `F09` 8K 融合复核 + `div8209_eos` 6 次 |
| `k` | **4** | `spec_dec.K = 4` |
| `chunk` | **2048** | 冻结配置（8K 必须分块 prefill） |
| `max_new` | **64** | 冻结配置（`other_short` 判失败） |
| 解码 | **greedy（argmax）** | 冻结 |
| tokenizer | `Qwen/Qwen2.5-1.5B-Instruct`（`local_files_only=True`；`add_special_tokens=True`） | `family_input.TOKENIZER_ID` |
| EOS 规则 | **命中 EOS 立即停止并记录实际长度**；未命中且未达 `max_new` ⇒ `other_short` ⇒ **判失败** | `family_input.eos_config()`（`EOS_ID_EXPECTED=151645` 运行时核对） |
| target RoPE | **CPU FP32**（`target_rope=cpu_fp32`） | 冻结配置；本轮 66 次运行逐次核对 |
| AR 口径 | **target-only**（进程**从不加载** draft） | `g5_measurement_protocol §3.1/§3.5` |
| Spec 口径 | 双模型（target + draft，算法必需） | 同上 |
| 诊断干预 | **全部关闭**：`qk_fp32=False`、`prob_fp32=False`；**不安装** serial-QK / row-aligned wrapper | 本轮 66 次运行产物 `diag_precision_after_run` 实测一致；`row_aligned_wrap_active` 非真 |
| 进程纪律 | **一个 `(condition, family, block)` = 一个全新子进程**；父进程只编排；权重/tokenizer 加载**不进**计时 | `§3.5/§4/§5` |
| 重复 | **每 `(family, condition)` 共 5 次 measured reps**，跨 **2 个随机 block**，**3+2** | 本轮授权（与 proposal 定稿一致） |
| warm-up | **每 `(family, condition, block)` 各 1 次并丢弃**（＝每个条件单元进程 1 次） | 本轮授权 |
| block 随机化 | 每 block 内条件顺序由固定 seed `shuffle`，seed 记档可重放；**≥2** block | `§4` |
| cooldown | 条件之间默认 10 s，**不进计时** | `§5` |
| 计时边界 | 两端 `torch.cuda.synchronize()`；区间内禁 `empty_cache`/`nvidia-smi`/文件 IO/文本 decode/诊断；`prefill` 用包装区间、`decode = total − prefill` | `§5` + `docs/formal_runner_spec_2026-10-06.md` §4 |
| 显存口径 | `peak_allocated`（主）/`peak_reserved`（辅，**≠物理占用**）/`nvidia_smi_used`（物理参考） | `§6` |
| speedup 口径 | **`speedup = T_AR / T_Spec`**（分子 AR、分母 Spec；**>1 ⇒ Spec 更快**）；`AR-primary` 是**比较基线** | 本轮修正后的 `§3.1` |
| 数据：pilot | **F09 / F03 / F11 / F04**（均已用于开发；stratum 各一） | 本轮授权 |
| 数据：正式 | **`FF01–FF16`**：`docs/prompt_families_formal_manifest.json`（`formal-v1`） | 本轮构建 + 18 项 CPU 校验通过 |
| 输入重建 | 按 `source/block/window/tokenizer` **重建 token IDs**，再用 `prefixes[ctx].token_sha1` 与 `n_tokens` 校验（manifest **不存 ids**） | `formal-v1` 字段说明 + `family_input.load_family_ids(verify_hash=True)` |
| 声明资格 / 硬停 | `§7b.4` 十条声明资格；`§7b.5` 环境拒绝与补跑规则；`§7` 不稳定单元处理；v2 §5.2 Q3 硬停清单 | 已冻结，不改 |

## 2. **不在**本次冻结范围内的（独立待办，互不阻塞 BF16 主线）

| 项 | 说明 |
| --- | --- |
| V-only FP8 档（A1/A2） | 需 **自己的**控制逻辑门 + 后端变更后的 40 窗真实文本质量回归；按 per-KV 独立放行 |
| 8K 路线选择（A/B/C） | **pilot 后**按 `g5_measurement_protocol §11b` 的触发条件择一并登记；**不预判** |
| 生成质量口径 | 可选研究；**不得**用固定文本 NLL/PPL 或自生成文本 PPL 声称"分歧后质量等同" |
| 数值机制研究（五类算子等） | 解释性研究，不阻塞正式矩阵 |
| k 扫描 / 额外诊断 | 不属正式范围 |

## 3. 本记录依据的只读核对（可复查）

1. **既有运行的配置一致性**：`div8209_b6` 60 次 + `F09` 6 次 + `div8209_eos` 6 次的产物中，与用例无关的字段
   （`arm`/`k`/`chunk`/`kv`/`dtypes`/`target_rope`/EOS id）与上表**逐字段相符**；
   `max_new` 在覆盖轮/复核/补测中按**预先声明**取值（16/13/5/2、64、24），**非**运行时改动；正式矩阵一律 64。
2. **融合路径证据**：`F09` 6 次运行的 `fused_by_phase={'prefill':112}`（AR）/`{...:208}`（prod,serial）、
   `fused_s_max_by_phase={'prefill':2048}`、`input_tokens=8192`；非 prefill 融合 = 无。
3. **诊断关闭证据**：66/66 次运行的 `diag_precision_after_run = {qk_fp32: False, prob_fp32: False}` 且 `stable=True`。
4. **数据集合证据**：`formal-v1` 18/18 校验通过（身份唯一、与已消费集合身份级不相交且 8192 hash 全不同、
   三档前缀可重建、2K⊂4K⊂8K、位置不重叠、集合内分层 4/4/4/4、每 block ≤5、无 token ids 泄漏、
   `source_manifest_sha256` 与当前 `v1-draft` 一致）。

## 4. 变更流程（任何一项改动都走同一条路）

1. 记录改动内容与理由（写进新版本的冻结记录，旧记录保留不覆盖）；
2. 更新 `engine_baseline_commit`；
3. 重新跑：治理守卫（`verify_family_independence_cpu`，当前 34/0）、环境预检（`stage2b_preflight_cpu`，当前 rc=0）、
   全量 CPU 自测（当前 25/25）、正式集合校验（`scripts/verify_formal_families_cpu.py`，当前 18/0）；
4. 在报告中写明"哪一次运行用了哪一个 baseline"。
