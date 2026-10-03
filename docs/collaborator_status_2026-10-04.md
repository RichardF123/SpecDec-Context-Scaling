> SpecDec 课程项目 | 协作者阶段性交付 | 2026-10-04
> 状态口径：本报告汇总已有开发与诊断证据，不是正式 benchmark 结果；旧 FAIL/CHECK 不因新诊断而改写。

## 一页结论

**我们已经有可运行的 Qwen2.5 推理引擎、V-only FP8 KV、greedy 投机解码和测量工具，但尚未开始正式的多 family 性能实验。** 当前最重要的工作不是再加一个优化开关，而是批准分层正确性协议，并在正式配置下分别验证 BF16 与 V-only FP8。已生成五页课程 proposal 提交版 PDF，**截至本报告未确认已提交**。

| 工作流 | 当前状态 | 已验证的边界 |
| --- | --- | --- |
| 自写引擎与 target 1.5B | L1/L2、G1 历史 PASS | CPU FP32 参考对齐；BF16 GPU 512-token HF 对齐；不代表任意长度/后端均逐位相同 |
| 8K 运行 | G2 单模型 PASS；G4-B3 双模型可行性 PASS | 能完成 smoke，不等于长期稳定、可重复的正式 8K 计时 |
| KV 压缩 | G3 V-only FP8 方案已实现并测质量 | K 保留 BF16，V 用 FP8；逻辑 KV 约 1.32x，整机收益有限 |
| SpecDec | G4-B5 主循环与 B6 FP32 PASS；BF16 历史门 CHECK | serial oracle 在 5 个 BF16 family 上逐 token 匹配 AR；production batched 路径仍可重复地产生分歧 |
| 测量系统与正式实验 | G5 工具已有；pilot/final 未验收 | 环境、8K 稳定性、正确性口径及记录字段需先过门；不得声明正式加速 |

## 研究问题与实验范围

课程研究问题已按导师意见收窄为：**在固定模型、精度、提案长度和推理配置下，上下文长度如何影响 greedy SpecDec 相对 AR 的解码速度？** 固定 RTX 4060 Laptop 8GB、Qwen2.5-1.5B-Instruct target、Qwen2.5-0.5B-Instruct draft、BF16 计算、`k=4`、chunked prefill 2048、最多生成 64 token。主实验是 2K/4K/8K × AR/Spec 的 **6 个 BF16 KV 条件**；另设 8K × AR/Spec 的 **2 个 V-only FP8 KV 附加条件**。至少 16 个 prompt families，同一 family 使用嵌套前缀；每条件 1 次 warm-up、5 次测量，分布在至少两个随机化 block。FP8 不是第二条主研究轴，不恢复历史 12 条件 factorial。

正式 AR 仅 target 常驻；SpecDec 同时常驻 target 与 draft。正式速度只取 production batched verification，不把 serial oracle、row-aligned 诊断或 profiler 插桩耗时混入分子分母。若两条生成轨迹不同，速度比只能作为**相同输入/配置、不同轨迹的计时比较**，不能声称“相同输出的无损加速”。上述实验计划来自课程 proposal；尚未证明所有入口条件已满足。

## 已完成的实现与可复核数字

| 阶段 | 做了什么 | 结果和单位 | 如何解读 |
| --- | --- | --- | --- |
| L1/L2 | 实现 Embedding、RMSNorm、RoPE、GQA、SwiGLU、KV cache、LM head 与 greedy generate，并与参考模型对齐 | 0.5B 前向与缓存/生成历史 PASS | 证明基础数据链路和索引语义，不是速度结果 |
| G1 | 复制 Qwen2.5-1.5B 权重，测 full-forward、cached-vs-full 与生成 | CPU FP32 最大 logits 差约 `3.1e-05` / `2.0e-05`；4/4 生成一致 | 1.5B target 的历史正确性门 PASS |
| GPU BF16 基线 | 512-token ours/HF 对照 | logits 最大差 `0.000`、argmax 100%、NLL/PPL 一致 | 本机 FP16 路径曾产生 NaN，正式质量口径改用 BF16 |
| G2/G4-B3 | 8K chunked prefill；1.5B+0.5B 双模型驻留 smoke | 单模型 G2 三轮稳定、峰值约 `6.83 GiB`；B3 四配置 PASS，8K 双模型 PyTorch allocated peak 约 `6.16 GiB` | 脚本、阶段和统计口径不同，两个峰值不能直接作优化前后比较；均不等于正式实验可行 |
| G3 | 对 target KV 做非对称压缩 | K BF16，V E4M3，per-token/per-KV-head FP32 scale；8K target KV `224.00 -> 169.75 MiB`，约 `1.32x` | 仅逻辑 KV 字节，非整模型显存；8K 运行本身不依赖 FP8 |
| G3 质量 | WikiText-2 test，40 个不重叠 2048-token 窗口 | V-only `Delta NLL +0.000397`、相对 `Delta PPL +0.040%`；历史 paired eager 峰值只降约 `0.06 GiB`，decode 未加速 | 质量结论限既有配置；K-only/full FP8 是负结果，K-FP8 pilot `Delta NLL >= +2.4`，不可外推到其他量化法 |
| G4-B4/B5 | cache trim/consistent、proposal/verify/accept/repair、EOS 回归 | 结构单测与主循环 smoke PASS | 控制逻辑已实现；不等于生产 BF16 路径与 AR 总是相同输出 |
| G4-B6 | exact-token 对照 | CPU FP32 `20/20` 组合 PASS；旧 GPU BF16 `19/20`，CHECK | 旧 BF16 分歧记录保留，不以 margin 自动豁免 |

这里的 `GiB` 是 2^30 字节、`MiB` 是 2^20 字节；`allocated` 是 PyTorch 当前/峰值跟踪的张量分配，`reserved` 是分配器池，设备侧占用又是另一口径。`NLL` 越低越好，`PPL = exp(mean NLL)`；`Delta PPL` 是相对未压缩 BF16 KV 的变化。不同脚本、GPU 状态、backend 或生成轨迹下的秒数不能直接拼接为正式 speedup。

## 正确性诊断：查到了什么，没查到什么

生产 SpecDec 一轮让 draft 提议最多 4 token，target 以 `M=4` 批量验证；AR 则让同一 target 连续以 `M=1` 逐 token 前向。两者数学目标相同，但 BF16 下**执行形状不同不保证数值逐位相同**。此前 F13 等输入出现可重复的 AR/production token 分歧；例如 F13 首分歧在生成索引 47，另外四个 protocol-validation families 的首分歧分别是 F01@14、F02@7、F08@52、F10@8。这些不是正式 family-level 分歧率，也不能笼统归罪于 cuDNN。

同一历史的局部因果实验确认，attention QK 以及 `o_proj`、`gate_proj`、`up_proj`、`down_proj` 五类操作在被测样本上存在 `M=1`/`M=4` 形状敏感性。控制内容和布局后差异仍出现；**没有捕获到底层 CUDA kernel 名或 tile 证据，也没有证明五个操作解释每一次生产分歧**。row-aligned 的局部修复在另三/四个 protocol families 上未通过算法比较，历史 FAIL 保留。

随后用诊断性 **serial target oracle** 保持 accept、replacement、rollback、cache-commit 逻辑，却把 target verification 改为逐 token `M=1`。F13、F01、F02、F08、F10 共 5 个 BF16 family 各生成 64 token，与各自 target-only AR **5/5 完整 token IDs 相同**，两次独立进程重复一致；调用几何审计显示生成期 `s=1`、零次 block verification，cache/round/finite 检查通过。这支持**这些输入上的控制逻辑**，不是所有输入的证明，更不是 V-only FP8 的证据。serial 耗时只用于诊断，不进入性能表。

生产 BF16 batched verification 仍可能分歧。因此“serial exact IDs PASS”不能被写成“production exact IDs PASS”。分层正确性协议 `docs/specdec_correctness_protocol_v2_2026-10-04.md` 仍为 **DRAFT / NEEDS_DECISION**：把 exact-token 门放到 serial oracle 是执行路径上的实质改变，须由小组确认，并向导师如实说明；旧 B6-bf16 CHECK 19/20 不改写。

## 测量系统、可行性与未完成工作

已实现独立进程运行、P-state/系统 commit 预检、warm-up、分阶段计时、生成 token 与停止原因、逐轮接受统计、cache/finite/结构检查、PyTorch allocated/reserved 峰值和设备侧内存记录、失败落盘及单独 profiler pass。计时与插桩分离，产物须带配置、版本与输入溯源。测量工具“能记录”不代表正式 16-family 矩阵已经运行，也不代表 8K 双模型长期稳定：历史 8K eager 测量曾有 OOM、原生崩溃和显著时长波动；prefill-only cuDNN 是可选内存缓解路径，尚须按最终配置完成正确性、稳定性与口径验收。

探索性 2K BF16 测量曾得到 decode speedup 约 `0.406x`（Spec/AR 吞吐比），说明那组输入和当时配置**未加速**；这不是主实验结论。8K 探针速度波动大，不能据此宣布长上下文已出现或消失加速。不得以单次无 OOM 或单算子节省替代完整双模型正式运行证据。

**接下来按依赖顺序：**

1. 小组确认分层正确性路线、production 分歧如何标注、质量评估口径；不把 v2 草案自动视为已冻结，也不采用缺乏证据的“首分歧 index=0”自动门槛。
2. 预登记并核对新 family 的来源与 hash，分别补 BF16 和 V-only FP8 的 Gate A/Gate B 证据；任一 finite、结构、重复性或合法停止检查失败即停并保留原始产物。
3. 固定唯一正式 attention/precision 配置，在四-family pilot 中检验 2K/4K/8K 的资源、记录字段、稳定性和计时边界；8K 失败不得用改参数后混表掩盖。
4. 门槛通过后运行六个 BF16 主条件；两格 8K FP8 附加条件另行过门。汇总 paired family-level 中位数与 IQR，报告 acceptance、输出分歧、失败和显存；不挑最快 run。

建议分工与 proposal 一致：Fang Chuming 负责引擎/target/KV 与配置回归；Deng Xiyue 负责 SpecDec/cache/生产分歧和接受率测量；Cheng Shuyue 负责 prompt families、编排、schema、重复性、汇总和图表。所有人共同审阅协议变更、正确性与最终解释。

## 证据索引与复现边界

| 主题 | 仓库记录 |
| --- | --- |
| 课程范围和角色 | `docs/specdec_course_proposal_2026-10-04_final.tex`；`docs/pdf/specdec_course_proposal_2026-10-04_final.pdf`；提交状态待确认 |
| 历史实验数字 | `docs/data_ledger.md`；`docs/daily_report_2026-09-08b_G1G2G3_FP8KV.md`；`docs/plan_g4_next_steps.md` |
| 正确性阶段 | `docs/serial_oracle_stage1_f13_2026-10-03.md`；`docs/serial_oracle_stage2_2026-10-03.md`；`docs/row_aligned_candidate_report_2026-10-03.md` |
| 形状敏感性 | `docs/gemm_shape_sensitivity_mechanism_2026-10-03.md`；`docs/div8209_serial_l0_closure_2026-10-02.md` |
| 协议待决 | `docs/specdec_correctness_protocol_v2_2026-10-04.md`；`docs/specdec_protocol_v2_evidence_inventory_2026-10-04.md` |
| 正式测量口径 | `docs/g5_measurement_protocol.md`；`docs/prompt_families_manifest.json`；`docs/cudnn_prefill_stage_report.md` |

若运行配置或 backend 改变，请重新验证相应正确性与质量，不把旧结果直接迁移；任何新的测试结论应指向新产物，而非覆盖历史 FAIL/CHECK。
