# SpecDec：小显存 GPU 上的上下文长度与推理性能研究

在 RTX 4060 Laptop 8GB 上实现 Qwen2.5 推理引擎，研究固定模型对、精度与推理配置时，上下文长度如何影响 greedy speculative decoding 相对自回归解码（AR）的性能。DSAI 4207 课程项目。

团队实现模型前向、KV cache、量化辅助函数和投机解码控制循环；张量运算依赖 PyTorch，权重与 tokenizer 来自 Hugging Face，prefill 融合 attention 使用库提供的 cuDNN 路径。**自写推理引擎不等于所有 CUDA 算子均为手写。**

## 项目简介

大模型生成文本包含两个工作阶段：**prefill** 处理已有上下文并建立 KV cache，**decode** 使用缓存逐步生成新 token。上下文变长时，两部分的计算量、缓存访问与临时张量需求都会变化。在 8GB 笔记本 GPU 上，理论上更省 target 调用的算法，也可能被 draft 成本、缓存操作或资源压力抵消。

本项目围绕这一实际问题搭建了可以检查内部数据流的推理系统：先实现并对齐模型，再加入缓存、量化和投机解码，最后建立可复现的测量流程。我们不预设 SpecDec 必然加速，也不预设 KV 压缩必然降低整机延迟；负结果同样需要解释清楚。

| 问题 | 项目如何回答 |
|---|---|
| 上下文变长后，SpecDec 相对 AR 的收益如何变化？ | 固定模型对与推理配置，在 2K / 4K / 8K 比较 decode 与端到端延迟，这是主实验 |
| 为什么接受率高仍可能不加速？ | 结合接受长度、轮数、target 调用量和独立阶段诊断，分析省下的 target 工作能否抵消额外成本 |
| KV 压缩是否改变 8K 条件的资源与性能？ | 使用 V-only FP8 作为次要消融，分别测逻辑缓存、显存、延迟与固定文本质量 |
| 批量验证与单步 AR 是否保持同样输出？ | 区分控制逻辑正确性与有限精度数值行为，保留完整 token 序列及分歧证据 |

**为什么自建引擎？** 我们需要控制 cache 的布局与裁剪、量化发生的位置、验证块的形状和计时边界。自建引擎提供这些内部接口；参考库用于校验，而不是仅用一个整机速度数字推断原因。

## 系统如何工作

模型前向的数据流为：

```text
文本 → tokenizer → token IDs [B, S]
     → Embedding [B, S, hidden]
     → 多层 DecoderLayer（RMSNorm / attention / MLP / residual）
     → 最终 RMSNorm → LM head → logits [B, S, vocab]
     → greedy argmax → 新 token → 更新 cache → 下一步
```

`B` 是 batch 大小，`S` 是本次输入长度，不一定是全部历史长度。Attention 中的 RoPE 使用对应位置，GQA 让多个 query heads 共享 KV heads；各层缓存历史 K/V，避免每生成一步都重新计算整个前缀。因果 mask 阻止当前位置读取未来 token。

**Chunked prefill** 将长输入按块处理，并累积各层 KV cache。它不是截断上下文：后续块仍可关注已有历史。分块限制本次 query 长度，prefill-only cuDNN 融合路径进一步减少显式 attention 中间张量；这与是否启用 SpecDec 是两件不同的事。

投机解码的每轮流程为：

```text
已提交前缀
→ draft 按自回归方式提出最多 k 个候选
→ target 批量验证这些候选所对应的预测位置
→ 接受连续匹配的前缀
→ 遇到首个拒绝时，由 target 提供替换 token
→ 按实现契约处理缓存与必要修复
→ 继续下一轮，或因 EOS / 长度预算停止
```

缓存中可能暂时包含尚未提交的候选，因此 `spec_cache.py` 的长度检查和裁剪不是附属功能，而是算法正确性的一部分。未接受候选不能成为后续生成的有效历史。当前流程的细节与出口契约以 [SpecDec 设计](docs/design_freeze_g4_specdec.md) 和 [cache 契约](docs/design_g4_cache_api.md) 为准。

V-only FP8 则改变缓存的**存储表示**：K 保持 BF16；每个 token、每个 KV head 的 128 维 V 用 max-abs scale 归一化，再舍入为 E4M3。读取时反量化回计算精度。压缩来自 FP8 payload，而非简单除以 scale；scale 元数据和反量化成本都必须计入结果。

## 当前状态（2026-10-07）

**BF16 正确性在已测试范围内通过；正式配置与输入集合已登记。正式 runner 由用户编写，pilot 和正式性能采集尚未完成。** 当前没有正式加速结论，也不声称生产 SpecDec 与 AR 始终逐 token 一致。

| 工作 | 已完成 | 边界与待办 |
|---|---|---|
| 自写引擎 | Embedding、RMSNorm、RoPE、GQA、SwiGLU、KV-cache、LM head、greedy generation、chunked prefill；参考实现对齐 | 历史 CPU FP32 与 GPU BF16 对齐证据有各自输入和配置范围 |
| 投机解码 | Draft 提议、target 批量验证、拒绝修复、cache 回滚、EOS 与长度处理 | 限定范围 BF16 B6 PASS，不等于生产批量验证与 AR 全局 exact-id 等价 |
| 正确性取证 | 类型覆盖、8K 融合复核、真实 EOS 补测；serial oracle 在对应测试中与 AR 完整一致 | serial 是诊断控制，不进入正式 speedup；生产批量验证仍可能出现可重复的数值分歧 |
| V-only FP8 KV | K 保持 BF16，V 使用 FP8 E4M3、per-token × per-head scale、group=128 | 历史 eager 质量与压缩结果已记录；新后端独立准入与质量回归待完成 |
| 测量与数据准备 | 阶段计时、显存记录、环境门、失败落盘、独立进程诊断；正式 16-family 清单及 CPU 校验 | 工具准备完成不等于正式实验完成；还需实现 runner 并通过 pilot |

下一步：**编写 runner → CPU dry-run 与记录校验 → 四-family pilot → 按协议处理稳定性及失败 → 获准后正式采集。** 不用历史 eager 波动提前否决当前 cuDNN-prefill 配置的 8K 条件。

## 实现与验证历程

各阶段解决的是不同问题，不能把前一阶段的 PASS 当作后一阶段的结论。

| 阶段 | 测试目的与方法 | 已获得的证据 |
|---|---|---|
| L1 / L2：基础引擎 | 对照参考 logits；比较缓存前向与全量重算；比较 greedy token IDs | 0.5B 基础前向与缓存生成对齐，为后续功能建立基线 |
| G1：1.5B target | 复制参考权重，分别检查 full-forward、cached-vs-full、生成一致性 | 历史 CPU FP32 最大 logits 差约 `3.1e-05` / `2.0e-05`，四组生成一致；GPU BF16 短窗口另有对齐证据 |
| G2：长上下文可运行性 | 8K chunked prefill 后继续生成，多轮记录有限性、序列与显存 | 历史单 target operational 测试无 OOM、三轮稳定；不代表当前双模型正式计时已经稳定 |
| G3：KV 压缩 | 量化 roundtrip、缓存结构、重复确定性、chunk 对照和真实文本 NLL/PPL | 全 K/V FP8 质量损失明显，转为 K BF16 + V FP8；量化方案与证据边界已记录 |
| G4：投机解码 | Draft 对齐、双模型可行性、cache API、主循环分支、EOS、AR/Spec 对照 | 已实现完整 greedy 流程；通过分层正确性取证区分控制逻辑与生产数值差异 |
| G5：测量准备 | 独立进程、环境门、同步计时、记录格式、失败分类、输入重建 | 测量工具与正式输入集合已准备；用户编写正式 runner 后仍需 pilot 检验 |

### 为什么正确性定位花了较多时间？

同一个 target 的单步 AR 与批量验证，在数学结构上应具有对应的因果预测位置，但 BF16 下不保证不同执行形状逐位相同。局部误差可能改变接近并列的 argmax；一旦选择不同 token，两条路径的后续历史也会不同。

我们因此不能只看最终文本是否相同：需要检查预测位置、缓存长度、接受/拒绝分支，再用同历史中间张量对照隔离数值因素。**Serial oracle** 保留投机控制流程，但用单步 target 决策作为诊断参照。在对应样本上与 AR 一致，支持控制逻辑；它不是生产批量路径，也不能拿来证明生产 SpecDec 的速度或全局输出等价。

当前有限范围 BF16 B6 证据包含 60 次类型覆盖、6 次 8K 融合复核和独立 6 次真实 EOS 补测。历史不完整用例与 FAIL/CHECK 保留；真实 EOS 补测支持其实际执行分支，其他分支还需结合 CPU 控制流证据阅读。详见 [B6 状态与证据](docs/b6_status_and_evidence_index_2026-10-05.md)。

## 研究范围与实验条件

主问题只有一个：**上下文增长如何改变固定配置下的 SpecDec 性能？** 自写引擎是实验基础，KV 压缩是次要消融，不把所有实现模块作为并列研究问题。

| 实验线 | 上下文 token 数 | Target KV | 解码策略 | 条件数 |
|---|---|---|---|---|
| BF16 主实验 | 2048 / 4096 / 8192 | 未量化 BF16 | AR / SpecDec | 6 |
| 次要压缩消融 | 8192 | K BF16 + V FP8 | AR / SpecDec | 2 |

旧的 **3×2×2、12 条件矩阵是历史计划**，不是当前主实验规格。FP8 按自己的准入条件放行，不阻塞 BF16 主线。

当前登记配置：

- Target：`Qwen/Qwen2.5-1.5B-Instruct`；draft：`Qwen/Qwen2.5-0.5B-Instruct`；计算精度 BF16。
- Attention：`cudnn_prefill` 仅融合符合长度策略的 prefill；decode / propose / verify / repair 保持 eager。
- `k=4`、`chunk=2048`、`max_new=64`；greedy argmax，命中 EOS 即停并记录实际长度；诊断精度干预关闭。
- AR 使用 target-only 进程；SpecDec 使用双模型进程。加载不进入推理计时。
- Pilot：开发用 family `F09/F03/F11/F04`；正式：`FF01–FF16`，按清单重建输入并核对长度/hash。
- 每个 family × condition 合计 5 次 measured，跨两个随机测量 block 按 3+2 分配；每个 `(family, condition, block)` 独立进程做一次 warm-up 并丢弃。

配置冻结不等于正式采集批准。见[配置记录](docs/config_freeze_2026-10-06.md)、[runner 规格](docs/formal_runner_spec_2026-10-06.md)和[准入清单](docs/bf16_formal_admission_2026-10-05.md)。恢复运行规则与引擎基线/runner 版本关系仍需在实现时明确。

### 数据与重复测量如何组织？

正式 family 来自 WikiText-2 train 的预定窗口，pilot 使用已参与开发的 family。每个正式 family 从同一窗口取 2K、4K、8K 嵌套前缀；长度越长，输入包含的信息越多，续写起点也不同。相同长度下的 AR/Spec 对照使用同一输入。

`formal-v1` 从 22 个未使用候选中按确定性规则选出 16 个 family；集合内相对分层为 4/4/4/4，每个源 block 不超过 5 个。清单记录来源、位置和 hash，不按正式性能结果挑选样本。CPU 校验通过 18 项检查，包括 48 个前缀的重建、消费隔离和嵌套关系。

这里有两种不同的 block：**数据源 block** 表示文本截取区域；**测量 block** 表示随机化运行的一轮。主实验共 `16 × 6 × 5 = 480` 次 measured；按每个 `(family, condition, measurement block)` 一个进程组织时，共 192 个进程任务及对应 warm-up。此数量不包含 pilot、失败处置与 FP8 消融。

## 已有发现与结论边界

| 发现 | 实测结果 | 适用范围 |
|---|---|---|
| V-only FP8 固定文本质量 | 独立 40 个 2048-token 窗口：平均 ΔNLL `+0.000397`，相对 ΔPPL `+0.040%` | Qwen2.5-1.5B、BF16 compute、历史 eager 路径；不证明分歧后的生成质量等同 |
| Target 8K 逻辑 KV 压缩 | `224.00 → 169.75 MiB`，约 `1.32×`，含 scale 元数据 | 仅 target KV，不是整模型或双模型显存压缩比 |
| 压缩不必然加速 | 历史 paired eager 测试峰值仅降约 `0.06 GiB`，decode 未加速 | 反量化有成本；不能直接外推到当前后端 |
| K-FP8 负结果 | 被测配置下 ΔNLL 约 `+2.4` 或更高，V-only 明显更温和 | 仅本模型、在线 max-abs E4M3 与被测实现，不宣称所有 K 量化不可行 |
| 批量验证数值行为 | 同历史局部样本中，QK 与 projection 对 `M=1` / `M=4` 形状敏感；生产轨迹可能分歧 | 控制逻辑正确与逐 token 数值复现是不同问题；未证明统一、唯一的 kernel 级根因 |

正式性能指标为 `speedup = T_AR / T_Spec`，大于 1 表示 SpecDec 更快。若输出轨迹不同，须同时报告分歧、实际长度、接受率和调用量，不能称为“相同输出的无损加速”。插桩、serial oracle 和 profiler 计时不作为正式速度数据。

显存分别记录 PyTorch `allocated`、`reserved` 和设备侧占用；`reserved` 不等于物理驻留，不能仅凭它超过 8 GiB 就断言分页或物理超限。旧 FAIL/CHECK 保留，后续 PASS 不覆盖历史记录。

## 指标如何阅读

| 指标 | 单位或定义 | 回答什么问题 |
|---|---|---|
| Prefill latency | 秒；AR 为 target prefill，Spec 包含 target 与 draft prefill | 处理输入、建立缓存需要多久？ |
| Decode latency | 秒；按冻结边界计时，记录实际生成长度 | 生成阶段需要多久？ |
| Decode throughput | 实际生成 token 数 / decode 秒数，token/s | 每秒生成多少 token？EOS 提前停止时不能仍按 64 计算 |
| Decode / e2e speedup | 对应边界的 `T_AR / T_Spec`，无量纲 | Spec 是否更快？`0.5×` 表示时间约为 AR 的两倍 |
| Acceptance rate | 接受候选数 / 提议候选数 | Draft 与 target 在当前轨迹上有多匹配？不是正确率或质量分数 |
| Accepted length / round counts | 每轮接受长度、轮数、target/draft 调用次数 | 节省了多少调用，额外工作来自哪里？ |
| NLL / PPL | 固定真实文本上的平均负对数似然；`PPL = exp(mean NLL)` | 改变 KV 表示后，对真实下一 token 的预测损失如何变化？ |
| Logical KV size | bytes / MiB，含 scale 等元数据 | 缓存表示本身压缩了多少？ |
| Peak allocated / reserved | MiB 或 GiB，分别报告 | 张量活跃分配与分配器池的峰值是多少？ |
| Sequence divergence | 完整 token IDs、共同前缀、首分歧位置、长度与 stop reason | AR/Spec 是否走了同一轨迹？不能只比较打印文本片段 |

`1 MiB = 2^20 bytes`，`1 GiB = 2^30 bytes`。所有结果需注明模型、精度、backend、输入、计时边界和代码版本；跨配置、跨工具的数字不能直接拼成一个 speedup。

### 性能分析与正式计时分开

阶段插桩和 profiler 用于解释成本，例如 draft 提议、target 验证、repair、cache 复制、矩阵乘法与反量化分别花了多少时间。它们会改变运行开销，因此应在独立诊断 pass 中执行，不混入正式 measured 数据。融合 attention 的内部步骤也不一定能在 profiler 中拆成独立算子。

Pilot 的作用是验证 runner、计时边界、记录完整性和当前 2K/4K/8K 的稳定性，不是挑选“更容易加速”的正式 family。只有这些工作完成并满足准入要求后，才开始正式采集；目前尚不能用探索性测量宣称加速，也不能将轨迹分歧直接解释成质量下降。

## 本地环境与检查

以下为项目已使用的 Windows / PowerShell 环境命令，不是正式实验启动命令。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-local.txt
python -m pip install -e .
python -m specdec.cli env-check
```

本机目标环境：PyTorch 2.6.0+cu124、RTX 4060 Laptop 8GB。GPU 测试串行运行；环境检查失败时保留失败证据，不绕过护栏伪装成功。

```powershell
# CPU 正式输入校验；需要本地数据与 tokenizer
python -X utf8 scripts/verify_formal_families_cpu.py

# 历史基础对齐工具，不代替当前完整准入
python -m specdec.engine.align_ours
python -m specdec.engine.test_cache_equiv
python -m specdec.engine.align_generate
```

离线模式可使用 `$env:HF_HUB_OFFLINE = "1"`，前提是所需文件已缓存。Manifest 只存来源、长度和 hash，不包含完整 token IDs。原始运行产物主要保存在本地；GitHub 文档同步不等于原始数据全部上传。

## 代码与文档导航

| 路径 | 用途 |
|---|---|
| `specdec/engine/model_self.py` | 自写模型、attention、KV-cache、量化接入与生成 |
| `specdec/engine/spec_dec.py` | Greedy SpecDec 控制循环 |
| `specdec/engine/spec_cache.py` | Cache 长度、裁剪与一致性契约 |
| `specdec/engine/` | 对齐、正确性、数值诊断与测量工具 |
| `specdec/kernels/` | 隔离的 CUDA / Triton 算子研究，不默认等于正式配置所用路径 |
| `scripts/build_formal_families.py` | 确定性构建正式 family 清单 |
| `scripts/verify_formal_families_cpu.py` | 输入、消费隔离、hash、分层与嵌套校验 |

- [课程 proposal](docs/pdf/specdec_course_proposal_2026-10-04_final.pdf)：主 6 条件 + 次要 2 条件；存在提交版 PDF 不代表已确认提交。
- [正式输入清单](docs/prompt_families_formal_manifest.json)：`formal-v1`；16 个 family、三档前缀，18 项 CPU 校验通过。分层为候选集合内相对标签。
- [G5 测量协议](docs/g5_measurement_protocol.md)：计时、随机化、环境与失败记录。
- [B6 状态与证据](docs/b6_status_and_evidence_index_2026-10-05.md)、[分层正确性协议](docs/specdec_correctness_protocol_v2_2026-10-04.md)：正确性声明范围。
- [正确性归因与处理报告（2026-10-07，19 页 PDF）](docs/pdf/specdec_correctness_localization_report_2026-10-07.pdf)：从分歧发现、同历史定位和因果干预，到逐行候选评估、serial oracle、B6/EOS 收尾与最终分层处理。生成源为 `scripts/generate_correctness_localization_report.py`；保留生产数值差异与历史 FAIL/CHECK，不宣称全局 exact-id 等价。
- [10 月 6 日阶段记录](docs/stage_checkpoint_2026-10-06.md)、[协作者阶段报告](docs/collaborator_status_2026-10-04.md)：历史进展与证据索引；历史“下一步”不替代当前配置。
- [数据台账](docs/data_ledger.md)：实测结果、配置与历史 FAIL/CHECK。
- [FP8 设计](docs/design_freeze_fp8_kv.md)、[SpecDec 设计](docs/design_freeze_g4_specdec.md)、[cache 契约](docs/design_g4_cache_api.md)。
- [投机解码教学](docs/specdec_explained.md)（[PDF](docs/pdf/specdec_speculative_decoding_explained.pdf)）：原理、张量与代码；历史 exact-id 说明应结合当前数值证据阅读。
- [协作者 PR 评审](docs/collaborator_pr_review_2026-10-02.md)、[kernel 研究说明](specdec/kernels/README.md)。

旧版 v6/v7/v7.1 计划与阶段日报保留在 `docs/` 供复盘；其中 12 条件、旧精度口径和历史验收不作为当前正式执行合同。性能与质量声明须指向真实测量及对应配置，不把诊断结果或外推估计写成正式实验结果。
