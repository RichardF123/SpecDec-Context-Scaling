# SpecDec：上下文长度与推理性能研究

在 RTX 4060 Laptop 8GB 上实现 Qwen2.5 推理引擎，研究固定模型对、精度与推理配置时，上下文长度如何影响 greedy speculative decoding 相对自回归解码（AR）的性能。DSAI 4207 课程项目。

团队实现模型前向、KV cache、量化辅助函数和投机解码控制循环；张量运算依赖 PyTorch，权重与 tokenizer 来自 Hugging Face，prefill 融合 attention 使用库提供的 cuDNN 路径。**自写推理引擎不等于所有 CUDA 算子均为手写。**

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
- [10 月 6 日阶段记录](docs/stage_checkpoint_2026-10-06.md)、[协作者阶段报告](docs/collaborator_status_2026-10-04.md)：历史进展与证据索引；历史“下一步”不替代当前配置。
- [数据台账](docs/data_ledger.md)：实测结果、配置与历史 FAIL/CHECK。
- [FP8 设计](docs/design_freeze_fp8_kv.md)、[SpecDec 设计](docs/design_freeze_g4_specdec.md)、[cache 契约](docs/design_g4_cache_api.md)。
- [投机解码教学](docs/specdec_explained.md)（[PDF](docs/pdf/specdec_speculative_decoding_explained.pdf)）：原理、张量与代码；历史 exact-id 说明应结合当前数值证据阅读。
- [协作者 PR 评审](docs/collaborator_pr_review_2026-10-02.md)、[kernel 研究说明](specdec/kernels/README.md)。

旧版 v6/v7/v7.1 计划与阶段日报保留在 `docs/` 供复盘；其中 12 条件、旧精度口径和历史验收不作为当前正式执行合同。性能与质量声明须指向真实测量及对应配置，不把诊断结果或外推估计写成正式实验结果。
