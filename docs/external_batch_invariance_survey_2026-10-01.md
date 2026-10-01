# 外部方案核查：batch/shape 不变性与投机解码无损性（只读调研，2026-10-01）

> 只读调研：**未安装任何东西**、**未迁移**、**未做任何性能结论**、未修改仓库代码。
> 目的：回答"现成算子/框架是否覆盖我们观察到的**形状相关数值差异**"，以及本机环境（Windows +
> RTX 4060 Laptop sm_89 + PyTorch 2.6.0+cu124 + transformers 5.16.1）的兼容性证据。

## 1. PyTorch 官方文档

* **明确不保证**：数值精度说明写明 PyTorch **不保证**数学等价计算的逐位一致，且 CPU/GPU、跨版本都可能不同；
  其中有一节 **"Batched computations or slice computations"** 直接指出
  `(A@B)[0]` **不保证**与 `A[0]@B[0]` 逐位相同、`A.sum(-1)[0]` **不保证**与 `A[:,0].sum()` 相同
  —— 这正是我们观察到的**形状依赖**现象在官方文档中的对应条目。
* **已有旋钮**：`torch.use_deterministic_algorithms(True)`（针对**原子性/运行间**不确定，**不是**形状依赖）；
  `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction`（bf16 GEMM 归约精度，**默认开启**）；
  `torch.backends.cuda.matmul.allow_tf32`（1.12 起默认 False，对 fp32 无效）；
  SDPA 各后端各自确定，但**跨后端不保证逐位一致**。
* **证据缺失**：`CUBLAS_WORKSPACE_CONFIG` 在当前官方文档中**未检索到**（不作为依据引用）。

## 2. Thinking Machines `batch_invariant_ops`（2025-09-10 博客 / MIT 仓库）

* 覆盖：`aten::mm`/`addmm`（**线性层**，固定 tile 的 Triton kernel）、`aten::mean.dim`（**间接覆盖 RMSNorm**）；
  **attention 在博客里设计但"未包含在代码发布中"**（FlexAttention 的固定 split-size 改动待上游）。
* 与本问题最相关的一点：博客明确讨论**"缓存里有 0 个 token（prefill）还是 999 个 token（decode）"时归约顺序必须一致**，
  并指出 vLLM 的 Triton prefix-prefill kernel 破坏了该性质；修复需要**先更新 KV/page table 再算 attention**，
  且要用**固定 split 大小（不是固定 split 数）**。
* 已知限制（有 issue 证据）：在 **PyTorch 2.6 + 公开 Triton 3.x 上会 `TypeError`**（issue #14，修复 PR 未合并）；
  fp32 输入默认走 TF32、精度反而下降（issue #23/PR #24，含 RoPE `cos()` 误差达 1.35 的记录）；
  其 vLLM 示例要求 **PyTorch ≥ 2.9**；博客验证是 Qwen3-235B/8B on vLLM，**未提及 Windows**。

## 3. vLLM batch invariance 与投机解码"无损"

* 文档：**beta** 功能，要求 **NVIDIA CC ≥ 8.0**（sm_89 形式上满足）；`VLLM_BATCH_INVARIANT=1`；
  会禁用 custom all-reduce、序列并行/async TP；**明确提示有性能代价**；已测模型列表**含
  `Qwen/Qwen2.5-1.5B-Instruct`**。
* 投机解码页面：目标并行验证"**algorithmically lossless and theoretically lossless up to floating-point
  precision. Minor variations may occur due to hardware numerics or batch size changes affecting numerical
  stability.**" ⇒ **官方自己把"无损"限定在浮点/批大小层面**，不是逐位保证。
* 跟踪 issue #27433：**"Speculative decoding support (this might be hard)" 未勾选**、prefix caching 未勾选；
  未合并 PR #52522 报告 BI 关闭时 EAGLE3/DFlash/DSpark 在 GSM8K 上有 exact-BI 不匹配（401/187/118 ÷ 1319），
  BI 开启后为 0（吞吐 −25%/−5%/−3%），但验证环境是 **≥32 GB 显存的 CUDA GPU**。

## 4. 四个问题的回答

| # | 问题 | 结论 |
| --- | --- | --- |
| 1 | 哪些现成算子覆盖**线性/归一化/attention**？ | **线性 + 归一化（经 mean）** 在 TML 已发布库中覆盖；**attention 只有设计/在 vLLM 内部**，未随库发布。vLLM 在自家引擎内文档化了线性/attention 的 batch invariance。 |
| 2 | 是否覆盖 **token 分块 / 历史缓存**相关的归约差异？ | **唯一有明确文档的地方**就是这一点（博客 §"cache 里 0 vs 999 token"），但**修复未随代码发布**；vLLM 把 **speculative decoding 支持标为未完成**。⇒ **没有任何现成算子直接覆盖"1 token vs 4 token 同历史"的差异**。 |
| 3 | Windows + RTX 4060(sm_89) + PyTorch 2.6 的兼容性证据？ | vLLM 只要求 CC ≥ 8.0（形式上满足）；**两者都没有 Windows 支持证据**（文档/安装路径都是 Linux/CUDA 中心）。TML 库还有**反向证据**：PyTorch 2.6 + 公开 Triton 会 `TypeError`，其示例要求 PyTorch ≥ 2.9。**无任何 sm_89 或 8GB 笔记本 GPU 上的验证记录。** |
| 4 | 哪些需额外验证 / 哪些明显不适合？ | **需验证**：`allow_bf16_reduced_precision_reduction=False` 是否改变我们对齐行的差值（有文档、未实测）；TML 的 override 在 `transformers` 里是否真被派发；sm_89 上 Triton kernel 能否绕过 issue #14。**明显不适合**：在本仓库 PyTorch 2.6 上装 Triton kernel（有文档化的 `TypeError`）；把 `use_deterministic_algorithms`/cuDNN determinism 当作本问题的解法；把 vLLM 的"lossless"当成逐位保证（其文档自认不是）。 |

## 5. 明确未知

1. 当前官方对 `CUBLAS_WORKSPACE_CONFIG` 的正式说明（未检索到）；
2. **`transformers` eager attention（fp32 softmax 路径）是否也有 1-vs-4 token 的同类差异** ——
   无公开来源讨论；**本轮我们自己用 HF 实测回答了它**（见报告 §9.40：**有**，且量级同阶）；
3. TML 的 attention/FlexAttention 改动是否已上游（仓库 2025-11-04 后无活动）；
4. vLLM #52522 是否已合并、BI 是否对更多 attention 后端受限；
5. 两个项目在 sm_89 / Windows 上的任何 kernel 级验证（**证据缺失**，不等于被否定）。
