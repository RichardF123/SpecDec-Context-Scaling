# SpecDec — 从零实现的小型 LLM 推理引擎（课程项目）

> 从零实现 Qwen2 系推理引擎 + KV-cache 解码 + 优化阶梯（量化 → 投机解码），
> 在 8GB 笔记本 GPU 上做可复现测量。
> 权威文档：`docs/plan_v6.md`、`docs/stage_report_l1_l2.md`、`docs/data_ledger.md`。

## 项目状态

- ✅ **L1 — 从零前向**（RMSNorm / RoPE / GQA attention / SwiGLU / 24 层组装 /
  词表绑定打分）：对齐门 **max|Δ| = 0.000**（与 transformers fp32 逐位一致）。
- ✅ **L2 — KV-cache 解码 + 贪心生成**：缓存等价 ≈ 2e-5（浮点噪声），
  **生成对齐 PASS**（整条 token 序列与官方模型数学一致）。
- ✅ **L4 — int8 / packed-int4 权重量化**：Qwen2.5-1.5B 的 196/196 个目标
  Linear 已替换。4090D 实测中，INT8 模型 storage ↓42.4%、peak VRAM ↓38.4%、
  PPL +1.13%（PASS）；INT4 PPL +21.67%（FAIL，保留为诚实负结果）。
- 🔄 下一步 **L4b（fp8 KV 量化）/ L5（投机解码）**；RTX 4060 8GB 上的最终
  hardware-specific latency / peak VRAM 结论将在项目最终出数阶段按同协议复跑。
- 性能基线（Qwen2.5-0.5B）：CPU fp32 decode ≈ 9.2 tok/s；GPU fp16 decode
  ≈ 10.6 tok/s（latency-bound 的朴素解码——是后续各级优化要超越的起点）。

## 项目简介

**在做什么**：从零手写 Qwen2 架构推理引擎（RMSNorm / RoPE / GQA attention /
SwiGLU / KV-cache 解码 / 贪心生成），再沿 **KV-cache → int8/int4 量化 → fp8 KV
量化 → 投机解码** 逐级优化，用可复现测量表回答研究问题：
> 投机解码在 8GB 小卡上随上下文变长到哪里失效？KV 量化能否延长它的获利区间？

**意义**：课程要求"原创 + hands-on LLM system"——我们交付自写引擎 + 质量×成本
测量表（decode tokens/s / 显存 / 接受率 / ppl），而不是跑教程报一个数字；结果
可作为小模型部署的决策表；亲手实现 transformer 与推理引擎也正是 AI Infra
方向的核心技能。

**技术栈**：Python + PyTorch 2.6（cu124）；自写引擎 `engine/model_self.py`（OURS）；
HuggingFace 仅用于权重加载与 tokenizer（LIB）；对齐/缓存/生成测试 vs transformers
eager；规划 int8/int4（L4）、fp8 KV（L4b）、固定链投机解码（L5）、受控 CUDA kernel 轨、
Spec-Bench 风格测量协议 + `results.jsonl`。

**为什么这么做**：调 API/库只能看整机表现，研究问题（缓存、量化、投机解码的收益）
需要引擎内部控制权（缓存布局、解码循环、计时插桩）——这是自建引擎的必要性而非炫技；
"从零"配上对齐基线（0.000）才让每个结论可复现、可归因；同时遵守风险分层纪律
（只承诺纯 PyTorch MUST，高方差项为加分轨）。

## 快速开始

```powershell
# 1) 建环境（方式 A：conda / 方式 B：venv）
conda create -n specdec python=3.11 -y ; conda activate specdec
# 或：
py -3.11 -m venv .venv ; .\.venv\Scripts\Activate.ps1

# 2) 安装依赖（必须装 CUDA 版 —— CPU wheel 会让 cuda=False，静默失败）
pip install -U pip
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-local.txt
pip install -e .

# 3) 验证机器（第 0 步门）
python -m specdec.cli env-check
```
预期输出：`cuda available: True`、`NVIDIA GeForce RTX 4060 Laptop GPU`、显存约 8GiB。

> 若 HuggingFace 下载慢/失败，先执行：`set HF_HOME=D:\hf-cache` 和
> `set HF_ENDPOINT=https://hf-mirror.com`，再跑下面的验收命令。
> Windows 说明：官方 `triton` 没有 Windows 版——kernel 轨（L3）在 WSL2 里做，
> 或按 go/no-go 规则保持 eager（详见 `docs/plan_v6.md`）。

## 验收命令（仓库根目录 + venv 已激活）

```powershell
python -m specdec.engine.align_ours          # L1 门 —— 预期 0.000，PASS
python -m specdec.engine.test_cache_equiv    # L2 缓存==全量 —— 预期 < 1e-3，PASS
python -m specdec.engine.align_generate      # L2 生成对齐 —— 预期 PASS
python -m specdec.engine.bench_basic         # 性能基线（CPU，有 GPU 则含 GPU）

# A6：轻量工程门（无需下载 1.5B 权重）
python -m specdec.quant.test_int8
python -m specdec.quant.test_int4

# A6：真实权重集成门（默认 Qwen2.5-0.5B）
python -m specdec.quant.align_int8
python -m specdec.quant.align_int4

# A6：1.5B 系统成本与质量（CUDA；每种 mode 单独进程运行）
python -m specdec.quant.bench_l4 --mode fp16
python -m specdec.quant.bench_l4 --mode int8
python -m specdec.quant.bench_l4 --mode int4
python -m specdec.quant.eval_l4_ppl --mode fp16
python -m specdec.quant.eval_l4_ppl --mode int8
python -m specdec.quant.eval_l4_ppl --mode int4
```

## 文档

- `docs/plan_v6.md` — 权威项目方案（MUST/SHOULD 分层、门体系、时间线）
- `docs/stage_report_l1_l2.md` — 阶段总结（决策历程、debug 实战、经验教训）
- `docs/data_ledger.md` — 实测数据台账（正确性 + 性能，每条带条件）
- `docs/L1_code_teaching.md` — `engine/model_self.py` 逐行教学
- `docs/debug_lesson_alignment.md` — 差分调试方法论

## 包结构

```
specdec/
  cli.py          CLI：env-check、run
  config.py       RunConfig + config_hash / run_id（复现哈希链）
  results.py      只追加 results.jsonl 台账
  engine/         model_self.py（OURS 自写）+ 对齐/测试/基准工具
  kernels/        （L3 轨，可选）
  quant/          L4：int8 / packed-int4（OURS，已实现）
  kv_quant/       Phase 4b：KV 量化（OURS）
  spec_dec/       Phase 5：投机解码（OURS）
  bench/          Phase 7：基准测试
```

诚实性规则：`results.jsonl` 只追加真实测量数据，绝不写假行。引擎代码由团队
自己编写；只有权重加载 / tokenizer / 测试工具借用第三方库（OURS vs LIB 边界
见 `docs/plan_v6.md`）。
