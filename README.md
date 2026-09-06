# SpecDec — 从零实现的小型 LLM 推理引擎（课程项目）

> 从零实现 Qwen2 系推理引擎 + KV-cache 解码 + 优化阶梯（量化 → 投机解码），
> 在 8GB 笔记本 GPU 上做可复现测量。
> 权威文档：`docs/plan_v6.md`、`docs/stage_report_l1_l2.md`、`docs/data_ledger.md`。

## 项目状态

- ✅ **L1 — 从零前向**（RMSNorm / RoPE / GQA attention / SwiGLU / 24 层组装 /
  词表绑定打分）：对齐门 **max|Δ| = 0.000**（与 transformers fp32 逐位一致）。
- ✅ **L2 — KV-cache 解码 + 贪心生成**：缓存等价 ≈ 2e-5（浮点噪声），
  **生成对齐 PASS**（整条 token 序列与官方模型数学一致）。
- 🔄 下一步 **L4（int8/int4 权重量化）**；之后 L4b（KV 量化）/ L5（投机解码）；
  CUDA kernel 轨为受控加分项（见 `docs/plan_v6.md`）。
- 性能基线（Qwen2.5-0.5B）：CPU fp32 decode ≈ 9.2 tok/s；GPU fp16 decode
  ≈ 10.6 tok/s（latency-bound 的朴素解码——是后续各级优化要超越的起点）。

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
  quant/          Phase 4：int8/int4（OURS）
  kv_quant/       Phase 4b：KV 量化（OURS）
  spec_dec/       Phase 5：投机解码（OURS）
  bench/          Phase 7：基准测试
```

诚实性规则：`results.jsonl` 只追加真实测量数据，绝不写假行。引擎代码由团队
自己编写；只有权重加载 / tokenizer / 测试工具借用第三方库（OURS vs LIB 边界
见 `docs/plan_v6.md`）。
