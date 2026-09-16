# SpecDec — 从零实现的小型 LLM 推理引擎（课程项目）

> 从零实现 Qwen2 系推理引擎 + KV-cache 解码 + 优化阶梯（KV 压缩 → 投机解码），
> 在 8GB 笔记本 GPU 上做可复现测量。
> 权威文档：`docs/plan_v7_1_protocol_freeze.md`（实验协议冻结 + Amendment v7.2）、`docs/data_ledger.md`（数据台账）、
> `docs/design_freeze_fp8_kv.md`（FP8 KV 设计冻结 v1.3）、`docs/design_freeze_g4_specdec.md`（G4 协议 v0.3）、
> `docs/plan_g4_next_steps.md`（G4 阶段状态）。

## 项目状态（2026-09-15 快照）

- ✅ **引擎正确性基线**：L1 从零前向对齐 0.000（fp32 CPU，vs transformers）；L2 KV-cache + 贪心生成对齐 PASS；
- ✅ **G1 — target 1.5B correctness**（fp32 CPU）：full-forward `3.1e-05`、cached-vs-full `2.0e-05`、argmax same、生成序列一致 → **PASS**；
- ✅ **BF16 GPU 对齐（质量口径）**：OURS vs HF 512-token `max|Δ|=0`、argmax 100%、NLL/PPL 相等。本机 CUDA FP16 对该 Qwen 路径会产生 NaN，因此**正式质量计算一律 BF16**；
- ✅ **G2 — 8K × 1.5B operational**（chunked prefill、3 runs 稳定、无 OOM）→ **PASS**；
- ✅ **G3 — FP8 KV FROZEN**：正式方案 = **Asymmetric V-only FP8 KV**（K cache BF16 不量化 + V cache FP8 E4M3，per-token×per-head scale，group=128）。独立 40 窗（不与 pilot 重叠）结果：`ΔNLL +0.000397`、`relative ΔPPL +0.040%`；V2 生成前缀 32/32、V3 chunk 行为一致；8K KV cache 224.00 → 169.75 MiB（**1.32×**）。**诚实边界**：eager decode 无加速且有 V 反量化开销；整机峰值仅降 ~0.06 GiB；1.32× 仅指 KV cache 逻辑压缩；K-FP8 / full-FP8 为 negative ablation（ΔNLL ≥ +2.4，**限定 Qwen2.5-1.5B + 在线 max-abs E4M3 + eager attention，不外推**）；
- ✅ **G4-0 — draft 正确性**：**CONDITIONAL PASS（BF16 shape drift 已表征）**——OURS full==HF full、OURS cached==HF cached；HF 自身 full/cached 亦同处分歧；FP32 恢复一致 ⇒ 非引擎 bug、非不确定性。block-vs-serial 被测 argmax 一致、cache 终长一致、**无 off-by-one**；但 cache 数值差 max\|Δ\|≈2.0 → exact-id 门保留为风险；
- ✅ **G4-B3 — 双模型 8GB 可行性**：**PASS 4/4**（1.5B+0.5B 同驻 bf16；ctx512 alloc 4.34/4.336 GiB；ctx8192 alloc 6.16 GiB、resv 8.92 GiB；无 OOM、finite、3 轮稳定）；
- ✅ **G4-B4 — cache 操作 API**：**PASS**（用户实现 `spec_cache.py`：`cache_length` / `trim_cache` / `check_consistent`；base 与 v_fp8 两档全 True）；
- ✅ **G4-B5 — greedy SpecDec 主循环**：**PASS**（用户实现 `spec_dec.py`；T1–T4 单测 + 真实投机 smoke：rounds 8 / proposed 27 / accepted 11 / mismatch 5 / all_accept 3；每轮不变量 `tgt_len==drf_len==pre+(m+1 or Kp)` 全成立）；
- ✅ **G4-B6 — correctness gate（两档并列登记）**：
  - **B6-fp32 = 实现正确性证据 → PASS**：10 case × 2 KV 档 = **20/20 组合逐 token id 全等**（CPU/float32，参照 = 同引擎 cached AR；总 216/348；finite、确定性、非空过、case 与档位跑满）。判据在运行前入库；"dtype 不是被测对象"已明确；
  - **B6-bf16 = 性能档行为 → CHECK 19/20**：唯一分歧 `natural-2@7` 经实测归因为 **bf16 数值不可判定位置**（候选对精确并列 22.8750；signal 0.1491 ≤ noise 0.1491），fp32 真值 = spec 侧 token。**严格口径：不设例外、不放宽 exact-id 门**，该记录原样保留并作为已知数值限制披露；
- ✅ **EOS 路径 bug 修复**：`_propose` 的 EOS 检查位置导致 draft cache 少 1，全接受分支 `trim_cache` 越界（生成以 EOS 正常收尾时必踩）→ 已修 + 纯逻辑回归测试（含旧实现对照复现 ValueError）；
- 🔄 **下一步**：≥16 prompt families（2K⊂4K⊂8K）→ G5 instrumentation → **Part 4 测量（含 AR 基线同档）** → 定 final 规模 → pilot/final → 图与归因（break-even）；
- 测量纪律：只记录实测（数据见 `docs/data_ledger.md`）；跨脚本显存数字不作比较；进度与决策变更走 amendment/日志而非改写已冻结方案。

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
```

G4 门（投机解码；GPU 脚本请在交互式终端串行运行）：

```powershell
python -m specdec.engine.g4_b3_dual_model_feasibility   # B3 双模型 8GB 可行性 —— PASS 4/4
python -m specdec.engine.g4_b4_cache_tests              # B4 cache API —— PASS（base + v_fp8）
python -m specdec.engine.g4_b5_helper_tests             # B5 辅助函数 T1–T4 —— PASS
python -m specdec.engine.g4_b5_smoke                    # B5 主循环 smoke —— PASS
python -m specdec.engine.g4_b5_eos_branch_tests         # EOS 分支回归（纯逻辑 CPU，秒级）—— PASS
python -m specdec.engine.g4_b6_correctness              # B6-bf16 exact-id 门 —— CHECK 19/20（已归因）
python -m specdec.engine.g4_b6_fp32_correctness         # B6-fp32 实现正确性门（CPU/float32）—— PASS 20/20
python -m specdec.engine.g4_b6_tie_diag                 # 分歧位置的 signal-vs-noise 归因
```

`g4_b6_fp32_correctness` 支持 `set B6F_LIMIT=N` 跑小样（仅验证环境/耗时；**小样永远不会打印 PASS**）。设 `set HF_HUB_OFFLINE=1` 可避免 tokenizer 联网抖动。

## 文档

- `docs/plan_v7_1_protocol_freeze.md` — 实验协议冻结（v7.1 + §0 **Amendment v7.2**：Compute=BF16、KV 因子、12 conditions）
- `docs/plan_v7.md` — v7 方案（方向、承诺、裁剪序）；`docs/plan_v6.md` — v6/v6.1 历史方案
- `docs/data_ledger.md` — 实测数据台账（§3.4 G1、§3.5.x FP8 KV、§4.5.1 G2、§4.5.2 B3、§4.5.3 B4、§4.5.4 B5、§4.5.5 B6-bf16、§4.5.6 B6-fp32）
- `docs/status_note_fp8_v4.md` — FP8 KV：根因、修复、正式判定、G4 前置清单
- `docs/design_freeze_fp8_kv.md` — FP8 KV 设计冻结（v1.0→v1.3）
- `docs/design_freeze_g4_specdec.md` — G4 投机解码协议（v0.3：logits 对齐、回滚语义、`_propose` 出口不变量、EOS 规则）
- `docs/design_g4_cache_api.md` — cache 接口契约；`docs/plan_g4_next_steps.md` — G4 七阶段规划与状态
- `docs/daily_report_2026-09-15b_G4_B5B6_EOSfix.md` — 本日日报（B5/B6/EOS 修复）
- `docs/daily_report_2026-09-15.md`、`docs/daily_report_2026-09-08b_G1G2G3_FP8KV.md` — 阶段归档日报
- `docs/stage_report_l1_l2.md`、`docs/L1_code_teaching.md`、`docs/debug_lesson_alignment.md` — 阶段与教学文档

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
