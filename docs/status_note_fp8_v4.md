# G3 / FP8 KV 状态说明（2026-09-08 晚）

> 范围：V4 超常 PPL 根因、修复清单、当前有效 V2/V3/V4 结果、G3 状态判定、下一步选型建议。
> 纪律：G4 SpecDec **不启动**；当前 V4 数字**不**作为最终结论写入 ledger（待选型后定稿）。

## 1. 根因（两类，叠加导致 NaN / 千万级 PPL）

1. **引擎静默缺 q/k/v bias（主因）**：Qwen2.5 checkpoint 里 `q/k/v_proj` 实际带 bias，
   但 config 无 `attention_bias` 字段；`build_from_config` 旧默认 `False` + `load_state_dict(strict=False)`
   静默跳过这些权重 → 引擎在无 bias 下前向，GPU 低精度输出严重跑偏（旧 fp16 V2/V3/V4
   全部受影响；其中 V2/V3 是"自己对自己"自洽，结构/确定性项仍有效，但行为数字不可信）。
2. **平台 dtype 问题（次因）**：本机 HF CUDA `fp16` 对该 Qwen 路径本身产生 NaN
   （短 prompt 即 non-finite，argmax `'!'`）；`bf16` 正常（同 prompt argmax `" Paris"`）。
   → 质量/NLL 一律用 bf16。

## 2. 已修复（commit `3d93921`）

| 修复 | 内容 |
|---|---|
| attention bias | `align_ours.build_from_config`：`model_type=="qwen2"` 且未显式指定时默认 `attn_bias=True`；另有从 HF 模块实测 bias 的检测路径 |
| RMSNorm | 内部 fp32 计算，再转回原 dtype（与 HF 一致）|
| softmax | `torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)` |
| FP8 反量化 | `dequantize_kv_fp8(q, scale, dtype=x.dtype)`，不再硬编码 fp16 → bf16 运行时解压回 bf16 |
| harness dtype | V2/V3/V4 切 `torch.bfloat16`；V4 增 `LOSS_ROWS=64`（按位置小块算 loss 防 OOM）|

修复后短 prompt（bf16）核对：`missing=[] unexpected=[]`、finite True/True、`maxdiff≈0.25`、
argmax `Paris / Paris` —— **bf16 基线路径恢复到合理状态（短 prompt 级别）**。

## 3. 当前有效结果（bf16，用户 GPU 实测）

**V2（512 tok prompt, chunk 1024, decode 32, bf16）**
- 结构 PASS（fmt/dtype/scale 形状、确定性、finite）；
- 行为分歧大：agreeing prefix **1/32**；step logits max|Δ| ≈ 6.7 → 11.4（随步增长）。
- base 续写 `' spans many decades of research into learning syst'`；fp8 `' spans decades research decades …'`。

**V3（4096 tok, chunk 1024 vs 512, bf16）**
- fp8 KV（反量化后）max|Δ| overall **20.25**（层 0/14/27：4.75 / 14.38 / 20.25）；
- 两 chunk 布局 decode16 **不一致** → **CHECK**（chunk-size 敏感成立，机制见 §5-D 说明）。

**V4（Wikitext-2 test, 40×2048 窗, chunk 512, bf16 base KV vs FP8 KV）**
- mean NLL base **4.7446** / fp8 **6.0879**；
- PPL base **114.96** / fp8 **440.50**；ΔPPL **+325.54**；relative **+283.2%**；
- per-window ΔPPL mean +398.3（min +72.5, max +2432.7）；logits finite True。
- 注：base PPL 114.96 偏高（Instruct 模型 + 裸 wikitext），但 finite 且稳定，可作为相对基准。

**仍有效（不受本组 bug 影响）**：V1 往返单测 PASS（纯张量数学）；V2 结构/确定性项；V4 口径
审计结论（按 token 平均 NLL、label shift、reduction=sum、W−1、同窗同口径 —— 全部正确）。

**作废/待复核**：ledger §3.5.1 的 fp16 时代 V2/V3 行为数字（受 bias+fp16 双重污染），
需按 bf16 口径重新登记（下节）；ledger §3.5.2 V4 仍空。

## 4. G3 状态判定

**CHECK（不是 clean PASS，也不是最终 negative/mixed）**
- 能跑通 ✅；V1 ✅；V2 结构 ✅；
- 质量门 ❌：FP8 相对 ΔPPL **+283%** 过大；
- 稳定性门 ❌：V3 chunk-size 敏感（序列不一致）。
- 结论：**FP8 KV（per-token × per-kv-head, E4M3）当前实现质量不可接受**，但**尚未到"宣判不可行"**
  —— 存在低成本高信息量的改进实验（§5），做完再定稿（改进冻结 或 如实登记 negative/mixed）。

## 5. 设计选型（先实验后定稿，保持报告纪律：无人工阈值、报 base/fp8 PPL + ΔPPL% + V2 分歧 + V3 敏感）

| 方案 | 内容 | 作用 | 成本/风险 |
|---|---|---|---|
| **A 冻结现状** | 把 per-token×per-head E4M3 登记为 negative/mixed | 立即推进 G4 | 放弃 FP8 质量；RQ3 只能讲 feasibility |
| **B 细粒度 scale** | per-token × per-head × **组内分组**（head_dim 64 切成 2×32 或 4×16，每组独立 scale） | 直接降量化误差（最可能把 +283% 拉回可接受）| 元数据 16B→32B(g32)/64B(g16) /token/layer；压缩比 1.88×→1.6×；改动集中在 quantize/dequant 与 scale 形状 |
| **C per-channel 诊断** | 每 (head, 64 维中每维) 一个 scale（精度优先诊断）| 判定"大 ΔPPL 是否 scale 粒度造成" | decode 单步时 per-channel 需跨 token 统计 → 需静态/校准 scale 或两遍，复杂；**只作诊断不冻结** |
| **D 语义一致化** | **本 chunk 的 K/V 也在 attention 前量化→反量化**（"所有 token 都经 fp8 视图"），对齐生产 FP8-KV 语义 | 消除 chunk-size 敏感（V3 稳定）；语义 = 真·FP8-compressed KV | 本 chunk K/V 也带 fp8 噪声（更真实）；compute 略增；改动在 fp8 读路径 |

**最终路线（2026-09-08 用户拍板, 覆盖上文“推荐顺序”; A 暂不选）**：
`① 基线验证（bf16 对齐 + G1 回归）→ ② D 语义一致化 → ③ B 细粒度 scale → ④ 依 V3/V4 定 FP8 去留 → 才考虑 G4`
理由：先确认"基础链路已对齐"再动 FP8 方案, 否则异常无法归因; +283% 混有"粒度过粗"与
"chunk 语义不一致"两个可修因素, 现阶段冻结 negative/mixed 过早。

## 6. 下一步动作（既定路线 ①→④, 等你执行/确认）

- [ ] **①a** GPU 单跑 `python -m specdec.engine.gpu_bf16_align_probe`
      （中等长度 bf16 ours-vs-HF: finite / max|Δ| / argmax% / NLL / control Paris）
- [ ] **①b** `python -m specdec.engine.g1_target_15b`（fp32 CPU 回归, 应仍 PASS 三检查）
- [ ] 基线过 → **② D** 语义一致化（本 chunk K/V 也在 attention 前量化→反量化）→ 重跑 V3（预期消融稳定）+ V4
- [ ] **③ B** per-head 内 2×32 分组 scale（GROUP=32）→ 重跑 V4 看 ΔPPL%; 不足再 GROUP=16
- [ ] **④** 依 V3/V4 结果: 改进版定稿 → G3 PASS(改进) 或 如实登记 negative/mixed → 才进 G4
- [ ] 全部按 bf16 口径记入 ledger（§3.5.1 修正行 + §3.5.2 定稿; 旧 fp16 行为数字标注作废）
