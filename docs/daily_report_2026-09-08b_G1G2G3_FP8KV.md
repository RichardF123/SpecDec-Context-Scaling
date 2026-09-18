# SpecDec 日报（归档）2026-09-08b —— G1/G2/G3 + FP8 KV 定案全程

> 本文件归档 2026-09-08 后半程（本会话）全部工作、实测数据与结论。
> 同日前半程见 `daily_report_2026-09-08.md`（分块 prefill 完工/v6.1，commit `be0b448`）。
> 数字来源：用户本机（RTX 4060 Laptop 8GB）运行回贴，全部为 **BF16** 口径（CUDA fp16 对本 Qwen 路径 NaN，见 §2.3）。
> 数据账本见 `data_ledger.md`（§3.4/§4.5.1/§3.5–3.5.4）；设计修订见 `design_freeze_fp8_kv.md` v1.0→v1.3。

## 一、一句话
**从 v7.1 Priority-1 起步，跑完 G1→G2→G3：修出基础链路（bias/RMSNorm/softmax/RoPE/bf16 平台），用 pilot 隔离出"K-FP8 损坏质量、V-FP8 无损"，正式冻结 Asymmetric V-only FP8 KV（K BF16 + V FP8 gs128），独立 40 窗 ΔNLL +0.000397、8K KV 压缩 1.32×。G4 未启动。**

## 二、主线时间线（commit `dd351bd`→`244b4b6` + 未提交的 v_fp8 定案实现）
| 阶段 | 动作 | 关键 commit |
|---|---|---|
| G2 minimal | 8K×1.5B chunked prefill 可运行（6.83 GiB） | dd351bd |
| G2 正式门 | 3×128-tok 稳定、correctness sanity、峰值伪影定位（跨 run 存留 last_logits）→ **PASS** | a2af09f…5c40396 |
| FP8 设计冻结 | D1–D8 v1.0 → 批准 | a73ff35, 2af7cc0 |
| 平台根因 | CUDA fp16 NaN；bias 缺失（strict=False 掩盖）；RMSNorm/softmax/RoPE fp32 化；质量口径→bf16 | 3d93921, ac9c51f 等 |
| V1–V3 fp16 时代 | 被上述 bug 污染 → 作废重测（§3.5.2 备注） | 709b02c… |
| bf16 对齐 | ours vs HF 512-tok max\|Δ\|=0 / argmax 100% / NLL=PPL 相等 | a9f6925, gpu_bf16_align_probe |
| D 语义一致化 | 当前 chunk K/V 先量化→反量化再进 attention；存储复用同一 payload | ac9c51f |
| B 分组 scale | group32 质量反降 → 判定"分组方向证伪" | 071f414, 0c35a2b |
| pilot K/V 隔离 | canary(gs1 无损)、真实 K/V 指标、离线 attention 代理、9 配置 4 窗 NLL → **K-FP8 是全部伤害、V-FP8 无损** | 0c35a2b…82e51a9 |
| 正式定案实现 | fmt="v_fp8"（K bf16 + V FP8 gs128）；V4 独立窗口 40..79；8K paired operational | （未提交，本批） |

## 三、发现与修复（均已在引擎验证）
| 问题 | 根因 | 修复 |
|---|---|---|
| V4 PPL≈1e7 / NaN | ① 引擎静默缺 q/k/v bias（config 无 attention_bias 但 checkpoint 有）；② 本机 CUDA fp16 对该 Qwen 路径本身 NaN | ① bias 默认/实测检测（`align_ours`）；② 质量口径统一 **bf16** |
| RMSNorm/softmax 低精度漂移 | fp16/bf16 内算方差/softmax | 内部 **fp32** 计算再转回 dtype |
| 512-token 对齐 79% | RoPE 位置/频率/角度用 bf16（256 后不精确） | 位置/频率/cos/sin **fp32** 计算后转 dtype → 对齐 100% |
| V4 OOM | 整块 512×151936 logits 转 fp32 | LOSS_ROWS 小批计算，口径不变 |
| G2 峰值 6.83→7.62 假漂移 | 上 run 的 last_logits 跨 empty_cache/reset 存活 | run 间 `del` → 3/3 稳定 6.83 |
| G1 切分空张量 | 固定第 32 token 切分，prompt 可不足 | 按实际长度动态切分 |
| V3 chunk 敏感 | current-chunk K/V 以 bf16 特权参加 attention | D 语义一致化 |

## 四、数据与结论（详细见 data_ledger）
### G1（§3.4）PASS
full-forward 3.099e-05 / cached-vs-full 2.038e-05 / argmax same / generation equal（fp32 CPU）。

### G2（§4.5.1）PASS
8K×1.5B chunked prefill，3 runs 稳定、无 OOM、峰值记录；7.5 GiB 只是本地观察线，协议判据=no OOM+记录+稳定。

### FP8 negative ablations（pilot 级）
- full-FP8 per-head(128)：ΔNLL≈+2.41 / rel ΔPPL≈+771%；
- full-FP8 gs32：rel ΔPPL≈+2158%（更差）→ 细分组不改善真实质量；
- K-only 各粒度（128/64/32/16/1）：ΔNLL≥+2.4（gs32≈+4.1）；
- V-only gs32/128：ΔNLL ≤ +0.002；
- K1/V1 逐元素哨兵 = 0（框架正确性证据）。
- 归因：K 进 QKᵀ 经 softmax 放大；V 在 softmax 后线性加权，本模型对 V 宽容。**限定 Qwen2.5-1.5B + 在线 max-abs E4M3 + eager attention，不外推。**

### 正式 Asymmetric V-only FP8 KV（§3.5.3）→ G3 FROZEN
- V2：prefix 32/32、prefill logits max|Δ| 0.4375、deterministic、finite；
- V3：chunk1024/512 decode16 一致；localize 相对 base 额外 coupling K +0.379 / V +0.891（不要求原始 KV bitwise 同，bf16 GEMM-shape 漂移）；
- **V4 独立窗口（offset 40..79，40×2048，chunk512，与 pilot 0..3 disjoint）**：ΔNLL **+0.000397**；PPL base 10.1041 / V-only 10.1081；absolute ΔPPL +0.0040；relative **+0.040%**；per-window ΔPPL mean +0.0044（min −0.0332, max +0.0370）；finite；
- 8K operational（同脚本 paired）：KV logical 224.00 → 169.75 MiB（**1.320×**，75.78%）；alloc 5.07 → 5.01 GiB；prefill 相近、decode 略慢（每步 V 反量化开销）；
- 记账：每 token/layer 776 B（K 512 + V 256 + v_scale 8）vs 1024 B；**1.32× 仅指 KV cache 逻辑压缩，不是整模型/可行性收益**；跨脚本显存数字不可比。

## 五、本会话文件/验证入口
引擎：`model_self.py`（fmt fp16/fp8/v_fp8、GROUP_SIZE=32、GROUP_V=128、混合缓存、set_fp8_matrices/groups）。
验证：`gpu_bf16_align_probe`、`g1_target_15b`、`fp8_kv_v1_roundtrip`、`fp8_kv_v2_short_window`、`fp8_kv_v3_chunked_ablation`、`fp8_kv_v3_localize`、`fp8_kv_v4_dppl`（正式 40..79）、`fp8_kv_group_probe`（pilot）、`fp8_scale_diag`（弃置探索：分组误差近同）、`g2_operational/g2_peak_probe`、**新增 `fp8_vonly_v1_roundtrip`（V-only gs128 直接结构测试，待跑）与 `fp8_vonly_8k_operational`（base vs v_fp8 paired）**。
文档：`design_freeze_fp8_kv.md` v1.0→v1.3（pilot-driven amendment）、`status_note_fp8_v4.md`（§7 判定、§8 G4 清单）、`data_ledger.md`（§3.4/§3.5–3.5.4/§4.5.1）。

## 六、未决 / 下一步（收尾时点更新）
- [x] `fp8_vonly_v1_roundtrip` 已 GPU 实测 **[1]–[7] 全 PASS → G3 FROZEN**（见 §七）；
- [ ] B2 协议批准（`design_freeze_g4_specdec.md` v0.1 DRAFT → v1.0）后：B3 双模型 8GB 可行性 → B4 cache API → B5 greedy 主循环 → B6 correctness gate；
- [ ] G4-0：0.5B draft 的 G1 式三检查（bf16 行为）可与 B3 一并推进；
- [ ] V2/V3/V4 旧 fp16 时代行为行已在台账标注作废；正式行见 §3.5.3。
- 结论诚实声明：微小负 Δ 不解读为量化提升；+0.040% 不作显著损失；未启动 G4；跨脚本显存不比较。

## 七、晚间续（2026-09-08 会话末收尾）

### 阶段 A 完成：G3 正式冻结
- `fp8_vonly_v1_roundtrip.py`：补 **[7] current==stored** 断言（测试钩子记录首 chunk/层 V 量化结果，与最终 cache 段逐位比较，证写路径无二次量化/错位）；修 prompt 为 `make_text` 96 token、修 map448 索引；实测 **[1]–[7] 全 PASS**；
- **G3: PASS / FROZEN**：正式 = **Asymmetric V-only FP8 KV**（K BF16 + V FP8 E4M3，per-token×per-head scale，gs128）；negative = K-FP8/full-FP8；
- 复核冻结口径：质量目标达到（正式 40 窗 ΔNLL +0.000397、rel ΔPPL +0.040%）；工程稳定达到（V2 32/32、V3 chunk 一致、deterministic、finite）；KV 压缩 1.32×；整机峰值仅 ~0.06 GiB；**eager decode 无加速且 V-only 有反量化开销**；是否改变 SpecDec 可行性 → G4 dual-model 实测。

### 阶段 B 已做（仅文档/讲解，无实现代码）
- B1：SpecDec 原理与数据流讲解（token/logits/K-V cache 状态机、bonus、EOS、预算<mismatch、为何不保证加速）；
- B2：`docs/design_freeze_g4_specdec.md` **v0.1 DRAFT**（k=4/greedy/exact-equivalence 门/cache 不变式与 trim 语义/B3 前置门/范围排除）；
- B3–B6 未实现；待协议批准后推进。

### 文档/进度指针
- 状态：`status_note_fp8_v4.md` §7（A4 冻结确认）§8（G4 前置清单）；`design_freeze_g4_specdec.md`（新）；`data_ledger.md` §3.5.3/§3.5.4 + 变更日志；
- 当日代码/文档已随 `c7eb1d1` 与本收尾快照 commit 入库并推送。

## 八、G4-0 归因登记（续，会话更晚）
- **G4-0: CONDITIONAL PASS — BF16 shape drift characterized**（非完整 G4 PASS）
- 证据：OURS full==HF full、OURS cached==HF cached（max|Δ|=0）；HF 自身 cached/full 亦同处分歧；
  所有路径重复确定；FP32 下四条路径全一致 → 非 ENGINE BUG / NONDETERMINISM，属 BF16 不同计算形状舍入。
- block-vs-serial 探针：被测位置 argmax 一致、cache 终长一致、无 off-by-one；**但 block/serial cache
  数值差 max|Δ|≈2.0** → B6 exact-token-id 门保留，多轮累积仍可能翻转。
- 原始失败样本（cached=' for' vs full=' since', 第 8 步）保留。登记落点：design_freeze_g4 §8.1 / ledger。

—— 2026-09-08 全天归档终。
