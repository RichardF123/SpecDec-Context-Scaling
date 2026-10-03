# SpecDec 阶段状态同步（2026-10-02）

本记录汇总截至 `f6335f5` 的已有结果与后续分析，不包含新 GPU 实验。不修改 proposal、生产默认、正式验收门槛或旧 FAIL/CHECK。

## 当前研究范围与状态

- 以 [2026-10-01 定稿](proposal_freeze_2026-10-01.md) 为准：主实验 2K/4K/8K × AR/Spec、base BF16，共 6 条件；补充实验为 8K V-only FP8 × AR/Spec，共 2 条件。
- 引擎、KV cache、V-only FP8、greedy SpecDec、测量和诊断工具已有实现与阶段验证。
- 生产 BF16 的 AR/Spec 严格逐 token 一致性仍未通过。历史 FP32 B6 通过不能替代此门。
- 正式实验尚未开展，不作正式 speedup、最终 break-even 或全条件稳定性声明。
- G3 独立 40 窗结果保留：ΔNLL +0.000397，relative ΔPPL +0.040%；target 8K 逻辑 KV 224.00 → 169.75 MiB，约 1.32×。这不是整机压缩率或速度收益。

## 最新证据

1. [RoPE 来源对齐](c1_rope_source_alignment_2026-10-01.md)：诊断中使用 CPU FP32 构造 target inv_freq，所测 C1 的 56/56 KV 张量与完整末位置 logits 与 HF 逐位一致。不能据此声称所有执行路径已一致；生产默认未改。
2. [真实 AR/Spec](ar_spec_aligned_rope_2026-10-01.md)：F13、ctx=8192、chunk=2048、k=4、max_new=64、base BF16、all-eager，target RoPE 对齐而 draft 保持原样。完整序列只在零基索引 47（预测位置 8239）不同：AR=6702、Spec=6652。
3. [现场复现与四格定位](divasc_index47_localization_2026-10-01.md)：新捕获复现上述序列；历史来源与当前执行方式分开控制，结果如下。

| 起点缓存历史 | single | block |
|---|---:|---:|
| AR | 6702 | 6702 |
| Spec | 6652 | 6652 |

真实路径的重放行与捕获行逐元素一致；每格重复一致。相同历史下 single/block logits 仍有数值差异，但没有改变本位置 argmax。相同执行方式下更换历史则改变选择。

AR 历史下 6702/6652 的 logits 为 20.125/20.0；Spec 历史下均为 20.125，实际 argmax 选择 6652。候选 5470 已被拒绝，差异发生在 target 替换判断，不是忘记拒绝候选。

两侧 prefill 缓存位置 0..8191 逐位一致；生成缓存最早从位置 8192 出现差异。长度 8237 的快照中 55/56 张量不同。本决策点的行映射、替换规则、截断边界和尾部泄漏检查未发现问题。

## 结论边界

- 已有受控证据支持：缓存历史影响位置 8239 的选择。相同 token 前缀并不保证不同计算路径构造的 KV 数值逐位相同。
- 尚未定位最初产生差异的具体算子；不能把局部历史效应升级成“全部问题最终根因已确定”。
- 本轮 all-eager 仍有分歧，不能直接把问题归咎于 cuDNN；这也不意味着所有后端相互等价。
- 本位置未发现逻辑错误，不代表全流程逻辑审计完成。margin 不是数值差异的免责门槛。
- 后续 token 再次相同不代表缓存恢复相同；局部修复也不保证其他位置、其他家族通过。
- 原始大张量/运行 JSON 的本地路径见各报告；本次同步不新增上传这些运行产物，公开报告不等于完整原始数据发布。

## 下一步建议修正（未执行）

定位报告 §7 原建议只处理 mismatch 轮，不能保证建立串行一致历史。本说明补充分析，不改写历史运行结果：

1. 默认关闭的诊断需要从相同 prefill 起点开始，覆盖全接受轮与拒绝轮，因为两种分支都会保留块内 KV。
2. 每轮必须从轮前 `pre` 的完整缓存快照重放实际提交的 token，而不是在已保留这些 token 的 `pre+m` 缓存后重复追加。
3. 同步刷新 target `saved_next`，避免串行缓存配旧块 logits；保持实际输出、EOS、长度和缓存不变量可核对。
4. 优先验证共同 token 前缀下位置 8237 的缓存是否与 AR 逐位一致，以及原判断位置是否恢复选择 6702。若前缀提前分歧，不能强行比较不同逻辑历史并归因。
5. 完整 64-token 一致性是额外观察，不作为历史效应成立的唯一判据。即使缓存历史一致，当前块计算仍可能在其他位置改变选择；新分歧需单独记录。

这只是待执行的因果诊断，不是已验证修复，也不是最终性能方案。串行重放增加 target 工作量，不得隐藏其成本或把外部 AR 缓存免费提供给正式 Spec 性能测试。保持串行 GPU、现有资源护栏和严格正确性门；未经后续授权不执行新实验。

## KV Optimization Final GPU Closeout（2026-10-02，独立补充线）

这是一条与上述 production BF16 AR/Spec correctness 调查相互独立的 KV
optimization engineering extension。最终 GPU 工作已经完成；本节只同步已归档结果，
不修改 proposal、正式 6+2 范围、历史 FAIL/CHECK 或 correctness 状态。

- Paged KV storage、BF16/V-FP8 direct paged CUDA、`q_len=1..4`、
  decode/propose/verify/repair direct route、target verify 和 pointer-table reuse 已完成，
  real-model / SpecDec parity 通过。Prefill 保持既有 materialized/fused path。
- 最终 microbenchmark 的 direct path 在所有 12 个配置中都不快于 materialized
  reference；已验证收益是避免 full-history materialization，并显著减少
  attention-call temporary allocation/peak delta，不是整机 VRAM 或 latency speedup。
- A–D 在 6 个小规模 exact-input cases 中保持 exact output；V-FP8 的 combined
  final payload ratio 为 `0.83046875`。H2O 50% 的 E/F exact rate 均为
  `0.6666667`、mean token agreement `0.875`，因此不得称为 lossless；G 的 25%
  retention 是 aggressive stress。
- H2O paired interaction audit：quantization-only exact `12/12`；keep-set/state
  change `12/12`；9 个 `retention_changed_output_same`，3 个
  `retention_diverged_before_output`（均来自 ratio `0.25`）。这是有边界的
  retention-mediated evidence，不是完整因果证明。
- H2O score capture 与 compaction 仍是 reference/materialized 路径，不能声明
  H2O latency improvement。

权威总结和 canonical `_final` 文件分别见
[`kv_optimization_closeout_2026-10-02.md`](kv_optimization_closeout_2026-10-02.md)
与 [`results/kv_closeout_2026-10-02/`](../results/kv_closeout_2026-10-02/)。
