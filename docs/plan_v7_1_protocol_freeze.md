# SpecDec v7.1 —— Experimental Protocol Freeze

> 目标：不改变方向、不加功能、不扩 scope；只修复影响正式实验**有效性/可解释性/可复现性**的协议缺口。
> 依据：v7（方向已过审）+ 2026-09-08 日报已证实事实 + data ledger。

---

## 0. Amendment v7.2（2026-09-08，protocol amendment —— 覆盖下述各条的旧表述，历史文本保留不改写）

**修订动因**：G3 pilot 证明"full-FP8 K/V"在本模型不可用（K-FP8 各粒度 ΔNLL≥+2.4），V-only FP8 无损；同时本机 CUDA FP16 对本 Qwen 路径 NaN → 正式质量口径必须 BF16。故把"KV precision={FP16,FP8}"的旧因子设计修订为下述结构。

**修订内容（生效于正式 factorial；旧 FP16 时代数字 = 历史 operational evidence，不进正式汇总）**：
1. Compute dtype：正式质量与性能实验**统一 BF16**（不做"性能 FP16 / 质量 BF16"双口径）。
2. KV representation（正式 factorial 因子，2 档）：
   - `base_bf16`：K=BF16，V=BF16（质量参考，uncompressed）；
   - `v_fp8`：K=BF16，V=FP8 E4M3，per-token × per-KV-head scale，group=128（G3 正式方案）。
3. ΔNLL/ΔPPL 一律**相对 base_bf16** 报告；质量参考 = uncompressed BF16 KV（不再是 FP16 KV）。
4. AR 与 SpecDec 在**相同 KV representation** 下比较（同档内比；不跨档做分母）。
5. full-FP8 K/V 与 K-FP8 仅保留为 **G3 negative ablation**，**不进入正式 factorial**。
6. 正式 factorial = **3 context lengths × 2 KV representations × 2 decoding methods = 12 conditions**（本协议 §5/§6 已是 12 conditions；历史文档中的"18 格"为 v6/v7 内部 bookkeeping，已弃用，正式统计一律按 12 conditions 计）。
7. 带宽归因口径（修正旧句"eager 下显存带宽不是瓶颈"）：当前 eager 实现**未观察到 V-only FP8 的延迟收益**——反量化开销超过了 KV 内存访问减少的潜在收益；本实验未独立隔离显存带宽，因此**不作更强归因**。
8. G3 正式门（更新 §7 的 G3 定义口）：FP8 方案定案 = **Asymmetric V-only FP8 KV**（§0 第 2 项），质量/工程/显存证据见 `data_ledger.md` §3.5.3–3.5.4 与 `status_note_fp8_v4.md`；原"full-FP8"表述仅作历史与 negative ablation 引用。

---

## 1. v7 → v7.1 Change Log（仅协议级修改）

| # | 变更 |
|---|---|
| P1 | 执行顺序重排：L4 weight 量化移出主线；P1=1.5B correctness + 8K operational |
| P2 | 新增 **Context Construction Protocol**：paired prompt families（2K/4K/8K 同 family，nested prefix）|
| P3 | 冻结生成参数与 spec 参数（k 等）|
| P4 | Spec k 选择规则前置：**k=4 by design**（禁止 post hoc 选 k）|
| P5 | Speedup 数学定义：decode-only 与 e2e 分离；禁止跨 precision 分母 |
| P6 | Prefill 处理：decode-only=primary，e2e=secondary（含 target/draft prefill 组成）|
| P7 | FP8 改称 **FP8-compressed KV**；格式/缩放/scale lifetime/反量化/记账/质量均设设计冻结门 |
| P8 | 质量协议冻结：数据集/窗口/绝对+相对 ΔPPL%，不设事后阈值 |
| P9 | 统计协议两级聚合（reps→median；families→median+IQR），禁 fastest run |
| P10 | Pilot(4 fam×12 cond×3 reps) 与 Final(≥16 fam×12 cond×5 reps) 分离 |
| P11 | 条件执行交错/随机化 + 温度/时钟/VRAM/时间戳记录 |
| P12 | Break-even：相邻点跨 1 才触发 ≤2 adaptive probes；区间报告优先 |
| P13 | Failure Analysis 增加 F6（decode 快但 e2e 掉）、F7（FP8 只解决 feasibility）|
| P14 | G2 重定义 = **8K Operational Feasibility**（非"实现 chunked prefill"）|
| P15 | 事实边界：Plan=intended/protocol；Ledger=measured only；日报不是机器数据唯一来源 |

## 2. Frozen Research Questions（不再扩展）
- **RQ1**：How does context length affect speculative-decoding speedup on an 8GB consumer GPU?
- **RQ2**：Is the degradation mainly associated with reduced accepted-token benefit or increased draft/verification overhead?
- **RQ3**：Does FP8-compressed KV cache change the profitable context range of speculative decoding?
- 禁止新 RQ；kernel/weight/scaling/alt-drafts/dynamic trees 仅 supporting/optional。

## 3. Current Verified State（与日报一致，严格三分）
**Completed**
- 基础 eager engine；0.5B development correctness（L1 对齐 0.000、L2 缓存等价、生成对齐 PASS、4K chunked-prefill 等价 PASS）；
- 分块 prefill 实现（`prefill_chunked(ids, chunk=1024)` + need_logits）；8K 一次性 OOM 根因已定位；transformers eager 与 OUR baseline sanity 完成。

**Partially verified**
- 8K **operational** feasibility：chunked-prefill 正确性✅；**1.5B target + 8K + 峰值显存 + 时长 + 稳定多次运行 = 未完成**；
- 正式基准基础设施：有 bench/probe 脚本，但 **results schema / resume / config-hash 冻结未完成**。

**Not started**
- 1.5B target correctness gate；FP8 KV（设计+实现）；fixed-chain spec-dec；prompt families；pilot；final factorial。

## 4. Experimental Protocol
**4.1 Context Construction（paired families）**
- base docs：公共开放长文本（Wikipedia/公开报告语料拼接；不依赖事实评测，仅作解码上下文）；
- 每个 **family i** 含 3 个变体：2K/4K/8K，构造 = **nested prefix**（2K ⊂ 4K ⊂ 8K 为同一文本链前段），**query/任务指令固定**并置于末尾（如 "Continue the text."）；
- 截取策略：**prefix-controlled**（保留前 N tokens，任务恒为续写 → 不引入 QA 证据位置混淆）；可加简短 filler 使长度精确；±tolerance ≤32 tokens，记录实际 token 数；
- tokenizer 固定；family manifest JSON 版本化（id、源、长度、hash、query、task 类型）；
- 正式最低 **16 families**；Pilot 用 4 families。

**4.2 Generation（冻结）**
greedy；temperature=0；top-p off；max_new_tokens=固定（如 128，记入 manifest）；EOS 处理=遇到 eos 停但记录生成长度，condition 间同策略；tokenizer 版本固定；BOS/EOS policy 同 tokenizer 默认、两端一致；padding=none（batch=1）；prompt token 数按 manifest 记录；first-token latency 单独记（TTFT 列）。

**4.3 Speculative decoding（冻结）**
- proposal length **k=4**（by design，见 §Spec-k）；verification=greedy 语义：按序比较 target greedy token 与 draft token，接受一致前缀；首个不一致处取 target token；
- rejection 后：丢弃被拒 draft 段（target cache 不rollback 错误段；draft cache rollback 到接受前缀处）；draft cache 每步随其自身前缀更新；
- draft 与 target 同词表；draft 在 context 上先做 draft prefill（见 e2e 定义）。

**4.4 FP8（FP8-compressed KV，P2 设计冻结门）**
- 名称：**FP8-compressed KV cache**（非 native FP8 attention）；格式 E4M3（设计门内复核 E5M2）；缩放 scheme/scale lifetime/反量化点（读时 fp8→fp16 再进 eager attention）/记账（payload+scale 元数据计入；理论 bytes/token 与实测峰值显存分开报）/ΔPPL protocol —— 全部在实现前于设计冻结门定死。

**4.5 Timing & Speedup（冻结）**
- decode-only：`S_decode(L,q)=T_AR_decode/T_SPEC_decode`；e2e：`S_e2e(L,q)=T_AR_total/T_SPEC_total`；q∈{FP16,FP8}，**同精度分母**（FP8 Spec 对 FP8 AR；禁 FP8 Spec/FP16 AR）。
- AR total=target prefill+decode；Spec total=target prefill+draft prefill+speculative decode（实现是否共享 target/draft 的 context 处理需在代码中明示并一致）。

**4.6 Quality（冻结，无事后阈值）**
dataset=Wikitext-2 test；窗口=非重叠固定上下文窗口；每 condition ≥32 窗口；FP16 同窗口 reference；报告 absolute PPL + relative ΔPPL%；保留 δ 仅当正式 FP8 前预注册（本 freeze 默认不设人工 PASS/FAIL δ）。

**4.7 Statistics（冻结）**
- Level A：每 (family×ctx×KV×decode) ≥5 reps（最低 3）→ **median latency**；reps 不是独立样本；
- Level B：每 condition 跨 families 报 median+IQR（box/violin/point overlay，每 family 一个点）；禁 fastest-run、禁单一平均 tok/s。

## 5. Pilot Protocol
4 families × 12 conditions × 3 reps；只查：OOM、instrumentation 正确、cache/spec correctness、timing variance、FP8 stability、schema、resume/recovery、thermal。Pilot 数字不进入最终主表（除非协议未变且注明）。

## 6. Final Factorial Protocol
≥16 families × 12 conditions × 5 reps（可扩 24/32 仅在资源充足时）；12 conditions 表（3 ctx ×2 KV×2 decode）列：Prompt Family ID / Context tokens / KV format / Decode method / Spec k / Output tokens / Prefill time / Decode time / Total time / Draft time / Verify time / Acceptance / tokens/s / Peak VRAM / Temperature / Repetition ID / Config hash。resume/recovery 测试通过后才允许整批。

## 7. Gates
- **G1** 1.5B target correctness：OUR full vs HF full <1e-3；OUR cached vs OUR full <1e-3+argmax；greedy 序列一致协议 → 写入 ledger。
- **G2** 8K Operational Feasibility：1.5B + 8K + chunked prefill + 不 OOM + 完成规定 decode 长度 + peak VRAM 记录 + 连续多次稳定 + correctness sanity PASS。
- **G3** FP8 design freeze（格式/缩放/反量化/记账/质量协议定死）+ FP8 quality sanity。
- **G4** Spec correctness：k=4 下 output equivalence PASS（greedy verify 无损 sanity）。
- **G5** Instrumentation：results schema / config hash / resume / write 测试 PASS。
- G1–G5 全过 + Freeze Checklist 全勾 → Pilot → Final。

## 8. Updated Execution Order
- **Priority 1**：① 1.5B target correctness（G1）；② 8K chunked-prefill operational validation（G2）。
- **Priority 2**：③ FP8 KV（G3）；④ Speculative decoding（G4）；⑤ instrumentation（G5）。
- **Priority 3**：⑥ prompt families 冻结；⑦ pilot；⑧ final factorial + quality sanity + attribution/failure/figures + reproducibility/report。
- **Optional only**：weight int8/int4（若已开始：≤4–6h timebox 最小 sanity，超时即停）；CUDA K1（不进入主线）。
- 明确：**weight quantization 不得阻塞 FP8 KV / SpecDec / benchmark。**

## 9. Scope / Cut Order
1 砍 CUDA K1；2 砍 weight 量化（int4 删除；int8 仅 ≤4–6h sanity）；3 砍 FP8 secondary QA 与不稳定 adaptive probes；保 controlled eval/reps/failure/repro/figures。超时预案同 v7（最小版本 2K/4K×FP16×AR/SpecDec + 诚实 negative）。

## 10. Freeze Checklist（全勾才 Final）
- [ ] Target 1.5B correctness PASS
- [x] 8K operational feasibility PASS（ledger §4.5.1，2026-09-08）
- [ ] paired prompt-family protocol frozen
- [ ] ≥16 final prompt families frozen
- [ ] generation settings frozen
- [ ] Spec k frozen（k=4）
- [x] FP8 format/scaling frozen（design_freeze_fp8_kv.md v1.0，2026-09-08）
- [x] G3 formal PASS / FROZEN（2026-09-08：Asymmetric V-only FP8 KV = base_bf16×v_fp8 因子落地；见 §0 Amendment v7.2 与 ledger §3.5.3）
- [ ] speedup equations frozen
- [ ] decode-only vs E2E definitions frozen
- [ ] quality dataset/protocol frozen
- [ ] repetitions/aggregation frozen
- [ ] instrumentation schema frozen
- [ ] pilot completed
- [ ] no unexplained OOM / cache mismatch
- [ ] results.jsonl write/resume tested
- [ ] main code/config commit hash frozen

## 11. Professor/TA Adversarial Review
- **当前是否还有 critical flaw？** 设计层面无致命缺陷；剩余最大不确定性在执行可行性（8K@1.5B、FP8/spec 实现正确性），协议已把每个点设为 gate。
- **最大 confounder？** context 长度与内容混淆 → 已用 nested-prefix families 消除；次之：laptop thermal → 交错+记录；再次：eager 实现中 decode-only 与 e2e 的口径混用 → 已分离。
- **哪个指标最容易被误解？** "net speedup"（须注明 decode-only 还是 e2e）与 acceptance（不是质量指标）；ppl 仅指量化质量。
- **哪个工程任务最可能超时？** FP8 KV + spec-dec 的 instrumentation/resume 韧性，以及 8K@1.5B 的墙钟/稳定性。
- **若 FP8 不加速，项目是否仍成立？** 成立——F3/F7 属有效 negative（feasibility/mixed）；只需如实归因与解释。
- **若 8K 跑不了，如何合法收缩 RQ？** 报告 8K 为 implementation-limited（F5 区分 algorithmic/implementation/hardware），主结论合法表述为"context up to 4K"（RQ1 范围内收缩+limitation）；仅当 4K 也不可行才考虑 pivot。

## 12. Final Verdict
**READY FOR IMPLEMENTATION**
原因：方向与协议已冻结、无 critical 设计缺陷；但 G1（1.5B correctness）与 G2（8K operational）尚未完成，Pilot/Final 均不可启动。下一步 = 执行 Priority 1（G1 → G2），随后按 freeze checklist 推进。
