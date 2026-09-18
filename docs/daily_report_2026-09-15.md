# SpecDec 日报 2026-09-15

> 前情：2026-09-08 已完成 G1/G2/G3（G3 = Asymmetric V-only FP8 KV，FROZEN），见 `daily_report_2026-09-08b_G1G2G3_FP8KV.md`。
> 本日：G4 推进到"协议修订 → G4-0 归因 → B3 双模型可行性 → B4 cache API"，并开始 B5（`spec_dec` 待写）。

## 一、今日一句话
**协议按 G3 事实修订（v7.2：Compute=BF16、KV 因子 {base_bf16, v_fp8}、12 conditions）；G4-0 归因为 CONDITIONAL PASS（BF16 shape drift，非引擎 bug）；B3 双模型 8GB 可行性 PASS（4/4）；B4 cache API PASS（用户实现 `spec_cache.py`，两档 KV 全 True）；`spec_dec.py` 已建待写，B5/B6 测试已就绪。**

## 二、今日完成（按门）
| 项 | 结果 | 证据/落点 |
|---|---|---|
| **协议修订 v7.2** | 完成 | `plan_v7_1_protocol_freeze.md` §0：BF16 compute；KV 因子 `base_bf16`/`v_fp8`；质量参考=uncompressed BF16；Δ 相对 base_bf16；AR/SpecDec 同档；full/K-FP8 仅 negative ablation；正式=**12 conditions**；带宽归因改为"反量化开销 > 省内存潜在收益，未独立隔离带宽，不作更强归因" |
| **G4 设计审查 → v0.2** | 完成 | `design_freeze_g4_specdec.md` v0.2：**取消 bonus**；logits 对齐规则（`[saved_next, rows[0..Kp-2]]` 判 d1..dKp，`rows[Kp-1]` 不判）；输入/template 一致；cache trim/rollback 精确语义；越界抛错；OOM `finally`；base 与 v_fp8 共用同一逻辑 |
| **G4-0 0.5B draft** | **CONDITIONAL PASS** | OURS full==HF full（max\|Δ\|=0）、OURS cached==HF cached；HF 自身 cached/full 亦同处分歧；重复确定；FP32 恢复 → 归因 **BF16 shape drift（非 ENGINE BUG / 非 NONDETERMINISM）**；block-vs-serial 被测位置 argmax 全一致、cache 终长一致、**无 off-by-one**；但 block/serial cache 数值差 max\|Δ\|≈2.0 → **B6 exact-id 门保留为风险**（不写成"B6 无风险"） |
| **B3 双模型 8GB** | **PASS（4/4）** | 1.5B+0.5B 同驻 bf16；ctx512：alloc 4.34/4.336 GiB；ctx8192：alloc 6.16 GiB、resv 8.92 GiB；base/v_fp8 均无 OOM、finite、3 轮稳定、smoke 首 token target/draft 一致；权重常驻后 free 2.00 GiB（smi 5.76） |
| **B4 cache API** | **PASS** | 用户实现 `spec_cache.py`（65 行）；`g4_b4_cache_tests`：base(0.5B) 与 v_fp8(1.5B) 全 True（len/append/trim/raises=3/3/idem/orig-unchanged/v_scale-sync）；坏 cache 被 `check_consistent` 抓到 |
| **B5/B6 测试** | 已就绪（未运行） | `g4_b5_smoke.py`（结构不变量+确定性）、`g4_b6_correctness.py`（10 类 case × 两档 KV 的 exact token-id 门） |
| **B5 主循环** | **未实现** | `specdec/engine/spec_dec.py` 已建文件，**0 行**；代码指导（签名/五步循环/对齐规则/回滚长度/6 个坑）已给出 |

## 三、关键数据（详见 `data_ledger.md` §4.5.2/§4.5.3）
| 条件 | alloc 峰值 | resv 峰值 | 结果 |
|---|---|---|---|
| ctx512 base_bf16（双模型） | 4.34 GiB（3/3 同） | 4.58 | 无 OOM / finite / 稳定 → PASS |
| ctx512 v_fp8（双模型） | 4.336 GiB（3/3） | 4.58 | PASS（vs base −0.003 GiB）|
| ctx8192 base_bf16 | 6.16 GiB（3/3 同） | 8.92 | PASS（resv > 8 GiB 物理：WDDM 池超订，未 OOM）|
| ctx8192 v_fp8 | 6.16 GiB（3/3 同） | 8.92 | PASS（alloc 峰值与 base 相同，不宣称峰值收益）|
> 不跨实验比较显存：≠ G2 fp16 单模型 6.83、≠ G3 单模型 paired 数字。

## 四、今日新增/修改文件
**代码（全部为测试/诊断；核心 `spec_cache.py` 由用户实现）**
- 新增：`g4_0_draft_correctness.py`、`g4_0_drift_diag.py`、`g4_block_vs_serial.py`、`g4_b3_dual_model_feasibility.py`、`g4_b4_cache_tests.py`、`g4_b5_smoke.py`、`g4_b6_correctness.py`
- 用户实现：`spec_cache.py`（B4 PASS）；`spec_dec.py`（空文件，待写）

**文档**
- 新增：`design_g4_cache_api.md`（cache 接口契约）、`plan_g4_next_steps.md`（七阶段规划）
- 修改：`plan_v7_1_protocol_freeze.md`（Amendment v7.2）、`design_freeze_g4_specdec.md`（v0.2 + §8.1 G4-0 登记）、`data_ledger.md`（§4.5.2 B3、§4.5.3 B4、变更日志）、`daily_report_2026-09-08b_G1G2G3_FP8KV.md`（§八 归因登记）

**非本会话创建（未改动）**：`docs/specdec_frozen_submission_proposal_3members.md`、`scripts/generate_specdec_proposal_pdf.py`

## 五、今日排查与经验（都值得写进报告方法学）
1. **`v_fp8` 不能用于 0.5B**：`GROUP_V=128` 要求 `head_dim%128==0`；0.5B head_dim=64 → 引擎断言拦下。故 `v_fp8` 是 **target 专属**格式，draft 恒 base；B4 的 v_fp8 段改用 1.5B。
2. **"静默退出"实为网络异常**：`AutoTokenizer.from_pretrained` 会联网 `list_repo_templates`，网络抖动抛 `httpx.RemoteProtocolError`（曾被误判为驱动杀进程）。修法：harness 一律**本地优先**（`local_files_only=True`）。
3. **引擎 cache 是原地推进的**：`_run` 复用同一批 dict（`cache["k"]=cat(...)`）→ `pre` 必须在 block forward **之前**取；已写进 `spec_dec` 指导与测试注释。
4. **trim 到 0 的契约定死为非法**：状态机里真实 token 数恒 >0，允许 0 会掩盖 `pre+m` 算错的 bug；文档+测试+实现同步（`new_len<=0` 抛 `ValueError`）。
5. **测试自身的几处 bug 已修**：`raises` 期望值（0 属非法）、m=0..4 需先把 cache 扩到 `pre+4` 再裁、mask 广播维度、`cache_lenth` 拼写。
6. **加载显存优化**：HF 权重先 CPU 加载、只把 ours 搬上 GPU（GPU 峰值 ≈6.9→3.9 GiB 可用），对 8GB 卡更稳。

## 六、未提交状态（等你确认后提交）
工作树含全部上述新增/修改（含 Amendment v7.2、G4 doc v0.2、7 个 harness、2 篇文档、台账/日报），以及你的 `spec_cache.py`、`spec_dec.py`。远端 `main` 仍是 `da2fac7`（09-08 快照）。**未自动提交**。

## 七、风险
1. **B6 exact-id 门**：BF16 下 block-built 与 serial-built cache 数值差 ≈2.0 → 多轮累积可能翻转 token；B6 未过前不得测 acceptance / pilot / final；
2. **双模型 8K 峰值 6.16 GiB**：长 decode / spec-dec 多轮后峰值是否会涨需实测（B3 只跑 5 tok smoke）；
3. `spec_dec` 状态机（对齐/回滚）尚未实现与验证。

## 八、明日计划（顺序）
1. 写 `spec_dec.py`（指导已给：签名 / 五步 / `[saved_next, rows[:Kp-1]]` / `pre+m+1` 与 `pre+m` 回滚 / 6 个坑）；
2. 跑 `g4_b5_smoke` → `g4_b6_correctness`（两档 KV，exact token-id）；不一致先看 margin 再定位；
3. B6 通过后做 Part 4 小规模测量（acceptance/延迟/峰值，warm-up + ≥3 次 + 中位数）；
4. 并行：G5 instrumentation 规格、≥16 prompt families 设计、10/7 提案（队名/分工）核对。
