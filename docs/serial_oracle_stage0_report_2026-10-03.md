# 串行 Target Verification Oracle — Stage 0 报告（2026-10-03）

> 设计文档（**代码之前冻结**）：`docs/serial_target_oracle_design_2026-10-03.md`
> 本报告只记录**已经发生的事实**；凡未运行的一律标注"未运行/未验证"。

## 0. 状态声明（严格区分五类）

| 类别 | 内容 |
| --- | --- |
| **已存在（上一轮及更早）** | `spec_dec.spec_decode_greedy` 的 accept/replacement/rollback/EOS/截断控制逻辑；`spec_cache`；`spec_dec._verify`（block，`s=Kp`）；`spec_dec._step`；row-aligned `_VERIFY_WRAP`（默认关闭）；`smoke_hooks.DecisionHooks`；`g5_8k_smoke_cudnn` harness；`target_rope_scope` |
| **本次新增·已实现且已验证通过（纯 CPU）** | ① `spec_dec` 的**可插拔 target verifier**（`_VERIFY_MODE` / `set_target_verify_mode` / `target_verify_mode` / `VerifyModeError` / `_step_rows` / `_verify_serial` / `_SERIAL_TRACE`）；② `serial_oracle.py` 的调用几何审计（Stage 0 用 `_Audit` + GPU 用**阶段感知、按实例分组**的 `install_run_audit` / `run_audit_summary`）；③ `serial_oracle_stage0.py`（**PASS 42/0**）；④ `serial_oracle_audit_selftest_cpu.py`（**PASS 19/0**） |
| **本次新增·已实现但未验证（需要 GPU）** | `serial_oracle_gpu.py`（Stage 1/2 运行器）、`serial_oracle_aggregate.py`（只读汇总器）、`scripts/so_batch.ps1`（顺序批运行）。**语法/接口已核对，但尚未产生任何一次成功的 GPU 运行** |
| **设计完成·未实现** | Stage 1/2 的 5-family 结论；任何关于"serial == AR"的数值结论 |
| **未验证的核心命题** | "若 target verification 完全走与 AR 相同的逐 token `M=1` 路径，SpecDec 的控制逻辑是否生成与 target-only AR 完全相同的 token IDs" —— **尚无任何 GPU 证据** |

## 1. 交付物清单（文件 / 函数 / 契约）

| 文件 | 新增/改动 | 作用 |
| --- | --- | --- |
| `specdec/engine/spec_dec.py` | 新增 `_VERIFY_MODE`、`VERIFY_MODES`、`VerifyModeError`、`set_target_verify_mode`、`target_verify_mode`、`_step_rows`、`_verify_serial`、`_SERIAL_TRACE`/`_SERIAL_TRACE_PRE`、`serial_trace`、`clear_serial_trace`；`_verify` 内部分派 | 串行 verifier + 每轮 trace；**默认 `"block"` 时生产路径逐字不变** |
| `specdec/engine/serial_oracle.py` | 新增 `CallGeometryError`、`_Audit` / `target_call_audit` / `draft_call_audit`（Stage 0 用）、`GeometryAuditError`、`install_run_audit`、`run_audit_summary`、`_label_of`（GPU 用） | 调用几何审计：**非 prefill 阶段的 `s>1` 立即抛错** |
| `specdec/engine/serial_oracle_stage0.py` | 新增 | Stage 0：13 例 + 2 负对照 + 1 接入层用例，**脚本化 stub + 真实控制逻辑** |
| `specdec/engine/serial_oracle_audit_selftest_cpu.py` | 新增 | 审计器**自身**的自测（含反空泛与"必须能失败"） |
| `specdec/engine/serial_oracle_gpu.py` | 新增 | Stage 1/2 运行器（一个进程 = 一个 (case, mode, rep)） |
| `specdec/engine/serial_oracle_aggregate.py` | 新增 | 只读汇总 + 按**预登记口径**判定 |
| `scripts/so_batch.ps1` | 新增 | 顺序批运行（不并发、不自动重试、逐行落日志） |

### 1.1 串行 verifier 的 contract（与 block **完全一致**）

```
_verify(target, cache, cands) -> (rows, cache_blk)     rows.shape == (kp, V)
   └─ _VERIFY_MODE == "serial" ⇒ _verify_serial(target, cache, cands)
```
* 输入：`cache`（全部层的 KV，token 维长度 = `pre`）、`cands`（本轮候选 token，长度 `kp`）；
* 输出：`rows[j]` = 喂入 `cands[j]` **之后**位置 `pre+j` 的 logits 行（`(kp, V)`）；
  追加后的 cache（长度 `pre+kp`）；
* **唯一区别**：block 用**一次** `_run(s=kp)`；serial 用 **kp 次** `_run(s=1)`（与 AR 单步完全相同）。

### 1.2 自启动到结束的完整 I/O（一句话链路）

`prefill`（分块，每次 `s=chunk`）→ `saved_next` → 每轮：`_propose`（draft 逐 token，`s=1`）→
`_verify`（**mode 分派**）→ `g` → `m` → `emitted` → `trim_cache` + `_step`（replacement）或
`tgt_cache = tgt_blk`（全接受）→ EOS/预算停止。

## 2. Stage 0 结果：**PASS 42 / FAIL 0**（纯 CPU、无模型、无 GPU）

产物：`output/g5/attn/serial_oracle_stage0_20261003T134928Z.json`（首轮失败记录
`serial_oracle_stage0_20261003T134138Z.json` 与第二轮 `..._20261003T134239Z.json` 均**保留**）。

脚本索引约定（由 `spec_dec` 语义**推导**，不是猜的）：`a[P-1] = saved_next/drf_lg`；
`cands[j] = a_draft[P-1+j]`；`g[j] = a_target[P-1+j]`；**block 与 serial 消费同一批 `kp` 行**。

| # | 用例 | 该用例在证明什么 | 结果 |
| --- | --- | --- | --- |
| 1 | `m=0` | 第一个候选被拒时 emitted = 单个 replacement；target generation 调用全 `s=1`；审计器 `all_m1` | ok（4 检查） |
| 2 | `m=1` | 部分接受 + replacement 的拼接顺序 | ok（2） |
| 3 | `m=Kp-1` | 只在最后一位分歧 | ok（2） |
| 4 | `m=Kp` | **全接受**：`emitted == cands`，`saved_next = rows[Kp-1]`，**bonus 下一轮才 emit** | ok（4） |
| 5 | EOS@replacement | EOS 只经 replacement 路径进入，被接受前缀内无 EOS | ok（2） |
| 6 | EOS@candidate | draft 提前 break ⇒ `Kp < k_eff` 且本轮结束 | ok（2） |
| 7 | 预算截断 | `Kp = k_eff < K`，emitted 受 `max_new` 约束 | ok（2） |
| 8 | **block vs serial 全循环等价** | 同一脚本下 gen / 逐轮控制流 / stats 逐项相同；serial 的 `step_argmax` 复现 block 的 `rows`；**全接受轮的 bonus 链**一致；block 确实出现 `s=Kp` 调用 | ok（6） |
| 9/10/11 | cache 提交与回滚 | `tgt_len == drf_len == pre+(m+1\|Kp)`；被拒候选**不残留**；`tgt_len == prompt_len + 累计 emitted` | ok（4） |
| 12 | 异常恢复 | 上下文异常后 `_run` 恢复（按 `__func__`+`__self__` 核对）、审计不再安装、模式回到 `block`、异常照常抛出 | ok（4） |
| 13 | serial 关闭 | block 路径出现 `s=Kp` 的 generation 调用；`serial_trace` 为空 | ok（3） |
| 14 | **负对照** | 几何审计**必须能失败**：block 模式 + 强制审计 ⇒ `CallGeometryError` | ok（1） |
| 15 | **负对照** | serial 与 row-aligned wrapper **互斥**：两种安装顺序都被 `VerifyModeError` 拒绝 | ok（2） |
| 16 | **接入层等价性** | 用与 `smoke_hooks` **相同手法**（替换 `spec_dec._verify`）包装后：每轮恰好一次调用、serial 路径确实执行、结果与未包装**逐位相同** | ok（4） |

### 2.1 审计器自测：**PASS 19 / FAIL 0**（`serial_oracle_audit_selftest_20261003T135435Z.json`）

`all_target_calls_m1 = True` 是**结论性证据**，因此审计器本身必须先被检验：

* `prefill` 阶段允许多次大调用（真实 harness 是**分块** prefill）；非 `prefill` 阶段 `s>1` ⇒ **抛错**；
* **反空泛**：零调用 / 零 generation 调用 ⇒ `sane=False`、`certifies_m1=False`、`target=None`
  （**绝不**因"集合为空"而判 True）；
* **身份识别**：无 `smoke_hooks` 注册表 ⇒ 标签 `unknown` 且**不通过**（绝不按 head_dim/调用顺序猜）；
  有注册表 ⇒ 按 `is` 判定 target/draft，并用层数（28 vs 24）与参数量交叉核对；
* `uninstall()` 用 `is` 核查类属性已还原；类属性替换/恢复逐项核对。

## 3. 缺陷与修正（全部保留首次失败记录，**未放宽任何门**）

| # | 缺陷 | 类别 | 表现与发现方式 | 修正 |
| --- | --- | --- | --- | --- |
| D1 | Stage 0 `c05` 期望 `m=0` | **我的断言错**（非代码错） | 首轮 FAIL 35/2。由脚本约定 `g[0]=a_t[P-1]` 与 `cands[0]=a_d[P-1]` 相等 ⇒ 实际 `m=1`，EOS 落在 replacement 位 | 改为精确核对"EOS 经 replacement 路径进入"（`m<Kp` 且被接受前缀无 EOS） |
| D2 | Stage 0 `c12` 用 `t12._run is orig12` 核对恢复 | **无效的身份比较** | 同上轮 FAIL。`_run` 是类属性（bound method），每次访问都是**新对象** ⇒ 恒为 False | 改为核对 `__func__` + `__self__`，并加 `not isinstance(_run, _Audit)` |
| D3 | serial 分派点放在**主循环** | **集成缺陷**（Stage 0 纯逻辑测试**发现不了**） | 在接入 GPU 前逐项核对 `smoke_hooks` 挂载点时发现：harness 是把 `spec_dec._verify` 换成包装器来推进轮次记账的 ⇒ 主循环绕过 `_verify` 会**静默掏空** `begin_round`/`on_round_start`/`on_verify_pre`/`verify` 阶段的 `rows` 有限性检查，并使 `smoke_checks.ALWAYS_REQUIRED` 的 `("verify","target")` 缺失 ⇒ `checks.passed=False` | `_verify` 成为**唯一分派点**（模式在 `_verify` **内部**选择），主循环调用形式与生产逐字相同；新增用例 16 把该性质固化为回归测试 |
| D4 | `_step_rows` 复用模块级 `_step` | **记录口径缺陷**（非数值） | 生产里 `_step` 被 `smoke_hooks` 包装为 `repair` 阶段 + `saved_next` 角色 ⇒ serial 每轮 `kp` 次校验被误标为 repair，`("repair","target")` 计数从 1 次/轮抬到 `kp+1` 次/轮 ⇒ 与 block 记录**不可比** | `_step_rows` 直接调用 `model._run`（数值语义与 `_step` 完全相同）⇒ 两模式插桩足迹逐项相同 |
| D5 | GPU 运行器后处理 `aud.uninstall()` 写成属性访问 | **运行器 bug** | 首次批运行（6 次尝试，全部被环境门拒绝）后 `AttributeError`，导致**连记录都没落盘** | 改为 `aud["uninstall"]()`；整段后处理包 try/except，**异常也照常落盘**并标出 |

> `serial_oracle_stage0_20261003T134138Z.json`（35/2）与
> `serial_oracle_audit_selftest_20261003T135324Z.json`（18/1）作为首次失败记录**永久保留**，
> 未被覆盖；两处修正**只改断言/记录方式，未改任何被测逻辑**。

## 4. 与冻结设计的偏离（**实现期发现，均在任何 GPU 运行之前**）

冻结设计文档 §2.4 写的是"**允许恰好一次** prefill 调用（`s == prompt_len`）"。实现时核对真实
harness 后发现该假设**不成立**：`g5_8k_smoke_cudnn` 的 prefill 是**分块**的
（`ctx=8192, chunk=2048` ⇒ 多次 `s=2048`）。因此审计判据改为**按阶段**：

* `phase == "prefill"`：允许任意 `s`（分块 prefill）；
* **其他任何阶段**（`verify` / `repair` / `decode` / `propose` / 阶段未知）：必须 `s == 1`，否则**抛错**。

判据的**意图未变且更强**（"generation 期间禁止任何 `s>1` 的 target 前向"），且**不放宽**：
阶段未知（`None`）同样按"非 prefill"处理 ⇒ 不能被静默放过。本偏离已在 `serial_oracle.py`
模块文档与 `serial_oracle_audit_selftest_cpu.py` 用例 a01 中固化。
**冻结文档本身未被改写**（它仍是预登记判据的依据）。

## 5. 与真实 harness 的接入核对（GPU 运行前逐项确认）

真实运行期 `_run` 的包装链（**由外到内**）：

```
smoke_hooks._run_hook        (DecisionHooks.install(), harness 行 475；先决策点有限性检查)
  └─ target_rope_scope.scoped_run   (实例识别 + CPU-FP32 inv_freq 作用域；harness 行 350-355)
       └─ serial_oracle.hook        (★ 本轮的调用几何审计，装在 H.main() **之前** ⇒ 最内层)
            └─ model_self.QwenLM._run（原始实现）
```
* 审计装在**最内层**仍能看到**每一次**调用（链上每一环都会调用下一环）；
* `_run` hook 必须是**普通函数**（`orig(model, x, cache_list, need_logits)` 位置调用），
  与 `smoke_hooks` 同一手法（否则 `instance._run(...)` 不会重新绑定）；
* `spec_dec` 侧的包装链（**由外到内**）：harness 的 `_verify_struct_spy`
  → `DecisionHooks._verify` → 真实 `_verify`（**mode 分派**）→ `_verify_serial`；
  `spec_dec._step` 被 `DecisionHooks` 包装（`repair` 阶段），但 `_step_rows` 已改为不经它；
* 因此 serial 路径的 harness 记录（轮次、覆盖、有限性）与 block 路径**同口径**。

### 5.1 记账恒等式（**非空泛**证据）

审计器实测的 target generation 调用数必须等于解码器记账推导值，否则"几何结论无效"：

| 模式 | 恒等式 | 理由 |
| --- | --- | --- |
| `ar` | `target_calls == decode_steps` | 每步一次 `s=1` |
| `serial` | `target_calls == proposed + rollbacks` | 每轮 `kp` 次逐 token + 每次回滚一次 `_step` |
| `prod` | `target_calls == target_block_calls + rollbacks` | 每轮一次 block + 每次回滚一次 `_step` |

## 6. 环境阻断记录（Stage 1 **未开始**，与代码/数值无关）

第一次批运行（22:00:36–22:01:36，6 次尝试）与单次试探（22:06:52）**全部被 harness 的环境门**
在**加载模型之前**拒绝：

```
[smoke] !! 环境不合格，未加载：环境不合格：commit_ok=False pstate=True
commit 可用 15.2 GiB < 安全线 16 GiB（上限 55.6 GiB，已用 40.4 GiB）   # 批运行期间
commit 可用 14.9 GiB < 安全线 16 GiB（上限 55.6 GiB，已用 40.7 GiB）   # 单次试探
GPU 上其它 C+G 进程=22（Edge/WebView、Teams、Outlook、豆包、网易云音乐、Codex node 等）
pstate: qualified=True（sm_median 2415–2460 MHz ≥ 2000；mem 8000 ≥ 7500）
```

* **未加载任何模型、未消耗 GPU 计算**（每次仅 ~10 秒的环境自检 + ~6 秒 P-state 负载采样）；
* 按预登记口径这属于 **`CHECK_NEEDS_REVIEW`（环境不合格）**，**不是** `SERIAL_ORACLE_FAIL`；
* 汇总器实测：`div8209_so_aggregate_stage1_20261003T140725Z.json` ⇒ `CHECK_NEEDS_REVIEW`，
  `needs_review` 明确列出"环境不合格（未加载模型）"与"缺 AR/serial 运行"两组原因；
* **未**使用 `SMOKE_ALLOW_UNQUALIFIED` 绕过环境门（那属于放宽安全门，禁止）。
* 保留的日志：`output/g5/attn/div8209_so_F13_{ar,prod,serial}_rep{1,2}.log`、
  `div8209_so_batch_20261003T140036Z.log`、`..._20261003T140106Z.log`、
  `div8209_so_F13_ar_rep1.log`（试探）与对应的 `smoke8k_eager_*.json`（环境不合格记录）。

## 7. 未决问题与下一步（**未完成，不得当作已完成**）

1. **Stage 1（F13）**：3 模式 × 2 独立进程 = 6 runs。需环境门通过（可用 commit ≥ 16 GiB）。
   预期校验点：`prod` 复现历史首分歧 `index=47` 且 `gen_hash == e90dad2b3fcf1c65`，
   `ar` 复现 `gen_hash == 6e7522bf2521ad7b`（历史锚点，用于证明新运行器未改动 AR/production 路径）。
2. **Stage 2**：F01/F02/F08/F10 × {AR×2, serial×2} = 16 runs（不复跑 production/row-aligned，引用上一轮结果）。
3. **Stage 1/2 报告 + §9.63/§4.78 文档更新 + 5-family 判定**。
4. 未决：serial 模式的耗时（每轮 `kp` 次 `s=1` 前向，预期比 block 慢但**不作为**准入判据——
   本阶段只回答"控制逻辑能否生成与 AR 相同的 token IDs"，不含速度结论）。

**当前不得声称**：serial oracle 与 AR 在真实模型上 token 一致（**未运行**）；也不得启动 B6/pilot/factorial。
