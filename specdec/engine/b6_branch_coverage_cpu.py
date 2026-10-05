"""b6_branch_coverage_cpu.py — **B6 分支覆盖 + 严格验收**的只读 CPU 检查器（2026-10-05，C-3；rev.2 修漏洞）。

## 为什么需要
`design_freeze_g4_specdec.md:83` 要求 B6 覆盖 10 类 prompt 形态与若干分支；
**自然语言 prompt 不保证触发某个分支** ⇒ 覆盖必须由**实际 trace**证明。
本模块是字面 prompt 用例的**下游校验**：
`stage_check` / `stage2b_verdict` 以 manifest family（`F\\d{2}`）为单位，**不适用**于字面用例。

## rev.2 修的四个**验收器漏洞**（都不是模型/算法问题）
1. **完整序列一致性**：旧版只看重叠区间的 `first_divergence` ⇒ `AR=[1,2,3,4]` 与 `serial=[1,2]`
   会被判"无分歧"⇒ 可 PASS。**现在**：严格门 = **完整 token 列表相等**（长度不同也算不一致），
   并报告共同前缀、两侧长度、长度差异位置。production 同样如实记录，但**不**自动判控制逻辑失败。
2. **硬检查必须验值**：旧版只要求字段**存在** ⇒ `self_checks_all_ok=False`、`cache_structure_ok=False`、
   `prefill_logits_finite=False`、`round_invariants.all_ok=False` 仍可 PASS。**现在**：逐项验值
   （调用计数恒等式 / 有限性 / cache 结构 / 控制不变量 / 运行完成状态 / 杀开关复位 / 停止合法性），
   并核对 `gen_len`↔实际长度↔`max_new`↔`stop_reason`↔EOS 的自洽性。
   缺失 ⇒ **证据不足（不得 PASS）**；显式 False ⇒ **CASE_FAIL**；类型非法 ⇒ **非 PASS**。
3. **AR 的融合 prefill 验收**：旧版对 AR 强制 `expect_prefill_fused=False` ⇒ 误拒合理融合。
   **现在**：AR 与 Spec 走**同一 target prefill 策略**判定；按实现的分派规则核对调用证据
   （`strategy`/`requested_backend`/`fused_by_phase`/`s_max_by_phase`/`fallbacks`/`eager_reason_kinds`）；
   现有统计**无法区分** target/draft 注意力调用时**明确标注证据边界**，不臆造；**禁止**非 prefill 阶段融合。
   `expect=None` **不等于**"不检查策略"：它要求"要么真的融合、要么有**记录的**回退理由"，否则证据不足。
4. **跨模式 / 跨重复的输入与配置匹配**：同 (case, rep) 的 AR/prod/serial 必须同输入 hash、同实际长度、
   同 target dtype/KV、同 backend 与 prefill 策略、同 chunk/k/max_new/EOS 与已冻结数值配置；
   同模式跨 rep 亦然。**AR 不加载 draft 是预期差异**（`kv_format.draft="not_loaded"`），不误判。

## 产物选择（第二节）
`find_runs()` **不**静默覆盖同一 `(mode, rep)` 的多份产物、**不**偷偷取"最后一份"：
重复产物、非法 `rep`、身份字段（tag/case/mode）与请求不一致 ⇒ **显式报告并判非 PASS**。

退出码：`0` 全部 `CASE_PASS` · `1` 存在 INCOMPLETE/FAIL · `3` 参数非法（不写产物）。
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
MODES = ("ar", "prod", "serial")
REPS = (1, 2)
# 非 prefill 阶段**禁止融合**
NON_PREFILL_PHASES = ("propose", "verify", "repair", "decode", "generate")
LEGAL_STOPS = ("max_new_reached", "eos")
# 冻结候选配置中与用例无关的部分（逐 run、跨模式/跨 rep 必须一致）
FROZEN_CONFIG = dict(k=4, chunk=2048, kv="base_bf16", model_dtype="torch.bfloat16",
                     target_kv_payload="base_bf16")
CONFIG_COMPARE_KEYS = ("k", "chunk", "kv", "max_new", "arm", "target_rope", "bf16_red_red",
                       "struct_check")
# **必需字段**（存在性；缺失 ⇒ 证据不足）
REQUIRED_PATHS = (
    "case", "mode", "rep", "tag", "run_id", "stage",
    "smoke.input.source_kind", "smoke.input.ctx", "smoke.input.ids_hash",
    "smoke.gen", "smoke.stop_reason", "smoke.specdec_stats",
    "smoke.cache_structure_ok", "smoke.cache_structure_n_checks",
    "smoke.round_invariants.all_ok", "smoke.prefill_logits_finite",
    "smoke.checks_passed", "smoke.gen_ids_in_vocab",
    "smoke.attention_stats.strategy", "smoke.attention_stats.requested_backend",
    "smoke.attention_stats.fused_by_phase",
    "accounting.matches", "self_checks_all_ok", "verify_mode_after_run",
    # 冻结数值配置的证据：诊断精度开关必须**记录**（缺 ⇒ 证据不足，不得 PASS）
    "diag_precision_before_run", "diag_precision_after_run", "diag_precision_stable",
)


# ------------------------------------------------------------------ 基础读取
def _dig(rec: dict, path: str):
    cur = rec
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def load_artifact(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def find_runs(tag: str, case: str, outdir: Path = OUTDIR) -> tuple:
    """收集某 case 的产物：返回 `(runs, problems)`。

    ⚠️ **不静默**：同一 `(mode, rep)` 的多份产物**全部保留**并记入 `problems`；
    非法 `rep`、身份字段与请求不一致（tag/case/mode）同样记入 `problems`。
    `runs[(mode, rep)]` 只在**唯一**时给出该记录；重复时置 `None` 并附 `files` 列表。
    """
    runs: dict = {}
    files: dict = {}
    problems: list = []
    for p in sorted(glob.glob(str(Path(outdir) / f"{tag}_{case}_*_rep*_*.json"))):
        name = Path(p).name
        if any(x in name for x in ("aggregate", "verdict", "margins", "coverage", "selftest",
                                   "b6_branch_coverage")):
            continue
        try:
            rec = load_artifact(Path(p))
        except Exception as e:                                        # noqa: BLE001
            problems.append(f"{name}: 无法解析（{type(e).__name__}）")
            continue
        if not isinstance(rec, dict) or rec.get("tool") != "serial_oracle_gpu":
            continue
        # 身份字段核对：必须与**请求的** tag/case 一致
        if rec.get("tag") != tag:
            problems.append(f"{name}: 记录内 tag={rec.get('tag')!r} ≠ 请求 {tag!r}")
            continue
        if rec.get("case") != case:
            problems.append(f"{name}: 记录内 case={rec.get('case')!r} ≠ 请求 {case!r}")
            continue
        mode, rep = rec.get("mode"), rec.get("rep")
        if mode not in MODES:
            problems.append(f"{name}: 非法 mode={mode!r}")
            continue
        if not isinstance(rep, int) or isinstance(rep, bool):
            problems.append(f"{name}: 非法 rep={rep!r}（必须为整数）")
            continue
        files.setdefault((mode, rep), []).append(name)
        runs[(mode, rep)] = rec
    for key, fl in files.items():
        if len(fl) > 1:
            problems.append(f"{key[0]}/rep{key[1]}: **重复产物 {len(fl)} 份** {fl} ⇒ 不静默择一")
            runs[key] = None
    return runs, problems


# ------------------------------------------------------------------ 字段/配置
def missing_required_fields(rec: dict) -> list:
    """缺字段：**空容器不算缺失**（`fused_by_phase={}` 是合法取值："没有任何阶段融合"）。
    只有 `None` / 空串算没有记录；`smoke.gen` 空列表例外（无生成 = 无证据）。"""
    miss = []
    for path in REQUIRED_PATHS:
        v = _dig(rec, path)
        if v is None or (isinstance(v, str) and not v.strip()):
            miss.append(path)
    gen = _dig(rec, "smoke.gen")
    if isinstance(gen, list) and not gen:
        miss.append("smoke.gen（为空 ⇒ 无生成证据）")
    return miss


def type_problems(rec: dict) -> list:
    """类型非法 ⇒ 非 PASS（既不是 FAIL 也不是 PASS）。"""
    bad = []
    checks = [("smoke.gen", list), ("smoke.input.ctx", int), ("smoke.input.ids_hash", str),
              ("accounting.matches", bool), ("self_checks_all_ok", bool),
              ("smoke.cache_structure_ok", bool), ("smoke.prefill_logits_finite", bool),
              ("smoke.round_invariants.all_ok", bool), ("smoke.checks_passed", bool),
              ("smoke.gen_ids_in_vocab", bool), ("smoke.cache_structure_n_checks", int),
              ("smoke.attention_stats.fused_by_phase", dict),
              ("smoke.attention_stats.strategy", str),
              ("smoke.attention_stats.requested_backend", str)]
    for path, typ in checks:
        v = _dig(rec, path)
        if v is None:
            continue
        if typ is int and isinstance(v, bool):        # bool 是 int 的子类 ⇒ 显式排除
            bad.append(f"{path}: 类型非法（bool，期望 int）")
        elif not isinstance(v, typ):
            bad.append(f"{path}: 类型非法（{type(v).__name__}，期望 {typ.__name__}）")
    return bad


def config_violations(rec: dict, *, expect_max_new: int | None = None) -> list:
    bad = []
    sc = rec.get("smoke_config") or {}
    inp = (_dig(rec, "smoke.input") or {})
    dt = inp.get("dtypes") or {}
    if sc.get("k") != FROZEN_CONFIG["k"]:
        bad.append(f"k={sc.get('k')}≠{FROZEN_CONFIG['k']}")
    if sc.get("chunk") != FROZEN_CONFIG["chunk"]:
        bad.append(f"chunk={sc.get('chunk')}≠{FROZEN_CONFIG['chunk']}")
    if sc.get("kv") != FROZEN_CONFIG["kv"]:
        bad.append(f"kv={sc.get('kv')}≠{FROZEN_CONFIG['kv']}")
    if sc.get("struct_check") is not True:
        bad.append("struct_check≠True")
    if dt.get("model") != FROZEN_CONFIG["model_dtype"]:
        bad.append(f"model_dtype={dt.get('model')}≠{FROZEN_CONFIG['model_dtype']}")
    if dt.get("target_kv_payload") != FROZEN_CONFIG["target_kv_payload"]:
        bad.append(f"target_kv_payload={dt.get('target_kv_payload')}")
    if expect_max_new is not None and sc.get("max_new") != expect_max_new:
        bad.append(f"max_new={sc.get('max_new')}≠用例声明 {expect_max_new}")
    dp = rec.get("diag_precision_after_run")
    if isinstance(dp, dict):
        for key in ("qk_fp32", "prob_fp32"):
            if dp.get(key) is not False:
                bad.append(f"diag.{key}={dp.get(key)!r}≠False")
    # 缺失一律归"证据不足"；只有**显式为 False** 才判配置漂移
    if rec.get("diag_precision_stable") is False:
        bad.append("diag_precision 运行前后不一致（配置漂移）")
    return bad


# ------------------------------------------------------------------ 硬检查（验值）
def hard_checks(rec: dict) -> list:
    """返回 `[(name, state, detail)]`，`state ∈ {ok, fail, missing, illegal}`。

    **逐项验值**，不以"字段存在"代替"检查通过"。
    """
    mode = rec.get("mode")
    rows = []

    def add(name, value, *, expect=True, label=None):
        """逐项验值。

        ⚠️ 期望值为**字符串**时（如 `stage=="complete"`、`verify_mode_after_run=="block"`）
        必须**直接比较字符串**；首版误用 `bool(value) is bool(expect)` ⇒ `"yes"` 与 `"block"`
        会被判成"相等" ⇒ 类型非法/值不符仍 PASS（自测 G14b 抓到）。
        """
        if value is None:
            rows.append((name, "missing", "字段缺失 ⇒ 证据不足"))
        elif isinstance(expect, bool):
            if not isinstance(value, bool):
                rows.append((name, "illegal", f"类型非法：{type(value).__name__}（期望 bool）"))
            elif value is expect:
                rows.append((name, "ok", label or f"{name}={value}"))
            else:
                rows.append((name, "fail", label or f"{name}={value}（期望 {expect}）"))
        else:                                   # 期望字符串
            if not isinstance(value, str):
                rows.append((name, "illegal", f"类型非法：{type(value).__name__}（期望 str）"))
            elif value == expect:
                rows.append((name, "ok", label or f"{name}={value!r}"))
            else:
                rows.append((name, "fail", label or f"{name}={value!r}（期望 {expect!r}）"))

    add("self_checks_all_ok", rec.get("self_checks_all_ok"))
    add("accounting_matches（调用计数恒等式）", _dig(rec, "accounting.matches"))
    add("stage_complete", rec.get("stage"), expect="complete",
        label=f"stage={rec.get('stage')!r}（期望 'complete'）")
    add("kill_switch_restored（verify_mode_after_run==block）",
        rec.get("verify_mode_after_run"), expect="block",
        label=f"verify_mode_after_run={rec.get('verify_mode_after_run')!r}")
    if mode in ("prod", "serial"):
        add("harness_spec_mode", (rec.get("smoke_config") or {}).get("mode"), expect="spec",
            label=f"smoke_config.mode={(rec.get('smoke_config') or {}).get('mode')!r}")
    if mode == "serial":
        add("serial_dispatched_at_end", rec.get("verify_mode_at_run_end"), expect="serial",
            label=f"verify_mode_at_run_end={rec.get('verify_mode_at_run_end')!r}")
    add("cache_structure_ok", _dig(rec, "smoke.cache_structure_ok"))
    n_ck = _dig(rec, "smoke.cache_structure_n_checks")
    if n_ck is None:
        rows.append(("cache_structure_n_checks>0", "missing", "字段缺失"))
    elif not isinstance(n_ck, int) or isinstance(n_ck, bool):
        rows.append(("cache_structure_n_checks>0", "illegal", f"类型非法：{type(n_ck).__name__}"))
    elif n_ck <= 0:
        rows.append(("cache_structure_n_checks>0", "fail", f"n_checks={n_ck}（必须 >0）"))
    else:
        rows.append(("cache_structure_n_checks>0", "ok", f"n_checks={n_ck}"))
    add("prefill_logits_finite", _dig(rec, "smoke.prefill_logits_finite"))
    add("round_invariants_all_ok", _dig(rec, "smoke.round_invariants.all_ok"))
    add("harness_checks_passed", _dig(rec, "smoke.checks_passed"))
    add("gen_ids_in_vocab", _dig(rec, "smoke.gen_ids_in_vocab"))
    # 有限性覆盖（harness 的逐阶段/逐模型检查）
    fin = _dig(rec, "smoke.checks.finiteness_coverage")
    if isinstance(fin, dict):
        n = fin.get("n_checked")
        by = fin.get("by_phase_model") or {}
        bad = [k for k, v in by.items() if isinstance(v, dict) and v.get("ok") is not True]
        if not n:
            rows.append(("finiteness_n_checked>0", "fail", f"n_checked={n}"))
        elif bad:
            rows.append(("finiteness_by_phase_model_ok", "fail", f"未通过：{bad}"))
        else:
            rows.append(("finiteness_n_checked>0 / by_phase_model ok", "ok", f"n_checked={n}"))
    else:
        rows.append(("finiteness_coverage", "missing", "smoke.checks.finiteness_coverage 缺失"))
    # harness 的必需检查覆盖
    unc = _dig(rec, "smoke.checks.uncovered")
    if unc is None:
        rows.append(("harness_checks_uncovered_empty", "missing", "字段缺失"))
    elif unc:
        rows.append(("harness_checks_uncovered_empty", "fail", f"未覆盖必需检查：{unc}"))
    else:
        rows.append(("harness_checks_uncovered_empty", "ok", "无未覆盖必需检查"))
    # 决策覆盖
    dc = _dig(rec, "smoke.decision_coverage")
    if dc is None:
        rows.append(("decision_coverage", "missing", "字段缺失（旧产物可能没有）"))
    elif dc is True:
        rows.append(("decision_coverage", "ok", "ok"))
    else:
        rows.append(("decision_coverage", "fail", f"decision_coverage={dc}"))
    return rows


def consistency_checks(rec: dict) -> list:
    """`gen_len` ↔ 实际列表长度 ↔ `max_new` ↔ `stop_reason` ↔ EOS 的自洽性（沿用冻结停止规则）。"""
    rows = []
    gen = list(_dig(rec, "smoke.gen") or [])
    gl = _dig(rec, "smoke.gen_len")
    es = _dig(rec, "smoke.end_state") or {}
    stop = _dig(rec, "smoke.stop_reason")
    mn_cfg = (rec.get("smoke_config") or {}).get("max_new")
    mn_end = es.get("max_new")
    eos_id = es.get("eos_id")
    if gl is None:
        rows.append(("gen_len_present", "missing", "smoke.gen_len 缺失"))
    elif gl != len(gen):
        rows.append(("gen_len==len(gen)", "fail", f"gen_len={gl} len(gen)={len(gen)}"))
    else:
        rows.append(("gen_len==len(gen)", "ok", f"{gl}"))
    if mn_end is not None and mn_cfg is not None and mn_end != mn_cfg:
        rows.append(("end_state.max_new==config.max_new", "fail", f"{mn_end}≠{mn_cfg}"))
    elif mn_end is not None and mn_cfg is not None:
        rows.append(("end_state.max_new==config.max_new", "ok", f"{mn_end}"))
    if stop is None:
        rows.append(("stop_reason_present", "missing", "stop_reason 缺失"))
    elif stop not in LEGAL_STOPS:
        rows.append(("stop_reason_legal", "fail",
                     f"stop_reason={stop!r} 不在冻结规则 {LEGAL_STOPS}（other_short 等一律判失败）"))
    else:
        rows.append(("stop_reason_legal", "ok", stop))
        if stop == "max_new_reached":
            if mn_cfg is not None and len(gen) != mn_cfg:
                rows.append(("stop==max_new_reached⇒len==max_new", "fail",
                             f"len={len(gen)} max_new={mn_cfg}"))
            else:
                rows.append(("stop==max_new_reached⇒len==max_new", "ok", f"len={len(gen)}"))
        if stop == "eos":
            if es.get("eos_seen") is not True:
                rows.append(("stop==eos⇒eos_seen", "fail", f"eos_seen={es.get('eos_seen')!r}"))
            elif eos_id is None:
                rows.append(("stop==eos⇒last_token==eos_id", "missing",
                             "产物未记录 eos_id ⇒ 无法核对（证据边界）"))
            elif not gen or int(gen[-1]) != int(eos_id):
                rows.append(("stop==eos⇒last_token==eos_id", "fail",
                             f"last={gen[-1] if gen else None} eos_id={eos_id}"))
            else:
                rows.append(("stop==eos⇒last_token==eos_id", "ok", f"{eos_id}"))
    return rows


# ------------------------------------------------------------------ 后端/融合策略
def backend_report(rec: dict, *, expect_backend: str | None, expect_prefill_fused) -> dict:
    """**同一策略**适用于 AR / production / serial（不得对 AR 强制 False）。

    `expect_prefill_fused`：
      * `True`  ⇒ 必须真的触发 prefill 融合；
      * `False` ⇒ 不得融合，且需有**记录的**回退理由（否则证据不足）；
      * `None`  ⇒ **按实现的分派规则**：要么真的融合、要么有**记录的**回退理由；
                  若现有统计两者都看不出 ⇒ **证据不足**（**不得**解释成"不检查策略"）。
    `arm == "eager"` 时融合恒为 0 是**合法配置**（该臂本就不请求融合）。
    """
    att = (_dig(rec, "smoke.attention_stats") or {})
    arm = (rec.get("smoke_config") or {}).get("arm")
    fused = att.get("fused_by_phase") or {}
    s_max = att.get("s_max_by_phase") or {}
    fallbacks = att.get("fallbacks") or []
    reasons = att.get("eager_reason_kinds") or {}
    problems, notes, insufficient = [], [], []
    if expect_backend and expect_backend.upper() not in str(att.get("requested_backend") or "").upper():
        problems.append(f"requested_backend={att.get('requested_backend')!r} 未包含 {expect_backend!r}")
    # ① 禁止非 prefill 融合（对所有臂、所有模式）
    bad_fused = {ph: n for ph, n in fused.items() if ph in NON_PREFILL_PHASES and n}
    if bad_fused:
        problems.append(f"非 prefill 阶段出现融合调用：{bad_fused}（**禁止**）")
    prefill_fused = int(fused.get("prefill") or 0)
    has_fallback_evidence = bool(fallbacks) or bool(reasons)
    # ② 按 arm 与实际分派结果判定
    if arm == "eager":
        if prefill_fused:
            problems.append(f"arm=eager 却出现融合 {fused}（配置漂移）")
        elif str(att.get("strategy")) != "eager":
            problems.append(f"arm=eager 但 strategy={att.get('strategy')!r}")
        else:
            notes.append("arm=eager：融合为 0 属**合法配置**（该臂不请求融合）")
        # ⚠️ 用例**声明**应验证融合，但本次跑的是 eager 臂 ⇒ 该期望**未被检验**（不得静默当通过）
        if expect_prefill_fused is True:
            insufficient.append("用例声明应触发/验证 prefill 融合，但本次 `arm=eager` ⇒ "
                                "该期望**未被检验**（需在候选后端臂上重跑该用例）")
    else:
        if expect_prefill_fused is True and prefill_fused == 0:
            problems.append(f"用例声明**应触发** prefill 融合，但 fused_by_phase={fused} "
                            f"（s_max_by_phase={s_max}）")
        elif expect_prefill_fused is False and prefill_fused > 0:
            problems.append(f"用例声明**不应**融合，但出现 prefill 融合 {fused}")
        elif expect_prefill_fused is None:
            if prefill_fused > 0:
                notes.append("按分派规则：实际触发了 prefill 融合")
            elif has_fallback_evidence:
                notes.append(f"按分派规则：未融合但**有记录的回退理由** "
                             f"（fallbacks={fallbacks[:2]} reasons={list(reasons)[:3]}）")
            else:
                # ⚠️ **证据不足**（≠ 明确失败）：分派规则无法判定 ⇒ CASE_INCOMPLETE，不得 PASS。
                insufficient.append("既无 prefill 融合、也无记录的回退理由 ⇒ "
                                    "无法判定分派规则是否合理（证据边界，不得当作通过）")
    boundary = ("现有 `attention_stats` 的 `by_phase`/`fused_by_phase` **不区分 target/draft** "
                "⇒ target 与 draft 各自的注意力调用证据在**本产物内不可分**（证据边界，不臆造）；"
                "`smoke.checks.finiteness_coverage.by_phase_model` 仅覆盖有限性检查。")
    return dict(arm=arm, requested_backend=att.get("requested_backend"), strategy=att.get("strategy"),
                calls=att.get("calls"), fused_by_phase=fused, eager_by_phase=att.get("eager_by_phase"),
                s_max_by_phase=s_max, fused_s_max_by_phase=att.get("fused_s_max_by_phase"),
                fallbacks=fallbacks[:3], eager_reason_kinds=reasons,
                expect_prefill_fused=expect_prefill_fused, problems=problems, notes=notes,
                insufficient=insufficient, evidence_boundary=boundary)


# ------------------------------------------------------------------ 相等性 / 一致性
def strict_equality(gen_a: list, gen_b: list) -> dict:
    """**完整 token 列表**相等判定（长度不同 ⇒ 不一致），并报告共同前缀与长度差异位置。"""
    cp = 0
    for x, y in zip(gen_a, gen_b):
        if x != y:
            break
        cp += 1
    equal = bool(list(gen_a) == list(gen_b))
    first_diff = None if equal else (cp if cp < min(len(gen_a), len(gen_b)) else None)
    return dict(equal=equal, common_prefix=cp, len_a=len(gen_a), len_b=len(gen_b),
                first_token_diff=first_diff,
                length_difference_position=(None if len(gen_a) == len(gen_b)
                                            else min(len(gen_a), len(gen_b))),
                kind=("EQUAL" if equal else
                      ("PREFIX_LENGTH_MISMATCH" if first_diff is None else "TOKEN_DIFF")))


_CMP_PATHS = ("smoke.input.ids_hash", "smoke.input.ctx", "smoke.input.dtypes.model",
              "smoke.input.dtypes.target_kv_payload", "smoke.kv_format",
              "smoke_config.k", "smoke_config.chunk", "smoke_config.max_new",
              "smoke_config.kv", "smoke_config.arm", "smoke_config.target_rope",
              "smoke_config.bf16_red_red", "smoke_config.struct_check",
              "smoke.input.eos_config.eos_id")


def _cmp_view(rec: dict) -> dict:
    """可比性视图（AR **不加载 draft** 属预期差异 ⇒ 只比 target 侧）。"""
    v = {p: _dig(rec, p) for p in _CMP_PATHS}
    kvf = v.get("smoke.kv_format")
    v["smoke.kv_format"] = (dict(kvf, draft="<draft-exempt>") if isinstance(kvf, dict) else kvf)
    v["draft_loaded"] = bool(isinstance(kvf, dict) and kvf.get("draft") not in (None, "not_loaded"))
    return v


def comparability_problems(runs: dict, *, modes=MODES, reps=REPS) -> list:
    """跨模式（同 rep）与跨重复（同模式）的输入/配置匹配核对。"""
    bad = []
    for rep in reps:
        base = runs.get(("ar", rep))
        if base is None:
            continue
        vb = _cmp_view(base)
        for mode in modes:
            if mode == "ar":
                continue
            other = runs.get((mode, rep))
            if other is None:
                continue
            vo = _cmp_view(other)
            for k in vb:
                if k == "draft_loaded":
                    continue
                if vb[k] != vo[k]:
                    bad.append(f"rep{rep}: AR 与 {mode} 的 {k} 不一致（{vb[k]!r} vs {vo[k]!r}）")
        # AR 的 draft 必须是 not_loaded（否则说明 AR 加载了 draft = 基线不纯）
        if vb.get("draft_loaded"):
            bad.append(f"rep{rep}: AR 侧 draft 已加载（kv_format.draft≠not_loaded）⇒ 基线不纯")
    for mode in modes:
        r1, r2 = runs.get((mode, 1)), runs.get((mode, 2))
        if r1 is None or r2 is None:
            continue
        v1, v2 = _cmp_view(r1), _cmp_view(r2)
        for k in v1:
            if v1[k] != v2[k]:
                bad.append(f"{mode}: rep1 与 rep2 的 {k} 不一致（{v1[k]!r} vs {v2[k]!r}）")
    return bad


# ------------------------------------------------------------------ 分支覆盖
def branch_observations(rec: dict) -> dict:
    rounds = list((_dig(rec, "smoke.rounds") or []))
    gen = list((_dig(rec, "smoke.gen") or []))
    k = int((rec.get("smoke_config") or {}).get("k") or 4)
    stop = _dig(rec, "smoke.stop_reason")
    obs = dict(all_accept=0, first_reject=0, mid_reject=0, rollback=0, repair=0,
               trim_any=0, trim_last_round=0, eos_stop=0, len_not_multiple_of_k=0,
               rounds=len(rounds), gen_len=len(gen))
    for i, rr in enumerate(rounds):
        m, kp = int(rr.get("m") or 0), int(rr.get("Kp") or 0)
        if rr.get("all_accept") or (kp and m == kp):
            obs["all_accept"] += 1
        if m == 0:
            obs["first_reject"] += 1
        elif 0 < m < kp:
            obs["mid_reject"] += 1
        if rr.get("rollback"):
            obs["rollback"] += 1
        if rr.get("repair"):
            obs["repair"] += 1
        if kp and kp < k:
            obs["trim_any"] += 1
            if i == len(rounds) - 1:
                obs["trim_last_round"] += 1
    if stop == "eos":
        obs["eos_stop"] = 1
    if gen and len(gen) % k != 0:
        obs["len_not_multiple_of_k"] = 1
    return obs


# ------------------------------------------------------------------ 判定
_STATE_RANK = dict(ok=0, missing=1, illegal=2, fail=3)


def evaluate_case(case_id: str, declared: dict, runs: dict, *, expect_backend: str | None = None,
                  expect_config: dict | None = None, run_problems: list | None = None) -> dict:
    expect_config = expect_config or {}
    expect_max_new = expect_config.get("max_new")
    required_modes = list(expect_config.get("modes") or MODES)
    res: dict = dict(case_id=case_id, declared=dict(declared), per_mode={}, branches={},
                     reasons=[], hard_checks={}, consistency={}, verdict=None,
                     duplicate_or_identity_problems=list(run_problems or []))
    field_problems, config_problems, backend_problems, type_problems_ = [], [], [], []
    backend_insufficient: list = []
    hard_fail, hard_missing, hard_illegal = [], [], []
    cons_fail, cons_missing, cons_illegal = [], [], []
    for mode in required_modes:
        for rep in REPS:
            rec = runs.get((mode, rep))
            if rec is None:
                if (mode, rep) in runs:
                    continue          # 重复产物：已由 duplicate problems 报出
                field_problems.append(f"{mode}/rep{rep}: 产物缺失")
                continue
            miss = missing_required_fields(rec)
            if miss:
                field_problems.append(f"{mode}/rep{rep}: 缺字段 {miss}")
            tp = type_problems(rec)
            if tp:
                type_problems_.append(f"{mode}/rep{rep}: {tp}")
            cfgbad = config_violations(rec, expect_max_new=expect_max_new)
            if cfgbad:
                config_problems.append(f"{mode}/rep{rep}: {cfgbad}")
            # 同一策略适用于所有模式（**不**对 AR 强制 False）
            br = backend_report(rec, expect_backend=expect_backend,
                                expect_prefill_fused=declared.get("expect_prefill_fused"))
            if br["problems"]:
                backend_problems.append(f"{mode}/rep{rep}: {br['problems']}")
            if br.get("insufficient"):
                backend_insufficient.append(f"{mode}/rep{rep}: {br['insufficient']}")
            res["per_mode"].setdefault(mode, {})[f"rep{rep}"] = br
            for name, state, detail in hard_checks(rec):
                res["hard_checks"][f"{mode}/rep{rep}/{name}"] = dict(state=state, detail=detail)
                if state == "fail":
                    hard_fail.append(f"{mode}/rep{rep}: {name} → {detail}")
                elif state == "missing":
                    hard_missing.append(f"{mode}/rep{rep}: {name} → {detail}")
                elif state == "illegal":
                    hard_illegal.append(f"{mode}/rep{rep}: {name} → {detail}")
            for name, state, detail in consistency_checks(rec):
                res["consistency"][f"{mode}/rep{rep}/{name}"] = dict(state=state, detail=detail)
                if state == "fail":
                    cons_fail.append(f"{mode}/rep{rep}: {name} → {detail}")
                elif state == "missing":
                    cons_missing.append(f"{mode}/rep{rep}: {name} → {detail}")
                elif state == "illegal":
                    cons_illegal.append(f"{mode}/rep{rep}: {name} → {detail}")
    # 分支覆盖
    obs_total = {}
    for mode in ("prod", "serial"):
        for rep in REPS:
            rec = runs.get((mode, rep))
            if rec is None:
                continue
            for kk, vv in branch_observations(rec).items():
                if isinstance(vv, int) and kk not in ("rounds", "gen_len"):
                    obs_total[kk] = obs_total.get(kk, 0) + vv
    for branch in (declared.get("branches") or []):
        n = int(obs_total.get(branch, 0))
        res["branches"][branch] = dict(observed=n, verdict="COVERED" if n > 0 else "UNCOVERED")
    # 严格门（**完整序列**）+ production 只记录 + 跨 rep 一致
    gate = dict(serial_vs_ar={}, production_vs_ar={}, repeats={})
    for rep in REPS:
        ar, se, pr = runs.get(("ar", rep)), runs.get(("serial", rep)), runs.get(("prod", rep))
        if ar and se:
            d = strict_equality(list(_dig(se, "smoke.gen") or []), list(_dig(ar, "smoke.gen") or []))
            gate["serial_vs_ar"][f"rep{rep}"] = d
            if not d["equal"]:
                res["reasons"].append(
                    f"serial/rep{rep} 与 AR **完整序列不相等**（{d['kind']}：共同前缀 {d['common_prefix']}、"
                    f"len {d['len_a']} vs {d['len_b']}、长度差异位置 {d['length_difference_position']}）")
        if ar and pr:
            gate["production_vs_ar"][f"rep{rep}"] = strict_equality(
                list(_dig(pr, "smoke.gen") or []), list(_dig(ar, "smoke.gen") or []))
    for mode in required_modes:
        g1, g2 = runs.get((mode, 1)), runs.get((mode, 2))
        if g1 is None or g2 is None:
            gate["repeats"][mode] = None
            continue
        d = strict_equality(list(_dig(g1, "smoke.gen") or []), list(_dig(g2, "smoke.gen") or []))
        gate["repeats"][mode] = d["equal"]
        if not d["equal"]:
            res["reasons"].append(f"{mode} 两次独立重复**完整序列不相等**（{d['kind']}）")
    cmp_bad = comparability_problems(runs, modes=required_modes, reps=REPS)
    gate["comparability_problems"] = cmp_bad
    res["gate"] = gate
    # ---- 汇总理由 ----
    if res["duplicate_or_identity_problems"]:
        res["reasons"].append("**产物选择问题**：" + "；".join(res["duplicate_or_identity_problems"][:4]))
    if res["hard_checks"]:
        bystate = {}
        for k, v in res["hard_checks"].items():
            bystate.setdefault(v["state"], []).append(k)
        res["hard_checks_summary"] = {k: len(v) for k, v in bystate.items()}
    for lst, head in ((hard_fail, "**硬检查失败**"), (cons_fail, "**自洽性失败**"),
                      (config_problems, "**配置违规**"), (backend_problems, "**后端/融合问题**"),
                      (cmp_bad, "**跨模式/跨重复不一致**"),
                      (type_problems_, "**类型非法**"), (field_problems, "**缺字段**"),
                      (hard_missing, "**硬检查字段缺失**"), (cons_missing, "**自洽性字段缺失**"),
                      (hard_illegal, "**硬检查类型非法**"), (cons_illegal, "**自洽性类型非法**"),
                      (backend_insufficient, "**后端/融合证据不足**")):
        if lst:
            res["reasons"].append(head + "：" + "；".join(lst[:4]) + ("…" if len(lst) > 4 else ""))
    uncovered = [b for b, v in res["branches"].items() if v["verdict"] == "UNCOVERED"]
    if uncovered:
        res["reasons"].append(f"**未覆盖分支**（未触发即不得算通过）：{uncovered}")
    # ---- 判定：FAIL > INCOMPLETE > PASS ----
    fail_like = bool(res["duplicate_or_identity_problems"] or hard_fail or cons_fail
                     or config_problems or backend_problems or cmp_bad or type_problems_
                     or any(not d["equal"] for d in gate["serial_vs_ar"].values())
                     or any(v is False for v in gate["repeats"].values()))
    incomplete_like = bool(field_problems or hard_missing or cons_missing or hard_illegal
                           or cons_illegal or uncovered or backend_insufficient
                           or any(v is None for v in gate["repeats"].values()))
    if fail_like:
        res["verdict"] = "CASE_FAIL"
    elif incomplete_like:
        res["verdict"] = "CASE_INCOMPLETE"
    else:
        res["verdict"] = "CASE_PASS"
    res["production_divergence_note"] = (
        "production 与 AR 的差异（含**长度不同**）**只作发散轨迹记录**：不判控制逻辑失败、"
        "不自动归为无害浮点误差、不声明同输出无损加速")
    res["branch_observations_total"] = obs_total
    return res


# ------------------------------------------------------------------ CLI
def main() -> int:
    ap = argparse.ArgumentParser(description="B6 分支覆盖 + 严格验收（只读 CPU）")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--declared", default=None,
                    help="用例声明 JSON：{cases:{case_id:{branches,max_new,expect_prefill_fused}},"
                         "set_requirements:{branches,types}}；省略则用内建 `b6_cases.py`")
    ap.add_argument("--cases", default=None)
    ap.add_argument("--expect-backend", default=None)
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--list-cases", action="store_true")
    a = ap.parse_args()
    from . import b6_cases as BC
    if a.list_cases:
        for c in BC.B6_CASES:
            print(f"  {c['case_id']:10s} kind={c['kind']:26s} max_new={c['max_new']:2d} "
                  f"branches={list(c['branches'])} expect_prefill_fused={c['expect_prefill_fused']}")
        return 0
    if a.declared:
        dpath = Path(a.declared)
        if not dpath.exists():
            print(f"[b6cov] 声明文件不存在：{dpath} ⇒ INVALID_ARGUMENTS（退出码 3，不写产物）")
            return 3
        try:
            declared = json.loads(dpath.read_text(encoding="utf-8-sig"))
        except Exception as e:                                        # noqa: BLE001
            print(f"[b6cov] 声明文件不可解析：{e} ⇒ INVALID_ARGUMENTS（退出码 3，不写产物）")
            return 3
    else:
        declared, dpath = BC.declared_document(), None
    if not isinstance(declared, dict) or not declared:
        print("[b6cov] 声明为空 ⇒ INVALID_ARGUMENTS（退出码 3，不写产物）")
        return 3
    if isinstance(declared.get("cases"), dict):
        case_map, set_req = declared["cases"], (declared.get("set_requirements") or {})
    else:
        case_map, set_req = declared, {}
    if not case_map:
        print("[b6cov] 声明里没有任何 case ⇒ INVALID_ARGUMENTS（退出码 3，不写产物）")
        return 3
    cases = ([x.strip() for x in a.cases.split(",") if x.strip()] if a.cases else sorted(case_map))
    missing_decl = [c for c in cases if c not in case_map]
    if missing_decl:
        print(f"[b6cov] 用例未在声明中出现：{missing_decl} ⇒ INVALID_ARGUMENTS（退出码 3，不写产物）")
        return 3
    out = dict(tool="b6_branch_coverage_cpu", tag=a.tag, cases=cases, expect_backend=a.expect_backend,
               declared_file=(str(dpath) if dpath else "内建 b6_cases.py（单一来源）"),
               set_requirements=set_req, run_utc=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
               cases_result={})
    print(f"== B6 分支覆盖 / 严格验收 == tag={a.tag} cases={cases} expect_backend={a.expect_backend}")
    for c in cases:
        spec = dict(case_map.get(c) or {})
        runs, probs = find_runs(a.tag, c, Path(a.outdir))
        r = evaluate_case(c, spec, runs, expect_backend=a.expect_backend,
                          expect_config=dict(max_new=spec.get("max_new"),
                                             modes=spec.get("modes") or list(MODES)),
                          run_problems=probs)
        out["cases_result"][c] = r
        print(f"  [{c}] verdict={r['verdict']}  分支={ {b: v['verdict'] for b, v in r['branches'].items()} }")
        for x in r["reasons"]:
            print(f"      · {x}")
        sa = {k: (v["kind"], v["first_token_diff"], v["length_difference_position"])
              for k, v in r["gate"]["serial_vs_ar"].items()}
        pv = {k: v["kind"] for k, v in r["gate"]["production_vs_ar"].items()}
        print(f"      serial vs AR（严格）：{sa}   production vs AR（**只记录**）：{pv}")
    set_branches, types_seen = {}, set()
    for c, r in out["cases_result"].items():
        for b, n in (r.get("branch_observations_total") or {}).items():
            set_branches[b] = set_branches.get(b, 0) + int(n)
        if case_map.get(c, {}).get("kind"):
            types_seen.add(case_map[c]["kind"])
    set_branch_verdict = {b: ("COVERED" if set_branches.get(b, 0) > 0 else "UNCOVERED")
                          for b in (set_req.get("branches") or [])}
    set_type_verdict = {t: ("COVERED" if t in types_seen else "UNCOVERED")
                        for t in (set_req.get("types") or [])}
    out["set_result"] = dict(branches=set_branch_verdict, types=set_type_verdict,
                             observed=dict(set_branches),
                             note="集合级要求未满足 ⇒ 整体不得判 PASS（未触发即未覆盖）")
    if set_branch_verdict or set_type_verdict:
        print(f"  集合级：分支={set_branch_verdict}")
        print(f"  集合级：类型={set_type_verdict}")
    verdicts = [r["verdict"] for r in out["cases_result"].values()]
    n_pass = verdicts.count("CASE_PASS")
    set_ok = (all(v == "COVERED" for v in set_branch_verdict.values())
              and all(v == "COVERED" for v in set_type_verdict.values()))
    out["summary"] = dict(n_cases=len(verdicts), n_pass=n_pass,
                          n_incomplete=verdicts.count("CASE_INCOMPLETE"),
                          n_fail=verdicts.count("CASE_FAIL"),
                          set_requirements_ok=bool(set_ok),
                          all_pass=bool(n_pass == len(verdicts) and verdicts and set_ok))
    print(f"== 汇总：CASE_PASS={n_pass}/{len(verdicts)} "
          f"INCOMPLETE={out['summary']['n_incomplete']} FAIL={out['summary']['n_fail']} "
          f"集合级要求={out['summary']['set_requirements_ok']}")
    if not a.no_write:
        p = Path(a.outdir) / f"{a.tag}_b6_branch_coverage_{out['run_utc']}.json"
        out["artifact_json"] = str(p)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[b6cov] RESULT_JSON={p}")
    return 0 if out["summary"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
