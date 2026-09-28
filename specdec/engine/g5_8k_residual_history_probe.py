"""g5_8k_residual_history_probe.py — base_bf16 干预组**残余分歧**的缓存历史单变量探针。

问题（只问一件事）
  在 tag `qk32c_20260928T025500Z`/`qk32` 组里，base_bf16 的 AR 与 SpecDec 在
  **生成 index 52（输入位置 8243）** 首次分歧：AR=279、Spec=64871。
  两条路径在**同一位置、同一当前 token、同一精度、同一前向入口**下只差**缓存内容**
  （AR 历史 vs SpecDec 历史）。
  ⇒ **只改变缓存历史，是否足以复现不同的 argmax？**

为什么这个对照是干净的
  * 位置：两条路径都在 cache 长度 **8243**（位置 0..8242）之上、把当前 token 放在位置 **8243**；
    该行的 logits 预测位置 8244 = 生成 index **52**。
  * 当前 token：AR 喂 `gen[51]`，SpecDec 的 repair 喂上一轮 `g[0]`，二者同为 **323**（前 52 个 token 一致）。
  * **同一个单 token 前向入口**：引擎的 `spec_dec._step(model, token, cache)` 体就是
    `model._run(model.embed_tokens([[token]]), cache, need_logits=True)`，
    与 AR 解码单步（`g5_8k_smoke_cudnn` 的 AR 循环）**是同一个表达式**。两次重放都用 `_step`。
  * 精度：两组都用 target-only QK FP32（`diag_qk_scope.QKScopeHook`），重放时同样开启；draft 不参与。
  * 前缀：分歧前 52 个 token 完全相同（由捕获断言核对，不靠假设）。

先证明再判断
  每次重放都必须**逐元素等于**其真实行（`elementwise_equal=True`）才允许进入结论；
  否则标 `REPLAY_MISMATCH` 并停止。

用法：
  RP_MODE=ar|spec|compare  RP_KV=base_bf16（默认）  RP_TAG=... python -m specdec.engine.g5_8k_residual_history_probe
"""
from __future__ import annotations

import glob
import json
import os
import time
import traceback
from pathlib import Path

import torch

from .. import results_schema as S
from . import diag_cache_cmp as CC
from . import diag_fields as DF
from . import diag_qk_scope as QS
from . import model_self as M
from . import spec_cache
from . import spec_dec
from .smoke_checks import CheckRegistry, prefill_finiteness_live
from .smoke_hooks import DecisionHooks
from . import g5_8k_spec_variance_probe as P5V

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TARGET, DRAFT = P5V.TARGET, P5V.DRAFT

MODE = os.environ.get("RP_MODE", "ar").strip().lower()
KV = os.environ.get("RP_KV", "base_bf16").strip()
TAG = os.environ.get("RP_TAG", "reshist")
REF_TAG = os.environ.get("RP_REF_TAG", "qk32c_20260928T025500Z")
REF_GROUP = os.environ.get("RP_REF_GROUP", "qk32")
CTX = int(os.environ.get("RP_CTX", "8192"))
CHUNK = int(os.environ.get("RP_CHUNK", "2048"))
MAXNEW = int(os.environ.get("RP_MAXNEW", "64"))
K = int(os.environ.get("RP_K", "4"))
ARM = os.environ.get("RP_ARM", "cudnn_prefill")

# 本轮**只**定位 base_bf16 的 index 52 / 输入位置 8243
DIV_INDEX = int(os.environ.get("RP_DIV_INDEX", "52"))
CACHE_LEN_BEFORE = int(os.environ.get("RP_CACHE_LEN_BEFORE", "8243"))   # = 8192 + 51
WATCH = [279, 64871, 323]
assert KV == "base_bf16", "本轮只定位 base_bf16"
assert MAXNEW == 64 and K == 4, "不得缩短 max_new"


def _write(rec: dict, run_id: str) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = OUTDIR / f"{TAG}_{KV}_{MODE}_{run_id}.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)
    return p


def _ref(mode: str) -> dict:
    p = [x for x in glob.glob(str(OUTDIR / f"{REF_TAG}_{REF_GROUP}_{KV}_{mode}_*.json"))
         if "analysis" not in x and "revision" not in x][0]
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _row(t: torch.Tensor) -> dict:
    r = t.detach().float().reshape(-1).cpu()
    f = DF.row_fields(r, watch=WATCH)
    top2 = torch.topk(r, 2).values
    f["margin_top1_top2"] = float(top2[0] - top2[1])
    f["tie_at_top"] = bool(float(top2[0]) == float(top2[1]))
    return f


def _to_cpu(cache):
    """**真正独立**的 CPU 副本：`.detach().cpu().clone()`（`.cpu()` 对 CPU 张量是 no-op）。"""
    return [{k: (v.detach().cpu().clone() if torch.is_tensor(v) else v) for k, v in d.items()}
            for d in cache]


def _step_row(tgt, tok: int, cache):
    """用引擎自己的单 token 入口重放一次，返回 (row_fp32, 被改写的缓存, 工作副本)。"""
    work = CC.deep_clone(cache)
    with torch.inference_mode(), M.attention_phase("repair"):
        lg, new_cache = spec_dec._step(tgt, tok, work)
    return lg[0, -1].detach().clone(), new_cache, work


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec: dict = dict(
        run_id=run_id, tool="g5_8k_residual_history_probe", mode=MODE, kv=KV, tag=TAG,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git=S.git_meta(ROOT),
        config=dict(ctx=CTX, chunk=CHUNK, max_new=MAXNEW, k=K, greedy=True,
                    attention_strategy=ARM, target_kv=KV, draft_kv="base_bf16",
                    divergence_gen_index=DIV_INDEX, cache_len_before=CACHE_LEN_BEFORE,
                    precision_group="qk32 (target-only QK FP32)", watch=WATCH),
        reference=dict(tag=REF_TAG, group=REF_GROUP, kv=KV),
        question="只改变缓存历史（同位置/同 token/同精度/同一单 token 前向入口），"
                 "是否足以复现不同的 argmax？",
        scope="只定位 base_bf16 index 52 / 输入位置 8243；不做泛化 2×2、不扩展 v_fp8、"
              "不改 backend、不缩短 max_new、不改算法或验收标准；不做性能测量。")
    print(f"[probe] run_id={run_id} mode={MODE} kv={KV} tag={TAG}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    if MODE == "compare":
        return _compare(rec, run_id)

    ok, why = P5V.commit_guard()
    pf = P5V.PF.qualify()
    rec["commit_guard"] = dict(ok=ok, why=why, line_gib=float(P5V.MIN_COMMIT_GIB),
                              overridden=bool(float(P5V.MIN_COMMIT_GIB) < 16.0))
    rec["pstate"] = pf
    print(f"[probe] commit: {why}")
    if not ok or pf.get("qualified") is not True:
        rec["stage"] = "refused"
        _write(rec, run_id)
        return 3

    hooks = qk = None
    try:
        rec["diag_precision_baseline"] = M.set_diag_preset("eager")
        rec["attention_scope_proof"] = QS.static_scope_proof()
        from .lowmem_load import load_ours_bf16
        tgt, tok, tinfo = load_ours_bf16(TARGET)
        drf = None
        if MODE == "spec":
            drf, _t2, _di = load_ours_bf16(DRAFT)
            drf.set_kv_format("fp16")
        rec["load"] = dict(target=tinfo, draft=None if drf is None else "loaded")
        tgt.set_kv_format(spec_dec._fmt(KV))
        M.set_attention_strategy(ARM)
        M.reset_attention_stats()
        ids = P5V.load_ids(tok)[:, :CTX].contiguous()
        prompt_len = int(ids.shape[1])
        rec["input"] = dict(ctx=prompt_len, ids_hash=P5V._gen_hash([int(x) for x in ids[0].tolist()]))
        assert prompt_len + DIV_INDEX - 1 == CACHE_LEN_BEFORE, "位置自洽检查失败"

        ref_ar = _ref("ar")
        ref_spec = _ref("spec")
        ref_gen = [int(x) for x in ref_spec["run"]["gen"]]
        ref_ar_gen = [int(x) for x in ref_ar["run"]["gen"]]
        ref_round = next(r for r in ref_spec["rounds"] if r["pre"] == CACHE_LEN_BEFORE)
        fed_token = ref_gen[DIV_INDEX - 1]           # = gen[51]
        rec["expectation"] = dict(
            ref_ar_gen_head=ref_ar_gen[:DIV_INDEX + 1],
            ref_spec_token_at_div=ref_gen[DIV_INDEX],
            fed_token=fed_token,
            ref_round=ref_round,
            note="分歧前 52 个 token 应完全一致；当前 token 两条路径都应为 fed_token")

        reg = CheckRegistry()
        torch.cuda.reset_peak_memory_stats()
        acc: dict = {}
        inner = P5V.timed_chunked_prefill(acc, CHUNK, tgt)
        hooks = DecisionHooks(tgt=tgt, drf=drf, registry=reg, prefill_impl=inner)
        hooks.install()
        qk = QS.QKScopeHook(tgt=tgt, enabled=True)     # target-only QK FP32（与 qk32 组一致）
        qk.install()

        if MODE == "ar":
            snap, real_row, gen_head = _run_ar(tgt, ids, inner, reg, rec, fed_token, prompt_len)
            real_src = "AR 真实 decode 单步（tgt._run 单 token）"
        else:
            snap, real_row, captured = _run_spec(tgt, drf, ids, reg, rec, tok.eos_token_id,
                                                 ref_round)
            real_src = "SpecDec 真实 target repair（spec_dec._step）"

        # ---- 重放前先留**真正独立**的 CPU 参照（顺序很关键：若放在重放之后，检查又会恒真）----
        ref_cpu = _to_cpu(snap)
        cp = OUTDIR / f"{TAG}_{KV}_{MODE}_{run_id}.cache.pt"
        torch.save(ref_cpu, str(cp))

        # ---- 重放：同一个单 token 前向入口 ----
        replay_row, _newc, work = _step_row(tgt, fed_token, snap)
        cmp_ = DF.compare_rows(real_row, replay_row)
        # 重放之后取第二份副本 → 与参照逐层逐键比较（这才是**可失败**的完整性检查）
        snap_cpu = _to_cpu(snap)
        integ = CC.integrity_check(ref_cpu, snap_cpu)
        integ["workcopy_shares_storage_with_snapshot"] = bool(
            any(snap[i]["k"].data_ptr() == work[i]["k"].data_ptr() for i in range(len(snap))))
        integ["shapes_are_independent_copies"] = not integ["workcopy_shares_storage_with_snapshot"]
        rec["snapshot_integrity"] = integ
        rvr = dict(cmp_)          # `compare_rows` 已含 `note` ⇒ 必须 update，不能用 dict(**)
        rvr["must_be_equal"] = "必须 elementwise_equal=True，否则判 REPLAY_MISMATCH"
        rec["single_token_replay"] = dict(
            entry_point="spec_dec._step(model, token, cache)",
            entry_point_body="model._run(model.embed_tokens([[token]]), cache, need_logits=True)",
            entry_point_identical_to_ar_decode=True,
            ar_decode_expression="tgt._run(tgt.embed_tokens(torch.tensor([[nxt]])), cache, need_logits=True)",
            token=fed_token, token_position=CACHE_LEN_BEFORE,
            predicted_position=CACHE_LEN_BEFORE + 1, gen_index=DIV_INDEX,
            cache_len_before=spec_cache.cache_length(snap),
            real_row_source=real_src, real_row=_row(real_row), replay_row=_row(replay_row),
            replay_vs_real=rvr,
            replay_ok=bool(cmp_["elementwise_equal"]),
        )
        rows_dump = dict(real=real_row.detach().float().cpu(),
                         replay=replay_row.detach().float().cpu())
        rec["cache_meta"] = CC.metadata(snap)
        rp = OUTDIR / f"{TAG}_{KV}_{MODE}_{run_id}.rows.pt"
        torch.save(rows_dump, str(rp))
        rec["cache_dump"] = str(cp)
        rec["rows_dump"] = str(rp)
        rec["qk_scope_report"] = qk.report()
        qk.remove()
        qk = None
        hooks.remove()
        hooks = None

        pf_f = prefill_finiteness_live(reg, ["target"] if MODE == "ar" else ["target", "draft"])
        rec["prefill_logits_finite"] = pf_f
        own_gen = gen_head if MODE == "ar" else list(rec.get("spec_emitted") or [])
        rec["own_emitted_matches_reference"] = bool(
            own_gen[:DIV_INDEX] == ref_gen[:DIV_INDEX])
        rec["run"] = dict(
            gen=own_gen,
            gen_len=len(own_gen),
            max_new=MAXNEW, stop_reason=("captured" if MODE == "ar" else "captured_after_repair"),
            peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2 ** 20, 3),
            peak_reserved_mib=round(torch.cuda.max_memory_reserved() / 2 ** 20, 3),
            attention_stats=M.attention_stats(),
            timing_note="计时/显存仅作诊断记录；本轮不做加速或减速声明。")
        bad = (not rec["single_token_replay"]["replay_ok"]) or (not integ["all_equal"])
        rec["stage"] = "complete" if not bad else "replay_mismatch"
        p = _write(rec, run_id)
        s = rec["single_token_replay"]
        print(f"[probe] cache_len={s['cache_len_before']} token={s['token']} "
              f"pos={s['token_position']} -> 预测位置 {s['predicted_position']} (index {DIV_INDEX})")
        print(f"[probe] real  : argmax={s['real_row']['argmax_id']} max={s['real_row']['max_score']} "
              f"tied={s['real_row']['n_max_tied']}{s['real_row']['tied_max_tokens']} "
              f"margin={s['real_row']['margin_top1_top2']}")
        print(f"[probe] replay: argmax={s['replay_row']['argmax_id']} max={s['replay_row']['max_score']} "
              f"tied={s['replay_row']['n_max_tied']}{s['replay_row']['tied_max_tokens']} "
              f"margin={s['replay_row']['margin_top1_top2']}")
        print(f"[probe] replay vs real: elementwise_equal={cmp_['elementwise_equal']} "
              f"max|Δ|={cmp_['abs_err_max']} argmax_same={cmp_['argmax_same']}")
        print(f"[probe] snapshot_integrity all_equal={integ['all_equal']} "
              f"max|Δ|={integ['max_abs_diff']} tensors={integ['n_tensors_compared']}")
        print(f"[probe] RESULT_JSON={p}")
        return 0 if not bad else 5
    except Exception:
        rec["stage"] = "crashed"
        rec["traceback"] = traceback.format_exc()
        try:
            if qk is not None:
                qk.remove()
            if hooks is not None:
                hooks.remove()
        except Exception:                                        # noqa: BLE001
            pass
        M.set_diag_precision(qk32=False, prob32=False)
        p = _write(rec, run_id)
        tb = rec["traceback"]
        print(f"[probe] !! 崩溃（已落盘 {p}）：\n{tb[-800:]}")
        return 2


def _run_ar(tgt, ids, inner, reg, rec, fed_token, prompt_len):
    """自由 AR（target-only QK FP32）推进到 cache 长度 = CACHE_LEN_BEFORE，捕获该处缓存。"""
    ref_ar_gen = rec["expectation"]["ref_ar_gen_head"]
    with torch.inference_mode(), M.attention_phase("prefill"):
        lg, cache = inner(tgt, ids)
    reg.check_finite(lg, phase="prefill", model="target", role="prefill_lg", where="AR prefill")
    nxt = int(lg.reshape(-1, lg.shape[-1])[-1].argmax())
    gen = [nxt]
    steps = []
    n_feed = CACHE_LEN_BEFORE - prompt_len          # 51
    while len(steps) < n_feed:
        with torch.inference_mode(), M.attention_phase("decode"):
            lg, cache = tgt._run(tgt.embed_tokens(torch.tensor([[nxt]], device=ids.device)),
                                 cache, need_logits=True)
        reg.check_finite(lg, phase="decode", model="target", role="decode_lg",
                         round_=len(steps), step=len(steps) + 1, where="AR 单步（紧接 argmax）")
        nxt = int(lg.reshape(-1, lg.shape[-1])[-1].argmax())
        gen.append(nxt)
        steps.append(dict(step=len(steps) + 1, argmax=nxt, gen_index=len(gen) - 1))
    clen = spec_cache.cache_length(cache)
    ok_prefix = (gen[:DIV_INDEX] == ref_ar_gen[:DIV_INDEX])
    rec["ar_trajectory"] = dict(
        trajectory_mode="free_generation", teacher_forced=False, cache_len=clen,
        gen_head=gen[:DIV_INDEX + 1], ref_head=ref_ar_gen[:DIV_INDEX + 1],
        fed_token_expected=fed_token, fed_token_actual=gen[DIV_INDEX - 1],
        matches_reference_prefix=bool(ok_prefix),
        note="自由生成（喂自己的 argmax）；必须与前 52 个 token 逐 token 一致才可继续")
    assert clen == CACHE_LEN_BEFORE, f"缓存长度 {clen} != {CACHE_LEN_BEFORE}"
    assert gen[DIV_INDEX - 1] == fed_token, f"当前 token {gen[DIV_INDEX-1]} != {fed_token}"
    assert ok_prefix, "AR 自由轨迹与参考不一致 ⇒ 不是真实轨迹复现"
    snap = CC.deep_clone(cache)                     # ★ 独立快照（deep_clone，**重放之前**）
    # 真实行 = AR 循环**自然**走出的下一步（与 `_step` 是同一个表达式），**不是**重放出来的
    with torch.inference_mode(), M.attention_phase("decode"):
        lg, _c2 = tgt._run(tgt.embed_tokens(torch.tensor([[fed_token]], device=ids.device)),
                           cache, need_logits=True)
    reg.check_finite(lg, phase="decode", model="target", role="decode_lg",
                     round_=len(steps), step=len(steps) + 1,
                     where="AR 真正决定 index 52 的那一步（紧接被 argmax 使用）")
    real_row = lg[0, -1].detach().clone()
    ref_head = rec["expectation"]["ref_ar_gen_head"]
    rec["ar_trajectory"]["real_next_argmax"] = int(real_row.argmax())
    rec["ar_trajectory"]["ref_next_token"] = ref_head[DIV_INDEX]
    assert int(real_row.argmax()) == ref_head[DIV_INDEX], "AR 真实决策与参考不一致"
    return snap, real_row, gen


def _run_spec(tgt, drf, ids, reg, rec, eos, ref_round):
    """真实 SpecDec（target-only QK FP32）跑到 cache_len=8243 的那一轮并捕获 repair 前的缓存。

    ⚠️ **不在这里安装 DecisionHooks**：`main()` 已经装了一份。再装第二份会因为
    `smoke_hooks._HOOK_REGISTRY` 以**类**为键而被覆盖，形成
    `_run_hook → qk._wrapper → _run_hook(第二份) → _wrapper → …` 的**无限递归**
    （首跑就在这里 RecursionError）。捕获逻辑改为在 `_verify_spy` 里完成。
    """
    held: dict = {}
    _orig_verify, _orig_step = spec_dec._verify, spec_dec._step
    rounds: list[dict] = []

    def _verify_spy(model, cache, cands, *a, **k):
        # 拿到 blk + repair 之后，下一轮的 verify 一进来就停（与预算无关）
        if "blk" in held and "repair" in held:
            raise _Captured()
        pre = spec_cache.cache_length(cache)          # 就地改写**之前**的长度
        if pre == CACHE_LEN_BEFORE and "pre_snap" not in held:
            held["pre_snap"] = CC.deep_clone(cache)   # ★ 独立快照（deep_clone）
            held["cands"] = [int(x) for x in cands]
            held["round"] = reg.round + 1
        rows, blk = _orig_verify(model, cache, cands, *a, **k)
        if "pre_snap" in held and "blk" not in held:
            held["blk"] = CC.deep_clone(blk)          # ★ 独立快照（deep_clone）
            held["verify_rows"] = rows.detach().clone()
        return rows, blk

    def _step_spy(model, token, cache, *a, **k):
        # ⚠️ 长度必须在**调用之前**读：`_step` 会就地改写传入的 dict 把 token 追加进去，
        #    调用之后再读拿到的是追加后的长度（首跑就把 8243 误记成 8244）。
        len_before = spec_cache.cache_length(cache)
        lg, c = _orig_step(model, token, cache, *a, **k)
        if model is tgt and "pre_snap" in held and "repair" not in held:
            held["repair"] = lg[0, -1].detach().clone()
            held["repair_token"] = int(token)
            held["repair_cache_len_before"] = len_before
            held["repair_cache_len_after"] = spec_cache.cache_length(c)
        return lg, c

    spec_dec._verify, spec_dec._step = _verify_spy, _step_spy
    try:
        spec_dec.spec_decode_greedy(tgt, drf, ids, MAXNEW, k=K, eos_id=eos,
                                    target_kv=KV, draft_kv="base_bf16",
                                    log=lambda d: rounds.append(dict(d)))
    except _Captured:
        pass
    finally:
        spec_dec._verify, spec_dec._step = _orig_verify, _orig_step

    logged = next((r for r in rounds if r["pre"] == CACHE_LEN_BEFORE), None)
    checks = dict(
        captured=("pre_snap" in held and "blk" in held and "repair" in held),
        pre=spec_cache.cache_length(held["pre_snap"]) if "pre_snap" in held else None,
        expect_pre=CACHE_LEN_BEFORE,
        matches_round=bool(logged and logged["round"] == ref_round["round"]),
        matches_Kp=bool(logged and logged["Kp"] == ref_round["Kp"]),
        matches_cands=bool(logged and list(logged["cands"]) == list(ref_round["cands"])),
        matches_g=bool(logged and list(logged["g"]) == list(ref_round["g"])),
        matches_emitted=bool(logged and list(logged["emitted"]) == list(ref_round["emitted"])),
        logged=logged, ref=ref_round,
        repair_token=held.get("repair_token"),
        repair_cache_len_before=held.get("repair_cache_len_before"),
        repair_cache_len_after=held.get("repair_cache_len_after"),
    )
    checks["all_match"] = all(checks[k] for k in
                              ("captured", "matches_round", "matches_Kp", "matches_cands",
                               "matches_g", "matches_emitted"))
    rec["capture_check"] = checks
    assert checks["all_match"], f"捕获条件不符：{ {k: v for k, v in checks.items() if k.startswith('matches') or k == 'captured'} }"
    # repair 前的缓存 = 该轮 block 缓存按 pre+m 截断（与 spec_dec 的做法一致）
    pre_m = CACHE_LEN_BEFORE + int(ref_round["m"])
    snap = spec_cache.trim_cache(held["blk"], pre_m)
    # 交叉核对：`trim(blk, pre+m)` 应与"verify 之前的那份快照"逐张量一致
    xchk = CC.compare_caches(held["pre_snap"], snap, dequantize=False)
    xchk["note"] = ("verify 前快照 vs trim(tgt_blk, pre+m)：本次 m=%d，二者应逐张量一致；"
                    "不一致说明截断口径或 block 语义理解有误" % ref_round["m"])
    rec["spec_repair_cache"] = dict(
        source="trim_cache(独立快照(tgt_blk), pre + m)",
        pre=ref_round["pre"], m=ref_round["m"], trimmed_to=pre_m,
        length=spec_cache.cache_length(snap),
        cross_check_vs_pre_verify_snapshot=xchk)
    emitted: list = []
    for r in rounds:
        emitted += [int(x) for x in r["emitted"]]
    rec["spec_emitted"] = emitted
    real_row = held["repair"]
    rec["rounds"] = rounds
    return snap, real_row, held


class _Captured(Exception):
    pass


def _compare(rec: dict, run_id: str) -> int:
    """纯 CPU：比较 AR 与 Spec 两份缓存历史下的**同一单 token 前向**结果。"""
    found = {}
    for mode in ("ar", "spec"):
        js = [x for x in glob.glob(str(OUTDIR / f"{TAG}_{KV}_{mode}_*.json"))
              if "compare" not in x]
        if js:
            found[mode] = json.loads(Path(sorted(js)[-1]).read_text(encoding="utf-8"))
    rec["sources"] = {m: r.get("run_id") for m, r in found.items()}
    if len(found) != 2 or any(r.get("stage") != "complete" for r in found.values()):
        rec["stage"] = "missing_or_bad_source"
        rec["source_stages"] = {m: r.get("stage") for m, r in found.items()}
        p = _write(rec, run_id)
        print(f"[probe] !! 来源缺失或未完成 ⇒ 停下 -> {p}")
        return 5

    a, b = found["ar"], found["spec"]
    ra, rb = a["single_token_replay"], b["single_token_replay"]
    checks = dict(
        both_replay_ok=bool(ra["replay_ok"] and rb["replay_ok"]),
        same_entry_point=bool(ra["entry_point"] == rb["entry_point"]),
        same_token=bool(ra["token"] == rb["token"]),
        same_token_position=bool(ra["token_position"] == rb["token_position"]),
        same_cache_len=bool(ra["cache_len_before"] == rb["cache_len_before"]),
        same_precision=bool(a["qk_scope_report"]["enabled"] and b["qk_scope_report"]["enabled"]),
        same_predicted_position=bool(ra["predicted_position"] == rb["predicted_position"]),
        same_gen_index=bool(ra["gen_index"] == rb["gen_index"]),
    )
    checks["all_controls_equal"] = all(v for k, v in checks.items() if k != "both_replay_ok")

    dumps = {}
    for m, r in found.items():
        c = glob.glob(str(OUTDIR / f"{TAG}_{KV}_{m}_{r['run_id']}.cache.pt"))
        if c:
            dumps[m] = torch.load(c[-1], map_location="cpu")
    hist = CC.compare_caches(dumps["ar"], dumps["spec"], dequantize=False) if len(dumps) == 2 else {}
    if hist:
        hist["kind"] = "cache_history"
        hist["note"] = "base_bf16 ⇒ 原始 bf16 载荷比较（无需反量化）"

    rowcmp = None
    if len(dumps) == 2:
        rj = {m: torch.load(sorted(glob.glob(str(OUTDIR / f"{TAG}_{KV}_{m}_{found[m]['run_id']}.rows.pt")))[-1],
                            map_location="cpu") for m in ("ar", "spec")}
        rowcmp = dict(DF.compare_rows(rj["ar"]["replay"], rj["spec"]["replay"]))
        rowcmp.update(
            ar_replay=_row(rj["ar"]["replay"]), spec_replay=_row(rj["spec"]["replay"]),
            comparison_note="两侧都是**重放行**（已各自证明等于真实行），因此这是"
                            "『只改缓存历史』的单变量对照")

    same_argmax = None
    if rowcmp:
        same_argmax = bool(rowcmp["ar_replay"]["argmax_id"] == rowcmp["spec_replay"]["argmax_id"])
    verdict = ("HISTORY_ALONE_SUFFICES" if (checks["all_controls_equal"]
                                           and checks["both_replay_ok"]
                                           and same_argmax is False)
               else ("HISTORY_ALONE_NOT_ENOUGH" if same_argmax is True else "INCONCLUSIVE"))
    rec.update(control_checks=checks, cache_history_compare=hist, row_compare=rowcmp,
               same_argmax=same_argmax, verdict=verdict,
               verdict_rule="先满足：两次重放各自逐元素等于真实行、且入口/token/位置/长度/精度全同；"
                            "否则 INCONCLUSIVE。再判断：缓存历史不同是否足以产生不同 argmax。",
               stage="complete")
    p = _write(rec, run_id)
    print(f"[probe] controls: {json.dumps(checks, ensure_ascii=False)}")
    print(f"[probe] cache history: all_equal={hist.get('all_equal')} max|Δ|={hist.get('max_abs_diff')} "
          f"n={hist.get('n_tensors_compared')}")
    if rowcmp:
        print(f"[probe] AR replay  : argmax={rowcmp['ar_replay']['argmax_id']} "
              f"max={rowcmp['ar_replay']['max_score']} tied={rowcmp['ar_replay']['n_max_tied']}"
              f"{rowcmp['ar_replay']['tied_max_tokens']} margin={rowcmp['ar_replay']['margin_top1_top2']}")
        print(f"[probe] Spec replay: argmax={rowcmp['spec_replay']['argmax_id']} "
              f"max={rowcmp['spec_replay']['max_score']} tied={rowcmp['spec_replay']['n_max_tied']}"
              f"{rowcmp['spec_replay']['tied_max_tokens']} margin={rowcmp['spec_replay']['margin_top1_top2']}")
        print(f"[probe] rows elementwise_equal={rowcmp['elementwise_equal']} "
              f"argmax_same={rowcmp['argmax_same']} max|Δ|={rowcmp['abs_err_max']}")
    print(f"[probe] VERDICT={verdict}")
    print(f"[probe] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
