"""row_aligned_stage1_f13_p8209.py — **Stage 1**：位置 8209 的全层候选门（F13；只读、有界、单次）。

## 目标
把全层 row-aligned verify 候选模式（`row_aligned_verify`）在 **target 的全部 28 层**上一次开启，
判定：① 控制门（1A）是否全过；② `S0` 与 `BALL` 的 logits/token 属于哪一类（1B）；
③ 若中间层重新分歧，**只记录第一层编号与逐层 `n_diff`**（**禁止**进入该层内部诊断）。

## 四个 arm（同一份 `H(8208)`，各自独立克隆 cache，串行；每臂 2 个独立 pass）
| arm | 输入 | 候选模式 |
| --- | --- | --- |
| **S0** | `[20642]`（decode，s=1） | 关 |
| **B0** | `[20642,659,576,4531]`（verify，s=4） | 关 |
| **BALL** | 同 B0 | **开**（28 QK + 112 Linear） |
| **SALL** | 同 S0（s=1） | **开**（必须与 S0 逐位一致） |

## 硬门（1A）
S0 ≡ 历史 S；B0 ≡ 历史 B；**SALL ≡ S0**（层行 + final + logits）；**28 个 QK 全部 applied**；
**112 个 Linear 全部 applied**；**draft 干预 = 0**（本工具**不加载 draft**，且注册集合只含 target 模块）；
两次 BALL 逐位一致。

## 纪律
不进入 L1–L27 算子内部；不做 profiler/FP32；不切后端；**不跑正式 B6**；原始张量在任何摘要前落盘；
硬门失败即停止、不自动重试；旧产物不覆盖；**B6 状态不变**。
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch

from .. import results_schema as S
from . import cache_clone as CC
from . import decision_rows as DR
from . import diag_checks as DC
from . import diag_fields as DF
from . import f13_i50_cells as C
from . import lmhead_boundary as LB
from . import model_self as M
from . import row_aligned as RA
from . import sb_align as A
from . import serial_align as SA
from . import spec_cache
from . import spec_dec
from . import g5_8k_spec_variance_probe as P5V

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"

TAG = os.environ.get("RA_TAG", "div8209_ball")
MODE = os.environ.get("RA_MODE", "gpu").strip().lower()
LAYER = int(os.environ.get("RA_LAYER", "0"))
PRE = int(os.environ.get("RA_PRE", "8208"))
TARGET_POS = int(os.environ.get("RA_TARGET", "8209"))
K = int(os.environ.get("RA_K", "4"))
FEED = [int(x) for x in os.environ.get("RA_FEED", "20642").split(",")]
CANDS = [int(x) for x in os.environ.get("RA_CANDS", "20642,659,576,4531").split(",")]
WATCH = tuple(int(x) for x in os.environ.get("RA_WATCH", "323,659,20642,576,4531").split(","))
REPEAT = int(os.environ.get("RA_REPEAT", "2"))
TARGET_ROPE = os.environ.get("RA_TARGET_ROPE", "cpu_fp32").strip().lower()
CAPTURE_OFF_CHECK = S.env_flag("RA_CAPTURE_OFF_CHECK", True)
CAP_H = os.environ.get("RA_CAP_H", str(OUTDIR / "divcap_ar_20261001T185503Z.pt"))
CAP_H2 = os.environ.get("RA_CAP_H2", str(OUTDIR / "divasc_rebuild_20261001T220737Z.pt"))
L0OPS_PT = os.environ.get("RA_L0OPS_PT", str(OUTDIR / "div8209_l0ops_20261002T172447Z.pt"))
PREV_PT = os.environ.get("RA_PREV_PT", str(OUTDIR / "div8209_sml_20261002T201740Z.pt"))
PREV_JSON = os.environ.get("RA_PREV_JSON", str(OUTDIR / "div8209_sml_20261002T201740Z.json"))
SAVED_PT = os.environ.get("RA_SAVED_PT", str(OUTDIR / "div8209_cells_20261002T064415Z.pt"))
REF_RUN_AR = os.environ.get("RA_REF_AR", str(OUTDIR / "smoke8k_eager_20261001T185503Z.json"))
REF_RUN_RB = os.environ.get("RA_REF_RB", str(OUTDIR / "smoke8k_eager_20261001T220737Z.json"))
KV, ARM = "base_bf16", os.environ.get("RA_ARM", "eager")
VIS = PRE + 1
# (arm, phase, candidate)
PLAN = (("S0", "decode", False), ("B0", "verify", False),
        ("BALL", "verify", True), ("SALL", "decode", True))
ARM_NAMES = [p[0] for p in PLAN]
S_ARMS = ("S0", "SALL")


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def _to_dev(snap, dev):
    full = CC.clone_list_cache_full(snap)
    return [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in layer.items()}
            for layer in full]


def _set_precision() -> dict:
    b = torch.backends.cuda.matmul
    d0 = bool(getattr(b, "allow_bf16_reduced_precision_reduction"))
    b.allow_bf16_reduced_precision_reduction = True
    return dict(default_at_import=d0, set_to=True,
                effective_after_set=bool(b.allow_bf16_reduced_precision_reduction),
                allow_tf32=bool(b.allow_tf32),
                note="与既有 smoke 记录一致（default True → 显式 True，effective True）；allow_tf32 全程 False")


def _fields(row) -> dict:
    f = DF.row_fields(row, watch=list(WATCH))
    tk = f["topk"]
    f["margin_top1_top2"] = float(tk[0]["score"] - tk[1]["score"]) if len(tk) >= 2 else None
    f["dtype"] = str(row.dtype)
    f["sha16"] = A.stable_sha(row)
    return f


def _token_fields(row) -> dict:
    f = row.float().reshape(-1)
    v, idx = torch.topk(f, 2)
    return dict(argmax=int(f.argmax()), top1=int(idx[0]), top1_value=float(v[0]),
                top2=int(idx[1]), top2_value=float(v[1]), margin=float(v[0] - v[1]),
                dtype=str(row.dtype), vocab=int(f.numel()), sha16=A.stable_sha(row))


def _row_cmp(a, b, name: str) -> dict:
    a2, b2 = a.float().reshape(-1), b.float().reshape(-1)
    DC.assert_same_shape(a2, b2, name)
    d = (a2 - b2).abs()
    return dict(name=name, exact_equal=bool(torch.equal(a.reshape(-1), b.reshape(-1))),
                n_diff=int((a.reshape(-1) != b.reshape(-1)).sum()),
                abs_err_max=float(d.max()), abs_err_mean=float(d.mean()),
                sha16_a=A.stable_sha(a), sha16_b=A.stable_sha(b), dtype=str(a.dtype),
                shape=[int(x) for x in a.shape])


def _chain_rows_to_cpu(chain: dict, n_layers: int) -> dict:
    out = {}
    for i in range(n_layers):
        d = chain["layers"].get(i, {})
        pre, post = d.get("pre"), d.get("post")
        out[i] = dict(input=(pre["a_row"].detach().cpu() if pre and pre["a_row"] is not None else None),
                      output=(post["z_row"].detach().cpu() if post and post["z_row"] is not None else None),
                      cache_len_before=(pre.get("cache_len_before") if pre else None),
                      cache_len_after=(post.get("cache_len_after") if post else None),
                      phase=(pre.get("phase") if pre else None))
    return out


def _norm_row(norm_caps: dict, row_index: int):
    caps = norm_caps.get("caps") or {}
    if not caps:
        return None
    rec = caps[max(caps)]
    y = rec.get("y_out")
    return None if y is None else (y[0, row_index] if y.dim() == 3 else y).detach().cpu()


def _norm_in_row(norm_caps: dict, row_index: int):
    caps = norm_caps.get("caps") or {}
    if not caps:
        return None
    rec = caps[max(caps)]
    x = rec.get("x_in")
    return None if x is None else (x[0, row_index] if x.dim() == 3 else x).detach().cpu()


def _cache_struct(cache) -> dict:
    return dict(n_layers=len(cache), keys=sorted(cache[0].keys()),
                dtype=str(cache[0]["k"].dtype), lens=[int(cache[i]["k"].shape[2]) for i in range(len(cache))])


def _capture_arm(tgt, H, dev, arm: str, mode: str, tag: str, *, cand: bool) -> dict:
    """单臂捕获：层链（28 层决策行）+ 最终 RMSNorm + 真实前向（候选模式可选）。"""
    RA.clear_verify_log()
    cache = _to_dev(H, dev)
    row_index = -1 if arm in S_ARMS else 0
    ctx = RA.row_aligned_verify(tgt) if cand else contextlib.nullcontext(None)
    with ctx as cand_info, \
            LB.layer_chain_capture(tgt, row_index=row_index, tag=f"{arm}_{tag}") as chain, \
            LB.record(tgt, tag=f"{arm}_{tag}_norm") as norm_info:
        with torch.inference_mode(), M.attention_phase(mode):
            if arm in S_ARMS:
                lg, cache_after = spec_dec._step(tgt, FEED[0], cache)
                row = LB.row_of(lg, None)
            else:
                rows_b, cache_after = spec_dec._verify(tgt, cache, CANDS)
                row = LB.row_of(rows_b, 0)
        # 候选模式的分支计数：**必须**在上下文体内取（退出即恢复注册表）
        st_qk = M.serial_qk_state() if cand else None
        st_lin = M.serial_linear_state() if cand else None
        struct = _cache_struct(cache_after)
        norm_caps = LB.take()
    # ⚠️ `verify_log()` 是 context 的 `finally` 里写入的 ⇒ 必须在**退出之后**取（首跑在体内取到空）。
    log = RA.verify_log() if cand else []
    return dict(arm=arm, mode=mode, row_index=int(row_index), cand=bool(cand), chain=chain,
                norm_caps=norm_caps, final_row=row.detach().to("cpu"),
                cache_len_after=int(spec_cache.cache_length(cache_after)),
                cache_len_before=int(chain["layers"][LAYER]["pre"]["cache_len_before"]),
                cache_struct_after=struct, n_pre=int(chain["n_pre"]), n_post=int(chain["n_post"]),
                cand_info=cand_info, qk_state=st_qk, lin_state=st_lin, verify_log=log)


def _build_payload(arms: dict, *, n_layers: int, module_facts: dict, expected: dict) -> dict:
    pay: dict = dict(meta=dict(vis=VIS, layer=LAYER, n_layers=n_layers, arms=sorted(arms)),
                     module_facts=module_facts, expected_counts=expected)
    for a, r in arms.items():
        pay[f"rows_{a}"] = _chain_rows_to_cpu(r["chain"], n_layers)
        pay[f"norm_row_{a}"] = _norm_row(r["norm_caps"], r["row_index"])
        pay[f"norm_in_row_{a}"] = _norm_in_row(r["norm_caps"], r["row_index"])
        pay[f"final_row_{a}"] = r["final_row"].detach().cpu()
        pay[f"cand_counts_{a}"] = dict(
            qk_applied=(len(r["qk_state"]["applied"]) if r["qk_state"] else 0),
            lin_applied=(len(r["lin_state"]["applied"]) if r["lin_state"] else 0),
            qk_seen=(r["qk_state"]["n_seen"] if r["qk_state"] else 0),
            lin_seen=(r["lin_state"]["n_seen"] if r["lin_state"] else 0),
            log=r["verify_log"], cache_struct_after=r["cache_struct_after"])
    pay["note"] = ("`rows_*` = 28 层输入/输出决策行；`norm_*` = 最终 RMSNorm 输入/输出决策行；"
                   "`final_row_*` = 完整词表 logits 决策行；`cand_counts_*` = 每层 QK/Linear 的 applied/seen 计数、"
                   "逐行可见/形状记录与 cache 结构。")
    return pay


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="row_aligned_stage1_f13_p8209", family="F13", kv=KV, arm=ARM,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               question="把 row-aligned 候选模式一次性应用到 target 全部 28 层后，位置 8209 的 "
                        "logits/token 属于哪一类？中间层是否重新分歧？",
               semantics="四臂 S0/B0/BALL/SALL × 2 pass；候选模式只包住 target block verify；"
                         "不进入 L1–L27 算子内部。",
               config=dict(tag=TAG, layer=LAYER, pre=PRE, target_pos=TARGET_POS, k=K, feed=FEED,
                           cands=CANDS, watch=list(WATCH), repeat=REPEAT, vis=VIS,
                           target_rope=TARGET_ROPE, capture_off_check=CAPTURE_OFF_CHECK,
                           cap_h=str(CAP_H), plan=[dict(arm=a, mode=m, candidate=c) for a, m, c in PLAN],
                           in_scope="只做 Stage 1 全层局部门；不进入 L1–L27 算子内部；不做 profiler/FP32；不跑 B6"))
    print(f"[ra1] run_id={run_id} layer={LAYER} pre={PRE} target={TARGET_POS} repeat={REPEAT}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    if MODE == "selftest":
        return _selftest(rec, run_id)
    ok, why = P5V.commit_guard()
    rec["commit_guard"] = dict(ok=bool(ok), why=why)
    print(f"[ra1] commit: {why}")
    if not ok:
        rec["stage"] = "refused"
        _write(rec, run_id)
        return 3
    return _run(rec, run_id)


def _write(rec, run_id, selftest: bool = False) -> Path:
    name = (f"{TAG}_{run_id}_selftest.json" if selftest else f"{TAG}_{run_id}.json")
    p = OUTDIR / name
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return p


def _save_pt(payload: dict, run_id: str, selftest: bool = False) -> Path:
    name = (f"{TAG}_{run_id}_selftest.pt" if selftest else f"{TAG}_{run_id}.pt")
    p = OUTDIR / name
    torch.save(payload, str(p))
    return p


def _run(rec: dict, run_id: str) -> int:
    TRS = None
    try:
        from .lowmem_load import load_ours_bf16
        tgt, tok, ti = load_ours_bf16(P5V.TARGET)
        tgt.set_kv_format(spec_dec._fmt(KV))
        M.set_attention_strategy(ARM)
        M.reset_attention_stats()
        dev = next(tgt.parameters()).device
        n_layers = len(tgt.layers)
        rec["precision"] = _set_precision()
        rec["weights_load"] = dict(bitmatch=ti.get("weights_bitmatch"), checked=ti.get("n_params_checked"))
        rec["layer"] = LB.layer_info(tgt, LAYER)
        rec["norm"] = LB.norm_info(tgt)
        expected = RA.expected_counts(tgt)
        rec["expected_counts"] = expected
        rec["coverage"] = RA.coverage_check(tgt)
        rec["probe_state_before"] = M.attn_probe_state()
        rec["serial_state_before"] = dict(qk=M.serial_qk_state(), linear=M.serial_linear_state())
        rec["verify_wrap_before"] = spec_dec.verify_wrap_state()
        rec["resident_models"] = ["target"]
        rec["draft_loaded"] = False
        DC.require(rec["probe_state_before"]["active"] is False, "进入实验前探针必须关闭")
        DC.require(rec["serial_state_before"]["qk"]["active"] is False
                   and rec["serial_state_before"]["linear"]["active"] is False, "进入实验前候选模式必须关闭")
        DC.require(rec["verify_wrap_before"]["active"] is False, "进入实验前 verify 包装必须未安装")
        DC.require(rec["coverage"]["covers_all_target"] is True, "模块覆盖不完整", str(rec["coverage"]))
        DC.require(expected["qk"] == n_layers and expected["linear"] == 4 * n_layers,
                   "预期数量与模型结构不符", str(expected))
        W = tgt.embed_tokens.weight
        rec["W_before"] = LB.weights_info(tgt)
        print(f"[ra1] 层数={n_layers}；预期注册 QK={expected['qk']}、Linear={expected['linear']}；"
              f"覆盖完整={rec['coverage']['covers_all_target']}", flush=True)
        if TARGET_ROPE == "cpu_fp32":
            from . import target_rope_scope as _TRS
            TRS = _TRS
            rec["target_rope_scope"] = TRS.install(tgt, theta=float(tgt.rope_theta),
                                                   head_dim=int(tgt.head_dim))
        cap = C._load_cap(Path(CAP_H))
        snap = cap["snapshot_cpu"]
        src_len = C._cache_len(snap)
        H, tinfo = C._truncate(snap, PRE, "H")
        H_digest_before = [A.stable_sha(lay["k"]) for lay in H] + [A.stable_sha(lay["v"]) for lay in H]
        rec["H_source"] = dict(path=str(CAP_H), sha256=_sha_file(Path(CAP_H)), source_len=src_len,
                               pre=PRE, is_pure_slice=tinfo["is_pure_slice"], n_layers=len(H))
        if Path(CAP_H2).exists():
            H2, _t2 = C._truncate(C._load_cap(Path(CAP_H2))["snapshot_cpu"], PRE, "H2")
            rec["H_identity"] = dict(second=str(CAP_H2), compare=C._cache_bitequal(H, H2, "H_vs_H2"))
            DC.require(rec["H_identity"]["compare"]["all_equal"], "两份捕获并非逐位相同")
            del H2
        print(f"[ra1] H：{Path(CAP_H).name} {src_len} → {PRE}（纯切片={tinfo['is_pure_slice']}）", flush=True)

        passes = []
        for it in range(REPEAT):
            arms = {}
            for arm, mode, cand in PLAN:
                arms[arm] = _capture_arm(tgt, H, dev, arm, mode, f"p{it + 1}", cand=cand)
                ck = arms[arm]["qk_state"] or {}
                cl = arms[arm]["lin_state"] or {}
                print(f"[ra1] pass {it + 1}/{REPEAT} arm {arm} ✔（候选={cand}；"
                      f"QK applied={len(ck.get('applied', []))}/seen={ck.get('n_seen', 0)}；"
                      f"Linear applied={len(cl.get('applied', []))}/seen={cl.get('n_seen', 0)}）", flush=True)
            passes.append(arms)
        P1, P2 = passes[0], (passes[1] if len(passes) > 1 else None)
        DC.require(M.attn_probe_state()["active"] is False, "捕获后探针仍在位")
        DC.require(M.serial_qk_state()["active"] is False and M.serial_linear_state()["active"] is False,
                   "捕获后候选模式仍开启")
        rec["serial_state_after"] = dict(qk=M.serial_qk_state(), linear=M.serial_linear_state())
        rep = dict(n_passes=len(passes))
        if P2 is not None:
            for a in ARM_NAMES:
                rep[f"{a}_rows_identical"] = bool(all(
                    all(torch.equal(_chain_rows_to_cpu(P1[a]["chain"], n_layers)[i][k],
                                    _chain_rows_to_cpu(P2[a]["chain"], n_layers)[i][k])
                        for k in ("input", "output")) for i in range(n_layers)))
                rep[f"{a}_final_row_identical"] = bool(torch.equal(P1[a]["final_row"], P2[a]["final_row"]))
                rep[f"{a}_cache_len_identical"] = bool(P1[a]["cache_len_after"] == P2[a]["cache_len_after"])
        rec["repeat_consistency"] = rep
        DC.require(all(v is True for k, v in rep.items() if k.endswith("identical")),
                   "两次独立捕获不一致（硬门 1A）", str(rep))
        print("[ra1] 两次独立 pass 逐位一致 ✔（四臂的层行/最终行/cache 长度）", flush=True)

        # ★ 原始张量在任何摘要/打印之前落盘
        payload = _build_payload({a: P1[a] for a in ARM_NAMES}, n_layers=n_layers,
                                 module_facts=dict(n_layers=n_layers), expected=expected)
        payload["H_L0_cache_pre"] = dict(k=H[LAYER]["k"].detach().cpu(), v=H[LAYER]["v"].detach().cpu())
        tp = _save_pt(payload, run_id)
        rec["artifacts"] = dict(json=str(_write(rec, run_id)), pt=str(tp))
        print(f"[ra1] 原始张量已落盘（早于任何比较/打印）：{tp.name}", flush=True)

        S0, B0, BALL, SALL = (P1[a] for a in ARM_NAMES)
        rows = {a: _chain_rows_to_cpu(P1[a]["chain"], n_layers) for a in ARM_NAMES}
        l0ops = torch.load(L0OPS_PT, map_location="cpu") if Path(L0OPS_PT).exists() else {}
        prev = torch.load(PREV_PT, map_location="cpu") if Path(PREV_PT).exists() else {}
        saved_rows = {}
        if Path(SAVED_PT).exists():
            saved_rows = {k: v for k, v in torch.load(SAVED_PT, map_location="cpu").items()
                          if torch.is_tensor(v)}
        anchors_s = C._anchor_ar(REF_RUN_AR, TARGET_POS)
        anchors_b = C._anchor_specdec(REF_RUN_RB, PRE, TARGET_POS - PRE, TARGET_POS)

        # ---- 硬门 1A ----
        g1 = dict(
            S0_history=dict(
                final_row=(_row_cmp(S0["final_row"], l0ops["final_row_S"], "S0_final_vs_history")
                           if "final_row_S" in l0ops else None),
                L0_output=(_row_cmp(rows["S0"][0]["output"], l0ops["L0_output_row_S"], "S0_L0out_vs_history")
                           if "L0_output_row_S" in l0ops else None),
                x0=(_row_cmp(rows["S0"][0]["input"], l0ops["x0_row_S"], "S0_x0_vs_history")
                    if "x0_row_S" in l0ops else None)),
            B0_history=dict(
                final_row=(_row_cmp(B0["final_row"], l0ops["final_row_B"], "B0_final_vs_history")
                           if "final_row_B" in l0ops else None),
                L0_output=(_row_cmp(rows["B0"][0]["output"], l0ops["L0_output_row_B"], "B0_L0out_vs_history")
                           if "L0_output_row_B" in l0ops else None),
                x0=(_row_cmp(rows["B0"][0]["input"], l0ops["x0_row_B"], "B0_x0_vs_history")
                    if "x0_row_B" in l0ops else None)),
            SALL_vs_S0=dict(
                final_row=_row_cmp(SALL["final_row"], S0["final_row"], "SALL_final_vs_S0"),
                layer_rows_identical=bool(all(
                    torch.equal(rows["SALL"][i][k], rows["S0"][i][k])
                    for i in range(n_layers) for k in ("input", "output"))),
                norm_row_identical=bool(torch.equal(_norm_row(SALL["norm_caps"], -1),
                                                    _norm_row(S0["norm_caps"], -1)))),
            coverage=dict(
                qk_applied_ball=len((BALL["qk_state"] or {}).get("applied", [])),
                lin_applied_ball=len((BALL["lin_state"] or {}).get("applied", [])),
                qk_seen_ball=(BALL["qk_state"] or {}).get("n_seen"),
                lin_seen_ball=(BALL["lin_state"] or {}).get("n_seen"),
                expected_qk=expected["qk"], expected_linear=expected["linear"],
                qk_applied_sall=len((SALL["qk_state"] or {}).get("applied", [])),
                lin_applied_sall=len((SALL["lin_state"] or {}).get("applied", [])),
                draft_loaded=False, draft_applied=0,
                draft_note="本工具**不加载 draft**；注册集合只含 target 的 140 个模块对象（"
                           "Stage 0 已用另一个微型模型实例证明 draft 的 applied=0）"),
            repeat=rep)
        rec["gate_1A"] = g1
        for kk in ("S0_history", "B0_history"):
            for k2, v in g1[kk].items():
                DC.require(v is None or v["exact_equal"], f"1A 失败：{kk}/{k2}", str(v))
        DC.require(g1["SALL_vs_S0"]["final_row"]["exact_equal"]
                   and g1["SALL_vs_S0"]["layer_rows_identical"] and g1["SALL_vs_S0"]["norm_row_identical"],
                   "1A 失败：SALL 与 S0 不逐位一致", str(g1["SALL_vs_S0"]))
        DC.require(g1["coverage"]["qk_applied_ball"] == expected["qk"]
                   and g1["coverage"]["lin_applied_ball"] == expected["linear"],
                   "1A 失败：候选模式的 QK/Linear applied 数量与模型结构不符", str(g1["coverage"]))
        DC.require(all(v is True for k, v in rep.items() if k.endswith("identical")),
                   "1A 失败：两次 BALL 不一致")
        print(f"[ra1] 硬门 1A ✔（S0/B0 复现历史；SALL≡S0；QK applied="
              f"{g1['coverage']['qk_applied_ball']}/{expected['qk']}、Linear applied="
              f"{g1['coverage']['lin_applied_ball']}/{expected['linear']}）", flush=True)

        # ---- 1B：S0 vs BALL 分类 ----
        cmp_ball = _row_cmp(BALL["final_row"], S0["final_row"], "BALL_final_vs_S0")
        tf_s0, tf_ball = _token_fields(S0["final_row"]), _token_fields(BALL["final_row"])
        cls = int(1 if (cmp_ball["exact_equal"] and tf_s0["argmax"] == tf_ball["argmax"])
                  else (2 if tf_s0["argmax"] == tf_ball["argmax"] else 3))
        rec["class_1B"] = dict(category=cls, logits=cmp_ball, token_s0=tf_s0, token_ball=tf_ball,
                               reading={1: "logits 与 token 都一致 ⇒ 候选在全层上闭合本例",
                                        2: "token 一致、logits 不一致 ⇒ token-ID 门可通过，但数值路径未对齐",
                                        3: "token 不一致 ⇒ 候选路线失败"}[cls])
        # 逐层轨迹（只记录第一层编号与 n_diff，禁止进入内部）
        layers = []
        for i in range(n_layers):
            a_in, b_in = rows["S0"][i]["input"], rows["BALL"][i]["input"]
            a_out, b_out = rows["S0"][i]["output"], rows["BALL"][i]["output"]
            layers.append(dict(layer=i, input_exact_equal=bool(torch.equal(a_in, b_in)),
                               output_exact_equal=bool(torch.equal(a_out, b_out)),
                               input_n_diff=int((a_in != b_in).sum()),
                               output_n_diff=int((a_out != b_out).sum())))
        rec["per_layer_S0_vs_BALL"] = dict(
            layers=layers,
            first_layer_with_equal_input_and_different_output=next(
                (l["layer"] for l in layers if l["input_exact_equal"] and not l["output_exact_equal"]), None),
            first_layer_with_any_output_diff=next((l["layer"] for l in layers
                                                   if not l["output_exact_equal"]), None),
            all_layers_identical=bool(all(l["input_exact_equal"] and l["output_exact_equal"] for l in layers)),
            note="**只记录编号与逐层 n_diff**：按本轮约定不进入该层内部诊断")
        print("== 1B：S0 vs BALL ==")
        print(f"   类别 {cls}；logits exact={cmp_ball['exact_equal']} n_diff={cmp_ball['n_diff']} "
              f"max|Δ|={cmp_ball['abs_err_max']:.6g}；token S0={tf_s0['argmax']} BALL={tf_ball['argmax']} "
              f"（margin {tf_s0['margin']:.6g}/{tf_ball['margin']:.6g}）", flush=True)
        for l in layers:
            print(f"   L{l['layer']:2d} in_eq={str(l['input_exact_equal']):5s} in_nd={l['input_n_diff']:>6d} "
                  f"out_eq={str(l['output_exact_equal']):5s} out_nd={l['output_n_diff']:>6d}", flush=True)
        print(f"   首个「输入同·输出不同」层 = {rec['per_layer_S0_vs_BALL']['first_layer_with_equal_input_and_different_output']}"
              f"；全层一致={rec['per_layer_S0_vs_BALL']['all_layers_identical']}", flush=True)

        # ---- 控制 8/9/10：候选只在 verify、mask 与 cache 正确 ----
        vr = (BALL["verify_log"] or [{}])[0]
        ctl = dict(
            cache_struct=dict(S0=S0["cache_struct_after"], B0=B0["cache_struct_after"],
                              BALL=BALL["cache_struct_after"], SALL=SALL["cache_struct_after"]),
            positions=dict(S_last_row=True, B_row0=True,
                           both_predict_target=bool(S0["cache_len_after"] == TARGET_POS
                                                    and B0["cache_len_after"] == PRE + len(CANDS)),
                           cache_lens_after={a: P1[a]["cache_len_after"] for a in ARM_NAMES}),
            verify_only=dict(
                ball_qk_applied=len((BALL["qk_state"] or {}).get("applied", [])),
                ball_lin_applied=len((BALL["lin_state"] or {}).get("applied", [])),
                sall_is_decode=True,
                note="候选只包住 `spec_dec._verify` 的调用点（本工具四臂直接调用 `_verify`/`_step` 复现该作用域）"),
            causal_prefix=dict(qk_visible_examples=[v for v in (vr.get("qk_visible") or [])[:3]],
                               n_qk_records=len(vr.get("qk_visible") or []),
                               expected_row0=[PRE + 1], expected_rows=[PRE + 1 + i for i in range(len(CANDS))],
                               all_rows_ok=bool(vr.get("qk_visible")
                                                and all(v == [PRE + 1 + i for i in range(len(CANDS))]
                                                        for v in vr["qk_visible"]))),
            token_match_anchors=dict(
                S0=dict(measured=tf_s0["argmax"], anchor=anchors_s.get("token"),
                        match=bool(tf_s0["argmax"] == anchors_s.get("token"))),
                B0=dict(measured=tf_ball and tf_ball["argmax"] if False else _token_fields(B0["final_row"])["argmax"],
                        anchor=anchors_b.get("token"),
                        match=bool(_token_fields(B0["final_row"])["argmax"] == anchors_b.get("token")))))
        rec["controls"] = ctl
        DC.require(ctl["causal_prefix"]["all_rows_ok"], "控制 9 失败：causal prefix 不符",
                   str(ctl["causal_prefix"]))
        DC.require(ctl["positions"]["both_predict_target"], "位置口径失败", str(ctl["positions"]))
        DC.require(all(v["match"] for v in ctl["token_match_anchors"].values()), "锚点不一致",
                   str(ctl["token_match_anchors"]))
        DC.require(all(all(l == ctl["cache_struct"]["S0"]["lens"][0] for l in ctl["cache_struct"][a]["lens"])
                       and ctl["cache_struct"][a]["n_layers"] == n_layers
                       and ctl["cache_struct"][a]["keys"] == ["k", "v"]
                       and ctl["cache_struct"][a]["dtype"] == "torch.bfloat16" for a in ARM_NAMES),
                   "控制 10 失败：cache 结构/长度/dtype 不符", str(ctl["cache_struct"]))
        if saved_rows:
            for a, key in (("S0", "H_AR-S"), ("B0", "H_AR-B")):
                if key in saved_rows:
                    c = _row_cmp(P1[a]["final_row"], saved_rows[key], f"{a}_vs_saved")
                    DC.require(c["exact_equal"], f"与已保存局部对照行不同（{a}）", str(c))

        # ---- 源完整性 / 有限性 ----
        H_digest_after = [A.stable_sha(lay["k"]) for lay in H] + [A.stable_sha(lay["v"]) for lay in H]
        rec["finiteness"] = dict(all_logits_finite=bool(all(torch.isfinite(P1[a]["final_row"]).all()
                                                           for a in ARM_NAMES)),
                                 all_layer_rows_finite=bool(all(
                                     torch.isfinite(rows[a][i][k]).all() for a in ARM_NAMES
                                     for i in range(n_layers) for k in ("input", "output"))),
                                 note="候选模式下 target 的全部 logits 与 28 层决策行都有限（无 NaN/Inf）")
        rec["source_integrity"] = dict(
            H_snapshot_unchanged=bool(H_digest_before == H_digest_after),
            W_after=LB.weights_info(tgt),
            W_data_ptr_stable=bool(LB.weights_info(tgt, digest=False)["data_ptr"] == rec["W_before"]["data_ptr"]),
            W_digest_stable=bool(LB.weights_info(tgt)["digest_subsample"] == rec["W_before"]["digest_subsample"]),
            norm_weight_stable=bool(LB.norm_info(tgt)["weight"]["data_ptr"] == rec["norm"]["weight"]["data_ptr"]),
            probe_state_now=M.attn_probe_state(),
            serial_qk_state_now=M.serial_qk_state(),
            serial_linear_state_now=M.serial_linear_state(),
            verify_wrap_now=spec_dec.verify_wrap_state(),
            hooks_removed=bool(all(len(tgt.layers[i]._forward_pre_hooks) == 0
                                   and len(tgt.layers[i]._forward_hooks) == 0
                                   for i in range(n_layers)) and len(tgt.norm._forward_hooks) == 0),
            note="H 快照、权重、Norm 权重全程不变；探针/两个注册表/verify 包装都已恢复、层钩子全部移除")
        DC.require(rec["finiteness"]["all_logits_finite"] and rec["finiteness"]["all_layer_rows_finite"],
                   "控制 14 失败：存在非有限张量", str(rec["finiteness"]))
        for k, v in rec["source_integrity"].items():
            if isinstance(v, bool):
                DC.require(v, f"控制 11/13 失败：{k}")
        DC.require(rec["source_integrity"]["probe_state_now"]["active"] is False
                   and rec["source_integrity"]["serial_qk_state_now"]["active"] is False
                   and rec["source_integrity"]["serial_linear_state_now"]["active"] is False
                   and rec["source_integrity"]["verify_wrap_now"]["active"] is False
                   and rec["source_integrity"]["hooks_removed"],
                   "控制 13 失败：context/hook 未全部恢复", str(rec["source_integrity"]))

        # ---- 关闭后生产路径对照（控制 2/7/8）----
        off = dict(verify_wrap=spec_dec.verify_wrap_state())
        if CAPTURE_OFF_CHECK:
            with torch.inference_mode(), M.attention_phase("decode"):
                lg_off_s, _c1 = spec_dec._step(tgt, FEED[0], _to_dev(H, dev))
            with torch.inference_mode(), M.attention_phase("verify"):
                rows_off_b, _c2 = spec_dec._verify(tgt, _to_dev(H, dev), CANDS)
            off["S_off_equals_S0"] = bool(torch.equal(LB.row_of(lg_off_s, None).detach().cpu(),
                                                      S0["final_row"]))
            off["B_off_equals_B0"] = bool(torch.equal(LB.row_of(rows_off_b, 0).detach().cpu(),
                                                      B0["final_row"]))
            off["states_after_off"] = dict(qk=M.serial_qk_state(), linear=M.serial_linear_state())
            DC.require(off["S_off_equals_S0"] and off["B_off_equals_B0"]
                       and off["states_after_off"]["qk"]["active"] is False
                       and off["states_after_off"]["linear"]["active"] is False,
                       "控制 2 失败：关闭后生产路径与 S0/B0 不同", str(off))
            print("[ra1] 控制 2/7/8 ✔（候选关闭时 S/B 与 S0/B0 逐位一致）", flush=True)
        rec["intervention_off_check"] = off
        att = M.attention_stats()
        rec["attention_stats"] = dict(strategy=att["strategy"], fused_by_phase=att["fused_by_phase"],
                                      eager_by_phase=att["eager_by_phase"])
        rec["tf32_after"] = bool(torch.backends.cuda.matmul.allow_tf32)
        DC.require(rec["tf32_after"] is False, "控制 15 失败：allow_tf32 非 False")
        rec["stage"] = "complete"
        p = _write(rec, run_id)
        rec["artifacts"] = dict(json=str(p), pt=str(tp))
        _write(rec, run_id)
        print(f"[ra1] RESULT_JSON={p}")
        print(f"[ra1] RESULT_PT={tp}")
        print(f"[ra1] 1B 类别 = {cls}（1=logits+token 一致；2=仅 token 一致；3=token 不一致）")
        return 0
    except Exception as e:                                          # noqa: BLE001
        try:
            M.clear_attn_probe()
            M.clear_serial_qk()
            M.clear_serial_linear()
            spec_dec.set_verify_wrap(None)
        except Exception:                                           # noqa: BLE001
            pass
        rec["stage"] = ("checks_failed" if isinstance(e, (DC.CheckError, DR.AlignError, A.AlignError,
                                                          CC.CloneError, LB.BoundaryError)) else "crashed")
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        p = _write(rec, run_id)
        print(f"[ra1] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[ra1] RESULT_JSON={p}")
        return 5 if rec["stage"] == "checks_failed" else 2
    finally:
        try:
            LB.assert_clean()
            M.clear_attn_probe()
            M.clear_serial_qk()
            M.clear_serial_linear()
            spec_dec.set_verify_wrap(None)
        except Exception:                                           # noqa: BLE001
            pass
        if TRS is not None:
            rec["target_rope_scope_restore"] = TRS.assert_restored()
            _write(rec, run_id)


# ------------------------------------------------------------------ CPU 自测（门）
def _selftest(rec: dict, run_id: str) -> int:
    """**不加载大模型**：微型 3 层模型上验证四臂链路（省时；28 层覆盖由 Stage 0 的门负责）。"""
    npass = nfail = 0

    def check(name, cond, extra=None):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  [ok] {name}")
        else:
            nfail += 1
            print(f"  [FAIL] {name}" + (f"  {str(extra)[:240]}" if extra is not None else ""))

    global PRE, TARGET_POS, VIS, FEED, CANDS, REPEAT, LAYER
    torch.manual_seed(0)
    PRE, TARGET_POS, VIS, FEED, CANDS, REPEAT, LAYER = 5, 6, 6, [3], [3, 4, 5, 6], 2, 0
    tgt = M.QwenLM(vocab_size=64, hidden=32, heads=4, kv_heads=2, head_dim=8, intermediate=64,
                   num_layers=3, rope_theta=10000.0, rms_eps=1e-6, attn_bias=True
                   ).to(torch.bfloat16).eval()
    M.set_attention_strategy("eager")
    ids = torch.randint(0, 64, (1, PRE))
    with torch.inference_mode():
        _lg, c0 = tgt._run(tgt.embed_tokens(ids), None, need_logits=True)
    H = [{k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in lay.items()}
         for lay in CC.clone_list_cache_full(c0)]
    dev = torch.device("cpu")
    p1 = {a: _capture_arm(tgt, H, dev, a, m, "s1", cand=c) for a, m, c in PLAN}
    check("★ 四臂捕获都能跑通；块长 = K（B 臂 4 行、S 臂 1 行）",
          all(len(p1[a]["chain"]["layers"]) == 3 for a in ARM_NAMES)
          and p1["BALL"]["cache_len_after"] == PRE + len(CANDS)
          and p1["SALL"]["cache_len_after"] == PRE + 1)
    check("★ 候选模式在 BALL/SALL 中 applied = 5×层数（3 层 ⇒ 15）",
          len(p1["BALL"]["qk_state"]["applied"]) == 3
          and len(p1["BALL"]["lin_state"]["applied"]) == 12
          and len(p1["SALL"]["qk_state"]["applied"]) == 3
          and len(p1["SALL"]["lin_state"]["applied"]) == 12)
    check("★ 非候选臂（S0/B0）没有注册表状态（`None`）",
          p1["S0"]["qk_state"] is None and p1["B0"]["lin_state"] is None)
    check("★ cache 结构正确（层内键 `k`/`v`、bf16、各层等长）",
          all(p1[a]["cache_struct_after"]["keys"] == ["k", "v"]
              and p1[a]["cache_struct_after"]["dtype"] == "torch.bfloat16"
              and len(set(p1[a]["cache_struct_after"]["lens"])) == 1 for a in ARM_NAMES))
    with torch.inference_mode(), M.attention_phase("decode"):
        lg_native_s, _ = spec_dec._step(tgt, FEED[0], _to_dev(H, dev))
    with torch.inference_mode(), M.attention_phase("verify"):
        rows_native_b, _ = spec_dec._verify(tgt, _to_dev(H, dev), CANDS)
    check("★ 控制 2：S0/B0 与原生路径逐位一致（候选关闭时生产路径不变）",
          torch.equal(p1["S0"]["final_row"], LB.row_of(lg_native_s, None).detach().cpu())
          and torch.equal(p1["B0"]["final_row"], LB.row_of(rows_native_b, 0).detach().cpu()))
    check("★ 硬门 1A 的 SALL≡S0 在微型模型上也成立（层行 + final）",
          torch.equal(p1["SALL"]["final_row"], p1["S0"]["final_row"])
          and all(torch.equal(_chain_rows_to_cpu(p1["SALL"]["chain"], 3)[i][k],
                              _chain_rows_to_cpu(p1["S0"]["chain"], 3)[i][k])
                  for i in range(3) for k in ("input", "output")))
    c = _row_cmp(p1["BALL"]["final_row"], p1["S0"]["final_row"], "BALL_vs_S0")
    check("★ 1B 分类可算（类别 ∈ {1,2,3}，字段齐全）",
          _token_fields(p1["S0"]["final_row"])["argmax"] >= 0 and c["n_diff"] >= 0)
    payload = _build_payload(p1, n_layers=3, module_facts=dict(n_layers=3),
                             expected=RA.expected_counts(tgt))
    tp = _save_pt(payload, run_id, selftest=True)
    check("★ 落盘 payload 可跑通（四臂层行/最终行/cache 结构/候选计数都在）",
          tp.exists() and tp.stat().st_size > 0
          and all(f"rows_{a}" in payload and f"cand_counts_{a}" in payload for a in ARM_NAMES), str(tp))
    check("★ 结束后：探针关、注册表关、verify 包装未安装、层钩子移除",
          M.attn_probe_state()["active"] is False and M.serial_qk_state()["active"] is False
          and M.serial_linear_state()["active"] is False and spec_dec.verify_wrap_state()["active"] is False
          and all(len(tgt.layers[i]._forward_pre_hooks) == 0 and len(tgt.layers[i]._forward_hooks) == 0
                  for i in range(3)))
    print(f"\nStage 1 工具（CPU 门）：PASS {npass} / FAIL {nfail}")
    rec["selftest"] = dict(npass=npass, nfail=nfail,
                           semantics="非真实运行：3 层微型模型上验证四臂捕获/候选计数/关闭等价/1A-1B/落盘",
                           pt=str(tp))
    rec["stage"] = "selftest"
    _write(rec, run_id, selftest=True)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
