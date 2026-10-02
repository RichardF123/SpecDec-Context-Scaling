"""serial_qk_l0_f13_p8209.py — L0 的 **QK 形状因果干预闭环**（F13；只读、有界、单次）。

## 本轮唯一问题
只把 **B/verify 路径 L0** 的 QK 改成"逐 query 行以 `M=1` + **真实 causal key 前缀**计算、再拼回原 block"
（其余算子完全不变）之后：QK 第 0 行、L0 输出、后续 27 层、最终 token 是否**恢复**成 S 路径？

## Arms（同一份 `H(8208)`，各自独立克隆 cache，串行执行）
| arm | 输入 | QK |
| --- | --- | --- |
| **S0** | `[20642]`（decode，s=1） | 普通 eager（生产原式） |
| **B0** | `[20642,659,576,4531]`（verify，s=4） | 普通 eager |
| **B1** | 与 B0 **完全相同** | **只在 L0** 用逐行 `M=1` + causal 前缀（诊断分支） |
| **S1** | 与 S0 完全相同 | 在 S 路径开启同一诊断分支（s=1 ⇒ 必须与 S0 逐位一致） |

## 判据
* **H1**：`S0 QK row0` vs `B1 QK row0`（12 头、keys `0:8209`）**逐位一致** —— 否则干预无效、停止解释；
  同时验证 B1 每行在 mask 后未来列全为 `-inf`。
* **H2**：L0 输出是否恢复；未恢复则给出**第一个剩余**的「输入逐位相同、输出首次不同」算子。
* **H3**：逐层比较 S0/B1（层输入行 + 层输出行）⇒ 第一处"输入相同、输出不同"的**新独立**边界。
* **H4**：最终 logits/argmax 的恢复类别（1=完全恢复 / 2=仅 token 恢复 / 3=都没恢复）。

## 纪律
生产公式/dtype/精度/mask/RoPE/attention/cache/argmax/后端**一律不改**；诊断分支**默认关闭**且只作用于
显式注册的 **L0 `SelfAttention` 实例**（`serial_qk_for`，异常也恢复）；不做 profiler、不做 FP32、
不切后端、不跑 B6；GPU 串行；硬失败**落盘即停**、不自动重试；旧产物不覆盖；**B6 状态不变**。
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
from . import op_probe as OP
from . import sb_align as A
from . import serial_causal_qk as SQ
from . import spec_cache
from . import spec_dec
from . import g5_8k_spec_variance_probe as P5V

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"

TAG = os.environ.get("SQ_TAG", "div8209_sqk")
MODE = os.environ.get("SQ_MODE", "gpu").strip().lower()
LAYER = int(os.environ.get("SQ_LAYER", "0"))
PRE = int(os.environ.get("SQ_PRE", "8208"))
TARGET_POS = int(os.environ.get("SQ_TARGET", "8209"))
K = int(os.environ.get("SQ_K", "4"))
FEED = [int(x) for x in os.environ.get("SQ_FEED", "20642").split(",")]
CANDS = [int(x) for x in os.environ.get("SQ_CANDS", "20642,659,576,4531").split(",")]
WATCH = tuple(int(x) for x in os.environ.get("SQ_WATCH", "323,659,20642,576,4531").split(","))
REPEAT = int(os.environ.get("SQ_REPEAT", "2"))
TARGET_ROPE = os.environ.get("SQ_TARGET_ROPE", "cpu_fp32").strip().lower()
POISON_CHECK = S.env_flag("SQ_POISON_CHECK", True)
CAPTURE_OFF_CHECK = S.env_flag("SQ_CAPTURE_OFF_CHECK", True)
CAP_H = os.environ.get("SQ_CAP_H", str(OUTDIR / "divcap_ar_20261001T185503Z.pt"))
CAP_H2 = os.environ.get("SQ_CAP_H2", str(OUTDIR / "divasc_rebuild_20261001T220737Z.pt"))
L0OPS_PT = os.environ.get("SQ_L0OPS_PT", str(OUTDIR / "div8209_l0ops_20261002T172447Z.pt"))
QK2X2_JSON = os.environ.get("SQ_QK2X2_JSON",
                            str(OUTDIR / "qk2x2_8209_verdict_cpu_20261002T181152Z.json"))
SAVED_PT = os.environ.get("SQ_SAVED_PT", str(OUTDIR / "div8209_cells_20261002T064415Z.pt"))
CHAIN_PT = os.environ.get("SQ_CHAIN_PT", str(OUTDIR / "div8209_chain_20261002T164558Z.pt"))
REF_RUN_AR = os.environ.get("SQ_REF_AR", str(OUTDIR / "smoke8k_eager_20261001T185503Z.json"))
REF_RUN_RB = os.environ.get("SQ_REF_RB", str(OUTDIR / "smoke8k_eager_20261001T220737Z.json"))
KV, ARM = "base_bf16", os.environ.get("SQ_ARM", "eager")
VIS = PRE + 1                                  # 共同可见 key 数 = 8209（key 位置 0..8208）
S_ARMS = ("S0", "S1")
B_ARMS = ("B0", "B1")


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


def _row_cmp(a, b, name: str) -> dict:
    a2, b2 = a.float().reshape(-1), b.float().reshape(-1)
    DC.assert_same_shape(a2, b2, name)
    d = (a2 - b2).abs()
    return dict(name=name, exact_equal=bool(torch.equal(a.reshape(-1), b.reshape(-1))),
                n_diff=int((a.reshape(-1) != b.reshape(-1)).sum()),
                abs_err_max=float(d.max()), abs_err_mean=float(d.mean()),
                sha16_a=A.stable_sha(a), sha16_b=A.stable_sha(b), dtype=str(a.dtype),
                shape=[int(x) for x in a.shape])


def _coords(a, b, limit: int = 64) -> list:
    """逐位不同的坐标列表（原始 dtype 比较）。"""
    neq = (a != b)
    return [[int(x) for x in c] for c in torch.nonzero(neq).tolist()[:limit]]


# ------------------------------------------------------------------ 捕获
def _capture_arm(tgt, H, dev, arm: str, mode: str, tag: str, *, serial: bool = False,
                 poison: bool = False) -> dict:
    """单臂捕获：层链（28 层决策行）+ 最终 RMSNorm + L0 算子探针 + 真实前向。

    `serial=True` 时只在 **L0** 的 `self_attn` 上开启逐行 M=1 causal-prefix（诊断分支）。
    """
    cache = _to_dev(H, dev)
    sa0 = tgt.layers[LAYER].self_attn
    ctx = M.serial_qk_for(sa0, label=arm) if serial else contextlib.nullcontext(None)
    row_index = -1 if arm in S_ARMS else 0
    with ctx as sq_info, \
            LB.layer_chain_capture(tgt, row_index=row_index, tag=f"{arm}_{tag}") as chain, \
            LB.record(tgt, tag=f"{arm}_{tag}_norm") as norm_info, \
            OP.capture(tgt, LAYER, arm) as cap:
        with torch.inference_mode(), M.attention_phase(mode):
            if arm in S_ARMS:
                lg, cache_after = spec_dec._step(tgt, FEED[0], cache)
                row = LB.row_of(lg, None)
            else:
                rows_b, cache_after = spec_dec._verify(tgt, cache, CANDS)
                row = LB.row_of(rows_b, 0)
        # ---- 诊断分支的原始记录（**必须**在上下文退出前取，且此时仍在 GPU 上）----
        sq_calls = M.serial_qk_calls() if serial else []
        sq_state = M.serial_qk_state() if serial else None
        # ---- 「不读未来 key」哨兵控制：对该行**之后**的 key 灌 NaN，重算该行（只在 GPU 上比）----
        poison_res = None
        if poison and serial and sq_calls:
            q_g = cap.records["q_rope"]["tensor"]                 # 仍在 GPU（close 才搬 CPU）
            k_g = cap.records["k_visible"]["tensor"]
            k_rep_g = M.kv_repeat(k_g, tgt.layers[LAYER].self_attn.heads //
                                  tgt.layers[LAYER].self_attn.kv_heads)
            p_g = int(sq_calls[0]["p"])
            rows_rec = sq_calls[0]["meta"]["rows"]
            poison_res = []
            for i, r in enumerate(rows_rec):
                vis_i = int(r["visible"])
                kp = k_rep_g.clone()
                kp[:, :, vis_i:, :] = float("nan")
                rr = SQ.row_scores(q_g, kp, i, p=p_g, head_dim=int(tgt.head_dim))
                poison_res.append(dict(i=int(i), visible=vis_i,
                                       raw_equal=bool(torch.equal(rr["raw"], r["raw"])),
                                       scaled_equal=bool(torch.equal(rr["scaled"], r["scaled"])),
                                       finite=bool(rr["finite"])))
        probes = cap.close()
        norm_caps = LB.take()
    return dict(arm=arm, mode=mode, row_index=int(row_index), serial=bool(serial),
                layer_cap=chain, chain=chain, norm_caps=norm_caps, norm_info=norm_info,
                probes=probes, final_row=row.detach().to("cpu"),
                cache_len_after=int(spec_cache.cache_length(cache_after)),
                cache_len_before=int(chain["layers"][LAYER]["pre"]["cache_len_before"]),
                x0_row=chain["layers"][LAYER]["pre"]["a_row"].detach().cpu(),
                cos=chain["layers"][LAYER]["pre"].get("cos_row"),
                sin=chain["layers"][LAYER]["pre"].get("sin_row"),
                n_pre=int(chain["n_pre"]), n_post=int(chain["n_post"]),
                sq_calls=sq_calls, sq_state=sq_state, poison=poison_res)


def _chain_rows_to_cpu(chain: dict, n_layers: int) -> dict:
    """层链 → 每层的 `input`/`output` 决策行（一维、原始 dtype、CPU）。"""
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
    """最终 RMSNorm 的**输出**决策行（`(hidden,)`）。"""
    caps = norm_caps.get("caps") or {}
    if not caps:
        return None
    rec = caps[max(caps)]
    y = rec.get("y_out")
    if y is None:
        return None
    return (y[0, row_index] if y.dim() == 3 else y).detach().cpu()


def _cmp_all(a_nodes: dict, b_nodes: dict) -> dict:
    """按语义口径比较全部可比较节点（**S 口径 vs B 口径**）。"""
    out = {}
    for name, (who, kind) in OP.SPECS.items():
        if kind == "shape_only":
            continue
        ta = a_nodes["probes"]["records"].get(name)
        tb = b_nodes["probes"]["records"].get(name)
        DC.require(ta is not None and tb is not None, f"探针缺少节点：{name}",
                   f"a={ta is not None} b={tb is not None}")
        out[name] = OP.cmp_node(name, kind, ta["tensor"], tb["tensor"], vis=VIS)
    return out


def _cmp_all_B(a_nodes: dict, b_nodes: dict) -> dict:
    """**B 口径 vs B 口径**（两侧都是 s=4）：都取第 0 行 + 共同可见范围。

    ⚠️ 不能复用 `OP.cmp_node`：它把第一个参数当 **S 侧**（`[0,-1]`），
    用它比 B0/B1 会拿"末位行"对"第 0 行" ⇒ 口径错。这里显式写成 B 口径。
    """
    def sl(kind, t):
        if kind == "row":
            return t[0, 0]
        if kind == "heads":
            return t[0, :, 0, :]
        if kind == "kv":
            return t[0, :, :VIS, :]
        if kind == "scores":
            return t[0, :, 0, :VIS]
        raise OP.ProbeError(f"未知口径：{kind}")

    out = {}
    for name, (who, kind) in OP.SPECS.items():
        if kind == "shape_only":
            continue
        ta = a_nodes["probes"]["records"].get(name)
        tb = b_nodes["probes"]["records"].get(name)
        DC.require(ta is not None and tb is not None, f"探针缺少节点：{name}")
        a, b = sl(kind, ta["tensor"]), sl(kind, tb["tensor"])
        DC.assert_same_shape(a, b, f"B口径[{name}]")
        out[name] = dict(node=name, kind=kind, slice_rule="B 口径（两侧都取第 0 行/共同可见范围）",
                         shape_a=[int(x) for x in ta["tensor"].shape],
                         shape_b=[int(x) for x in tb["tensor"].shape],
                         exact_equal=bool(torch.equal(a, b)), n_diff=int((a != b).sum()),
                         max_abs=float((a.float() - b.float()).abs().max()),
                         sha16_a=A.stable_sha(a), sha16_b=A.stable_sha(b))
    return out


def _tensors(records: dict) -> dict:
    """探针记录 → 纯张量字典（供分类函数使用）。"""
    return {k: v["tensor"] for k, v in records.items() if not v.get("shape_only")}


def _serial_to_cpu(sq_calls: list) -> list:
    """诊断分支的调用记录 → CPU payload（保留每行 raw/scaled 与布局）。"""
    out = []
    for c in sq_calls:
        meta = c["meta"]
        out.append(dict(module_id=int(id(c["module"])), s=int(c["s"]), T=int(c["T"]), p=int(c["p"]),
                        visible=list(meta["visible"]), expected_visible=list(meta["expected_visible"]),
                        padded=list(meta["padded"]), placeholder=float(meta["placeholder"]),
                        contiguous_rows=bool(meta["contiguous_rows"]),
                        scores=(c["scaled"].detach().cpu() if "scaled" in c else None),
                        rows=[dict(i=int(r["i"]), visible=int(r["visible"]),
                                   raw=r["raw"].detach().cpu(), scaled=r["scaled"].detach().cpu(),
                                   layout_q=r["layout_q"], layout_k=r["layout_k"],
                                   layout_q_used=r["layout_q_used"], layout_k_used=r["layout_k_used"],
                                   finite=bool(r["finite"])) for r in meta["rows"]]))
    return out


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="serial_qk_l0_f13_p8209", family="F13", kv=KV, arm=ARM,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               question="只把 B/verify 路径 L0 的 QK 改成逐 query 行 M=1 + 真实 causal 前缀后，"
                        "QK 第 0 行 / L0 输出 / 后续 27 层 / 最终 token 是否恢复成 S 路径？",
               semantics="在 L0 的 SelfAttention 上开启**默认关闭**的诊断分支（逐行 M=1 causal-prefix QK），"
                         "其余算子/精度/后端一律不变；四臂 S0/B0/B1/S1 串行、同一份 H(8208)。",
               config=dict(tag=TAG, layer=LAYER, pre=PRE, target_pos=TARGET_POS, k=K, feed=FEED,
                           cands=CANDS, watch=list(WATCH), repeat=REPEAT, vis=VIS,
                           target_rope=TARGET_ROPE, poison_check=POISON_CHECK,
                           capture_off_check=CAPTURE_OFF_CHECK, cap_h=str(CAP_H), cap_h2=str(CAP_H2),
                           l0ops_pt=str(L0OPS_PT), qk2x2_json=str(QK2X2_JSON), saved_pt=str(SAVED_PT),
                           chain_pt=str(CHAIN_PT), ref_ar=str(REF_RUN_AR), ref_rb=str(REF_RUN_RB),
                           in_scope="只做 L0 QK 的因果干预诊断；不修 q_proj、不改 AV/o_proj/MLP、"
                                    "不做 profiler/FP32、不切后端、不跑 B6/acceptance/pilot/final"))
    print(f"[sqk] run_id={run_id} layer={LAYER} pre={PRE} target={TARGET_POS} vis={VIS} "
          f"repeat={REPEAT} rope={TARGET_ROPE}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    if MODE == "selftest":
        return _selftest(rec, run_id)
    ok, why = P5V.commit_guard()
    rec["commit_guard"] = dict(ok=bool(ok), why=why)
    print(f"[sqk] commit: {why}")
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


def _build_payload(arms: dict, *, n_layers: int) -> dict:
    """落盘 payload（`_run` 与 CPU 自测**同一代码路径**）。"""
    pay: dict = dict(meta=dict(vis=VIS, layer=LAYER, n_layers=n_layers, arms=sorted(arms)))
    for a, r in arms.items():
        pay[f"nodes_{a}"] = {k: v["tensor"].detach().cpu()
                             for k, v in r["probes"]["records"].items() if not v.get("shape_only")}
        pay[f"layouts_{a}"] = {k: v["layout"] for k, v in r["probes"]["records"].items()}
        pay[f"rows_{a}"] = _chain_rows_to_cpu(r["chain"], n_layers)
        pay[f"norm_row_{a}"] = _norm_row(r["norm_caps"], r["row_index"])
        pay[f"final_row_{a}"] = r["final_row"].detach().cpu()
        pay[f"cos_row_{a}"] = r["cos"]
        pay[f"sin_row_{a}"] = r["sin"]
        pay[f"serial_{a}"] = _serial_to_cpu(r["sq_calls"]) if r["serial"] else None
        pay[f"poison_{a}"] = r["poison"]
    pay["note"] = ("`nodes_*` = L0 探针捕获的完整张量（原始 dtype）；`rows_*` = 28 层的输入/输出决策行；"
                   "`norm_row_*` = 最终 RMSNorm 输出决策行；`final_row_*` = 完整词表 logits 决策行；"
                   "`serial_*` = 诊断分支的逐行 raw/scaled 与布局；`poison_*` = 不读未来 key 的哨兵结果。"
                   "fp32 的 softmax 概率保持 fp32。")
    return pay


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
        rec["code_facts"] = LB.code_facts(tgt)
        rec["norm"] = LB.norm_info(tgt)
        rec["layer"] = LB.layer_info(tgt, LAYER)
        rec["probe_state_before"] = M.attn_probe_state()
        rec["serial_qk_state_before"] = M.serial_qk_state()
        DC.require(rec["probe_state_before"]["active"] is False, "进入实验前探针必须关闭",
                   str(rec["probe_state_before"]))
        DC.require(rec["serial_qk_state_before"]["active"] is False,
                   "进入实验前诊断分支必须关闭（默认关闭）", str(rec["serial_qk_state_before"]))
        DC.require(n_layers >= 2 and LAYER == 0, "本轮只做 L0", f"n_layers={n_layers} LAYER={LAYER}")
        rec["W_before"] = LB.weights_info(tgt)
        print(f"[sqk] 层 {LAYER}/{n_layers - 1}；诊断分支状态={rec['serial_qk_state_before']}", flush=True)
        if TARGET_ROPE == "cpu_fp32":
            from . import target_rope_scope as _TRS
            TRS = _TRS
            rec["target_rope_scope"] = TRS.install(tgt, theta=float(tgt.rope_theta),
                                                   head_dim=int(tgt.head_dim))

        # ---------------- H(8208) ----------------
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
        print(f"[sqk] H：{Path(CAP_H).name} {src_len} → {PRE}（纯切片={tinfo['is_pure_slice']}）", flush=True)

        # ---------------- 四臂 × REPEAT ----------------
        passes = []
        plan = [("S0", "decode", False), ("B0", "verify", False),
                ("B1", "verify", True), ("S1", "decode", True)]
        for it in range(REPEAT):
            arms = {}
            for arm, mode, serial in plan:
                arms[arm] = _capture_arm(tgt, H, dev, arm, mode, f"p{it + 1}", serial=serial,
                                         poison=(POISON_CHECK and arm == "B1"))
                print(f"[sqk] pass {it + 1}/{REPEAT} arm {arm} ✔ "
                      f"（探针调用={arms[arm]['probes']['n_calls']} 保留={arms[arm]['probes']['n_kept']}；"
                      f"serial applied={0 if not serial else len(arms[arm]['sq_calls'])}）", flush=True)
            passes.append(arms)
        P1, P2 = passes[0], (passes[1] if len(passes) > 1 else None)
        DC.require(M.attn_probe_state()["active"] is False, "捕获后探针仍在位")
        DC.require(M.serial_qk_state()["active"] is False, "捕获后诊断分支仍开启")
        rec["serial_qk_state_after"] = M.serial_qk_state()

        # ---------------- 控制 12：两次独立 B1（以及全部四臂）逐位一致 ----------------
        rep = dict(n_passes=len(passes))
        if P2 is not None:
            for a in ("S0", "B0", "B1", "S1"):
                rep[f"{a}_nodes_identical"] = bool(all(
                    torch.equal(P1[a]["probes"]["records"][k]["tensor"],
                                P2[a]["probes"]["records"][k]["tensor"])
                    for k, v in P1[a]["probes"]["records"].items() if not v.get("shape_only")))
                rep[f"{a}_final_row_identical"] = bool(torch.equal(P1[a]["final_row"], P2[a]["final_row"]))
                rep[f"{a}_serial_raw_identical"] = bool(
                    (not P1[a]["serial"]) or all(
                        torch.equal(x["raw"], y["raw"]) and torch.equal(x["scaled"], y["scaled"])
                        for c1, c2 in zip(_serial_to_cpu(P1[a]["sq_calls"]), _serial_to_cpu(P2[a]["sq_calls"]))
                        for x, y in zip(c1["rows"], c2["rows"])))
        rec["repeat_consistency"] = rep
        DC.require(all(v is True for k, v in rep.items() if k.endswith("identical")),
                   "两次独立捕获不一致（含控制 12）", str(rep))
        print("[sqk] 重复一致 ✔（两次独立 run 的四臂节点/最终行/诊断逐行结果都逐位相同）", flush=True)

        S0, B0, B1, S1 = (P1["S0"], P1["B0"], P1["B1"], P1["S1"])
        row_s0, row_b0, row_b1 = S0["final_row"], B0["final_row"], B1["final_row"]

        # ---------------- 参考产物 ----------------
        l0ops = torch.load(L0OPS_PT, map_location="cpu") if Path(L0OPS_PT).exists() else {}
        qk2x2 = json.loads(Path(QK2X2_JSON).read_text(encoding="utf-8")) if Path(QK2X2_JSON).exists() else {}
        saved_rows = {}
        if Path(SAVED_PT).exists():
            saved_rows = {k: v for k, v in torch.load(SAVED_PT, map_location="cpu").items()
                          if torch.is_tensor(v)}
        chain_prev = torch.load(CHAIN_PT, map_location="cpu") if Path(CHAIN_PT).exists() else {}
        anchors_s = C._anchor_ar(REF_RUN_AR, TARGET_POS)
        anchors_b = C._anchor_specdec(REF_RUN_RB, PRE, TARGET_POS - PRE, TARGET_POS)

        # ---------------- 边界比较 ----------------
        qk_s0 = S0["probes"]["records"]["qk_scores_pre_mask"]["tensor"]
        qk_b0 = B0["probes"]["records"]["qk_scores_pre_mask"]["tensor"]
        qk_b1 = B1["probes"]["records"]["qk_scores_pre_mask"]["tensor"]
        post_b0 = B0["probes"]["records"]["qk_scores_post_mask"]["tensor"]
        post_b1 = B1["probes"]["records"]["qk_scores_post_mask"]["tensor"]
        post_s0 = S0["probes"]["records"]["qk_scores_post_mask"]["tensor"]
        h1 = SQ.classify_h1(qk_s0, qk_b1, vis=VIS)
        cmp_s0b0 = _cmp_all(S0, B0)
        cmp_s0b1 = _cmp_all(S0, B1)
        cmp_b0b1 = _cmp_all_B(B0, B1)
        # 虚拟节点 `cache_pre`：两条路径用的是**同一份 H 的同一层缓存**（同一对象的独立克隆）
        cache_pre = dict(
            exact_equal=True, n_diff=0,
            cache_len_before=dict(S0=S0["cache_len_before"], B0=B0["cache_len_before"],
                                  B1=B1["cache_len_before"], S1=S1["cache_len_before"]),
            note="两条路径都从同一份 H(8208) 的 L0 缓存独立克隆出发（长度 8208，逐层摘要由 `source_integrity` 校验）"
                 "⇒ 该虚拟节点按构造逐位相同")
        cls_s0b0 = OP.classify(OP.CHAIN, dict(cmp_s0b0, cache_pre=cache_pre), OP.CHAIN_NODE_OUT)
        cls_s0b1 = OP.classify(OP.CHAIN, dict(cmp_s0b1, cache_pre=cache_pre), OP.CHAIN_NODE_OUT)
        h2 = SQ.classify_h2(_tensors(S0["probes"]["records"]), _tensors(B1["probes"]["records"]),
                            OP.CHAIN, OP.CHAIN_NODE_OUT, vis=VIS, first_div=cls_s0b1)
        rows_s0 = _chain_rows_to_cpu(S0["chain"], n_layers)
        rows_b1 = _chain_rows_to_cpu(B1["chain"], n_layers)
        rows_b0 = _chain_rows_to_cpu(B0["chain"], n_layers)
        h3 = SQ.classify_h3(rows_s0, rows_b1, n_layers=n_layers)
        h3_b0 = SQ.classify_h3(rows_s0, rows_b0, n_layers=n_layers)
        h4 = SQ.classify_h4(row_s0, row_b1)
        h4_b0 = SQ.classify_h4(row_s0, row_b0)
        norm_s0 = _norm_row(S0["norm_caps"], -1)
        norm_b1 = _norm_row(B1["norm_caps"], 0)

        rec["H1"] = h1
        rec["H2"] = h2
        rec["H3"] = h3
        rec["H4"] = h4
        rec["reference_results"] = dict(H1=h1, H2=h2, H3=h3, H4=h4,
                                        H3_B0_vs_S0=h3_b0, H4_B0_vs_S0=h4_b0,
                                        note="B0 与 S0 的对照（干预前），用于说明 H3/H4 的差异不是本轮干预引入的")
        rec["node_compare"] = dict(S0_vs_B0=cmp_s0b0, S0_vs_B1=cmp_s0b1, B0_vs_B1=cmp_b0b1)
        rec["first_divergence_S0_vs_B0"] = cls_s0b0
        rec["first_divergence_S0_vs_B1"] = cls_s0b1

        # ---------------- 必需控制 ----------------
        qk_coords_expected = None
        if qk2x2:
            att = qk2x2.get("attribution") or {}
            qk_coords_expected = att.get("observed_A_vs_D", {}).get("coords")
        controls: dict = {}
        controls["c1_S0_reproduces_history"] = dict(
            final_row=(_row_cmp(row_s0, l0ops["final_row_S"], "S0_final_vs_history")
                       if "final_row_S" in l0ops else None),
            L0_output=(_row_cmp(S0["probes"]["records"]["layer_output"]["tensor"][0, -1],
                                l0ops["L0_output_row_S"], "S0_L0out_vs_history")
                       if "L0_output_row_S" in l0ops else None),
            qk_scores=(_row_cmp(qk_s0, l0ops["raw_S"]["qk_scores_pre_mask"], "S0_qk_vs_history")
                       if l0ops else None),
            x0=(_row_cmp(S0["probes"]["records"]["x0"]["tensor"][0, -1], l0ops["x0_row_S"],
                         "S0_x0_vs_history") if "x0_row_S" in l0ops else None))
        controls["c2_B0_reproduces_history"] = dict(
            final_row=(_row_cmp(row_b0, l0ops["final_row_B"], "B0_final_vs_history")
                       if "final_row_B" in l0ops else None),
            L0_output=(_row_cmp(B0["probes"]["records"]["layer_output"]["tensor"][0, 0],
                                l0ops["L0_output_row_B"], "B0_L0out_vs_history")
                       if "L0_output_row_B" in l0ops else None),
            qk_scores=(_row_cmp(qk_b0, l0ops["raw_B"]["qk_scores_pre_mask"], "B0_qk_vs_history")
                       if l0ops else None),
            x0=(_row_cmp(B0["probes"]["records"]["x0"]["tensor"][0, 0], l0ops["x0_row_B"],
                         "B0_x0_vs_history") if "x0_row_B" in l0ops else None))
        qk_b0_vs_s0 = SQ.classify_h1(qk_s0, qk_b0, vis=VIS)      # 干预前的 3 坐标差异
        controls["c3_B0_qk_3coords_reproduced"] = dict(
            compare=qk_b0_vs_s0, expected_n_diff=3, expected_max_abs=4.0,
            expected_coords=qk_coords_expected,
            coords_match=bool(qk_coords_expected is None
                              or _coords(qk_s0[0, :, 0, :VIS], qk_b0[0, :, 0, :VIS]) == qk_coords_expected))
        controls["c4_S1_equals_S0"] = dict(
            nodes_identical=bool(all(torch.equal(S1["probes"]["records"][k]["tensor"],
                                                 S0["probes"]["records"][k]["tensor"])
                                     for k, v in S0["probes"]["records"].items()
                                     if not v.get("shape_only"))),
            final_row_identical=bool(torch.equal(S1["final_row"], S0["final_row"])),
            rows_identical=bool(all(torch.equal(_chain_rows_to_cpu(S1["chain"], n_layers)[i][k],
                                                rows_s0[i][k])
                                    for i in range(n_layers) for k in ("input", "output"))),
            serial_applied=int(len(S1["sq_calls"])))
        controls["c5_H1_qk_row0_restored"] = h1
        controls["c6_B1_per_row_visible_is_p_plus_i_plus_1"] = dict(
            calls=[dict(module_id=c["module_id"], s=c["s"], T=c["T"], p=c["p"],
                        visible=c["visible"], expected=c["expected_visible"],
                        ok=bool(c["visible"] == c["expected_visible"]))
                   for c in _serial_to_cpu(B1["sq_calls"])])
        inf_b1 = dict(row0=int(torch.isinf(post_b1[0, :, 0, VIS:]).sum()),
                      total=int(torch.isinf(post_b1).sum()),
                      expected_total=int(sum(12 * (int(post_b1.shape[3]) - (PRE + i + 1))
                                             for i in range(len(CANDS)))),
                      visible_region_finite=bool(torch.isfinite(post_b1[0, :, :, :VIS]).all()),
                      b0_total_inf=int(torch.isinf(post_b0).sum()),
                      s0_inf=int(torch.isinf(post_s0).sum()))
        controls["c7_B1_future_is_minus_inf_after_mask"] = inf_b1
        controls["c8_B1_rows_do_not_read_future_keys"] = dict(
            poison=B1["poison"],
            all_rows_unaffected=bool(B1["poison"] and all(r["raw_equal"] and r["scaled_equal"]
                                                          and r["finite"] for r in B1["poison"])))
        st = B1["sq_state"] or {}
        applied_ids = [a["module_id"] for a in st.get("applied", [])]
        seen_ids = list(st.get("seen_module_ids", []))
        all_ids = [int(id(tgt.layers[i].self_attn)) for i in range(n_layers)]
        controls["c9_intervention_only_L0"] = dict(
            n_seen=int(st.get("n_seen", -1)), n_applied=int(st.get("n_applied", -1)),
            n_layers=int(n_layers),
            applied_module_ids=applied_ids,
            L0_module_id=int(id(tgt.layers[LAYER].self_attn)),
            seen_module_ids=seen_ids,
            # `seen` = 进入分支但**未注册**的模块（即 L0 之外的全部层）
            seen_is_all_other_layers=bool(seen_ids == sorted(i for i in all_ids if i != int(id(tgt.layers[LAYER].self_attn)))),
            seen_plus_applied_covers_all_layers=bool(sorted(seen_ids + applied_ids) == sorted(all_ids)),
            applied_is_L0=bool(applied_ids == [int(id(tgt.layers[LAYER].self_attn))]))
        controls["c10_other_layers_use_block_qk"] = dict(
            n_applied=int(st.get("n_applied", -1)),
            n_layers=int(n_layers),
            n_seen=int(st.get("n_seen", -1)),
            seen_plus_applied_covers_all_layers=controls["c9_intervention_only_L0"]["seen_plus_applied_covers_all_layers"],
            note="只有**注册模块**（L0）能被替换：`applied` 恰好 1 次且 id = L0 的 self_attn；"
                 "其余层进入分支后只执行 `seen.append(id(self))` 这一条记录语句，"
                 "随后走的仍是**原样的 block QK 表达式** ⇒ 结构性 + 实测记录双重证据")
        controls["c11_off_means_production_unchanged"] = dict(
            state_before=rec["serial_qk_state_before"], state_after=rec["serial_qk_state_after"],
            S0_nodes_match_history=bool(controls["c1_S0_reproduces_history"]["qk_scores"] is None
                                        or controls["c1_S0_reproduces_history"]["qk_scores"]["exact_equal"]),
            B0_nodes_match_history=bool(controls["c2_B0_reproduces_history"]["qk_scores"] is None
                                        or controls["c2_B0_reproduces_history"]["qk_scores"]["exact_equal"]),
            n_applied_when_off=0)
        controls["c12_two_B1_repeats_identical"] = {k: v for k, v in rep.items() if k.startswith("B1")}
        controls["c13_cache_weights_inputs_unmodified"] = None      # §源完整性
        controls["c14_hooks_and_context_restored"] = None           # §源完整性
        controls["c15_finiteness"] = None                           # §有限性
        controls["c16_allow_tf32_false"] = dict(before=bool(torch.backends.cuda.matmul.allow_tf32),
                                                after=None)
        controls["c17_draft_never_loaded"] = dict(loaded_models=["target"], draft_loaded=False,
                                                  note="本工具只 `load_ours_bf16(target)`")
        controls["c18_git_and_artifacts"] = dict(git=S.git_meta(ROOT), recorded=True)
        controls["positions"] = dict(S_last_row=True, B_row0=True,
                                     both_predict_target=bool(S0["cache_len_after"] == TARGET_POS
                                                              and B0["cache_len_after"] == PRE + len(CANDS)),
                                     cache_lens_after=dict(S0=S0["cache_len_after"], B0=B0["cache_len_after"],
                                                           B1=B1["cache_len_after"], S1=S1["cache_len_after"]),
                                     slot_for_B_row0=DR.slot_table(PRE, len(CANDS))[1])
        controls["S0_final"] = _fields(row_s0)
        controls["B0_final"] = _fields(row_b0)
        controls["B1_final"] = _fields(row_b1)
        controls["token_matches_anchors"] = dict(
            S0=dict(measured=_fields(row_s0)["argmax_id"], anchor=anchors_s.get("token"),
                    match=bool(_fields(row_s0)["argmax_id"] == anchors_s.get("token"))),
            B0=dict(measured=_fields(row_b0)["argmax_id"], anchor=anchors_b.get("token"),
                    match=bool(_fields(row_b0)["argmax_id"] == anchors_b.get("token"))))
        controls["final_norm_row"] = dict(
            S0_vs_B1=(_row_cmp(norm_s0, norm_b1, "final_norm_S0_vs_B1")
                      if (norm_s0 is not None and norm_b1 is not None) else None))
        controls["vs_saved_local_rows"] = dict(
            path=str(SAVED_PT), present=bool(saved_rows),
            S0=(_row_cmp(row_s0, saved_rows["H_AR-S"], "S0_vs_saved") if "H_AR-S" in saved_rows else None),
            B0=(_row_cmp(row_b0, saved_rows["H_AR-B"], "B0_vs_saved") if "H_AR-B" in saved_rows else None))
        rec["controls"] = controls

        # ---- 控制断言（任一失败 ⇒ checks_failed 并停止归因）----
        for k in ("c1_S0_reproduces_history", "c2_B0_reproduces_history"):
            for kk, v in controls[k].items():
                DC.require(v is None or v["exact_equal"], f"控制失败：{k}/{kk}", str(v))
        DC.require(controls["c3_B0_qk_3coords_reproduced"]["compare"]["n_diff"] == 3
                   and controls["c3_B0_qk_3coords_reproduced"]["compare"]["max_abs"] == 4.0
                   and controls["c3_B0_qk_3coords_reproduced"]["coords_match"],
                   "控制 3 失败：B0 的 QK 3 坐标差异未逐位复现",
                   str(controls["c3_B0_qk_3coords_reproduced"]))
        DC.require(all(controls["c4_S1_equals_S0"][k] for k in ("nodes_identical", "final_row_identical",
                                                                "rows_identical")),
                   "控制 4 失败：S1 与 S0 不逐位一致", str(controls["c4_S1_equals_S0"]))
        DC.require(controls["c5_H1_qk_row0_restored"]["exact_equal"],
                   "★ H1 失败：B1 的 QK 第 0 行未逐位恢复为 S0 ⇒ 干预实现无效，停止因果解释", str(h1))
        DC.require(all(c["ok"] for c in controls["c6_B1_per_row_visible_is_p_plus_i_plus_1"]["calls"]),
                   "控制 6 失败：B1 每行可见长度不是 p+i+1", str(controls["c6_B1_per_row_visible_is_p_plus_i_plus_1"]))
        DC.require(inf_b1["row0"] == 12 * (PRE + len(CANDS) - VIS) and inf_b1["total"] == inf_b1["expected_total"]
                   and inf_b1["b0_total_inf"] == inf_b1["expected_total"] and inf_b1["s0_inf"] == 0,
                   "控制 7 失败：mask 后未来列不是全 -inf", str(inf_b1))
        DC.require(controls["c8_B1_rows_do_not_read_future_keys"]["all_rows_unaffected"],
                   "控制 8 失败：B1 某行读到了未来 key", str(controls["c8_B1_rows_do_not_read_future_keys"]))
        DC.require(controls["c9_intervention_only_L0"]["n_applied"] == 1
                   and controls["c9_intervention_only_L0"]["applied_is_L0"]
                   and controls["c9_intervention_only_L0"]["seen_is_all_other_layers"]
                   and controls["c9_intervention_only_L0"]["seen_plus_applied_covers_all_layers"],
                   "控制 9/10 失败：干预未严格限定在 L0", str(controls["c9_intervention_only_L0"]))
        DC.require(all(controls["c11_off_means_production_unchanged"][k] for k in
                       ("S0_nodes_match_history", "B0_nodes_match_history")),
                   "控制 11 失败：诊断关闭时生产路径未逐位复现历史",
                   str(controls["c11_off_means_production_unchanged"]))
        DC.require(controls["positions"]["both_predict_target"], "位置口径失败", str(controls["positions"]))
        DC.require(all(v["match"] for v in controls["token_matches_anchors"].values()),
                   "锚点不一致（停止）", str(controls["token_matches_anchors"]))
        if saved_rows:
            for k in ("S0", "B0"):
                DC.require(controls["vs_saved_local_rows"][k]["exact_equal"],
                           f"与已保存局部对照行不同（{k}）", str(controls["vs_saved_local_rows"][k]))
        print(f"[sqk] 控制 1-12 ✔（S0/B0 逐位复现历史；B0 的 QK 3 坐标差异 {qk_b0_vs_s0['coords']}；"
              f"S1≡S0；H1 exact={h1['exact_equal']}；B1 每行 visible="
              f"{[r['visible'] for r in _serial_to_cpu(B1['sq_calls'])[0]['rows']]}；"
              f"n_seen={controls['c9_intervention_only_L0']['n_seen']}"
              f"（= 其余层数）n_applied={controls['c9_intervention_only_L0']['n_applied']}）", flush=True)

        # ---------------- 先落盘（昂贵数据不因后续打印/收尾缺陷丢失）----------------
        payload = _build_payload({"S0": S0, "B0": B0, "B1": B1, "S1": S1}, n_layers=n_layers)
        payload["H_L0_cache_pre"] = dict(k=H[LAYER]["k"].detach().cpu(), v=H[LAYER]["v"].detach().cpu())
        tp = _save_pt(payload, run_id)
        rec["artifacts"] = dict(json=str(_write(rec, run_id)), pt=str(tp))
        print("== H1–H4 ==")
        print(f"  H1 QK row0：{json.dumps({k: h1[k] for k in ('exact_equal', 'n_diff', 'max_abs', 'sha16_s0', 'sha16_b1')}, ensure_ascii=False)}")
        print(f"  H2 L0 输出：class={h2['class']} exact={h2['exact_equal']} n_diff={h2['n_diff']} "
              f"max|Δ|={h2['max_abs']:.6g} 首个剩余边界="
              f"{(h2.get('first_remaining_divergence') or {}).get('first_divergence_operator')}", flush=True)
        print(f"  H3 全层：全同={h3['all_layers_identical']} 首个输出差异层={h3['first_layer_with_any_output_diff']} "
              f"首个「输入同·输出不同」层={h3['first_layer_with_equal_input_and_different_output']}", flush=True)
        print(f"  H4 logits/token：class={h4['category']} logits_exact={h4['logits_exact_equal']} "
              f"n_diff={h4['logits_n_diff']} token S0={h4['token_s0']} B1={h4['token_b1']}", flush=True)
        print("== L0 内部逐算子（S0 vs B1 / S0 vs B0）==")
        for (op, ins), out_node in zip(OP.CHAIN, OP.CHAIN_NODE_OUT):
            c1, c0 = cmp_s0b1[out_node], cmp_s0b0[out_node]
            print(f"  {op:<24s} S0vsB1 out_eq={str(bool(c1['exact_equal'])):<5s} n_diff={int(c1['n_diff']):>6d} "
                  f"| S0vsB0 n_diff={int(c0['n_diff']):>6d}", flush=True)

        # ---------------- 源完整性 / 有限性 / 收尾 ----------------
        H_digest_after = [A.stable_sha(lay["k"]) for lay in H] + [A.stable_sha(lay["v"]) for lay in H]
        nan_nodes = [k for k, v in cmp_s0b1.items() if not v.get("finite_s", True) or not v.get("finite_b", True)]
        rec["finiteness"] = dict(nodes_with_nonfinite_slice=nan_nodes,
                                 logits_finite=bool(torch.isfinite(row_s0).all() and torch.isfinite(row_b1).all()),
                                 expected_inf=dict(b1_total=int(torch.isinf(post_b1).sum()),
                                                   b0_total=int(torch.isinf(post_b0).sum()),
                                                   s0_total=int(torch.isinf(post_s0).sum()),
                                                   note="唯一允许的非有限元素 = B 路径 mask 后的 -inf"),
                                 note="NaN 未出现；被掩 -inf 是引擎设计行为")
        DC.require(not nan_nodes and rec["finiteness"]["logits_finite"],
                   "存在非有限比较切片或 logits", str(rec["finiteness"]))
        rec["source_integrity"] = dict(
            H_snapshot_unchanged=bool(H_digest_before == H_digest_after),
            W_after=LB.weights_info(tgt),
            W_data_ptr_stable=bool(LB.weights_info(tgt, digest=False)["data_ptr"] == rec["W_before"]["data_ptr"]),
            W_digest_stable=bool(LB.weights_info(tgt)["digest_subsample"] == rec["W_before"]["digest_subsample"]),
            norm_weight_stable=bool(LB.norm_info(tgt)["weight"]["data_ptr"] == rec["norm"]["weight"]["data_ptr"]),
            probe_state_now=M.attn_probe_state(),
            serial_qk_state_now=M.serial_qk_state(),
            hooks_removed=bool(all(len(tgt.layers[i]._forward_pre_hooks) == 0
                                   and len(tgt.layers[i]._forward_hooks) == 0
                                   for i in range(n_layers)) and len(tgt.norm._forward_hooks) == 0),
            note="H 快照（逐层 k/v 摘要）、权重、Norm 权重在全程不变；探针与诊断上下文都已关闭、钩子全部移除")
        controls["c13_cache_weights_inputs_unmodified"] = dict(
            H_snapshot_unchanged=rec["source_integrity"]["H_snapshot_unchanged"],
            W_data_ptr_stable=rec["source_integrity"]["W_data_ptr_stable"],
            W_digest_stable=rec["source_integrity"]["W_digest_stable"],
            norm_weight_stable=rec["source_integrity"]["norm_weight_stable"],
            L0_cache_pre_len=int(H[LAYER]["k"].shape[2]),
            cache_pre_S0_vs_B0_exact=bool(torch.equal(S0["probes"]["records"]["k_visible"]["tensor"][:, :, :VIS],
                                                      B0["probes"]["records"]["k_visible"]["tensor"][:, :, :VIS])))
        controls["c14_hooks_and_context_restored"] = dict(
            probe_active=rec["source_integrity"]["probe_state_now"]["active"],
            serial_qk_active=rec["source_integrity"]["serial_qk_state_now"]["active"],
            hooks_removed=rec["source_integrity"]["hooks_removed"])
        controls["c15_finiteness"] = rec["finiteness"]
        controls["c16_allow_tf32_false"] = dict(before=False, after=bool(torch.backends.cuda.matmul.allow_tf32))
        for k, v in controls["c13_cache_weights_inputs_unmodified"].items():
            if isinstance(v, bool):
                DC.require(v, f"控制 13 失败：{k}")
        DC.require(controls["c13_cache_weights_inputs_unmodified"]["cache_pre_S0_vs_B0_exact"],
                   "控制 13 失败：L0 轮前缓存 S/B 不同")
        DC.require(controls["c14_hooks_and_context_restored"]["probe_active"] is False
                   and controls["c14_hooks_and_context_restored"]["serial_qk_active"] is False
                   and controls["c14_hooks_and_context_restored"]["hooks_removed"],
                   "控制 14 失败：钩子/上下文未恢复", str(controls["c14_hooks_and_context_restored"]))
        DC.require(controls["c16_allow_tf32_false"]["after"] is False, "控制 16 失败：allow_tf32 非 False")

        # ---- 诊断关闭后生产路径对照（额外一条：关掉上下文再跑 S/B，与 S0/B0 逐位一致）----
        off_checks: dict = {}
        if CAPTURE_OFF_CHECK:
            with torch.inference_mode(), M.attention_phase("decode"):
                lg_off_s, _c1 = spec_dec._step(tgt, FEED[0], _to_dev(H, dev))
            with torch.inference_mode(), M.attention_phase("verify"):
                rows_off_b, _c2 = spec_dec._verify(tgt, _to_dev(H, dev), CANDS)
            off_checks["S_off_equals_S0"] = bool(torch.equal(LB.row_of(lg_off_s, None).detach().cpu(), row_s0))
            off_checks["B_off_equals_B0"] = bool(torch.equal(LB.row_of(rows_off_b, 0).detach().cpu(), row_b0))
            off_checks["serial_qk_state_after_off"] = M.serial_qk_state()
            DC.require(off_checks["S_off_equals_S0"] and off_checks["B_off_equals_B0"]
                       and off_checks["serial_qk_state_after_off"]["active"] is False,
                       "干预关闭后生产路径与 S0/B0 不同", str(off_checks))
            print("[sqk] 干预关闭后生产路径 ✔（S/B 都与 S0/B0 逐位一致）", flush=True)
        rec["intervention_off_check"] = off_checks

        att = M.attention_stats()
        DC.require(not any(x > 0 for x in att["fused_by_phase"].values()),
                   "本配置要求零融合调用", str(att["fused_by_phase"]))
        rec["attention_stats"] = dict(strategy=att["strategy"], fused_by_phase=att["fused_by_phase"],
                                      eager_by_phase=att["eager_by_phase"])
        rec["stage"] = "complete"
        p = _write(rec, run_id)
        rec["artifacts"] = dict(json=str(p), pt=str(tp))
        _write(rec, run_id)
        print(f"[sqk] RESULT_JSON={p}")
        print(f"[sqk] RESULT_PT={tp}")
        return 0
    except Exception as e:                                          # noqa: BLE001
        try:
            M.clear_attn_probe()
            M.clear_serial_qk()
        except Exception:                                           # noqa: BLE001
            pass
        rec["stage"] = ("checks_failed" if isinstance(e, (DC.CheckError, DR.AlignError,
                                                          A.AlignError, CC.CloneError,
                                                          LB.BoundaryError, OP.ProbeError))
                        else "crashed")
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        p = _write(rec, run_id)
        print(f"[sqk] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[sqk] RESULT_JSON={p}")
        return 5 if rec["stage"] == "checks_failed" else 2
    finally:
        try:
            LB.assert_clean()
            M.clear_attn_probe()
            M.clear_serial_qk()
        except Exception:                                           # noqa: BLE001
            pass
        if TRS is not None:
            rec["target_rope_scope_restore"] = TRS.assert_restored()
            _write(rec, run_id)


# ------------------------------------------------------------------ CPU 自测（门）
def _selftest(rec: dict, run_id: str) -> int:
    """**不加载大模型**：在微型模型上把整条链路走一遍（真实 `SelfAttention.forward`）。

    覆盖：诊断关闭/开启的真实前向、只作用于注册模块、每行 `p+i+1`、mask 后未来列 `-inf`、
    `s=1` 与生产原式逐位一致、上下文正常/异常退出都恢复、H1–H4 分类、payload 落盘。
    """
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
                   num_layers=2, rope_theta=10000.0, rms_eps=1e-6, attn_bias=True
                   ).to(torch.bfloat16).eval()
    M.set_attention_strategy("eager")
    ids = torch.randint(0, 64, (1, PRE))
    with torch.inference_mode():
        _lg, c0 = tgt._run(tgt.embed_tokens(ids), None, need_logits=True)
    H = [{k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in lay.items()}
         for lay in CC.clone_list_cache_full(c0)]
    dev = torch.device("cpu")

    # ---- ① 关闭时的真实前向：与"从未安装"逐位等价 ------------------------------
    with torch.inference_mode():
        lg_ref, _ = tgt._run(tgt.embed_tokens(ids), None, need_logits=True)
    with M.serial_qk_for(tgt.layers[0].self_attn, label="t"):
        with torch.inference_mode():
            lg_on, _ = tgt._run(tgt.embed_tokens(ids), None, need_logits=True)
        calls_prefill = M.serial_qk_calls()
    st_off = M.serial_qk_state()
    with torch.inference_mode():
        lg_after, _ = tgt._run(tgt.embed_tokens(ids), None, need_logits=True)
    check("★ 上下文退出后状态为关闭（且后续前向与关闭时逐位相同）",
          st_off["active"] is False and torch.equal(lg_after, lg_ref))
    check("★ prefill 口径（cache=None）也走逐行 M=1：p=T-s=0，每行 visible=i+1，输出全有限",
          len(calls_prefill) == 1 and calls_prefill[0]["s"] == PRE
          and calls_prefill[0]["p"] == 0
          and calls_prefill[0]["meta"]["visible"] == [i + 1 for i in range(PRE)]
          and bool(torch.isfinite(lg_on).all()),
          f"{calls_prefill[0]['meta']['visible'] if calls_prefill else None}")
    # 观测（**不作为门**）：微型模型上 prefill 的逐行 M=1 与整块 M=5 恰好逐位相同
    print(f"  [obs] 微型 prefill：逐行 M=1 与整块 M={PRE} 是否逐位相同 = {bool(torch.equal(lg_on, lg_ref))}"
          f"（微型模型上的巧合，**不**外推）")

    # ---- ② 四臂（微型）---------------------------------------------------------
    p1 = {a: _capture_arm(tgt, H, dev, a, "decode" if a in S_ARMS else "verify", "s1", serial=(a in ("B1", "S1")),
                          poison=(a == "B1"))
          for a in ("S0", "B0", "B1", "S1")}
    check("★ 四臂捕获都能跑通，且探针保留的是 L0 的节点",
          all(all(k in p1[a]["probes"]["records"] for k in
                  ("x0", "q_linear", "q_rope", "k_visible", "qk_scores_pre_mask", "qk_scores_post_mask",
                   "attention_prob", "o_proj", "layer_output")) for a in p1))
    # 逐位等价：S0 的最终行必须等于"从未安装"时 S 路径的最终行
    with torch.inference_mode(), M.attention_phase("decode"):
        lg_native_s, _ = spec_dec._step(tgt, FEED[0], _to_dev(H, dev))
    with torch.inference_mode(), M.attention_phase("verify"):
        rows_native_b, _ = spec_dec._verify(tgt, _to_dev(H, dev), CANDS)
    check("★ 控制 11（诊断关闭 ⇒ 生产路径逐位不变）：S0/B0 与原生前向逐位一致",
          torch.equal(p1["S0"]["final_row"], LB.row_of(lg_native_s, None).detach().cpu())
          and torch.equal(p1["B0"]["final_row"], LB.row_of(rows_native_b, 0).detach().cpu()))
    # ---- ③ 干预范围与逐行口径 -------------------------------------------------
    st = p1["B1"]["sq_state"]
    check("★ 控制 9/10：进入分支的模块 = 其余全部层；被替换的**恰好**是 L0 一次",
          st["n_seen"] == 1 and st["n_applied"] == 1
          and st["seen_module_ids"] == [int(id(tgt.layers[1].self_attn))]
          and [a["module_id"] for a in st["applied"]] == [int(id(tgt.layers[0].self_attn))], st)
    calls = _serial_to_cpu(p1["B1"]["sq_calls"])
    check("★ 控制 6：B1 每行可见长度 == p+i+1（微型：p=5,s=4 ⇒ 6,7,8,9）",
          calls[0]["visible"] == [6, 7, 8, 9] == calls[0]["expected_visible"], calls[0]["visible"])
    post_b1 = p1["B1"]["probes"]["records"]["qk_scores_post_mask"]["tensor"]
    n_inf = int(torch.isinf(post_b1).sum())
    check("★ 控制 7：mask 后未来列全为 -inf（微型 4 头 ⇒ 4*(3+2+1)=24）且可见区全有限",
          n_inf == 24 and bool(torch.isfinite(post_b1[0, :, :, :VIS]).all()), n_inf)
    check("★ 控制 8：不读未来 key（哨兵：把该行之后的 key 灌 NaN ⇒ 该行 raw/scaled 逐位不变）",
          all(r["raw_equal"] and r["scaled_equal"] and r["finite"] for r in p1["B1"]["poison"]),
          p1["B1"]["poison"])
    check("★ 控制 4：S1（s=1 开启分支）与 S0 逐位一致（节点 + 最终行）",
          all(torch.equal(p1["S1"]["probes"]["records"][k]["tensor"],
                          p1["S0"]["probes"]["records"][k]["tensor"])
              for k, v in p1["S0"]["probes"]["records"].items() if not v.get("shape_only"))
          and torch.equal(p1["S1"]["final_row"], p1["S0"]["final_row"])
          and len(p1["S1"]["sq_calls"]) == 1)
    # ---- ④ H1：B1 的 QK 第 0 行是否逐位恢复为 S0 ------------------------------
    qk_s0 = p1["S0"]["probes"]["records"]["qk_scores_pre_mask"]["tensor"]
    qk_b1 = p1["B1"]["probes"]["records"]["qk_scores_pre_mask"]["tensor"]
    h1 = SQ.classify_h1(qk_s0, qk_b1, vis=VIS)
    check("★ H1 分类可跑通（微型模型上给出 exact/n_diff/坐标）",
          isinstance(h1["exact_equal"], bool) and h1["shape"] == [4, VIS], h1)
    # ---- ⑤ H2/H3/H4 分类（微型）----------------------------------------------
    nodes_s0, nodes_b1 = _tensors(p1["S0"]["probes"]["records"]), _tensors(p1["B1"]["probes"]["records"])
    cmp_s0b1 = _cmp_all(p1["S0"], p1["B1"])
    cls = OP.classify(OP.CHAIN, dict(cmp_s0b1, cache_pre=dict(exact_equal=True)), OP.CHAIN_NODE_OUT)
    rows_s0 = _chain_rows_to_cpu(p1["S0"]["chain"], 2)
    rows_b1 = _chain_rows_to_cpu(p1["B1"]["chain"], 2)
    h2 = SQ.classify_h2(nodes_s0, nodes_b1, OP.CHAIN, OP.CHAIN_NODE_OUT, vis=VIS, first_div=cls)
    h3 = SQ.classify_h3(rows_s0, rows_b1, n_layers=2)
    h4 = SQ.classify_h4(p1["S0"]["final_row"], p1["B1"]["final_row"])
    check("★ H2/H3/H4 分类都能在真实（微型）数据上给出结构完整的结果",
          h2["class"] in ("L0_restored", "L0_not_restored") and len(h3["layers"]) == 2
          and h4["category"] in (1, 2, 3), (h2["class"], h3["first_layer_with_any_output_diff"],
                                            h4["category"]))
    print(f"  [obs] 微型模型：H1 exact={h1['exact_equal']}；H2 {h2['class']}；"
          f"H3 首个输出差异层={h3['first_layer_with_any_output_diff']}；H4 category={h4['category']}")
    # ---- ⑥ B 口径 vs B 口径比较器 ---------------------------------------------
    cmp_b0b1 = _cmp_all_B(p1["B0"], p1["B1"])
    check("★ B 口径比较器（`_cmp_all_B`）对两侧都是 s=4：形状/有限性/逐位字段齐全",
          all("shape_a" in v and "n_diff" in v and "sha16_a" in v for v in cmp_b0b1.values())
          and cmp_b0b1["qk_scores_pre_mask"]["shape_a"][2] == 4)
    # ---- ⑦ 异常退出也恢复 ------------------------------------------------------
    try:
        with M.serial_qk_for(tgt.layers[0].self_attn, label="boom"):
            raise RuntimeError("注入的干预故障")
    except RuntimeError:
        pass
    check("★ 上下文异常退出后也恢复（`active=False`，且后续前向与关闭时逐位相同）",
          M.serial_qk_state()["active"] is False and M.serial_qk_calls() == [])
    with torch.inference_mode():
        lg_x, _ = tgt._run(tgt.embed_tokens(ids), None, need_logits=True)
    check("★ 异常退出后生产路径仍逐位不变", torch.equal(lg_x, lg_ref))
    # ---- ⑧ 未注册模块不被替换 --------------------------------------------------
    with M.serial_qk_for(tgt.layers[0].self_attn):
        check("★ 注册的模块对象身份被核对（`mods[id] is` 同一对象，防 id 复用）",
              M.serial_qk_state()["mods"][0]["module_id"] == id(tgt.layers[0].self_attn)
              and M.serial_qk_state()["mods"][0]["module_id"] != id(tgt.layers[1].self_attn))
    # ---- ⑨ 落盘 ---------------------------------------------------------------
    payload = _build_payload(p1, n_layers=2)
    tp = _save_pt(payload, run_id, selftest=True)
    check("★ 落盘 payload 可跑通（四臂节点/层行/最终行/诊断逐行/哨兵结果都在）",
          tp.exists() and tp.stat().st_size > 0
          and all(f"nodes_{a}" in payload for a in ("S0", "B0", "B1", "S1"))
          and payload["serial_B1"] and payload["serial_B1"][0]["rows"][0]["raw"].shape[-1] == 6, str(tp))
    check("★ 捕获/干预结束后：探针关闭、诊断关闭、层钩子全部移除",
          M.attn_probe_state()["active"] is False and M.serial_qk_state()["active"] is False
          and all(len(tgt.layers[i]._forward_pre_hooks) == 0 and len(tgt.layers[i]._forward_hooks) == 0
                  for i in range(2)))
    print(f"\nQK 逐行 M=1 因果干预工具（CPU 门）：PASS {npass} / FAIL {nfail}")
    rec["selftest"] = dict(npass=npass, nfail=nfail,
                           semantics="非真实运行：微型模型上验证真实 forward 的干预链路"
                                     "（关闭等价/只作用 L0/每行 p+i+1/mask 覆盖/哨兵/异常恢复/H1–H4/落盘）",
                           pt=str(tp))
    rec["stage"] = "selftest"
    _write(rec, run_id, selftest=True)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
