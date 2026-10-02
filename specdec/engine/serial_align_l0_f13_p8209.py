"""serial_align_l0_f13_p8209.py — L0 的 **QK + o_proj 联合对齐**（F13；只读、有界、单次）。

## 本轮唯一目标
在**保留** L0 serial-causal QK 干预的基础上，把 L0 的 `self_attn.o_proj` 也按行以 **M=1**
调用**同一个 `nn.Linear`** 再拼回 `(1,4,1536)`，判定 `o_proj`、L0 输出、后续层与最终 logits
能否进一步恢复。**只干预 L0 的 QK 与 L0 的 `o_proj`，其他算子一律不动。**

## 五个 arm（同一份 `H(8208)`，各自独立克隆 cache，串行执行）
| arm | 输入 | serial-causal QK | serial o_proj（逐行 M=1） |
| --- | --- | --- | --- |
| **S0** | `[20642]` | 关 | 关 |
| **B0** | `[20642,659,576,4531]` | 关 | 关 |
| **BQ** | 同 B0 | **开** | 关 |
| **BQO** | 同 B0 | **开** | **开** |
| **S1** | 同 S0 | **开** | **开** |

## 判据
* **H1**：S0 == 历史 S0、B0 == 历史 B0、**BQ == 上一轮 QK-only 干预**、S1 == S0（任一不成立即停止解释）。
* **H2**：`heads_merged` 决策行逐位相同、`o_proj` 决策行逐位一致（否则干预实现无效、停止归因）；
  并记录 BQ 的 2 个差异是否被完全消除。
* **H3**：L0 输出是否完全恢复；未恢复则给出**第一个持续到出口**的独立边界。
* **H4**：逐层（28 层）输入/输出决策行轨迹（**不**对输入已不同的层做独立来源判定）。
* **H5**：最终 logits / argmax / top-2 margin 的恢复类别（1/2/3）。

## 纪律
生产公式/dtype/精度/mask/RoPE/attention/cache/argmax/后端**一律不改**；两个诊断分支都**默认关闭**、
只作用于显式注册的模块实例、异常也恢复；不做 profiler/FP32、不切后端、**不跑 B6**；GPU 串行；
硬失败**落盘即停**、不自动重试；旧产物不覆盖；**B6 状态不变**。
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
from . import serial_align as SA
from . import serial_causal_qk as SQ
from . import spec_cache
from . import spec_dec
from . import g5_8k_spec_variance_probe as P5V

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"

TAG = os.environ.get("SLA_TAG", "div8209_sla")
MODE = os.environ.get("SLA_MODE", "gpu").strip().lower()
LAYER = int(os.environ.get("SLA_LAYER", "0"))
PRE = int(os.environ.get("SLA_PRE", "8208"))
TARGET_POS = int(os.environ.get("SLA_TARGET", "8209"))
K = int(os.environ.get("SLA_K", "4"))
FEED = [int(x) for x in os.environ.get("SLA_FEED", "20642").split(",")]
CANDS = [int(x) for x in os.environ.get("SLA_CANDS", "20642,659,576,4531").split(",")]
WATCH = tuple(int(x) for x in os.environ.get("SLA_WATCH", "323,659,20642,576,4531").split(","))
REPEAT = int(os.environ.get("SLA_REPEAT", "2"))
TARGET_ROPE = os.environ.get("SLA_TARGET_ROPE", "cpu_fp32").strip().lower()
POISON_CHECK = S.env_flag("SLA_POISON_CHECK", True)
CAPTURE_OFF_CHECK = S.env_flag("SLA_CAPTURE_OFF_CHECK", True)
CAP_H = os.environ.get("SLA_CAP_H", str(OUTDIR / "divcap_ar_20261001T185503Z.pt"))
CAP_H2 = os.environ.get("SLA_CAP_H2", str(OUTDIR / "divasc_rebuild_20261001T220737Z.pt"))
L0OPS_PT = os.environ.get("SLA_L0OPS_PT", str(OUTDIR / "div8209_l0ops_20261002T172447Z.pt"))
PREV_PT = os.environ.get("SLA_PREV_PT", str(OUTDIR / "div8209_sqk_20261002T184729Z.pt"))
PREV_JSON = os.environ.get("SLA_PREV_JSON", str(OUTDIR / "div8209_sqk_20261002T184729Z.json"))
QK2X2_JSON = os.environ.get("SLA_QK2X2_JSON",
                            str(OUTDIR / "qk2x2_8209_verdict_cpu_20261002T181152Z.json"))
SAVED_PT = os.environ.get("SLA_SAVED_PT", str(OUTDIR / "div8209_cells_20261002T064415Z.pt"))
REF_RUN_AR = os.environ.get("SLA_REF_AR", str(OUTDIR / "smoke8k_eager_20261001T185503Z.json"))
REF_RUN_RB = os.environ.get("SLA_REF_RB", str(OUTDIR / "smoke8k_eager_20261001T220737Z.json"))
KV, ARM = "base_bf16", os.environ.get("SLA_ARM", "eager")
VIS = PRE + 1
S_ARMS = ("S0", "S1")
PLAN = (("S0", "decode", False, False), ("B0", "verify", False, False), ("BQ", "verify", True, False),
        ("BQO", "verify", True, True), ("S1", "decode", True, True))


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


def _row_pair(a_nodes: dict, b_nodes: dict, name: str) -> dict:
    """S 口径（末位行）vs B 口径（第 0 行）的 `row` 类节点比较。"""
    return _row_cmp(a_nodes["probes"]["records"][name]["tensor"][0, -1],
                    b_nodes["probes"]["records"][name]["tensor"][0, 0], name)


def _cmp_all(a_nodes: dict, b_nodes: dict) -> dict:
    out = {}
    for name, (who, kind) in OP.SPECS.items():
        if kind == "shape_only":
            continue
        ta = a_nodes["probes"]["records"].get(name)
        tb = b_nodes["probes"]["records"].get(name)
        DC.require(ta is not None and tb is not None, f"探针缺少节点：{name}")
        out[name] = OP.cmp_node(name, kind, ta["tensor"], tb["tensor"], vis=VIS)
    return out


def _cmp_all_B(a_nodes: dict, b_nodes: dict) -> dict:
    """B 口径 vs B 口径（两侧都取第 0 行）。

    ⚠️ 首跑缺陷：这里曾写成 `return {"row": t[0,0], "heads": t[0,:,0,:], ...}[kind]` ——
    **dict 字面量会把四个分支全部求值**，对 3 维张量取 4 个索引直接 IndexError（白付一次加载）。
    必须用 `if/elif` 只求值命中的那一支。
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
        a = sl(kind, a_nodes["probes"]["records"][name]["tensor"])
        b = sl(kind, b_nodes["probes"]["records"][name]["tensor"])
        out[name] = dict(node=name, exact_equal=bool(torch.equal(a, b)),
                         n_diff=int((a != b).sum()),
                         max_abs=float((a.float() - b.float()).abs().max()),
                         sha16_a=A.stable_sha(a), sha16_b=A.stable_sha(b))
    return out


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
    if y is None:
        return None
    return (y[0, row_index] if y.dim() == 3 else y).detach().cpu()


def _norm_in_row(norm_caps: dict, row_index: int):
    caps = norm_caps.get("caps") or {}
    if not caps:
        return None
    rec = caps[max(caps)]
    x = rec.get("x_in")
    if x is None:
        return None
    return (x[0, row_index] if x.dim() == 3 else x).detach().cpu()


def _tensors(records: dict) -> dict:
    return {k: v["tensor"] for k, v in records.items() if not v.get("shape_only")}


def _capture_arm(tgt, H, dev, arm: str, mode: str, tag: str, *, serial_qk: bool, serial_lin: bool,
                 poison: bool = False) -> dict:
    """单臂捕获：QK/o_proj 干预（可选）+ 28 层链 + 最终 RMSNorm + L0 探针 + 真实前向。"""
    cache = _to_dev(H, dev)
    sa0 = tgt.layers[LAYER].self_attn
    ctx_qk = M.serial_qk_for(sa0, label=arm) if serial_qk else contextlib.nullcontext(None)
    ctx_lin = M.serial_linear_for(sa0.o_proj, label=arm) if serial_lin else contextlib.nullcontext(None)
    row_index = -1 if arm in S_ARMS else 0
    with ctx_qk, ctx_lin, \
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
        sq_calls = M.serial_qk_calls() if serial_qk else []
        sl_calls = M.serial_linear_calls() if serial_lin else []
        sq_state = M.serial_qk_state() if serial_qk else None
        sl_state = M.serial_linear_state() if serial_lin else None
        poison_res = None
        if poison and serial_qk and sq_calls:
            q_g = cap.records["q_rope"]["tensor"]
            k_g = cap.records["k_visible"]["tensor"]
            k_rep_g = M.kv_repeat(k_g, sa0.heads // sa0.kv_heads)
            p_g = int(sq_calls[0]["p"])
            poison_res = []
            for i, r in enumerate(sq_calls[0]["meta"]["rows"]):
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
    return dict(arm=arm, mode=mode, row_index=int(row_index), serial_qk=bool(serial_qk),
                serial_lin=bool(serial_lin), chain=chain, norm_caps=norm_caps, norm_info=norm_info,
                probes=probes, final_row=row.detach().to("cpu"),
                cache_len_after=int(spec_cache.cache_length(cache_after)),
                cache_len_before=int(chain["layers"][LAYER]["pre"]["cache_len_before"]),
                x0_row=chain["layers"][LAYER]["pre"]["a_row"].detach().cpu(),
                n_pre=int(chain["n_pre"]), n_post=int(chain["n_post"]),
                sq_calls=sq_calls, sl_calls=sl_calls, sq_state=sq_state, sl_state=sl_state,
                poison=poison_res)


def _serial_qk_to_cpu(sq_calls: list) -> list:
    out = []
    for c in sq_calls:
        meta = c["meta"]
        out.append(dict(module_id=int(id(c["module"])), s=int(c["s"]), T=int(c["T"]), p=int(c["p"]),
                        visible=list(meta["visible"]), expected_visible=list(meta["expected_visible"]),
                        padded=list(meta["padded"]), placeholder=float(meta["placeholder"]),
                        contiguous_rows=bool(meta["contiguous_rows"]),
                        scores=c["scaled"].detach().cpu(),
                        rows=[dict(i=int(r["i"]), visible=int(r["visible"]),
                                   raw=r["raw"].detach().cpu(), scaled=r["scaled"].detach().cpu(),
                                   layout_q_used=r["layout_q_used"], layout_k_used=r["layout_k_used"],
                                   finite=bool(r["finite"])) for r in meta["rows"]]))
    return out


def _serial_lin_to_cpu(sl_calls: list) -> list:
    out = []
    for c in sl_calls:
        meta = c["meta"]
        out.append(dict(module_id=int(id(c["module"])), s=int(meta["s"]),
                        cat_dim=int(meta["cat_dim"]), contiguous_rows=bool(meta["contiguous_rows"]),
                        per_row_input_shapes=list(meta["per_row_input_shapes"]),
                        out_features=int(meta["out_features"]),
                        rows=[dict(r) for r in meta["rows"]],
                        out=c["out"].detach().cpu()))
    return out


def _build_payload(arms: dict, *, n_layers: int) -> dict:
    pay: dict = dict(meta=dict(vis=VIS, layer=LAYER, n_layers=n_layers, arms=sorted(arms)))
    for a, r in arms.items():
        pay[f"nodes_{a}"] = {k: v["tensor"].detach().cpu()
                             for k, v in r["probes"]["records"].items() if not v.get("shape_only")}
        pay[f"layouts_{a}"] = {k: v["layout"] for k, v in r["probes"]["records"].items()}
        pay[f"rows_{a}"] = _chain_rows_to_cpu(r["chain"], n_layers)
        pay[f"norm_row_{a}"] = _norm_row(r["norm_caps"], r["row_index"])
        pay[f"norm_in_row_{a}"] = _norm_in_row(r["norm_caps"], r["row_index"])
        pay[f"final_row_{a}"] = r["final_row"].detach().cpu()
        pay[f"serial_qk_{a}"] = _serial_qk_to_cpu(r["sq_calls"]) if r["serial_qk"] else None
        pay[f"serial_lin_{a}"] = _serial_lin_to_cpu(r["sl_calls"]) if r["serial_lin"] else None
        pay[f"poison_{a}"] = r["poison"]
    pay["note"] = ("`nodes_*` = L0 探针捕获的完整张量（原始 dtype）；`rows_*` = 28 层输入/输出决策行；"
                   "`norm_in_row_*`/`norm_row_*` = 最终 RMSNorm 的输入/输出决策行；"
                   "`final_row_*` = 完整词表 logits 决策行；`serial_qk_*`/`serial_lin_*` = 两个诊断分支的"
                   "逐行原始记录（含布局与逐行输入形状）；`poison_*` = 不读未来 key 的哨兵结果。")
    return pay


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="serial_align_l0_f13_p8209", family="F13", kv=KV, arm=ARM,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               question="在保留 L0 serial-causal QK 干预的基础上，再把 L0 的 o_proj 按行以 M=1 调用同一 "
                        "nn.Linear 后拼回，o_proj / L0 输出 / 后续层 / 最终 logits 能否进一步恢复？",
               semantics="两个**默认关闭**的实例级诊断分支（逐行 M=1 的 causal-prefix QK 与逐行 M=1 的 "
                         "o_proj），五臂 S0/B0/BQ/BQO/S1 串行、同一份 H(8208)、各臂独立克隆 cache。",
               config=dict(tag=TAG, layer=LAYER, pre=PRE, target_pos=TARGET_POS, k=K, feed=FEED,
                           cands=CANDS, watch=list(WATCH), repeat=REPEAT, vis=VIS,
                           target_rope=TARGET_ROPE, poison_check=POISON_CHECK,
                           capture_off_check=CAPTURE_OFF_CHECK, cap_h=str(CAP_H), cap_h2=str(CAP_H2),
                           l0ops_pt=str(L0OPS_PT), prev_pt=str(PREV_PT), prev_json=str(PREV_JSON),
                           qk2x2_json=str(QK2X2_JSON), saved_pt=str(SAVED_PT),
                           ref_ar=str(REF_RUN_AR), ref_rb=str(REF_RUN_RB),
                           plan=[dict(arm=a, mode=m, qk=q, o_proj=o) for a, m, q, o in PLAN],
                           in_scope="只干预 L0 的 QK 与 L0 的 o_proj；不改 Q/K/V projection、softmax、AV、MLP、"
                                    "权重、mask、RoPE、cache、argmax、后端；不做 profiler/FP32；不跑 B6"))
    print(f"[sla] run_id={run_id} layer={LAYER} pre={PRE} target={TARGET_POS} vis={VIS} "
          f"repeat={REPEAT} rope={TARGET_ROPE}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    if MODE == "selftest":
        return _selftest(rec, run_id)
    ok, why = P5V.commit_guard()
    rec["commit_guard"] = dict(ok=bool(ok), why=why)
    print(f"[sla] commit: {why}")
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
        rec["code_facts"] = LB.code_facts(tgt)
        rec["norm"] = LB.norm_info(tgt)
        rec["layer"] = LB.layer_info(tgt, LAYER)
        sa0 = tgt.layers[LAYER].self_attn
        rec["oproj_facts"] = SA.module_facts(sa0.o_proj)         # 从**实际模块**读取（不靠常量）
        rec["qkv_facts"] = dict(q=SA.module_facts(sa0.q_proj), k=SA.module_facts(sa0.k_proj),
                                v=SA.module_facts(sa0.v_proj))
        rec["probe_state_before"] = M.attn_probe_state()
        rec["serial_qk_state_before"] = M.serial_qk_state()
        rec["serial_linear_state_before"] = M.serial_linear_state()
        for k in ("probe_state_before",):
            DC.require(rec[k]["active"] is False, "进入实验前探针必须关闭", str(rec[k]))
        DC.require(rec["serial_qk_state_before"]["active"] is False
                   and rec["serial_linear_state_before"]["active"] is False,
                   "进入实验前两个诊断分支都必须关闭")
        DC.require(n_layers >= 2 and LAYER == 0, "本轮只做 L0", f"n_layers={n_layers} LAYER={LAYER}")
        rec["W_before"] = LB.weights_info(tgt)
        print(f"[sla] o_proj 事实：{rec['oproj_facts']}", flush=True)
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
        print(f"[sla] H：{Path(CAP_H).name} {src_len} → {PRE}（纯切片={tinfo['is_pure_slice']}）", flush=True)

        # ---------------- 五臂 × REPEAT ----------------
        passes = []
        for it in range(REPEAT):
            arms = {}
            for arm, mode, sq, sl in PLAN:
                arms[arm] = _capture_arm(tgt, H, dev, arm, mode, f"p{it + 1}", serial_qk=sq,
                                         serial_lin=sl, poison=(POISON_CHECK and arm == "BQO"))
                print(f"[sla] pass {it + 1}/{REPEAT} arm {arm} ✔（探针={arms[arm]['probes']['n_calls']}/"
                      f"{arms[arm]['probes']['n_kept']}；QK applied={len(arms[arm]['sq_calls'])} "
                      f"o_proj applied={len(arms[arm]['sl_calls'])}）", flush=True)
            passes.append(arms)
        P1, P2 = passes[0], (passes[1] if len(passes) > 1 else None)
        DC.require(M.attn_probe_state()["active"] is False, "捕获后探针仍在位")
        DC.require(M.serial_qk_state()["active"] is False and M.serial_linear_state()["active"] is False,
                   "捕获后诊断分支仍开启")
        rec["serial_state_after"] = dict(qk=M.serial_qk_state(), linear=M.serial_linear_state())

        rep = dict(n_passes=len(passes))
        if P2 is not None:
            for a in [p[0] for p in PLAN]:
                rep[f"{a}_nodes_identical"] = bool(all(
                    torch.equal(P1[a]["probes"]["records"][k]["tensor"],
                                P2[a]["probes"]["records"][k]["tensor"])
                    for k, v in P1[a]["probes"]["records"].items() if not v.get("shape_only")))
                rep[f"{a}_final_row_identical"] = bool(torch.equal(P1[a]["final_row"], P2[a]["final_row"]))
                rep[f"{a}_layer_rows_identical"] = bool(all(
                    torch.equal(_chain_rows_to_cpu(P1[a]["chain"], n_layers)[i][kk],
                                _chain_rows_to_cpu(P2[a]["chain"], n_layers)[i][kk])
                    for i in range(n_layers) for kk in ("input", "output")))
        rec["repeat_consistency"] = rep
        DC.require(all(v is True for k, v in rep.items() if k.endswith("identical")),
                   "两次独立捕获不一致（含控制 12）", str(rep))
        print("[sla] 重复一致 ✔（五臂 × 2 pass 的节点/最终行/层行都逐位相同）", flush=True)

        S0, B0, BQ, BQO, S1 = (P1["S0"], P1["B0"], P1["BQ"], P1["BQO"], P1["S1"])

        # ★★★ 先把**原始张量**落盘，再做任何比较 ★★★
        #   本轮首跑教训：比较代码里的一个下标缺陷（`_cmp_all_B` 的 dict 字面量分派）在
        #   10 次前向全部跑完之后崩溃，把昂贵的捕获数据全部丢掉。落盘必须早于比较。
        payload = _build_payload({a: P1[a] for a in [p[0] for p in PLAN]}, n_layers=n_layers)
        payload["H_L0_cache_pre"] = dict(k=H[LAYER]["k"].detach().cpu(), v=H[LAYER]["v"].detach().cpu())
        payload["oproj_facts"] = rec["oproj_facts"]
        tp = _save_pt(payload, run_id)
        rec["artifacts"] = dict(json=str(_write(rec, run_id)), pt=str(tp))
        print(f"[sla] 原始张量已落盘（早于比较）：{tp.name}", flush=True)

        # ---------------- 参考产物 ----------------
        l0ops = torch.load(L0OPS_PT, map_location="cpu") if Path(L0OPS_PT).exists() else {}
        prev = torch.load(PREV_PT, map_location="cpu") if Path(PREV_PT).exists() else {}
        prev_j = json.loads(Path(PREV_JSON).read_text(encoding="utf-8")) if Path(PREV_JSON).exists() else {}
        qk2x2 = json.loads(Path(QK2X2_JSON).read_text(encoding="utf-8")) if Path(QK2X2_JSON).exists() else {}
        saved_rows = {}
        if Path(SAVED_PT).exists():
            saved_rows = {k: v for k, v in torch.load(SAVED_PT, map_location="cpu").items()
                          if torch.is_tensor(v)}
        anchors_s = C._anchor_ar(REF_RUN_AR, TARGET_POS)
        anchors_b = C._anchor_specdec(REF_RUN_RB, PRE, TARGET_POS - PRE, TARGET_POS)

        # ---------------- 边界比较 ----------------
        n_s0b0 = _cmp_all(S0, B0)
        n_s0bq = _cmp_all(S0, BQ)
        n_s0bqo = _cmp_all(S0, BQO)
        n_bqbqo = _cmp_all_B(BQ, BQO)
        cache_pre = dict(exact_equal=True, n_diff=0,
                         cache_len_before=dict(S0=S0["cache_len_before"], B0=B0["cache_len_before"],
                                               BQ=BQ["cache_len_before"], BQO=BQO["cache_len_before"],
                                               S1=S1["cache_len_before"]),
                         note="五臂都从同一份 H(8208) 的 L0 缓存独立克隆出发（长度 8208）⇒ 该虚拟节点按构造逐位相同")
        pers_s0bqo = SA.chain_persistence(dict(n_s0bqo, cache_pre=cache_pre), OP.CHAIN, OP.CHAIN_NODE_OUT)
        pers_s0b0 = SA.chain_persistence(dict(n_s0b0, cache_pre=cache_pre), OP.CHAIN, OP.CHAIN_NODE_OUT)

        # ---- H1：旧结果复现 ----
        h1 = dict(
            S0_history=dict(
                final_row=(_row_cmp(S0["final_row"], l0ops["final_row_S"], "S0_final_vs_history")
                           if "final_row_S" in l0ops else None),
                L0_output=(_row_cmp(S0["probes"]["records"]["layer_output"]["tensor"][0, -1],
                                    l0ops["L0_output_row_S"], "S0_L0out_vs_history")
                           if "L0_output_row_S" in l0ops else None),
                qk_scores=(_row_cmp(S0["probes"]["records"]["qk_scores_pre_mask"]["tensor"],
                                    l0ops["raw_S"]["qk_scores_pre_mask"], "S0_qk_vs_history")
                           if l0ops else None),
                x0=(_row_cmp(S0["probes"]["records"]["x0"]["tensor"][0, -1], l0ops["x0_row_S"],
                             "S0_x0_vs_history") if "x0_row_S" in l0ops else None)),
            B0_history=dict(
                final_row=(_row_cmp(B0["final_row"], l0ops["final_row_B"], "B0_final_vs_history")
                           if "final_row_B" in l0ops else None),
                L0_output=(_row_cmp(B0["probes"]["records"]["layer_output"]["tensor"][0, 0],
                                    l0ops["L0_output_row_B"], "B0_L0out_vs_history")
                           if "L0_output_row_B" in l0ops else None),
                qk_scores=(_row_cmp(B0["probes"]["records"]["qk_scores_pre_mask"]["tensor"],
                                    l0ops["raw_B"]["qk_scores_pre_mask"], "B0_qk_vs_history")
                           if l0ops else None),
                x0=(_row_cmp(B0["probes"]["records"]["x0"]["tensor"][0, 0], l0ops["x0_row_B"],
                             "B0_x0_vs_history") if "x0_row_B" in l0ops else None)),
            BQ_prev_round=dict(
                final_row=(_row_cmp(BQ["final_row"], prev["final_row_B1"], "BQ_final_vs_prev")
                           if "final_row_B1" in prev else None),
                L0_output=(_row_cmp(BQ["probes"]["records"]["layer_output"]["tensor"][0, 0],
                                    prev["nodes_B1"]["layer_output"][0, 0], "BQ_L0out_vs_prev")
                           if prev else None),
                nodes_all_identical=bool(prev and all(
                    torch.equal(BQ["probes"]["records"][k]["tensor"], prev["nodes_B1"][k])
                    for k in prev["nodes_B1"] if k in BQ["probes"]["records"]
                    and not BQ["probes"]["records"][k].get("shape_only"))),
                layer_rows_identical=bool(prev and all(
                    torch.equal(_chain_rows_to_cpu(BQ["chain"], n_layers)[i][kk], prev["rows_B1"][i][kk])
                    for i in range(n_layers) for kk in ("input", "output"))),
                qk_row0_vs_S0=_row_cmp(BQ["probes"]["records"]["qk_scores_pre_mask"]["tensor"][0, :, 0, :VIS],
                                      S0["probes"]["records"]["qk_scores_pre_mask"]["tensor"][0, :, 0, :VIS],
                                      "BQ_qk_row0_vs_S0"),
                prev_expected=dict(L0_output_n_diff=(prev_j.get("H2") or {}).get("n_diff"),
                                   final_token=(((prev_j.get("controls") or {}).get("B1_final") or {})
                                                .get("argmax_id")),
                                   logits_n_diff=(prev_j.get("H4") or {}).get("logits_n_diff"))),
            S1_vs_S0=dict(
                nodes_identical=bool(all(torch.equal(S1["probes"]["records"][k]["tensor"],
                                                     S0["probes"]["records"][k]["tensor"])
                                         for k, v in S0["probes"]["records"].items()
                                         if not v.get("shape_only"))),
                final_row_identical=bool(torch.equal(S1["final_row"], S0["final_row"])),
                layer_rows_identical=bool(all(
                    torch.equal(_chain_rows_to_cpu(S1["chain"], n_layers)[i][kk],
                                _chain_rows_to_cpu(S0["chain"], n_layers)[i][kk])
                    for i in range(n_layers) for kk in ("input", "output"))),
                serial_applied=dict(qk=int(len(S1["sq_calls"])), o_proj=int(len(S1["sl_calls"])))))
        rec["H1"] = h1

        # ---- H2：o_proj 局部闭环 ----
        heads_s0 = S0["probes"]["records"]["heads_merged"]["tensor"][0, -1]
        heads_bqo = BQO["probes"]["records"]["heads_merged"]["tensor"][0, 0]
        op_s0 = S0["probes"]["records"]["o_proj"]["tensor"][0, -1]
        op_bqo = BQO["probes"]["records"]["o_proj"]["tensor"][0, 0]
        op_bq = BQ["probes"]["records"]["o_proj"]["tensor"][0, 0]
        h2 = SA.classify_h2_oproj(heads_s0, heads_bqo, op_s0, op_bqo, op_bq)
        rec["H2"] = h2

        # ---- H3：L0 是否完全恢复 ----
        l0_s0 = S0["probes"]["records"]["layer_output"]["tensor"][0, -1]
        l0_bqo = BQO["probes"]["records"]["layer_output"]["tensor"][0, 0]
        l0_b0 = B0["probes"]["records"]["layer_output"]["tensor"][0, 0]
        l0_bq = BQ["probes"]["records"]["layer_output"]["tensor"][0, 0]
        h3 = SA.classify_h3_l0(l0_s0, l0_bqo, pers_s0bqo, b0_l0_row=l0_b0)
        h3["bq_context"] = dict(n_diff_S0_vs_BQ=int((l0_s0 != l0_bq).sum()))
        rec["H3"] = h3

        # ---- H4：后续层 ----
        rows_s0 = _chain_rows_to_cpu(S0["chain"], n_layers)
        rows_bqo = _chain_rows_to_cpu(BQO["chain"], n_layers)
        rows_b0 = _chain_rows_to_cpu(B0["chain"], n_layers)
        h4 = SA.classify_h4_layers(rows_s0, rows_bqo, n_layers=n_layers)
        h4_b0 = SA.classify_h4_layers(rows_s0, rows_b0, n_layers=n_layers)
        rec["H4"] = h4

        # ---- H5：最终结果 ----
        h5 = SA.classify_h5_final(S0["final_row"], BQO["final_row"])
        h5_b0 = SA.classify_h5_final(S0["final_row"], B0["final_row"])
        h5_bq = SA.classify_h5_final(S0["final_row"], BQ["final_row"])
        rec["H5"] = h5

        # ---- 单独观察的 MLP 边界 ----
        mlp = SA.mlp_boundary_report(n_s0bqo, boundary_op=None)
        mlp["attention_residual_exact_equal"] = bool(n_s0bqo["attention_residual"]["exact_equal"])
        mlp["post_attention_rmsnorm_exact_equal"] = bool(n_s0bqo["post_attention_rmsnorm"]["exact_equal"])
        mlp["mlp_gate_proj_input_equal"] = bool(n_s0bqo["post_attention_rmsnorm"]["exact_equal"])
        mlp["mlp_up_proj_input_equal"] = bool(n_s0bqo["post_attention_rmsnorm"]["exact_equal"])
        mlp["per_operator_input_equal"] = {
            "gate_proj": bool(n_s0bqo["post_attention_rmsnorm"]["exact_equal"]),
            "up_proj": bool(n_s0bqo["post_attention_rmsnorm"]["exact_equal"]),
            "silu_activation": bool(n_s0bqo["mlp_gate_proj"]["exact_equal"]),
            "gated_product": bool(n_s0bqo["mlp_activation"]["exact_equal"]
                                  and n_s0bqo["mlp_up_proj"]["exact_equal"]),
            "down_proj": bool(n_s0bqo["mlp_gated_product"]["exact_equal"]),
            "mlp_residual_add": bool(n_s0bqo["attention_residual"]["exact_equal"]
                                     and n_s0bqo["mlp_down_proj"]["exact_equal"])}
        rec["MLP_boundary"] = mlp

        # ---- 参考对照（B0 / BQ）----
        rec["reference_results"] = dict(
            H3_BQ_only=dict(n_diff_S0_vs_BQ=int((l0_s0 != l0_bq).sum())),
            H3_B0=dict(n_diff_S0_vs_B0=int((l0_s0 != l0_b0).sum())),
            H4_B0_vs_S0=h4_b0, H5_B0_vs_S0=h5_b0, H5_BQ_vs_S0=h5_bq,
            first_divergence_S0_vs_B0=pers_s0b0,
            note="同一 run 内的 B0（无干预）与 BQ（只 QK）对照，用于说明本轮 BQO 的增量效果")
        rec["node_compare"] = dict(S0_vs_B0=n_s0b0, S0_vs_BQ=n_s0bq, S0_vs_BQO=n_s0bqo,
                                   BQ_vs_BQO=n_bqbqo)

        # ---------------- 必需控制（20 项）----------------
        controls: dict = {}
        controls["c1_S0_reproduces_history"] = h1["S0_history"]
        controls["c2_B0_reproduces_history"] = h1["B0_history"]
        controls["c3_BQ_reproduces_prev_round"] = h1["BQ_prev_round"]
        controls["c4_S1_equals_S0"] = h1["S1_vs_S0"]
        qk_row0_bqo_vs_s0 = _row_cmp(BQO["probes"]["records"]["qk_scores_pre_mask"]["tensor"][0, :, 0, :VIS],
                                     S0["probes"]["records"]["qk_scores_pre_mask"]["tensor"][0, :, 0, :VIS],
                                     "BQO_qk_row0_vs_S0")
        controls["c5_BQO_qk_row0_equals_S0"] = qk_row0_bqo_vs_s0
        controls["c6_BQO_heads_merged_row0_equals_S0"] = _row_cmp(heads_s0, heads_bqo,
                                                                 "BQO_heads_merged_vs_S0")
        controls["c7_BQO_oproj_row0_equals_S0"] = _row_cmp(op_s0, op_bqo, "BQO_oproj_vs_S0")
        sl_calls_cpu = _serial_lin_to_cpu(BQO["sl_calls"])
        controls["c8_serial_oproj_per_row_shape"] = dict(
            calls=[dict(module_id=c["module_id"], s=c["s"], cat_dim=c["cat_dim"],
                        per_row_input_shapes=c["per_row_input_shapes"],
                        all_rows_1x1_in=bool(all(sh == [1, 1, rec["oproj_facts"]["in_features"]]
                                                 for sh in c["per_row_input_shapes"])))
                   for c in sl_calls_cpu])
        w_now = SA.module_facts(sa0.o_proj)
        controls["c9_serial_oproj_same_module_weight_dtype"] = dict(
            facts_before=rec["oproj_facts"], facts_after=w_now,
            same_module_id=bool(rec["oproj_facts"]["module_id"] == w_now["module_id"]),
            same_weight_data_ptr=bool(rec["oproj_facts"]["weight_data_ptr"] == w_now["weight_data_ptr"]),
            weight_dtype=w_now["weight_dtype"],
            call_module_ids=[c["module_id"] for c in sl_calls_cpu],
            call_module_is_oproj=bool([c["module_id"] for c in sl_calls_cpu]
                                      == [int(id(sa0.o_proj))]),
            row_input_dtypes=[r["input_layout"]["dtype"] for r in (sl_calls_cpu[0]["rows"]
                                                                  if sl_calls_cpu else [])],
            no_bias=bool(w_now["has_bias"] is False))
        sl_state = BQO["sl_state"] or {}
        applied_ids = [a["module_id"] for a in sl_state.get("applied", [])]
        seen_ids = list(sl_state.get("seen_module_ids", []))
        oproj_all = [int(id(tgt.layers[i].self_attn.o_proj)) for i in range(n_layers)]
        controls["c10_intervention_only_L0_oproj"] = dict(
            n_applied=int(sl_state.get("n_applied", -1)), n_seen=int(sl_state.get("n_seen", -1)),
            n_layers=int(n_layers), applied_module_ids=applied_ids,
            L0_oproj_module_id=int(id(sa0.o_proj)),
            seen_is_all_other_oproj=bool(seen_ids == sorted(i for i in oproj_all
                                                            if i != int(id(sa0.o_proj)))),
            seen_plus_applied_covers_all=bool(sorted(seen_ids + applied_ids) == sorted(oproj_all)),
            applied_is_L0_oproj=bool(applied_ids == [int(id(sa0.o_proj))]))
        controls["c11_other_layers_oproj_block_path"] = dict(
            n_applied=int(sl_state.get("n_applied", -1)), n_layers=int(n_layers),
            note="只有注册模块（L0 的 o_proj）能被替换；其余 27 层的 o_proj 进入分支后只执行一条计数语句，"
                 "随后仍走**原样的一次调用**")
        controls["c12_two_BQO_repeats_identical"] = {k: v for k, v in rep.items() if k.startswith("BQO")}
        controls["c13_off_means_production_unchanged"] = dict(
            state_before=dict(qk=rec["serial_qk_state_before"], linear=rec["serial_linear_state_before"]),
            state_after=rec["serial_state_after"],
            S0_matches_history=bool(controls["c1_S0_reproduces_history"] is None
                                    or controls["c1_S0_reproduces_history"]["qk_scores"] is None
                                    or controls["c1_S0_reproduces_history"]["qk_scores"]["exact_equal"]),
            n_applied_when_off=0)
        controls["c14_hooks_and_context_restored"] = None      # §源完整性
        controls["c15_cache_inputs_weights_unmodified"] = None  # §源完整性
        controls["c16_finiteness"] = None                       # §有限性
        controls["c17_allow_tf32_false"] = dict(before=False, after=None)
        controls["c18_draft_never_loaded"] = dict(loaded_models=["target"], draft_loaded=False,
                                                  note="本工具只 `load_ours_bf16(target)`")
        controls["c19_git_and_artifacts"] = dict(git=S.git_meta(ROOT), recorded=True)
        controls["c20_old_artifacts_not_overwritten"] = dict(
            prev_pt=str(PREV_PT), prev_pt_mtime=None, prev_json=str(PREV_JSON),
            l0ops_pt=str(L0OPS_PT), note="新一轮用**新 tag**；旧 JSON/PT 只读、不做任何写入")
        controls["positions"] = dict(S_last_row=True, B_row0=True,
                                     both_predict_target=bool(S0["cache_len_after"] == TARGET_POS
                                                             and B0["cache_len_after"] == PRE + len(CANDS)),
                                     cache_lens_after=dict(S0=S0["cache_len_after"], B0=B0["cache_len_after"],
                                                           BQ=BQ["cache_len_after"], BQO=BQO["cache_len_after"],
                                                           S1=S1["cache_len_after"]),
                                     slot_for_B_row0=DR.slot_table(PRE, len(CANDS))[1])
        controls["S0_final"], controls["B0_final"] = _fields(S0["final_row"]), _fields(B0["final_row"])
        controls["BQ_final"], controls["BQO_final"] = _fields(BQ["final_row"]), _fields(BQO["final_row"])
        controls["token_matches_anchors"] = dict(
            S0=dict(measured=controls["S0_final"]["argmax_id"], anchor=anchors_s.get("token"),
                    match=bool(controls["S0_final"]["argmax_id"] == anchors_s.get("token"))),
            B0=dict(measured=controls["B0_final"]["argmax_id"], anchor=anchors_b.get("token"),
                    match=bool(controls["B0_final"]["argmax_id"] == anchors_b.get("token"))))
        controls["final_norm_row"] = dict(
            S0_vs_BQO=_row_cmp(_norm_row(S0["norm_caps"], -1), _norm_row(BQO["norm_caps"], 0),
                               "final_norm_S0_vs_BQO"),
            S0_vs_B0=_row_cmp(_norm_row(S0["norm_caps"], -1), _norm_row(B0["norm_caps"], 0),
                              "final_norm_S0_vs_B0"),
            final_norm_input_S0_vs_BQO=_row_cmp(_norm_in_row(S0["norm_caps"], -1),
                                                _norm_in_row(BQO["norm_caps"], 0),
                                                "final_norm_input_S0_vs_BQO"))
        controls["vs_saved_local_rows"] = dict(
            path=str(SAVED_PT), present=bool(saved_rows),
            S0=(_row_cmp(S0["final_row"], saved_rows["H_AR-S"], "S0_vs_saved") if "H_AR-S" in saved_rows else None),
            B0=(_row_cmp(B0["final_row"], saved_rows["H_AR-B"], "B0_vs_saved") if "H_AR-B" in saved_rows else None))
        rec["controls"] = controls

        # ---- 控制断言 ----
        for ck in ("c1_S0_reproduces_history", "c2_B0_reproduces_history"):
            for kk, v in controls[ck].items():
                DC.require(v is None or not isinstance(v, dict) or v.get("exact_equal") is True,
                           f"控制失败：{ck}/{kk}", str(v))
        prev_bq = controls["c3_BQ_reproduces_prev_round"]
        DC.require(prev_bq["nodes_all_identical"] and prev_bq["layer_rows_identical"]
                   and (prev_bq["final_row"] is None or prev_bq["final_row"]["exact_equal"])
                   and (prev_bq["L0_output"] is None or prev_bq["L0_output"]["exact_equal"]),
                   "控制 3 失败：BQ 未逐位复现上一轮的 QK-only 干预", str(prev_bq))
        DC.require(all(controls["c4_S1_equals_S0"][k] for k in
                       ("nodes_identical", "final_row_identical", "layer_rows_identical")),
                   "控制 4 失败：S1 与 S0 不逐位一致", str(controls["c4_S1_equals_S0"]))
        DC.require(controls["c5_BQO_qk_row0_equals_S0"]["exact_equal"],
                   "控制 5 失败：BQO 的 QK 第 0 行未与 S0 逐位一致", str(controls["c5_BQO_qk_row0_equals_S0"]))
        DC.require(controls["c6_BQO_heads_merged_row0_equals_S0"]["exact_equal"],
                   "控制 6 失败：BQO 的 heads_merged 决策行未与 S0 逐位一致",
                   str(controls["c6_BQO_heads_merged_row0_equals_S0"]))
        DC.require(controls["c7_BQO_oproj_row0_equals_S0"]["exact_equal"],
                   "★ 控制 7 失败：BQO 的 o_proj 决策行未与 S0 逐位一致 ⇒ 干预实现无效，停止归因",
                   str(controls["c7_BQO_oproj_row0_equals_S0"]))
        DC.require(all(c["all_rows_1x1_in"] for c in controls["c8_serial_oproj_per_row_shape"]["calls"]),
                   "控制 8 失败：serial o_proj 的每行输入不是 [1,1,in_features]",
                   str(controls["c8_serial_oproj_per_row_shape"]))
        c9 = controls["c9_serial_oproj_same_module_weight_dtype"]
        DC.require(c9["same_module_id"] and c9["same_weight_data_ptr"] and c9["call_module_is_oproj"]
                   and c9["no_bias"] and all(d == "torch.bfloat16" for d in c9["row_input_dtypes"]),
                   "控制 9 失败：serial o_proj 未使用同一模块/同一权重/dtype 不符", str(c9))
        DC.require(controls["c10_intervention_only_L0_oproj"]["n_applied"] == 1
                   and controls["c10_intervention_only_L0_oproj"]["applied_is_L0_oproj"]
                   and controls["c10_intervention_only_L0_oproj"]["seen_is_all_other_oproj"]
                   and controls["c10_intervention_only_L0_oproj"]["seen_plus_applied_covers_all"],
                   "控制 10/11 失败：o_proj 干预未严格限定在 L0", str(controls["c10_intervention_only_L0_oproj"]))
        DC.require(controls["positions"]["both_predict_target"], "位置口径失败", str(controls["positions"]))
        DC.require(all(v["match"] for v in controls["token_matches_anchors"].values()),
                   "锚点不一致（停止）", str(controls["token_matches_anchors"]))
        if saved_rows:
            for k in ("S0", "B0"):
                DC.require(controls["vs_saved_local_rows"][k]["exact_equal"],
                           f"与已保存局部对照行不同（{k}）")
        print(f"[sla] 控制 1-13 ✔（S0/B0 复现历史；BQ ≡ 上一轮；S1≡S0；QK row0={controls['c5_BQO_qk_row0_equals_S0']['exact_equal']}；"
              f"heads_merged={controls['c6_BQO_heads_merged_row0_equals_S0']['exact_equal']}；"
              f"o_proj row0={controls['c7_BQO_oproj_row0_equals_S0']['exact_equal']}；"
              f"逐行输入={controls['c8_serial_oproj_per_row_shape']['calls'][0]['per_row_input_shapes']}）",
              flush=True)

        # ---------------- 先落盘（已在上面完成；此处只做打印）----------------
        print("== H1–H5 ==")
        print(f"  H1 S0/B0 复现历史=True；BQ ≡ 上一轮（节点={prev_bq['nodes_all_identical']}，"
              f"层行={prev_bq['layer_rows_identical']}）；S1≡S0={controls['c4_S1_equals_S0']['nodes_identical']}")
        print(f"  H2 heads_merged exact={h2['heads_merged_exact_equal']}；o_proj exact={h2['o_proj_exact_equal']}"
              f"（n_diff={h2['o_proj_n_diff']}, max|Δ|={h2['o_proj_max_abs']:.6g}）；"
              f"BQ 的 2 差异被消除={h2['o_proj_2diffs_removed']}；class={h2['class_']}", flush=True)
        print(f"  H3 L0 输出：class={h3['class_']} exact={h3['exact_equal']} n_diff={h3['n_diff']} "
              f"max|Δ|={h3['max_abs']:.6g}；首个持续边界={h3['first_persistent_divergence_operator']}"
              f"（全部边界候选={h3['boundaries']}）", flush=True)
        print(f"  H4 全层：全同={h4['all_layers_identical']} 首个输出差异层={h4['first_layer_with_any_output_diff']} "
              f"首个「输入同·输出不同」层={h4['first_layer_with_equal_input_and_different_output']}", flush=True)
        print(f"  H5 logits：class={h5['category']} exact={h5['logits_exact_equal']} n_diff={h5['logits_n_diff']} "
              f"max|Δ|={h5['logits_max_abs']:.6g} margin S0={h5['margin_s0']:.6g}/BQO={h5['margin_x']:.6g} "
              f"token {h5['token_s0']} vs {h5['token_x']}", flush=True)
        print(f"  MLP：attention_residual 同={mlp['attention_residual_exact_equal']} "
              f"post_attn_norm 同={mlp['post_attention_rmsnorm_exact_equal']} "
              f"首个输出不同的 MLP 算子={mlp['first_mlp_output_that_differs']}", flush=True)
        print("== L0 逐算子（S0vsBQO / S0vsBQ / S0vsB0 的 n_diff）==", flush=True)
        for (op, ins), out_node in zip(OP.CHAIN, OP.CHAIN_NODE_OUT):
            c_o, c_q, c_0 = n_s0bqo[out_node], n_s0bq[out_node], n_s0b0[out_node]
            print(f"  {op:<24s} out_eq={str(bool(c_o['exact_equal'])):<5s} n_diff={int(c_o['n_diff']):>6d} "
                  f"| BQ {int(c_q['n_diff']):>6d} | B0 {int(c_0['n_diff']):>6d}", flush=True)

        # ---------------- 源完整性 / 有限性 ----------------
        H_digest_after = [A.stable_sha(lay["k"]) for lay in H] + [A.stable_sha(lay["v"]) for lay in H]
        nan_nodes = [k for k, v in n_s0bqo.items() if not v.get("finite_s", True) or not v.get("finite_b", True)]
        rec["finiteness"] = dict(nodes_with_nonfinite_slice=nan_nodes,
                                 logits_finite=bool(all(torch.isfinite(x["final_row"]).all() for x in P1.values())),
                                 expected_inf=dict(
                                     BQO=int(torch.isinf(BQO["probes"]["records"]["qk_scores_post_mask"]["tensor"]).sum()),
                                     B0=int(torch.isinf(B0["probes"]["records"]["qk_scores_post_mask"]["tensor"]).sum()),
                                     S0=int(torch.isinf(S0["probes"]["records"]["qk_scores_post_mask"]["tensor"]).sum()),
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
            oproj_weight_stable=bool(rec["oproj_facts"]["weight_data_ptr"] == w_now["weight_data_ptr"]),
            qkv_weight_stable=bool(all(rec["qkv_facts"][k]["weight_data_ptr"]
                                       == SA.module_facts(getattr(sa0, f"{k}_proj"))["weight_data_ptr"]
                                       for k in ("q", "k", "v"))),
            probe_state_now=M.attn_probe_state(),
            serial_qk_state_now=M.serial_qk_state(),
            serial_linear_state_now=M.serial_linear_state(),
            hooks_removed=bool(all(len(tgt.layers[i]._forward_pre_hooks) == 0
                                   and len(tgt.layers[i]._forward_hooks) == 0
                                   for i in range(n_layers)) and len(tgt.norm._forward_hooks) == 0),
            note="H 快照、全部权重、Norm 权重在全程不变；探针与两个诊断上下文都已关闭、层钩子全部移除")
        controls["c14_hooks_and_context_restored"] = dict(
            probe_active=rec["source_integrity"]["probe_state_now"]["active"],
            serial_qk_active=rec["source_integrity"]["serial_qk_state_now"]["active"],
            serial_linear_active=rec["source_integrity"]["serial_linear_state_now"]["active"],
            hooks_removed=rec["source_integrity"]["hooks_removed"])
        controls["c15_cache_inputs_weights_unmodified"] = dict(
            H_snapshot_unchanged=rec["source_integrity"]["H_snapshot_unchanged"],
            W_data_ptr_stable=rec["source_integrity"]["W_data_ptr_stable"],
            W_digest_stable=rec["source_integrity"]["W_digest_stable"],
            norm_weight_stable=rec["source_integrity"]["norm_weight_stable"],
            oproj_weight_stable=rec["source_integrity"]["oproj_weight_stable"],
            qkv_weight_stable=rec["source_integrity"]["qkv_weight_stable"],
            L0_cache_pre_len=int(H[LAYER]["k"].shape[2]),
            cache_pre_all_arms_equal=bool(len(set(x["cache_len_before"]
                                                  for x in (S0, B0, BQ, BQO, S1))) == 1))
        controls["c16_finiteness"] = rec["finiteness"]
        controls["c17_allow_tf32_false"] = dict(before=False, after=bool(torch.backends.cuda.matmul.allow_tf32))
        for k, v in controls["c15_cache_inputs_weights_unmodified"].items():
            if isinstance(v, bool):
                DC.require(v, f"控制 15 失败：{k}")
        DC.require(controls["c14_hooks_and_context_restored"]["probe_active"] is False
                   and controls["c14_hooks_and_context_restored"]["serial_qk_active"] is False
                   and controls["c14_hooks_and_context_restored"]["serial_linear_active"] is False
                   and controls["c14_hooks_and_context_restored"]["hooks_removed"],
                   "控制 14 失败：钩子/上下文未恢复", str(controls["c14_hooks_and_context_restored"]))
        DC.require(controls["c17_allow_tf32_false"]["after"] is False, "控制 17 失败：allow_tf32 非 False")

        off_checks: dict = {}
        if CAPTURE_OFF_CHECK:
            with torch.inference_mode(), M.attention_phase("decode"):
                lg_off_s, _c1 = spec_dec._step(tgt, FEED[0], _to_dev(H, dev))
            with torch.inference_mode(), M.attention_phase("verify"):
                rows_off_b, _c2 = spec_dec._verify(tgt, _to_dev(H, dev), CANDS)
            off_checks["S_off_equals_S0"] = bool(torch.equal(LB.row_of(lg_off_s, None).detach().cpu(),
                                                             S0["final_row"]))
            off_checks["B_off_equals_B0"] = bool(torch.equal(LB.row_of(rows_off_b, 0).detach().cpu(),
                                                             B0["final_row"]))
            off_checks["states_after_off"] = dict(qk=M.serial_qk_state(), linear=M.serial_linear_state())
            DC.require(off_checks["S_off_equals_S0"] and off_checks["B_off_equals_B0"]
                       and off_checks["states_after_off"]["qk"]["active"] is False
                       and off_checks["states_after_off"]["linear"]["active"] is False,
                       "干预关闭后生产路径与 S0/B0 不同", str(off_checks))
            print("[sla] 干预关闭后生产路径 ✔（S/B 都与 S0/B0 逐位一致）", flush=True)
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
        print(f"[sla] RESULT_JSON={p}")
        print(f"[sla] RESULT_PT={tp}")
        return 0
    except Exception as e:                                          # noqa: BLE001
        try:
            M.clear_attn_probe()
            M.clear_serial_qk()
            M.clear_serial_linear()
        except Exception:                                           # noqa: BLE001
            pass
        rec["stage"] = ("checks_failed" if isinstance(e, (DC.CheckError, DR.AlignError,
                                                          A.AlignError, CC.CloneError,
                                                          LB.BoundaryError, OP.ProbeError))
                        else "crashed")
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        p = _write(rec, run_id)
        print(f"[sla] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[sla] RESULT_JSON={p}")
        return 5 if rec["stage"] == "checks_failed" else 2
    finally:
        try:
            LB.assert_clean()
            M.clear_attn_probe()
            M.clear_serial_qk()
            M.clear_serial_linear()
        except Exception:                                           # noqa: BLE001
            pass
        if TRS is not None:
            rec["target_rope_scope_restore"] = TRS.assert_restored()
            _write(rec, run_id)


# ------------------------------------------------------------------ CPU 自测（门）
def _selftest(rec: dict, run_id: str) -> int:
    """**不加载大模型**：微型模型上把整条链路走一遍（真实 `SelfAttention.forward`、真实 `nn.Linear`）。"""
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
    sa0 = tgt.layers[0].self_attn

    # ---- ① 关闭时逐位等价 + 异常恢复 -------------------------------------------
    with torch.inference_mode():
        lg_ref, _ = tgt._run(tgt.embed_tokens(ids), None, need_logits=True)
    with M.serial_linear_for(sa0.o_proj, label="t"):
        with torch.inference_mode():
            lg_on, _ = tgt._run(tgt.embed_tokens(ids), None, need_logits=True)
        lin_calls = M.serial_linear_calls()
    with torch.inference_mode():
        lg_after, _ = tgt._run(tgt.embed_tokens(ids), None, need_logits=True)
    check("★ 关闭时生产路径逐位不变 + 上下文退出后恢复",
          M.serial_linear_state()["active"] is False and torch.equal(lg_after, lg_ref)
          and M.serial_linear_calls() == [])
    check("★ prefill 口径下 serial o_proj 也逐行调用（s=PRE ⇒ 每行一次，输入 [1,1,hidden]）",
          len(lin_calls) == 1 and lin_calls[0]["meta"]["s"] == PRE
          and all(sh == [1, 1, 32] for sh in lin_calls[0]["meta"]["per_row_input_shapes"])
          and bool(torch.isfinite(lg_on).all()), lin_calls[0]["meta"]["per_row_input_shapes"] if lin_calls else None)

    # ---- ② 两个 context 嵌套 ---------------------------------------------------
    with M.serial_qk_for(sa0, label="n"), M.serial_linear_for(sa0.o_proj, label="n"):
        with torch.inference_mode(), M.attention_phase("verify"):
            _rb, _ = spec_dec._verify(tgt, _to_dev(H, dev), CANDS)
        st_qk, st_lin = M.serial_qk_state(), M.serial_linear_state()
    check("★ QK 与 o_proj 两个 context **可嵌套**，且各自只替换自己的模块",
          st_qk["n_applied"] == 1 and st_lin["n_applied"] == 1
          and st_qk["applied"][0]["module_id"] == id(sa0)
          and st_lin["applied"][0]["module_id"] == id(sa0.o_proj)
          and M.serial_qk_state()["active"] is False and M.serial_linear_state()["active"] is False)
    check("★ 只作用于指定模块：进入分支但未注册的 layer-1 o_proj 被计数（其余层仍走原路径）",
          st_lin["n_seen"] == 1 and st_lin["seen_module_ids"] == [int(id(tgt.layers[1].self_attn.o_proj))],
          st_lin)
    for bad_ctx in (M.serial_qk_for(sa0), M.serial_linear_for(sa0.o_proj)):
        try:
            with bad_ctx:
                raise RuntimeError("注入的干预故障")
        except RuntimeError:
            pass
    check("★ 两个 context 异常退出也都恢复",
          M.serial_qk_state()["active"] is False and M.serial_linear_state()["active"] is False)

    # ---- ③ 五臂（微型）--------------------------------------------------------
    p1 = {}
    for arm, mode, sq, sl in PLAN:
        p1[arm] = _capture_arm(tgt, H, dev, arm, mode, "s1", serial_qk=sq, serial_lin=sl,
                               poison=(arm == "BQO"))
    check("★ 五臂捕获都能跑通，L0 探针节点齐全", all(
        all(k in p1[a]["probes"]["records"] for k in
            ("x0", "q_rope", "k_visible", "qk_scores_pre_mask", "attention_prob", "attention_times_v",
             "heads_merged", "o_proj", "attention_residual", "post_attention_rmsnorm", "mlp_gate_proj",
             "mlp_up_proj", "mlp_activation", "mlp_gated_product", "mlp_down_proj", "layer_output"))
        for a in p1))
    with torch.inference_mode(), M.attention_phase("decode"):
        lg_native_s, _ = spec_dec._step(tgt, FEED[0], _to_dev(H, dev))
    with torch.inference_mode(), M.attention_phase("verify"):
        rows_native_b, _ = spec_dec._verify(tgt, _to_dev(H, dev), CANDS)
    check("★ 控制 13（关闭 ⇒ 生产路径逐位不变）：S0/B0 与原生前向逐位一致",
          torch.equal(p1["S0"]["final_row"], LB.row_of(lg_native_s, None).detach().cpu())
          and torch.equal(p1["B0"]["final_row"], LB.row_of(rows_native_b, 0).detach().cpu()))
    check("★ 控制 4：S1（两个分支都开、s=1）与 S0 逐位一致",
          all(torch.equal(p1["S1"]["probes"]["records"][k]["tensor"],
                          p1["S0"]["probes"]["records"][k]["tensor"])
              for k, v in p1["S0"]["probes"]["records"].items() if not v.get("shape_only"))
          and torch.equal(p1["S1"]["final_row"], p1["S0"]["final_row"])
          and len(p1["S1"]["sq_calls"]) == 1 and len(p1["S1"]["sl_calls"]) == 1)
    sl = _serial_lin_to_cpu(p1["BQO"]["sl_calls"])
    check("★ 控制 8：serial o_proj 每行实际输入形状为 [1,1,hidden]（微型 hidden=32）",
          len(sl) == 1 and sl[0]["per_row_input_shapes"] == [[1, 1, 32]] * 4
          and sl[0]["cat_dim"] == 1 and sl[0]["out_features"] == 32)
    check("★ 控制 9：逐行调用用的是**同一个**模块实例（同一权重 data_ptr、同一 dtype、无 bias）",
          [c["module_id"] for c in sl] == [int(id(sa0.o_proj))]
          and SA.module_facts(sa0.o_proj)["has_bias"] is False
          and all(r["input_layout"]["dtype"] == "torch.bfloat16" for r in sl[0]["rows"])
          and all(r["input_shape"] == [1, 1, 32] for r in sl[0]["rows"]))
    check("★ 控制 10/11：o_proj 干预只作用于 L0（applied=1、seen=其余层）",
          p1["BQO"]["sl_state"]["n_applied"] == 1 and p1["BQO"]["sl_state"]["n_seen"] == 1
          and p1["BQO"]["sl_state"]["applied"][0]["module_id"] == int(id(sa0.o_proj))
          and p1["BQO"]["sl_state"]["seen_module_ids"] == [int(id(tgt.layers[1].self_attn.o_proj))])
    # ---- ④ H1–H5 分类（真实微型数据）------------------------------------------
    heads_s0 = p1["S0"]["probes"]["records"]["heads_merged"]["tensor"][0, -1]
    heads_bqo = p1["BQO"]["probes"]["records"]["heads_merged"]["tensor"][0, 0]
    op_s0 = p1["S0"]["probes"]["records"]["o_proj"]["tensor"][0, -1]
    op_bqo = p1["BQO"]["probes"]["records"]["o_proj"]["tensor"][0, 0]
    op_bq = p1["BQ"]["probes"]["records"]["o_proj"]["tensor"][0, 0]
    h2 = SA.classify_h2_oproj(heads_s0, heads_bqo, op_s0, op_bqo, op_bq)
    n_s0bqo = _cmp_all(p1["S0"], p1["BQO"])
    pers = SA.chain_persistence(dict(n_s0bqo, cache_pre=dict(exact_equal=True)), OP.CHAIN, OP.CHAIN_NODE_OUT)
    h3 = SA.classify_h3_l0(p1["S0"]["probes"]["records"]["layer_output"]["tensor"][0, -1],
                           p1["BQO"]["probes"]["records"]["layer_output"]["tensor"][0, 0], pers,
                           b0_l0_row=p1["B0"]["probes"]["records"]["layer_output"]["tensor"][0, 0])
    h4 = SA.classify_h4_layers(_chain_rows_to_cpu(p1["S0"]["chain"], 2),
                               _chain_rows_to_cpu(p1["BQO"]["chain"], 2), n_layers=2)
    h5 = SA.classify_h5_final(p1["S0"]["final_row"], p1["BQO"]["final_row"])
    mlp = SA.mlp_boundary_report(n_s0bqo, boundary_op=None)
    check("★ H2/H3/H4/H5 分类都能在真实（微型）数据上给出结构完整的结果",
          h2["class_"] in ("local_closure_ok", "o_proj_not_restored", "heads_differ")
          and h3["class_"] in ("L0_restored", "L0_not_restored") and len(h4["layers"]) == 2
          and h5["category"] in (1, 2, 3) and len(mlp["per_operator"]) == 6,
          (h2["class_"], h3["class_"], h4["first_layer_with_any_output_diff"], h5["category"]))
    print(f"  [obs] 微型模型：H2 {h2['class_']}；H3 {h3['class_']}（首个持续边界="
          f"{h3['first_persistent_divergence_operator']}）；H4 首个输出差异层="
          f"{h4['first_layer_with_any_output_diff']}；H5 category={h5['category']}")
    # ---- ⑤ 落盘 ---------------------------------------------------------------
    payload = _build_payload(p1, n_layers=2)
    tp = _save_pt(payload, run_id, selftest=True)
    check("★ 落盘 payload 可跑通（五臂节点/层行/最终行/两个分支的逐行记录/哨兵结果都在）",
          tp.exists() and tp.stat().st_size > 0
          and all(f"nodes_{a}" in payload for a in ("S0", "B0", "BQ", "BQO", "S1"))
          and payload["serial_lin_BQO"][0]["per_row_input_shapes"] == [[1, 1, 32]] * 4
          and payload["serial_qk_BQO"][0]["rows"][0]["raw"].shape[-1] == 6, str(tp))
    check("★ 捕获/干预结束后：探针关闭、两个分支关闭、层钩子全部移除",
          M.attn_probe_state()["active"] is False and M.serial_qk_state()["active"] is False
          and M.serial_linear_state()["active"] is False
          and all(len(tgt.layers[i]._forward_pre_hooks) == 0 and len(tgt.layers[i]._forward_hooks) == 0
                  for i in range(2)))
    print(f"\nQK + o_proj 联合对齐工具（CPU 门）：PASS {npass} / FAIL {nfail}")
    rec["selftest"] = dict(npass=npass, nfail=nfail,
                           semantics="非真实运行：微型模型上验证真实 forward 的双干预链路"
                                     "（关闭等价/嵌套/只作用指定模块/逐行输入形状/异常恢复/H1–H5/落盘）",
                           pt=str(tp))
    rec["stage"] = "selftest"
    _write(rec, run_id, selftest=True)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
