"""serial_mlp_l0_f13_p8209.py — L0 的**全部持续 GEMM 联合对齐**（F13；只读、有界、单次）。

## 本轮唯一目标（L0 逐算子闭环的**最后一次**）
在保留 L0 `serial-causal QK` + `serial o_proj` 的基础上，继续把 L0 MLP 的
`gate_proj` / `up_proj` / `down_proj` 按行以 **M=1** 调用**同一个 `nn.Linear`**，判定：
① gate/up 对齐后 MLP 中间状态是否恢复；② `down_proj` 是否新的独立边界；
③ 全部持续 GEMM 对齐后 **L0 输出能否逐位恢复**；④ 若恢复，后续**第一处**重新分歧的层；
⑤ 最终 logits/token 如何变化。

**不得**进入 L1–L27 的算子级诊断（若 L0 恢复，只记录**第一处**分歧层，不深入其内部）。

## 六个 arm（同一份 `H(8208)`，各自独立克隆 cache，串行执行）
| arm | 输入 | serial QK | serial o_proj | serial MLP（逐行 M=1） |
| --- | --- | --- | --- | --- |
| **S0** | `[20642]` | 关 | 关 | 关 |
| **B0** | `[20642,659,576,4531]` | 关 | 关 | 关 |
| **BQO** | 同 B0 | 开 | 开 | 关（须逐位复现上一轮） |
| **BQOGU** | 同 B0 | 开 | 开 | **gate + up** |
| **BQOGUD** | 同 B0 | 开 | 开 | **gate + up + down** |
| **S1** | 同 S0 | 开 | 开 | **gate + up + down**（s=1 控制） |

## 干预定义（与 `serial_linear_for` 多目标一致）
对目标 Linear 的 block 输入 `x [1,4,Din]`：`rows=[module(x[:, i:i+1, :]) for i in range(4)]; y=torch.cat(rows,dim=1)`
—— 每行**实际**以 `[1,1,Din]` 调用**原来的同一个 `nn.Linear`**；不先算 M=4 再切；不改 matmul/einsum/FP32/kernel；
拼接不做额外算术；dtype/权重/bias/模块实例不变；**对象身份精确匹配**（不用类名匹配全部模块）。

## 纪律
其他算子（Q/K/V projection、softmax、AV、SiLU、乘法、残差、mask、RoPE、cache、argmax）与后端**一律不改**；
三个诊断分支都**默认关闭**、实例级、异常安全；不做 profiler/FP32、不切后端、**不跑 B6**；GPU 串行；
**原始张量在任何摘要或打印之前先落盘**；硬失败落盘即停、不自动重试；旧产物不覆盖；**B6 状态不变**。
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

TAG = os.environ.get("SML_TAG", "div8209_sml")
MODE = os.environ.get("SML_MODE", "gpu").strip().lower()
LAYER = int(os.environ.get("SML_LAYER", "0"))
PRE = int(os.environ.get("SML_PRE", "8208"))
TARGET_POS = int(os.environ.get("SML_TARGET", "8209"))
K = int(os.environ.get("SML_K", "4"))
FEED = [int(x) for x in os.environ.get("SML_FEED", "20642").split(",")]
CANDS = [int(x) for x in os.environ.get("SML_CANDS", "20642,659,576,4531").split(",")]
WATCH = tuple(int(x) for x in os.environ.get("SML_WATCH", "323,659,20642,576,4531").split(","))
REPEAT = int(os.environ.get("SML_REPEAT", "2"))
TARGET_ROPE = os.environ.get("SML_TARGET_ROPE", "cpu_fp32").strip().lower()
POISON_CHECK = S.env_flag("SML_POISON_CHECK", True)
CAPTURE_OFF_CHECK = S.env_flag("SML_CAPTURE_OFF_CHECK", True)
CAP_H = os.environ.get("SML_CAP_H", str(OUTDIR / "divcap_ar_20261001T185503Z.pt"))
CAP_H2 = os.environ.get("SML_CAP_H2", str(OUTDIR / "divasc_rebuild_20261001T220737Z.pt"))
L0OPS_PT = os.environ.get("SML_L0OPS_PT", str(OUTDIR / "div8209_l0ops_20261002T172447Z.pt"))
PREV_PT = os.environ.get("SML_PREV_PT", str(OUTDIR / "div8209_sla_20261002T192540Z.pt"))
PREV_JSON = os.environ.get("SML_PREV_JSON", str(OUTDIR / "div8209_sla_20261002T192540Z.json"))
SAVED_PT = os.environ.get("SML_SAVED_PT", str(OUTDIR / "div8209_cells_20261002T064415Z.pt"))
REF_RUN_AR = os.environ.get("SML_REF_AR", str(OUTDIR / "smoke8k_eager_20261001T185503Z.json"))
REF_RUN_RB = os.environ.get("SML_REF_RB", str(OUTDIR / "smoke8k_eager_20261001T220737Z.json"))
KV, ARM = "base_bf16", os.environ.get("SML_ARM", "eager")
VIS = PRE + 1
S_ARMS = ("S0", "S1")
# (arm, phase, serial QK, serial o_proj, serial MLP: None | "gu" | "gud")
PLAN = (("S0", "decode", False, False, None),
        ("B0", "verify", False, False, None),
        ("BQO", "verify", True, True, None),
        ("BQOGU", "verify", True, True, "gu"),
        ("BQOGUD", "verify", True, True, "gud"),
        ("S1", "decode", True, True, "gud"))
ARM_NAMES = [p[0] for p in PLAN]


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
    """决策行的 argmax 与 **top-2**（token、值、margin）——用原始 dtype 的 float 视图计算。"""
    f = row.float().reshape(-1)
    v, idx = torch.topk(f, 2)
    return dict(argmax_id=int(f.argmax()), argmax_value=float(f.max()),
                top1_id=int(idx[0]), top1_value=float(v[0]),
                top2_id=int(idx[1]), top2_value=float(v[1]),
                margin=float(v[0] - v[1]), dtype=str(row.dtype), vocab=int(f.numel()),
                sha16=A.stable_sha(row))


def _row_cmp(a, b, name: str) -> dict:
    a2, b2 = a.float().reshape(-1), b.float().reshape(-1)
    DC.assert_same_shape(a2, b2, name)
    d = (a2 - b2).abs()
    return dict(name=name, exact_equal=bool(torch.equal(a.reshape(-1), b.reshape(-1))),
                n_diff=int((a.reshape(-1) != b.reshape(-1)).sum()),
                abs_err_max=float(d.max()), abs_err_mean=float(d.mean()),
                sha16_a=A.stable_sha(a), sha16_b=A.stable_sha(b), dtype=str(a.dtype),
                shape=[int(x) for x in a.shape])


def _cmp_all(a_nodes: dict, b_nodes: dict) -> dict:
    out = {}
    for name, (who, kind) in OP.SPECS.items():
        if kind == "shape_only":
            continue
        out[name] = OP.cmp_node(name, kind, a_nodes["probes"]["records"][name]["tensor"],
                                b_nodes["probes"]["records"][name]["tensor"], vis=VIS)
    return out


def _cmp_all_B(a_nodes: dict, b_nodes: dict) -> dict:
    """B 口径 vs B 口径（两侧都取第 0 行）。**必须用 if/elif**：dict 字面量会把所有分支求值。"""
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
        a, b = sl(kind, a_nodes["probes"]["records"][name]["tensor"]), \
            sl(kind, b_nodes["probes"]["records"][name]["tensor"])
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
    return None if y is None else (y[0, row_index] if y.dim() == 3 else y).detach().cpu()


def _norm_in_row(norm_caps: dict, row_index: int):
    caps = norm_caps.get("caps") or {}
    if not caps:
        return None
    rec = caps[max(caps)]
    x = rec.get("x_in")
    return None if x is None else (x[0, row_index] if x.dim() == 3 else x).detach().cpu()


def _row(nodes_arm: dict, name: str, which_end: int):
    """探针节点在**决策行**上的切片（S 臂用 `-1`、B 臂用 `0`）。模块级，供 `_run` 与 CPU 自测共用。"""
    return nodes_arm["probes"]["records"][name]["tensor"][0, which_end]


def _mlp_modules(tgt, layer: int, which):
    mlp = tgt.layers[layer].mlp
    if which is None:
        return ()
    if which == "gu":
        return (mlp.gate_proj, mlp.up_proj)
    if which == "gud":
        return (mlp.gate_proj, mlp.up_proj, mlp.down_proj)
    raise ValueError(f"未知 MLP 干预组合：{which}")


def _capture_arm(tgt, H, dev, arm: str, mode: str, tag: str, *, qk: bool, oproj: bool, mlp,
                 poison: bool = False) -> dict:
    cache = _to_dev(H, dev)
    sa0 = tgt.layers[LAYER].self_attn
    lin_mods = tuple(([sa0.o_proj] if oproj else []) + list(_mlp_modules(tgt, LAYER, mlp)))
    ctx_qk = M.serial_qk_for(sa0, label=arm) if qk else contextlib.nullcontext(None)
    ctx_lin = M.serial_linear_for(*lin_mods, label=arm) if lin_mods else contextlib.nullcontext(None)
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
        sq_calls = M.serial_qk_calls() if qk else []
        sl_calls = M.serial_linear_calls() if lin_mods else []
        sq_state = M.serial_qk_state() if qk else None
        sl_state = M.serial_linear_state() if lin_mods else None
        poison_res = None
        if poison and qk and sq_calls:
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
    return dict(arm=arm, mode=mode, row_index=int(row_index), qk=bool(qk), oproj=bool(oproj),
                mlp=mlp, n_lin_mods=len(lin_mods), chain=chain, norm_caps=norm_caps,
                norm_info=norm_info, probes=probes, final_row=row.detach().to("cpu"),
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
                        contiguous_rows=bool(meta["contiguous_rows"]), scores=c["scaled"].detach().cpu(),
                        rows=[dict(i=int(r["i"]), visible=int(r["visible"]),
                                   raw=r["raw"].detach().cpu(), scaled=r["scaled"].detach().cpu(),
                                   layout_q_used=r["layout_q_used"], layout_k_used=r["layout_k_used"],
                                   finite=bool(r["finite"])) for r in meta["rows"]]))
    return out


def _serial_lin_to_cpu(sl_calls: list) -> list:
    out = []
    for c in sl_calls:
        meta = c["meta"]
        out.append(dict(module_id=int(id(c["module"])), module_label=f"{type(c['module']).__name__}",
                        s=int(meta["s"]), cat_dim=int(meta["cat_dim"]),
                        contiguous_rows=bool(meta["contiguous_rows"]),
                        per_row_input_shapes=list(meta["per_row_input_shapes"]),
                        out_features=int(meta["out_features"]),
                        rows=[dict(r) for r in meta["rows"]], out=c["out"].detach().cpu()))
    return out


def _build_payload(arms: dict, *, n_layers: int, module_facts: dict) -> dict:
    pay: dict = dict(meta=dict(vis=VIS, layer=LAYER, n_layers=n_layers, arms=sorted(arms)),
                     module_facts=module_facts)
    for a, r in arms.items():
        pay[f"nodes_{a}"] = {k: v["tensor"].detach().cpu()
                             for k, v in r["probes"]["records"].items() if not v.get("shape_only")}
        pay[f"layouts_{a}"] = {k: v["layout"] for k, v in r["probes"]["records"].items()}
        pay[f"rows_{a}"] = _chain_rows_to_cpu(r["chain"], n_layers)
        pay[f"norm_row_{a}"] = _norm_row(r["norm_caps"], r["row_index"])
        pay[f"norm_in_row_{a}"] = _norm_in_row(r["norm_caps"], r["row_index"])
        pay[f"final_row_{a}"] = r["final_row"].detach().cpu()
        pay[f"serial_qk_{a}"] = _serial_qk_to_cpu(r["sq_calls"]) if r["qk"] else None
        pay[f"serial_lin_{a}"] = _serial_lin_to_cpu(r["sl_calls"]) if r["n_lin_mods"] else None
        pay[f"poison_{a}"] = r["poison"]
    pay["note"] = ("`nodes_*` = L0 探针捕获的完整张量（原始 dtype）；`rows_*` = 28 层输入/输出决策行；"
                   "`norm_in_row_*`/`norm_row_*` = 最终 RMSNorm 输入/输出决策行；`final_row_*` = 完整词表 logits；"
                   "`serial_qk_*`/`serial_lin_*` = 诊断分支的逐行原始记录；`poison_*` = 不读未来 key 的哨兵；"
                   "`module_facts` = 从实际模块读出的 gate/up/down/o_proj 事实。")
    return pay


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="serial_mlp_l0_f13_p8209", family="F13", kv=KV, arm=ARM,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               question="在保留 L0 QK + o_proj 对齐的基础上，再把 L0 MLP 的 gate/up/down 按行以 M=1 调用同一 "
                        "nn.Linear，MLP 中间状态与 L0 输出能否逐位恢复？down_proj 是否新的独立边界？",
               semantics="三个**默认关闭**的实例级诊断分支（逐行 M=1 的 causal-prefix QK、逐行 M=1 的 o_proj、"
                         "逐行 M=1 的 MLP 投影，多目标共用同一注册表），六臂串行、同一份 H(8208)。",
               config=dict(tag=TAG, layer=LAYER, pre=PRE, target_pos=TARGET_POS, k=K, feed=FEED,
                           cands=CANDS, watch=list(WATCH), repeat=REPEAT, vis=VIS,
                           target_rope=TARGET_ROPE, poison_check=POISON_CHECK,
                           capture_off_check=CAPTURE_OFF_CHECK, cap_h=str(CAP_H), cap_h2=str(CAP_H2),
                           l0ops_pt=str(L0OPS_PT), prev_pt=str(PREV_PT), prev_json=str(PREV_JSON),
                           saved_pt=str(SAVED_PT), ref_ar=str(REF_RUN_AR), ref_rb=str(REF_RUN_RB),
                           plan=[dict(arm=a, mode=m, qk=q, o_proj=o, mlp=x) for a, m, q, o, x in PLAN],
                           in_scope="只干预 L0 的 QK / o_proj / gate_proj / up_proj / down_proj；"
                                    "不进 L1–L27 算子级诊断；不做 profiler/FP32；不跑 B6"))
    print(f"[sml] run_id={run_id} layer={LAYER} pre={PRE} target={TARGET_POS} vis={VIS} repeat={REPEAT}",
          flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    if MODE == "selftest":
        return _selftest(rec, run_id)
    ok, why = P5V.commit_guard()
    rec["commit_guard"] = dict(ok=bool(ok), why=why)
    print(f"[sml] commit: {why}")
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
        sa0, mlp0 = tgt.layers[LAYER].self_attn, tgt.layers[LAYER].mlp
        facts = dict(o_proj=SA.module_facts(sa0.o_proj), gate_proj=SA.module_facts(mlp0.gate_proj),
                     up_proj=SA.module_facts(mlp0.up_proj), down_proj=SA.module_facts(mlp0.down_proj),
                     q_proj=SA.module_facts(sa0.q_proj), k_proj=SA.module_facts(sa0.k_proj),
                     v_proj=SA.module_facts(sa0.v_proj))
        rec["module_facts"] = facts
        rec["probe_state_before"] = M.attn_probe_state()
        rec["serial_state_before"] = dict(qk=M.serial_qk_state(), linear=M.serial_linear_state())
        DC.require(rec["probe_state_before"]["active"] is False, "进入实验前探针必须关闭")
        DC.require(rec["serial_state_before"]["qk"]["active"] is False
                   and rec["serial_state_before"]["linear"]["active"] is False,
                   "进入实验前三个诊断分支都必须关闭")
        DC.require(n_layers >= 2 and LAYER == 0, "本轮只做 L0", f"n_layers={n_layers}")
        DC.require(facts["gate_proj"]["out_features"] == facts["up_proj"]["out_features"]
                   and facts["down_proj"]["in_features"] == facts["gate_proj"]["out_features"]
                   and facts["down_proj"]["out_features"] == facts["gate_proj"]["in_features"],
                   "MLP 三个投影的维度关系不符（gate/up: h→i；down: i→h）", str(facts))
        rec["W_before"] = LB.weights_info(tgt)
        print(f"[sml] gate={facts['gate_proj']['in_features']}→{facts['gate_proj']['out_features']} "
              f"up={facts['up_proj']['in_features']}→{facts['up_proj']['out_features']} "
              f"down={facts['down_proj']['in_features']}→{facts['down_proj']['out_features']}"
              f"（bias: {[facts[k]['has_bias'] for k in ('gate_proj', 'up_proj', 'down_proj')]}）", flush=True)
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
        print(f"[sml] H：{Path(CAP_H).name} {src_len} → {PRE}（纯切片={tinfo['is_pure_slice']}）", flush=True)

        # ---------------- 六臂 × REPEAT ----------------
        passes = []
        for it in range(REPEAT):
            arms = {}
            for arm, mode, qk, oproj, mlp in PLAN:
                arms[arm] = _capture_arm(tgt, H, dev, arm, mode, f"p{it + 1}", qk=qk, oproj=oproj,
                                         mlp=mlp, poison=(POISON_CHECK and arm == "BQOGUD"))
                print(f"[sml] pass {it + 1}/{REPEAT} arm {arm} ✔（探针={arms[arm]['probes']['n_calls']}/"
                      f"{arms[arm]['probes']['n_kept']}；QK={len(arms[arm]['sq_calls'])} "
                      f"Linear={len(arms[arm]['sl_calls'])}/{arms[arm]['n_lin_mods']}）", flush=True)
            passes.append(arms)
        P1, P2 = passes[0], (passes[1] if len(passes) > 1 else None)
        DC.require(M.attn_probe_state()["active"] is False, "捕获后探针仍在位")
        DC.require(M.serial_qk_state()["active"] is False and M.serial_linear_state()["active"] is False,
                   "捕获后诊断分支仍开启")
        rec["serial_state_after"] = dict(qk=M.serial_qk_state(), linear=M.serial_linear_state())
        rep = dict(n_passes=len(passes))
        if P2 is not None:
            for a in ARM_NAMES:
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
                   "两次独立捕获不一致（含控制 15）", str(rep))
        print("[sml] 重复一致 ✔（六臂 × 2 pass 的节点/最终行/层行都逐位相同）", flush=True)

        # ★★★ 原始张量在任何摘要/打印之前先落盘（控制 24）★★★
        payload = _build_payload({a: P1[a] for a in ARM_NAMES}, n_layers=n_layers, module_facts=facts)
        payload["H_L0_cache_pre"] = dict(k=H[LAYER]["k"].detach().cpu(), v=H[LAYER]["v"].detach().cpu())
        tp = _save_pt(payload, run_id)
        rec["artifacts"] = dict(json=str(_write(rec, run_id)), pt=str(tp))
        print(f"[sml] 原始张量已落盘（早于任何比较/打印）：{tp.name}", flush=True)

        S0, B0, BQO, BQOGU, BQOGUD, S1 = (P1[a] for a in ARM_NAMES)

        # ---------------- 参考产物 ----------------
        l0ops = torch.load(L0OPS_PT, map_location="cpu") if Path(L0OPS_PT).exists() else {}
        prev = torch.load(PREV_PT, map_location="cpu") if Path(PREV_PT).exists() else {}
        prev_j = json.loads(Path(PREV_JSON).read_text(encoding="utf-8")) if Path(PREV_JSON).exists() else {}
        saved_rows = {}
        if Path(SAVED_PT).exists():
            saved_rows = {k: v for k, v in torch.load(SAVED_PT, map_location="cpu").items()
                          if torch.is_tensor(v)}
        anchors_s = C._anchor_ar(REF_RUN_AR, TARGET_POS)
        anchors_b = C._anchor_specdec(REF_RUN_RB, PRE, TARGET_POS - PRE, TARGET_POS)

        n_s0b0 = _cmp_all(S0, B0)
        n_s0bqo = _cmp_all(S0, BQO)
        n_s0bqogu = _cmp_all(S0, BQOGU)
        n_s0bqogud = _cmp_all(S0, BQOGUD)
        cache_pre = dict(exact_equal=True, n_diff=0,
                         cache_len_before={a: P1[a]["cache_len_before"] for a in ARM_NAMES},
                         note="六臂都从同一份 H(8208) 的 L0 缓存独立克隆出发（长度 8208）")
        pers_gu = SA.chain_persistence(dict(n_s0bqogu, cache_pre=cache_pre), OP.CHAIN, OP.CHAIN_NODE_OUT)
        pers_gud = SA.chain_persistence(dict(n_s0bqogud, cache_pre=cache_pre), OP.CHAIN, OP.CHAIN_NODE_OUT)

        # 决策行切片统一用**模块级** `_row(nodes_arm, name, end)`（S 臂 -1、B 臂 0）

        # ---- H1 ----
        h1 = dict(
            S0_history=dict(
                final_row=(_row_cmp(S0["final_row"], l0ops["final_row_S"], "S0_final_vs_history")
                           if "final_row_S" in l0ops else None),
                L0_output=(_row_cmp(_row(S0, "layer_output", -1), l0ops["L0_output_row_S"],
                                    "S0_L0out_vs_history") if "L0_output_row_S" in l0ops else None),
                qk_scores=(_row_cmp(S0["probes"]["records"]["qk_scores_pre_mask"]["tensor"],
                                    l0ops["raw_S"]["qk_scores_pre_mask"], "S0_qk_vs_history") if l0ops else None),
                x0=(_row_cmp(S0["x0_row"], l0ops["x0_row_S"], "S0_x0_vs_history") if "x0_row_S" in l0ops else None)),
            B0_history=dict(
                final_row=(_row_cmp(B0["final_row"], l0ops["final_row_B"], "B0_final_vs_history")
                           if "final_row_B" in l0ops else None),
                L0_output=(_row_cmp(_row(B0, "layer_output", 0), l0ops["L0_output_row_B"],
                                    "B0_L0out_vs_history") if "L0_output_row_B" in l0ops else None),
                qk_scores=(_row_cmp(B0["probes"]["records"]["qk_scores_pre_mask"]["tensor"],
                                    l0ops["raw_B"]["qk_scores_pre_mask"], "B0_qk_vs_history") if l0ops else None),
                x0=(_row_cmp(B0["x0_row"], l0ops["x0_row_B"], "B0_x0_vs_history") if "x0_row_B" in l0ops else None)),
            BQO_prev_round=dict(
                nodes_all_identical=bool(prev and all(torch.equal(BQO["probes"]["records"][k]["tensor"],
                                                                  prev["nodes_BQO"][k])
                                                      for k in prev["nodes_BQO"]
                                                      if k in BQO["probes"]["records"])),
                final_row=(_row_cmp(BQO["final_row"], prev["final_row_BQO"], "BQO_final_vs_prev")
                           if "final_row_BQO" in prev else None),
                layer_rows_identical=bool(prev and all(
                    torch.equal(_chain_rows_to_cpu(BQO["chain"], n_layers)[i][kk], prev["rows_BQO"][i][kk])
                    for i in range(n_layers) for kk in ("input", "output"))),
                L0_output=(_row_cmp(_row(BQO, "layer_output", 0), prev["nodes_BQO"]["layer_output"][0, 0],
                                    "BQO_L0out_vs_prev") if prev else None),
                prev_expected=dict(L0_output_n_diff=(prev_j.get("H3") or {}).get("n_diff"),
                                   token=(((prev_j.get("controls") or {}).get("BQO_final") or {})
                                          .get("argmax_id")),
                                   logits_n_diff=(prev_j.get("H5") or {}).get("logits_n_diff"))),
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
                applied=dict(qk=int(len(S1["sq_calls"])), linear=int(len(S1["sl_calls"])))))
        rec["H1"] = h1

        # ---- H2：gate/up 局部闭环（在 BQOGU 上判定）----
        h2 = dict(
            post_attention_rmsnorm=_row_cmp(_row(S0, "post_attention_rmsnorm", -1),
                                            _row(BQOGU, "post_attention_rmsnorm", 0), "H2_post_attn_norm"),
            gate_proj=_row_cmp(_row(S0, "mlp_gate_proj", -1), _row(BQOGU, "mlp_gate_proj", 0), "H2_gate"),
            up_proj=_row_cmp(_row(S0, "mlp_up_proj", -1), _row(BQOGU, "mlp_up_proj", 0), "H2_up"),
            silu=_row_cmp(_row(S0, "mlp_activation", -1), _row(BQOGU, "mlp_activation", 0), "H2_silu"),
            gated_product=_row_cmp(_row(S0, "mlp_gated_product", -1), _row(BQOGU, "mlp_gated_product", 0),
                                   "H2_gated"),
            reference_BQO=dict(
                gate_proj=_row_cmp(_row(S0, "mlp_gate_proj", -1), _row(BQO, "mlp_gate_proj", 0), "BQO_gate"),
                up_proj=_row_cmp(_row(S0, "mlp_up_proj", -1), _row(BQO, "mlp_up_proj", 0), "BQO_up"),
                gated_product=_row_cmp(_row(S0, "mlp_gated_product", -1), _row(BQO, "mlp_gated_product", 0),
                                       "BQO_gated")))
        h2["class_"] = ("gate_up_local_closure_ok"
                        if all(h2[k]["exact_equal"] for k in ("post_attention_rmsnorm", "gate_proj",
                                                              "up_proj", "silu", "gated_product"))
                        else "gate_up_not_restored")
        rec["H2"] = h2

        # ---- H3：down_proj 独立边界（在 BQOGU 上判定）----
        gp_gu = h2["gated_product"]
        down_gu = _row_cmp(_row(S0, "mlp_down_proj", -1), _row(BQOGU, "mlp_down_proj", 0), "H3_down_BQOGU")
        h3 = dict(gated_product=dict(row=gp_gu),
                  down_proj=dict(row=down_gu),
                  down_proj_class=("independent_boundary" if (gp_gu["exact_equal"] and not down_gu["exact_equal"])
                                   else ("both_equal" if (gp_gu["exact_equal"] and down_gu["exact_equal"]) else
                                         "gate_up_closure_failed_inputs_differ")),
                  reference_BQO=dict(down_proj=_row_cmp(_row(S0, "mlp_down_proj", -1),
                                                        _row(BQO, "mlp_down_proj", 0), "BQO_down")))
        rec["H3"] = h3

        # ---- H4：完整 L0 是否恢复 ----
        l0_s0, l0_b0 = _row(S0, "layer_output", -1), _row(B0, "layer_output", 0)
        l0_bqo, l0_gu, l0_gud = (_row(BQO, "layer_output", 0), _row(BQOGU, "layer_output", 0),
                                 _row(BQOGUD, "layer_output", 0))
        h4 = dict(S0_vs_BQOGUD=_row_cmp(l0_s0, l0_gud, "H4_L0out_S0_vs_BQOGUD"),
                  S0_vs_BQOGU=_row_cmp(l0_s0, l0_gu, "H4_L0out_S0_vs_BQOGU"),
                  S0_vs_BQO=_row_cmp(l0_s0, l0_bqo, "H4_L0out_S0_vs_BQO"),
                  S0_vs_B0=_row_cmp(l0_s0, l0_b0, "H4_L0out_S0_vs_B0"),
                  trajectory=dict(B0=int((l0_s0 != l0_b0).sum()), BQO=int((l0_s0 != l0_bqo).sum()),
                                  BQOGU=int((l0_s0 != l0_gu).sum()), BQOGUD=int((l0_s0 != l0_gud).sum())),
                  persistence_bqogud=pers_gud)
        h4["class_"] = "L0_restored" if h4["S0_vs_BQOGUD"]["exact_equal"] else "L0_not_restored"
        rows_s0 = _chain_rows_to_cpu(S0["chain"], n_layers)
        h4["BQOGUD_L1_input_equals_S0_L1_input"] = bool(torch.equal(rows_s0[1]["input"],
                                                                    _chain_rows_to_cpu(BQOGUD["chain"],
                                                                                       n_layers)[1]["input"]))
        rec["H4"] = h4

        # ---- H5：后续层第一处重新分歧（仅当 L0 恢复；不进入层内部）----
        h5 = SA.classify_h4_layers(rows_s0, _chain_rows_to_cpu(BQOGUD["chain"], n_layers),
                                   n_layers=n_layers)
        h5_bqo = SA.classify_h4_layers(rows_s0, _chain_rows_to_cpu(BQO["chain"], n_layers),
                                       n_layers=n_layers)
        h5["applicable"] = bool(h4["class_"] == "L0_restored")
        h5["note"] = ("仅当 L0 完全恢复时，`first_layer_with_equal_input_and_different_output` 才是"
                      "『后续第一处重新分歧层』；否则只作为逐层 `n_diff` 轨迹如实记录，**不**对输入已不同的层归因")
        rec["H5"] = h5

        # ---- H6：最终结果（S0 vs B0 / BQO / BQOGU / BQOGUD）----
        h6 = {}
        for a, label in ((B0, "B0"), (BQO, "BQO"), (BQOGU, "BQOGU"), (BQOGUD, "BQOGUD")):
            c = SA.classify_h5_final(S0["final_row"], a["final_row"])
            c["fields_s0"] = _token_fields(S0["final_row"])
            c["fields_x"] = _token_fields(a["final_row"])
            h6[label] = c
        rec["H6"] = h6

        # ---------------- 必需控制（24 项）----------------
        controls: dict = {}
        controls["c1_S0_reproduces_history"] = h1["S0_history"]
        controls["c2_B0_reproduces_history"] = h1["B0_history"]
        controls["c3_BQO_reproduces_prev_round"] = h1["BQO_prev_round"]
        controls["c4_S1_equals_S0"] = h1["S1_vs_S0"]
        qk_row0 = lambda arm: _row_cmp(arm["probes"]["records"]["qk_scores_pre_mask"]["tensor"][0, :, 0, :VIS],
                                       S0["probes"]["records"]["qk_scores_pre_mask"]["tensor"][0, :, 0, :VIS],
                                       "qk_row0_vs_S0")
        controls["c5_qk_row0_equals_S0"] = {a: qk_row0(x) for a, x in
                                            (("BQOGU", BQOGU), ("BQOGUD", BQOGUD))}
        attn_nodes = ("qk_scores_pre_mask", "qk_scores_post_mask", "attention_times_v", "heads_merged",
                      "o_proj", "attention_residual", "post_attention_rmsnorm")
        controls["c6_attention_subblock_rows_equals_S0"] = {
            a: {n: bool(n_s0bqogud[n]["exact_equal"] if a == "BQOGUD" else n_s0bqogu[n]["exact_equal"])
                for n in attn_nodes} for a in ("BQOGU", "BQOGUD")}
        sl_gud = _serial_lin_to_cpu(BQOGUD["sl_calls"])
        sl_gu = _serial_lin_to_cpu(BQOGU["sl_calls"])
        by_id = {int(id(mlp0.gate_proj)): "gate_proj", int(id(mlp0.up_proj)): "up_proj",
                 int(id(mlp0.down_proj)): "down_proj", int(id(sa0.o_proj)): "o_proj"}
        shapes_by_mod = {}
        for c in sl_gud:
            shapes_by_mod[by_id.get(c["module_id"], f"unknown{c['module_id']}")] = c["per_row_input_shapes"]
        controls["c7_c9_per_row_input_shapes"] = dict(
            BQOGU={by_id.get(c["module_id"], "?"): c["per_row_input_shapes"] for c in sl_gu},
            BQOGUD=shapes_by_mod,
            gate_ok=bool(shapes_by_mod.get("gate_proj") == [[1, 1, facts["gate_proj"]["in_features"]]] * 4),
            up_ok=bool(shapes_by_mod.get("up_proj") == [[1, 1, facts["up_proj"]["in_features"]]] * 4),
            down_ok=bool(shapes_by_mod.get("down_proj") == [[1, 1, facts["down_proj"]["in_features"]]] * 4),
            o_proj_ok=bool(shapes_by_mod.get("o_proj") == [[1, 1, facts["o_proj"]["in_features"]]] * 4))
        facts_after = dict(gate_proj=SA.module_facts(mlp0.gate_proj), up_proj=SA.module_facts(mlp0.up_proj),
                           down_proj=SA.module_facts(mlp0.down_proj), o_proj=SA.module_facts(sa0.o_proj))
        controls["c10_same_modules_weights_dtype"] = dict(
            same_module_ids=bool(all(facts[k]["module_id"] == facts_after[k]["module_id"] for k in facts_after)),
            same_weight_ptrs=bool(all(facts[k]["weight_data_ptr"] == facts_after[k]["weight_data_ptr"]
                                      for k in facts_after)),
            no_bias=bool(all(facts_after[k]["has_bias"] is False for k in ("gate_proj", "up_proj", "down_proj"))),
            weight_dtypes=sorted({facts_after[k]["weight_dtype"] for k in facts_after}),
            call_module_ids=sorted({c["module_id"] for c in sl_gud}),
            call_ids_are_L0=bool(sorted({c["module_id"] for c in sl_gud})
                                 == sorted(int(id(m)) for m in (sa0.o_proj, mlp0.gate_proj,
                                                                mlp0.up_proj, mlp0.down_proj))),
            row_input_dtypes=sorted({r["input_layout"]["dtype"] for c in sl_gud for r in c["rows"]}))
        controls["c11_gate_up_rows_equal_S0"] = dict(
            gate=h2["gate_proj"], up=h2["up_proj"], silu=h2["silu"], gated=h2["gated_product"])
        controls["c12_down_row_equal_S0"] = _row_cmp(_row(S0, "mlp_down_proj", -1),
                                                     _row(BQOGUD, "mlp_down_proj", 0), "BQOGUD_down_vs_S0")
        sl_state = BQOGUD["sl_state"] or {}
        applied_ids = [a["module_id"] for a in sl_state.get("applied", [])]
        seen_ids = list(sl_state.get("seen_module_ids", []))
        all_lin = [int(id(tgt.layers[i].mlp.gate_proj)) for i in range(n_layers)] + \
                  [int(id(tgt.layers[i].mlp.up_proj)) for i in range(n_layers)] + \
                  [int(id(tgt.layers[i].mlp.down_proj)) for i in range(n_layers)] + \
                  [int(id(tgt.layers[i].self_attn.o_proj)) for i in range(n_layers)]
        l0_registered = [int(id(m)) for m in (sa0.o_proj, mlp0.gate_proj, mlp0.up_proj, mlp0.down_proj)]
        controls["c13_only_L0_instances"] = dict(
            n_applied=int(sl_state.get("n_applied", -1)), n_seen=int(sl_state.get("n_seen", -1)),
            n_call_sites=int(len(all_lin)), applied_module_ids=sorted(applied_ids),
            applied_is_L0_set=bool(sorted(applied_ids) == sorted(l0_registered)),
            seen_is_all_others=bool(sorted(seen_ids) == sorted(set(all_lin) - set(l0_registered))),
            seen_plus_applied_covers_all=bool(sorted(seen_ids + applied_ids) == sorted(all_lin)),
            applied_labels=[a.get("module_label") for a in sl_state.get("applied", [])]) 
        controls["c14_other_layers_block_path"] = dict(
            n_applied=int(sl_state.get("n_applied", -1)), n_call_sites=int(len(all_lin)),
            note="只有 L0 的 4 个实例被替换；其余 108 个调用点进入分支后只执行一条计数语句，"
                 "随后仍走**原样的一次调用**（对象身份匹配，不用类名匹配）")
        controls["c15_two_BQOGUD_repeats_identical"] = {k: v for k, v in rep.items() if k.startswith("BQOGUD")}
        controls["c16_off_means_production_unchanged"] = dict(state_before=rec["serial_state_before"])
        controls["c17_hooks_and_context_restored"] = None
        controls["c18_cache_inputs_weights_unmodified"] = None
        controls["c19_finiteness"] = None
        controls["c20_allow_tf32_false"] = dict(before=False, after=None)
        controls["c21_draft_never_loaded"] = dict(loaded_models=["target"], draft_loaded=False,
                                                  note="本工具只 `load_ours_bf16(target)`")
        controls["c22_git_and_artifacts"] = dict(git=S.git_meta(ROOT), recorded=True)
        controls["c23_old_artifacts_not_overwritten"] = dict(
            prev_pt=str(PREV_PT), l0ops_pt=str(L0OPS_PT), saved_pt=str(SAVED_PT),
            note="新 tag `div8209_sml`；旧产物只读")
        controls["c24_raw_saved_before_summary"] = None      # §落盘顺序（用 mtime 佐证）
        controls["positions"] = dict(S_last_row=True, B_row0=True,
                                     both_predict_target=bool(S0["cache_len_after"] == TARGET_POS
                                                             and B0["cache_len_after"] == PRE + len(CANDS)),
                                     cache_lens_after={a: P1[a]["cache_len_after"] for a in ARM_NAMES},
                                     slot_for_B_row0=DR.slot_table(PRE, len(CANDS))[1])
        controls["tokens"] = {a: _token_fields(P1[a]["final_row"]) for a in ARM_NAMES}
        controls["final_norm_row"] = dict(
            S0_vs_BQOGUD=_row_cmp(_norm_row(S0["norm_caps"], -1), _norm_row(BQOGUD["norm_caps"], 0),
                                  "final_norm_S0_vs_BQOGUD"),
            final_norm_input_S0_vs_BQOGUD=_row_cmp(_norm_in_row(S0["norm_caps"], -1),
                                                   _norm_in_row(BQOGUD["norm_caps"], 0),
                                                   "final_norm_in_S0_vs_BQOGUD"))
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
        pv = controls["c3_BQO_reproduces_prev_round"]
        DC.require(pv["nodes_all_identical"] and pv["layer_rows_identical"]
                   and (pv["final_row"] is None or pv["final_row"]["exact_equal"])
                   and (pv["L0_output"] is None or pv["L0_output"]["exact_equal"]),
                   "控制 3 失败：BQO 未逐位复现上一轮", str(pv))
        DC.require(all(controls["c4_S1_equals_S0"][k] for k in
                       ("nodes_identical", "final_row_identical", "layer_rows_identical")),
                   "控制 4 失败：S1 与 S0 不逐位一致", str(controls["c4_S1_equals_S0"]))
        DC.require(all(v["exact_equal"] for v in controls["c5_qk_row0_equals_S0"].values()),
                   "控制 5 失败：BQOGU/BQOGUD 的 QK row0 未与 S0 逐位一致")
        DC.require(all(all(v.values()) for v in controls["c6_attention_subblock_rows_equals_S0"].values()),
                   "控制 6 失败：attention 子块决策行未继续与 S0 一致",
                   str(controls["c6_attention_subblock_rows_equals_S0"]))
        per_row = controls["c7_c9_per_row_input_shapes"]
        DC.require(per_row["gate_ok"] and per_row["up_ok"] and per_row["down_ok"] and per_row["o_proj_ok"],
                   "控制 7/8/9 失败：逐行输入形状不符", str(per_row))
        c10 = controls["c10_same_modules_weights_dtype"]
        DC.require(c10["same_module_ids"] and c10["same_weight_ptrs"] and c10["no_bias"]
                   and c10["call_ids_are_L0"] and c10["row_input_dtypes"] == ["torch.bfloat16"],
                   "控制 10 失败：模块/权重/dtype 不符", str(c10))
        DC.require(all(controls["c11_gate_up_rows_equal_S0"][k]["exact_equal"]
                       for k in ("gate", "up", "silu", "gated")),
                   "★ 控制 11 失败：gate/up/SiLU/gated 决策行未逐位恢复 ⇒ serial Linear 无效，停止解释",
                   str(controls["c11_gate_up_rows_equal_S0"]))
        DC.require(controls["c12_down_row_equal_S0"]["exact_equal"],
                   "★ 控制 12 失败：BQOGUD 的 down_proj 决策行未与 S0 逐位一致", str(controls["c12_down_row_equal_S0"]))
        c13 = controls["c13_only_L0_instances"]
        DC.require(c13["applied_is_L0_set"] and c13["seen_is_all_others"]
                   and c13["seen_plus_applied_covers_all"],
                   "控制 13/14 失败：干预未严格限定在 L0 的 4 个实例", str(c13))
        DC.require(controls["positions"]["both_predict_target"], "位置口径失败")
        if saved_rows:
            for k in ("S0", "B0"):
                DC.require(controls["vs_saved_local_rows"][k]["exact_equal"],
                           f"与已保存局部对照行不同（{k}）")
        print(f"[sml] 控制 1-14 ✔（S0/B0 复现历史；BQO ≡ 上一轮；S1≡S0；gate/up/SiLU/gated 逐位恢复="
              f"{h2['class_']}；down(在 BQOGUD)={controls['c12_down_row_equal_S0']['exact_equal']}；"
              f"applied={c13['n_applied']}（L0 的 4 个实例）、seen={c13['n_seen']}）", flush=True)

        # ---------------- 打印 H1–H6 ----------------
        print("== H1–H6 ==")
        print(f"  H1 S0/B0 复现历史；BQO ≡ 上一轮（节点={pv['nodes_all_identical']} "
              f"层行={pv['layer_rows_identical']}）；S1≡S0={controls['c4_S1_equals_S0']['nodes_identical']}")
        print(f"  H2 gate/up 局部闭环：{h2['class_']}（post-norm={h2['post_attention_rmsnorm']['exact_equal']} "
              f"gate={h2['gate_proj']['exact_equal']} up={h2['up_proj']['exact_equal']} "
              f"silu={h2['silu']['exact_equal']} gated={h2['gated_product']['exact_equal']}；"
              f"BQO 对照 gate n_diff={h2['reference_BQO']['gate_proj']['n_diff']}/up="
              f"{h2['reference_BQO']['up_proj']['n_diff']}/gated={h2['reference_BQO']['gated_product']['n_diff']}）",
              flush=True)
        print(f"  H3 down_proj（在 BQOGU 中）：{h3['down_proj_class']}"
              f"（gated 逐位相同={gp_gu['exact_equal']}；down n_diff={down_gu['n_diff']} "
              f"max|Δ|={down_gu['abs_err_max']:.6g}；BQO 对照 down n_diff="
              f"{h3['reference_BQO']['down_proj']['n_diff']}）", flush=True)
        print(f"  H4 L0 输出：{h4['class_']}（轨迹 B0={h4['trajectory']['B0']} → BQO={h4['trajectory']['BQO']} → "
              f"BQOGU={h4['trajectory']['BQOGU']} → BQOGUD={h4['trajectory']['BQOGUD']}；"
              f"BQOGUD L1 input == S0 L1 input：{h4['BQOGUD_L1_input_equals_S0_L1_input']}）", flush=True)
        if h5["applicable"]:
            print(f"  H5 后续第一处重新分歧层 = {h5['first_layer_with_equal_input_and_different_output']}"
                  f"（首个输出差异层 = {h5['first_layer_with_any_output_diff']}）", flush=True)
        else:
            print(f"  H5 L0 未恢复 ⇒ 只记录逐层 n_diff（首个输出差异层 = "
                  f"{h5['first_layer_with_any_output_diff']}），**不**对输入已不同的层归因", flush=True)
        for k, v in h6.items():
            print(f"  H6 S0 vs {k}：class={v['category']} logits_exact={v['logits_exact_equal']} "
                  f"n_diff={v['logits_n_diff']} max|Δ|={v['logits_max_abs']:.6g} "
                  f"mean|Δ|={v['logits_mean_abs']:.6g} token {v['token_s0']} vs {v['token_x']} "
                  f"margin {v['margin_s0']:.6g}/{v['margin_x']:.6g}", flush=True)
        print("== L0 逐算子（S0vsBQOGUD | S0vsBQOGU | S0vsBQO | S0vsB0 的 n_diff）==", flush=True)
        for (op, ins), out_node in zip(OP.CHAIN, OP.CHAIN_NODE_OUT):
            c1_, c2_, c3_, c4_ = (n_s0bqogud[out_node], n_s0bqogu[out_node], n_s0bqo[out_node],
                                  n_s0b0[out_node])
            print(f"  {op:<24s} out_eq={str(bool(c1_['exact_equal'])):<5s} n_diff={int(c1_['n_diff']):>6d} "
                  f"| GU {int(c2_['n_diff']):>6d} | BQO {int(c3_['n_diff']):>6d} | B0 {int(c4_['n_diff']):>6d}",
                  flush=True)

        # ---------------- 源完整性 / 有限性 / 落盘顺序 ----------------
        H_digest_after = [A.stable_sha(lay["k"]) for lay in H] + [A.stable_sha(lay["v"]) for lay in H]
        nan_nodes = [k for k, v in n_s0bqogud.items() if not v.get("finite_s", True) or not v.get("finite_b", True)]
        rec["finiteness"] = dict(nodes_with_nonfinite_slice=nan_nodes,
                                 logits_finite=bool(all(torch.isfinite(P1[a]["final_row"]).all()
                                                        for a in ARM_NAMES)),
                                 expected_inf=dict(
                                     BQOGUD=int(torch.isinf(BQOGUD["probes"]["records"]["qk_scores_post_mask"]["tensor"]).sum()),
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
            all_diag_weights_stable=bool(all(facts[k]["weight_data_ptr"] == facts_after[k]["weight_data_ptr"]
                                             for k in facts_after)),
            probe_state_now=M.attn_probe_state(),
            serial_qk_state_now=M.serial_qk_state(),
            serial_linear_state_now=M.serial_linear_state(),
            hooks_removed=bool(all(len(tgt.layers[i]._forward_pre_hooks) == 0
                                   and len(tgt.layers[i]._forward_hooks) == 0
                                   for i in range(n_layers)) and len(tgt.norm._forward_hooks) == 0),
            note="H 快照、全部权重、Norm 权重在全程不变；探针与三个诊断上下文都已关闭、层钩子全部移除")
        controls["c17_hooks_and_context_restored"] = dict(
            probe_active=rec["source_integrity"]["probe_state_now"]["active"],
            serial_qk_active=rec["source_integrity"]["serial_qk_state_now"]["active"],
            serial_linear_active=rec["source_integrity"]["serial_linear_state_now"]["active"],
            hooks_removed=rec["source_integrity"]["hooks_removed"])
        controls["c18_cache_inputs_weights_unmodified"] = dict(
            H_snapshot_unchanged=rec["source_integrity"]["H_snapshot_unchanged"],
            W_data_ptr_stable=rec["source_integrity"]["W_data_ptr_stable"],
            W_digest_stable=rec["source_integrity"]["W_digest_stable"],
            norm_weight_stable=rec["source_integrity"]["norm_weight_stable"],
            all_diag_weights_stable=rec["source_integrity"]["all_diag_weights_stable"],
            L0_cache_pre_len=int(H[LAYER]["k"].shape[2]),
            cache_pre_all_arms_equal=bool(len(set(P1[a]["cache_len_before"] for a in ARM_NAMES)) == 1))
        controls["c19_finiteness"] = rec["finiteness"]
        controls["c20_allow_tf32_false"] = dict(before=False, after=bool(torch.backends.cuda.matmul.allow_tf32))
        pt_m, js_m = tp.stat().st_mtime, Path(rec["artifacts"]["json"]).stat().st_mtime
        controls["c24_raw_saved_before_summary"] = dict(
            pt_mtime=pt_m, json_mtime=js_m, pt_written_first=bool(pt_m <= js_m),
            note="`_save_pt` 在六臂捕获 + 重复一致性之后、**任何比较/摘要/打印之前**执行；"
                 "JSON 在全部计算之后写（因此 .pt 的 mtime 早于 JSON）")
        for k, v in controls["c18_cache_inputs_weights_unmodified"].items():
            if isinstance(v, bool):
                DC.require(v, f"控制 18 失败：{k}")
        DC.require(controls["c17_hooks_and_context_restored"]["probe_active"] is False
                   and controls["c17_hooks_and_context_restored"]["serial_qk_active"] is False
                   and controls["c17_hooks_and_context_restored"]["serial_linear_active"] is False
                   and controls["c17_hooks_and_context_restored"]["hooks_removed"],
                   "控制 17 失败：钩子/上下文未恢复", str(controls["c17_hooks_and_context_restored"]))
        DC.require(controls["c20_allow_tf32_false"]["after"] is False, "控制 20 失败：allow_tf32 非 False")
        DC.require(controls["c24_raw_saved_before_summary"]["pt_written_first"],
                   "控制 24 失败：原始张量未在摘要之前落盘", str(controls["c24_raw_saved_before_summary"]))

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
                       "控制 16 失败：关闭后生产路径与 S0/B0 不同", str(off_checks))
            print("[sml] 控制 16 ✔（开关全关后 S/B 与 S0/B0 逐位一致）", flush=True)
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
        print(f"[sml] RESULT_JSON={p}")
        print(f"[sml] RESULT_PT={tp}")
        return 0
    except Exception as e:                                          # noqa: BLE001
        try:
            M.clear_attn_probe()
            M.clear_serial_qk()
            M.clear_serial_linear()
        except Exception:                                           # noqa: BLE001
            pass
        rec["stage"] = ("checks_failed" if isinstance(e, (DC.CheckError, DR.AlignError, A.AlignError,
                                                          CC.CloneError, LB.BoundaryError, OP.ProbeError))
                        else "crashed")
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        p = _write(rec, run_id)
        print(f"[sml] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[sml] RESULT_JSON={p}")
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
    """**不加载大模型**：微型模型上验证六臂链路、多目标 Linear、身份隔离与 H1–H6 分类。"""
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
    sa0, mlp0 = tgt.layers[0].self_attn, tgt.layers[0].mlp

    with torch.inference_mode():
        lg_ref, _ = tgt._run(tgt.embed_tokens(ids), None, need_logits=True)
    # ① 关闭 ⇒ 逐位不变；异常退出 ⇒ 恢复
    with M.serial_linear_for(sa0.o_proj, mlp0.gate_proj, mlp0.up_proj, mlp0.down_proj, label="t"):
        with torch.inference_mode():
            lg_on, _ = tgt._run(tgt.embed_tokens(ids), None, need_logits=True)
        calls_on = M.serial_linear_calls()
        st_on = M.serial_linear_state()
    with torch.inference_mode():
        lg_after, _ = tgt._run(tgt.embed_tokens(ids), None, need_logits=True)
    check("★ 关闭 ⇒ 生产路径逐位不变；退出后状态恢复",
          M.serial_linear_state()["active"] is False and torch.equal(lg_after, lg_ref))
    check("★ 多目标注册：一次 `serial_linear_for(4 个模块)` 全部生效（每层各调用一次）",
          st_on["n_mods"] == 4 and st_on["n_applied"] == 4 and len(calls_on) == 4
          and sorted(a["module_id"] for a in st_on["applied"]) ==
          sorted(int(id(m)) for m in (sa0.o_proj, mlp0.gate_proj, mlp0.up_proj, mlp0.down_proj)),
          st_on)
    dims = {int(id(mlp0.gate_proj)): [1, 1, 32], int(id(mlp0.up_proj)): [1, 1, 32],
            int(id(mlp0.down_proj)): [1, 1, 64], int(id(sa0.o_proj)): [1, 1, 32]}
    calls_cpu = _serial_lin_to_cpu(calls_on)          # 原始记录只有 `module`；CPU 版才有 `module_id`
    check("★ 每个目标模块的逐行输入形状符合各自 in_features（gate/up/o=32、down=64）",
          all(c["per_row_input_shapes"] == [dims[c["module_id"]]] * PRE for c in calls_cpu)
          and sorted(c["module_id"] for c in calls_cpu) == sorted(dims),
          [(c["module_id"], c["per_row_input_shapes"]) for c in calls_cpu])
    check("★ 身份隔离：未注册的 layer-1 对应模块被计数而不被替换（seen = 其余调用点）",
          st_on["n_seen"] == 4 and set(st_on["seen_module_ids"]) == {
              int(id(tgt.layers[1].self_attn.o_proj)), int(id(tgt.layers[1].mlp.gate_proj)),
              int(id(tgt.layers[1].mlp.up_proj)), int(id(tgt.layers[1].mlp.down_proj))},
          st_on["seen_module_ids"])
    # ② 嵌套 QK + 多 Linear
    with M.serial_qk_for(sa0, label="n"), \
            M.serial_linear_for(sa0.o_proj, mlp0.gate_proj, mlp0.up_proj, mlp0.down_proj, label="n"):
        with torch.inference_mode(), M.attention_phase("verify"):
            _rb, _ = spec_dec._verify(tgt, _to_dev(H, dev), CANDS)
        st_qk, st_lin = M.serial_qk_state(), M.serial_linear_state()
    check("★ QK 与多目标 Linear 三个 context 可嵌套，且各自只替换自己的模块",
          st_qk["n_applied"] == 1 and st_lin["n_applied"] == 4
          and M.serial_qk_state()["active"] is False and M.serial_linear_state()["active"] is False)
    for bad in (M.serial_qk_for(sa0), M.serial_linear_for(mlp0.down_proj)):
        try:
            with bad:
                raise RuntimeError("注入的干预故障")
        except RuntimeError:
            pass
    check("★ 异常退出也都恢复", M.serial_qk_state()["active"] is False
          and M.serial_linear_state()["active"] is False)
    # ③ 六臂
    p1 = {a: _capture_arm(tgt, H, dev, a, m, "s1", qk=q, oproj=o, mlp=x,
                          poison=(a == "BQOGUD"))
          for a, m, q, o, x in PLAN}
    check("★ 六臂捕获都能跑通；L0 探针节点齐全（含 MLP 三个投影）", all(
        all(k in p1[a]["probes"]["records"] for k in
            ("x0", "qk_scores_pre_mask", "heads_merged", "o_proj", "attention_residual",
             "post_attention_rmsnorm", "mlp_gate_proj", "mlp_up_proj", "mlp_activation",
             "mlp_gated_product", "mlp_down_proj", "layer_output")) for a in p1))
    with torch.inference_mode(), M.attention_phase("decode"):
        lg_native_s, _ = spec_dec._step(tgt, FEED[0], _to_dev(H, dev))
    with torch.inference_mode(), M.attention_phase("verify"):
        rows_native_b, _ = spec_dec._verify(tgt, _to_dev(H, dev), CANDS)
    check("★ 控制 16：S0/B0（开关全关）与原生前向逐位一致",
          torch.equal(p1["S0"]["final_row"], LB.row_of(lg_native_s, None).detach().cpu())
          and torch.equal(p1["B0"]["final_row"], LB.row_of(rows_native_b, 0).detach().cpu()))
    check("★ 控制 4：S1（全开、s=1）与 S0 逐位一致",
          all(torch.equal(p1["S1"]["probes"]["records"][k]["tensor"],
                          p1["S0"]["probes"]["records"][k]["tensor"])
              for k, v in p1["S0"]["probes"]["records"].items() if not v.get("shape_only"))
          and torch.equal(p1["S1"]["final_row"], p1["S0"]["final_row"])
          and len(p1["S1"]["sq_calls"]) == 1 and len(p1["S1"]["sl_calls"]) == 4)
    # ④ H1–H6 分类（真实微型数据）
    n_gu, n_gud = _cmp_all(p1["S0"], p1["BQOGU"]), _cmp_all(p1["S0"], p1["BQOGUD"])
    h2 = dict(gate_proj=_row_cmp(_row(p1["S0"], "mlp_gate_proj", -1), _row(p1["BQOGU"], "mlp_gate_proj", 0), "g"),
              up_proj=_row_cmp(_row(p1["S0"], "mlp_up_proj", -1), _row(p1["BQOGU"], "mlp_up_proj", 0), "u"))
    pers = SA.chain_persistence(dict(n_gud, cache_pre=dict(exact_equal=True)), OP.CHAIN, OP.CHAIN_NODE_OUT)
    h4 = dict(S0_vs_BQOGUD=_row_cmp(_row(p1["S0"], "layer_output", -1), _row(p1["BQOGUD"], "layer_output", 0),
                                    "l0"))
    h5 = SA.classify_h4_layers(_chain_rows_to_cpu(p1["S0"]["chain"], 2),
                               _chain_rows_to_cpu(p1["BQOGUD"]["chain"], 2), n_layers=2)
    h6 = SA.classify_h5_final(p1["S0"]["final_row"], p1["BQOGUD"]["final_row"])
    check("★ H2/H3/H4/H5/H6 分类都能在真实（微型）数据上给出结构完整的结果",
          isinstance(h2["gate_proj"]["exact_equal"], bool) and len(pers["per_operator"]) == 25
          and h4["S0_vs_BQOGUD"]["shape"] == [32] and len(h5["layers"]) == 2
          and h6["category"] in (1, 2, 3)
          and isinstance(_token_fields(p1["S0"]["final_row"])["margin"], float),
          (h4["S0_vs_BQOGUD"]["exact_equal"], h6["category"]))
    print(f"  [obs] 微型模型：gate/up 逐位恢复={h2['gate_proj']['exact_equal']}/{h2['up_proj']['exact_equal']}；"
          f"L0 逐位恢复={h4['S0_vs_BQOGUD']['exact_equal']}；首个持续边界="
          f"{pers['first_persistent_divergence_operator']}；H6 类别={h6['category']}")
    # ⑤ 落盘
    payload = _build_payload(p1, n_layers=2, module_facts=dict(gate=SA.module_facts(mlp0.gate_proj),
                                                               down=SA.module_facts(mlp0.down_proj)))
    tp = _save_pt(payload, run_id, selftest=True)
    sl_gud = payload["serial_lin_BQOGUD"]
    check("★ 落盘 payload 可跑通（六臂节点/层行/最终行/三个分支的逐行记录/哨兵/模块事实都在）",
          tp.exists() and tp.stat().st_size > 0
          and all(f"nodes_{a}" in payload for a in ARM_NAMES)
          and len(sl_gud) == 4
          and all(len(c["per_row_input_shapes"]) == len(CANDS) for c in sl_gud)
          and sorted(c["per_row_input_shapes"][0][2] for c in sl_gud) == [32, 32, 32, 64]
          and all(all(sh == [1, 1, c["per_row_input_shapes"][0][2]] for sh in c["per_row_input_shapes"])
                  for c in sl_gud), str(tp))
    check("★ 结束后：探针关、三个分支关、层钩子移除",
          M.attn_probe_state()["active"] is False and M.serial_qk_state()["active"] is False
          and M.serial_linear_state()["active"] is False
          and all(len(tgt.layers[i]._forward_pre_hooks) == 0 and len(tgt.layers[i]._forward_hooks) == 0
                  for i in range(2)))
    print(f"\nL0 全部持续 GEMM 对齐工具（CPU 门）：PASS {npass} / FAIL {nfail}")
    rec["selftest"] = dict(npass=npass, nfail=nfail,
                           semantics="非真实运行：微型模型上验证六臂链路、多目标 Linear（gud）、身份隔离、"
                                     "嵌套 context、异常恢复、H1–H6 分类与落盘",
                           pt=str(tp))
    rec["stage"] = "selftest"
    _write(rec, run_id, selftest=True)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
