"""attention_backend_shape_probe.py — 已有 attention 后端下的**形状敏感性观测**（诊断性、只读）。

## 目的（Step 3 候选 ⑤）
问：换用**已有的** attention 后端（`eager` / `sdpa` / `cudnn_prefill` 等，由 `model_self` 的
`set_attention_strategy` 提供）之后，**位置 8209 的 S（单步 M=1）与 B（四行块 M=4）是否仍然分歧**？

**这不是正确性证明**：
* 换后端会**整体改变**数值路径（不只是形状选择）⇒ 即使 token 一致，也不能说明"形状问题已修复"；
* 本工具只作**观测**：记录每个后端下 S/B 的 argmax、logits 差异、以及 L0 决策行差异；
* 生产默认后端**不改变**（每个策略用完即恢复为 `eager`），不写入任何默认值。

**纪律**：不做 FP32、不启用 cuDNN/SDPA 于生产路径、不改权重/cache/mask、不跑 B6；结果先落盘再摘要。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch

from .. import results_schema as S
from . import cache_clone as CC
from . import diag_checks as DC
from . import f13_i50_cells as C
from . import lmhead_boundary as LB
from . import model_self as M
from . import op_probe as OP
from . import sb_align as A
from . import spec_dec
from . import g5_8k_spec_variance_probe as P5V

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"

TAG = os.environ.get("AB_TAG", "div8209_backend")
PRE = int(os.environ.get("AB_PRE", "8208"))
TARGET_POS = int(os.environ.get("AB_TARGET", "8209"))
FEED = [int(x) for x in os.environ.get("AB_FEED", "20642").split(",")]
CANDS = [int(x) for x in os.environ.get("AB_CANDS", "20642,659,576,4531").split(",")]
TARGET_ROPE = os.environ.get("AB_TARGET_ROPE", "cpu_fp32").strip().lower()
CAP_H = os.environ.get("AB_CAP_H", str(OUTDIR / "divcap_ar_20261001T185503Z.pt"))
BACKENDS = [x for x in os.environ.get("AB_BACKENDS", "eager,sdpa").split(",") if x]
KV = "base_bf16"
VIS = PRE + 1


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


def _cmp(a, b, name) -> dict:
    DC.assert_same_shape(a, b, name)
    d = (a.float() - b.float()).abs()
    return dict(name=name, exact_equal=bool(torch.equal(a, b)), n_diff=int((a != b).sum()),
                max_abs=float(d.max()), sha16_a=A.stable_sha(a), sha16_b=A.stable_sha(b))


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec: dict = dict(run_id=run_id, tool="attention_backend_shape_probe", family="F13", kv=KV,
                     timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     git=S.git_meta(ROOT),
                     question="换用已有 attention 后端后，位置 8209 的 S(M=1) 与 B(M=4) 是否仍然分歧？",
                     semantics="只作观测：每个后端跑 S/B 两臂并比较 argmax 与决策行；**不**把后端切换当正确性证明；"
                                "生产默认后端不变（用完恢复 eager）。",
                     config=dict(tag=TAG, pre=PRE, target_pos=TARGET_POS, feed=FEED, cands=CANDS,
                                 backends=BACKENDS, target_rope=TARGET_ROPE, cap_h=str(CAP_H),
                                 in_scope="后端形状稳定性观测；不改生产默认、不做 FP32、不跑 B6"))
    print(f"[ab] run_id={run_id} backends={BACKENDS} pre={PRE}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    TRS = None
    try:
        from .lowmem_load import load_ours_bf16
        tgt, tok, ti = load_ours_bf16(P5V.TARGET)
        tgt.set_kv_format(spec_dec._fmt(KV))
        rec["weights_load"] = dict(bitmatch=ti.get("weights_bitmatch"), checked=ti.get("n_params_checked"))
        rec["strategy_default_at_start"] = M.attention_strategy()
        if TARGET_ROPE == "cpu_fp32":
            from . import target_rope_scope as _TRS
            TRS = _TRS
            rec["target_rope_scope"] = TRS.install(tgt, theta=float(tgt.rope_theta),
                                                   head_dim=int(tgt.head_dim))
        cap = C._load_cap(Path(CAP_H))
        H, tinfo = C._truncate(cap["snapshot_cpu"], PRE, "H")
        rec["H_source"] = dict(path=str(CAP_H), sha256=_sha_file(Path(CAP_H)), pre=PRE,
                               is_pure_slice=tinfo["is_pure_slice"], n_layers=len(H))
        DC.require(tinfo["is_pure_slice"] and C._cache_len(H) == PRE, "H 截断不符")
        dev = next(tgt.parameters()).device

        def _arm(arm: str, mode: str, layer_caps=None):
            cache = _to_dev(H, dev)
            with OP.capture(tgt, 0, arm) as cc:
                with torch.inference_mode(), M.attention_phase(mode):
                    if arm == "S":
                        lg, _c = spec_dec._step(tgt, FEED[0], cache)
                        row = LB.row_of(lg, None)
                    else:
                        rows_b, _c = spec_dec._verify(tgt, cache, CANDS)
                        row = LB.row_of(rows_b, 0)
                nodes = {k: v["tensor"] for k, v in cc.records.items() if not v.get("shape_only")}
            return row.detach().cpu(), nodes

        out = {}
        for bk in BACKENDS:
            entry: dict = dict(backend=bk)
            try:
                M.set_attention_strategy(bk)
                M.reset_attention_stats()
                s_row, s_nodes = _arm("S", "decode")
                b_row, b_nodes = _arm("B", "verify")
                st = M.attention_stats()
                entry.update(
                    attention_stats=dict(strategy=st["strategy"], fused_by_phase=st["fused_by_phase"],
                                         eager_by_phase=st["eager_by_phase"],
                                         s_max_by_phase=st.get("s_max_by_phase")),
                    S=dict(argmax=int(s_row.argmax()), sha16=A.stable_sha(s_row)),
                    B=dict(argmax=int(b_row.argmax()), sha16=A.stable_sha(b_row)),
                    tokens_match=bool(int(s_row.argmax()) == int(b_row.argmax())),
                    final_row=_cmp(s_row, b_row, f"{bk}_S_vs_B_final_row"),
                    L0_output=_cmp(s_nodes["layer_output"][0, -1], b_nodes["layer_output"][0, 0],
                                   f"{bk}_L0out_S_vs_B"),
                    L0_input=_cmp(s_nodes["x0"][0, -1], b_nodes["x0"][0, 0], f"{bk}_L0in_S_vs_B"),
                    qk_row0=_cmp(s_nodes["qk_scores_pre_mask"][0, :, 0, :VIS],
                                 b_nodes["qk_scores_pre_mask"][0, :, 0, :VIS], f"{bk}_qk_row0"),
                    note="观测：该后端下 S/B 是否仍分歧；**不**代表「修复」——后端整体改变了数值路径")
            except Exception as e:                                        # noqa: BLE001
                entry["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            finally:
                M.set_attention_strategy("eager")                          # 用完恢复，不写入默认
            out[bk] = entry
            print(f"[ab] {bk:<14} S={entry.get('S', {}).get('argmax')} B={entry.get('B', {}).get('argmax')} "
                  f"tokens_match={entry.get('tokens_match')} "
                  f"L0out_n_diff={((entry.get('L0_output') or {}).get('n_diff'))} "
                  f"err={entry.get('error', '')[:60]}", flush=True)
        rec["backends"] = out
        rec["strategy_restored"] = M.attention_strategy()
        DC.require(rec["strategy_restored"] == "eager", "策略未恢复为 eager", rec["strategy_restored"])
        rec["stage"] = "complete"
        p = OUTDIR / f"{TAG}_{run_id}.json"
        rec["artifacts"] = dict(json=str(p))
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[ab] RESULT_JSON={p}")
        return 0
    except Exception as e:                                                # noqa: BLE001
        rec["stage"] = ("checks_failed" if isinstance(e, DC.CheckError) else "crashed")
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        p = OUTDIR / f"{TAG}_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[ab] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[ab] RESULT_JSON={p}")
        return 5 if rec["stage"] == "checks_failed" else 2
    finally:
        try:
            M.set_attention_strategy("eager")
            LB.assert_clean()
            M.clear_attn_probe()
            M.clear_serial_qk()
            M.clear_serial_linear()
        except Exception:                                                 # noqa: BLE001
            pass
        if TRS is not None:
            try:
                rec["target_rope_scope_restore"] = TRS.assert_restored()
            except Exception:                                             # noqa: BLE001
                pass


if __name__ == "__main__":
    raise SystemExit(main())
