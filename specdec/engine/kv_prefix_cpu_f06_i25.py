"""kv_prefix_cpu_f06_i25.py — 用**已有产物**做 L27 K/V 数值核验（**纯 CPU，只读**）。

为什么能做：两臂的 cache 都已落盘且是**原始 bf16 K/V**（RoPE 后、GQA 扩展前，形状 `(1,2,L,128)`）：
  * AR 臂  : `…_F06_ar_*.cache.pt`        = list[28] dict(k,v)，长度 **8216**（位置 0..8215）
  * Spec 臂: `…_F06_spec_*.cache.pt`      = {round11_snap_pre(8215), round11_blk(8219), round12_*}
比较三类对象（L27 为主，L0 作对照）：
  P1 **共同前缀**  : AR[0:8215] vs Spec.snap_pre（8215）—— 同一 token 序列，但由**不同调用形状**产生
  P2 **位置 8215** : AR[8215] vs Spec.blk[8215]（token=27685）—— 同位置、同 token，**不同 forward 调用**
  P3 **块内新位置**: Spec.blk[8216..8218] —— **AR 无对应项**（不可比较，只记录存在性）

不加载模型、不碰 GPU、不写正式结果；不修改任何输入张量。
"""
from __future__ import annotations

import glob
import hashlib
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TAG = "f06i25b_20260929T130000Z"
FAMILY = "F06"
LAYERS = (27, 0)          # 27 = 主对象；0 = 对照点（不构成逐层扫描）
PREFIX_LEN = 8215
POS_8215 = 8215
AR_LEN = 8216


def _newest(pat):
    return Path(sorted(glob.glob(str(OUTDIR / pat)))[-1])


def _sha(p: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:n]


def _kv(entry):
    """从 cache 条目里取 (k, v)。"""
    if isinstance(entry, dict):
        return entry.get("k"), entry.get("v")
    if isinstance(entry, (list, tuple)) and len(entry) == 2:
        return entry[0], entry[1]
    raise TypeError(f"未知 cache 条目类型：{type(entry).__name__}")


def _cmp(a, b, name):
    if tuple(a.shape) != tuple(b.shape):
        raise AssertionError(f"{name}: shape 不一致 {tuple(a.shape)} vs {tuple(b.shape)}")
    d = (a.float() - b.float()).abs()
    return dict(name=name, shape=[int(x) for x in a.shape], dtype_a=str(a.dtype),
                dtype_b=str(b.dtype), elementwise_equal=bool(torch.equal(a, b)),
                abs_err_max=float(d.max()), abs_err_mean=float(d.mean()),
                max_err_index=[int(x) for x in
                               torch.unravel_index(torch.argmax(d.reshape(-1)), d.shape)],
                finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                n_diff=int((d > 0).sum()))


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    ar_p = _newest(f"{TAG}_{FAMILY}_ar_*.cache.pt")
    sp_p = _newest(f"{TAG}_{FAMILY}_spec_*.cache.pt")
    rec = dict(run_id=run_id, tool="kv_prefix_cpu_f06_i25",
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               scope="只用已有 cache.pt 做 L27（+L0 对照）K/V 数值核验；不加载模型、不碰 GPU、"
                     "不修改输入、不写正式结果。",
               sources=dict(ar=dict(path=str(ar_p), bytes=ar_p.stat().st_size, sha256_16=_sha(ar_p)),
                            spec=dict(path=str(sp_p), bytes=sp_p.stat().st_size, sha256_16=_sha(sp_p))),
               layers=list(LAYERS), prefix_len=PREFIX_LEN)
    print(f"[kv] AR   cache: {ar_p.name} ({ar_p.stat().st_size / 1e6:.1f} MB)")
    print(f"[kv] Spec cache: {sp_p.name} ({sp_p.stat().st_size / 1e6:.1f} MB)")
    ar = torch.load(ar_p, map_location="cpu")
    sp = torch.load(sp_p, map_location="cpu")
    rec["structure"] = dict(ar_top=type(ar).__name__, ar_len=len(ar),
                            ar_entry_keys=sorted(ar[0].keys()) if isinstance(ar[0], dict) else None,
                            spec_top=sorted(sp.keys()))
    print(f"[kv] AR 顶层 {type(ar).__name__} len={len(ar)} 条目键={sorted(ar[0].keys())}")
    print(f"[kv] Spec 顶层键={sorted(sp.keys())}")
    snap = sp["round11_snap_pre"]
    blk = sp["round11_blk"]

    # ---- 结构与不变量断言（失败即停） ----
    k_ar, v_ar = _kv(ar[27])
    k_snap, v_snap = _kv(snap[27])
    k_blk, v_blk = _kv(blk[27])
    for nm, t, want_len in (("AR.k", k_ar, AR_LEN), ("AR.v", v_ar, AR_LEN),
                            ("snap.k", k_snap, PREFIX_LEN), ("snap.v", v_snap, PREFIX_LEN),
                            ("blk.k", k_blk, 8219), ("blk.v", v_blk, 8219)):
        assert t.dim() == 4 and t.shape[1] == 2 and t.shape[3] == 128, f"{nm} 形状异常 {tuple(t.shape)}"
        assert t.shape[2] == want_len, f"{nm} 长度应为 {want_len}，实测 {t.shape[2]}"
        assert str(t.dtype) == "torch.bfloat16", f"{nm} dtype 应为 bf16，实测 {t.dtype}"
        assert torch.isfinite(t).all(), f"{nm} 含非有限元素"
    rec["invariants"] = dict(ar_len=int(k_ar.shape[2]), snap_len=int(k_snap.shape[2]),
                             blk_len=int(k_blk.shape[2]), dtype="torch.bfloat16",
                             shape_ok=True, finite_ok=True)
    print(f"[kv] 断言通过：AR={AR_LEN} snap_pre={PREFIX_LEN} blk=8219，全部 bf16 且 finite")

    ar_snap_k = k_ar.clone()
    out = {}
    for li in LAYERS:
        ka, va = _kv(ar[li])
        ks, vs = _kv(snap[li])
        kb, vb = _kv(blk[li])
        e = dict(
            P1_prefix_k=_cmp(ka[:, :, :PREFIX_LEN, :], ks, "P1_prefix_k"),
            P1_prefix_v=_cmp(va[:, :, :PREFIX_LEN, :], vs, "P1_prefix_v"),
            P2_pos8215_k=_cmp(ka[:, :, POS_8215:POS_8215 + 1, :], kb[:, :, POS_8215:POS_8215 + 1, :],
                              "P2_pos8215_k"),
            P2_pos8215_v=_cmp(va[:, :, POS_8215:POS_8215 + 1, :], vb[:, :, POS_8215:POS_8215 + 1, :],
                              "P2_pos8215_v"),
            P3_blk_new_positions=dict(
                exists=True, length=int(kb.shape[2]) - POS_8215 - 1,
                note="块内 8216/8217/8218 三个新位置：**AR 无对应项**（AR 未产生这些位置的 K/V）"
                     "⇒ 不可比较，只记录存在性"),
            ar_has_pos8216=dict(length=int(ka.shape[2]),
                                note="AR cache 长度 8216 ⇒ 只含位置 0..8215；位置 8216 是 AR 本次" 
                                     "**前向的查询位置**，其 K/V 由该前向产生但**未保存在本次 cache.pt**（"
                                     "保存发生在写入之前）"))
        out[f"L{li}"] = e
    rec["compare"] = out

    # 输入未被修改
    rec["inputs_unmodified"] = dict(
        ar_k_unchanged=bool(torch.equal(k_ar, ar_snap_k)),
        spec_files_readonly=True)
    assert rec["inputs_unmodified"]["ar_k_unchanged"], "AR K 张量被原地修改"

    p = OUTDIR / f"{TAG}_kvprefix_{run_id}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for li in LAYERS:
        e = out[f"L{li}"]
        print(f"\n== L{li} ==")
        for k in ("P1_prefix_k", "P1_prefix_v", "P2_pos8215_k", "P2_pos8215_v"):
            v = e[k]
            print(f"  {k:16s} equal={v['elementwise_equal']} max|Δ|={v['abs_err_max']:.6g} "
                  f"mean={v['abs_err_mean']:.6g} n_diff={v['n_diff']}/{torch.tensor(v['shape']).prod().item()}"
                  f" idx={v['max_err_index']}")
        print(f"  P3 块内新位置：{e['P3_blk_new_positions']['note']}")
    print(f"\n[kv] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
