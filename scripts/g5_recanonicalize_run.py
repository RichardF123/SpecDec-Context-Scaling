"""g5_recanonicalize_run.py — 把已被修正的映射规则**重新套用**到既有 part4 分片上，生成新版 canonical。

这是"**重新记账**"，不是"重新测量"：
  - 只读 `output/part4/part4_*_<run_id>*.jsonl`（**绝不改写**）；
  - 计时数字仍是当初那批测量值；被修正的是**映射/角色/状态分层/跨进程等价性**；
  - 输出到**新 schema 版本**的新文件名（唯一命名，拒绝覆盖），旧 canonical 原样保留作审计证据；
  - 每行 `notes` 里写明"由哪些分片重建"，可回溯。

为什么需要它：schema v1.1 的 canonical 存在已知错记（summary `*_med` 未映射、
跨进程等价性写成凭空否定的 `ar_equal=False`、within-run/between-block 措辞混用、
dual-control 误标声明资格）。**原地改写会销毁审计证据**，所以一律重建新文件。

用法
  python scripts/g5_recanonicalize_run.py --run-id g5stab_20260922T040625Z
  python scripts/g5_recanonicalize_run.py --run-id g5stab_20260922T040625Z --dry-run
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from specdec import results_schema as S          # noqa: E402

PART4 = ROOT / "output" / "part4"
G5DIR = ROOT / "output" / "g5"
PAT = re.compile(r"part4_[0-9a-f]+_(?P<rid>g5stab(?:_[A-Za-z0-9]+)?)_b(?P<b>\d+)_c(?P<c>\d+)_"
                 r"(?P<role>pri|ctl)_(?P<ctx>\d+)_(?P<kv>[a-z0-9_]+)_(?P<mode>AR|SpecDec)_"
                 r"(?P<res>target_only|dual_control|na)\.jsonl$")


def load_shards(run_id: str) -> tuple[dict, dict]:
    """读分片（只读）。

    返回 `(summaries, runs)`：
      summaries: {(ctx, kv, mode, res, block): summary_row}
      runs     : {(ctx, kv, mode, res): [每 rep 的 run 行]}
    只读、绝不改写原始分片。
    """
    summ, runs = {}, {}
    for f in sorted(glob.glob(str(PART4 / f"part4_*_{run_id}_b*_c*_*.jsonl"))):
        name = Path(f).name
        m = PAT.search(name)
        if not m:
            continue
        res = None if m.group("res") == "na" else m.group("res")
        cell = (int(m.group("ctx")), m.group("kv"), m.group("mode"), res)
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") == "summary":
                summ[cell + (int(m.group("b")),)] = dict(r, _shard=name)
            elif r.get("kind") == "run":
                runs.setdefault(cell, []).append(r)
    return summ, runs


def pooled_seconds(runs: dict, cell: tuple) -> dict | None:
    """全 measured rep 合并中位数（与 probe / 离线报表同一口径）。"""
    rs = runs.get(cell) or []
    if not rs:
        return None
    def m(k):
        v = [x.get(k) for x in rs if isinstance(x.get(k), (int, float)) and x.get(k) == x.get(k)]
        return statistics.median(v) if v else None
    return dict(ar_decode=m("t_decode_s"), spec_decode=m("t_decode_s"),
                ar_total=m("t_total_s"), spec_total=m("t_total_s"))


def between_block(shards: dict) -> dict:
    """按 probe 的口径重算 between-block：同 cell 两 block 的 decode tok/s 中位数相对差。"""
    per_cell: dict = {}
    for (ctx, kv, mode, res, b), r in shards.items():
        per_cell.setdefault((ctx, kv, mode, res), {})[b] = r.get("decode_tok_s_med")
    out = {}
    for cell, meds in per_cell.items():
        vals = [v for v in meds.values() if isinstance(v, (int, float)) and v == v]
        if len(vals) < 2:
            continue
        hi, lo = max(vals), min(vals)
        rel = 100.0 * (hi - lo) / hi if hi else None
        out[cell] = dict(rel=rel, check=bool(rel is not None and rel > S.BETWEEN_BLOCK_REL_LIMIT))
    return out


def build_rows(run_id: str, shards: dict, bt: dict) -> list[tuple]:
    rows = []
    for (ctx, kv, mode, res, b), summ in sorted(shards.items(), key=lambda x: str(x[0])):
        info = bt.get((ctx, kv, mode, res), {})
        meta = dict(
            run_id=run_id, block_index=b, condition_index=0, seed=None, condition_order=[],
            timestamp_utc=None, mode=mode, ar_residency=(res if mode == "AR" else None),
            target_kv=kv, context_requested=ctx, context_actual=ctx,
            max_new_tokens=summ.get("max_new", 64), reps_expected=5,
            reps_measured=summ.get("reps"), returncode=0, env_qualified=True,
            pstate_qualified=True, between_block_rel_diff_pct=info.get("rel"),
            between_block_check=info.get("check"))
        row = S.build_g5_row(dict(summ, kind="summary"), meta)
        row["notes"] = (f"[RECANON {S.SCHEMA_VERSION}] 由 part4 分片重建（**未重新测量**）："
                        f"{summ.get('_shard')}；原始 run={run_id}")
        rows.append(((ctx, kv, mode, res, b), row))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="把修正后的映射规则重新套用到既有 part4 分片")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    shards, runs = load_shards(args.run_id)
    if not shards:
        print(f"未找到 run_id={args.run_id} 的 part4 分片")
        return 2
    print(f"[recanon] run_id={args.run_id} 分片 {len(shards)} 个（rep 行 "
          f"{sum(len(v) for v in runs.values())} 条）→ schema {S.SCHEMA_VERSION}")
    bt = between_block(shards)
    print("\n[recanon] between-block（重算）：")
    for cell, v in sorted(bt.items(), key=lambda x: str(x[0])):
        print(f"  {cell}: rel={v['rel']:.1f}% -> {'CHECK' if v['check'] else 'OK'}")

    rows = build_rows(args.run_id, shards, bt)
    S.fill_cross_process_equality([r for _, r in rows])

    print("\n[recanon] 跨进程等价性：")
    for key, r in rows:
        if r.get("mode") == "SpecDec":
            print(f"  {key}: ar_equal={r['ar_equal']} certified={r['ar_equal_certified']} "
                  f"({r['equality_reason']})")

    print("\n[recanon] 状态分层与声明资格：")
    for key, r in rows:
        print(f"  {key[0]:<6} {key[1]:<10} {key[2]:<8} {str(key[3] or '-'):<13} b{key[4]} "
              f"ms={r['measurement_status']:<11} wr={r['within_run_status']:<6} "
              f"bb={str(r['between_block_status']):<5} env={str(r['environment_status']):<5} "
              f"role={str(r['condition_role']):<18} claim={r['speedup_claim_allowed']}")
        if r["between_block_check_reason"]:
            print(f"          between: {r['between_block_check_reason']}")
        if r["within_run_check_reason"]:
            print(f"          within : {r['within_run_check_reason']}")

    print("\n[recanon] speedup（按 block 配对 + 全 rep 合并中位数；与离线报表同一口径）：")
    cells = {(k[0], k[1]) for k in shards}
    for ctx, kv in sorted(cells):
        ar_cell = (ctx, kv, "AR", S.PRIMARY_AR_RESIDENCY)
        sp_cell = (ctx, kv, "SpecDec", None)
        ar_rows = [r for k, r in rows if k[:4] == ar_cell]
        sp_rows = [r for k, r in rows if k[:4] == sp_cell]
        if not ar_rows or not sp_rows:
            continue
        pa, ps = pooled_seconds(runs, ar_cell), pooled_seconds(runs, sp_cell)
        pooled = None
        if pa and ps:
            pooled = dict(ar_decode=pa["ar_decode"], spec_decode=ps["spec_decode"],
                          ar_total=pa["ar_total"], spec_total=ps["spec_total"])
        su = S.speedup_claim_over_blocks(ar_rows, sp_rows, pooled_seconds=pooled)
        per = "  ".join(f"blk{x['block_index']}:×{x['speedup_decode']:.3f}"
                        for x in su["per_block"] if x.get("speedup_decode") is not None)
        pooled_txt = (f"合并 decode ×{su['pooled_decode']:.3f} e2e ×{su['pooled_e2e']:.3f}"
                      if su["pooled_decode"] else "合并 n/a")
        print(f"  ctx={ctx} {kv}  {per} | {pooled_txt}")
        print(f"    allowed={su['allowed']}")
        for why in su["blockers"]:
            print(f"      {why}")

    if args.dry_run:
        print("\n[recanon] --dry-run：未写入任何文件")
        return 0

    seed = "\n".join(f"{k}|{r['measurement_status']}|{r['ar_equal']}" for k, r in rows)
    cfg_hash = "re" + hashlib.sha256(seed.encode()).hexdigest()[:10]
    out = S.g5_path(G5DIR, cfg_hash, args.run_id)
    if out.exists():
        raise SystemExit(f"[拒绝覆盖] {out} 已存在")
    for _, r in rows:
        ref = next((rr for kk, rr in rows
                    if kk[:4] == (r["context_actual"], r["target_kv"], "AR",
                                  S.PRIMARY_AR_RESIDENCY)
                    and kk[4] == r["block_index"]), None) if r["mode"] == "SpecDec" else None
        S.write_g5_run(out, r, append=True, ref_row=ref)
    print(f"\n[recanon] 写出 {len(S.read_g5_rows(out))} 行 -> {out}")
    print("[recanon] 旧 canonical（v1.1）**原样保留**作审计证据；本次未重新测量任何时间。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
