"""replay_f06_i25_samehistory.py — 同一份 cache 历史下，**块验证行 vs 单步路径**对照（§4 的判决实验）。

为什么要这一步
  上一阶段已经证明：F06/index25 的两侧决策在**同一输入 token(659)、同一输入位置(8216)**下，
  唯一实质差异是 token **576 的 logits 相差正好 0.125**（AR 17.0 / SpecDec 17.125），
  使顶部并列集合从 {715,4710} 变成 {576,715,4710}，于是"取较小下标"选出 576 而不是 715。

  但这仍**不能**判断差异来自"**当前这一步的计算路径不同**（s=1 vs s=4）"还是
  "**两侧 cache 历史本身不同**"。判据（任务 §4）：
    * **同一份 cache** 上，块行与单步仍有差异  ⇒ 支持调查**当前计算路径**；
    * 同一份 cache 上差异**消失**          ⇒ 支持调查 **cache/saved_next 历史**。

做法（全部在 **SpecDec 自己的 cache** 上，不引入任何新开关、不改精度）
  cache = 上一阶段保存的 `round11_snap_pre`（长度 8215，位置 0..8214）
  (a) **块路径（真实）**：`_verify(tgt, clone(cache), cands=[27685,659,715,4882])` → rows[1]
      即真实决策行（输入位置 8216、预测 8217）。
  (b) **单步路径（同历史）**：`_step(tgt, 27685, clone(cache))` → cache 8216，
      再 `_step(tgt, 659, ...)` → 位置 8216 的 logits（**同一预测位置**）。
  (c) **块路径重放**与真实 rows[1] 必须逐元素相等（否则重放机制有问题，判 REPLAY_MISMATCH）。

只比较**同一预测位置**的行，不比"任意末行"。
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import torch

from .. import results_schema as S
from . import diag_cache_cmp as CC
from . import diag_fields as DF
from . import family_input as FI
from . import model_self as M
from . import spec_dec
from . import g5_8k_spec_variance_probe as P5V

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
SRC_TAG = "f06i25b_20260929T130000Z"
FAMILY = "F06"
KV = "base_bf16"
DIV_INDEX = 25
EXPECT_AR, EXPECT_SPEC = 715, 576
ROUND = 11
CAND0 = 27685          # 该轮 cands[0]，也是位置 8215 上的 token
FED = 659              # 位置 8216 上的 token（两侧相同）


def _newest(pat):
    return Path(sorted(glob.glob(str(OUTDIR / pat)))[-1])


def _cell(row):
    r = row.detach().float().reshape(-1).cpu()
    f = DF.row_fields(r, watch=[EXPECT_AR, EXPECT_SPEC, 4710, 4882])
    t2 = torch.topk(r, 2).values
    f["margin_top1_top2"] = float(t2[0] - t2[1])
    f["tie_at_top"] = bool(float(t2[0]) == float(t2[1]))
    return f


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="replay_f06_i25_samehistory", tag=SRC_TAG,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               scope="同一份 SpecDec cache 上，块路径 vs 单步路径（同预测位置）——§4 的判决实验。"
                     "不加精度开关、不改算法、不做性能结论。")
    print(f"[sam] run_id={run_id}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    ok, why = P5V.commit_guard()
    print(f"[sam] commit: {why}")
    if not ok:
        rec["stage"] = "refused"
        (OUTDIR / f"{SRC_TAG}_samehistory_{run_id}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return 3

    rec["diag_precision"] = M.set_diag_preset("eager")
    from .lowmem_load import load_ours_bf16
    tgt, tok, tinfo = load_ours_bf16(P5V.TARGET)
    tgt.set_kv_format(spec_dec._fmt(KV))
    M.set_attention_strategy("cudnn_prefill")

    # 真实记录 + 保存的 cache 快照
    j = json.loads(_newest(f"{SRC_TAG}_{FAMILY}_spec_*.json").read_text(encoding="utf-8"))
    dec = j["decision"]
    assert int(dec["deciding_round"]) == ROUND, dec["deciding_round"]
    cands = [int(x) for x in dec["cands"]]
    assert cands[0] == CAND0 and cands[1] == FED, cands
    saved = torch.load(_newest(f"{SRC_TAG}_{FAMILY}_spec_*.cache.pt"), map_location="cpu")
    snap = saved[f"round{ROUND}_snap_pre"]
    rows_pt = torch.load(_newest(f"{SRC_TAG}_{FAMILY}_spec_*.rows.pt"), map_location="cpu")
    real_rows = rows_pt[f"round{ROUND}_rows"]
    rec["source"] = dict(json=str(_newest(f"{SRC_TAG}_{FAMILY}_spec_*.json")), round=ROUND,
                         cands=cands, cache_len=int(CC.cache_len(snap)),
                         snap_len_from_record=int(dec["pre"]))
    print(f"[sam] cache len={CC.cache_len(snap)} (record {dec['pre']}) cands={cands}")

    def to_gpu(cache):
        return [{k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in d.items()}
                for d in cache]

    with torch.inference_mode():
        # (a) 真实块路径重放
        a_work = to_gpu(snap)
        with M.attention_phase("verify"):
            rows_a, _blk = spec_dec._verify(tgt, a_work, cands)
        row_a = rows_a[1].detach().float().cpu()
        # (b) 同历史单步路径：先喂 cands[0]，再喂 659
        b_work = to_gpu(snap)
        with M.attention_phase("repair"):
            _lg1, c1 = spec_dec._step(tgt, CAND0, b_work)
        with M.attention_phase("repair"):
            lg2, _c2 = spec_dec._step(tgt, FED, c1)
        row_b = lg2[0, -1].detach().float().cpu()

    # ⚠️ `compare_rows` 自身含 `note` 键 ⇒ 一律 `dict(x)` + `update()`，不用 `dict(**x, note=...)`
    #    （这个坑本会话已经踩到第三次）。
    cmp_ab = dict(DF.compare_rows(row_a, row_b))
    cmp_ab["comparison_note"] = "同一 cache、同一预测位置（8216→8217）"
    cmp_real = dict(DF.compare_rows(real_rows[1].float(), row_a))
    cmp_real["comparison_note"] = "必须 elementwise_equal=True，否则重放机制存疑"
    rec.update(
        cache_len=int(CC.cache_len(snap)),
        block_row=_cell(row_a), single_row=_cell(row_b),
        block_vs_single=cmp_ab,
        block_replay_vs_real=cmp_real,
        key_candidates={str(t): dict(block=float(row_a[t]), single=float(row_b[t]),
                                     delta=float(row_a[t] - row_b[t]))
                        for t in (EXPECT_AR, EXPECT_SPEC, 4710, 4882)},
        stage="complete",
        verdict=("SAME_HISTORY_STILL_DIFFERS__SUPPORTS_CURRENT_COMPUTE_PATH"
                 if not cmp_ab["elementwise_equal"] else
                 "SAME_HISTORY_DIFF_DISAPPEARS__SUPPORTS_HISTORY_BRANCH"),
        verdict_rule="同 cache 上块行与单步仍有差异 ⇒ 支持调查当前计算路径；差异消失 ⇒ 支持调查 cache/saved_next 历史。"
                     "这只是**定位分支**，不等于最终根因，也允许两者共同作用。")
    p = OUTDIR / f"{SRC_TAG}_samehistory_{run_id}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    rp = OUTDIR / f"{SRC_TAG}_samehistory_{run_id}.pt"
    torch.save(dict(block_row=row_a, single_row=row_b), str(rp))
    rec["rows_dump"] = str(rp)

    print("== 同一 cache（长度 %d）上、同一预测位置 8217 ==" % rec["cache_len"])
    for nm in ("block_row", "single_row"):
        c = rec[nm]
        print(f"  {nm:10s}: argmax={c['argmax_id']} max={c['max_score']} "
              f"tied={c['n_max_tied']}{c['tied_max_tokens']} margin={c['margin_top1_top2']}")
    print(f"  块 vs 单步: elementwise_equal={cmp_ab['elementwise_equal']} "
          f"argmax_same={cmp_ab['argmax_same']} max|Δ|={cmp_ab['abs_err_max']}")
    print(f"  块重放 vs 真实行: elementwise_equal={cmp_real['elementwise_equal']} "
          f"max|Δ|={cmp_real['abs_err_max']}")
    print("  关键候选:", json.dumps(rec["key_candidates"], ensure_ascii=False))
    print(f"  VERDICT={rec['verdict']}")
    print(f"[sam] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
