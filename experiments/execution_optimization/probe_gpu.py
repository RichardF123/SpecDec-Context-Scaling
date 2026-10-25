"""Phase O 的 GPU correctness probe：optimized vs reference vs fixture 三向对照。

判定（全部必须通过；任一失败 ⇒ PHASE_O_REJECTED_FOR_CORRECTNESS）：
  * optimized generated_ids / generated_length / stop_reason == reference fixture；
  * optimized 每轮 (Kp, m, pre, cands, g, emitted, tgt_len, drf_len) == reference 轨迹；
  * optimized stats == reference stats（rounds/proposed/accepted/mismatch/rollbacks/...）；
  * reference 运行本身与 fixture 一致（同进程复现性）；AR 与 fixture 一致；
  * 逐轮 cache 长度不变量 & 末轮 tgt_len=drf_len=ctx+gen；`check_consistent` 通过；
  * logits 有限（`finite_check=True`；propose/verify/repair 的每次 _run 后检查）；
  * attention 策略断言通过（eager decode、cudnn_prefill 仅 prefill）。
另记录：reference 每轮 host 同步次数（.item/.tolist/.cpu/int）vs optimized ledger（diagnostic，不用于性能）。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from experiments.formal import worker as W  # noqa: E402
from experiments.formal.environment import check_environment, check_gpu_environment  # noqa: E402
from experiments.formal.plan import Condition, Job  # noqa: E402
from experiments.formal.runtime_info import collect_runtime_info  # noqa: E402
from specdec.engine import model_self, spec_dec, target_rope_scope  # noqa: E402

from experiments.execution_optimization import config  # noqa: E402
from experiments.execution_optimization import fast_spec_dec as F  # noqa: E402

FAMILIES = config.FROZEN["dev_families"]
CONTEXTS = config.FROZEN["contexts"]


class SyncCounter:
    """统计 .item()/.tolist()/.cpu()/int(tensor) 的调用次数（host 同步类操作）。"""

    def __init__(self):
        self.n = 0

    def __enter__(self):
        self._orig = (torch.Tensor.item, torch.Tensor.tolist, torch.Tensor.cpu, torch.Tensor.__int__)
        counter = self

        def item(self_):
            counter.n += 1
            return counter._orig[0](self_)

        def tolist(self_):
            counter.n += 1
            return counter._orig[1](self_)

        def cpu(self_, *a, **k):
            counter.n += 1
            return counter._orig[2](self_, *a, **k)

        def __int__(self_):
            counter.n += 1
            return counter._orig[3](self_)

        torch.Tensor.item = item
        torch.Tensor.tolist = tolist
        torch.Tensor.cpu = cpu
        torch.Tensor.__int__ = __int__
        return self

    def __exit__(self, *exc):
        (torch.Tensor.item, torch.Tensor.tolist, torch.Tensor.cpu, torch.Tensor.__int__) = self._orig
        return False


def run_reference(target, draft, ids, eos_id):
    trace = []
    prompt = torch.tensor([ids], dtype=torch.long, device=torch.device("cuda"))
    out = spec_dec.spec_decode_greedy(
        target, draft, prompt, 64, k=4, eos_id=eos_id,
        target_kv="base_bf16", draft_kv="base_bf16", log=trace.append)
    return out, trace


def compare_traces(ref_trace, opt_trace):
    diffs = []
    if len(ref_trace) != len(opt_trace):
        diffs.append(dict(kind="round_count", ref=len(ref_trace), opt=len(opt_trace)))
        return diffs
    for i, (a, b) in enumerate(zip(ref_trace, opt_trace)):
        for key in ("Kp", "m", "pre", "tgt_len", "drf_len"):
            if a[key] != b[key]:
                diffs.append(dict(kind="trace", round=i, key=key, ref=a[key], opt=b[key]))
        for key in ("cands", "g", "emitted"):
            if a[key] != b[key]:
                diffs.append(dict(kind="trace", round=i, key=key, ref=a[key], opt=b[key]))
    return diffs


def probe(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    fixtures_dir = out.parent / "fixtures"
    if not fixtures_dir.exists():
        fixtures_dir = Path(out / "fixtures")
    env = check_environment()
    gpu = check_gpu_environment()
    target, tokenizer, target_info = W.load_target()
    draft, dtok, draft_info = W.load_draft()
    if dtok.eos_token_id != tokenizer.eos_token_id:
        raise RuntimeError("draft/target EOS 不一致")
    rope = W.configure_target(target)
    eos_id = tokenizer.eos_token_id
    rows = []
    try:
        for family in FAMILIES:
            for ctx in CONTEXTS:
                fixture_file = fixtures_dir / f"{family}_{ctx}.json"
                fixture = json.loads(fixture_file.read_text(encoding="utf-8"))
                ids, input_meta = W.load_input(family, ctx)
                row = dict(family=family, context=ctx, ok=True, errors=[])

                # 1) AR
                job_ar = Job(family, Condition(f"L{ctx}_AR_base_bf16", ctx, "AR", "base_bf16"), 0, 1)
                ar = W.run_ar_once(target, tokenizer, ids, job_ar)
                row["ar_matches_fixture"] = ar["generated_ids"] == fixture["ar"]["generated_ids"]
                row["ar_stop_matches"] = ar["stop_reason"] == fixture["ar"]["stop_reason"]
                if not row["ar_matches_fixture"]:
                    row["ok"] = False
                    row["errors"].append("AR tokens differ from fixture")
                if not row["ar_stop_matches"]:
                    row["ok"] = False
                    row["errors"].append("AR stop reason differs")

                # 2) reference SpecDec（带同步计数）
                job_sp = Job(family, Condition(f"L{ctx}_SpecDec_base_bf16", ctx, "SpecDec", "base_bf16"), 0, 1)
                W.check_runtime_config(target, draft, job_sp)
                model_self.reset_attention_stats()
                with SyncCounter() as counter:
                    with W.spec_prefill_scope(target, draft), W.spec_generation_scope():
                        ref, ref_trace = run_reference(target, draft, ids, eos_id)
                model_self.assert_strategy_used()
                ref_stats = model_self.attention_stats()
                row["reference"] = dict(gen_len=len(ref["gen"]), rounds=ref["stats"]["rounds"],
                                        sync_ops=counter.n,
                                        sync_per_round=counter.n / ref["stats"]["rounds"],
                                        s_max_by_phase=ref_stats["s_max_by_phase"],
                                        by_phase=ref_stats["by_phase"])
                row["reference_matches_fixture"] = (
                    ref["gen"] == fixture["specdec_reference"]["generated_ids"]
                    and ref["stats"] == fixture["specdec_reference"]["stats"])
                if not row["reference_matches_fixture"]:
                    row["ok"] = False
                    row["errors"].append("reference run differs from fixture")

                # 3) optimized
                model_self.reset_attention_stats()
                opt = F.spec_decode_greedy_optimized(
                    target, draft, torch.tensor([ids], dtype=torch.long,
                                                device=torch.device("cuda")),
                    64, k=4, eos_id=eos_id, finite_check=True, capture_cache=True)
                model_self.assert_strategy_used()
                opt_stats = model_self.attention_stats()
                row["optimized"] = dict(gen_len=len(opt["gen"]), rounds=opt["stats"]["rounds"],
                                        sync_ops=opt["ledger"]["sync"], sync_per_round=opt["ledger"]["sync_per_round"],
                                        h2d=opt["ledger"]["h2d"],
                                        s_max_by_phase=opt_stats["s_max_by_phase"],
                                        by_phase=opt_stats["by_phase"])
                row["optimized_matches_reference"] = (opt["gen"] == ref["gen"]
                                                      and opt["stats"] == ref["stats"])
                if not row["optimized_matches_reference"]:
                    row["ok"] = False
                    row["errors"].append("optimized tokens/stats differ from reference")
                diffs = compare_traces(ref_trace, opt["control"])
                row["trace_diffs"] = len(diffs)
                if diffs:
                    row["ok"] = False
                    row["errors"].append(f"control trace diffs: {diffs[:5]}")
                row["optimized_matches_fixture"] = (
                    opt["gen"] == fixture["specdec_reference"]["generated_ids"])
                if not row["optimized_matches_fixture"]:
                    row["ok"] = False
                    row["errors"].append("optimized tokens differ from fixture")

                # 4) cache/长度不变量
                fc = opt.get("final_cache") or {}
                expected_len = ctx + len(opt["gen"])
                row["final_cache"] = fc
                if fc.get("tgt_len") != expected_len or fc.get("drf_len") != expected_len:
                    row["ok"] = False
                    row["errors"].append(f"final cache len {fc} != ctx+gen {expected_len}")
                if not (fc.get("tgt_consistent") and fc.get("drf_consistent")):
                    row["ok"] = False
                    row["errors"].append("cache consistency failed")
                if fixture["specdec_reference"]["final_tgt_len"] != expected_len:
                    row["ok"] = False
                    row["errors"].append("fixture final length != ctx+gen")
                # 与 reference trace 的末轮长度对照
                if ref_trace and (opt["control"][-1]["tgt_len"] != ref_trace[-1]["tgt_len"]
                                  or opt["control"][-1]["drf_len"] != ref_trace[-1]["drf_len"]):
                    row["ok"] = False
                    row["errors"].append("final lens differ from reference trace")

                rows.append(row)
                status = "OK" if row["ok"] else "FAIL"
                print(f"{status} {family} ctx={ctx}: ref_sync/round={row['reference']['sync_per_round']:.2f} "
                      f"opt_sync/round={row['optimized']['sync_per_round']:.2f} "
                      f"gen={len(opt['gen'])} rounds={opt['stats']['rounds']}", flush=True)
    finally:
        target_rope_scope.restore()

    result = dict(
        schema_version="ot-probe-v1",
        utc=datetime.now(timezone.utc).isoformat(),
        environment=dict(commit=env, gpu=gpu),
        runtime=collect_runtime_info(),
        load=dict(target=target_info, draft=draft_info),
        frozen=config.FROZEN,
        rows=rows,
        all_ok=all(r["ok"] for r in rows),
    )
    (out / "probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                    encoding="utf-8")
    with (out / "probe_rows.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "family", "context", "ok", "ar_matches_fixture", "reference_matches_fixture",
            "optimized_matches_reference", "optimized_matches_fixture", "trace_diffs",
            "ref_sync_per_round", "opt_sync_per_round", "opt_h2d", "errors"])
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(family=r["family"], context=r["context"], ok=r["ok"],
                                 ar_matches_fixture=r["ar_matches_fixture"],
                                 reference_matches_fixture=r["reference_matches_fixture"],
                                 optimized_matches_reference=r["optimized_matches_reference"],
                                 optimized_matches_fixture=r["optimized_matches_fixture"],
                                 trace_diffs=r["trace_diffs"],
                                 ref_sync_per_round=r["reference"]["sync_per_round"],
                                 opt_sync_per_round=r["optimized"]["sync_per_round"],
                                 opt_h2d=r["optimized"]["h2d"],
                                 errors="; ".join(r["errors"])))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.out)
    print("ALL_OK" if result["all_ok"] else "PROBE_FAILED")
    return 0 if result["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
