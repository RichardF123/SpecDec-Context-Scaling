"""Collect position-wise acceptance and target-margin evidence.

The diagnostic reproduces the frozen block-verification algorithm.  It is not a
timing benchmark: top-k, rank computation, and scalar extraction intentionally
run outside any performance claim.  Rows after the first mismatch are retained
as counterfactual block outputs but excluded from primary acceptance analysis.
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

from experiments.execution_optimization import config  # noqa: E402
from experiments.formal import worker as W  # noqa: E402
from experiments.formal.environment import check_environment, check_gpu_environment  # noqa: E402
from experiments.formal.plan import Condition, Job  # noqa: E402
from experiments.formal.runtime_info import collect_runtime_info  # noqa: E402
from specdec.engine import model_self, spec_cache, spec_dec, target_rope_scope  # noqa: E402


def _last_row(logits: torch.Tensor) -> torch.Tensor:
    return logits.reshape(-1, logits.shape[-1])[-1]


def _top_evidence(logits: torch.Tensor, candidate: int) -> dict:
    """Return decision evidence without changing the model state.

    Rank uses strict-greater logits.  Ties therefore share the same lower-bound
    rank; the exact chosen argmax id is separately recorded.
    """
    row = logits.float()
    values, indices = torch.topk(row, k=2)
    candidate_logit = row[candidate]
    rank = int((row > candidate_logit).sum().item()) + 1
    return {
        "top1_id": int(indices[0].item()),
        "top2_id": int(indices[1].item()),
        "top1_logit": float(values[0].item()),
        "top2_logit": float(values[1].item()),
        "margin_top1_top2": float((values[0] - values[1]).item()),
        "candidate_logit": float(candidate_logit.item()),
        "candidate_gap_from_top1": float((values[0] - candidate_logit).item()),
        "candidate_rank_lower": rank,
    }


def _prefill(model, prompt: torch.Tensor):
    logits, cache = W.test_chunked_prefill(model, prompt, chunk=2048, diagnostic=False)
    return logits.unsqueeze(1), cache


def _reference(target, draft, prompt, eos_id: int) -> dict:
    model_self.reset_attention_stats()
    with W.spec_prefill_scope(target, draft), W.spec_generation_scope():
        out = spec_dec.spec_decode_greedy(
            target, draft, prompt, config.FROZEN["max_new"],
            k=config.FROZEN["k"], eos_id=eos_id,
            target_kv="base_bf16", draft_kv="base_bf16",
        )
    model_self.assert_strategy_used()
    return out


def diagnose_case(target, draft, prompt: torch.Tensor, eos_id: int) -> dict:
    target.set_kv_format("fp16")
    draft.set_kv_format("fp16")
    tgt_lg, tgt_cache = _prefill(target, prompt)
    drf_lg, drf_cache = _prefill(draft, prompt)
    saved_next = tgt_lg[:, -1:]
    drf_lg = drf_lg[:, -1:]
    generated: list[int] = []
    positions: list[dict] = []
    pending_positions: list[dict] = []
    rounds: list[dict] = []
    stats = dict(rounds=0, proposed=0, accepted=0, mismatch_rounds=0,
                 all_accept_rounds=0, target_block_calls=0, draft_calls=0, rollbacks=0)

    while len(generated) < config.FROZEN["max_new"]:
        round_index = stats["rounds"]
        k_eff = min(config.FROZEN["k"], config.FROZEN["max_new"] - len(generated))
        pre = spec_cache.cache_length(tgt_cache)
        candidates: list[int] = []
        draft_rows: list[torch.Tensor] = []
        prop_lg = drf_lg

        with torch.inference_mode(), model_self.attention_phase("propose"):
            for position in range(k_eff):
                proposal_row = _last_row(prop_lg)
                token = int(proposal_row.argmax().item())
                candidates.append(token)
                # Keep the existing tensor alive, but do not launch diagnostic top-k/rank
                # kernels between model calls.  Evidence is computed after generation.
                draft_rows.append(proposal_row.detach())
                ids = torch.tensor([[token]], dtype=torch.long, device=proposal_row.device)
                prop_lg, drf_cache = draft._run(
                    draft.embed_tokens(ids), drf_cache, need_logits=True)
                if token == eos_id:
                    break

        kp = len(candidates)
        if not kp:
            break
        with model_self.attention_phase("verify"):
            rows, target_block_cache = spec_dec._verify(target, tgt_cache, candidates)
        judgments = [saved_next] + [rows[j] for j in range(kp - 1)]
        target_rows = [_last_row(row).detach() for row in judgments]
        target_ids = [int(row.argmax().item()) for row in target_rows]
        mismatch = 0
        while mismatch < kp and target_ids[mismatch] == candidates[mismatch]:
            mismatch += 1

        for position in range(kp):
            decision_relevant = mismatch == kp or position <= mismatch
            accepted = position < mismatch
            first_mismatch = mismatch < kp and position == mismatch
            pending_positions.append({
                "round": round_index,
                "pre": pre,
                "kp": kp,
                "position": position + 1,
                "candidate_id": candidates[position],
                "accepted": accepted,
                "first_mismatch": first_mismatch,
                "decision_relevant": decision_relevant,
                "post_mismatch_counterfactual": not decision_relevant,
                "draft_logits": draft_rows[position],
                "target_logits": target_rows[position],
            })

        if mismatch < kp:
            replacement = target_ids[mismatch]
            emitted = candidates[:mismatch] + [replacement]
            tgt_cache = spec_cache.trim_cache(target_block_cache, pre + mismatch)
            drf_cache = spec_cache.trim_cache(drf_cache, pre + mismatch)
            with model_self.attention_phase("repair"):
                saved_next, tgt_cache = spec_dec._step(target, replacement, tgt_cache)
                drf_lg, drf_cache = spec_dec._step(draft, replacement, drf_cache)
            stats["mismatch_rounds"] += 1
            stats["rollbacks"] += 1
        else:
            emitted = list(candidates)
            tgt_cache = target_block_cache
            drf_cache = spec_cache.trim_cache(drf_cache, pre + kp)
            saved_next = rows[kp - 1]
            drf_lg = prop_lg
            stats["all_accept_rounds"] += 1

        generated.extend(emitted)
        stats["rounds"] += 1
        stats["proposed"] += kp
        stats["accepted"] += mismatch
        stats["target_block_calls"] += 1
        stats["draft_calls"] += kp
        rounds.append({
            "round": round_index, "pre": pre, "kp": kp, "m": mismatch,
            "candidates": candidates, "target_ids": target_ids, "emitted": emitted,
            "tgt_len": spec_cache.cache_length(tgt_cache),
            "drf_len": spec_cache.cache_length(drf_cache),
        })
        if emitted and emitted[-1] == eos_id:
            break

    # Only now launch diagnostic top-k/rank kernels.  They cannot perturb any
    # subsequent generation decision because the full trajectory is complete.
    for pending in pending_positions:
        candidate = pending["candidate_id"]
        draft_evidence = _top_evidence(pending.pop("draft_logits"), candidate)
        target_evidence = _top_evidence(pending.pop("target_logits"), candidate)
        pending.update(
            target_top1_id=target_evidence["top1_id"],
            target_top2_id=target_evidence["top2_id"],
            target_margin=target_evidence["margin_top1_top2"],
            draft_token_target_rank=target_evidence["candidate_rank_lower"],
            draft_token_target_gap=target_evidence["candidate_gap_from_top1"],
            draft_margin=draft_evidence["margin_top1_top2"],
        )
        positions.append(pending)

    return {
        "generated_ids": generated,
        "stats": stats,
        "positions": positions,
        "rounds": rounds,
        "final_target_cache": spec_cache.cache_length(tgt_cache),
        "final_draft_cache": spec_cache.cache_length(drf_cache),
    }


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    environment = check_environment()
    environment["gpu"] = check_gpu_environment()
    target, tokenizer, target_info = W.load_target()
    draft, draft_tokenizer, draft_info = W.load_draft()
    if tokenizer.eos_token_id != draft_tokenizer.eos_token_id:
        raise RuntimeError("target/draft EOS mismatch")
    rope = W.configure_target(target)
    all_positions: list[dict] = []
    cases: list[dict] = []
    try:
        for family in config.FROZEN["dev_families"]:
            for ctx in config.FROZEN["contexts"]:
                ids, input_meta = W.load_input(family, ctx)
                prompt = torch.tensor([ids], dtype=torch.long, device="cuda")
                reference = _reference(target, draft, prompt, tokenizer.eos_token_id)
                model_self.reset_attention_stats()
                diagnostic = diagnose_case(target, draft, prompt, tokenizer.eos_token_id)
                model_self.assert_strategy_used()
                same = diagnostic["generated_ids"] == reference["gen"]
                stats_same = diagnostic["stats"] == reference["stats"]
                expected_len = ctx + len(diagnostic["generated_ids"])
                cache_ok = (diagnostic["final_target_cache"] == expected_len
                            and diagnostic["final_draft_cache"] == expected_len)
                first_diff = next((i for i, (a, b) in enumerate(zip(
                    reference["gen"], diagnostic["generated_ids"])) if a != b), None)
                if not cache_ok:
                    raise RuntimeError(
                        f"{family}/{ctx} diagnostic cache invariant failed: "
                        f"target={diagnostic['final_target_cache']} "
                        f"draft={diagnostic['final_draft_cache']} expected={expected_len}")
                for row in diagnostic["positions"]:
                    all_positions.append({"family": family, "ctx": ctx, **row})
                case = {
                    "family": family, "ctx": ctx,
                    "input_sha1": input_meta["token_sha1"],
                    "generated_length": len(diagnostic["generated_ids"]),
                    "rounds": diagnostic["stats"]["rounds"],
                    "proposed": diagnostic["stats"]["proposed"],
                    "accepted": diagnostic["stats"]["accepted"],
                    "acceptance": diagnostic["stats"]["accepted"] / diagnostic["stats"]["proposed"],
                    "mismatch_rounds": diagnostic["stats"]["mismatch_rounds"],
                    "all_accept_rounds": diagnostic["stats"]["all_accept_rounds"],
                    "reference_tokens_equal": same,
                    "reference_stats_equal": stats_same,
                    "reference_common_prefix": (
                        len(reference["gen"]) if same else int(first_diff or 0)),
                    "cache_ok": cache_ok,
                }
                cases.append(case)
                compare = "equal" if same and stats_same else f"diverged@{first_diff}"
                print(f"OK {family} {ctx}: accept={case['acceptance']:.3f} "
                      f"rounds={case['rounds']} mismatch={case['mismatch_rounds']} "
                      f"reference={compare}", flush=True)
    finally:
        target_rope_scope.restore()

    def write_csv(path: Path, rows: list[dict]):
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output / "positions.csv", all_positions)
    write_csv(output / "cases.csv", cases)
    manifest = {
        "schema_version": "acceptance-diagnostics-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "development-only; not a timing benchmark",
        "families": list(config.FROZEN["dev_families"]),
        "contexts": list(config.FROZEN["contexts"]),
        "configuration": config.FROZEN,
        "environment": environment,
        "runtime": collect_runtime_info(),
        "load": {"target": target_info, "draft": draft_info},
        "rope": rope,
        "cases": cases,
        "reference_equal_cases": sum(row["reference_tokens_equal"] and row["reference_stats_equal"]
                                     for row in cases),
        "all_cache_ok": all(row["cache_ok"] for row in cases),
        "interpretation": {
            "primary_rows": "decision_relevant=True only",
            "mismatch_margin": "small target margin indicates a close decision, not proof of numerical causation",
            "post_mismatch": "retained for audit but excluded because logits condition on a rejected draft prefix",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (ROOT / "output" / "acceptance_diagnostics" /
                             (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                              + "_" + uuid4().hex[:8]))
    result = run(output)
    print("OUTPUT:", output)
    print("REFERENCE_EQUAL_CASES:", result["reference_equal_cases"], "/", len(result["cases"]))
    print("ALL_CACHE_OK:", result["all_cache_ok"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

