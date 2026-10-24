"""CPU-only readiness audit for the frozen confirmatory full plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

from experiments.formal.input_sets import select_input_set
from experiments.formal.runtime_info import collect_runtime_info
from specdec.engine import family_input

from .plan import CONTEXTS, build_plan, validate_plan
from .run import PREREGISTRATION, SOURCE_FILES, source_sha256

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "output/confirmatory_readiness/2026-10-25/full_readiness.json",
    )
    args = parser.parse_args()

    manifest_path, families = select_input_set("formal")
    manifest = family_input.load_manifest(manifest_path)
    tokenizer = AutoTokenizer.from_pretrained(
        family_input.TOKENIZER_ID,
        revision=family_input.TOKENIZER_REVISION,
        local_files_only=True,
    )

    hash_rows = []
    for family in families:
        for ctx in CONTEXTS:
            ids, meta = family_input.load_family_ids(
                tokenizer, family, manifest=manifest, ctx=ctx, verify_hash=True
            )
            hash_rows.append(
                {
                    "family": family,
                    "ctx": ctx,
                    "tokens": len(ids),
                    "token_sha1": meta["token_sha1"],
                    "manifest_token_sha1": meta["manifest_token_sha1"],
                    "match": meta["token_sha1"] == meta["manifest_token_sha1"],
                }
            )

    tasks = build_plan(families, contexts=CONTEXTS, seed=4207)
    plan = validate_plan(tasks, families, contexts=CONTEXTS)
    runtime = collect_runtime_info()
    prereg_hash = sha256(PREREGISTRATION)
    manifest_hash = sha256(manifest_path)

    checks = {
        "formal_family_set_exact": tuple(families) == tuple(f"FF{i:02d}" for i in range(1, 17)),
        "all_48_input_hashes_match": len(hash_rows) == 48 and all(r["match"] for r in hash_rows),
        "plan_192_tasks": plan["n_tasks"] == 192,
        "plan_480_measured": plan["n_measured"] == 480,
        "plan_48_family_context_pairs": plan["n_pairs"] == 48,
        "ab_ba_8_8_per_context": all(
            row == {"AB": 8, "BA": 8} for row in plan["pair_order_by_context"].values()
        ),
        "all_96_block_pairs_adjacent": plan["adjacent_pair_units"] == 96,
        "block_pair_units_interleaved": (
            plan["block_interleave"]["max_same_block_run"] <= 2
            and plan["block_interleave"]["transitions"] >= 95
        ),
        "preregistration_present": PREREGISTRATION.exists(),
        "all_frozen_sources_present": all((ROOT / rel).exists() for rel in SOURCE_FILES),
    }
    code_ready = all(checks.values())
    execution_ready_now = code_ready and not runtime["dirty_worktree"]

    result = {
        "schema_version": "confirmatory-full-readiness-v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cpu_only": True,
        "gpu_touched": False,
        "seed": 4207,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "preregistration": str(PREREGISTRATION),
        "preregistration_sha256": prereg_hash,
        "source_sha256": source_sha256(),
        "runtime": runtime,
        "plan_summary": plan,
        "input_hashes": hash_rows,
        "checks": checks,
        "code_ready_for_clean_freeze": code_ready,
        "execution_ready_now": execution_ready_now,
        "blocking_reason": (
            None if execution_ready_now else
            "worktree must be committed and clean; execute with --expected-commit equal to HEAD"
            if code_ready and runtime["dirty_worktree"] else
            "one or more readiness checks failed"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({
        "out": str(args.out.resolve()),
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "code_ready_for_clean_freeze": code_ready,
        "execution_ready_now": execution_ready_now,
        "blocking_reason": result["blocking_reason"],
    }, ensure_ascii=False, indent=2))
    return 0 if code_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
