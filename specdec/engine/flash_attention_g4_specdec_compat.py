"""End-to-end compatibility gate for FlashAttention on the latest G4 workflow.

The test keeps the draft model eager and compares:

* eager BF16 target versus Flash BF16 target under cached AR;
* eager-target SpecDec versus Flash-target SpecDec;
* per-round acceptance and rollback state;
* cache lengths/invariants after cached q_len=4 verification;
* constructed one-round state-machine cases covering m=0..K.

This is a correctness diagnostic, not a performance benchmark.
"""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import torch

from . import spec_cache, spec_dec
from .g4_b4_cache_tests import load_ours
from .g4_part4_pilot_measure import load_text


TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
CONTEXTS = tuple(
    int(value) for value in os.environ.get(
        "FLASH_G4_COMPAT_CTX", "512,2048,8192"
    ).split(",") if value.strip()
)
MAX_NEW = int(os.environ.get("FLASH_G4_COMPAT_MAXNEW", "32"))
CHUNK = int(os.environ.get("FLASH_G4_COMPAT_CHUNK", "2048"))
K = spec_dec.K
BACKENDS = ("eager", "triton_bf16_two_pass")


def synchronize():
    torch.cuda.synchronize()
    return time.perf_counter()


def chunked_prefill(model, ids):
    cache = None
    logits = None
    with torch.inference_mode():
        for start in range(0, ids.shape[1], CHUNK):
            piece = ids[:, start:start + CHUNK]
            logits, cache = model._run(
                model.embed_tokens(piece),
                cache,
                need_logits=(start + CHUNK >= ids.shape[1]),
            )
            spec_cache.check_consistent(cache)
    return logits, cache


def cache_lengths(cache):
    spec_cache.check_consistent(cache)
    return [int(layer["k"].shape[2]) for layer in cache]


def run_ar(model, ids, eos_id, backend):
    model.set_kv_format("fp16")
    model.set_attention_backend(backend)
    started = synchronize()
    logits, cache = chunked_prefill(model, ids)
    prefill_s = synchronize() - started
    prefill_length = spec_cache.cache_length(cache)
    decisions = []
    tokens = []
    finite = bool(torch.isfinite(logits).all())
    started = synchronize()
    with torch.inference_mode():
        for index in range(MAX_NEW):
            row = logits[:, -1].float().cpu()
            decisions.append(row)
            token = int(logits[:, -1].argmax())
            tokens.append(token)
            if token == eos_id or index == MAX_NEW - 1:
                break
            logits, cache = model._run(
                model.embed_tokens(torch.tensor([[token]], device=ids.device)),
                cache,
                need_logits=True,
            )
            spec_cache.check_consistent(cache)
            finite &= bool(torch.isfinite(logits).all())
    decode_s = synchronize() - started
    return {
        "backend": backend,
        "tokens": tokens,
        "decision_logits": decisions,
        "finite": finite,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "prefill_cache_length": prefill_length,
        "final_cache_length": spec_cache.cache_length(cache),
        "layer_cache_lengths": cache_lengths(cache),
    }


def logits_comparison(reference, candidate):
    rows = []
    for eager, flash in zip(reference, candidate):
        difference = flash - eager
        denominator = torch.linalg.vector_norm(eager).clamp_min(1e-12)
        rows.append({
            "max_abs": float(difference.abs().max()),
            "relative_l2": float(
                torch.linalg.vector_norm(difference) / denominator
            ),
            "argmax_same": bool(eager.argmax() == flash.argmax()),
        })
    return {
        "steps": len(rows),
        "same_length": len(reference) == len(candidate),
        "all_argmax_same": all(row["argmax_same"] for row in rows),
        "worst_max_abs": max((row["max_abs"] for row in rows), default=0.0),
        "worst_relative_l2": max(
            (row["relative_l2"] for row in rows), default=0.0
        ),
        "per_step": rows,
    }


def validate_round_cache_semantics(trace):
    checks = []
    for row in trace:
        if row["m"] < row["Kp"]:
            expected = row["pre"] + row["m"] + 1
        else:
            expected = row["pre"] + row["Kp"]
        checks.append({
            "round": row["round"],
            "m": row["m"],
            "pre": row["pre"],
            "expected_length": expected,
            "target_length": row["tgt_len"],
            "draft_length": row["drf_len"],
            "passed": (
                row["tgt_len"] == expected
                and row["drf_len"] == expected
            ),
        })
    return checks


def run_specdec(target, draft, ids, eos_id, backend):
    target.set_attention_backend(backend)
    draft.set_attention_backend("eager")
    monitor = {
        "target_q_lengths": [],
        "finite": True,
        "cache_consistent": True,
    }
    target_id = id(target)
    original_prefill = spec_dec._prefill
    original_verify = spec_dec._verify
    original_step = spec_dec._step

    def monitored_prefill(model, prompt):
        logits, cache = chunked_prefill(model, prompt)
        monitor["finite"] &= bool(torch.isfinite(logits).all())
        try:
            spec_cache.check_consistent(cache)
        except RuntimeError:
            monitor["cache_consistent"] = False
            raise
        return logits, cache

    def monitored_verify(model, cache, candidates):
        if id(model) == target_id:
            monitor["target_q_lengths"].append(len(candidates))
        rows, cache_block = original_verify(model, cache, candidates)
        monitor["finite"] &= bool(torch.isfinite(rows).all())
        try:
            spec_cache.check_consistent(cache_block)
        except RuntimeError:
            monitor["cache_consistent"] = False
            raise
        return rows, cache_block

    def monitored_step(model, token, cache):
        logits, next_cache = original_step(model, token, cache)
        monitor["finite"] &= bool(torch.isfinite(logits).all())
        try:
            spec_cache.check_consistent(next_cache)
        except RuntimeError:
            monitor["cache_consistent"] = False
            raise
        return logits, next_cache

    trace = []
    spec_dec._prefill = monitored_prefill
    spec_dec._verify = monitored_verify
    spec_dec._step = monitored_step
    try:
        started = synchronize()
        output = spec_dec.spec_decode_greedy(
            target,
            draft,
            ids,
            MAX_NEW,
            k=K,
            eos_id=eos_id,
            target_kv="base_bf16",
            draft_kv="base_bf16",
            log=lambda row: trace.append(row),
        )
        elapsed_s = synchronize() - started
    finally:
        spec_dec._prefill = original_prefill
        spec_dec._verify = original_verify
        spec_dec._step = original_step

    monitor["round_cache_checks"] = validate_round_cache_semantics(trace)
    monitor["round_cache_semantics"] = all(
        row["passed"] for row in monitor["round_cache_checks"]
    )
    return {
        "backend": backend,
        "tokens": output["gen"],
        "stats": output["stats"],
        "trace": trace,
        "monitor": monitor,
        "elapsed_s": elapsed_s,
        "final_target_cache_length": trace[-1]["tgt_len"] if trace else ids.shape[1],
        "final_draft_cache_length": trace[-1]["drf_len"] if trace else ids.shape[1],
    }


def compare_specdec(eager, flash):
    stat_keys = (
        "rounds", "proposed", "accepted", "mismatch_rounds",
        "all_accept_rounds", "rollbacks", "target_block_calls", "draft_calls",
    )
    trace_keys = ("Kp", "m", "pre", "tgt_len", "drf_len", "cands", "g", "emitted")
    stats_equal = {
        key: eager["stats"].get(key) == flash["stats"].get(key)
        for key in stat_keys
    }
    traces_equal = (
        len(eager["trace"]) == len(flash["trace"])
        and all(
            all(left.get(key) == right.get(key) for key in trace_keys)
            for left, right in zip(eager["trace"], flash["trace"])
        )
    )
    return {
        "tokens_identical": eager["tokens"] == flash["tokens"],
        "stats_equal": stats_equal,
        "all_stats_equal": all(stats_equal.values()),
        "trace_equal": traces_equal,
        "target_cache_length_equal": (
            eager["final_target_cache_length"]
            == flash["final_target_cache_length"]
        ),
        "draft_cache_length_equal": (
            eager["final_draft_cache_length"]
            == flash["final_draft_cache_length"]
        ),
        "both_finite": (
            eager["monitor"]["finite"] and flash["monitor"]["finite"]
        ),
        "both_cache_consistent": (
            eager["monitor"]["cache_consistent"]
            and flash["monitor"]["cache_consistent"]
            and eager["monitor"]["round_cache_semantics"]
            and flash["monitor"]["round_cache_semantics"]
        ),
        "flash_q_lengths": flash["monitor"]["target_q_lengths"],
        "observed_m": [row["m"] for row in flash["trace"]],
        "observed_pre": [row["pre"] for row in flash["trace"]],
    }


def constructed_m_coverage():
    """Exercise every rollback branch without depending on model acceptance."""
    vocabulary = 64
    prompt_length = 7
    candidates = [10, 11, 12, 13]
    corrective = 42

    class DummyModel:
        def set_kv_format(self, _fmt):
            return None

        def embed_tokens(self, ids):
            return ids

    def make_cache(values):
        values = torch.tensor(values, dtype=torch.float32).view(1, 1, -1, 1)
        return [
            {
                "k": values.expand(1, 2, -1, 4).clone() + layer * 1000,
                "v": values.expand(1, 2, -1, 4).clone() + layer * 1000 + 500,
            }
            for layer in range(2)
        ]

    def extend(cache, values):
        addition = make_cache(values)
        return [
            {
                key: torch.cat([old[key], new[key]], dim=2)
                for key in ("k", "v")
            }
            for old, new in zip(cache, addition)
        ]

    def cache_payload_is(cache, expected):
        expected = torch.tensor(expected, dtype=torch.float32)
        spec_cache.check_consistent(cache)
        if spec_cache.cache_length(cache) != len(expected):
            return False
        for layer_index, layer in enumerate(cache):
            for key, offset in (("k", 0), ("v", 500)):
                observed = layer[key][0, 0, :, 0]
                wanted = expected + layer_index * 1000 + offset
                if not torch.equal(observed, wanted):
                    return False
        return True

    def logits_for(token, rows=1):
        logits = torch.full((1, rows, vocabulary), -1000.0)
        logits[..., token] = 1000.0
        return logits

    original = (
        spec_dec._prefill, spec_dec._propose,
        spec_dec._verify, spec_dec._step,
    )
    records = []
    try:
        for accepted in range(K + 1):
            target = DummyModel()
            draft = DummyModel()
            desired = [
                candidates[index] if index < accepted else corrective
                for index in range(K)
            ]

            prefix = list(range(prompt_length))

            def fake_prefill(model, _ids):
                model.last_cache = make_cache(prefix)
                return logits_for(desired[0]), model.last_cache

            def fake_propose(draft_model, cache, _logits, k_eff, _eos):
                proposed = candidates[:k_eff]
                draft_model.last_cache = extend(cache, proposed)
                return proposed, draft_model.last_cache, logits_for(1)

            def fake_verify(target_model, cache, proposed):
                rows = torch.full((len(proposed), vocabulary), -1000.0)
                for index in range(len(proposed)):
                    token = desired[index + 1] if index + 1 < K else 1
                    rows[index, token] = 1000.0
                target_model.last_cache = extend(cache, proposed)
                return rows, target_model.last_cache

            def fake_step(model, token, cache):
                model.last_cache = extend(cache, [token])
                return logits_for(token), model.last_cache

            spec_dec._prefill = fake_prefill
            spec_dec._propose = fake_propose
            spec_dec._verify = fake_verify
            spec_dec._step = fake_step
            trace = []
            result = spec_dec.spec_decode_greedy(
                target,
                draft,
                torch.ones(1, prompt_length, dtype=torch.long),
                K,
                k=K,
                eos_id=(corrective if accepted < K else -1),
                log=lambda row: trace.append(row),
            )
            expected_length = (
                prompt_length + accepted + 1
                if accepted < K else prompt_length + K
            )
            expected_payload = prefix + candidates[:accepted]
            if accepted < K:
                expected_payload.append(corrective)
            row = trace[0]
            passed = bool(
                len(trace) == 1
                and row["m"] == accepted
                and row["pre"] == prompt_length
                and row["tgt_len"] == expected_length
                and row["drf_len"] == expected_length
                and result["stats"]["rollbacks"] == (accepted < K)
                and result["stats"]["mismatch_rounds"] == (accepted < K)
                and result["stats"]["all_accept_rounds"] == (accepted == K)
                and cache_payload_is(target.last_cache, expected_payload)
                and cache_payload_is(draft.last_cache, expected_payload)
            )
            records.append({
                "m": accepted,
                "pre": row["pre"],
                "target_length": row["tgt_len"],
                "draft_length": row["drf_len"],
                "expected_length": expected_length,
                "expected_payload": expected_payload,
                "target_payload_ok": cache_payload_is(
                    target.last_cache, expected_payload,
                ),
                "draft_payload_ok": cache_payload_is(
                    draft.last_cache, expected_payload,
                ),
                "rollbacks": result["stats"]["rollbacks"],
                "passed": passed,
            })
    finally:
        (
            spec_dec._prefill, spec_dec._propose,
            spec_dec._verify, spec_dec._step,
        ) = original
    return {
        "records": records,
        "covered_m": [row["m"] for row in records if row["passed"]],
        "passed": all(row["passed"] for row in records),
    }


def strip_heavy(record):
    return {
        key: value for key, value in record.items()
        if key != "decision_logits"
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    torch.manual_seed(0)
    rollback_coverage = constructed_m_coverage()
    print(
        "constructed rollback coverage: "
        + ", ".join(
            f"m={row['m']}:{'PASS' if row['passed'] else 'CHECK'}"
            for row in rollback_coverage["records"]
        ),
        flush=True,
    )

    text, source = load_text()
    if source.startswith("FALLBACK-SYNTHETIC"):
        raise RuntimeError(
            "real WikiText input is required; place data/wikitext-2-raw-v1.zip "
            "or set P4_TEXT"
        )
    from transformers import AutoTokenizer
    from .g4_b4_cache_tests import _from_pretrained
    tokenizer = _from_pretrained(
        AutoTokenizer, TARGET, trust_remote_code=True,
    )
    all_ids = tokenizer(
        text[:400_000], return_tensors="pt",
    ).input_ids
    if all_ids.shape[1] < max(CONTEXTS):
        raise RuntimeError(
            f"corpus has {all_ids.shape[1]} tokens, need {max(CONTEXTS)}"
        )

    print("[load] target 1.5B", flush=True)
    target, _ = load_ours(TARGET)
    print("[load] draft 0.5B", flush=True)
    draft, _ = load_ours(DRAFT)
    eos_id = tokenizer.eos_token_id
    context_results = {}
    overall = rollback_coverage["passed"]

    for context in CONTEXTS:
        ids = all_ids[:, :context].cuda()
        print(f"\n=== ctx={context} AR ===", flush=True)
        ar_records = {}
        for backend in BACKENDS:
            ar_records[backend] = run_ar(target, ids, eos_id, backend)
            row = ar_records[backend]
            print(
                f"{backend:24s} prefill={row['prefill_s']:.3f}s "
                f"decode={row['decode_s']:.3f}s "
                f"cache={row['final_cache_length']} finite={row['finite']}"
            )
        ar_compare = logits_comparison(
            ar_records["eager"]["decision_logits"],
            ar_records["triton_bf16_two_pass"]["decision_logits"],
        )
        ar_pass = bool(
            ar_records["eager"]["tokens"]
            == ar_records["triton_bf16_two_pass"]["tokens"]
            and ar_records["eager"]["finite"]
            and ar_records["triton_bf16_two_pass"]["finite"]
            and ar_records["eager"]["prefill_cache_length"] == context
            and ar_records["triton_bf16_two_pass"]["prefill_cache_length"] == context
            and ar_records["eager"]["final_cache_length"]
            == ar_records["triton_bf16_two_pass"]["final_cache_length"]
            and ar_compare["same_length"]
            and ar_compare["all_argmax_same"]
        )
        print(
            f"AR tokens_equal="
            f"{ar_records['eager']['tokens'] == ar_records['triton_bf16_two_pass']['tokens']} "
            f"worst_rel_l2={ar_compare['worst_relative_l2']:.6f} pass={ar_pass}"
        )

        print(f"=== ctx={context} SpecDec ===", flush=True)
        spec_records = {}
        for backend in BACKENDS:
            spec_records[backend] = run_specdec(
                target, draft, ids, eos_id, backend,
            )
            row = spec_records[backend]
            print(
                f"{backend:24s} elapsed={row['elapsed_s']:.3f}s "
                f"rounds={row['stats']['rounds']} "
                f"accepted={row['stats']['accepted']}/{row['stats']['proposed']} "
                f"rollback={row['stats']['rollbacks']} "
                f"q={row['monitor']['target_q_lengths']}"
            )
        spec_compare = compare_specdec(
            spec_records["eager"], spec_records["triton_bf16_two_pass"],
        )
        spec_pass = bool(
            spec_compare["tokens_identical"]
            and spec_compare["all_stats_equal"]
            and spec_compare["trace_equal"]
            and spec_compare["target_cache_length_equal"]
            and spec_compare["draft_cache_length_equal"]
            and spec_compare["both_finite"]
            and spec_compare["both_cache_consistent"]
            and all(1 <= q_len <= K for q_len in spec_compare["flash_q_lengths"])
            and K in spec_compare["flash_q_lengths"]
        )
        print(
            f"SpecDec tokens_equal={spec_compare['tokens_identical']} "
            f"stats_equal={spec_compare['all_stats_equal']} "
            f"trace_equal={spec_compare['trace_equal']} "
            f"cache_ok={spec_compare['both_cache_consistent']} pass={spec_pass}"
        )
        overall &= ar_pass and spec_pass
        context_results[str(context)] = {
            "ar": {
                "eager": strip_heavy(ar_records["eager"]),
                "flash": strip_heavy(ar_records["triton_bf16_two_pass"]),
                "comparison": ar_compare,
                "passed": ar_pass,
            },
            "specdec": {
                "eager": spec_records["eager"],
                "flash": spec_records["triton_bf16_two_pass"],
                "comparison": spec_compare,
                "passed": spec_pass,
            },
        }
        del ids, ar_records, spec_records
        gc.collect()
        torch.cuda.empty_cache()

    result = {
        "status": "latest_main_flash_g4_compatibility",
        "main_fresh_start_sha": "d4eb38967f5f51e7b2cd5292e38b6a06f1d8f897",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "target": TARGET,
        "draft": DRAFT,
        "contexts": CONTEXTS,
        "max_new_tokens": MAX_NEW,
        "k": K,
        "chunk": CHUNK,
        "text_source": source,
        "constructed_rollback_coverage": rollback_coverage,
        "context_results": context_results,
        "decision": (
            "PASS_FLASH_G4_COMPATIBILITY"
            if overall else "CHECK_FLASH_G4_COMPATIBILITY"
        ),
    }
    output = Path(os.environ.get(
        "SPECDEC_FLASH_G4_OUTDIR", "output/flash_g4",
    )) / "flash_attention_g4_specdec_compat.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\ndecision={result['decision']}\nJSON={output}")
    del target, draft
    gc.collect()
    torch.cuda.empty_cache()
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
