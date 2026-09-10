"""Pilot for recent-BF16 residual sizes in PC-INT8 K cache.

Uses only the original pilot windows 0..3.  No new quality candidate may use
the already-read confirmation windows 40..79; a selected candidate proceeds to
fresh offsets 80..119.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch

from .fp8_kv_v4_dppl import ensure_data
from .kv_policy_pilot_v2 import MODEL, WINDOW, load_model

DTYPE = torch.bfloat16
CHUNK = 512
LOSS_ROWS = 64
N_WINDOWS = 4
WINDOW_OFFSET = 0


@dataclass(frozen=True)
class Policy:
    name: str
    fmt: str
    residual: int = 0


POLICIES = (
    Policy("base_bf16", "fp16"),
    Policy("current_vfp8", "v_fp8"),
    Policy("kpcint8_b32_r0_vfp8", "k_pc_int8_v_fp8", 0),
    Policy("kpcint8_b32_r128_vfp8", "k_pc_int8_v_fp8", 128),
    Policy("kpcint8_b32_r256_vfp8", "k_pc_int8_v_fp8", 256),
)


def cache_logical_bytes(cache_list) -> int:
    return sum(
        int(value.numel()) * value.element_size()
        for layer in cache_list
        for value in layer.values()
        if isinstance(value, torch.Tensor)
    )


def window_stats(model, ids_w: torch.Tensor, policy: Policy):
    model.set_kv_head_policy(2, 2)
    model.set_k_residual_length(policy.residual)
    model.set_kv_format(policy.fmt)
    cache = None
    nll_sum = 0.0
    n_tok = 0
    finite = True
    with torch.inference_mode():
        for start in range(0, WINDOW, CHUNK):
            piece = ids_w[start:start + CHUNK]
            logits, cache = model._run(
                model.embed_tokens(piece.unsqueeze(0)), cache, need_logits=True,
            )
            finite &= bool(torch.isfinite(logits).all().item())
            targets = ids_w[start + 1:start + 1 + CHUNK]
            rows = targets.shape[0]
            for row_start in range(0, rows, LOSS_ROWS):
                row_end = min(row_start + LOSS_ROWS, rows)
                log_probs = torch.log_softmax(
                    logits[0, row_start:row_end].to(torch.float32), dim=-1,
                )
                idx = torch.arange(row_end - row_start, device=logits.device)
                nll_sum += -log_probs[idx, targets[row_start:row_end]].sum().item()
                del log_probs, idx
            n_tok += rows
            del logits
    return nll_sum, n_tok, finite, cache_logical_bytes(cache)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    torch.manual_seed(0)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tokenizer.model_max_length = 10**9
    with open(ensure_data(), encoding="utf-8") as stream:
        all_ids = torch.tensor(tokenizer(stream.read())["input_ids"], dtype=torch.long)
    needed = (WINDOW_OFFSET + N_WINDOWS) * WINDOW
    if all_ids.numel() < needed:
        raise RuntimeError(f"dataset has {all_ids.numel()} tokens, need {needed}")

    model = load_model()
    print(
        f"residual pilot: model=1.5B native_kv_heads=2 windows=0..3 "
        f"W={WINDOW} chunk={CHUNK}", flush=True,
    )
    print("confirmation windows 40..79 are not read; next fresh set is 80..119\n")
    records = []
    started = time.perf_counter()
    for policy in POLICIES:
        total_nll = 0.0
        total_tokens = 0
        finite = True
        logical_bytes = None
        policy_started = time.perf_counter()
        for wi in range(N_WINDOWS):
            offset = (WINDOW_OFFSET + wi) * WINDOW
            ids_w = all_ids[offset:offset + WINDOW].cuda()
            nll, tokens, fin, cache_bytes = window_stats(model, ids_w, policy)
            total_nll += nll
            total_tokens += tokens
            finite &= fin
            logical_bytes = cache_bytes
            del ids_w
            torch.cuda.empty_cache()
        mean_nll = total_nll / total_tokens
        ppl = math.exp(mean_nll)
        bytes_per_token_layer = logical_bytes / (WINDOW * len(model.layers))
        record = {
            "name": policy.name,
            "nll": mean_nll,
            "ppl": ppl,
            "finite": finite,
            "bytes_per_token_layer": bytes_per_token_layer,
            "cache_mib": logical_bytes / 2**20,
            "wall_s": time.perf_counter() - policy_started,
        }
        records.append(record)
        print(
            f"{policy.name:30s} NLL={mean_nll:.6f} PPL={ppl:.4f} "
            f"cache={record['cache_mib']:.2f}MiB "
            f"B/tok/layer={bytes_per_token_layer:.1f} finite={finite} "
            f"wall={record['wall_s']:.0f}s", flush=True,
        )

    base = records[0]
    current = records[1]
    print("\n=== residual pilot comparison ===")
    for record in records:
        dnll = record["nll"] - base["nll"]
        relative_ppl = (record["ppl"] / base["ppl"] - 1.0) * 100.0
        compression = base["bytes_per_token_layer"] / record["bytes_per_token_layer"]
        bytes_vs_current = record["bytes_per_token_layer"] / current["bytes_per_token_layer"]
        print(
            f"{record['name']:30s} dNLL={dnll:+.6f} "
            f"rel_dPPL={relative_ppl:+.3f}% compression={compression:.3f}x "
            f"bytes_vs_current={bytes_vs_current:.3f}x",
        )
    print(f"\ntotal wall={time.perf_counter() - started:.0f}s")
    print("Select at most one residual size; any confirmation uses fresh windows 80..119.")
    return 0 if all(record["finite"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
