"""Pilot: native/reduced GQA KV heads x asymmetric KV quantization.

This is candidate screening only.  It deliberately uses the existing pilot
WikiText-2 windows 0..3 and never touches formal confirmation windows 40..79.
No candidate is promoted automatically; compare dNLL first and logical bytes
second, then confirm at most one selected candidate on the frozen formal set.

Run after ``kv_policy_v1``:
    python -m specdec.engine.kv_policy_pilot
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch

from .align_ours import build_from_config
from .fp8_kv_v4_dppl import ensure_data

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DTYPE = torch.bfloat16
WINDOW = 2048
CHUNK = 512
LOSS_ROWS = 64
N_WINDOWS = 4
WINDOW_OFFSET = 0


@dataclass(frozen=True)
class Policy:
    name: str
    fmt: str
    k_heads: int
    v_heads: int


# Complete small factorial requested by the team: native/reduced K heads x
# native/reduced V heads x the current/new one-byte quantization policy.
POLICIES = (
    Policy("base_bf16_h2_h2", "fp16", 2, 2),
    Policy("vfp8_h2_h2_current", "v_fp8", 2, 2),
    Policy("vfp8_h1_h2", "v_fp8", 1, 2),
    Policy("vfp8_h2_h1", "v_fp8", 2, 1),
    Policy("vfp8_h1_h1", "v_fp8", 1, 1),
    Policy("ki8_vf8_h2_h2", "k_int8_v_fp8", 2, 2),
    Policy("ki8_vf8_h1_h2", "k_int8_v_fp8", 1, 2),
    Policy("ki8_vf8_h2_h1", "k_int8_v_fp8", 2, 1),
    Policy("ki8_vf8_h1_h1", "k_int8_v_fp8", 1, 1),
)


def load_model():
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    native = int(cfg.num_key_value_heads)
    if native != 2:
        raise RuntimeError(f"probe is frozen for Qwen2.5-1.5B kv_heads=2, got {native}")
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, config=cfg, attn_implementation="eager",
        torch_dtype=DTYPE, trust_remote_code=True,
    ).to("cuda").eval()
    ours = build_from_config(cfg)
    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    if cfg.to_dict().get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    missing, unexpected = ours.load_state_dict(sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing}, unexpected={unexpected}")
    del hf
    torch.cuda.empty_cache()
    return ours.to("cuda", DTYPE).eval()


def cache_logical_bytes(cache_list) -> int:
    return sum(
        int(value.numel()) * value.element_size()
        for layer in cache_list
        for value in layer.values()
        if isinstance(value, torch.Tensor)
    )


def window_stats(model, ids_w: torch.Tensor, policy: Policy):
    model.set_kv_head_policy(policy.k_heads, policy.v_heads)
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
        f"model=1.5B q_heads=12 native_kv_heads=2 dtype={DTYPE} "
        f"windows={WINDOW_OFFSET}..{WINDOW_OFFSET + N_WINDOWS - 1} "
        f"W={WINDOW} chunk={CHUNK}", flush=True,
    )
    print("screening only; formal windows 40..79 remain untouched\n", flush=True)

    records = []
    started = time.perf_counter()
    for policy in POLICIES:
        total_nll = 0.0
        total_tokens = 0
        finite = True
        logical_bytes = None
        p0 = time.perf_counter()
        for wi in range(N_WINDOWS):
            offset = (WINDOW_OFFSET + wi) * WINDOW
            window = all_ids[offset:offset + WINDOW].cuda()
            nll, tokens, fin, cache_bytes = window_stats(model, window, policy)
            total_nll += nll
            total_tokens += tokens
            finite &= fin
            logical_bytes = cache_bytes
            del window
            torch.cuda.empty_cache()
        mean_nll = total_nll / total_tokens
        ppl = math.exp(mean_nll)
        bytes_per_token_layer = logical_bytes / (WINDOW * len(model.layers))
        record = dict(
            name=policy.name, fmt=policy.fmt, k_heads=policy.k_heads,
            v_heads=policy.v_heads, nll=mean_nll, ppl=ppl, finite=finite,
            cache_mib=logical_bytes / 2**20,
            bytes_per_token_layer=bytes_per_token_layer,
            wall_s=time.perf_counter() - p0,
        )
        records.append(record)
        print(
            f"{policy.name:22s} NLL={mean_nll:.6f} PPL={ppl:.4f} "
            f"cache={record['cache_mib']:.2f}MiB "
            f"B/tok/layer={bytes_per_token_layer:.1f} finite={finite} "
            f"wall={record['wall_s']:.0f}s", flush=True,
        )

    base = records[0]
    print("\n=== comparison vs base (pilot evidence only) ===")
    for record in records:
        record["dNLL"] = record["nll"] - base["nll"]
        record["relative_dPPL_pct"] = (record["ppl"] / base["ppl"] - 1.0) * 100.0
        record["compression_x"] = base["bytes_per_token_layer"] / record["bytes_per_token_layer"]
        print(
            f"{record['name']:22s} dNLL={record['dNLL']:+.6f} "
            f"rel_dPPL={record['relative_dPPL_pct']:+.3f}% "
            f"compression={record['compression_x']:.3f}x",
        )
    print(f"\ntotal wall={time.perf_counter() - started:.0f}s")
    print("No automatic PASS/winner. Select at most one candidate using pilot dNLL first,")
    print("then confirm it on frozen windows 40..79 before changing the G3 design.")
    return 0 if all(r["finite"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
