"""Second-round pilot: native heads with better asymmetric KV quantization.

Uses only WikiText-2 pilot windows 0..3.  It excludes the failed head-reduction
and per-token K-INT8 variants from round 1 and does not read formal windows.
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


POLICIES = (
    Policy("base_bf16", "fp16"),
    Policy("current_vfp8", "v_fp8"),
    Policy("candidate_vint4_g32", "v_int4"),
    Policy("candidate_kpcint8_b32_vfp8", "k_pc_int8_v_fp8"),
)


def load_model():
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    if int(cfg.num_key_value_heads) != 2:
        raise RuntimeError("pilot is frozen for Qwen2.5-1.5B native kv_heads=2")
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, config=cfg, attn_implementation="eager",
        torch_dtype=DTYPE, trust_remote_code=True,
    ).to("cuda").eval()
    ours = build_from_config(cfg)
    state = {key.removeprefix("model."): value for key, value in hf.state_dict().items()}
    if cfg.to_dict().get("tie_word_embeddings", False):
        state.pop("lm_head.weight", None)
    missing, unexpected = ours.load_state_dict(state, strict=True)
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
    model.set_kv_head_policy(2, 2)
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
    # The corpus is tokenized on CPU and sliced before model execution.  Raising
    # this tokenizer-only warning limit does not change token IDs or model context.
    tokenizer.model_max_length = 10**9
    with open(ensure_data(), encoding="utf-8") as stream:
        all_ids = torch.tensor(tokenizer(stream.read())["input_ids"], dtype=torch.long)
    needed = (WINDOW_OFFSET + N_WINDOWS) * WINDOW
    if all_ids.numel() < needed:
        raise RuntimeError(f"dataset has {all_ids.numel()} tokens, need {needed}")

    model = load_model()
    print(
        f"model=1.5B native_kv_heads=2 dtype={DTYPE} "
        f"windows={WINDOW_OFFSET}..{WINDOW_OFFSET + N_WINDOWS - 1} "
        f"W={WINDOW} chunk={CHUNK}", flush=True,
    )
    print("round-2 screening only; formal windows 40..79 remain untouched\n", flush=True)

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
        record = {
            "name": policy.name,
            "nll": mean_nll,
            "ppl": ppl,
            "finite": finite,
            "cache_mib": logical_bytes / 2**20,
            "bytes_per_token_layer": bytes_per_token_layer,
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
    print("\n=== comparison (round-2 pilot evidence only) ===")
    for record in records:
        dnll = record["nll"] - base["nll"]
        relative_ppl = (record["ppl"] / base["ppl"] - 1.0) * 100.0
        compression = base["bytes_per_token_layer"] / record["bytes_per_token_layer"]
        versus_current = record["bytes_per_token_layer"] / current["bytes_per_token_layer"]
        print(
            f"{record['name']:30s} dNLL={dnll:+.6f} "
            f"rel_dPPL={relative_ppl:+.3f}% compression={compression:.3f}x "
            f"bytes_vs_current={versus_current:.3f}x",
        )
    print(f"\ntotal wall={time.perf_counter() - started:.0f}s")
    print("No automatic winner. Compare quality first; confirm at most one candidate.")
    return 0 if all(record["finite"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
