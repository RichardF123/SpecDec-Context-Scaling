"""eval_l4_ppl.py — L4 weight-quantization quality gate.

Measures perplexity on one fixed local text corpus for fp16 / int8 / int4.
The corpus is frozen before inspecting L4 PPL results.

Run:
    python -m specdec.quant.eval_l4_ppl --mode fp16
    python -m specdec.quant.eval_l4_ppl --mode int8
    python -m specdec.quant.eval_l4_ppl --mode int4
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from specdec.engine.align_ours import build_from_config
from specdec.quant.int8 import quantize_model_int8
from specdec.quant.int4 import quantize_model_int4


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Frozen before L4 perplexity results are inspected.
QUALITY_TEXT = """
Scientific experiments are useful only when their measurements can be repeated.
A careful comparison changes one factor at a time, records the conditions of
each run, and reports negative results instead of hiding them.

Modern language models predict tokens from context. During autoregressive
generation, the model repeatedly processes new tokens while attending to the
previous sequence. Caching previously computed keys and values avoids repeating
much of this work.

Quantization reduces the numerical precision used to represent model weights.
Lower precision can reduce memory consumption, but aggressive quantization may
also introduce approximation error. A useful inference system therefore has to
measure both efficiency and quality.

Computer systems often contain trade-offs between storage, computation, and
communication. An optimization that saves memory does not necessarily reduce
latency if it introduces additional conversion or synchronization overhead.

Machine learning evaluation should distinguish implementation correctness from
model quality and system performance. Two implementations may produce nearly
the same predictions while having very different memory usage or execution
time.

In a well-designed benchmark, hardware, model configuration, random seeds,
input data, and measurement protocol are recorded explicitly. This makes it
possible for another machine to reproduce the experiment and explain any
remaining differences.

Long-context inference increases the amount of information stored in the
attention cache. As the context becomes longer, memory traffic and attention
work can become increasingly important components of decoding latency.

Speculative decoding uses a smaller model to propose future tokens and a larger
model to verify them. Its benefit depends on how many proposals are accepted
and how much verification costs relative to ordinary decoding.

Reliable engineering does not assume that every optimization will improve
every metric. A method that reduces memory but decreases throughput can still
be useful when memory capacity is the binding resource.

The purpose of this evaluation is to compare several numerical representations
of the same language model under identical inputs. The text and evaluation
procedure remain fixed across all modes so that changes in perplexity reflect
the quantization path rather than changes in the test data.
"""


def build_model(mode: str, group_size: int):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        local_files_only=True,
    )
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        local_files_only=True,
    )

    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=cfg,
        attn_implementation="eager",
        torch_dtype=torch.float32,
        trust_remote_code=True,
        local_files_only=True,
    )
    hf.eval()

    try:
        hf_q = hf.model.layers[0].self_attn.q_proj
        attn_bias = getattr(hf_q, "bias", None) is not None
    except Exception:
        attn_bias = False

    ours = build_from_config(cfg, attn_bias_force=attn_bias)

    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    if cfg.to_dict().get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)

    res = ours.load_state_dict(sd, strict=False)
    if res.missing_keys:
        raise RuntimeError(f"missing weights: {res.missing_keys}")

    del hf

    if mode == "int8":
        replaced = quantize_model_int8(ours)
        print("int8 replaced:", replaced)
    elif mode == "int4":
        replaced = quantize_model_int4(ours, group_size=group_size)
        print("int4 replaced:", replaced, "group_size:", group_size)

    ours = ours.to(device="cuda", dtype=torch.float16).eval()
    return ours, tok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fp16", "int8", "int4"), required=True)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("GATE: FAIL — CUDA required")
        return 1

    torch.manual_seed(0)

    print("== L4 A6 quality evidence ==")
    print("model:", MODEL_ID)
    print("mode:", args.mode)
    print("gpu:", torch.cuda.get_device_name(0))
    print("dtype: fp16 compute")
    print("corpus: frozen local text")
    print("[1/2] loading OURS engine + weights")

    model, tok = build_model(args.mode, args.group_size)

    ids = tok(
        QUALITY_TEXT,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[:, : args.max_tokens]

    if ids.shape[1] < 2:
        raise RuntimeError("quality corpus produced too few tokens")

    inputs = ids[:, :-1].to("cuda")
    targets = ids[:, 1:].to("cuda")

    print("[2/2] computing perplexity")
    with torch.no_grad():
        logits = model(inputs)
        loss_sum = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="sum",
        )

    token_count = targets.numel()
    mean_nll = float(loss_sum) / token_count
    ppl = math.exp(mean_nll)

    print("target tokens:", token_count)
    print(f"mean NLL: {mean_nll:.6f}")
    print(f"PPL: {ppl:.6f}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
