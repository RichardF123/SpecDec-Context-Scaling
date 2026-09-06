"""align_cache_diag.py — where do full-forward and cached-decode diverge?

Full path: one call with 10 tokens.  Cache path: prefill 9, then step 1 token.
Hooks are attached ONLY during the run whose layers we want to capture.
Run:  python -m specdec.engine.align_cache_diag
"""
from __future__ import annotations

import torch

from .model_self import QwenLM


def main() -> int:
    torch.manual_seed(0)
    model = QwenLM(vocab_size=5000, hidden=64, heads=8, kv_heads=2, head_dim=8,
                   intermediate=256, num_layers=2, rope_theta=1_000_000.0)
    model.eval()
    ids = torch.randint(0, 5000, (1, 10))

    caps_full, caps_step = [], []

    def attach(cap):
        hs = []
        for layer in model.layers:
            def hook(m, inp, out, cap=cap):
                cap.append(out if isinstance(out, torch.Tensor) else out[0])
            hs.append(layer.register_forward_hook(hook))
        return hs

    # --- full path (10 tokens), capture layer outputs ---
    hs = attach(caps_full)
    with torch.no_grad():
        full = model(ids)
    for h in hs:
        h.remove()

    # --- cache path: prefill 9 (no capture), then step token 10 (capture) ---
    with torch.no_grad():
        x9 = model.embed_tokens(ids[:, :9])
        _, caches = model._run(x9, None)
    hs = attach(caps_step)
    with torch.no_grad():
        x10 = model.embed_tokens(ids[:, 9:])
        step_logits, _ = model._run(x10, caches)
    for h in hs:
        h.remove()

    diff = (full[:, -1] - step_logits[:, -1]).abs().max().item()
    print(f"final logits diff (last pos): {diff:.3e}")
    print(f"captured layers: full={len(caps_full)} step={len(caps_step)}")
    print("layer  full-vs-step max|Δ| (last position)")
    for i, (cf, cs) in enumerate(zip(caps_full, caps_step)):
        d = (cf[:, -1] - cs[:, -1]).abs().max().item()
        print(f"L{i:<2d}    {d:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
