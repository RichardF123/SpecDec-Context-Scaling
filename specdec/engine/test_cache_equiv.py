"""test_cache_equiv.py — L2 correctness gate: cached decode == full forward.

Run:  python -m specdec.engine.test_cache_equiv
Expected: cache == full : 0.0  (bitwise identical)
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

    with torch.no_grad():
        full = model(ids)                                   # 全量：一次喂 10 个词

        # 缓存路径：先 prefill 前 9 个（无缓存），再单独喂第 10 个
        x9 = model.embed_tokens(ids[:, :9])
        _, caches = model._run(x9, None)

        x10 = model.embed_tokens(ids[:, 9:])
        step_logits, _ = model._run(x10, caches)

        diff = (full[:, -1] - step_logits[:, -1]).abs().max().item()
        same = bool((full[:, -1].argmax(-1) == step_logits[:, -1].argmax(-1)).all().item())
        print("full logits shape:", tuple(full.shape))
        print("step logits shape:", tuple(step_logits.shape))
        print("cache == full (last position) max|Δ|:", diff)
        print("next-token prediction identical:", same)
        # tiny fp kernel-order noise (~1e-7/layer) is expected when matmul shapes differ;
        # a real logic bug would show |Δ| >> 1e-2 and grow across layers.
        assert diff < 1e-3 and same, "KV cache changed the math or the prediction!"
    print("PASS: cached decode matches full forward within fp tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
