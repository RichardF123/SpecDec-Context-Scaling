"""engine.sampling — sampling utilities for decode (OURS). L1 uses greedy only."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """Argmax over last dim. logits: (..., V)."""
    return torch.argmax(logits, dim=-1)


def top_k_top_p(logits: torch.Tensor, temperature: float = 1.0,
                top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Sample one token id per row. logits: (..., V). top_k=0 disables; top_p=1.0 disables."""
    logits = logits / max(temperature, 1e-6)
    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        topk = torch.topk(logits, k, dim=-1)
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, topk.indices, topk.values)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
