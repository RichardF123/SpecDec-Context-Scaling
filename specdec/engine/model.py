"""engine.model — from-scratch Qwen2-family forward pass (L1, no KV cache yet).

OURS: architecture, attention, RoPE, RMSNorm, SwiGLU, forward loop are written here.
LIB: weight tensors are loaded from a HuggingFace checkpoint (loading/serialization only).

Sub-module names intentionally mirror HF key names so the HF state_dict can be copied
directly; the math is our own.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HF-style 'rotate half' rope convention (Qwen2 uses this, not interleaved)."""
    d = x.shape[-1]
    h = d // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B,H,S,D), cos/sin: (1,1,S,D)."""
    return (x * cos) + (rotate_half(x) * sin)


def build_rope_cache(head_dim: int, seq_len: int, theta: float, device, dtype):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float64) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float64)
    freqs = torch.outer(positions, inv_freq).to(dtype)          # (S, head_dim/2)
    cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0)            # (1,1,S,hd/2)
    sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)
    # broadcast over the full head_dim when multiplied with the concatenated rotate_half form
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """x: (B, kv_heads, S, head_dim) -> repeat each head n_rep times (GQA)."""
    if n_rep == 1:
        return x
    b, kv, s, d = x.shape
    return x.unsqueeze(2).expand(b, kv, n_rep, s, d).reshape(b, kv * n_rep, s, d)


class SelfAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)

    def forward(self, x: torch.Tensor, cos, sin) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = repeat_kv(k, self.heads // self.kv_heads)
        v = repeat_kv(v, self.heads // self.kv_heads)
        attn = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        # causal mask (upper triangle) - Qwen2 has no windowing for the base model
        mask = torch.full((s, s), float("-inf"), device=x.device, dtype=x.dtype)
        mask = torch.triu(mask, diagonal=1)
        attn = attn + mask
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(x.dtype)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, s, self.heads * self.head_dim)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int,
                 intermediate: int, rms_eps: float):
        super().__init__()
        self.self_attn = SelfAttention(hidden, heads, kv_heads, head_dim)
        self.mlp = MLP(hidden, intermediate)
        self.input_layernorm = RMSNorm(hidden, rms_eps)
        self.post_attention_layernorm = RMSNorm(hidden, rms_eps)

    def forward(self, x: torch.Tensor, cos, sin) -> torch.Tensor:
        h = self.input_layernorm(x)
        x = x + self.self_attn(h, cos, sin)
        h = self.post_attention_layernorm(x)
        x = x + self.mlp(h)
        return x


class Qwen2Engine(nn.Module):
    """From-scratch Qwen2-style CausalLM container (no KV cache yet = L1).

    Weights are copied in from a HF checkpoint by matching sub-module names.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.hidden = int(cfg["hidden_size"])
        self.heads = int(cfg["num_attention_heads"])
        self.kv_heads = int(cfg.get("num_key_value_heads", self.heads))
        self.head_dim = int(cfg.get("head_dim", self.hidden // self.heads))
        self.layers_n = int(cfg["num_hidden_layers"])
        self.intermediate = int(cfg["intermediate_size"])
        self.rms_eps = float(cfg.get("rms_norm_eps", 1e-6))
        self.rope_theta = float(cfg.get("rope_theta", 10000.0))
        self.tie_word_embeddings = bool(cfg.get("tie_word_embeddings", False))

        self.embed_tokens = nn.Embedding(int(cfg["vocab_size"]), self.hidden)
        self.layers = nn.ModuleList([
            DecoderLayer(self.hidden, self.heads, self.kv_heads, self.head_dim,
                         self.intermediate, self.rms_eps)
            for _ in range(self.layers_n)
        ])
        self.norm = RMSNorm(self.hidden, self.rms_eps)
        # lm_head created at load time ONLY when embeddings are not tied.
        self.lm_head: Optional[nn.Linear] = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, s = input_ids.shape
        cos, sin = build_rope_cache(self.head_dim, s, self.rope_theta,
                                    input_ids.device, input_ids.dtype)
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        if self.tie_word_embeddings or self.lm_head is None:
            logits = F.linear(x, self.embed_tokens.weight)
        else:
            logits = self.lm_head(x)
        return logits  # (B, S, V)


def load_from_hf(state_dict: dict, cfg: dict) -> Qwen2Engine:
    """Copy HF Qwen2 state_dict into our engine (weight LOADING only).

    Accepts both HF prefixes ("model.embed_tokens.*") and bare ones. HF Qwen2
    checkpoints with tied embeddings omit lm_head.weight; our forward then reads
    logits from the embedding weight directly.
    """
    if any(k.startswith("model.") for k in state_dict):
        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
    engine = Qwen2Engine(cfg)
    has_lm = any(k.startswith("lm_head.") for k in state_dict)
    if has_lm:
        engine.lm_head = nn.Linear(engine.hidden, int(cfg["vocab_size"]), bias=False)
    engine.load_state_dict(state_dict, strict=False)
    return engine
