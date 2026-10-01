"""Install custom kernels on the live engine classes.

Replaces RMSNORM.forward, apply_rope, MLP.forward, and SelfAttention.forward.
Cache layout, FP8 quant, and the attention-phase counter stay on the engine
path. Only the math calls change.
"""

from __future__ import annotations

_ORIG: dict = {}
_PACK = "torch"

_PACKS = {
    "torch": dict(rms=False, rope=False, swiglu=False, attn=False),
    "elementwise": dict(rms=True, rope=True, swiglu=True, attn=False),
    "attention": dict(rms=False, rope=False, swiglu=False, attn=True),
    "all": dict(rms=True, rope=True, swiglu=True, attn=True),
}


def _model():
    from specdec.engine import model_self
    return model_self


def _capture():
    if _ORIG:
        return
    m = _model()
    _ORIG["rope"] = m.apply_rope
    _ORIG["rms"] = m.RMSNORM.forward
    _ORIG["mlp"] = m.MLP.forward
    _ORIG["attn"] = m.SelfAttention.forward


def _rms_forward(self, x):
    from specdec.kernels.ops import rmsnorm
    return rmsnorm(x, self.weight, self.eps, backend="cuda")


def _rope(x, cos, sin):
    from specdec.kernels.ops import apply_rope
    return apply_rope(x, cos, sin, backend="cuda")


def _mlp_forward(self, x):
    from specdec.kernels.ops import swiglu
    gate = self.gate_proj(x)
    up = self.up_proj(x)
    return self.down_proj(swiglu(gate, up, backend="cuda"))


def _attn_forward(self, x, cos, sin, cache=None):
    """Vendor SelfAttention.forward with the eager score tensor replaced."""
    import torch

    from specdec.kernels.ops import attention as attention_op

    m = _model()
    b, s, _ = x.shape
    q = self.q_proj(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
    q = m.apply_rope(q, cos, sin)
    k = m.apply_rope(k, cos, sin)
    fmt = cache.get("fmt", self.kv_format) if cache is not None else self.kv_format
    if fmt == "v_fp8":
        k_on, v_on, gk, gv = False, True, m.GROUP_SIZE, m.GROUP_V
    elif fmt == "fp8":
        k_on, v_on, gk, gv = self.fp8_k, self.fp8_v, self.fp8_g_k, self.fp8_g_v
    else:
        k_on = v_on = False
        gk = gv = m.GROUP_SIZE
    qk_cur = qv_cur = ks_cur = vs_cur = None
    if k_on:
        qk_cur, ks_cur = m.quantize_kv_fp8(k, gk)
        k_cur16 = m.dequantize_kv_fp8(qk_cur, ks_cur, x.dtype)
    else:
        k_cur16 = k
    if v_on:
        qv_cur, vs_cur = m.quantize_kv_fp8(v, gv)
        v_cur16 = m.dequantize_kv_fp8(qv_cur, vs_cur, x.dtype)
    else:
        v_cur16 = v
    if cache is None:
        k_all, v_all = k_cur16, v_cur16
    elif fmt == "fp16":
        k = torch.cat([cache["k"], k], dim=2)
        v = torch.cat([cache["v"], v], dim=2)
        cache["k"] = k
        cache["v"] = v
        k_all, v_all = k, v
    else:
        if "k_scale" in cache:
            k_all = torch.cat([m.dequantize_kv_fp8(cache["k"], cache["k_scale"], x.dtype), k_cur16], dim=2)
        else:
            k_all = torch.cat([cache["k"], k_cur16], dim=2)
            cache["k"] = k_all
        if "v_scale" in cache:
            v_all = torch.cat([m.dequantize_kv_fp8(cache["v"], cache["v_scale"], x.dtype), v_cur16], dim=2)
        else:
            v_all = torch.cat([cache["v"], v_cur16], dim=2)
            cache["v"] = v_all
    t_len = k_all.shape[2] if cache is not None else s
    path, reason = m.decide_attention_path(m._ATTENTION_PHASE, s)
    m._record_attention_call(path, m._ATTENTION_PHASE, s, t_len, reason, self.heads, self.kv_heads, self.head_dim)
    if path == "fused":
        out = m._fused_prefill_attention(q, k_all, v_all, self.head_dim)
    else:
        out = attention_op(q, k_all, v_all, backend="cuda")
    out = out.transpose(1, 2).contiguous().view(b, s, self.heads * self.head_dim)
    if cache is None:
        if fmt in ("fp8", "v_fp8"):
            nc = {"fmt": fmt}
            if k_on:
                nc["k"], nc["k_scale"] = qk_cur, ks_cur
            else:
                nc["k"] = k_all
            if v_on:
                nc["v"], nc["v_scale"] = qv_cur, vs_cur
            else:
                nc["v"] = v_all
            new_cache = nc
        else:
            new_cache = {"k": k_all, "v": v_all}
    elif fmt in ("fp8", "v_fp8"):
        if k_on:
            cache["k"] = torch.cat([cache["k"], qk_cur], dim=2)
            cache["k_scale"] = torch.cat([cache["k_scale"], ks_cur], dim=2)
        if v_on:
            cache["v"] = torch.cat([cache["v"], qv_cur], dim=2)
            cache["v_scale"] = torch.cat([cache["v_scale"], vs_cur], dim=2)
        new_cache = cache
    else:
        new_cache = cache
    return self.o_proj(out), new_cache


def install(pack: str) -> str:
    """Activate a kernel pack. Returns the pack name."""
    if pack not in _PACKS:
        raise ValueError(f"unknown pack {pack!r}; choose from {sorted(_PACKS)}")
    global _PACK
    _capture()
    restore()
    flags = _PACKS[pack]
    m = _model()
    if flags["rope"]:
        m.apply_rope = _rope
    if flags["rms"]:
        m.RMSNORM.forward = _rms_forward
    if flags["swiglu"]:
        m.MLP.forward = _mlp_forward
    if flags["attn"]:
        m.SelfAttention.forward = _attn_forward
    _PACK = pack
    return pack


def restore() -> None:
    global _PACK
    if not _ORIG:
        return
    m = _model()
    m.apply_rope = _ORIG["rope"]
    m.RMSNORM.forward = _ORIG["rms"]
    m.MLP.forward = _ORIG["mlp"]
    m.SelfAttention.forward = _ORIG["attn"]
    _PACK = "torch"


def current_pack() -> str:
    return _PACK
