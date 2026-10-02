import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import contextlib

from . import paged_cache, spec_cache
class RMSNORM(nn.Module):
    def __init__(self, hidden: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.to(torch.float32)
        variance = torch.mean(x32 * x32, dim=-1, keepdim=True)
        normed = x32 * torch.rsqrt(variance + self.eps)
        return self.weight * normed.to(orig_dtype)

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, hidden: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, hidden))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
         return self.weight[token_ids]

class MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        down = self.down_proj(gate * up)
        return down

def Build_Rope_cache(head_dim: int, seq_len: int, theta:float, device: torch.device,dtype: torch.dtype,start_pos: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    calc_dtype = torch.float32
    exponents = torch.arange(0, head_dim, 2, device=device, dtype=calc_dtype) / head_dim
    inv_freq = 1.0 / (theta ** exponents)
    postitions = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=calc_dtype)
    freqs = torch.outer(postitions, inv_freq)
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1).unsqueeze(0).unsqueeze(0)
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1).unsqueeze(0).unsqueeze(0)
    return cos.to(dtype), sin.to(dtype)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    h = d // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)

def kv_repeat(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, kv, s, d = x.shape
    return x.unsqueeze(2).expand(b, kv, n_rep, s, d).reshape(b, kv * n_rep, s, d)

E4M3_MAX = 448.0
GROUP_SIZE = 32
GROUP_V = 128

def quantize_kv_fp8(x: torch.Tensor, group_size: int = GROUP_SIZE):
    d = x.shape[-1]
    assert d % group_size == 0, f"head_dim {d} % group_size {group_size} != 0"
    g = d // group_size
    xg = x.to(torch.float32).view(*x.shape[:-1], g, group_size)
    m = xg.abs().amax(dim=-1, keepdim=True)
    s = m / E4M3_MAX
    s = torch.where(s == 0, torch.ones_like(s), s)
    q = (xg / s).to(torch.float8_e4m3fn)
    q = q.reshape_as(x)
    return q, s.squeeze(-1)

def dequantize_kv_fp8(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float16):
    g = scale.shape[-1]
    qg = q.to(torch.float32).view(*q.shape[:-1], g, q.shape[-1] // g)
    hat = qg * scale.unsqueeze(-1)
    return hat.reshape_as(q).to(dtype)

ATTENTION_STRATEGIES = ("eager", "cudnn_prefill")
PREFILL_FUSED_MIN_S = 2
SHORT_BLOCK_POLICY = "eager_fallback"
FUSED_BACKEND_NAME = "CUDNN_ATTENTION"

ATTENTION_STRATEGY = "eager"
_ATTENTION_PHASE = "other"
_ATTENTION_STATS: dict | None = None
_MAX_FALLBACK_LOGGED = 32

# ---------------------------------------------------------------------------
# PagedAttention v1d routing/usage statistics.
#
# Direct page-aware CUDA is deliberately narrow:
#   - paged_v0 cache
#   - BF16/base K/V (fmt == "fp16")
#   - q_len == 1
#   - CUDA
#   - phase in decode/propose/repair/other
#
# In particular, `verify` is excluded even when kp == 1 so SpecDec target
# verification stays on the already-validated materialized/reference path.
PAGED_ATTENTION_MODES = ("off", "materialized_cuda", "direct_cuda")
PAGED_DIRECT_PHASES = ("decode", "propose", "repair", "other")
_PAGED_ATTENTION_STATS: dict | None = None


def _fresh_paged_attention_stats() -> dict:
    return {
        "direct_cuda_calls": 0,
        "materialized_cuda_calls": 0,
        "fallback_materialize_calls": 0,
        "direct_by_phase": {},
        "materialized_by_phase": {},
        "fallback_by_phase": {},
    }


def reset_paged_attention_stats() -> None:
    global _PAGED_ATTENTION_STATS
    _PAGED_ATTENTION_STATS = _fresh_paged_attention_stats()


def paged_attention_stats() -> dict:
    import copy
    global _PAGED_ATTENTION_STATS
    if _PAGED_ATTENTION_STATS is None:
        _PAGED_ATTENTION_STATS = _fresh_paged_attention_stats()
    return copy.deepcopy(_PAGED_ATTENTION_STATS)


def _record_paged_attention(kind: str, phase: str) -> None:
    global _PAGED_ATTENTION_STATS
    if _PAGED_ATTENTION_STATS is None:
        _PAGED_ATTENTION_STATS = _fresh_paged_attention_stats()
    st = _PAGED_ATTENTION_STATS
    if kind == "direct_cuda":
        st["direct_cuda_calls"] += 1
        bucket = st["direct_by_phase"]
    elif kind == "materialized_cuda":
        st["materialized_cuda_calls"] += 1
        bucket = st["materialized_by_phase"]
    elif kind == "fallback_materialize":
        st["fallback_materialize_calls"] += 1
        bucket = st["fallback_by_phase"]
    else:
        raise ValueError(f"unknown paged attention stat kind {kind!r}")
    bucket[phase] = bucket.get(phase, 0) + 1

DIAG_QK_FP32 = False
DIAG_PROB_FP32 = False


def diag_precision() -> dict:
    return dict(qk_fp32=DIAG_QK_FP32, prob_fp32=DIAG_PROB_FP32)


def set_diag_precision(qk32=None, prob32=None) -> dict:
    global DIAG_QK_FP32, DIAG_PROB_FP32
    if qk32 is not None:
        DIAG_QK_FP32 = bool(qk32)
    if prob32 is not None:
        DIAG_PROB_FP32 = bool(prob32)
    return diag_precision()

DIAG_PRESETS = {
    "eager": (False, False),
    "qk32": (True, False),
    "prob32": (False, True),
    "fp32": (True, True),
}


def set_diag_preset(name: str) -> dict:
    if name not in DIAG_PRESETS:
        raise ValueError(f"未知诊断精度预设 {name!r}（可选 {tuple(DIAG_PRESETS)}）")
    return set_diag_precision(*DIAG_PRESETS[name])


def _fresh_stats() -> dict:
    return {"strategy": ATTENTION_STRATEGY, "requested_backend": FUSED_BACKEND_NAME,
            "calls": {"eager": 0, "fused": 0},
            "by_phase": {}, "fused_by_phase": {}, "eager_by_phase": {},
            "s_max_by_phase": {}, "fused_s_max_by_phase": {},
            "fallbacks": [], "fallback_count": 0, "eager_reason_kinds": {},
            "phases_seen": [], "errors": []}


def set_attention_strategy(name: str) -> str:
    global ATTENTION_STRATEGY
    if name not in ATTENTION_STRATEGIES:
        raise ValueError(f"未知 attention 策略 {name!r}（可选 {ATTENTION_STRATEGIES}）")
    ATTENTION_STRATEGY = name
    if _ATTENTION_STATS is not None:
        _ATTENTION_STATS["strategy"] = name
    return ATTENTION_STRATEGY


def attention_strategy() -> str:
    return ATTENTION_STRATEGY


@contextlib.contextmanager
def attention_phase(name: str):
    global _ATTENTION_PHASE
    prev = _ATTENTION_PHASE
    _ATTENTION_PHASE = name
    st = _stats()
    if name not in st["phases_seen"]:
        st["phases_seen"].append(name)
    try:
        yield
    finally:
        _ATTENTION_PHASE = prev


def current_attention_phase() -> str:
    return _ATTENTION_PHASE


def _stats() -> dict:
    global _ATTENTION_STATS
    if _ATTENTION_STATS is None:
        _ATTENTION_STATS = _fresh_stats()
    return _ATTENTION_STATS


def reset_attention_stats() -> None:
    global _ATTENTION_STATS
    _ATTENTION_STATS = _fresh_stats()


def attention_stats() -> dict:
    import copy
    st = copy.deepcopy(_stats())
    st["strategy"] = ATTENTION_STRATEGY
    return st


def decide_attention_path(phase: str, s: int, strategy: str | None = None) -> tuple[str, str | None]:
    strat = ATTENTION_STRATEGY if strategy is None else strategy
    if strat != "cudnn_prefill":
        return "eager", f"strategy={strat}"
    if phase != "prefill":
        return "eager", f"phase={phase!r} != 'prefill'（仅 prefill 允许融合）"
    if s < PREFILL_FUSED_MIN_S:
        return "eager", f"s={s} < PREFILL_FUSED_MIN_S={PREFILL_FUSED_MIN_S}（cuDNN 不支持 seq_len=1）"
    return "fused", None


def _kv_keep_mask(s: int, T: int, device) -> torch.Tensor:
    drop = torch.triu(torch.ones(s, T, device=device, dtype=torch.bool), diagonal=(T - s) + 1)
    return ~drop


def _sdpa(q, k, v, **kw):
    return F.scaled_dot_product_attention(q, k, v, **kw)


def _fused_prefill_attention(q, k, v, head_dim: int):
    import torch.nn.attention as A
    s, T = q.shape[2], k.shape[2]
    keep = _kv_keep_mask(s, T, q.device)
    with A.sdpa_kernel([A.SDPBackend.CUDNN_ATTENTION]):
        out = _sdpa(q, k, v, attn_mask=keep, dropout_p=0.0, scale=head_dim ** -0.5,
                    enable_gqa=(q.shape[1] != k.shape[1]))
    return out


def _record_attention_call(path: str, phase: str, s: int, T: int, reason: str | None,
                           heads: int, kv_heads: int, head_dim: int) -> None:
    st = _stats()
    st["calls"][path] += 1
    st["by_phase"][phase] = st["by_phase"].get(phase, 0) + 1
    key = "fused_by_phase" if path == "fused" else "eager_by_phase"
    st[key][phase] = st[key].get(phase, 0) + 1
    smax = st["s_max_by_phase"]
    if s > smax.get(phase, 0):
        smax[phase] = s
    if path == "fused":
        fsmax = st["fused_s_max_by_phase"]
        if s > fsmax.get(phase, 0):
            fsmax[phase] = s
    if path == "eager" and ATTENTION_STRATEGY == "cudnn_prefill":
        st["fallback_count"] += 1
        kind = "other"
        if reason and "PREFILL_FUSED_MIN_S" in reason:
            kind = "short_block_seq_len_lt_min"
        elif reason and reason.startswith("phase="):
            kind = "phase_not_prefill"
        st["eager_reason_kinds"][kind] = st["eager_reason_kinds"].get(kind, 0) + 1
        if len(st["fallbacks"]) < _MAX_FALLBACK_LOGGED:
            st["fallbacks"].append(dict(phase=phase, s=s, T=T, reason=reason, kind=kind,
                                        heads=heads, kv_heads=kv_heads, head_dim=head_dim,
                                        policy=SHORT_BLOCK_POLICY))


def assert_strategy_used() -> dict:
    st = attention_stats()
    if ATTENTION_STRATEGY == "cudnn_prefill" and st["calls"]["fused"] == 0:
        raise RuntimeError(
            "策略=cudnn_prefill 但 fused 调用数为 0 —— 说明没有任何调用点标注了 "
            "attention_phase('prefill')，**这属于静默不生效**，必须显式排查。"
            f"（已见阶段：{st['phases_seen']}；回退数：{st['fallback_count']}）")
    return st


def chunked_prefill(model, ids, chunk: int = 1024, *, need_logits_last: bool = True):
    with attention_phase("prefill"):
        cache, lg, n = None, None, ids.shape[1]
        for st in range(0, n, chunk):
            lg, cache = model._run(model.embed_tokens(ids[:, st:st + chunk]), cache,
                                   need_logits=(need_logits_last and st + chunk >= n))
    return lg, cache


class SelfAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim:int, attn_bias: bool = False):
        super().__init__()
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias = attn_bias)
        self.k_proj = nn.Linear(hidden, kv_heads *head_dim, bias = attn_bias)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias = attn_bias)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias = False)
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.kv_format = "fp16" 
        self.cache_backend = "contiguous"
        self.cache_page_size = paged_cache.DEFAULT_PAGE_SIZE
        self.paged_attention_mode = "off"
        # H2O v0 reference-score capture. Disabled by default.
        self.h2o_capture_scores = False
        self.h2o_last_score = None
        self.fp8_g_k = GROUP_SIZE
        self.fp8_g_v = GROUP_SIZE
        self.fp8_k = True
        self.fp8_v = True

    def forward(self, x: torch.Tensor, cos, sin, cache = None) -> torch.Tensor:
        cache_was_none = cache is None
        self.h2o_last_score = None
        cache_is_paged = paged_cache.is_paged_layer(cache)
        if cache is not None and cache_is_paged != (self.cache_backend == "paged_v0"):
            raise ValueError("cache object/backend mismatch; start a fresh cache after switching backend")
        paged_mode = self.cache_backend == "paged_v0"
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        fmt = cache.get("fmt", self.kv_format) if cache is not None else self.kv_format
        if paged_mode and fmt == "fp8":
            raise ValueError("paged_v0 does not support full fp8 K/V")
        if fmt == 'v_fp8':
            k_on, v_on, gk, gv = False, True, GROUP_SIZE, GROUP_V
        elif fmt == 'fp8':
            k_on, v_on, gk, gv = self.fp8_k, self.fp8_v, self.fp8_g_k, self.fp8_g_v
        else:
            k_on = v_on = False
            gk = gv = GROUP_SIZE
        qk_cur = qv_cur = ks_cur = vs_cur = None
        if k_on:
            qk_cur, ks_cur = quantize_kv_fp8(k, gk)
            k_cur16 = dequantize_kv_fp8(qk_cur, ks_cur, x.dtype)
        else:
            k_cur16 = k
        if v_on:
            qv_cur, vs_cur = quantize_kv_fp8(v, gv)
            v_cur16 = dequantize_kv_fp8(qv_cur, vs_cur, x.dtype)
        else:
            v_cur16 = v
        paged_cuda_decode = False
        if paged_mode:
            if cache_was_none:
                cache = paged_cache.create_layer_cache(fmt, self.cache_page_size)
            if fmt == "fp16":
                paged_cache.append_current(cache, k, v)
            else:
                paged_cache.append_current(cache, k_cur16, qv_cur, vs_cur)

            # H2O v0 reference scorer. This is intentionally outside the
            # page-aware CUDA kernel: v0 validates policy/retention semantics
            # before optimizing score collection.
            if self.h2o_capture_scores:
                if fmt != "fp16":
                    raise ValueError("H2O v0 score capture supports fp16/base cache only")
                if s != 1:
                    raise ValueError("H2O v0 score capture supports q_len=1 decode only")
                if b != 1:
                    raise ValueError("H2O v0 score capture currently requires batch size 1")
                k_score = paged_cache.materialize_k(cache)
                k_score_rep = kv_repeat(k_score, self.heads // self.kv_heads)
                score_logits = torch.matmul(
                    q.float(), k_score_rep.float().transpose(-2, -1)
                ) / (self.head_dim ** 0.5)
                score_prob = torch.softmax(score_logits, dim=-1, dtype=torch.float32)
                # GQA/global-token adaptation: one scalar importance per
                # resident token, shared by the layer's paged KV storage.
                self.h2o_last_score = (
                    score_prob.mean(dim=(0, 1, 2)).detach().contiguous()
                )

            # PagedAttention v1d: direct CUDA remains single-token BF16 only,
            # and is phase-gated.  SpecDec target verification stays on the
            # materialized/reference path even when kp == 1.
            paged_cuda_decode = (
                not cache_was_none
                and fmt == "fp16"
                and s == 1
                and q.is_cuda
                and _ATTENTION_PHASE in PAGED_DIRECT_PHASES
                and self.paged_attention_mode in ("materialized_cuda", "direct_cuda")
            )

            if paged_cuda_decode and self.paged_attention_mode == "direct_cuda":
                # Direct page-aware kernel: do not materialize full-history K/V.
                _record_paged_attention("direct_cuda", _ATTENTION_PHASE)
                k_all = None
                v_all = None
            else:
                if paged_cuda_decode and self.paged_attention_mode == "materialized_cuda":
                    _record_paged_attention("materialized_cuda", _ATTENTION_PHASE)
                else:
                    _record_paged_attention("fallback_materialize", _ATTENTION_PHASE)
                materialized = paged_cache.materialize_layer(cache)
                k_all = materialized["k"]
                if fmt == "v_fp8":
                    v_all = dequantize_kv_fp8(
                        materialized["v"], materialized["v_scale"], x.dtype)
                else:
                    v_all = materialized["v"]
        elif cache is None:
            k_all, v_all = k_cur16, v_cur16
        elif fmt == 'fp16':
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
            cache["k"] = k
            cache["v"] = v
            k_all, v_all = k, v
        else:
            if "k_scale" in cache:
                k_all = torch.cat([dequantize_kv_fp8(cache["k"], cache["k_scale"], x.dtype), k_cur16], dim=2)
            else:
                k_all = torch.cat([cache["k"], k_cur16], dim=2)
                cache["k"] = k_all
            if "v_scale" in cache:
                v_all = torch.cat([dequantize_kv_fp8(cache["v"], cache["v_scale"], x.dtype), v_cur16], dim=2)
            else:
                v_all = torch.cat([cache["v"], v_cur16], dim=2)
                cache["v"] = v_all
        if paged_mode:
            T = paged_cache.logical_length(cache)
        else:
            T = k_all.shape[2] if cache is not None else s
        path, reason = decide_attention_path(_ATTENTION_PHASE, s)
        _record_attention_call(path, _ATTENTION_PHASE, s, T, reason,
                               self.heads, self.kv_heads, self.head_dim)
        if path == "fused":
            out = _fused_prefill_attention(q, k_all, v_all, self.head_dim)
        elif paged_cuda_decode:
            from specdec.kernels import ops as kernel_ops
            if self.paged_attention_mode == "direct_cuda":
                out = kernel_ops.paged_attention_decode(q, cache, backend="cuda", validate=False)
            elif self.paged_attention_mode == "materialized_cuda":
                out = kernel_ops.attention(q, k_all, v_all, backend="cuda")
            else:
                raise RuntimeError(
                    f"unexpected paged attention mode {self.paged_attention_mode!r}")
        else:
            k_rep = kv_repeat(k_all, self.heads // self.kv_heads)
            v_rep = kv_repeat(v_all, self.heads // self.kv_heads)
            if DIAG_QK_FP32:
                scores = torch.matmul(q.float(), k_rep.float().transpose(-2, -1)) / (self.head_dim ** 0.5)
            else:
                scores = torch.matmul(q, k_rep.transpose(-2, -1)) / (self.head_dim ** 0.5)
            p = T - s
            if cache_was_none:
                mask = torch.triu(torch.ones(s, T, device=x.device, dtype=torch.bool), diagonal=1)
                scores = scores.masked_fill(mask, float('-inf'))
            elif s == 1:
                pass
            else:
                mask = torch.triu(torch.ones(s, T, device=x.device, dtype=torch.bool), diagonal=p + 1)
                scores = scores.masked_fill(mask, float('-inf'))
            attn = torch.softmax(scores, dim=-1, dtype=torch.float32)
            if DIAG_PROB_FP32:
                out = torch.matmul(attn, v_rep.float())
            else:
                attn = attn.to(q.dtype)
                out = torch.matmul(attn, v_rep)
            out = out.to(q.dtype)
        out = out.transpose(1, 2).contiguous().view(b, s, self.heads * self.head_dim)
        if paged_mode:
            new_cache = cache
        elif cache_was_none:
            if fmt in ('fp8', 'v_fp8'):
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
        elif fmt in ('fp8', 'v_fp8'):
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
    
    
class DecoderLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int, intermediate: int, attn_bias: bool = False):
        super().__init__()
        self.self_attn = SelfAttention(hidden, heads, kv_heads, head_dim, attn_bias)
        self.mlp = MLP(hidden, intermediate)
        self.input_layernorm = RMSNORM(hidden)
        self.post_attention_layernorm = RMSNORM(hidden)

    def forward(self, x: torch.Tensor, cos, sin, cache = None) -> torch.Tensor:
        atten_out, new_attn_cache = self.self_attn(self.input_layernorm(x), cos, sin, cache)
        x = x + atten_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_attn_cache

class QwenLM(nn.Module):
    def __init__(self, vocab_size: int, hidden: int, heads: int, kv_heads: int, head_dim: int, intermediate: int, num_layers: int, rope_theta: float = 1_000_000.0, rms_eps: float = 1e-6, attn_bias: bool = False):
        super().__init__()
        self.embed_tokens = Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList([DecoderLayer(hidden, heads, kv_heads, head_dim, intermediate, attn_bias) for _ in range(num_layers)])
        self.norm = RMSNORM(hidden)
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.kv_format = "fp16"
        self.cache_backend = "contiguous"
        self.cache_page_size = paged_cache.DEFAULT_PAGE_SIZE
        self.paged_attention_mode = "off"

    def _run(self, x, cache_list=None, need_logits=True, start_pos_override: int | None = None):
        if start_pos_override is None:
            start_pos = spec_cache.cache_length(cache_list)
        else:
            if (not isinstance(start_pos_override, int)
                    or isinstance(start_pos_override, bool)
                    or start_pos_override < 0):
                raise ValueError("start_pos_override must be a non-negative int")
            start_pos = start_pos_override
        cos, sin = Build_Rope_cache(self.head_dim, seq_len=x.size(1), theta=self.rope_theta, device=x.device, dtype=x.dtype, start_pos=start_pos)
        new_cache_list = []
        for i, layer in enumerate(self.layers):
            cache = cache_list[i] if cache_list is not None else None
            x, new_cache = layer(x, cos, sin, cache)
            new_cache_list.append(new_cache)
        x = self.norm(x)
        if need_logits:
            logits = x @ self.embed_tokens.weight.T
            return logits, new_cache_list
        return None, new_cache_list

    def prefill_chunked(self, token_ids, chunk: int = 1024) -> tuple[torch.Tensor, list[dict]]:
        cache_list = None
        for st in range(0, token_ids.size(1), chunk):
            piece = token_ids[:, st:st + chunk]
            _, cache_list = self._run(self.embed_tokens(piece), cache_list, need_logits=False)
        return cache_list

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(token_ids)
        logits, _ = self._run(x)
        return logits

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int, eos_id: int = None) -> torch.Tensor:
        gen = prompt_ids
        with torch.no_grad():
            logits, cache_list = self._run(self.embed_tokens(prompt_ids), None)
            for _ in range(max_new_tokens):
                next_id = logits[:, -1].argmax(dim=-1)
                if eos_id is not None and (next_id == eos_id).all():
                    break
                gen = torch.cat([gen, next_id.unsqueeze(1)], dim=1)
                logits, cache_list = self._run(self.embed_tokens(next_id.unsqueeze(1)), cache_list)
        return gen

    def set_kv_format(self, fmt: str) -> None:
        assert fmt in ("fp16", "fp8", "v_fp8"), fmt
        if self.cache_backend == "paged_v0" and fmt == "fp8":
            raise ValueError("paged_v0 does not support full fp8 K/V")
        if fmt == "v_fp8":
            assert self.head_dim % GROUP_V == 0, f"head_dim {self.head_dim} % GROUP_V != 0"
        self.kv_format = fmt
        for layer in self.layers:
            a = layer.self_attn
            a.kv_format = fmt
            if fmt == "v_fp8":
                a.fp8_k, a.fp8_v, a.fp8_g_v = False, True, GROUP_V
            elif fmt == "fp8":
                a.fp8_k, a.fp8_v = True, True
                a.fp8_g_k, a.fp8_g_v = GROUP_SIZE, GROUP_SIZE

    def set_cache_backend(self, backend: str, page_size: int = 16) -> None:
        if backend not in ("contiguous", "paged_v0"):
            raise ValueError(f"unknown cache backend {backend!r}")
        if backend == "paged_v0":
            if self.kv_format == "fp8":
                raise ValueError("paged_v0 does not support full fp8 K/V")
            if (not isinstance(page_size, int) or isinstance(page_size, bool)
                    or page_size <= 0):
                raise ValueError("page_size must be a positive int")
        self.cache_backend = backend
        self.cache_page_size = page_size
        for layer in self.layers:
            layer.self_attn.cache_backend = backend
            layer.self_attn.cache_page_size = page_size

    def set_paged_attention_mode(self, mode: str) -> None:
        """Select the experimental paged decode attention path.

        off:
            Existing paged_v0 behavior. Full K/V is materialized and the
            normal engine attention path is used.

        materialized_cuda:
            For BF16 q_len=1 decode with an existing paged cache, materialize
            K/V and run the existing contiguous CUDA online-softmax attention.

        direct_cuda:
            For the same case, read paged K/V directly with the page-aware
            CUDA kernel and do not materialize full-history K/V.

        Direct/materialized CUDA comparison paths are used only for
        q_len=1 BF16 paged calls in decode/propose/repair/other phases.
        Prefill, verify, V-FP8, multi-token, and CPU calls keep the existing
        materialized engine path.
        """
        allowed = PAGED_ATTENTION_MODES
        if mode not in allowed:
            raise ValueError(
                f"unknown paged attention mode {mode!r}; choose from {allowed}")
        self.paged_attention_mode = mode
        for layer in self.layers:
            layer.self_attn.paged_attention_mode = mode

    def set_h2o_score_capture(self, enabled: bool) -> None:
        """Enable/disable H2O v0 reference attention-mass capture."""
        flag = bool(enabled)
        for layer in self.layers:
            layer.self_attn.h2o_capture_scores = flag
            if not flag:
                layer.self_attn.h2o_last_score = None

    def pop_h2o_scores(self) -> list[torch.Tensor]:
        """Return one captured global token-mass vector per decoder layer."""
        out = []
        for index, layer in enumerate(self.layers):
            a = layer.self_attn
            value = a.h2o_last_score
            if value is None:
                raise RuntimeError(
                    f"missing H2O score capture for decoder layer {index}")
            out.append(value)
            a.h2o_last_score = None
        return out

    def set_fp8_groups(self, k_group: int | None = None, v_group: int | None = None) -> None:
        if k_group is not None:
            assert self.head_dim % k_group == 0
        if v_group is not None:
            assert self.head_dim % v_group == 0
        for layer in self.layers:
            a = layer.self_attn
            if k_group is not None:
                a.fp8_g_k = k_group
            if v_group is not None:
                a.fp8_g_v = v_group

    def set_fp8_matrices(self, k_on: bool | None = None, v_on: bool | None = None,
                         k_group: int | None = None, v_group: int | None = None) -> None:
        for layer in self.layers:
            a = layer.self_attn
            if k_on is not None:
                a.fp8_k = bool(k_on)
            if v_on is not None:
                a.fp8_v = bool(v_on)
            if k_group is not None:
                assert self.head_dim % k_group == 0
                a.fp8_g_k = k_group
            if v_group is not None:
                assert self.head_dim % v_group == 0
                a.fp8_g_v = v_group

if __name__ == "__main__":
    ids = torch.randint(0, 151936, (1, 10))
    full = model(ids)
    x9 = model.embed_tokens(ids[:, :9])
    _, caches = model._run(x9, None)
    x10 = model.embed_tokens(ids[:, 9:])
    step_logits, _ = model._run(x10, caches)
    print((full[:, -1] - step_logits[:, -1]).abs().max().item())
