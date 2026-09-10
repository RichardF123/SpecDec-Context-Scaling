import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
    # Calculate the inverse frequencies
    exponents = torch.arange(0, head_dim, 2, device=device, dtype=calc_dtype) / head_dim
    inv_freq = 1.0 / (theta ** exponents)
    # Create the position indices
    postitions = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=calc_dtype)
    # Compute the outer product to get the angles
    freqs = torch.outer(postitions, inv_freq)  # (seq_len, head_dim/2)
    # Compute the cosine and sine values
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
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
INT8_MAX = 127.0
INT4_MAX = 7.0
GROUP_SIZE = 32      # full-FP8 ablation 默认组粒度
GROUP_V = 128        # 正式 Asymmetric V-only: V 的 group size(=1 group/head)
GROUP_K_INT8 = 128   # K-INT8 候选: per-token x per-head scale
GROUP_V_INT4 = 32    # V-INT4 候选: per-token x per-head-dim group
K_CHANNEL_BLOCK = 32 # K-INT8 候选: per-channel scale over completed token blocks

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

def quantize_kv_int8(x: torch.Tensor, group_size: int = GROUP_K_INT8):
    """Symmetric RTN INT8 for KV tensors, grouped along head_dim.

    With group_size=head_dim this is per-token x per-KV-head scaling.  INT8 keeps
    the same one-byte payload as FP8 but provides substantially finer resolution
    around zero, which is the property under test for the K-sensitive path.
    """
    d = x.shape[-1]
    assert d % group_size == 0, f"head_dim {d} % group_size {group_size} != 0"
    g = d // group_size
    xg = x.to(torch.float32).view(*x.shape[:-1], g, group_size)
    m = xg.abs().amax(dim=-1, keepdim=True)
    scale = m / INT8_MAX
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(xg / scale).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return q.reshape_as(x), scale.squeeze(-1)

def dequantize_kv_int8(q: torch.Tensor, scale: torch.Tensor,
                       dtype: torch.dtype = torch.float16):
    g = scale.shape[-1]
    qg = q.to(torch.float32).view(*q.shape[:-1], g, q.shape[-1] // g)
    return (qg * scale.unsqueeze(-1)).reshape_as(q).to(dtype)

def quantize_kv_int4_packed(x: torch.Tensor, group_size: int = GROUP_V_INT4):
    """Symmetric signed INT4 with two head-dimension values packed per byte."""
    d = x.shape[-1]
    assert d % group_size == 0, f"head_dim {d} % group_size {group_size} != 0"
    assert d % 2 == 0, f"head_dim {d} must be even for INT4 packing"
    groups = d // group_size
    xg = x.to(torch.float32).view(*x.shape[:-1], groups, group_size)
    maximum = xg.abs().amax(dim=-1, keepdim=True)
    scale = maximum / INT4_MAX
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    signed = torch.round(xg / scale).clamp(-INT4_MAX, INT4_MAX).to(torch.int16)
    signed = signed.reshape_as(x)
    biased = (signed + 8).to(torch.uint8)
    packed = biased[..., 0::2] | (biased[..., 1::2] << 4)
    return packed, scale.squeeze(-1)

def dequantize_kv_int4_packed(packed: torch.Tensor, scale: torch.Tensor,
                              dtype: torch.dtype = torch.float16):
    d = packed.shape[-1] * 2
    low = (packed & 0x0F).to(torch.int16) - 8
    high = (packed >> 4).to(torch.int16) - 8
    signed = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], d)
    groups = scale.shape[-1]
    qg = signed.to(torch.float32).view(*signed.shape[:-1], groups, d // groups)
    return (qg * scale.unsqueeze(-1)).reshape_as(signed).to(dtype)

def quantize_kv_int8_channel_blocks(x: torch.Tensor,
                                    block_size: int = K_CHANNEL_BLOCK):
    """Quantize complete token blocks with one INT8 scale per K channel.

    ``x`` is (batch, kv_heads, tokens, head_dim), and tokens must be a whole
    number of blocks.  Scales are (batch, kv_heads, blocks, head_dim).  The
    caller retains an incomplete tail in BF16, matching the residual-cache idea
    used by practical asymmetric KV quantizers.
    """
    b, h, tokens, d = x.shape
    assert tokens % block_size == 0, (tokens, block_size)
    blocks = tokens // block_size
    xb = x.to(torch.float32).view(b, h, blocks, block_size, d)
    maximum = xb.abs().amax(dim=3)
    scale = maximum / INT8_MAX
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(xb / scale.unsqueeze(3)).clamp(
        -INT8_MAX, INT8_MAX,
    ).to(torch.int8)
    return q.reshape_as(x), scale

def dequantize_kv_int8_channel_blocks(q: torch.Tensor, scale: torch.Tensor,
                                      dtype: torch.dtype = torch.float16,
                                      block_size: int = K_CHANNEL_BLOCK):
    b, h, tokens, d = q.shape
    assert tokens % block_size == 0, (tokens, block_size)
    blocks = tokens // block_size
    qb = q.to(torch.float32).view(b, h, blocks, block_size, d)
    return (qb * scale.unsqueeze(3)).reshape_as(q).to(dtype)

def append_kv_int8_channel_cache(
    q_cache: torch.Tensor | None,
    scale_cache: torch.Tensor | None,
    residual: torch.Tensor | None,
    current: torch.Tensor,
    block_size: int = K_CHANNEL_BLOCK,
    residual_length: int = 0,
):
    """Append K values while retaining a recent BF16 residual window.

    Older values are quantized only in complete ``block_size`` groups.  Once the
    cache is long enough, the residual therefore stays in
    ``[residual_length, residual_length + block_size)``.  A nonzero recent
    residual follows the mechanism used by practical KV quantizers: recent K
    values remain exact while their memory cost is amortized over long context.
    """
    assert residual_length >= 0
    pending = current if residual is None else torch.cat([residual, current], dim=2)
    quantizable = max(0, pending.shape[2] - residual_length)
    full_tokens = (quantizable // block_size) * block_size
    full = pending[:, :, :full_tokens]
    new_residual = pending[:, :, full_tokens:]
    if full_tokens:
        q_new, scale_new = quantize_kv_int8_channel_blocks(full, block_size)
        q_cache = q_new if q_cache is None else torch.cat([q_cache, q_new], dim=2)
        scale_cache = (
            scale_new if scale_cache is None
            else torch.cat([scale_cache, scale_new], dim=2)
        )
    if q_cache is None:
        b, h, _, d = current.shape
        q_cache = torch.empty((b, h, 0, d), device=current.device, dtype=torch.int8)
        scale_cache = torch.empty(
            (b, h, 0, d), device=current.device, dtype=torch.float32,
        )
    return q_cache, scale_cache, new_residual

def read_kv_int8_channel_cache(q_cache: torch.Tensor, scale_cache: torch.Tensor,
                               residual: torch.Tensor, dtype: torch.dtype,
                               block_size: int = K_CHANNEL_BLOCK):
    completed = dequantize_kv_int8_channel_blocks(
        q_cache, scale_cache, dtype, block_size,
    )
    return torch.cat([completed, residual], dim=2)

def reduce_kv_heads(x: torch.Tensor, target_heads: int) -> torch.Tensor:
    """Mean-pool groups of KV heads for a diagnostic GQA->MQA ablation.

    This changes the pretrained model function and is therefore never enabled by
    default or treated as a drop-in compression method.  It exists only to answer
    whether reducing Qwen2.5-1.5B's two native KV heads is viable without uptraining.
    """
    source_heads = x.shape[1]
    assert 0 < target_heads <= source_heads, (source_heads, target_heads)
    assert source_heads % target_heads == 0, (source_heads, target_heads)
    if target_heads == source_heads:
        return x
    b, _, s, d = x.shape
    return x.reshape(b, target_heads, source_heads // target_heads, s, d).mean(dim=2)

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
        # 策略旋钮: 'fp8'(full K/V ablation) 时生效; 'v_fp8'(正式) 时 V 固定 GROUP_V
        self.fp8_g_k = GROUP_SIZE
        self.fp8_g_v = GROUP_SIZE
        self.fp8_k = True
        self.fp8_v = True
        # Native by default.  Values below native kv_heads are diagnostic only:
        # directly collapsing pretrained GQA heads changes model semantics.
        self.cache_k_heads = kv_heads
        self.cache_v_heads = kv_heads
        self.k_residual_length = 0

    def forward(self, x: torch.Tensor, cos, sin, cache = None) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = reduce_kv_heads(k, self.cache_k_heads)
        v = reduce_kv_heads(v, self.cache_v_heads)
        fmt = cache.get("fmt", self.kv_format) if cache is not None else self.kv_format
        if cache is not None:
            assert cache["k"].shape[1] == k.shape[1], \
                "K head policy changed while a cache was live"
            assert cache["v"].shape[1] == v.shape[1], \
                "V head policy changed while a cache was live"
        # 模式语义: 'fp16'=base(bf16 KV); 'v_fp8'=正式 V-only;
        # 其余格式仅为候选/消融，不改变默认正式路径。
        if fmt == 'v_fp8':
            k_quant, v_quant, gk, gv = 'none', 'fp8', GROUP_SIZE, GROUP_V
        elif fmt == 'k_int8_v_fp8':
            k_quant, v_quant, gk, gv = 'int8', 'fp8', GROUP_K_INT8, GROUP_V
        elif fmt == 'v_int4':
            k_quant, v_quant, gk, gv = 'none', 'int4', GROUP_SIZE, GROUP_V_INT4
        elif fmt == 'k_pc_int8_v_fp8':
            k_quant, v_quant, gk, gv = 'pc_int8', 'fp8', K_CHANNEL_BLOCK, GROUP_V
        elif fmt == 'fp8':
            k_quant = 'fp8' if self.fp8_k else 'none'
            v_quant = 'fp8' if self.fp8_v else 'none'
            gk, gv = self.fp8_g_k, self.fp8_g_v
        else:
            k_quant = v_quant = 'none'
            gk = gv = GROUP_SIZE
        qk_cur = qv_cur = ks_cur = vs_cur = None
        if k_quant == 'fp8':
            qk_cur, ks_cur = quantize_kv_fp8(k, gk)
            k_cur16 = dequantize_kv_fp8(qk_cur, ks_cur, x.dtype)
        elif k_quant == 'int8':
            qk_cur, ks_cur = quantize_kv_int8(k, gk)
            k_cur16 = dequantize_kv_int8(qk_cur, ks_cur, x.dtype)
        else:
            k_cur16 = k
        if v_quant == 'fp8':
            qv_cur, vs_cur = quantize_kv_fp8(v, gv)
            v_cur16 = dequantize_kv_fp8(qv_cur, vs_cur, x.dtype)
        elif v_quant == 'int4':
            qv_cur, vs_cur = quantize_kv_int4_packed(v, gv)
            v_cur16 = dequantize_kv_int4_packed(qv_cur, vs_cur, x.dtype)
        else:
            v_cur16 = v

        # K cache/read.  The per-channel candidate quantizes completed token
        # blocks and retains an incomplete tail in BF16; all other formats keep
        # the original append behavior.
        if k_quant == 'pc_int8':
            q_hist = None if cache is None else cache['k']
            s_hist = None if cache is None else cache['k_scale']
            residual = None if cache is None else cache['k_residual']
            residual_length = (
                self.k_residual_length if cache is None
                else int(cache.get('k_residual_length', 0))
            )
            qk_store, ks_store, k_residual = append_kv_int8_channel_cache(
                q_hist, s_hist, residual, k, K_CHANNEL_BLOCK, residual_length,
            )
            k_all = read_kv_int8_channel_cache(
                qk_store, ks_store, k_residual, x.dtype, K_CHANNEL_BLOCK,
            )
        elif cache is None:
            k_all = k_cur16
        elif fmt == 'fp16':
            k = torch.cat([cache["k"], k], dim=2)
            k_all = k
        else:
            if "k_scale" in cache:
                if cache["k"].dtype == torch.int8:
                    k_hist = dequantize_kv_int8(cache["k"], cache["k_scale"], x.dtype)
                else:
                    k_hist = dequantize_kv_fp8(cache["k"], cache["k_scale"], x.dtype)
                k_all = torch.cat([k_hist, k_cur16], dim=2)
            else:
                k_all = torch.cat([cache["k"], k_cur16], dim=2)

        # V cache/read.  Packed INT4 is identified by uint8 payload; existing
        # FP8 formats retain their original dequantization path.
        if cache is None:
            v_all = v_cur16
        elif fmt == 'fp16':
            v_all = torch.cat([cache["v"], v], dim=2)
        elif "v_scale" in cache:
            if cache["v"].dtype == torch.uint8:
                v_hist = dequantize_kv_int4_packed(
                    cache["v"], cache["v_scale"], x.dtype,
                )
            else:
                v_hist = dequantize_kv_fp8(
                    cache["v"], cache["v_scale"], x.dtype,
                )
            v_all = torch.cat([v_hist, v_cur16], dim=2)
        else:
            v_all = torch.cat([cache["v"], v_cur16], dim=2)
        assert self.heads % k_all.shape[1] == 0, (self.heads, k_all.shape[1])
        assert self.heads % v_all.shape[1] == 0, (self.heads, v_all.shape[1])
        k_rep = kv_repeat(k_all, self.heads // k_all.shape[1])
        v_rep = kv_repeat(v_all, self.heads // v_all.shape[1])
        scores = torch.matmul(q, k_rep.transpose(-2, -1)) / (self.head_dim ** 0.5)
        T = k_all.shape[2] if cache is not None else s
        p = T - s
        if cache is None:
            mask = torch.triu(torch.ones(s, s, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float('-inf'))
        elif s == 1:
            pass
        else:
            mask = torch.triu(torch.ones(s, T, device=x.device, dtype=torch.bool), diagonal=p + 1)
            scores = scores.masked_fill(mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v_rep)
        out = out.transpose(1, 2).contiguous().view(b, s, self.heads * self.head_dim)
        quantized_formats = (
            'fp8', 'v_fp8', 'k_int8_v_fp8', 'v_int4', 'k_pc_int8_v_fp8',
        )
        if fmt in quantized_formats:
            new_cache = {"fmt": fmt}
            if k_quant == 'pc_int8':
                new_cache["k"] = qk_store
                new_cache["k_scale"] = ks_store
                new_cache["k_residual"] = k_residual
                new_cache["k_residual_length"] = residual_length
            elif k_quant != 'none':
                new_cache["k"] = (
                    qk_cur if cache is None
                    else torch.cat([cache["k"], qk_cur], dim=2)
                )
                new_cache["k_scale"] = (
                    ks_cur if cache is None
                    else torch.cat([cache["k_scale"], ks_cur], dim=2)
                )
            else:
                new_cache["k"] = k_all
            if v_quant != 'none':
                new_cache["v"] = (
                    qv_cur if cache is None
                    else torch.cat([cache["v"], qv_cur], dim=2)
                )
                new_cache["v_scale"] = (
                    vs_cur if cache is None
                    else torch.cat([cache["v_scale"], vs_cur], dim=2)
                )
            else:
                new_cache["v"] = v_all
        else:
            new_cache = {"k": k_all, "v": v_all}
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

    def _run(self,x, cache_list = None, need_logits = True):  
        start_pos = 0
        if cache_list is not None and cache_list[0] is not None:
            first_cache = cache_list[0]
            start_pos = first_cache["k"].shape[2]
            if "k_residual" in first_cache:
                start_pos += first_cache["k_residual"].shape[2]
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
        """缓存格式: 'fp16'=base(bf16 KV); 'v_fp8'=正式 Asymmetric V-only
        (K bf16 不量化 + V FP8 E4M3 group=GROUP_V=128, 每 head 1 group);
        'k_int8_v_fp8'=候选(K INT8 RTN + V FP8, 均 per-token x per-head);
        'v_int4'=候选(K bf16 + packed V INT4 group32);
        'k_pc_int8_v_fp8'=候选(K token-block/per-channel INT8 + V FP8);
        'fp8'=full FP8 K/V(negative ablation; K/V 粒度可经 set_fp8_groups/matrices 调)。"""
        assert fmt in (
            "fp16", "fp8", "v_fp8", "k_int8_v_fp8", "v_int4",
            "k_pc_int8_v_fp8",
        ), fmt
        if fmt in ("v_fp8", "k_int8_v_fp8", "k_pc_int8_v_fp8"):
            assert self.head_dim % GROUP_V == 0, f"head_dim {self.head_dim} % GROUP_V != 0"
        if fmt == "k_int8_v_fp8":
            assert self.head_dim % GROUP_K_INT8 == 0, \
                f"head_dim {self.head_dim} % GROUP_K_INT8 != 0"
        if fmt == "v_int4":
            assert self.head_dim % GROUP_V_INT4 == 0, \
                f"head_dim {self.head_dim} % GROUP_V_INT4 != 0"
        self.kv_format = fmt
        for layer in self.layers:
            a = layer.self_attn
            a.kv_format = fmt
            if fmt in ("v_fp8", "k_int8_v_fp8", "k_pc_int8_v_fp8"):
                a.fp8_k, a.fp8_v, a.fp8_g_v = False, True, GROUP_V
            elif fmt == "fp8":
                a.fp8_k, a.fp8_v = True, True
                a.fp8_g_k, a.fp8_g_v = GROUP_SIZE, GROUP_SIZE
            # 'fp16': 仅记录格式; fp8 旋钮不影响 base 路径

    def set_kv_head_policy(self, k_heads: int | None = None,
                           v_heads: int | None = None) -> None:
        """Set stored K/V head counts for a controlled diagnostic ablation.

        None restores the checkpoint-native count.  Reducing heads applies a
        deterministic mean merge and changes the pretrained model; such results
        are screening evidence only unless an explicit training procedure exists.
        Call only before starting a fresh cache.
        """
        native = self.layers[0].self_attn.kv_heads
        kh = native if k_heads is None else int(k_heads)
        vh = native if v_heads is None else int(v_heads)
        assert native % kh == 0 and self.layers[0].self_attn.heads % kh == 0
        assert native % vh == 0 and self.layers[0].self_attn.heads % vh == 0
        for layer in self.layers:
            layer.self_attn.cache_k_heads = kh
            layer.self_attn.cache_v_heads = vh

    def set_k_residual_length(self, tokens: int = 0) -> None:
        """Keep this many most-recent K tokens in BF16 for PC-INT8 formats.

        The value is captured in each new cache, so changing it never mutates a
        live cache's representation.  It is ignored by all non-PC-INT8 formats.
        """
        tokens = int(tokens)
        assert tokens >= 0
        for layer in self.layers:
            layer.self_attn.k_residual_length = tokens

    def set_fp8_groups(self, k_group: int | None = None, v_group: int | None = None) -> None:
        """诊断开关: K/V 各自 FP8 scale 分组粒度(须整除 head_dim)。默认 None=不动。
        group=1 => 逐元素(对 bf16 无损); 128 => per-head; 32 => 正式值。仅 fp8 模式生效。"""
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
        """非对称策略开关: 分别决定 K/V 是否量化进 FP8。None=不动。
        正式默认 k_on=False(bf16 K), v_on=True(FP8 V, group32)。"""
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
    full = model(ids)                                  # 全量 10 词
    x9 = model.embed_tokens(ids[:, :9])
    _, caches = model._run(x9, None)                   # prefill 前 9
    x10 = model.embed_tokens(ids[:, 9:])
    step_logits, _ = model._run(x10, caches)           # 单独喂第 10 个
    print((full[:, -1] - step_logits[:, -1]).abs().max().item())   # 应 0.0
