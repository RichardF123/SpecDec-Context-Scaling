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
GROUP_SIZE = 32      # full-FP8 ablation 默认组粒度
GROUP_V = 128        # 正式 Asymmetric V-only: V 的 group size(=1 group/head)

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
        # FlashAttention is opt-in.  The default path must remain importable on
        # CPU and native Windows installations that do not provide Triton.
        self.attention_backend = "eager"
        # 策略旋钮: 'fp8'(full K/V ablation) 时生效; 'v_fp8'(正式) 时 V 固定 GROUP_V
        self.fp8_g_k = GROUP_SIZE
        self.fp8_g_v = GROUP_SIZE
        self.fp8_k = True
        self.fp8_v = True

    def forward(self, x: torch.Tensor, cos, sin, cache = None) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        fmt = cache.get("fmt", self.kv_format) if cache is not None else self.kv_format
        # 模式语义: 'fp16'=base(bf16 KV 计算); 'v_fp8'=正式 V-only(K bf16 + V FP8 GROUP_V);
        #           'fp8'=full FP8 K/V(negative ablation, 粒度经 fp8_g_*/fp8_k/v 旋钮)
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
        if cache is None:
            k_all, v_all = k_cur16, v_cur16
        elif fmt == 'fp16':
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
            cache["k"] = k
            cache["v"] = v
            k_all, v_all = k, v
        else:
            # fp8 混合缓存读取: 有 *_scale 键 = fp8 payload(反量化), 否则 = bf16 全长历史
            if "k_scale" in cache:
                k_all = torch.cat([dequantize_kv_fp8(cache["k"], cache["k_scale"],
                                                     x.dtype), k_cur16], dim=2)
            else:
                k_all = torch.cat([cache["k"], k_cur16], dim=2)
                cache["k"] = k_all                      # bf16 全长写回
            if "v_scale" in cache:
                v_all = torch.cat([dequantize_kv_fp8(cache["v"], cache["v_scale"],
                                                     x.dtype), v_cur16], dim=2)
            else:
                v_all = torch.cat([cache["v"], v_cur16], dim=2)
                cache["v"] = v_all
        hybrid_eager = (
            self.attention_backend == "triton_bf16_g4_hybrid"
            and cache is not None
            and 2 <= s <= 4
        )
        if self.attention_backend == "eager" or hybrid_eager:
            k_rep = kv_repeat(k_all, self.heads // self.kv_heads)
            v_rep = kv_repeat(v_all, self.heads // self.kv_heads)
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
        elif self.attention_backend in (
            "triton_bf16_two_pass", "triton_bf16_g4_hybrid",
        ):
            supported = (
                fmt == "fp16"
                and q.is_cuda
                and q.dtype == torch.bfloat16
                and self.head_dim == 128
                and k_all.shape[1] == self.kv_heads
                and v_all.shape[1] == self.kv_heads
            )
            if not supported:
                raise RuntimeError(
                    f"{self.attention_backend} requires CUDA BF16, head_dim=128, "
                    "fmt='fp16', and checkpoint-native KV heads"
                )
            # Lazy import preserves the existing eager/CPU/Windows path.
            from .flash_attention_two_pass_triton import (
                flash_attention_bf16_two_pass,
            )
            out = flash_attention_bf16_two_pass(
                q.contiguous(),
                k_all.contiguous(),
                v_all.contiguous(),
                causal=True,
            )
        else:
            raise RuntimeError(f"unknown attention backend: {self.attention_backend!r}")
        out = out.transpose(1, 2).contiguous().view(b, s, self.heads * self.head_dim)
        if cache is None:
            if fmt in ('fp8', 'v_fp8'):
                nc = {"fmt": fmt}
                if k_on:
                    nc["k"], nc["k_scale"] = qk_cur, ks_cur
                else:
                    nc["k"] = k_all                    # bf16(K 不量化)
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
            # else: K 已在读路径按 bf16 全长写回
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

    def _run(self,x, cache_list = None, need_logits = True):  
        start_pos = 0
        if cache_list is not None and cache_list[0] is not None:
            start_pos = cache_list[0]["k"].shape[2]
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
        'fp8'=full FP8 K/V(negative ablation; K/V 粒度可经 set_fp8_groups/matrices 调)。"""
        assert fmt in ("fp16", "fp8", "v_fp8"), fmt
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
            # 'fp16': 仅记录格式; fp8 旋钮不影响 base 路径

    def set_attention_backend(self, backend: str = "eager") -> None:
        """Select eager, pure Triton, or the G4-compatible hybrid backend.

        The hybrid uses Triton for prompt prefill and cached q_len=1, and the
        existing eager expression for cached q_len=2..4 verification. Triton
        deliberately does not accept compressed KV. Call this only between
        fresh runs; do not change attention semantics while consuming a cache.
        """
        if backend not in (
            "eager",
            "triton_bf16_two_pass",
            "triton_bf16_g4_hybrid",
        ):
            raise ValueError(
                "attention backend must be 'eager', "
                "'triton_bf16_two_pass', or "
                "'triton_bf16_g4_hybrid'"
            )
        for layer in self.layers:
            layer.self_attn.attention_backend = backend

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
