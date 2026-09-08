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
        variance = torch.mean(x * x, dim=-1, keepdim=True)
        normed = 1 / torch.sqrt(variance + self.eps)
        return normed * x * self.weight

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
    # Calculate the inverse frequencies
    exponents = torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim
    inv_freq = 1.0 / (theta ** exponents)
    # Create the position indices
    postitions = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=dtype)
    # Compute the outer product to get the angles
    freqs = torch.outer(postitions, inv_freq)  # (seq_len, head_dim/2)
    # Compute the cosine and sine values
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    return cos, sin

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

def quantize_kv_fp8(x: torch.Tensor):
    m = x.abs().amax(dim=-1, keepdim=True)
    s = m.to(torch.float32) / E4M3_MAX
    s = torch.where(s == 0, torch.ones_like(s), s)
    q = (x.to(torch.float32) / s).to(torch.float8_e4m3fn)
    return q, s.squeeze(-1)

def dequantize_kv_fp8(q: torch.Tensor, scale: torch.Tensor):
    return (q.to(torch.float32) * scale.unsqueeze(-1)).to(torch.float16)

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

    def forward(self, x: torch.Tensor, cos, sin, cache = None) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        fmt = cache.get("fmt", self.kv_format) if cache is not None else self.kv_format
        if cache is None:
            k_all, v_all = k, v
        elif fmt == 'fp16':
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
            cache["k"] = k
            cache["v"] = v
            k_all, v_all = k, v
        else:
            past_k = dequantize_kv_fp8(cache["k"], cache["k_scale"])
            past_v = dequantize_kv_fp8(cache["v"], cache["v_scale"])
            k_all = torch.cat([past_k, k], dim=2)
            v_all = torch.cat([past_v, v], dim=2)        
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
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_rep)
        out = out.transpose(1, 2).contiguous().view(b, s, self.heads * self.head_dim)
        if cache is None:
            if fmt == 'fp8':
                qk, ks = quantize_kv_fp8(k)
                qv, vs = quantize_kv_fp8(v)
                new_cache = {"k": qk, "v": qv, "k_scale": ks, "v_scale": vs, "fmt": fmt}
            else:                
                new_cache = {"k": k_all, "v": v_all}
        elif fmt == 'fp8':
            qk, ks = quantize_kv_fp8(k)
            qv, vs = quantize_kv_fp8(v)
            cache['k'] = torch.cat([cache['k'], qk], dim=2)
            cache['v'] = torch.cat([cache['v'], qv], dim=2)
            cache['k_scale'] = torch.cat([cache['k_scale'], ks], dim=-1)
            cache['v_scale'] = torch.cat([cache['v_scale'], vs], dim=-1)
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
        assert fmt in ("fp16", "fp8"), fmt
        self.kv_format = fmt
        for layer in self.layers:
            layer.self_attn.kv_format = fmt
    
if __name__ == "__main__":
    ids = torch.randint(0, 151936, (1, 10))
    full = model(ids)                                  # 全量 10 词
    x9 = model.embed_tokens(ids[:, :9])
    _, caches = model._run(x9, None)                   # prefill 前 9
    x10 = model.embed_tokens(ids[:, 9:])
    step_logits, _ = model._run(x10, caches)           # 单独喂第 10 个
    print((full[:, -1] - step_logits[:, -1]).abs().max().item())   # 应 0.0