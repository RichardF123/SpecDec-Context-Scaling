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

def Build_Rope_cache(head_dim: int, seq_len: int, theta:float, device: torch.device,dtype: torch.dtype):
    # Calculate the inverse frequencies
    exponents = torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim
    inv_freq = 1.0 / (theta ** exponents)
    # Create the position indices
    postitions = torch.arange(seq_len, device=device, dtype=dtype)
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

    def forward(self, x: torch.Tensor, cos, sin) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = kv_repeat(k, self.heads // self.kv_heads)
        v = kv_repeat(v, self.heads // self.kv_heads)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.ones(s, s, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, s, self.heads * self.head_dim)
        return self.o_proj(out)
    
class DecoderLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int, intermediate: int, attn_bias: bool = False):
        super().__init__()
        self.self_attn = SelfAttention(hidden, heads, kv_heads, head_dim, attn_bias)
        self.mlp = MLP(hidden, intermediate)
        self.input_layernorm = RMSNORM(hidden)
        self.post_attention_layernorm = RMSNORM(hidden)

    def forward(self, x: torch.Tensor, cos, sin) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class QwenLM(nn.Module):
    def __init__(self, vocab_size: int, hidden: int, heads: int, kv_heads: int, head_dim: int, intermediate: int, num_layers: int, rope_theta: float = 1_000_000.0, rms_eps: float = 1e-6, attn_bias: bool = False):
        super().__init__()
        self.embed_tokens = Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList([DecoderLayer(hidden, heads, kv_heads, head_dim, intermediate, attn_bias) for _ in range(num_layers)])
        self.norm = RMSNORM(hidden)
        self.head_dim = head_dim
        self.rope_theta = rope_theta 

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(token_ids)
        cos, sin = Build_Rope_cache(self.head_dim, seq_len=x.size(1), theta=self.rope_theta, device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return x @ self.embed_tokens.weight.T
if __name__ == "__main__":
    torch.manual_seed(0)
model = QwenLM(vocab_size=5000, hidden=64, heads=8, kv_heads=2, head_dim=8,
               intermediate=256, num_layers=2, rope_theta=1_000_000.0)
ids = torch.randint(0, 5000, (2, 16))
logits = model(ids)
print("logits shape:", logits.shape)          # 应 (2, 16, 5000)
print("no NaN:", bool(torch.isfinite(logits).all()))