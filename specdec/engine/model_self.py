import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import contextlib
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

# ---------------------------------------------------------------- attention 后端策略（实验性分支）
# 目的：**单独**验证"融合 attention 能否降低 prefill 的临时显存压力"。
# 纪律（对应本次任务的硬要求）：
#   ① 默认 `eager`，行为与历史完全一致；不把新后端宣布为正式冻结方案；
#   ② **只允许 prefill 阶段**走融合路径 —— AR 单步 / draft 起草 / target 块验证 / 替换修复
#      一律保持原路径；
#   ③ 阶段由调用方**显式标注**。**不得**用 `q_len > 1` 判断 prefill：
#      `spec_dec._verify` 的候选验证也是多 token（s = kp ∈ [1..k]）；
#   ④ 不支持或短块 → **预先定义**的回退策略，并把**原因**记录进统计（不静默回退）；
#   ⑤ **不吞异常**：融合路径的 OOM / "无可用 kernel" / 实现错误一律向上抛，
#      绝不用笼统 `except Exception` 掩盖。
ATTENTION_STRATEGIES = ("eager", "cudnn_prefill")
PREFILL_FUSED_MIN_S = 2          # cuDNN SDPA 不支持 seq_len == 1（见台账 §4.17.x）
SHORT_BLOCK_POLICY = "eager_fallback"   # 唯一预定义策略：短块回退 eager 并记录原因
FUSED_BACKEND_NAME = "CUDNN_ATTENTION"

ATTENTION_STRATEGY = "eager"
_ATTENTION_PHASE = "other"       # 调用方显式设置；默认**不是** prefill
_ATTENTION_STATS: dict | None = None
_MAX_FALLBACK_LOGGED = 32

# ---------------------------------------------------------------------------
# **仅用于诊断**的 eager 精度干预开关（两个可独立移除的舍入点，默认全部关闭）
#   R1 = QK 分数被舍入成 bf16（matmul 输入 bf16 ⇒ 输出 bf16）
#   R2 = softmax 概率被 `.to(dtype)` 转回 bf16
# 第三个舍入点 R3（AV 输出被舍入成 bf16）**在整模型层面无法单独移除**：
# 下游的 `o_proj` 是 bf16 权重，fp32 的 attention 输出喂不进去
# （实测：`RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != BFloat16`）。
# 因此整模型层面 `R3 缺失` 与 `R3 存在但立刻被 o_proj 的输入转换再舍入一次` 不可区分；
# R3 只能靠 `g5_attn_real_tensor_forensics.py` 的**逐层张量**变体阶梯单独测量。
# **默认 (False, False) 时，eager 分支执行的算子与顺序与历史完全一致**（逐位不变）；
# 打开后只用于"在真实模型上逐项分离差异来源"，**不得**用于正式测量。
DIAG_QK_FP32 = False
DIAG_PROB_FP32 = False


def diag_precision() -> dict:
    return dict(qk_fp32=DIAG_QK_FP32, prob_fp32=DIAG_PROB_FP32)


def set_diag_precision(qk32=None, prob32=None) -> dict:
    """设置诊断用精度干预开关；返回设置后的状态。**不改变默认行为。**"""
    global DIAG_QK_FP32, DIAG_PROB_FP32
    if qk32 is not None:
        DIAG_QK_FP32 = bool(qk32)
    if prob32 is not None:
        DIAG_PROB_FP32 = bool(prob32)
    return diag_precision()


# 整模型层面可运行的预设（名字与逐层工具的变体一致；`out32` 只在逐层工具里有意义）
DIAG_PRESETS = {
    "eager": (False, False),      # R1 在、R2 在
    "qk32": (True, False),        # 只移除 R1
    "prob32": (False, True),      # 只移除 R2
    "fp32": (True, True),         # R1、R2 都移除（R3 仍受 o_proj 约束）
}


def set_diag_preset(name: str) -> dict:
    if name not in DIAG_PRESETS:
        raise ValueError(f"未知诊断精度预设 {name!r}（可选 {tuple(DIAG_PRESETS)}）")
    return set_diag_precision(*DIAG_PRESETS[name])


def _fresh_stats() -> dict:
    return {"strategy": ATTENTION_STRATEGY, "requested_backend": FUSED_BACKEND_NAME,
            "calls": {"eager": 0, "fused": 0},
            "by_phase": {}, "fused_by_phase": {}, "eager_by_phase": {},
            # 每阶段见过的最大 query 长度 —— 用来**正面证明**
            # "候选验证的 s>1 也仍然是 eager"（而不是只靠"没有 fused"反证）
            "s_max_by_phase": {}, "fused_s_max_by_phase": {},
            "fallbacks": [], "fallback_count": 0, "eager_reason_kinds": {},
            "phases_seen": [], "errors": []}


def set_attention_strategy(name: str) -> str:
    """显式切换策略。返回切换后的策略名。"""
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
    """显式标注当前执行阶段。**唯一**决定"是否允许融合"的输入（除策略本身）。

    合法阶段：`prefill` / `decode` / `propose` / `verify` / `repair` / `other`。
    只有 `prefill` 允许融合；其余一律 eager —— 这正是"候选验证不被误用新路径"的机制保证。
    """
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
    """返回统计（深拷贝），用于证明阶段分流与回退原因。"""
    import copy
    st = copy.deepcopy(_stats())
    st["strategy"] = ATTENTION_STRATEGY
    return st


def decide_attention_path(phase: str, s: int, strategy: str | None = None) -> tuple[str, str | None]:
    """**纯函数**决定走哪条路径。返回 `(path, reason)`，`path ∈ {"eager","fused"}`。

    可 CPU 测试；这是"阶段分流"的唯一裁决点。
    """
    strat = ATTENTION_STRATEGY if strategy is None else strategy
    if strat != "cudnn_prefill":
        return "eager", f"strategy={strat}"
    if phase != "prefill":
        return "eager", f"phase={phase!r} != 'prefill'（仅 prefill 允许融合）"
    if s < PREFILL_FUSED_MIN_S:
        return "eager", f"s={s} < PREFILL_FUSED_MIN_S={PREFILL_FUSED_MIN_S}（cuDNN 不支持 seq_len=1）"
    return "fused", None


def _kv_keep_mask(s: int, T: int, device) -> torch.Tensor:
    """偏移因果掩码的**保留**形式（SDPA 语义：True = 参与）。

    语义必须与 eager 路径完全一致：**第 i 个 query 只能访问 key_position ≤ (T − s) + i**。
    本式对三种情形统一成立：cache 为 None（T==s ⇒ j ≤ i）、s==1（j ≤ T−1，全允许）、
    以及非方阵的分块情况（**此时 `is_causal=True` 不等价**，见台账 §4.17）。
    """
    drop = torch.triu(torch.ones(s, T, device=device, dtype=torch.bool), diagonal=(T - s) + 1)
    return ~drop


def _sdpa(q, k, v, **kw):
    """`F.scaled_dot_product_attention` 的**一级间接**。

    存在的理由：让测试可以**安全地** spy/替换（直接改 `F.scaled_dot_product_attention`
    会污染全局 `torch.nn.functional`，那是不可接受的副作用）。
    """
    return F.scaled_dot_product_attention(q, k, v, **kw)


def _fused_prefill_attention(q, k, v, head_dim: int):
    """强制 CUDNN_ATTENTION 后端的融合 prefill attention。

    - GQA 用 `enable_gqa=True` 原生处理（**不做** `kv_repeat` 展开，这是省显存的一半来源）；
    - 掩码用 **bool keep-mask**（不是 `is_causal`，非方阵时二者不等价）；
    - **不捕获任何异常**：无可用 kernel / OOM 直接抛给调用方。
    """
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
        # **口径**：`fallback_count` = 策略=cudnn_prefill 时的**全部非融合调用**，
        # 它包含两大类，必须靠 `eager_reason_kinds` 分开读，否则会误读：
        #   short_block_seq_len_lt_min —— **短块策略真的触发了**（prefill 里 s < 2）；
        #   phase_not_prefill         —— 非 prefill 阶段**本就应该** eager（正常路径，不是回退）。
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
    """**防止"忘了标注阶段"导致静默不融合**：策略已切到 cudnn_prefill 但一次都没融合 ⇒ 抛错。"""
    st = attention_stats()
    if ATTENTION_STRATEGY == "cudnn_prefill" and st["calls"]["fused"] == 0:
        raise RuntimeError(
            "策略=cudnn_prefill 但 fused 调用数为 0 —— 说明没有任何调用点标注了 "
            "attention_phase('prefill')，**这属于静默不生效**，必须显式排查。"
            f"（已见阶段：{st['phases_seen']}；回退数：{st['fallback_count']}）")
    return st


def chunked_prefill(model, ids, chunk: int = 1024, *, need_logits_last: bool = True):
    """分块 prefill 的**规范实现**：把 attention 阶段显式标为 `'prefill'`。

    与各处 harness 原有的循环语义一致：逐块 `model._run`，**仅最后一块**取 logits。
    阶段标注是本函数存在的主要理由 —— 否则 `cudnn_prefill` 策略会静默不生效。
    """
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
        T = k_all.shape[2] if cache is not None else s
        # ---- 阶段分流（唯一裁决点；见文件头策略块）----
        path, reason = decide_attention_path(_ATTENTION_PHASE, s)
        _record_attention_call(path, _ATTENTION_PHASE, s, T, reason,
                               self.heads, self.kv_heads, self.head_dim)
        if path == "fused":
            # 融合 prefill：不做 kv_repeat 展开；掩码为 bool keep-mask
            out = _fused_prefill_attention(q, k_all, v_all, self.head_dim)
        else:
            k_rep = kv_repeat(k_all, self.heads // self.kv_heads)
            v_rep = kv_repeat(v_all, self.heads // self.kv_heads)
            # ---- R1：QK 分数精度。默认（DIAG_QK_FP32=False）与历史**逐位一致** ----
            if DIAG_QK_FP32:
                scores = torch.matmul(q.float(), k_rep.float().transpose(-2, -1)) / (self.head_dim ** 0.5)
            else:
                scores = torch.matmul(q, k_rep.transpose(-2, -1)) / (self.head_dim ** 0.5)
            p = T - s
            if cache is None:
                mask = torch.triu(torch.ones(s, T, device=x.device, dtype=torch.bool), diagonal=1)
                scores = scores.masked_fill(mask, float('-inf'))
            elif s == 1:
                pass
            else:
                mask = torch.triu(torch.ones(s, T, device=x.device, dtype=torch.bool), diagonal=p + 1)
                scores = scores.masked_fill(mask, float('-inf'))
            attn = torch.softmax(scores, dim=-1, dtype=torch.float32)
            # ---- R2：概率是否回转 bf16（R3 见策略块说明：整模型层面无法单独移除）----
            if DIAG_PROB_FP32:
                out = torch.matmul(attn, v_rep.float())
            else:
                attn = attn.to(q.dtype)                              # 默认路径：与历史一致
                out = torch.matmul(attn, v_rep)
            out = out.to(q.dtype)      # o_proj 是 bf16 ⇒ 这一步在默认路径是**无操作**（无拷贝）
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
