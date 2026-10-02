import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import contextlib

# ---------------------------------------------------------------------------
# 诊断探针（**默认 None ⇒ 完全不介入**；仅用于只读捕获，不改任何计算）
#
# 为什么需要它：Q/K/V 的 reshape、RoPE、KV 拼接、QK^T、mask、softmax、AV、head merge
# 都是**函数式算子**（没有对应的 nn.Module），普通 module hook 抓不到。
# 这里提供最小侵入的探针点：`_probe(name, tensor, who)` 只在**安装时**回调一次，
# 回调只做 `detach().clone()`（只读），**不**改张量、不改顺序、不改 dtype。
# 关闭（`clear_attn_probe()`）后 `_ATTN_PROBE is None` ⇒ 每个探针点只做一次身份判断，
# 不构造参数、不碰张量 ⇒ 与改动前**逐位等价**（由开启/关闭逐位控制与跨轮复现控制证明）。
# ---------------------------------------------------------------------------
_ATTN_PROBE = None


def set_attn_probe(fn) -> dict:
    """安装只读诊断探针；返回安装前的状态（供调用方核对）。"""
    global _ATTN_PROBE
    prev_active = _ATTN_PROBE is not None
    _ATTN_PROBE = fn
    return dict(installed=_ATTN_PROBE is not None, had_previous=prev_active)


def clear_attn_probe() -> dict:
    """移除探针（幂等）。"""
    global _ATTN_PROBE
    _ATTN_PROBE = None
    return dict(probe_is_none=_ATTN_PROBE is None)


def attn_probe_state() -> dict:
    return dict(active=_ATTN_PROBE is not None)


def _probe(name: str, t, who) -> None:
    """探针点：**关闭时零副作用**（只做一次身份判断）。"""
    if _ATTN_PROBE is not None:
        _ATTN_PROBE(name, t, who)


# ---------------------------------------------------------------------------
# 诊断性**因果干预**：QKᵀ 的逐行 M=1 causal-prefix（**默认关闭 = `None`**）
#
# 为什么需要它：已证明位置 8209 的 L0 里，`q_proj` 的瞬时差异被 RoPE 吸收，
# 而 QKᵀ 在 Q/K 输入逐位一致时**重新**产生差异（M=1/N=8209 与 M=4/N=8212 的执行几何不同）。
# 本轮要判定"把 QK 几何恢复成逐行 M=1 + 真实 causal 前缀"是否足以消除 L0/全层/token 的分歧。
#
# 纪律（与探针一致）：
#   * **默认关闭**：`_SERIAL_QK is None` ⇒ 分支只做两次身份判断，不碰张量 ⇒ 与改动前**逐位等价**；
#   * 只作用于**显式注册的模块实例**（`id(self)` 命中且同一对象）；
#   * 注册期间**强引用**这些模块（避免 `id` 复用），退出时（含异常）清空；
#   * 记录 `seen`（**所有**进入分支的 self_attn，含未注册的）与 `calls`（真正被替换的），
#     用来证明"干预只作用于 L0、其余 27 层仍走原 block 路径"；
#   * **不**改公式/dtype/精度/mask/softmax/AV/cache/argmax，**不**先算完整 M=4 再切片。
# ---------------------------------------------------------------------------
_SERIAL_QK = None


def set_serial_qk(*modules, label: str = "serial_qk") -> dict:
    """开启逐行 M=1 causal-prefix QK，**只**对给定 `SelfAttention` 模块实例生效。"""
    global _SERIAL_QK
    DC_REQ = [m for m in modules]
    if not DC_REQ:
        raise ValueError("必须显式给出至少一个 SelfAttention 模块实例")
    _SERIAL_QK = dict(label=str(label), mods={id(m): m for m in DC_REQ},
                      seen=[], calls=[])
    return serial_qk_state()


def clear_serial_qk() -> dict:
    """关闭干预（幂等）。"""
    global _SERIAL_QK
    _SERIAL_QK = None
    return dict(serial_qk_is_none=_SERIAL_QK is None)


def serial_qk_state() -> dict:
    """当前干预状态：是否开启、注册了哪些模块、进入分支/被替换的调用记录。"""
    if _SERIAL_QK is None:
        return dict(active=False, n_mods=0, seen_module_ids=[], applied=[],
                    note="默认关闭 ⇒ QK 走生产原式")
    calls = _SERIAL_QK["calls"]
    return dict(active=True, label=_SERIAL_QK["label"], n_mods=len(_SERIAL_QK["mods"]),
                mods=[dict(module_id=int(k), label=_label_of(v)) for k, v in _SERIAL_QK["mods"].items()],
                n_seen=len(_SERIAL_QK["seen"]), seen_module_ids=sorted(set(_SERIAL_QK["seen"])),
                n_applied=len(calls),
                applied=[dict(module_id=int(id(c["module"])), s=int(c["s"]), T=int(c["T"]),
                              p=int(c["p"]), visible=list(c["meta"]["visible"]),
                              contiguous_rows=bool(c["meta"]["contiguous_rows"]))
                         for c in calls],
                note="`seen` = 进入该分支的**所有** self_attn（含未注册的，只计数不改计算）；"
                     "`applied` = 真正被替换为逐行 M=1 的调用")


def _label_of(m) -> str:
    return f"{type(m).__name__}@{hex(id(m))}"


@contextlib.contextmanager
def serial_qk_for(*modules, label: str = "serial_qk"):
    """`with serial_qk_for(target.layers[0].self_attn): ...` —— 退出（含异常）必定关闭。"""
    info = set_serial_qk(*modules, label=label)
    try:
        yield info
    finally:
        clear_serial_qk()


def _serial_qk_branch(self, q, k_rep, s: int, T: int):
    """诊断分支：命中注册模块 ⇒ 逐行 M=1 causal-prefix；否则记为 `seen` 并返回 `None`。"""
    ent = _SERIAL_QK["mods"].get(id(self))
    if ent is self:                                   # 同一个对象（防 `id` 复用）
        from . import serial_causal_qk as SCQ
        scores, meta = SCQ.serial_causal_scores(q, k_rep, p=T - s, head_dim=self.head_dim)
        _SERIAL_QK["calls"].append(dict(module=self, s=int(s), T=int(T), p=int(T - s),
                                        meta=meta, scaled=scores))
        return scores
    _SERIAL_QK["seen"].append(int(id(self)))          # 只计数：这些层仍走原 block QK
    return None


def serial_qk_calls() -> list:
    """诊断用：返回本次注册期间的**原始**调用记录（含每行 raw/scaled 与布局，仍在原设备）。

    ⚠️ 必须在 `with serial_qk_for(...)` **体内**调用（退出后注册表被清空）。
    """
    return [] if _SERIAL_QK is None else list(_SERIAL_QK["calls"])


# ---------------------------------------------------------------------------
# 诊断性**因果干预 ②**：`o_proj` 的逐行 M=1 调用（**默认关闭 = `None`**）
#
# 为什么需要它：上一轮把 L0 的 QK 几何恢复成逐行 M=1 后，`heads_merged` 决策行已与 S 路径**逐位相同**，
# 但 `o_proj` 输出仍有 2 个坐标不同 ⇒ `o_proj` 是本例 L0 中**首个持续到出口**的 shape-sensitive 边界。
# 本轮把 `self.o_proj(out)` 这一次 M=4 调用换成**四次** M=1 的**同一模块**调用再拼回。
#
# 纪律与 `_SERIAL_QK` 完全一致：默认关闭（关闭时只多一次身份判断）、只作用于显式注册的模块实例、
# 注册期间强引用、退出（含异常）即清空、`seen`/`calls` 分别记录"进入但未注册"与"真正被替换"的次数。
# ---------------------------------------------------------------------------
_SERIAL_LINEAR = None


def set_serial_linear(*modules, label: str = "serial_linear") -> dict:
    """开启"逐行 M=1 调用同一 `nn.Linear`"，**只**对给定模块实例生效（如 `layers[0].self_attn.o_proj`）。"""
    global _SERIAL_LINEAR
    mods = [m for m in modules]
    if not mods:
        raise ValueError("必须显式给出至少一个 nn.Linear 模块实例")
    _SERIAL_LINEAR = dict(label=str(label), mods={id(m): m for m in mods}, seen=[], calls=[])
    return serial_linear_state()


def clear_serial_linear() -> dict:
    """关闭该干预（幂等）。"""
    global _SERIAL_LINEAR
    _SERIAL_LINEAR = None
    return dict(serial_linear_is_none=_SERIAL_LINEAR is None)


def serial_linear_state() -> dict:
    if _SERIAL_LINEAR is None:
        return dict(active=False, n_mods=0, seen_module_ids=[], applied=[],
                    note="默认关闭 ⇒ Linear 走生产原式的一次调用")
    calls = _SERIAL_LINEAR["calls"]
    return dict(active=True, label=_SERIAL_LINEAR["label"], n_mods=len(_SERIAL_LINEAR["mods"]),
                mods=[dict(module_id=int(k)) for k in _SERIAL_LINEAR["mods"]],
                n_seen=len(_SERIAL_LINEAR["seen"]), seen_module_ids=sorted(set(_SERIAL_LINEAR["seen"])),
                n_applied=len(calls),
                applied=[dict(module_id=int(id(c["module"])), s=int(c["meta"]["s"]),
                              per_row_input_shapes=list(c["meta"]["per_row_input_shapes"]))
                         for c in calls],
                note="`seen` = 进入该分支但**未注册**的 Linear（只计数，仍走原调用）；"
                     "`applied` = 真正被替换为逐行 M=1 的调用")


@contextlib.contextmanager
def serial_linear_for(*modules, label: str = "serial_linear"):
    """`with serial_linear_for(target.layers[0].self_attn.o_proj): ...` —— 退出（含异常）必定关闭。"""
    info = set_serial_linear(*modules, label=label)
    try:
        yield info
    finally:
        clear_serial_linear()


def _serial_linear_branch(mod, x):
    """诊断分支：命中注册模块 ⇒ 逐行 M=1 调用同一模块；否则记为 `seen` 并返回 `None`。"""
    ent = _SERIAL_LINEAR["mods"].get(id(mod))
    if ent is mod:                                    # 同一个对象（防 `id` 复用）
        from . import serial_align as SA
        out, meta = SA.serial_linear_rows(mod, x)
        _SERIAL_LINEAR["calls"].append(dict(module=mod, meta=meta, out=out))
        return out
    _SERIAL_LINEAR["seen"].append(int(id(mod)))       # 只计数：这些 Linear 仍走原调用
    return None


def serial_linear_calls() -> list:
    """诊断用：返回本次注册期间的**原始**调用记录（含逐行元数据与输出）。

    ⚠️ 必须在 `with serial_linear_for(...)` **体内**调用。
    """
    return [] if _SERIAL_LINEAR is None else list(_SERIAL_LINEAR["calls"])



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
        _probe("mlp_input", x, self)                       # 诊断（默认关闭）
        g = self.gate_proj(x)
        _probe("mlp_gate_proj", g, self)                   # 诊断（默认关闭）
        gate = F.silu(g)
        _probe("mlp_activation", gate, self)               # 诊断（默认关闭）
        up = self.up_proj(x)
        _probe("mlp_up_proj", up, self)                    # 诊断（默认关闭）
        gp = gate * up
        _probe("mlp_gated_product", gp, self)              # 诊断（默认关闭）
        down = self.down_proj(gp)
        _probe("mlp_down_proj", down, self)                # 诊断（默认关闭）
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
        ql = self.q_proj(x)
        _probe("q_linear", ql, self)                       # 诊断（默认关闭）
        q = ql.view(b, s, self.heads, self.head_dim).transpose(1, 2)
        _probe("q_reshaped", q, self)                      # 诊断（默认关闭）
        kl = self.k_proj(x)
        _probe("k_linear", kl, self)                       # 诊断（默认关闭）
        k = kl.view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        _probe("k_reshaped", k, self)                      # 诊断（默认关闭）
        vl = self.v_proj(x)
        _probe("v_linear", vl, self)                       # 诊断（默认关闭）
        v = vl.view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        _probe("v_reshaped", v, self)                      # 诊断（默认关闭）
        q = apply_rope(q, cos, sin)
        _probe("q_rope", q, self)                          # 诊断（默认关闭）
        k = apply_rope(k, cos, sin)
        _probe("k_rope", k, self)                          # 诊断（默认关闭）
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
        _probe("k_visible", k_all, self)                   # 诊断（默认关闭）
        _probe("v_visible", v_all, self)                   # 诊断（默认关闭）
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
            _probe("k_repeated", k_rep, self)              # 诊断（默认关闭；只记 shape）
            _probe("v_repeated", v_rep, self)              # 诊断（默认关闭；只记 shape）
            # ---- R1：QK 分数精度。默认（DIAG_QK_FP32=False）与历史**逐位一致** ----
            # 诊断性因果干预（**默认关闭**）：`_SERIAL_QK is None` ⇒ `_sq is None`，
            # 只多一次全局身份判断，下面走的就是**原样的生产表达式**（逐位等价）。
            _sq = (_serial_qk_branch(self, q, k_rep, s, T)
                   if (_SERIAL_QK is not None and not DIAG_QK_FP32) else None)
            if _sq is not None:
                scores = _sq                                                # 诊断：逐行 M=1 causal-prefix
            elif DIAG_QK_FP32:
                scores = torch.matmul(q.float(), k_rep.float().transpose(-2, -1)) / (self.head_dim ** 0.5)
            else:
                scores = torch.matmul(q, k_rep.transpose(-2, -1)) / (self.head_dim ** 0.5)
            p = T - s
            _probe("qk_scores_pre_mask", scores, self)     # 诊断（默认关闭）
            if cache is None:
                mask = torch.triu(torch.ones(s, T, device=x.device, dtype=torch.bool), diagonal=1)
                scores = scores.masked_fill(mask, float('-inf'))
            elif s == 1:
                pass
            else:
                mask = torch.triu(torch.ones(s, T, device=x.device, dtype=torch.bool), diagonal=p + 1)
                scores = scores.masked_fill(mask, float('-inf'))
            _probe("qk_scores_post_mask", scores, self)    # 诊断（默认关闭）
            attn = torch.softmax(scores, dim=-1, dtype=torch.float32)
            _probe("attention_prob", attn, self)           # 诊断（默认关闭；**fp32** 保持原 dtype）
            # ---- R2：概率是否回转 bf16（R3 见策略块说明：整模型层面无法单独移除）----
            if DIAG_PROB_FP32:
                out = torch.matmul(attn, v_rep.float())
            else:
                attn = attn.to(q.dtype)                              # 默认路径：与历史一致
                out = torch.matmul(attn, v_rep)
            _probe("attention_times_v", out, self)         # 诊断（默认关闭）
            out = out.to(q.dtype)      # o_proj 是 bf16 ⇒ 这一步在默认路径是**无操作**（无拷贝）
        out = out.transpose(1, 2).contiguous().view(b, s, self.heads * self.head_dim)
        _probe("heads_merged", out, self)                  # 诊断（默认关闭）
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
        # 诊断性因果干预 ②（**默认关闭**）：注册的 `o_proj` 改为**逐行 M=1** 调用同一模块再拼回。
        # 关闭时 `_SERIAL_LINEAR is None` ⇒ 只多一次身份判断，下面走的就是**原样的生产调用**。
        _sl = (_serial_linear_branch(self.o_proj, out) if _SERIAL_LINEAR is not None else None)
        op = _sl if _sl is not None else self.o_proj(out)
        _probe("o_proj", op, self)                         # 诊断（默认关闭）
        return op, new_cache
    
    
class DecoderLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int, intermediate: int, attn_bias: bool = False):
        super().__init__()
        self.self_attn = SelfAttention(hidden, heads, kv_heads, head_dim, attn_bias)
        self.mlp = MLP(hidden, intermediate)
        self.input_layernorm = RMSNORM(hidden)
        self.post_attention_layernorm = RMSNORM(hidden)

    def forward(self, x: torch.Tensor, cos, sin, cache = None) -> torch.Tensor:
        _probe("x0", x, self)                              # 诊断（默认关闭）
        n1 = self.input_layernorm(x)
        _probe("input_rmsnorm", n1, self)                  # 诊断（默认关闭）
        atten_out, new_attn_cache = self.self_attn(n1, cos, sin, cache)
        _probe("attn_out", atten_out, self)                # 诊断（默认关闭；= o_proj 输出）
        xr = x + atten_out
        _probe("attention_residual", xr, self)             # 诊断（默认关闭）
        n2 = self.post_attention_layernorm(xr)
        _probe("post_attention_rmsnorm", n2, self)         # 诊断（默认关闭）
        mo = self.mlp(n2)
        _probe("mlp_out", mo, self)                        # 诊断（默认关闭；= down_proj 输出）
        z = xr + mo
        _probe("layer_output", z, self)                    # 诊断（默认关闭）
        return z, new_attn_cache

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
