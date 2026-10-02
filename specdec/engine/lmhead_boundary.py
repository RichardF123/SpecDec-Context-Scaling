"""lmhead_boundary.py — **LM head 边界的默认关闭捕获**（最终 RMSNorm 之后、权重乘法之前）。

## 边界在哪里（以**实际代码**为准，不假定有独立 `lm_head` 模块）
`model_self.QwenLM._run`（`model_self.py:485-499`）最后两步是：

```
x = self.norm(x)                       # 最终 RMSNorm（fp32 内部计算 → 回到输入 dtype = bf16）
if need_logits:
    logits = x @ self.embed_tokens.weight.T     # ★ 生产表达式：**绑定权重**（tied embedding）
```

⇒ 本模块的边界定义是：**`self.norm` 的输出**（形状 `(1, T, hidden)`、bf16），它就是权重乘法的**左操作数**。
**没有** `lm_head` 模块：输出权重就是 `self.embed_tokens.weight`（`(vocab, hidden)`、bf16）。

## 捕获怎么实现（为什么不改 `_run`）
用 `model.norm.register_forward_hook(...)`：`PyTorch` 官方钩子只**读取**模块的输入/输出，
既不改 `_run`、也不改任何张量，移除后逐位回到安装前。**默认不挂**（`enabled=False` 时一个钩子都不挂）。

## 诚实边界
* 捕获的是**该次前向**的最终 RMSNorm 输出；它**不**等于"自由生成时那一行的隐藏态"——
  两者只有在"同历史 + 同输入 + 同执行形状"下才是同一件事（由调用方核对）。
* 本模块**不**做任何精度替换：投影一律用生产表达式与生产 dtype（bf16）。
* 捕获张量会**克隆**保留（`detach().clone()`），不会被后续前向/缓存操作改写。
"""
from __future__ import annotations

import contextlib
import inspect

import torch

from . import model_self as M

# 生产表达式（用于源码级事实核对与文档；投影一律走 `project`）
PROJECTION_EXPRESSION = "x @ self.embed_tokens.weight.T"
NORM_CALL = "x = self.norm(x)"


class BoundaryError(RuntimeError):
    """LM head 边界捕获的实现/契约错误（不是引擎错误）。"""


_S: dict = dict(installed=False, enabled=False, handle=None, model=None, caps={}, n_calls=0,
                baseline_hooks=None)


def code_facts(model=None) -> dict:
    """源码级事实：表达式、norm 调用、以及"是否存在独立 lm_head"。**CPU 可测**。"""
    try:
        src = inspect.getsource(M.QwenLM._run)
    except Exception as e:                                          # noqa: BLE001
        src = f"<{type(e).__name__}: {e}>"
    facts = dict(module="specdec.engine.model_self", symbol="QwenLM._run",
                 source_has_expression=bool(PROJECTION_EXPRESSION in src),
                 source_has_norm_call=bool(NORM_CALL in src),
                 projection_expression=PROJECTION_EXPRESSION,
                 weight_source="self.embed_tokens.weight（tied embedding，**没有** lm_head 模块）",
                 boundary="self.norm 的输出（最终 RMSNorm 之后、权重乘法之前）",
                 norm_internal="fp32 内部计算后 `normed.to(orig_dtype)`（dtype 回到输入 dtype）")
    if model is not None:
        facts.update(has_lm_head_module=bool(hasattr(model, "lm_head")),
                     has_norm=bool(hasattr(model, "norm")),
                     weight_key="embed_tokens.weight",
                     weight_dtype=str(model.embed_tokens.weight.dtype),
                     weight_shape=[int(x) for x in model.embed_tokens.weight.shape])
    return facts


def weights_info(model, *, digest=True) -> dict:
    """实际输出权重的 dtype/shape/身份/摘要。

    * **身份**用 `is` + `id()` + `data_ptr()`（最强、最便宜）；
    * **摘要**用**抽样** sha16（全量 466 MB 的 sha 没有必要且会多占内存）⇒ 明确标注"抽样"。
    """
    from . import sb_align as A
    W = model.embed_tokens.weight
    d = dict(shape=[int(x) for x in W.shape], dtype=str(W.dtype), device=str(W.device),
             is_contiguous=bool(W.is_contiguous()), stride=[int(x) for x in W.stride()],
             id=id(W), data_ptr=int(W.data_ptr()),
             is_embed_tokens_weight=bool(W is model.embed_tokens.weight),
             n_params=int(W.numel()),
             note="`is`/`id`/`data_ptr` 是身份证据；`digest_subsample` 是**抽样**摘要，不是全量张量摘要")
    if digest:
        d["digest_subsample"] = A.stable_sha(W[::97, ::7])
        d["digest_subsample_shape"] = [int(x) for x in W[::97, ::7].shape]
    return d


def _layout(t) -> dict:
    if not torch.is_tensor(t):
        return dict(kind="not_a_tensor", type=str(type(t)))
    return dict(shape=[int(x) for x in t.shape], stride=[int(x) for x in t.stride()],
                dtype=str(t.dtype), device=str(t.device),
                is_contiguous=bool(t.is_contiguous()),
                is_view=bool(getattr(t, "_base", None) is not None))


def _normal_clone(t):
    """克隆成**普通张量**（不是 inference tensor）。

    为什么必须这样：`_run` 是在 `torch.inference_mode()` 里调用的，钩子里直接 `.clone()`
    得到的是 **inference tensor**；它一旦被拿到 `inference_mode` **之外**与普通张量（如权重 `W`）做运算，
    PyTorch 会报 `RuntimeError: Inference tensors cannot be saved for backward`（自测首跑即此错）。
    `torch.inference_mode(False)` 是官方逃生口 ⇒ 在它里面克隆即可得到普通张量。
    """
    with torch.inference_mode(False):
        return t.detach().clone()


def _normal_cache_clone(cache):
    """缓存的**普通张量**深拷贝（保留全部键）——用于"轮前状态"快照。

    ⚠️ **口径**（首版在这里踩过）：`QwenLM._run` 逐层传给 `DecoderLayer` 的是
    `cache_list[i]`，也就是**单层的 cache dict**（`{"k","v"}`，fp8 时还有 `*_scale`/`fmt`），
    **不是**整模型的 `list[dict]`。所以这里同时支持两种形态：`dict` 与 `list[dict]`。

    为什么不用 `cache_clone.clone_list_cache_full`：它基于 `copy.deepcopy`（面向 `list[dict]`），
    对 **inference tensor** 仍会产出 inference tensor；后续若要在 `inference_mode` 之外与普通张量
    做运算（比较/投影）会报错。这里逐张量 `_normal_clone`，其余键深拷贝。
    """
    import copy as _copy
    if isinstance(cache, dict):
        return {k: (_normal_clone(v) if torch.is_tensor(v) else _copy.deepcopy(v))
                for k, v in cache.items()}
    return [{k: (_normal_clone(v) if torch.is_tensor(v) else _copy.deepcopy(v))
             for k, v in layer.items()} for layer in cache]


def _cache_len_of(cache) -> int | None:
    """缓存长度（同时支持单层 `dict` 与整模型 `list[dict]`）。"""
    if cache is None:
        return None
    d = cache[0] if isinstance(cache, (list, tuple)) else cache
    k = d.get("k") if isinstance(d, dict) else None
    return int(k.shape[2]) if torch.is_tensor(k) else None


def norm_info(model) -> dict:
    """最终 RMSNorm 的实际参数与代码路径（**不复用任何"应该怎样"的假设**）。

    `model_self.RMSNORM.forward`：`orig=x.dtype` → `x32=x.float()` →
    `mean(x32*x32, dim=-1)` → `x32*rsqrt(var+eps)` → `.to(orig_dtype)` → `weight * normed`
    ⇒ **fp32 内部归约 + 输出回到输入 dtype**，且是**逐行**运算（归约维是最后一维 hidden）。
    """
    n = model.norm
    w = getattr(n, "weight", None)
    from . import sb_align as A
    d = dict(module_class=type(n).__name__, is_model_norm=bool(n is model.norm),
             eps=(None if getattr(n, "eps", None) is None else float(n.eps)),
             weight=None if w is None else dict(
                 shape=[int(x) for x in w.shape], dtype=str(w.dtype), device=str(w.device),
                 stride=[int(x) for x in w.stride()], is_contiguous=bool(w.is_contiguous()),
                 id=id(w), data_ptr=int(w.data_ptr()), n_params=int(w.numel()),
                 digest_subsample=A.stable_sha(w[::7])),
             code_path=("orig_dtype=x.dtype; x32=x.float(); var=mean(x32*x32,dim=-1,keepdim=True); "
                        "normed=x32*rsqrt(var+eps); return weight*normed.to(orig_dtype)"),
             row_wise="归约维是最后一维（hidden）⇒ 每一行的输出只依赖该行输入",
             note="本模块就是**最终** RMSNorm（`_run` 里 `x = self.norm(x)`）；它同时是权重乘法的输入边")
    return d


def norm_forward(model, x, *, phase: str | None = None) -> tuple:
    """用**模型现有的** `model.norm` 模块实例计算最终 RMSNorm（**不重写公式**）。

    * 输入 `x` 必须是**原生 dtype**（bf16）且形状 `(1,T,hidden)` 或 `(1,hidden)`；
      dtype 不一致会**报错**（历史上踩过"拿 fp32 统计副本去投影/Norm"的坑）；
    * 在 `torch.inference_mode()` 内调用（与 `_run` 的调用环境一致）；
    * 返回 `(y, meta)`，`y` 是**普通张量**克隆（原始 dtype），`meta` 含输入/输出的 layout。
    """
    if not torch.is_tensor(x):
        raise BoundaryError("norm_forward 只接受张量")
    w = getattr(model.norm, "weight", None)
    if w is not None and x.dtype != w.dtype:
        raise BoundaryError(f"输入 dtype={x.dtype} 与 Norm 权重 dtype={w.dtype} 不一致"
                            f"（不要拿 fp32 统计副本做 Norm/投影）")
    if x.dim() not in (2, 3):
        raise BoundaryError(f"输入维度应为 2 或 3，得到 {list(x.shape)}")
    ctx = M.attention_phase(phase) if phase else contextlib.nullcontext()
    with torch.inference_mode(), ctx:
        y = model.norm(x)
    if not torch.is_tensor(y):
        raise BoundaryError("model.norm 未返回张量")
    if tuple(y.shape) != tuple(x.shape):
        raise BoundaryError(f"Norm 输出形状与输入不同：{list(y.shape)} vs {list(x.shape)}")
    meta = dict(input=_layout(x), output=_layout(y), dtype_in=str(x.dtype), dtype_out=str(y.dtype),
                dtype_preserved=bool(y.dtype == x.dtype),
                module_class=type(model.norm).__name__,
                note="调用的是模型**现有** Norm 模块；未改公式、未改 dtype 规则")
    return _normal_clone(y), meta


def layer_info(model, index: int) -> dict:
    """第 `index` 层（0 基）的**实际**结构与代码路径（不假定层数、不假定子模块名）。"""
    lay = model.layers[index]
    n_layers = len(model.layers)
    subs = {}
    for nm in ("input_layernorm", "post_attention_layernorm"):
        m = getattr(lay, nm, None)
        w = getattr(m, "weight", None)
        if w is not None:
            subs[nm] = dict(shape=[int(x) for x in w.shape], dtype=str(w.dtype),
                            data_ptr=int(w.data_ptr()), eps=float(getattr(m, "eps", 0.0)))
    sa = getattr(lay, "self_attn", None)
    subs["self_attn"] = dict(
        class_name=(type(sa).__name__ if sa is not None else None),
        heads=int(getattr(sa, "heads", -1)), kv_heads=int(getattr(sa, "kv_heads", -1)),
        head_dim=int(getattr(sa, "head_dim", -1)),
        q_proj_dtype=(str(sa.q_proj.weight.dtype) if sa is not None else None))
    return dict(index=int(index), n_layers=int(n_layers), is_last=bool(index == n_layers - 1),
                class_name=type(lay).__name__, submodules=subs,
                code_path=("atten_out, new_cache = self.self_attn(self.input_layernorm(x), cos, sin, cache); "
                           "x = x + atten_out; x = x + self.mlp(self.post_attention_layernorm(x)); "
                           "return x, new_attn_cache"),
                note="层输入 `x` 就是上一层的输出；最后一层的输出**直接**进入 `self.norm`（`_run` 的下一行）")


@contextlib.contextmanager
def embedding_capture(model, *, tag="embed"):
    """只读捕获 embedding 的输出与输入 token id（用于核对"对应行"是否同一 token、同一位置）。"""
    rec: dict = dict(tag=tag, n_calls=0)
    def hook(module, inputs, output):
        rec["n_calls"] += 1
        ids = inputs[0] if inputs else None
        rec["ids"] = ([int(v) for v in ids.reshape(-1).tolist()] if torch.is_tensor(ids) else None)
        rec["ids_layout"] = _layout(ids) if torch.is_tensor(ids) else None
        rec["out"] = _normal_clone(output) if torch.is_tensor(output) else None
        rec["out_layout"] = _layout(output) if torch.is_tensor(output) else None
    h = model.embed_tokens.register_forward_hook(hook)
    try:
        yield rec
    finally:
        h.remove()


@contextlib.contextmanager
def layer_capture(model, index: int, *, tag="layer", clone_cache: bool = True):
    """在指定层上挂**只读** pre/post 钩子，取「层输入 / cos / sin / **轮前缓存**」与「层输出」。

    ⚠️ **关键纪律**：层的 `forward` 会**就地追加** `cache["k"]/["v"]`（`self_attn` 里
    `cache["k"] = torch.cat([...], dim=2)`）⇒ "轮前状态"**只能**在 `forward` 执行**之前**取。
    本函数用 `register_forward_pre_hook` 取**入口**状态并立即做**独立深拷贝**；
    **不**把 post 钩子里的缓存当作轮前状态（那份已经被追加过）。
    """
    rec: dict = dict(tag=tag, index=int(index), n_pre=0, n_post=0, pre=None, post=None)
    lay = model.layers[index]

    def pre_hook(module, inputs):
        rec["n_pre"] += 1
        ins = list(inputs) + [None, None, None]
        x, cos, sin, cache = ins[0], ins[1], ins[2], ins[3]
        rec["pre"] = dict(
            a=(_normal_clone(x) if torch.is_tensor(x) else None), a_layout=_layout(x),
            cos=(_normal_clone(cos) if torch.is_tensor(cos) else None),
            sin=(_normal_clone(sin) if torch.is_tensor(sin) else None),
            cos_layout=_layout(cos), sin_layout=_layout(sin),
            cache_pre=(_normal_cache_clone(cache) if (clone_cache and cache is not None) else None),
            cache_len_before=_cache_len_of(cache),
            phase=M.current_attention_phase(),
            module_is_layer=bool(module is lay),
            note="入口状态：`a` 是层输入（= 上一层输出）；`cache_pre` 是**未追加**的独立深拷贝")

    def post_hook(module, inputs, output):
        rec["n_post"] += 1
        z = output[0] if isinstance(output, (tuple, list)) and output else output
        cache_after = output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else None
        rec["post"] = dict(
            z=(_normal_clone(z) if torch.is_tensor(z) else None), z_layout=_layout(z),
            cache_len_after=_cache_len_of(cache_after),
            same_cache_object=bool(cache_after is inputs[3]) if len(inputs) > 3 else None,
            phase=M.current_attention_phase(),
            note="`z` 是层输出；若该层是最后一层，`z` **就是** 随后 `self.norm` 的输入")

    h1 = lay.register_forward_pre_hook(pre_hook)
    h2 = lay.register_forward_hook(post_hook)
    try:
        yield rec
    finally:
        h1.remove()
        h2.remove()


def layer_forward(model, index: int, x, cos, sin, cache, *, phase: str | None = None) -> tuple:
    """用**模型现有的**第 `index` 层做一次前向（**不重写公式**）。

    * `cache` 必须由调用方给出**独立克隆**（本函数会就地追加它）；
    * 返回 `(z, cache_len_after, meta)`，`z` 是**普通张量**克隆（原始 dtype）；
    * `meta` 记录：输入/输出 layout、实际 phase、**阶段分流的判定**（引擎纯函数）、
      期望的掩码可见范围（按 `model_self.py:239-247/415-423` 的规则**由本函数计算并记录**）、缓存长度变化。
    """
    if not torch.is_tensor(x):
        raise BoundaryError("layer_forward 只接受张量输入")
    if x.dim() != 3:
        raise BoundaryError(f"层输入应为 (1,T,H)，得到 {list(x.shape)}")
    w = getattr(getattr(model.layers[index], "self_attn", None), "q_proj", None)
    if w is not None and x.dtype != w.weight.dtype:
        raise BoundaryError(f"输入 dtype={x.dtype} 与层权重 dtype={w.weight.dtype} 不一致")
    s = int(x.shape[1])
    pre_len = _cache_len_of(cache)
    T = (s if pre_len is None else pre_len + s)
    p = T - s
    path, reason = M.decide_attention_path((phase or "other"), s)
    ctx = M.attention_phase(phase) if phase else contextlib.nullcontext()
    with torch.inference_mode(), ctx:
        z, cache_after = model.layers[index](x, cos, sin, cache)
    if not torch.is_tensor(z):
        raise BoundaryError("层未返回张量输出")
    meta = dict(input=_layout(x), output=_layout(z), dtype_in=str(x.dtype), dtype_out=str(z.dtype),
                phase=(phase or M.current_attention_phase()), dispatch=dict(path=path, reason=reason),
                s=s, T=T, p=p,
                cache_len_before=pre_len, cache_len_after=_cache_len_of(cache_after),
                cache_is_single_layer_dict=bool(isinstance(cache, dict)),
                mask_rule=("cache is None ⇒ triu(diagonal=1)" if pre_len is None else
                           ("s == 1 ⇒ 不加掩码（`elif s == 1: pass`），只可能看到 ≤ 自身位置"
                            if s == 1 else
                            f"s > 1 ⇒ triu(diagonal=p+1)（p={p}）⇒ 第 j 行可见 key ≤ p+j")),
                visible_last_key_row0=((T - 1) if s == 1 else p),
                source="掩码/分流规则来自 `model_self.py:239-247` 与 `:401-423`（本函数**只读地**照此记录，不参与计算）")
    return _normal_clone(z), _cache_len_of(cache_after), meta


@contextlib.contextmanager
def layer_chain_capture(model, *, indices=None, row_index: int = 0, tag: str = "chain",
                        want_rope: bool = True):
    """对**每一层**同时挂 pre/post 只读钩子，只记录**决策行**与元数据（全层边界捕获）。

    * 只保存 `a_row`/`z_row`（`(hidden,)`、**原始 dtype**、在 GPU 上 `detach().clone()`），
      **不**保存整段 `(1,S,hidden)` 激活；
    * **不在钩子里做 `.cpu()`**：那会每层引入一次同步、可能改变执行轨迹；`.cpu()` 由调用方在
      **整个前向结束之后**统一做（数据量很小：28 层 × 2 × 1536 × 2B ≈ 172 KB/条路径）；
    * 只读：钩子返回 `None`，不改任何张量、不改 cache、不改顺序；
    * `finally` 里移除全部钩子（异常也会移除）。
    """
    idx = list(indices if indices is not None else range(len(model.layers)))
    rec: dict = dict(tag=tag, indices=[int(i) for i in idx], row_index=int(row_index),
                     want_rope=bool(want_rope), layers={}, n_pre=0, n_post=0)
    handles = []

    def mk_pre(i):
        def hook(module, inputs):
            rec["n_pre"] += 1
            ins = list(inputs) + [None, None, None]
            x, cos, sin, cache = ins[0], ins[1], ins[2], ins[3]
            d = rec["layers"].setdefault(i, {})
            row = (x[0, row_index] if (torch.is_tensor(x) and x.dim() == 3) else None)
            d["pre"] = dict(
                a_row=(_normal_clone(row) if row is not None else None),
                a_row_layout=_layout(row),
                a_shape=([int(v) for v in x.shape] if torch.is_tensor(x) else None),
                s=(int(x.shape[1]) if torch.is_tensor(x) else None),
                phase=M.current_attention_phase(),
                cache_len_before=_cache_len_of(cache),
                cache_keys=(sorted(cache.keys()) if isinstance(cache, dict) else None),
                cos_row=((_normal_clone(cos[0, 0, row_index]) if (want_rope and torch.is_tensor(cos)
                                                                  and cos.dim() == 4) else None)),
                sin_row=((_normal_clone(sin[0, 0, row_index]) if (want_rope and torch.is_tensor(sin)
                                                                  and sin.dim() == 4) else None)),
                module_is_layer=bool(module is model.layers[i]))
            return None
        return hook

    def mk_post(i):
        def hook(module, inputs, output):
            rec["n_post"] += 1
            z = output[0] if isinstance(output, (tuple, list)) and output else output
            cache_after = output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else None
            row = (z[0, row_index] if (torch.is_tensor(z) and z.dim() == 3) else None)
            d = rec["layers"].setdefault(i, {})
            d["post"] = dict(
                z_row=(_normal_clone(row) if row is not None else None),
                z_row_layout=_layout(row),
                z_shape=([int(v) for v in z.shape] if torch.is_tensor(z) else None),
                cache_len_after=_cache_len_of(cache_after),
                same_cache_object=(bool(cache_after is inputs[3]) if len(inputs) > 3 else None),
                phase=M.current_attention_phase())
            return None
        return hook

    for i in idx:
        handles.append(model.layers[i].register_forward_pre_hook(mk_pre(i)))
    for i in idx:
        handles.append(model.layers[i].register_forward_hook(mk_post(i)))
    try:
        yield rec
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:                                       # noqa: BLE001
                pass


def to_cpu_rows(rec: dict) -> dict:
    """把链式捕获里的**决策行**统一搬到 CPU（**前向结束之后**才调用）。"""
    out = {}
    for i, d in rec["layers"].items():
        e = {}
        for k in ("pre", "post"):
            v = d.get(k)
            if v is None:
                e[k] = None
                continue
            e[k] = dict(v)
            for rk in ("a_row", "z_row", "cos_row", "sin_row"):
                if torch.is_tensor(e[k].get(rk)):
                    e[k][rk] = e[k][rk].detach().to("cpu")
        out[i] = e
    return dict(tag=rec["tag"], indices=rec["indices"], row_index=rec["row_index"],
                n_pre=rec["n_pre"], n_post=rec["n_post"], layers=out)


def install(model, *, enabled: bool = True, tag: str = "cap") -> dict:
    """挂/不挂捕获钩子。**必须在 `finally` / `remove()` 里移除**。

    `enabled=False` ⇒ **一个钩子都不挂**，`enabled=False` 的运行与"从未安装过"逐位等价（CPU 自测覆盖）。
    """
    if _S["installed"]:
        raise BoundaryError("已经安装过（同一时刻只允许一个捕获）")
    _S["model"] = model
    _S["baseline_hooks"] = sorted(int(k) for k in model.norm._forward_hooks.keys())
    if not enabled:
        _S.update(installed=False, enabled=False, handle=None)
        return dict(installed=False, enabled=False, tag=tag, baseline_hooks=_S["baseline_hooks"],
                    note="诊断关闭 ⇒ 不挂任何钩子；行为与未安装逐位等价")
    _S["caps"] = {}
    _S["n_calls"] = 0

    def hook(module, inputs, output):
        _S["n_calls"] += 1
        i = _S["n_calls"]
        rec = dict(call_index=i, tag=tag,
                   x_in=(_normal_clone(inputs[0]) if inputs and torch.is_tensor(inputs[0]) else None),
                   y_out=(_normal_clone(output) if torch.is_tensor(output) else None),
                   x_in_layout=_layout(inputs[0]) if inputs and torch.is_tensor(inputs[0]) else None,
                   y_out_layout=_layout(output) if torch.is_tensor(output) else None)
        _S["caps"][i] = rec
        return None                     # 钩子**只读**：返回 None 不改变输出

    _S["handle"] = model.norm.register_forward_hook(hook)
    _S["installed"] = True
    _S["enabled"] = True
    return dict(installed=True, enabled=True, tag=tag, baseline_hooks=_S["baseline_hooks"],
                hook_id=int(_S["handle"].id),
                semantics="只读记录 `self.norm` 的输入与输出；不改 `_run`、不改张量、不改 dtype",
                boundary=code_facts(model)["boundary"])


def take() -> dict:
    """取出本次安装期间的所有捕获（按调用序号），并清空缓存。"""
    caps = dict(_S["caps"])
    _S["caps"] = {}
    return dict(n_calls=_S["n_calls"], caps=caps)


def remove() -> dict:
    """移除钩子并核对：`model.norm` 上的钩子集合回到安装前基线（逐 id 比较）。"""
    if _S["handle"] is None:
        return dict(removed=False, note="未安装过（无需移除）", baseline_hooks=_S["baseline_hooks"])
    try:
        _S["handle"].remove()
    finally:
        _S["handle"] = None
        _S["installed"] = False
    now = sorted(int(k) for k in _S["model"].norm._forward_hooks.keys())
    return dict(removed=True, hooks_after=now, baseline_hooks=_S["baseline_hooks"],
                ok=bool(now == (_S["baseline_hooks"] or [])))


def assert_clean() -> dict:
    res = remove()
    if res.get("removed") and not res.get("ok"):
        raise BoundaryError(f"钩子未完全移除：{res}")
    if _S["handle"] is not None:
        raise BoundaryError("钩子仍然存在")
    return res


@contextlib.contextmanager
def record(model, *, tag: str = "cap", enabled: bool = True):
    """with 体里捕获**该模型**最终 RMSNorm 的输出；**异常也会**移除钩子（`finally`）。"""
    info = install(model, enabled=enabled, tag=tag)
    try:
        yield info
    finally:
        assert_clean()


def row_of(lg, index: int | None = None):
    """取决策行为**普通张量**（一维 `(V,)`）。

    * `lg` 可以是 `(1,T,V)`（`_run` 的原始输出）或 `(T,V)`（`_verify` 已切片后的 `rows`）；
    * `index=None` ⇒ **最后一行**（S 路径：`(1,1,V)` 的唯一一行）；否则取第 `index` 行（B 路径：`rows[0]`）；
    * 一律在 `inference_mode(False)` 里克隆 ⇒ 可以在 `inference_mode` 之外参与比较/哈希（见 `_normal_clone`）。
    """
    with torch.inference_mode(False):
        t = lg[0] if lg.dim() == 3 else lg
        return t[-1 if index is None else index].detach().clone()


def layout(t) -> dict:
    """张量的形状/步长/dtype/设备/是否连续/是否为 view（对外公开，供工具记录）。"""
    return _layout(t)


def project(h, W, *, take_row: int | None = None, phase: str | None = None) -> tuple:
    """按**生产表达式** `h @ W.T` 投影（bf16、`inference_mode`），返回 `(完整 logits, 指定行)`。

    * 输入 `h`：`(1,T,H)` 或 `(1,H)`；`W`：`(V,H)`（即 `model.embed_tokens.weight`）。
    * **保留完整词表投影**：不做任何"只取 323/659 两行"的裁剪（那会改变 GEMM 形状）。
    * ⚠️ 行口径（既有工具踩过两次的坑）：
      `(1,1,V)[0,0]` 是**一维行** ✓；但 `(1,V)[0,0]` 是**标量** ✗ ⇒ 两句都写 `[0,0]` 会得到
      "有时是行、有时是标量"的隐患。本函数按维度显式取行，并 `require` 结果是**一维、长度 = V**。
    """
    if h.dim() not in (2, 3):
        raise BoundaryError(f"h 维度应为 2 或 3，得到 {list(h.shape)}")
    if not (W.dim() == 2 and h.shape[-1] == W.shape[1]):
        raise BoundaryError(f"形状不匹配：h={list(h.shape)} W={list(W.shape)}")
    ctx = M.attention_phase(phase) if phase else contextlib.nullcontext()
    with torch.inference_mode(), ctx:
        lg = h @ W.T                       # ★ 与 model_self._run 完全同一表达式与 dtype
    V = int(W.shape[0])
    if lg.dim() == 2:                      # (1,H) @ (H,V) -> (1,V)
        if lg.shape[0] != 1:
            raise BoundaryError(f"(1,V) 形状异常：{list(lg.shape)}")
        row, full = lg[0], lg
    elif lg.dim() == 3:
        if take_row is None:
            raise BoundaryError("三维输入必须显式给 take_row")
        row, full = lg[0, take_row], lg
    else:
        raise BoundaryError(f"logits 维度异常：{list(lg.shape)}")
    if row.dim() != 1 or int(row.numel()) != V:
        raise BoundaryError(f"取到的行应为 (V,)={V}，得到 {list(row.shape)}")
    return full, row


# ------------------------------------------------------------------ CPU 自测（门）
def _selftest() -> int:
    """**不加载大模型**：核对捕获点、只读性、钩子移除、关闭等价、投影取行口径。"""
    np_ = nf = 0

    def check(name, cond, extra=None):
        nonlocal np_, nf
        if cond:
            np_ += 1
            print(f"  [ok] {name}")
        else:
            nf += 1
            print(f"  [FAIL] {name}" + (f"  {str(extra)[:220]}" if extra is not None else ""))

    torch.manual_seed(0)
    m = M.QwenLM(vocab_size=64, hidden=32, heads=4, kv_heads=2, head_dim=8, intermediate=64,
                 num_layers=2, rope_theta=10000.0, rms_eps=1e-6, attn_bias=True
                 ).to(torch.bfloat16).eval()
    W = m.embed_tokens.weight
    facts = code_facts(m)
    print("== 1) 边界事实（源码级）==")
    print(f"  {facts['projection_expression']}")
    check("★ 生产表达式确实是 `x @ self.embed_tokens.weight.T`（源码内出现）",
          facts["source_has_expression"] and facts["source_has_norm_call"], facts)
    check("★ 模型**没有**独立 `lm_head` 模块（输出权重 = `embed_tokens.weight`，绑定权重）",
          facts["has_lm_head_module"] is False and facts["has_norm"] is True, facts)

    ids = torch.randint(0, 64, (1, 5))
    x = m.embed_tokens(ids)
    with torch.inference_mode(), record(m, tag="t") as info:
        lg, cache = m._run(x, None, need_logits=True)
        got = take()
    print("== 2) 捕获点：最终 RMSNorm 之后、权重乘法之前 ==")
    check("★ 每次前向恰好捕获 1 次，且安装了 1 个只读钩子",
          got["n_calls"] == 1 and info["installed"] and len(got["caps"]) == 1, got.get("n_calls"))
    c = got["caps"][1]
    y, x_in = c["y_out"], c["x_in"]
    check("★ 捕获形状/dtype 正确：`(1,T,hidden)`、bf16",
          list(y.shape) == [1, 5, 32] and y.dtype == torch.bfloat16, c["y_out_layout"])
    check("★ 捕获的 y 就是 `self.norm` 的输出（`norm(x_in) == y` 逐位）",
          bool(torch.equal(m.norm(x_in), y)))
    check("★ ★ 捕获点确实在权重乘法**之前**：`y @ W.T` 与 `_run` 返回的 logits **逐位相同**",
          bool(torch.equal(y @ W.T, lg)))
    check("★ norm 确实做了归一化（输入与输出不同，别把 pre-norm 当 post-norm）",
          not bool(torch.equal(x_in, y)))
    check("★ 捕获张量是**克隆**（与钩子内部张量不共享存储）",
          bool(y.data_ptr() != x_in.data_ptr()) and bool(torch.equal(y, y.clone())))
    check("★ 捕获张量是**普通张量**（不是 inference tensor）⇒ 可在 `inference_mode` 之外直接使用",
          (not y.is_inference()) and (not x_in.is_inference())
          and bool(torch.isfinite(y @ W.T).all()), (y.is_inference(), x_in.is_inference()))

    print("== 3) 钩子移除 / 异常恢复 / 关闭等价 ==")
    check("★ `record` 退出后钩子集合回到基线（逐 id）", bool(list(m.norm._forward_hooks.keys()) == []))
    try:
        with record(m, tag="boom"):
            raise RuntimeError("注入的前向故障")
    except RuntimeError:
        pass
    check("★ with 体内抛异常后钩子仍被移除（`finally`）",
          bool(list(m.norm._forward_hooks.keys()) == []))
    with torch.inference_mode():
        lg_plain, _ = m._run(x, None, need_logits=True)
    off = install(m, enabled=False, tag="off")
    try:
        with torch.inference_mode():
            lg_off, _ = m._run(x, None, need_logits=True)
    finally:
        assert_clean()
    check("★ **关闭诊断**（`enabled=False`，一个钩子都不挂）时与未安装**逐位等价**",
          off["installed"] is False and bool(torch.equal(lg_plain, lg_off)))

    print("== 4) 投影取行口径（既有工具踩过的 `[0,0]` 陷阱）==")
    h3 = y[:, -1:, :]                                  # (1,1,32) ← 引擎 S 路径的真实形状
    _full1, row1 = project(h3, W, take_row=0)
    _full2, row2 = project(y[:, -1, :], W)             # 同一行，但走 (1,H) 二维形式
    h4 = torch.cat([y[:, 0:1, :], y[:, 1:4, :]], dim=1)   # (1,4,32) ← 引擎 B 路径的真实形状
    full4, row4 = project(h4, W, take_row=0)
    check("★ 三维 (1,1,H) 与二维 (1,H) 两条路径都返回**一维、长度=V**的行",
          row1.dim() == 1 and row1.numel() == 64 and row2.dim() == 1 and row2.numel() == 64
          and row4.dim() == 1 and row4.numel() == 64, (list(row1.shape), list(row2.shape)))
    check("★ 保留**完整词表**投影（不被裁成两行）",
          full4.shape[-1] == 64 and int(W.shape[0]) == 64)
    check("★ 行口径一致：四行投影的第 0 行 == `full4[0, 0]`",
          bool(torch.equal(row4, full4[0, 0])) and list(full4.shape) == [1, 4, 64])
    # ⚠️ 这一条**只记录、不断言**相等：`(1,1,H)@W.T`（三维批量）与 `(1,H)@W.T`（二维）
    #    是**两种调用形状**，可能落到不同 kernel ⇒ 数值上是否逐位相同本身就是本轮的观察对象之一。
    #    引擎两条真实路径用的都是**三维**形式（`_run` 里 x 一定是 (1,T,H)），因此下面的格一律用三维。
    same_2d_3d = bool(torch.equal(row1, row2))
    print(f"  [obs] 三维 (1,1,H) 与二维 (1,H) 的同一条投影：逐位相同={same_2d_3d}")
    check("★ 二维/三维两种写法都必须给出**同一行的可行结果**（有限、argmax 一致即可）",
          bool(torch.isfinite(row1).all()) and bool(torch.isfinite(row2).all())
          and int(row1.argmax()) == int(row2.argmax()))
    bad = 0
    try:
        project(h4, W)                                  # 三维却不给 take_row ⇒ 必须报错
    except BoundaryError:
        bad += 1
    check("★ 三维输入不给 `take_row` 时**报错**（而不是猜最后一行）", bad == 1)
    print("== 5) Norm 边界原语（`norm_info` / `norm_forward`）==")
    ni = norm_info(m)
    check("★ `norm_info` 记录实际 weight/eps/dtype/布局，并写明**逐行**与 fp32 内部归约",
          ni["module_class"] == "RMSNORM" and ni["weight"]["dtype"] == "torch.bfloat16"
          and ni["weight"]["shape"] == [32] and ni["eps"] is not None
          and "dim=-1" in ni["code_path"], ni)
    y2, meta2 = norm_forward(m, y)                       # ★ 用**原始 bf16** 捕获张量
    check("★ `norm_forward` 用现有模块复算 ⇒ 与捕获的 y **逐位相同**（重放忠实）",
          bool(torch.equal(y2, y)), meta2)
    check("★ `norm_forward` 保持原始 dtype（输出 bf16）并记录输入/输出 layout",
          meta2["dtype_preserved"] and y2.dtype == torch.bfloat16
          and meta2["output"]["shape"] == [1, 5, 32], meta2)
    bad = 0
    try:
        norm_forward(m, y.float())                       # fp32 统计副本 ⇒ 必须报错
    except BoundaryError:
        bad += 1
    check("★ 传入 fp32（统计副本）时**报错**（不允许悄悄换 dtype 做 Norm）", bad == 1)
    y4, _m4 = norm_forward(m, torch.cat([y[:, 0:1, :], y[:, 1:4, :]], dim=1))
    row_wise_obs = bool(torch.equal(y4[0, 0], y[0, 0]))
    print(f"  [obs] 逐行性（CPU 小模型）：4 行与 1 行两种形状下第 0 行输出逐位相同={row_wise_obs}")
    check("★ `norm_forward` 对多行输入形状保持 (1,4,32) 且有限",
          list(y4.shape) == [1, 4, 32] and bool(torch.isfinite(y4).all()))
    print("== 6) 层边界原语（`layer_info` / `layer_capture` / `layer_forward` / `embedding_capture`）==")
    M.set_attention_strategy("eager")
    Llast = len(m.layers) - 1
    li = layer_info(m, Llast)
    check("★ `layer_info` 认出最后一层与真实子模块 dtype/形状",
          li["is_last"] is True and li["n_layers"] == len(m.layers)
          and li["submodules"]["self_attn"]["q_proj_dtype"] == "torch.bfloat16", li)
    # 捕获位置：同一前向内挂「L0 + L1(=最后一层) + 最终 norm」三个只读钩子
    with layer_capture(m, 0, tag="L0") as c0, layer_capture(m, Llast, tag="Llast") as cL, \
            record(m, tag="norm") as _ni:
        with torch.inference_mode():
            lg3, cache3 = m._run(x, None, need_logits=True)
        norm_cap = take()
    n_in = norm_cap["caps"][1]["x_in"]
    check("★ 捕获链逐位成立：L0 输出 == 最后一层输入；最后一层输出 == 最终 Norm 的输入",
          bool(torch.equal(c0["post"]["z"], cL["pre"]["a"]))
          and bool(torch.equal(cL["post"]["z"], n_in)), None)
    check("★ `layer_forward` 的元数据：phase / 阶段分流 / s,T,p / 掩码可见范围 / 缓存长度都有",
          (lambda meta: meta["phase"] == "decode" and meta["dispatch"]["path"] == "eager"
           and meta["s"] == 1 and meta["T"] == 1 and meta["cache_len_before"] is None
           and "triu" in meta["mask_rule"] and meta["visible_last_key_row0"] == 0)(_meta_probe(m, x)),
          _meta_probe(m, x))
    check("★ 钩子在 with 退出后已移除（pre/post 两层都回基线）",
          len(m.layers[Llast]._forward_pre_hooks) == 0 and len(m.layers[Llast]._forward_hooks) == 0)
    try:
        with layer_capture(m, Llast, tag="boom"):
            raise RuntimeError("注入的层捕获故障")
    except RuntimeError:
        pass
    check("★ with 体内抛异常后层钩子仍被移除",
          len(m.layers[Llast]._forward_pre_hooks) == 0 and len(m.layers[Llast]._forward_hooks) == 0)
    # 轮前缓存：必须取**入口**状态（post 里的缓存已被就地追加过）
    x2 = m.embed_tokens(torch.tensor([[7]], device=ids.device))
    with layer_capture(m, Llast, tag="c") as cc:
        with torch.inference_mode():
            _lg4, cache4 = m._run(x2, cache3, need_logits=True)
    pre = cc["pre"]["cache_pre"]
    # ⚠️ 口径：层收到/返回的是**单层 cache dict**（`{"k","v"}`），而 `_run` 返回的是 `list[dict]`
    #    ⇒ 这里 pre 用 `pre["k"]`、模型级用 `cache3[Llast]["k"]`（首版写成 `pre[0]["k"]` 即 KeyError: 0）。
    _pre_len = _cache_len_of(pre)
    check("★ 轮前缓存取自 pre 钩子且**未被追加**：长度 = 入口长度，后长度 = 入口+1",
          _pre_len == cc["pre"]["cache_len_before"]
          and cc["post"]["cache_len_after"] == cc["pre"]["cache_len_before"] + 1
          and bool(torch.equal(pre["k"], cache3[Llast]["k"][:, :, :_pre_len]))
          and cc["post"]["same_cache_object"] is True,
          (cc["pre"]["cache_len_before"], cc["post"]["cache_len_after"], _pre_len))
    with embedding_capture(m, tag="e") as ec:
        with torch.inference_mode():
            _ = m.embed_tokens(ids)
    check("★ `embedding_capture` 记录输入 token id 与输出 layout",
          ec["n_calls"] == 1 and ec["ids"] == [int(v) for v in ids.reshape(-1).tolist()]
          and ec["out_layout"]["shape"] == [1, 5, 32], ec.get("ids"))
    print("== 7) 全层链式捕获（`layer_chain_capture`）==")
    from . import spec_dec as _SD                       # 自测用：真实单步（会经过 embed_tokens）
    _tok = int(ids[0, 0])
    with torch.inference_mode():
        _lg0, _seed = m._run(m.embed_tokens(torch.randint(0, 64, (1, 3))), None, need_logits=True)
    _seed_clone = [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in d.items()} for d in _seed]
    with embedding_capture(m, tag="chain_e") as ce, \
            layer_chain_capture(m, row_index=0, tag="chain1") as ch:
        with torch.inference_mode(), M.attention_phase("decode"):
            _SD._step(m, _tok, _seed_clone)
    c1 = to_cpu_rows(ch)
    check("★ 每层 pre/post 各触发 1 次；行形状 (hidden,)、原始 dtype、已搬到 CPU",
          c1["n_pre"] == len(m.layers) and c1["n_post"] == len(m.layers)
          and all(tuple(c1["layers"][i]["pre"]["a_row"].shape) == (32,) for i in c1["layers"])
          and all(str(c1["layers"][i]["pre"]["a_row"].dtype) == "torch.bfloat16"
                  for i in c1["layers"]))
    check("★ 边界链自洽：L0 输入行 == embedding 行；layer[i] 输出行 == layer[i+1] 输入行（逐位）",
          bool(torch.equal(c1["layers"][0]["pre"]["a_row"].to("cpu"), ce["out"][0, 0].detach().to("cpu")))
          and all(bool(torch.equal(c1["layers"][i]["post"]["z_row"],
                                   c1["layers"][i + 1]["pre"]["a_row"]))
                  for i in range(len(m.layers) - 1)))
    check("★ 钩子在 with 退出后全部移除",
          all(len(m.layers[i]._forward_pre_hooks) == 0 and len(m.layers[i]._forward_hooks) == 0
              for i in range(len(m.layers))))
    try:
        with layer_chain_capture(m, row_index=0, tag="boom"):
            raise RuntimeError("注入的链式捕获故障")
    except RuntimeError:
        pass
    check("★ 链式捕获 with 体内抛异常后钩子也全部移除",
          all(len(m.layers[i]._forward_pre_hooks) == 0 and len(m.layers[i]._forward_hooks) == 0
              for i in range(len(m.layers))))
    print(f"\nLM head 边界捕获（CPU 门）：PASS {np_} / FAIL {nf}")
    return 0 if nf == 0 else 1


def _meta_probe(model, x) -> dict:
    """自测用：在**无缓存单步**下调用 `layer_forward`，返回其 meta（只读，不参与分析）。"""
    L = len(model.layers) - 1
    cos, sin = M.Build_Rope_cache(model.head_dim, int(x.shape[1]), model.rope_theta,
                                  x.device, x.dtype, 0)
    with M.attention_phase("decode"):
        _z, _c, meta = layer_forward(model, L, x[:, :1, :], cos[:, :, :1, :], sin[:, :, :1, :], None)
    return meta


if __name__ == "__main__":
    raise SystemExit(_selftest())
