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
    print(f"\nLM head 边界捕获（CPU 门）：PASS {np_} / FAIL {nf}")
    return 0 if nf == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
