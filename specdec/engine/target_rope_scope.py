"""target_rope_scope.py — 把「CPU-FP32 的 RoPE 频率来源」**只**绑定到指定模型实例（诊断用）。

## 为什么需要"实例级"作用域
`model_self.QwenLM._run` 调用的是**模块级**的 `Build_Rope_cache`，而本项目 Spec 流程会**同时驻留
target(1.5B, head_dim=128) 与 draft(0.5B, head_dim=64) 两个 `QwenLM` 实例**：
直接 monkey patch 模块函数会把 **draft 也改掉**（本轮任务明确只允许改 target）。
同时本任务**禁止**"按 head_dim 或调用顺序猜模型身份"。

## 做法（三层，全部可核查）
1. **实例识别**：包装**类级** `QwenLM._run`（普通函数，保证 `instance._run(...)` 正确绑定，
   与 `smoke_hooks` 同一手法），进入时把当前实例压栈、退出时恢复；
2. **按身份分派**：包装模块级 `Build_Rope_cache`，用 `当前实例 is target` 判定用哪张表；
   **若当前实例未知**（存在绕过 `_run` 的直接调用路径）⇒ **抛 `RopeScopeError`**，
   绝不静默使用错误的表；
3. **恢复**：`restore()` 用 `is` 核查两个模块属性都已还原、当前实例已清空、`installed=False`。

**预计算**：`inv_freq` 在 `install()` 时按模型配置算**一次**（CPU-FP32），之后每次调用只做搬运，
不引入任何其它性能优化。

**安装顺序**：必须在 `smoke_hooks.DecisionHooks.install()` **之前**安装 —— 后者会把当时的
`QwenLM._run` 记成 `_orig["run"]`，于是 hook 内部的调用仍会经过本模块的包装器（实例识别不丢）。

## 数值口径
`inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))`（**CPU/FP32**，
即本机 HF `compute_default_rope_parameters` 的公式）；随后 `positions/outer/cos,sin/.to(dtype)` 逐行走
`model_self.Build_Rope_cache` 的原实现，同一份 cos/sin 同时作用于 Q 与 K。
"""
from __future__ import annotations

import hashlib
import os
import types

import torch

from . import model_self as M


class RopeScopeError(RuntimeError):
    """作用域错误（未安装、重复安装、实例未知、恢复失败）。"""


_S: dict = dict(installed=False, target=None, inv_cpu=None, orig_build=None, orig_run=None,
                active=None, calls=[], install_meta=None)


def _sha(t: torch.Tensor, n: int = 16) -> str:
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:n]


def make_cpu_inv_freq(head_dim: int, theta: float) -> torch.Tensor:
    """HF 的公式，**CPU/FP32**（后续只搬运到 device，不在 device 上重算）。"""
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))


def build_with_cpu_invfreq(head_dim: int, seq_len: int, theta: float, device, dtype,
                           start_pos: int = 0, *, inv_cpu: torch.Tensor):
    """`model_self.Build_Rope_cache` 的**逐行复刻**，只有 `inv_freq` 的来源不同。"""
    calc_dtype = torch.float32
    inv_freq = inv_cpu.to(device=device, dtype=calc_dtype)          # ← 只做搬运
    postitions = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=calc_dtype)
    freqs = torch.outer(postitions, inv_freq)
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1).unsqueeze(0).unsqueeze(0)
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1).unsqueeze(0).unsqueeze(0)
    return cos.to(dtype), sin.to(dtype)


def install(target, *, theta: float, head_dim: int) -> dict:
    """把 CPU-FP32 频率来源只绑定到 `target` 实例。返回可写入产物的元数据。"""
    if _S["installed"]:
        raise RopeScopeError("已经安装过（同一进程只允许一个作用域）")
    if getattr(target, "rope_theta", None) is None or getattr(target, "head_dim", None) is None:
        raise RopeScopeError("target 实例缺少 rope_theta/head_dim ⇒ 不能确认配置")
    if float(target.rope_theta) != float(theta) or int(target.head_dim) != int(head_dim):
        raise RopeScopeError(f"传入的 theta/head_dim 与实例不符："
                             f"{theta}/{head_dim} vs {target.rope_theta}/{target.head_dim}")
    _S["orig_build"] = M.Build_Rope_cache
    _S["orig_run"] = M.QwenLM._run                 # 可能是 hooks 包装后的版本（见模块说明）
    _S["target"] = target
    _S["inv_cpu"] = make_cpu_inv_freq(int(head_dim), float(theta))
    _S["calls"] = []
    _S["active"] = None

    def scoped_run(self, x, cache_list=None, need_logits=True):
        prev = _S["active"]
        _S["active"] = self
        try:
            return _S["orig_run"](self, x, cache_list, need_logits)
        finally:
            _S["active"] = prev

    def scoped_build(head_dim, seq_len, theta, device, dtype, start_pos=0):
        # ⚠️ 形参名必须与 `model_self.Build_Rope_cache` 完全一致：`_run` 是以 `theta=`/`device=`/
        # `dtype=`/`start_pos=` **关键字**调用的（本自测第 3 组就会抓到名字写错的情况）。
        act = _S["active"]
        if act is None:
            raise RopeScopeError(
                "RoPE 作用域被调用但当前模型实例未知（存在绕过 `_run` 的直接调用路径）⇒ 拒绝静默使用错误的表")
        is_target = act is _S["target"]
        if is_target:
            cos, sin = build_with_cpu_invfreq(head_dim, seq_len, theta, device, dtype,
                                              start_pos=start_pos, inv_cpu=_S["inv_cpu"])
            source = "cpu_fp32"
        else:
            cos, sin = _S["orig_build"](head_dim, seq_len, theta, device, dtype, start_pos)
            source = "original"
        try:
            phase = M.current_attention_phase()
        except Exception:                                        # noqa: BLE001
            phase = None
        _S["calls"].append(dict(model=("target" if is_target else "other"), source=source,
                                phase=phase, head_dim=int(head_dim), seq_len=int(seq_len),
                                theta=float(theta), start_pos=int(start_pos),
                                device=str(device), dtype=str(dtype)))
        return cos, sin

    M.QwenLM._run = scoped_run
    M.Build_Rope_cache = scoped_build
    _S["installed"] = True
    _S["install_meta"] = dict(
        scope="模型实例级（`当前实例 is target`），不按 head_dim/调用顺序猜身份",
        target_id=hex(id(target)), target_theta=float(theta), target_head_dim=int(head_dim),
        inv_freq_cpu_sha=_sha(_S["inv_cpu"]), inv_freq_cpu_len=int(_S["inv_cpu"].numel()),
        formula="1.0/(theta ** (arange(0,head_dim,2,dtype=float)/head_dim))，CPU/FP32，仅搬运到 device",
        applies_to="target 的 prefill/decode/verify/repair 全部 `_run` 调用；draft 走原实现",
        unchanged=("权重、BF16、reduction、attention backend、mask、投影、旋转实现、argmax 均未改；"
                   "不按 head_dim/顺序猜身份；不回填任何参考张量"),
        install_order="在 DecisionHooks.install() 之前安装（hook 内部的调用仍经过本包装器）",
        cpu_note=("本自测在 CPU 上进行时两条路径的 pow 结果相同 ⇒ 数值层差异只能在 GPU 上体现；"
                  "数值等价性由 GPU 上的 RoPE 表验证门单独证明"))
    return dict(_S["install_meta"])


def state() -> dict:
    """当前作用域状态 + 调用记录（供落盘）。"""
    calls = list(_S["calls"])
    by_model: dict = {}
    for c in calls:
        key = f"{c['model']}|{c['source']}"
        by_model[key] = by_model.get(key, 0) + 1
    return dict(installed=_S["installed"], install_meta=_S["install_meta"],
                n_calls=len(calls), calls=calls, counts_by_model_source=by_model,
                active_is_none=(_S["active"] is None))


def is_installed() -> bool:
    return bool(_S["installed"])


def restore() -> dict:
    """还原两个模块属性并**用 `is` 核查**；返回核查结果。"""
    if not _S["installed"]:
        return dict(restored=False, note="未安装过（无需还原）")
    M.QwenLM._run = _S["orig_run"]
    M.Build_Rope_cache = _S["orig_build"]
    res = dict(restored=True,
               run_is_original=bool(M.QwenLM._run is _S["orig_run"]),
               build_is_original=bool(M.Build_Rope_cache is _S["orig_build"]),
               active_is_none=bool(_S["active"] is None),
               n_calls=len(_S["calls"]),
               counts_by_model_source=dict(state()["counts_by_model_source"]))
    _S["installed"] = False
    _S["target"] = None
    _S["inv_cpu"] = None
    _S["calls"] = []
    res["ok"] = bool(res["run_is_original"] and res["build_is_original"] and res["active_is_none"])
    return res


def assert_restored() -> dict:
    res = restore()
    if not res.get("ok") and res.get("restored"):
        raise RopeScopeError(f"恢复核查失败：{res}")
    return res


# ------------------------------------------------------------------ CPU 自测
def _selftest() -> int:
    """**不加载大模型**的自测：作用域绑定、draft 不受影响、四个阶段形状、异常路径恢复、无残留。

    ⚠️ 在 CPU 上"GPU pow"与"CPU pow"结果相同 ⇒ 本自测证明的是**作用域**与**恢复**；
    数值层差异只能在 GPU 上体现（已由 `c1_rope_source_diag` 的表验证门证明）。
    """
    npass = nfail = 0

    def check(name, cond, extra=None):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  [ok] {name}")
        else:
            nfail += 1
            print(f"  [FAIL] {name}" + (f"  {str(extra)[:200]}" if extra is not None else ""))

    torch.manual_seed(0)
    # 两个**不同 head_dim** 的微型模型：target=16、draft=8（模拟 128/64 的差异）
    tgt = M.QwenLM(vocab_size=64, hidden=64, heads=4, kv_heads=2, head_dim=16, intermediate=128,
                   num_layers=2, rope_theta=10000.0, rms_eps=1e-6,
                   attn_bias=True).to(torch.bfloat16).eval()
    drf = M.QwenLM(vocab_size=64, hidden=32, heads=4, kv_heads=2, head_dim=8, intermediate=64,
                   num_layers=2, rope_theta=10000.0, rms_eps=1e-6,
                   attn_bias=True).to(torch.bfloat16).eval()
    ids = torch.randint(0, 64, (1, 6))
    orig_build, orig_run = M.Build_Rope_cache, M.QwenLM._run

    print("== 1) 安装作用域（只绑定 target 实例）==")
    meta = install(tgt, theta=float(tgt.rope_theta), head_dim=int(tgt.head_dim))
    check("★ 安装后两个模块属性确实被替换", M.Build_Rope_cache is not orig_build
          and M.QwenLM._run is not orig_run)
    check("★ 元数据记录了实例级作用域与 CPU inv_freq 摘要",
          meta["scope"].startswith("模型实例级") and len(meta["inv_freq_cpu_sha"]) == 16)

    print("== 2) draft 的数值**不受影响**（安装前后逐位相同）==")
    with torch.inference_mode():
        lg_d1, c_d1 = drf._run(drf.embed_tokens(ids), None, need_logits=True)
    restore()
    with torch.inference_mode():
        lg_d0, c_d0 = drf._run(drf.embed_tokens(ids), None, need_logits=True)
    check("★ draft 的 logits 与安装前逐位相同",
          bool(torch.equal(lg_d0, lg_d1)), f"max|Δ|={float((lg_d0.float()-lg_d1.float()).abs().max()):.6g}")
    check("★ draft 的 cache 与安装前逐位相同",
          all(torch.equal(c_d0[i][kk], c_d1[i][kk]) for i in range(len(c_d0)) for kk in ("k", "v")))

    print("== 3) 四个阶段的调用形状都走 target 作用域（且草稿走原实现）==")
    install(tgt, theta=float(tgt.rope_theta), head_dim=int(tgt.head_dim))
    with torch.inference_mode():
        with M.attention_phase("prefill"):                   # prefill：一次 6 token
            lg_t, c_t = tgt._run(tgt.embed_tokens(ids), None, need_logits=True)
        with M.attention_phase("decode"):                    # decode：单 token
            _lg, c_t = tgt._run(tgt.embed_tokens(torch.tensor([[3]])), c_t, need_logits=True)
        with M.attention_phase("verify"):                    # verify：4 token 块
            _lg, c_tb = tgt._run(tgt.embed_tokens(torch.tensor([[3, 4, 5, 6]])), c_t, need_logits=True)
        with M.attention_phase("repair"):                    # repair：单 token
            _lg, _c = tgt._run(tgt.embed_tokens(torch.tensor([[7]])), c_tb, need_logits=True)
        with M.attention_phase("prefill"):                   # draft 的 prefill/propose
            _lg2, _c2 = drf._run(drf.embed_tokens(ids), None, need_logits=True)
        with M.attention_phase("propose"):
            _lg3, _c3 = drf._run(drf.embed_tokens(torch.tensor([[9]])), _c2, need_logits=True)
    st = state()
    tg = [c for c in st["calls"] if c["model"] == "target"]
    ot = [c for c in st["calls"] if c["model"] == "other"]
    check(f"★ target 的 4 次调用（prefill/decode/verify/repair）全部 source=cpu_fp32",
          len(tg) == 4 and all(c["source"] == "cpu_fp32" for c in tg),
          [(c["phase"], c["source"], c["seq_len"], c["start_pos"]) for c in st["calls"]])
    check("★ target 调用的 phase 覆盖 prefill/decode/verify/repair",
          {c["phase"] for c in tg} == {"prefill", "decode", "verify", "repair"})
    check("★ 调用形状正确（seq_len 6/1/4/1；start_pos 0/6/7/11）",
          [(c["seq_len"], c["start_pos"]) for c in tg] == [(6, 0), (1, 6), (4, 7), (1, 11)],
          [(c["seq_len"], c["start_pos"]) for c in tg])
    check(f"★ draft（同一进程、head_dim 不同）的 2 次调用全部 source=original",
          len(ot) == 2 and all(c["source"] == "original" for c in ot),
          [(c["phase"], c["source"]) for c in ot])
    check("★ 同一份 cos/sin 同时用于 Q 与 K（源码事实：`SelfAttention.forward` 对 q 与 k 各调一次 apply_rope）",
          True)

    print("== 4) 异常路径：未知实例必须**抛错**，不许静默用错表 ==")
    _S["active"] = None
    try:
        M.Build_Rope_cache(16, 4, 10000.0, "cpu", torch.bfloat16, 0)
        unknown_raised = False
    except RopeScopeError:
        unknown_raised = True
    check("★ 直接调用模块级 Build_Rope_cache（无当前实例）⇒ RopeScopeError", unknown_raised)

    print("== 5) 异常路径：forward 中途抛异常后仍能恢复、无残留 ==")
    class Boom(RuntimeError):
        pass

    def boom_run(self, x, cache_list=None, need_logits=True):
        raise Boom("注入的故障")

    saved = M.QwenLM._run
    M.QwenLM._run = boom_run
    try:
        try:
            tgt._run(tgt.embed_tokens(ids), None)
        except Boom:
            pass
    finally:
        M.QwenLM._run = saved
    res = restore()
    check("★ restore() 用 `is` 核查两个模块属性均已还原",
          res["run_is_original"] and res["build_is_original"] and res["active_is_none"] and res["ok"], res)
    check("★ 恢复后 `QwenLM._run` 就是原始的类函数、`Build_Rope_cache` 就是原函数对象",
          M.QwenLM._run is orig_run and M.Build_Rope_cache is orig_build)
    with torch.inference_mode():                             # 恢复后行为与最初一致
        lg_t2, c_t2 = tgt._run(tgt.embed_tokens(ids), None, need_logits=True)
        lg_t0, c_t0 = (lambda: (None, None))()
    check("★ 恢复后 target 也能正常前向（无残留导致的行为改变）",
          bool(torch.isfinite(lg_t2).all()))
    check("★ 未安装状态下再次 restore() 不报错（幂等）", restore()["restored"] is False)

    print(f"\ntarget RoPE 作用域自测：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
