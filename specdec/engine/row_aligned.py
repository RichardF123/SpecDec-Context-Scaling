"""row_aligned.py — **全层 row-aligned verify 候选模式**（默认关闭；只作用于 target 的 block verify）。

## 候选模式定义（与项目冻结的规格逐项一致）
对 `target` 的**全部 28 层**：
* **Attention QK**：每个 query 行分别以 **M=1** + **真实 causal key 前缀 `p+i+1`** 计算，
  用**同一个** `torch.matmul`、**同一个** BF16 dtype、**同一个**缩放，然后拼回 block scores，
  继续走**原有** mask / softmax / AV；
* **Linear**：`o_proj`、`gate_proj`、`up_proj`、`down_proj` 四个 Linear 逐行以 **M=1**
  调用**原来的同一个模块**，再按 token 行拼回。

**不对齐**：`q_proj`、`k_proj`、`v_proj`、RoPE、softmax、AV、RMSNorm、SiLU、gated product、
residual、cache、argmax。**不**改成显式 matmul/einsum/FP32/手写 kernel。

## 作用范围与安全属性
* **默认关闭**：不进入上下文时 `model_self` 的两个注册表都是 `None` ⇒ 生产路径**一字不变**；
* **只作用于 target**：注册的是**该模型实例**的模块对象（按 `id` + `is` 身份匹配，**不**按类名/名称匹配）；
* **draft 永不启用**：draft 的模块从不出现在注册表里 ⇒ 它进入分支只会被计数、**永不被替换**；
* **只在 target block verify 阶段使用**：由 `spec_dec.set_verify_wrap(row_aligned_verify)` 安装在
  `spec_dec._verify(...)` 的调用点（AR/decode 路径不经过那里）；
* **AR/decode 的 s=1 与普通路径逐位一致**：`serial_*` 在 s=1 时就是"整块一次调用"，
  由本模块的 CPU 门与 Stage 1 的 `SALL == S0` 逐位门共同证明；
* **正常/异常退出都恢复**：`contextlib.contextmanager` + `finally`。

## 预期注册数量必须**从模型结构推导**
`qk = len(model.layers)`、`linear = 4 × len(model.layers)`、`total = 5 × len(model.layers)`。
本仓库 target 是 28 层 ⇒ 28 / 112 / 140（由 `expected_counts()` 现算，**不写死**）。
"""
from __future__ import annotations

import contextlib

import torch

from . import diag_checks as DC
from . import model_self as M


# ------------------------------------------------------------------ 模块集合（按对象身份）
def all_qk_modules(model) -> tuple:
    """每层的 `self_attn`（QK 的宿主模块）。"""
    return tuple(layer.self_attn for layer in model.layers)


def all_linear_modules(model) -> tuple:
    """每层的 `o_proj` / `gate_proj` / `up_proj` / `down_proj`（顺序固定：层 → 四个投影）。"""
    mods = []
    for layer in model.layers:
        mods += [layer.self_attn.o_proj, layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj]
    return tuple(mods)


def expected_counts(model) -> dict:
    """**从模型结构推导**的预期数量（不写死数字）。"""
    n = len(model.layers)
    return dict(n_layers=int(n), qk=int(n), linear=int(4 * n), total=int(5 * n),
                note="qk = 层数；linear = 4×层数（o_proj/gate_proj/up_proj/down_proj）；total = 5×层数")


def model_module_ids(model) -> dict:
    """该模型自己的 QK/Linear 模块 `id` 集合（供"只作用于 target"核对）。"""
    return dict(qk=[int(id(m)) for m in all_qk_modules(model)],
                linear=[int(id(m)) for m in all_linear_modules(model)])


def coverage_check(target, other=None, *, registered=None) -> dict:
    """注册集合是否**恰好**覆盖 target 的 QK+Linear，且**完全不重叠** `other`（draft）。"""
    tgt_ids = model_module_ids(target)
    want = set(tgt_ids["qk"]) | set(tgt_ids["linear"])
    reg = registered if registered is not None else want
    out = dict(target_qk=len(tgt_ids["qk"]), target_linear=len(tgt_ids["linear"]),
               target_total=len(want), registered_total=len(reg),
               covers_all_target=bool(reg == want),
               missing=sorted(want - set(reg))[:8], extra=sorted(set(reg) - want)[:8])
    if other is not None:
        oth = model_module_ids(other)
        out["other_total"] = len(set(oth["qk"]) | set(oth["linear"]))
        out["overlap_with_other"] = len(reg & (set(oth["qk"]) | set(oth["linear"])))
        out["other_untouched"] = bool(out["overlap_with_other"] == 0)
    return out


# ------------------------------------------------------------------ 候选模式上下文
_VERIFY_LOG: list = []


def clear_verify_log() -> dict:
    global _VERIFY_LOG
    n = len(_VERIFY_LOG)
    _VERIFY_LOG = []
    return dict(cleared=n)


def verify_log() -> list:
    """每次（verify 调用内的）模块级干预结束后累积的记录摘要。"""
    return list(_VERIFY_LOG)


@contextlib.contextmanager
def row_aligned_verify(target, *, log: bool = True):
    """在 `with` 体内：target 全部层的 QK + 4 个 Linear 走逐行 M=1；退出（含异常）必定恢复。

    只应在 **target block verify** 的调用点使用（见 `spec_dec.set_verify_wrap`）。
    """
    qk_mods, lin_mods = all_qk_modules(target), all_linear_modules(target)
    exp = expected_counts(target)
    DC.require(len(qk_mods) == exp["qk"] and len(lin_mods) == exp["linear"],
               "模块集合与模型结构不一致", f"{len(qk_mods)}/{len(lin_mods)} vs {exp}")
    with M.serial_qk_for(*qk_mods, label="row_aligned") as qk_info, \
            M.serial_linear_for(*lin_mods, label="row_aligned") as lin_info:
        try:
            yield dict(expected=exp, qk_mods=len(qk_mods), linear_mods=len(lin_mods),
                       qk_installed=qk_info.get("n_mods"), linear_installed=lin_info.get("n_mods"))
        finally:
            if log:
                st_qk, st_lin = M.serial_qk_state(), M.serial_linear_state()
                _VERIFY_LOG.append(dict(
                    qk_applied=int(st_qk.get("n_applied", -1)), qk_seen=int(st_qk.get("n_seen", -1)),
                    lin_applied=int(st_lin.get("n_applied", -1)), lin_seen=int(st_lin.get("n_seen", -1)),
                    qk_applied_ids=[a["module_id"] for a in st_qk.get("applied", [])],
                    lin_applied_ids=[a["module_id"] for a in st_lin.get("applied", [])],
                    qk_visible=[a.get("visible") for a in st_qk.get("applied", [])],
                    lin_shapes=[a.get("per_row_input_shapes") for a in st_lin.get("applied", [])]))


def row_aligned_verify_dict(target) -> dict:
    """`spec_dec` 用的包装入口：把上下文按 `target` 绑定（可重复调用、默认不启用）。"""
    return row_aligned_verify(target)


# ------------------------------------------------------------------ CPU 门（Stage 0）
def _selftest() -> int:
    """**不加载大模型**：28 层微型模型 + 一个"draft"微型模型，验证 Stage 0 的 10 项要求。"""
    from . import cache_clone as CC

    npass = nfail = 0

    def check(name, cond, extra=None):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  [ok] {name}")
        else:
            nfail += 1
            print(f"  [FAIL] {name}" + (f"  {str(extra)[:240]}" if extra is not None else ""))

    torch.manual_seed(0)
    kw = dict(vocab_size=64, hidden=32, heads=4, kv_heads=2, head_dim=8, intermediate=64,
              rope_theta=10000.0, rms_eps=1e-6, attn_bias=True)
    tgt = M.QwenLM(num_layers=28, **kw).to(torch.bfloat16).eval()       # 与真实 target 同层数
    drf = M.QwenLM(num_layers=28, **kw).to(torch.bfloat16).eval()       # 充当 draft（另一个实例）
    M.set_attention_strategy("eager")
    ids = torch.randint(0, 64, (1, 5))
    with torch.inference_mode():
        _lg, c0 = tgt._run(tgt.embed_tokens(ids), None, need_logits=True)
    H = [{k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in lay.items()}
         for lay in CC.clone_list_cache_full(c0)]

    exp = expected_counts(tgt)
    check("★ 预期注册数量**从模型结构推导**：qk=28、linear=112、total=140（本仓库 target 层数=28）",
          exp == dict(n_layers=28, qk=28, linear=112, total=140,
                      note=exp["note"]) or (exp["qk"] == 28 and exp["linear"] == 112
                                            and exp["total"] == 140), exp)
    check("★ 模块集合：28 个 `self_attn` + 每层四个 Linear（o/gate/up/down），且全部互不相同",
          len(all_qk_modules(tgt)) == 28 and len(all_linear_modules(tgt)) == 112
          and len({id(m) for m in all_qk_modules(tgt) + all_linear_modules(tgt)}) == 140)

    # ① 默认关闭
    check("★ 默认关闭：两个注册表都是 `None`（生产路径一字不变）",
          M.serial_qk_state()["active"] is False and M.serial_linear_state()["active"] is False)
    # ② 只作用于 target + draft 完全不重叠
    check("★ 只作用于 target：注册集合 = target 的 140 个（`covers_all_target`），与 draft 0 重叠",
          coverage_check(tgt, drf)["covers_all_target"] is True
          and coverage_check(tgt, drf)["other_untouched"] is True, coverage_check(tgt, drf))

    # ③ 打开候选：计数是否符合结构（block verify s=4）
    with torch.inference_mode(), M.attention_phase("verify"):
        with row_aligned_verify(tgt) as info:
            _rb, _c = tgt._run(tgt.embed_tokens(torch.tensor([[3, 4, 5, 6]])), _to_dev(H, torch.device("cpu")),
                               need_logits=True)
            st_qk, st_lin = M.serial_qk_state(), M.serial_linear_state()
    log = verify_log()
    check("★ 全层覆盖：28 个 QK 全部 applied、112 个 Linear 全部 applied（一次 block verify）",
          st_qk["n_applied"] == 28 and st_lin["n_applied"] == 112
          and len(log) == 1 and log[0]["qk_applied"] == 28 and log[0]["lin_applied"] == 112,
          (st_qk["n_applied"], st_lin["n_applied"]))
    check("★ `seen` 计数为 0（注册集合已覆盖全部被拦调用点：28 QK 宿主 + 112 Linear）",
          st_qk["n_seen"] == 0 and st_lin["n_seen"] == 0, (st_qk["n_seen"], st_lin["n_seen"]))
    check("★ 退出后恢复：两个注册表回到 `None`；`verify_log()` 有 1 条记录",
          M.serial_qk_state()["active"] is False and M.serial_linear_state()["active"] is False)
    vr = verify_log()[0]
    check("★ 每层 QK 每行都用真实 causal prefix：`visible = p+i+1`（微型 H=5 key、block s=4 ⇒ p=5 ⇒ 6,7,8,9）",
          all(v == [6, 7, 8, 9] for v in vr["qk_visible"]), vr["qk_visible"][:2])
    check("★ 每个 Linear 逐行输入形状为 [1,1,Din]（gate/up/o=32、down=64；每层 4 个）",
          all(sh == [[1, 1, 32]] * 4 if i % 4 != 3 else sh == [[1, 1, 64]] * 4
              for i, sh in enumerate(vr["lin_shapes"])), vr["lin_shapes"][:4])
    clear_verify_log()

    # ④ draft 零干预：注册表装着 target 的模块时跑 draft 前向 ⇒ applied 必须为 0
    with torch.inference_mode(), M.attention_phase("verify"):
        with row_aligned_verify(tgt):
            _r2, _c2 = drf._run(drf.embed_tokens(torch.tensor([[3, 4, 5, 6]])),
                                _to_dev(H, torch.device("cpu")), need_logits=True)
            d_qk, d_lin = M.serial_qk_state(), M.serial_linear_state()
    check("★ draft 零干预：draft 前向的 `applied = 0`（只被计数），target 的才被替换",
          len(d_qk["applied"]) == 0 and len(d_lin["applied"]) == 0
          and d_qk["n_seen"] == 28 and d_lin["n_seen"] == 112,
          (len(d_qk["applied"]), d_qk["n_seen"], d_lin["n_seen"]))
    clear_verify_log()

    # ⑤ AR/decode（s=1）开关前后逐位一致
    with torch.inference_mode(), M.attention_phase("decode"):
        lg_off, cache_off = tgt._run(tgt.embed_tokens(torch.tensor([[7]])), _to_dev(H, torch.device("cpu")),
                                     need_logits=True)
    with torch.inference_mode(), M.attention_phase("decode"):
        with row_aligned_verify(tgt):
            lg_on, cache_on = tgt._run(tgt.embed_tokens(torch.tensor([[7]])),
                                       _to_dev(H, torch.device("cpu")), need_logits=True)
    check("★ AR/decode（s=1）开启候选与关闭逐位一致（logits 与缓存）",
          torch.equal(lg_off, lg_on)
          and all(torch.equal(cache_off[i][k], cache_on[i][k])
                  for i in range(len(cache_off)) for k in ("k", "v")))
    clear_verify_log()

    # ⑥ 多上下文嵌套 + 正常/异常退出恢复
    with row_aligned_verify(tgt):
        with row_aligned_verify(tgt):
            check("★ 多上下文嵌套（同一 target 两层）不报错且状态一致",
                  M.serial_qk_state()["n_mods"] == 28 and M.serial_linear_state()["n_mods"] == 112)
        check("★ 内层退出后外层仍生效", M.serial_qk_state()["n_mods"] == 28)
    check("★ 正常退出后全部关闭",
          M.serial_qk_state()["active"] is False and M.serial_linear_state()["active"] is False)
    try:
        with row_aligned_verify(tgt):
            raise RuntimeError("注入的候选模式故障")
    except RuntimeError:
        pass
    check("★ 异常退出也恢复（两个注册表都为 `None`，且 `verify_log()` 仍记录到该次）",
          M.serial_qk_state()["active"] is False and M.serial_linear_state()["active"] is False
          and len(verify_log()) >= 1)
    clear_verify_log()

    # ⑦ 旧的单层 L0 模式行为不变（只注册 L0 的 5 个模块 ⇒ 其余层不被替换）
    sa0 = tgt.layers[0].self_attn
    mlp0 = tgt.layers[0].mlp
    with torch.inference_mode(), M.attention_phase("verify"):
        with M.serial_qk_for(sa0, label="L0only"), \
                M.serial_linear_for(sa0.o_proj, mlp0.gate_proj, mlp0.up_proj, mlp0.down_proj,
                                    label="L0only"):
            _r3, _c3 = tgt._run(tgt.embed_tokens(torch.tensor([[3, 4, 5, 6]])),
                                _to_dev(H, torch.device("cpu")), need_logits=True)
            l_qk, l_lin = M.serial_qk_state(), M.serial_linear_state()
    check("★ 旧的单层 L0 模式行为不变：只 applied 1 个 QK + 4 个 Linear，其余层只被计数",
          l_qk["n_applied"] == 1 and l_lin["n_applied"] == 4
          and l_qk["n_seen"] == 27 and l_lin["n_seen"] == 108,
          (l_qk["n_applied"], l_qk["n_seen"], l_lin["n_applied"], l_lin["n_seen"]))
    check("★ 旧模式的 `applied` 就是 L0 的 5 个模块（对象身份）",
          l_qk["applied"][0]["module_id"] == int(id(sa0))
          and sorted(a["module_id"] for a in l_lin["applied"])
          == sorted(int(id(m)) for m in (sa0.o_proj, mlp0.gate_proj, mlp0.up_proj, mlp0.down_proj)))
    print(f"\n全层 row-aligned 候选模式（Stage 0 CPU 门）：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


def _to_dev(snap, dev):
    from . import cache_clone as CC
    full = CC.clone_list_cache_full(snap)
    return [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in layer.items()}
            for layer in full]


def main() -> int:
    print("== Stage 0：全层 row-aligned verify 候选模式的 CPU/微型模型门（**不加载大模型**）==")
    return _selftest()


if __name__ == "__main__":
    raise SystemExit(main())
