"""qk_scope_selftest_cpu.py — QK FP32 干预的**范围与生效性** CPU 自测（不加载大模型）。

要回答的问题（对应任务"先检查，不要直接启动 GPU"）
  1. 开关实际改变哪些算子？输出 dtype 变不变？
  2. 在哪些执行阶段生效？**融合 prefill 是否绕过它？**
  3. 能否把干预**限定到 target**、让 draft 保持原精度？
  4. 异常/卸载后开关是否复位（不能污染后续运行）？
"""
from __future__ import annotations

import inspect

import torch

from . import model_self as M
from . import diag_qk_scope as S


def _tiny_model(heads=4, kv_heads=2, head_dim=32, layers=2, hidden=None):
    hidden = hidden or heads * head_dim
    m = M.QwenLM(vocab_size=64, hidden=hidden, heads=heads, kv_heads=kv_heads,
                 head_dim=head_dim, intermediate=hidden * 2, num_layers=layers)
    # ⚠️ 必须转 bf16：生产是 bf16 权重，默认 fp32 下 QK 本来就在 fp32 算，
    # 开关会变成**无操作**（第一版自测就是这样拿到 max|d|=0.0 的假结论）。
    return m.to(torch.bfloat16)


def _run() -> int:
    npass = nfail = 0

    def check(name, cond, extra=""):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  [ok] {name}")
        else:
            nfail += 1
            print(f"  [FAIL] {name} {extra}")

    print("== 1. 开关的唯一读取点与算子范围（静态可复查） ==")
    proof = S.static_scope_proof()
    check("模块里 DIAG_QK_FP32 的读取点共 2 处（含状态读取器）",
          proof["n_qk_read_sites"] == 2, proof["qk_read_sites"])
    check("★ **影响计算**的读取点只有 1 处（排除 diag_precision 这类读取器）",
          proof["n_qk_compute_read_sites"] == 1, proof["qk_compute_read_sites"])
    check("该唯一读取点位于 attention 的 eager 分支（forward 内）",
          proof["qk_read_only_in_eager_branch"] is True, proof["qk_compute_read_sites"])
    check("★ 融合 prefill 的源码里**没有**该开关名 ⇒ 融合 prefill 绕过它",
          proof["fused_bypasses_qk"] is True and proof["fused_bypasses_prob"] is True)
    check("输出 dtype 明确标为不变（干预不改变张量类型）",
          proof["output_dtype_unchanged"] is True)
    for op in ("o_proj / MLP / RMSNorm / lm_head / 权重 / 缓存 dtype",):
        check(f"明确不改动的部分含：{op}", op in proof["unchanged_ops"])

    print("== 2. 阶段分流：哪些阶段吃到这个开关 ==")
    cov = proof["coverage_by_phase"]
    check("prefill s=4 ⇒ fused（开关**无效**）", cov["prefill_s4"]["path"] == "fused")
    check("prefill s=1 ⇒ eager（cuDNN 不支持 s=1）", cov["prefill_s1"]["path"] == "eager")
    for ph in ("decode", "verify", "repair"):
        for s in (1, 4):
            check(f"{ph} s={s} ⇒ eager（开关**有效**）",
                  cov[f"{ph}_s{s}"]["path"] == "eager")
    check("propose 也是 eager（但本工具把 draft 排除在作用域外）",
          cov["propose_s1"]["path"] == "eager" and cov["propose_s4"]["path"] == "eager")
    check("生效阶段写明 decode/verify/repair",
          all(any(k in e for e in proof["effective_stages"])
              for k in ("decode", "verify", "repair")))
    check("未生效阶段写明融合 prefill",
          any("prefill" in e for e in proof["not_effective_stages"]))

    print("== 3. 开关确实改变 eager 数值（真实 tiny 模型，CPU） ==")
    torch.manual_seed(0)
    m = _tiny_model()
    m.eval()
    T = 512
    x = torch.randn(1, T, m.embed_tokens.weight.shape[1]).to(torch.bfloat16)
    with torch.inference_mode(), M.attention_phase("decode"):
        M.set_diag_precision(qk32=False, prob32=False)
        out_off, _ = m._run(x, None, need_logits=True)
        M.set_diag_precision(qk32=True)
        out_on, _ = m._run(x, None, need_logits=True)
        M.set_diag_precision(qk32=False)
    check("两条路径输出 dtype 相同（都是 bf16）⇒ 干预不改变 dtype",
          out_off.dtype == out_on.dtype == torch.bfloat16,
          f"{out_off.dtype} vs {out_on.dtype}")
    check("两条路径形状相同", tuple(out_off.shape) == tuple(out_on.shape))
    check("★ QK FP32 确实改变了 eager 的数值输出（elementwise_equal=False）",
          not torch.equal(out_off, out_on))
    d = (out_off.float() - out_on.float()).abs().max().item()
    check("差异非零且有限", d > 0 and d == d, f"max|d|={d}")
    check("关闭后与「再跑一次关闭」逐位一致（开关可逆、无残留）",
          torch.equal(out_off, _rerun(m, x)))

    print("== 4. per-model 作用域：target 开、draft 关 ==")
    seen = {}

    class StubM:
        """记录调用瞬间观察到的全局开关值。"""
        def __init__(self, name):
            self.name = name

        def _run(self, x, cache_list=None, need_logits=True):
            seen[self.name] = M.diag_precision()["qk_fp32"]
            return x, cache_list

    tgt, drf = StubM("target"), StubM("draft")
    h = S.QKScopeHook(tgt=tgt, enabled=True, run_owner=StubM)
    h.install()
    try:
        tgt._run(1)
        drf._run(1)
        tgt._run(1)
    finally:
        rep = h.report()
        h.remove()
    check("★ target 调用时开关为 True", seen["target"] is True)
    check("★ draft 调用时开关为 False（保持原精度）", seen["draft"] is False)
    check("卸载后全局开关复位为 False", M.diag_precision()["qk_fp32"] is False)
    check("统计：target 调用数 = flag_on 数", rep["calls_target"] == 2 and rep["flag_on_calls"] == 2)
    check("统计：其它模型调用数 = flag_off 数",
          rep["calls_other"] == 1 and rep["flag_off_calls"] == 1)
    check("报告标明 scope=target-only", rep["scope"] == "target-only")

    print("== 5. 控制组（enabled=False）：谁都不开 ==")
    seen.clear()
    tgt2, drf2 = StubM("target"), StubM("draft")
    h2 = S.QKScopeHook(tgt=tgt2, enabled=False, run_owner=StubM)
    h2.install()
    try:
        tgt2._run(1)
        drf2._run(1)
        rep2 = h2.report()
    finally:
        h2.remove()
    check("控制组：target 调用时开关为 False", seen["target"] is False)
    check("控制组：draft 调用时开关为 False", seen["draft"] is False)
    check("控制组 flag_on_calls=0", rep2["flag_on_calls"] == 0)
    check("控制组报告 scope=control", rep2["scope"].startswith("control"))

    print("== 6. 异常安全：包装内抛异常也必须复位 ==")

    class Boom:
        def _run(self, x, cache_list=None, need_logits=True):
            raise RuntimeError("boom")

    bo = Boom()
    h3 = S.QKScopeHook(tgt=bo, enabled=True, run_owner=Boom)
    h3.install()
    try:
        bo._run(1)
        raised = False
    except RuntimeError:
        raised = True
    finally:
        h3.remove()
    check("异常向外传播（不吞掉）", raised is True)
    check("★ 异常后全局开关已复位为 False", M.diag_precision()["qk_fp32"] is False)

    print("== 7. 脏状态拒绝安装（避免把干预泄进控制组） ==")
    M.set_diag_precision(qk32=True)
    try:
        S.QKScopeHook(tgt=StubM("t"), enabled=False, run_owner=StubM).install()
        refused = False
    except AssertionError:
        refused = True
    finally:
        M.set_diag_precision(qk32=False)
    check("★ 全局开关不干净时 install() 直接断言失败", refused is True)
    check("清理后开关为 False", M.diag_precision()["qk_fp32"] is False)

    print(f"\nCPU 自测：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


def _rerun(m, x):
    with torch.inference_mode(), M.attention_phase("decode"):
        M.set_diag_precision(qk32=False, prob32=False)
        out, _ = m._run(x, None, need_logits=True)
    return out


if __name__ == "__main__":
    raise SystemExit(_run())
