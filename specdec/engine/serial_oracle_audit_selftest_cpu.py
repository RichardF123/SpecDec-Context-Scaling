"""serial_oracle_audit_selftest_cpu.py — **调用几何审计器自身**的纯 CPU 自测（不加载模型、不用 GPU）。

## 为什么必须有这一层
运行产物里"`all_target_calls_m1 = True`"是**结论性证据**。若审计器本身装错位置、或对空集合
求 `all(...)`，就会得到**空泛为真**的假证据。因此本自测针对审计器逐条检验：

1. **阶段判定**：`prefill` 阶段允许多次大调用（真实 harness 是分块 prefill；Stage 0 的 `_Audit`
   "只允许一次大调用"在真实运行里会误报，故另写一套阶段感知版本）；
2. **硬失败**：非 `prefill` 阶段出现 `s>1` ⇒ 立即抛 `CallGeometryError`（必须**能**失败）；
3. **记录模式**：不抛错，但必须如实报出 `generation_seq_lens` 含 >1；
4. **反空泛**：零调用 / 零 generation 调用 ⇒ `ok=False` 且 `target_all_generation_m1 is None`
   （**不得**因集合为空而判 True）；
5. **身份识别**：无 `smoke_hooks` 注册表时标签必须是 `unknown` 且 `ok=False`（绝不猜）；
   有注册表时按 `is` 判定 target/draft，并用层数/参数量交叉核对；
6. **恢复**：`uninstall()` 必须把 `owner_cls._run` 还原为原函数（用 `is` 核查）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from . import serial_oracle as SO

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"


class StubQwen(nn.Module):
    """最小可用的"类 QwenLM"：`_run` 返回 `(None, cache)`，cache = `[{k:(1,2,T,4)}]*layers`。"""
    def __init__(self, layers: int, dim: int):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1, dim))
        self.layers = int(layers)

    def _run(self, x, cache_list=None, need_logits=True):
        s = int(x.shape[1])
        old = list(cache_list or [])
        out = []
        for i in range(self.layers):
            base = old[i]["k"] if i < len(old) else torch.zeros(1, 2, 0, 4)
            out.append(dict(k=torch.cat([base, torch.zeros(1, 2, s, 4)], dim=2)))
        return None, out


class _FakeHooks:
    def __init__(self, tgt, drf):
        self.tgt, self.drf = tgt, drf


def _ck(name, cond, extra=None):
    return (name, bool(cond), extra)


def main() -> int:
    from . import model_self as M
    from . import smoke_hooks as SH

    checks: list = []
    # 参数量必须可区分（审计用它交叉核对身份）：target 2800 > draft 2400
    tgt, drf = StubQwen(28, 2800), StubQwen(24, 2400)
    T = torch.zeros(1, 1, 8)                       # token 张量（审计只看 shape[1]）

    # ==== 1) 记录模式：prefill 分块（3×2048）+ verify 逐 token(4×1) + repair 1 次 ====
    AUD = SO.install_run_audit(StubQwen, enforce_m1_outside_prefill=False)
    saved_reg = dict(SH._HOOK_REGISTRY)
    try:
        with M.attention_phase("prefill"):
            c = None
            for _ in range(3):
                c = tgt._run(torch.zeros(1, 2048, 8), c)[1]
        with M.attention_phase("verify"):
            for _ in range(4):
                c = tgt._run(T, c)[1]
        with M.attention_phase("repair"):
            c = tgt._run(T, c)[1]
        s_no_reg = SO.run_audit_summary(AUD["state"])
    finally:
        res1 = AUD["uninstall"]()
        SH._HOOK_REGISTRY.clear()
        SH._HOOK_REGISTRY.update(saved_reg)
    ti = list(s_no_reg["per_instance"].values())[0]
    checks += [_ck("a01_prefill_phase_allows_multiple_large_calls",
                   ti["by_phase"]["prefill"]["n"] == 3
                   and ti["by_phase"]["prefill"]["max_seq_len"] == 2048, ti["by_phase"]),
               _ck("a02_generation_seq_lens_recorded", ti["generation_seq_lens"] == [1],
                   ti["generation_seq_lens"]),
               _ck("a03_all_generation_m1_true_when_clean",
                   ti["all_generation_m1"] is True and ti["nonvacuous"] is True, ti),
               _ck("a04_n_layers_from_returned_cache", ti["n_layers_first"] == 28, ti["n_layers_first"]),
               _ck("a05_no_registry_label_is_unknown_and_not_ok",
                   ti["label"] == "unknown" and s_no_reg["ok"] is False,
                   (ti["label"], s_no_reg["ok"])),
               _ck("a06_uninstall_restored", res1["pre_uninstall_was_our_hook"] is True
                   and res1["restored_is_original"] is True, res1)]

    # ==== 2) 记录模式 + 注册表：两个实例按身份分组，层数可区分 ====
    AUD = SO.install_run_audit(StubQwen, enforce_m1_outside_prefill=False)
    SH._HOOK_REGISTRY[StubQwen] = _FakeHooks(tgt, drf)
    try:
        with M.attention_phase("prefill"):
            ct = tgt._run(torch.zeros(1, 2048, 8), None)[1]
            cd = drf._run(torch.zeros(1, 2048, 8), None)[1]
        with M.attention_phase("verify"):
            ct = tgt._run(torch.zeros(1, 4, 8), ct)[1]           # ← block 几何（s=4）
        with M.attention_phase("propose"):
            cd = drf._run(T, cd)[1]
        s2 = SO.run_audit_summary(AUD["state"])
    finally:
        res2 = AUD["uninstall"]()
        SH._HOOK_REGISTRY.clear()
        SH._HOOK_REGISTRY.update(saved_reg)
    checks += [_ck("a07_two_instances_grouped_by_identity",
                   s2["n_instances"] == 2 and set(s2["labels"]) == {"target", "draft"}, s2["labels"]),
               _ck("a08_block_geometry_is_reported_as_not_m1",
                   s2["target"]["generation_seq_lens"] == [4]
                   and s2["target"]["all_generation_m1"] is False, s2["target"]),
               _ck("a09_layer_counts_differ_and_params_ordered",
                   s2["cross_check"]["layer_counts_differ"] is True
                   and s2["cross_check"]["target_params_gt_draft"] is True, s2["cross_check"]),
               _ck("a10_block_geometry_does_not_certify_m1_but_audit_is_sane",
                   # 记录模式下 target 的 generation 里有 s=4 ⇒ **不得**认证 M=1（`ok`/`certifies_m1` False），
                   # 但审计器本身必须"装对了"（`sane` True）—— 两个判据语义不同，不能混用。
                   s2["ok"] is False and s2["certifies_m1"] is False and s2["sane"] is True,
                   dict(ok=s2["ok"], sane=s2["sane"], certifies_m1=s2["certifies_m1"])),
               _ck("a11_uninstall_restored_2", res2["restored_is_original"] is True, res2)]

    # ==== 3) 强制模式：非 prefill 阶段 s>1 ⇒ **必须抛错**（负对照：审计器必须能失败） ====
    AUD = SO.install_run_audit(StubQwen, enforce_m1_outside_prefill=True)
    SH._HOOK_REGISTRY[StubQwen] = _FakeHooks(tgt, drf)
    raised3, msg3 = None, None
    try:
        with M.attention_phase("prefill"):
            c3 = tgt._run(torch.zeros(1, 2048, 8), None)[1]
        try:
            with M.attention_phase("verify"):
                tgt._run(torch.zeros(1, 4, 8), c3)                # ← 必须在这里抛
            raised3 = "NO_RAISE"
        except SO.CallGeometryError as ex:
            raised3, msg3 = "CallGeometryError", str(ex)[:200]
        s3 = SO.run_audit_summary(AUD["state"])
    finally:
        AUD["uninstall"]()
        SH._HOOK_REGISTRY.clear()
        SH._HOOK_REGISTRY.update(saved_reg)
    checks += [_ck("a12_enforce_raises_on_s_gt_1_outside_prefill",
                   raised3 == "CallGeometryError", (raised3, msg3)),
               _ck("a13_violation_recorded_with_phase_and_seq_len",
                   len(s3["violations"]) == 1 and s3["violations"][0]["seq_len"] == 4
                   and s3["violations"][0]["phase"] == "verify", s3["violations"]),
               _ck("a14_failed_call_still_recorded",
                   4 in s3["target"]["by_phase"].get("verify", {}).get("seq_lens", []),
                   s3["target"]["by_phase"])]

    # ==== 4) 强制模式 + 干净几何（prefill 分块 + 全部 M=1）⇒ 唯一一次 ok=True ====
    AUD = SO.install_run_audit(StubQwen, enforce_m1_outside_prefill=True)
    SH._HOOK_REGISTRY[StubQwen] = _FakeHooks(tgt, drf)
    try:
        with M.attention_phase("prefill"):
            ct = tgt._run(torch.zeros(1, 2048, 8), None)[1]
            cd = drf._run(torch.zeros(1, 2048, 8), None)[1]
        for _ in range(6):
            with M.attention_phase("verify"):
                ct = tgt._run(T, ct)[1]
            with M.attention_phase("propose"):
                cd = drf._run(T, cd)[1]
        with M.attention_phase("repair"):
            ct = tgt._run(T, ct)[1]
        s4 = SO.run_audit_summary(AUD["state"])
    finally:
        AUD["uninstall"]()
        SH._HOOK_REGISTRY.clear()
        SH._HOOK_REGISTRY.update(saved_reg)
    checks += [_ck("a15_clean_geometry_is_ok", s4["ok"] is True and s4["sane"] is True, s4),
               _ck("a16_target_generation_counts_m1_only",
                   s4["target"]["generation_seq_lens"] == [1]
                   and s4["target"]["n_generation_calls"] == 7, s4["target"]),
               _ck("a17_draft_and_target_both_recorded",
                   s4["draft"]["n_generation_calls"] == 6 and s4["target"]["n_generation_calls"] == 7,
                   (s4["draft"]["n_generation_calls"], s4["target"]["n_generation_calls"]))]

    # ==== 5) 反空泛：零调用 ⇒ 不得判 True ====
    empty = dict(installed=True, owner="x", enforce_m1_outside_prefill=True,
                 per_instance={}, order=[], n_calls=0, violations=[])
    s5 = SO.run_audit_summary(empty)
    checks += [_ck("a18_empty_audit_is_not_ok",
                   s5["ok"] is False and s5["sane"] is False and s5["target"] is None, s5)]

    # ==== 6) 恢复后的类属性必须就是原函数 ====
    orig_run = StubQwen._run
    AUD = SO.install_run_audit(StubQwen, enforce_m1_outside_prefill=True)
    mid = StubQwen._run
    res6 = AUD["uninstall"]()
    checks += [_ck("a19_class_attr_replaced_then_restored",
                   mid is not orig_run and StubQwen._run is orig_run, res6)]

    npass = sum(1 for _, ok, _ in checks if ok)
    nfail = len(checks) - npass
    print("== 调用几何审计器自测（纯 CPU、不加载模型）==")
    for nm, ok, extra in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {nm}"
              + (f"  {str(extra)[:220]}" if (extra is not None and not ok) else ""))
    print(f"\n审计器自测：PASS {npass} / FAIL {nfail}")

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(tool="serial_oracle_audit_selftest_cpu", run_id=run_id,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               diagnostic_only=True, gpu_used=False, model_loaded=False,
               semantics=("对 `serial_oracle.install_run_audit`/`run_audit_summary` 自身做检验："
                          "阶段判定（prefill 分块）、非 prefill 阶段 s>1 必须抛错、记录模式如实报出、"
                          "反空泛（零调用不得判 True）、身份识别（无注册表⇒unknown 且不通过）、"
                          "层数交叉核对、类属性恢复"),
               npass=npass, nfail=nfail,
               decision=("AUDITOR_SELFTEST_PASS" if nfail == 0 else "AUDITOR_SELFTEST_FAIL"),
               checks=[dict(name=n, ok=bool(o), extra=(None if o else e)) for n, o, e in checks],
               summaries=dict(no_registry=s_no_reg, two_instances=s2, enforced_violation=s3,
                              clean=s4, empty=s5))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = OUTDIR / f"serial_oracle_audit_selftest_{run_id}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[audit-selftest] RESULT_JSON={p}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
