"""stage2b_verdict_selftest_cpu.py — Stage 2b 判决工具的 **CPU 自测**（纯 CPU；不加载模型、不使用 GPU）。

## 目的：证明"**不可能因为漏读/错读数据而虚假 PASS**"
每个用例都在**临时目录**里合成一套 Stage 2b 产物（**绝不触碰真实产物**），再调用 `stage2b_verdict.evaluate()`
检查判决是否符合预期：

| 用例 | 构造 | 期望判决 |
| --- | --- | --- |
| A 错误 tag | 请求 `div8209_TYPO`（产物在 `div8209_p2`） | `CHECK_NEEDS_REVIEW`（**不**读旧 tag） |
| B 缺新 family | 少一个 family 的任何产物 | `CHECK_NEEDS_REVIEW` |
| C 缺重复 | 某 family/serial 只有 rep1 | `CHECK_NEEDS_REVIEW` |
| D 重复编号冲突 | 同一 `(family, serial, rep1)` 两份产物 | `CHECK_NEEDS_REVIEW` |
| E 首次 AR 两次不一致（新 family） | AR rep1 ≠ rep2 | `BASELINE_REGRESSION` |
| F 旧 family 锚点不符 | F13 的 AR 输出改成别的 token | `BASELINE_REGRESSION` |
| G production 与 AR 分歧但结构正常 | production gen 在 index 7 后不同 | **不**判 serial 失败；`NUMERICAL_DIVERGENCE_RECORDED` 已记录 |
| H 新 family 数据完整 | 四个 family × 3 模式 × 2 重复 | **真正纳入判决**（`included_families` 含全部）；`SERIAL_ORACLE_PASS` |
| I manifest 输入 hash 不符 | 记录的 `input.family.token_sha1` 与 manifest 不一致 | `CHECK_NEEDS_REVIEW` |
| J tag 内含越界 family | 额外放入 F13 的产物 | `CHECK_NEEDS_REVIEW`；加 `--allow-off-scope` 后可判 PASS |
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

from . import stage2b_verdict as SV

NEW = ("F09", "F03", "F11", "F04")
MODES = ("ar", "prod", "serial")
REPS = 2
TAG_GOOD = "div8209_p2"
TAG_WRONG = "div8209_TYPO"
# 真实 manifest 的 8192 token_sha1（自测里用真值，使"输入 hash 校验"路径被真正走到）
REAL_SHA = dict(F09="e5dbe83616ea348a", F03="92755adeb5715d47",
                F11="2c14a59130440e9a", F04="e9004b83814fe321",
                F13="9a278731d744a859")
RUNID = "20261004T000000Z"


def _gen_for(family: str, mode: str, rep: int, *, variant: int = 0) -> list:
    base = (hash(family) % 1000) + 100
    if mode == "prod" and variant == 1:
        g = [base + i for i in range(64)]
        g[7] = base + 999          # production 与 AR 在 index 7 分歧
        return g
    return [base + i for i in range(64)]


def _rounds_and_trace(gen: list, *, pre0: int = 8192, kp: int = 4) -> tuple:
    rounds, trace = [], []
    n = len(gen) // kp
    for i in range(n):
        cands = list(gen[i * kp:(i + 1) * kp])
        pre = pre0 + i * kp
        nxt = list(gen[(i + 1) * kp:(i + 2) * kp]) or [cands[0]]
        bonus = nxt[0]
        rounds.append(dict(round=i, Kp=kp, m=kp, pre=pre, cands=cands, g=list(cands),
                           emitted=list(cands), tgt_len=pre + kp, drf_len=pre + kp,
                           rollback=False, repair=False, correction=None, all_accept=True,
                           tgt_len_after=pre + kp, drf_len_after=pre + kp))
        trace.append(dict(kp=kp, pre=pre, mode="serial",
                          step_cache_lens=list(range(pre + 1, pre + kp + 1)),
                          step_argmax=cands[1:] + [bonus]))
    return rounds, trace


def _mk(tag: str, family: str, mode: str, rep: int, *, gen: list | None = None,
        stop: str = "max_new_reached", token_sha1: str | None = None,
        self_hash: str | None = None, batches: bool | None = None,
        model_run: bool = True, env_blocked: bool = False) -> dict:
    gen = list(gen if gen is not None else _gen_for(family, mode, rep))
    rounds, trace = _rounds_and_trace(gen)
    seq_lens = [1, 4] if (batches if batches is not None else mode == "prod") else [1]
    by_phase = dict(prefill=dict(n=4, seq_lens=[2048] * 4, max_seq_len=2048),
                    **({"verify": dict(n=len(rounds), seq_lens=[4] * len(rounds), max_seq_len=4)}
                       if mode == "prod" else
                       {"verify": dict(n=len(rounds), seq_lens=[1] * len(rounds), max_seq_len=1)}))
    n_gen = 3 * len(rounds) + 1
    rec = dict(
        tool=SV.TOOL, tag=tag, case=family, mode=mode, rep=rep, run_id=RUNID,
        diagnostic_only=True, model_run_happened=bool(model_run), env_qualified=not env_blocked,
        blocked_by=("environment" if env_blocked else None),
        verify_mode_before_run=("serial" if mode == "serial" else "block"),
        verify_mode_at_run_end=("serial" if mode == "serial" else "block"),
        verify_mode_after_run="block", row_aligned_wrap_active=False,
        smoke_rc=0, stage="complete",
        serial_trace=(trace if mode == "serial" else []),
        audit=dict(n_calls=238, n_instances=2, sane=True, certifies_m1=True, violations=[],
                   labels={"target": ["1"], "draft": ["2"]},
                   target=dict(label="target", label_source="smoke_hooks._HOOK_REGISTRY.tgt",
                               n_params=1_500_000_000, n_layers_first=28, n_calls=n_gen,
                               n_prefill_calls=4, n_generation_calls=n_gen,
                               generation_seq_lens=seq_lens,
                               all_generation_m1=(seq_lens == [1]), nonvacuous=True,
                               by_phase=by_phase),
                   cross_check=dict(layer_counts_differ=True, target_params_gt_draft=True,
                                    target_n_layers=28, draft_n_layers=24)),
        accounting=dict(mode=mode, target_generation_calls_measured=n_gen, expected=n_gen,
                        matches=True),
        self_checks=dict(smoke_rc_zero=True, audit_certifies_target_generation_m1=True),
        self_checks_all_ok=True,
        provenance=dict(hash_verified=True, family_id=family),
        commit_guard=dict(ok=not env_blocked, why="selftest", overridden=False),
        smoke=dict(gen=gen, gen_len=len(gen), stop_reason=stop, gen_ids_in_vocab=True,
                   prefill_logits_finite=True, n_rounds=len(rounds), rounds=rounds,
                   # ⚠️ `_row` 读的是 runner **扁平化**后的 `smoke.checks_passed`（不是 harness 的
                   #    `smoke.checks.passed`）；首版漏了这个字段 ⇒ 硬检查 13 误报失败（自测抓到）。
                   checks_passed=True, checks_failures=[],
                   round_invariants=dict(all_ok=True, mode=mode, per_round=[]),
                   specdec_stats=dict(rounds=len(rounds), proposed=len(rounds) * 4,
                                      accepted=len(rounds) * 4, rollbacks=0,
                                      mismatch_rounds=0, all_accept_rounds=len(rounds),
                                      target_block_calls=len(rounds), draft_calls=len(rounds) * 4),
                   cache_structure_ok=True, cache_structure_n_checks=50,
                   mem_timeline_last=dict(pt_peak_allocated_mib=7800.0, pt_peak_reserved_mib=10648.0),
                   input=dict(source_kind="manifest",
                              family=dict(family_id=family,
                                          token_sha1=(token_sha1 or REAL_SHA.get(family, "deadbeef")),
                                          hash_verified=True)),
                   checks=dict(passed=True, n_required=14, uncovered=[], failures=[],
                               finiteness_coverage=dict(n_checked=210, first_bad=None,
                                                        by_phase_model={"prefill/target": dict(ok=True),
                                                                        "verify/target": dict(ok=True)})),
                   end_state=dict(gen_hash=(self_hash or SV.AGG._gen_hash(gen)), gen_len=len(gen),
                                  stop_reason=stop, eos_seen=False)))
    return rec


def _write(outdir: Path, recs: list) -> None:
    for r in recs:
        # ⚠️ 文件名必须用**每条记录自己的 run_id**：首版写死模块常量 ⇒ "重复编号冲突"用例
        #    的第二份产物把第一份**覆盖**掉了，冲突根本没落到磁盘（自测 D 抓到）。
        name = f"{r['tag']}_{r['case']}_{r['mode']}_rep{r['rep']}_{r.get('run_id', RUNID)}.json"
        (outdir / name).write_text(json.dumps(r, ensure_ascii=False, default=str), encoding="utf-8")


def _full_set(tag: str, families=NEW, *, prod_variant: int = 0, ar_override=None) -> list:
    recs = []
    for fam in families:
        for rep in (1, 2):
            g = (ar_override or {}).get((fam, rep))
            recs.append(_mk(tag, fam, "ar", rep, gen=g))
            recs.append(_mk(tag, fam, "prod", rep, gen=_gen_for(fam, "prod", rep, variant=prod_variant)))
            recs.append(_mk(tag, fam, "serial", rep))
    return recs


def main() -> int:
    checks: list = []

    def ck(name, cond, extra=None):
        checks.append((name, bool(cond), extra))

    tmp = Path(tempfile.mkdtemp(prefix="s2b_selftest_"))
    try:
        # ---------- H) 完整数据 ⇒ 真正纳入判决 ----------
        _write(tmp, _full_set(TAG_GOOD))
        sH = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmp)
        ck("H_完整数据_判决=PASS", sH["decision"] == "SERIAL_ORACLE_PASS", sH["decision"])
        ck("H_四个 family 全部纳入判决", set(sH["included_families"]) == set(NEW),
           sH["included_families"])
        ck("H_期望三元组数=24", sH["n_triples_expected"] == 24, sH["n_triples_expected"])
        ck("H_实得三元组数=24", sH["n_triples_found"] == 24, sH["n_triples_found"])
        ck("H_无 needs_review", not sH["needs_review"], sH["needs_review"])

        # ---------- A) 错误 tag ----------
        sA = SV.evaluate(TAG_WRONG, list(NEW), list(MODES), REPS, outdir=tmp)
        ck("A_错误tag_不PASS", sA["decision"] == "CHECK_NEEDS_REVIEW", sA["decision"])
        ck("A_错误tag_读到0条产物", sA["n_runs_found_total"] == 0, sA["n_runs_found_total"])
        ck("A_错误tag_报缺24个三元组", len([x for x in sA["needs_review"] if x.startswith("缺运行")]) == 24,
           len(sA["needs_review"]))

        # ---------- B) 缺新 family ----------
        tmpB = tmp / "b"; tmpB.mkdir()
        _write(tmpB, [r for r in _full_set(TAG_GOOD) if r["case"] != "F11"])
        sB = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpB)
        ck("B_缺family_不PASS", sB["decision"] == "CHECK_NEEDS_REVIEW", sB["decision"])
        ck("B_指出缺的是F11", any("F11" in x for x in sB["needs_review"]), sB["needs_review"][:3])

        # ---------- C) 缺重复 ----------
        tmpC = tmp / "c"; tmpC.mkdir()
        _write(tmpC, [r for r in _full_set(TAG_GOOD)
                      if not (r["case"] == "F03" and r["mode"] == "serial" and r["rep"] == 2)])
        sC = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpC)
        ck("C_缺重复_不PASS", sC["decision"] == "CHECK_NEEDS_REVIEW", sC["decision"])
        ck("C_指出缺F03_serial_rep_2",
           any("F03" in x and "serial" in x and "rep=2" in x for x in sC["needs_review"]),
           sC["needs_review"][:3])

        # ---------- D) 重复编号冲突 ----------
        tmpD = tmp / "d"; tmpD.mkdir()
        recs = _full_set(TAG_GOOD)
        dup = _mk(TAG_GOOD, "F04", "serial", 1)
        dup["run_id"] = "20261004T000001Z"
        _write(tmpD, recs + [dup])
        sD = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpD)
        ck("D_重复编号_不PASS", sD["decision"] == "CHECK_NEEDS_REVIEW", sD["decision"])
        ck("D_报冲突", any("重复编号冲突" in x for x in sD["needs_review"]), sD["needs_review"][:3])

        # ---------- E) 首次 AR 两次不一致（新 family）----------
        tmpE = tmp / "e"; tmpE.mkdir()
        bad = _gen_for("F09", "ar", 2)
        bad[10] += 500
        _write(tmpE, _full_set(TAG_GOOD, ar_override={("F09", 2): bad}))
        sE = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpE)
        ck("E_AR两次不一致=BASELINE_REGRESSION", sE["decision"] == "BASELINE_REGRESSION", sE["decision"])
        ck("E_未误判为serial失败", not sE["serial_failures"], sE["serial_failures"][:2])

        # ---------- F) 旧 family 锚点不符 ----------
        tmpF = tmp / "f"; tmpF.mkdir()
        wrong = [100 + i for i in range(64)]        # 与登记锚点 6e7522bf… 不符
        _write(tmpF, [_mk(TAG_GOOD, "F13", "ar", 1, gen=wrong),
                      _mk(TAG_GOOD, "F13", "ar", 2, gen=wrong),
                      _mk(TAG_GOOD, "F13", "prod", 1), _mk(TAG_GOOD, "F13", "prod", 2),
                      _mk(TAG_GOOD, "F13", "serial", 1, gen=wrong),
                      _mk(TAG_GOOD, "F13", "serial", 2, gen=wrong)])
        sF = SV.evaluate(TAG_GOOD, ["F13"], ["ar", "prod", "serial"], REPS, outdir=tmpF)
        ck("F_旧family锚点不符=BASELINE_REGRESSION", sF["decision"] == "BASELINE_REGRESSION", sF["decision"])
        ck("F_锚点类别为登记输出锚点",
           sF["per_family"]["F13"]["anchor_class"] == "registered_output_anchor",
           sF["per_family"]["F13"]["anchor_class"])

        # ---------- G) production 与 AR 分歧但结构正常 ----------
        tmpG = tmp / "g"; tmpG.mkdir()
        _write(tmpG, _full_set(TAG_GOOD, prod_variant=1))
        sG = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpG)
        ck("G_分歧不判serial失败", not sG["serial_failures"], sG["serial_failures"][:2])
        ck("G_判决仍PASS（分歧只记录）", sG["decision"] == "SERIAL_ORACLE_PASS", sG["decision"])
        ck("G_分歧分类为NUMERICAL_DIVERGENCE_RECORDED",
           all(v["production_divergence_class"] == "NUMERICAL_DIVERGENCE_RECORDED"
               for v in sG["per_family"].values()),
           {k: v["production_divergence_class"] for k, v in sG["per_family"].items()})
        ck("G_记录首分歧位置", all(v["production_divergence"]["first_divergence"][0] == 7
                                   for v in sG["per_family"].values()),
           {k: v["production_divergence"]["first_divergence"] for k, v in sG["per_family"].items()})

        # ---------- I) manifest 输入 hash 不符 ----------
        tmpI = tmp / "i"; tmpI.mkdir()
        _write(tmpI, _full_set(TAG_GOOD, ar_override={}) and
               [dict(r, smoke=dict(r["smoke"], input=dict(source_kind="manifest",
                                                          family=dict(family_id=r["case"],
                                                                      token_sha1="WRONG_SHA",
                                                                      hash_verified=True))))
                for r in _full_set(TAG_GOOD)])
        sI = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpI)
        ck("I_输入hash不符_不PASS", sI["decision"] == "CHECK_NEEDS_REVIEW", sI["decision"])

        # ---------- J) tag 内含越界 family ----------
        tmpJ = tmp / "j"; tmpJ.mkdir()
        _write(tmpJ, _full_set(TAG_GOOD) + [_mk(TAG_GOOD, "F13", "ar", 1), _mk(TAG_GOOD, "F13", "ar", 2)])
        sJ = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpJ)
        ck("J_越界family_不PASS", sJ["decision"] == "CHECK_NEEDS_REVIEW", sJ["decision"])
        sJ2 = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpJ, allow_off_scope=True)
        ck("J_显式allow后可PASS", sJ2["decision"] == "SERIAL_ORACLE_PASS", sJ2["decision"])

        # ---------- 环境拒绝 ⇒ 缺重复 ----------
        tmpK = tmp / "k"; tmpK.mkdir()
        recs = [r for r in _full_set(TAG_GOOD)
                if not (r["case"] == "F09" and r["mode"] == "ar" and r["rep"] == 2)]
        eb = _mk(TAG_GOOD, "F09", "ar", 2, model_run=False, env_blocked=True)
        _write(tmpK, recs + [eb])
        sK = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpK)
        ck("K_环境拒绝造成缺重复_不PASS", sK["decision"] == "CHECK_NEEDS_REVIEW", sK["decision"])
        ck("K_环境拒绝被单独报告", sK["n_env_blocked"] == 1, sK["n_env_blocked"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    npass = sum(1 for _, ok, _ in checks if ok)
    nfail = len(checks) - npass
    print("== Stage 2b 判决工具自测（纯 CPU；合成产物在临时目录，未触碰真实产物）==")
    for nm, ok, extra in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {nm}" + (f"  {str(extra)[:160]}" if not ok else ""))
    print(f"\nStage 2b 工具自测：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
