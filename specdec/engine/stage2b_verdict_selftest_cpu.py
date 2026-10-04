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
# 合成产物的 prompt 长度：真实产物 `smoke.end_state.prompt_len` 存在（如 F13 prod = 8192）。
# 合成里统一取 8192，使"绝对位置 = prompt_len + 输出内下标"可被**精确断言**。
PROMPT_LEN = 8192
# 诊断记录里"位置来源"的合法标注（新工具 `gb6_diag_select` 写的就是这一口径）。
# ⚠️ 消费端要求它**含 `first_divergence`**：这正是为了拒绝"找不到 prod 就退回末 token"的记录。
ORIGIN_OK = "first_divergence(ar_vs_prod)"


def _diag(tag: str, families=(None,), **over) -> dict:
    """合成一份**形状与真实生产者一致**的 GB6 诊断记录（按 family 索引）。

    生产者 = `correctness_protocol_margins.py` + `gb6_diag_select.py`，其必需字段见
    `SV.GB6_FROM_DIAGNOSTIC`（数值）与 `SV.GB6_DIAG_PROVENANCE`（来源/配置）。
    """
    fams = list(NEW) if families == (None,) else list(families)
    out = {}
    for f in fams:
        d = dict(family=f, runs_tag=tag, ar_top1=1, ar_top2=2, margin=0.5, tied_at_max=1,
                 cache_len_after=8255, position=7, prompt_len=PROMPT_LEN,
                 position_origin=ORIGIN_OK)
        d.update(over)
        out[f] = d
    return out


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
                                  stop_reason=stop, eos_seen=False,
                                  # ⚠️ 忠实于真实产物：`end_state.prompt_len` / `max_new` /
                                  #    `target_cache_len` / `draft_cache_len` / `n_rounds` /
                                  #    `n_decode_steps` 都存在
                                  #    （见 output/g5/attn/div8209_so_F13_ar_rep2_*.json：
                                  #     AR 的 n_decode_steps=63 = 64 token - 首 token；
                                  #     production 的 n_decode_steps=0）。
                                  #    首版合成缺这些字段 ⇒ GB6 的 `absolute_position` 与
                                  #    `ar_work.decode_steps` 永远推导不出（**夹具不忠实**，自测抓到）。
                                  max_new=len(gen), prompt_len=PROMPT_LEN,
                                  n_decode_steps=(0 if mode != "ar" else max(0, len(gen) - 1)),
                                  target_cache_len=PROMPT_LEN + len(gen),
                                  draft_cache_len=PROMPT_LEN + len(gen), n_rounds=len(rounds))))
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
        # ⚠️ 本轮规则收紧后：分歧**本身**仍不判失败（本用例要守的就是这一点），但分歧必须完成
        #    GB6 记账（需独立诊断 pass）才能放行；故此处按新规则同时给出齐全诊断，
        #    只验证"分歧不判失败"，"缺记账不放行"由 CE4/CE4a 覆盖。
        tmpG = tmp / "g"; tmpG.mkdir()
        _write(tmpG, _full_set(TAG_GOOD, prod_variant=1))
        sG = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpG)
        ck("G_分歧不判serial失败", not sG["serial_failures"], sG["serial_failures"][:2])
        ck("G_缺GB6记账时不放行", sG["decision"] == "CHECK_NEEDS_REVIEW"
           and not sG["serial_failures"] and not sG["baseline_failures"], sG["decision"])
        sG2 = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpG,
                          gb6_diagnostics=_diag(TAG_GOOD))
        ck("G_记账齐全后分歧只记录_判决PASS", sG2["decision"] == "SERIAL_ORACLE_PASS", sG2["decision"])
        sG = sG2
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

        # ---------- 环境拒绝 ⇒ 缺重复（旧用例 K，保留）----------
        tmpK = tmp / "k"; tmpK.mkdir()
        recs = [r for r in _full_set(TAG_GOOD)
                if not (r["case"] == "F09" and r["mode"] == "ar" and r["rep"] == 2)]
        eb = _mk(TAG_GOOD, "F09", "ar", 2, model_run=False, env_blocked=True)
        _write(tmpK, recs + [eb])
        sK = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpK)
        ck("K_环境拒绝造成缺重复_不PASS", sK["decision"] == "CHECK_NEEDS_REVIEW", sK["decision"])
        ck("K_环境拒绝被单独报告", sK["n_env_blocked"] == 1, sK["n_env_blocked"])

        # ================= 反例回归（CE1–CE5；旧实现在这些用例上会**通过**）=================
        # ---- CE1：空 family 集合 + 不存在的 tag（旧实现：expected=0 ⇒ 空泛 PASS）----
        sCE1 = SV.evaluate("definitely_missing_stage2b_tag", [], list(MODES), REPS)
        ck("CE1_空family_不PASS", sCE1["decision"] != "SERIAL_ORACLE_PASS", sCE1["decision"])
        ck("CE1_报参数非法", bool(sCE1["argument_errors"]), sCE1["argument_errors"])
        ck("CE1_期望三元组为0但不放行", sCE1["n_triples_expected"] == 0
           and sCE1["decision"] == "CHECK_NEEDS_REVIEW", sCE1["n_triples_expected"])
        # ---- CE2：只给 AR×2 且 modes=[ar]（旧实现：prod/serial 为 0 仍 PASS）----
        tmpCE2 = tmp / "ce2"; tmpCE2.mkdir()
        _write(tmpCE2, [_mk(TAG_GOOD, "F13", "ar", 1), _mk(TAG_GOOD, "F13", "ar", 2)])
        sCE2 = SV.evaluate(TAG_GOOD, ["F13"], ["ar"], REPS, outdir=tmpCE2)
        ck("CE2_模式不全_不PASS", sCE2["decision"] != "SERIAL_ORACLE_PASS", sCE2["decision"])
        ck("CE2_报模式非法", any("模式" in e for e in sCE2["argument_errors"]), sCE2["argument_errors"])
        sCE2b = SV.evaluate(TAG_GOOD, ["F13"], list(MODES), REPS, outdir=tmpCE2, allow_off_scope=True)
        ck("CE2b_三种模式齐全但缺prod_serial_不PASS",
           sCE2b["decision"] == "CHECK_NEEDS_REVIEW", sCE2b["decision"])
        ck("CE2b_逐模式存在性报缺",
           any("prod 模式没有任何可用产物" in x for x in sCE2b["needs_review"])
           and any("serial 模式没有任何可用产物" in x for x in sCE2b["needs_review"]),
           sCE2b["needs_review"][:4])
        # ---- CE3：stage_check 必须真跑硬门（历史 F13：D6 缺失 + 非 clean complete ⇒ STOP）----
        try:
            from . import stage_check as SC
            r13 = SV.attach_internal(SV.load_runs("div8209_so")[-1]) if SV.load_runs("div8209_so") else None
            f13_ser = [SV.attach_internal(x) for x in SV.load_runs("div8209_so")
                       if x.get("case") == "F13" and x.get("mode") == "serial"
                       and x.get("model_run_happened")]
            if f13_ser:
                for r in f13_ser:
                    r["_ar_stop"] = "max_new_reached"
                reasons13 = SC._gate_serial("F13", f13_ser, reps=2, ar_ref=None)
                ck("CE3_历史F13_serial_被硬门拦下", bool(reasons13), reasons13[:2])
                ck("CE3_理由含D6缺失",
                   any("18_d6_mode_restored_after_cleanup" in x for x in reasons13),
                   reasons13[:4])
                ck("CE3_理由含非clean_complete",
                   any("complete_with_selfcheck_failures" in x for x in reasons13), reasons13[:4])
            else:
                ck("CE3_历史F13_serial_被硬门拦下", False, "无真实 F13 serial 产物可测")
        except Exception as e:                                            # noqa: BLE001
            ck("CE3_历史F13_serial_被硬门拦下", False, f"{type(e).__name__}: {str(e)[:120]}")
        # ---- CE4：production 分歧时 GB6 必需字段（来自独立诊断）缺失 ⇒ 不 PASS ----
        tmpCE4 = tmp / "ce4"; tmpCE4.mkdir()
        _write(tmpCE4, _full_set(TAG_GOOD, prod_variant=1))
        sCE4a = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCE4)   # 无诊断
        ck("CE4_分歧但无GB6诊断_不PASS", sCE4a["decision"] == "CHECK_NEEDS_REVIEW", sCE4a["decision"])
        ck("CE4_报缺诊断字段",
           any("GB6 必需字段**实际为空**" in x for x in sCE4a["needs_review"]), sCE4a["needs_review"][:3])
        gb6_full = _diag(TAG_GOOD)
        sCE4b = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCE4,
                            gb6_diagnostics=gb6_full)
        ck("CE4b_诊断齐全_可PASS", sCE4b["decision"] == "SERIAL_ORACLE_PASS", sCE4b["decision"])
        ck("CE4b_GB6记录含推导字段",
           all(sCE4b["per_family"][f]["gb6"]["runs_derived"].get("first_divergence_index") == 7
               for f in NEW),
           {f: sCE4b["per_family"][f]["gb6"]["runs_derived"].get("first_divergence_index")
            for f in NEW})
        ck("CE4b_绝对位置=prompt_len+下标",
           all(sCE4b["per_family"][f]["gb6"]["runs_derived"].get("absolute_position")
               == PROMPT_LEN + 7 for f in NEW),
           {f: sCE4b["per_family"][f]["gb6"]["runs_derived"].get("absolute_position") for f in NEW})
        ck("CE4b_轮号与两侧token均记账",
           all(isinstance(sCE4b["per_family"][f]["gb6"]["runs_derived"].get("round_index"), int)
               and sCE4b["per_family"][f]["gb6"]["runs_derived"].get("ar_token") is not None
               and sCE4b["per_family"][f]["gb6"]["runs_derived"].get("prod_token") is not None
               for f in NEW),
           {f: (sCE4b["per_family"][f]["gb6"]["runs_derived"].get("round_index"),
                sCE4b["per_family"][f]["gb6"]["runs_derived"].get("ar_token"),
                sCE4b["per_family"][f]["gb6"]["runs_derived"].get("prod_token")) for f in NEW})
        ck("CE4b_无残留缺失字段",
           all(not sCE4b["per_family"][f]["gb6"]["missing"] for f in NEW),
           {f: sCE4b["per_family"][f]["gb6"]["missing"] for f in NEW})
        gb6_wrongfam = {f: dict(gb6_full[f], family="F13") for f in NEW}       # 跨 family 迁移 ⇒ 拒绝
        sCE4c = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCE4,
                            gb6_diagnostics=gb6_wrongfam)
        ck("CE4c_跨family诊断被拒_不PASS", sCE4c["decision"] == "CHECK_NEEDS_REVIEW", sCE4c["decision"])
        # ---- CE4d：诊断位置与产物推导**不一致** ⇒ 不 PASS（位置交叉核对必须生效）----
        gb6_badpos = {f: dict(gb6_full[f], position=9) for f in NEW}
        sCE4d = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCE4,
                            gb6_diagnostics=gb6_badpos)
        ck("CE4d_诊断位置不符_不PASS", sCE4d["decision"] == "CHECK_NEEDS_REVIEW", sCE4d["decision"])
        ck("CE4d_报位置不一致",
           any("位置与产物推导不一致" in x for x in sCE4d["needs_review"]), sCE4d["needs_review"][:3])
        # ---- CE4e：诊断 tag 不符 ⇒ 拒绝（禁止跨配置迁移旧记录）----
        gb6_wrongtag = {f: dict(gb6_full[f], runs_tag="div8209_so") for f in NEW}
        sCE4e = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCE4,
                            gb6_diagnostics=gb6_wrongtag)
        ck("CE4e_跨tag诊断被拒_不PASS", sCE4e["decision"] == "CHECK_NEEDS_REVIEW", sCE4e["decision"])
        # ---- CE4f：诊断缺 `prompt_len`（**来源/配置**字段不全）⇒ 拒绝（本轮收紧：坐标口径
        #      的对照必须建立在"两侧同一 prompt_len"之上，缺了就不能采用）----
        gb6_nopl = {f: {k: v for k, v in gb6_full[f].items() if k != "prompt_len"} for f in NEW}
        sCE4f = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCE4,
                            gb6_diagnostics=gb6_nopl)
        ck("CE4f_缺prompt_len诊断被拒_不PASS", sCE4f["decision"] == "CHECK_NEEDS_REVIEW",
           sCE4f["decision"])
        ck("CE4f_理由为来源配置字段不全",
           any("来源/配置" in x and "prompt_len" in x for x in sCE4f["needs_review"]),
           sCE4f["needs_review"][:2])
        # ---- CE4g：位置不符（同坐标系：诊断 position 是输出序列内下标）⇒ 不 PASS ----
        gb6_idx_bad = {f: dict(gb6_full[f], position=9) for f in NEW}
        sCE4g = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCE4,
                            gb6_diagnostics=gb6_idx_bad)
        ck("CE4g_位置与产物不符_不PASS", sCE4g["decision"] == "CHECK_NEEDS_REVIEW", sCE4g["decision"])
        ck("CE4g_标注为输出下标坐标系",
           all(sCE4g["per_family"][x]["gb6"].get("position_frame") == "output_index" for x in NEW),
           {x: sCE4g["per_family"][x]["gb6"].get("position_frame") for x in NEW})
        # ---- CE4h：诊断**没有任何 tag** ⇒ 拒绝（不得默认"同配置"）----
        gb6_notag = {f: {k: v for k, v in gb6_full[f].items() if k != "runs_tag"} for f in NEW}
        sCE4h = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCE4,
                            gb6_diagnostics=gb6_notag)
        ck("CE4h_无tag诊断被拒_不PASS", sCE4h["decision"] == "CHECK_NEEDS_REVIEW", sCE4h["decision"])
        ck("CE4h_理由为无tag可核对",
           any("没有可核对的 tag" in x for x in sCE4h["needs_review"]), sCE4h["needs_review"][:2])
        # ---- CE4i：**文件顶层** runs_tag 必须下推到每条 family 记录（真实 margins 产物就是这个形态）----
        fp = tmp / "margins_real_shape.json"
        fp.write_text(json.dumps(dict(tool="correctness_protocol_margins", runs_tag=TAG_GOOD,
                                      families={f: {k: v for k, v in gb6_full[f].items()
                                                    if k != "runs_tag"} for f in NEW}),
                                 ensure_ascii=False), encoding="utf-8")
        loaded = SV.load_gb6_diagnostics(str(fp))
        ck("CE4i_顶层tag下推到各family",
           all((loaded.get(f) or {}).get("runs_tag") == TAG_GOOD for f in NEW),
           {f: (loaded.get(f) or {}).get("runs_tag") for f in NEW})
        sCE4i = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCE4,
                            gb6_diagnostics=loaded)
        ck("CE4i_文件式诊断可PASS", sCE4i["decision"] == "SERIAL_ORACLE_PASS", sCE4i["decision"])
        fp2 = tmp / "margins_wrong_tag.json"
        fp2.write_text(json.dumps(dict(tool="correctness_protocol_margins", runs_tag="div8209_proto",
                                       families={f: {k: v for k, v in gb6_full[f].items()
                                                     if k != "runs_tag"} for f in NEW}),
                                  ensure_ascii=False), encoding="utf-8")
        sCE4j = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCE4,
                            gb6_diagnostics=SV.load_gb6_diagnostics(str(fp2)))
        ck("CE4j_旧配置诊断(div8209_proto)被拒_不PASS",
           sCE4j["decision"] == "CHECK_NEEDS_REVIEW"
           and any("禁止跨配置迁移" in x for x in sCE4j["needs_review"]), sCE4j["decision"])
        # ---- CE4k：诊断的 prompt_len 与产物不一致（不同输入配置）⇒ 不 PASS ----
        gb6_badpl = {f: dict(gb6_full[f], prompt_len=4096) for f in NEW}
        sCE4k = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCE4,
                            gb6_diagnostics=gb6_badpl)
        ck("CE4k_prompt_len不符_不PASS", sCE4k["decision"] == "CHECK_NEEDS_REVIEW", sCE4k["decision"])
        ck("CE4k_报输入配置不一致",
           any("prompt_len 与产物不一致" in x for x in sCE4k["needs_review"]), sCE4k["needs_review"][:2])
        # ---- CE4l：AR 模式产物**没有** `smoke.rounds`（真实情形）时，决策轮号必须取自
        #      divergent（production）侧的 rounds，而**不是**变成 None ----
        tmpCE4L = tmp / "ce4l"; tmpCE4L.mkdir()
        recs = _full_set(TAG_GOOD, prod_variant=1)
        for r in recs:
            if r.get("mode") == "ar":
                r["smoke"].pop("rounds", None)          # 真实 AR 产物就是没有 rounds
        _write(tmpCE4L, recs)
        sCE4l = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCE4L,
                            gb6_diagnostics=gb6_full)
        rd_l = {f: sCE4l["per_family"][f]["gb6"]["runs_derived"] for f in NEW}
        ck("CE4l_AR无rounds时轮号取自prod侧",
           all(rd_l[f].get("round_index") is not None and rd_l[f].get("round_index_source") == "prod"
               and rd_l[f].get("round_index_ar") is None for f in NEW),
           {f: (rd_l[f].get("round_index"), rd_l[f].get("round_index_source"),
                rd_l[f].get("round_index_ar")) for f in NEW})
        ck("CE4l_该情形仍可PASS", sCE4l["decision"] == "SERIAL_ORACLE_PASS", sCE4l["decision"])
        # ---- CE5：环境拒绝**只报告**；有完整有效重复时不得因此阻断 ----
        tmpCE5 = tmp / "ce5"; tmpCE5.mkdir()
        extra_eb = _mk(TAG_GOOD, "F09", "ar", 1, model_run=False, env_blocked=True)
        extra_eb["run_id"] = "20261004T000009Z"
        _write(tmpCE5, _full_set(TAG_GOOD) + [extra_eb])
        sCE5 = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCE5)
        ck("CE5_环境拒绝不阻断（有效重复齐全）", sCE5["decision"] == "SERIAL_ORACLE_PASS", sCE5["decision"])
        ck("CE5_环境拒绝被单列报告", sCE5["n_env_blocked"] == 1 and len(sCE5["env_blocked_records"]) == 1,
           (sCE5["n_env_blocked"], sCE5["env_blocked_records"]))
        ck("CE5_不再是needs_review噪声",
           not any("环境拒绝" in x for x in sCE5["needs_review"]), sCE5["needs_review"][:3])

        # ================= 本轮（CF 系列）：三个**已复现**判决漏洞的回归 =================
        # ---- CF1：运行侧 GB6 字段缺失（`missing=['absolute_position']`）必须阻断 ----
        #      旧实现算出了 `missing` 却**从不使用**它 ⇒ 记录不完整仍然 PASS。
        tmpCF1 = tmp / "cf1"; tmpCF1.mkdir()
        recs_cf1 = _full_set(TAG_GOOD, prod_variant=1)
        for r in recs_cf1:
            r["smoke"]["end_state"].pop("prompt_len", None)   # ⇒ absolute_position 推导不出
        _write(tmpCF1, recs_cf1)
        sCF1 = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCF1,
                           gb6_diagnostics=gb6_full)
        g1 = sCF1["per_family"]["F09"]["gb6"]
        ck("CF1_旧实现会PASS的情形现在不PASS", sCF1["decision"] == "CHECK_NEEDS_REVIEW",
           sCF1["decision"])
        ck("CF1_missing确含absolute_position", "absolute_position" in (g1.get("missing") or []),
           g1.get("missing"))
        ck("CF1_阻断理由点名运行侧缺字段",
           any("GB6 必需字段**实际为空**" in x and "absolute_position" in x
               for x in sCF1["needs_review"]), sCF1["needs_review"][:3])
        ck("CF1_运行侧缺失被单列", "absolute_position" in (g1.get("from_runs_missing") or []),
           g1.get("from_runs_missing"))
        # ---- CF1b：字典存在但**关键值全为 None** 不算字段齐全（AR 工作量） ----
        tmpCF1b = tmp / "cf1b"; tmpCF1b.mkdir()
        recs_cf1b = _full_set(TAG_GOOD, prod_variant=1)
        for r in recs_cf1b:
            if r.get("mode") == "ar":
                r["smoke"]["end_state"].pop("n_decode_steps", None)     # 无 decode steps
                r["smoke"]["specdec_stats"].pop("decode_steps", None)
                r["audit"]["target"].pop("n_generation_calls", None)    # 无 target calls
        _write(tmpCF1b, recs_cf1b)
        sCF1b = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCF1b,
                            gb6_diagnostics=gb6_full)
        g1b = sCF1b["per_family"]["F09"]["gb6"]
        ck("CF1b_AR工作量关键值全None不算齐全",
           {"ar_work.decode_steps", "ar_work.target_gen_calls"} <= set(g1b.get("missing") or []),
           g1b.get("missing"))
        ck("CF1b_该情形不PASS", sCF1b["decision"] == "CHECK_NEEDS_REVIEW", sCF1b["decision"])
        # ---- CF1c：AR 工作量必须按 decode steps/target calls 记录，且**不填造接受率** ----
        sCF1c = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCE4,
                            gb6_diagnostics=gb6_full)
        aw = sCF1c["per_family"]["F09"]["gb6"]["runs_derived"].get("ar_work") or {}
        ck("CF1c_AR工作量记decode_steps与target_calls",
           aw.get("decode_steps") is not None and aw.get("target_gen_calls") is not None, aw)
        ck("CF1c_AR不填造接受率",
           aw.get("proposed") is None and aw.get("accepted") is None
           and "not_applicable" in str(aw.get("acceptance")), aw)
        # ---- CF1d：无分歧时**无需** GB6 诊断（`missing` 必须为空、不得阻断）----
        tmpCF1d = tmp / "cf1d"; tmpCF1d.mkdir()
        _write(tmpCF1d, _full_set(TAG_GOOD))              # prod 与 AR 逐 token 相同
        sCF1d = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCF1d)
        g1d = sCF1d["per_family"]["F09"]["gb6"]
        ck("CF1d_无分歧时missing为空", not g1d.get("missing"), g1d.get("missing"))
        ck("CF1d_无分歧时无需诊断即可PASS", sCF1d["decision"] == "SERIAL_ORACLE_PASS",
           sCF1d["decision"])
        # ---- CF2：新 family AR 的 `smoke.stop_reason` 与 `end_state.stop_reason` 矛盾必须阻断 ----
        tmpCF2 = tmp / "cf2"; tmpCF2.mkdir()
        recs_cf2 = _full_set(TAG_GOOD)
        for r in recs_cf2:
            if r.get("mode") == "ar":
                r["smoke"]["end_state"]["stop_reason"] = "eos"     # smoke 说 max_new_reached
        _write(tmpCF2, recs_cf2)
        sCF2 = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCF2)
        pf2 = sCF2["per_family"]["F03"]
        ck("CF2_旧实现会PASS的情形现在不PASS", sCF2["decision"] == "CHECK_NEEDS_REVIEW",
           sCF2["decision"])
        ck("CF2_矛盾被列为自报状态矛盾",
           bool(pf2["ar_checks"].get("per_row")) and
           all(rr["contradictions"].get("stop_smoke_vs_end_state")
               for rr in pf2["ar_checks"]["per_row"]), pf2["ar_checks"].get("per_row"))
        ck("CF2_stop_matches_end_state检查为False",
           pf2["ar_checks"].get("stop_matches_end_state") is False,
           pf2["ar_checks"].get("stop_matches_end_state"))
        ck("CF2_理由为自报状态相互矛盾",
           any("自报状态相互矛盾" in x for x in sCF2["needs_review"]), sCF2["needs_review"][:3])
        ck("CF2_不误判为基线回归", not sCF2["baseline_failures"], sCF2["baseline_failures"][:2])
        # ---- CF2b：新 family AR **缺** `end_state.stop_reason` ⇒ 归为"证据缺失" ----
        tmpCF2b = tmp / "cf2b"; tmpCF2b.mkdir()
        recs_cf2b = _full_set(TAG_GOOD)
        for r in recs_cf2b:
            if r.get("mode") == "ar":
                r["smoke"]["end_state"].pop("stop_reason", None)
        _write(tmpCF2b, recs_cf2b)
        sCF2b = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCF2b)
        ck("CF2b_缺stop_reason归为证据缺失_不PASS",
           sCF2b["decision"] == "CHECK_NEEDS_REVIEW"
           and any("证据缺失" in x and "end_state.stop_reason" in x
                   for x in sCF2b["needs_review"]), sCF2b["needs_review"][:3])
        # ---- CF2c：新 family AR 的独立 hash 与 end_state hash 矛盾 ⇒ 阻断 ----
        tmpCF2c = tmp / "cf2c"; tmpCF2c.mkdir()
        recs_cf2c = _full_set(TAG_GOOD)
        for r in recs_cf2c:
            if r.get("mode") == "ar":
                r["smoke"]["end_state"]["gen_hash"] = "deadbeefdeadbeef"
        _write(tmpCF2c, recs_cf2c)
        sCF2c = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCF2c)
        ck("CF2c_hash矛盾被列为矛盾并阻断",
           sCF2c["decision"] == "CHECK_NEEDS_REVIEW"
           and any("自报状态相互矛盾" in x for x in sCF2c["needs_review"]),
           sCF2c["needs_review"][:2])
        # ---- CF2d：两次重复之间不一致仍按"基线回归"（确定性失败，分类不能被改坏）----
        #      注意：必须把该 rep 的**自报字段一起改**，否则同时制造出"自报矛盾"，
        #      测的就不是"两次之间不一致"这一类了（下面的 CF2e 专门测两者的优先级）。
        tmpCF2d = tmp / "cf2d"; tmpCF2d.mkdir()
        recs_cf2d = _full_set(TAG_GOOD)
        for r in recs_cf2d:
            if r.get("mode") == "ar" and int(r.get("rep") or 0) == 2:
                g2 = [x + 1 for x in r["smoke"]["gen"]]
                r["smoke"]["gen"] = g2
                r["smoke"]["end_state"]["gen_hash"] = SV.AGG._gen_hash(g2)
        _write(tmpCF2d, recs_cf2d)
        sCF2d = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCF2d)
        ck("CF2d_两次重复不一致=BASELINE_REGRESSION",
           sCF2d["decision"] == "BASELINE_REGRESSION", sCF2d["decision"])
        ck("CF2d_分类为两次之间不一致",
           any("两次" in x for x in sCF2d["baseline_failures"]), sCF2d["baseline_failures"][:2])
        # ---- CF2e：既"两次不一致"又"自报矛盾"时，`CHECK_NEEDS_REVIEW`（证据不可信）优先于
        #      `BASELINE_REGRESSION`：**不得**在记录本身矛盾时声称"基线回归" ----
        tmpCF2e = tmp / "cf2e"; tmpCF2e.mkdir()
        recs_cf2e = _full_set(TAG_GOOD)
        for r in recs_cf2e:
            if r.get("mode") == "ar" and int(r.get("rep") or 0) == 2:
                r["smoke"]["gen"] = [x + 1 for x in r["smoke"]["gen"]]   # 自报字段**不**跟着改
        _write(tmpCF2e, recs_cf2e)
        sCF2e = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCF2e)
        ck("CF2e_自报矛盾优先于基线回归",
           sCF2e["decision"] == "CHECK_NEEDS_REVIEW"
           and any("自报状态相互矛盾" in x for x in sCF2e["needs_review"]), sCF2e["decision"])
        # ---- CF3：诊断**位置来源**必须自证为首分歧；"末 token"口径的记录必须被拒 ----
        gb6_legacy = _diag(TAG_GOOD, position_origin="legacy_last_index（无分歧 ⇒ 末位；非首分歧）")
        sCF3 = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCE4,
                           gb6_diagnostics=gb6_legacy)
        ck("CF3_末token口径诊断被拒_不PASS", sCF3["decision"] == "CHECK_NEEDS_REVIEW",
           sCF3["decision"])
        ck("CF3_拒绝理由点名position_origin",
           any("position_origin" in x and "first_divergence" in x for x in sCF3["needs_review"]),
           sCF3["needs_review"][:2])
        # CF3b：诊断缺 position_origin（旧 margins 产物就是这种形态）⇒ 拒绝
        gb6_noorigin = {f: {k: v for k, v in gb6_full[f].items() if k != "position_origin"}
                        for f in NEW}
        sCF3b = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCE4,
                            gb6_diagnostics=gb6_noorigin)
        ck("CF3b_缺position_origin被拒_不PASS", sCF3b["decision"] == "CHECK_NEEDS_REVIEW",
           sCF3b["decision"])
        # CF3c：两侧 token 不一致 ⇒ 拒绝（运行侧与诊断侧必须相互一致）
        gb6_badtok = {f: dict(gb6_full[f], ar_token=999999) for f in NEW}
        sCF3c = SV.evaluate(TAG_GOOD, list(NEW), list(MODES), REPS, outdir=tmpCE4,
                            gb6_diagnostics=gb6_badtok)
        ck("CF3c_两侧token不一致被拒_不PASS", sCF3c["decision"] == "CHECK_NEEDS_REVIEW",
           sCF3c["decision"])
        ck("CF3c_理由点名token不一致",
           any("token 与运行侧产物不一致" in x for x in sCF3c["needs_review"]),
           sCF3c["needs_review"][:2])
        # CF3d：位置正确 + 两侧字段齐全 ⇒ **完整记账**并放行（正向用例）----
        rd_ok = sCE4b["per_family"]["F09"]["gb6"]
        ck("CF3d_正向：两侧字段齐全且无残留缺失",
           (not rd_ok.get("missing")) and (not rd_ok.get("from_runs_missing"))
           and rd_ok.get("from_diagnostic_available") == sorted(SV.GB6_FROM_DIAGNOSTIC),
           (rd_ok.get("missing"), rd_ok.get("from_runs_missing"),
            rd_ok.get("from_diagnostic_available")))
        ck("CF3d_正向：位置与两侧token全部对得上",
           rd_ok["runs_derived"]["first_divergence_index"] == rd_ok["diagnostic_provenance"]["position"]
           and rd_ok["runs_derived"]["round_index"] is not None, rd_ok.get("runs_derived"))
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
