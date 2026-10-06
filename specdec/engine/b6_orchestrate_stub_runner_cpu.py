"""b6_orchestrate_stub_runner_cpu.py — **仅供无 GPU 编排自测**的假运行器。

## 纪律（防伪造）
* 本模块**不是**取证工具：它写出的产物**不是**任何真实 run 的证据；
* 双重结构保护：`tag` 必须以 `SELFTEST` 开头、且必须显式给出 `STUB_ORCH_OUTDIR`，
  否则**拒绝运行**（退出码 3），不会往 `output/g5/attn` 或任何真实 tag 写文件；
* 产物**逐字复用** `b6_branch_coverage_selftest_cpu` 的合成夹具构造器（`_mk`/`_round`），
  因此与服务端验收器（CHK）的字段口径始终一致，不会出现"假夹具自己漂移"。

## 用 `STUB_ORCH_SCENARIO` 选择场景（每个场景对应一条编排验收路径）
`ok` · `hardfail`（硬检查失败）· `missing_rep2`（rep2 不写产物）· `duplicate`（同槽位两份产物）·
`repmismatch`（rep2 序列不同）· `serialmismatch`（serial≠AR）· `prodonly`（只有 production 不同）·
`nofuse`（候选臂但无融合、有回退理由）· `covgap`（声明分支未触发）· `rc_nonzero`（非 0 退出）。
`arm` 取自 `SO_ARM`（**不**硬编码），因此本自测同时验证 `so_batch.ps1` 是否真的把 `SO_ARM`/`SO_MAXNEW`/
`SO_PROMPT` 传到了运行器。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from .b6_branch_coverage_selftest_cpu import _mk, _round

SCEN = (os.environ.get("STUB_ORCH_SCENARIO") or "ok").strip()
TAG = (os.environ.get("SO_TAG") or "").strip()
CASE = (os.environ.get("SO_CASE") or "").strip()
MODE = (os.environ.get("SO_MODE") or "").strip().lower()
REP = int(os.environ.get("SO_REP") or "1")
MAXNEW = int(os.environ.get("SO_MAXNEW") or "64")
ARM = os.environ.get("SO_ARM") or "eager"
PROMPT = os.environ.get("SO_PROMPT") or ""
PROMPT_ID = (os.environ.get("SO_PROMPT_ID") or "").strip()
OUTDIR = (os.environ.get("STUB_ORCH_OUTDIR") or "").strip()

SCENARIOS = ("ok", "hardfail", "missing_rep2", "duplicate", "repmismatch", "serialmismatch",
             "prodonly", "nofuse", "covgap", "rc_nonzero")


def _gen() -> list:
    g = list(range(200, 200 + MAXNEW))
    if SCEN == "repmismatch" and REP == 2:
        g[4] = 888
    if SCEN == "serialmismatch" and MODE == "serial":
        g[3] = 777
    if SCEN == "prodonly" and MODE == "prod":
        g[5] = 4242
    return g


def main() -> int:
    if not TAG.startswith("SELFTEST"):
        print(f"[stub] 拒绝：tag={TAG!r} 不以 SELFTEST 开头（防止伪造真实取证）", file=sys.stderr)
        return 3
    if not OUTDIR:
        print("[stub] 拒绝：未给出 STUB_ORCH_OUTDIR（不往默认产物目录写）", file=sys.stderr)
        return 3
    if SCEN not in SCENARIOS:
        print(f"[stub] 未知场景 {SCEN!r}（可选 {SCENARIOS}）", file=sys.stderr)
        return 3
    if SCEN == "rc_nonzero":
        print("[stub] 场景 rc_nonzero：模拟运行器非 0 退出", file=sys.stderr)
        return 3
    if SCEN == "missing_rep2" and REP == 2:
        print("[stub] 场景 missing_rep2：rep2 故意不写产物（退出码仍为 0）")
        return 0
    rounds = ([_round(0, 4, 4)] if SCEN == "covgap" else [_round(0, 4, 4), _round(1, 3, 2)])
    kw: dict = {}
    if SCEN == "nofuse":
        kw.update(fused_by_phase={},
                  fallbacks=[dict(phase="decode", s=1, reason="phase='decode' != 'prefill'",
                                  kind="phase_not_prefill")],
                  reasons={"phase_not_prefill": 30})
    if SCEN == "hardfail":
        kw.update(self_ok=False)
    ids_hash = (hashlib.sha1(PROMPT.encode("utf-8")).hexdigest()[:16] if PROMPT
                else "stubhash" + hashlib.sha1(CASE.encode()).hexdigest()[:8])
    rec = _mk(CASE, MODE, REP, gen=_gen(), rounds=rounds, stop="max_new_reached",
              ctx=12, k=4, max_new=MAXNEW, arm=ARM, ids_hash=ids_hash,
              # ⚠️ 记录内的 tag 必须是**请求的假 tag**（`_mk` 默认写死真实 tag）：
              #    否则 `find_runs` 的身份核对会（正确地）判"记录内 tag ≠ 请求" ⇒ 自测假失败。
              tag_in_record=TAG,
              source_kind=("literal_prompt" if PROMPT else "manifest"),
              overrides=({"smoke.input.prompt_id": PROMPT_ID} if PROMPT_ID else None), **kw)
    d = Path(OUTDIR)
    d.mkdir(parents=True, exist_ok=True)
    names = ["{0}_{1}_{2}_rep{3}_STUB{4}.json".format(TAG, CASE, MODE, REP, s)
             for s in (("A", "B") if SCEN == "duplicate" else ("A",))]
    for n in names:
        (d / n).write_text(json.dumps(rec, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[stub] 场景 {SCEN}：写入 {len(names)} 份产物 tag={TAG} case={CASE} mode={MODE} "
          f"rep={REP} arm={ARM} max_new={MAXNEW} prompt_id={PROMPT_ID or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
