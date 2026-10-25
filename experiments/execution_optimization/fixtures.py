"""Phase O1：为 4 个开发 family × 3 个上下文建立 reference fixture。

fixture 内容（每个 (family, ctx) 一条）：
  * input token_sha1（与 manifest 校验过的）；
  * AR generated ids / stop；
  * reference SpecDec generated ids / stop；
  * reference SpecDec 每轮控制轨迹（Kp/m/pre/cands/g/emitted/tgt_len/drf_len）；
  * stats（rounds/proposed/accepted/mismatch/rollback/block calls）；
  * attention 策略统计（计数与 s_max_by_phase）；
  * 有限性/缓存不变量：由 `diagnose_spec`（hooks）在所有 64 步上强制检查；
  * final cache 长度（= ctx + generated_length 的逐轮不变量）。

fixture 只用于 O2 的等价性判定，**不作为任何性能结果**。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from experiments.formal import worker as W  # noqa: E402
from experiments.formal.diagnostics import diagnose_spec  # noqa: E402
from experiments.formal.environment import check_environment, check_gpu_environment  # noqa: E402
from experiments.formal.plan import Condition, Job  # noqa: E402
from experiments.formal.runtime_info import collect_runtime_info  # noqa: E402
from experiments.formal.worker import test_chunked_prefill  # noqa: E402
from specdec.engine import model_self, target_rope_scope  # noqa: E402

from experiments.execution_optimization import config  # noqa: E402

FAMILIES = config.FROZEN["dev_families"]
CONTEXTS = config.FROZEN["contexts"]


def prefill_impl(model, ids):
    logits, cache = test_chunked_prefill(model, ids, chunk=2048, diagnostic=False)
    return logits.unsqueeze(1), cache


def build(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    env = check_environment()
    gpu = check_gpu_environment()
    target, tokenizer, target_info = W.load_target()
    draft, dtok, draft_info = W.load_draft()
    if dtok.eos_token_id != tokenizer.eos_token_id:
        raise RuntimeError("draft/target EOS 不一致")
    rope = W.configure_target(target)
    eos_id = tokenizer.eos_token_id
    manifest = []

    try:
        for family in FAMILIES:
            for ctx in CONTEXTS:
                ids, input_meta = W.load_input(family, ctx)
                job_ar = Job(family, Condition(f"L{ctx}_AR_base_bf16", ctx, "AR", "base_bf16"), 0, 1)
                ar = W.run_ar_once(target, tokenizer, ids, job_ar)
                job_sp = Job(family, Condition(f"L{ctx}_SpecDec_base_bf16", ctx, "SpecDec", "base_bf16"), 0, 1)
                runtime_config = W.check_runtime_config(target, draft, job_sp)
                sp = W.run_once(target, draft, tokenizer, ids, job_sp)
                inp = torch.tensor([ids], dtype=torch.long, device=torch.device("cuda"))
                diag = diagnose_spec(target, draft, inp, eos_id, prefill_impl)
                if diag["generated_ids"] != sp["generated_ids"]:
                    raise RuntimeError(f"{family}_{ctx}: diagnose 与 run_once 输出不一致")
                if diag["stats"] != sp["spec_stats"]:
                    raise RuntimeError(f"{family}_{ctx}: diagnose 与 run_once stats 不一致")

                trace = diag["trace"]
                for row in trace:
                    if row["tgt_len"] != row["pre"] + len(row["emitted"]):
                        raise RuntimeError("fixture 逐轮 cache 长度不变量失败")
                    if row["drf_len"] != row["tgt_len"]:
                        raise RuntimeError("fixture draft/target cache 长度不一致")
                last = trace[-1] if trace else None
                if last is not None:
                    expected_len = ctx + len(diag["generated_ids"])
                    if last["tgt_len"] != expected_len:
                        raise RuntimeError("fixture 末轮 cache 长度 != ctx+gen")

                fixture = dict(
                    schema_version="ot-reference-fixture-v1",
                    family=family, context_tokens=ctx,
                    input=dict(token_sha1=input_meta["token_sha1"],
                               hash_verified=input_meta.get("hash_verified"),
                               tokens=ctx),
                    configuration=runtime_config,
                    eos_id=eos_id,
                    ar=dict(generated_ids=ar["generated_ids"],
                            generated_length=ar["generated_length"],
                            stop_reason=ar["stop_reason"]),
                    specdec_reference=dict(
                        generated_ids=diag["generated_ids"],
                        generated_length=len(diag["generated_ids"]),
                        stop_reason=sp["stop_reason"],
                        stats=diag["stats"],
                        trace=trace,
                        final_tgt_len=last["tgt_len"] if last else ctx,
                        final_drf_len=last["drf_len"] if last else ctx,
                    ),
                    attention_stats=dict(
                        calls=diag["attention_stats"]["calls"],
                        by_phase=diag["attention_stats"]["by_phase"],
                        s_max_by_phase=diag["attention_stats"]["s_max_by_phase"],
                        eager_by_phase=diag["attention_stats"]["eager_by_phase"],
                        fused_by_phase=diag["attention_stats"]["fused_by_phase"],
                        strategy=diag["attention_stats"]["strategy"],
                    ),
                    decision_coverage=diag["decision_coverage"],
                )
                path = out / "fixtures" / f"{family}_{ctx}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")
                manifest.append(dict(family=family, context=ctx, file=str(path),
                                     ar_len=len(ar["generated_ids"]),
                                     spec_len=len(diag["generated_ids"]),
                                     same_ar_spec=ar["generated_ids"] == diag["generated_ids"]))
                print(f"fixture {family} ctx={ctx}: AR={len(ar['generated_ids'])} "
                      f"Spec={len(diag['generated_ids'])} rounds={diag['stats']['rounds']}", flush=True)
    finally:
        target_rope_scope.restore()

    meta = dict(
        schema_version="ot-fixtures-manifest-v1",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        families=list(FAMILIES), contexts=list(CONTEXTS),
        environment=dict(commit=env, gpu=gpu),
        load=dict(target=target_info, draft=draft_info),
        rope=rope,
        runtime=collect_runtime_info(),
        frozen=config.FROZEN,
        execution_mode=config.execution_mode(),
        cases=manifest,
    )
    (out / "manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or (ROOT / "output" / "specdec_execution_optimization" /
                       (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]))
    meta = build(out)
    print("OUTPUT:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
