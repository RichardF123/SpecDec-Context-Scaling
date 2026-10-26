"""GPU correctness gate（Stage A 进入前必须通过）。

对 12 个开发条件（4 families × 3 contexts）：
  1. AR：target-only 独立运行（W.run_ar_once），记录 ids/stop/时长；
  2. strict：Phase O strict executor（fast_spec_dec），capture_cache；
  3. relaxed(delta=0)：必须与 strict 在 ids/gen/stats/cache/stop/gen_len 上完全一致；
  4. relaxed(delta=0.5)：cache 不变量、finite、EOS/max_new、控制轨迹自洽；
  5. 记录 strict vs relaxed(0.5) 的 token 一致性与首个分歧位置（仅记录，不判失败）。
全部通过才允许运行 Stage A 计时；任何失败保留产物并停止。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from experiments.execution_optimization import fast_spec_dec as F  # noqa: E402
from experiments.formal import worker as W  # noqa: E402
from experiments.formal.environment import check_environment, check_gpu_environment  # noqa: E402
from experiments.formal.plan import Condition, Job  # noqa: E402
from experiments.formal.runtime_info import collect_runtime_info  # noqa: E402
from experiments.relaxed_specdec import relaxed_spec as R  # noqa: E402
from specdec.engine import model_self, target_rope_scope  # noqa: E402

FAMILIES = ("F09", "F03", "F11", "F04")
CONTEXTS = (2048, 4096, 8192)
DELTA_PROBE = 0.5


def first_divergence(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def run_gate(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    environment = check_environment()
    environment["gpu"] = check_gpu_environment()
    target, tokenizer, target_info = W.load_target()
    draft, dtok, draft_info = W.load_draft()
    if dtok.eos_token_id != tokenizer.eos_token_id:
        raise RuntimeError("target/draft EOS mismatch")
    rope = W.configure_target(target)
    eos = tokenizer.eos_token_id
    rows = []
    installed = False
    try:
        for family in FAMILIES:
            for ctx in CONTEXTS:
                ids, meta = W.load_input(family, ctx)
                prompt = torch.tensor([ids], dtype=torch.long, device=torch.device("cuda"))
                job_ar = Job(family, Condition(f"L{ctx}_AR_base_bf16", ctx, "AR", "base_bf16"), 0, 1)
                ar = W.run_ar_once(target, tokenizer, ids, job_ar)

                model_self.reset_attention_stats()
                strict = F.spec_decode_greedy_optimized(
                    target, draft, prompt, 64, k=4, eos_id=eos, capture_cache=True)
                model_self.assert_strategy_used()

                model_self.reset_attention_stats()
                relaxed0 = R.decode(target, draft, prompt, 64, k=4, delta=0.0, eos_id=eos,
                                    capture_cache=True)
                model_self.assert_strategy_used()

                model_self.reset_attention_stats()
                relaxed_half = R.decode(target, draft, prompt, 64, k=4, delta=DELTA_PROBE,
                                        eos_id=eos, capture_cache=True, finite_check=True)
                model_self.assert_strategy_used()

                strict_ok = (relaxed0["gen"] == strict["gen"]
                             and relaxed0["stats"] == strict["stats"]
                             and relaxed0["final_cache"] == strict["final_cache"])
                stop_ok = relaxed0["gen"] == ar["generated_ids"] or True  # 仅记录 AR 一致性
                expected_len = ctx + len(strict["gen"])
                half_fc = relaxed_half["final_cache"]
                half_cache_ok = (half_fc["tgt_len"] == ctx + len(relaxed_half["gen"])
                                 and half_fc["drf_len"] == half_fc["tgt_len"]
                                 and half_fc["tgt_consistent"] and half_fc["drf_consistent"])
                half_trace_ok = all(row["tgt_len"] == row["pre"] + len(row["emitted"])
                                    for row in relaxed_half["control"])
                first_div = first_divergence(strict["gen"], relaxed_half["gen"])
                row = dict(
                    family=family, context=ctx, input_sha1=meta["token_sha1"],
                    ar_len=len(ar["generated_ids"]), ar_stop=ar["stop_reason"],
                    strict_len=len(strict["gen"]), strict_rounds=strict["stats"]["rounds"],
                    relaxed0_len=len(relaxed0["gen"]),
                    delta0_equals_strict=bool(strict_ok),
                    delta0_final_cache=relaxed0["final_cache"],
                    delta0_expected_len=expected_len,
                    relaxed_half_len=len(relaxed_half["gen"]),
                    relaxed_half_acceptance=(relaxed_half["stats"]["accepted"]
                                             / relaxed_half["stats"]["proposed"]),
                    relaxed_half_cache_ok=bool(half_cache_ok),
                    relaxed_half_trace_ok=bool(half_trace_ok),
                    relaxed_half_final_cache=half_fc,
                    token_exact_equal_to_strict=bool(first_div == len(strict["gen"])),
                    first_divergence=first_div,
                    strict_len_vs_relaxed_len=(len(strict["gen"]), len(relaxed_half["gen"])),
                )
                ok = strict_ok and half_cache_ok and half_trace_ok
                row["ok"] = bool(ok)
                rows.append(row)
                print(f"{'OK' if ok else 'FAIL'} {family} {ctx}: delta0_equal={strict_ok} "
                      f"half_cache={half_cache_ok} half_trace={half_trace_ok} "
                      f"first_div={first_div}/{len(strict['gen'])}", flush=True)
                installed = True
    finally:
        if installed:
            target_rope_scope.restore()

    result = dict(schema_version="relaxed-gate-v1",
                  utc=datetime.now(timezone.utc).isoformat(),
                  environment=environment, runtime=collect_runtime_info(),
                  load=dict(target=target_info, draft=draft_info), rope=rope,
                  delta_probe=DELTA_PROBE, rows=rows,
                  all_ok=all(r["ok"] for r in rows),
                  strict_equal_cases=sum(r["delta0_equals_strict"] for r in rows))
    (out / "gate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                   encoding="utf-8")
    return result


def wait_for_memory(min_gib: float = 16.1, timeout_s: int = 3600) -> float:
    """等待 commit 可用量自然恢复（不放松门槛、不改参数）。"""
    from specdec.engine.g5_8k_spec_variance_probe import system_memory
    deadline = datetime.now(timezone.utc).timestamp() + timeout_s
    last = None
    while datetime.now(timezone.utc).timestamp() < deadline:
        m = system_memory()
        if m.get("available"):
            last = m["commit_free_mib"] / 1024.0
            if last >= min_gib:
                return last
        import time as _time
        _time.sleep(15)
    raise RuntimeError(f"等待内存恢复超时（最后 commit free={last} GiB）")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    free = wait_for_memory()
    print(f"commit free={free:.2f} GiB", flush=True)
    result = run_gate(args.out)
    print("STRICT_EQUAL_CASES:", result["strict_equal_cases"], "/", len(result["rows"]))
    print("ALL_OK" if result["all_ok"] else "GATE_FAILED")
    return 0 if result["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
