"""Gate 2（GPU correctness gate，非计时）：all_accept_oracle 在 12 个开发条件上的一次性检查。

检查项：
  * 有限性（propose/verify 的每次 _run 后 isfinite；异常即失败）；
  * cache 一致性：逐轮 tgt_len == drf_len == pre + len(emitted)；末轮 = ctx + len(gen)；
  * `spec_cache.check_consistent` 通过；
  * attention 策略断言（`assert_strategy_used`）通过；
  * 每轮 emitted == candidates（所有非 EOS 候选确实被接受）；
  * rollback / mismatch / repair 计数 == 0；acceptance_rate == 1.0（除非 EOS/max_new 截断）；
  * 轮数与理论下界的关系：当前契约 m==kp 只发出 kp 个 candidate（不额外发 bonus token），
    因此 64 token / k=4 的理论最少轮数 = ceil(64/4) = 16（不是 13；13 只在每轮发 5 个 token 时成立）。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from experiments.formal import worker as W  # noqa: E402
from experiments.formal.environment import check_environment, check_gpu_environment  # noqa: E402
from experiments.formal.runtime_info import collect_runtime_info  # noqa: E402
from experiments.relaxed_specdec import all_accept_oracle as A  # noqa: E402
from specdec.engine import model_self, target_rope_scope  # noqa: E402

FAMILIES = ("F09", "F03", "F11", "F04")
CONTEXTS = (2048, 4096, 8192)
K = 4
MAX_NEW = 64


def expected_rounds(gen_len: int) -> int:
    return (gen_len + K - 1) // K


def wait_for_memory(min_gib: float = 16.1, timeout_s: int = 3600) -> float:
    from specdec.engine.g5_8k_spec_variance_probe import system_memory
    import time as _time
    deadline = _time.time() + timeout_s
    last = None
    while _time.time() < deadline:
        m = system_memory()
        if m.get("available"):
            last = m["commit_free_mib"] / 1024.0
            if last >= min_gib:
                return last
        _time.sleep(20)
    raise RuntimeError(f"等待内存恢复超时（最后 commit free={last} GiB）")


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
                model_self.reset_attention_stats()
                result = A.decode(target, draft, prompt, MAX_NEW, k=K, eos_id=eos,
                                  finite_check=True, capture_cache=True)
                model_self.assert_strategy_used()
                stats = result["stats"]
                control = result["control"]
                gen_len = len(result["gen"])
                fc = result["final_cache"]
                round_ok = all(row["tgt_len"] == row["drf_len"] == row["pre"] + len(row["emitted"])
                               for row in control)
                emitted_ok = all(row["emitted"] == row["cands"] for row in control)
                counts_ok = (stats["mismatch_rounds"] == 0 and stats["repair_count"] == 0
                             and stats["rollback_count"] == 0)
                acceptance_ok = (stats["accepted"] == stats["proposed"])
                ok = (fc["tgt_len"] == ctx + gen_len and fc["drf_len"] == ctx + gen_len
                      and fc["tgt_consistent"] and fc["drf_consistent"]
                      and round_ok and emitted_ok and counts_ok and acceptance_ok
                      and stats["rounds"] == expected_rounds(gen_len))
                row = dict(family=family, context=ctx, input_sha1=meta["token_sha1"],
                           gen_len=gen_len, stop=("eos" if (result["gen"] and result["gen"][-1] == eos) else "max_new_reached"),
                           rounds=stats["rounds"], expected_rounds=expected_rounds(gen_len),
                           acceptance_rate=stats["accepted"] / stats["proposed"],
                           mismatch_rounds=stats["mismatch_rounds"],
                           repair_count=stats["repair_count"],
                           rollback_count=stats["rollback_count"],
                           distinct_emitted=sum(1 for r in control if r["emitted"] != r["cands"]),
                           round_ok=round_ok, emitted_ok=emitted_ok,
                           cache_ok=bool(fc["tgt_consistent"] and fc["drf_consistent"]
                                         and fc["tgt_len"] == ctx + gen_len),
                           final_cache=fc, ok=bool(ok))
                rows.append(row)
                print(f"{'OK' if ok else 'FAIL'} {family} {ctx}: gen={gen_len} rounds={stats['rounds']} "
                      f"(expect {expected_rounds(gen_len)}) acceptance={row['acceptance_rate']:.3f} "
                      f"cache={fc['tgt_len']}/{fc['drf_len']} rounds_ok={round_ok} emitted_ok={emitted_ok}",
                      flush=True)
                installed = True
    finally:
        if installed:
            target_rope_scope.restore()

    result = dict(schema_version="all-accept-gate-v1",
                  utc=datetime.now(timezone.utc).isoformat(),
                  environment=environment, runtime=collect_runtime_info(),
                  load=dict(target=target_info, draft=draft_info), rope=rope,
                  rows=rows, all_ok=all(r["ok"] for r in rows),
                  theory=dict(k=K, max_new=MAX_NEW,
                              note=("当前契约 m==kp 只发出 kp 个 candidate（不额外发 bonus token）；"
                                    "因此 64 token 的理论最少轮数 = ceil(64/4) = 16；"
                                    "13 轮仅适用于每轮发 5 个 token（kp + bonus）的契约")))
    (out / "gate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                   encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    free = wait_for_memory()
    print(f"commit free={free:.2f} GiB", flush=True)
    result = run_gate(args.out)
    print("ALL_OK" if result["all_ok"] else "GATE_FAILED")
    return 0 if result["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
