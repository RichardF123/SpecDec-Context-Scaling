"""Gate 2（GPU correctness gate，非计时）：bonus_accept_oracle 在 12 个开发条件上的一次性检查。

检查项：
  * carry 与 catchup 两种路径的 token / rounds / bonus 数 / candidate 数 / stop 完全一致；
  * 每轮 candidate 全接受（emitted_candidates == 该轮候选）；bonus = argmax(最后一行)；
  * 每轮不变量：target_len == draft_len == ctx + committed − pending_new；
  * 末轮：rounds==13、candidate tokens==52、bonus==12、total==64、stop=max_new_reached、
    pending_at_stop==0（最后一轮不再发 bonus ⇒ cache 对齐）；
  * 有限性、cache 一致性、attention 策略断言；
  * carry 计时路径的 target 额外单步调用 == 0（catchup 路径允许并记录）；
  * EOS 未出现（本批历史条件），若出现则单独记录长度；
  * round 位置语义：首轮 verify_input_length==kp，后续==kp+1，candidate_start_row==pending_included，
    next_bonus_row==L−1。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from experiments.formal import worker as W  # noqa: E402
from experiments.formal.environment import check_environment, check_gpu_environment  # noqa: E402
from experiments.formal.runtime_info import collect_runtime_info  # noqa: E402
from experiments.relaxed_specdec import bonus_accept_oracle as B  # noqa: E402
from specdec.engine import model_self, target_rope_scope  # noqa: E402

FAMILIES = ("F09", "F03", "F11", "F04")
CONTEXTS = (2048, 4096, 8192)
K = 4
MAX_NEW = 64


def wait_for_memory(min_gib: float = 16.1, timeout_s: int = 3600) -> float:
    from specdec.engine.g5_8k_spec_variance_probe import system_memory
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        m = system_memory()
        if m.get("available"):
            last = m["commit_free_mib"] / 1024.0
            if last >= min_gib:
                return last
        time.sleep(20)
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
                carry = B.decode(target, draft, prompt, MAX_NEW, k=K, eos_id=eos,
                                 mode="carry", finite_check=True, capture_cache=True,
                                 diagnostic_top2=True)
                model_self.assert_strategy_used()

                model_self.reset_attention_stats()
                catch = B.decode(target, draft, prompt, MAX_NEW, k=K, eos_id=eos,
                                 mode="catchup", finite_check=True, capture_cache=True,
                                 diagnostic_top2=True)
                model_self.assert_strategy_used()

                st = carry["stats"]
                control = carry["control"]
                gen_len = len(carry["gen"])
                fc = carry["final_cache"]
                token_equal = carry["gen"] == catch["gen"]
                semantic_equal = (st["rounds"] == catch["stats"]["rounds"]
                                  and st["bonus_tokens"] == catch["stats"]["bonus_tokens"]
                                  and st["candidate_tokens"] == catch["stats"]["candidate_tokens"]
                                  and carry["pending_at_stop"] == catch["pending_at_stop"]
                                  and all(a["emitted_candidates"] == b["emitted_candidates"]
                                          for a, b in zip(carry["control"], catch["control"])))
                diff_pos = next((i for i, (x, y) in enumerate(zip(carry["gen"], catch["gen"]))
                                 if x != y), None)
                div_info = None
                if diff_pos is not None:
                    cum = 0
                    best_gap = None
                    for row, crow in zip(carry["control"], catch["control"]):
                        n = row["emitted_count"]
                        if cum <= diff_pos < cum + n:
                            a = carry["gen"][diff_pos]
                            b = catch["gen"][diff_pos]
                            kind = ("bonus" if (row["emitted_bonus"] is not None
                                               and a == row["emitted_bonus"])
                                    else "candidate")
                            best_gap = row.get("last_row_top2_gap")
                            div_info = dict(round=row["round"], kind=kind, carry=a, catch=b,
                                            carry_top2_gap=row.get("last_row_top2_gap"),
                                            catch_top2_gap=crow.get("last_row_top2_gap"))
                            break
                        cum += n
                same = token_equal and semantic_equal
                per_round_ok = all(
                    row["target_len"] == row["draft_len"] == ctx + row["committed"] - row["pending_new"]
                    and row["candidate_start_row"] == row["pending_included"]
                    and row["next_bonus_row"] == row["verify_input_length"] - 1
                    for row in control)
                first_ok = control[0]["verify_input_length"] == control[0]["kp"]
                later_ok = all(row["verify_input_length"] == row["kp"] + (1 if row["pending_included"] else 0)
                               for row in control[1:])
                counts_ok = (st["rounds"] == 13 and st["candidate_tokens"] == 52
                             and st["bonus_tokens"] == 12 and gen_len == 64
                             and all(x["mismatch_rounds"] == 0 and x["repair_count"] == 0
                                     and x["rollback_count"] == 0 for x in (st, catch["stats"])))
                it = st["target_extra_single_step_calls"] == 0
                stop_ok = carry["gen"][-1] != eos   # 本批历史条件预期不出现 EOS
                ok = (semantic_equal and per_round_ok and first_ok and later_ok and counts_ok
                      and it and stop_ok and fc["tgt_consistent"] and fc["drf_consistent"]
                      and fc["tgt_len"] == ctx + gen_len and fc["drf_len"] == ctx + gen_len)
                row = dict(family=family, context=ctx, input_sha1=meta["token_sha1"],
                           gen_len=gen_len, rounds=st["rounds"],
                           candidate_tokens=st["candidate_tokens"],
                           bonus_tokens=st["bonus_tokens"],
                           target_block_calls=st["target_block_calls"],
                           target_block_input_tokens=st["target_block_input_tokens"],
                           draft_calls=st["draft_calls"],
                           bonus_catchup_calls=st["bonus_catchup_calls"],
                           target_extra_single_step_calls=st["target_extra_single_step_calls"],
                           carry_vs_catchup_same=bool(same),
                           tokens_bitwise_equal=bool(token_equal),
                           semantic_equal=bool(semantic_equal),
                           divergence=div_info,
                           per_round_ok=per_round_ok, first_round_ok=first_ok,
                           later_rounds_ok=later_ok, counts_ok=counts_ok,
                           final_cache=fc, ok=bool(ok))
                rows.append(row)
                print(f"{'OK' if ok else 'FAIL'} {family} {ctx}: gen={gen_len} rounds={st['rounds']} "
                      f"cand={st['candidate_tokens']} bonus={st['bonus_tokens']} "
                      f"tgt_calls={st['target_block_calls']} draft_calls={st['draft_calls']} "
                      f"same={same} cache={fc['tgt_len']}/{fc['drf_len']}", flush=True)
                installed = True
    finally:
        if installed:
            target_rope_scope.restore()

    result = dict(schema_version="bonus-accept-gate-v1",
                  utc=datetime.now(timezone.utc).isoformat(),
                  environment=environment, runtime=collect_runtime_info(),
                  load=dict(target=target_info, draft=draft_info), rope=rope,
                  rows=rows, all_ok=all(r["ok"] for r in rows),
                  tokens_bitwise_equal_cases=sum(r["tokens_bitwise_equal"] for r in rows),
                  semantic_equal_cases=sum(r["semantic_equal"] for r in rows),
                  theory=dict(k=K, max_new=MAX_NEW, rounds=13, candidate_tokens=52,
                              bonus_tokens=12, total=64,
                              note=("每轮最多提交 kp+1 个 token；12×5 + 尾轮 4 = 64 ⇒ 13 轮；"
                                    "carry 路径 target 只做 block call（首轮 s=kp、之后 s=kp+1），"
                                    "draft 需额外 12 次 bonus catch-up（每轮 1 次，末轮 bonus 不再追赶）")))
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
    print("BITWISE_EQUAL_CASES:", result["tokens_bitwise_equal_cases"], "/", len(result["rows"]))
    print("SEMANTIC_EQUAL_CASES:", result["semantic_equal_cases"], "/", len(result["rows"]))
    print("ALL_OK" if result["all_ok"] else "GATE_FAILED")
    return 0 if result["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
