"""2K 端到端 Graph 兑现门：同批 AR / eager bonus / graph bonus 三模式。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from experiments.execution_optimization import config  # noqa: E402
from experiments.relaxed_specdec.run_dev import wait_for_memory  # noqa: E402
from experiments.cuda_graph_e2e import design  # noqa: E402

FAMILY = "F09"
CTX = 2048
REPS = 5
WARMUP = 1


def mem_ledger(tag: str) -> dict:
    free, total = torch.cuda.mem_get_info()
    return dict(tag=tag, allocated_gib=torch.cuda.memory_allocated() / 2**30,
                reserved_gib=torch.cuda.memory_reserved() / 2**30,
                free_gib=free / 2**30, total_gib=total / 2**30)


def time_once(fn, n: int = 1, trials: int = 3) -> dict:
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(n):
            fn()
        en.record()
        en.synchronize()
        samples.append(st.elapsed_time(en) / n)
    samples.sort()
    return dict(n=n, trials=trials, median_ms=samples[len(samples) // 2],
                min_ms=samples[0], max_ms=samples[-1], samples=samples)


def run_graph_child(out: Path) -> dict:
    from experiments.formal import worker as W
    from experiments.formal.environment import check_environment, check_gpu_environment
    from experiments.formal.worker import test_chunked_prefill
    from experiments.relaxed_specdec import bonus_accept_oracle as B
    from experiments.cuda_graph_e2e import graph_state as GS
    from experiments.cuda_graph_e2e import capture_rounds as CR
    from experiments.cuda_graph_e2e import gate_gpu as G
    from specdec.engine import model_self, target_rope_scope

    out.mkdir(parents=True, exist_ok=True)
    env = check_environment(16)
    env["gpu"] = check_gpu_environment()
    ledger = [mem_ledger("start")]
    target, tokenizer, tinfo = W.load_target()
    ledger.append(mem_ledger("after_target_load"))
    draft, dtok, dinfo = W.load_draft()
    ledger.append(mem_ledger("after_draft_load"))
    rope = W.configure_target(target)
    installed = True
    try:
        ids, meta = W.load_input(FAMILY, CTX)
        prompt = torch.tensor([ids], dtype=torch.long, device=torch.device("cuda"))
        tgt_lg, tgt_cache = test_chunked_prefill(target, prompt, chunk=2048, diagnostic=False)
        drf_lg, drf_cache = test_chunked_prefill(draft, prompt, chunk=2048, diagnostic=False)
        ledger.append(mem_ledger("after_prefill"))
        # eager bonus oracle 参考（同批）
        model_self.reset_attention_stats()
        ref = B.decode(target, draft, prompt, design.MAX_NEW, k=4,
                       eos_id=tokenizer.eos_token_id, mode="carry", capture_cache=True)
        model_self.assert_strategy_used()
        ledger.append(mem_ledger("after_eager_reference"))
        state_t = GS.BigBuffers(target)
        state_d = GS.BigBuffers(draft)
        state_t.load_from_cache(tgt_cache)
        state_d.load_from_cache(drf_cache)
        io = GS.StaticIO()
        io.load_draft_logits0(drf_lg)
        snap_io = io.snapshot()
        ledger.append(mem_ledger("after_buffers"))

        from experiments.cuda_graph_local import rope_tables as RT
        tables = {}
        for r in range(design.ROUNDS):
            s = design.VERIFY_S[r]
            start = design.VERIFY_START[r]
            tables[(s, start)] = RT.build_rope_tables(target, [s], start_pos=start,
                                                      dtype=torch.bfloat16)[(s, start)]
        cap = CR.capture_all({"target": target, "draft": draft}, state_d, state_t, io,
                             shared_pool=False, rope_tables=tables)
        graphs = cap["graphs"]
        ledger.append(mem_ledger("after_all_capture"))
        capture_s = cap["capture_s"]

        def reset():
            state_t.load_from_cache(tgt_cache)
            state_d.load_from_cache(drf_cache)
            io.restore(snap_io)

        reset()
        gate = G.run_gate(graphs, io, state_d, state_t, ref)
        ledger.append(mem_ledger("after_gate"))
        if not gate["ok"]:
            (out / "failure.json").write_text(json.dumps(dict(
                mode="graph", error="E2E_GRAPH_2K_GATE_FAILED", gate=gate),
                ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            raise RuntimeError(f"E2E_GRAPH_2K_GATE: FAIL {gate}")
        reset_t = time_once(reset, n=1, trials=3)

        def assemble_loop():
            for r in range(design.ROUNDS):
                CR.assemble_verify_tokens(io, r)

        assemble_t = time_once(assemble_loop, n=1, trials=3)
        # 首次请求：capture + 一次完整 replay
        reset()
        first = time_once(lambda: CR.replay_pipeline(graphs, io, None), n=1, trials=1)
        # measured reps：每 rep 先 reset（恢复真实 prefill cache），再完整 13 轮 replay
        reps = []
        for rep in range(REPS):
            reset()
            torch.cuda.synchronize()
            st = torch.cuda.Event(enable_timing=True)
            en = torch.cuda.Event(enable_timing=True)
            wall0 = time.perf_counter()
            st.record()
            CR.replay_pipeline(graphs, io, None)
            en.record()
            en.synchronize()
            wall1 = time.perf_counter()
            gen = io.gen.clone()
            reps.append(dict(rep=rep, gpu_ms=st.elapsed_time(en),
                             wall_ms=(wall1 - wall0) * 1000.0,
                             gen=[int(x) for x in gen[0].tolist()]))
        ledger.append(mem_ledger("after_measured"))
        result = dict(
            schema_version="cuda-graph-e2e-2k-v1",
            experiment_class=design.EXPERIMENT_CLASS, scope=design.SCOPE,
            formal_result_allowed=False, quality_claim_allowed=False,
            production_claim_allowed=False,
            mode="graph", utc=datetime.now(timezone.utc).isoformat(),
            environment=env, input_meta=meta,
            load=dict(target=tinfo, draft=dinfo), rope=rope,
            graph_count=len(graphs), shared_pool=cap["shared_pool"],
            capture_s=capture_s,
            first_request_replay_ms=first["median_ms"],
            first_request_total_s=capture_s + first["median_ms"] / 1000.0,
            reset_ms=reset_t["median_ms"],
            assemble_ms=assemble_t["median_ms"],
            reps=reps,
            ledger=ledger,
            gate=gate,
            ref_gen=ref["gen"],
            ref_control=[dict(round=c["round"], kp=c["kp"], emitted_candidates=c["emitted_candidates"],
                              emitted_bonus=c["emitted_bonus"], emitted_count=c["emitted_count"])
                         for c in ref["control"]],
            status="ok",
        )
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                   default=str), encoding="utf-8")
        return result
    finally:
        if installed:
            target_rope_scope.restore()


def run_simple_child(out: Path, mode: str) -> dict:
    from experiments.formal import worker as W
    from experiments.formal.environment import check_environment, check_gpu_environment
    from experiments.formal.plan import Condition, Job
    from experiments.relaxed_specdec import bonus_accept_oracle as B
    from experiments.formal.worker import test_chunked_prefill
    from specdec.engine import model_self, target_rope_scope

    out.mkdir(parents=True, exist_ok=True)
    env = check_environment(16)
    env["gpu"] = check_gpu_environment()
    target, tokenizer, tinfo = W.load_target()
    draft = None
    if mode == "eager":
        draft, dtok, dinfo = W.load_draft()
        if dtok.eos_token_id != tokenizer.eos_token_id:
            raise RuntimeError("EOS mismatch")
    rope = W.configure_target(target)
    installed = True
    try:
        ids, meta = W.load_input(FAMILY, CTX)
        if mode == "ar":
            job = Job(FAMILY, Condition(f"L{CTX}_AR_base_bf16", CTX, "AR", "base_bf16"), 0, 1)
            W.run_ar_once(target, tokenizer, ids, job)
            reps = [W.run_ar_once(target, tokenizer, ids, job) for _ in range(REPS)]
            result = dict(
                schema_version="cuda-graph-e2e-2k-v1", experiment_class=design.EXPERIMENT_CLASS,
                scope=design.SCOPE, formal_result_allowed=False, quality_claim_allowed=False,
                production_claim_allowed=False,
                mode="ar", utc=datetime.now(timezone.utc).isoformat(),
                environment=env, input_meta=meta, load=dict(target=tinfo),
                reps=[dict(rep=i, prefill_s=r["prefill_s"], decode_s=r["decode_s"],
                           total_s=r["total_s"], gen=r["generated_ids"]) for i, r in enumerate(reps)],
                status="ok")
        else:
            prompt = torch.tensor([ids], dtype=torch.long, device=torch.device("cuda"))
            B.decode(target, draft, prompt, design.MAX_NEW, k=4,
                     eos_id=tokenizer.eos_token_id, mode="carry")
            reps = []
            for rep in range(REPS):
                model_self.reset_attention_stats()
                out_run = B.decode(target, draft, prompt, design.MAX_NEW, k=4,
                                   eos_id=tokenizer.eos_token_id, mode="carry",
                                   measure_timing=True, capture_cache=True)
                model_self.assert_strategy_used()
                reps.append(dict(rep=rep, prefill_s=out_run["timing"]["prefill_s"],
                                 decode_s=out_run["timing"]["decode_s"],
                                 total_s=out_run["timing"]["total_s"],
                                 gen=out_run["gen"],
                                 stats=out_run["stats"]))
            result = dict(
                schema_version="cuda-graph-e2e-2k-v1", experiment_class=design.EXPERIMENT_CLASS,
                scope=design.SCOPE, formal_result_allowed=False, quality_claim_allowed=False,
                production_claim_allowed=False,
                mode="eager", utc=datetime.now(timezone.utc).isoformat(),
                environment=env, input_meta=meta, load=dict(target=tinfo, draft=dinfo),
                rope=rope, reps=reps, status="ok")
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                   default=str), encoding="utf-8")
        return result
    finally:
        if installed:
            target_rope_scope.restore()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--mode", choices=("ar", "eager", "graph"))
    args = parser.parse_args()
    if args.mode:
        run_graph_child(args.out) if args.mode == "graph" else run_simple_child(args.out, args.mode)
        return 0
    run_dir = args.out or (ROOT / "output" / "cuda_graph_e2e" /
                           (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]))
    run_dir.mkdir(parents=True, exist_ok=True)
    modes = ("ar", "eager", "graph")
    (run_dir / "manifest.json").write_text(json.dumps(dict(
        schema_version="cuda-graph-e2e-2k-manifest-v1",
        experiment_class=design.EXPERIMENT_CLASS, scope=design.SCOPE,
        formal_result_allowed=False, quality_claim_allowed=False,
        production_claim_allowed=False,
        utc=datetime.now(timezone.utc).isoformat(),
        family=FAMILY, context=CTX, k=design.K,
        max_new=design.MAX_NEW, rounds=design.ROUNDS,
        reps=REPS, warmup=WARMUP, modes=list(modes),
        schedule=design.check_schedule()), ensure_ascii=False, indent=2), encoding="utf-8")
    print("OUTPUT:", run_dir, flush=True)
    for i, mode in enumerate(modes, 1):
        out = run_dir / mode
        if (out / "result.json").exists():
            print(f"[{i}/{len(modes)}] {mode} 已完成，跳过", flush=True)
            continue
        free = wait_for_memory()
        print(f"[{i}/{len(modes)}] {mode} commit free={free:.2f} GiB", flush=True)
        log_path = run_dir / f"{mode}.log"
        cmd = [sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()),
               "--mode", mode, "--out", str(out)]
        with log_path.open("a", encoding="utf-8") as log:
            with subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True,
                                  encoding="utf-8", errors="replace") as proc:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                rc = proc.wait()
        if rc != 0 and not (out / "result.json").exists():
            raise RuntimeError(f"{mode} failed rc={rc}; see {log_path}")
        time.sleep(3)
    print("COMPLETE:", run_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
