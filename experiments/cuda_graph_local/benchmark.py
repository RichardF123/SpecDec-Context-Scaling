"""局部 CUDA Graph A/B 计时（one model per process：draft 或 target，避免双模型 8K 显存污染）。

每个 combo = (family, ctx, model)：
  draft  -> A1/A2/A3；target -> B1/B2。
每进程只加载一个模型 → eager/graph 计时不受另一模型驻留与图池碎片影响。
"""
from __future__ import annotations

import argparse
import contextlib
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

FAMILIES = ("F09", "F04")
CONTEXTS = config.FROZEN["contexts"]
N_REPLAY = {"A1": 60, "A2": 20, "A3": 16, "B1": 30, "B2": 24}
TRIALS = 5


def run_combo(family: str, ctx: int, model_kind: str, out_dir: Path) -> dict:
    from experiments.formal import worker as W
    from experiments.formal.environment import check_environment, check_gpu_environment
    from experiments.formal.runtime_info import collect_runtime_info
    from experiments.formal.worker import test_chunked_prefill
    from experiments.cuda_graph_local import fixtures as F
    from experiments.cuda_graph_local import local_ops as L
    from experiments.cuda_graph_local import draft_graph as DG
    from experiments.cuda_graph_local import verify_graph as VG
    from experiments.cuda_graph_local import gate_gpu as G
    from experiments.cuda_graph_local import rope_tables as RT
    from specdec.engine import target_rope_scope

    out_dir.mkdir(parents=True, exist_ok=True)
    environment = check_environment(16)
    environment["gpu"] = check_gpu_environment()
    if model_kind == "draft":
        model, tokenizer, info = W.load_draft()
        rope = None
        installed = False
    else:
        model, tokenizer, info = W.load_target()
        rope = W.configure_target(model)
        installed = True
    try:
        ids, input_meta = W.load_input(family, ctx)
        prompt = torch.tensor([ids], dtype=torch.long, device=torch.device("cuda"))
        lg, cache = test_chunked_prefill(model, prompt, chunk=2048, diagnostic=False)
        del lg
        toks = F.load_tokens(family, ctx)
        init = L.clone_cache(cache)
        del cache
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        ctx_len = int(prompt.shape[1])
        base = dict(schema_version="cuda-graph-local-combo-v2",
                    experiment_class="operator_mechanism_diagnostic",
                    formal_result_allowed=False, quality_claim_allowed=False,
                    full_specdec_speedup_claim_allowed=False,
                    utc=datetime.now(timezone.utc).isoformat(),
                    family=family, context=ctx, model=model_kind,
                    input_sha1=input_meta["token_sha1"], history_len=ctx_len,
                    tokens=toks, environment=environment,
                    runtime=collect_runtime_info(), load={model_kind: info}, rope=rope)
        pool = [toks["bonus0"], *toks["a1"], *toks["a2"], *toks["a3"], *toks["b1"], *toks["b2"]]

        def make_alt(tokens):
            for t in pool:
                if t != tokens[-1]:
                    return list(tokens[:-1]) + [t]
            raise RuntimeError("无可用替代 token")

        if model_kind == "draft":
            comps = [DG.build_draft_step(model, init, toks["a1"][0]),
                     DG.build_draft_chain(model, init, toks["a2"]),
                     DG.build_draft_chain(model, init, toks["a3"])]
        else:
            comps = [VG.build_verify(model, init, toks["b1"]),
                     VG.build_verify(model, init, toks["b2"])]
        # 阶段 A：先在不含任何 graph pool 的干净 allocator 状态下测 eager，
        # 避免"先 capture 再测 eager"造成的碎片化偏差（实测会膨胀 2x+）。
        eager_results = {comp["name"]: L.time_trials(comp["eager"], n=N_REPLAY[comp["name"]],
                                                    trials=TRIALS)
                         for comp in comps}
        rows = []
        for comp in comps:
            name = comp["name"]
            alt = make_alt(comp["tokens"])
            tok_tensor = torch.tensor([comp["tokens"]], dtype=torch.long,
                                      device=torch.device("cuda"))
            if model_kind == "target":
                tables = RT.build_rope_tables(model, [comp["M"]], start_pos=ctx_len,
                                              dtype=torch.bfloat16)
                rope_ctx = RT.patched_rope(tables)
            else:
                rope_ctx = contextlib.nullcontext()
            with rope_ctx:
                t0 = time.perf_counter()
                graph, outs, mem = L.capture(comp["capture_fn"])
                capture_ms = (time.perf_counter() - t0) * 1000.0
                gate = G.gate_component(comp, graph, outs, alt)
                st = torch.cuda.Event(enable_timing=True)
                en = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                st.record()
                graph.replay()
                en.record()
                en.synchronize()
                first_replay_ms = st.elapsed_time(en)
                n = N_REPLAY[name]
                eager_t = eager_results[name]     # 阶段 A 的干净状态测量
                graph_t = L.time_trials(graph.replay, n=n, trials=TRIALS)

                def effective():
                    comp["static_tok"].copy_(tok_tensor)
                    graph.replay()

                eff_t = L.time_trials(effective, n=n, trials=TRIALS)

                def copy_only():
                    comp["static_tok"].copy_(tok_tensor)

                copy_t = L.time_trials(copy_only, n=n * 4, trials=TRIALS)
                peak_alloc = torch.cuda.max_memory_allocated()
                peak_reserved = torch.cuda.max_memory_reserved()
            rows.append(dict(
                component=name, kind=comp["kind"], tokens_in=len(comp["tokens"]),
                M=comp.get("M"), n_steps=comp.get("n_steps"),
                capture_ms=capture_ms, first_replay_ms=first_replay_ms,
                graph=dict(eager=eager_t, replay_kernel_only=graph_t,
                           replay_plus_copy=eff_t, static_copy=copy_t),
                pool_alloc_delta=mem["pool_alloc_delta"],
                pool_reserved_delta=mem["pool_reserved_delta"],
                peak_allocated_gib=peak_alloc / 2**30,
                peak_reserved_gib=peak_reserved / 2**30,
                gate=gate))
            del graph, outs
            torch.cuda.empty_cache()
            if not gate["ok"]:
                raise RuntimeError(f"{family}/{ctx}/{model_kind}/{name} correctness gate FAILED")
        base["components"] = rows
        base["status"] = "ok"
        (out_dir / "result.json").write_text(json.dumps(base, ensure_ascii=False, indent=2,
                                                       default=str), encoding="utf-8")
        return base
    except Exception as exc:  # noqa: BLE001
        import traceback
        (out_dir / "failure.json").write_text(json.dumps(dict(
            family=family, context=ctx, model=model_kind, error=repr(exc),
            traceback=traceback.format_exc()), ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    finally:
        if installed:
            target_rope_scope.restore()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--family")
    parser.add_argument("--ctx", type=int)
    parser.add_argument("--model", choices=("draft", "target"))
    args = parser.parse_args()
    if args.family:
        run_combo(args.family, args.ctx, args.model, args.out)
        return 0
    run_dir = args.out or (ROOT / "output" / "cuda_graph_local" /
                           (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]))
    run_dir.mkdir(parents=True, exist_ok=True)
    combos = [(fam, c, m) for fam in FAMILIES for c in CONTEXTS for m in ("draft", "target")]
    (run_dir / "manifest.json").write_text(json.dumps(dict(
        schema_version="cuda-graph-local-manifest-v2",
        experiment_class="operator_mechanism_diagnostic",
        formal_result_allowed=False, quality_claim_allowed=False,
        full_specdec_speedup_claim_allowed=False,
        utc=datetime.now(timezone.utc).isoformat(),
        families=list(FAMILIES), contexts=list(CONTEXTS), n_replay=N_REPLAY, trials=TRIALS,
        combos=[dict(family=f, context=c, model=m, dir=str(run_dir / f"{f}_{c}_{m}"))
                for f, c, m in combos],
        notes=["每个进程只加载一个模型（draft 或 target），避免 8K 双模型显存污染",
               "历史长度 = 该 combo 的 ctx；token 来自真实 bonus oracle 轨迹",
               "kernel-only=只 replay；replay+copy=copy_ token 后 replay；capture 单独记录"]),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("OUTPUT:", run_dir, flush=True)
    for i, (family, ctx, model_kind) in enumerate(combos, 1):
        combo_dir = run_dir / f"{family}_{ctx}_{model_kind}"
        if (combo_dir / "result.json").exists():
            print(f"[{i}/{len(combos)}] {family}_{ctx}_{model_kind} 已完成，跳过", flush=True)
            continue
        free = wait_for_memory()
        print(f"[{i}/{len(combos)}] {family}_{ctx}_{model_kind} commit free={free:.2f} GiB", flush=True)
        log_path = run_dir / f"{family}_{ctx}_{model_kind}.log"
        cmd = [sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()),
               "--family", family, "--ctx", str(ctx), "--model", model_kind,
               "--out", str(combo_dir)]
        with log_path.open("a", encoding="utf-8") as log:
            with subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True,
                                  encoding="utf-8", errors="replace") as proc:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                rc = proc.wait()
        if rc != 0 and not (combo_dir / "result.json").exists():
            raise RuntimeError(f"{family}_{ctx}_{model_kind} failed rc={rc}; see {log_path}")
        time.sleep(3)
    print("COMPLETE:", run_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
