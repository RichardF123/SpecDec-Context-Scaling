"""specdec CLI (Phase 0 scaffold): env-check + run stub.

Imports are lazy so `specdec run --help` / `specdec env-check` work even
before torch is installed (env-check reports what is missing).
"""
from __future__ import annotations

import argparse
import json
import sys


def _cmd_env_check(_args: argparse.Namespace) -> int:
    print("== specdec env-check ==")
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        print(f"[FAIL] torch import failed: {exc}")
        return 1
    print(f"torch {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(
            f"VRAM: {props.total_memory / 2**30:.1f} GiB | "
            f"capability: {props.major}.{props.minor}"
        )
        try:
            import triton

            print(f"triton {triton.__version__} (import OK)")
        except Exception as exc:
            print(f"[WARN] triton import failed: {exc}")
            print("       per go/no-go rule: if Triton is unavailable/slow -> LOCK EAGER")
    else:
        print("[WARN] no CUDA GPU visible — CPU-only path must be used for demo/eval")
    try:
        import transformers

        print(f"transformers {transformers.__version__}")
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] transformers import failed: {exc}")
    print("[OK] env-check done.")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .config import RunConfig

    if args.config:
        with open(args.config, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        allowed = RunConfig.__dataclass_fields__
        cfg = RunConfig(**{k: v for k, v in data.items() if k in allowed})
    else:
        cfg = RunConfig(stage=args.stage, seed=args.seed)

    rec = cfg.to_record()
    print(f"stage={cfg.stage}  model={cfg.model_id}  seed={cfg.seed}")
    print(f"config_hash={rec['config_hash']}")
    print(f"run_id={rec['run_id']}")
    print(
        "[NOT-IMPLEMENTED] engine pipeline lands in Phase 1 (L1 forward). "
        "Scaffold check only — NO result row appended (honesty rule)."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="specdec", description="From-scratch LLM inference engine (course project)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("env-check", help="check torch/triton/transformers on this machine")

    run_p = sub.add_parser("run", help="run a ladder stage (stub in Phase 0)")
    run_p.add_argument("--stage", default="L1", help="L1..L5, L4b")
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--config", default=None, help="path to a JSON RunConfig")

    args = parser.parse_args(argv)
    if args.command == "env-check":
        return _cmd_env_check(args)
    if args.command == "run":
        return _cmd_run(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
