from transformers import AutoTokenizer
import torch
from .records import save_record, save_failure, save_diagnostic
from .diagnostics import diagnose_ar, diagnose_spec
from .environment import (
    check_environment,
    check_gpu_environment,
    EnvironmentRefused,
)
import time
import traceback
import argparse
import hashlib
from .runtime_info import collect_runtime_info, require_formal_version
from .runtime_audit import capture_boundary, assess_boundaries
from pathlib import Path
from uuid import uuid4
from contextlib import contextmanager
from specdec.engine import spec_dec
from .records import save_record, save_failure
from dataclasses import asdict
from .plan import build_jobs
from .input_sets import select_input_set

from specdec.engine import model_self
from specdec.engine.spec_cache import cache_length, check_consistent
from specdec.engine import target_rope_scope
from specdec.engine.lowmem_load import load_ours_bf16
from specdec.engine.family_input import(
    TOKENIZER_ID,
    TOKENIZER_REVISION,
    eos_config,
    load_family_ids,
    load_manifest,
)

def load_input(family_id, context_tokens, *, input_set="pilot"):
    manifest_path, families = select_input_set(input_set)
    if family_id not in families:
        raise ValueError(f"Family {family_id} 不属于 {input_set} 输入集合")
    manifest = load_manifest(manifest_path)
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID,
        revision = TOKENIZER_REVISION,
        local_files_only = True,
    )

    ids, metadata = load_family_ids(
        tokenizer,
        family_id,
        manifest = manifest,
        ctx = context_tokens,
        verify_hash = True,
    )

    if len(ids) != context_tokens:
        raise ValueError("输入的token数不符合任务要求")

    eos = eos_config(tokenizer)
    if not eos["matches_expected"]:
        raise ValueError(f"eos_token_id {eos['eos_id']} 不符合预期 {eos['eos_id_expected']}")
    metadata = dict(metadata)
    metadata.update(
        input_set=input_set,
        manifest_path=str(manifest_path),
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    return ids, metadata
def load_target():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，无法加载目标模型")

    model, tokenizer, info = load_ours_bf16(
        "Qwen/Qwen2.5-1.5B-Instruct",
        verify = True,
    )

    if info["n_params_checked"] <= 0:
        raise RuntimeError("模型参数检查失败，可能是模型文件损坏或不完整")
    
    if not info["weights_bitmatch"]:
        raise RuntimeError("模型权重位数不匹配，可能是模型文件损坏或不完整")

    parameter = next(model.parameters())

    if parameter.device.type != "cuda":
        raise RuntimeError("模型没有加载到GPU上，可能是CUDA配置问题")

    if parameter.dtype != torch.bfloat16:
        raise RuntimeError("模型权重不是bfloat16类型，可能是模型文件损坏或不完整")
    return model, tokenizer, info

def load_draft():
    model, tokenizer, info = load_ours_bf16(
        "Qwen/Qwen2.5-0.5B-Instruct",
        verify = True,
    )

    if info["n_params_checked"] <= 0:
        raise RuntimeError("Draft 权重检查未执行")

    if not info["weights_bitmatch"]:
        raise RuntimeError(
            f"Draft 权重不一致： {info['mismatch_keys']}"
        )

    parameter = next(model.parameters())
    if parameter.device.type != "cuda":
        raise RuntimeError("Draft 不在CUDA上")
    if parameter.dtype != torch.bfloat16:
        raise RuntimeError("Draft 不是 BF16")

    model.set_kv_format("fp16")
    return model, tokenizer, info

def configure_target(model):
    model.set_kv_format("fp16")
    model_self.set_diag_preset("eager")
    model_self.set_attention_strategy("cudnn_prefill")

    rope_info = target_rope_scope.install(
        model,
        theta = float(model.rope_theta),
        head_dim = int(model.head_dim),
    )
    return rope_info

def check_runtime_config(target, draft, job):
    precision = model_self.diag_precision()

    if model_self.attention_strategy() != "cudnn_prefill":
        raise RuntimeError("Attention 策略与冻结配置不符")
    
    if precision != {"qk_fp32": False, "prob_fp32": False}:
        raise RuntimeError("诊断精度干预没有关闭")
    
    if spec_dec.target_verify_mode() != "block":
        raise RuntimeError("Target 验证模式不是production block")
    
    if spec_dec.verify_wrap_state()["active"]:
        raise RuntimeError("存在额外验证包装")

    if (draft is not None) != (job.condition.mode == "SpecDec"):
        raise RuntimeError("模型常驻状态与任务模式不符")

    models = [target] if draft is None else [target, draft]
    for model in models:
        if model.kv_format != "fp16":
            raise RuntimeError("不是未量化 KV 基线")

        for parameter in model.parameters():
            if parameter.device.type != "cuda":
                raise RuntimeError("存在非CUDA权重")
            if parameter.dtype != torch.bfloat16:
                raise RuntimeError("存在非 BF 16 权重")
    return {
        "attention_strategy": model_self.attention_strategy(),
        "diag_precision": precision,
        "target_verify_mode": spec_dec.target_verify_mode(),
        "target_kv_engine": target.kv_format,
        "draft_kv_engine": None if draft is None else draft.kv_format,
        "compute_dtype": "bfloat16",
        "draft_resident": draft is not None,
    }
    

def test_chunked_prefill(model, input_ids, chunk = 2048, diagnostic = True):
    n_tokens = input_ids.shape[1]
    cache = None
    logits = None

    with torch.inference_mode():
        for start in range(0, n_tokens, chunk):
            end = min(start + chunk, n_tokens)
            piece = input_ids[:, start:end]
            is_last = end == n_tokens

            with model_self.attention_phase("prefill"):
                logits, cache = model._run(
                    model.embed_tokens(piece),
                    cache,
                    need_logits = is_last,
                )
            if diagnostic:
                check_consistent(cache)
                assert cache_length(cache) == end
                print(f"已处理 {start}:{end}，cache={cache_length(cache)}")

    assert logits is not None

    if diagnostic:
        assert torch.isfinite(logits).all().item()
        model_self.assert_strategy_used()

        stats = model_self.attention_stats()
        assert stats["fused_by_phase"].get("prefill", 0) > 0
        print("融合调用：", stats["fused_by_phase"])

    next_logits = logits[:, -1, :].clone()
    return next_logits, cache

def test_ar_decode(model, next_logits, cache, eos_id, max_new = 16, diagnostic = True):
    generated = []
    stop_reason = "max_new_reached"
    prompt_length = cache_length(cache)
    decode_calls = 0
    device = next(model.parameters()).device

    with torch.inference_mode():
        for index in range(max_new):
            if diagnostic:
                assert torch.isfinite(next_logits).all().item()
            token = int(next_logits[0].argmax().item())
            generated.append(token)

            if token == eos_id:
                stop_reason = "eos"
                break
            if index == max_new - 1:
                break

            step_ids = torch.tensor(
                [[token]],dtype = torch.long, device = device
            )
            with model_self.attention_phase("decode"):
                logits, cache = model._run(
                    model.embed_tokens(step_ids), cache,
                    need_logits = True,
                )

            next_logits = logits[:, -1, :]
            decode_calls += 1
            if diagnostic:
                check_consistent(cache)
                assert cache_length(cache) == prompt_length + decode_calls
    return generated, stop_reason

@contextmanager
def spec_prefill_scope(target,draft):
    original = spec_dec._prefill
    calls = []

    def adapted_prefill(model, input_ids):
        if model is target:
            role = "target"
        elif model is draft:
            role = "draft"
        else:
            raise RuntimeError("Prefill 收到未登记的模型")
        torch.cuda.synchronize()
        started = time.perf_counter()
        next_logits, cache = test_chunked_prefill(
            model,
            input_ids,
            chunk = 2048,
            diagnostic = False,
        )

        torch.cuda.synchronize()
        calls.append({
            "role": role,
            "seconds": time.perf_counter() - started,
        })

        return next_logits.unsqueeze(1), cache

    spec_dec._prefill = adapted_prefill

    try:
        yield calls
    finally:
        spec_dec._prefill = original

@contextmanager
def spec_generation_scope():
    originals = {
        "_propose": spec_dec._propose,
        "_verify":  spec_dec._verify,
        "_step": spec_dec._step,
    }
    
    def wrap(function, phase):
        def wrapped(*args, **kwargs):
            with model_self.attention_phase(phase):
                return function(*args, **kwargs)
        return wrapped

    spec_dec._propose = wrap(originals["_propose"], "propose")
    spec_dec._verify = wrap(originals["_verify"], "verify")
    spec_dec._step = wrap(originals["_step"], "repair")

    try:
        yield
    finally:
        spec_dec._propose = originals["_propose"]
        spec_dec._verify = originals["_verify"]
        spec_dec._step = originals["_step"]

def run_ar_once(model, tokenizer, token_ids, job):
    if job.condition.mode != "AR":
        raise ValueError("当前接口只支持 AR")

    if job.condition.target_kv != "base_bf16":
        raise ValueError("当前接口只支持未量化 BF16 KV")

    if len(token_ids) != job.condition.context_tokens:
        raise ValueError("输入长度与Job不一致")
    device = next(model.parameters()).device
    input_ids = torch.tensor(
        [token_ids], dtype = torch.long, device = device
    )
    model_self.reset_attention_stats()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    next_logits, cache = test_chunked_prefill(
        model, input_ids, chunk = 2048, diagnostic = False
    )

    torch.cuda.synchronize()
    prefill_end = time.perf_counter()

    generated, stop_reason = test_ar_decode(
        model,
        next_logits,
        cache,
        tokenizer.eos_token_id,
        max_new = 64,
        diagnostic = False,
    )

    torch.cuda.synchronize()
    end = time.perf_counter()

    prefill_s = prefill_end - start
    total_s = end - start
    decode_s  = total_s - prefill_s

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    model_self.assert_strategy_used()

    return {
        "job": asdict(job),
        "max_new": 64,
        "generated_ids": generated,
        "generated_length": len(generated),
        "stop_reason": stop_reason,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "total_s": total_s,
        "decode_tokens_per_s": len(generated) / decode_s,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "diagnostic_checks_executed": False,
        "attention_stats": model_self.attention_stats(),
    }

def run_spec_once(target, draft, tokenizer, token_ids, job):
    if job.condition.mode != "SpecDec" or draft is None:
        raise ValueError("SpecDec 任务必须有 Draft")
    if job.condition.target_kv != "base_bf16":
        raise ValueError("当前只支持未量化 BF16 KV")
    if len(token_ids) != job.condition.context_tokens:
        raise ValueError("输入长度与Job不一致")
    if spec_dec.target_verify_mode() != "block":
        raise ValueError("不能使用serial oracle 计时")
    if spec_dec.verify_wrap_state()["active"]:
        raise RuntimeError("不能使用诊断 verify wrapper 计时")

    device = next(target.parameters()).device
    input_ids = torch.tensor([token_ids],dtype = torch.long, device = device)
    model_self.reset_attention_stats()

    with (
        spec_prefill_scope(target, draft) as calls,
        spec_generation_scope(),
    ):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        output = spec_dec.spec_decode_greedy(
            target, draft, input_ids, 64,
            k = 4, eos_id = tokenizer.eos_token_id,
            target_kv = "base_bf16", draft_kv = "base_bf16",
        ) 
        torch.cuda.synchronize()
        total_s = time.perf_counter() - started
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()

    if [item["role"] for item in calls] != ["target", "draft"]:
        raise RuntimeError("Prefill 调用顺序或次数异常")
    model_self.assert_strategy_used()
    generated = output['gen']
    prefill_s = sum(item["seconds"] for item in calls)
    decode_s = total_s - prefill_s
    stop = "eos" if generated and generated[-1] == tokenizer.eos_token_id else "max_new_reached"

    return{
        "job": asdict(job), "max_new": 64,
        "generated_ids": generated, "generated_length": len(generated),
        "stop_reason": stop,
        "prefill_s": prefill_s, "decode_s": decode_s, "total_s": total_s,
        "decode_tokens_per_s": len(generated) / decode_s,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "diagnostic_checks_executed": False,
        "prefill_components": calls,
        "spec_stats": output["stats"],
        "attention_stats": model_self.attention_stats()
    }         

def run_once(model, draft, tokenizer, token_ids, job):
    if job.condition.mode == "AR":
        if draft is not None:
            raise ValueError("AR 任务不应加载 draft")
        return run_ar_once(model, tokenizer, token_ids, job)

    if job.condition.mode == "SpecDec":
        return run_spec_once(
            model, draft, tokenizer, token_ids, job
        )
    raise ValueError(f"未知模式：{job.condition.mode}")

def run_diagnostic(model, draft, tokenizer, token_ids, job):
    device = next(model.parameters()).device
    input_ids = torch.tensor(
        [token_ids], dtype = torch.long, device = device
    )

    def prefill_impl(current_model, ids):
        next_logits, cache = test_chunked_prefill(
            current_model,
            ids,
            chunk = 2048,
            diagnostic = True,
        )
        return next_logits.unsqueeze(1), cache

    if job.condition.mode == "AR":
        if draft is not None:
            raise ValueError("AR诊断不应该加载draft")
        return diagnose_ar(
            model, input_ids, tokenizer.eos_token_id, prefill_impl
        )
    if job.condition.mode == "SpecDec":
        if draft is None:
            raise ValueError("SpecDec 诊断缺少draft")
        return diagnose_spec(
            model, draft, input_ids,
            tokenizer.eos_token_id, prefill_impl,
        )
    raise ValueError(f"未知模式： {job.condition.mode}")

def execute_job(model, draft, tokenizer, token_ids, job, provenance, run_id, output_dir, progress,
                *, experiment_stage="development"):
    print("输出目录：", output_dir)
    progress["stage"] = "diagnostic"
    diagnostic = run_diagnostic(
        model, draft, tokenizer, token_ids, job
    )
    diagnostic.update(provenance)
    diagnostic["run_id"] = run_id
    diagnostic["job"] = asdict(job)
    diagnostic["experiment_stage"] = experiment_stage
    progress["stage"] = "save_diagnostic"
    save_diagnostic(
        diagnostic,
        output_dir / "diagnostics" / "result.json"
    )
    print("开始 warm_up: 结果不计入 measured")
    progress["stage"] = "warmup"
    warmup = run_once(model, draft, tokenizer, token_ids, job)
    del warmup

    results = []

    for rep_index in range(job.measured_reps):
        progress["rep_index"] = rep_index
        progress["stage"] = "monitor_start"
        monitor_start = capture_boundary()
        progress["stage"] = "measured"
        result = run_once(model, draft, tokenizer, token_ids, job)
        progress["stage"] = "monitor_end"
        monitor_end = capture_boundary()
        result["runtime_environment_audit"] = assess_boundaries(
            monitor_start, monitor_end
        )
        result.update(provenance)
        result["diagnostic_file"] = "diagnostics/result.json"
        result["same_mode_diagnostic_tokens_equal"] = (
            result["generated_ids"] == diagnostic["generated_ids"]
        )
        result["run_id"] = run_id
        result["rep_index"] = rep_index
        result["run_kind"] = "measured"
        result["experiment_stage"] = experiment_stage
        result["eos_id"] = tokenizer.eos_token_id

        filename = (
            f"{job.family_id}_{job.condition.condition_id}"
            f"_b{job.measurement_block}_r{rep_index}.json"
        )
        progress["stage"] = "save_record"
        save_record(
            result,
            output_dir / filename,
            tokenizer.eos_token_id,
        )
        progress["stage"] = "compare_diagnostic"
        if not result["same_mode_diagnostic_tokens_equal"]:
            raise RuntimeError("同模式诊断与计时输出不一致，保留结果并停止")
        results.append(result)

        print(
            f"Measured {rep_index + 1} / {job.measured_reps}:"
            f"prefill = {result['prefill_s']:.3f}s,"
            f"decode = {result['decode_s']:.3f}s"
        )
    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-set", choices=("pilot", "formal"), default="pilot")
    parser.add_argument(
        "--family", required = True,
    )
    parser.add_argument(
        "--ctx", required = True, type = int,
        choices = (2048, 4096, 8192) 
    )
    parser.add_argument(
        "--mode", required = True, choices = ("AR", "SpecDec"), 
    )
    parser.add_argument(
        "--block", required = True, type = int, choices = (0, 1), 
    )
    parser.add_argument(
        "--dry-run", action = "store_true"
    )
    parser.add_argument("--output-dir",type = Path)
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    _, families = select_input_set(args.input_set)
    if args.family not in families:
        parser.error(f"--family {args.family} 不属于 --input-set {args.input_set}")
    if args.input_set == "formal" and not args.dry_run and not args.expected_commit:
        parser.error("正式采集必须由批次入口传入 --expected-commit")
    return args

if __name__ == "__main__":
    args = parse_args()
    manifest_path, families = select_input_set(args.input_set)
    jobs = build_jobs(families)

    job = next(
        item for item in jobs
        if item.family_id == args.family
        and item.condition.context_tokens == args.ctx
        and item.condition.mode == args.mode
        and item.measurement_block == args.block
    )

    if args.dry_run:
        print("输入集合：", args.input_set, "清单：", manifest_path)
        print(asdict(job))
        raise SystemExit(0)
    run_id = uuid4().hex
    experiment_stage = "formal" if args.input_set == "formal" else "development"
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parents[2]
            / "output" / ("formal_worker" if experiment_stage == "formal" else "formal_development") / run_id
        )
    progress = {"stage": "environment"}
    environment = None
    rope_installed = False

    try:
        environment = check_environment()
        progress["stage"] = "runtime_info"
        runtime_info = collect_runtime_info()
        if experiment_stage == "formal":
            require_formal_version(runtime_info, args.expected_commit)
        progress["stage"] = "gpu_preflight"
        environment["gpu"] = check_gpu_environment()
        progress["stage"] = "load_input"
        ids, metadata = load_input(
            job.family_id, job.condition.context_tokens,
            input_set=args.input_set,
        )

        progress["stage"] = "load_target"
        model, tokenizer, load_info = load_target()
        draft = None
        draft_info = None

        if job.condition.mode == "SpecDec":
            progress["stage"] = "load_draft"
            draft, draft_tokenizer, draft_info = load_draft()

            if draft_tokenizer.eos_token_id != tokenizer.eos_token_id:
                raise ValueError("Target 与 Draft 的EOS ID 不一致")
            
        progress["stage"] = "configure"
        rope_info = configure_target(model)
        rope_installed = True
        config_info = check_runtime_config(model, draft, job)

        provenance = {
            "input": metadata,
            "load": load_info,
            "draft_load": draft_info,
            "rope": rope_info,
            "environment": environment,
            "runtime": runtime_info,
            "configuration": config_info,
            "collection": {
                "input_set": args.input_set,
                "expected_commit": args.expected_commit,
                "speedup_claim_allowed": False,
                "claim_status": "raw_collection_pending_analysis",
            },
        }
        results = execute_job(
            model, draft, tokenizer, ids, job, provenance,
            run_id, output_dir, progress,
            experiment_stage=experiment_stage,
        )
        print("Measured 记录数：", len(results))

    except Exception as exc:
        failure_class = (
            "ENV_REFUSED" if isinstance(exc, EnvironmentRefused)
            else "OOM" if isinstance(exc, torch.cuda.OutOfMemoryError)
            else "ERROR"
        )
        failure = {
            "run_id": run_id, "job": asdict(job),
            **progress,
            "failure_class": failure_class,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "environment": environment,
            "experiment_stage": experiment_stage,
            "environment_failure_evidence": getattr(exc, "evidence", None),
        }
        save_failure(failure, output_dir / "failure.json")
        raise
    finally:
        if rope_installed:
            target_rope_scope.restore()
