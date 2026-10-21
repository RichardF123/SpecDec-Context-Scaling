import torch
from specdec.engine.spec_cache import cache_length, check_consistent
from specdec.engine import model_self, spec_dec

from specdec.engine.smoke_checks import (
    CheckRegistry,
    verify_decision_coverage,
    expected_map,
)
from specdec.engine.smoke_hooks import DecisionHooks

def diagnose_spec(target, draft, input_ids, eos_id, prefill_impl):
    if spec_dec.target_verify_mode() != "block":
        raise RuntimeError("诊断必须使用 production block 验证")
    if spec_dec.verify_wrap_state()["active"]:
        raise RuntimeError("不能叠加 row-aligned 验证包装")

    registry = CheckRegistry()
    trace = []

    def check_round(row):
        kp = row["Kp"]
        cands = row["cands"]
        decisions = row["g"]

        if not 1 <= kp <= 4:
            raise RuntimeError("候选数量越界")
        if len(cands) != kp or len(decisions) != kp:
            raise RuntimeError("候选或判断行数量错误")

        accepted = 0
        while accepted < kp and cands[accepted] == decisions[accepted]:
            accepted += 1

        if row["m"] != accepted:
            raise RuntimeError("接受长度与候选比较结果不一致")

        emitted = (
            cands[:accepted] + [decisions[accepted]]
            if accepted < kp else list(cands)
        )

        if row["emitted"] != emitted:
            raise RuntimeError("接受前缀或替换token错误")

        expected = row["pre"] + len(emitted)

        if row["tgt_len"] != expected or row["drf_len"] != expected:
            raise RuntimeError("逐轮缓存长度不变量失败")
        if trace and row["pre"] != trace[-1]["tgt_len"]:
            raise RuntimeError("跨轮缓存长度不一致")
        trace.append(row)
    
    def check_prefill_cache(role, logits, cache):
        check_consistent(cache)
        if cache_length(cache) != input_ids.shape[1]:
            raise RuntimeError(f"{role} prefill 缓存长度错误")

    def check_verify_cache(cache, cands):
        check_consistent(cache)

    hooks = DecisionHooks(
        tgt = target,
        drf = draft,
        registry = registry,
        prefill_impl = prefill_impl,
        on_prefill_post = check_prefill_cache,
        on_verify_pre = check_verify_cache,
    ) 

    model_self.reset_attention_stats()
    try:
        hooks.install()
        originals = dict(hooks._orig)

        with torch.inference_mode():
            output = spec_dec.spec_decode_greedy(
                target, draft, input_ids, 64,
                k = 4, eos_id = eos_id,
                target_kv = "base_bf16",
                draft_kv = "base_bf16",
                log = check_round,
            )

        coverage = verify_decision_coverage(
            registry,
            draft_calls_expected = output["stats"]["draft_calls"],
            rounds_expected = output["stats"]["rounds"],
        )

        if not coverage["ok"] or not trace:
            raise RuntimeError("决策检查覆盖不足或逐轮记录为空")
        expected = expected_map(
            mismatch_rounds = output["stats"]["mismatch_rounds"],
            mode = "spec", 
        )

        for key, requirement in expected.items():
            if not requirement["required"]:
                continue

            slot = registry.calls.get(key)
            if slot is None or not slot["ok"]:
                raise RuntimeError(f"必需有限性检查缺失或失败： {key}")
        model_self.assert_strategy_used()
        return {
            "diagnostic_only": True,
            "generated_ids": output["gen"],
            "stats": output["stats"],
            "trace": trace,
            "decision_coverage": coverage,
            "finiteness_coverage": registry.coverage(),
            "attention_stats": model_self.attention_stats(),
        }
    except Exception as exc:
        exc.evidence = {
            "diagnostic_only": True,
            "trace": trace,
            "finiteness_coverage": registry.coverage(),
            "attention_stats": model_self.attention_stats(),
        }
        raise
    finally:
        hooks.remove()
        if "originals" in locals():
            restored = (
                spec_dec._prefill is originals["prefill"]
                and spec_dec._propose is originals["propose"]
                and spec_dec._verify is originals["verify"]
                and spec_dec._step is originals["step"]
                and model_self.QwenLM._run is originals["run"]
            )
            if not restored:
                raise RuntimeError("诊断 hooks 未完整恢复")

def diagnose_ar(target, input_ids, eos_id, prefill_impl):
    registry = CheckRegistry()
    generated = []
    decode_calls = 0
    stop_reason = "max_new_reached"
    prompt_length = input_ids.shape[1]
    model_self.reset_attention_stats()

    try:
        with torch.inference_mode():
            logits, cache = prefill_impl(target, input_ids)
            check_consistent(cache)
            if cache_length(cache) != prompt_length:
                raise RuntimeError("AR prefill 缓存长度错误")

            for index in range(64):
                phase = "prefill" if index == 0 else "decode"
                registry.check_finite(
                    logits, phase = phase, model = "target",
                    role = "decision_logits",
                    where = f"选择第 {index} 个输出token之前",
                )

                token = int(logits[0, -1].argmax().item())
                generated.append(token)

                if token == eos_id:
                    stop_reason = "eos"
                    break
                if index == 63:
                    break

                step_ids = torch.tensor(
                    [[token]], dtype = torch.long, device = input_ids.device
                )

                with model_self.attention_phase("decode"):
                    logits, cache = target._run(
                        target.embed_tokens(step_ids),
                        cache, need_logits = True,
                    )

                decode_calls += 1
                check_consistent(cache)
                if cache_length(cache) != prompt_length + decode_calls:
                    raise RuntimeError("AR decode 缓存长度错误")
        if registry.n_checked != len(generated):
            raise RuntimeError("AR 决策有限性检查覆盖不足")
        model_self.assert_strategy_used()

        return {
            "diagnostic_only": True,
            "generated_ids": generated,
            "stop_reason": stop_reason,
            "decode_calls": decode_calls,
            "finiteness_coverage": registry.coverage(),
            "attention_stats": model_self.attention_stats(),
        }
    except Exception as exc:
        exc.evidence = {
            "diagnostic_only": True,
            "generated_ids": generated,
            "finiteness_coverage": registry.coverage(),
        }
        raise
            