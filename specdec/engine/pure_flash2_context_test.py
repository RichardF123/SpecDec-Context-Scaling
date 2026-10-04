"""CPU-only regression checks for pure_flash2_context state restoration.

No FlashAttention import or CUDA execution is required because this test only
enters/exits the runtime context without calling model.forward().
"""
from __future__ import annotations

from .model_self import QwenLM
from .pure_flash2_backend import pure_flash2_context


def _make_model() -> QwenLM:
    return QwenLM(
        vocab_size=64,
        hidden=256,
        heads=2,
        kv_heads=1,
        head_dim=128,
        intermediate=512,
        num_layers=2,
    )


def _snapshot(model: QwenLM):
    return {
        "model": (
            model.kv_format,
            model.cache_backend,
            model.cache_page_size,
            model.paged_attention_mode,
        ),
        "layers": [
            (
                layer.self_attn.kv_format,
                layer.self_attn.cache_backend,
                layer.self_attn.cache_page_size,
                layer.self_attn.paged_attention_mode,
                hasattr(layer.self_attn, "_pure_flash2_enabled"),
                hasattr(layer.self_attn, "_pure_flash2_max_seqlen"),
                hasattr(layer.self_attn, "_pure_flash2_original_forward"),
            )
            for layer in model.layers
        ],
    }


def _configure_nondefault(model: QwenLM) -> None:
    model.set_kv_format("v_fp8")
    model.set_cache_backend("paged_v0", page_size=32)
    model.set_paged_attention_mode("flash_paged_cuda")


def _assert_forced_inside(model: QwenLM) -> None:
    assert model.kv_format == "fp16"
    assert model.cache_backend == "contiguous"
    assert model.cache_page_size == 16
    assert model.paged_attention_mode == "off"
    for layer in model.layers:
        a = layer.self_attn
        assert a.kv_format == "fp16"
        assert a.cache_backend == "contiguous"
        assert a.cache_page_size == 16
        assert a.paged_attention_mode == "off"
        assert a._pure_flash2_enabled is True


def main() -> int:
    model = _make_model()
    _configure_nondefault(model)
    before = _snapshot(model)

    with pure_flash2_context(model, max_seqlen=64):
        _assert_forced_inside(model)
    assert _snapshot(model) == before

    try:
        with pure_flash2_context(model, max_seqlen=64):
            _assert_forced_inside(model)
            raise RuntimeError("intentional-test-exception")
    except RuntimeError as exc:
        assert str(exc) == "intentional-test-exception"
    else:
        raise AssertionError("expected intentional exception")

    assert _snapshot(model) == before
    print("PURE_FLASH2_CONTEXT_RESTORE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
