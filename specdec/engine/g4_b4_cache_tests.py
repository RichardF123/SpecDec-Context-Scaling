"""g4_b4_cache_tests.py — B4: cache append/trim/rollback 测试（对用户实现的 spec_cache 契约）

前置：`specdec/engine/spec_cache.py` 已按 `docs/design_g4_cache_api.md` 实现
（cache_length / trim_cache / check_consistent）。
覆盖:
  1. None 语义      2. 真实 cache 构建 长度正确  3. append 后长度增长
  4. trim 到 L / L+1 / 1 token / 多 token / 0(非法) / 越界 / 负数
  5. mismatch 在 m=0..3 后长度 6. base 与 v_fp8 均可 trim、v_scale 同步
  7. check_consistent 抓坏 cache  8. 原 cache 不被原地修改
模型用法(按协议):
  - base_bf16 cache: 0.5B draft(轻) 上测;
  - v_fp8 cache: **只能在 1.5B target 上测**(GROUP_V=128 要求 head_dim%128==0;0.5B head_dim=64)。
Run（交互式 GPU 终端, 单跑; 先实现 spec_cache）: python -m specdec.engine.g4_b4_cache_tests
"""
from __future__ import annotations

import gc
import sys
import traceback

import torch

from .align_ours import build_from_config

MODEL_SMALL = "Qwen/Qwen2.5-0.5B-Instruct"     # base_bf16 用
MODEL_TARGET = "Qwen/Qwen2.5-1.5B-Instruct"    # v_fp8 用(target 专属格式)
DTYPE = torch.bfloat16
TOKENS = 32
CHUNK = 16


def _from_pretrained(cls, model_name: str, **kw):
    """先试本地缓存(local_files_only=True),失败再联网。
    原因: transformers 加载 tokenizer 时会联网 list_repo_templates;网络抖动会抛
    httpx.RemoteProtocolError, 与被测逻辑无关。"""
    try:
        return cls.from_pretrained(model_name, local_files_only=True, **kw)
    except Exception as e:  # noqa: BLE001
        print(f"[load] local cache miss/fail ({type(e).__name__}: {e}); "
              f"retrying online ...", flush=True)
        return cls.from_pretrained(model_name, **kw)


def load_ours(model_name: str):
    """测试用 loader: HF 权重先在 CPU 加载 -> 拷给 ours(CPU) -> 只把 ours 搬上 GPU。
    这样 GPU 峰值 ≈ 单模型(避免 hf+ours 同时在卡上)。"""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    free0, _ = torch.cuda.mem_get_info()
    print(f"[load] {model_name}: HF on CPU (cuda free={free0/2**30:.2f} GiB)",
          flush=True)
    cfg = _from_pretrained(AutoConfig, model_name, trust_remote_code=True)
    hf = _from_pretrained(AutoModelForCausalLM, model_name, config=cfg,
                          attn_implementation="eager", torch_dtype=DTYPE,
                          trust_remote_code=True)          # 注意: 不上 GPU
    ours = build_from_config(cfg)
    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    if cfg.to_dict().get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    ours.load_state_dict(sd, strict=True)
    del hf, sd
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[load] weights copied on CPU; moving ours to cuda ...", flush=True)
    ours = ours.to("cuda", DTYPE).eval()
    free1, _ = torch.cuda.mem_get_info()
    print(f"[load] {model_name} resident, cuda free={free1/2**30:.2f} GiB", flush=True)
    tok = _from_pretrained(AutoTokenizer, model_name, trust_remote_code=True)
    return ours, tok


def build_cache(ours, ids, fmt):
    ours.set_kv_format(fmt)
    cache = None
    with torch.inference_mode():
        for st in range(0, ids.shape[1], CHUNK):
            piece = ids[:, st:st + CHUNK]
            _, cache = ours._run(ours.embed_tokens(piece), cache, need_logits=False)
    return cache


def append_ones(ours, cache, n):
    with torch.inference_mode():
        _, cache = ours._run(
            ours.embed_tokens(torch.ones(1, n, dtype=torch.long, device="cuda")),
            cache, need_logits=False)
    return cache


def test_one_mode(spec_cache, ours, tok, fmt: str, label: str) -> bool:
    ids = torch.randint(100, 1000, (1, TOKENS)).cuda()
    ok = True
    base = build_cache(ours, ids, fmt)
    L0 = spec_cache.cache_length(base)
    spec_cache.check_consistent(base)
    L0_ok = L0 == TOKENS

    grown = append_ones(ours, base, 3)
    L1 = spec_cache.cache_length(grown)
    spec_cache.check_consistent(grown)
    app_ok = L1 == L0 + 3

    t1 = spec_cache.trim_cache(grown, L0 + 1)
    spec_cache.check_consistent(t1)
    t1_ok = spec_cache.cache_length(t1) == L0 + 1
    t0 = spec_cache.trim_cache(grown, L0)
    spec_cache.check_consistent(t0)
    t0_ok = spec_cache.cache_length(t0) == L0

    sync = True
    if "v_scale" in t0[0]:
        sync = all(l["v_scale"].shape[2] == l["v"].shape[2] for l in t0)

    # 非法:超长 / 负数 / 0(状态机里真实长度恒 >0)
    raises = 0
    for bad in (L1 + 5, -1, 0):
        try:
            spec_cache.trim_cache(grown, bad)
        except ValueError:
            raises += 1
    idem = spec_cache.cache_length(spec_cache.trim_cache(grown, L1)) == L1
    orig_unchanged = (spec_cache.cache_length(grown) == L1
                      and grown[0]["k"].shape[2] == L1)

    print(f"[{label}] len={L0}(ok={L0_ok}) append->{L1}(ok={app_ok}) "
          f"trim->L0+1={t1_ok} trim->L0={t0_ok} v_scale-sync={sync} "
          f"raises={raises}/3 idem={idem} orig-unchanged={orig_unchanged}")
    ok &= all((L0_ok, app_ok, t1_ok, t0_ok, sync, raises == 3, idem, orig_unchanged))
    del base, grown, t1, t0
    torch.cuda.empty_cache()
    return ok


def run_base(spec_cache) -> bool:
    """0.5B 上的 base_bf16 段 + mismatch 裁剪长度段(独立进程运行以避免显存累积)。"""
    ok = True
    print("[stage] loading 0.5B ...", flush=True)
    ours_s, tok_s = load_ours(MODEL_SMALL)
    ok &= test_one_mode(spec_cache, ours_s, tok_s, "fp16", "base_bf16")

    # 先取 pre,再延长 4(模拟 block 多算),然后裁回 pre+m。
    # 注意: 引擎 _run 【原地】推进 cache(dict 复用) -> 必须先取 pre, 否则 base==blk。
    ids = torch.randint(100, 1000, (1, TOKENS)).cuda()
    base = build_cache(ours_s, ids, "fp16")
    pre = spec_cache.cache_length(base)
    blk = append_ones(ours_s, base, 4)
    m_ok = spec_cache.cache_length(blk) == pre + 4
    for m in (0, 1, 2, 3, 4):
        r = spec_cache.trim_cache(blk, pre + m)
        spec_cache.check_consistent(r)
        m_ok &= spec_cache.cache_length(r) == pre + m
    print(f"[2] trim for mismatch m=0..4 (pre={pre}, blk={pre+4} back to pre+m): {m_ok}",
          flush=True)
    ok &= m_ok
    del base, blk, ours_s, tok_s
    gc.collect()
    torch.cuda.empty_cache()
    return ok


def run_v_fp8(spec_cache) -> bool:
    """1.5B 上的 v_fp8 段(独立进程运行)。"""
    ok = True
    print("[stage] loading 1.5B (v_fp8 is target-only) ...", flush=True)
    ours_t, tok_t = load_ours(MODEL_TARGET)
    free, _ = torch.cuda.mem_get_info()
    print(f"[stage] building v_fp8 cache (cuda free={free/2**30:.2f} GiB) ...",
          flush=True)
    ok &= test_one_mode(spec_cache, ours_t, tok_t, "v_fp8", "v_fp8")

    ids = torch.randint(100, 1000, (1, TOKENS)).cuda()
    bad = build_cache(ours_t, ids, "v_fp8")
    bad[3]["v"] = bad[3]["v"][:, :, :-1]
    try:
        spec_cache.check_consistent(bad)
        caught = False
    except RuntimeError:
        caught = True
    print(f"[3] check_consistent catches bad cache: {caught}", flush=True)
    ok &= caught
    del bad, ours_t, tok_t
    gc.collect()
    torch.cuda.empty_cache()
    return ok


def main(mode: str = "all") -> int:
    from . import spec_cache  # 用户实现后导入
    torch.manual_seed(0)
    ok = True

    # [1] None 语义(不需要模型)
    ok &= spec_cache.cache_length(None) == 0
    ok &= spec_cache.trim_cache(None, 0) is None
    print(f"[1] None semantics: {ok}", flush=True)

    try:
        if mode in ("all", "base"):
            ok &= run_base(spec_cache)
        if mode in ("all", "v_fp8"):
            ok &= run_v_fp8(spec_cache)
    except Exception:
        traceback.print_exc()
        print("\nB4: CHECK (exception above)", flush=True)
        return 2

    print("\nB4:", "PASS" if ok else "CHECK", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    _mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if _mode not in ("all", "base", "v_fp8"):
        print("usage: python -m specdec.engine.g4_b4_cache_tests [all|base|v_fp8]",
              flush=True)
        raise SystemExit(2)
    raise SystemExit(main(_mode))
