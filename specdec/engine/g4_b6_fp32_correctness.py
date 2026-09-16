"""g4_b6_fp32_correctness.py — B6-fp32：高精度档下的 SpecDec **实现**正确性门（预登记）

【本门验证什么 / 不验证什么】（运行前冻结，入库留证）
- 被测对象 = `spec_decode_greedy` 的**实现**：控制流(propose→verify→accept/rollback)、
  cache 语义(prefill / `trim_cache(pre+m)` / 长度不变量)、logits 对齐
  (`[saved_next, rows[0..kp-2]]` 判 `cands[0..kp-1]`；`rows[kp-1]` 不判)、
  替换 token 的**单步重算**、EOS 与 `budget<k` 边界。**dtype 不是被测对象。**
- 为什么需要它：bf16 档（B6-bf16，§4.5.5）在 `natural-2@7` 处把候选对压成**精确并列**
  (22.8750/22.8750)，实测形状噪声 0.250–0.281（2–4 ULP）**大于**真实 gap 0.1491（1.2 ULP）
  ⇒ 该位置上 bf16 **无法表示决策**，门在该处不是良定义的（不是实现缺陷，也不是"放宽"问题）。
- 本门把**同一份实现**放到决策良定义的配置：CPU / float32 / 双模型 eager / batch=1 / k=4 /
  greedy。fp32 相对误差 ~1e-7，与 1e-1 量级的决策 gap 差 5–6 个数量级。
- 判据（与 B6-bf16 **完全一致，不放宽一个字**）：逐 token id 全等（参照 = 同引擎 cached AR
  `target_ref`，不比文本）；logits 全 finite；重复运行确定；vacuity guard（accepted>0）；
  **两个 KV 档、全部 case 都必须过**，否则 CHECK。
- 失败归因（预登记，只用于解释，**不改判据**）：若仍分歧，在该 prefix 上取 fp32 三路径
  (full / cached / AR-history) 的同一候选对 gap：若三路径 argmax 一致且
  `|gap| > 路径间 spread` ⇒ 判为实现层缺陷；否则如实登记为"fp32 下仍不可判定"。两种情况
  **门都不过**。
- 报告口径：B6-bf16 = 性能档行为（如实披露 19/20 + 归因）；B6-fp32 = **实现正确性证据**。

Run（交互式终端, 单跑）:
  先小样验证:  set B6F_LIMIT=2  &&  python -m specdec.engine.g4_b6_fp32_correctness
  全量:        python -m specdec.engine.g4_b6_fp32_correctness
"""
from __future__ import annotations

import gc
import os
import time
import traceback

import torch

from .align_ours import build_from_config
from .g4_b4_cache_tests import _from_pretrained
from .g4_b6_correctness import CASES, DRAFT, K, TARGET, target_ref
from .g4_b6_tie_diag import at_prefix_cached, at_prefix_full, at_prefix_hist

DTYPE = torch.float32
DEV = "cpu"
LIMIT = int(os.environ.get("B6F_LIMIT", "0") or 0)


def load_cpu(model_name: str):
    """HF 权重按 fp32 载入 CPU -> 拷进 ours(CPU)。峰值内存 ≈ 两个模型之和（32 GB 足够）。"""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {model_name}: HF fp32 load (CPU) ...", flush=True)
    cfg = _from_pretrained(AutoConfig, model_name, trust_remote_code=True)
    hf = _from_pretrained(AutoModelForCausalLM, model_name, config=cfg,
                          attn_implementation="eager", torch_dtype=DTYPE,
                          trust_remote_code=True)
    ours = build_from_config(cfg)
    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    if cfg.to_dict().get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    ours.load_state_dict(sd, strict=True)
    del hf, sd
    gc.collect()
    ours = ours.to(DEV, DTYPE).eval()
    print(f"[load] {model_name}: ours resident (fp32 CPU)", flush=True)
    tok = _from_pretrained(AutoTokenizer, model_name, trust_remote_code=True)
    return ours, tok


def gap_audit(tgt, tok, ids, prefix_ids, a, b, split):
    """分歧位置的 fp32 三路径归因（只解释，不改判据）。"""
    prefix = torch.cat([ids, torch.tensor([prefix_ids], dtype=ids.dtype, device=DEV)], dim=1)
    paths = {"full": at_prefix_full(tgt, prefix),
             "cached": at_prefix_cached(tgt, prefix),
             "AR-history": at_prefix_hist(tgt, prefix, split)}
    gaps = {}
    for nm, row in paths.items():
        am = int(row.argmax())
        g = float(row[b]) - float(row[a])
        gaps[nm] = g
        print(f"      fp32 {nm:<10s} argmax={tok.decode([am])!r} "
              f"gap(spec-ref)={g:+.3e} | spec@{float(row[a]):.5f} ref@{float(row[b]):.5f}",
              flush=True)
    spread = max(gaps.values()) - min(gaps.values())
    signal = abs(gaps["full"])
    uniq = {int(r.argmax()) for r in paths.values()}
    if len(uniq) == 1 and signal > spread:
        why = (f"实现层缺陷 (fp32 三路径 argmax 同解 且 |gap|={signal:.3e} > "
               f"spread={spread:.3e})")
    else:
        why = (f"fp32 下仍不可判定 (|gap|={signal:.3e}, spread={spread:.3e}, "
               f"argmax 种类={len(uniq)})")
    print(f"      => 归因: {why} —— 门仍不过", flush=True)


def main() -> int:
    from .spec_dec import spec_decode_greedy
    torch.manual_seed(0)

    try:
        probe = torch.zeros(4, dtype=torch.float32).to(torch.float8_e4m3fn)
        fp8_cpu = True
        print(f"[env] CPU float8_e4m3fn cast: OK ({probe.dtype})", flush=True)
    except Exception as e:  # noqa: BLE001
        fp8_cpu = False
        print(f"[env] CPU float8_e4m3fn cast: UNSUPPORTED ({type(e).__name__}: {e}) "
              f"-> v_fp8 档跳过(该档不得据此记为通过)", flush=True)

    cases = CASES[:LIMIT] if LIMIT else CASES
    print(f"[env] torch={torch.__version__} threads={torch.get_num_threads()} "
          f"device={DEV} dtype=fp32 cases={len(cases)}/{len(CASES)} limit={LIMIT or 'all'}",
          flush=True)

    tgt, tok = load_cpu(TARGET)
    drf, _ = load_cpu(DRAFT)
    eos = tok.eos_token_id

    tiers = ["base_bf16"] + (["v_fp8"] if fp8_cpu else [])
    ok = True
    finite_ok = True
    rows = []
    for kv in tiers:
        print(f"\n== target_kv={kv} (fp32 CPU) ==", flush=True)
        for label, prompt, n in cases:
            t0 = time.perf_counter()
            ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
            with torch.inference_mode():
                lg, _ = tgt._run(tgt.embed_tokens(ids), None, need_logits=True)
            finite_ok &= bool(torch.isfinite(lg).all().item())
            del lg

            ref = target_ref(tgt, ids, n, eos)
            out = spec_decode_greedy(tgt, drf, ids, n, k=K, eos_id=eos, target_kv=kv)
            gen = out["gen"]
            st = out["stats"]
            same = gen == ref
            rows.append((kv, label, n, len(gen), len(ref), same,
                         st.get("accepted"), st.get("proposed")))
            print(f"  [{label:10s}] n={n:2d} gen={len(gen):2d} ref={len(ref):2d} "
                  f"equal={same} accepted={st.get('accepted')}/{st.get('proposed')} "
                  f"rounds={st.get('rounds')} ({time.perf_counter()-t0:.1f}s)", flush=True)
            if not same:
                ok = False
                for i, (a, b) in enumerate(zip(gen, ref)):
                    if a != b:
                        print(f"    first diff @{i}: spec={tok.decode([a])!r} "
                              f"ref={tok.decode([b])!r}", flush=True)
                        gap_audit(tgt, tok, ids, ref[:i], a, b, int(ids.shape[1]))
                        break
                else:
                    print(f"    length differs (spec={len(gen)} ref={len(ref)})", flush=True)

        for label, prompt, n in (CASES[0], CASES[5]):
            ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
            a = spec_decode_greedy(tgt, drf, ids, n, k=K, eos_id=eos, target_kv=kv)["gen"]
            b = spec_decode_greedy(tgt, drf, ids, n, k=K, eos_id=eos, target_kv=kv)["gen"]
            print(f"  determinism[{label}]: {a == b}", flush=True)
            ok &= (a == b)

    print("\n== summary ==")
    tot_acc = tot_prop = 0
    for kv, label, n, lg_, lr, same, acc, prop in rows:
        tot_acc += acc or 0
        tot_prop += prop or 0
        print(f"  {kv:9s} {label:10s} n={n:2d} gen={lg_:2d} ref={lr:2d} equal={same} "
              f"acc={acc}/{prop}")
    print(f"logits finite (all cases): {finite_ok}")
    print(f"total accepted/proposed = {tot_acc}/{tot_prop}")
    ok &= finite_ok
    vacuous = (tot_prop > 0 and tot_acc == 0)
    if vacuous:
        print("WARNING: vacuous — 无候选被接受, 等价性未被真正检验", flush=True)
    ok &= (not vacuous)
    ok &= (len(tiers) == 2)
    if len(tiers) != 2:
        print(f"NOTE: 只跑了 {len(tiers)}/2 个 KV 档 -> 不得判 PASS", flush=True)

    del tgt, drf
    gc.collect()
    print(f"\nB6-fp32: {'PASS' if ok else 'CHECK'} "
          f"(tiers={len(tiers)}/2, cases={len(cases)}/{len(CASES)}, k={K}, exact token-id)")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
