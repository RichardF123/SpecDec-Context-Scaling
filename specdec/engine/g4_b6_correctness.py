"""g4_b6_correctness.py — B6: SpecDec exact-token-id correctness gate（唯一放行证据）

判据（协议 v0.2 §8）: 对每个 case、每个 KV 档:
    spec_decode_greedy(...)["gen"]  ==  target_only_greedy(ids, n) 的生成段（逐 token，不比文本）
  另加: 多次运行确定性; 全 logits finite; 覆盖 10 类 prompt 形态。
参照实现 = **同一引擎的 target-only 贪婪 generate**（cached AR）。若出现不一致:
  打印首个分歧位置 + 该前缀下 target 的 top-1/top-2 margin（区分 near-tie 漂移 vs 真 bug）,
  **不得**改成放宽门或换参照。
Run（交互式 GPU 终端, 单跑; 先实现 spec_dec）: python -m specdec.engine.g4_b6_correctness
"""
from __future__ import annotations

import gc
import traceback

import torch

from .g4_b4_cache_tests import load_ours

TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
K = 4

CASES = [
    ("natural-1", "The capital of France is", 16),
    ("natural-2", "Photosynthesis is the process by which", 16),
    ("math", "3 + 5 = 8. And 7 times 6 is", 16),
    ("code", "def add(a, b):\n    return a + b\n\nprint(add(2, 3))", 16),
    ("short", "Hello", 16),
    ("long", "The history of artificial intelligence and neural networks spans many "
             "decades of research into learning systems that process language with "
             "attention over long sequences of tokens in large models.", 16),
    ("len13", "Q: what is the largest planet?\nA:", 13),          # 非 4 倍数
    ("len5", "Once upon a time", 5),                              # 小预算
    ("len2", "The answer is", 2),                                 # budget < k
    ("eos-prone", "Repeat the word cat forever: cat cat cat cat cat cat", 16),
]


def target_ref(tgt, ids, n, eos):
    """target-only greedy（cached AR，同引擎，即协议参照）。"""
    return tgt.generate(ids, n, eos_id=eos)[0, ids.shape[1]:].tolist()


def divergence_diag(tgt, tok, ids, gen, ref, idx):
    """在首个分歧前缀处打印 target 的 argmax(决策) 与 top-1/top-2 margin。
    注意: 决策路径用 argmax; 并列时 topk 的返回顺序不等于 argmax(取最小 index)。"""
    prefix = torch.cat([ids, torch.tensor([gen[:idx]], device=ids.device)], dim=1)
    with torch.inference_mode():
        lg, _ = tgt._run(tgt.embed_tokens(prefix), None, need_logits=True)
    row = lg[0, -1]
    am = int(row.argmax())
    v, i = row.topk(2)
    tied = int((row == row.max()).sum())
    print(f"    diag @prefix_len={prefix.shape[1]}: argmax={tok.decode([am])!r} "
          f"tied_at_max={tied} | top1={tok.decode([int(i[0])])!r}({v[0]:.3f}) "
          f"top2={tok.decode([int(i[1])])!r}({v[1]:.3f}) margin={float(v[0]-v[1]):.3f} "
          f"| spec picked {tok.decode([gen[idx]])!r} (ref {tok.decode([ref[idx]])!r})")


def main() -> int:
    from .spec_dec import spec_decode_greedy
    torch.manual_seed(0)

    print("[stage] loading target 1.5B ...", flush=True)
    tgt, tok = load_ours(TARGET)
    print("[stage] loading draft 0.5B ...", flush=True)
    drf, _ = load_ours(DRAFT)
    eos = tok.eos_token_id

    ok = True
    finite_ok = True
    rows = []
    for kv in ("base_bf16", "v_fp8"):
        print(f"\n== target_kv={kv} ==", flush=True)
        for label, prompt, n in CASES:
            ids = tok(prompt, return_tensors="pt").input_ids.cuda()
            # finite sanity: target 在该 prompt 上的 logits 必须 finite
            with torch.inference_mode():
                lg, _ = tgt._run(tgt.embed_tokens(ids), None, need_logits=True)
            finite_ok &= bool(torch.isfinite(lg).all().item())
            del lg
            torch.cuda.empty_cache()

            ref = target_ref(tgt, ids, n, eos)
            out = spec_decode_greedy(tgt, drf, ids, n, k=K, eos_id=eos, target_kv=kv)
            gen = out["gen"]
            same = gen == ref
            st = out["stats"]
            rows.append((kv, label, n, len(gen), len(ref), same,
                         st.get("accepted"), st.get("proposed")))
            print(f"  [{label:10s}] n={n:2d} gen={len(gen):2d} ref={len(ref):2d} "
                  f"equal={same} accepted={st.get('accepted')}/{st.get('proposed')} "
                  f"rounds={st.get('rounds')}", flush=True)
            if not same:
                ok = False
                for i, (a, b) in enumerate(zip(gen, ref)):
                    if a != b:
                        print(f"    first diff @{i}: spec={tok.decode([a])!r} "
                              f"ref={tok.decode([b])!r}")
                        divergence_diag(tgt, tok, ids, gen, ref, i)
                        break
                else:
                    print(f"    length differs (spec={len(gen)} ref={len(ref)})")
            ok &= same

        # 确定性(用两个代表 case)
        for label, prompt, n in (CASES[0], CASES[5]):
            ids = tok(prompt, return_tensors="pt").input_ids.cuda()
            a = spec_decode_greedy(tgt, drf, ids, n, k=K, eos_id=eos, target_kv=kv)["gen"]
            b = spec_decode_greedy(tgt, drf, ids, n, k=K, eos_id=eos, target_kv=kv)["gen"]
            print(f"  determinism[{label}]: {a == b}")
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
        print("WARNING: vacuous — 没有任何候选被接受, 等价性未被真正检验 "
              "(需先修 draft/align 后再判 B6)", flush=True)
    ok &= (not vacuous)

    del tgt, drf
    gc.collect()
    torch.cuda.empty_cache()
    print("\nB6:", "PASS" if ok else "CHECK",
          "(不一致时先看 margin: 近贴边属 BF16 漂移类, 需记录而非放宽门)")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
