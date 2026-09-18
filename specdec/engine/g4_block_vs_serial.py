"""g4_block_vs_serial.py — target block-forward vs serial cached-forward 一致性诊断

目的（分类 D / B6 前置）: 同一真实 prefix 后, "一次 block forward 验证 4 个 token" vs
"同 4 token 逐 token cached forward" 在每个位置的 argmax/数值是否一致; 同时验证 v0.2
logits 对齐规则（防 off-by-one）。

语义（本文件内的索引约定）:
  block: 一次 _run(prefix + [t1..t4]) -> 行 b0..b3; b_j 是"输入 t_{j+1}"的输出行,
         预测 t_{j+2}（因果 mask 下只依赖 prefix+t1..t_{j+1}）。
  serial: 从 prefix cache 依次喂 t1..t4, a_j = 喂入 t_{j+1} 后的输出行, 同样预测 t_{j+2};
          a3 = 喂入 t4 后预测"t4 之后"。
  对齐判定（v0.2）: t1 用 saved pre_logits(prefix 之末) 判定;
      t_{j+2} (j=0..2) 用 b_j 判定, 与 serial 参照 a_j 逐位一致;
      全接受后的续接 = b3 与 a3 一致。
报告每 case: 每个对齐位置 argmax 是否一致、max|Δ|、block/serial 的 top-1/2 与 margin;
          终态 block-cache vs serial-cache max|Δ| 与长度; 确定性由重复运行外部确认。
候选来源: 各模型自身 serial-greedy(K 个) —— 全接受情形; 1.5B 另用 0.5B-draft-greedy —— 真实用法。
Run（交互式 GPU 终端, 单跑）:  python -m specdec.engine.g4_block_vs_serial
"""
from __future__ import annotations

import torch

from .align_ours import build_from_config

DTYPE = torch.bfloat16
K = 4


def load_ours(model_name: str):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        model_name, config=cfg, attn_implementation="eager",
        torch_dtype=DTYPE, trust_remote_code=True).to("cuda").eval()
    ours = build_from_config(cfg)
    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    if cfg.to_dict().get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    ours.load_state_dict(sd, strict=True)
    del hf
    torch.cuda.empty_cache()
    ours = ours.to("cuda", DTYPE).eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return ours, tok


def pref_capture(ours, ids):
    with torch.inference_mode():
        lg, cache = ours._run(ours.embed_tokens(ids), None, need_logits=True)
    return lg[0, -1].to(torch.float32), cache


def serial_feed(ours, cache, cands):
    """逐 token 喂 cands; 返回 a_j(j=0..K-1) fp32 与终 cache."""
    a = []
    cur = cache
    with torch.inference_mode():
        for t in cands:
            lg, cur = ours._run(
                ours.embed_tokens(torch.tensor([[t]], device=cands.device)),
                cur, need_logits=True)
            a.append(lg[0, -1].to(torch.float32))
    return a, cur


def block_feed(ours, ids, cands):
    full = torch.cat([ids, cands.view(1, -1)], dim=1)
    with torch.inference_mode():
        lg, cache = ours._run(ours.embed_tokens(full), None, need_logits=True)
    rows = lg[0, -4:].to(torch.float32)          # b0..b3
    return rows, cache


def greedy_next(ours, ids, n):
    toks = []
    with torch.inference_mode():
        lg, cache = ours._run(ours.embed_tokens(ids), None, need_logits=True)
        for _ in range(n):
            t = int(lg[0, -1].argmax())
            toks.append(t)
            lg, cache = ours._run(
                ours.embed_tokens(torch.tensor([[t]], device=ids.device)),
                cache, need_logits=True)
    return toks


def cache_kv_maxdiff(cache_a, cache_b):
    m = 0.0
    for la, lb in zip(cache_a, cache_b):
        for key in ("k", "v"):
            ka = la[key].to(torch.float32)
            kb = lb[key].to(torch.float32)
            m = max(m, (ka - kb).abs().max().item())
    return m


def top2(lg):
    v, idx = lg.topk(2)
    return int(idx[0]), float(v[0]), int(idx[1]), float(v[1]), float(v[0] - v[1])


def case_report(model_name, ours, tok, prompt, cands, label):
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    pre, cache_p = pref_capture(ours, ids)
    a, cache_serial = serial_feed(ours, cache_p, torch.tensor(cands, device="cuda"))
    rows, cache_block = block_feed(ours, ids, torch.tensor(cands, device="cuda"))

    print(f"\n[{label}] {model_name}\n  prompt={prompt[:44]!r}\n  cands(t1..t4)="
          f"{[tok.decode([c])[:6] for c in cands]}")
    # t1 由 saved pre_logits 判定 —— 两法同源恒等, 仅打印其 argmax
    am_pre = int(pre.argmax())
    print(f"  t1 saved-pre argmax = {tok.decode([am_pre])!r} (两法同源)")
    eq_all = True
    for j in range(K - 1):                       # 判定 t2..t4: b0..b2 vs a0..a2
        am_b, v1b, t2b, v2b, mb = top2(rows[j])
        am_a, v1a, t2a, v2a, ma = top2(a[j])
        eq = am_b == am_a
        eq_all &= eq
        print(f"  t{j+2}: b{j} argmax={tok.decode([am_b])!r} vs a{j} argmax="
              f"{tok.decode([am_a])!r} equal={eq} | max|Δ|="
              f"{(rows[j]-a[j]).abs().max().item():.3e} | b top1/2 "
              f"{t2b}({v2b:.1f})/{v1b}({mb:.2f}) | a top1/2 {t2a}({v2a:.1f})/"
              f"{v1a}({ma:.2f})")
    # 全接受后的续接: b3 vs a3（都预测 t4 之后）
    am_b, v1b, t2b, v2b, mb = top2(rows[K - 1])
    am_a, v1a, t2a, v2a, ma = top2(a[K - 1])
    eqc = am_b == am_a
    eq_all &= eqc
    print(f"  after-accept b3 vs a3: argmax {tok.decode([am_b])!r} == "
          f"{tok.decode([am_a])!r}: {eqc} | max|Δ|="
          f"{(rows[K-1]-a[K-1]).abs().max().item():.3e}")
    print(f"  final cache max|Δ|(block vs serial)="
          f"{cache_kv_maxdiff(cache_block, cache_serial):.3e}  len block="
          f"{cache_block[0]['k'].shape[2]} serial={cache_serial[0]['k'].shape[2]}  "
          f"all-aligned-equal={eq_all}")


def main() -> int:
    torch.manual_seed(0)
    DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
    TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
    prompts = ["The capital of France is",
               "The history of artificial intelligence and neural networks spans many decades.",
               "def add(a, b):\n    return a + b\n\nprint(add(2, 3))"]
    for p in prompts:
        for mname, mlabel in ((DRAFT, "0.5B"), (TARGET, "1.5B")):
            ours, tok = load_ours(mname)
            ids = tok(p, return_tensors="pt").input_ids.cuda()
            g = greedy_next(ours, ids, K)
            case_report(mname, ours, tok, p, g, f"{mlabel}-self-greedy")
            if mlabel == "1.5B":
                d5, tok5 = load_ours(DRAFT)
                g5 = greedy_next(d5, tok5(p, return_tensors="pt").input_ids.cuda(), K)
                case_report(mname, ours, tok, p, g5, "1.5B-cands=0.5B-draft-greedy")
                del d5
                torch.cuda.empty_cache()
            del ours
            torch.cuda.empty_cache()
    print("\n判定依据: 各对齐位 argmax 全一致 + 终态 cache 长度一致 -> block 与 serial 一致"
          "（B6 无 off-by-one 风险）; 任一不一致且重复稳定 -> TARGET BLOCKER 候选, 需进一步定位")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
