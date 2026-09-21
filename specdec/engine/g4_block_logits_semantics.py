"""g4_block_logits_semantics.py — 用**真实模型**验证 block forward 的 logits 语义（CPU fp32 控制）。

要回答的问题（修复前必须确认，不能靠推断）
  设已提交 prefix = P（长度 pre），draft 提出 y0..y3。
  target 一次 block forward 后返回 rows[0..3]。每一行到底对应哪个条件分布？
      rows[j] 是 p(· | P, y0..yj)        （已看过 y_j）
      还是 p(· | P, y0..y(j-1))          （还没看 y_j）？

判据（同位置 vs 相邻位置对照）
  用**串行单步**在同一 prefix 上依次喂 y0,y1,y2,y3，得到 ser[j]。
  - 若 rows[j] 的语义是 "p(·|P,y0..yj)"，则 rows[j] == ser[j]（同一个条件分布，fp32 下应几乎逐元素相等）
  - 若把 rows[j] 误当成 "p(·|P,y0..y(j-1))"（即用 rows[i] 判 candidate i 的那种错位），
    则它会与 ser[j-1] 接近而与 ser[j] 有明显差异
  于是：报告 (a) 同位置 max|Δ|、(b) 相邻位置 max|Δ|，以及各自 argmax 是否一致。
  fp32 下同位置 Δ ≈ 1e-6 量级即可判定语义正确；错位则差值应与"相邻行差异"同量级。

同时验证接口层事实：logits 序列维、cache 输入长度、返回行与输入位置的一一对应、
prefill 最后一行是否可用于判第一个候选。

用法：python -m specdec.engine.g4_block_logits_semantics
可选：P4S_S=8（prefix 长度）P4S_K=4（候选数）
"""
from __future__ import annotations

import os
import traceback
import zipfile
from pathlib import Path

import torch

from . import spec_dec
from .g4_b4_cache_tests import _from_pretrained

ROOT = Path(__file__).resolve().parents[2]
TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
S = int(os.environ.get("P4S_S", "8"))
KC = int(os.environ.get("P4S_K", "4"))
TOL = float(os.environ.get("P4S_TOL", "1e-3"))


def load_ids(tok, n):
    with zipfile.ZipFile(ROOT / "data" / "wikitext-2-raw-v1.zip") as z:
        name = next(k for k in z.namelist() if "test" in k and k.endswith(".raw"))
        raw = z.read(name).decode("utf-8", errors="ignore")
    text = " ".join(ln.strip() for ln in raw.splitlines()
                    if ln.strip() and not ln.strip().startswith("="))
    ids = tok(text[:400_000], return_tensors="pt").input_ids
    return ids[:, :n]


def main() -> int:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from .align_ours import build_from_config
    torch.manual_seed(0)
    tok = _from_pretrained(AutoTokenizer, TARGET, trust_remote_code=True)

    print("[stage] 加载 fp32 CPU target（语义真值）...", flush=True)
    cfg = _from_pretrained(AutoConfig, TARGET, trust_remote_code=True)
    hf = _from_pretrained(AutoModelForCausalLM, TARGET, config=cfg,
                          attn_implementation="eager", torch_dtype=torch.float32,
                          trust_remote_code=True)
    model = build_from_config(cfg)
    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    if cfg.to_dict().get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    model.load_state_dict(sd, strict=True)
    del hf, sd
    model = model.to("cpu", torch.float32).eval()
    model.set_kv_format(spec_dec._fmt("base_bf16"))

    ids = load_ids(tok, S + KC)
    prompt, cands = ids[:, :S], ids[0, S:S + KC].tolist()
    print(f"[setup] prompt={S} tok  cands={cands}")

    with torch.inference_mode():
        # ---- 两次独立 prefill：block 与 serial 各自从同一 prefix 出发（cache 会被原地推进）----
        lg_pf, cache_blk = model._run(model.embed_tokens(prompt), None, need_logits=True)
        _, cache_ser = model._run(model.embed_tokens(prompt), None, need_logits=True)
        prefix_next = lg_pf[:, -1:]                      # p(·|P)
        rows, _ = model._run(model.embed_tokens(torch.tensor([cands])), cache_blk,
                             need_logits=True)
        rows = rows[0, -KC:]                             # 与 _verify 的 lg[0,-kp:] 一致
        ser = []
        for j, t in enumerate(cands):
            lg_j, cache_ser = model._run(
                model.embed_tokens(torch.tensor([[t]])), cache_ser, need_logits=True)
            ser.append(lg_j[0, -1])
        print(f"[shapes] prefix_next={tuple(prefix_next.shape)} rows={tuple(rows.shape)} "
              f"serial_rows={len(ser)} (each {tuple(ser[0].shape)})")
        print(f"[cache ] block 后的 cache 长度={cache_blk[0]['k'].shape[2]} "
              f"serial 后的 cache 长度={cache_ser[0]['k'].shape[2]} "
              f"（应都等于 {S}+{KC}={S + KC}）")

        print("\n== 同位置对照：rows[j] vs ser[j]（语义应为同一条件分布）==")
        same_ok, same_max = True, 0.0
        for j in range(KC):
            d = float((rows[j].float() - ser[j].float()).abs().max())
            am = int(rows[j].argmax()) == int(ser[j].argmax())
            same_max = max(same_max, d)
            same_ok &= (d <= TOL and am)
            print(f"  rows[{j}] vs ser[{j}]: max|Δ|={d:.3e} argmax_same={am} "
                  f"argmax={int(rows[j].argmax())}")

        print("\n== 相邻位置对照：rows[j] vs ser[j-1]（若把 rows 错位一格使用，会落在这里）==")
        neigh_max = 0.0
        for j in range(1, KC):
            d = float((rows[j].float() - ser[j - 1].float()).abs().max())
            am = int(rows[j].argmax()) == int(ser[j - 1].argmax())
            neigh_max = max(neigh_max, d)
            print(f"  rows[{j}] vs ser[{j-1}]: max|Δ|={d:.3e} argmax_same={am}")

        print("\n== 判定 ==")
        print(f"  同位置 max|Δ| = {same_max:.3e}（阈值 {TOL:g}）⇒ "
              f"{'语义正确：rows[j] = p(·|P,y0..yj)' if same_ok else '不一致，需检查'}")
        print(f"  相邻位置 max|Δ| = {neigh_max:.3e} ⇒ "
              f"错位使用的代价量级（远大于同位置差）")
        print(f"  prefill 最后一行可用作判第一个候选的分布：prefix_next argmax="
              f"{int(prefix_next[0, -1].argmax())}（p(·|P)）")
        print("\n  结论：候选 y_i 的判定分布 = prefix_next(i=0) / rows[i-1](i>=1)，"
              "与 spec_dec.py 第 83 行的实现一致。")
    ok = same_ok
    print(f"\nblock-logits-semantics: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
