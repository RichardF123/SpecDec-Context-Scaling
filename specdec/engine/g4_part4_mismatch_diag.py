"""g4_part4_mismatch_diag.py — 定位历史 Part 4 中 SpecDec != AR 的三个条件（确定性诊断）。

目标条件（`exploratory_run_1_fixed_order` 中曾不等）：
  ctx=512  + base_bf16 + SpecDec
  ctx=2048 + base_bf16 + SpecDec
  ctx=2048 + v_fp8    + SpecDec

对每个条件做确定性诊断（AR 与 SpecDec 用**同一 target 模型、同一 KV 格式、同一输入 token 序列**）：
  1. 首个不同 token 的位置与两侧 token id
  2. 该位置 AR 的 top-1/top-2 token 与 logit、**margin**
  3. SpecDec 实际选择的 token；它来自哪个 speculative round、是"被接受"还是"替换(bonus 位置)"
  4. 该轮的 draft block（候选）、target block 判定行、accepted length m、分支
  5. **block-vs-serial**：同 prefix 单步前向的 argmax 与 max|Δ|（查行索引/对齐错位）
  6. **FP32 控制**：同 prefix 在 CPU fp32 上复算 top-1/top-2/margin（查是否只是 bf16 近贴边）
  7. 结构性校验：cache 长度不变量、emitted 长度 ≤ k、g 是否与捕获的判定行 argmax 一致

归因分类（证据门槛写死，不猜）
  CACHE_ROLLBACK_OR_BOOKKEEPING : 出现 cache 长度不变量违反
  BONUS_TOKEN_OR_LENGTH         : 某轮 emitted 长度 > kp（本设计无 bonus）
  TARGET_BLOCK_LOGIT_INDEX      : **结构性**签名 —— 判定行更贴近"相邻位置"的 serial 行，
                                  而不是它自己的位置（真实错位一格实测 max|Δ| ≈ 14–17，
                                  而同位置 fp32 max|Δ| ≈ 5e-5）
  EOS_LENGTH_HANDLING           : 仅在尾部因长度/EOS 不同
  BF16_NEAR_TIE                 : 结构校验全过、非错位，且 **|fp32 margin| ≤ 同位置 block-vs-serial
                                  max|Δ|（现场噪声）** ⇒ 该位置在 bf16 下不可判定
  UNCLASSIFIED_NEEDS_INSPECTION : 以上都不满足（明确不猜）

**不要**仅凭"block-vs-serial 的 argmax 不同"就判索引错误：bf16 的形状噪声（实测同位置
0.25–0.51）本身就足以在真实间隔只有 0.02–0.05 的位置翻面。判据是"信号 vs 现场噪声"，外加
"是否更贴近相邻位置"这一结构性检查。`P4D_NOISE`（默认 0.30）只作参考常数。

用法（交互式 GPU 终端；预计 5–15 分钟）
  python -m specdec.engine.g4_part4_mismatch_diag
可选：P4D_CTX=512,2048  P4D_MAXNEW=64  P4D_FP32=1|0  P4D_NOISE=0.30  P4D_KVDIAG=v_fp8
"""
from __future__ import annotations

import os
import traceback
import zipfile
from pathlib import Path

import torch

from . import spec_dec
from .g4_b4_cache_tests import _from_pretrained, load_ours

ROOT = Path(__file__).resolve().parents[2]
TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
K = spec_dec.K

CASES = [(512, "base_bf16"), (2048, "base_bf16"), (2048, "v_fp8")]
CTXS = [int(x) for x in os.environ.get("P4D_CTX", "").split(",") if x.strip()]
MAXNEW = int(os.environ.get("P4D_MAXNEW", "64"))
USE_FP32 = os.environ.get("P4D_FP32", "1") not in ("0", "false", "False")
NOISE = float(os.environ.get("P4D_NOISE", "0.30"))
CHUNK = int(os.environ.get("P4D_CHUNK", "2048"))


def load_ids(tok, ctx: int) -> torch.Tensor:
    z = ROOT / "data" / "wikitext-2-raw-v1.zip"
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if "test" in n and n.endswith(".raw"))
        raw = zf.read(name).decode("utf-8", errors="ignore")
    text = " ".join(ln.strip() for ln in raw.splitlines()
                    if ln.strip() and not ln.strip().startswith("="))
    ids = tok(text[:400_000], return_tensors="pt").input_ids
    assert ids.shape[1] >= ctx, (ids.shape[1], ctx)
    return ids[:, :ctx]


def chunked_prefill(model, ids, chunk=CHUNK):
    with torch.inference_mode():
        cache, lg, n = None, None, ids.shape[1]
        for st in range(0, n, chunk):
            lg, cache = model._run(model.embed_tokens(ids[:, st:st + chunk]), cache,
                                   need_logits=(st + chunk >= n))
    return lg, cache


def top2(row) -> dict:
    r = row.float()
    v, i = r.topk(2)
    return dict(top1=int(i[0]), top1_logit=float(v[0]), top2=int(i[1]), top2_logit=float(v[1]),
                margin=float(v[0] - v[1]))


def ar_run(tgt, ids, n, eos, kv):
    """AR 同档基线，逐 token 捕获判定行（position -> logits）。"""
    tgt.set_kv_format(spec_dec._fmt(kv))
    lg, cache = chunked_prefill(tgt, ids)
    gen, rows = [], []
    cur = lg[:, -1:]
    for i in range(n):
        rows.append(cur[0, -1].detach().cpu())
        tok = int(cur.reshape(-1, cur.shape[-1])[-1].argmax())
        gen.append(tok)
        if eos is not None and tok == eos:
            break
        if i == n - 1:
            break
        cur, cache = spec_dec._step(tgt, tok, cache)
    return gen, rows


def spec_run_capture(tgt, drf, ids, n, eos, kv):
    """SpecDec，捕获每轮：saved_next / rows / cands / m / emitted / 分支 + 结构不变量。"""
    events: list[dict] = []
    orig_pre, orig_ver, orig_step = spec_dec._prefill, spec_dec._verify, spec_dec._step

    def prefill(model, ids_):
        lg, cache = orig_pre(model, ids_)
        if model is tgt:
            events.append(dict(kind="prefill", lg=lg[:, -1:].detach().cpu())
                          )
        return lg, cache

    def verify(target, cache, cands):
        rows, blk = orig_ver(target, cache, cands)
        events.append(dict(kind="verify", cands=list(cands),
                           rows=rows.detach().cpu(), kp=len(cands)))
        return rows, blk

    def step(model, token, cache):
        lg, c = orig_step(model, token, cache)
        if model is tgt:
            events.append(dict(kind="step", token=int(token), lg=lg.detach().cpu()))
        return lg, c

    spec_dec._prefill, spec_dec._verify, spec_dec._step = prefill, verify, step
    trace: list[dict] = []
    try:
        out = spec_dec.spec_decode_greedy(tgt, drf, ids, n, k=K, eos_id=eos,
                                          target_kv=kv, log=lambda d: trace.append(d))
    finally:
        spec_dec._prefill, spec_dec._verify, spec_dec._step = orig_pre, orig_ver, orig_step
    return out, trace, events


def rounds_from_events(events) -> list[dict]:
    """把事件流重组成轮次：每轮 = (saved_next, verify 行, m, cands)。

    **全接受路径不调用 `_step`**：下一轮 `saved_next` 直接取本轮 `rows[kp-1]`（block 最后一行）；
    错配路径才由随后的 `_step` 覆盖。这里必须与实现一致，否则 `saved` 会变成 None（假证据）。
    """
    rounds, last_tgt = [], None
    for e in events:
        if e["kind"] in ("prefill", "step"):
            last_tgt = e["lg"][0, -1]
        elif e["kind"] == "verify":
            rounds.append(dict(saved=last_tgt, rows=e["rows"], cands=e["cands"], kp=e["kp"]))
            last_tgt = e["rows"][e["kp"] - 1]
    return rounds


def main() -> int:
    from transformers import AutoTokenizer
    torch.manual_seed(0)
    tok = _from_pretrained(AutoTokenizer, TARGET, trust_remote_code=True)
    eos = tok.eos_token_id
    cases = [(c, kv) for (c, kv) in CASES if (not CTXS or c in CTXS)]

    print(f"[env] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} "
          f"noise_band={NOISE} fp32_control={USE_FP32}", flush=True)
    print(f"[cases] {cases}", flush=True)
    print("[stage] loading target 1.5B + draft 0.5B ...", flush=True)
    tgt, _ = load_ours(TARGET)
    drf, _ = load_ours(DRAFT)

    tgt32 = None
    if USE_FP32:
        print("[stage] loading fp32 CPU target（仅用于分歧点 margin 控制）...", flush=True)
        from transformers import AutoConfig, AutoModelForCausalLM
        from .align_ours import build_from_config
        cfg = _from_pretrained(AutoConfig, TARGET, trust_remote_code=True)
        hf = _from_pretrained(AutoModelForCausalLM, TARGET, config=cfg,
                              attn_implementation="eager", torch_dtype=torch.float32,
                              trust_remote_code=True)
        tgt32 = build_from_config(cfg)
        sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
        if cfg.to_dict().get("tie_word_embeddings", False):
            sd.pop("lm_head.weight", None)
        tgt32.load_state_dict(sd, strict=True)
        del hf, sd
        tgt32 = tgt32.to("cpu", torch.float32).eval()

    verdicts = []
    for ctx, kv in cases:
        print(f"\n===== ctx={ctx} kv={kv} SpecDec vs AR =====", flush=True)
        ids = load_ids(tok, ctx).cuda()
        ar_gen, ar_rows = ar_run(tgt, ids, MAXNEW, eos, kv)
        out, trace, events = spec_run_capture(tgt, drf, ids, MAXNEW, eos, kv)
        sg = out["gen"]
        d = next((i for i, (a, b) in enumerate(zip(sg, ar_gen)) if a != b), None)
        print(f"  AR len={len(ar_gen)} SpecDec len={len(sg)} first_diff_index={d}")
        if d is None:
            print("  => 本条件**未复现**不一致（该条件记为已通过；历史不一致未在此复现）")
            verdicts.append((ctx, kv, "NO_DIVERGENCE", {}))
            continue
        # ---- 轮次归属 ----
        rounds = rounds_from_events(events)
        own = None
        for ti, t in enumerate(trace):
            off = d - (t["pre"] - ctx)
            if 0 <= off < len(t["emitted"]):
                own = dict(round=ti, off=off, kp=t["Kp"], m=t["m"], pre=t["pre"],
                           cands=t["cands"], g=t["g"], emitted=t["emitted"],
                           branch=("accepted" if off < t["m"] else "replacement"))
                break
        print(f"  AR[{d}]={ar_gen[d]} ({tok.decode([ar_gen[d]])!r})  "
              f"SpecDec[{d}]={sg[d]} ({tok.decode([sg[d]])!r})")
        ev: dict = {"d": d}
        if own:
            print(f"  [round] r={own['round']} Kp={own['kp']} m={own['m']} pre={own['pre']} "
                  f"off={own['off']} branch={own['branch']}")
            print(f"          cands={own['cands']} g={own['g']} emitted={own['emitted']}")
            r = rounds[own["round"]]
            judge = r["saved"] if own["off"] == 0 else r["rows"][own["off"] - 1]
            jt = top2(judge)
            print(f"  [judge row] top1={jt['top1']}({jt['top1_logit']:.4f} "
                  f"{tok.decode([jt['top1']])!r}) top2={jt['top2']}({jt['top2_logit']:.4f} "
                  f"{tok.decode([jt['top2']])!r}) margin={jt['margin']:.4f}")
            ev["judge"] = jt
            # 结构性校验
            g_recomp = [int((r["saved"] if j == 0 else r["rows"][j - 1]).argmax())
                        for j in range(r["kp"])]
            ev["g_recomputed_ok"] = (g_recomp == own["g"])
            ev["emitted_len_ok"] = all(len(t["emitted"]) <= t["Kp"] for t in trace)
            ev["invariant_violations"] = [
                t for t in trace
                if t["tgt_len"] != t["drf_len"] or
                t["tgt_len"] != t["pre"] + (t["m"] + 1 if t["m"] < t["Kp"] else t["Kp"])]
            # ---- 语义位置判别：判定行 vs **同位置 / 相邻位置**的 serial 前向 ----
            def serial_row(k):
                if k < 0:
                    return None
                pr = torch.cat([ids, torch.tensor([sg[:k]], dtype=ids.dtype,
                                                  device=ids.device)], dim=1)
                lg_, _ = chunked_prefill(tgt, pr)
                return lg_[0, -1].detach().cpu()

            pref = torch.cat([ids, torch.tensor([sg[:d]], dtype=ids.dtype, device=ids.device)],
                             dim=1)
            for nm, k in (("own", d), ("prev", d - 1), ("next", d + 1)):
                r_ = serial_row(k)
                if r_ is None:
                    continue
                dd = float((judge.float() - r_.float()).abs().max())
                am = int(judge.argmax()) == int(r_.argmax())
                ev[f"bvs_{nm}_max_abs"] = dd
                ev[f"bvs_{nm}_argmax_same"] = am
                print(f"  [serial {nm:<4}] argmax={int(r_.argmax())} "
                      f"max|Δ vs judge row|={dd:.4f} argmax_same={am}")
            ev["judge_row_pos"] = int(ids.shape[1]) + d - 1
            print(f"  [位置] 判定行位置 = {ev['judge_row_pos']}；"
                  f"prefix 长度 = {int(ids.shape[1])}；分歧 index = {d}")
            print(f"  [structural] g_recomputed_ok={ev['g_recomputed_ok']} "
                  f"emitted_len_ok={ev['emitted_len_ok']} "
                  f"invariant_violations={len(ev['invariant_violations'])}")
        ar_ev = top2(ar_rows[d]) if d < len(ar_rows) else None
        if ar_ev:
            ev["ar"] = ar_ev
            print(f"  [AR bf16 ] top1={ar_ev['top1']}({ar_ev['top1_logit']:.4f} "
                  f"{tok.decode([ar_ev['top1']])!r}) top2={ar_ev['top2']}({ar_ev['top2_logit']:.4f} "
                  f"{tok.decode([ar_ev['top2']])!r}) margin={ar_ev['margin']:.4f}")
        if tgt32 is not None:
            with torch.inference_mode():
                lg32, _ = chunked_prefill(tgt32, pref.cpu())
            f32 = top2(lg32[0, -1])
            ev["fp32"] = f32
            print(f"  [FP32 ctrl] top1={f32['top1']}({f32['top1_logit']:.4f} "
                  f"{tok.decode([f32['top1']])!r}) top2={f32['top2']}({f32['top2_logit']:.4f} "
                  f"{tok.decode([f32['top2']])!r}) margin={f32['margin']:.4f}")
            print(f"              fp32 argmax == AR bf16 token: "
                  f"{f32['top1'] == ar_gen[d]} | == SpecDec token: {f32['top1'] == sg[d]}")
        # ---- 分类（判别式按证据强度排序；判据不用人为常数，见下）----
        # signal = |fp32 margin|（该决策点的真实间隔）
        # noise  = 同一位置 block-vs-serial 的 max|Δ|（bf16 形状噪声的**现场实测值**）
        # 真实"错位一格"的签名是结构性的：判定行会更贴近**相邻位置**而不是它自己的位置。
        # 参考量级（g4_block_logits_semantics 实测，真实模型 fp32）：
        #   同位置 max|Δ| ≈ 5e-5（argmax 相同）；错位一格 max|Δ| ≈ 14–17（argmax 不同）。
        own_ok = ev.get("bvs_own_argmax_same", True)
        own_d = ev.get("bvs_own_max_abs", float("nan"))
        nb = [ev.get(k) for k in ("bvs_prev_max_abs", "bvs_next_max_abs")
              if isinstance(ev.get(k), (int, float))]
        off_by_one = (own_ok is False) and bool(nb) and min(nb) < own_d
        signal = abs(ev["fp32"]["margin"]) if ev.get("fp32") else None
        if ev.get("invariant_violations"):
            v = "CACHE_ROLLBACK_OR_BOOKKEEPING"
        elif not ev.get("emitted_len_ok", True):
            v = "BONUS_TOKEN_OR_LENGTH"
        elif off_by_one:
            v = "TARGET_BLOCK_LOGIT_INDEX"
        elif signal is not None and own_d == own_d and signal <= own_d:
            v = "BF16_NEAR_TIE"
        elif len(sg) != len(ar_gen) and d >= min(len(sg), len(ar_gen)) - 1:
            v = "EOS_LENGTH_HANDLING"
        else:
            v = "UNCLASSIFIED_NEEDS_INSPECTION"
        ev["signal_fp32_margin"] = signal
        ev["noise_own_max_abs"] = own_d
        print(f"  => 归因：**{v}**")
        print(f"     判别：signal=|fp32 margin|={signal if signal is None else round(signal, 4)}  vs  "
              f"noise=同位置 block-vs-serial max|Δ|={own_d:.4f}"
              f"（参考：错位一格约 14–17）")
        print(f"     off_by_one={off_by_one}（own_argmax_same={own_ok}，"
              f"相邻 max|Δ|={[round(x, 3) for x in nb]}）"
              f"  噪声带常数参考 P4D_NOISE={NOISE}")
        if v == "BF16_NEAR_TIE":
            print(f"     证据：真实间隔 {signal:.4f} ≤ 现场噪声 {own_d:.4f} ⇒ 该位置在 bf16 下不可判定；"
                  f"结构校验全过、非错位（相邻位置差值 "
                  f"{[round(x, 2) for x in nb]} 远大于同位置）")
        elif v == "UNCLASSIFIED_NEEDS_INSPECTION":
            print(f"     若 signal 明显大于 noise 且 own_argmax_same=False，则不是近贴边，"
                  f"需人工查实现（**不得豁免**）")
        verdicts.append((ctx, kv, v, ev))

    print("\n===== 汇总 =====")
    for ctx, kv, v, ev in verdicts:
        extra = ""
        if ev:
            extra = (f"  first_diff={ev.get('d')} "
                     f"fp32_margin={ev.get('fp32', {}).get('margin')} "
                     f"bvs_maxΔ={ev.get('block_vs_serial_max_abs')}")
        print(f"  ctx={ctx:<5} {kv:<10} {v}{extra}")
    print("\n注意：若结论为实现错误，必须重跑 G4 B5、G4 B6、三个失败条件与 8K 正确性对照；"
          "若为 bf16 近贴边，上面的 margin 就是证据（不得仅凭猜测豁免）。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
