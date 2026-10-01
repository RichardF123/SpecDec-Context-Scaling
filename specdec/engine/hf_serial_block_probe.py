"""hf_serial_block_probe.py — **HF 单步 / 块计算对照**（外部实现，独立诊断入口）。

回答：不调用自研 Spec 状态机、只用 HF target、**相同初始缓存 + 相同 token 输入**，
单步（s=1）与块（s=4）前向是否也产生数值差异？

模式（`HFSB_MODE`）
  * `selftest`：CPU + **微型 HF 模型**，验证 ① 行对齐 ② 缓存深拷贝不共享存储 ③ mask/可见范围
    ④ 快照未被改写 ⑤ 有限性；不做任何结论外推。
  * `run`：GPU BF16、Qwen2.5-1.5B-Instruct、`attn_implementation="eager"`、未量化 KV；
    用 HF 自己分块构建 8K 历史到 `pre`，再深拷贝出 S/B 两臂。

对齐语义（必须不能错位）
  * Serial 第 j 步：喂第 j 个 token，`T_s = pre+j+1`，可见 `key ≤ pre+j`；
  * Block 第 j 行：一次喂 4 个 token，`T_b = pre+4`，可见 `key ≤ pre+j`（因果 mask）；
  ⇒ **可见集合相同、T 不同**；两者都预测位置 `pre+j+1`。
  本工具用 `output_attentions=True` 对 4-token 调用做**实际张量级** mask 支撑集检查。

来源（只作固定输入，不用于"强制正确答案"）：
  输入 family、前缀 token、`pre`、4 个 token 均取自**已登记**的 F13/index50 取证记录（文件名见 JSON）。
"""
from __future__ import annotations

import glob
import json
import os
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TAG = "hfsb_20261001"
TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
MODE = os.environ.get("HFSB_MODE", "selftest").strip().lower()
FAMILY = os.environ.get("HFSB_FAMILY", "F13").strip().upper()
CTX = int(os.environ.get("HFSB_CTX", "8192"))
CHUNK = int(os.environ.get("HFSB_CHUNK", "2048"))
PRE = int(os.environ.get("HFSB_PRE", "8240"))
TOKENS = [int(x) for x in os.environ.get("HFSB_TOKENS", "220,16,16,1154").split(",")]
TINY = os.environ.get("HFSB_TINY", "1") == "1"


def _newest(pat):
    return Path(sorted(glob.glob(str(OUTDIR / pat)))[-1])


def _sha(t: torch.Tensor, n: int = 16) -> str:
    import hashlib
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:n]


def _cache_len(c) -> int:
    return int(c.get_seq_length())


def clone_cache(c):
    """**深拷贝到独立存储**（HF v5 `DynamicCache.layers[i].keys/values`；无 legacy 接口）。"""
    new = type(c)()
    for lay in c.layers:
        d = lay.__class__()
        for a in ("keys", "values"):
            t = getattr(lay, a, None)
            if t is not None:
                setattr(d, a, t.detach().clone())
        for a in ("is_initialized", "is_sliding", "layer_idx"):
            if hasattr(lay, a):
                try:
                    setattr(d, a, getattr(lay, a))
                except Exception:                                  # noqa: BLE001
                    pass
        new.layers.append(d)
    return new


def _kv_digest(c) -> dict:
    out = {}
    for i, lay in enumerate(c.layers):
        for a in ("keys", "values"):
            t = getattr(lay, a)
            out[f"L{i}_{a[0]}"] = dict(shape=[int(x) for x in t.shape], dtype=str(t.dtype),
                                       finite=bool(torch.isfinite(t).all()), sha=_sha(t))
    return out


def _row_fields(r) -> dict:
    x = r.detach().float().reshape(-1)
    t2 = torch.topk(x, 2).values
    am = int(torch.argmax(x))
    tied = [int(i) for i in torch.nonzero(x == x[am]).reshape(-1)]
    return dict(argmax=am, max_score=float(x.max()), margin=float(t2[0] - t2[1]),
                n_tied_max=len(tied), tie=True if len(tied) > 1 else False,
                top5=[int(i) for i in torch.topk(x, 5).indices],
                finite=bool(torch.isfinite(x).all()), sha=_sha(r))


def _cmp(a, b) -> dict:
    x, y = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    d = (x - y).abs()
    return dict(elementwise_equal=bool(torch.equal(x, y)), max_abs=float(d.max()),
                mean_abs=float(d.mean()), n_diff=int((d > 0).sum()),
                a_argmax=int(torch.argmax(x)), b_argmax=int(torch.argmax(y)),
                same_argmax=bool(int(torch.argmax(x)) == int(torch.argmax(y))))


def _mask_support(attn_row) -> dict:
    """从 attention 概率（已 softmax，形状 `(heads, k_len)`）得到**逐 head 的可见支撑**。

    ⚠️ 必须**逐 head** 统计：把 `(heads, k_len)` 展平后比首尾下标会被"两臂 k_len 不同"骗到
    （serial 行 k_len=7、block 行 k_len=10，尾部 3 个是被 mask 的 0 槽位）—— 首跑就踩了这个。
    """
    per_head = []
    for h in range(attn_row.shape[0]):
        nz = torch.nonzero(attn_row[h].reshape(-1) > 0).reshape(-1)
        per_head.append(dict(count=int(nz.numel()),
                             last=int(nz[-1]) if nz.numel() else None,
                             first=int(nz[0]) if nz.numel() else None))
    return dict(per_head=per_head, k_len=int(attn_row.shape[-1]),
                counts=[x["count"] for x in per_head], lasts=[x["last"] for x in per_head])


def _selftest() -> int:
    from transformers import Qwen2Config, Qwen2ForCausalLM
    npass = nfail = 0

    def check(name, cond, extra=None):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  [ok] {name}")
        else:
            nfail += 1
            print(f"  [FAIL] {name}" + (f"  {str(extra)[:200]}" if extra is not None else ""))

    cfg = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=512)
    cfg._attn_implementation = "eager"       # transformers 5.x：构造前设置 config 字段
    m = Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval()
    torch.manual_seed(0)
    prompt = torch.randint(0, 128, (1, 6))
    with torch.inference_mode():
        out = m(prompt, use_cache=True)
    print(f"  （自测）attn impl = {getattr(m.config, '_attn_implementation', None)}")
    C = out.past_key_values
    check("prefill 后 cache 长度 = 6", _cache_len(C) == 6, _cache_len(C))
    snap = _kv_digest(C)
    cs, cb = clone_cache(C), clone_cache(C)
    check("★ 两臂深拷贝不共享存储",
          not any(cs.layers[i].keys.data_ptr() == cb.layers[i].keys.data_ptr()
                  for i in range(len(cs.layers))))
    toks = [10, 11, 12, 13]
    S, B, attn_s, attn_b = [], [], [], []
    for j, t in enumerate(toks):
        with torch.inference_mode():
            o = m(torch.tensor([[t]]), past_key_values=cs, use_cache=True,
                  position_ids=torch.tensor([[6 + j]]),
                  attention_mask=torch.ones(1, 7 + j, dtype=torch.long), output_attentions=True)
        S.append(o.logits[0, -1].clone())
        attn_s.append(o.attentions[-1][0, :, 0, :].clone())      # 最后一层、第 0 个 head
    with torch.inference_mode():
        ob = m(torch.tensor([toks]), past_key_values=cb, use_cache=True,
               position_ids=torch.arange(6, 10).view(1, -1),
               attention_mask=torch.ones(1, 10, dtype=torch.long), output_attentions=True)
    B = [ob.logits[0, j].clone() for j in range(4)]
    attn_b = [ob.attentions[-1][0, :, j, :].clone() for j in range(4)]
    check("serial 各步 cache 长度递增到 10", _cache_len(cs) == 10, _cache_len(cs))
    check("block 后 cache 长度 = 10", _cache_len(cb) == 10, _cache_len(cb))
    check("★ 行对齐：serial 第 j 步 ↔ block 第 j 行，均预测 pre+j+1",
          all(tuple(x.shape) == tuple(B[j].shape) for j, x in enumerate(S)))
    sup_ok = True
    for j in range(4):
        a = _mask_support(attn_s[j])
        bb = _mask_support(attn_b[j])
        if a["counts"] != bb["counts"] or a["lasts"] != bb["lasts"] or a["lasts"][0] != 6 + j:
            sup_ok = False
            print(f"    j={j}: serial counts/lasts={a['counts']}/{a['lasts']} "
                  f"block={bb['counts']}/{bb['lasts']}（k_len {a['k_len']} vs {bb['k_len']}）")
    check("★ 实际 mask 支撑集一致（逐 head：可见数与最后可见 key 都相同，且 = pre+j）", sup_ok)
    check("★ 原始快照未被改写", _kv_digest(C) == snap)
    check("logits 有限", all(bool(torch.isfinite(x.float()).all()) for x in S + B))
    d = [_cmp(S[j], B[j]) for j in range(4)]
    print("  （自测信息，不做全局外推）serial vs block 的 max|Δ| =",
          [round(x["max_abs"], 6) for x in d])
    print(f"\n短输入自测：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


def _run() -> int:
    import transformers
    from transformers import AutoModelForCausalLM, DynamicCache
    from . import family_input as FI
    from . import g5_8k_spec_variance_probe as P5V

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="hf_serial_block_probe", family=FAMILY, ctx=CTX, chunk=CHUNK,
               pre=PRE, tokens=TOKENS, model=TARGET,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               versions=dict(torch=torch.__version__, cuda=torch.version.cuda,
                             transformers=transformers.__version__),
               semantics="HF 外部对照：相同初始缓存 + 相同 token 输入，s=1 串行 vs s=4 块；"
                         "**固定输入 teacher-forced 数值对照**，不是自由生成，也不是完整 SpecDec 验收。")
    print(f"[hfsb] run_id={run_id} family={FAMILY} pre={PRE} tokens={TOKENS}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    ok, why = P5V.commit_guard()
    print(f"[hfsb] commit: {why}")
    if not ok:
        rec["stage"] = "refused"
        (OUTDIR / f"{TAG}_{FAMILY}_{run_id}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return 3
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True)
        ids_list, fam_meta = FI.load_family_ids(tok, FAMILY, ctx=CTX)
        ids = torch.tensor([ids_list], device="cuda")
        # 前缀来源：已登记的 AR 捕获记录（固定输入，不用于强制答案）
        cap_ar = _newest("f13i50_20261001_capture_ar_*.pt")
        prefix_full = list(torch.load(cap_ar, map_location="cpu")["prefix_tokens"])
        A_len = PRE - CTX
        A_req = len(prefix_full) >= A_len
        if not A_req:
            raise RuntimeError(f"捕获前缀只有 {len(prefix_full)} 个 token，少于需要的 {A_len}")
        prefix = prefix_full[:A_len]        # 只需前 A_len 个就能把 cache 建到 pre
        rec["sources"] = dict(family_manifest=fam_meta, prefix_pt=str(cap_ar),
                              prefix_note=f"取已登记捕获 prefix_tokens 的**前 {A_len} 个**"
                                          f"（到 index{A_len - 1}；该记录共 {len(prefix_full)} 个）",
                              step_tokens=TOKENS)
        # 标准输入块（与其它工具一致）：治理守卫按 `input.family` / `input.source_kind` 识别消费方
        rec["input"] = dict(source_kind="manifest", family=fam_meta,
                            ids_hash=_sha(ids.detach().cpu().reshape(-1).float()),
                            ctx=int(ids.shape[1]), eos_id=int(tok.eos_token_id))
        model = AutoModelForCausalLM.from_pretrained(TARGET, torch_dtype=torch.bfloat16,
                                                     attn_implementation="eager").to("cuda").eval()
        impl = getattr(model.config, "_attn_implementation", None)
        rec["impl"] = dict(attn_implementation=str(impl))
        A_req = (str(impl) == "eager")
        if not A_req:
            raise RuntimeError(f"attn_implementation 必须是 eager，实测 {impl!r}")
        # ---- HF 自己分块构建到 pre（不导入自研缓存）----
        cache = DynamicCache()
        pos = 0
        seq = torch.cat([ids, torch.tensor([prefix], device="cuda")], dim=1)
        assert int(seq.shape[1]) == PRE, f"seq 长度 {int(seq.shape[1])} != pre {PRE}"
        with torch.inference_mode():
            for st in range(0, int(seq.shape[1]), CHUNK):
                piece = seq[:, st:st + CHUNK]
                n = int(piece.shape[1])
                o = model(piece, past_key_values=cache, use_cache=True,
                          position_ids=torch.arange(st, st + n, device="cuda").view(1, -1),
                          attention_mask=torch.ones(1, st + n, dtype=torch.long, device="cuda"))
                pos = st + n
        rec["initial_cache"] = dict(cache_len=_cache_len(cache), n_layers=len(cache.layers),
                                    kv_digest=_kv_digest(cache))
        if _cache_len(cache) != PRE:
            raise RuntimeError(f"HF 历史 cache 长度 {_cache_len(cache)} != pre {PRE}")
        snap_digest = json.dumps(rec["initial_cache"]["kv_digest"], sort_keys=True)
        # ---- 共享同一初始缓存（先克隆，再深拷贝；不执行任何自研状态机）----
        C0 = clone_cache(cache)

        def arm_serial(c):
            rows, masks = [], []
            for j, t in enumerate(TOKENS):
                with torch.inference_mode():
                    o = model(torch.tensor([[t]], device="cuda"), past_key_values=c, use_cache=True,
                              position_ids=torch.tensor([[PRE + j]], device="cuda"),
                              attention_mask=torch.ones(1, PRE + j + 1, dtype=torch.long, device="cuda"),
                              output_attentions=True)
                rows.append(o.logits[0, -1].clone())
                masks.append(_mask_support(o.attentions[-1][0, :, 0, :]))
            return rows, masks, c

        def arm_block(c):
            with torch.inference_mode():
                o = model(torch.tensor([TOKENS], device="cuda"), past_key_values=c, use_cache=True,
                          position_ids=torch.arange(PRE, PRE + len(TOKENS), device="cuda").view(1, -1),
                          attention_mask=torch.ones(1, PRE + len(TOKENS), dtype=torch.long, device="cuda"),
                          output_attentions=True)
            rows = [o.logits[0, j].clone() for j in range(len(TOKENS))]
            masks = [_mask_support(o.attentions[-1][0, :, j, :]) for j in range(len(TOKENS))]
            return rows, masks, c

        S1, Sm1, cs1 = arm_serial(clone_cache(C0))
        B1, Bm1, cb1 = arm_block(clone_cache(C0))
        S2, _, cs2 = arm_serial(clone_cache(C0))
        B2, _, cb2 = arm_block(clone_cache(C0))
        if snap_digest != json.dumps(_kv_digest(cache), sort_keys=True):
            raise RuntimeError("原始 HF 历史 cache 被改写")
        rec["alignment"] = dict(
            rule="serial 第 j 步 T=pre+j+1、可见 key ≤ pre+j；block 第 j 行 T=pre+4、可见 key ≤ pre+j；"
                 "两者都预测 pre+j+1",
            serial_rows=[dict(j=j, pred_pos=PRE + j + 1, T=PRE + j + 1, mask_support=Sm1[j])
                         for j in range(len(TOKENS))],
            block_rows=[dict(j=j, pred_pos=PRE + j + 1, T=PRE + len(TOKENS), mask_support=Bm1[j])
                        for j in range(len(TOKENS))],
            mask_support_equal=all(Sm1[j]["counts"] == Bm1[j]["counts"]
                                   and Sm1[j]["lasts"] == Bm1[j]["lasts"]
                                   for j in range(len(TOKENS))),
            mask_note="`mask_support` 取自 **实际 attention 概率张量**（>0 支撑集）",
            row_source="block 行 j 与 serial 第 j 步**对齐到同一预测位置**；错位比较被禁止")
        if not rec["alignment"]["mask_support_equal"]:
            raise RuntimeError("实际 mask 支撑集不一致 ⇒ 不能做同前缀比较")
        rec["rows"] = dict(
            serial=[dict(j=j, pred_pos=PRE + j + 1, **_row_fields(S1[j])) for j in range(len(TOKENS))],
            block=[dict(j=j, pred_pos=PRE + j + 1, **_row_fields(B1[j])) for j in range(len(TOKENS))])
        rec["compare_cross_shape"] = [dict(j=j, cluster="serial_vs_block", **_cmp(S1[j], B1[j]))
                                      for j in range(len(TOKENS))]
        rec["compare_same_path_repeat"] = (
            [dict(j=j, cluster="serial_rep1_vs_rep2", **_cmp(S1[j], S2[j])) for j in range(len(TOKENS))]
            + [dict(j=j, cluster="block_rep1_vs_rep2", **_cmp(B1[j], B2[j])) for j in range(len(TOKENS))])
        rec["kv_after"] = dict(
            serial_vs_block=[dict(layer=i, keys_equal=bool(torch.equal(cs1.layers[i].keys,
                                                                      cb1.layers[i].keys)),
                                  values_equal=bool(torch.equal(cs1.layers[i].values,
                                                                cb1.layers[i].values)))
                             for i in range(len(cs1.layers))],
            serial_repeat_equal=all(torch.equal(cs1.layers[i].keys, cs2.layers[i].keys)
                                    and torch.equal(cs1.layers[i].values, cs2.layers[i].values)
                                    for i in range(len(cs1.layers))),
            block_repeat_equal=all(torch.equal(cb1.layers[i].keys, cb2.layers[i].keys)
                                   and torch.equal(cb1.layers[i].values, cb2.layers[i].values)
                                   for i in range(len(cb1.layers))),
            cache_len=dict(serial=_cache_len(cs1), block=_cache_len(cb1)))
        rec["resources"] = dict(pt_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                                pt_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20,
                                cuda_free_mib=torch.cuda.mem_get_info()[0] / 2 ** 20)
        rec["stage"] = "complete"
        p = OUTDIR / f"{TAG}_{FAMILY}_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        rec["artifacts"] = dict(json=str(p))
        print("[hfsb] 对齐：", json.dumps(rec["alignment"]["mask_support_equal"]))
        for x in rec["compare_cross_shape"]:
            print(f"  j={x['j']} serial_vs_block: same_argmax={x['same_argmax']} "
                  f"argmax {x['a_argmax']} vs {x['b_argmax']} max|Δ|={x['max_abs']:.6g} "
                  f"mean={x['mean_abs']:.6g} n_diff={x['n_diff']}")
        print("  同路径重复一致：", json.dumps(
            [x["elementwise_equal"] for x in rec["compare_same_path_repeat"]]))
        print("  KV serial vs block 逐层相同：",
              json.dumps([(x["keys_equal"], x["values_equal"]) for x in rec["kv_after"]["serial_vs_block"]]))
        print(f"[hfsb] RESULT_JSON={p}")
        return 0
    except Exception as e:                                        # noqa: BLE001
        rec["stage"] = "crashed"
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        p = OUTDIR / f"{TAG}_{FAMILY}_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[hfsb] !! 停止：{e}")
        print(f"[hfsb] RESULT_JSON={p}")
        return 2


if __name__ == "__main__":
    if MODE == "selftest":
        raise SystemExit(_selftest())
    raise SystemExit(_run())
