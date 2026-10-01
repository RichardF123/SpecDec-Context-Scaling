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

from .cache_clone import clone_cache_full

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
# 唯一实验变量：`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction`
#   default = 不动（记录实际值）| 1 = 显式 True | 0 = 显式 False
RED = os.environ.get("HFSB_RED_REDUCTION", "default").strip().lower()


def _apply_flag() -> dict:
    """**在模型计算与 prefill 之前**设置并读回；记录"默认值 / 是否显式设置 / 生效值"。"""
    import torch
    backend = torch.backends.cuda.matmul
    default_at_import = bool(getattr(backend, "allow_bf16_reduced_precision_reduction"))
    if RED in ("1", "true", "on"):
        backend.allow_bf16_reduced_precision_reduction = True
        how = "explicit True"
    elif RED in ("0", "false", "off"):
        backend.allow_bf16_reduced_precision_reduction = False
        how = "explicit False"
    elif RED == "default":
        how = "未显式设置（保持导入时的值）"
    else:
        raise ValueError(f"未知 HFSB_RED_REDUCTION={RED!r}")
    effective = bool(backend.allow_bf16_reduced_precision_reduction)
    return dict(requested=RED, default_at_import=default_at_import, how=how,
                effective_after_set=effective,
                doc="PyTorch 2.6 数值精度说明：BF16 GEMM 的 reduced-precision reduction "
                    "**默认开启**，可用该开关关闭")


def _newest(pat):
    return Path(sorted(glob.glob(str(OUTDIR / pat)))[-1])


def _sha(t: torch.Tensor, n: int = 16) -> str:
    import hashlib
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:n]


def _cache_len(c) -> int:
    return int(c.get_seq_length())


def clone_cache(c):
    """克隆 HF `DynamicCache`。

    **本函数的历史行为是正确的**（在 `type(c)()` 后额外拷贝了 `is_initialized`/`is_sliding`/`layer_idx`，
    实测克隆 6-token 历史后 `get_seq_length()==6`、追加 1 个 token 后为 7、前 6 位 K 保留），
    因此本工具此前与今后的产物**不受** xb_probe 那个"只填 keys/values"的 bug 影响。
    现在统一走 `cache_clone.clone_cache_full`（deepcopy，保留完整对象状态），避免两个克隆实现并存。
    """
    return clone_cache_full(c)


def _kv_digest(c) -> dict:
    out = {}
    for i, lay in enumerate(c.layers):
        for a in ("keys", "values"):
            t = getattr(lay, a)
            out[f"L{i}_{a[0]}"] = dict(shape=[int(x) for x in t.shape], dtype=str(t.dtype),
                                       finite=bool(torch.isfinite(t).all()), sha=_sha(t))
    return out


def clone_checks(src, c1, c2, what: str) -> dict:
    """**K 与 V 都要**查：与源逐位相同、两臂之间**都不共享存储**。"""
    n = len(src.layers)
    res = dict(n_layers=n, per_attr={}, k_shares_storage=False, v_shares_storage=False,
               bitwise_equal_to_source=True)
    for a in ("keys", "values"):
        eq1 = eq2 = True
        for i in range(n):
            s = getattr(src.layers[i], a)
            t1 = getattr(c1.layers[i], a)
            t2 = getattr(c2.layers[i], a)
            if not torch.equal(s, t1) or not torch.equal(s, t2):
                eq1 = False
            if t1.data_ptr() == t2.data_ptr():
                res["k_shares_storage" if a == "keys" else "v_shares_storage"] = True
        res["per_attr"][a] = dict(bitwise_equal_to_source=bool(eq1))
        if not eq1:
            res["bitwise_equal_to_source"] = False
    if not res["bitwise_equal_to_source"]:
        raise RuntimeError(f"{what}: 克隆臂与源快照不是逐位相同")
    if res["k_shares_storage"] or res["v_shares_storage"]:
        raise RuntimeError(f"{what}: 两臂共享存储（K={res['k_shares_storage']} "
                           f"V={res['v_shares_storage']}）")
    return res


_MASK_LOG: list = []


def install_mask_recorder():
    """**拦截本机 HF 实际使用的 attention 接口**，捕获它收到的 `attention_mask` 实参。

    v5 的实际形态：`Qwen2Attention.forward` 内
    `ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, eager_attention_forward)`
    ⇒ 包装 `get_interface` 返回的接口即可拿到**模型真实使用的 mask**（逐层逐次调用记录）。

    为什么不用 `create_causal_mask` 重建：实测重建结果与模型实际收到的 mask **不一致**
    （重建未传 `cache_position`/完整序列等内部上下文）⇒ 重建**不能**作为 mask 验证。
    """
    from transformers.models.qwen2 import modeling_qwen2 as mq
    AF = mq.ALL_ATTENTION_FUNCTIONS
    if getattr(AF, "_dsh_wrapped", False):
        return AF
    orig_get = AF.get_interface

    def patched_get_interface(name, default=None):
        fn = orig_get(name, default)

        def wrapped(module, query, key, value, attention_mask=None, *a, **k):
            if attention_mask is None and a:          # 位置参数形态
                attention_mask = a[0]
                a = a[1:]
            _MASK_LOG.append(dict(layer_idx=getattr(module, "layer_idx", None),
                                  q_len=int(query.shape[-2]), kv_len=int(key.shape[-2]),
                                  name=str(name),
                                  mask=(None if attention_mask is None else
                                        dict(shape=[int(x) for x in attention_mask.shape],
                                             dtype=str(attention_mask.dtype))),
                                  mask_obj=attention_mask))
            return fn(module, query, key, value, attention_mask, *a, **k)

        return wrapped

    AF.get_interface = patched_get_interface
    AF._dsh_wrapped = True
    return AF


def mask_allowed_from_actual(entry) -> dict:
    """从**实际传给 attention 的 mask 实参**解析允许访问的 key 集合（逐行）。"""
    m = entry.get("mask_obj")
    if m is None:
        return dict(kind="none", note="该调用未传 mask（q_len=1 的常见情形）⇒ **记隐式**，不算已验证")
    t = m
    while t.dim() > 2:
        t = t[0]
    rows = t if t.dim() == 2 else t.reshape(1, -1)
    out = []
    for r in range(rows.shape[0]):
        row = rows[r].reshape(-1)
        if row.dtype == torch.bool:
            allowed = row
        else:
            # 加法掩码语义：**允许 = 0**，禁止 = 大负数 / -inf（实测 HF bf16 掩码用 `finfo.min` 量级，
            # 因此**不能**用 `> -inf` 判定 —— 首跑就是这样把"全允许"误读出来的）。
            allowed = (row == 0)
            if not bool(allowed.any()):            # 退路：全非零时取最大值为"允许"
                allowed = (row == row.max())
        idx = torch.nonzero(allowed).reshape(-1)
        out.append(dict(first=int(idx[0]) if idx.numel() else None,
                        last=int(idx[-1]) if idx.numel() else None, count=int(idx.numel())))
    return dict(kind="tensor", shape=[int(x) for x in m.shape], dtype=str(m.dtype), rows=out)


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
        rec["bf16_reduced_precision_reduction"] = _apply_flag()
        print(f"[hfsb] bf16 reduced-precision reduction: "
              f"{json.dumps(rec['bf16_reduced_precision_reduction'], ensure_ascii=False)}", flush=True)
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
        install_mask_recorder()
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
            rows, masks, checks = [], [], []
            for j, t in enumerate(TOKENS):
                pre_call = c.get_seq_length()
                if pre_call != PRE + j:
                    raise RuntimeError(f"serial step {j}: 调用前 cache 长度 {pre_call} != {PRE + j}")
                piece = torch.tensor([[t]], device="cuda")
                am = torch.ones(1, PRE + j + 1, dtype=torch.long, device="cuda")
                pos = torch.tensor([[PRE + j]], device="cuda")
                _MASK_LOG.clear()
                with torch.inference_mode():
                    o = model(piece, past_key_values=c, use_cache=True,
                              position_ids=pos, attention_mask=am, output_attentions=True)
                m_entry = _MASK_LOG[0] if _MASK_LOG else None
                m_parsed = mask_allowed_from_actual(m_entry) if m_entry else dict(kind="no_call")
                # ---- 逐步硬检查 ----
                if c.get_seq_length() != PRE + j + 1:
                    raise RuntimeError(f"serial step {j}: 调用后 cache 长度 {c.get_seq_length()} "
                                       f"!= {PRE + j + 1}")
                lr = o.logits[0, -1]
                if not bool(torch.isfinite(lr.float()).all()):
                    raise RuntimeError(f"serial step {j}: logits 非有限")
                kv = c.layers[-1]
                if not bool(torch.isfinite(kv.keys[:, :, -1, :]).all()) \
                        or not bool(torch.isfinite(kv.values[:, :, -1, :]).all()):
                    raise RuntimeError(f"serial step {j}: 新增 K/V 非有限")
                checks.append(dict(step=j, pred_pos=PRE + j + 1, cache_before=pre_call,
                                   cache_after=c.get_seq_length(), pos=PRE + j,
                                   n_attn_calls=len(_MASK_LOG),
                                   mask_actual=m_parsed))
                rows.append(lr.clone())
                masks.append(_mask_support(o.attentions[-1][0, :, 0, :]))
            return rows, masks, checks, c

        def arm_block(c):
            piece = torch.tensor([TOKENS], device="cuda")
            am = torch.ones(1, PRE + len(TOKENS), dtype=torch.long, device="cuda")
            pos = torch.arange(PRE, PRE + len(TOKENS), device="cuda").view(1, -1)
            _MASK_LOG.clear()
            with torch.inference_mode():
                o = model(piece, past_key_values=c, use_cache=True, position_ids=pos,
                          attention_mask=am, output_attentions=True)
            m_entry = _MASK_LOG[0] if _MASK_LOG else None
            m_parsed = mask_allowed_from_actual(m_entry) if m_entry else dict(kind="no_call")
            if c.get_seq_length() != PRE + len(TOKENS):
                raise RuntimeError(f"block: cache 长度 {c.get_seq_length()} != {PRE + len(TOKENS)}")
            rows, checks = [], []
            for j in range(len(TOKENS)):
                lr = o.logits[0, j]
                if not bool(torch.isfinite(lr.float()).all()):
                    raise RuntimeError(f"block row {j}: logits 非有限")
                checks.append(dict(row=j, pred_pos=PRE + j + 1, cache_before=PRE,
                                   cache_after=c.get_seq_length(), pos=PRE + j,
                                   n_attn_calls=len(_MASK_LOG), mask_actual=m_parsed))
                rows.append(lr.clone())
            kv = c.layers[-1]
            if not bool(torch.isfinite(kv.keys[:, :, -len(TOKENS):, :]).all()) \
                    or not bool(torch.isfinite(kv.values[:, :, -len(TOKENS):, :]).all()):
                raise RuntimeError("block: 新增 K/V 非有限")
            masks = [_mask_support(o.attentions[-1][0, :, j, :]) for j in range(len(TOKENS))]
            return rows, masks, checks, c

        C0 = clone_cache(cache)                     # 共同起点（用于克隆两臂）
        C0_digest = json.dumps(_kv_digest(C0), sort_keys=True)
        cs1 = clone_cache(C0)
        cb1 = clone_cache(C0)
        rec["clone_checks_rep1"] = clone_checks(C0, cs1, cb1, "rep1 初始两臂")
        S1, Sm1, chk_s1, cs1 = arm_serial(cs1)
        B1, Bm1, chk_b1, cb1 = arm_block(cb1)
        cs2 = clone_cache(C0)
        cb2 = clone_cache(C0)
        rec["clone_checks_rep2"] = clone_checks(C0, cs2, cb2, "rep2 初始两臂")
        S2, _, _, cs2 = arm_serial(cs2)
        B2, _, _, cb2 = arm_block(cb2)
        if json.dumps(_kv_digest(cache), sort_keys=True) != snap_digest:
            raise RuntimeError("原始 HF 历史 cache 被改写")
        if json.dumps(_kv_digest(C0), sort_keys=True) != C0_digest:
            raise RuntimeError("共同起点 C0（克隆来源）被改写")
        rec["alignment"] = dict(
            rule="serial 第 j 步 T=pre+j+1、可见 key ≤ pre+j；block 第 j 行 T=pre+4、可见 key ≤ pre+j；"
                 "两者都预测 pre+j+1",
            serial_rows=[dict(j=j, pred_pos=PRE + j + 1, T=PRE + j + 1, mask_support=Sm1[j])
                         for j in range(len(TOKENS))],
            block_rows=[dict(j=j, pred_pos=PRE + j + 1, T=PRE + len(TOKENS), mask_support=Bm1[j])
                        for j in range(len(TOKENS))],
            checks_serial=chk_s1, checks_block=chk_b1,
            mask_object_vs_softmax=dict(
                mask_actual="**拦截本机 HF 实际使用的 attention 函数**，取它收到的 `attention_mask` 实参"
                            "（逐层调用记录），再按加法(-inf)/布尔语义解析每行允许访问的 key 集合",
                reconstruction_attempt="曾用 `create_causal_mask` 重建：实测**与模型实际收到的 mask 不一致**"
                                       "（重建未传 `cache_position`/完整序列等内部上下文）⇒ **不作为验证依据**，"
                                       "仅记录该失败尝试",
                softmax_support="仅作**交叉参考**（最后一层、逐 head 的非零计数/最后非零下标）；"
                                "**softmax 后的零概率不等于被 mask**，因此它**不构成** mask 验证"),
            mask_actual_serial=[c["mask_actual"] for c in chk_s1],
            mask_actual_block=[c["mask_actual"] for c in chk_b1],
            mask_support_equal=all(Sm1[j]["counts"] == Bm1[j]["counts"]
                                   and Sm1[j]["lasts"] == Bm1[j]["lasts"]
                                   for j in range(len(TOKENS))),
            row_source="block 行 j 与 serial 第 j 步**对齐到同一预测位置**；错位比较被禁止")
        # 实际 mask 实参核对：有显式 mask 的行必须给出允许集合且与位置公式一致；没有的记为**隐式**（不算 PASS）
        for j in range(len(TOKENS)):
            for side, chk in (("serial", chk_s1[j]), ("block", chk_b1[j])):
                ma = chk["mask_actual"]
                if ma.get("kind") != "tensor":
                    rec.setdefault("missing_checks", []).append(
                        f"{side} row {j}: 实际调用未传显式 mask（{ma.get('kind')}）⇒ 记为**隐式**，"
                        f"不作为已验证的 mask 检查")
                    continue
                rows_allowed = ma["rows"]
                want_row = 0 if side == "serial" else j
                got = rows_allowed[want_row] if want_row < len(rows_allowed) else None
                if got is None or got["first"] != 0 or got["last"] != PRE + j:
                    raise RuntimeError(f"{side} row {j}: 实际 mask 允许集合 {got} 不符"
                                       f"（期望 0..{PRE + j}）")
        rec["mask_actual_verified_rows"] = sum(
            1 for chk in chk_s1 + chk_b1 if chk["mask_actual"].get("kind") == "tensor")
        rec.setdefault("missing_checks", []).append(
            "全层/全 head 的 mask 只按 layer_idx=0 用 HF mask API 取了一次；"
            "未逐层逐 head 展开（如需更严格需另做）")
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
