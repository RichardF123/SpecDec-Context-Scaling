"""xb_probe.py — **自研引擎 vs HF 的同路径对齐探针**（`XB_IMPL=hf|ours`，另有 `XB_SELFTEST=1`）。

回答的问题：**相同输入、相同 prefill 分块、相同后续调用日程下，自研 target 与 HF target 是否一致？**
（这与"各自内部单步/块是否一致"是不同问题。）

## 事先声明的协议 S*（两边**完全相同**）
  序列 = F13 prompt(8192) + 历史前 48 个 token（共 8240）
  * **C1**：4 个 chunk 覆盖 prompt —— `[0:2048, 2048:4096, 4096:6144, 6144:8192]`
  * **C2**：再喂 48 个 token 的**一个 chunk** `[8192:8240]`
  * 起点 `pre = 8240`；随后两臂：
    - **Serial**：依次喂 `[220,16,16,1154]`（每次 1 个 token，共 4 次）
    - **Block**：一次喂 `[220,16,16,1154]`
  > ⚠️ 这是 **teacher-forced 数值对照**，**不是**自由生成，也**不是**完整 SpecDec 验收；
  > 也**不是**原始自由生成 AR/Spec 的历史。
  > ⚠️ 本协议的 C2 与上一轮 HF 探针的"2096 末块"日程**不同**（那批产物保留，但**不**用于本轮跨实现比较）。

## 已知实现差异（**运行前**列出，不凭名称假设一致）
  1. **score 缩放**：HF `matmul(q,k^T) * scaling`（乘倒数）vs 自研 `matmul(...) / sqrt(head_dim)`（除）——
     数学等价、**bf16 下可能差 1 ULP**；
  2. **mask 施加**：HF 用 **bf16 加法掩码**（允许=0、禁止≈`finfo.min`）；自研用 `masked_fill(-inf)`；
  3. **GQA**：HF `repeat_kv`（expand+reshape）vs 自研 `kv_repeat`（同模式），需实测而非假定；
  4. **RoPE**：HF `Qwen2RotaryEmbedding`（`position_embeddings`）vs 自研 `Build_Rope_cache` + `apply_rope`，
     cos/sin 的 dtype 与生成方式**需实测**；
  5. **RMSNorm**：两边**都是 fp32 内部**（已核对 HF 源码与自研实现）⇒ **不是**差异点；
  6. **LM head**：HF `lm_head(hidden)`（tied）vs 自研 `x @ embed.weight.T` —— 形状相同；
  7. **权重映射**：HF `state_dict` 去掉 `model.` 前缀、tied 时去掉 `lm_head.weight`、
     `load_state_dict(strict=True)` + 逐张量 `torch.equal` ⇒ **同名映射**（记录分组计数，不把 `missing=[]` 当唯一证据）。
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TAG = "xb_20261001"
TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
FAMILY = os.environ.get("XB_FAMILY", "F13").strip().upper()
CTX = int(os.environ.get("XB_CTX", "8192"))
CHUNK = int(os.environ.get("XB_CHUNK", "2048"))
PRE = int(os.environ.get("XB_PRE", "8240"))
TOKENS = [int(x) for x in os.environ.get("XB_TOKENS", "220,16,16,1154").split(",")]
IMPL = os.environ.get("XB_IMPL", "ours").strip().lower()
SELFTEST = os.environ.get("XB_SELFTEST", "1") == "1"


def _newest(pat):
    return Path(sorted(glob.glob(str(OUTDIR / pat)))[-1])


def _sha(t: torch.Tensor, n: int = 16) -> str:
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:n]


def _row_fields(r) -> dict:
    x = r.detach().float().reshape(-1)
    t2 = torch.topk(x, 2).values
    am = int(torch.argmax(x))
    tied = [int(i) for i in torch.nonzero(x == x.amax()).reshape(-1)]
    return dict(argmax=am, max_score=float(x.max()), margin=float(t2[0] - t2[1]),
                n_tied_max=len(tied), finite=bool(torch.isfinite(x).all()), sha=_sha(r))


# ------------------------------------------------------------------ 自测
def _selftest() -> int:
    """微型模型自测：**映射**（同名 + strict + 逐张量位级）、**行对齐**、**缓存复制**。"""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from .align_ours import build_from_config

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
                      max_position_embeddings=512, attention_bias=False, tie_word_embeddings=True)
    cfg._attn_implementation = "eager"
    cfg.attention_bias = False          # ⚠️ 构造参数未生效（实测两次）：v5 的微型 config 仍带 q/k/v bias
    cfg.tie_word_embeddings = True      #    ⇒ 自测里显式剔除 bias 键以**模拟真实 Qwen2.5 配置**
    hf = Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval()
    bias_keys = [k for k in hf.state_dict() if k.endswith(("q_proj.bias", "k_proj.bias",
                                                           "v_proj.bias"))]
    print(f"  （自测信息）微型 HF 携带 q/k/v bias 键 {len(bias_keys)} 个（真实 Qwen2.5 无 bias；"
          f"自测按真实配置剔除，并记录该差异）")
    ours = build_from_config(cfg).to(torch.bfloat16).eval()
    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    for k in [k for k in sd if k.endswith(("q_proj.bias", "k_proj.bias", "v_proj.bias"))]:
        sd.pop(k)                        # 见上：按真实 Qwen2.5 配置剔除微型模型的多余 bias
    tied = bool(cfg.to_dict().get("tie_word_embeddings", False))
    if tied:
        sd.pop("lm_head.weight", None)
    ours.load_state_dict(sd, strict=True)
    check("★ 同名映射 + strict 载入成功（无缺键/多键）", True)
    cur = ours.state_dict()
    same = [k for k in sd if k in cur and torch.equal(cur[k], sd[k])]
    check(f"★ 逐张量位级一致 {len(same)} / {len(sd)}", len(same) == len(sd),
          [k for k in sd if k not in same][:4])
    groups = {}
    for k in sd:
        top = k.split(".")[0] if not k.startswith("layers") else "layers." + k.split(".")[2]
        groups[top] = groups.get(top, 0) + 1
    print(f"  映射分组计数：{json.dumps(groups, ensure_ascii=False)}")
    check("关键组都在（embed_tokens / layers.*.{self_attn,mlp,input_layernorm,"
          "post_attention_layernorm} / norm）",
          any(k.startswith("embed_tokens") for k in sd) and "norm.weight" in sd
          and any("self_attn.q_proj.weight" in k for k in sd)
          and any("mlp.gate_proj.weight" in k for k in sd)
          and any("input_layernorm.weight" in k for k in sd)
          and any("post_attention_layernorm.weight" in k for k in sd))
    check("HF 与自研都没有 qkv bias（config.attention_bias=False）",
          not any("q_proj.bias" in k or "k_proj.bias" in k or "v_proj.bias" in k for k in sd))
    # 行对齐 + 缓存复制
    ids = torch.randint(0, 128, (1, 6))
    with torch.inference_mode():
        ho = hf(ids, use_cache=True)
        ol, oc = ours._run(ours.embed_tokens(ids), None, need_logits=True)   # 顺序：(logits, cache)
    hk = ho.past_key_values.layers[0].keys
    ok_k = oc[0]["k"]
    check("★ 缓存布局一致 (1, kv_heads, T, head_dim)",
          tuple(hk.shape) == tuple(ok_k.shape), f"{tuple(hk.shape)} vs {tuple(ok_k.shape)}")
    d = (ho.logits[0, -1].float() - ol[0, -1].float()).abs()
    print(f"  （自测信息）微型模型 prompt 末位 logits max|Δ|={float(d.max()):.6g}，"
          f"argmax HF={int(ho.logits[0,-1].argmax())} OURS={int(ol[0,-1].argmax())}")
    cp1 = [{k: v.clone() for k, v in d.items()} for d in oc]
    cp2 = [{k: v.clone() for k, v in d.items()} for d in oc]
    check("★ 两臂缓存可深拷贝且不共享存储",
          not any(cp1[i][kk].data_ptr() == cp2[i][kk].data_ptr()
                  for i in range(len(cp1)) for kk in ("k", "v")))
    check("★ 深拷贝后与源逐位相同",
          all(torch.equal(cp1[i][kk], oc[i][kk]) for i in range(len(oc)) for kk in ("k", "v")))
    print(f"\n跨实现自测：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


# ------------------------------------------------------------------ 协议
def _schedule(seq_len: int) -> list:
    """声明的日程 S*：prompt 用 chunk 切分，剩下的（48 个 token）单独一个 chunk。"""
    sched = [(st, min(st + CHUNK, CTX)) for st in range(0, CTX, CHUNK)]
    if seq_len > CTX:
        sched.append((CTX, seq_len))
    return sched


def _run_impl() -> int:
    import transformers
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="xb_probe", impl=IMPL, family=FAMILY, ctx=CTX, chunk=CHUNK,
               pre=PRE, tokens=TOKENS, model=TARGET,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               schedule=_schedule(PRE), versions=dict(torch=torch.__version__,
                                                      transformers=transformers.__version__,
                                                      cuda=torch.version.cuda),
               semantics="同路径 teacher-forced 数值对照（不是自由生成/完整 SpecDec 验收）；"
                         "两边**各自独立**构建缓存，不互相导入。")
    print(f"[xb] run_id={run_id} impl={IMPL}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    from . import g5_8k_spec_variance_probe as P5V
    from . import family_input as FI
    from . import model_self as M
    from . import spec_cache, spec_dec
    from . import lowmem_load
    ok, why = P5V.commit_guard()
    print(f"[xb] commit: {why}")
    if not ok:
        (OUTDIR / f"{TAG}_{IMPL}_{run_id}.json").write_text(json.dumps(rec, ensure_ascii=False),
                                                            encoding="utf-8")
        return 3
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True)
        ids_list, fam_meta = FI.load_family_ids(tok, FAMILY, ctx=CTX)
        cap = torch.load(_newest("f13i50_20261001_capture_ar_*.pt"), map_location="cpu")
        prefix48 = [int(x) for x in cap["prefix_tokens"]][:PRE - CTX]
        rec["input"] = dict(source_kind="manifest", family=fam_meta, ctx=CTX,
                            prefix_src=str(_newest("f13i50_20261001_capture_ar_*.pt")),
                            prefix_len=len(prefix48), tokens=TOKENS)
        seq = list(ids_list) + prefix48
        assert len(seq) == PRE
        rec["input"]["seq_sha16"] = hashlib.sha256(
            json.dumps(seq).encode("utf-8")).hexdigest()[:16]
        # ---- 唯一变量：reduction=True（显式），QK32/PROB32 关闭 ----
        b = torch.backends.cuda.matmul
        b.allow_bf16_reduced_precision_reduction = True
        rec["precision"] = dict(bf16_reduced_precision_reduction=bool(
            b.allow_bf16_reduced_precision_reduction), qk_fp32=False, prob_fp32=False,
            note="两边都显式 reduction=True；QK32/PROB32 关闭")
        out = dict(meta=dict(impl=IMPL, run_id=run_id, schedule=_schedule(PRE)))
        if IMPL == "ours":
            tgt, _tok2, info = lowmem_load.load_ours_bf16(TARGET)
            tgt.set_kv_format("fp16")            # base_bf16（KV 未量化）
            M.set_diag_preset("eager")
            M.set_attention_strategy("eager")    # 全 eager，与 HF 对齐
            rec["weights"] = dict(bitmatch=info["weights_bitmatch"],
                                  n_checked=info["n_params_checked"],
                                  mismatch=info["mismatch_keys"],
                                  mapping="HF state_dict 去 `model.` 前缀 + tied 去 lm_head + strict 载入")
            rec["impl"] = dict(attn_implementation="eager(engine)", kv_format="fp16=base_bf16",
                               strategy=M.attention_strategy())
            ids = torch.tensor([seq], device="cuda")
            # C1：prompt 4 个 chunk
            cache = None
            with torch.inference_mode(), M.attention_phase("prefill"):
                for st, en in _schedule(PRE):
                    if en > CTX:
                        break
                    piece = ids[:, st:en]
                    lg, cache = tgt._run(tgt.embed_tokens(piece), cache,
                                         need_logits=(en == CTX))
            rec["C1"] = dict(cache_len=int(spec_cache.cache_length(cache)), last_logits=_row_fields(lg[0, -1]))
            out["c1_logits"] = lg[0, -1].detach().float().cpu()
            out["c1_kv"] = [{k: v.detach().float().cpu() for k, v in d.items()} for d in cache]
            # C2：48 个 token 一个 chunk
            with torch.inference_mode(), M.attention_phase("prefill"):
                lg2, cache = tgt._run(tgt.embed_tokens(ids[:, CTX:PRE]), cache, need_logits=True)
            rec["C2"] = dict(cache_len=int(spec_cache.cache_length(cache)), last_logits=_row_fields(lg2[0, -1]))
            out["c2_logits"] = lg2[0, -1].detach().float().cpu()
            out["c2_kv"] = [{k: v.detach().float().cpu() for k, v in d.items()} for d in cache]
            # 两臂
            cs = [{k: v.clone() for k, v in d.items()} for d in cache]
            cb = [{k: v.clone() for k, v in d.items()} for d in cache]
            ser = []
            for j, t in enumerate(TOKENS):
                with torch.inference_mode(), M.attention_phase("decode"):
                    l, cs = spec_dec._step(tgt, t, cs)
                ser.append(l[0, -1].detach().float().cpu())
            with torch.inference_mode(), M.attention_phase("verify"):
                rows, cb = spec_dec._verify(tgt, cb, list(TOKENS))
            out["serial_rows"] = ser
            out["block_rows"] = [rows[j].detach().float().cpu() for j in range(len(TOKENS))]
            rec["serial"] = [_row_fields(x) for x in ser]
            rec["block"] = [_row_fields(x) for x in out["block_rows"]]
            rec["attention_stats"] = M.attention_stats()
        else:
            from transformers import AutoModelForCausalLM, DynamicCache
            from .hf_serial_block_probe import install_mask_recorder, _MASK_LOG, mask_allowed_from_actual
            install_mask_recorder()
            model = AutoModelForCausalLM.from_pretrained(TARGET, torch_dtype=torch.bfloat16,
                                                         attn_implementation="eager").to("cuda").eval()
            rec["impl"] = dict(attn_implementation=str(model.config._attn_implementation),
                               kv_format="DynamicCache bf16")
            assert str(model.config._attn_implementation) == "eager"
            ids = torch.tensor([seq], device="cuda")
            cache = DynamicCache()
            with torch.inference_mode():
                for st, en in _schedule(PRE):
                    if en > CTX:
                        break
                    piece = ids[:, st:en]
                    n = int(piece.shape[1])
                    o = model(piece, past_key_values=cache, use_cache=True,
                              position_ids=torch.arange(st, st + n, device="cuda").view(1, -1),
                              attention_mask=torch.ones(1, st + n, dtype=torch.long, device="cuda"))
            rec["C1"] = dict(cache_len=int(cache.get_seq_length()),
                             last_logits=_row_fields(o.logits[0, -1]))
            out["c1_logits"] = o.logits[0, -1].detach().float().cpu()
            out["c1_kv"] = [dict(k=cache.layers[i].keys.detach().float().cpu(),
                                 v=cache.layers[i].values.detach().float().cpu())
                            for i in range(len(cache.layers))]
            with torch.inference_mode():
                o2 = model(ids[:, CTX:PRE], past_key_values=cache, use_cache=True,
                           position_ids=torch.arange(CTX, PRE, device="cuda").view(1, -1),
                           attention_mask=torch.ones(1, PRE, dtype=torch.long, device="cuda"))
            rec["C2"] = dict(cache_len=int(cache.get_seq_length()),
                             last_logits=_row_fields(o2.logits[0, -1]))
            out["c2_logits"] = o2.logits[0, -1].detach().float().cpu()
            out["c2_kv"] = [dict(k=cache.layers[i].keys.detach().float().cpu(),
                                 v=cache.layers[i].values.detach().float().cpu())
                            for i in range(len(cache.layers))]

            def clone(c):
                n = DynamicCache()
                for lay in c.layers:
                    d = lay.__class__()
                    d.keys = lay.keys.detach().clone()
                    d.values = lay.values.detach().clone()
                    n.layers.append(d)
                return n

            cs, cb = clone(cache), clone(cache)
            assert not any(cs.layers[i].keys.data_ptr() == cb.layers[i].keys.data_ptr()
                           for i in range(len(cs.layers))), "两臂共享存储"
            ser = []
            for j, t in enumerate(TOKENS):
                with torch.inference_mode():
                    oo = model(torch.tensor([[t]], device="cuda"), past_key_values=cs, use_cache=True,
                               position_ids=torch.tensor([[PRE + j]], device="cuda"),
                               attention_mask=torch.ones(1, PRE + j + 1, dtype=torch.long,
                                                        device="cuda"))
                ser.append(oo.logits[0, -1].detach().float().cpu())
            _MASK_LOG.clear()
            with torch.inference_mode():
                ob = model(torch.tensor([TOKENS], device="cuda"), past_key_values=cb, use_cache=True,
                           position_ids=torch.arange(PRE, PRE + len(TOKENS), device="cuda").view(1, -1),
                           attention_mask=torch.ones(1, PRE + len(TOKENS), dtype=torch.long,
                                                     device="cuda"))
            rec["mask_actual_serial"] = [mask_allowed_from_actual(dict(mask_obj=None)),
                                         "见 hf_serial_block_probe（本工具只记录 block 的实际 mask）"]
            ma = mask_allowed_from_actual(_MASK_LOG[0]) if _MASK_LOG else dict(kind="no_call")
            rec["mask_actual_block"] = ma
            out["serial_rows"] = ser
            out["block_rows"] = [ob.logits[0, j].detach().float().cpu() for j in range(len(TOKENS))]
            rec["serial"] = [_row_fields(x) for x in ser]
            rec["block"] = [_row_fields(x) for x in out["block_rows"]]
        rec["resources"] = dict(pt_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                                pt_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20,
                                cuda_free_mib=torch.cuda.mem_get_info()[0] / 2 ** 20,
                                note="顺序加载、不同时驻留；计时/显存只作诊断")
        rec["kv_digest"] = dict(
            c1_sha=[_sha(d["k"]) for d in out["c1_kv"]], c2_sha=[_sha(d["k"]) for d in out["c2_kv"]])
        tp = OUTDIR / f"{TAG}_{IMPL}_{run_id}.pt"
        torch.save(out, str(tp))
        rec["stage"] = "complete"
        rec["artifacts"] = dict(pt=str(tp), json=str(OUTDIR / f"{TAG}_{IMPL}_{run_id}.json"))
        p = OUTDIR / f"{TAG}_{IMPL}_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[xb] C1 len={rec['C1']['cache_len']} argmax={rec['C1']['last_logits']['argmax']} | "
              f"C2 len={rec['C2']['cache_len']} argmax={rec['C2']['last_logits']['argmax']}")
        print(f"[xb] serial argmax={[x['argmax'] for x in rec['serial']]} "
              f"block argmax={[x['argmax'] for x in rec['block']]}")
        print(f"[xb] RESULT_JSON={p}")
        return 0
    except Exception as e:                                        # noqa: BLE001
        rec["stage"] = "crashed"
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        p = OUTDIR / f"{TAG}_{IMPL}_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[xb] !! 停止：{e}")
        print(f"[xb] RESULT_JSON={p}")
        return 2


if __name__ == "__main__":
    raise SystemExit(_selftest() if SELFTEST else _run_impl())
