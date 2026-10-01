"""unified_geometry_run.py — UCG 原型运行（`UG_MODE=ar|spec`, `UG_FAMILY=F13|F06`）。

**独立诊断入口**：只包 `spec_dec._verify` / `spec_dec._step`，**不改**生产默认 backend/精度/tie-breaking。
目标语义 = **B（AR 与 Spec 同时改用统一几何后彼此一致）**；**不**宣称复现旧 AR 的逐 token。

固定：BF16 / 未量化 KV / 原模型对 / chunk=2048 / k=4 / max_new=64 / 原 EOS 规则 / 全 eager。
AR 每步 `[t, pad, pad, pad]` 取 row 0 → trim 到 pre+1（**不做 teacher forcing、不复用旧候选/日程**）。
"""
from __future__ import annotations

import glob
import json
import os
import time
import traceback
from pathlib import Path

import torch

from .. import results_schema as S
from . import model_self as M
from . import sb_align as A
from . import spec_cache
from . import spec_dec
from . import unified_geometry as U
from . import family_input as FI
from . import g5_8k_spec_variance_probe as P5V
from .smoke_checks import CheckRegistry
from .smoke_hooks import DecisionHooks

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TAG = "ucg_20261001"
CTX = int(os.environ.get("UG_CTX", "8192"))
CHUNK = int(os.environ.get("UG_CHUNK", "2048"))
K = int(os.environ.get("UG_K", "4"))
MAXNEW = int(os.environ.get("UG_MAXNEW", "64"))
KV, ARM = "base_bf16", "eager"
MODE = os.environ.get("UG_MODE", "ar").strip().lower()
FAMILY = os.environ.get("UG_FAMILY", "F13").strip().upper()


def _newest(pat):
    return Path(sorted(glob.glob(str(OUTDIR / pat)))[-1])


def _cache_len(c):
    return int(spec_cache.cache_length(c))


def _old_seq(family: str, mode: str):
    """旧（s=1/s=Kp 混合几何）的生成序列，用于**单独记录**"新几何 vs 旧几何"的变化。"""
    for p in sorted(glob.glob(str(OUTDIR / "smoke8k_eager_*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        fam = ((r.get("input") or {}).get("family") or {}).get("family_id")
        if fam == family and (r.get("config") or {}).get("mode") == mode:
            last = r
    try:
        return (last.get("run") or {}).get("gen") or []
    except NameError:
        return []


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="unified_geometry_run", mode=MODE, family=FAMILY, arm=ARM, kv=KV,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               config=dict(ctx=CTX, chunk=CHUNK, k=K, max_new=MAXNEW, kv=KV, arm=ARM,
                           diag_precision="qk_fp32=False, prob_fp32=False",
                           block=U.KB, pad_id=U.PAD_ID),
               target_semantics="**B**：AR 与 Spec 同时改用统一几何后彼此一致；"
                                "**不**宣称复现旧 AR 的逐 token（旧 AR 与新 AR 的差异单独记录）",
               scope="独立诊断入口；只包 `_verify`/`_step`；不改生产默认 backend/精度/门槛；"
                     "不做 speedup 声明（AR 侧 target 计算量约 4×，属正确性控制）。")
    print(f"[ucg] run_id={run_id} mode={MODE} family={FAMILY}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    ok, why = P5V.commit_guard()
    print(f"[ucg] commit: {why}")
    if not ok:
        rec["stage"] = "refused"
        (OUTDIR / f"{TAG}_{FAMILY}_{MODE}_{run_id}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return 3
    hooks = None
    ov, os_ = spec_dec._verify, spec_dec._step
    patched = False
    try:
        rec["diag_precision"] = M.set_diag_preset("eager")
        from .lowmem_load import load_ours_bf16
        tgt, tok, ti = load_ours_bf16(P5V.TARGET)
        drf = None
        if MODE == "spec":
            drf, _t, _i = load_ours_bf16(P5V.DRAFT)
            drf.set_kv_format(spec_dec._fmt(KV))
        tgt.set_kv_format(spec_dec._fmt(KV))
        M.set_attention_strategy(ARM)
        M.reset_attention_stats()
        rec["weights"] = dict(bitmatch=ti.get("weights_bitmatch"), checked=ti.get("n_params_checked"))
        ids_list, fam_meta = FI.load_family_ids(tok, FAMILY, ctx=CTX)
        ids = torch.tensor([ids_list], device="cuda")
        A.require(int(ids.shape[1]) == CTX, "prompt 长度必须 8192", str(int(ids.shape[1])))
        rec["input"] = dict(family=fam_meta, ids_hash=P5V._gen_hash([int(x) for x in ids[0].tolist()]),
                            eos_id=int(tok.eos_token_id))
        inner = P5V.timed_chunked_prefill({}, CHUNK, tgt)
        reg = CheckRegistry()
        plans = []
        hooks = DecisionHooks(tgt=tgt, drf=drf, registry=reg, prefill_impl=inner,
                              on_verify_post=lambda rows: None)
        hooks.install()
        # ⚠️ 顺序关键：hooks 安装后模块属性已被替换 ⇒ 一律用**安装前**捕获的原始 `_verify` 作为底层调用，
        #    并由 U.*_padded 自己显式标注相位（否则会自我递归，或把 repair 记成 verify；两者都真踩过）。
        orig_verify = ov

        def verify_patched(model, cache, cands, *a, **kk):
            cands = list(cands)
            plans.append(U.verify_plan(_cache_len(cache), cands))
            return U.verify_padded(model, cache, cands, call=orig_verify)

        def step_patched(model, token, cache, *a, **kk):
            plans.append(U.repair_plan(_cache_len(cache), int(token)))
            return U.step_padded(model, cache, int(token), call=orig_verify, phase="repair")

        if MODE == "spec":
            spec_dec._verify, spec_dec._step = verify_patched, step_patched
            patched = True

        torch.cuda.reset_peak_memory_stats()
        if MODE == "ar":
            with torch.inference_mode(), M.attention_phase("prefill"):
                lg, cache = inner(tgt, ids)
            A.require(_cache_len(cache) == CTX, "prefill 后 cache 必须 = ctx", str(_cache_len(cache)))
            rec["initial_state"] = dict(cache_len=CTX,
                                        digest=A.require_full_cache(cache, expect_len=CTX,
                                                                    n_layers=len(cache),
                                                                    what="AR prefill 后"),
                                        next_logits=dict(argmax=int(lg.reshape(-1, lg.shape[-1])[-1].argmax()),
                                                         sha=A.stable_sha(lg)))
            gen, steps = [], []
            nxt = int(lg.reshape(-1, lg.shape[-1])[-1].argmax())
            for i in range(MAXNEW):
                gen.append(nxt)
                if nxt == int(tok.eos_token_id):
                    stop = "eos"
                    break
                if i == MAXNEW - 1:
                    stop = "max_new_reached"
                    break
                pre = _cache_len(cache)
                lg2, cache = U.step_padded(tgt, cache, nxt, call=orig_verify, phase="decode")
                plans.append(U.ar_step_plan(pre, nxt))
                A.require(_cache_len(cache) == pre + 1, "AR 步后 cache 必须 = pre+1",
                          f"{_cache_len(cache)} vs {pre + 1}")
                A.require(bool(torch.isfinite(lg2.float()).all()), "AR 步 logits 非有限")
                steps.append(dict(step=i, fed=nxt, pre=pre, cache_len_after=_cache_len(cache),
                                  next_argmax=int(lg2.reshape(-1).argmax())))
                nxt = int(lg2.reshape(-1).argmax())
            else:
                stop = "loop_end"
            out = dict(gen=gen, stats=dict(mode="AR-UCG", steps=len(steps)))
            rec["steps_tail"] = steps[-3:]
            rec["n_target_block_calls"] = len(steps)
        else:
            out = spec_dec.spec_decode_greedy(tgt, drf, ids, MAXNEW, k=K,
                                              eos_id=int(tok.eos_token_id),
                                              target_kv=KV, draft_kv="base_bf16")
            gen = out["gen"]
            stop = ("eos" if gen and gen[-1] == int(tok.eos_token_id)
                    else ("max_new_reached" if len(gen) >= MAXNEW else "loop_end"))
            rec["rounds"] = out["stats"]

        A.require(gen, "生成序列为空")
        gen = [int(x) for x in gen]
        geo = U.require_geometry_uniform(plans) if plans else dict(n_plans=0)
        rec["geometry"] = geo
        rec["result"] = dict(gen=gen, gen_len=len(gen), stop_reason=stop,
                             eos_id=int(tok.eos_token_id), max_new=MAXNEW,
                             gen_sha=A.stable_sha(torch.tensor(gen)))
        old = _old_seq(FAMILY, MODE)
        if old:
            n = min(len(old), len(gen))
            first = next((i for i in range(n) if int(old[i]) != gen[i]), None)
            rec["new_vs_old_geometry"] = dict(
                note="**单独记录**：新几何(UCG) vs 旧几何（生产）在同一输入/同一 family 下的序列变化",
                old_len=len(old), new_len=len(gen), identical=bool(list(old[:n]) == gen[:n]),
                first_divergence_index=first,
                old_token=(int(old[first]) if first is not None else None),
                new_token=(gen[first] if first is not None else None))
        att = M.attention_stats()
        A.require(not any(v > 0 for v in att["fused_by_phase"].values()),
                  "本配置要求零融合调用", str(att["fused_by_phase"]))
        rec["attention_stats"] = dict(strategy=att["strategy"], fused_by_phase=att["fused_by_phase"],
                                      eager_by_phase=att["eager_by_phase"],
                                      s_max_by_phase=att["s_max_by_phase"])
        rec["resources"] = dict(pt_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                                pt_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20,
                                cuda_free_mib=torch.cuda.mem_get_info()[0] / 2 ** 20,
                                note="含诊断插桩；**不**作 speedup 或正式性能结论")
        rec["stage"] = "complete"
        p = OUTDIR / f"{TAG}_{FAMILY}_{MODE}_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        rec["artifacts"] = dict(json=str(p))
        print(f"[ucg] 完成：gen_len={len(gen)} stop={stop} 计划数={len(plans)} "
              f"几何一致={geo.get('s')}x{geo.get('T_minus_pre')} fused={att['fused_by_phase']}")
        if "new_vs_old_geometry" in rec:
            nv = rec["new_vs_old_geometry"]
            print(f"[ucg] 新几何 vs 旧几何：identical={nv['identical']} "
                  f"首次差异 index={nv['first_divergence_index']}")
        print(f"[ucg] gen={gen}")
        print(f"[ucg] RESULT_JSON={p}")
        return 0
    except Exception as e:                                        # noqa: BLE001
        if patched:
            spec_dec._verify, spec_dec._step = ov, os_
        if hooks is not None:
            try:
                hooks.remove()
            except Exception:                                     # noqa: BLE001
                pass
        rec["stage"] = ("checks_failed" if isinstance(e, (U.URError, A.AlignError)) else "crashed")
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        M.set_diag_precision(qk32=False, prob32=False)
        p = OUTDIR / f"{TAG}_{FAMILY}_{MODE}_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[ucg] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[ucg] RESULT_JSON={p}")
        return 5 if rec["stage"] == "checks_failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
