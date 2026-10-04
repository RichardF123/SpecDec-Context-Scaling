"""correctness_protocol_margins.py — 决策位置的 **top-1/top-2/margin** 提取 + profiler 可用性检查。

## 为什么需要
协议要求每次运行记录"决策位置 top-1/top-2 与 margin"（用于判断分歧是否发生在**近并列**处）。
harness 的产物只存生成 token IDs 与统计量，不存决策行 ⇒ 这里**按需重建**：
对每个 family，用 **AR 的口径**（分块 prefill + 逐 token `M=1` 单步）走到
**首个分歧位置**，记录该位置的 top-1/top-2 与 margin，以及 AR/production 在该位置各自选的 token。

## 位置从哪来（反例 CF3 的修复）
位置**只能**来自"同 tag、同 family 的 `AR` 与 `production` 产物首个不同 token 的输出内下标"，
由纯 CPU 模块 `gb6_diag_select.py` 选定并核对（输入 hash / ctx / prompt_len / KV 格式 / 计算配置）。
旧实现在找不到 divergent 侧（模式名写成 `aligned`/`spec`，而 Stage 2b 用 `prod`/`serial`）时
把位置**退化为末 token**——真实 F13 实测 `pos=63` 而真值 `47`。现在：
  * 找不到 divergent 侧 ⇒ 抛 `MISSING_ARTIFACT` 并**拒绝**出记录；
  * tag/family/配置不符 ⇒ 抛 `PROVENANCE_MISMATCH` / `CONFIG_MISMATCH` 并**拒绝**；
  * AR 与 production **逐 token 相同**（无分歧）⇒ `TOKEN_EQUAL`，**不记录位置**
    （GB6 此时不需要诊断）；只有显式 `--allow-last-index-when-equal` 才走 legacy 末位，
    且 `position_origin` 会标成 `legacy_last_index`，消费端（GB6）会**拒绝**该记录。

## 模式
`--ar-mode` / `--div-mode`（默认沿用历史模式名 `ar`/`spec`）。Stage 2b 用
**`--stage2b`**（等价于 `--ar-mode ar --div-mode prod`）。默认值刻意保持历史模式名：
这样在 Stage 2b 上**不会**悄悄取到一份"看起来能用"的产物，而是明确报缺。

## profiler 可用性
只**检查**本机是否已有 `nsys`/`ncu` 可执行文件（**不安装**）；并在进程内试一次 `torch.profiler`
CUDA activity 是否真的产出 kernel 事件。任一不可用 ⇒ 记 `PROFILER_UNAVAILABLE`，**不重试**、不阻塞。

**纪律**：不加载 draft（AR 口径只需 target）；不改 cache/mask/RoPE/权重；不做 FP32；
**诊断与正式计时分离**——本工具产出**不进入**任何性能/加速比结论；结果先落盘再摘要。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import torch

from .. import results_schema as S
from . import cache_clone as CC
from . import diag_checks as DC
from . import f13_i50_cells as C
from . import gb6_diag_select as GDS
from . import lmhead_boundary as LB
from . import model_self as M
from . import sb_align as A
from . import spec_dec
from . import spec_cache
from . import g5_8k_spec_variance_probe as P5V

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"

TAG = os.environ.get("PM_TAG", "div8209_proto")
FAMILIES = [x for x in os.environ.get("PM_FAMILIES", "F01,F02,F08,F10").split(",") if x]
PROTO_TAG = os.environ.get("PM_RUNS_TAG", "div8209_proto")
CTX = int(os.environ.get("PM_CTX", "8192"))
CHUNK = int(os.environ.get("PM_CHUNK", "2048"))
MAXNEW = int(os.environ.get("PM_MAXNEW", "64"))
TARGET_ROPE = os.environ.get("PM_TARGET_ROPE", "cpu_fp32").strip().lower()
# Stage 2b：模式名参数化（默认沿用历史模式名；见模块 docstring）
AR_MODE = os.environ.get("PM_AR_MODE", "ar")
DIV_MODE = os.environ.get("PM_DIV_MODE", "spec")
KV = "base_bf16"


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def _profiler_status() -> dict:
    """只**检查**，不安装：nsys/ncu 是否存在 + torch.profiler 是否真能给出 CUDA kernel 事件。"""
    nsys = shutil.which("nsys")
    ncu = shutil.which("ncu")
    got_kernel = False
    detail = None
    try:
        a = torch.randn(1, 12, 1, 128, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(1, 12, 8209, 128, dtype=torch.bfloat16, device="cuda")
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(2):
                torch.matmul(a, b.transpose(-2, -1))
        torch.cuda.synchronize()
        got_kernel = any(("CUDA" in str(getattr(e, "device_type", "")))
                         for e in prof.events())
        detail = dict(n_events=len(list(prof.events())),
                      n_cuda_events=sum(1 for e in prof.events()
                                        if "CUDA" in str(getattr(e, "device_type", ""))))
    except Exception as e:                                                # noqa: BLE001
        detail = dict(error=f"{type(e).__name__}: {str(e)[:200]}")
    return dict(nsys=nsys, ncu=ncu, torch_profiler_cuda_kernel_events=bool(got_kernel),
                detail=detail,
                status=("PROFILER_AVAILABLE" if (nsys or ncu or got_kernel) else "PROFILER_UNAVAILABLE"))


def main() -> int:
    ap = argparse.ArgumentParser(description="GB6 诊断：决策位置 top-1/top-2/margin（只读产物 + 重建）")
    ap.add_argument("--tag", default=TAG, help="写入本次记录用的 tag（默认 PM_TAG）")
    ap.add_argument("--runs-tag", default=PROTO_TAG, help="读取哪一批 run 产物（默认 PM_RUNS_TAG）")
    ap.add_argument("--families", default=",".join(FAMILIES), help="逗号分隔（默认 PM_FAMILIES）")
    ap.add_argument("--ar-mode", default=AR_MODE, help="AR 侧模式名（默认 ar）")
    ap.add_argument("--div-mode", default=DIV_MODE, help="divergent 侧模式名（Stage 2b = prod）")
    ap.add_argument("--stage2b", action="store_true",
                    help="Stage 2b 专用入口：等价于 --ar-mode ar --div-mode prod")
    ap.add_argument("--ctx", type=int, default=CTX)
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--allow-last-index-when-equal", action="store_true",
                    help="**仅** legacy 诊断用途：无分歧时退化为末位（位置来源会标成 legacy，"
                         "GB6 消费端会拒绝该记录）")
    a = ap.parse_args()
    ar_mode = "ar" if a.stage2b else a.ar_mode
    div_mode = "prod" if a.stage2b else a.div_mode
    families = [x.strip() for x in a.families.split(",") if x.strip()]
    outdir = Path(a.outdir)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec: dict = dict(run_id=run_id, tool="correctness_protocol_margins", tag=a.tag,
                     timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     git=S.git_meta(ROOT), families=families, runs_tag=a.runs_tag,
                     ar_mode=ar_mode, div_mode=div_mode,
                     stage2b_entry=bool(a.stage2b),
                     protocol="post_shape_localization_v1",
                     position_policy=("first_divergence(AR vs %s) —— 同 tag/同 family/同配置；"
                                      "缺 divergent 侧即拒绝，**不**退化为末 token" % div_mode),
                     timing_separation="本产物**不进入**任何性能/加速比/计时结论（GB6 记账专用）",
                     semantics="按 AR 口径重建**首分歧位置**并读 top-1/top-2/margin；"
                               "检查 profiler 可用性；不加载 draft")
    print(f"[pm] run_id={run_id} families={families} ar_mode={ar_mode} div_mode={div_mode} "
          f"runs_tag={a.runs_tag}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    TRS = None
    try:
        rec["profiler_status"] = _profiler_status()
        print(f"[pm] profiler: {rec['profiler_status']['status']} "
              f"(nsys={rec['profiler_status']['nsys']} ncu={rec['profiler_status']['ncu']} "
              f"torch_cuda_events={rec['profiler_status']['torch_profiler_cuda_kernel_events']})", flush=True)
        from .lowmem_load import load_ours_bf16
        tgt, tok, ti = load_ours_bf16(P5V.TARGET)
        tgt.set_kv_format(spec_dec._fmt(KV))
        M.set_attention_strategy("eager")
        rec["weights_load"] = dict(bitmatch=ti.get("weights_bitmatch"), checked=ti.get("n_params_checked"))
        if TARGET_ROPE == "cpu_fp32":
            from . import target_rope_scope as _TRS
            TRS = _TRS
            rec["target_rope_scope"] = TRS.install(tgt, theta=float(tgt.rope_theta),
                                                   head_dim=int(tgt.head_dim))
        from . import family_input as FI
        out = {}
        for fam in families:
            entry: dict = dict(family=fam)
            # ---- 1) 纯 CPU：配对 + 来源/配置核对（**失败即拒绝**，不退化、不借用别处 margin）----
            try:
                sel = GDS.select_divergent_pair(
                    tag=a.runs_tag, family=fam, outdir=outdir, ar_mode=ar_mode, div_mode=div_mode,
                    ctx=int(a.ctx), allow_last_index_when_equal=bool(a.allow_last_index_when_equal))
            except GDS.PairSelectionError as e:
                out[fam] = dict(entry, status="SELECTION_REFUSED", error_code=e.code, error=e.detail,
                                position_origin=None,
                                note="配对/来源核对失败 ⇒ **拒绝**出记录：不借用旧 tag、旧 backend、"
                                     "其他 family 的 margin，也不退化为末 token")
                rec.setdefault("selection_errors", {})[fam] = dict(code=e.code, detail=e.detail)
                print(f"[pm] {fam}: 配对被拒 [{e.code}] {e.detail}", flush=True)
                continue
            pair = dict(ar=sel["ar"]["file"], div=sel["div"]["file"],
                        ar_mode=ar_mode, div_mode=div_mode, runs_tag=a.runs_tag)
            entry.update(pair=pair, status=sel["status"], position_origin=sel["position_origin"],
                         first_divergence_index=sel["first_divergence_index"],
                         config_checks=sel["config_checks"], round_index=sel["round_index"])
            if sel["status"] != "DIVERGENT":
                # 无分歧 ⇒ GB6 **不需要**诊断：如实记录，**不**伪造位置
                entry.update(position=None, error=None,
                             note="AR 与 production 逐 token 相同 ⇒ 无首分歧；不记录决策位置，"
                                  "GB6 在此 family 上不需要诊断 pass")
                out[fam] = entry
                print(f"[pm] {fam}: AR 与 {div_mode} 无分歧 ⇒ 不记录位置（GB6 不需要诊断）", flush=True)
                continue
            pos = int(sel["first_divergence_index"])
            ar_gen = GDS.load_gen(outdir / sel["ar"]["file"])
            prod_gen = GDS.load_gen(outdir / sel["div"]["file"])
            try:
                # 正确的签名：`load_family_ids(tok, family_id, *, manifest=None, ctx, verify_hash=True)`
                ids, prov = FI.load_family_ids(tok, fam, ctx=int(a.ctx), verify_hash=True)
            except Exception as e:                                        # noqa: BLE001
                ids, prov, entry["provenance_error"] = None, None, f"{type(e).__name__}: {str(e)[:300]}"
            if ids is None:
                out[fam] = dict(entry, error="family_input 输入不可用（校验失败或签名不符）")
                print(f"[pm] {fam}: 输入不可用 ⇒ 跳过（**不换样本**）：{entry['provenance_error']}", flush=True)
                continue
            dev = next(tgt.parameters()).device
            ids = ids.to(dev) if torch.is_tensor(ids) else torch.tensor([ids], device=dev)
            # 载入的输入必须与产物记录的上下文长度一致（否则位置无意义）
            if sel["prompt_len"] is not None and int(ids.shape[1]) != int(sel["prompt_len"]):
                raise DC.CheckError(
                    f"{fam}: 载入输入长度 {int(ids.shape[1])} ≠ 产物 prompt_len {sel['prompt_len']}"
                    f"（输入配置不一致 ⇒ 停止，不产记录）")
            # 按 AR 口径重建：分块 prefill（与 AR 相同）+ 逐 token 单步走到 `pos`
            acc: dict = {}
            inner = P5V.timed_chunked_prefill(acc, CHUNK, tgt)
            with torch.inference_mode(), M.attention_phase("prefill"):
                lg, cache = inner(tgt, ids)
            nxt = int(lg.reshape(-1, lg.shape[-1])[-1].argmax())
            for i in range(pos):
                tok_i = int(ar_gen[i])
                with torch.inference_mode(), M.attention_phase("decode"):
                    lg, cache = spec_dec._step(tgt, tok_i, cache)
                nxt = int(lg.reshape(-1, lg.shape[-1])[-1].argmax())
            row = lg.reshape(-1, lg.shape[-1])[-1].float()
            v, idx = torch.topk(row, 2)
            entry.update(
                provenance=prov, position=int(pos), prompt_len=int(ids.shape[1]),
                ar_top1=int(idx[0]), ar_top2=int(idx[1]),
                ar_top1_value=float(v[0]), ar_top2_value=float(v[1]), margin=float(v[0] - v[1]),
                argmax=int(row.argmax()), argmax_value=float(row.max()),
                tied_at_max=int((row == row.max()).sum()),
                ar_token=(ar_gen[pos] if pos < len(ar_gen) else None),
                diag_token=None,
                prod_token=(prod_gen[pos] if pos < len(prod_gen) else None),
                cache_len_after=int(spec_cache.cache_length(cache)),
                note="按 **AR 口径**（分块 prefill + 逐 token M=1）重建**首分歧位置**"
                     "（同 tag/同 family 的 AR vs %s；位置由 gb6_diag_select 选定并核对配置）"
                     % div_mode)
            out[fam] = entry
            print(f"[pm] {fam}: pos={pos}（首分歧）prompt_len={entry['prompt_len']} "
                  f"AR top1={entry['ar_top1']}({entry['ar_top1_value']:.4g}) "
                  f"top2={entry['ar_top2']}({entry['ar_top2_value']:.4g}) margin={entry['margin']:.4g} "
                  f"AR/PROD token={entry['ar_token']}/{entry['prod_token']}", flush=True)
        rec["families"] = out
        n_ok = sum(1 for v in out.values() if v.get("status") in ("DIVERGENT", "TOKEN_EQUAL",
                                                                  "TOKEN_EQUAL_LEGACY_LAST_INDEX"))
        refused = sorted(rec.get("selection_errors") or {})
        rec["stage"] = ("selection_failed" if refused else "complete")
        rec["n_families_ok"] = n_ok
        rec["families_refused"] = refused
        p = outdir / f"{a.tag}_margins_{run_id}.json"
        rec["artifacts"] = dict(json=str(p))
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[pm] RESULT_JSON={p}")
        if refused:
            print(f"[pm] !! 有 family 因配对/来源核对被拒：{refused}"
                  f" ⇒ 这些 family **不能**用于 GB6 记账（退出码 4）")
            return 4
        return 0
    except Exception as e:                                                # noqa: BLE001
        rec["stage"] = ("checks_failed" if isinstance(e, DC.CheckError) else "crashed")
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        p = outdir / f"{a.tag}_margins_{run_id}.json"
        rec["artifacts"] = dict(json=str(p))
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[pm] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[pm] RESULT_JSON={p}")
        return 5 if rec["stage"] == "checks_failed" else 2
    finally:
        try:
            LB.assert_clean()
            M.clear_attn_probe()
            M.clear_serial_qk()
            M.clear_serial_linear()
            spec_dec.set_verify_wrap(None)
        except Exception:                                                 # noqa: BLE001
            pass
        if TRS is not None:
            try:
                rec["target_rope_scope_restore"] = TRS.assert_restored()
            except Exception:                                             # noqa: BLE001
                pass


if __name__ == "__main__":
    raise SystemExit(main())
