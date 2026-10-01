"""c1_rope_source_diag.py — **只对齐 RoPE 频率来源**后的 C1 诊断（`L0D_MODE=gate|run|compare`）。

## 本轮唯一回答的问题
"把 `inv_freq` 的来源换成 CPU-FP32（本机 HF 的实际公式）之后，自研**完整 prefill** 的 C1
缓存与 logits 是否与 HF 一致？"（不宣称 CPU 更准；不预设能解决 AR/Spec 分歧。）

## 唯一干预（接入点、原理、恢复方式）
* **接入点**：`model_self.QwenLM._run` 里的一句
  `cos, sin = Build_Rope_cache(self.head_dim, seq_len=..., theta=self.rope_theta, device=..., dtype=..., start_pos=start_pos)`
  （`model_self.py:489`）。全仓库只有这一处在 prefill 路径上调用它
  （另外两处是 `align_debug_attn.py` / `fp8_kv_group_probe.py` 的独立诊断，本工具不动它们）。
* **原理**：`Build_Rope_cache` 只改**第一行**——`inv_freq` 在 **CPU** 上以 FP32 按 HF 的公式算好，
  再 `.to(device)` **只做搬运**；其余（`positions` 的 `arange`、`torch.outer`、`cos/sin`、`.to(dtype)` 的
  BF16 转换）**逐行沿用自研原实现**。因此差异源**只有** `inv_freq` 的计算设备。
* **Q/K 同时生效**：`_run` 每次算一份 cos/sin 并传给每一层的 `self_attn(x, cos, sin, cache)`，
  而 `SelfAttention.forward` 对 **q 与 k 都用同一个 cos/sin** 调 `apply_rope`
  （`model_self.py:354-355`）⇒ 不是"只改 K"。
* **不改**：权重、BF16、`allow_bf16_reduced_precision_reduction`、attention backend、mask、投影、
  旋转实现（`apply_rope`/`rotate_half`）、argmax；**不回填**任何参考缓存/hidden/logits。
* **恢复方式**：只替换模块属性 `model_self.Build_Rope_cache`，进程内 `try/finally` 还原；
  还原后用**原函数**重新生成 chunk-0 的表，与 b0db582 落盘的**自研侧实际表**逐位比较（`restore_check`）
  ——既证明"无全局状态残留"，也证明"相对基线只改了这一处"。进程结束即无残留（本诊断不写任何磁盘状态）。

## 三个模式
* `gate`  ：**不加载模型**，先过表验证门（CPU inv_freq == 已记录的 HF inv_freq；四个 chunk 的 cos/sin
            与已捕获的 HF 实际表逐位一致；dtype/shape/位置覆盖/有限性；布局按语义归一化）。
* `run`   ：门通过才执行**一次**自研完整 target prefill（四个 chunk、空缓存起步、到 8192 立即停止；
            不加载 draft、不跑 C2/decode）。
* `compare`：**只读**比较（新 vs 旧自研 vs HF）。
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import torch

from .l0_k_replay import (HF_JSON, HF_PT, OURS_JSON, OURS_PT, OUTDIR, SCHED, TARGET, _cmp_node,
                          _latest, _save, _seq, _sha)

TAG = "c1rope"
CTX = 8192
# ---- 固定引用（不"自动选最新"，除本工具自己产出的 gate 之外）----
L0_HF_JSON = OUTDIR / "l0_replay_hf_20261001T143734Z.json"     # 记录 HF 实际 inv_freq
L0_HF_DUMP = OUTDIR / "l0_replay_hf_dump_20261001T143734Z.pt"  # HF 实际使用的 cos/sin
L0_OURS_DUMP = OUTDIR / "l0_replay_ours_dump_20261001T143510Z.pt"
L0_TABLE_JSON = OUTDIR / "l0_replay_table_20261001T144149Z.json"
MODE = os.environ.get("L0D_MODE", "compare").strip().lower()


def _norm(t: torch.Tensor) -> torch.Tensor:
    """布局归一化：把 (1,1,seq,dim) 与 (1,seq,dim) 都变成 (seq,dim)（**按语义**，不掩盖数值差异）。"""
    return t.reshape(-1, t.shape[-1])


def inv_freq_cpu(head_dim: int, theta: float) -> torch.Tensor:
    """**本机 HF 的确切公式**（`Qwen2RotaryEmbedding.compute_default_rope_parameters`）：
    `1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))`，在 **CPU** 上、FP32。"""
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))


def inv_freq_gpu(head_dim: int, theta: float) -> torch.Tensor:
    """自研原实现的做法（`Build_Rope_cache` 第一行）：在 **GPU** 上算同一个式子。"""
    exponents = torch.arange(0, head_dim, 2, device="cuda", dtype=torch.float32) / head_dim
    return 1.0 / (theta ** exponents)


def build_rope_cache_cpu_invfreq(head_dim: int, seq_len: int, theta: float, device, dtype,
                                 start_pos: int = 0, *, inv_cpu: torch.Tensor | None = None):
    """自研 `Build_Rope_cache` 的**逐行复刻**，只把 `inv_freq` 的来源换成 CPU-FP32 后再搬到 device。"""
    calc_dtype = torch.float32
    if inv_cpu is None:
        inv_cpu = inv_freq_cpu(head_dim, theta)
    inv_freq = inv_cpu.to(device=device, dtype=calc_dtype)          # ← 只做搬运
    postitions = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=calc_dtype)
    freqs = torch.outer(postitions, inv_freq)
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1).unsqueeze(0).unsqueeze(0)
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1).unsqueeze(0).unsqueeze(0)
    return cos.to(dtype), sin.to(dtype)


# ------------------------------------------------------------------ gate
def _gate(ts: str) -> int:
    rec: dict = dict(tool="c1_rope_source_diag", mode="gate", run_utc=ts,
                     semantics="表验证门：不加载模型；只验证 CPU-FP32 inv_freq 与四个 chunk 的 cos/sin",
                     fixed=dict(l0_hf_json=str(L0_HF_JSON), l0_hf_dump=str(L0_HF_DUMP),
                                l0_ours_dump=str(L0_OURS_DUMP), l0_table_json=str(L0_TABLE_JSON)))
    for p in (L0_HF_JSON, L0_HF_DUMP, L0_OURS_DUMP, L0_TABLE_JSON, HF_JSON, HF_PT, OURS_JSON, OURS_PT):
        if not p.exists():
            print(f"[c1r] !! 缺少固定引用：{p}")
            rec["stage"] = "missing_reference"
            _save(rec, f"{TAG}_gate", ts)
            return 5
    # ---- 0) 输入与配置核实 ----
    seq, meta = _seq()
    hj = json.load(open(HF_JSON, encoding="utf-8"))     # 旧 HF C1 快照记录（输入 hash / C1 数值）
    h0 = json.load(open(L0_HF_JSON, encoding="utf-8"))  # b0db582 的 HF 侧重放记录（config + 实际 inv_freq）
    rec["input"] = dict(source_kind="manifest", **meta)
    rec["config"] = dict(
        theta_hf=float(h0["config"]["rope_theta_from_config"]),
        head_dim_hf=int(h0["config"]["head_dim"]),
        head_dim_key_present=bool(h0["config"]["head_dim_key_present"]),
        head_dim_from_attn_module=int(h0["config"]["head_dim_from_attn_module"]),
        rope_parameters=h0["config"]["rope_parameters"],
        rms_norm_eps=float(h0["config"]["rms_norm_eps"]),
        hf_apply_rotary=h0["config"]["apply_rotary_pos_emb"],
        reduction=bool(h0["precision"]["bf16_reduced_precision_reduction"]),
        allow_tf32=bool(h0["precision"]["allow_tf32"]),
        source_note="config/精度取自 b0db582 的 HF 侧重放记录；input hash 与旧 HF C1 记录交叉核对")
    print(f"[c1r] theta={rec['config']['theta_hf']} head_dim={rec['config']['head_dim_hf']} "
          f"seq_sha16={meta['seq_sha16']}")
    assert meta["seq_sha16"] == hj["input"]["seq_sha16"] == h0["input"]["seq_sha16"] \
        == "679ad27f88c66887", "输入 hash 与既有记录不一致"
    theta, dim = rec["config"]["theta_hf"], rec["config"]["head_dim_hf"]

    # ---- 1) CPU inv_freq 与"已记录的 HF inv_freq"一致 ----
    inv_cpu = inv_freq_cpu(dim, theta)
    inv_gpu = inv_freq_gpu(dim, theta).cpu()
    rec_hf = torch.tensor(h0["inv_freq"]["values"], dtype=torch.float32)
    rec["inv_freq"] = dict(
        cpu_sha=_sha(inv_cpu), gpu_sha=_sha(inv_gpu), recorded_hf_sha=h0["inv_freq"]["sha"],
        cpu_equals_recorded=bool(torch.equal(inv_cpu, rec_hf)),
        gpu_equals_recorded=bool(torch.equal(inv_gpu, rec_hf)),
        cpu_vs_gpu_diff_idx=[int(i) for i in torch.nonzero(inv_cpu != inv_gpu).reshape(-1).tolist()],
        shape=list(inv_cpu.shape), dtype=str(inv_cpu.dtype),
        nonfinite=int((~torch.isfinite(inv_cpu)).sum()))
    print(f"[c1r] inv_freq: CPU sha={_sha(inv_cpu)} 记录HF sha={h0['inv_freq']['sha']} "
          f"一致={rec['inv_freq']['cpu_equals_recorded']}；GPU 一致={rec['inv_freq']['gpu_equals_recorded']}"
          f"；CPU vs GPU 差异下标={rec['inv_freq']['cpu_vs_gpu_diff_idx']}")

    # ---- 2) 四个 chunk 的表 vs 已捕获的 HF 实际表 ----
    hd = torch.load(L0_HF_DUMP, map_location="cpu")
    od = torch.load(L0_OURS_DUMP, map_location="cpu")
    per = {}
    ok = bool(rec["inv_freq"]["cpu_equals_recorded"]) and rec["inv_freq"]["nonfinite"] == 0
    for st, en in SCHED:
        n = en - st
        cos, sin = build_rope_cache_cpu_invfreq(dim, n, theta, "cuda", torch.bfloat16,
                                                start_pos=st, inv_cpu=inv_cpu)
        cH, sH = hd[st]["cos"].to("cuda"), hd[st]["sin"].to("cuda")
        cO, sO = od[st]["cos"].to("cuda"), od[st]["sin"].to("cuda")
        pos_ok = bool(torch.equal(hd[st]["positions"].to("cuda"),
                                  torch.arange(st, en, device="cuda").view(1, -1)))
        d = dict(
            chunk=[st, en], positions=dict(start=st, end=en - 1, n=n, match_hf_dump=pos_ok),
            cos=dict(dtype=str(cos.dtype), shape=list(cos.shape), layout_hf=list(cH.shape),
                     nonfinite=int((~torch.isfinite(cos)).sum()),
                     bitequal_hf=bool(torch.equal(_norm(cos), _norm(cH))),
                     n_diff_hf=int((_norm(cos) != _norm(cH)).sum()),
                     bitequal_ours_dump=bool(torch.equal(_norm(cos), _norm(cO))),
                     n_diff_ours_dump=int((_norm(cos) != _norm(cO)).sum())),
            sin=dict(dtype=str(sin.dtype), shape=list(sin.shape), layout_hf=list(sH.shape),
                     nonfinite=int((~torch.isfinite(sin)).sum()),
                     bitequal_hf=bool(torch.equal(_norm(sin), _norm(sH))),
                     n_diff_hf=int((_norm(sin) != _norm(sH)).sum()),
                     bitequal_ours_dump=bool(torch.equal(_norm(sin), _norm(sO))),
                     n_diff_ours_dump=int((_norm(sin) != _norm(sO)).sum())))
        per[str(st)] = d
        ok = ok and pos_ok and d["cos"]["bitequal_hf"] and d["sin"]["bitequal_hf"] \
            and d["cos"]["nonfinite"] == 0 and d["sin"]["nonfinite"] == 0
        print(f"[c1r] chunk [{st},{en})：cos==HF表 {d['cos']['bitequal_hf']}"
              f"（vs 自研旧表 不同 {d['cos']['n_diff_ours_dump']}）、"
              f"sin==HF表 {d['sin']['bitequal_hf']}（vs 自研旧表 不同 {d['sin']['n_diff_ours_dump']}）；"
              f"dtype={d['cos']['dtype']} shape={d['cos']['shape']} 位置覆盖={pos_ok}")
    rec["chunks"] = per
    rec["gate_pass"] = bool(ok)
    rec["stage"] = "complete" if ok else "gate_failed"
    print(f"\n[c1r] ★ 表验证门：{'通过' if ok else '**未通过**'}"
          + ("" if ok else " ⇒ 落盘停止，不继续完整 prefill，不追加其它修改"))
    _save(rec, f"{TAG}_gate", ts)
    return 0 if ok else 6


# ------------------------------------------------------------------ run
def _run(ts: str) -> int:
    from . import g5_8k_spec_variance_probe as P5V
    from . import model_self as M
    from . import lowmem_load, spec_cache
    gate_p = _latest(f"{TAG}_gate_*.json")
    gate = json.load(open(gate_p, encoding="utf-8"))
    rec: dict = dict(tool="c1_rope_source_diag", mode="run", run_utc=ts, gate_json=gate_p.name,
                     prior_attempt=dict(
                         json=os.environ.get("C1R_PRIOR_ATTEMPT", ""),
                         note="上一次尝试在**任何层前向之前**因工具包装函数参数名不匹配而崩溃"
                              "（`Build_Rope_cache` 是以 `theta=` 关键字调用的）；未计算任何层、未建立任何缓存；"
                              "崩溃产物已保留 ⇒ 本次仍属**第一次** C1 诊断运行"),
                     semantics="唯一干预：inv_freq 改为 CPU-FP32（HF 公式）后再搬到 GPU；"
                               "其余逐行沿用自研原实现；一次完整 prefill 到 C1(len 8192) 即停")
    if not gate.get("gate_pass"):
        rec["stage"] = "refused"
        rec["reason"] = "表验证门未通过 ⇒ 不执行完整 prefill"
        _save(rec, f"{TAG}_run", ts)
        print("[c1r] !! 拒绝执行（门未通过）")
        return 6
    ok, why = P5V.commit_guard()
    print(f"[c1r] commit: {why}")
    rec["commit_guard"] = dict(ok=bool(ok), why=why)
    if not ok:
        rec["stage"] = "aborted_commit_guard"
        _save(rec, f"{TAG}_run", ts)
        return 3
    try:
        seq, meta = _seq()
        rec["input"] = dict(source_kind="manifest", **meta)
        theta = float(gate["config"]["theta_hf"])
        dim = int(gate["config"]["head_dim_hf"])
        inv_cpu = inv_freq_cpu(dim, theta)
        tgt, _tok, info = lowmem_load.load_ours_bf16(TARGET)
        tgt.set_kv_format("fp16")
        M.set_diag_preset("eager")
        M.set_attention_strategy("eager")
        b = torch.backends.cuda.matmul
        b.allow_bf16_reduced_precision_reduction = True
        rec["weights"] = dict(bitmatch=info["weights_bitmatch"], n_checked=info["n_params_checked"],
                              mismatch=info["mismatch_keys"], load_path=info["load_path"])
        rec["config"] = dict(rope_theta_from_model=float(tgt.rope_theta), head_dim=int(tgt.head_dim),
                             theta_used=theta, dim_used=dim, attn_strategy=str(M.attention_strategy()),
                             kv_format="fp16=base_bf16",
                             diag=dict(qk_fp32=bool(M.DIAG_QK_FP32), prob_fp32=bool(M.DIAG_PROB_FP32)),
                             precision=dict(bf16_reduced_precision_reduction=bool(
                                 b.allow_bf16_reduced_precision_reduction),
                                 allow_tf32=bool(b.allow_tf32)))
        assert float(tgt.rope_theta) == theta and int(tgt.head_dim) == dim, "模型参数与门记录不一致"
        # ---- 唯一干预：替换模块属性；进程内 try/finally 还原 ----
        orig = M.Build_Rope_cache
        calls = []

        def diag_build(head_dim, seq_len, theta, device, dtype, start_pos=0):
            # ⚠️ 参数名必须与 `model_self.Build_Rope_cache` **完全一致**：`_run` 是用
            # `theta=`、`device=`、`dtype=`、`start_pos=` 这些**关键字**调用的
            # （首版改名成 `theta_` 导致 `unexpected keyword argument 'theta'`，在任何层前向之前就崩了）
            calls.append(dict(head_dim=int(head_dim), seq_len=int(seq_len), theta=float(theta),
                              device=str(device), dtype=str(dtype), start_pos=int(start_pos)))
            return build_rope_cache_cpu_invfreq(head_dim, seq_len, theta, device, dtype,
                                                start_pos=start_pos, inv_cpu=inv_cpu)

        M.Build_Rope_cache = diag_build
        try:
            ids = torch.tensor([seq], device="cuda")
            cache = None
            with torch.inference_mode(), M.attention_phase("prefill"):
                for st, en in SCHED:
                    lg, cache = tgt._run(tgt.embed_tokens(ids[:, st:en]), cache,
                                         need_logits=(en == CTX))
        finally:
            M.Build_Rope_cache = orig
        restored = M.Build_Rope_cache is orig
        # 还原核查：用**原函数**重算 chunk-0 表，必须与 b0db582 落盘的自研实际表逐位相同
        od = torch.load(L0_OURS_DUMP, map_location="cpu")
        c0, s0 = M.Build_Rope_cache(dim, 2048, tgt.rope_theta, "cuda", torch.bfloat16, start_pos=0)
        rec["intervention"] = dict(
            injection_point="model_self.QwenLM._run 的 Build_Rope_cache 调用（model_self.py:489）",
            how="只在 CPU-FP32 上算 inv_freq，再 .to(device) 搬运；positions/outer/cos/sin/bf16 cast 逐行走原实现",
            applies_to="同一份 cos/sin 同时用于 Q 与 K（model_self.py:354-355）",
            calls=calls, n_calls=len(calls), restored=bool(restored),
            restore_check=dict(
                ours_dump_cos_bitequal=bool(torch.equal(_norm(c0), _norm(od[0]["cos"].to("cuda")))),
                ours_dump_sin_bitequal=bool(torch.equal(_norm(s0), _norm(od[0]["sin"].to("cuda"))))),
            no_disk_state="本诊断不写任何模块级/磁盘全局状态；进程退出即完全消失")
        print(f"[c1r] 干预：Build_Rope_cache 调用 {len(calls)} 次；已还原={restored}；"
              f"还原核查（原函数 == 落盘的自研旧表）="
              f"{rec['intervention']['restore_check']['ours_dump_cos_bitequal']}/"
              f"{rec['intervention']['restore_check']['ours_dump_sin_bitequal']}")
        assert restored and all(rec["intervention"]["restore_check"].values()), "还原核查未通过"
        # ---- 结果与检查 ----
        clen = int(spec_cache.cache_length(cache))
        rec["C1"] = dict(cache_len=clen, n_layers=len(cache))
        assert clen == CTX, f"cache 长度 {clen} != {CTX}"
        layers = []
        for i, d in enumerate(cache):
            layers.append(dict(layer=i, k_shape=list(d["k"].shape), v_shape=list(d["v"].shape),
                               k_dtype=str(d["k"].dtype), v_dtype=str(d["v"].dtype),
                               k_finite=bool(torch.isfinite(d["k"]).all()),
                               v_finite=bool(torch.isfinite(d["v"]).all()),
                               k_len=int(d["k"].shape[2]), v_len=int(d["v"].shape[2]),
                               k_sha=_sha(d["k"]), v_sha=_sha(d["v"])))
        rec["layers"] = layers
        assert all(x["k_finite"] and x["v_finite"] and x["k_len"] == CTX and x["v_len"] == CTX
                   for x in layers), "存在非有限值或长度不符"
        lg = lg[0, -1].detach().float().cpu()
        rec["logits"] = dict(shape=[int(lg.numel())], finite=bool(torch.isfinite(lg).all()),
                             argmax=int(lg.argmax()),
                             top2_margin=float(torch.topk(lg, 2).values.diff().abs().item()),
                             n_tied_max=int((lg == lg.max()).sum()), sha=_sha(lg))
        rec["attention_stats"] = M.attention_stats()
        rec["tables"] = dict(inv_freq_cpu_sha=_sha(inv_cpu), inv_freq_dim=dim, inv_freq_theta=theta,
                             per_call=[dict(seq_len=c["seq_len"], start_pos=c["start_pos"]) for c in calls])
        rec["resources"] = dict(pt_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                                pt_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20,
                                cuda_free_mib=torch.cuda.mem_get_info()[0] / 2 ** 20,
                                note="只作诊断，不作性能声明")
        out = dict(c1_logits=lg, c1_kv=[{k: v.detach().float().cpu() for k, v in d.items()}
                                        for d in cache],
                   inv_freq_cpu=inv_cpu,
                   meta=dict(run_id=ts, intervention="cpu_fp32_inv_freq", ctx=CTX, schedule=SCHED))
        assert all(torch.equal(out["c1_kv"][i][kk].to(torch.bfloat16).float(), out["c1_kv"][i][kk])
                   for i in range(len(out["c1_kv"])) for kk in ("k", "v")), "存的不是精确 bf16 值"
        pp = OUTDIR / f"{TAG}_new_dump_{ts}.pt"
        torch.save(out, str(pp))
        rec["dump_pt"] = str(pp)
        rec["stage"] = "complete"
        _save(rec, f"{TAG}_run", ts)
        print(f"[c1r] C1 len={clen} argmax={rec['logits']['argmax']} "
              f"margin={rec['logits']['top2_margin']:.6g} | dump={pp.name}")
        return 0
    except Exception as e:                                        # noqa: BLE001
        rec["stage"] = "crashed"
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        _save(rec, f"{TAG}_run", ts)
        print(f"[c1r] !! 停止：{e}")
        return 2


# ------------------------------------------------------------------ compare（只读）
def _cmp_logits(name, a, b) -> dict:
    x, y = a.float(), b.float()
    d = (x - y).abs()
    fa, fb = torch.topk(x, 2).values, torch.topk(y, 2).values
    return dict(pair=name, bitequal=bool(torch.equal(x, y)), n_diff=int((x != y).sum()),
                numel=int(x.numel()), max_abs=float(d.max()), mean_abs=float(d.mean()),
                argmax_a=int(x.argmax()), argmax_b=int(y.argmax()),
                same_argmax=bool(int(x.argmax()) == int(y.argmax())),
                margin_a=float((fa[0] - fa[1]).abs()), margin_b=float((fb[0] - fb[1]).abs()),
                n_tied_max_a=int((x == x.max()).sum()), n_tied_max_b=int((y == y.max()).sum()))


def _cmp_kv(name, kvA, kvB) -> dict:
    rows = []
    for i in range(min(len(kvA), len(kvB))):
        for kk in ("k", "v"):
            x, y = kvA[i][kk].float(), kvB[i][kk].float()
            d = (x - y).abs()
            rows.append(dict(layer=i, attr=kk, bitequal=bool(torch.equal(x, y)),
                             n_diff=int((x != y).sum()), numel=int(x.numel()),
                             max_abs=float(d.max()), mean_abs=float(d.mean())))
    per_layer = [dict(layer=i,
                      bitequal_all=all(r["bitequal"] for r in rows if r["layer"] == i),
                      n_diff=sum(r["n_diff"] for r in rows if r["layer"] == i))
                 for i in range(min(len(kvA), len(kvB)))]
    return dict(pair=name, rows=rows, per_layer=per_layer, n_tensors=len(rows),
                n_bitequal=sum(1 for r in rows if r["bitequal"]),
                n_tensors_all_bitwise=all(r["bitequal"] for r in rows),
                max_abs=float(max(r["max_abs"] for r in rows)),
                n_diff_total=int(sum(r["n_diff"] for r in rows)))


def _compare(ts: str) -> int:
    rp = _latest(f"{TAG}_run_*.json")
    rj = json.load(open(rp, encoding="utf-8"))
    if rj.get("stage") != "complete" or not rj.get("dump_pt") or not Path(rj["dump_pt"]).exists():
        print(f"[c1r] !! 最新的 run 记录 {rp.name} 不是完成的运行（stage={rj.get('stage')}）⇒ 拒绝比较")
        return 5
    new = torch.load(Path(rj["dump_pt"]), map_location="cpu")
    old = torch.load(OURS_PT, map_location="cpu")
    hf = torch.load(HF_PT, map_location="cpu")
    ojs = json.load(open(OURS_JSON, encoding="utf-8"))
    hjs = json.load(open(HF_JSON, encoding="utf-8"))
    rec: dict = dict(tool="c1_rope_source_diag", mode="compare", run_utc=ts,
                     new_run=rj.get("run_id") or rj.get("run_utc"),
                     new_json=rp.name, new_dump=Path(rj["dump_pt"]).name,
                     refs=dict(old_ours=OURS_JSON.name, hf=HF_JSON.name),
                     intervention=rj.get("intervention"),
                     input=dict(seq_sha16_new=rj["input"]["seq_sha16"],
                                seq_sha16_old=ojs["input"]["seq_sha16"],
                                seq_sha16_hf=hjs["input"]["seq_sha16"]))
    assert rec["input"]["seq_sha16_new"] == rec["input"]["seq_sha16_old"] == rec["input"]["seq_sha16_hf"]
    print("== K/V：全部 28 层 ×(k,v) = 56 个张量 ==")
    rec["kv_new_vs_old"] = _cmp_kv("new_vs_old_ours", new["c1_kv"], old["c1_kv"])
    rec["kv_new_vs_hf"] = _cmp_kv("new_vs_hf", new["c1_kv"], hf["c1_kv"])
    rec["kv_old_vs_hf"] = _cmp_kv("old_ours_vs_hf", old["c1_kv"], hf["c1_kv"])
    for key, tag in (("kv_old_vs_hf", "旧自研 vs HF"), ("kv_new_vs_hf", "新自研 vs HF"),
                     ("kv_new_vs_old", "新自研 vs 旧自研")):
        x = rec[key]
        print(f"  {tag:<16} 逐位相同张量 {x['n_bitequal']}/{x['n_tensors']}；"
              f"不同元素合计 {x['n_diff_total']}；max|Δ|={x['max_abs']:.6g}")
    print("== C1 末位 logits（151936 维）==")
    rec["logits_new_vs_old"] = _cmp_logits("new_vs_old_ours", new["c1_logits"], old["c1_logits"])
    rec["logits_new_vs_hf"] = _cmp_logits("new_vs_hf", new["c1_logits"], hf["c1_logits"])
    rec["logits_old_vs_hf"] = _cmp_logits("old_ours_vs_hf", old["c1_logits"], hf["c1_logits"])
    for key, tag in (("logits_old_vs_hf", "旧自研 vs HF"), ("logits_new_vs_hf", "新自研 vs HF"),
                     ("logits_new_vs_old", "新自研 vs 旧自研")):
        x = rec[key]
        print(f"  {tag:<16} 逐位相同={x['bitequal']} 不同元素 {x['n_diff']}/{x['numel']} "
              f"max|Δ|={x['max_abs']:.6g} mean|Δ|={x['mean_abs']:.6g} | "
              f"argmax {x['argmax_a']} vs {x['argmax_b']} 同={x['same_argmax']} | "
              f"margin {x['margin_a']:.6g}/{x['margin_b']:.6g} 并列 {x['n_tied_max_a']}/{x['n_tied_max_b']}")
    print("== L0 K：此前的 60 个差异 ==")
    l0 = dict(old_vs_hf_n_diff=int((old["c1_kv"][0]["k"] != hf["c1_kv"][0]["k"]).sum()),
              new_vs_hf_n_diff=int((new["c1_kv"][0]["k"] != hf["c1_kv"][0]["k"]).sum()),
              new_vs_old_n_diff=int((new["c1_kv"][0]["k"] != old["c1_kv"][0]["k"]).sum()))
    l0["l0_k_60_removed"] = bool(l0["new_vs_hf_n_diff"] == 0 and l0["old_vs_hf_n_diff"] == 60)
    rec["l0_k"] = l0
    print(f"  旧自研 vs HF 不同元素={l0['old_vs_hf_n_diff']}；新自研 vs HF 不同元素={l0['new_vs_hf_n_diff']}；"
          f"新自研 vs 旧自研={l0['new_vs_old_n_diff']} ⇒ 60 个差异已消除={l0['l0_k_60_removed']}")
    # ---- 结论（按预先声明的口径，不夸大）----
    all_bit = rec["kv_new_vs_hf"]["n_tensors_all_bitwise"] and rec["logits_new_vs_hf"]["bitequal"]
    none_bit = rec["kv_new_vs_hf"]["n_bitequal"] == 0 and not rec["logits_new_vs_hf"]["bitequal"]
    if all_bit:
        verdict = ("全部 K/V 与 logits 逐位相同 ⇒ **仅对齐频率来源，在该样本与该配置下足以消除 C1 跨实现差异**"
                   "（不推及其他样本/配置/backend；不宣布 AR/Spec 严格一致已修复；不宣布可启动正式实验）")
    elif l0["l0_k_60_removed"]:
        verdict = ("L0 的 60 个差异已消除，但 C1 仍有剩余差异 ⇒ 分开报告：已消除＝L0 K（及相应传播部分）；"
                   "剩余＝下列新自研 vs HF 的 K/V 与 logits 差异；**不得**把 max|Δ| 下降称为完全修复")
    else:
        verdict = ("L0 的差异**未**消除 ⇒ 停止，先核查干预是否被忠实执行（见 intervention/restore_check），"
                   "不继续向后归因")
    rec["verdict"] = verdict
    rec["not_claimed"] = ["AR/Spec 严格一致性已修复", "所有输入或 backend 已通过",
                          "正式实验可以启动", "此前所有差异只有这一个原因"]
    print(f"\n[c1r] ★ {verdict}")
    _save(rec, f"{TAG}_compare", ts)
    return 0


def main() -> int:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if MODE == "gate":
        return _gate(ts)
    if MODE == "run":
        return _run(ts)
    if MODE == "compare":
        return _compare(ts)
    print(f"[c1r] !! 非法 L0D_MODE={MODE!r}（可选 gate|run|compare）")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
