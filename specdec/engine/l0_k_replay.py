"""l0_k_replay.py — **第 0 层 K 生成的最小重放**：节点对照 + 复现门 + 条件式 RoPE 表替换。

## 为什么需要它
既有产物只能告诉我们"L0 K 有 60 个元素不同、L0 V 逐位相同"。**不能**据此判断差异出现在
`input RMSNorm / K 投影 / RoPE` 的哪一步 —— `input_layernorm` 在投影**之前**（勘误之二 §11.1），
所以必须做**节点级实测**。

## 只做什么
每个实现**一次**原形状重放：`Embedding → input_layernorm → k_proj/v_proj → RoPE`，**只到第 0 层**。
**不**跑其余 27 层、**不**算 attention、**不**算 LM head、**不**做生成。
保持既有运行的全部条件：bf16 权重、`reduction=True`、QK32/PROB32 关闭、原 token ids/位置编号、
`chunk=2048`、原投影调用形状与张量布局、原 RoPE 调用方式；theta/head_dim **从 config 与模型实例读取**，不手写。
★ **不得缩小矩阵形状**：每个 chunk 都按**原始 2048 行**计算，比较与列表都在**计算完成之后**再做。

## 复现门（先过门，再谈归因）
重放出的 **RoPE 后 K** 必须与**该实现自己**的既有 C1 快照对应 chunk **逐位相同**；
额外的对照门：**V** 也必须逐位相同（V 在两侧本就逐位相等，是重放忠实性的控制项）。
任一侧未过门 ⇒ 落盘 `stage="gate_failed"` 并停止；**不**改精度、不换 kernel、不扩大扫描、不加新阈值。

## 结论口径（第一个已检查且不同的节点）
节点顺序：`embed → norm → k_pre → (cos/sin) → k_post`。
`L0_MODE=compare` 会给出 `earliest_differing_node` 与 `rope_substitution_allowed`；
**只有** 复现门全过 **且** `k_pre` 两侧逐位相同 **且** `k_post` 不同 时，才允许 `L0_MODE=subst`。

## 模式
* `L0_MODE=ours`  ：自研侧重放（1 次 GPU 加载；顺序执行，不与其他实现同时驻留）
* `L0_MODE=hf`    ：HF 侧重放（同上）
* `L0_MODE=compare`：**纯 CPU** 节点对照（不加载任何模型）
* `L0_MODE=subst` ：条件式替换对照（固定同一份 RoPE 前 K、同一旋转实现，只换 cos/sin）；
                   条件不满足时**自己拒绝**并落盘 `refused`（rc=6）
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
TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
MODE = os.environ.get("L0_MODE", "compare").strip().lower()
TAG = "l0_replay"
CHUNK = 2048
CTX = 8192
PRE = 8240
# ---- 固定原始路径（不"自动选最新"）----
OURS_PT = OUTDIR / "xb_20261001_ours_20261001T132915Z.pt"
HF_PT = OUTDIR / "xb_20261001_hf_20261001T132947Z.pt"
OURS_JSON = OUTDIR / "xb_20261001_ours_20261001T132915Z.json"
HF_JSON = OUTDIR / "xb_20261001_hf_20261001T132947Z.json"
PREFIX_PT = OUTDIR / "f13i50_20261001_capture_ar_20261001T072313Z.pt"
SCHED = [(0, 2048), (2048, 4096), (4096, 6144), (6144, 8192)]


def _save(rec: dict, prefix: str, ts: str) -> Path:
    p = OUTDIR / f"{prefix}_{ts}.json"
    if p.exists():
        k = 1
        while (OUTDIR / f"{prefix}_{ts}_r{k}.json").exists():
            k += 1
        p = OUTDIR / f"{prefix}_{ts}_r{k}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[l0r] RESULT_JSON={p}")
    return p


def _seq() -> tuple:
    from transformers import AutoTokenizer
    from . import family_input as FI
    tok = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True)
    ids_list, fam_meta = FI.load_family_ids(tok, "F13", ctx=CTX)
    cap = torch.load(PREFIX_PT, map_location="cpu")
    prefix = [int(x) for x in cap["prefix_tokens"]][:PRE - CTX]
    seq = list(ids_list) + prefix
    sha = hashlib.sha256(json.dumps(seq).encode("utf-8")).hexdigest()[:16]
    assert sha == "679ad27f88c66887", f"输入序列 sha16={sha} 与既有产物不一致"
    meta = dict(family=fam_meta, ctx=CTX, prefix_pt=str(PREFIX_PT), prefix_len=len(prefix),
                seq_sha16=sha, tokens=[], note="只消费**已消费过的** F13；不引入任何新样本")
    return seq, meta


def _nodes_meta(name: str, t: torch.Tensor) -> dict:
    return dict(name=name, dtype=str(t.dtype), shape=list(t.shape), stride=list(t.stride()),
                numel=int(t.numel()), n_nonfinite=int((~torch.isfinite(t)).sum()),
                min=float(t.float().min()), max=float(t.float().max()),
                sha=_sha(t))


def _sha(t: torch.Tensor, n: int = 16) -> str:
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:n]


# ------------------------------------------------------------------ 自研侧
def _run_ours(ts: str) -> int:
    from . import g5_8k_spec_variance_probe as P5V
    from . import model_self as M
    from . import lowmem_load
    rec: dict = dict(tool="l0_k_replay", mode="ours", run_utc=ts, chunk=CHUNK, ctx=CTX, pre=PRE,
                     source_pt=str(OURS_PT), source_json=str(OURS_JSON),
                     semantics="只到第 0 层的 K/V 生成重放；不跑其余层/attention/LM head/生成")
    ok, why = P5V.commit_guard()
    print(f"[l0r] commit: {why}")
    rec["commit_guard"] = dict(ok=bool(ok), why=why)
    if not ok:
        rec["stage"] = "aborted_commit_guard"
        _save(rec, f"{TAG}_ours", ts)
        return 3
    try:
        seq, meta = _seq()
        tgt, _tok, info = lowmem_load.load_ours_bf16(TARGET)
        tgt.set_kv_format("fp16")
        M.set_diag_preset("eager")
        M.set_attention_strategy("eager")
        rec["input"] = dict(source_kind="manifest", **meta)
        rec["weights"] = dict(bitmatch=info["weights_bitmatch"], n_checked=info["n_params_checked"],
                              mismatch=info["mismatch_keys"])
        rec["config"] = dict(rope_theta_from_model=float(tgt.rope_theta),
                             head_dim_from_model=int(tgt.head_dim), kv_heads=int(tgt.layers[0].self_attn.kv_heads),
                             heads=int(tgt.layers[0].self_attn.heads),
                             rms_eps=float(tgt.layers[0].input_layernorm.eps),
                             k_proj_bias=bool(tgt.layers[0].self_attn.k_proj.bias is not None),
                             attn_impl=str(M.attention_strategy()),
                             diag=dict(qk_fp32=bool(M.DIAG_QK_FP32), prob_fp32=bool(M.DIAG_PROB_FP32)))
        b = torch.backends.cuda.matmul
        b.allow_bf16_reduced_precision_reduction = True
        rec["precision"] = dict(bf16_reduced_precision_reduction=bool(b.allow_bf16_reduced_precision_reduction),
                                allow_tf32=bool(b.allow_tf32),
                                fp32_precision=str(getattr(b, "fp32_precision", "<无该属性>")))
        print(f"[l0r] ours: theta={tgt.rope_theta} head_dim={tgt.head_dim} "
              f"bias={rec['config']['k_proj_bias']} reduction={rec['precision']['bf16_reduced_precision_reduction']}")
        snap = torch.load(OURS_PT, map_location="cpu")["c1_kv"]
        sa = tgt.layers[0].self_attn
        dump, gates, tables = {}, {}, {}
        for st, en in SCHED:
            ids = torch.tensor([seq[st:en]], device="cuda")
            pos = torch.arange(st, en, device="cuda").view(1, -1)
            with torch.inference_mode(), M.attention_phase("prefill"):
                emb = tgt.embed_tokens(ids)
                h = tgt.layers[0].input_layernorm(emb)
                k_pre = sa.k_proj(h).view(1, en - st, sa.kv_heads, sa.head_dim).transpose(1, 2)
                v = sa.v_proj(h).view(1, en - st, sa.kv_heads, sa.head_dim).transpose(1, 2)
                cos, sin = M.Build_Rope_cache(sa.head_dim, en - st, tgt.rope_theta, emb.device,
                                              emb.dtype, start_pos=st)
                k_post = M.apply_rope(k_pre, cos, sin)
            kref, vref = snap[0]["k"][:, :, st:en, :], snap[0]["v"][:, :, st:en, :]
            gates[st] = dict(k_bitequal=bool(torch.equal(k_post.float().cpu(), kref)),
                             v_bitequal=bool(torch.equal(v.float().cpu(), vref)),
                             k_max_abs=float((k_post.float().cpu() - kref).abs().max()),
                             v_max_abs=float((v.float().cpu() - vref).abs().max()),
                             snapshot_k_sha=_sha(kref), replay_k_sha=_sha(k_post.cpu()))
            dump[st] = dict(embed=emb.cpu(), norm=h.cpu(), k_pre=k_pre.cpu(), v=v.cpu(),
                            cos=cos.cpu(), sin=sin.cpu(), k_post=k_post.cpu(),
                            ids=ids.cpu(), positions=pos.cpu())
            tables[st] = dict(cos=_nodes_meta("cos", cos), sin=_nodes_meta("sin", sin))
            print(f"[l0r]   chunk [{st},{en}) gate: K={gates[st]['k_bitequal']} V={gates[st]['v_bitequal']} "
                  f"(K max|Δ|={gates[st]['k_max_abs']:.6g})")
        rec["gates"] = gates
        rec["tables"] = tables
        rec["nodes"] = {st: dict(embed=_nodes_meta("embed", dump[st]["embed"]),
                                 norm=_nodes_meta("norm", dump[st]["norm"]),
                                 k_pre=_nodes_meta("k_pre", dump[st]["k_pre"]),
                                 k_post=_nodes_meta("k_post", dump[st]["k_post"]),
                                 v=_nodes_meta("v", dump[st]["v"])) for st, _ in SCHED}
        gate_ok = all(g["k_bitequal"] and g["v_bitequal"] for g in gates.values())
        rec["gate_all_pass"] = bool(gate_ok)
        dp = OUTDIR / f"{TAG}_ours_dump_{ts}.pt"
        torch.save(dump, str(dp))
        rec["dump_pt"] = str(dp)
        rec["stage"] = "complete" if gate_ok else "gate_failed"
        _save(rec, f"{TAG}_ours", ts)
        print(f"[l0r] ours 复现门：{'全过' if gate_ok else '**未过**'}；dump={dp.name}")
        return 0 if gate_ok else 7
    except Exception as e:                                        # noqa: BLE001
        rec["stage"] = "crashed"
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        _save(rec, f"{TAG}_ours", ts)
        print(f"[l0r] !! 停止：{e}")
        return 2


# ------------------------------------------------------------------ HF 侧
def _run_hf(ts: str) -> int:
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    from . import g5_8k_spec_variance_probe as P5V
    rec: dict = dict(tool="l0_k_replay", mode="hf", run_utc=ts, chunk=CHUNK, ctx=CTX, pre=PRE,
                     source_pt=str(HF_PT), source_json=str(HF_JSON), versions=dict(
                         torch=torch.__version__, transformers=transformers.__version__,
                         cuda=torch.version.cuda),
                     prior_attempt=dict(
                         json=os.environ.get("L0_PRIOR_ATTEMPT", ""),
                         note="上一次尝试在**任何前向之前**因工具缺陷崩溃（读取不存在的 `config.head_dim`，"
                              "v5 的生产 config.json 没有该键、head_dim 由 hidden/heads 推出）；"
                              "崩溃产物已保留，未执行任何重放 ⇒ 本次仍属**第一组**原形状重放"),
                     semantics="只到第 0 层的 K/V 生成重放；不跑其余层/attention/LM head/生成")
    ok, why = P5V.commit_guard()
    print(f"[l0r] commit: {why}")
    rec["commit_guard"] = dict(ok=bool(ok), why=why)
    if not ok:
        rec["stage"] = "aborted_commit_guard"
        _save(rec, f"{TAG}_hf", ts)
        return 3
    try:
        seq, meta = _seq()
        cfg = AutoConfig.from_pretrained(TARGET, trust_remote_code=True)
        m = AutoModelForCausalLM.from_pretrained(TARGET, config=cfg, torch_dtype=torch.bfloat16,
                                                 attn_implementation="eager").to("cuda").eval()
        assert str(m.config._attn_implementation) == "eager"
        rec["input"] = dict(source_kind="manifest", **meta)
        inv = m.model.rotary_emb.inv_freq.detach().float().cpu()
        # v5 的生产 config.json **没有** `head_dim` 键（head_dim 由 hidden/heads 推出）⇒ 不能直接取属性
        head_dim_cfg = int(getattr(cfg, "head_dim", 0) or (cfg.hidden_size // cfg.num_attention_heads))
        rec["config"] = dict(rope_parameters=dict(cfg.rope_parameters),
                             rope_theta_from_config=float(cfg.rope_parameters["rope_theta"]),
                             head_dim=head_dim_cfg,
                             head_dim_from_attn_module=int(m.model.layers[0].self_attn.head_dim),
                             head_dim_key_present=bool(hasattr(cfg, "head_dim")),
                             rms_norm_eps=float(cfg.rms_norm_eps),
                             k_proj_bias=bool(m.model.layers[0].self_attn.k_proj.bias is not None),
                             attn_implementation=str(m.config._attn_implementation),
                             attention_scaling=float(getattr(m.model.rotary_emb, "attention_scaling", -1)),
                             rotary_cls=type(m.model.rotary_emb).__name__,
                             apply_rotary_pos_emb=f"{apply_rotary_pos_emb.__module__}."
                                                  f"{getattr(apply_rotary_pos_emb, '__qualname__', '?')}",
                             hub_kernels_env=os.environ.get("USE_HUB_KERNELS", "<未设置>"))
        b = torch.backends.cuda.matmul
        b.allow_bf16_reduced_precision_reduction = True
        rec["precision"] = dict(bf16_reduced_precision_reduction=bool(b.allow_bf16_reduced_precision_reduction),
                                allow_tf32=bool(b.allow_tf32),
                                fp32_precision=str(getattr(b, "fp32_precision", "<无该属性>")))
        rec["inv_freq"] = dict(shape=list(inv.shape), sha=_sha(inv), min=float(inv.min()),
                               max=float(inv.max()), values=[float(x) for x in inv])
        print(f"[l0r] hf: rope_theta={rec['config']['rope_theta_from_config']} "
              f"head_dim={rec['config']['head_dim']} scaling={rec['config']['attention_scaling']} "
              f"apply_rotary={rec['config']['apply_rotary_pos_emb']}")
        snap = torch.load(HF_PT, map_location="cpu")["c1_kv"]
        sa = m.model.layers[0].self_attn
        dump, gates, tables = {}, {}, {}
        for st, en in SCHED:
            ids = torch.tensor([seq[st:en]], device="cuda")
            pos = torch.arange(st, en, device="cuda").view(1, -1)
            with torch.inference_mode():
                emb = m.model.embed_tokens(ids)
                h = m.model.layers[0].input_layernorm(emb)
                hs = (*emb.shape[:-1], -1, sa.head_dim)
                k_pre = sa.k_proj(h).view(hs).transpose(1, 2)
                v = sa.v_proj(h).view(hs).transpose(1, 2)
                cos, sin = m.model.rotary_emb(emb, pos)
                _, k_post = apply_rotary_pos_emb(k_pre, k_pre, cos, sin)
            kref, vref = snap[0]["k"][:, :, st:en, :], snap[0]["v"][:, :, st:en, :]
            gates[st] = dict(k_bitequal=bool(torch.equal(k_post.float().cpu(), kref)),
                             v_bitequal=bool(torch.equal(v.float().cpu(), vref)),
                             k_max_abs=float((k_post.float().cpu() - kref).abs().max()),
                             v_max_abs=float((v.float().cpu() - vref).abs().max()),
                             snapshot_k_sha=_sha(kref), replay_k_sha=_sha(k_post.cpu()))
            dump[st] = dict(embed=emb.cpu(), norm=h.cpu(), k_pre=k_pre.cpu(), v=v.cpu(),
                            cos=cos.cpu(), sin=sin.cpu(), k_post=k_post.cpu(),
                            ids=ids.cpu(), positions=pos.cpu())
            tables[st] = dict(cos=_nodes_meta("cos", cos), sin=_nodes_meta("sin", sin))
            print(f"[l0r]   chunk [{st},{en}) gate: K={gates[st]['k_bitequal']} V={gates[st]['v_bitequal']} "
                  f"(K max|Δ|={gates[st]['k_max_abs']:.6g})")
        rec["gates"] = gates
        rec["tables"] = tables
        rec["nodes"] = {st: dict(embed=_nodes_meta("embed", dump[st]["embed"]),
                                 norm=_nodes_meta("norm", dump[st]["norm"]),
                                 k_pre=_nodes_meta("k_pre", dump[st]["k_pre"]),
                                 k_post=_nodes_meta("k_post", dump[st]["k_post"]),
                                 v=_nodes_meta("v", dump[st]["v"])) for st, _ in SCHED}
        gate_ok = all(g["k_bitequal"] and g["v_bitequal"] for g in gates.values())
        rec["gate_all_pass"] = bool(gate_ok)
        dp = OUTDIR / f"{TAG}_hf_dump_{ts}.pt"
        torch.save(dump, str(dp))
        rec["dump_pt"] = str(dp)
        rec["stage"] = "complete" if gate_ok else "gate_failed"
        _save(rec, f"{TAG}_hf", ts)
        print(f"[l0r] hf 复现门：{'全过' if gate_ok else '**未过**'}；dump={dp.name}")
        return 0 if gate_ok else 7
    except Exception as e:                                        # noqa: BLE001
        rec["stage"] = "crashed"
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        _save(rec, f"{TAG}_hf", ts)
        print(f"[l0r] !! 停止：{e}")
        return 2


# ------------------------------------------------------------------ 节点对照（纯 CPU）
def _latest(pat: str) -> Path:
    ps = sorted(glob.glob(str(OUTDIR / pat)))
    if not ps:
        raise FileNotFoundError(pat)
    return Path(ps[-1])


def _pick_dumps() -> tuple:
    op = Path(os.environ["L0_OURS_DUMP"]) if os.environ.get("L0_OURS_DUMP") else _latest(f"{TAG}_ours_dump_*.pt")
    hp = Path(os.environ["L0_HF_DUMP"]) if os.environ.get("L0_HF_DUMP") else _latest(f"{TAG}_hf_dump_*.pt")
    return op, hp


def _cmp_node(name: str, a: torch.Tensor, b: torch.Tensor) -> dict:
    """节点比较。⚠️ z两侧布局可能不同（HF 的 cos/sin 是 `(1,seq,dim)`，自研是 `(1,1,seq,dim)`）：
    先 `broadcast_tensors` **按语义对位**，再做逐位与差量比较；`layout_same` 单独记录，
    绝不让"形状不同"冒充"数值不同"（首版比较因此把 cos/sin 的 `bitequal` 记成 False，已更正）。"""
    x, y = a.float(), b.float()
    xb, yb = torch.broadcast_tensors(x, y)
    d = (xb - yb).abs()
    return dict(node=name, shape_a=list(a.shape), shape_b=list(b.shape), layout_same=bool(
        tuple(a.shape) == tuple(b.shape)), dtype_a=str(a.dtype), dtype_b=str(b.dtype),
                bitequal=bool(torch.equal(xb, yb)), max_abs=float(d.max()),
                mean_abs=float(d.mean()), n_diff=int((d > 0).sum()), numel=int(d.numel()),
                sha_a=_sha(a), sha_b=_sha(b))


def _run_compare(ts: str) -> int:
    op, hp = _pick_dumps()
    oj = json.load(open(_latest(f"{TAG}_ours_*.json"), encoding="utf-8"))
    hj = json.load(open(_latest(f"{TAG}_hf_*.json"), encoding="utf-8"))
    rec: dict = dict(tool="l0_k_replay", mode="compare", run_utc=ts, ours_dump=str(op), hf_dump=str(hp),
                     ours_json=os.path.basename(_latest(f"{TAG}_ours_*.json")),
                     hf_json=os.path.basename(_latest(f"{TAG}_hf_*.json")),
                     ours_stage=oj.get("stage"), hf_stage=hj.get("stage"),
                     gate=dict(ours_all_pass=oj.get("gate_all_pass"), hf_all_pass=hj.get("gate_all_pass")))
    print(f"[l0r] 节点对照：ours={op.name} hf={hp.name}")
    print(f"[l0r] 复现门：ours={rec['gate']['ours_all_pass']} hf={rec['gate']['hf_all_pass']}")
    o, h = torch.load(op, map_location="cpu"), torch.load(hp, map_location="cpu")
    nodes, rowsum = {}, {}
    for st, en in SCHED:
        nodes[st] = {nm: _cmp_node(nm, o[st][nm], h[st][nm])
                     for nm in ("embed", "norm", "k_pre", "cos", "sin", "k_post", "v")}
        print(f"\n  chunk [{st},{en})：")
        for nm in ("embed", "norm", "k_pre", "cos", "sin", "k_post", "v"):
            c = nodes[st][nm]
            print(f"    {nm:<7} 逐位相同={str(c['bitequal']):<5} max|Δ|={c['max_abs']:.6g} "
                  f"不同元素={c['n_diff']}/{c['numel']}")
    rec["nodes"] = nodes
    # ---- 结论：第一个已检查且不同的节点（按数据流顺序）----
    order = ["embed", "norm", "k_pre", "k_post"]
    first = None
    for nm in order:
        if not all(nodes[st][nm]["bitequal"] for st, _ in SCHED):
            first = nm
            break
    rec["earliest_differing_node"] = first if first else "无（embed/norm/k_pre/k_post 全部逐位相同）"
    gate_ok = bool(rec["gate"]["ours_all_pass"]) and bool(rec["gate"]["hf_all_pass"])
    kpre_same = all(nodes[st]["k_pre"]["bitequal"] for st, _ in SCHED)
    kpost_same = all(nodes[st]["k_post"]["bitequal"] for st, _ in SCHED)
    rec["k_pre_bitwise_equal"] = bool(kpre_same)
    rec["k_post_bitwise_equal"] = bool(kpost_same)
    rec["rope_substitution_allowed"] = bool(gate_ok and kpre_same and not kpost_same)
    rec["verdict"] = (
        "复现门未全过（或某侧 gate_failed）⇒ 不允许分析节点差异" if not gate_ok else
        "embed 输出已不同 ⇒ 停在该节点，不跳到 RoPE" if not all(nodes[st]["embed"]["bitequal"] for st, _ in SCHED) else
        "input RMSNorm 输出已不同 ⇒ 停在该节点，不跳到 RoPE" if not all(nodes[st]["norm"]["bitequal"] for st, _ in SCHED) else
        "RoPE 前 K 已不同 ⇒ 不能把 RoPE 表当唯一原因；记录后停止（不做 kernel 调优）"
        if not kpre_same else
        "RoPE 前 K 逐位相同、RoPE 后不同 ⇒ 允许进入条件式 RoPE 替换对照（区分表值 vs 旋转运算）"
        if not kpost_same else
        "k_post 也逐位相同 ⇒ 重放环境与被检查路径未显示差异（与快照差异并存，属未解释项）")
    print(f"\n[l0r] ★ 第一个已检查且不同的节点：{rec['earliest_differing_node']}")
    print(f"[l0r] ★ 判定：{rec['verdict']}")
    print(f"[l0r] ★ RoPE 替换对照是否允许：{rec['rope_substitution_allowed']}")
    _save(rec, f"{TAG}_compare", ts)
    return 0


# ------------------------------------------------------------------ 条件式 RoPE 表替换
def _run_subst(ts: str) -> int:
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    from . import model_self as M
    rec: dict = dict(tool="l0_k_replay", mode="subst", run_utc=ts,
                     semantics="固定同一份 RoPE 前 K + 同一旋转实现，只替换 cos/sin（A=自研表，B=HF 表）；"
                               "不叠加任何其它改动；不加载任何模型权重")
    op, hp = _pick_dumps()
    oj = json.load(open(_latest(f"{TAG}_ours_*.json"), encoding="utf-8"))
    hj = json.load(open(_latest(f"{TAG}_hf_*.json"), encoding="utf-8"))
    cj = json.load(open(_latest(f"{TAG}_compare_*.json"), encoding="utf-8"))
    rec["inputs"] = dict(ours_dump=str(op), hf_dump=str(hp), compare_json=os.path.basename(
        _latest(f"{TAG}_compare_*.json")))
    # ---- 条件检查（自己拒绝，不靠调用者）----
    cond = dict(ours_gate=bool(oj.get("gate_all_pass")), hf_gate=bool(hj.get("gate_all_pass")),
                k_pre_bitwise_equal=bool(cj.get("k_pre_bitwise_equal")),
                k_post_differ=not bool(cj.get("k_post_bitwise_equal")),
                compare_allows=bool(cj.get("rope_substitution_allowed")))
    rec["conditions"] = cond
    print(f"[l0r] 替换对照的前置条件：{json.dumps(cond, ensure_ascii=False)}")
    if not all(cond.values()):
        rec["stage"] = "refused"
        rec["reason"] = "前置条件不满足（复现门未过 / RoPE 前 K 非逐位相同 / k_post 无差异）⇒ 不做替换对照"
        _save(rec, f"{TAG}_subst", ts)
        print("[l0r] !! 拒绝执行替换对照")
        return 6
    o, h = torch.load(op, map_location="cpu"), torch.load(hp, map_location="cpu")
    ko_all = torch.load(OURS_PT, map_location="cpu")["c1_kv"][0]["k"]
    kh_all = torch.load(HF_PT, map_location="cpu")["c1_kv"][0]["k"]
    out = {}
    tot = dict(cos_diff=0, sin_diff=0, cos_entries=0, sin_entries=0)
    idx_cos, idx_sin = [], []
    for st, en in SCHED:
        k_pre = o[st]["k_pre"].to("cuda")                       # 同一份 RoPE 前 K（两侧逐位相同）
        cO, sO = o[st]["cos"].to("cuda"), o[st]["sin"].to("cuda")     # 自研原生布局 (1,1,seq,128)
        cH, sH = h[st]["cos"].to("cuda"), h[st]["sin"].to("cuda")     # HF 原生布局 (1,seq,128)
        cO2, cH2 = cO.reshape(-1, cO.shape[-1]), cH.reshape(-1, cH.shape[-1])
        sO2, sH2 = sO.reshape(-1, sO.shape[-1]), sH.reshape(-1, sH.shape[-1])
        with torch.inference_mode():
            # (0) 表本身：按**同一形状**逐位比较 + 差异落在哪个频率下标（dim % 64）
            tb = dict(shape_ours=list(cO.shape), shape_hf=list(cH.shape),
                      cos_bitequal=bool(torch.equal(cO2, cH2)), sin_bitequal=bool(torch.equal(sO2, sH2)),
                      cos_n_diff=int((cO2 != cH2).sum()), sin_n_diff=int((sO2 != sH2).sum()),
                      cos_numel=int(cO2.numel()), sin_numel=int(sO2.numel()))
            dj = torch.nonzero(cO2 != cH2, as_tuple=False)   # (pos, dim)
            sj = torch.nonzero(sO2 != sH2, as_tuple=False)
            tb["cos_diff_by_freq_index"] = {int(i): int((dj[:, 1] % 64 == i).sum())
                                            for i in sorted({int(x) for x in dj[:, 1] % 64})}
            tb["sin_diff_by_freq_index"] = {int(i): int((sj[:, 1] % 64 == i).sum())
                                            for i in sorted({int(x) for x in sj[:, 1] % 64})}
            for r in dj.tolist()[:5000]:
                idx_cos.append([st, int(r[0]), int(r[1]), float(cO2[r[0], r[1]]), float(cH2[r[0], r[1]])])
            for r in sj.tolist()[:5000]:
                idx_sin.append([st, int(r[0]), int(r[1]), float(sO2[r[0], r[1]]), float(sH2[r[0], r[1]])])
            # (1) 旋转实现语义：同一 K + **同一张表的值**（各按各的原生布局）⇒ 两侧旋转函数输出是否逐位相同
            kA_eng = M.apply_rope(k_pre, cO, sO)
            _, kA_hf = apply_rotary_pos_emb(k_pre, k_pre, cO2.unsqueeze(0), sO2.unsqueeze(0))
            sem = dict(bitequal=bool(torch.equal(kA_eng, kA_hf)),
                       shape_ours=list(kA_eng.shape), shape_hf=list(kA_hf.shape),
                       max_abs=float((kA_eng.float() - kA_hf.float()).abs().max()),
                       note="同一张表（自研表的值）分别喂给两侧旋转实现，各自原生布局")
            # (2) A/B：同一份 K、同一旋转实现（自研 apply_rope），只换表
            kB_eng = M.apply_rope(k_pre, cH, sH)
            ab = dict(ab_bitequal=bool(torch.equal(kA_eng, kB_eng)),
                      n_diff=int((kA_eng != kB_eng).sum()),
                      max_abs=float((kA_eng.float() - kB_eng.float()).abs().max()))
            # (3) 逐元素解释既有差异：A 是否等于自研快照、B 是否等于 HF 快照
            ko = ko_all[:, :, st:en, :]
            kh = kh_all[:, :, st:en, :]
            expl = dict(A_equals_ours_snapshot=bool(torch.equal(kA_eng.float().cpu(), ko)),
                        B_equals_hf_snapshot=bool(torch.equal(kB_eng.float().cpu(), kh)),
                        A_equals_hf_snapshot=bool(torch.equal(kA_eng.float().cpu(), kh)),
                        B_equals_ours_snapshot=bool(torch.equal(kB_eng.float().cpu(), ko)),
                        snapshot_diff_n=int((ko != kh).sum()),
                        ab_diff_n=int((kA_eng != kB_eng).sum()),
                        same_diff_positions=bool(torch.equal((ko != kh).cpu(), (kA_eng != kB_eng).cpu())))
        out[st] = dict(table=tb, rotation_semantics=sem, ab=ab, explains=expl)
        tot["cos_diff"] += tb["cos_n_diff"]
        tot["sin_diff"] += tb["sin_n_diff"]
        tot["cos_entries"] += tb["cos_numel"]
        tot["sin_entries"] += tb["sin_numel"]
        print(f"[l0r]   chunk [{st},{en})：表 cos 不同 {tb['cos_n_diff']}/{tb['cos_numel']}、"
              f"sin 不同 {tb['sin_n_diff']}/{tb['sin_numel']}；"
              f"旋转实现语义一致={sem['bitequal']}；A/B 不同={ab['n_diff']}；"
              f"A==ours快照 {expl['A_equals_ours_snapshot']}、B==hf快照 {expl['B_equals_hf_snapshot']}、"
              f"差异位置集合相同={expl['same_diff_positions']}")
    rec["chunks"] = out
    rec["totals"] = tot
    rec["table_diffs"] = dict(cos_sample=idx_cos[:200], sin_sample=idx_sin[:200],
                              cos_n_listed=len(idx_cos), sin_n_listed=len(idx_sin))
    allok = all(out[st]["explains"]["A_equals_ours_snapshot"] and out[st]["explains"]["B_equals_hf_snapshot"]
                and out[st]["explains"]["same_diff_positions"] for st, _ in SCHED)
    rec["explains_all_existing_diffs"] = bool(allok)
    rec["stage"] = "complete"
    print(f"\n[l0r] ★ 只换 cos/sin 是否**逐元素**解释既有 60 个差异：{allok}")
    _save(rec, f"{TAG}_subst", ts)
    return 0


def _run_table(ts: str) -> int:
    """**表构造诊断**（不是重放、不是替换对照、不加载任何权重、不做任何前向）。

    只做一件事：把两侧**实际构造 cos/sin 的算式**各自在**它们实际使用的设备**上跑一遍，
    逐位比较 `inv_freq` / `freqs` / `cos,sin(fp32)` / `cos,sin(bf16)`，并给出差异所在的频率下标。
    输入只有**真实 config 里的** theta/head_dim（不手工猜），以及既有 dump 里已保存的表（用于核对重建是否等价）。

    为什么不算超限：不含模型前向、不含 K、不含快照对照、不改任何精度旋钮、不引入新 family/长度/生成。
    """
    import json as _json
    rec: dict = dict(tool="l0_k_replay", mode="table", run_utc=ts,
                     semantics="表构造诊断：两侧算式在实际设备上重算并逐位比较；无模型、无前向、无 K")
    hj = _json.load(open(_latest(f"{TAG}_hf_*.json"), encoding="utf-8"))
    oj = _json.load(open(_latest(f"{TAG}_ours_*.json"), encoding="utf-8"))
    theta_hf = float(hj["config"]["rope_theta_from_config"])
    theta_ours = float(oj["config"]["rope_theta_from_model"])
    dim_hf = int(hj["config"]["head_dim"])
    dim_ours = int(oj["config"]["head_dim_from_model"])
    rec["params"] = dict(theta_hf=theta_hf, theta_ours=theta_ours, dim_hf=dim_hf, dim_ours=dim_ours,
                         note="全部来自真实 config / 模型实例；两侧 theta 一致才继续")
    print(f"[l0r] table: theta hf={theta_hf} ours={theta_ours}；head_dim hf={dim_hf} ours={dim_ours}")
    if theta_hf != theta_ours or dim_hf != dim_ours:
        rec["stage"] = "refused"
        rec["reason"] = "两侧 theta/head_dim 不一致 ⇒ 拒绝比较（不猜参数）"
        _save(rec, f"{TAG}_table", ts)
        return 6

    def _diff(name, a, b):
        m = (a != b)
        d = (a.float() - b.float()).abs()
        return dict(name=name, shape_a=list(a.shape), shape_b=list(b.shape),
                    bitequal=bool(torch.equal(a, b)), n_diff=int(m.sum()), numel=int(a.numel()),
                    max_abs=float(d.max()),
                    idx=[list(x) for x in torch.nonzero(m, as_tuple=False).tolist()][:64],
                    ours=[float(x) for x in a.float()[m][:16].tolist()],
                    hf=[float(x) for x in b.float()[m][:16].tolist()])

    half = dim_hf // 2
    exp_cpu = torch.arange(0, dim_hf, 2, dtype=torch.float32) / dim_hf          # HF: 在 CPU 上算
    exp_gpu = torch.arange(0, dim_hf, 2, device="cuda", dtype=torch.float32) / dim_hf   # 自研: 在 GPU 上算
    with torch.inference_mode():
        inv_hf = (1.0 / (theta_hf ** exp_cpu))                     # HF: CPU pow + CPU 除法（留在 CPU）
        inv_ours = (1.0 / (theta_ours ** exp_gpu))                 # 自研: GPU pow + GPU 除法
        rec["inv_freq"] = _diff("inv_freq", inv_ours, inv_hf.to("cuda"))
        print(f"[l0r] inv_freq（GPU 算 vs CPU 算）：逐位相同={rec['inv_freq']['bitequal']} "
              f"不同下标={rec['inv_freq']['idx']}")
        if rec["inv_freq"]["idx"]:
            for i in rec["inv_freq"]["idx"][:4]:
                print(f"         idx={i}: gpu={float(inv_ours[i]):.17g} cpu={float(inv_hf[i]):.17g}")
        pos = torch.arange(0, CTX, device="cuda", dtype=torch.float32).view(1, -1)
        # HF 的写法（按源码逐句复刻）：inv_freq[None,:,None].expand(1,-1,1) @ position_ids[:,None,:] → (1,64,8192) → transpose
        ib = inv_hf.to("cuda")[None, :, None].expand(1, half, 1)
        ia = inv_ours[None, :, None].expand(1, half, 1)
        with torch.autocast("cuda", enabled=False):               # HF 的 forward 里就是这么包的
            f_mat_gpu = (ia @ pos[:, None, :]).transpose(1, 2)     # matmul + GPU 侧 inv_freq
            f_mat_cpu = (ib @ pos[:, None, :]).transpose(1, 2)     # matmul + CPU 侧 inv_freq（= HF 实况）
        f_out_gpu = torch.outer(pos.reshape(-1), inv_ours).unsqueeze(0)   # outer（自研的写法）
        f_out_cpu = torch.outer(pos.reshape(-1), inv_hf.to("cuda")).unsqueeze(0)
        rec["freqs"] = dict(matmul_gpuinv_vs_outer_gpuinv=_diff("matmul_vs_outer", f_mat_gpu, f_out_gpu),
                            matmul_cpuinv_vs_outer_cpuinv=_diff("matmul_vs_outer_cpuinv", f_mat_cpu, f_out_cpu),
                            hf_chain_vs_ours_chain=_diff("hf_chain_vs_ours_chain", f_out_gpu, f_mat_cpu))
        print(f"[l0r] freqs：matmul vs outer（同一 inv_freq）逐位相同="
              f"{rec['freqs']['matmul_gpuinv_vs_outer_gpuinv']['bitequal']}；"
              f"HF 链 vs 自研链 逐位相同={rec['freqs']['hf_chain_vs_ours_chain']['bitequal']} "
              f"不同下标={rec['freqs']['hf_chain_vs_ours_chain']['idx'][:6]}")

        def _table(freqs, ours_layout):
            emb = torch.cat((freqs, freqs), dim=-1)
            c, s = emb.cos(), emb.sin()                            # fp32（HF 再乘 attention_scaling=1.0，乘 1.0 精确）
            if ours_layout:                                        # 自研：unsqueeze 成 (1,1,seq,dim)
                c, s = c.unsqueeze(0).unsqueeze(0), s.unsqueeze(0).unsqueeze(0)
            else:                                                  # HF：保持 (1,seq,dim)
                c, s = c.unsqueeze(0), s.unsqueeze(0)
            return c.to(torch.bfloat16), s.to(torch.bfloat16)

        cA, sA = _table(f_out_gpu, True)      # 自研链（GPU pow + outer）
        cB, sB = _table(f_mat_cpu, False)     # HF 链（CPU pow + GPU matmul）
        cA2, sA2 = cA.reshape(-1, dim_hf), sA.reshape(-1, dim_hf)
        cB2, sB2 = cB.reshape(-1, dim_hf), sB.reshape(-1, dim_hf)
        rec["tables"] = dict(cos=_diff("cos_bf16", cA2, cB2), sin=_diff("sin_bf16", sA2, sB2),
                             cos_by_freq_index={}, sin_by_freq_index={})
        for nm, o2, h2 in (("cos", cA2, cB2), ("sin", sA2, sB2)):
            nz = torch.nonzero(o2 != h2, as_tuple=False)
            rec["tables"][f"{nm}_by_freq_index"] = {
                int(i): int((nz[:, 1] % half == i).sum()) for i in sorted({int(x) for x in nz[:, 1] % half})}
        print(f"[l0r] 重建表（全 8192 位置）：cos 不同={rec['tables']['cos']['n_diff']} 个，"
              f"按频率下标={rec['tables']['cos_by_freq_index']}；sin 不同={rec['tables']['sin']['n_diff']} 个，"
              f"按频率下标={rec['tables']['sin_by_freq_index']}")
        # 与既有 dump 里**实际使用**的表核对（重建是否等价）
        op, hp = _pick_dumps()
        o, h = torch.load(op, map_location="cpu"), torch.load(hp, map_location="cpu")
        vv = {}
        for st, en in SCHED:
            vv[st] = dict(
                recon_ours_equals_dump=bool(torch.equal(
                    cA2[st:en], o[st]["cos"].reshape(-1, dim_hf).to("cuda"))),
                recon_hf_equals_dump=bool(torch.equal(
                    cB2[st:en], h[st]["cos"].reshape(-1, dim_hf).to("cuda"))),
                recon_ours_sin_equals_dump=bool(torch.equal(
                    sA2[st:en], o[st]["sin"].reshape(-1, dim_hf).to("cuda"))),
                recon_hf_sin_equals_dump=bool(torch.equal(
                    sB2[st:en], h[st]["sin"].reshape(-1, dim_hf).to("cuda"))))
            print(f"[l0r]   与 dump 核对 chunk [{st},{en})：重建自研表==dump "
                  f"{vv[st]['recon_ours_equals_dump']}/{vv[st]['recon_ours_sin_equals_dump']}、"
                  f"重建 HF 表==dump {vv[st]['recon_hf_equals_dump']}/{vv[st]['recon_hf_sin_equals_dump']}"
                  f"（cos/sin）")
        rec["recon_vs_dump"] = vv
    rec["stage"] = "complete"
    _save(rec, f"{TAG}_table", ts)
    return 0


def main() -> int:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    torch.manual_seed(0)
    if MODE == "ours":
        return _run_ours(ts)
    if MODE == "hf":
        return _run_hf(ts)
    if MODE == "compare":
        return _run_compare(ts)
    if MODE == "subst":
        return _run_subst(ts)
    if MODE == "table":
        return _run_table(ts)
    print(f"[l0r] !! 非法 L0_MODE={MODE!r}（可选 ours|hf|compare|subst|table）")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
