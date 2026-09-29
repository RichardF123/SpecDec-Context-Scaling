"""layerwise_f06_i25.py — F06/base_bf16/index25 的**逐层**对照：块路径(s=4) vs 串行路径(s=1)。

边界（对应任务）
  * 只研究这一个案例；**不**重复 16 次生成、**不**做全矩阵；
  * **不改**精度/模型/KV/backend/argmax 规则/验收门槛；**不**启用 QK32/prob32；**不**加补偿或容差逻辑；
  * **不**改写 `forward`、**不**换计算表达式：只用**只读**的 `nn.Module.register_forward_hook`
    与（函数式操作的）**只读** `torch.softmax` 包装；不改变任何张量的 shape/dtype/计算顺序；
  * 不通过缩短 max_new 改变路径：本工具**不跑生成**，直接在上一轮保存的**真实 round-11 cache**
    （长度 8215）上重放，块长度=真实 4、候选=真实 `[27685,659,715,4882]`。

两个对照臂（同一起始 cache，独立深拷贝）
  A 真实块验证：`spec_dec._verify(tgt, clone(cache), cands)` → 取**输入位置 8216** 的那一行（rows[1]）
  B 串行路径：`_step(tgt,27685,·)` → cache 8216；再 `_step(tgt,659,·)` → s=1 那一行（位置 8216）

对齐口径（§五）
  两侧都在**输入位置 8216 / 预测位置 8217**；attention 只在**共同有效 key 范围 [0, 8216]** 上比数值，
  块路径被 mask 的未来位置（key 8217–8218）单独报告其屏蔽情况（不参与数值比较）。
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import torch

from .. import results_schema as S
from . import diag_cache_cmp as CC
from . import diag_fields as DF
from . import model_self as M
from . import spec_dec
from . import g5_8k_spec_variance_probe as P5V

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
SRC_TAG = "f06i25b_20260929T130000Z"
FAMILY = "F06"
ROUND = 11
CAND0, FED = 27685, 659
KEY_RANGE_END = 8216          # 两侧共同有效的最后一个 key 位置（含）
ALIGN_ROW_BLOCK = 1           # 块路径中对齐到输入位置 8216 的行
EXPECT_AR, EXPECT_SPEC = 715, 576
OPS = ("input_layernorm", "q_proj", "k_proj", "v_proj", "o_proj",
       "post_attention_layernorm", "gate_proj", "up_proj", "down_proj")


def _newest(pat):
    return Path(sorted(glob.glob(str(OUTDIR / pat)))[-1])


class Tap:
    """只读观察：记录模块的 (输入, 输出) 与 softmax 的 (输入=分数, 输出=概率)。"""

    def __init__(self, row_idx: int):
        self.row = row_idx
        self.mod: dict = {}
        self.sm: list = []
        self.handles: list = []
        self._orig_softmax = None
        self._in_softmax = False

    def _mk(self, name):
        def hook(module, inputs, output):
            if isinstance(output, torch.Tensor) and output.dim() == 3:
                o = output[0, self.row].detach().float().cpu()
                i = (inputs[0][0, self.row].detach().float().cpu()
                     if isinstance(inputs[0], torch.Tensor) and inputs[0].dim() == 3 else None)
                self.mod[name] = dict(input=i, output=o)
        return hook

    def install(self, tgt):
        self.handles.append(tgt.embed_tokens.register_forward_hook(self._mk("embed_tokens")))
        for li, layer in enumerate(tgt.layers):
            for op in OPS:
                mod = (layer.self_attn if op in ("q_proj", "k_proj", "v_proj", "o_proj")
                       else layer.mlp if op in ("gate_proj", "up_proj", "down_proj")
                       else layer)
                self.handles.append(mod.__getattr__(op).register_forward_hook(
                    self._mk(f"L{li}.{op}")))
        self.handles.append(tgt.norm.register_forward_hook(self._mk("final_norm")))

        # ---- 函数式操作：只读包装 torch.softmax（attention 的分数→概率）----
        # eager 路径的 scores 已含 1/sqrt(d) 缩放与 mask；softmax 输入即"缩放+屏蔽后的分数"。
        self._orig_softmax = torch.softmax

        def _softmax(inp, *a, **k):
            out = self._orig_softmax(inp, *a, **k)
            try:
                if (not self._in_softmax and isinstance(inp, torch.Tensor)
                        and inp.dim() == 4 and inp.shape[-1] > 1):
                    self._in_softmax = True
                    tt = min(int(inp.shape[-1]), KEY_RANGE_END + 1)
                    self.sm.append(dict(
                        scores=inp[0, :, self.row, :tt].detach().float().cpu(),
                        probs=out[0, :, self.row, :tt].detach().float().cpu(),
                        T=int(inp.shape[-1]), s=int(inp.shape[2]),
                        masked_tail_all_neg_inf=bool(
                            torch.isneginf(inp[0, :, self.row, tt:]).all())
                        if inp.shape[-1] > tt else None))
            except Exception:                                    # noqa: BLE001
                pass
            finally:
                self._in_softmax = False
            return out

        torch.softmax = _softmax

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        if self._orig_softmax is not None:
            torch.softmax = self._orig_softmax


def _cmp_tensor(a, b, name: str, *, prefix: str = "") -> dict:
    if a is None or b is None:
        return dict(name=name, comparable=False, reason="有一侧缺该观察点")
    if tuple(a.shape) != tuple(b.shape):
        return dict(name=name, comparable=False,
                    reason=f"shape 不同 {tuple(a.shape)} vs {tuple(b.shape)}")
    d = (a - b).abs()
    mx = float(d.max())
    flat = int(d.reshape(-1).argmax())
    idx = [int(x) for x in torch.unravel_index(torch.tensor(flat), d.shape)]
    return dict(name=name, comparable=True, shape=[int(x) for x in a.shape],
                dtype=str(a.dtype), elementwise_equal=bool(torch.equal(a, b)),
                abs_err_max=mx, abs_err_mean=float(d.mean()),
                max_err_index=idx, a_at_max=float(a[tuple(idx)]), b_at_max=float(b[tuple(idx)]),
                a_absmax=float(a.abs().max()), b_absmax=float(b.abs().max()))


def _run_arm(tgt, snap_cpu, arm: str):
    """执行一个对照臂并返回观察集合。"""
    def to_gpu(c):
        return [{k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in d.items()}
                for d in c]
    tap = Tap(ALIGN_ROW_BLOCK if arm == "block" else 0)
    tap.install(tgt)
    try:
        with torch.inference_mode():
            if arm == "block":
                work = to_gpu(snap_cpu)
                with M.attention_phase("verify"):
                    rows, _blk = spec_dec._verify(tgt, work, [CAND0, FED, 715, 4882])
                logits = rows[ALIGN_ROW_BLOCK].detach().float().cpu()
                n_calls = 1
            else:
                work = to_gpu(snap_cpu)
                with M.attention_phase("repair"):
                    spec_dec._step(tgt, CAND0, work)
                # ⚠️ 第二步前清空 softmax 捕获：否则会累积 2 步 × 28 层 = 56 条，
                #    与块的 28 条 zip 起来会错位到**第一步**（位置 8215、T=8216），
                #    变成"形状不同 ⇒ non-comparable"，而首版把 non-comparable 误印成"相同"。
                tap.sm.clear()
                with M.attention_phase("repair"):
                    _lg, _c = spec_dec._step(tgt, FED, work)
                logits = _lg[0, -1].detach().float().cpu()
                n_calls = 2
    finally:
        tap.remove()
    return dict(logits=logits, mod=tap.mod, softmax=tap.sm, n_calls=n_calls)


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="layerwise_f06_i25", tag=SRC_TAG,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               config=dict(family=FAMILY, kv="base_bf16", round=ROUND, cands=[CAND0, FED, 715, 4882],
                           align_input_position=KEY_RANGE_END, align_predicted_position=8217,
                           block_row_index=ALIGN_ROW_BLOCK,
                           key_range_compared=[0, KEY_RANGE_END],
                           source_cache=str(_newest(f"{SRC_TAG}_{FAMILY}_spec_*.cache.pt"))),
               scope="逐层定位 F06/index25 的块 vs 串行差异；只读插桩；不改精度/算法/门槛；无性能结论。")
    print(f"[lw] run_id={run_id}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    ok, why = P5V.commit_guard()
    print(f"[lw] commit: {why}")
    if not ok:
        rec["stage"] = "refused"
        (OUTDIR / f"{SRC_TAG}_layerwise_{run_id}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return 3

    rec["diag_precision"] = M.set_diag_preset("eager")     # QK32/prob32 关闭
    from .lowmem_load import load_ours_bf16
    tgt, tok, _ti = load_ours_bf16(P5V.TARGET)
    tgt.set_kv_format(spec_dec._fmt("base_bf16"))
    M.set_attention_strategy("cudnn_prefill")              # 与生产一致；本工具只跑 eager 路径(verify/repair)

    saved = torch.load(_newest(f"{SRC_TAG}_{FAMILY}_spec_*.cache.pt"), map_location="cpu")
    snap = saved[f"round{ROUND}_snap_pre"]
    real_rows = torch.load(_newest(f"{SRC_TAG}_{FAMILY}_spec_*.rows.pt"), map_location="cpu")
    real_row = real_rows[f"round{ROUND}_rows"][ALIGN_ROW_BLOCK].float()
    rec["cache_len"] = int(CC.cache_len(snap))
    rec["cache_layers_all_same_len"] = len({int(d["k"].shape[2]) for d in snap}) == 1
    print(f"[lw] cache len={rec['cache_len']} all_same_len={rec['cache_layers_all_same_len']}")

    # ---- 无插桩基线（§三）：先做一次不带 hook 的重放 ----
    def to_gpu(c):
        return [{k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in d.items()}
                for d in c]
    with torch.inference_mode(), M.attention_phase("verify"):
        rows0, _b0 = spec_dec._verify(tgt, to_gpu(snap), [CAND0, FED, 715, 4882])
    plain_block = rows0[ALIGN_ROW_BLOCK].detach().float().cpu()
    rec["instrumentation_check"] = dict(
        plain_block_vs_saved_real=_cmp_tensor(plain_block, real_row, "plain_block_vs_saved_real"),
        note="无插桩重放必须与上一轮保存的真实决策行逐元素相同")

    A = _run_arm(tgt, snap, "block")
    B = _run_arm(tgt, snap, "single")
    rec["instrumentation_check"]["instrumented_block_vs_saved_real"] = _cmp_tensor(
        A["logits"], real_row, "instrumented_block_vs_saved_real")
    rec["instrumentation_check"]["instrumented_block_vs_plain"] = _cmp_tensor(
        A["logits"], plain_block, "instrumented_block_vs_plain")

    # ---- 结果层 ----
    la, lb = DF.row_fields(A["logits"], watch=[EXPECT_AR, EXPECT_SPEC, 4710]), \
        DF.row_fields(B["logits"], watch=[EXPECT_AR, EXPECT_SPEC, 4710])
    rec["result"] = dict(
        block=dict(**{k: la[k] for k in ("argmax_id", "max_score", "n_max_tied", "tied_max_tokens")},
                   note="期望 argmax=576（三并列）"),
        single=dict(**{k: lb[k] for k in ("argmax_id", "max_score", "n_max_tied", "tied_max_tokens")},
                    note="期望 argmax=4710（无并列）"),
        logits_cmp=_cmp_tensor(A["logits"], B["logits"], "final_logits_row"),
        reproduces=dict(block_576=(la["argmax_id"] == EXPECT_SPEC),
                        single_4710=(lb["argmax_id"] == 4710)),
        key_candidates={str(t): dict(block=float(A["logits"][t]), single=float(B["logits"][t]),
                                     delta=float(A["logits"][t] - B["logits"][t]))
                        for t in (EXPECT_AR, EXPECT_SPEC, 4710, 4882)})

    # ---- 逐层对照 ----
    rows = []
    for key in sorted(A["mod"], key=lambda s: (0 if s == "embed_tokens" else
                                               1 if s == "final_norm" else
                                               2 + int(s[1:].split(".")[0]), s)):
        if key not in B["mod"]:
            continue
        ca = A["mod"][key]
        cb = B["mod"][key]
        inp = _cmp_tensor(ca["input"], cb["input"], "input")
        out = _cmp_tensor(ca["output"], cb["output"], "output")
        rows.append(dict(where=key, input=inp, output=out))
    # attention 分数/概率（只比共同有效 key 范围）
    sm = []
    for i, (sa, sb) in enumerate(zip(A["softmax"], B["softmax"])):
        sm.append(dict(layer=i,
                       scores=_cmp_tensor(sa["scores"], sb["scores"], f"L{i}.scores(keys<={KEY_RANGE_END})"),
                       probs=_cmp_tensor(sa["probs"], sb["probs"], f"L{i}.probs"),
                       block_T=sa["T"], single_T=sb["T"],
                       block_masked_tail_all_neg_inf=sa["masked_tail_all_neg_inf"]))
    rec["per_layer"] = rows
    rec["attention_scores_probs"] = sm
    first_diff = [r["where"] for r in rows if r["output"]["comparable"]
                  and not r["output"]["elementwise_equal"]]
    sm_cmp = [s for s in sm if s["scores"]["comparable"]]
    sm_noncmp = [s["layer"] for s in sm if not s["scores"]["comparable"]]
    first_sm = [s["layer"] for s in sm_cmp if not s["scores"]["elementwise_equal"]]
    sm_equal = [s["layer"] for s in sm_cmp if s["scores"]["elementwise_equal"]]
    rec["summary"] = dict(
        n_layers_with_module_diff=len(first_diff), first_diff_layers=first_diff[:6],
        n_layers_with_score_diff=len(first_sm), first_score_diff_layers=first_sm[:6],
        n_score_pairs_comparable=len(sm_cmp), n_score_pairs_noncomparable=sm_noncmp,
        n_score_pairs_equal=len(sm_equal), first_score_equal_layers=sm_equal[:6],
        note="只报告**首个**出现差异的层与操作；首个非零差异不等于 bug，也不等于主要根因")
    rec["stage"] = "complete"
    p = OUTDIR / f"{SRC_TAG}_layerwise_{run_id}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    print("== 插桩检查（§三） ==")
    for k in ("plain_block_vs_saved_real", "instrumented_block_vs_saved_real",
              "instrumented_block_vs_plain"):
        c = rec["instrumentation_check"][k]
        print(f"  {k}: elementwise_equal={c.get('elementwise_equal')} max|Δ|={c.get('abs_err_max')}")
    print("== 结果层 ==")
    print(f"  block : argmax={la['argmax_id']} tied={la['n_max_tied']}{la['tied_max_tokens']}")
    print(f"  single: argmax={lb['argmax_id']} tied={lb['n_max_tied']}{lb['tied_max_tokens']}")
    print(f"  logits: elementwise_equal={rec['result']['logits_cmp']['elementwise_equal']} "
          f"max|Δ|={rec['result']['logits_cmp']['abs_err_max']}")
    print("== 逐层（只列有差异的模块输出） ==")
    shown = 0
    for r in rows:
        o = r["output"]
        if o["comparable"] and not o["elementwise_equal"]:
            print(f"  {r['where']:28s} out max|Δ|={o['abs_err_max']:.6g} "
                  f"in_equal={r['input'].get('elementwise_equal')} idx={o['max_err_index']}")
            shown += 1
            if shown >= 14:
                print("  ...")
                break
    if shown == 0:
        print("  （所有模块输出逐元素相同）")
    print("== attention 分数/概率（共同有效 key 范围） ==")
    shown = 0
    for s in sm:
        if s["scores"]["comparable"] and not s["scores"]["elementwise_equal"]:
            print(f"  L{s['layer']:02d} scores max|Δ|={s['scores']['abs_err_max']:.6g} "
                  f"probs max|Δ|={s['probs']['abs_err_max']:.6g} "
                  f"block_T={s['block_T']} single_T={s['single_T']} "
                  f"masked_tail_all_-inf={s['block_masked_tail_all_neg_inf']}")
            shown += 1
            if shown >= 6:
                print("  ...")
                break
    if shown == 0:
        if sm_cmp:
            print(f"  （{len(sm_cmp)} 层**可比较且逐元素相同**；"
                  f"不可比较 {len(sm_noncmp)} 层：{[s['layer'] for s in sm if not s['scores']['comparable']][:6]}）")
        else:
            print(f"  （**全部不可比较**：{len(sm_noncmp)} 层 —— 形状/范围未对齐，不得当作「相同」）")
    print(f"[lw] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
