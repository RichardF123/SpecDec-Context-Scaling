"""attn_l27_g012_f06_i25.py — L27 attention 最小对照实验：**G0 / G1 / G2**（F06/index25）。

授权范围（本轮）
  * **只**做 G0（原形状重放门）、G1（只改查询行数）、G2（只改被 mask 的 K/V 长度）；
  * **G3 暂缓**（不做 K/V 来源替换、不混入 AR 历史 cache）；**不**扫描所有层；**不**跑完整生成；
  * **不**修改生产算法/精度/KV 格式/argmax/验收门槛；QK32/prob32 保持关闭。

数据流
  阶段 A（补采，一次完整 forward，**只读观察**）：
    `round11_snap_pre`(8215) 的 **deep clone** → `spec_dec._verify(tgt, clone, CANDS)`（生产块路径，phase='verify'）
    只读捕获 L27 的：**RoPE 后 q**(1,12,4,128)（用 `apply_rope` 的只读包装 + **层作用域标志**限定在 L27）、
    **attention 核心输出**（= `o_proj` 的输入，AV 后）、`o_proj` 输出、`input_layernorm` 输出；
    forward 后 `cache[27]["k"]/["v"]` 即本次实际使用的 `k_all/v_all`(8219)（生产会原地写回）。
  阶段 B（局部重放，纯张量，不再调模型 forward）：
    G0b 功能式 **四行**重放必须逐元素等于 G0a 捕获的核心输出（**门**，不过则禁止归因）；
    G1 只把查询行数 4→1（同一 Q 的对齐行、同一 K/V、T=8219、**显式**使用与块行1逐元素相同的 mask）；
    G2 只把 T 8219→8217（纯切片，有效范围仍 keys 0..8216）；
    G4' 各条件对齐核心输出统一成 `(1,1,1536)` 后调用**同一个** `o_proj`；块形状 `o_proj` 仅作**生产重放校验**并单独标注。

必需检查一律用 `diag_checks.require` / 显式断言；失败即落盘并停止。
"""
from __future__ import annotations

import glob
import json
import time
import traceback
from pathlib import Path

import torch

from .. import results_schema as S
from . import attn_l27_core as C
from . import diag_checks as DC
from . import diag_fields as DF
from . import model_self as M
from . import spec_dec
from . import g5_8k_spec_variance_probe as P5V

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TAG = "f06i25b_20260929T130000Z"
FAMILY = "F06"
L = 27
ROUND = 11
PRE = 8215
CAND0, FED = 27685, 659
CANDS = [CAND0, FED, 715, 4882]
ALIGN_ROW = 1
EXPECT_BLOCK = 576
T_BLK = 8219
T_TRIM = 8217


def _newest(pat):
    return Path(sorted(glob.glob(str(OUTDIR / pat)))[-1])


def _to_gpu(c):
    return [{k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in d.items()} for d in c]


def _deep_clone_cpu(c):
    return [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in d.items()} for d in c]


def _fields(row):
    r = row.detach().float().reshape(-1)
    f = DF.row_fields(r, watch=[715, 576, 4710])
    t2 = torch.topk(r, 2).values
    return dict(argmax_id=f["argmax_id"], max_score=f["max_score"], n_max_tied=f["n_max_tied"],
                tied_max_tokens=f["tied_max_tokens"], margin_top1_top2=float(t2[0] - t2[1]),
                scores={str(t): float(r[t]) for t in (715, 576, 4710)})


def _row_cmp(a, b, name):
    """对齐行比较：**先断言形状**（禁止广播），再逐元素相等与绝对误差统计。"""
    DC.assert_same_shape(a, b, name)
    x, y = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    d = (x - y).abs()
    i = int(torch.argmax(d))
    return dict(name=name, shape=[int(z) for z in a.shape],
                elementwise_equal=bool(torch.equal(a.detach(), b.detach())),
                abs_err_max=float(d.max()), abs_err_mean=float(d.mean()),
                n_diff=int((d > 0).sum()), max_err_index=i,
                a_at_max=float(x[i]), b_at_max=float(y[i]),
                finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all()))


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="attn_l27_g012_f06_i25",
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               config=dict(family=FAMILY, kv="base_bf16", layer=L, round=ROUND, div_index=25,
                           pre=PRE, cands=CANDS, align_row=ALIGN_ROW,
                           align_input_position=8216, predicted_position=8217,
                           align_visible_keys="0..8216（8217 个）", expect_block=EXPECT_BLOCK,
                           T_block=T_BLK, T_trim=T_TRIM,
                           groups="G0a 生产重放门 / G0b 功能式四行重放门 / G1 查询行数 / G2 KV 长度；G3 暂缓",
                           diag_precision="qk_fp32=False, prob_fp32=False（关闭）"),
               scope="L27 attention 最小对照（G0/G1/G2）；不改生产实现/精度/KV/argmax/门槛；"
                     "不跑生成；不扫描其它层；无性能结论。")
    print(f"[g012] run_id={run_id}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    ok, why = P5V.commit_guard()
    print(f"[g012] commit: {why}")
    if not ok:
        rec["stage"] = "refused"
        (OUTDIR / f"{TAG}_attng012_{run_id}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return 3

    hooks: list = []
    orig_rope = M.apply_rope
    rope_patched = False
    try:
        rec["diag_precision"] = M.set_diag_preset("eager")
        DC.require(rec["diag_precision"].get("qk_fp32") is False and
                   rec["diag_precision"].get("prob_fp32") is False,
                   "QK32/prob32 必须处于关闭状态", str(rec["diag_precision"]))

        snap_p = _newest(f"{TAG}_{FAMILY}_spec_*.cache.pt")
        rows_p = _newest(f"{TAG}_{FAMILY}_spec_*.rows.pt")
        l27_p = _newest(f"{TAG}_l27_*.pt")
        rec["sources"] = dict(cache=str(snap_p), rows=str(rows_p), l27=str(l27_p),
                              note="源产物只读；生产 forward 会把 cache **原地改长**，故必须先 deep clone")
        snap_all = torch.load(snap_p, map_location="cpu")
        rows_saved = torch.load(rows_p, map_location="cpu")
        l27_saved = torch.load(l27_p, map_location="cpu")
        real_row = rows_saved[f"round{ROUND}_rows"][ALIGN_ROW].detach().float().cpu()
        rec["real_block_row"] = _fields(real_row)
        DC.require(rec["real_block_row"]["argmax_id"] == EXPECT_BLOCK,
                   "真实块行 argmax 必须是 576", str(rec["real_block_row"]["argmax_id"]))
        print(f"[g012] 真实块行 argmax={EXPECT_BLOCK} ✔ 并列={rec['real_block_row']['n_max_tied']}"
              f"{rec['real_block_row']['tied_max_tokens']} "
              f"margin={rec['real_block_row']['margin_top1_top2']}")

        from .lowmem_load import load_ours_bf16
        tgt, tok, ti = load_ours_bf16(P5V.TARGET)
        rec["weights"] = dict(bitmatch=ti.get("weights_bitmatch"), checked=ti.get("n_params_checked"))
        tgt.set_kv_format(spec_dec._fmt("base_bf16"))
        M.set_attention_strategy("cudnn_prefill")
        DC.require(M.attention_strategy() == "cudnn_prefill", "attention 策略必须是 cudnn_prefill")
        DC.require(M.diag_precision()["qk_fp32"] is False and
                   M.diag_precision()["prob_fp32"] is False, "精度开关必须关闭")

        # ---------------- 阶段 A：补采（一次完整 forward，只读观察） ----------------
        snap = _deep_clone_cpu(snap_all[f"round{ROUND}_snap_pre"])
        DC.require(len(snap) == 28, "cache 层数必须是 28", str(len(snap)))
        DC.require(int(snap[L]["k"].shape[2]) == PRE and int(snap[L]["v"].shape[2]) == PRE,
                   "轮前 cache 长度必须是 8215", str(int(snap[L]["k"].shape[2])))
        DC.require(str(snap[L]["k"].dtype) == "torch.bfloat16", "cache 必须是 bf16")
        snap_before = _deep_clone_cpu(snap)                  # 独立快照（事后证明未被覆盖）
        blk_saved = snap_all[f"round{ROUND}_blk"]

        held: dict = {}
        scope = dict(in_l27=False)

        def spy_rope(x, cos, sin):
            out = orig_rope(x, cos, sin)                     # 返回的仍是原函数结果（只读观察）
            if scope["in_l27"] and x.dim() == 4 and x.shape[3] == C.HEAD_DIM:
                if x.shape[1] == C.N_HEADS and "q_rope" not in held:
                    held["q_rope"] = out.detach().clone()
                elif x.shape[1] == C.N_KV_HEADS and "k_rope_new" not in held:
                    held["k_rope_new"] = out.detach().clone()
            return out

        def grab(key, take_input):
            def h(mod, inp, out):
                held[key] = (inp[0] if take_input else out).detach().clone()
            return h

        oproj_mod = tgt.layers[L].self_attn.o_proj
        hooks.append(tgt.layers[L].register_forward_pre_hook(lambda m, i: scope.update(in_l27=True)))
        hooks.append(tgt.layers[L].register_forward_hook(lambda m, i, o: scope.update(in_l27=False)))
        hooks.append(oproj_mod.register_forward_hook(grab("core_out", True)))   # o_proj 输入 = 核心输出
        hooks.append(oproj_mod.register_forward_hook(grab("oproj_out", False)))
        hooks.append(tgt.layers[L].input_layernorm.register_forward_hook(grab("ln_out", False)))
        gpu_cache = _to_gpu(snap)                            # 注意：这是**新 dict**，原地改长发生在这份上
        M.apply_rope = spy_rope
        rope_patched = True
        with torch.inference_mode(), M.attention_phase("verify"):
            rows, cache_blk = spec_dec._verify(tgt, gpu_cache, CANDS)
        M.apply_rope = orig_rope
        rope_patched = False
        for h in hooks:
            h.remove()
        hooks.clear()

        inst_row = rows[ALIGN_ROW].detach().float().cpu()
        k_all = cache_blk[L]["k"].detach().clone()           # 生产返回的 cache 即本次实际使用的 k_all
        v_all = cache_blk[L]["v"].detach().clone()
        rec["capture"] = dict(q_rope=DC.layout(held["q_rope"]), core_out=DC.layout(held["core_out"]),
                              oproj_out=DC.layout(held["oproj_out"]), ln_out=DC.layout(held["ln_out"]),
                              k_all=DC.layout(k_all), v_all=DC.layout(v_all),
                              k_rope_new=DC.layout(held["k_rope_new"]))

        # ---- 必需检查（阶段 A） ----
        DC.require(bool(torch.equal(inst_row, real_row)),
                   "重放最终 logits 未逐元素复现真实块行")
        DC.require(list(held["q_rope"].shape) == [1, C.N_HEADS, 4, 128],
                   "RoPE 后 q 形状必须是 (1,12,4,128)", str(list(held["q_rope"].shape)))
        DC.require(list(held["core_out"].shape) == [1, 4, 1536],
                   "核心输出形状必须是 (1,4,1536)", str(list(held["core_out"].shape)))
        DC.require(list(k_all.shape) == [1, 2, T_BLK, 128],
                   "k_all 形状必须是 (1,2,8219,128)", str(list(k_all.shape)))
        DC.require(str(k_all.dtype) == "torch.bfloat16" and
                   str(held["q_rope"].dtype) == "torch.bfloat16", "Q/K 必须是 bf16")
        DC.require(torch.equal(k_all.cpu(), blk_saved[L]["k"].clone()) and
                   torch.equal(v_all.cpu(), blk_saved[L]["v"].clone()),
                   "本次重算的 k_all/v_all 与既有 blk cache 不一致")
        DC.require(int(cache_blk[L]["k"].shape[2]) == T_BLK,
                   "生产应把 cache 原地改长为 8219", str(int(cache_blk[L]["k"].shape[2])))
        DC.require(torch.equal(cache_blk[L]["k"][:, :, :PRE, :].cpu(), snap_before[L]["k"]),
                   "历史 cache（前 8215 项）被覆盖")
        DC.require(torch.equal(cache_blk[L]["k"].cpu(), snap_before[L]["k"]) is False,
                   "语义检查：改长后的 cache 不应等于旧快照（否则说明未真正追加）")
        DC.require(DC.finite_all(held["core_out"]) and DC.finite_all(held["oproj_out"]),
                   "捕获输出含非有限值")

        oproj_row_block = held["oproj_out"][0, ALIGN_ROW].detach().float().cpu()
        rec["G0a"] = dict(
            replay_final_logits_equals_real=True, replay_argmax=rec["real_block_row"]["argmax_id"],
            k_all_matches_saved_blk_cache=True, history_cache_intact=True,
            q_shape=list(held["q_rope"].shape), core_shape=list(held["core_out"].shape),
            kv_T=int(k_all.shape[2]),
            oproj_block_shape_row_vs_l27_saved=_row_cmp(
                oproj_row_block, l27_saved["block_atten_out"].float(), "oproj_block_row_vs_l27_saved"),
            note="块形状 o_proj 输出**仅用于生产重放校验**，不参与 G1/G2 归因（§五）")
        DC.require(rec["G0a"]["oproj_block_shape_row_vs_l27_saved"]["elementwise_equal"],
                   "块形状 o_proj 输出未逐元素复现已存 atten_out 行")
        print("[g012] G0a 生产重放 ✔：最终 logits == 真实块行（逐元素）；"
              "k_all/v_all == 既有 blk cache；历史 cache 前 8215 项未被覆盖；"
              "块形状 o_proj 行 == 已存 atten_out")

        # ---------------- 阶段 B：功能式重放门 + G1 + G2 ----------------
        q4 = held["q_rope"]
        keep4 = C.block_mask(4, T_BLK, q4.device)
        DC.require(list(keep4.shape) == [4, T_BLK], "mask 形状必须是 (4,8219)")
        DC.require(C.last_visible_key(keep4[ALIGN_ROW].cpu()) == 8216,
                   "对齐行最后可见 key 必须是 8216", str(C.last_visible_key(keep4[ALIGN_ROW].cpu())))
        DC.require(C.visible_count(keep4[ALIGN_ROW].cpu()) == 8217, "对齐行可见数必须是 8217")
        DC.require(not bool(keep4[ALIGN_ROW][8217]) and not bool(keep4[ALIGN_ROW][8218]),
                   "对齐行不得看见 8217/8218")

        core4 = C.core_forward(q4, k_all, v_all, keep4)
        av4 = core4["av"]
        DC.require(bool(torch.equal(C.flatten_core(av4), held["core_out"])),
                   "G0b 功能式四行重放未逐元素复现捕获的核心输出")
        sc4 = core4["scores"]
        keepF4 = keep4.expand_as(sc4)
        sc4_masked = sc4.masked_fill(~keepF4, float("-inf"))   # mask 施加后的 scores（生产 :418/:423 同义）
        rec["scores_checks"] = dict(
            raw_scores_shape=list(sc4.shape), mask_shape=list(keep4.shape),
            valid_positions_intact=True,
            note="**原始 scores 在语义上不过 mask**（生产在 softmax 之前才 masked_fill）；"
                 "因此「有效位置有限性」与「被 mask 位置 = -inf」两项都针对**施加 mask 之后**的 scores 检查。")
        DC.require(bool(torch.isfinite(sc4_masked.float()).masked_select(keepF4).all()),
                   "施加 mask 后，有效位置 scores 出现非有限值")
        DC.require(bool((sc4_masked.float().masked_select(~keepF4) == float("-inf")).all()),
                   "施加 mask 后，被 mask 位置必须恰为 -inf",
                   f"被 mask 位置数={int((~keepF4).sum())}")
        rec["G0b"] = dict(gate_passed=True, core_shape=list(held["core_out"].shape),
                          scores_shape=list(sc4.shape), mask_shape=list(keep4.shape),
                          note="功能式**四行**核心输出 == 生产捕获核心输出（逐元素）；"
                               "scores 只在有效位置查有限性，被 mask 位置为预期 -inf")
        print("[g012] G0b 门 ✔：功能式四行重放逐元素 == 生产捕获核心输出")

        # G1：只改查询行数（4 → 1）
        q1 = q4[:, :, ALIGN_ROW:ALIGN_ROW + 1, :].contiguous()
        DC.require(list(q1.shape) == [1, C.N_HEADS, 1, 128],
                   "G1 单查询 q 必须是 (1,12,1,128)", str(list(q1.shape)))
        DC.require(torch.equal(q1.reshape(-1), q4[:, :, ALIGN_ROW, :].reshape(-1)),
                   "G1 的 q 必须取自同一份 Q 的对齐行（不得另行生成）")
        keep1 = keep4[ALIGN_ROW:ALIGN_ROW + 1].clone()
        DC.require(torch.equal(keep1, keep4[ALIGN_ROW:ALIGN_ROW + 1]),
                   "G1 的 mask 必须与块行 1 逐元素相同")
        DC.require(C.last_visible_key(keep1[0].cpu()) == 8216 and
                   C.visible_count(keep1[0].cpu()) == 8217, "G1 有效可见范围必须是 0..8216")
        core1 = C.core_forward(q1, k_all, v_all, keep1)
        DC.require(DC.finite_all(core1["probs"]) and DC.finite_all(core1["av"]), "G1 输出含非有限值")

        # G2：只改 T（8219 → 8217），K/V 纯切片
        k2, v2 = C.truncate_kv(k_all, v_all, T_TRIM)
        DC.require(torch.equal(k2, k_all[:, :, :T_TRIM, :]) and torch.equal(v2, v_all[:, :, :T_TRIM, :]),
                   "G2 截断必须逐元素保留前 8217 项")
        keep2 = C.block_mask(1, T_TRIM, q1.device)
        DC.require(torch.equal(keep2[0], keep1[0][:T_TRIM]),
                   "G2 的 mask 必须是 G1 mask 截到 8217（有效范围一致）")
        DC.require(C.last_visible_key(keep2[0].cpu()) == 8216, "G2 有效可见范围必须是 0..8216")
        core2 = C.core_forward(q1, k2, v2, keep2)
        DC.require(DC.finite_all(core2["probs"]) and DC.finite_all(core2["av"]), "G2 输出含非有限值")

        cmp = dict(
            G1_vs_G0_core=_row_cmp(C.flatten_core(core1["av"])[0, 0],
                                   C.flatten_core(av4)[0, ALIGN_ROW], "G1_vs_G0_core"),
            G1_vs_G0_scores=_row_cmp(core1["scores"][0, :, 0, :], sc4[0, :, ALIGN_ROW, :],
                                     "G1_vs_G0_scores"),
            G1_vs_G0_probs=_row_cmp(core1["probs"][0, :, 0, :],
                                    core4["probs"][0, :, ALIGN_ROW, :], "G1_vs_G0_probs"),
            G2_vs_G1_core=_row_cmp(C.flatten_core(core2["av"])[0, 0],
                                   C.flatten_core(core1["av"])[0, 0], "G2_vs_G1_core"),
            G2_vs_G1_scores=_row_cmp(core2["scores"][0, :, 0, :T_TRIM],
                                     core1["scores"][0, :, 0, :T_TRIM], "G2_vs_G1_scores_first8217"),
            G2_vs_G1_probs=_row_cmp(core2["probs"][0, :, 0, :T_TRIM],
                                    core1["probs"][0, :, 0, :T_TRIM], "G2_vs_G1_probs_first8217"))

        oproj = tgt.layers[L].self_attn.o_proj
        with torch.inference_mode():
            o0 = oproj(C.flatten_core(av4)[:, ALIGN_ROW:ALIGN_ROW + 1, :])
            o1 = oproj(C.flatten_core(core1["av"]))
            o2 = oproj(C.flatten_core(core2["av"]))
        DC.require(list(o0.shape) == list(o1.shape) == list(o2.shape) == [1, 1, 1536],
                   "G4' 三个 o_proj 调用的形状必须完全一致",
                   f"{list(o0.shape)} {list(o1.shape)} {list(o2.shape)}")
        cmp["G4p_oproj_G1_vs_G0"] = _row_cmp(o1[0, 0], o0[0, 0], "G4p_oproj_G1_vs_G0")
        cmp["G4p_oproj_G2_vs_G1"] = _row_cmp(o2[0, 0], o1[0, 0], "G4p_oproj_G2_vs_G1")

        rec["G1"] = dict(q_shape=list(q1.shape), kv_T=T_BLK, dtype=str(q1.dtype),
                         valid_keys="0..8216（8217 个）", mask_explicit_same_as_block_row1=True,
                         probs_row_sum=float(core1["probs"].float().sum(-1).reshape(-1)[0]),
                         note="**刻意偏离**生产 `s==1` 分支（生产在 s==1 时**不施加** mask）："
                              "此处显式施加与块行 1 逐元素相同的 mask，以防看见未来 key 8217/8218")
        rec["G2"] = dict(q_shape=list(q1.shape), kv_T=T_TRIM, physical_len=T_TRIM,
                         valid_keys="0..8216（8217 个 = 全部物理 key）",
                         truncation_is_pure_slice=True, mask_all_visible=True,
                         probs_row_sum=float(core2["probs"].float().sum(-1).reshape(-1)[0]),
                         note="T=8217 时 0..8216 恰为全部 key ⇒ mask 全可见；有效范围与 G1 一致")
        rec["compare"] = cmp
        rec["verdict"] = dict(
            G1_query_rows_effect=dict(elementwise_equal=cmp["G1_vs_G0_core"]["elementwise_equal"],
                                      core_max=cmp["G1_vs_G0_core"]["abs_err_max"],
                                      core_n_diff=cmp["G1_vs_G0_core"]["n_diff"],
                                      scores_equal=cmp["G1_vs_G0_scores"]["elementwise_equal"],
                                      probs_equal=cmp["G1_vs_G0_probs"]["elementwise_equal"]),
            G2_masked_kv_len_effect=dict(elementwise_equal=cmp["G2_vs_G1_core"]["elementwise_equal"],
                                         core_max=cmp["G2_vs_G1_core"]["abs_err_max"],
                                         core_n_diff=cmp["G2_vs_G1_core"]["n_diff"],
                                         scores_equal=cmp["G2_vs_G1_scores"]["elementwise_equal"]),
            oproj_unified_shape=dict(G1_vs_G0_max=cmp["G4p_oproj_G1_vs_G0"]["abs_err_max"],
                                     G2_vs_G1_max=cmp["G4p_oproj_G2_vs_G1"]["abs_err_max"],
                                     G1_vs_G0_equal=cmp["G4p_oproj_G1_vs_G0"]["elementwise_equal"],
                                     G2_vs_G1_equal=cmp["G4p_oproj_G2_vs_G1"]["elementwise_equal"]),
            boundary=("输出**相同** ⇒ 仅表示『本次固定输入下未观察到该变量效应』，**不**代表所有输入或 kernel 正常；"
                      "输出**不同** ⇒ 仅表示该**受控变量**在本例产生数值差异，**不**等于 bug，"
                      "**不**证明导致 576/4710 翻转；**不**宣称解释原始 AR=715；不作性能声明、不放宽门槛、"
                      "不标正式 correctness 通过；**不**以某候选的 ULP 归一化整行误差。"))
        rec["stage"] = "complete"
        p = OUTDIR / f"{TAG}_attng012_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        tp = OUTDIR / f"{TAG}_attng012_{run_id}.pt"
        torch.save(dict(q_rope=q4.detach().float().cpu(), core_out=held["core_out"].detach().float().cpu(),
                        oproj_out_block=held["oproj_out"].detach().float().cpu(),
                        ln_out=held["ln_out"].detach().float().cpu(), mask4=keep4.cpu(),
                        G0_scores_row=sc4[0, :, ALIGN_ROW, :].detach().float().cpu(),
                        G0_probs_row=core4["probs"][0, :, ALIGN_ROW, :].detach().float().cpu(),
                        G0_core_row=C.flatten_core(av4)[0, ALIGN_ROW].detach().float().cpu(),
                        G1_core=C.flatten_core(core1["av"])[0, 0].detach().float().cpu(),
                        G1_probs_row=core1["probs"][0, :, 0, :].detach().float().cpu(),
                        G2_core=C.flatten_core(core2["av"])[0, 0].detach().float().cpu(),
                        oproj_G0=o0[0, 0].detach().float().cpu(), oproj_G1=o1[0, 0].detach().float().cpu(),
                        oproj_G2=o2[0, 0].detach().float().cpu()), str(tp))
        rec["artifacts"] = dict(json=str(p), pt=str(tp))
        print("== 对齐行（行 1）比较 ==")
        for k in ("G1_vs_G0_core", "G1_vs_G0_scores", "G1_vs_G0_probs",
                  "G2_vs_G1_core", "G2_vs_G1_scores", "G2_vs_G1_probs",
                  "G4p_oproj_G1_vs_G0", "G4p_oproj_G2_vs_G1"):
            v = cmp[k]
            print(f"  {k:24s} equal={v['elementwise_equal']} max|Δ|={v['abs_err_max']:.6g} "
                  f"mean={v['abs_err_mean']:.6g} n_diff={v['n_diff']}/{int(torch.tensor(v['shape']).prod())} "
                  f"shape={v['shape']}")
        print("== 判定 ==")
        print(f"  G1 查询行数效应：核心 equal={rec['verdict']['G1_query_rows_effect']['elementwise_equal']} "
              f"max={rec['verdict']['G1_query_rows_effect']['core_max']:.6g} "
              f"n_diff={rec['verdict']['G1_query_rows_effect']['core_n_diff']}")
        print(f"  G2 长度效应：核心 equal={rec['verdict']['G2_masked_kv_len_effect']['elementwise_equal']} "
              f"max={rec['verdict']['G2_masked_kv_len_effect']['core_max']:.6g} "
              f"n_diff={rec['verdict']['G2_masked_kv_len_effect']['core_n_diff']}")
        print(f"[g012] RESULT_JSON={p}")
        return 0
    except Exception as e:                                        # noqa: BLE001
        if rope_patched:
            M.apply_rope = orig_rope
        for h in hooks:
            try:
                h.remove()
            except Exception:                                     # noqa: BLE001
                pass
        rec["stage"] = ("checks_failed" if isinstance(e, DC.CheckError) else "crashed")
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        M.set_diag_precision(qk32=False, prob32=False)
        p = OUTDIR / f"{TAG}_attng012_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[g012] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[g012] RESULT_JSON={p}")
        return 5 if rec["stage"] == "checks_failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
