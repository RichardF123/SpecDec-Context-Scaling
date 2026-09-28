"""g5_8k_divergence_2x2.py — 用**真实两条路径的真实 cache 历史**做 2×2 定点对照（TOOL_VERSION=2）。

要回答的问题
  首个分歧（生成序列 index 31）处：分歧来自 **计算形状**（s=1 单步 vs 真实候选块），
  还是来自 **cache 历史**（真实 AR 路径 vs 真实 SpecDec 路径）？

本轮（v2）修正的两个**实验配置/证据**缺陷（v1 的结论不可直接沿用）
  ── 缺陷 1：**生成预算改变了被诊断的计算形状**
     v1 调 `spec_decode_greedy(..., 32, k=4)`，而原失败实验是 `max_new=64, k=4`。
     目标轮（pre=8222）之前已生成 30 个 token ⇒ `k_eff = min(4, 32-30) = 2`
     ⇒ **实际执行的是 Kp=2 的块**，而对照重放却用硬编码的 4 个候选。
     后果：v1 把"重放的 4 候选行"当成了"实测 rows[0]"来报（v_fp8 档实测 Kp=2 给 5880，
     重放 Kp=4 给 1083，`spec_actual_vs_replayed_block.argmax_same=false`）。
     v2：**恢复 max_new=64 / k=4**，只由捕获 hook 在该轮之后中止（不靠减少预算提前结束），
         并在捕获时**断言** round/cache_len/Kp/候选/位置与原实验记录一致，
         不一致即标 `NOT_REPRODUCED` 并停止（不继续套用旧归因）。
  ── 缺陷 2：**快照校验存在别名盲点**
     v1 的 `_clone` 只复制 dict，张量仍共享 ⇒ `torch.equal(snap_ref[i]['k'], snap[i]['k'])`
     比的是**同一个张量对象**，恒为 True —— 该检查不可能失败（空证据）；且只覆盖 k/v，
     漏掉 `v_scale`/`fmt` 等字段。
     v2：改用 `diag_cache_cmp.deep_clone`（逐张量独立副本）作为参照，
         完整性核对遍历**所有层的所有张量键** + `fmt` + 层数 + 长度，
         并有 CPU 自测证明"故意原地修改/只改 scale/键重绑都能被检测出来"，
         同时证明旧的浅拷贝参照**检测不到**。

2×2（每格给出**原始分数**、argmax、**完整并列集合**、margin、有限性）
  cache 历史（AR / SpecDec） ×  计算形状（single / 真实候选块）
  实测格子：AR-history×single = AR 真实决策；SpecDec-history×真实块 = SpecDec 真实 rows[0]。

纪律
  * 生产精度路径（`set_diag_preset("eager")` 且记录）；不改 dtype/k/chunk/量化/后端/tie-breaking/验收阈值；
  * 逐决策有限性检查保留（复用 `DecisionHooks` + `CheckRegistry`）；
  * 同一参数只跑一次；未复现就如实记录，不换 prompt、不挑样本；
  * 本工具**不做性能测量**；产物只用于正确性/数值取证。

用法
  python -m specdec.engine.g5_8k_divergence_2x2            # DIAG_MODE=ar（默认）
  DIAG_MODE=spec   ...                                    # 真实 SpecDec 历史
  DIAG_MODE=compare ...                                   # 纯 CPU：读两份 dump 出 2×2
环境变量：DIAG_KV=base_bf16|v_fp8、DIAG_TAG（把同名产物串起来）、DIAG_CTX、DIAG_CHUNK。
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import torch

from .. import results_schema as S
from . import diag_cache_cmp as CC
from . import diag_fields as DF
from . import model_self as M
from . import spec_cache
from . import spec_dec
from . import g5_8k_spec_variance_probe as P5V
from .smoke_checks import CheckRegistry
from .smoke_hooks import DecisionHooks

TOOL_VERSION = 2
ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TARGET, DRAFT = P5V.TARGET, P5V.DRAFT

MODE = os.environ.get("DIAG_MODE", "ar").strip().lower()
KV = os.environ.get("DIAG_KV", "base_bf16")
TAG = os.environ.get("DIAG_TAG", "div2x2")
CTX = int(os.environ.get("DIAG_CTX", "8192"))
CHUNK = int(os.environ.get("DIAG_CHUNK", "2048"))
PROMPT_LEN = CTX

# ★ 与原失败实验**完全一致**的生成配置（v1 错在把 max_new 改成 32）
MAX_NEW = int(os.environ.get("DIAG_MAX_NEW", "64"))
K = int(os.environ.get("DIAG_K", "4"))

PREFIX_TOKENS = int(os.environ.get("DIAG_PREFIX_TOKENS", "30"))
DIV_INDEX = 31                                   # 生成序列 index（不是 PREFIX_TOKENS）
REAL_ROUND = 8                                   # pre=8222 的那一轮在原实验里的轮号
WATCH = [1083, 5880, 5230]
BOUNDARY_CACHE_LEN = PROMPT_LEN + PREFIX_TOKENS  # 8222
SRC = {  # 只读来源（按明确文件名，不猜"最新"）
    "base_bf16": dict(ar="smoke8k_cudnn_prefill_20260927T135218Z.json",
                      spec="smoke8k_cudnn_prefill_20260927T135254Z.json"),
    "v_fp8": dict(ar="smoke8k_cudnn_prefill_20260927T135336Z.json",
                  spec="smoke8k_cudnn_prefill_20260927T135413Z.json"),
}


class _Captured(Exception):
    """在拿到目标轮次的真实验证行后主动中止解码，由外层捕获。**不靠削减生成预算停止。**"""


def _path(kind: str, run_id: str) -> Path:
    return OUTDIR / f"{TAG}_{KV}_{MODE}_{run_id}.{kind}"


def _write(rec: dict, run_id: str) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = _path("json", run_id)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)
    return p


def _to_cpu_copy(cache):
    """独立 CPU 副本：`.detach().cpu().clone()`（`.cpu()` 对 CPU 张量是 no-op，故必须 clone）。"""
    if cache is None:
        return None
    return [{k: (v.detach().cpu().clone() if torch.is_tensor(v) else v) for k, v in d.items()}
            for d in cache]


def _source(key: str) -> dict:
    return json.loads((OUTDIR / SRC[KV][key]).read_text(encoding="utf-8"))


def _round_record(src_json: dict, idx: int) -> dict:
    for r in src_json.get("rounds", []):
        if int(r.get("round", -1)) == idx:
            return r
    raise KeyError(f"来源 JSON 里没有 round {idx}")


def _row_extra(row: torch.Tensor) -> dict:
    """整行层面的补充量。**两个"间距"含义不同，不可混用**：

    * `margin_top1_top2` = 最大 − 第二大。**为 0 等价于"顶部并列"**（这才能替代旧 margin）。
    * `gap_to_next_distinct` = 最大 − 下一个**不同**的分数。顶部 2 路并列时它**不是 0**
      （而是到第三档的距离），所以**不能**用它判断是否并列。
    """
    x = row.reshape(-1).float()
    mx = float(x.max())
    below = x[x < mx]
    second = float(below.max()) if below.numel() else None
    top2 = torch.topk(x, 2).values if x.numel() >= 2 else None
    tie = bool(top2 is not None and float(top2[0]) == float(top2[1]))
    return dict(
        max_score=mx,
        second_distinct_score=second,
        gap_to_next_distinct=(None if second is None else mx - second),
        margin_top1_top2=(None if top2 is None else float(top2[0] - top2[1])),
        tie_at_top=tie,
        margin_note="`margin_top1_top2`==0 ⇔ 顶部并列；`gap_to_next_distinct` 是到下一个"
                    "**不同**分数的距离（并列时不为 0），二者不可混用",
    )


def _cell(row: torch.Tensor, watch=None) -> dict:
    f = DF.row_fields(row, watch=(WATCH if watch is None else watch))
    f.update(_row_extra(row))
    return f


def _block_rows(tgt, snap, cands):
    """在**独立副本**上跑一次真实候选块验证，返回 (rows, 工作副本)。"""
    work = CC.deep_clone(snap)
    with torch.inference_mode():
        rows, _blk = spec_dec._verify(tgt, work, list(cands))
    return rows.detach().clone(), work


def _single_row(tgt, snap, tok: int):
    work = CC.deep_clone(snap)
    with torch.inference_mode():
        lg, _ = tgt._run(tgt.embed_tokens(torch.tensor([[tok]], device="cuda")),
                         work, need_logits=True)
    return lg[0, -1].detach().clone()


def run_ar(tgt, ids, expected_ar, rec, reg):
    """真实 AR 路径：prefill + **自由生成**推进到边界（cache_len=8222），捕获该处 cache。

    ★ v2 改进：v1 直接喂固定前缀（teacher-forced）。v2 改为**自由生成**（喂自己的 argmax），
      并逐步与来源 AR 轨迹的前 30 个 token 交叉核对 ⇒ 这才叫"真实 AR 轨迹复现"。
      任何一步不一致 ⇒ `all_steps_agree=False`，外层标 NOT_REPRODUCED。
    """
    with torch.inference_mode(), M.attention_phase("prefill"):
        cache, lg, n = None, None, ids.shape[1]
        for st in range(0, n, CHUNK):
            lg, cache = tgt._run(tgt.embed_tokens(ids[:, st:st + CHUNK]), cache,
                                 need_logits=(st + CHUNK >= n))
    reg.check_finite(lg, phase="prefill", model="target", role="prefill_lg", where="AR prefill")
    gen = [int(lg.reshape(-1, lg.shape[-1])[-1].argmax())]
    steps = [dict(step=0, fed_token=None, argmax=gen[0], gen_index=0,
                  expected=expected_ar[0], agrees=(gen[0] == expected_ar[0]))]
    for i in range(1, PREFIX_TOKENS + 1):
        fed = gen[i - 1]                     # 自由生成：喂自己上一步的 argmax
        with torch.inference_mode(), M.attention_phase("decode"):
            lg, cache = tgt._run(tgt.embed_tokens(torch.tensor([[fed]], device="cuda")),
                                 cache, need_logits=True)
        reg.check_finite(lg, phase="decode", model="target", role="decode_lg",
                         round_=i, step=i, where="AR 单步：紧接被 argmax 使用")
        nxt = int(lg[0, -1].argmax())
        gen.append(nxt)
        exp = expected_ar[i] if i < len(expected_ar) else None
        steps.append(dict(step=i, fed_token=fed, argmax=nxt, gen_index=i,
                          expected=exp, agrees=(exp is not None and nxt == exp)))
    # 关键点：到这一步为止喂了 PREFIX_TOKENS 个 token ⇒ cache_len = 8192+30 = 8222
    clen = spec_cache.cache_length(cache)
    prefix_len = min(PREFIX_TOKENS, len(expected_ar))
    agree_steps = [s for s in steps if s["gen_index"] < prefix_len]
    rec["ar_path"] = dict(
        trajectory_mode="free_generation",
        teacher_forced=False,
        steps=steps,
        gen_head=gen,
        final_cache_len=clen,
        prefix_tokens_checked=prefix_len,
        all_steps_agree=all(s["agrees"] for s in agree_steps),
        n_steps_disagreeing=sum(1 for s in agree_steps if not s["agrees"]),
        note="自由生成（喂自己的 argmax）；逐步与来源 AR 轨迹交叉核对。"
             "未喂固定前缀，故**不**标为 teacher-forced replay",
    )
    return cache, steps


def capture_verdict(expected_round: dict, *, actual_round, actual_cache_len,
                    actual_cands, logged) -> dict:
    """把"这次捕获是否忠实复现了原实验的那一轮"做成**纯函数**，便于 CPU 自测。

    缺陷 1 的核心防线：v1 因为 max_new=32 使实际 Kp=2，却仍按原实验的 4 候选归因。
    这里逐字段核对；任一不符 ⇒ `all_match=False` ⇒ 外层标 NOT_REPRODUCED 并停止。
    """
    exp_cands = [int(x) for x in expected_round["cands"]]
    act_cands = [int(x) for x in (actual_cands or [])]
    v = dict(
        expect_round=int(expected_round["round"]),
        expect_cache_len=BOUNDARY_CACHE_LEN,
        expect_Kp=len(exp_cands),
        expect_cands=exp_cands,
        expect_g=[int(x) for x in expected_round["g"]],
        expect_emitted=[int(x) for x in expected_round["emitted"]],
        expect_predicted_token_position=BOUNDARY_CACHE_LEN + 1,
        actual_round=(None if actual_round is None else int(actual_round)),
        actual_cache_len=(None if actual_cache_len is None else int(actual_cache_len)),
        actual_Kp=len(act_cands),
        actual_cands=act_cands,
    )
    v["matches_round"] = (v["actual_round"] == v["expect_round"])
    v["matches_cache_len"] = (v["actual_cache_len"] == v["expect_cache_len"])
    v["matches_Kp"] = (v["actual_Kp"] == v["expect_Kp"])
    v["matches_cands"] = (act_cands == exp_cands)
    v["matches_g"] = bool(logged is not None and
                          [int(x) for x in logged.get("g", [])] == v["expect_g"])
    v["matches_emitted"] = bool(logged is not None and
                                [int(x) for x in logged.get("emitted", [])] == v["expect_emitted"])
    v["logged_round_record"] = logged
    v["all_match"] = all(v[k] for k in ("matches_round", "matches_cache_len", "matches_Kp",
                                        "matches_cands", "matches_g", "matches_emitted"))
    if not v["matches_Kp"]:
        v["kp_mismatch_note"] = (
            f"实际 Kp={v['actual_Kp']} ≠ 期望 Kp={v['expect_Kp']} ⇒ "
            "**本次捕获没有执行原实验的计算形状**（这正是 v1 max_new=32 的失败模式）")
    return v


def run_spec(tgt, drf, ids, rec, reg, eos, expected_round, expected_ar=None):
    """真实 SpecDec 路径：跑到 pre == 8222 那一轮，捕获**真实** cache 与**真实** rows。"""
    held: dict = {}

    def on_verify_pre(cache, cands):
        if spec_cache.cache_length(cache) == BOUNDARY_CACHE_LEN and "snap" not in held:
            held["snap"] = CC.deep_clone(cache)      # 就地修改**之前**、且**真正独立**的快照
            held["cands"] = [int(x) for x in cands]
            held["cache_len"] = spec_cache.cache_length(cache)
            held["round"] = reg.round
            held["snap_master_ptrs"] = [cache[0]["k"].data_ptr()]

    def on_verify_post(rows):
        if "snap" in held and "rows" not in held:
            held["rows"] = rows.detach().clone()     # 本轮**全部** Kp 行
            held["rows_shape"] = [int(x) for x in rows.shape]

    def on_round_start(r):
        held.setdefault("rounds_seen", []).append(r)
        if "snap" in held and "rows" in held:
            raise _Captured()                        # ★ 拿到真实行就停（与预算无关）

    hooks = DecisionHooks(tgt=tgt, drf=drf, registry=reg,
                          prefill_impl=P5V.timed_chunked_prefill({}, CHUNK, tgt),
                          on_verify_pre=on_verify_pre, on_verify_post=on_verify_post,
                          on_round_start=on_round_start)
    hooks.install()
    rounds: list[dict] = []
    try:
        out = spec_dec.spec_decode_greedy(tgt, drf, ids, MAX_NEW, k=K, eos_id=eos,
                                          target_kv=KV, draft_kv="base_bf16",
                                          log=lambda d: rounds.append(dict(d)))
    except _Captured:
        out = None
    finally:
        hooks.remove()

    rec["spec_path"] = dict(
        max_new=MAX_NEW, k=K,
        stop_policy="capture_hook（不靠减少 max_new 提前结束）",
        stop_reason=("captured" if "snap" in held and "rows" in held else "exhausted"),
        rounds=rounds,
        captured=("snap" in held),
        captured_cache_len=held.get("cache_len"),
        captured_round=held.get("round"),
        captured_cands=held.get("cands"),
        captured_rows_shape=held.get("rows_shape"),
    )
    if rounds:
        last = rounds[-1]
        rec["spec_path"]["last_round_logged"] = dict(
            round=last["round"], pre=last["pre"], Kp=last["Kp"], m=last["m"],
            cands=last["cands"], g=last["g"], emitted=last["emitted"])
    # ★ 捕获条件断言（与来源原实验记录逐字段核对）
    cap = dict(captured=("snap" in held and "rows" in held))
    if cap["captured"]:
        logged = next((r for r in rounds if int(r.get("round", -1)) == REAL_ROUND), None)
        cap.update(capture_verdict(expected_round, actual_round=held.get("round"),
                                   actual_cache_len=held.get("cache_len"),
                                   actual_cands=held.get("cands"), logged=logged))
        if expected_ar is not None:
            cap["ar_prefix_source"] = expected_ar[:PREFIX_TOKENS]
    rec["capture_check"] = cap
    held["rounds_log"] = rounds
    return held


def build_cells(tgt, snap, rec, *, real_rows=None, real_cands=None, block_cands=None) -> dict:
    """在**同一份独立快照**上跑 single 与真实候选块；每次都用 `deep_clone` ⇒ 互不污染。"""
    block_cands = list(block_cands if block_cands is not None else [])
    single_tok = int(block_cands[0])
    rows_dump: dict[str, torch.Tensor] = {}

    row_single = _single_row(tgt, snap, single_tok)
    rows_dump["single"] = row_single.detach().float().cpu()

    rows_block, work_block = _block_rows(tgt, snap, block_cands)
    row_block = rows_block[0]
    rows_dump["block"] = row_block.detach().float().cpu()
    rows_dump["block_all_rows"] = rows_block.detach().float().cpu()

    out = dict(single=_cell(row_single), block=_cell(row_block),
               single_equals_block=bool(torch.equal(row_single, row_block)),
               diff=DF.compare_rows(row_single, row_block))

    # 额外对照（**不能**冒充真实执行行）：更短的候选长度
    controls = {}
    for kp in (2, 3):
        if kp < len(block_cands):
            rc, _w = _block_rows(tgt, snap, block_cands[:kp])
            controls[f"block_kp{kp}"] = _cell(rc[0])
            rows_dump[f"block_kp{kp}"] = rc[0].detach().float().cpu()
            controls[f"block_kp{kp}"]["is_real_executed_row"] = False
            controls[f"block_kp{kp}"]["vs_block_diff"] = DF.compare_rows(rc[0], row_block)
    if controls:
        out["controls_extra_candidate_lengths"] = controls
        out["controls_note"] = ("这些是**额外对照**（更短候选），"
                                "**不得**当作原实验真实执行的那一行")

    out["block_cands"] = block_cands
    out["single_token"] = single_tok
    out["positions"] = DF.position_labels(PROMPT_LEN, DIV_INDEX, BOUNDARY_CACHE_LEN,
                                           len(block_cands))
    out["row0_position"] = dict(
        input_position=BOUNDARY_CACHE_LEN, predicted_token_position=BOUNDARY_CACHE_LEN + 1,
        answers_candidate_index=1,
        note="rows[0] 与 single 都位于输入位置 8222、都由 cands[0] 作为输入 token")

    if real_rows is not None:
        real0 = real_rows[0]
        rows_dump["real_rows"] = real_rows.detach().float().cpu()
        out["real_rows0"] = _cell(real0)
        out["real_rows_all"] = [_cell(real_rows[j]) for j in range(real_rows.shape[0])]
        out["real_actual_cands"] = list(real_cands)
        out["real_is_executed_shape"] = (len(real_cands) == len(block_cands))
        comp = dict(DF.compare_rows(real0, row_block))    # 可能已含 `note`，故用 update 而不是 dict(**)
        comp.update(
            same_cands=(list(real_cands) == list(block_cands)),
            real_Kp=len(real_cands), replay_Kp=len(block_cands),
            comparison_note="**逐元素**比较（不以 argmax 相同代替）；同候选同位置时应当逐元素相等",
        )
        out["real_vs_replay_block"] = comp
    out["_rows"] = rows_dump
    out["_work_block"] = work_block
    return out


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec: dict = dict(
        run_id=run_id, tool="g5_8k_divergence_2x2", tool_version=TOOL_VERSION,
        mode=MODE, kv=KV, tag=TAG,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git=S.git_meta(ROOT),
        config=dict(ctx=CTX, chunk=CHUNK, kv=KV, prefix_tokens=PREFIX_TOKENS,
                    max_new=MAX_NEW, k=K,
                    divergence_gen_index=DIV_INDEX,
                    boundary_cache_len=BOUNDARY_CACHE_LEN,
                    real_round=REAL_ROUND, watch=WATCH),
        fixed_defects=[
            "生成预算：v1 max_new=32 ⇒ 目标轮 k_eff=min(4,32-30)=2，改变了被诊断的计算形状；"
            "v2 恢复 max_new=64/k=4，只由捕获 hook 停止",
            "快照校验：v1 的 _clone 只复制 dict（张量共享）⇒ torch.equal 恒真（空证据）且只覆盖 k/v；"
            "v2 用 diag_cache_cmp.deep_clone + 全层全键（含 v_scale/fmt）核对",
        ],
        scope="定点诊断：真实 AR / 真实 SpecDec 的 cache 历史 × 计算形状 的 2×2。"
              "不做性能测量、不改生产精度、不改验收标准。")
    print(f"[2x2] run_id={run_id} mode={MODE} kv={KV} tag={TAG} tool_version={TOOL_VERSION}",
          flush=True)
    print(f"RUN_ID={run_id}", flush=True)

    if MODE == "compare":
        return _compare(rec, run_id)

    ok, why = P5V.commit_guard()
    pf = P5V.PF.qualify()
    rec["commit_guard"] = dict(ok=ok, why=why, line_gib=float(P5V.MIN_COMMIT_GIB),
                               overridden=bool(float(P5V.MIN_COMMIT_GIB) < 16.0))
    rec["pstate"] = pf
    rec["gpu_consumers"] = P5V.gpu_consumers()
    rec["pstate_note"] = "P-state 只是**环境合格门**（沿用既有纪律）；本轮不做性能测量，其吞吐数字不进入任何结论"
    print(f"[2x2] commit: {why}")
    print(f"[2x2] P-state: qualified={pf.get('qualified')} sm={pf.get('median') or pf.get('sm_median')}")
    if not ok or pf.get("qualified") is not True:
        rec["stage"] = "refused"
        _write(rec, run_id)
        print("[2x2] !! 环境不合格，未加载")
        return 3

    try:
        rec["diag_precision"] = M.set_diag_preset("eager")
        rec["attention_precision_variant"] = None
        M.set_attention_strategy("cudnn_prefill")

        src_ar = _source("ar")
        src_spec = _source("spec")
        gen_ar = src_ar["run"]["gen"]
        gen_spec = src_spec["run"]["gen"]
        expected_ar = list(gen_ar[:PREFIX_TOKENS])
        expected_round = _round_record(src_spec, REAL_ROUND)
        rec["prefix"] = dict(
            tokens=expected_ar,
            agrees=(list(gen_ar[:PREFIX_TOKENS]) == list(gen_spec[:PREFIX_TOKENS])),
            source_files=SRC[KV],
            source_max_new=dict(ar=src_ar["run"].get("max_new"), spec=src_spec["run"].get("max_new")),
            source_round=expected_round,
            observed={"ar_g31": gen_ar[DIV_INDEX], "spec_g31": gen_spec[DIV_INDEX]},
            expected_shapes=dict(real_spec_Kp=len(expected_round["cands"]),
                                 real_spec_s=len(expected_round["cands"]),
                                 ar_single_s=1),
            note="来源实验 4 份 run 都是 max_new=64；本轮必须与之对齐")
        print(f"[2x2] 来源 max_new={rec['prefix']['source_max_new']} "
              f"前缀一致={rec['prefix']['agrees']}；实测 g31: AR={gen_ar[DIV_INDEX]} "
              f"Spec={gen_spec[DIV_INDEX]}；原实验 round8 Kp={len(expected_round['cands'])}")
        if rec["prefix"]["source_max_new"]["spec"] != MAX_NEW:
            raise RuntimeError(f"生成预算未对齐：本轮 {MAX_NEW} vs 来源 "
                               f"{rec['prefix']['source_max_new']['spec']}")

        from .lowmem_load import load_ours_bf16
        tgt, tokz, tinfo = load_ours_bf16(TARGET)
        drf, dinfo = None, None
        if MODE == "spec":
            drf, _t2, dinfo = load_ours_bf16(DRAFT)
            drf.set_kv_format("fp16")
        rec["load"] = dict(target=tinfo, draft=dinfo)
        tgt.set_kv_format(spec_dec._fmt(KV))
        full = P5V.load_ids(tokz)
        ids = full[:, :CTX].contiguous()
        rec["input"] = dict(ids_hash=P5V._gen_hash([int(x) for x in ids[0].tolist()]), ctx=CTX)
        reg = CheckRegistry()

        real_rows = real_cands = None
        if MODE == "ar":
            snap, ar_steps = run_ar(tgt, ids, expected_ar, rec, reg)
            block_cands = list(expected_round["cands"])   # 反事实：用原实验的真实候选集
            rec["block_cands_source"] = ("来源 SpecDec 原实验 round 8 的真实候选（反事实对照）")
            if not rec["ar_path"]["all_steps_agree"]:
                rec["stage"] = "not_reproduced"
                rec["not_reproduced_reason"] = "自由 AR 轨迹与来源前 30 token 不完全一致"
                p = _write(rec, run_id)
                print(f"[2x2] !! NOT_REPRODUCED：自由 AR 与来源不一致 -> {p}")
                return 5
        else:
            held = run_spec(tgt, drf, ids, rec, reg, tokz.eos_token_id, expected_round,
                            expected_ar)
            if not rec["capture_check"]["all_match"]:
                rec["stage"] = "not_reproduced"
                rec["not_reproduced_reason"] = "捕获轮与原实验 round 8 记录不一致（见 capture_check）"
                p = _write(rec, run_id)
                print(f"[2x2] !! NOT_REPRODUCED：捕获条件不符 -> {p}")
                print(f"[2x2] capture_check={json.dumps(rec['capture_check'], ensure_ascii=False, default=str)[:600]}")
                return 5
            snap = held["snap"]
            real_rows = held["rows"]
            real_cands = list(held["cands"])
            block_cands = list(held["cands"])             # ★ 重放必须用**捕获到的真实候选**
            del drf, held
            torch.cuda.empty_cache()
            rec["block_cands_source"] = "捕获到的真实候选（与来源 round 8 一致）"

        rec["cache"] = CC.metadata(snap)
        _dump_path = _path("cache.pt", run_id)
        ref_cpu = _to_cpu_copy(snap)                      # **独立** CPU 参照（真实副本）
        torch.save(ref_cpu, str(_dump_path))
        rec["cache_dump"] = str(_dump_path)
        rec["cache_dump_note"] = ("独立 CPU 副本（逐张量 .cpu().clone()）；"
                                  "参照即来自该副本，故与 GPU 快照**不共享 storage**")
        print(f"[2x2] cache 快照：len={rec['cache']['length']} layers={rec['cache']['n_layers']} "
              f"-> {_dump_path.name}")

        cells = build_cells(tgt, snap, rec, real_rows=real_rows, real_cands=real_cands,
                            block_cands=block_cands)
        work_block = cells.pop("_work_block", None)

        # ---- 快照完整性（deep_clone 参照 + 全层全键）----
        snap_cpu = _to_cpu_copy(snap)
        integ = CC.integrity_check(ref_cpu, snap_cpu)
        integ["master_vs_workcopy_shared_storage"] = bool(
            work_block is not None and
            any(snap[i]["k"].data_ptr() == work_block[i]["k"].data_ptr()
                for i in range(len(snap))))
        integ["shapes_are_independent_copies"] = (not integ["master_vs_workcopy_shared_storage"])
        rec["snapshot_integrity"] = integ
        print(f"[2x2] 快照完整性：all_equal={integ['all_equal']} "
              f"max|Δ|={integ['max_abs_diff']} tensors={integ['n_tensors_compared']}")

        rows_dump = cells.pop("_rows", None)
        if rows_dump is not None:
            rp = _path("rows.pt", run_id)
            torch.save(rows_dump, str(rp))
            rec["rows_dump"] = str(rp)
            rec["rows_dump_keys"] = sorted(rows_dump.keys())
        rec["cells"] = cells
        rec["stage"] = "complete"
        p = _write(rec, run_id)
        for name in ("single", "block"):
            c = cells[name]
            print(f"[2x2] {name:6s}: argmax_id={c['argmax_id']} max={c['max_score']} "
                  f"tied={c['n_max_tied']}{c['tied_max_tokens']} "
                  f"gap_to_next={c['gap_to_next_distinct']} watch={c['watch_scores']}")
        if "real_rows0" in cells:
            c = cells["real_rows0"]
            print(f"[2x2] real  : argmax_id={c['argmax_id']} max={c['max_score']} "
                  f"tied={c['n_max_tied']}{c['tied_max_tokens']}")
            rv = cells["real_vs_replay_block"]
            print(f"[2x2] real vs replay-block: elementwise_equal={rv['elementwise_equal']} "
                  f"argmax_same={rv['argmax_same']} max|Δ|={rv['abs_err_max']} "
                  f"(real_Kp={rv['real_Kp']} replay_Kp={rv['replay_Kp']})")
        d = cells["diff"]
        print(f"[2x2] single vs block: elementwise_equal={d['elementwise_equal']} "
              f"argmax_same={d['argmax_same']} max|Δ|={d['abs_err_max']}")
        print(f"[2x2] RESULT_JSON={p}")
        return 0
    except Exception:
        rec["stage"] = "crashed"
        rec["traceback"] = traceback.format_exc()
        rec["alloc"] = P5V.allocator_state()
        p = _write(rec, run_id)
        print(f"[2x2] !! 崩溃（已落盘 {p}）：\n{rec['traceback']}")
        print(f"[2x2] RESULT_JSON={p}")
        return 2


def _compare(rec: dict, run_id: str) -> int:
    """纯 CPU：读两档两份 dump，做**全层全键**核对，并组装 2×2。"""
    import glob
    print(f"[2x2] compare: kv={KV} tag={TAG}")
    found = {}
    for mode in ("ar", "spec"):
        js = sorted(glob.glob(str(OUTDIR / f"{TAG}_{KV}_{mode}_*.json")))
        if js:
            found[mode] = json.loads(Path(js[-1]).read_text(encoding="utf-8"))
    rec["sources"] = {m: r.get("run_id") for m, r in found.items()}
    if len(found) != 2:
        rec["stage"] = "missing_source"
        p = _write(rec, run_id)
        print(f"[2x2] !! 缺少某一档的结果 ⇒ 停下（{list(found)}）-> {p}")
        return 5

    # 只接受 stage=complete 的来源，避免把失败记录当成证据
    bad = [m for m, r in found.items() if r.get("stage") != "complete"]
    rec["source_stages"] = {m: r.get("stage") for m, r in found.items()}
    if bad:
        rec["stage"] = "bad_source"
        p = _write(rec, run_id)
        print(f"[2x2] !! 来源不是 complete（{bad}）⇒ 停下 -> {p}")
        return 5

    dumps = {}
    for mode, r in found.items():
        c = sorted(glob.glob(str(OUTDIR / f"{TAG}_{KV}_{mode}_{r['run_id']}.cache.pt")))
        if c:
            dumps[mode] = torch.load(c[-1], map_location="cpu")
    hist = {}
    if len(dumps) == 2:
        # fp8 载荷**先反量化**（float32）再比；raw 载荷差在两个不同 scale 下没有物理含义
        hist = CC.compare_caches(dumps["ar"], dumps["spec"], dequantize=True)
        hist["kind"] = "cache_history"
        hist["note"] += "；两条路径在同一 token 前缀下若 scale 不同，raw 差不可比"
        print(f"[2x2] cache 历史比较：n={hist['n_tensors_compared']} "
              f"逐位相同={hist['all_equal']} max|Δ|={hist['max_abs_diff']}")
    rec["cache_history_compare"] = hist or dict(skipped="缺少 dump")
    rec["snapshot_integrity"] = {m: found[m].get("snapshot_integrity") for m in found}
    rec["capture_check"] = found.get("spec", {}).get("capture_check")
    rec["ar_path"] = found.get("ar", {}).get("ar_path")

    tbl = {}
    for mode in ("ar", "spec"):
        if mode in found:
            cs = found[mode].get("cells", {})
            tbl[mode] = {k: cs[k] for k in ("single", "block", "real_rows0")
                         if k in cs}
            if "controls_extra_candidate_lengths" in cs:
                tbl[mode]["controls_extra_candidate_lengths"] = list(
                    cs["controls_extra_candidate_lengths"].keys())
    rec["table_2x2"] = tbl
    rec["observed"] = {m: found[m].get("prefix", {}).get("observed") for m in found}
    rec["stage"] = "complete"
    p = _write(rec, run_id)
    print(f"[2x2] 2×2 表已组装 -> {p}")
    print(f"[2x2] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
