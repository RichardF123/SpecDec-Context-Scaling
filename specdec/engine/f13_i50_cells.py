"""f13_i50_cells.py — **同前缀四格局部对照**（历史来源 × 执行方式），不自由生成。

原用途：F13 index50 在 `pre=8240` 快照处的四格对照（`TARGET_POS=8242`）。
本轮（分歧定位）只**参数化**同一套逻辑到新的现场（`pre=8237`、`TARGET_POS=8239`），
默认值不变 ⇒ 未设环境变量时行为与原来一致。

四格定义（在快照长度 `PRE` 处）：
                    串行 S（s=1）          分块 B（s=4）
  真实 AR 历史 H_AR   H_AR-S                 H_AR-B
  真实 Spec 历史 H_SP H_SP-S                 H_SP-B

固定：相同权重/dtype/backend/位置规则；`max_new` 不参与（局部对照不生成）。
  * S：依次喂入 `FEED`（长度 = `TARGET_POS - PRE`），取**预测 `TARGET_POS`** 的那一行；
  * B：喂入已记录候选 `CANDS`，取 `rows[TARGET_POS-PRE-1]`（同样预测 `TARGET_POS`）；
  * 每格从**独立完整克隆**开始（`cache_clone.clone_list_cache_full` = deepcopy：保留**全部**层内键
    与长度信息；**不**只复制 K/V），并校验与源快照、与其它格**不共享存储**；源快照事后校验未被修改。

控制（只对真实运行里发生过的组合）：**H_AR-S 必须逐元素复现捕获的真实 AR 决策行**；
**H_SP-B 必须逐元素复现捕获的真实 Spec `rows[决策行]`**。另两格没有对应的真实捕获行 ——
那两种组合在真实运行里没有发生，本工具**不**为其编造参考。

本轮新增（**默认关闭**，由环境变量显式开启；不改变默认行为）：
  * `CELLS_REPEAT=2`：每格**重复一次**（每次独立克隆）并断言两次逐位相同（格内复现性）；
  * `CELLS_LEAK_CHECK=1`：**mask 泄漏检查**——把 block 的**尾部槽位**换成别的 token 后重跑，
    要求"决策行"**逐位不变**（无泄漏），同时要求**另一个槽位的行确实改变**（阳性对照，
    证明扰动真的生效，避免"什么都没变"的假通过）。
  * `CELLS_TARGET_ROPE=cpu_fp32`：只对 **target 实例**安装 `target_rope_scope`（与真实运行一致）。

**字段命名纪律**：本工具**不**使用 `own_max_abs`/`prev_max_abs`/`next_max_abs`/`fp32_margin`
（那是 `results_schema.classify_bf16_compare` 的字段，定义为"同位置 block-vs-serial"与"相邻位置"的比较）。
这里一律用**按比较对命名**的字段（如 `same_history_S_vs_B_*`、`same_exec_S_AR_vs_SP_*`、
`repeat_*`），避免把不同历史/不同比较塞进同名槽位。既有 `classify_bf16_compare` 需要
`fp32_margin`（只能由 fp32 运行得到），本轮边界**禁止**新增精度实验 ⇒ 本工具**不**调用它、
**不**给出任何 `BF16_NEAR_TIE` 判定。
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
from . import cache_clone as CC
from . import decision_rows as DR
from . import model_self as M
from . import sb_align as A
from . import spec_cache
from . import spec_dec
from . import g5_8k_spec_variance_probe as P5V

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TAG = os.environ.get("CELLS_TAG", "f13i50_20261001")
KV, ARM = "base_bf16", os.environ.get("CELLS_ARM", "eager")
PRE = int(os.environ.get("CELLS_PRE", "8240"))
K = int(os.environ.get("CELLS_K", "4"))
CHUNK = int(os.environ.get("CELLS_CHUNK", "2048"))
FEED = [int(x) for x in os.environ.get("CELLS_FEED", "220,16").split(",")]
CANDS = [int(x) for x in os.environ.get("CELLS_CANDS", "220,16,16,1154").split(",")]
TARGET_POS = int(os.environ.get("CELLS_TARGET", "8242"))
WATCH = tuple(int(x) for x in os.environ.get("CELLS_WATCH", "16,17,20").split(","))
PROBE_A = os.environ.get("CELLS_PROBE_A", "row_8241")
PROBE_B = os.environ.get("CELLS_PROBE_B", "saved_next")
REPEAT = int(os.environ.get("CELLS_REPEAT", "1"))
LEAK_CHECK = S.env_flag("CELLS_LEAK_CHECK", False)
TARGET_ROPE = os.environ.get("CELLS_TARGET_ROPE", "default").strip().lower()
SELFTEST = S.env_flag("CELLS_SELFTEST", False)
CAP_AR = os.environ.get("CELLS_CAP_AR", "")
CAP_SP = os.environ.get("CELLS_CAP_SP", "")
# 只读"数据流核对"模式（**不加载模型**）：用真实捕获值核对"行 ↔ 位置 ↔ 输出 token"的对应关系
CELLS_MODE = os.environ.get("CELLS_MODE", "gpu").strip().lower()
REF_RUN_AR = os.environ.get("CELLS_REF_AR", "")      # 捕获对应的**运行记录**（提供 rounds/ar_inv/gen）
REF_RUN_SP = os.environ.get("CELLS_REF_SP", "")
# 前缀策略：strict（默认，要求两臂逻辑前缀逐 token 相同）| record（不等也照算，但明确标注"非同一前缀"）
PREFIX_POLICY = os.environ.get("CELLS_PREFIX_POLICY", "strict").strip().lower()
# 本轮捕获把 AR 侧的行存在 `row_target`/`row_prev` 键下（位置写在 JSON 元数据里）；
# 旧捕获用 `row_<预测位置>`（如 `row_8242`）。默认沿用旧约定，可用环境变量切换。
AR_ROW_KEY = os.environ.get("CELLS_AR_ROW_KEY", "")    # 空 ⇒ 用 f"row_{TARGET_POS}"
# 决策槽位：`slot_table(pre,K)` 中预测 TARGET_POS 的那一槽；`row_index = slot-1`
DEC_SLOT = TARGET_POS - PRE
ROW_INDEX = DEC_SLOT - 1


def _newest(pat):
    return Path(sorted(glob.glob(str(OUTDIR / pat)))[-1])


def _resolve_caps() -> tuple:
    """固定引用优先（显式路径）；否则退回"该 TAG 的最新捕获"（原行为）。"""
    ca = Path(CAP_AR) if CAP_AR else _newest(f"{TAG}_capture_ar_*.pt")
    cs = Path(CAP_SP) if CAP_SP else _newest(f"{TAG}_capture_spec_*.pt")
    return ca, cs


def _load_cap(p: Path) -> dict:
    """读捕获：张量在 `.pt`；本轮另存了同名 `.json` 元数据（pre/cands/g/round 等）。

    两者合并（元数据放在 `_meta`，**不**覆盖同名张量键）。
    """
    cap = dict(torch.load(p, map_location="cpu"))
    jp = p.with_suffix(".json")
    cap["_meta"] = (json.loads(jp.read_text(encoding="utf-8")) if jp.exists() else {})
    return cap


def _mv(cap: dict, key, default=None):
    """先取 JSON 元数据、再取 .pt 里的同名键（兼容旧捕获格式）。"""
    m = cap.get("_meta") or {}
    return m.get(key, cap.get(key, default))


def _ar_row_key() -> str:
    return AR_ROW_KEY or f"row_{TARGET_POS}"


def _declared_len(cap: dict, *keys: str) -> tuple:
    """在多个**历史命名**里找一个"声明的长度"字段（旧捕获用 `snapshot_len`，本轮用 `snapshot_cache_len`）。

    找不到就返回 `(None, None)`，由调用方退回"从快照推导"并**标注来源**——
    这样既不破坏旧格式，也不把"推导值"冒充"声明值"。
    """
    for k in keys:
        v = _mv(cap, k)
        if v is not None:
            return int(v), k
    return None, None


def _cache_len(c):
    return int(spec_cache.cache_length(c))


def _summary(r, tag) -> dict:
    x = r.detach().float().reshape(-1)
    t5, t2 = torch.topk(x, 5), torch.topk(x, 2).values
    return dict(cell=tag, argmax=int(torch.argmax(x)), max_score=float(x.max()),
                margin_bf16=float(t2[0] - t2[1]),
                top5=[dict(token=int(i), logit=float(v)) for i, v in zip(t5.indices, t5.values)],
                watch={str(t): float(x[t]) for t in WATCH},
                finite=bool(torch.isfinite(x).all()), sha=A.stable_sha(r), n_elem=int(x.numel()),
                field_note="margin_bf16 = 该格**自身 bf16 logits** 的 top1-top2；"
                           "不是 fp32 margin，也不用于任何通过/豁免判定")


def _cmp(a, b, name) -> dict:
    x, y = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    A.require(x.numel() == y.numel(), f"{name}: 长度不一致")
    d = (x - y).abs()
    return dict(name=name, elementwise_equal=bool(torch.equal(x, y)), max_abs=float(d.max()),
                mean_abs=float(d.mean()), n_diff=int((d > 0).sum()),
                a_argmax=int(torch.argmax(x)), b_argmax=int(torch.argmax(y)),
                same_argmax=bool(int(torch.argmax(x)) == int(torch.argmax(y))))


def _cell_cache(snap):
    """从 CPU 快照造一份**独立完整**的设备缓存（deepcopy ⇒ 与源不共享任何存储）。

    设备取 target 参数的设备（GPU 运行 = cuda；CPU 自测 = cpu），不硬编码 `"cuda"`。
    """
    full = CC.clone_list_cache_full(snap)
    return [{k: (v.to(_CELL_DEV) if torch.is_tensor(v) else v) for k, v in layer.items()}
            for layer in full]


_CELL_DEV = torch.device("cpu")


def _run_cell(tgt, snap, mode, *, feed=None, cands=None, tag="cell"):
    """跑一格，返回 `(row, summary, extra)`；`extra` 含长度/位置/mask 检查。

    `mode="S"`：依次单步喂 `feed`，取最后一次的末位行（预测 TARGET_POS）；
    `mode="B"`：一次块前向喂 `cands`，取 `rows[ROW_INDEX]`。
    """
    feed = list(FEED if feed is None else feed)
    cands = list(CANDS if cands is None else cands)
    cache = _cell_cache(snap)
    DR.require_distinct_storage(cache, snap, f"{tag}: 与源快照")
    extra: dict = {}
    if mode == "S":
        rows = []
        for tk in feed:
            with torch.inference_mode(), M.attention_phase("decode"):
                lg, cache = spec_dec._step(tgt, tk, cache)
            rows.append(lg[0, -1].detach().clone())
        A.require(_cache_len(cache) == PRE + len(feed),
                  f"{tag}: 串行后 cache 应为 {PRE + len(feed)}", str(_cache_len(cache)))
        row = rows[-1]
        extra["cache_len_after"] = _cache_len(cache)
        extra["fed"] = feed
        extra["no_mask_branch"] = ("s==1 ⇒ 引擎 `elif s == 1: pass`（不加掩码）；"
                                   "单步只可能看到 ≤ 自身位置")
    else:
        with torch.inference_mode(), M.attention_phase("verify"):
            rr, blk = spec_dec._verify(tgt, cache, cands)
        A.require(int(rr.shape[0]) == len(cands), f"{tag}: rows 形状异常", str(list(rr.shape)))
        row = rr[ROW_INDEX].detach().clone()
        extra["cache_len_after_block"] = _cache_len(blk)
        extra["block_rows_argmax"] = [int(rr[j].argmax()) for j in range(rr.shape[0])]
        extra["block_rows_cpu"] = rr.detach().float().cpu().clone()
        extra["fed"] = cands
        # 位置/可见范围检查（用既有决策行工具，不改其定义）
        tab = DR.slot_table(PRE, len(cands))
        dslot = tab[DEC_SLOT]
        A.require(dslot["predicted_pos"] == TARGET_POS, "槽位表位置错误", str(dslot))
        A.require(dslot["visible_last_key"] == TARGET_POS - 1,
                  "决策行可见范围应为 target-1", str(dslot))
        A.require(dslot["source"] == f"rows[{ROW_INDEX}]", "决策行来源错误", str(dslot))
        extra["slot"] = dslot
        # rollback 边界：trim 到 PRE 必须逐位复现起点缓存（trim 只做切片，不重算）
        tr = spec_cache.trim_cache(blk, PRE)
        A.require(_cache_len(tr) == PRE, "trim 后长度应为 PRE", str(_cache_len(tr)))
        extra["rollback_trim_equals_pre"] = bool(
            all(torch.equal(tr[i][kk], snap[i][kk].to(tr[i][kk].device))
                for i in range(len(tr)) for kk in ("k", "v")))
    return row, _summary(row, tag), extra


def _leak_check(tgt, snap, tag, ref_rows) -> dict:
    """**mask 泄漏检查**：把决策行**之后**的尾槽 token 换掉，决策行必须逐位不变。

    阳性对照：被替换的尾槽**自身**的行必须**改变**（证明扰动真的生效）。
    没有阳性对照的"没变化"是空证据，不能算通过。
    """
    changed_slots = list(range(ROW_INDEX + 1, len(CANDS)))
    if not changed_slots:
        return dict(applied=False, reason="没有位于决策行之后的槽位可改")
    tail = list(CANDS)
    for j in changed_slots:
        tail[j] = (tail[j] + 1) % 100000
    cache = _cell_cache(snap)
    with torch.inference_mode(), M.attention_phase("verify"):
        rr2, _blk2 = spec_dec._verify(tgt, cache, tail)
    dec_unchanged = bool(torch.equal(rr2[ROW_INDEX].detach().float().cpu(),
                                     ref_rows[ROW_INDEX].detach().float().cpu()))
    tail_changed = [bool(not torch.equal(rr2[j].detach().float().cpu(),
                                        ref_rows[j].detach().float().cpu())) for j in changed_slots]
    return dict(applied=True, changed_slots=changed_slots, tail_used=tail,
                decision_row_index=ROW_INDEX, decision_row_unchanged=dec_unchanged,
                positive_control_tail_rows_changed=tail_changed,
                positive_control_ok=bool(all(tail_changed)),
                decision_row_max_abs_vs_ref=float((rr2[ROW_INDEX].detach().float().cpu()
                                                   - ref_rows[ROW_INDEX].detach().float().cpu()
                                                   ).abs().max()),
                note=("决策行 rows[%d]（预测 %d，可见 ≤ %d）在尾槽被替换后**逐位不变** ⇒ 无尾部泄漏；"
                      "尾槽自身的行必须改变（阳性对照），否则本检查无效"
                      % (ROW_INDEX, TARGET_POS, TARGET_POS - 1)))


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="f13_i50_cells", family="F13", kv=KV, arm=ARM,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=S.git_meta(ROOT),
               config=dict(tag=TAG, pre=PRE, k=K, chunk=CHUNK, feed=FEED, cands=CANDS,
                           target_pos=TARGET_POS, dec_slot=DEC_SLOT, row_index=ROW_INDEX,
                           watch=list(WATCH), repeat=REPEAT, leak_check=LEAK_CHECK,
                           target_rope=TARGET_ROPE,
                           diag_precision="qk_fp32=False, prob_fp32=False"),
               semantics="同前缀四格局部对照（历史来源 × 执行方式）；**不做**自由生成；"
                         "四格各自的输入前缀相同（到 PRE-1 为止），但**分块/串行各自新增 KV 后"
                         "数值历史可能不同**。",
               field_naming="不使用 own_max_abs/prev_max_abs/next_max_abs/fp32_margin；"
                            "所有比较按比较对命名；不调用 classify_bf16_compare（缺 fp32_margin）。")
    print(f"[cells] run_id={run_id} pre={PRE} target={TARGET_POS} dec_slot={DEC_SLOT} "
          f"feed={FEED} cands={CANDS} repeat={REPEAT} leak={LEAK_CHECK} rope={TARGET_ROPE}",
          flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    if SELFTEST:
        return _selftest(rec, run_id)
    if CELLS_MODE == "cpu_prov":
        return _provenance_cpu(rec, run_id)
    ok, why = P5V.commit_guard()
    rec["commit_guard"] = dict(ok=bool(ok), why=why)
    print(f"[cells] commit: {why}")
    if not ok:
        rec["stage"] = "refused"
        _write(rec, run_id)
        return 3
    return _run(rec, run_id)


# ------------------------------------------------------------------ 只读数据流核对（CPU）
def _provenance_cpu(rec, run_id) -> int:
    """**不加载模型**：用真实捕获值核对「缓存 → 输入 → 判断行 → 拒绝/替换 → 输出 token」的对应关系。

    数据来源全部是**真实捕获/真实记录**：捕获的 `rows`/`row_target`/`snapshot_cpu`/`gen`，
    以及两次真实运行记录里的 `rounds`/`ar_inv`。**不**用任何重算值冒充原始值。
    """
    ca, cs = _resolve_caps()
    cap_ar, cap_sp = _load_cap(ca), _load_cap(cs)
    rec["sources"] = dict(capture_ar=str(ca), capture_spec=str(cs),
                          ref_run_ar=REF_RUN_AR, ref_run_spec=REF_RUN_SP,
                          ar_row_key=_ar_row_key())
    gen_sp = list(_mv(cap_sp, "gen") or [])
    gen_ar = list(_mv(cap_ar, "gen") or [])
    pre = int(_mv(cap_sp, "pre"))
    cands = list(_mv(cap_sp, "cands") or [])
    rows = cap_sp.get("rows")
    rows_argmax = ([int(rows[j].argmax()) for j in range(rows.shape[0])] if rows is not None else [])
    kp = _mv(cap_sp, "Kp")
    if kp is None:
        br = _mv(cap_sp, "block_round")
        kp = (br.get("Kp") if isinstance(br, dict) else None) or len(cands)
    kp = int(kp)
    tab = DR.slot_table(pre, kp)
    # (1) 槽位表：每一槽的**判断行来源**与**它预测的位置**、以及该行可见的最大 key
    slots = [dict(s, g_token=(list(_mv(cap_sp, "g") or [])[s["slot"]]
                              if s["slot"] < len(_mv(cap_sp, "g") or []) else None))
             for s in tab]
    # (2) rows[j] 的 argmax ↔ **真实生成序列**在该行预测位置上的 token
    rows_vs_gen = []
    for j, a in enumerate(rows_argmax):
        pos = pre + 1 + j
        gi = pos - 8192
        rows_vs_gen.append(dict(row_index=j, predicts_pos=pos, rows_argmax=a,
                                gen_token=(gen_sp[gi] if 0 <= gi < len(gen_sp) else None),
                                match=bool(0 <= gi < len(gen_sp) and gen_sp[gi] == a)))
    # (3) 若给了运行记录，再用**当轮记录**的 g 交叉核对一次
    rounds_g, g_vs_rows = [], []
    if REF_RUN_SP and Path(REF_RUN_SP).exists():
        rj = json.loads(Path(REF_RUN_SP).read_text(encoding="utf-8"))
        rr = next((x for x in rj["rounds"] if int(x["pre"]) == pre), None)
        if rr is not None:
            rounds_g = list(rr["g"])
            g_vs_rows = [dict(row_index=j, rows_argmax=rows_argmax[j], g_at_j_plus_1=rounds_g[j + 1],
                              match=bool(rounds_g[j + 1] == rows_argmax[j]))
                         for j in range(len(rows_argmax)) if j + 1 < len(rounds_g)]
            rec["round_record"] = dict(round=rr["round"], pre=rr["pre"], Kp=rr["Kp"], m=rr["m"],
                                       cands=rr["cands"], g=rr["g"], emitted=rr["emitted"],
                                       tgt_len=rr["tgt_len"], drf_len=rr["drf_len"])
    # (4) AR 侧真实行 ↔ 位置 ↔ 输出 token
    row_target = cap_ar.get(_ar_row_key())
    ar_row_argmax = int(row_target.argmax()) if row_target is not None else None
    gi47 = TARGET_POS - 8192
    ar_row_check = dict(row_key=_ar_row_key(),
                        at_position=_mv(cap_ar, "row_target_at_position"),
                        predicts=_mv(cap_ar, "row_target_predicts"),
                        input_token=_mv(cap_ar, "row_target_input_token"),
                        cache_len_after=_mv(cap_ar, "row_target_cache_len_after"),
                        row_sha=(A.stable_sha(row_target) if row_target is not None else None),
                        row_argmax=ar_row_argmax,
                        gen_token_at_target=(gen_ar[gi47] if 0 <= gi47 < len(gen_ar) else None),
                        match=bool(ar_row_argmax is not None and 0 <= gi47 < len(gen_ar)
                                   and ar_row_argmax == gen_ar[gi47]))
    # (5) rollback / 截断边界（用真实记录的 pre/m 与 tgt_len）
    rb = {}
    if rec.get("round_record"):
        rr = rec["round_record"]
        rb = dict(pre=rr["pre"], m=rr["m"], kept_len=rr["pre"] + rr["m"],
                  tgt_len_after=rr["tgt_len"], expected_tgt_len=rr["pre"] + rr["m"] + 1,
                  consistent=bool(rr["tgt_len"] == rr["pre"] + rr["m"] + 1),
                  emitted_equals_cands_prefix_plus_correction=bool(
                      rr["emitted"] == rr["cands"][:rr["m"]] + [rr["g"][rr["m"]]]),
                  correction_token=rr["g"][rr["m"]], rejected_candidate=rr["cands"][rr["m"]])
    # (6) 结构检查：快照完整性与两臂前缀一致
    snap_ar, snap_sp = cap_ar["snapshot_cpu"], cap_sp["snapshot_cpu"]
    # (6b) 两臂历史**最早在哪个位置/层**开始不同（只读；用于给"历史差异"定位到起点）
    first_div, per_layer_first = None, []
    for i in range(len(snap_ar)):
        f_l = None
        for kk in ("k", "v"):
            neq = (snap_ar[i][kk] != snap_sp[i][kk])          # (1, kv_heads, T, head_dim)
            # ⚠️ 归约维必须正确：先去 kv_heads(dim=1)，再去 head_dim(最后一维) ⇒ 留下 (T,)
            #    （首版误写 `.any(dim=1).any(dim=1)`，把 head_dim 当成了位置 ⇒ 得出"prefill 也不同"的**错误**中间结论）
            pos_any = neq.any(dim=1).any(dim=-1).reshape(-1)
            nz = torch.nonzero(pos_any, as_tuple=False)
            if nz.numel():
                p0 = int(nz.reshape(-1)[0])
                f_l = p0 if f_l is None else min(f_l, p0)
        per_layer_first.append(dict(layer=i, first_diff_pos=f_l,
                                    shape=list(snap_ar[i]["k"].shape)))
        if f_l is not None:
            first_div = f_l if first_div is None else min(first_div, f_l)
        if f_l is not None:
            first_div = f_l if first_div is None else min(first_div, f_l)
    hist = dict(first_diff_pos_any_layer=first_div,
                first_diff_meaning=("位置 p 之前**所有层**的 K/V 逐位相同 ⇒ 两臂历史从 p 开始在数值上分叉；"
                                    "8192 = 第一个生成 token 的位置"),
                prefill_positions_identical=bool(first_div is None or first_div >= 8192),
                per_layer_first_diff_pos=[x for x in per_layer_first if x["first_diff_pos"] is not None],
                n_layers_diverging=int(sum(1 for x in per_layer_first if x["first_diff_pos"] is not None)),
                note="只读计算；**不**据此做逐层归因，只说明分叉**起点**")
    structure = dict(
        ar_snap_len=_cache_len(snap_ar), sp_snap_len=_cache_len(snap_sp),
        ar_n_layers=len(snap_ar), sp_n_layers=len(snap_sp),
        keys_ar=sorted(snap_ar[0].keys()), keys_sp=sorted(snap_sp[0].keys()),
        prefix_equal=bool(list(_mv(cap_ar, "prefix_tokens") or [])
                          == list(_mv(cap_sp, "prefix_tokens") or [])),
        prefix_n=len(_mv(cap_ar, "prefix_tokens") or []),
        visible_last_key_for_decision=tab[DEC_SLOT]["visible_last_key"],
        decision_predicts=tab[DEC_SLOT]["predicted_pos"],
        decision_source=tab[DEC_SLOT]["source"])
    rec["provenance"] = dict(pre=pre, Kp=kp, cands=cands, rows_argmax=rows_argmax,
                            slots=slots, rows_vs_gen=rows_vs_gen, g_vs_rows=g_vs_rows,
                            ar_row=ar_row_check, rollback=rb, structure=structure,
                            history_first_divergence=hist,
                            note="`g`（当轮候选判定结果）由 `spec_dec.spec_decode_greedy` 在 hooks 返回"
                                 "**之后**计算，引擎接口未向 hook 暴露 ⇒ 捕获文件里没有该字段；"
                                 "本模式改用 ①真实捕获的 rows_argmax ②真实生成序列 ③运行记录里的 g "
                                 "三者交叉核对，不用重算值冒充原始值。")
    checks = dict(
        rows_match_gen=all(x["match"] for x in rows_vs_gen),
        rows_match_recorded_g=(all(x["match"] for x in g_vs_rows) if g_vs_rows else None),
        ar_row_matches_gen=bool(ar_row_check["match"]),
        decision_slot_is_the_target=bool(tab[DEC_SLOT]["predicted_pos"] == TARGET_POS
                                         and tab[DEC_SLOT]["visible_last_key"] == TARGET_POS - 1),
        rollback_consistent=(rb.get("consistent") if rb else None),
        rollback_emitted_rule=(rb.get("emitted_equals_cands_prefix_plus_correction") if rb else None),
        snapshot_complete=bool(_cache_len(snap_ar) == PRE == _cache_len(snap_sp)
                               and structure["keys_ar"] == structure["keys_sp"]),
        prefix_equal=structure["prefix_equal"])
    rec["checks"] = checks
    rec["stage"] = "complete"
    p = _write(rec, run_id)
    print("== 数据流核对（只读，真实捕获值）==")
    print(f"  pre={pre} Kp={kp} cands={cands} rows_argmax={rows_argmax}")
    for s in slots:
        print(f"    slot{s['slot']}: {s['source']:<10} 预测位置={s['predicted_pos']} "
              f"可见≤{s['visible_last_key']}")
    for x in rows_vs_gen:
        print(f"    rows[{x['row_index']}] argmax={x['rows_argmax']} ↔ 位置 {x['predicts_pos']} 的"
              f"真实 token={x['gen_token']}  一致={x['match']}")
    for x in g_vs_rows:
        print(f"    运行记录 g[{x['row_index']+1}]={x['g_at_j_plus_1']} ↔ rows[{x['row_index']}] "
              f"argmax={x['rows_argmax']}  一致={x['match']}")
    print(f"  AR 行：{json.dumps({k: v for k, v in ar_row_check.items() if k != 'row_sha'}, ensure_ascii=False)}")
    print(f"  rollback：{json.dumps(rb, ensure_ascii=False)}")
    print(f"  历史分叉起点：任意层最早位置={hist['first_diff_pos_any_layer']}"
          f"（prefill 位置是否完全相同={hist['prefill_positions_identical']}；"
          f"分叉层数={hist['n_layers_diverging']}）")
    print(f"   前若干层最早分叉位置：{[ (x['layer'], x['first_diff_pos']) for x in hist['per_layer_first_diff_pos'][:6] ]}")
    print(f"  ★ checks：{json.dumps(checks, ensure_ascii=False)}")
    print(f"[cells] RESULT_JSON={p}")
    return 0 if all(v is not False for v in checks.values()) else 5


def _write(rec, run_id, selftest: bool = False) -> Path:
    name = (f"{TAG}_cells_selftest_{run_id}.json" if selftest else f"{TAG}_cells_{run_id}.json")
    p = OUTDIR / name
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return p


def _run(rec, run_id) -> int:
    TRS = None
    global _CELL_DEV
    try:
        from .lowmem_load import load_ours_bf16
        tgt, tok, ti = load_ours_bf16(P5V.TARGET)
        tgt.set_kv_format(spec_dec._fmt(KV))
        M.set_attention_strategy(ARM)
        M.reset_attention_stats()
        # ⚠️ 格子缓存的设备必须跟随 target 实例（否则 CPU 快照会被搬到 CPU，与 cuda 模型 `cat` 冲突）
        _CELL_DEV = next(tgt.parameters()).device
        rec["cell_device"] = str(_CELL_DEV)
        if TARGET_ROPE == "cpu_fp32":
            from . import target_rope_scope as _TRS
            TRS = _TRS
            rec["target_rope_scope"] = TRS.install(tgt, theta=float(tgt.rope_theta),
                                                   head_dim=int(tgt.head_dim))
        rec["weights"] = dict(bitmatch=ti.get("weights_bitmatch"), checked=ti.get("n_params_checked"))
        ca, cs = _resolve_caps()
        cap_ar = _load_cap(ca)
        cap_sp = _load_cap(cs)
        rows_sp = cap_sp.get("rows")
        rows_ar = cap_ar.get("rows")
        rec["sources"] = dict(capture_ar=str(ca), capture_spec=str(cs),
                              ar_row_key=_ar_row_key(),
                              captured_AR_row_sha=A.stable_sha(cap_ar[_ar_row_key()]),
                              captured_SP_rows_sha=(A.stable_sha(rows_sp[ROW_INDEX])
                                                    if rows_sp is not None else None),
                              captured_AR_rows_present=rows_ar is not None,
                              AR_capture_meta=cap_ar.get("_meta") or {},
                              SP_capture_meta=cap_sp.get("_meta") or {})
        pref_ar = list(_mv(cap_ar, "prefix_tokens") or [])
        pref_sp = list(_mv(cap_sp, "prefix_tokens") or [])
        A.require(pref_ar and pref_sp, "捕获缺少 prefix_tokens（在 JSON 元数据里）",
                  f"ar={len(pref_ar)} sp={len(pref_sp)}")
        _prefix_equal = bool(pref_ar == pref_sp)
        if PREFIX_POLICY == "strict":
            A.require(_prefix_equal, "两次捕获的 token 前缀不一致",
                      f"{pref_ar[:6]} vs {pref_sp[:6]}")
        else:
            A.require(len(pref_ar) == len(pref_sp), "两臂前缀长度不一致（无法逐 token 对位）",
                      f"{len(pref_ar)} vs {len(pref_sp)}")
        A.require(len(pref_ar) == TARGET_POS - 8192,
                  f"前缀长度应为 {TARGET_POS - 8192}", str(len(pref_ar)))
        rec["prefix"] = dict(n=len(pref_ar), tokens=pref_ar, equal=_prefix_equal,
                             policy=PREFIX_POLICY, tokens_sp=pref_sp,
                             note=("两臂捕获的共同前缀逐 token 相同（到 TARGET_POS-1 为止）"
                                   if _prefix_equal else
                                   "⚠️ 两臂逻辑前缀**不同** ⇒ 下面的四格**不是**同一前缀对照，"
                                   "只能作为「不同历史 + 不同前缀」的描述"))
        snap_ar, snap_sp = cap_ar["snapshot_cpu"], cap_sp["snapshot_cpu"]
        A.require(snap_ar is not None and snap_sp is not None, "捕获缺少 snapshot_cpu")
        A.require(_cache_len(snap_ar) == PRE and _cache_len(snap_sp) == PRE,
                  "两份快照长度必须都是 PRE", f"{_cache_len(snap_ar)} / {_cache_len(snap_sp)}")
        # 快照完整性：长度可从 K 的形状推出、层内键集合一致、且与真实运行的报告长度一致
        sp_pre_rec, pre_src = _declared_len(cap_sp, "pre")
        if sp_pre_rec is None:
            sp_pre_rec, pre_src = _declared_len(cap_sp, "snapshot_cache_len", "snapshot_len")
        if sp_pre_rec is None:
            sp_pre_rec, pre_src = _cache_len(snap_sp), "derived_from_snapshot"
        ar_len_rec, ar_len_src = _declared_len(cap_ar, "snap_cache_len", "snapshot_cache_len",
                                               "snapshot_len")
        if ar_len_rec is None:
            ar_len_rec, ar_len_src = _cache_len(snap_ar), "derived_from_snapshot"
        sp_g = list(_mv(cap_sp, "g") or [])
        A.require(_cache_len(snap_ar) == PRE and _cache_len(snap_sp) == PRE,
                  "两份快照长度必须都是 PRE", f"{_cache_len(snap_ar)} / {_cache_len(snap_sp)}")
        A.require(_cache_len(snap_sp) == int(sp_pre_rec), "Spec 快照长度与捕获记录不符")
        A.require(_cache_len(snap_ar) == int(ar_len_rec), "AR 快照长度与捕获记录不符")
        rec["snapshot_integrity"] = dict(
            ar_len=_cache_len(snap_ar), sp_len=_cache_len(snap_sp), n_layers=[len(snap_ar), len(snap_sp)],
            keys_ar=sorted(snap_ar[0].keys()), keys_sp=sorted(snap_sp[0].keys()),
            sp_declared_pre=sp_pre_rec, pre_source=pre_src,
            ar_declared_len=ar_len_rec, ar_len_source=ar_len_src,
            non_tensor_keys_preserved=sorted({k for lay in snap_ar for k in lay
                                              if not torch.is_tensor(lay[k])}),
            clone_helper="cache_clone.clone_list_cache_full (deepcopy，保留全部层内键)")
        # 两份历史 KV 的逐层差异（只作描述，不据此定位根因层）
        dig_ar = A.require_full_cache(snap_ar, expect_len=PRE, n_layers=len(snap_ar), what="H_AR")
        dig_sp = A.require_full_cache(snap_sp, expect_len=PRE, n_layers=len(snap_sp), what="H_SP")
        rec["history_kv"] = dict(compare=A.cache_digest_diff(dig_ar, dig_sp),
                                 note="只报逐层**摘要**差异（sha16），**不**据此自动定位根因层，"
                                      "也不冒充完整张量逐位验证（本轮另做逐张量 torch.equal）")
        # 逐张量**逐位**比较两份历史（比摘要强；只看是否相同，不做逐层归因）
        bit = []
        for i in range(len(snap_ar)):
            for kk in ("k", "v"):
                ta, tb = snap_ar[i][kk], snap_sp[i][kk]
                bit.append(dict(layer=i, attr=kk, shape_equal=tuple(ta.shape) == tuple(tb.shape),
                                bitequal=bool(torch.equal(ta, tb)),
                                n_diff=int((ta != tb).sum()) if ta.shape == tb.shape else None,
                                max_abs=float((ta.float() - tb.float()).abs().max())
                                if ta.shape == tb.shape else None))
        rec["history_kv"]["elementwise"] = dict(
            rows=bit, n_tensors=len(bit), n_bitequal=sum(1 for r in bit if r["bitequal"]),
            all_equal=all(r["bitequal"] for r in bit))
        probes = dict()
        if PROBE_A in cap_ar and PROBE_B in cap_sp:
            probes[f"ar_{PROBE_A}_vs_sp_{PROBE_B}"] = _cmp(
                cap_ar[PROBE_A], cap_sp[PROBE_B], f"{PROBE_A}_vs_{PROBE_B}")
        rec["probe_rows"] = probes
        # 真实决策行的来源核对（用真实捕获的 rows 与捕获记录里的 g）
        g_rec = list(sp_g)
        rows_argmax = ([int(rows_sp[j].argmax()) for j in range(rows_sp.shape[0])]
                       if rows_sp is not None else None)
        rec["provenance"] = dict(
            sp_pre=sp_pre_rec, sp_Kp=_mv(cap_sp, "Kp"), sp_cands=_mv(cap_sp, "cands"),
            sp_round=_mv(cap_sp, "round"), sp_decision_slot=_mv(cap_sp, "decision_slot"),
            sp_row_index_for_decision=_mv(cap_sp, "row_index_for_decision"),
            sp_rows_argmax=rows_argmax, sp_recorded_g=g_rec,
            ar_row_at_position=_mv(cap_ar, "row_target_at_position"),
            ar_row_predicts=_mv(cap_ar, "row_target_predicts"),
            ar_row_input_token=_mv(cap_ar, "row_target_input_token"),
            ar_row_cache_len_after=_mv(cap_ar, "row_target_cache_len_after"),
            rows_match_g=([bool(g_rec[j + 1] == rows_argmax[j]) for j in range(len(rows_argmax))
                           if j + 1 < len(g_rec)] if rows_argmax is not None else []),
            note="rows[j] 应当预测位置 pre+1+j；与真实记录的 g[j+1] 逐个核对（真实值，不是重算）。"
                 "`saved_next` 的张量本体不经引擎接口暴露（未捕获）→ 其位置由 g[0]=位置 pre 的 token "
                 "与 slot_table 证实，不伪造。")
        res, summaries, extras = {}, {}, {}
        for tag, snap, mode in (("H_AR-S", snap_ar, "S"), ("H_AR-B", snap_ar, "B"),
                                ("H_SP-S", snap_sp, "S"), ("H_SP-B", snap_sp, "B")):
            row, summ, extra = _run_cell(tgt, snap, mode, tag=tag)
            res[tag] = row.cpu()
            summaries[tag] = summ
            extras[tag] = extra
            if REPEAT > 1:
                for it in range(REPEAT - 1):
                    row2, _s2, _e2 = _run_cell(tgt, snap, mode, tag=f"{tag}/rep{it+2}")
                    summaries[tag][f"repeat{it+2}_elementwise_equal"] = bool(
                        torch.equal(row.float().cpu(), row2.float().cpu()))
        rec["cells"] = summaries
        rec["cell_checks"] = extras
        # 源快照未被污染
        rec["source_untouched"] = dict(
            ar_len=_cache_len(snap_ar), sp_len=_cache_len(snap_sp),
            ar_ok=bool(_cache_len(snap_ar) == PRE and _cache_len(snap_sp) == PRE))
        A.require(rec["source_untouched"]["ar_ok"], "源快照被污染（长度改变）")
        # ---- 控制：只对"真实运行里发生过的组合"要求逐元素复现 ----
        # Spec 侧控制**只在"捕获的 rows 确实由这张快照算出"时才成立**：
        #   旧捕获格式没有 `block_round` ⇒ 按原行为默认成立；
        #   新捕获（重建实验）带 `block_round.pre` ⇒ 只有当它 == PRE 时 rows 才出自该快照。
        _br = _mv(cap_sp, "block_round")
        _rows_pre = int(_br["pre"]) if isinstance(_br, dict) and "pre" in _br else None
        _sp_ctrl_ok = bool(rows_sp is not None and (True if _rows_pre is None else _rows_pre == PRE))
        rec["controls"] = dict(
            H_AR_S_vs_captured_AR_row=_cmp(res["H_AR-S"], cap_ar[_ar_row_key()],
                                           "H_AR-S_vs_captured_AR"),
            H_SP_B_vs_captured_SP_row=(_cmp(res["H_SP-B"], rows_sp[ROW_INDEX],
                                            "H_SP-B_vs_captured_SP")
                                       if rows_sp is not None else None),
            H_SP_B_control_applicable=_sp_ctrl_ok,
            H_SP_B_control_rows_pre=_rows_pre,
            H_SP_B_rows_present=rows_sp is not None,
            note="只有这两格有对应的**真实捕获行**；H_AR-B / H_SP-S 是干预组合，"
                 "真实运行里没有发生 ⇒ 不设参考、不编造参考。"
                 "Spec 侧控制要求：捕获里确有 `rows`，且 `block_round.pre == PRE`"
                 "（否则那批 rows 不是由本快照算出的）。")
        A.require(rec["controls"]["H_AR_S_vs_captured_AR_row"]["elementwise_equal"],
                  "控制失败：H_AR-S 未逐元素复现真实 AR 决策行",
                  str(rec["controls"]["H_AR_S_vs_captured_AR_row"]))
        if _sp_ctrl_ok:
            A.require(rec["controls"]["H_SP_B_vs_captured_SP_row"]["elementwise_equal"],
                      "控制失败：H_SP-B 未逐元素复现真实 Spec 决策行",
                      str(rec["controls"]["H_SP_B_vs_captured_SP_row"]))
            print("[cells] 控制 ✔ H_AR-S == 真实 AR 行、H_SP-B == 真实 Spec 决策行（均逐元素）")
        else:
            rec["controls"]["H_SP_B_control_skipped"] = dict(
                reason=("捕获里没有 `rows`（本轮该长度附近没有发生块验证）"
                        if rows_sp is None else
                        "捕获的 rows 来自长度 %s 的缓存，而非本快照（PRE=%d）" % (_rows_pre, PRE)),
                rows_present=rows_sp is not None, rows_pre=_rows_pre, PRE=PRE)
            print(f"[cells] 控制 ⚠ H_SP-B 的参考**不适用**（rows_present={rows_sp is not None}，"
                  f"rows_pre={_rows_pre}，PRE={PRE}）；只保留 H_AR-S 控制")
        # ---- 四格两两比较 ----
        rec["compare"] = dict(
            same_history_S_vs_B_AR=_cmp(res["H_AR-S"], res["H_AR-B"], "H_AR_S_vs_B"),
            same_history_S_vs_B_SP=_cmp(res["H_SP-S"], res["H_SP-B"], "H_SP_S_vs_B"),
            same_exec_S_AR_vs_SP=_cmp(res["H_AR-S"], res["H_SP-S"], "S_AR_vs_SP"),
            same_exec_B_AR_vs_SP=_cmp(res["H_AR-B"], res["H_SP-B"], "B_AR_vs_SP"))
        rec["verdict"] = dict(
            execution_shape_same_history_AR=dict(
                same_argmax=rec["compare"]["same_history_S_vs_B_AR"]["same_argmax"],
                max_abs=rec["compare"]["same_history_S_vs_B_AR"]["max_abs"]),
            execution_shape_same_history_SP=dict(
                same_argmax=rec["compare"]["same_history_S_vs_B_SP"]["same_argmax"],
                max_abs=rec["compare"]["same_history_S_vs_B_SP"]["max_abs"]),
            history_same_exec_S=dict(same_argmax=rec["compare"]["same_exec_S_AR_vs_SP"]["same_argmax"],
                                     max_abs=rec["compare"]["same_exec_S_AR_vs_SP"]["max_abs"]),
            history_same_exec_B=dict(same_argmax=rec["compare"]["same_exec_B_AR_vs_SP"]["same_argmax"],
                                     max_abs=rec["compare"]["same_exec_B_AR_vs_SP"]["max_abs"]),
            rule="同一历史下 S/B 不同 ⇒ 支持「执行形状足以产生差异」；相同执行方式下两历史不同 ⇒ "
                 "支持「缓存历史影响决定」；两者并存或依赖历史时如实报告**交互**，不强行选唯一原因。"
                 "margin 大小**不能**单独判定逻辑错误或数值误差；sha 一致只叫**摘要一致**；"
                 "后续 token 相同**不等于**缓存状态相同。")
        if LEAK_CHECK:
            rec["leak_check"] = dict(
                ar=_leak_check(tgt, snap_ar, "H_AR", extras["H_AR-B"]["block_rows_cpu"]),
                sp=_leak_check(tgt, snap_sp, "H_SP", extras["H_SP-B"]["block_rows_cpu"]))
        att = M.attention_stats()
        A.require(not any(v > 0 for v in att["fused_by_phase"].values()),
                  "本配置要求零融合调用", str(att["fused_by_phase"]))
        rec["attention_stats"] = dict(strategy=att["strategy"], fused_by_phase=att["fused_by_phase"],
                                      eager_by_phase=att["eager_by_phase"])
        rec["stage"] = "complete"
        p = _write(rec, run_id)
        tp = OUTDIR / f"{TAG}_cells_{run_id}.pt"
        torch.save(dict(res), str(tp))
        rec["artifacts"] = dict(json=str(p), pt=str(tp))
        print("== 四格 ==")
        for tag in ("H_AR-S", "H_AR-B", "H_SP-S", "H_SP-B"):
            s = summaries[tag]
            print(f"  {tag:8s} argmax={s['argmax']:6d} margin_bf16={s['margin_bf16']:.6g} "
                  f"watch={s['watch']} top5={[x['token'] for x in s['top5']]}")
        print("== 对照 ==")
        for k, v in rec["compare"].items():
            print(f"  {k:26s} same_argmax={v['same_argmax']} max|Δ|={v['max_abs']:.6g} "
                  f"mean={v['mean_abs']:.6g} n_diff={v['n_diff']} ({v['a_argmax']} vs {v['b_argmax']})")
        print(f"  history_kv 逐张量逐位相同 {rec['history_kv']['elementwise']['n_bitequal']}/"
              f"{rec['history_kv']['elementwise']['n_tensors']}")
        _write(rec, run_id)
        print(f"[cells] RESULT_JSON={p}")
        return 0
    except Exception as e:                                        # noqa: BLE001
        rec["stage"] = ("checks_failed" if isinstance(e, (DR.AlignError, A.AlignError,
                                                          CC.CloneError)) else "crashed")
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        p = _write(rec, run_id)
        print(f"[cells] !! 停止（stage={rec['stage']}）：{e}")
        print(f"[cells] RESULT_JSON={p}")
        return 5 if rec["stage"] == "checks_failed" else 2
    finally:
        if TRS is not None:
            rec["target_rope_scope_restore"] = TRS.assert_restored()
            _write(rec, run_id)


# ------------------------------------------------------------------ CPU 自测
def _selftest(rec: dict, run_id: str) -> int:
    """**不加载大模型**：验证本工具新增逻辑——完整快照、独立克隆、四格复现性、mask 泄漏检查。"""
    npass = nfail = 0

    def check(name, cond, extra=None):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  [ok] {name}")
        else:
            nfail += 1
            print(f"  [FAIL] {name}" + (f"  {str(extra)[:200]}" if extra is not None else ""))

    global PRE, TARGET_POS, DEC_SLOT, ROW_INDEX, FEED, CANDS, _CELL_DEV
    torch.manual_seed(0)
    _CELL_DEV = torch.device("cpu")                       # 自测在 CPU 上跑（不占显存）
    tiny = M.QwenLM(vocab_size=64, hidden=32, heads=4, kv_heads=2, head_dim=8, intermediate=64,
                    num_layers=2, rope_theta=10000.0, rms_eps=1e-6, attn_bias=True
                    ).to(torch.bfloat16).eval()
    ids = torch.randint(0, 64, (1, 5))
    with torch.inference_mode():
        _lg, cache0 = tiny._run(tiny.embed_tokens(ids), None, need_logits=True)
    snap = _snap_like(cache0)
    # 小尺寸下的四格参数：PRE=5、喂 2 个 token 预测位置 7、块喂 4 个候选取 rows[1]
    PRE, TARGET_POS, DEC_SLOT, ROW_INDEX = 5, 7, 2, 1
    FEED, CANDS = [3, 4], [3, 4, 5, 6]
    check("★ 快照保留**全部层内键**与长度信息（deepcopy，不只 K/V）",
          sorted(snap[0].keys()) == sorted(cache0[0].keys()) and _cache_len(snap) == 5,
          f"keys={sorted(snap[0].keys())} len={_cache_len(snap)}")
    snap_with_meta = _snap_like(cache0)
    snap_with_meta[0]["fmt"] = "fp16"                     # 非张量元数据必须被保留
    snap2 = CC.clone_list_cache_full(snap_with_meta)
    check("★ 非张量元数据键被保留且可克隆", snap2[0].get("fmt") == "fp16")
    row_s, sum_s, ex_s = _run_cell(tiny, snap, "S", tag="t/S")
    row_b, sum_b, ex_b = _run_cell(tiny, snap, "B", tag="t/B")
    check("★ 单步格：两次喂入后长度 = PRE+2、行有限",
          ex_s["cache_len_after"] == PRE + 2 and bool(torch.isfinite(row_s).all()), ex_s)
    check("★ 分块格：块后长度 = PRE+4、决策槽/可见范围检查通过",
          ex_b["cache_len_after_block"] == PRE + 4
          and ex_b["slot"]["predicted_pos"] == TARGET_POS
          and ex_b["slot"]["visible_last_key"] == TARGET_POS - 1, ex_b.get("slot"))
    check("★ rollback 边界：trim 到 PRE 逐位复现起点缓存", bool(ex_b["rollback_trim_equals_pre"]))
    row_s2, _, _ = _run_cell(tiny, snap, "S", tag="t/S2")
    row_b2, _, _ = _run_cell(tiny, snap, "B", tag="t/B2")
    check("★ 格内复现性：同一格重复一次逐位相同",
          bool(torch.equal(row_s, row_s2)) and bool(torch.equal(row_b, row_b2)))
    check("★ 独立克隆：源快照未被任何一格污染",
          _cache_len(snap) == PRE and bool(torch.equal(snap[0]["k"], cache0[0]["k"])))
    lk = _leak_check(tiny, snap, "t/leak", ex_b["block_rows_cpu"])
    check("★ mask 泄漏检查：尾槽被替换后**决策行逐位不变**",
          bool(lk.get("applied")) and bool(lk["decision_row_unchanged"]), lk)
    check("★ 阳性对照：被替换的尾槽自身的行**确实改变**（扰动生效）",
          bool(lk.get("positive_control_ok")), lk.get("positive_control_tail_rows_changed"))
    print(f"\n四格工具自测：PASS {npass} / FAIL {nfail}")
    rec["selftest"] = dict(npass=npass, nfail=nfail, semantics="非真实运行：只验证工具逻辑")
    rec["stage"] = "selftest"
    _write(rec, run_id, selftest=True)
    return 0 if nfail == 0 else 1


def _snap_like(cache):
    """自测用：和 `g5_8k_smoke_cudnn._snap_cpu_full` 同口径的完整快照。"""
    full = CC.clone_list_cache_full(cache)
    return [{k: (v.detach().to("cpu") if torch.is_tensor(v) else v) for k, v in layer.items()}
            for layer in full]


if __name__ == "__main__":
    raise SystemExit(main())
