"""div8209_preflight_cpu.py — 位置 8209 局部的**飞行前检查**（纯 CPU、只读、不加载模型、不写产物）。

## 为什么需要
上一轮（2026-10-01）与本轮（2026-10-02）各**白付过一次模型加载**：
* 10-01：改了捕获读取的默认路径，却直到 GPU 跑起来才发现旧捕获的键名不同；
* 10-02：四格工具的锚点键（下划线）与格标签（连字符）没做映射，格**算完**才 `KeyError`。
两次都是"配置/读取/命名"类缺陷，**都可以在加载模型之前用真实产物查出来**。

本脚本因此**只做飞行前检查**：用**真实产物**跑一遍 GPU run 将要调用的**同一批函数**
（`f13_i50_cells` 的 `_load_cap` / `_truncate` / `_cache_bitequal` / `_first_diff_pos_of_history` /
`_spec_control_applicability` / `_anchor_ar` / `_anchor_specdec`），核对：
① 捕获读取（长度/层数/`prefix_tokens`/有无 `rows`/有无参考行）；
② `CELLS_PRE`/`CELLS_TARGET`/`DEC_SLOT`/`ROW_INDEX`/`FEED` 的**内在一致性**；
③ 共同前缀是否逐 token 相同、截断是否为**纯切片**；
④ 两份（三份）历史在 `[0,PRE)` 上的逐位关系；
⑤ 三个 **argmax 锚点**的取值是否与运行记录一致（并把锚点打印出来供人工核对）。

**边界**：只读、无 GPU、不写任何文件；失败即退出（由调用方决定是否继续）。
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "output" / "g5" / "attn"
# 与 GPU run **逐项一致**的配置（改这里就等于改将被检查的配置）
os.environ.update(dict(
    CELLS_TAG="div8209", CELLS_PRE="8208", CELLS_TARGET="8209", CELLS_K="4",
    CELLS_CHUNK="2048", CELLS_FEED="20642", CELLS_CANDS="20642,659,576,4531",
    CELLS_WATCH="323,659,20642,576,4531", CELLS_REPEAT="2", CELLS_LEAK_CHECK="1",
    CELLS_TARGET_ROPE="cpu_fp32", CELLS_TRUNC="1", CELLS_PREFIX_TAKE="1",
    CELLS_CONTROL_POLICY="anchors", CELLS_CTX="8192",
    CELLS_CAP_AR=str(OUT / "divcap_ar_20261001T185503Z.pt"),
    CELLS_CAP_SP=str(OUT / "divcap_spec_20261001T185705Z.pt"),
    CELLS_CAP_RB=str(OUT / "divasc_rebuild_20261001T220737Z.pt"),
    CELLS_REF_AR=str(OUT / "smoke8k_eager_20261001T185503Z.json"),
    CELLS_REF_SP=str(OUT / "smoke8k_eager_20261001T185705Z.json"),
    CELLS_REF_RB=str(OUT / "smoke8k_eager_20261001T220737Z.json"),
))

from specdec.engine import f13_i50_cells as C                     # noqa: E402
from specdec.engine import sb_align as _A                         # noqa: E402


def stable(t) -> str:
    """张量摘要（跨进程约定，与其它工具一致）。"""
    return _A.stable_sha(t)

# 期望的锚点（真实运行记录里的 argmax；由运行记录本身提供，不硬编码结论）
EXPECT = dict(H_AR_S=323, H_AR_B=659, H_SP_B=323)


def _lmhead_section() -> None:
    """LM head 边界工具（`lmhead_boundary_f13_p8209`）的飞行前检查（纯 CPU、只读）。"""
    from specdec.engine import lmhead_boundary_f13_p8209 as L
    from specdec.engine import lmhead_boundary as LB
    print("\n================ LM head 边界工具（位置 8209）================")
    print(f"  配置：PRE={L.PRE} TARGET={L.TARGET_POS} DEC_SLOT={L.DEC_SLOT} "
          f"ROW_INDEX={L.ROW_INDEX} FEED={L.FEED} CANDS={L.CANDS} REPEAT={L.REPEAT} "
          f"ROPE={L.TARGET_ROPE}")
    assert L.DEC_SLOT == 1 and L.ROW_INDEX == 0 and len(L.FEED) == 1
    assert L.PRE + L.DEC_SLOT == L.TARGET_POS
    assert L.PRE + 1 + L.ROW_INDEX == L.TARGET_POS
    # ① H 捕获 + 纯切片截断（与已保存四格同一份）
    cap = C._load_cap(Path(L.CAP_H))
    snap = cap["snapshot_cpu"]
    H, t = C._truncate(snap, L.PRE, "H")
    print(f"  H：{Path(L.CAP_H).name} 源长={t['source_len']} → {t['pre']} 纯切片={t['is_pure_slice']} "
          f"层={len(H)} dtype={H[0]['k'].dtype}")
    assert t["is_pure_slice"] and C._cache_len(H) == L.PRE
    # ② 第二份捕获（重建历史）在共同前缀上逐位一致
    if Path(L.CAP_H2).exists():
        H2, _t2 = C._truncate(C._load_cap(Path(L.CAP_H2))["snapshot_cpu"], L.PRE, "H2")
        eq = C._cache_bitequal(H, H2, "H_vs_H2")
        print(f"  H vs 第二份捕获：{eq['n_bitequal']}/{eq['n_tensors']} 逐位相同")
        assert eq["all_equal"]
    # ③ 已保存的局部对照行（用于步骤 1 的控制比较）
    assert Path(L.SAVED_PT).exists(), L.SAVED_PT
    saved = torch.load(L.SAVED_PT, map_location="cpu")
    keys = sorted(k for k, v in saved.items() if torch.is_tensor(v))
    print(f"  已保存对照行：{Path(L.SAVED_PT).name} keys={keys} "
          f"shapes={ {k: list(saved[k].shape) for k in keys} }")
    assert "H_AR-S" in keys and "H_AR-B" in keys
    assert list(saved["H_AR-S"].shape) == [151936]
    # ④ 锚点（真实运行记录的 argmax）
    a_s = C._anchor_ar(L.REF_RUN_AR, L.TARGET_POS)
    a_b = C._anchor_specdec(L.REF_RUN_RB, L.PRE, L.DEC_SLOT, L.TARGET_POS)
    print(f"  锚点：S={a_s.get('token')}（{a_s.get('source')}） / B={a_b.get('token')}（{a_b.get('source')}）")
    assert a_s["available"] and a_s["token"] == 323
    assert a_b["available"] and a_b["token"] == 659
    # ⑤ 边界事实（源码级）+ 投影取行口径
    facts = LB.code_facts()
    assert facts["source_has_expression"] and facts["source_has_norm_call"]
    print(f"  边界：{facts['boundary']}；表达式 `{facts['projection_expression']}`（源码内确认）")
    print("  [lmh-preflight] H/截断/历史一致性/已保存行/锚点/边界事实 全部符合预期 ⇒ 可进入 GPU 运行")


def _norm_section() -> None:
    """最终 Norm 边界工具（`norm_boundary_f13_p8209`）的飞行前检查（纯 CPU、只读）。"""
    from specdec.engine import norm_boundary_f13_p8209 as N
    from specdec.engine import lmhead_boundary as LB
    print("\n================ 最终 Norm 边界工具（位置 8209）================")
    print(f"  配置：PRE={N.PRE} TARGET={N.TARGET_POS} FEED={N.FEED} CANDS={N.CANDS} "
          f"REPEAT={N.REPEAT} ROPE={N.TARGET_ROPE}")
    assert N.DEC_SLOT == 1 and N.ROW_INDEX == 0 and len(N.FEED) == 1
    assert N.PRE + N.DEC_SLOT == N.TARGET_POS
    cap = C._load_cap(Path(N.CAP_H))
    H, t = C._truncate(cap["snapshot_cpu"], N.PRE, "H")
    print(f"  H：{Path(N.CAP_H).name} {t['source_len']} → {t['pre']} 纯切片={t['is_pure_slice']}")
    assert t["is_pure_slice"] and C._cache_len(H) == N.PRE
    if Path(N.CAP_H2).exists():
        H2, _t2 = C._truncate(C._load_cap(Path(N.CAP_H2))["snapshot_cpu"], N.PRE, "H2")
        eq = C._cache_bitequal(H, H2, "H_vs_H2")
        print(f"  H vs 第二份捕获：{eq['n_bitequal']}/{eq['n_tensors']} 逐位相同")
        assert eq["all_equal"]
    saved = torch.load(N.SAVED_PT, map_location="cpu")
    assert "H_AR-S" in saved and "H_AR-B" in saved
    prev = torch.load(N.LMH_PT, map_location="cpu")
    keys = sorted(prev.get("rows", {}).keys())
    print(f"  上一轮 LM head run：{Path(N.LMH_PT).name} rows keys={keys}")
    assert "HS_1" in keys and "HB_4" in keys, "缺少跨 run 交叉核对所需的行"
    a_s = C._anchor_ar(N.REF_RUN_AR, N.TARGET_POS)
    a_b = C._anchor_specdec(N.REF_RUN_RB, N.PRE, N.DEC_SLOT, N.TARGET_POS)
    print(f"  锚点：S={a_s.get('token')} / B={a_b.get('token')}")
    assert a_s["token"] == 323 and a_b["token"] == 659
    ni = LB.code_facts()
    print(f"  Norm 边界：{ni['boundary']}；表达式 `{ni['projection_expression']}`（源码内确认）")
    assert ni["source_has_expression"] and ni["source_has_norm_call"]
    print("  [nb-preflight] H/截断/历史一致/已保存行/跨 run 参考行/锚点/边界事实 全部符合预期 ⇒ 可进入 GPU 运行")


def _layer_section() -> None:
    """L27 边界工具（`layer_boundary_f13_p8209`）的飞行前检查（纯 CPU、只读）。"""
    from specdec.engine import layer_boundary_f13_p8209 as LY
    from specdec.engine import lmhead_boundary as LB
    print("\n================ L27 边界工具（位置 8209）================")
    print(f"  配置：LAYER={LY.LAYER} PRE={LY.PRE} TARGET={LY.TARGET_POS} FEED={LY.FEED} "
          f"CANDS={LY.CANDS} REPEAT={LY.REPEAT} ROPE={LY.TARGET_ROPE}")
    assert LY.DEC_SLOT == 1 and LY.ROW_INDEX == 0 and len(LY.FEED) == 1
    assert LY.PRE + LY.DEC_SLOT == LY.TARGET_POS
    cap = C._load_cap(Path(LY.CAP_H))
    H, t = C._truncate(cap["snapshot_cpu"], LY.PRE, "H")
    print(f"  H：{Path(LY.CAP_H).name} {t['source_len']} → {t['pre']} 纯切片={t['is_pure_slice']} "
          f"层数={len(H)}")
    assert t["is_pure_slice"] and C._cache_len(H) == LY.PRE
    if Path(LY.CAP_H2).exists():
        H2, _t2 = C._truncate(C._load_cap(Path(LY.CAP_H2))["snapshot_cpu"], LY.PRE, "H2")
        eq = C._cache_bitequal(H, H2, "H_vs_H2")
        print(f"  H vs 第二份捕获：{eq['n_bitequal']}/{eq['n_tensors']} 逐位相同")
        assert eq["all_equal"]
    saved = torch.load(LY.SAVED_PT, map_location="cpu")
    assert "H_AR-S" in saved and "H_AR-B" in saved
    nrm = torch.load(LY.NORM_PT, map_location="cpu")
    nkeys = sorted(nrm.get("proj_rows_raw", {}).keys())
    lmh = torch.load(LY.LMH_PT, map_location="cpu")
    lkeys = sorted(lmh.get("rows", {}).keys())
    print(f"  上一轮 Norm run：proj_rows_raw keys={nkeys}；LM head run：rows keys={lkeys}")
    assert {"NA", "ND"} <= set(nkeys), "缺少单行尾部参考（NA/ND）"
    assert {"HS_1", "HB_1"} <= set(lkeys), "缺少单行 LM head 参考（HS_1/HB_1）"
    a_s = C._anchor_ar(LY.REF_RUN_AR, LY.TARGET_POS)
    a_b = C._anchor_specdec(LY.REF_RUN_RB, LY.PRE, LY.DEC_SLOT, LY.TARGET_POS)
    print(f"  锚点：S={a_s.get('token')} / B={a_b.get('token')}")
    assert a_s["token"] == 323 and a_b["token"] == 659
    ni = LB.code_facts()
    assert ni["source_has_expression"] and ni["source_has_norm_call"]
    print("  [lyr-preflight] H/截断/历史一致/已保存行/单行参考（Norm 与 LM head）/锚点/边界事实 全部符合预期 ⇒ 可进入 GPU 运行")


def _chain_section() -> None:
    """全层边界工具（`layer_chain_f13_p8209`）的飞行前检查（纯 CPU、只读）。"""
    from specdec.engine import layer_chain_f13_p8209 as CH
    print("\n================ 全层边界工具（位置 8209）================")
    print(f"  配置：PRE={CH.PRE} TARGET={CH.TARGET_POS} FEED={CH.FEED} CANDS={CH.CANDS} "
          f"REPEAT={CH.REPEAT} ROPE={CH.TARGET_ROPE}")
    assert CH.DEC_SLOT == 1 and CH.ROW_INDEX == 0 and len(CH.FEED) == 1
    assert CH.PRE + CH.DEC_SLOT == CH.TARGET_POS
    cap = C._load_cap(Path(CH.CAP_H))
    H, t = C._truncate(cap["snapshot_cpu"], CH.PRE, "H")
    print(f"  H：{Path(CH.CAP_H).name} {t['source_len']} → {t['pre']} 纯切片={t['is_pure_slice']} "
          f"层数={len(H)}")
    assert t["is_pure_slice"] and C._cache_len(H) == CH.PRE
    if Path(CH.CAP_H2).exists():
        H2, _t2 = C._truncate(C._load_cap(Path(CH.CAP_H2))["snapshot_cpu"], CH.PRE, "H2")
        eq = C._cache_bitequal(H, H2, "H_vs_H2")
        print(f"  H vs 第二份捕获：{eq['n_bitequal']}/{eq['n_tensors']} 逐位相同")
        assert eq["all_equal"]
    assert Path(CH.SAVED_PT).exists()
    l27 = torch.load(CH.L27_PT, map_location="cpu")
    print(f"  上一轮 L27 run 张量：{sorted(k for k in l27 if k.endswith('_raw'))[:6]} ...")
    assert {"aS_raw", "aB4_raw", "zS_raw", "zB4_raw"} <= set(l27), "缺少跨工具核对所需的 L27 行"
    a_s = C._anchor_ar(CH.REF_RUN_AR, CH.TARGET_POS)
    a_b = C._anchor_specdec(CH.REF_RUN_RB, CH.PRE, CH.DEC_SLOT, CH.TARGET_POS)
    print(f"  锚点：S={a_s.get('token')} / B={a_b.get('token')}")
    assert a_s["token"] == 323 and a_b["token"] == 659
    print("  [chain-preflight] H/截断/历史一致/已保存行/L27 跨工具参考/锚点 全部符合预期 ⇒ 可进入 GPU 运行")


def _l0op_section() -> None:
    """L0 算子边界工具（`l0_operator_f13_p8209`）的飞行前检查（纯 CPU、只读）。"""
    from specdec.engine import l0_operator_f13_p8209 as L0T
    from specdec.engine import op_probe as OP
    from specdec.engine import model_self as M
    print("\n================ L0 算子边界工具（位置 8209）================")
    print(f"  配置：LAYER={L0T.LAYER} PRE={L0T.PRE} TARGET={L0T.TARGET_POS} VIS={L0T.VIS} "
          f"REPEAT={L0T.REPEAT} ROPE={L0T.TARGET_ROPE}")
    assert L0T.LAYER == 0 and L0T.VIS == L0T.PRE + 1 == L0T.TARGET_POS
    assert len(L0T.FEED) == 1
    print(f"  探针状态（应为关闭）：{M.attn_probe_state()}")
    assert M.attn_probe_state()["active"] is False, "进入实验前探针必须是关闭的"
    print(f"  链定义：{len(OP.CHAIN)} 个算子、{len(OP.SPECS)} 个探针名"
          f"（其中 shape_only={sorted(k for k in OP.SPECS if OP.SPECS[k][1] == 'shape_only')}）")
    assert len(OP.CHAIN) == len(OP.CHAIN_NODE_OUT)
    for (op, ins), outn in zip(OP.CHAIN, OP.CHAIN_NODE_OUT):
        assert outn in OP.SPECS, (op, outn)
    cap = C._load_cap(Path(L0T.CAP_H))
    H, t = C._truncate(cap["snapshot_cpu"], L0T.PRE, "H")
    print(f"  H：{Path(L0T.CAP_H).name} {t['source_len']} → {t['pre']} 纯切片={t['is_pure_slice']}")
    assert t["is_pure_slice"] and C._cache_len(H) == L0T.PRE
    ch = torch.load(L0T.CHAIN_PT, map_location="cpu")
    assert "rows_S_raw" in ch and "rows_B_raw" in ch, "缺少上一轮全层捕获的 L0 参考行"
    l0_in_s = ch["rows_S_raw"][0]["input"]
    l0_out_s = ch["rows_S_raw"][0]["output"]
    # ⚠️ 口径断言（防复发）：上一轮保存的是 **1 维决策行**，不是 (1,s,hidden)
    assert l0_in_s.dim() == 1 and l0_out_s.dim() == 1, (l0_in_s.shape, l0_out_s.shape)
    assert ch["rows_B_raw"][0]["input"].dim() == 1
    print(f"  上一轮 L0 参考：input sha16={stable(l0_in_s)}、"
          f"output sha16={stable(l0_out_s)}（形状 {list(l0_in_s.shape)} 1 维决策行）；"
          f"L0 输出 S/B n_diff="
          f"{int((l0_out_s.reshape(-1) != ch['rows_B_raw'][0]['output'].reshape(-1)).sum())}")
    assert int((l0_out_s.reshape(-1) != ch["rows_B_raw"][0]["output"].reshape(-1)).sum()) == 111
    a_s = C._anchor_ar(L0T.REF_RUN_AR, L0T.TARGET_POS)
    a_b = C._anchor_specdec(L0T.REF_RUN_RB, L0T.PRE, L0T.TARGET_POS - L0T.PRE, L0T.TARGET_POS)
    print(f"  锚点：S={a_s.get('token')} / B={a_b.get('token')}")
    assert a_s["token"] == 323 and a_b["token"] == 659
    print("  [l0op-preflight] H/截断/历史一致/上一轮 L0 参考/锚点/链与探针表 全部符合预期 ⇒ 可进入 GPU 运行")


def _qk2x2_section() -> None:
    """QKᵀ 2×2 工具（`qk_shape_f13_p8209`）的飞行前检查（纯 CPU、只读）。"""
    from specdec.engine import qk_shape_f13_p8209 as QT
    from specdec.engine import qk_shape_2x2 as QK
    from specdec.engine import model_self as M
    print("\n================ QKᵀ 2×2 工具（位置 8209）================")
    print(f"  配置：LAYER={QT.LAYER} PRE={QT.PRE} VIS={QT.VIS} REPEAT={QT.REPEAT} "
          f"ROPE={QT.TARGET_ROPE}；GQA heads={QK.HEADS} kv_heads={QK.KV_HEADS} "
          f"head_dim={QK.HEAD_DIM} repeat={QK.HEADS // QK.KV_HEADS}")
    assert QT.LAYER == 0 and QT.VIS == QK.N_COMMON == QT.PRE + 1
    assert len(QT.FEED) == 1
    assert M.attn_probe_state()["active"] is False
    print(f"  六格：{[list(c) for c in QK.CELLS]}")
    assert [c[0] for c in QK.CELLS] == ["A", "B", "C", "D", "E", "F"]
    assert [(c[1], c[2]) for c in QK.CELLS] == [(1, 8209), (1, 8212), (4, 8209), (4, 8212),
                                                (4, 8209), (4, 8212)]
    assert (QK.HEAD_DIM ** 0.5) == math.sqrt(QK.HEAD_DIM), "幂与 sqrt 取值不同（需记录）"
    p = torch.load(QT.L0OPS_PT, map_location="cpu")
    need = ["q_rope", "k_visible", "qk_scores_pre_mask"]
    print(f"  上一轮 L0 算子 run：{Path(QT.L0OPS_PT).name}；raw_S 节点数={len(p['raw_S'])}")
    assert all(k in p["raw_S"] and k in p["raw_B"] for k in need), "缺少固定数据所需的 live 张量"
    print(f"    q_rope S={tuple(p['raw_S']['q_rope'].shape)} B={tuple(p['raw_B']['q_rope'].shape)}；"
          f"k_visible S={tuple(p['raw_S']['k_visible'].shape)} B={tuple(p['raw_B']['k_visible'].shape)}")
    assert tuple(p["raw_S"]["q_rope"].shape) == (1, QK.HEADS, 1, QK.HEAD_DIM)
    assert tuple(p["raw_B"]["q_rope"].shape) == (1, QK.HEADS, 4, QK.HEAD_DIM)
    assert tuple(p["raw_S"]["k_visible"].shape) == (1, QK.KV_HEADS, 8209, QK.HEAD_DIM)
    assert tuple(p["raw_B"]["k_visible"].shape) == (1, QK.KV_HEADS, 8212, QK.HEAD_DIM)
    cap = C._load_cap(Path(QT.CAP_H))
    H, t = C._truncate(cap["snapshot_cpu"], QT.PRE, "H")
    print(f"  H：{Path(QT.CAP_H).name} {t['source_len']} → {t['pre']} 纯切片={t['is_pure_slice']}")
    assert t["is_pure_slice"] and C._cache_len(H) == QT.PRE
    print("  [qk2x2-preflight] 配置/六格/幂与 sqrt/上一轮固定数据形状/H 全部符合预期 ⇒ 可进入 GPU 运行")


def _serial_qk_section() -> None:
    """QK 逐行 M=1 causal-prefix 因果干预工具（`serial_qk_l0_f13_p8209`）的飞行前检查（纯 CPU、只读）。"""
    from specdec.engine import serial_qk_l0_f13_p8209 as QT
    from specdec.engine import serial_causal_qk as SQ
    from specdec.engine import model_self as M
    print("\n================ QK 逐行 M=1 因果干预（位置 8209）================")
    print(f"  配置：LAYER={QT.LAYER} PRE={QT.PRE} VIS={QT.VIS} REPEAT={QT.REPEAT} "
          f"ROPE={QT.TARGET_ROPE} POISON={QT.POISON_CHECK}；四臂={'S0,B0,B1,S1'}")
    assert QT.LAYER == 0 and QT.VIS == QT.PRE + 1 and QT.REPEAT >= 2, "四臂与重复次数必须满足判据"
    assert len(QT.FEED) == 1 and len(QT.CANDS) == 4
    # 诊断分支必须**默认关闭**，且 API 齐全
    assert M.serial_qk_state()["active"] is False, "进入实验前诊断分支必须关闭"
    assert hasattr(M, "serial_qk_for") and hasattr(M, "serial_qk_calls") and hasattr(M, "clear_serial_qk")
    for fn in ("row_visible", "row_scores", "serial_causal_scores", "classify_h1", "classify_h2",
               "classify_h3", "classify_h4", "causal_mask"):
        assert hasattr(SQ, fn), f"serial_causal_qk 缺少 {fn}"
    # 纯逻辑口径（不加载模型）：p+i+1、占位值有限、mask 恰好覆盖占位列
    p, s, T = 8208, 4, 8212
    assert [SQ.row_visible(p, i, T) for i in range(s)] == [8209, 8210, 8211, 8212]
    m = SQ.causal_mask(s, T, device=torch.device("cpu"))
    assert all(bool(m[i, SQ.row_visible(p, i, T):].all())
               and not bool(m[i, :SQ.row_visible(p, i, T)].any()) for i in range(s))
    print(f"  逐行 causal 口径：visible={[SQ.row_visible(p, i, T) for i in range(s)]}"
          f"（p={p} s={s} T={T}）✔；mask 恰好覆盖占位列 ✔")
    # 参考产物齐全（本轮所有"必须复现"控制都依赖它们）
    qk2x2 = json.loads(Path(QT.QK2X2_JSON).read_text(encoding="utf-8"))
    coords = qk2x2["attribution"]["observed_A_vs_D"]["coords"]
    assert len(coords) == 3, f"QK 2×2 结论里的差异坐标数不是 3：{coords}"
    l0ops = torch.load(QT.L0OPS_PT, map_location="cpu")
    for k in ("final_row_S", "final_row_B", "L0_output_row_S", "L0_output_row_B", "x0_row_S", "x0_row_B"):
        assert k in l0ops, f"上一轮 L0 算子 run 缺少 {k}"
    assert "qk_scores_pre_mask" in l0ops["raw_S"] and "qk_scores_pre_mask" in l0ops["raw_B"]
    print(f"  参考产物：{Path(QT.L0OPS_PT).name}（S/B 的 final/L0out/x0/QK）+ "
          f"{Path(QT.QK2X2_JSON).name}（3 坐标 {coords}）✔")
    cap = C._load_cap(Path(QT.CAP_H))
    H, t = C._truncate(cap["snapshot_cpu"], QT.PRE, "H")
    assert t["is_pure_slice"] and C._cache_len(H) == QT.PRE and len(H) == 28
    print(f"  H：{Path(QT.CAP_H).name} {t['source_len']} → {t['pre']} 纯切片={t['is_pure_slice']}"
          f"（{len(H)} 层）✔")
    print("  [sqk-preflight] 配置/默认关闭/API/逐行 causal 口径/mask 覆盖/参考产物/H 全部符合预期 "
          "⇒ 可进入 GPU 运行")


def _serial_align_section() -> None:
    """QK + o_proj 联合对齐工具（`serial_align_l0_f13_p8209`）的飞行前检查（纯 CPU、只读）。"""
    from specdec.engine import serial_align_l0_f13_p8209 as QT
    from specdec.engine import serial_align as SA
    from specdec.engine import model_self as M
    print("\n================ QK + o_proj 联合对齐（位置 8209）================")
    plan = [list(p) for p in QT.PLAN]
    print(f"  配置：LAYER={QT.LAYER} PRE={QT.PRE} VIS={QT.VIS} REPEAT={QT.REPEAT} "
          f"ROPE={QT.TARGET_ROPE} POISON={QT.POISON_CHECK}；五臂={plan}")
    assert QT.LAYER == 0 and QT.VIS == QT.PRE + 1 and QT.REPEAT >= 2
    assert [p[0] for p in plan] == ["S0", "B0", "BQ", "BQO", "S1"], "五臂定义不符"
    assert [(p[2], p[3]) for p in plan] == [(False, False), (False, False), (True, False),
                                            (True, True), (True, True)], "两个开关的组合不符"
    assert M.serial_qk_state()["active"] is False and M.serial_linear_state()["active"] is False
    for fn in ("set_serial_linear", "clear_serial_linear", "serial_linear_state", "serial_linear_for",
               "serial_linear_calls"):
        assert hasattr(M, fn), f"model_self 缺少 {fn}"
    for fn in ("module_facts", "serial_linear_rows", "chain_persistence", "classify_h2_oproj",
               "classify_h3_l0", "classify_h4_layers", "classify_h5_final", "mlp_boundary_report"):
        assert hasattr(SA, fn), f"serial_align 缺少 {fn}"
    # 上一轮 BQ 参考（本轮控制 3 依赖它）
    prev = torch.load(QT.PREV_PT, map_location="cpu")
    pj = json.loads(Path(QT.PREV_JSON).read_text(encoding="utf-8"))
    need = ("nodes_B1", "rows_B1", "final_row_B1", "norm_row_B1")
    assert all(k in prev for k in need), f"上一轮 .pt 缺少 {need}"
    assert pj["H2"]["n_diff"] == 61 and pj["H4"]["logits_n_diff"] == 125439
    tok = pj["controls"]["B1_final"]["argmax_id"]
    print(f"  上一轮 BQ 参考：L0 输出 n_diff={pj['H2']['n_diff']}、logits n_diff={pj['H4']['logits_n_diff']}、"
          f"token={tok}、QK row0 exact={pj['H1']['exact_equal']} ✔")
    o_proj = pj["controls"]["c9_intervention_only_L0"]["L0_module_id"] if False else None
    print(f"  本轮要复现的增量：BQ 的 o_proj 决策行在 heads_merged 逐位相同时仍有 2 个差异")
    l0ops = torch.load(QT.L0OPS_PT, map_location="cpu")
    for k in ("final_row_S", "final_row_B", "L0_output_row_S", "L0_output_row_B"):
        assert k in l0ops, f"历史 L0 算子 run 缺少 {k}"
    cap = C._load_cap(Path(QT.CAP_H))
    H, t = C._truncate(cap["snapshot_cpu"], QT.PRE, "H")
    assert t["is_pure_slice"] and C._cache_len(H) == QT.PRE and len(H) == 28
    print(f"  H：{Path(QT.CAP_H).name} {t['source_len']} → {t['pre']} 纯切片={t['is_pure_slice']}"
          f"（{len(H)} 层）✔")
    print("  [sla-preflight] 五臂定义/两个开关组合/默认关闭/API/上一轮 BQ 参考/历史行/H 全部符合预期 "
          "⇒ 可进入 GPU 运行")


def _serial_mlp_section() -> None:
    """L0 全部持续 GEMM 对齐工具（`serial_mlp_l0_f13_p8209`）的飞行前检查（纯 CPU、只读）。"""
    from specdec.engine import serial_mlp_l0_f13_p8209 as QT
    from specdec.engine import serial_align as SA
    from specdec.engine import model_self as M
    print("\n================ L0 全部持续 GEMM 对齐（位置 8209）================")
    plan = [list(p) for p in QT.PLAN]
    print(f"  配置：LAYER={QT.LAYER} PRE={QT.PRE} VIS={QT.VIS} REPEAT={QT.REPEAT} "
          f"ROPE={QT.TARGET_ROPE}；六臂={plan}")
    assert QT.LAYER == 0 and QT.VIS == QT.PRE + 1 and QT.REPEAT >= 2
    assert [p[0] for p in plan] == ["S0", "B0", "BQO", "BQOGU", "BQOGUD", "S1"], "六臂定义不符"
    assert [(p[2], p[3], p[4]) for p in plan] == [
        (False, False, None), (False, False, None), (True, True, None),
        (True, True, "gu"), (True, True, "gud"), (True, True, "gud")], "开关组合不符"
    assert M.serial_qk_state()["active"] is False and M.serial_linear_state()["active"] is False
    for fn in ("set_serial_linear", "serial_linear_for", "serial_linear_state", "serial_linear_calls"):
        assert hasattr(M, fn), f"model_self 缺少 {fn}"
    for fn in ("module_facts", "serial_linear_rows", "chain_persistence", "classify_h5_final"):
        assert hasattr(SA, fn), f"serial_align 缺少 {fn}"
    # 生产代码里三处 MLP 调用的分支必须都在（关闭时逐位等价由门与真机控制证明）
    src = (ROOT / "specdec" / "engine" / "model_self.py").read_text(encoding="utf-8")
    for pat in ("_serial_linear_branch(self.gate_proj, x)",
                "_serial_linear_branch(self.up_proj, x)",
                "_serial_linear_branch(self.down_proj, gp)",
                "_serial_linear_branch(self.o_proj, out)"):
        assert pat in src, f"model_self 缺少分支：{pat}"
    # 上一轮（SLA）参考：本轮控制 3 依赖它
    prev = torch.load(QT.PREV_PT, map_location="cpu")
    pj = json.loads(Path(QT.PREV_JSON).read_text(encoding="utf-8"))
    need = ("nodes_BQO", "rows_BQO", "final_row_BQO", "norm_row_BQO", "oproj_facts")
    assert all(k in prev for k in need), f"上一轮 .pt 缺少 {need}"
    assert pj["H3"]["n_diff"] == 21 and pj["H5"]["category"] == 2
    print(f"  上一轮 BQO 参考：L0 输出 n_diff={pj['H3']['n_diff']}、token={pj['H5']['token_x']}、"
          f"logits n_diff={pj['H5']['logits_n_diff']}、H2 o_proj exact={pj['H2']['o_proj_exact_equal']} ✔")
    _mp = {d["op"]: d for d in pj["MLP_boundary"]["per_operator"]}
    print(f"  上一轮遗留（本轮要消掉的）边界：gate_proj n_diff={_mp['gate_proj']['n_diff']}、"
          f"up_proj={_mp['up_proj']['n_diff']}（均满足「输入逐位相同、输出不同」）")
    l0ops = torch.load(QT.L0OPS_PT, map_location="cpu")
    for k in ("final_row_S", "final_row_B", "L0_output_row_S", "L0_output_row_B"):
        assert k in l0ops, f"历史 L0 算子 run 缺少 {k}"
    cap = C._load_cap(Path(QT.CAP_H))
    H, t = C._truncate(cap["snapshot_cpu"], QT.PRE, "H")
    assert t["is_pure_slice"] and C._cache_len(H) == QT.PRE and len(H) == 28
    print(f"  H：{Path(QT.CAP_H).name} {t['source_len']} → {t['pre']} 纯切片={t['is_pure_slice']}"
          f"（{len(H)} 层）✔")
    print("  [sml-preflight] 六臂定义/开关组合/默认关闭/API/四处生产分支/上一轮参考/历史行/H 全部符合预期 "
          "⇒ 可进入 GPU 运行")


def main() -> int:
    print(f"配置：PRE={C.PRE} TARGET={C.TARGET_POS} DEC_SLOT={C.DEC_SLOT} ROW_INDEX={C.ROW_INDEX} "
          f"CTX={C.CTX} FEED={C.FEED} CANDS={C.CANDS} WATCH={C.WATCH}")
    print(f"      TRUNC={C.TRUNC} PREFIX_TAKE={C.PREFIX_TAKE} POLICY={C.CONTROL_POLICY} "
          f"REPEAT={C.REPEAT} LEAK={C.LEAK_CHECK} ROPE={C.TARGET_ROPE}")
    assert C.DEC_SLOT == len(C.FEED) == 1, (C.DEC_SLOT, C.FEED)
    assert C.ROW_INDEX == 0 and C.PRE + C.DEC_SLOT == C.TARGET_POS

    ca, cs = C._resolve_caps()
    cap_ar, cap_sp = C._load_cap(ca), C._load_cap(cs)
    cap_rb = C._load_cap(Path(C.CAP_RB))
    print("\n== 捕获读取 ==")
    for tag, cap in (("H_AR", cap_ar), ("H_SP", cap_sp), ("H_RB", cap_rb)):
        snap = cap["snapshot_cpu"]
        print(f"  {tag}: len={C._cache_len(snap)} 层={len(snap)} "
              f"rows={'有' if cap.get('rows') is not None else '无'} "
              f"prefix_tokens={len(C._mv(cap, 'prefix_tokens') or [])} "
              f"参考行 `{C._ar_row_key()}`={'有' if C._ar_row_key() in cap else '无'}")

    need = C.TARGET_POS - C.CTX
    pa = list(C._mv(cap_ar, "prefix_tokens") or [])[:need]
    ps = list(C._mv(cap_sp, "prefix_tokens") or [])[:need]
    print(f"\n== prefix_tokens（只取前 {need} 个；捕获里原长 "
          f"{[len(C._mv(cap_ar, 'prefix_tokens') or []), len(C._mv(cap_sp, 'prefix_tokens') or [])]}）==")
    print(f"  相等={pa == ps}\n  H_AR={pa}\n  H_SP={ps}")
    assert pa == ps and len(pa) == need, "共同前缀逐 token 不相同或长度不符"

    sa, ta = C._truncate(cap_ar["snapshot_cpu"], C.PRE, "H_AR")
    ss, ts = C._truncate(cap_sp["snapshot_cpu"], C.PRE, "H_SP")
    sr, tr = C._truncate(cap_rb["snapshot_cpu"], C.PRE, "H_RB")
    print("\n== 截断（纯切片）==")
    for t in (ta, ts, tr):
        print(f"  {t['tag']}: {t['source_len']} → {t['pre']}，纯切片={t['is_pure_slice']}")
        assert t["is_pure_slice"]

    h_rb = C._cache_bitequal(sa, sr, "H_AR_vs_H_RB")
    h_sp = C._cache_bitequal(sa, ss, "H_AR_vs_H_SP")
    fd = C._first_diff_pos_of_history(sa, ss)
    print(f"\n== 历史关系（[0,{C.PRE})）==")
    print(f"  H_AR vs H_RB：{h_rb['n_bitequal']}/{h_rb['n_tensors']} 逐位相同，"
          f"全等={h_rb['all_equal']}，不同元素={h_rb['n_diff_total']}")
    print(f"  H_AR vs H_SP：{h_sp['n_bitequal']}/{h_sp['n_tensors']} 逐位相同，"
          f"不同元素={h_sp['n_diff_total']}，max|Δ|={h_sp['max_abs']:.6g}，"
          f"最早分叉位置={fd['first_diff_pos_any_layer']}")
    assert h_rb["all_equal"], "重建历史与 H_AR 在共同前缀上并非逐位相同 ⇒ H_AR-B 的锚点不可用"

    ctrl = C._spec_control_applicability(cap_sp, rows_present=cap_sp.get("rows") is not None,
                                         pre=C.PRE, row_index=C.ROW_INDEX, target_pos=C.TARGET_POS)
    print(f"\n== 逐元素参考的适用性 ==\n  H_SP-B 的 rows 参考：{json.dumps(ctrl, ensure_ascii=False)}")

    print("\n== argmax 锚点（真实运行记录）==")
    got = dict(H_AR_S=C._anchor_ar(C.REF_RUN_AR, C.TARGET_POS),
               H_AR_B=C._anchor_specdec(C.REF_RUN_RB, C.PRE, C.DEC_SLOT, C.TARGET_POS),
               H_SP_B=C._anchor_specdec(C.REF_RUN_SP, C.PRE, C.DEC_SLOT, C.TARGET_POS))
    for tag, a in got.items():
        assert a["available"], (tag, a)
        assert a["token"] == EXPECT[tag], (tag, a["token"], EXPECT[tag])
        print(f"  {tag}: 真实={a['token']}（{a['source']}）")
    print("[preflight] 四格工具：配置、捕获读取、截断、历史关系、参考适用性、三个锚点全部符合预期")
    _lmhead_section()
    _norm_section()
    _layer_section()
    _chain_section()
    _l0op_section()
    _qk2x2_section()
    _serial_qk_section()
    _serial_align_section()
    _serial_mlp_section()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
