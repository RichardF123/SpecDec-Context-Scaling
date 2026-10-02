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
import os
import sys
from pathlib import Path

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

# 期望的锚点（真实运行记录里的 argmax；由运行记录本身提供，不硬编码结论）
EXPECT = dict(H_AR_S=323, H_AR_B=659, H_SP_B=323)


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
    print("[preflight] 配置、捕获读取、截断、历史关系、参考适用性、三个锚点全部符合预期 ⇒ 可进入 GPU 运行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
