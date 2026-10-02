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
# 锚点键（下划线，JSON 友好）→ 格标签（连字符，既有约定）的**显式**映射
_ANCHOR_CELL_TAG = {"H_AR_S": "H_AR-S", "H_AR_B": "H_AR-B", "H_SP_S": "H_SP-S", "H_SP_B": "H_SP-B"}
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
# prompt 长度（第一个生成 token 的绝对位置）：`prefix_tokens` 的长度口径 = TARGET_POS - CTX
CTX = int(os.environ.get("CELLS_CTX", "8192"))

# ---------------------------------------------------------------------------------
# 2026-10-02 新增（**默认不改变原行为**）：把同一套四格逻辑用到"更早的位置"（新首分歧 8209）上。
# 三个开关都只在显式设置时才生效。
# ---------------------------------------------------------------------------------
# `CELLS_TRUNC=1`：先把两份快照沿 token 维**纯切片**到 PRE 再跑格。
#   用途：捕获长度 8237 > 新分歧轮的 PRE=8208 ⇒ 必须先在**共同前缀**上切一刀。
#   纪律：切片**不重算**任何位置，逐张量核对切片 == 源前缀；源长度另记，且"声明的捕获长度"
#   改为对**源**长度成立（否则会把"截断"误报成"快照长度不符"）。
TRUNC = S.env_flag("CELLS_TRUNC", False)
# `CELLS_PREFIX_TAKE=1`：`prefix_tokens` 只取前 `TARGET_POS - CTX` 个。
#   用途：捕获里的 `prefix_tokens` 长度是 47（为位置 8239 准备的），本轮的格只需要前 17 个
#   （位置 8192..8208）。原长度另记；**相等性要求不变**（仍要求两臂逐 token 相同）。
PREFIX_TAKE = S.env_flag("CELLS_PREFIX_TAKE", False)
# `CELLS_CONTROL_POLICY`：`strict`（默认 ⇒ 行为与原来完全一致：缺**逐元素**参考行即失败）
#   | `anchors`（缺逐元素参考行时，改用**真实运行记录里的 argmax 锚点**并显式标注
#   "argmax 级、非逐元素复现"。**绝不**编造参考张量，也**不**把锚点冒充逐元素控制）。
CONTROL_POLICY = os.environ.get("CELLS_CONTROL_POLICY", "strict").strip().lower()
# 第三个运行记录（重建运行的 smoke 记录）：给 `H_AR-B` 提供 argmax 锚点。
REF_RUN_RB = os.environ.get("CELLS_REF_RB", "")
# 第三份捕获（重建快照）：**只**用于核对"它与 H_AR 在 [0,PRE) 上是否逐位相同"，
# 从而判断"用重建运行的记录给 H_AR-B 当锚点"是否成立。
CAP_RB = os.environ.get("CELLS_CAP_RB", "")


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
                raw_dtype=str(r.dtype), raw_shape=list(r.shape),
                # ⚠️ 报告陷阱（2026-10-02 实测踩到）：`torch.topk` 在**并列**时给谁排第一是未定义的，
                #   而 `torch.argmax` 返回**最小索引**的那个并列者。所以"top5[0]"与"argmax"
                #   在并列（margin=0）时**可以不是同一个 token**，看起来像矛盾，其实不是。
                tie_note="`margin_bf16` = 该格**自身 logits** 的 top1-top2（统一 `.float()` 后比较）；"
                         "0 表示并列，此时 `argmax` 取**最小索引**而 `topk[0]` 的顺序未定义 ⇒ "
                         "以 `argmax`（引擎判定用的就是它）为准，`top5[0]` 只作参考")


def _cmp(a, b, name) -> dict:
    x, y = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    A.require(x.numel() == y.numel(), f"{name}: 长度不一致")
    d = (x - y).abs()
    return dict(name=name, elementwise_equal=bool(torch.equal(x, y)), max_abs=float(d.max()),
                mean_abs=float(d.mean()), n_diff=int((d > 0).sum()),
                a_argmax=int(torch.argmax(x)), b_argmax=int(torch.argmax(y)),
                same_argmax=bool(int(torch.argmax(x)) == int(torch.argmax(y))))


# ----------------------------------------------------- 2026-10-02 新增的只读小工具
def _truncate(snap, pre: int, tag: str) -> tuple:
    """把快照沿 token 维**纯切片**到长度 `pre`（`v[:, :, :pre]`），并逐张量核对"切片 == 源前缀"。

    * 为什么需要：本轮新分歧轮的 `PRE=8208` 比捕获长度 8237 短 ⇒ 必须先在**共同前缀**上切片；
    * 为什么不重算：切片是视图/拷贝，**不经过任何前向** ⇒ 位置 `0..pre-1` 的数值就是捕获时的数值；
    * 返回 `(切片, 记录)`；`记录` 里写明源长度、核对结果与"非张量键被保留"。
    """
    src_len = _cache_len(snap)
    A.require(src_len >= pre, f"{tag}: 源快照长度 {src_len} < PRE={pre}（无法截断）")
    out = [{k: (v[:, :, :pre] if torch.is_tensor(v) and v.dim() == 4 else v)
            for k, v in lay.items()} for lay in snap]
    pure = bool(all(torch.equal(out[i][kk], snap[i][kk][:, :, :pre])
                    for i in range(len(snap)) for kk in ("k", "v")))
    A.require(pure, f"{tag}: 截断后与源前缀不逐位相同（截断不是纯切片）")
    return out, dict(tag=tag, source_len=src_len, pre=pre, is_pure_slice=pure,
                     n_layers=len(out), keys=sorted(out[0].keys()),
                     note="纯切片（`v[:,:, :pre]`），未重算任何位置；逐张量 `torch.equal` 核对通过")


def _first_diff_pos_of_history(a, b) -> dict:
    """两份**等长**历史最早在哪个**位置**出现数值差异（归约维 = (kv_heads, head_dim)，再取位置）。"""
    A.require(len(a) == len(b), "两份历史层数不同")
    first, per_layer = None, []
    for i in range(len(a)):
        A.require(tuple(a[i]["k"].shape) == tuple(b[i]["k"].shape),
                  f"第 {i} 层 K 形状不同（无法逐位比较）")
        f = None
        for kk in ("k", "v"):
            neq = (a[i][kk] != b[i][kk])
            pos_any = neq.any(dim=1).any(dim=-1).reshape(-1)      # ⇒ 留下 (T,)
            nz = torch.nonzero(pos_any, as_tuple=False)
            if nz.numel():
                p = int(nz.reshape(-1)[0])
                f = p if f is None else min(f, p)
        per_layer.append(dict(layer=i, first_diff_pos=f))
        if f is not None:
            first = f if first is None else min(first, f)
    return dict(first_diff_pos_any_layer=first,
                n_layers_diverging=int(sum(1 for x in per_layer if x["first_diff_pos"] is not None)),
                per_layer_first=[x for x in per_layer if x["first_diff_pos"] is not None][:6])


def _cache_bitequal(a, b, name: str) -> dict:
    """两份等长缓存的**逐张量** `torch.equal`（不是 sha 摘要）。"""
    A.require(len(a) == len(b), f"{name}: 层数不同")
    n = 0
    neq = 0
    ndiff = 0
    mx = 0.0
    for i in range(len(a)):
        for kk in ("k", "v"):
            x, y = a[i][kk], b[i][kk]
            A.require(tuple(x.shape) == tuple(y.shape), f"{name}: 第 {i} 层 {kk} 形状不同")
            n += 1
            if torch.equal(x, y):
                neq += 1
            else:
                ndiff += int((x != y).sum())
                mx = max(mx, float((x.float() - y.float()).abs().max()))
    return dict(name=name, n_tensors=n, n_bitequal=neq, all_equal=bool(neq == n),
                n_diff_total=ndiff, max_abs=mx)


def _spec_control_applicability(cap_sp: dict, *, rows_present: bool, pre: int, row_index: int,
                                target_pos: int) -> dict:
    """判断捕获里的 `rows` 能否充当**本格 H_SP-B 的逐元素参考**（三条缺一不可）。

    ① 捕获里确有 `rows`；
    ② 这批 rows 出自**长度 = pre** 的缓存：
       新格式读 `meta.block_round.pre`；旧格式读 `meta.pre`（`div_pre`/`div_post` 是**同一轮**）；
    ③ `rows[row_index]` 预测的位置 `pre + 1 + row_index` 必须**正是** `target_pos`。

    ⚠️ **为什么必须有 ③**（2026-10-02 修正的陷阱）：logits 的**词表长度对所有位置都一样**，
    所以"拿一行预测 8238 的行去和预测 8209 的行比"在长度检查上**完全看不出来**，
    只会得到一个 `elementwise_equal=False` 的"控制失败"，把配置错配误报成数值/实现问题。
    """
    meta = cap_sp.get("_meta") or {}
    br = meta.get("block_round")
    if isinstance(br, dict) and "pre" in br:
        rows_pre, src = int(br["pre"]), "meta.block_round.pre"
    elif meta.get("pre") is not None:
        rows_pre, src = int(meta["pre"]), "meta.pre（旧格式：div_pre/div_post 同一轮）"
    else:
        rows_pre, src = None, "缺失（无法判定 ⇒ 不适用）"
    predicts = (None if rows_pre is None else rows_pre + 1 + row_index)
    ok = bool(rows_present and rows_pre == pre and predicts == target_pos)
    if not rows_present:
        why = "捕获里没有 `rows`"
    elif rows_pre is None:
        why = "捕获没有声明产出 rows 的那一轮的 pre"
    elif rows_pre != pre:
        why = f"那批 rows 来自长度 {rows_pre} 的缓存（本格 PRE={pre}）"
    elif predicts != target_pos:
        why = f"那批 rows[{row_index}] 预测位置 {predicts}，不是本格 TARGET_POS={target_pos}"
    else:
        why = None
    return dict(applicable=ok, rows_pre=rows_pre, rows_pre_source=src, rows_predicts=predicts,
                rows_index_used=row_index, target_pos=target_pos, reason=why)


def _anchor_ar(path: str, target_pos: int) -> dict:
    """AR 运行记录里的**串行单步**锚点：`run.gen[target_pos - CTX]`（argmax 级）。"""
    if not path or not Path(path).exists():
        return dict(available=False, why="未提供 AR 运行记录（CELLS_REF_AR）")
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    g = [int(x) for x in (j.get("run") or {}).get("gen", [])]
    i = target_pos - CTX
    if not (0 <= i < len(g)):
        return dict(available=False, why=f"记录里 gen 只有 {len(g)} 个 token，取不到 index {i}")
    return dict(available=True, elementwise=False, token=g[i], index=i, position=target_pos,
                fed_token=(g[i - 1] if i - 1 >= 0 else None),
                source=f"{Path(path).name}: run.gen[{i}]",
                semantics="真实 AR 运行的**串行单步**在同一历史（逐位相同）上产出的 token —— "
                          "**argmax 级**锚点，不是整行逐元素参考")


def _anchor_specdec(path: str, pre: int, dec_slot: int, target_pos: int) -> dict:
    """SpecDec 运行记录里的**块验证**锚点：`rounds[pre==PRE].g[dec_slot]`（= `rows[dec_slot-1]` 的 argmax）。"""
    if not path or not Path(path).exists():
        return dict(available=False, why="未提供该运行的记录")
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    hit = next((r for r in (j.get("rounds") or []) if int(r.get("pre", -1)) == pre), None)
    if hit is None:
        return dict(available=False, why=f"记录里没有 pre={pre} 的轮次")
    g = list(hit.get("g") or [])
    if dec_slot >= len(g):
        return dict(available=False, why=f"该轮 g 只有 {len(g)} 项，取不到 g[{dec_slot}]")
    return dict(available=True, elementwise=False, token=int(g[dec_slot]),
                round=hit.get("round"), pre=pre, Kp=hit.get("Kp"), m=hit.get("m"),
                cands=list(hit.get("cands") or []), g=[int(x) for x in g],
                source=f"{Path(path).name}: rounds[pre={pre}].g[{dec_slot}]（= rows[{dec_slot - 1}] 的 argmax）",
                semantics="真实 SpecDec 运行的**块验证** `rows[%d]` 的 argmax（预测位置 %d）—— "
                          "**argmax 级**锚点，不是整行逐元素参考" % (dec_slot - 1, target_pos))



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
        A.require(len(feed) == DEC_SLOT and _cache_len(cache) == TARGET_POS,
                  f"{tag}: 串行格喂入 {len(feed)} 个 token 后末位行应预测 {TARGET_POS}",
                  f"DEC_SLOT={DEC_SLOT} cache_len={_cache_len(cache)}")
        row = rows[-1]
        extra["cache_len_after"] = _cache_len(cache)
        extra["fed"] = feed
        # 2026-10-02 新增（只加字段，不改数值）：串行格的**输入位置 / 可见范围 / 行来源**
        extra["predicted_pos"] = _cache_len(cache)
        extra["visible_last_key"] = _cache_len(cache) - 1
        extra["row_source"] = f"串行 S：喂入 {feed} 后的**末位行**（预测位置 {_cache_len(cache)}）"
        extra["exec_helper"] = ("`spec_dec._step`（与 AR 模式同调用形态：`torch.inference_mode()` + "
                               "`M.attention_phase('decode')` + `_run(..., need_logits=True)`）")
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
                           target_rope=TARGET_ROPE, ctx=CTX, trunc=TRUNC,
                           prefix_take=PREFIX_TAKE, control_policy=CONTROL_POLICY,
                           cap_rb=(CAP_RB or None), ref_run_rb=(REF_RUN_RB or None),
                           diag_precision="qk_fp32=False, prob_fp32=False"),
               semantics="同前缀四格局部对照（历史来源 × 执行方式）；**不做**自由生成；"
                         "四格各自的输入前缀相同（到 PRE-1 为止），但**分块/串行各自新增 KV 后"
                         "数值历史可能不同**。",
               field_naming="不使用 own_max_abs/prev_max_abs/next_max_abs/fp32_margin；"
                            "所有比较按比较对命名；不调用 classify_bf16_compare（缺 fp32_margin）。")
    print(f"[cells] run_id={run_id} pre={PRE} target={TARGET_POS} dec_slot={DEC_SLOT} "
          f"feed={FEED} cands={CANDS} repeat={REPEAT} leak={LEAK_CHECK} rope={TARGET_ROPE} "
          f"trunc={TRUNC} prefix_take={PREFIX_TAKE} control_policy={CONTROL_POLICY}",
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
        cap_rb = _load_cap(Path(CAP_RB)) if CAP_RB else None
        rows_sp = cap_sp.get("rows")
        rows_ar = cap_ar.get("rows")
        _ar_key = _ar_row_key()
        _ar_row_available = bool(_ar_key in cap_ar)
        rec["sources"] = dict(capture_ar=str(ca), capture_spec=str(cs),
                              capture_third=(str(CAP_RB) if CAP_RB else None),
                              ar_row_key=_ar_key,
                              ar_reference_row_present=_ar_row_available,
                              captured_AR_row_sha=(A.stable_sha(cap_ar[_ar_key])
                                                   if _ar_row_available else None),
                              captured_SP_rows_sha=(A.stable_sha(rows_sp[ROW_INDEX])
                                                    if rows_sp is not None else None),
                              captured_AR_rows_present=rows_ar is not None,
                              AR_capture_meta=cap_ar.get("_meta") or {},
                              SP_capture_meta=cap_sp.get("_meta") or {},
                              RB_capture_meta=((cap_rb or {}).get("_meta") or {}),
                              control_policy=CONTROL_POLICY)
        pref_ar = list(_mv(cap_ar, "prefix_tokens") or [])
        pref_sp = list(_mv(cap_sp, "prefix_tokens") or [])
        A.require(pref_ar and pref_sp, "捕获缺少 prefix_tokens（在 JSON 元数据里）",
                  f"ar={len(pref_ar)} sp={len(pref_sp)}")
        _pref_raw = [len(pref_ar), len(pref_sp)]
        if PREFIX_TAKE:
            # 2026-10-02：捕获里的 `prefix_tokens` 是**更长**的一段（为更后的位置准备的）；
            # 本轮的格只需要到 `TARGET_POS-1` 为止 ⇒ 只取前 `TARGET_POS-CTX` 个。
            # 这是**截取**（不重算、不改内容），原长度记录在 `prefix.n_raw`。
            _need = TARGET_POS - CTX
            A.require(min(_pref_raw) >= _need,
                      f"prefix_tokens 太短：需要前 {_need} 个", str(_pref_raw))
            pref_ar, pref_sp = pref_ar[:_need], pref_sp[:_need]
        _prefix_equal = bool(pref_ar == pref_sp)
        if PREFIX_POLICY == "strict":
            A.require(_prefix_equal, "两次捕获的 token 前缀不一致",
                      f"{pref_ar[:6]} vs {pref_sp[:6]}")
        else:
            A.require(len(pref_ar) == len(pref_sp), "两臂前缀长度不一致（无法逐 token 对位）",
                      f"{len(pref_ar)} vs {len(pref_sp)}")
        A.require(len(pref_ar) == TARGET_POS - CTX,
                  f"前缀长度应为 {TARGET_POS - CTX}", str(len(pref_ar)))
        rec["prefix"] = dict(n=len(pref_ar), tokens=pref_ar, equal=_prefix_equal,
                             policy=PREFIX_POLICY, tokens_sp=pref_sp,
                             n_raw=_pref_raw, taken=(TARGET_POS - CTX) if PREFIX_TAKE else None,
                             take_note=("只取前 %d 个（位置 %d..%d）；捕获里的更长尾巴与本格无关"
                                        % (TARGET_POS - CTX, CTX, TARGET_POS - 1)
                                        if PREFIX_TAKE else None),
                             note=("两臂捕获的共同前缀逐 token 相同（到 TARGET_POS-1 为止）"
                                   if _prefix_equal else
                                   "⚠️ 两臂逻辑前缀**不同** ⇒ 下面的四格**不是**同一前缀对照，"
                                   "只能作为「不同历史 + 不同前缀」的描述"))
        snap_ar, snap_sp = cap_ar["snapshot_cpu"], cap_sp["snapshot_cpu"]
        A.require(snap_ar is not None and snap_sp is not None, "捕获缺少 snapshot_cpu")
        src_len = dict(ar=_cache_len(snap_ar), sp=_cache_len(snap_sp))
        rec["truncation"] = dict(applied=False, pre=PRE, source_len=src_len)
        if TRUNC:
            # 2026-10-02：捕获长度（8237）> 本轮 PRE（8208）⇒ 先沿 token 维**纯切片**到 PRE。
            snap_ar, t_ar = _truncate(snap_ar, PRE, "H_AR")
            snap_sp, t_sp = _truncate(snap_sp, PRE, "H_SP")
            rec["truncation"] = dict(
                applied=True, pre=PRE, source_len=src_len, ar=t_ar, sp=t_sp,
                note="沿 token 维纯切片到 PRE（**不重算**任何位置）；切片与源前缀逐张量 `torch.equal` "
                     "核对通过；源长度另记 ⇒ 「快照长度与捕获声明长度一致」的检查改为对**源长度**成立")
        A.require(_cache_len(snap_ar) == PRE and _cache_len(snap_sp) == PRE,
                  "两份快照长度必须都是 PRE", f"{_cache_len(snap_ar)} / {_cache_len(snap_sp)}")
        # ---- 2026-10-02 新增：第三份快照（重建历史）与 H_AR 在 [0,PRE) 上是否**逐位相同** ----
        # 为什么需要：本轮要用"重建运行"的记录给 `H_AR-B` 当 argmax 锚点，前提是
        # **重建历史与 H_AR 在被截断的这段上逐位相同**。这一点必须在本 run 里**实测**，
        # 不能靠另一份 CPU 脚本的结论口头带过。
        if cap_rb is not None:
            snap_rb = cap_rb["snapshot_cpu"]
            A.require(snap_rb is not None, "第三份捕获缺少 snapshot_cpu")
            rb_src_len = _cache_len(snap_rb)
            if TRUNC:
                snap_rb, _t_rb = _truncate(snap_rb, PRE, "H_RB")
            A.require(_cache_len(snap_rb) == PRE, "第三份快照截断后长度应为 PRE",
                      str(_cache_len(snap_rb)))
            rec["histories"] = dict(
                pre=PRE, source_len=dict(ar=src_len["ar"], sp=src_len["sp"], rb=rb_src_len),
                h_ar_vs_h_rb=_cache_bitequal(snap_ar, snap_rb, "H_AR_vs_H_RB"),
                h_ar_vs_h_sp=_cache_bitequal(snap_ar, snap_sp, "H_AR_vs_H_SP"),
                h_ar_vs_h_sp_first_div=_first_diff_pos_of_history(snap_ar, snap_sp),
                note="`H_AR_vs_H_RB` 全真 ⇒ 「用重建运行的记录给 H_AR-B 当锚点」成立；"
                     "`H_AR_vs_H_SP` 的**最早分叉位置**说明两份历史从哪里开始不同（token 内容相同、"
                     "KV 构造不同）；这里只描述**是否/从哪**不同，不做算子级归因")
            del snap_rb
            print(f"[cells] 历史一致性：H_AR vs H_RB 逐位相同 "
                  f"{rec['histories']['h_ar_vs_h_rb']['n_bitequal']}/"
                  f"{rec['histories']['h_ar_vs_h_rb']['n_tensors']}；"
                  f"H_AR vs H_SP 最早分叉位置="
                  f"{rec['histories']['h_ar_vs_h_sp_first_div']['first_diff_pos_any_layer']}")
        else:
            rec["histories"] = dict(pre=PRE, third_capture=None,
                                   note="未提供第三份捕获（CELLS_CAP_RB）⇒ 不做 H_AR/H_RB 一致性核对，"
                                        "也就**不**使用重建记录给 H_AR-B 当锚点")
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
        # 声明长度是**源**捕获的长度；截断模式下比对的必须是源长度（否则会把"截断"误报成"长度不符"）
        _sp_ref = src_len["sp"] if TRUNC else _cache_len(snap_sp)
        _ar_ref = src_len["ar"] if TRUNC else _cache_len(snap_ar)
        A.require(_sp_ref == int(sp_pre_rec), "Spec 快照长度与捕获记录不符",
                  f"{_sp_ref} vs {sp_pre_rec}（TRUNC={TRUNC}）")
        A.require(_ar_ref == int(ar_len_rec), "AR 快照长度与捕获记录不符",
                  f"{_ar_ref} vs {ar_len_rec}（TRUNC={TRUNC}）")
        rec["snapshot_integrity"] = dict(
            ar_len=_cache_len(snap_ar), sp_len=_cache_len(snap_sp), n_layers=[len(snap_ar), len(snap_sp)],
            keys_ar=sorted(snap_ar[0].keys()), keys_sp=sorted(snap_sp[0].keys()),
            sp_declared_pre=sp_pre_rec, pre_source=pre_src,
            ar_declared_len=ar_len_rec, ar_len_source=ar_len_src,
            truncated=bool(TRUNC), source_len=src_len,
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
        _sctrl = _spec_control_applicability(cap_sp, rows_present=rows_sp is not None, pre=PRE,
                                             row_index=ROW_INDEX, target_pos=TARGET_POS)
        _rows_pre = _sctrl["rows_pre"]
        _rows_pre_src = _sctrl["rows_pre_source"]
        _rows_predicts = _sctrl["rows_predicts"]
        _sp_ctrl_ok = bool(_sctrl["applicable"])
        # ---- 逐元素控制（默认 strict：缺参考行就失败；`anchors` 策略下改为记录 + argmax 锚点）----
        _ar_elem_avail = bool(_ar_row_available)
        if _ar_elem_avail or CONTROL_POLICY == "strict":
            _ar_elem = _cmp(res["H_AR-S"], cap_ar[_ar_key], "H_AR-S_vs_captured_AR")
        else:
            _ar_elem = None
        rec["controls"] = dict(
            H_AR_S_vs_captured_AR_row=_ar_elem,
            H_SP_B_vs_captured_SP_row=(_cmp(res["H_SP-B"], rows_sp[ROW_INDEX],
                                            "H_SP-B_vs_captured_SP")
                                       if rows_sp is not None else None),
            H_SP_B_control_applicable=_sp_ctrl_ok,
            H_SP_B_control_rows_pre=_rows_pre,
            H_SP_B_control_rows_pre_source=_rows_pre_src,
            H_SP_B_control_rows_predicts=_rows_predicts,
            H_SP_B_rows_present=rows_sp is not None,
            note="只有这两格有对应的**真实捕获行**；H_AR-B / H_SP-S 是干预组合，"
                 "真实运行里没有发生 ⇒ 不设参考、不编造参考。"
                 "Spec 侧控制要求：捕获里确有 `rows`，且这批 rows 出自 **PRE** 长度的缓存、"
                 "`rows[ROW_INDEX]` 预测的正是 **TARGET_POS**"
                 "（否则那批 rows 是别的长度/别的槽位的行，词表长度相同 ⇒ 长度检查抓不住这种错配）。")
        if not _ar_elem_avail:
            rec["controls"]["H_AR_S_elementwise_control"] = dict(
                available=False, policy=CONTROL_POLICY, ar_row_key=_ar_key,
                reason=(f"该捕获里没有 `{_ar_key}`（AR 侧只在一处位置存了参考行；"
                        f"本轮的 TARGET_POS={TARGET_POS} 处没有捕获整行）"),
                fallback=("用**真实运行记录**里的 argmax 锚点代替（argmax 级，**不是**逐元素复现）"
                          if CONTROL_POLICY == "anchors" else "strict 策略 ⇒ 视作失败"))
            if CONTROL_POLICY == "anchors":
                print(f"[cells] 控制 ⚠ 无 `{_ar_key}` 逐元素参考行 ⇒ 改用 argmax 锚点（argmax 级）")
            else:
                A.require(False, f"strict 策略要求存在逐元素参考行 `{_ar_key}`",
                          "若本轮确实没有捕获整行，请显式设置 CELLS_CONTROL_POLICY=anchors")
        if _ar_elem is not None:
            A.require(_ar_elem["elementwise_equal"],
                      "控制失败：H_AR-S 未逐元素复现真实 AR 决策行", str(_ar_elem))
        if _sp_ctrl_ok:
            A.require(rec["controls"]["H_SP_B_vs_captured_SP_row"]["elementwise_equal"],
                      "控制失败：H_SP-B 未逐元素复现真实 Spec 决策行",
                      str(rec["controls"]["H_SP_B_vs_captured_SP_row"]))
            print("[cells] 控制 ✔ H_AR-S == 真实 AR 行、H_SP-B == 真实 Spec 决策行（均逐元素）")
        else:
            rec["controls"]["H_SP_B_control_skipped"] = dict(
                reason=_sctrl["reason"] or "不适用",
                rows_present=rows_sp is not None, rows_pre=_rows_pre, PRE=PRE,
                rows_pre_source=_rows_pre_src, rows_predicts=_rows_predicts,
                rows_index_used=ROW_INDEX, target_pos=TARGET_POS)
            print(f"[cells] 控制 ⚠ H_SP-B 的**逐元素**参考**不适用**（rows_present={rows_sp is not None}，"
                  f"rows_pre={_rows_pre}，PRE={PRE}，rows 预测位置={_rows_predicts}）；"
                  f"改用 argmax 锚点（若策略允许）")
        # ---- 2026-10-02 新增：**argmax 锚点**（真实运行记录里的 argmax，非逐元素参考）----
        anch = dict(
            H_AR_S=_anchor_ar(REF_RUN_AR, TARGET_POS),
            H_SP_B=_anchor_specdec(REF_RUN_SP, PRE, DEC_SLOT, TARGET_POS),
            H_AR_B=(_anchor_specdec(REF_RUN_RB, PRE, DEC_SLOT, TARGET_POS)
                    if (REF_RUN_RB and cap_rb is not None) else
                    dict(available=False,
                         why=("未提供重建运行记录（CELLS_REF_RB）"
                              if not REF_RUN_RB else
                              "未提供第三份捕获（CELLS_CAP_RB）⇒ 无法核对「重建历史 == H_AR」，"
                              "故不使用该锚点"))),
            H_SP_S=dict(available=False, why="真实运行里没有「H_SP 历史 + 串行单步」这一组合 ⇒ 无锚点"))
        _rb_hist_ok = bool(rec.get("histories", {}).get("h_ar_vs_h_rb", {}).get("all_equal"))
        if anch["H_AR_B"].get("available") and not _rb_hist_ok:
            anch["H_AR_B"] = dict(available=False,
                                  why="重建历史与 H_AR 在 [0,PRE) 上**并非逐位相同** ⇒ 该锚点不适用于 H_AR")
        for tag, a in anch.items():
            if a.get("available"):
                # ⚠️ 锚点键用下划线（JSON 友好），格标签用连字符（既有约定）——必须显式映射，
                # 不能想当然（首跑就是因为这里写成 `summaries[tag]` 而在**格计算完成之后**崩掉：
                # 数值没丢，但完整产物没落盘 ⇒ 白付一次模型加载）。
                a["cell_tag"] = _ANCHOR_CELL_TAG[tag]
                a["measured_argmax"] = int(summaries[_ANCHOR_CELL_TAG[tag]]["argmax"])
                a["match"] = bool(int(a["measured_argmax"]) == int(a["token"]))
        rec["argmax_anchors"] = dict(
            cells=anch, third_capture_present=cap_rb is not None,
            history_identity_ok_for_H_AR_B=_rb_hist_ok,
            semantics="锚点 = **真实运行记录**里该位置的 token/argmax；用于核对「本格复现了真实运行的"
                      "**token 选择**」。**不是**整行逐元素复现 ⇒ 不能据此声称完整 logits 复现现场；"
                      "也不得把锚点当成逐元素控制。")
        for tag, a in anch.items():
            if a.get("available"):
                print(f"[cells] 锚点 {tag}: 真实={a['token']} 本格={a['measured_argmax']} "
                      f"一致={a['match']}（{a['source']}）")
            else:
                print(f"[cells] 锚点 {tag}: 不适用（{a.get('why')}）")
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
                 "后续 token 相同**不等于**缓存状态相同。",
            reading_rules=(
                "① 并列（`margin_bf16 == 0`）时 `torch.argmax` 取**最小索引**、`torch.topk` 的并列顺序"
                "未定义 ⇒ 以 `argmax` 为准（引擎判定用的就是 `argmax`），`top5[0]` 只作参考，"
                "两者在并列时不同**不是**矛盾；"
                "② 本工具**不**调用 `classify_bf16_compare`、**不**给 `fp32_margin`、**不**新增精度实验 ⇒ "
                "只能报告「测到的 logits 差是多少」，**不能**报告「差多少才算数值误差」；"
                "③ `same_history_S_vs_B_*` 是**同一份快照**上的对照 ⇒ 该历史、该位置上有效；"
                "别处/别的部分的内容差**不**否定它，它**也**不自动推广到别的位置或别份历史；"
                "④ 锚点只证明「复现了**token 选择**（argmax 级）」，**不**证明整行逐元素复现。"))
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
    check("★ 串行格记录**输入位置/可见范围/行来源**（末位行预测 TARGET_POS、只看到 ≤ TARGET_POS-1）",
          ex_s["predicted_pos"] == TARGET_POS and ex_s["visible_last_key"] == TARGET_POS - 1
          and "exec_helper" in ex_s and "row_source" in ex_s, ex_s)

    # ---- 2026-10-02 新增：截断 / 逐位比较 / argmax 锚点（全部纯 CPU）----
    t3, tinfo = _truncate(snap, 3, "t")
    check("★ 截断：纯切片到长度 3、逐张量核对通过、源长度记录在案",
          _cache_len(t3) == 3 and tinfo["is_pure_slice"] and tinfo["source_len"] == PRE
          and sorted(t3[0].keys()) == sorted(snap[0].keys()), tinfo)
    _raised = False
    try:
        _truncate(snap, PRE + 4, "t")
    except Exception:                                              # noqa: BLE001
        _raised = True
    check("★ 截断：请求长度大于源长度时**拒绝**（不静默补零/复制）", _raised)
    _same = _cache_bitequal(snap, CC.clone_list_cache_full(snap), "t_same")
    _mod = CC.clone_list_cache_full(snap)
    _before = _mod[1]["v"][0, 0, 2, 0].clone()
    _mod[1]["v"][0, 0, 2, 0] = _before + 1.0
    _mut_ok = bool(not torch.equal(_mod[1]["v"][0, 0, 2, 0], _before))
    _diff = _cache_bitequal(snap, _mod, "t_diff")
    _fd = _first_diff_pos_of_history(snap, _mod)
    check("★ 逐位比较：克隆 vs 源全真；改一个元素后可见且**最早位置=2**（阳性对照成立）",
          _same["all_equal"] and _mut_ok and not _diff["all_equal"]
          and _diff["n_diff_total"] >= 1 and _fd["first_diff_pos_any_layer"] == 2,
          (_same, _mut_ok, _diff, _fd))
    _js = OUTDIR / f"_selftest_anchor_{run_id}.json"
    _js.write_text(json.dumps(dict(run=dict(gen=[100, 101, 102, 103]),
                                   rounds=[dict(round=0, pre=2, Kp=4, m=1, cands=[1, 2, 3, 4],
                                                g=[100, 222, 3, 4])]), ensure_ascii=False),
                   encoding="utf-8")
    a_ar = _anchor_ar(str(_js), CTX + 2)
    a_sp = _anchor_specdec(str(_js), 2, 1, CTX + 3)
    a_no = _anchor_specdec(str(_js), 99, 1, CTX + 3)
    check("★ argmax 锚点：从真实运行记录取 `gen[..]` / `rounds[pre=..].g[..]`；缺失时**如实标不适用**",
          a_ar["available"] and a_ar["token"] == 102 and a_ar["elementwise"] is False
          and a_sp["available"] and a_sp["token"] == 222 and a_sp["round"] == 0
          and (not a_no["available"]) and "pre=99" in a_no["why"], (a_ar, a_sp, a_no))
    _js.unlink()
    # Spec 侧逐元素参考的**适用性**：同长度同槽位才适用；换长度/换槽位一律不适用
    _c_new = _spec_control_applicability({"_meta": dict(block_round=dict(pre=8237))},
                                         rows_present=True, pre=8237, row_index=1, target_pos=8239)
    _c_old = _spec_control_applicability({"_meta": dict(pre=8237)},
                                         rows_present=True, pre=8237, row_index=1, target_pos=8239)
    _c_bad = _spec_control_applicability({"_meta": dict(pre=8237)},
                                         rows_present=True, pre=8208, row_index=0, target_pos=8209)
    _c_none = _spec_control_applicability({"_meta": dict(pre=8237)},
                                          rows_present=False, pre=8237, row_index=1, target_pos=8239)
    check("★ Spec 侧逐元素参考**适用性**：同长度同槽=适用（新旧两种格式都识别）；"
          "换长度/换槽/无 rows=不适用（且给出具体原因）",
          _c_new["applicable"] and _c_old["applicable"] and (not _c_bad["applicable"])
          and _c_bad["rows_predicts"] == 8238 and (not _c_none["applicable"])
          and "8208" in _c_bad["reason"], (_c_new, _c_old, _c_bad, _c_none))
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
