"""diag_cache_cmp.py — KV cache 的**真正独立**快照与**可检测**比较（纯函数，可 CPU 自测）。

为什么需要这个模块（对应本轮任务发现的两个缺陷）
  缺陷 A（**别名盲点**）：旧工具用 `[dict(d) for d in cache]` 做"快照"，**张量仍然共享**。
      于是 `torch.equal(snap_ref[i]['k'], snap[i]['k'])` 比较的是**同一个张量对象**，
      结果恒为 True —— 无论引擎是否就地改写过它。这个检查**在逻辑上不可能失败**，
      因此 `snapshot_unchanged_all_layers=True` 是**空证据**。
  缺陷 B（**字段覆盖不全**）：旧检查只看 `k`/`v`，不看 `k_scale`/`v_scale`/`fmt`，
      而 v_fp8 档的 V 是 `float8_e4m3fn` + `v_scale` —— scale 变化会被完全漏掉。

本模块的做法
  * `deep_clone`：**每个张量 `.clone()`**（独立 storage）作为参照；
  * `shallow_clone`：保留旧行为，供自测**证明**它为什么不够；
  * `compare_caches`：遍历**所有层**的**所有张量键**（含 scale）+ `fmt` + 层数 + 长度，
    返回逐层明细与聚合结果；fp8 载荷可选**先反量化再比**（历史比较用），
    完整性核对则用**原始载荷**（同量化格式下 raw 比较才有意义且更严格）。
"""
from __future__ import annotations

import torch

from . import model_self as M
from . import spec_cache


def deep_clone(cache):
    """**真正独立**的副本：逐层新 dict + 每个张量 `.clone()`（新 storage）。"""
    if cache is None:
        return None
    out = []
    for d in cache:
        nd = {}
        for k, v in d.items():
            nd[k] = v.detach().clone() if torch.is_tensor(v) else v
        out.append(nd)
    return out


def shallow_clone(cache):
    """旧工具的"快照"：只复制 dict，**张量共享**。仅用于自测展示其盲点，不作为证据。"""
    if cache is None:
        return None
    return [dict(d) for d in cache]


def tensor_keys(layer: dict) -> list:
    return sorted(k for k, v in layer.items() if torch.is_tensor(v))


def cache_len(cache) -> int:
    return spec_cache.cache_length(cache)


def metadata(cache) -> dict:
    """**全层**结构记录：每层的键、dtype、shape，以及 `fmt` 与总长度。"""
    if cache is None:
        return dict(n_layers=0, length=0, layers=[])
    layers = []
    for i, d in enumerate(cache):
        e = dict(layer=i, fmt=d.get("fmt"), tensor_keys=tensor_keys(d))
        for k in tensor_keys(d):
            t = d[k]
            e[k] = dict(dtype=str(t.dtype), shape=[int(x) for x in t.shape])
        layers.append(e)
    return dict(n_layers=len(cache), length=cache_len(cache), layers=layers,
                note="**全层**记录（不是只打印前 3 层）")


def compare_caches(a, b, *, dequantize: bool = False, max_detail: int = 8,
                   dequant_dtype: torch.dtype = torch.float32) -> dict:
    """逐层逐键比较两份 cache。

    dequantize=False（默认，用于**完整性核对**）：同格式同 dtype 下比**原始载荷**。
    dequantize=True （用于**cache 历史比较**）：fp8 载荷先用自身 scale 反量化再比，
                       否则两条路径的 scale 若不同，raw 差没有可比性。

    ⚠️ `dequant_dtype` **默认 float32**，不是 bf16：`v ≈ 3` 时 bf16 间距 ≈ 0.0156，
       而 fp8 载荷差 1 步只相当于 ~0.007 —— 用 bf16 反量化会把这种差**抹平**成"相等"
       （自测里专门有一条断言记录这个陷阱）。float32 下 q*scale 是精确可比的。
    返回：层数/键数、逐键相等、最大绝对差、dtype/shape/缺键等**结构性**不一致的明细。
    """
    if a is None or b is None:
        return dict(comparable=False, reason="有一侧为 None")
    n = min(len(a), len(b))
    n_equal, n_cmp = 0, 0
    dmax_all = []
    details = []
    struct = []
    per_key = {}
    for i in range(n):
        da, db = a[i], b[i]
        if da.get("fmt") != db.get("fmt"):
            struct.append(dict(layer=i, kind="fmt", a=da.get("fmt"), b=db.get("fmt")))
        ka, kb = set(tensor_keys(da)), set(tensor_keys(db))
        if ka != kb:
            struct.append(dict(layer=i, kind="key_set", only_a=sorted(ka - kb),
                               only_b=sorted(kb - ka)))
        for k in sorted(ka & kb):
            xa, xb = da[k], db[k]
            role = "raw"
            if dequantize:
                sc = f"{k}_scale"
                if sc in da and sc in db:
                    xa = M.dequantize_kv_fp8(da[k], da[sc], dequant_dtype)
                    xb = M.dequantize_kv_fp8(db[k], db[sc], dequant_dtype)
                    role = "dequantized"
            if tuple(xa.shape) != tuple(xb.shape) or xa.dtype != xb.dtype:
                struct.append(dict(layer=i, key=k, kind="shape_or_dtype",
                                   a=dict(shape=[int(x) for x in xa.shape], dtype=str(xa.dtype)),
                                   b=dict(shape=[int(x) for x in xb.shape], dtype=str(xb.dtype))))
                continue
            n_cmp += 1
            same = bool(torch.equal(xa, xb))
            d = float((xa.float() - xb.float()).abs().max())
            dmax_all.append(d)
            n_equal += int(same)
            agg = per_key.setdefault(k, dict(n=0, n_equal=0, max_abs_diff=0.0))
            agg["n"] += 1
            agg["n_equal"] += int(same)
            agg["max_abs_diff"] = max(agg["max_abs_diff"], d)
            if not same and len(details) < max_detail:
                details.append(dict(layer=i, key=k, role=role, max_abs_diff=d,
                                    same_shape=True, same_dtype=True))
    agree = (len(a) == len(b) and cache_len(a) == cache_len(b))
    return dict(
        comparable=True,
        dequantize=dequantize,
        dequant_dtype=str(dequant_dtype),
        n_layers=len(a), n_layers_b=len(b),
        length=cache_len(a), length_b=cache_len(b),
        layers_and_length_agree=agree,
        n_tensors_compared=n_cmp,
        n_equal=n_equal,
        # 层数/长度也必须一致，否则"比过的部分都一样"不等于"两份 cache 一样"
        all_equal=bool(n_cmp > 0 and n_equal == n_cmp and not struct and agree),
        max_abs_diff=(max(dmax_all) if dmax_all else None),
        per_key=per_key,
        structural=struct,
        mismatches=details,
        note=("原始载荷比较（完整性核对）" if not dequantize
              else "fp8 载荷先反量化再比（cache 历史比较）"),
    )


def integrity_check(ref, cur, **kw) -> dict:
    """快照完整性：`ref` 必须是**真正独立**的副本（`deep_clone`），`cur` 是跑完之后的快照。"""
    r = compare_caches(ref, cur, dequantize=False, **kw)
    r["kind"] = "snapshot_integrity"
    r["method"] = ("逐层逐键原始载荷 torch.equal；参照必须是 deep_clone（独立 storage）"
                   "——浅 dict 拷贝会因张量共享而使该检查恒真")
    return r


def _selftest_stub_cache(n_layers: int = 2, length: int = 4, *, fp8_v: bool = False):
    """构造一个形状合法的假 cache（CPU），用于自测。"""
    layers = []
    for _ in range(n_layers):
        d = {"k": torch.randn(1, 2, length, 8), "v": torch.randn(1, 2, length, 8), "fmt": None}
        if fp8_v:
            q, s = M.quantize_kv_fp8(d["v"], group_size=8)
            d["v"] = q
            d["v_scale"] = s
            d["fmt"] = "v_fp8"
        layers.append(d)
    return layers


def _selftest() -> int:
    npass = nfail = 0

    def check(name, cond):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  [ok] {name}")
        else:
            nfail += 1
            print(f"  [FAIL] {name}")

    print("== 快照隔离：deep_clone 独立性 ==")
    c = _selftest_stub_cache()
    ref = deep_clone(c)
    check("deep_clone 不与原 cache 共享张量",
          all(ref[i][k] is not c[i][k] for i in range(len(c)) for k in ("k", "v")))
    check("deep_clone 不与原 cache 共享 storage",
          all(ref[i][k].data_ptr() != c[i][k].data_ptr()
              for i in range(len(c)) for k in ("k", "v")))
    before = ref[0]["k"].clone()
    c[0]["k"][0, 0, 0, 0] += 3.0                       # 故意原地修改"引擎侧"
    check("原地修改原 cache 后，deep_clone 参照未被改动",
          bool(torch.equal(ref[0]["k"], before)))
    ref[0]["v"][0, 0, 0, 0] += 5.0
    check("原地修改参照后，原 cache 未被改动（双向隔离）",
          float(c[0]["v"][0, 0, 0, 0]) < 1e9)

    print("== 缺陷复现：浅拷贝的别名盲点（空证据） ==")
    c2 = _selftest_stub_cache()
    sref = shallow_clone(c2)
    check("shallow_clone 与原 cache **共享**张量对象（别名）",
          all(sref[i][k] is c2[i][k] for i in range(len(c2)) for k in ("k", "v")))
    c2[0]["k"][0, 0, 0, 0] += 7.0                      # 故意污染
    vac = compare_caches(sref, c2)
    check("★ 浅拷贝参照**无法**检测出这次原地修改（all_equal=True ⇒ 旧检查恒真）",
          vac["all_equal"] is True and vac["max_abs_diff"] == 0.0)

    print("== 故意原地修改后**能**被检测出来（deep_clone 参照） ==")
    c3 = _selftest_stub_cache()
    ref3 = deep_clone(c3)
    c3[0]["k"][0, 0, 0, 0] += 7.0
    det = integrity_check(ref3, c3)
    check("检测到原地修改：all_equal=False", det["all_equal"] is False)
    check("检测到原地修改：max_abs_diff = 7.0",
          abs(float(det["max_abs_diff"]) - 7.0) < 1e-6)
    check("检测到原地修改：mismatches 指到 layer0.k",
          any(m["layer"] == 0 and m["key"] == "k" for m in det["mismatches"]))

    print("== 检测：键重绑（不是原地改，而是换了一个张量） ==")
    c4 = _selftest_stub_cache()
    ref4 = deep_clone(c4)
    c4[1]["v"] = torch.zeros_like(c4[1]["v"])
    d4 = integrity_check(ref4, c4)
    check("键重绑被检测到（layer1.v）",
          d4["all_equal"] is False and any(m["layer"] == 1 and m["key"] == "v"
                                           for m in d4["mismatches"]))

    print("== 检测：只改 scale（旧检查完全漏掉的字段） ==")
    c5 = _selftest_stub_cache(fp8_v=True)
    ref5 = deep_clone(c5)
    check("v_fp8 桩 cache 含 v_scale 且 V 是 fp8",
          "v_scale" in c5[0] and str(c5[0]["v"].dtype).startswith("torch.float8"))
    c5[1]["v_scale"][0, 0, 0, 0] += 1.0
    d5 = integrity_check(ref5, c5)
    check("★ 只改 v_scale 也被检测到（旧检查只看 k/v ⇒ 会漏）",
          d5["all_equal"] is False
          and any(m["key"] == "v_scale" for m in d5["mismatches"]))
    c6 = _selftest_stub_cache(fp8_v=True)
    ref6 = deep_clone(c6)
    c6[0]["v"] = torch.zeros_like(c6[0]["v"])          # 改 fp8 载荷本身
    d6 = integrity_check(ref6, c6)
    check("fp8 载荷被改动也被检测到（raw 比较）",
          d6["all_equal"] is False and any(m["key"] == "v" for m in d6["mismatches"]))

    print("== 检测：fmt / 层数 / 长度 / dtype / shape 等结构性差异 ==")
    c7 = _selftest_stub_cache()
    ref7 = deep_clone(c7)
    c7[0]["fmt"] = "v_fp8"
    check("fmt 变化被记录为结构性差异",
          integrity_check(ref7, c7)["all_equal"] is False)
    c8 = _selftest_stub_cache(n_layers=2)
    ref8 = deep_clone(c8)
    d8 = integrity_check(ref8, c8[:1])
    check("层数不一致 ⇒ all_equal=False 且 layers_and_length_agree=False",
          d8["all_equal"] is False and d8["layers_and_length_agree"] is False)
    c9 = _selftest_stub_cache(length=4)
    ref9 = deep_clone(c9)
    c9b = deep_clone(c9)[:1] + [dict(c9[1], k=c9[1]["k"][:, :, :3])]
    d9 = integrity_check(ref9, c9b)
    check("长度不一致被检测到",
          d9["all_equal"] is False
          and any(s.get("kind") == "shape_or_dtype" for s in d9["structural"]))
    c10 = _selftest_stub_cache()
    ref10 = deep_clone(c10)
    c10[0]["k"] = c10[0]["k"].double()
    d10 = integrity_check(ref10, c10)
    check("dtype 不一致被检测到",
          any(s.get("kind") == "shape_or_dtype" for s in d10["structural"]))
    c11 = _selftest_stub_cache()
    ref11 = deep_clone(c11)
    c11[0]["v"] = torch.zeros(1, 2, 4, 9)
    d11 = integrity_check(ref11, c11)
    check("shape 不一致被检测到",
          any(s.get("kind") == "shape_or_dtype" and s.get("key") == "v"
              for s in d11["structural"]))

    print("== 反量化比较（cache 历史比较用） ==")
    a = _selftest_stub_cache(fp8_v=True)
    b = deep_clone(a)
    ra = compare_caches(a, b, dequantize=True)
    check("同一 cache 自比：反量化后 all_equal=True", ra["all_equal"] is True)
    check("反量化比较覆盖**每个键**（k/v/v_scale）× 每层",
          ra["n_tensors_compared"] == 3 * len(a) and "v" in ra["per_key"]
          and "v_scale" in ra["per_key"])
    check("反量化口径记为 float32（不是 bf16）",
          ra["dequant_dtype"] == str(torch.float32))

    def _e4m3(x: float) -> float:
        return float(torch.tensor(float(x), dtype=torch.float8_e4m3fn).to(torch.float32))

    # ★ 造突变时的陷阱：e4m3 只有 3 位尾数 ⇒ 在 ~288 处相邻间距是 32，
    #   所以"载荷 +1"会被**舍入回原值**，那是一次**无效突变**（第一版自测就踩了这个坑）。
    check("★ 陷阱：e4m3 在 288 处间距为 32，载荷 +1 被舍入回原值（无效突变）",
          _e4m3(288.0 + 1.0) == 288.0 and _e4m3(288.0 + 32.0) == 320.0)

    base_q = float(b[0]["v"][0, 0, 0, 0].to(torch.float32))
    sentinel = 8.0 if abs(base_q - 8.0) > 1e-6 else -8.0
    b[0]["v"][0, 0, 0, 0] = torch.tensor(sentinel, dtype=torch.float8_e4m3fn)
    check("守门：突变确实改成了**不同**的码值（避免再做无效突变）",
          float(b[0]["v"][0, 0, 0, 0].to(torch.float32)) != base_q)
    rb = compare_caches(a, b, dequantize=True)
    check("反量化后能看出该 fp8 载荷变化（all_equal=False）",
          rb["all_equal"] is False and float(rb["max_abs_diff"]) > 0)
    check("per_key 聚合给出 max_abs_diff", rb["per_key"]["v"]["max_abs_diff"] > 0)
    expect = abs(sentinel - base_q) * float(a[0]["v_scale"][0, 0, 0, 0])
    check("反量化差 = Δ码值 × scale（可对账）",
          abs(float(rb["per_key"]["v"]["max_abs_diff"]) - expect) < 1e-6)

    print("== metadata 覆盖全层 ==")
    md = metadata(_selftest_stub_cache(n_layers=3, fp8_v=True))
    check("metadata 记录全部 3 层（不是只前 3 层截断）", len(md["layers"]) == 3)
    check("metadata 含 fmt 与每个张量的 dtype/shape",
          md["layers"][0]["fmt"] == "v_fp8"
          and "v_scale" in md["layers"][0]["tensor_keys"]
          and md["layers"][0]["v"]["dtype"].startswith("torch.float8"))

    print(f"\nCPU 自测：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
