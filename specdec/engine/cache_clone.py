"""cache_clone.py — HF `DynamicCache` 的**可靠克隆**与 8 项 CPU 硬测试（可 `python -m` 运行自测）。

## 为什么需要它（原来做了什么、为什么错）
`xb_probe.py` 早期自己手写了一个 clone：新建 `DynamicLayer` 后只填 `keys`/`values`。
本机 transformers（v5，`DynamicLayer.__dict__ = ['device','dtype','is_initialized','keys','values']`）
里 `is_initialized` 是一个**普通布尔属性**：新实例为 `False` ⇒ `get_seq_length()` **返回 0**，
第一次 `update()` 会**重新初始化并清空**已填入的 K/V。
**实测（本机）**：旧的 xb clone 克隆 6-token 历史后 `get_seq_length()==0`、`is_initialized==False`，
追加 1 个 token 后长度为 **1**（应为 7），且**前 6 位 K 没有保留**。
⇒ 那次运行的两臂**并不是从 8240-token 历史出发**，其下游对照无效。

## 现在如何修
统一改用 `clone_cache_full()`：**`copy.deepcopy` 保留完整对象状态**（含 `is_initialized`、
`device`、`dtype` 及各层元数据），不再手写"只复制部分字段"的版本。
`require_clone_invariants()` 把 8 项不变量做成**可复用断言**。

## 已被验证为正确的旧实现
`hf_serial_block_probe.clone_cache` 在 `type(c)()` 后**额外拷贝了 `is_initialized` 等元数据**，
实测（本机）克隆后 `get_seq_length()==6`、追加后 7、前 6 位 K 保留 ⇒ **它的历史结果不受该 bug 影响**。
本模块仍统一改用 deepcopy 版，以免"两种克隆并存"。

**引擎侧（自研 `list[dict]`）当时并没有这个 bug**：那次运行里层 dict 只含 `k`/`v` 两个张量键，
`[{k: v.clone()}]` 能正确复制；只有 HF 的 `DynamicCache` 因为多带 `is_initialized` 元数据而出错。
`spec_verify_index_selftest_cpu.clone_cache` 的浅拷贝 `[dict(d)]` 也**不是** bug：它由该自测第 ⑤ 项
显式验证过"引擎从不就地改写旧张量、只把新张量重新绑定到键上"。
"""
from __future__ import annotations

import copy

import torch


class CloneError(Exception):
    pass


def require(cond, what: str, detail=""):
    if not cond:
        raise CloneError(f"{what} 未满足。{detail}".strip())
    return True


def clone_cache_full(cache):
    """**保留完整对象状态**的克隆（deepcopy）。张量与标量元数据一起独立复制。"""
    return copy.deepcopy(cache)


def clone_cache_old_xb(cache):
    """**旧的错误实现**（只填 keys/values）—— 仅用于回归测试证明"新测试能抓住它"。"""
    new = type(cache)()
    for lay in cache.layers:
        d = lay.__class__()
        d.keys = lay.keys.detach().clone()
        d.values = lay.values.detach().clone()
        new.layers.append(d)
    return new


def clone_list_cache_full(cache):
    """自研引擎缓存（`list[dict]`，每层一个 dict）的克隆。

    必须用 deepcopy 而不是 `[{k: v.clone() for k, v in d.items()}]`：层 dict 里**可能有非张量键**
    （`fmt` 等），后者会 `AttributeError: 'str' object has no attribute 'clone'`。
    本机实际跑过的 xb 运行里缓存 dict 只含 `k`/`v`，所以那次引擎侧克隆**没有错**（错只在 HF 侧）。
    """
    return copy.deepcopy(cache)


def clone_list_cache_old_xb(cache):
    """**旧的引擎侧写法**（对每个值调 `.clone()`）——仅用于回归测试。"""
    return [{k: v.clone() for k, v in d.items()} for d in cache]


def require_list_clone_invariants(cache, clone, *, L: int, append_fn=None, n: int = 1) -> dict:
    """引擎 `list[dict]` 缓存的克隆不变量（与 HF 版同构：位级、长度、键集合、无共享存储、追加）。

    1. 层数一致；2. 每层键集合一致（**含非张量键**，值相等）；
    3. 所有张量项逐位相同且形状/类型一致；4. `k` 的序列维 == L；
    5. 源与克隆、克隆与克隆之间不共享张量存储；
    6. 追加 n 个 token 后长度 = L+n 且前 L 位保持；7. 推进副本不影响源与兄弟副本。
    """
    require(len(clone) == len(cache), "克隆层数不一致", f"{len(clone)} vs {len(cache)}")
    n_tensors = 0
    for i, (a, b) in enumerate(zip(cache, clone)):
        require(set(a) == set(b), f"L{i} 键集合不一致", str(sorted(set(a) ^ set(b))))
        for kk, ta in a.items():
            tb = b[kk]
            if not torch.is_tensor(ta):
                require(tb == ta, f"L{i}.{kk} 非张量元数据不一致", f"{ta!r} vs {tb!r}")
                continue
            require(torch.is_tensor(tb) and tuple(ta.shape) == tuple(tb.shape)
                    and ta.dtype == tb.dtype, f"L{i}.{kk} 形状/类型不一致")
            require(torch.equal(ta, tb), f"L{i}.{kk} 与源不是逐位相同")
            n_tensors += 1
    k0 = clone[0].get("k")
    require(torch.is_tensor(k0), "L0 缺少张量键 k")
    require(int(k0.shape[2]) == L, "K 的序列维与长度不一致", f"{int(k0.shape[2])} vs {L}")
    shared = [f"src-vs-clone L{i}.{kk}"
              for i in range(len(cache)) for kk, ta in cache[i].items()
              if torch.is_tensor(ta) and torch.is_tensor(clone[i].get(kk))
              and ta.data_ptr() == clone[i][kk].data_ptr()]
    require(not shared, "源与克隆共享张量存储", str(shared[:4]))
    app = {}
    if append_fn is not None:
        src_len0 = int(cache[0]["k"].shape[2])
        src_k0 = cache[0]["k"].detach().clone()
        other = clone_list_cache_full(clone)
        before = clone[0]["k"].detach().clone()
        append_fn(clone, n)
        require(int(clone[0]["k"].shape[2]) == L + n, "追加后长度应为 L+n",
                f"{int(clone[0]['k'].shape[2])} vs {L + n}")
        require(bool(torch.equal(clone[0]["k"][:, :, :L, :], before[:, :, :L, :])),
                "追加后前 L 个位置的 K 未保持原值")
        require(int(cache[0]["k"].shape[2]) == src_len0 and torch.equal(cache[0]["k"], src_k0),
                "推进副本影响了源缓存")
        require(int(other[0]["k"].shape[2]) == L, "推进一个副本影响了另一个副本")
        app = dict(len_after=int(clone[0]["k"].shape[2]), history_preserved=True,
                   source_untouched=True, sibling_untouched=True)
    return dict(n_layers=len(cache), L=L, n_tensors=n_tensors, appended=app)


def require_clone_invariants(cache, clone, *, L: int, append_fn=None, n: int = 1) -> dict:
    """**8 项克隆不变量**（`append_fn(cache, n) -> cache` 用于第 5/6/7 项）。

    1. 各层 K/V 与源逐位相同；2. `get_seq_length()` 与 K 的序列维一致；
    3. 初始化状态与必要元数据正确；4. 源与副本、副本之间（K 与 V）都不共享存储；
    5. 从长度 L 的副本追加 n 个 token 后长度 = L+n；6. 追加后前 L 个位置的 K/V 保持原值；
    7. 推进一个副本不影响源与另一个副本；8. 由调用方另行断言（旧实现必须被抓出）。
    """
    require(len(clone.layers) == len(cache.layers), "克隆层数不一致")
    # 1 + 2
    for i, (a, b) in enumerate(zip(cache.layers, clone.layers)):
        for kk in ("keys", "values"):
            ta, tb = getattr(a, kk), getattr(b, kk)
            require(ta is not None and tb is not None, f"L{i}.{kk} 缺失")
            require(torch.equal(ta, tb), f"L{i}.{kk} 与源不是逐位相同")
    require(clone.get_seq_length() == L, "克隆的 get_seq_length 与预期长度不符",
            f"{clone.get_seq_length()} vs {L}")
    require(int(clone.layers[0].keys.shape[-2]) == L, "K 的序列维与长度不一致",
            f"{int(clone.layers[0].keys.shape[-2])} vs {L}")
    # 3
    metas = []
    for i, b in enumerate(clone.layers):
        require(bool(getattr(b, "is_initialized", False)),
                f"L{i} 的 is_initialized=False（旧 clone 的典型症状）")
        metas.append(dict(is_initialized=bool(b.is_initialized),
                          device=str(getattr(b, "device", None)),
                          dtype=str(getattr(b, "dtype", None))))
    require(all(m["device"] == metas[0]["device"] and m["dtype"] == metas[0]["dtype"]
                for m in metas), "各层 device/dtype 元数据不一致")
    # 4
    shared = []
    for i in range(len(cache.layers)):
        for kk in ("keys", "values"):
            if getattr(cache.layers[i], kk).data_ptr() == getattr(clone.layers[i], kk).data_ptr():
                shared.append(f"src-vs-clone L{i}.{kk}")
    require(not shared, "源与克隆共享存储", str(shared[:4]))
    # 5 + 6 + 7（需要 append_fn）
    app = {}
    if append_fn is not None:
        src_len0 = cache.get_seq_length()
        src_k0 = cache.layers[0].keys.detach().clone()
        other = clone_cache_full(clone)
        before = clone.layers[0].keys.detach().clone()
        append_fn(clone, n)
        require(clone.get_seq_length() == L + n, "追加后长度应为 L+n",
                f"{clone.get_seq_length()} vs {L + n}")
        require(bool(torch.equal(clone.layers[0].keys[:, :, :L, :], before[:, :, :L, :])),
                "追加后前 L 个位置的 K 未保持原值")
        require(cache.get_seq_length() == src_len0 and torch.equal(cache.layers[0].keys, src_k0),
                "推进副本影响了源缓存")
        require(other.get_seq_length() == L, "推进一个副本影响了另一个副本")
        app = dict(len_after=clone.get_seq_length(), history_preserved=True,
                   source_untouched=True, sibling_untouched=True)
    return dict(n_layers=len(cache.layers), L=L, metas=metas, appended=app,
                old_impl_note="旧实现（只填 keys/values）会 is_initialized=False、get_seq_length=0，"
                              "追加后被重新初始化 ⇒ 由自测第 8 项抓出")


# ----------------------------------------------------------------- CPU 自测
def _selftest() -> int:
    import copy as _copy

    from transformers import Qwen2Config, Qwen2ForCausalLM

    npass = nfail = 0

    def check(name, cond, extra=None):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  [ok] {name}")
        else:
            nfail += 1
            print(f"  [FAIL] {name}" + (f"  {str(extra)[:200]}" if extra is not None else ""))

    def raises(fn):
        try:
            fn()
            return False
        except CloneError:
            return True

    torch.manual_seed(0)                      # ★ 种子在模型初始化**之前**
    cfg = Qwen2Config(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=256)
    cfg._attn_implementation = "eager"
    m = Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval()
    ids = torch.randint(0, 64, (1, 6))
    with torch.inference_mode():
        o = m(ids, use_cache=True)
    C = o.past_key_values
    L = 6

    print("== 1) 元数据核查（本机库里 is_initialized 是普通属性）==")
    lay = C.layers[0]
    check("DynamicLayer.__dict__ 含 is_initialized/keys/values/device/dtype",
          set(lay.__dict__) >= {"is_initialized", "keys", "values"}, sorted(lay.__dict__))
    check("源 is_initialized=True 且 get_seq_length==6",
          bool(lay.is_initialized) and C.get_seq_length() == L)

    print("== 2) 正确克隆：8 项不变量 ==")
    def append(cache, n):
        with torch.inference_mode():
            m(torch.tensor([[7]]), past_key_values=cache, use_cache=True,
              position_ids=torch.tensor([[cache.get_seq_length()]]),
              attention_mask=torch.ones(1, cache.get_seq_length() + 1, dtype=torch.long))

    cl = clone_cache_full(C)
    info = require_clone_invariants(C, cl, L=L, append_fn=append, n=1)
    check("★ 1/2/3/4 项通过（K/V 位级、长度、初始化状态、无共享存储）", info["n_layers"] == 2)
    check("★ 5/6/7 项通过（追加 L+1、历史保留、源与兄弟副本不受影响）",
          info["appended"]["len_after"] == L + 1 and info["appended"]["history_preserved"]
          and info["appended"]["source_untouched"] and info["appended"]["sibling_untouched"])
    # 4（副本之间）
    c2 = clone_cache_full(C)
    check("★ 两个副本之间（K 与 V）都不共享存储",
          not any(getattr(cl.layers[i], kk).data_ptr() == getattr(c2.layers[i], kk).data_ptr()
                  for i in range(len(C.layers)) for kk in ("keys", "values")))

    print("== 3) ★ 回归：旧实现必须被抓出（不是只测正例）==")
    old = clone_cache_old_xb(C)
    check("旧实现：get_seq_length()==0（源为 6）", old.get_seq_length() == 0, old.get_seq_length())
    check("旧实现：is_initialized=False", bool(old.layers[0].is_initialized) is False)
    check("★ require_clone_invariants 抓到旧实现（is_initialized=False）",
          raises(lambda: require_clone_invariants(C, old, L=L)))
    check("★ 旧实现长度不符也被抓到",
          raises(lambda: require_clone_invariants(C, old, L=L)))

    print("== 4) 微型模型端到端：prefill → 克隆 → 单步/块前向（长度与历史保留）==")
    cs, cb = clone_cache_full(C), clone_cache_full(C)
    tok = int(o.logits[0, -1].argmax())
    with torch.inference_mode():
        s1 = m(torch.tensor([[tok]]), past_key_values=cs, use_cache=True,
               position_ids=torch.tensor([[L]]), attention_mask=torch.ones(1, L + 1, dtype=torch.long))
        b1 = m(torch.tensor([[tok, tok, tok]]), past_key_values=cb, use_cache=True,
               position_ids=torch.arange(L, L + 3).view(1, -1),
               attention_mask=torch.ones(1, L + 3, dtype=torch.long))
    check("serial 克隆后长度 = L+1", cs.get_seq_length() == L + 1, cs.get_seq_length())
    check("block  克隆后长度 = L+3", cb.get_seq_length() == L + 3, cb.get_seq_length())
    check("前 L 位历史在 serial 后保留",
          bool(torch.equal(cs.layers[0].keys[:, :, :L, :], C.layers[0].keys[:, :, :L, :])))
    check("前 L 位历史在 block 后保留",
          bool(torch.equal(cb.layers[0].keys[:, :, :L, :], C.layers[0].keys[:, :, :L, :])))
    check("源缓存未被两条路径影响（长度仍 6、K 位级不变）",
          C.get_seq_length() == L
          and bool(torch.equal(C.layers[0].keys, o.past_key_values.layers[0].keys)))
    print(f"  （自测信息）微型模型 serial/block 的 logits max|Δ|="
          f"{float((s1.logits[0,-1].float()-b1.logits[0,0].float()).abs().max()):.6g}")

    print("== 5) 自研引擎 list[dict] 缓存：克隆不变量 + 两类错法被抓出 ==")
    from .align_ours import build_from_config
    cfg2 = Qwen2Config(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                       max_position_embeddings=256)
    cfg2._attn_implementation = "eager"
    eng = build_from_config(cfg2).to(torch.bfloat16).eval()
    with torch.inference_mode():
        _, ec = eng._run(eng.embed_tokens(ids), None, need_logits=True)
    check("引擎缓存为 list[dict] 且含张量键 k/v",
          isinstance(ec, list) and all({"k", "v"} <= set(d) for d in ec)
          and all(torch.is_tensor(d["k"]) for d in ec))
    ec[0]["fmt"] = "fp16"                     # 非张量键：旧写法会在此处崩

    def eappend(c, n):
        with torch.inference_mode():
            _, nc = eng._run(eng.embed_tokens(torch.full((1, n), 7)), c, need_logits=False)
        c[:] = nc                             # 引擎 `_run` 返回新 list（dict 对象就地复用）⇒ 原地替换

    ecl = clone_list_cache_full(ec)
    info_e = require_list_clone_invariants(ec, ecl, L=L, append_fn=eappend, n=1)
    check("★ 引擎侧 1-5 项通过（含非张量键、位级、无共享存储）",
          info_e["n_tensors"] == 2 * 2)   # 2 层 ×(k,v)=4 个张量项；首层的 fmt 是非张量项
    check("★ 引擎侧 6/7 项通过（追加 L+1、历史保留、源与兄弟副本不受影响）",
          info_e["appended"]["len_after"] == L + 1 and info_e["appended"]["history_preserved"]
          and info_e["appended"]["source_untouched"] and info_e["appended"]["sibling_untouched"])
    shallow = [dict(d) for d in ec]
    check("★ 引擎侧浅拷贝（共享张量存储）被抓出",
          raises(lambda: require_list_clone_invariants(ec, shallow, L=L)))
    partial = [{k: v for k, v in d.items() if k in ("k", "v")} for d in ec]
    check("★ 引擎侧丢键（丢掉非张量键 fmt）被抓出",
          raises(lambda: require_list_clone_invariants(ec, partial, L=L)))
    try:
        clone_list_cache_old_xb(ec)
        old_engine_crashed = False
    except AttributeError:
        old_engine_crashed = True
    check("旧引擎侧写法在有非张量键时 AttributeError（本机 xb 运行的实际缓存只含 k/v，故当时未触发）",
          old_engine_crashed)
    ec_clean = [{kk: v for kk, v in d.items() if kk != "fmt"} for d in ec]   # 实际运行时的形态
    oldc = clone_list_cache_old_xb(ec_clean)
    check("★ 引擎侧旧写法在**只有张量键**（= 实际 xb 运行的形态）时通过全部不变量 "
          "⇒ 那次运行**引擎侧克隆是正确的**，错只在 HF 侧",
          require_list_clone_invariants(ec_clean, oldc, L=L)["n_tensors"] == 4)

    print("== 6) 机制：克隆坏了会怎样改变**前向结果**（不是只改元数据）==")
    tok2 = int(o.logits[0, -1].argmax())
    with torch.inference_mode():
        r_ok = m(torch.tensor([[tok2]]), past_key_values=clone_cache_full(C), use_cache=True,
                 position_ids=torch.tensor([[L]]), attention_mask=torch.ones(1, L + 1, dtype=torch.long))
        r_attn = m(torch.tensor([[tok2]]), past_key_values=clone_cache_full(C), use_cache=True,
                   position_ids=torch.tensor([[L]]), attention_mask=torch.ones(1, L + 1, dtype=torch.long),
                   output_attentions=True)
        r_old = m(torch.tensor([[tok2]]), past_key_values=clone_cache_old_xb(C), use_cache=True,
                  position_ids=torch.tensor([[L]]), attention_mask=torch.ones(1, L + 1, dtype=torch.long))
        r_none = m(torch.tensor([[tok2]]), use_cache=True, position_ids=torch.tensor([[L]]),
                   attention_mask=torch.ones(1, 1, dtype=torch.long))
    d_old = (r_ok.logits[0, -1].float() - r_old.logits[0, -1].float()).abs()
    d_none = (r_old.logits[0, -1].float() - r_none.logits[0, -1].float()).abs()
    check("★ 坏克隆的单步 logits 与正确克隆**不同**（bug 可在前向观测，不只元数据）",
          float(d_old.max()) > 0, f"max|Δ|={float(d_old.max()):.6g}")
    check("★ 坏克隆的单步 logits == **完全没有历史**的单步 logits（历史被清空）",
          float(d_none.max()) == 0, f"max|Δ|={float(d_none.max()):.6g}")
    check("★ output_attentions=True/False 的 logits **逐位相同**（微型 CPU 检查；支持"
          "「hfsb 探针传 output_attentions=True 不改变数值」这一判断）",
          bool(torch.equal(r_ok.logits[0, -1], r_attn.logits[0, -1])),
          f"max|Δ|={float((r_ok.logits[0,-1].float()-r_attn.logits[0,-1].float()).abs().max()):.6g}")
    print(f"  （自测信息）正确克隆 argmax={int(r_ok.logits[0,-1].argmax())} "
          f"坏克隆 argmax={int(r_old.logits[0,-1].argmax())} "
          f"无历史 argmax={int(r_none.logits[0,-1].argmax())}")

    print(f"\n缓存克隆自测：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
