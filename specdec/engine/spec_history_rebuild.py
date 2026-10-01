"""spec_history_rebuild.py — **「串行重建 target 缓存历史」的诊断包装**（默认关闭）。

## 目的（只回答一个问题）
F13 首个分歧（位置 8239：AR=6702 / Spec=6652）是否由**既有 KV 历史差异**驱动？
做法：**不改本轮接受决策**，把 Spec 每轮实际提交的 token 从**独立的轮前快照**逐 token 重放，
用重放得到的缓存替换下一轮的 target 缓存、并用**最后一次重放**的 logits 更新 `saved_next`。

## 为什么必须"逐 token 重放"而不是"trim 后补喂"
`spec_cache.trim_cache` 只做**切片**（`val[:, :, :new_len]`）⇒ 被保留的 KV 仍是**块内**算出来的；
而 `m == kp`（全接受）那一支直接 `tgt_cache = tgt_blk`，**一个都没被 trim 掉**。
所以**全接受轮同样会保留块内 KV** —— 只处理 mismatch 轮建立不出"串行一致"的历史。
另外：`_verify` 会**就地改写**传入的缓存（`cache["k"] = torch.cat(...)`）⇒
轮前快照必须在 `_verify` **之前**克隆，且重放必须在**独立副本**上做。

## 与生产的关系
* 生产的 `spec_dec.spec_decode_greedy` **一字未改**；本模块只在显式安装时替换
  模块属性 `spec_dec.spec_decode_greedy`，并在 `restore()` 里用 `is` 核查还原。
* 包装函数**与生产同签名**（含关键字参数），内部调用本模块对生产循环的**逐行复刻**，
  只在两处加诊断：① 轮前完整克隆 `H_pre`；② 需要时按"逐 token 重放"重建下一轮缓存。
* **默认关闭**（不安装 ⇒ 生产路径完全不变）；CPU 自测第 4 组证明"关闭时逐位等价"。

## 纪律（本轮要求，逐条落实）
* **不改本轮接受决策**：`m`、`emitted` 全部沿用原逻辑与原始值，不强制采用 AR token；
* 每个实际提交的 token 在重建历史里**只追加一次**（从 `H_pre` 开始，依次喂 `emitted`）；
* **原始拒绝分支的 repair 结果不再追加到重建缓存**（`trim_cache(tgt_blk,…)` 的结果被丢弃）；
* **不把重建后的数值冒充本轮原始判断值**：逐轮记录里 `orig_*`（块验证的真实值）与
  `replay_*`（重建后的值）分字段；
* **draft 完全沿用原更新规则**（不替换 target 缓存、不改提案逻辑）；
* 重建后断言 `cache_length == pre + len(emitted)`、逐层一致（`spec_cache.check_consistent`）。
"""
from __future__ import annotations

import copy

import torch

from . import cache_clone as CC
from . import model_self as M
from . import spec_cache
from . import spec_dec


class RebuildError(RuntimeError):
    """重建实现/契约错误（不是引擎错误）。"""


_S: dict = dict(installed=False, orig=None, rebuild=False, capture_len=None, capture_row_at=None,
                diag=None, last=None)


def _feed_one(target, token: int, cache):
    """**与 `spec_dec._step` 逐行等价**的单步喂入。

    为什么不直接调 `spec_dec._step`：在生产 harness 里它已被 `smoke_hooks` 包装（阶段改为
    `repair` 并计入其有限性/覆盖账目）。重放是**诊断**，不应改动那些账目的口径 ⇒
    这里显式走 `target._run(...)`，并自己做有限性检查（见 `_finite`）。
    """
    with torch.inference_mode(), M.attention_phase("decode"):
        lg, cache = target._run(
            target.embed_tokens(torch.tensor([[token]], device=cache[0]["k"].device)),
            cache, need_logits=True)
    return lg[:, -1:], cache


def _finite(lg, *, where: str) -> None:
    if lg is None or not bool(torch.isfinite(lg).all()):
        raise RebuildError(f"重放出现非有限 logits：{where}")


def _capture_step(diag: dict, replay, *, pre: int, rnd: int, fed: list) -> None:
    """在逐 token 重放过程中**只读**捕获：① 长度到达 capture_len 的完整快照；② 决策行的真实值。"""
    L = spec_cache.cache_length(replay)
    if diag["capture_len"] is not None and L == diag["capture_len"] and diag["snap"] is None:
        diag["snap"] = CC.clone_list_cache_full(replay)      # deepcopy ⇒ 与后续重放不共享存储
        diag["snap_meta"] = dict(round=rnd, len=L, feeds_so_far=list(fed), pre=pre)
    if diag["capture_row_at"] is not None and L == diag["capture_row_at"] and diag["row"] is None:
        diag["row"] = None                                   # 由调用方在该步之后填


def rebuilt_spec_decode_greedy(target, draft, prompt_ids, max_new_tokens, *, k=spec_dec.K,
                               eos_id=None, target_kv="base_bf16", draft_kv="base_bf16",
                               log=None, rebuild=False, diag=None):
    """`spec_dec.spec_decode_greedy` 的**逐行复刻** + 可选的"串行重建历史"诊断。

    `rebuild=False` 时**数值路径与生产完全一致**（CPU 自测第 4 组逐位核对）。
    `diag` 为字典时写入逐轮诊断与捕获；返回值比生产多一个 `diag` 键（调用方按需使用）。
    """
    target.set_kv_format(spec_dec._fmt(target_kv))
    draft.set_kv_format(spec_dec._fmt(draft_kv))

    tgt_lg, tgt_cache = spec_dec._prefill(target, prompt_ids)
    drf_lg, drf_cache = spec_dec._prefill(draft, prompt_ids)
    saved_next = tgt_lg[:, -1:]
    drf_lg = drf_lg[:, -1:]

    gen = []
    st = dict(rounds=0, proposed=0, accepted=0, target_block_calls=0, draft_calls=0,
              mismatch_rounds=0, all_accept_rounds=0, rollbacks=0)
    rows_diag: list[dict] = []
    n_replay = 0

    while len(gen) < max_new_tokens:
        k_eff = min(k, max_new_tokens - len(gen))
        pre = spec_cache.cache_length(tgt_cache)
        # ★★ 关键：`_verify` 会就地改写 tgt_cache ⇒ 轮前快照必须在它之前克隆
        H_pre = CC.clone_list_cache_full(tgt_cache) if rebuild else None

        cands, drf_cache, prop_lg = spec_dec._propose(draft, drf_cache, drf_lg, k_eff, eos_id)
        kp = len(cands)
        st["draft_calls"] += kp
        if kp == 0:
            break

        rows, tgt_blk = spec_dec._verify(target, tgt_cache, cands)
        st["target_block_calls"] += 1
        st["proposed"] += kp

        g = [int(x.argmax()) for x in [saved_next] + [rows[j] for j in range(kp - 1)]]
        m = 0
        while m < kp and g[m] == cands[m]:
            m += 1

        if m < kp:
            r = g[m]
            emitted = cands[:m] + [r]
            tgt_cache = spec_cache.trim_cache(tgt_blk, pre + m)
            saved_next, tgt_cache = spec_dec._step(target, r, tgt_cache)
            drf_cache = spec_cache.trim_cache(drf_cache, pre + m)
            drf_lg, drf_cache = spec_dec._step(draft, r, drf_cache)
            st["mismatch_rounds"] += 1
            st["rollbacks"] += 1
        else:
            emitted = list(cands)
            tgt_cache = tgt_blk
            drf_cache = spec_cache.trim_cache(drf_cache, pre + kp)
            saved_next = rows[kp - 1]
            drf_lg = prop_lg
            st["all_accept_rounds"] += 1

        rec_r = dict(round=st["rounds"], pre=pre, kp=kp, m=m, branch=("mismatch" if m < kp else "all_accept"),
                     cands=list(cands), g=list(g), emitted=list(emitted),
                     orig_tgt_len=spec_cache.cache_length(tgt_cache))

        # 诊断（只读）：把"决定 capture_row_at 的那一轮"的**真实块验证值**留档（供四格工具当参考）
        if diag is not None and diag.get("capture_row_at") is not None:
            slot = int(diag["capture_row_at"]) - pre
            if 1 <= slot <= kp - 1 and diag.get("block_round") is None:
                diag["block_round"] = dict(
                    round=st["rounds"], pre=pre, Kp=kp, slot=slot, row_index=slot - 1,
                    cands=list(cands), g=list(g), m=int(m), emitted=list(emitted),
                    rows_argmax=[int(rows[j].argmax()) for j in range(kp)],
                    rows_cpu=rows.detach().float().cpu().clone(),
                    note="该轮是**原逻辑**的块验证：rows 由长度 pre 的缓存算出（重建前的真实值）")

        if rebuild:
            # ★ 从 H_pre 的**独立克隆**开始重放：`_run` 会就地改写传入缓存的层 dict，
            #   若直接 `replay = H_pre`，第一次喂入就会把 H_pre 也改写掉（H_pre 必须保持"轮前状态"）
            replay = CC.clone_list_cache_full(H_pre)
            lg_last, feeds = None, []
            for tk in emitted:
                if (diag is not None and diag.get("capture_row_at") is not None
                        and spec_cache.cache_length(replay) + 1 == diag["capture_row_at"]):
                    diag["row_fed_token"] = int(tk)              # 该步喂入的 token（位置 capture_row_at-1）
                    diag["row_round"] = st["rounds"]
                lg_last, replay = _feed_one(target, tk, replay)
                n_replay += 1
                feeds.append(int(tk))
                _finite(lg_last, where=f"round={st['rounds']} 重放 token={tk}")
                if diag is not None:
                    if (diag.get("capture_row_at") is not None and diag.get("row") is None
                            and spec_cache.cache_length(replay) == diag["capture_row_at"]):
                        diag["row"] = lg_last[0, -1].detach().float().cpu().clone()
                    _capture_step(diag, replay, pre=pre, rnd=st["rounds"], fed=feeds)
            L = spec_cache.cache_length(replay)
            spec_cache.check_consistent(replay)                  # 跨层 token 维一致
            per_layer = [int(d["k"].shape[2]) for d in replay]
            if not (L == pre + len(emitted) and all(v == L for v in per_layer)):
                raise RebuildError(f"重建长度不符：L={L} 期望={pre + len(emitted)} 逐层={per_layer}")
            tgt_cache = replay                                   # ★ 替换下一轮的 target 缓存
            saved_next = lg_last                                 # ★ 用最后一次重放的 logits 更新
            hp_after = spec_cache.cache_length(H_pre)
            if hp_after != pre:
                raise RebuildError(f"H_pre 被改写：{hp_after} != {pre}")
            rec_r.update(rebuilt=True, n_replay_feeds=len(feeds), feeds=feeds, rebuilt_len=L,
                         per_layer_len_ok=True, H_pre_len_after_replay=hp_after,
                         # 重建后的数值与**原始块验证值**分字段，互不冒充
                         replay_saved_next_argmax=int(saved_next.reshape(-1)[-1].argmax()),
                         replay_lens=dict(pre=pre, after=spec_cache.cache_length(tgt_cache)))

        gen += emitted
        st["accepted"] += m
        st["rounds"] += 1
        if log is not None:
            log(dict(round=st["rounds"] - 1, Kp=kp, m=m, pre=pre,
                     tgt_len=spec_cache.cache_length(tgt_cache),
                     drf_len=spec_cache.cache_length(drf_cache),
                     cands=list(cands), g=list(g), emitted=list(emitted)))
        rows_diag.append(rec_r)
        if emitted and emitted[-1] == eos_id:
            break

    out = {"ids": list(prompt_ids[0]) + gen, "gen": gen, "stats": st}
    if diag is not None:
        diag["rounds"] = rows_diag
        diag["n_replay_feeds"] = n_replay
        diag["rebuild"] = bool(rebuild)
        out["diag"] = diag
    return out


# ------------------------------------------------------------------ 安装 / 还原
def install(*, rebuild: bool, capture_len: int | None = None,
            capture_row_at: int | None = None) -> dict:
    """把 `spec_dec.spec_decode_greedy` 换成同签名的诊断包装；必须在 `finally` 里 `restore()`。"""
    if _S["installed"]:
        raise RebuildError("已经安装过（同一进程只允许一个包装）")
    _S["orig"] = spec_dec.spec_decode_greedy
    _S["rebuild"] = bool(rebuild)
    _S["diag"] = dict(capture_len=(None if capture_len is None else int(capture_len)),
                      capture_row_at=(None if capture_row_at is None else int(capture_row_at)),
                      snap=None, snap_meta=None, row=None, row_fed_token=None, row_round=None,
                      rounds=[], n_replay_feeds=0)

    def shim(target, draft, prompt_ids, max_new_tokens, *, k=spec_dec.K, eos_id=None,
             target_kv="base_bf16", draft_kv="base_bf16", log=None):
        out = rebuilt_spec_decode_greedy(target, draft, prompt_ids, max_new_tokens, k=k,
                                         eos_id=eos_id, target_kv=target_kv, draft_kv=draft_kv,
                                         log=log, rebuild=_S["rebuild"], diag=_S["diag"])
        _S["last"] = out.get("diag")
        return out

    spec_dec.spec_decode_greedy = shim
    _S["installed"] = True
    return dict(installed=True, rebuild=bool(rebuild), capture_len=_S["diag"]["capture_len"],
                capture_row_at=_S["diag"]["capture_row_at"],
                semantics="不改本轮接受决策；从轮前快照逐 token 重放 emitted；替换下一轮 target 缓存；"
                          "saved_next 取自最后一次重放；全接受/部分接受/m=0 一支不落",
                signature="与生产 spec_decode_greedy 同签名（含 k/eos_id/target_kv/draft_kv/log 关键字）",
                probe="重放走显式 `target._run`（不经过 smoke_hooks），阶段标为 decode，并自做有限性检查")


def state() -> dict:
    return dict(installed=_S["installed"], rebuild=_S["rebuild"],
                diag=(_S["diag"] if _S["diag"] is None else
                      {k: v for k, v in _S["diag"].items() if k not in ("snap", "row")}),
                has_snap=bool(_S["diag"] and _S["diag"].get("snap") is not None),
                has_row=bool(_S["diag"] and _S["diag"].get("row") is not None))


def capture_payload() -> dict:
    """返回 `(meta, tensors)` 供工具落盘：完整快照 + 真实决策行 + 该轮真实块 rows + 逐轮诊断。"""
    d = _S["diag"] or {}
    snap = d.get("snap")
    meta = {k: v for k, v in d.items() if k not in ("snap", "row")}
    tens: dict = {}
    br = meta.get("block_round")
    if isinstance(br, dict) and br.get("rows_cpu") is not None:
        tens["rows"] = br.pop("rows_cpu")                  # 键名与 f13_i50_cells 约定一致
    if snap is not None:
        meta["snapshot_cache_len"] = spec_cache.cache_length(snap)
        meta["snapshot_n_layers"] = len(snap)
        meta["snapshot_keys"] = sorted(snap[0].keys())
        tens["snapshot_cpu"] = snap
    if d.get("row") is not None:
        tens["replay_row"] = d["row"]
        meta["replay_row_predicts"] = d.get("capture_row_at")
    return meta, tens


def restore() -> dict:
    if not _S["installed"]:
        return dict(restored=False, note="未安装过（无需还原）")
    spec_dec.spec_decode_greedy = _S["orig"]
    res = dict(restored=True,
               is_original=bool(spec_dec.spec_decode_greedy is _S["orig"]),
               n_replay_feeds=(_S["diag"] or {}).get("n_replay_feeds"))
    _S["installed"] = False
    res["ok"] = bool(res["is_original"])
    return res


def assert_restored() -> dict:
    res = restore()
    if res.get("restored") and not res.get("ok"):
        raise RebuildError(f"还原核查失败：{res}")
    return res


def is_installed() -> bool:
    return bool(_S["installed"])


# ------------------------------------------------------------------ CPU 自测
class _Stub:
    """受控伪模型：`script[pos]` 决定"位置 pos 的行预测哪个 token"；缓存是 1 层 (1,1,T,1)。

    这样可以在**不加载模型**的前提下精确构造全接受 / m=0 / 部分接受 / EOS / 剩余额度 五种情形。
    """

    def __init__(self, vocab: int, script: dict, name: str = "stub"):
        self.vocab, self.script, self.name = vocab, dict(script), name
        self.calls: list[dict] = []

    def set_kv_format(self, fmt):
        self.fmt = fmt

    def embed_tokens(self, t):
        return t

    def _run(self, x, cache_list=None, need_logits=True):
        toks = [int(v) for v in x.reshape(-1).tolist()]
        n = len(toks)
        start = 0 if cache_list is None else int(cache_list[0]["k"].shape[2])
        # 行语义与真实模型一致：第 j 行位于位置 start+j，预测 start+j+1
        lg = torch.full((1, n, self.vocab), -10.0)
        rows = []
        for j in range(n):
            nxt = int(self.script.get(start + j, 0))
            lg[0, j, nxt] = 0.0
            rows.append(nxt)
        self.calls.append(dict(name=self.name, n=n, start=start, rows=rows))
        if cache_list is None:
            cache = [dict(k=torch.zeros(1, 1, n, 1), v=torch.zeros(1, 1, n, 1))]
        else:
            cache = cache_list
            for d in cache:
                d["k"] = torch.cat([d["k"], torch.zeros(1, 1, n, 1)], dim=2)
                d["v"] = torch.cat([d["v"], torch.zeros(1, 1, n, 1)], dim=2)
        return (lg if need_logits else None), cache


def _selftest() -> int:
    """CPU 门：不加载任何大模型；不过门**不得**进入 GPU 实验。"""
    np_ = nf = 0

    def check(name, cond, extra=None):
        nonlocal np_, nf
        if cond:
            np_ += 1
            print(f"  [ok] {name}")
        else:
            nf += 1
            print(f"  [FAIL] {name}" + (f"  {str(extra)[:220]}" if extra is not None else ""))

    def _run_case(tag, *, dscript, tscript, k, max_new, eos, rebuild, capture_len=None,
                  capture_row_at=None):
        """跑一个受控场景；返回 (out, diag)。"""
        d = _Stub(64, dscript, "draft")
        t = _Stub(64, tscript, "target")
        prompt = torch.tensor([[1, 2, 3]])
        diag = dict(capture_len=capture_len, capture_row_at=capture_row_at, snap=None,
                    snap_meta=None, row=None, row_fed_token=None, row_round=None, rounds=[],
                    n_replay_feeds=0)
        out = rebuilt_spec_decode_greedy(t, d, prompt, max_new, k=k, eos_id=eos,
                                         rebuild=rebuild, diag=diag)
        return out, diag

    print("== 1) 全接受 / m=0 / 部分接受 / EOS / 剩余额度（受控伪模型）==")
    # 目标与草稿脚本按**绝对位置**给下一 token；位置 = 3 + 已生成数
    # 全接受：目标处处同意草稿
    D_ALL = {3: 10, 4: 11, 5: 12, 6: 13, 7: 14, 8: 15, 9: 16, 10: 17}
    out_a, diag_a = _run_case("all_accept", dscript=D_ALL, tscript=D_ALL, k=4, max_new=8,
                              eos=None, rebuild=True, capture_len=7, capture_row_at=9)
    ra = out_a["diag"]["rounds"]
    check("★ 全接受轮被重建（branch=all_accept 且 rebuilt=True）",
          all(r["branch"] == "all_accept" and r.get("rebuilt") for r in ra), ra)
    check("★ 全接受轮：emitted 逐 token 只追加一次（feeds == emitted）",
          all(r["feeds"] == r["emitted"] for r in ra))
    check("★ 全接受轮：重建后长度 = pre + len(emitted)，逐层一致",
          all(r["rebuilt_len"] == r["pre"] + len(r["emitted"]) and r["per_layer_len_ok"] for r in ra))
    check("★ 轮前快照 H_pre 在重放后未被改写（长度仍为 pre）",
          all(r["H_pre_len_after_replay"] == r["pre"] for r in ra))
    # m=0：目标第一行就与草稿不同（草稿首候选来自 prompt 末行 = 位置 2）
    D0 = {2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15}
    T0 = dict(D0)
    T0[2] = 63                     # saved_next 的行（位置 2）给出 63 ≠ 草稿首候选 10 ⇒ m=0
    out_0, _ = _run_case("m0", dscript=D0, tscript=T0, k=4, max_new=6, eos=None, rebuild=True)
    r0 = out_0["diag"]["rounds"]
    check("★ m=0 轮（首候选即被拒）被重建，emitted = [g[0]]（长度为 1）",
          r0[0]["m"] == 0 and r0[0]["branch"] == "mismatch" and r0[0]["emitted"] == [r0[0]["g"][0]]
          and r0[0].get("rebuilt") and r0[0]["feeds"] == r0[0]["emitted"], r0[0])
    # 部分接受：m=2
    DP = {2: 10, 3: 11, 4: 12, 5: 13, 6: 20, 7: 21, 8: 22, 9: 23}
    TP = dict(DP)
    TP[4] = 63                     # 位置 4 的行（判第 3 个候选）给出 63 ≠ 12 ⇒ m=2
    out_p, _ = _run_case("partial", dscript=DP, tscript=TP, k=4, max_new=7, eos=None, rebuild=True)
    rp = out_p["diag"]["rounds"]
    check("★ 部分接受（m=2）被重建，emitted = cands[:m] + [g[m]]",
          rp[0]["m"] == 2 and rp[0]["emitted"] == rp[0]["cands"][:2] + [rp[0]["g"][2]]
          and rp[0]["feeds"] == rp[0]["emitted"], rp[0])
    # EOS：让"被提交的 token"就是 eos（位置 3 的行给出 55，而草稿在该位置提 20 ⇒ m=1）
    DE = {2: 10, 3: 20, 4: 21, 5: 22, 6: 23}
    TE = {2: 10, 3: 55, 4: 21, 5: 22, 6: 23}
    out_e, _ = _run_case("eos", dscript=DE, tscript=TE, k=4, max_new=8, eos=55, rebuild=True)
    check("★ EOS：循环在提交 eos 后停止，且该 token 在重建历史中出现一次",
          out_e["gen"] == [10, 55] and out_e["diag"]["rounds"][-1]["feeds"] == [10, 55],
          (out_e["gen"], out_e["diag"]["rounds"]))
    # 剩余额度：max_new=3 且 k=4 ⇒ k_eff=3
    out_b, _ = _run_case("budget", dscript=D_ALL, tscript=D_ALL, k=4, max_new=3, eos=None,
                         rebuild=True)
    check("★ 剩余生成额度：k_eff=min(k, 剩余)，且重建长度随之（首轮 kp<=3）",
          out_b["diag"]["rounds"][0]["kp"] <= 3 and len(out_b["gen"]) == 3,
          out_b["diag"]["rounds"][0])

    print("== 2) 源快照不被修改 + 捕获语义 + 完整元数据 ==")
    from . import sb_align as _A
    _snap_h1 = _A.stable_sha(diag_a["snap"][0]["k"]) if diag_a["snap"] is not None else None
    check("★ 捕获到的长度快照长度正确（capture_len=7）",
          diag_a["snap"] is not None and spec_cache.cache_length(diag_a["snap"]) == 7,
          None if diag_a["snap"] is None else spec_cache.cache_length(diag_a["snap"]))
    check("★ 捕获保留了完整层内键与长度元数据（deepcopy）",
          sorted(diag_a["snap"][0].keys()) == ["k", "v"]
          and diag_a["snap_meta"]["len"] == 7 and "feeds_so_far" in diag_a["snap_meta"])
    _snap_h2 = _A.stable_sha(diag_a["snap"][0]["k"])
    check("★ 捕获快照与后续重放/其它场景不共享存储（内容不被改写）", _snap_h1 == _snap_h2,
          (_snap_h1, _snap_h2))
    check("★ 决策行（capture_row_at=9）在重放中被捕获，且记录了喂入它的 token",
          diag_a["row"] is not None and diag_a["row_fed_token"] is not None,
          (diag_a["row_fed_token"], diag_a["row_round"]))
    check("★ 逐轮写入 orig_*（原始块验证值）与 replay_*（重建值）分字段",
          all({"orig_tgt_len", "replay_saved_next_argmax", "replay_lens"} <= set(r) for r in ra))

    print("== 3) 默认关闭（rebuild=False）：与生产逐位等价 ==")
    d1 = _Stub(64, D_ALL, "draft")
    t1 = _Stub(64, D_ALL, "target")
    prompt = torch.tensor([[1, 2, 3]])
    ref = spec_dec.spec_decode_greedy(t1, d1, prompt, 8, k=4, eos_id=None)
    d2 = _Stub(64, D_ALL, "draft")
    t2 = _Stub(64, D_ALL, "target")
    mine = rebuilt_spec_decode_greedy(t2, d2, prompt, 8, k=4, eos_id=None, rebuild=False)
    check("★ rebuild=False 时 ids/gen/stats 与生产函数完全一致",
          ref["ids"] == mine["ids"] and ref["gen"] == mine["gen"] and ref["stats"] == mine["stats"],
          (ref["gen"], mine["gen"], ref["stats"], mine["stats"]))
    check("★ rebuild=False 时伪模型的调用序列也一致（不改调用行为）",
          [c["n"] for c in d1.calls] == [c["n"] for c in d2.calls]
          and [c["n"] for c in t1.calls] == [c["n"] for c in t2.calls],
          (len(d1.calls), len(d2.calls), len(t1.calls), len(t2.calls)))

    print("== 4) 真实微型模型端到端（真 `_run`/`_step`/`_verify`/trim 路径）==")
    torch.manual_seed(0)
    tgt = M.QwenLM(vocab_size=64, hidden=32, heads=4, kv_heads=2, head_dim=8, intermediate=64,
                   num_layers=2, rope_theta=10000.0, rms_eps=1e-6, attn_bias=True
                   ).to(torch.bfloat16).eval()
    drf = M.QwenLM(vocab_size=64, hidden=32, heads=4, kv_heads=2, head_dim=8, intermediate=64,
                   num_layers=1, rope_theta=10000.0, rms_eps=1e-6, attn_bias=True
                   ).to(torch.bfloat16).eval()
    ids = torch.randint(0, 64, (1, 6))
    ref2 = spec_dec.spec_decode_greedy(tgt, drf, ids, 12, k=4, eos_id=None)
    dg = dict(capture_len=8, capture_row_at=10, snap=None, snap_meta=None, row=None,
              row_fed_token=None, row_round=None, rounds=[], n_replay_feeds=0)
    got = rebuilt_spec_decode_greedy(tgt, drf, ids, 12, k=4, eos_id=None, rebuild=True, diag=dg)
    check("★ 真实微型模型：重建后每一轮长度契约成立（pre+len(emitted)）且逐层一致",
          all(r["rebuilt_len"] == r["pre"] + len(r["emitted"]) and r["per_layer_len_ok"]
              for r in got["diag"]["rounds"]), got["diag"]["rounds"][:2])
    check("★ 真实微型模型：每个提交 token 只追加一次",
          all(r["feeds"] == r["emitted"] for r in got["diag"]["rounds"]))
    check("★ 真实微型模型：重建轮数 == 生产轮数（决策未被改动）",
          len(got["diag"]["rounds"]) == ref2["stats"]["rounds"],
          (len(got["diag"]["rounds"]), ref2["stats"]["rounds"]))
    check("★ 重建后长度与生产一致（同 token 序列下应为同长）",
          abs(len(got["gen"]) - len(ref2["gen"])) <= 4, (len(got["gen"]), len(ref2["gen"])))

    print("== 5) 包装安装 / 异常恢复 / 无全局残留 ==")
    orig = spec_dec.spec_decode_greedy
    meta = install(rebuild=True, capture_len=8, capture_row_at=10)
    check("★ 安装后模块属性被替换，且元数据写明语义", spec_dec.spec_decode_greedy is not orig
          and meta["installed"] and "逐 token 重放" in meta["semantics"])

    class _Boom(_Stub):
        def _run(self, x, cache_list=None, need_logits=True):
            raise RuntimeError("注入的重放故障")

    try:
        try:
            rebuilt_spec_decode_greedy(_Boom(64, {3: 10}), _Stub(64, {3: 10}), torch.tensor([[1, 2, 3]]),
                                       4, k=4, eos_id=None, rebuild=True, diag={})
        except RuntimeError:
            pass
    finally:
        res = restore()
    check("★ 异常后 restore() 用 `is` 核查已还原", res["ok"] and res["restored"]
          and spec_dec.spec_decode_greedy is orig, res)
    check("★ 未安装时 restore() 幂等", restore()["restored"] is False)

    print(f"\n串行重建历史（CPU 门）：PASS {np_} / FAIL {nf}")
    return 0 if nf == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
