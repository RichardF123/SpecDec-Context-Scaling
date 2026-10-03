"""serial_oracle.py — 串行 target oracle 的**调用几何审计**（默认不安装；只读、零副作用）。

## 目的
硬性证明：在 serial 模式下，generation 阶段的**每一次** target 前向都是 `seq_len == 1`
（与 AR 相同），且**没有** block `_run(s>1)`、没有 row-aligned wrapper、没有"内部先算 M=4 再切第 0 行"。

## 实现
临时包装 `model._run`（**只包 target**；draft 单独记录），逐个记录 `x.shape[1]`：
* 允许**恰好一次** prefill 调用（`s == prompt_len`；`attention_phase == "prefill"` 或首个大 `s`）；
* 其余（generation）调用必须 `s == 1`，否则**立即抛错**（不降级、不重试）；
* `exit` 时（含异常）恢复原 `_run`（`is` 同一对象核对）。

**不参与任何计算**：包装只记录形状后调用原函数。
"""
from __future__ import annotations

import contextlib

import torch


class CallGeometryError(RuntimeError):
    """调用几何违反串行 oracle 契约（硬失败）。"""


class _Audit:
    def __init__(self, model, *, role: str, allow_first_large: bool = True, prompt_len: int | None = None):
        self.model = model
        self.role = role
        self.allow_first_large = bool(allow_first_large)
        self.prompt_len = prompt_len
        self.calls: list = []
        self.n_large = 0
        self.orig = None

    def __call__(self, x, cache, *a, **kw):
        s = int(x.shape[1])
        self.calls.append(s)
        is_first = (len(self.calls) == 1)
        if s > 1:
            if not (is_first and self.allow_first_large):
                raise CallGeometryError(
                    f"{self.role}: generation 阶段出现 seq_len={s} > 1 的 target 前向（第 {len(self.calls)} 次调用）"
                    f" ⇒ 违反串行 oracle 契约（禁止 block _run(s>1)；禁止先算 M=4 再切第 0 行）")
            self.n_large += 1
        return self.orig(x, cache, *a, **kw)

    def summary(self) -> dict:
        gen = self.calls[1:] if self.calls else []
        return dict(role=self.role, n_calls=len(self.calls), per_call_seq_len=list(self.calls),
                    prefill_calls=int(self.n_large), generation_calls=len(gen),
                    generation_seq_lens=sorted(set(gen)),
                    all_generation_m1=bool(all(s == 1 for s in gen)),
                    max_generation_seq_len=(max(gen) if gen else None))


@contextlib.contextmanager
def target_call_audit(target, *, prompt_len: int | None = None, role: str = "target"):
    """包装 **target** 的 `_run`，强制 generation 调用 `s == 1`；退出（含异常）恢复原函数。"""
    aud = _Audit(target, role=role, allow_first_large=True, prompt_len=prompt_len)
    orig = target._run
    aud.orig = orig
    target._run = aud
    try:
        yield aud
    finally:
        if target._run is aud:
            target._run = orig
        else:                                                             # 被别的包装覆盖过：如实报错
            raise CallGeometryError(f"{role}: `_run` 未被本审计恢复（存在嵌套包装？）")


@contextlib.contextmanager
def draft_call_audit(draft, *, role: str = "draft"):
    """只**记录** draft 的调用形状（不受 `s==1` 断言约束）。"""
    aud = _Audit(draft, role=role, allow_first_large=True)
    orig = draft._run
    aud.orig = orig
    draft._run = aud
    try:
        yield aud
    finally:
        if draft._run is aud:
            draft._run = orig


# ===========================================================================
# GPU 运行器用的**阶段感知、按实例分组**的调用几何审计
#
# 与上面的 `_Audit` 的区别（为什么必须另写一套）：
#   * 真实 harness 的 prefill 是**分块**的（8192/2048 = 4 次 `s=2048`，外加可能的尾块），
#     而 `_Audit` 只允许"恰好一次大调用" ⇒ 会在真实运行里**误报**（这是 Stage 0 用的近似假设）。
#   * 真实运行里同一进程驻留 target(1.5B) 与 draft(0.5B) 两个实例 ⇒ 必须**按实例身份**分组，
#     并把"每一次调用"落到 `(阶段, seq_len)` 上，才能证明"target 的 generation 调用全部 M=1"。
#   * 判定口径改为：**只要阶段不是 `prefill`，就要求 `seq_len == 1`**（`prefill` 允许任意 `s`）。
#
# 安装方式与 `smoke_hooks` 完全一致：把 `owner_cls._run` 换成**普通函数**（保证 `instance._run(...)`
# 正确绑定），并保存原函数以便用 `is` 核查恢复。
# ===========================================================================
class GeometryAuditError(RuntimeError):
    """审计器自身的安装/使用错误（不是被审计对象的错误）。"""


def _n_params(model) -> int | None:
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:                                                     # noqa: BLE001
        return None


def _cache_len_of(cache_list) -> int | None:
    """cache 的 token 维长度（`cache[0]["k"].shape[2]`）；不可读时返回 None（只读，不改）。"""
    try:
        if not cache_list:
            return None
        return int(cache_list[0]["k"].shape[2])
    except Exception:                                                     # noqa: BLE001
        return None


def _label_of(model) -> tuple:
    """用**身份**判定该实例是 target 还是 draft（权威来源：`smoke_hooks._HOOK_REGISTRY`）。

    只读；若拿不到 ⇒ 返回 `("unknown", "unavailable")`，**绝不**按 head_dim/调用顺序猜。
    """
    try:
        from . import smoke_hooks as SH
        for cls, h in getattr(SH, "_HOOK_REGISTRY", {}).items():
            if isinstance(model, cls):
                if model is getattr(h, "tgt", None):
                    return "target", "smoke_hooks._HOOK_REGISTRY.tgt"
                if model is getattr(h, "drf", None):
                    return "draft", "smoke_hooks._HOOK_REGISTRY.drf"
    except Exception:                                                     # noqa: BLE001
        pass
    return "unknown", "unavailable"


def install_run_audit(owner_cls, *, enforce_m1_outside_prefill: bool = True) -> dict:
    """把阶段感知的调用几何审计装到 `owner_cls._run` 上（普通函数；按实例分组）。

    返回 `dict(hook=..., orig=..., state=..., uninstall=...)`；`state` 是**实时**字典（运行中持续写入）。
    `enforce_m1_outside_prefill=True` 时：任何**非 `prefill` 阶段**的 `s>1` 调用**立即抛错**
    （不降级、不重试）；`False`（记录模式）只记录，用于取得"生产 block 路径的几何"作对照。
    """
    from . import model_self as M                       # 只为读 `current_attention_phase`（只读）
    orig = owner_cls._run
    state: dict = dict(installed=True, owner=str(getattr(owner_cls, "__name__", owner_cls)),
                       enforce_m1_outside_prefill=bool(enforce_m1_outside_prefill),
                       per_instance={}, order=[], n_calls=0, violations=[])

    def hook(model, x, cache_list=None, need_logits=True):
        s = int(x.shape[1])
        try:
            phase = M.current_attention_phase()
        except Exception:                                                 # noqa: BLE001
            phase = None
        key = id(model)
        st = state["per_instance"].get(key)
        if st is None:
            label, src = _label_of(model)
            st = state["per_instance"][key] = dict(
                key=key, label=label, label_source=src, n_params=_n_params(model),
                n_layers_first=None, calls=[])
            state["order"].append(key)
        # ★ 先记录**再调用**：即使该次调用抛错（有限性/几何），这一笔也留在记录里。
        rec = dict(i=len(st["calls"]), seq_len=s, phase=phase,
                   cache_len_in=_cache_len_of(cache_list), cache_len_out=None, n_layers_out=None)
        st["calls"].append(rec)
        state["n_calls"] += 1
        if enforce_m1_outside_prefill and phase != "prefill" and s > 1:
            v = dict(role=st["label"], instance_key=key, call_index=rec["i"],
                     seq_len=s, phase=phase,
                     why="非 prefill 阶段的 target 前向必须 M=1（禁止 block _run(s>1)；"
                         "禁止先算 M=4 再切第 0 行）")
            state["violations"].append(v)
            raise CallGeometryError(f"{st['label']}: 第 {len(st['calls'])} 次调用 seq_len={s} > 1 "
                                    f"（phase={phase}）⇒ 违反调用几何契约：{v['why']}")
        lg, out_cache = orig(model, x, cache_list, need_logits)
        # 回填：层数只能从**返回的** cache 读（首次调用 `cache_list is None`）
        rec["cache_len_out"] = _cache_len_of(out_cache)
        try:
            rec["n_layers_out"] = (len(out_cache) if out_cache else None)
        except Exception:                                                 # noqa: BLE001
            rec["n_layers_out"] = None
        if st["n_layers_first"] is None:
            st["n_layers_first"] = rec["n_layers_out"]
        return lg, out_cache

    owner_cls._run = hook

    def uninstall() -> dict:
        pre = owner_cls._run
        owner_cls._run = orig
        state["installed"] = False
        return dict(pre_uninstall_was_our_hook=bool(pre is hook),
                    restored_is_original=bool(owner_cls._run is orig))

    return dict(hook=hook, orig=orig, state=state, uninstall=uninstall)


def _summ_instance(st: dict) -> dict:
    calls = st["calls"]
    gen = [c for c in calls if c["phase"] != "prefill"]
    by_phase: dict = {}
    for c in calls:
        k = str(c["phase"])
        d = by_phase.setdefault(k, dict(n=0, seq_lens=[], max_seq_len=0))
        d["n"] += 1
        d["seq_lens"].append(c["seq_len"])
        d["max_seq_len"] = max(d["max_seq_len"], c["seq_len"])
    gen_lens = sorted({c["seq_len"] for c in gen})
    return dict(label=st["label"], label_source=st["label_source"], instance_key=st["key"],
                n_params=st["n_params"], n_layers_first=st["n_layers_first"],
                n_calls=len(calls), n_prefill_calls=len(calls) - len(gen),
                n_generation_calls=len(gen),
                generation_seq_lens=gen_lens,
                generation_max_seq_len=(max(gen_lens) if gen_lens else None),
                # ★ 反空泛：**零次** generation 调用一律判 False（不得因为"集合为空"而通过）
                all_generation_m1=bool(gen) and all(c["seq_len"] == 1 for c in gen),
                nonvacuous=bool(gen),
                by_phase=by_phase,
                first_calls=[dict(i=c["i"], seq_len=c["seq_len"], phase=c["phase"]) for c in calls[:6]],
                last_calls=[dict(i=c["i"], seq_len=c["seq_len"], phase=c["phase"]) for c in calls[-4:]])


def run_audit_summary(state: dict) -> dict:
    """把 `state` 汇总成可落盘结构（含**反空泛**判据与 target 识别的一致性核对）。"""
    per = {str(k): _summ_instance(v) for k, v in state["per_instance"].items()}
    by_label: dict = {}
    for k, v in per.items():
        by_label.setdefault(v["label"], []).append(k)
    labels_unique = {lab: (len(ks) == 1) for lab, ks in by_label.items()}
    tgt = None
    if len(by_label.get("target", [])) == 1:
        tgt = per[by_label["target"][0]]
    dft = None
    if len(by_label.get("draft", [])) == 1:
        dft = per[by_label["draft"][0]]
    # 交叉核对：target/draft 的层数与参数量必须**可区分**（否则识别不可靠 ⇒ 明确报出，不猜）
    cross = dict(target_n_layers=(tgt or {}).get("n_layers_first"),
                 target_n_params=(tgt or {}).get("n_params"),
                 draft_n_layers=(per[by_label["draft"][0]]["n_layers_first"]
                                 if len(by_label.get("draft", [])) == 1 else None),
                 draft_n_params=(per[by_label["draft"][0]]["n_params"]
                                 if len(by_label.get("draft", [])) == 1 else None))
    cross["layer_counts_differ"] = bool(cross["target_n_layers"] is not None
                                        and cross["draft_n_layers"] is not None
                                        and cross["target_n_layers"] != cross["draft_n_layers"])
    cross["target_params_gt_draft"] = bool(cross["target_n_params"] is not None
                                           and cross["draft_n_params"] is not None
                                           and cross["target_n_params"] > cross["draft_n_params"])
    # ⚠️ `sane` **不**要求 draft 存在：AR 模式只驻留 target（实例数由运行器按模式断言：
    #    ar ⇒ 1；spec/serial ⇒ 2）。若把 draft 写进 `sane`，AR 会被误判为"审计没接上"。
    ok = bool(state["n_calls"] > 0 and tgt is not None and tgt["nonvacuous"]
              and labels_unique.get("target") and len(per) >= 1)
    # ★ `sane` = 安装与识别的健全性（每个模式都必须为 True，否则"审计没接上"，结论无效）；
    #   `certifies_m1` = `sane` **且** target 的 generation 调用**全部** M=1 且无违规
    #   ⇒ 这是"串行 oracle 契约成立"的**认证**判据（`ok` 与该字段同义，保留 `ok` 以便调用方读取）。
    sane = ok
    certifies_m1 = bool(sane and tgt is not None and tgt["all_generation_m1"] is True
                        and not state["violations"])
    return dict(role="class-level phase-aware call-geometry audit",
                enforce_m1_outside_prefill=bool(state["enforce_m1_outside_prefill"]),
                installed=bool(state["installed"]), owner=state["owner"],
                n_calls=int(state["n_calls"]), n_instances=len(per),
                labels=by_label, labels_unique=labels_unique,
                target=tgt, draft=dft, per_instance=per, cross_check=cross,
                violations=list(state["violations"]),
                target_identified=bool(tgt is not None),
                target_all_generation_m1=(bool(tgt["all_generation_m1"]) if tgt else None),
                sane=sane, certifies_m1=certifies_m1, ok=certifies_m1,
                note=("空泛保护：零调用或零 generation 调用一律**不通过**（否则"
                      "『没记录到任何调用』会被误当成『全部 M=1』）；`sane` 只表示审计器装对了，"
                      "`certifies_m1`/`ok` 才表示『target 的 generation 调用全部 M=1』"))
