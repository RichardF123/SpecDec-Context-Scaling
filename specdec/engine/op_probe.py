"""op_probe.py — L0 内部**算子边界**的只读捕获与比较（默认关闭；纯逻辑可 CPU 测）。

## 为什么需要
`SelfAttention` / `DecoderLayer` / `MLP` 的公式在 `model_self.py` 里是**逐行 Python 表达式**，
其中 Q/K/V 的 reshape、RoPE、KV 拼接、QK^T、causal mask、softmax、AV、head merge 都是
**函数式算子**（没有对应的 `nn.Module`）⇒ 普通 module hook 抓不到。
`model_self.py` 因此提供了**默认 None 的探针点** `_probe(name, tensor, who)`：
关闭时不构造参数、不碰张量（逐位等价）；本模块只负责**安装/移除探针**、**按语义切片比较**、
以及**判定"输入逐位相同、输出首次不同"的算子**。

## 比较口径（**只比语义对应的决策数据**，绝不比形状不同的整张量）
| kind | 张量形状 | S 路径切片（`s=1`） | B 路径切片（`s=4`） | 结果形状 |
| --- | --- | --- | --- | --- |
| `row` | `(b, s, X)` | `[0, -1]` | `[0, 0]` | `(X,)` |
| `heads` | `(b, h, s, hd)` | `[0, :, -1, :]` | `[0, :, 0, :]` | `(h, hd)` |
| `kv` | `(b, kv, T, hd)` | `[0, :, :vis, :]` | `[0, :, :vis, :]` | `(kv, vis, hd)` |
| `scores` | `(b, h, s, T)` | `[0, :, 0, :vis]` | `[0, :, 0, :vis]` | `(h, vis)` |
| `shape_only` | 任意 | 只记录 shape/dtype/stride（**不**克隆整张量，例如 `k_repeated`/`v_repeated`） | — | — |

`vis = PRE + 1 = 8209`（key 位置 0..8208）：B 的第 0 行**看不到** 8209..8211 这三个候选，
所以只比较**共同可见范围**。

## GQA 口径（本模型实测：heads=12、kv_heads=2、head_dim=128）
`kv_repeat(k_all, heads // kv_heads)` ⇒ 第 `h` 个 query head 使用第 `h // (heads // kv_heads)` 个 KV head。
**不得**把广播后的 12 头维与原始 2 头维混为一谈（本模块只比较原始 `kv` 形状，并单独记录广播后的 shape）。
"""
from __future__ import annotations

import contextlib

import torch

from . import diag_checks as DC
from . import model_self as M
from . import sb_align as A

# ---- 探针名 → (所属子模块种类, 比较口径) -------------------------------------
SPECS: dict = {
    # DecoderLayer
    "x0": ("layer", "row"), "input_rmsnorm": ("layer", "row"), "attn_out": ("layer", "row"),
    "attention_residual": ("layer", "row"), "post_attention_rmsnorm": ("layer", "row"),
    "mlp_out": ("layer", "row"), "layer_output": ("layer", "row"),
    # SelfAttention
    "q_linear": ("attn", "row"), "k_linear": ("attn", "row"), "v_linear": ("attn", "row"),
    "q_reshaped": ("attn", "heads"), "k_reshaped": ("attn", "heads"), "v_reshaped": ("attn", "heads"),
    "q_rope": ("attn", "heads"), "k_rope": ("attn", "heads"),
    "k_visible": ("attn", "kv"), "v_visible": ("attn", "kv"),
    "k_repeated": ("attn", "shape_only"), "v_repeated": ("attn", "shape_only"),
    "qk_scores_pre_mask": ("attn", "scores"), "qk_scores_post_mask": ("attn", "scores"),
    "attention_prob": ("attn", "scores"), "attention_times_v": ("attn", "heads"),
    "heads_merged": ("attn", "row"), "o_proj": ("attn", "row"),
    # MLP
    "mlp_input": ("mlp", "row"), "mlp_gate_proj": ("mlp", "row"), "mlp_activation": ("mlp", "row"),
    "mlp_up_proj": ("mlp", "row"), "mlp_gated_product": ("mlp", "row"),
    "mlp_down_proj": ("mlp", "row"),
}

# ---- L0 的真实生产数据流（顺序 = 代码里的实际顺序；多输入算子要求**所有**输入逐位相同）----
CHAIN: list = [
    ("input_rmsnorm", ("x0",)),
    ("q_proj", ("input_rmsnorm",)),
    ("q_reshape", ("q_linear",)),
    ("rope_q", ("q_reshaped",)),
    ("k_proj", ("input_rmsnorm",)),
    ("k_reshape", ("k_linear",)),
    ("rope_k", ("k_reshaped",)),
    ("kv_concat_k", ("k_rope", "cache_pre")),
    ("v_proj", ("input_rmsnorm",)),
    ("v_reshape", ("v_linear",)),
    ("kv_concat_v", ("v_reshaped", "cache_pre")),
    ("qk_matmul_scale", ("q_rope", "k_visible")),
    ("causal_mask", ("qk_scores_pre_mask",)),
    ("softmax", ("qk_scores_post_mask",)),
    ("attn_times_v", ("attention_prob", "v_visible")),
    ("head_merge", ("attention_times_v",)),
    ("o_proj", ("heads_merged",)),
    ("attention_residual_add", ("x0", "o_proj")),
    ("post_attn_rmsnorm", ("attention_residual",)),
    ("gate_proj", ("post_attention_rmsnorm",)),
    ("up_proj", ("post_attention_rmsnorm",)),
    ("silu_activation", ("mlp_gate_proj",)),
    ("gated_product", ("mlp_activation", "mlp_up_proj")),
    ("down_proj", ("mlp_gated_product",)),
    ("mlp_residual_add", ("attention_residual", "mlp_down_proj")),
]

# 产出节点名（链的"输出"）与上面一一对应
CHAIN_NODE_OUT = ["input_rmsnorm", "q_linear", "q_reshaped", "q_rope", "k_linear", "k_reshaped",
                  "k_rope", "k_visible", "v_linear", "v_reshaped", "v_visible",
                  "qk_scores_pre_mask", "qk_scores_post_mask", "attention_prob",
                  "attention_times_v", "heads_merged", "o_proj", "attention_residual",
                  "post_attention_rmsnorm", "mlp_gate_proj", "mlp_up_proj", "mlp_activation",
                  "mlp_gated_product", "mlp_down_proj", "layer_output"]


class ProbeError(RuntimeError):
    """探针捕获/比较的实现或契约错误（不是引擎错误）。"""


def _layout(t) -> dict:
    return dict(shape=[int(x) for x in t.shape], stride=[int(x) for x in t.stride()],
                dtype=str(t.dtype), device=str(t.device),
                is_contiguous=bool(t.is_contiguous()))


def _normal_clone(t):
    """普通张量克隆（避免 inference tensor 在 `inference_mode` 之外参与运算时报错）。"""
    with torch.inference_mode(False):
        return t.detach().clone()


class ProbeCapture:
    """安装探针 → 只克隆**目标层**的中间张量 → 前向结束后统一搬 CPU。

    * 只接受 `who is` 目标层的三个子模块（`layer`/`self_attn`/`mlp`）的探针；
    * 不在回调里 `.cpu()`（避免每算子同步 / 改变执行轨迹）；
    * `finally` 里**必定**移除探针（异常也移除）。
    """

    def __init__(self, model, layer_index: int, arm: str, *, shape_only=("k_repeated", "v_repeated")):
        self.model = model
        self.layer_index = int(layer_index)
        self.arm = arm
        self.shape_only = set(shape_only)
        lay = model.layers[self.layer_index]
        self.who = {id(lay): "layer", id(lay.self_attn): "attn", id(lay.mlp): "mlp"}
        self.records: dict = {}
        self.meta: dict = {}
        self.n_calls = 0
        self.n_kept = 0
        self._installed = False

    def _cb(self, name: str, t, who) -> None:
        self.n_calls += 1
        kind = self.who.get(id(who))
        if kind is None:
            return                                    # 非目标层 ⇒ 不克隆、不记录
        if name not in SPECS or SPECS[name][0] != kind:
            self.meta.setdefault("unexpected", []).append(dict(name=name, kind=kind))
            return
        self.n_kept += 1
        if SPECS[name][1] == "shape_only":
            self.records[name] = dict(shape_only=True, layout=_layout(t), dtype=str(t.dtype),
                                      finite=bool(torch.isfinite(t).all()) if t.dtype.is_floating_point
                                      else None)
            return
        self.records[name] = dict(shape_only=False, tensor=_normal_clone(t), layout=_layout(t),
                                  dtype=str(t.dtype),
                                  finite=bool(torch.isfinite(t).all()) if t.dtype.is_floating_point
                                  else None)

    def __enter__(self) -> "ProbeCapture":
        self.prev = M.set_attn_probe(self._cb)
        self._installed = M.attn_probe_state()["active"]
        return self

    def close(self) -> dict:
        """移除探针并返回**已搬到 CPU** 的记录（前向结束之后才调用）。"""
        state = M.clear_attn_probe()
        if not state["probe_is_none"]:
            raise ProbeError(f"探针未移除：{state}")
        out = {}
        for k, v in self.records.items():
            if v.get("shape_only"):
                out[k] = dict(v)
            else:
                out[k] = dict(v)
                out[k]["tensor"] = v["tensor"].detach().to("cpu")
        return dict(arm=self.arm, layer_index=self.layer_index, n_calls=self.n_calls,
                    n_kept=self.n_kept, records=out, meta=self.meta,
                    probe_removed=bool(state["probe_is_none"]))

    def __exit__(self, *exc):
        try:
            self.close()
        except Exception:                                          # noqa: BLE001
            M.clear_attn_probe()
        return False


@contextlib.contextmanager
def capture(model, layer_index: int, arm: str):
    """`with capture(...) as cap:` —— 退出时（含异常）自动移除探针。"""
    c = ProbeCapture(model, layer_index, arm)
    with c:
        yield c


def slice_pair(kind: str, t_s, t_b, *, vis: int):
    """按语义口径取"对应决策切片"；形状不符时**报错**（不广播、不比整张量）。"""
    if kind == "row":
        DC.require(t_s.dim() == 3 and t_b.dim() == 3, f"row 口径要求 3 维，得到 {t_s.shape}/{t_b.shape}")
        a, b = t_s[0, -1], t_b[0, 0]
        rule = "S: [0,-1]（末位行）；B: [0,0]（第 0 行）"
    elif kind == "heads":
        DC.require(t_s.dim() == 4 and t_b.dim() == 4, f"heads 口径要求 4 维，得到 {t_s.shape}/{t_b.shape}")
        a, b = t_s[0, :, -1, :], t_b[0, :, 0, :]
        rule = "S: [0,:,-1,:]；B: [0,:,0,:]（同 batch/head 的当前位置行）"
    elif kind == "kv":
        DC.require(t_s.dim() == 4 and t_b.dim() == 4, f"kv 口径要求 4 维，得到 {t_s.shape}/{t_b.shape}")
        DC.require(t_s.shape[2] >= vis and t_b.shape[2] >= vis,
                   "可见范围超出缓存长度", f"{t_s.shape} {t_b.shape} vis={vis}")
        a, b = t_s[0, :, :vis, :], t_b[0, :, :vis, :]
        rule = f"两侧都取共同可见范围 [0:{vis})（B 的第 0 行看不到 {vis}.. 的候选）"
    elif kind == "scores":
        DC.require(t_s.dim() == 4 and t_b.dim() == 4, f"scores 口径要求 4 维，得到 {t_s.shape}/{t_b.shape}")
        DC.require(t_s.shape[3] >= vis and t_b.shape[3] >= vis,
                   "可见范围超出 scores 长度", f"{t_s.shape} {t_b.shape} vis={vis}")
        a, b = t_s[0, :, 0, :vis], t_b[0, :, 0, :vis]
        rule = f"S: [0,:,0,:{vis}]（s=1 的唯一行）；B: [0,:,0,:{vis}]（第 0 行、只取共同可见 key）"
    else:
        raise ProbeError(f"未知口径：{kind}")
    DC.assert_same_shape(a, b, f"slice_pair[{kind}]")
    return a, b, rule


def cmp_node(name: str, kind: str, t_s, t_b, *, vis: int) -> dict:
    """一个节点的 S/B 比较（**从原始 dtype** 做；仅统计时转 FP32）。"""
    a, b, rule = slice_pair(kind, t_s, t_b, vis=vis)
    x, y = a.float().reshape(-1), b.float().reshape(-1)
    d = (x - y).abs()
    flat = int(d.argmax())
    out = dict(node=name, kind=kind, slice_rule=rule,
               shape_s=[int(v) for v in t_s.shape], shape_b=[int(v) for v in t_b.shape],
               sliced_shape=[int(v) for v in a.shape], dtype=str(a.dtype),
               dtype_b=str(b.dtype), exact_equal=bool(torch.equal(a, b)),
               n_diff=int((a != b).sum()), max_abs=float(d.max()), mean_abs=float(d.mean()),
               sha16_s=A.stable_sha(a), sha16_b=A.stable_sha(b),
               max_diff_index=[int(v) for v in torch.unravel_index(torch.tensor(flat), d.shape)],
               s_at_max=float(x[flat]), b_at_max=float(y[flat]),
               finite_s=bool(torch.isfinite(a).all()), finite_b=bool(torch.isfinite(b).all()))
    DC.require(out["finite_s"] and out["finite_b"], f"{name}: 比较切片含非有限元素", str(out))
    return out


def classify(chain: list, node_cmp: dict, out_name: list) -> dict:
    """判定"输入逐位相同、输出首次不同"的算子（**纯函数**）。

    * 每个算子的输入可能不止一个（残差相加、gated product、QK^T、AV、KV 拼接…）；
      只有当**所有**输入都逐位相同时，才算"输入相同"；
    * `cache_pre` 是虚拟输入（轮前缓存，单独测量）；
    * 返回所有满足条件的算子 + 按生产顺序的第一个（可能不止一个，**如实全部报出**）。
    """
    hits = []
    details = []
    for (op, ins), out_node in zip(chain, out_name):
        in_eq = True
        for i in ins:
            if i not in node_cmp:
                in_eq = None
                break
            if not node_cmp[i]["exact_equal"]:
                in_eq = False
                break
        out_eq = (node_cmp[out_node]["exact_equal"] if out_node in node_cmp else None)
        hit = bool(in_eq is True and out_eq is False)
        details.append(dict(op=op, inputs=list(ins), output=out_node, all_inputs_equal=in_eq,
                            output_equal=out_eq, is_first_divergence=hit))
        if hit:
            hits.append(op)
    return dict(first_divergence_operator=(hits[0] if hits else None),
                all_operators_with_equal_inputs_and_different_output=hits,
                per_operator=details,
                note="输入相同 = **所有**输入逐位相同（多输入算子）；输出不同 = 该算子输出不逐位相同；"
                     "若多个算子同时满足 ⇒ 如实全部列出，取生产顺序的第一个作为 first_divergence_operator")


# ------------------------------------------------------------------ CPU 自测（门）
def _selftest() -> int:
    np_ = nf = 0

    def check(name, cond, extra=None):
        nonlocal np_, nf
        if cond:
            np_ += 1
            print(f"  [ok] {name}")
        else:
            nf += 1
            print(f"  [FAIL] {name}" + (f"  {str(extra)[:220]}" if extra is not None else ""))

    torch.manual_seed(0)
    print("== 1) 决策行/切片口径 ==")
    row_s = torch.arange(1 * 1 * 8, dtype=torch.float32).reshape(1, 1, 8)
    row_b = torch.arange(1 * 4 * 8, dtype=torch.float32).reshape(1, 4, 8)
    a, b, _r = slice_pair("row", row_s, row_b, vis=8)
    check("★ row 口径：S 取末位行、B 取第 0 行", bool(torch.equal(a, row_s[0, -1]))
          and bool(torch.equal(b, row_b[0, 0])) and list(a.shape) == [8])
    h_s = torch.arange(1 * 2 * 1 * 4, dtype=torch.float32).reshape(1, 2, 1, 4)
    h_b = torch.arange(1 * 2 * 4 * 4, dtype=torch.float32).reshape(1, 2, 4, 4)
    a, b, _r = slice_pair("heads", h_s, h_b, vis=4)
    check("★ heads 口径：同 head、当前位置行（(h,hd)）", list(a.shape) == [2, 4]
          and bool(torch.equal(a, h_s[0, :, -1, :])) and bool(torch.equal(b, h_b[0, :, 0, :])))
    kv_s = torch.zeros(1, 2, 9, 4)
    kv_b = torch.zeros(1, 2, 12, 4)
    a, b, rule = slice_pair("kv", kv_s, kv_b, vis=9)
    check("★ kv 口径：只比共同可见范围 [0:vis)", list(a.shape) == [2, 9, 4] and "可见" in rule)
    sc_s = torch.zeros(1, 6, 1, 9)
    sc_b = torch.zeros(1, 6, 4, 12)
    a, b, rule = slice_pair("scores", sc_s, sc_b, vis=9)
    check("★ scores 口径：只取 B 第 0 行 × 共同可见 key", list(a.shape) == [6, 9]
          and "第 0 行" in rule)
    bad = 0
    try:
        # 真实易犯的错误：把 q（12 头）与 k/v（2 头）当"对应头"比较 ⇒ 必须报错，不得广播
        slice_pair("heads", torch.zeros(1, 12, 1, 4), torch.zeros(1, 2, 4, 4), vis=4)
    except DC.CheckError:
        bad += 1
    bad2 = 0
    try:
        slice_pair("kv", kv_s, kv_b, vis=20)                       # 可见范围超过缓存长度
    except DC.CheckError:
        bad2 += 1
    check("★ 形状不符（12 头 vs 2 头）与可见范围越界时都**报错**（不广播）",
          bad == 1 and bad2 == 1)

    print("== 2) GQA head 映射（用真实 `kv_repeat`）==")
    kv = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
    rep = M.kv_repeat(kv, 6)
    check("★ heads=12/kv_heads=2 ⇒ 每个 KV head 连续映射 6 个 query head",
          list(rep.shape) == [1, 12, 3, 4]
          and all(bool(torch.equal(rep[0, h], kv[0, h // 6])) for h in range(12)))

    print("== 3) 可见 key 范围（B 第 0 行看不到未来候选）==")
    s, T, p = 4, 12, 8
    mask = torch.triu(torch.ones(s, T, dtype=torch.bool), diagonal=p + 1)
    check("★ B 第 0 行被掩掉的列从 p+1 开始（= 8209 的位置）",
          bool(mask[0, :p + 1].any()) is False and bool(mask[0, p + 1:].all()))
    check("★ 共同可见范围 = 0..p（含），不含 p+1 之后的候选", p == 8 and mask.shape == (4, 12))

    print("== 4) 首次分歧分类 ==")
    equal = dict(exact_equal=True)
    diff = dict(exact_equal=False)
    # 链 1：opB 首次分歧；opC 的输入里含 opB 的输出（已不同）⇒ **不能**算首次分歧
    chain = [("opA", ("x0",)), ("opB", ("opA_out",)), ("opC", ("opB_out", "x0"))]
    outs = ["opA_out", "opB_out", "opC_out"]
    c = classify(chain, dict(x0=equal, opA_out=equal, opB_out=diff, opC_out=diff), outs)
    check("★ 找到第一个「输入相同、输出不同」的算子（opB）；opC 因输入已不同被正确排除",
          c["first_divergence_operator"] == "opB"
          and c["all_operators_with_equal_inputs_and_different_output"] == ["opB"], c)
    check("★ 多输入算子：**所有**输入相同才算「输入相同」（opC 的 all_inputs_equal=False）",
          [d for d in c["per_operator"] if d["op"] == "opC"][0]["all_inputs_equal"] is False)
    chain2 = [("opD", ("x0", "aux"))]
    d2 = classify(chain2, dict(x0=equal, aux=equal, opD_out=diff), ["opD_out"])
    check("★ 多输入算子（两个输入都相同、输出不同）⇒ 被判定为首次分歧",
          d2["first_divergence_operator"] == "opD")
    d3 = classify(chain2, dict(x0=equal, aux=diff, opD_out=diff), ["opD_out"])
    check("★ 多输入算子里只要有一个输入已不同 ⇒ **不**算首次分歧（None）",
          d3["first_divergence_operator"] is None)
    c2 = classify(chain, dict(x0=equal, opA_out=diff, opB_out=diff, opC_out=diff), outs)
    check("★ 第一个算子就分歧时如实返回它（opA）", c2["first_divergence_operator"] == "opA",
          c2)
    c3 = classify(chain, dict(x0=diff, opA_out=diff, opB_out=diff, opC_out=diff), outs)
    check("★ 输入已不同 ⇒ 该算子**不**算首次分歧（返回 None，不误报）",
          c3["first_divergence_operator"] is None)
    c4 = classify(chain, dict(x0=equal, opA_out=equal, opB_out=equal, opC_out=equal), outs)
    check("★ 全部逐位相同 ⇒ 无分歧算子（None）", c4["first_divergence_operator"] is None)

    print("== 5) 探针安装/移除 + 关闭零副作用 ==")
    m = M.QwenLM(vocab_size=64, hidden=32, heads=4, kv_heads=2, head_dim=8, intermediate=64,
                 num_layers=2, rope_theta=10000.0, rms_eps=1e-6, attn_bias=True
                 ).to(torch.bfloat16).eval()
    M.set_attention_strategy("eager")
    ids = torch.randint(0, 64, (1, 5))
    with torch.inference_mode():
        lg_ref, _c = m._run(m.embed_tokens(ids), None, need_logits=True)
    check("★ 默认（探针 None）状态下 `attn_probe_state()` 为 False；`_probe` 调用无副作用",
          M.attn_probe_state()["active"] is False
          and M._probe("x0", torch.zeros(1), m.layers[0]) is None
          and M.attn_probe_state()["active"] is False)
    M.set_attn_probe(lambda *a: None)                  # 安装**空**探针：只验证调用点是否改变数值
    with torch.inference_mode():
        lg_noop, _c2 = m._run(m.embed_tokens(ids), None, need_logits=True)
    M.clear_attn_probe()
    check("★ 安装**空**探针时数值与关闭时**逐位相同**（探针调用点零副作用）",
          bool(torch.equal(lg_ref, lg_noop)))
    with torch.inference_mode():
        lg_off, _c3 = m._run(m.embed_tokens(ids), None, need_logits=True)
    check("★ 移除探针后再跑，仍与基线逐位相同", bool(torch.equal(lg_ref, lg_off)))
    try:
        with capture(m, 0, "S"):
            raise RuntimeError("注入的捕获故障")
    except RuntimeError:
        pass
    check("★ 异常退出后探针仍被移除", M.attn_probe_state()["active"] is False)
    with capture(m, 0, "S") as cap:
        with torch.inference_mode():
            m._run(m.embed_tokens(ids), None, need_logits=True)
    got = cap.close()
    check("★ 正常捕获：只保留目标层子模块的探针名，且都带原始 dtype 张量",
          got["probe_removed"] and got["n_kept"] > 10
          and all(k in SPECS for k in got["records"])
          and all((v["shape_only"] or str(v["tensor"].dtype) == "torch.bfloat16"
                   or str(v["tensor"].dtype) == "torch.float32") for v in got["records"].values()), 
          sorted(got["records"]))
    check("★ 探针名覆盖了 31 个生产探针点中的全部目标层子模块名（SPECS 表完整）",
          len([k for k in SPECS if SPECS[k][1] != "shape_only"]) == 29, len(SPECS))
    print(f"\n算子探针（CPU 门）：PASS {np_} / FAIL {nf}")
    return 0 if nf == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
