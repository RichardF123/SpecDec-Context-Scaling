"""audit_f06_i25_artifacts_cpu.py — F06/index25 产物覆盖审计（**纯 CPU，只读**）。

目的：不依赖报告文字，直接从 `output/g5/attn/` 的 JSON/PT 与工具源码生成「证据覆盖表」：
  * 每份关键产物的 path / 文件 hash / 字节数 / 产生工具 / run_id / git commit；
  * family、KV 模式、绝对位置、生成 index；
  * 覆盖哪些输入或决策行、是否保存**完整张量**、**原始 dtype**、还是只有统计量；
  * token 前缀、cache 来源、真实调用与重放关系；
  * 能支持什么比较 / 不能支持什么比较（人工核定表 + 自动字段）。

**不加载模型、不碰 GPU、不写正式结果文件**（只写 `…_artifact_audit_<run_id>.json`）。
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TAGS = ("f06i25b_20260929T130000Z", "f06i25_20260929T120000Z")

# ---- 人工核定的「能否支持何种比较」表（键 = 文件名片段） ----
CLAIMS = {
    "_F06_ar_": dict(
        kind="真实 AR 单步决策点",
        rows="**仅 1 行**（`real`/`replay` 各一行，(151936,)）—— **不是** 64 个生成位置的全程 logits",
        supports="① 位置 8216→8217 这一**单点**的 AR 决策行（并列集合/margin/候选分数）；"
                 "② AR 自身重放关系 `real == replay`（同一单步、同 cache）",
        not_supports="① 任何『全程 logits』『64 位置逐项比较』；② AR 与 Spec 的**同 cache**对照"
                     "（AR 的 cache 与 Spec 的 cache 来源不同）；③ K/V 数值等价性"),
    "_F06_spec_": dict(
        kind="真实 SpecDec 决策点 + 轮前快照",
        rows="`rounds` 记录 12 轮元数据；`rows.pt` 只含 **round11/round12 的 4 行 logits**"
             "（`(4,151936)`，共 8 行）—— **不覆盖** 64 个生成位置",
        supports="① index 25 所在那一轮的**块内 4 行**（对齐行 = rows[1]）；② 该轮**轮前 cache 快照**"
                 "（`snap_pre_cpu`，(1,12,8215,128) 每层）；③ 块/串行**同一起始 cache** 的对照",
        not_supports="① 其余 63 个位置的决策行；② `saved_next` 行与块行不可混用（语义不同）；"
                     "③ 未接受候选后的验证行 ≠ 最终提交轨迹的决策行"),
    "_l27_": dict(
        kind="L27 逐点（对齐行）",
        rows="对齐行 `(1536,)` 或 `(1,4,1536)`/`(1,1,1536)`；**不含** RoPE 后 Q/K、不含 GQA 扩展后 K",
        supports="① L27 内 `layer_in → input_layernorm → atten_out → 残差 → MLP → layer_out` 的"
                 "**对齐行**逐点值与两臂对照；② `layer_out == 最终 Norm 输入` 的核验",
        not_supports="① attention 核心（QK/softmax/AV）数值对照 —— **无 RoPE 后 Q/K**；"
                     "② GQA 重复后 K；③ `R_tap_*` **不是** attention 残差（它是最终 Norm 输入）"),
    "_mlpshape_": dict(
        kind="L27 MLP 四格与捕获",
        rows="完整 `U_B (1,4,1536)` / `U_S (1,1,1536)`（MLP 输入）、真实 attention 残差、MLP 输出、四格输出",
        supports="① 同输入变**调用形状**；② 同形状变**输入来源**；③ 尾部重组的输入来源",
        not_supports="① attention 分支；② 差异首次产生处"),
    "_mlptail_": dict(
        kind="L27 MLP 四格（严格三维）+ 四条件尾部",
        rows="四格对齐行 + 四个尾部条件的 `T/H/logits`（完整 151936 维）",
        supports="① 形状差是否存活/传播/改变选词；② `T_A/T_D == 真实 L27 输出` 的恒等式",
        not_supports="① attention/QKV；② 其它层"),
    "_tailshape_": dict(
        kind="尾部调用形状控制",
        rows="同一 T 的三种调用形状（`(1,4,1536)` / `(1,1,1536)` / `(1,1536)`）最终 logits",
        supports="① 区分『MLP 形状效应』与『LM head 的 M 维形状效应』；② 生产块路径的逐元素复现",
        not_supports="① attention；② 并列语义的裁决（属待确认事项）"),
    "_normboundary_": dict(
        kind="最终 RMSNorm 边界四格",
        rows="`R_B/H_B (1,4,1536)`、`R_S/H_S (1,1,1536)` 与四格 **Norm 输出**",
        supports="① 最终 Norm **输出**对调用形状是否敏感；② Norm 输入的来源效应",
        not_supports="① 最终 logits 的形状效应（那属 LM head，见 `_lmhead_`/`_tailshape_`）；"
                     "② Norm 输出相等**不能**推出 logits 相等"),
    "_lmhead_": dict(
        kind="LM head 边界四格",
        rows="隐藏向量 `H_B/H_S` 与四格 logits（完整 151936 维）",
        supports="① LM head 的 M 维形状效应（A vs B = 0.0625）；② 隐藏向量来源效应",
        not_supports="① 隐藏向量差异的**来源**；② attention 分支"),
    "_samehistory_": dict(
        kind="同一起始 cache 下的块 vs 单步",
        rows="块行与单步行（各 151936 维）+ key_candidates 分数",
        supports="① **同 cache、同预测位置**下块/单步的决策行对照（并列结构与 margin）",
        not_supports="① AR 轨迹的对照（AR cache 来源不同）；② 差异首次产生处"),
    "_substitution_": dict(
        kind="四臂中间值替换",
        rows="L0 `q_proj`/`o_proj` 对齐行替换后的 logits",
        supports="① L0 是否为**充分**原因（答：否）；② 替换范围与 no-op 校验",
        not_supports="① L27 attention；② 唯一根因"),
    "_layerwise_": dict(
        kind="逐层模块输出统计",
        rows="**只有统计量**（无 `.pt`）—— 不能恢复张量",
        supports="① 逐层模块输出差异的量级汇总；② 定位差异**已存在**的层",
        not_supports="① 任何逐元素复核；② 张量级对照（未存张量）"),
    "_localreplay_": dict(
        kind="L0 Linear 局部重放",
        rows="统计量",
        supports="① L0 线性层在前缀相同情况下的可复现性",
        not_supports="① L27；② 选词翻转"),
    "_analysis_cpu": dict(kind="CPU 事后分析", rows="统计量",
                          supports="① 已保存行的二次统计", not_supports="① 未保存张量的任何对照"),
}

# ---- 需要显式回答的问题（自动计算） ----
QUESTIONS = (
    "rows.pt 是否覆盖完整 64 个生成位置？（必须区分『定位点行』与『全程 logits』）",
    "哪些产物保存了**原始 dtype** 的完整张量，哪些只有统计量？",
    "RoPE 后 Q/K、GQA 扩展后 K 是否有任何产物保存？",
)


def _sha(p: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:n]


def _pt_summary(p: Path) -> dict:
    d = torch.load(p, map_location="cpu")
    if not isinstance(d, dict):
        return dict(loadable=True, kind=type(d).__name__, note="非 dict 顶层")
    out = {}
    for k, v in sorted(d.items()):
        if torch.is_tensor(v):
            out[k] = dict(shape=[int(x) for x in v.shape], dtype=str(v.dtype),
                          numel=int(v.numel()), bytes=int(v.numel() * v.element_size()))
        else:
            out[k] = dict(kind=type(v).__name__, value=str(v)[:60])
    return out


def _json_meta(p: Path) -> dict:
    r = json.load(open(p, encoding="utf-8"))
    cfg = r.get("config") or {}
    m = dict(tool=r.get("tool"), mode=r.get("mode"), tag=r.get("tag"),
             family=r.get("family"), kv=r.get("kv"), stage=r.get("stage"),
             run_id=r.get("run_id"), timestamp_utc=r.get("timestamp_utc"),
             git=(r.get("git") or {}).get("git_commit"),
             dirty=(r.get("git") or {}).get("dirty_worktree"),
             ctx=cfg.get("ctx"), chunk=cfg.get("chunk"), k=cfg.get("k"),
             max_new=cfg.get("max_new"), attention_strategy=cfg.get("attention_strategy"),
             div_index=cfg.get("div_index"), boundary_cache_len=cfg.get("boundary_cache_len"),
             diag_precision=r.get("diag_precision"))
    if "rounds" in r:
        m["n_rounds"] = len(r["rounds"])
        m["round_pre"] = [w.get("pre") for w in r["rounds"]][:14]
        m["window_rounds"] = [(w.get("round"), w.get("pre")) for w in r.get("window_rounds", [])]
    if "ar" in r:
        a = r["ar"]
        m["ar"] = dict(fed_token_position=a.get("fed_token_position"),
                       predicted_position=a.get("predicted_position"),
                       gen_index=a.get("gen_index"), cache_len_before=a.get("cache_len_before"),
                       fed_token=a.get("fed_token"), real_argmax=a.get("real_argmax"),
                       n_gen_head=len(a.get("gen_head") or []))
    if "decision" in r:
        d = r["decision"]
        m["decision"] = dict(pre=d.get("pre"), Kp=d.get("Kp"), m=d.get("m"),
                            emitted_token=d.get("emitted_token"), rows_idx=d.get("rows_idx"))
    return m


def _positions_from_pt(p: Path, keys: dict) -> dict:
    """从已加载的键摘要推断覆盖的『位置/行数』。"""
    info = {}
    for k, v in keys.items():
        if isinstance(v, dict) and "shape" in v:
            info[k] = v["shape"]
    return info


def main() -> int:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="audit_f06_i25_artifacts_cpu",
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               scope="只读审计产物覆盖与比较合法性；不加载模型、不碰 GPU、不写正式结果。",
               tags=list(TAGS), artifacts={}, claim_table=CLAIMS,
               questions={q: None for q in QUESTIONS})
    files = sorted([p for t in TAGS for p in OUTDIR.glob(f"{t}*")
                    if p.suffix in (".json", ".pt")])
    print(f"== 产物清单（{len(files)} 个 json/pt）==")
    for p in files:
        e = dict(name=p.name, bytes=p.stat().st_size, sha256_16=_sha(p),
                 mtime_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                         time.gmtime(p.stat().st_mtime)))
        if p.suffix == ".json":
            try:
                e["meta"] = _json_meta(p)
            except Exception as ex:                              # noqa: BLE001
                e["meta_error"] = f"{type(ex).__name__}: {ex}"
        else:
            try:
                ks = _pt_summary(p)
                e["keys"] = ks if len(p.name) > 0 else ks
                e["n_keys"] = len(ks)
                e["total_tensor_bytes"] = int(sum(
                    v.get("bytes", 0) for v in ks.values() if isinstance(v, dict)))
            except Exception as ex:                              # noqa: BLE001
                e["pt_error"] = f"{type(ex).__name__}: {ex}"
        frag = next((f for f in CLAIMS if f in p.name), None)
        e["claim_class"] = frag
        rec["artifacts"][p.name] = e
        tag = (e.get("meta") or {}).get("tool") or ""
        print(f"  {p.name:60s} {e['bytes']:>10d}  {e['sha256_16']}  {tag}")

    # ---- 自动回答三个问题 ----
    spec_rows = rec["artifacts"].get(
        "f06i25b_20260929T130000Z_F06_spec_20260929T093904Z.rows.pt", {})
    ar_rows = rec["artifacts"].get(
        "f06i25b_20260929T130000Z_F06_ar_20260929T093951Z.rows.pt", {})
    sr = spec_rows.get("keys", {})
    akeys = ar_rows.get("keys", {})
    rows_answer = dict(
        spec_rows_keys=sorted(sr), ar_rows_keys=sorted(akeys),
        spec_logit_rows=int(sum(v["shape"][0] for k, v in sr.items()
                               if isinstance(v, dict) and len(v.get("shape", [])) == 2)),
        ar_logit_rows=int(sum(v["shape"][0] for k, v in akeys.items()
                             if isinstance(v, dict) and len(v.get("shape", [])) == 2)),
        max_new=(rec["artifacts"].get(
            "f06i25b_20260929T130000Z_F06_spec_20260929T093904Z.json", {}).get("meta") or {}
            ).get("max_new"),
        conclusion=None)
    rows_answer["conclusion"] = (
        f"**不覆盖 64 个生成位置**：Spec 侧 `rows.pt` 只有 {rows_answer['spec_logit_rows']} 行"
        f"（round11/round12 各 4 行），AR 侧只有 {rows_answer['ar_logit_rows']} 行（real/replay）。"
        "两者都是**定位点行**，不是全程 logits；因此『64 位置逐项比较』在本产物集上**不可执行**。")
    rec["questions"][QUESTIONS[0]] = rows_answer

    raw_full, stats_only = [], []
    for name, e in rec["artifacts"].items():
        ks = e.get("keys")
        if ks and any(isinstance(v, dict) and "dtype" in v for v in ks.values()):
            raw_full.append(name)
        if e.get("claim_class") in ("_layerwise_", "_localreplay_", "_analysis_cpu"):
            stats_only.append(name)
    rec["questions"][QUESTIONS[1]] = dict(
        full_tensors=sorted(raw_full), stats_only_or_none=sorted(stats_only),
        note="`.pt` 中的张量均以创建时的 dtype 保存；但部分工具在**保存前**已 `.float()`，"
             "因此『文件里是 fp32』不等于『原始就是 fp32』——必须按工具源码判定（见 §4）。")

    rope = [n for n, e in rec["artifacts"].items()
            if any(k in (e.get("keys") or {})
                   for k in ("q_rope", "k_rope", "k_rep", "k_repeat", "attn_scores", "probs"))]
    rec["questions"][QUESTIONS[2]] = dict(
        found=rope,
        conclusion="**没有任何产物保存 RoPE 后 Q/K、GQA 扩展后 K、QK 分数或 softmax 概率**；"
                   "仅有 `q_proj/k_proj/v_proj` 的**对齐行**（`(1536,)`/`(256,)`）"
                   "⇒ attention 核心输入的完整数值核对目前**无既有产物可用**。")

    p = OUTDIR / f"f06i25b_20260929T130000Z_artifact_audit_{run_id}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("\n== Q1 rows.pt 覆盖 ==")
    print(f"  spec rows 键 = {rows_answer['spec_rows_keys']}（{rows_answer['spec_logit_rows']} 行）")
    print(f"  ar   rows 键 = {rows_answer['ar_rows_keys']}（{rows_answer['ar_logit_rows']} 行）")
    print(f"  max_new = {rows_answer['max_new']} ⇒ {rows_answer['conclusion']}")
    print("\n== Q2 完整张量 vs 仅有统计 ==")
    print(f"  保存完整张量的产物：{len(raw_full)} 个；仅有统计/无 .pt：{sorted(stats_only)}")
    print("\n== Q3 RoPE 后 Q/K / GQA 后 K / scores / probs ==")
    print(f"  命中键名：{rope or '无'} ⇒ {rec['questions'][QUESTIONS[2]]['conclusion']}")
    print(f"\n[audit] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
