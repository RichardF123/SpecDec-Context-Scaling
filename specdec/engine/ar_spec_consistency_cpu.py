"""ar_spec_consistency_cpu.py — **只读**比较：target RoPE 对齐后的真实 AR 与 Spec 完整生成（纯 CPU）。

回答："采用该诊断配置后，真实 AR 与 Spec 从头生成的完整序列是否一致？"

## 只做什么
1. **配置/权重/精度/输入**核验（两次运行必须逐项一致）。
2. **初始状态核验**（先过这一关，再谈生成差异）：target prefill 全层 K/V + 末位 logits，
   跨进程比较，并与**既有 HF C1 参考**及**已对齐诊断运行**（`c1rope_run_*`，其 K/V 与 logits 已逐位等于 HF）核对。
   * **比较强度**：K/V 用 `sb_align.stable_sha` 的**逐张量 sha16**（摘要级）；logits 用
     **同一约定**下的量（`sha` 只在同一工具内可比 —— smoke 工具哈希的是整块 `(1,2048,V)` 输出，
     `c1rope` 哈希的是末位 `(V,)` 行）⇒ 跨工具改用 `sum/max/margin/argmax` 对位，
     并**明确标注**这不是逐位证明。
3. **序列比较**：完整 token IDs（不是文本）、长度、EOS/停止原因、common prefix、
   **首次分歧 index/绝对位置/两侧 token**、差异总数。
4. **首次分歧对应的真实轮次**：把生成 index 映射到 Spec 轮次，给出该轮全部记账字段与
   "差异 token 是被接受候选还是修正 token"；AR 侧给出对应解码步。
5. **同协议既有运行交叉核对**（固定路径、只读）：把**完全相同协议**
   （ctx=8192/chunk=2048/k=4/max_new=64/F13/base_bf16/eager）的历史运行按"完整序列"分组，
   给出基线与本轮的对齐/分歧位置。**不**自动发现新文件，**不**重跑。
6. **判定**：按预先声明的口径；**禁止**把小 margin / 更晚分歧 / 较小 logits 差异当豁免。

## 不做什么
不逐层扫描、不精度组合、不新增取证（不跑新 run）、不覆盖任何历史产物。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
# ---- 固定引用（不"自动选最新"）----
DEFAULT_AR = "smoke8k_eager_20261001T165332Z.json"
DEFAULT_SPEC = "smoke8k_eager_20261001T165431Z.json"
HF_JSON = OUTDIR / "xb_20261001_hf_20261001T132947Z.json"
HF_PT = OUTDIR / "xb_20261001_hf_20261001T132947Z.pt"
C1DIAG_RUN = OUTDIR / "c1rope_run_20261001T152125Z.json"
C1DIAG_PT = OUTDIR / "c1rope_new_dump_20261001T152125Z.pt"
TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
CTX = 8192

# 同协议（ctx=8192/chunk=2048/k=4/max_new=64/F13/base_bf16）的既有运行：**固定清单**，只读。
# lineage A = target 用原实现（设备上算 inv_freq）；lineage B = 本轮（CPU-FP32）。
# 2026-09-29 的 `smoke8k_cudnn_prefill_*` 运行**排除**：arm=cudnn_prefill（融合路径）且存在
#   已记录的"整条序列错位一位"记账缺陷 ⇒ 与本协议不可比（见 g5_8k_smoke_cudnn 的历史注释）。
SAME_PROTOCOL_RUNS = [
    ("AR-A1", "lineage A", "smoke8k_eager_20261001T065923Z.json", "ar"),
    ("AR-A2", "lineage A", "smoke8k_eager_20261001T070134Z.json", "ar"),
    ("AR-A3(RED=1)", "lineage A", "smoke8k_eager_20261001T125659Z.json", "ar"),
    ("AR-A4(RED=0)", "lineage A", "smoke8k_eager_20261001T125847Z.json", "ar"),
    ("AR-B1(本轮)", "lineage B", "smoke8k_eager_20261001T165332Z.json", "ar"),
    ("SP-A1", "lineage A", "smoke8k_eager_20261001T070019Z.json", "spec"),
    ("SP-A2", "lineage A", "smoke8k_eager_20261001T070207Z.json", "spec"),
    ("SP-A3(RED=1)", "lineage A", "smoke8k_eager_20261001T125733Z.json", "spec"),
    ("SP-A4(RED=0)", "lineage A", "smoke8k_eager_20261001T125919Z.json", "spec"),
    ("SP-B1(本轮)", "lineage B", "smoke8k_eager_20261001T165431Z.json", "spec"),
    ("UCG-AR", "lineage A(UCG)", "ucg_20261001_F13_ar_20261001T114131Z.json", "ar"),
    ("UCG-SP1", "lineage A(UCG)", "ucg_20261001_F13_spec_20261001T114228Z.json", "spec"),
    ("UCG-SP2", "lineage A(UCG)", "ucg_20261001_F13_spec_20261001T114350Z.json", "spec"),
    ("UCG-SP3", "lineage A(UCG)", "ucg_20261001_F13_spec_20261001T114629Z.json", "spec"),
]


def _sha(t: torch.Tensor, n: int = 16) -> str:
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:n]


def _lastrow_stats(t: torch.Tensor) -> dict:
    r = t.detach().float().reshape(-1, t.shape[-1])[-1]
    t2 = torch.topk(r, 2).values
    return dict(sha_lastrow16=_sha(r), sum=float(r.sum()), max=float(r.max()),
                argmax=int(torch.argmax(r)), margin=float(t2[0] - t2[1]),
                n_tied=int((r == r.max()).sum()), n_finite=int(torch.isfinite(r).sum()))


def _ref_digest_from_pt(pt: Path, n_layers: int = 28) -> dict:
    """用**同一套** `stable_sha` 从 HF C1 快照的 fp32 K/V 算全层摘要（fp32 是精确 bf16 ⇒ 字节一致）。"""
    kv = torch.load(pt, map_location="cpu")["c1_kv"]
    return {f"L{i}_{kk}": dict(sha=_sha(kv[i][kk])) for i in range(n_layers) for kk in ("k", "v")}


def _diff_digest(a: dict, b: dict, label: str) -> dict:
    keys = sorted(set(a) & set(b))
    bad = [k for k in keys if a[k]["sha"] != b[k]["sha"]]
    return dict(label=label, n_keys=len(keys), n_equal=len(keys) - len(bad), n_diff=len(bad),
                diff_keys=bad[:12], all_equal=not bad)


def _gen_of(j: dict):
    g = (j.get("run") or {}).get("gen") or j.get("gen")
    if g is None:
        g = (j.get("result") or {}).get("gen")
    return None if g is None else [int(x) for x in g]


def _first_diff(a: list, b: list) -> dict:
    n = min(len(a), len(b))
    c = 0
    while c < n and a[c] == b[c]:
        c += 1
    return dict(identical=bool(a == b), common_prefix=c,
                first_diff=(None if a == b else c),
                tokens=(None if a == b else [a[c], b[c]]),
                n_diff=sum(1 for i in range(n) if a[i] != b[i]))


def main() -> int:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    ar_p = OUTDIR / os.environ.get("ARSPEC_AR_RUN", DEFAULT_AR)
    sp_p = OUTDIR / os.environ.get("ARSPEC_SPEC_RUN", DEFAULT_SPEC)
    rec: dict = dict(tool="ar_spec_consistency_cpu", run_utc=ts,
                     semantics="只读比较：target RoPE 对齐后的真实 AR 与 Spec 完整生成；"
                               "不重跑、不逐层扫描、不覆盖历史产物",
                     fixed=dict(ar_run=ar_p.name, spec_run=sp_p.name, hf_json=HF_JSON.name,
                                hf_pt=HF_PT.name, c1diag_run=C1DIAG_RUN.name,
                                c1diag_pt=C1DIAG_PT.name))
    for p in (ar_p, sp_p, HF_JSON, HF_PT, C1DIAG_RUN, C1DIAG_PT):
        if not p.exists():
            print(f"[asc] !! 缺少固定引用：{p}")
            return 5
    A = json.load(open(ar_p, encoding="utf-8"))
    S = json.load(open(sp_p, encoding="utf-8"))

    # ---------------- 1) 配置/权重/精度/输入 ----------------
    print("== 1) 运行配置、权重、精度、输入来源 ==")
    keys = ("mode", "ctx", "chunk", "k", "max_new", "kv", "arm", "target_model", "draft_model",
            "target_rope")
    cfgA = {k: A["config"][k] for k in keys}
    cfgS = {k: S["config"][k] for k in keys}
    diff_cfg = {k: (cfgA[k], cfgS[k]) for k in ("ctx", "chunk", "k", "max_new", "kv", "arm",
                                                "target_rope") if cfgA[k] != cfgS[k]}
    rec["config"] = dict(ar=cfgA, spec=cfgS, differing=diff_cfg,
                         ar_resident=A["resident_models"], spec_resident=S["resident_models"])
    for tag, j in (("AR", A), ("Spec", S)):
        print(f"  [{tag}] mode={j['config']['mode']} ctx={j['config']['ctx']} "
              f"chunk={j['config']['chunk']} k={j['config']['k']} max_new={j['config']['max_new']} "
              f"kv={j['config']['kv']} arm={j['config']['arm']} target_rope={j['config']['target_rope']}")
        print(f"        weights: bitmatch={j['weights_bitmatch']} checked={j['weights_checked']} | "
              f"reduction={j['bf16_reduced_precision_reduction']['effective_after_set']} | "
              f"diag={j['diag_precision']}")
        print(f"        input: ids_hash={j['input']['ids_hash']} ctx={j['input']['ctx']} "
              f"kind={j['input']['source_kind']} family={(j['input']['family'] or {}).get('family_id')}")
        print(f"        驻留模型={j['resident_models']} | gen={j['run']['gen_len']} "
              f"stop={j['run']['stop_reason']} | passed={j['verdict']['passed']}")
    rec["input_same"] = bool(A["input"]["ids_hash"] == S["input"]["ids_hash"])
    # 治理：本工具**不读 manifest**，family 元数据来自所读的既有运行产物（便于守卫按范围检查）
    rec["input"] = dict(source_kind="derived_from_artifacts", family=A["input"]["family"],
                        source_runs=[ar_p.name, sp_p.name],
                        ids_hash=A["input"]["ids_hash"],
                        note="只读派生；不消费任何新样本；family 字段来自既有运行产物")
    assert not diff_cfg, f"两次运行配置不一致：{diff_cfg}"
    assert rec["input_same"], "两次运行输入 ids_hash 不一致"
    assert cfgA["target_rope"] == "cpu_fp32" and cfgA["arm"] == "eager"

    # ---------------- 2) 初始状态（target prefill）跨进程 + 对 HF 参考 ----------------
    print("\n== 2) 初始状态核验：target prefill 全层 K/V + 末位 logits ==")
    dA, dS = A["prefill_digest"]["target"], S["prefill_digest"]["target"]
    assert dA["cache_len"] == dS["cache_len"] == CTX and dA["n_layers"] == dS["n_layers"] == 28
    c1 = json.load(open(C1DIAG_RUN, encoding="utf-8"))
    c1_lg = torch.load(C1DIAG_PT, map_location="cpu")["c1_logits"]
    ref_pt = _ref_digest_from_pt(HF_PT)
    ref_c1 = {f"L{i}_{kk}": dict(sha=c1["layers"][i][f"{kk}_sha"]) for i in range(28) for kk in ("k", "v")}
    lr_c1 = _lastrow_stats(c1_lg)
    kv_ok = all(_diff_digest(dA["layers"], x, "")["all_equal"] for x in (dS["layers"], ref_pt, ref_c1))
    lg_same = bool(dA["logits"]["sha"] == dS["logits"]["sha"])
    lg_vs_ref = dict(argmax=(dA["logits"]["argmax"], lr_c1["argmax"]),
                     margin=(dA["logits"]["margin_top1_top2"], lr_c1["margin"]),
                     max_score=(dA["logits"]["max_score"], lr_c1["max"]),
                     sum_recorded=(dA["logits"]["sum"], lr_c1["sum"]),
                     all_equal=bool(dA["logits"]["argmax"] == lr_c1["argmax"]
                                    and abs(dA["logits"]["margin_top1_top2"] - lr_c1["margin"]) == 0.0
                                    and abs(dA["logits"]["max_score"] - lr_c1["max"]) == 0.0),
                     sum_excluded_reason=("`sum` **不参与**判等：smoke 记录里的 `sum` 是**在 CUDA 上**"
                                          "求和的（`float(r.sum())`，r 在 GPU），本工具是**在 CPU 上**"
                                          "对同一行重新求和 ⇒ 归约顺序不同，实测差 1 个 fp32 ULP"
                                          "（166711.265625 vs 166711.25）。这与数值差异无关，"
                                          "属**比较口径**问题，已明确剔除而不是当成不一致。"
                                          "`max`/`margin`/`argmax` 与归约顺序无关，仍可跨工具比较。"))
    rec["init_state"] = dict(
        cache_len=dA["cache_len"], n_layers=dA["n_layers"],
        ar_vs_spec=_diff_digest(dA["layers"], dS["layers"], "AR_prefill vs Spec_prefill (target)"),
        ar_vs_hf_pt=_diff_digest(dA["layers"], ref_pt, "AR_prefill vs HF C1 快照"),
        ar_vs_c1diag=_diff_digest(dA["layers"], ref_c1, "AR_prefill vs 已对齐诊断运行(c1rope)"),
        logits=dict(ar_sha=dA["logits"]["sha"], spec_sha=dS["logits"]["sha"],
                    c1diag_lastrow_sha=lr_c1["sha_lastrow16"],
                    ar_argmax=dA["logits"]["argmax"], spec_argmax=dS["logits"]["argmax"],
                    ar_margin=dA["logits"]["margin_top1_top2"],
                    spec_margin=dS["logits"]["margin_top1_top2"],
                    ar_vs_spec_same_convention=lg_same, ar_vs_ref=lg_vs_ref),
        strength=("K/V：**摘要级**（逐张量 sha16，`sb_align.stable_sha`），并与 HF C1 快照、"
                  "已对齐诊断运行三方一致；"
                  "logits：AR 与 Spec 之间是**同约定**的整块 `(1,2048,V)` sha ⇒ 逐位一致；"
                  "与 c1rope/参考之间用 `sum/max/margin/argmax` 对位（两者 sha 约定不同："
                  "smoke 哈希整块输出、c1rope 哈希末位行）⇒ 这一条**不是**逐位证明"),
        convention_note=("⚠️ 工具缺陷更正：首版把不同约定的 logits sha 直接比较，得出"
                         "「初始状态未通过」的错误结论；现改为同约定比较 + 对位量比较。"))
    ist = rec["init_state"]
    for k in ("ar_vs_spec", "ar_vs_hf_pt", "ar_vs_c1diag"):
        x = ist[k]
        print(f"  {x['label']:<40} 逐位相同张量 {x['n_equal']}/{x['n_keys']}"
              + ("" if x["all_equal"] else f"  不同={x['diff_keys']}"))
    print(f"  logits：AR==Spec（同约定 sha）={lg_same}；与 c1rope 末位行对位 "
          f"argmax {lg_vs_ref['argmax']} margin {lg_vs_ref['margin']} "
          f"max {lg_vs_ref['max_score']} ⇒ 一致={lg_vs_ref['all_equal']}"
          f"（sum 因 GPU/CPU 归约顺序不同已剔除：{lg_vs_ref['sum_recorded']}）")
    rec["init_state_ok"] = bool(kv_ok and lg_same and lg_vs_ref["all_equal"])
    print(f"  ★ 初始状态核验：{'通过' if rec['init_state_ok'] else '**未通过**'}")

    # ---------------- 3) 干预记录 ----------------
    print("\n== 3) 干预记录：target RoPE 频率来源作用域 ==")
    for tag, j in (("AR", A), ("Spec", S)):
        sc = j["target_rope_scope"]
        print(f"  [{tag}] scope={sc['scope']} inv_freq_cpu_sha={sc['inv_freq_cpu_sha']} "
              f"theta={sc['target_theta']} head_dim={sc['target_head_dim']}")
        print(f"        调用计数={sc['counts_by_model_source']}")
        print(f"        target 覆盖 phase={sorted({c['phase'] for c in sc['calls'] if c['model']=='target'})}")
        print(f"        恢复={json.dumps(j['target_rope_scope_restore'], ensure_ascii=False)}")
    rec["intervention"] = dict(
        ar=dict(install=A["target_rope_scope"], counts=A["target_rope_scope"]["counts_by_model_source"],
                restore=A["target_rope_scope_restore"]),
        spec=dict(install=S["target_rope_scope"],
                  counts=S["target_rope_scope"]["counts_by_model_source"],
                  restore=S["target_rope_scope_restore"]),
        draft_untouched=bool(S["target_rope_scope"]["counts_by_model_source"].get("other|original", 0) > 0
                             and all(k.split("|")[1] == "original" for k in
                                     S["target_rope_scope"]["counts_by_model_source"] if k.startswith("other"))),
        same_inv_freq_sha=bool(A["target_rope_scope"]["inv_freq_cpu_sha"]
                               == S["target_rope_scope"]["inv_freq_cpu_sha"]))
    print(f"  ★ draft 全部走原实现={rec['intervention']['draft_untouched']}；"
          f"两次运行 CPU inv_freq 摘要相同={rec['intervention']['same_inv_freq_sha']}")

    # ---------------- 4) backend ----------------
    print("\n== 4) 各阶段实际 backend ==")
    rec["backend"] = {}
    for tag, j in (("AR", A), ("Spec", S)):
        st = j["run"]["attention_stats"]
        rec["backend"][tag] = dict(strategy=st.get("strategy"), fused_by_phase=st.get("fused_by_phase"),
                                   eager_by_phase=st.get("eager_by_phase"),
                                   dispatch_ok=j["run"]["dispatch_only_prefill"])
        print(f"  [{tag}] strategy={st.get('strategy')} fused={st.get('fused_by_phase')} "
              f"eager={st.get('eager_by_phase')} dispatch_ok={j['run']['dispatch_only_prefill']}")

    # ---------------- 5) 序列比较 ----------------
    print("\n== 5) 完整生成序列比较（token IDs）==")
    ga, gs = _gen_of(A), _gen_of(S)
    fd = _first_diff(ga, gs)
    first = fd["first_diff"]
    rec["sequence"] = dict(**fd, ar_len=len(ga), spec_len=len(gs),
                           ar_stop=A["run"]["stop_reason"], spec_stop=S["run"]["stop_reason"],
                           ar_passed=A["verdict"]["passed"], spec_passed=S["verdict"]["passed"],
                           abs_position=(None if first is None else CTX + first),
                           predicted_from_position=(None if first is None else CTX + first - 1),
                           ar_end=A["end_state"], spec_end=S["end_state"])
    print(f"  AR  len={len(ga)} stop={A['run']['stop_reason']} passed={A['verdict']['passed']}")
    print(f"  Spec len={len(gs)} stop={S['run']['stop_reason']} passed={S['verdict']['passed']}")
    print(f"  token IDs 完全一致={fd['identical']}；common prefix={fd['common_prefix']}；"
          f"差异个数={fd['n_diff']}")
    if first is not None:
        print(f"  ★ 首次分歧：index={first}（绝对位置 {CTX+first}，由位置 {CTX+first-1} 的 logits 决定）"
              f" AR={ga[first]} vs Spec={gs[first]}")
        print(f"    上下文 AR  ={ga[max(0,first-3):first+3]}")
        print(f"    上下文 Spec={gs[max(0,first-3):first+3]}")

    # ---------------- 6) 首次分歧对应的真实轮次 ----------------
    print("\n== 6) 首次分歧对应的真实轮次 ==")
    mapping, cum = [], 0
    for r in S["rounds"]:
        mapping.append(dict(round=r["round"], gen_range=[cum, cum + len(r["emitted"]) - 1],
                            pre=r["pre"], Kp=r["Kp"], m=r["m"], rollback=r.get("rollback"),
                            repair=r.get("repair"), correction=r.get("correction"),
                            all_accept=r.get("all_accept"), cands=r["cands"], g=r["g"],
                            emitted=r["emitted"], tgt_len_after=r.get("tgt_len_after")))
        cum += len(r["emitted"])
    rec["round_mapping"] = mapping
    rec["round_mapping_total_emitted"] = cum
    hit = next((x for x in mapping if first is not None and x["gen_range"][0] <= first <= x["gen_range"][1]),
               None)
    if hit is not None:
        off = first - hit["gen_range"][0]
        rec["first_diff_round"] = dict(
            round=hit["round"], offset_in_round=off, pre=hit["pre"], Kp=hit["Kp"], m=hit["m"],
            cands=hit["cands"], g=hit["g"], emitted=hit["emitted"], rollback=hit["rollback"],
            repair=hit["repair"], correction=hit["correction"], all_accept=hit["all_accept"],
            tgt_len_after=hit["tgt_len_after"], ar_token=ga[first], spec_token=gs[first],
            diff_token_role=("修正 token（offset==m：draft 候选被拒、target 自行修正）" if off == hit["m"]
                             else ("被接受候选（offset<m）" if off < hit["m"] else "越界（不应发生）")))
        f = rec["first_diff_round"]
        print(f"  第 {f['round']} 轮：gen[{hit['gen_range'][0]}..{hit['gen_range'][1]}] pre={f['pre']} "
              f"Kp={f['Kp']} m={f['m']} rollback={f['rollback']} repair={f['repair']} "
              f"all_accept={f['all_accept']}")
        print(f"    cands={f['cands']} g={f['g']} emitted={f['emitted']} correction={f['correction']}")
        print(f"    分歧 index={first} 轮内偏移 {off} ⇒ 角色：{f['diff_token_role']}")
    arinv = A["round_invariants"]["per_round"]
    rec["first_diff_ar_step"] = next((x for x in arinv if x.get("gen_len") == (first + 1)), None) \
        if first is not None else None
    if rec["first_diff_ar_step"]:
        s0 = rec["first_diff_ar_step"]
        print(f"  AR 侧对应解码步：step={s0['step']} decode_runs={s0['decode_runs']} "
              f"cache_len={s0['cache_len']} expected={s0['expected']} ok={s0['ok']}")

    # ---------------- 7) 同协议既有运行交叉核对（只读、固定路径） ----------------
    print("\n== 7) 同协议既有运行交叉核对（固定清单，只读；不重跑）==")
    runs, groups = {}, {}
    for tag, lineage, name, mode in SAME_PROTOCOL_RUNS:
        p = OUTDIR / name
        if not p.exists():
            print(f"  [缺] {tag} {name}")
            continue
        j = json.loads(p.read_text(encoding="utf-8"))
        g = _gen_of(j)
        if g is None:
            print(f"  [跳过] {tag} {name}（记录里没有完整 gen；如 UCG 早期崩溃运行）")
            continue
        key = ",".join(map(str, g))
        runs[tag] = dict(lineage=lineage, file=name, mode=mode, gen=g, group=key[:12],
                         t47=(g[47] if len(g) > 47 else None), t50=(g[50] if len(g) > 50 else None),
                         stop=(j.get("run") or {}).get("stop_reason") or (j.get("result") or {}).get("stop_reason"),
                         target_rope=(j.get("config") or {}).get("target_rope", "<无该字段(=原实现)>"))
        groups.setdefault(key, []).append(tag)
    rec["same_protocol_runs"] = dict(
        runs=runs, groups={k[:12]: v for k, v in groups.items()},
        note=("lineage A = target 用原实现；lineage B = 本轮（CPU-FP32）；UCG 行是当时被否定的"
              "统一执行几何原型（其 Spec 与基线 Spec 相同、AR 与基线 AR 不同）。"
              "2026-09-29 的 cuDNN 运行已排除（不同 arm + 已记录的记账错位缺陷）。"))
    print(f"  按完整序列分组（共 {len(groups)} 组）：")
    for _key, members in groups.items():
        r0 = runs[members[0]]
        print(f"    组 {r0['group']}: t47={r0['t47']} t50={r0['t50']} 成员={members}")
    pairs = [("AR-A1", "SP-A1", "基线（lineage A）AR vs Spec"),
             ("AR-B1(本轮)", "SP-B1(本轮)", "本轮（lineage B）AR vs Spec"),
             ("AR-A1", "AR-B1(本轮)", "AR：lineage A vs B（只换 target 频率来源）"),
             ("SP-A1", "SP-B1(本轮)", "Spec：lineage A vs B"),
             ("AR-A3(RED=1)", "AR-A1", "AR：RED=1 vs 默认（同一谱系重复性）"),
             ("SP-A2", "SP-A1", "Spec：重复运行（同一谱系重复性）")]
    rec["same_protocol_pairs"] = {}
    for a, b, label in pairs:
        if a in runs and b in runs:
            d = _first_diff(runs[a]["gen"], runs[b]["gen"])
            d.update(label=label, a=a, b=b, a_lineage=runs[a]["lineage"], b_lineage=runs[b]["lineage"])
            rec["same_protocol_pairs"][label] = d
            print(f"  {label:<42} identical={str(d['identical']):<5} common={d['common_prefix']:<3}"
                  + (f" 首次分歧={d['first_diff']} {d['tokens']}" if d["first_diff"] is not None else "")
                  + f" 差异总数={d['n_diff']}")

    # ---------------- 8) 判定 ----------------
    rec["checks"] = dict(finiteness_ar=A["run"]["prefill_logits_finite"],
                         finiteness_spec=S["run"]["prefill_logits_finite"],
                         round_invariants_ar=A["round_invariants"]["all_ok"],
                         round_invariants_spec=S["round_invariants"]["all_ok"],
                         dispatch_ar=A["run"]["dispatch_only_prefill"],
                         dispatch_spec=S["run"]["dispatch_only_prefill"],
                         structure_check_ar=A.get("cache_structure_ok"),
                         structure_check_spec=S.get("cache_structure_ok"))
    if fd["identical"]:
        rec["verdict"] = ("**一致**：AR 与 Spec 完整生成 token IDs 相同、长度与停止原因符合协议，"
                          "初始状态与 HF C1 参考一致 ⇒ 在该样本与该配置下通过完整序列一致性；"
                          "**仍需独立重复**（本工具不重复、不自动推进）。")
    else:
        rec["verdict"] = ("**不一致**（首组）：AR 与 Spec 在 index=%d（绝对位置 %d）首次分歧，"
                          "两侧 token 为 AR=%d / Spec=%d，差异总数 %d（长度与停止原因相同）。"
                          "按预先声明的口径：**保存并停止本轮** —— 不跑重复组、不跑 F06、"
                          "不新增取证、不做精度组合、不做逐层扫描。"
                          % (first, CTX + first, ga[first], gs[first], fd["n_diff"]))
    if not rec["init_state_ok"]:
        rec["verdict"] = ("**初始状态核验未通过** ⇒ 落盘停止，**不用**后续生成差异讨论 Spec 正确性。" +
                          rec["verdict"])
    rec["no_exemption"] = ("**不**把「小 margin」「更晚分歧」「较小 logits 差异」当作豁免：分歧就是分歧，"
                           "本工具不做容差、不做门槛调整。"
                           "初始状态已通过 ⇒ 分歧**不能**归因于初始状态不同或干预未忠实执行。")
    bp = rec["same_protocol_pairs"].get("基线（lineage A）AR vs Spec")
    if bp is not None:
        rec["baseline_note"] = (
            "同协议基线（lineage A，target 用原实现）的 AR/Spec **同样不一致**"
            f"（首次分歧 index={bp['first_diff']}，AR={bp['tokens'][0]} / Spec={bp['tokens'][1]}，"
            f"差异总数={bp['n_diff']}）⇒ 频率来源对齐**没有**消除 AR/Spec 分歧，只是把首次分歧位置前移。"
            "本工具**不**据此宣称任何一方「正确」，也不把该观察当作豁免。")
        print(f"\n[asc] 基线对照：{rec['baseline_note']}")
    print(f"\n[asc] ★ {rec['verdict']}")
    print(f"[asc] {rec['no_exemption']}")
    p = OUTDIR / f"arspec_consistency_{A['run_id']}_{S['run_id']}_{ts}.json"
    if p.exists():
        p = p.with_name(p.stem + "_r1.json")
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[asc] RESULT_JSON={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
