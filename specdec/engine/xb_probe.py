"""xb_probe.py — **自研引擎 vs HF 的同路径对齐探针**（`XB_IMPL=hf|ours`，另有 `XB_SELFTEST=1`）。

回答的问题：**相同输入、相同 prefill 分块、相同后续调用日程下，自研 target 与 HF target 是否一致？**
（这与"各自内部单步/块是否一致"是不同问题。）

## 0. 本轮修正（**只动诊断工具**，不涉及生产模型/精度/后端/提案/严格门）
1. **克隆 bug（HF 侧）**：之前本文件手写了一个"只填 `keys`/`values`"的 HF 克隆。在本机 transformers（v5，
   `DynamicLayer.__dict__ = ['device','dtype','is_initialized','keys','values']`）里 `is_initialized` 是
   **普通布尔属性**：新实例为 `False` ⇒ `get_seq_length()` 返回 **0**，第一次 `update()` 会**重新初始化并清空**已填入的 K/V。
   实测：克隆 6-token 历史后追加 1 个 token，长度是 **1**（应为 7），前 6 位 K **没有保留**。
   ⇒ 上一次运行的两臂**不是从 8240-token 历史出发**，其"阶段 B：跨实现"对照 **INVALID**（见勘误文档）。
   现在两侧统一用 `cache_clone` 的 deepcopy 克隆，并把**克隆不变量断言**写进产物（`clone_check`）。
   **引擎侧当时并没有这个 bug**：那次运行的层 dict 只含 `k`/`v` 两个张量键，`[{k: v.clone()}]` 是正确的；
   错只在 HF 侧。（`spec_verify_index_selftest_cpu.clone_cache` 的浅拷贝也不是 bug，由它自己的哨兵项验证。）
2. **撤回一条无依据归因**："上一轮 HF 探针用的是 2096 末块" —— 读源码实测：旧探针的循环是
   `for st in range(0, seq_len, CHUNK)`，其日程与本协议的 S* **相同**：
   `[(0,2048),(2048,4096),(4096,6144),(6144,8192),(8192,8240)]`，末块 = **48**（五个 chunk）。
   ⇒ "prefill 分块日程敏感性"这一归因**不成立，必须撤回**；新旧 HF 结果的差异是**未解释的待查项**。
3. **阶段闸门**：默认只执行到 **C1** 并落盘；C2 与两臂必须由**显式授权**的运行（`XB_STAGE=c2|ab`）开启。见 §2。
4. **微型自测修正**：修掉"先删掉 HF 的 q/k/v bias，再断言两边都没有 bias"这种**自证式**做法。
   现在改为：**以实际模块为准**探测 bias，把 bias **映射**进引擎（不删除），并核查参数名集合、张量位级、实际模块属性。
   两条**实测**事实（本轮核对）：
   * `transformers 5.16.1` 的 `Qwen2Attention.__init__` **硬编码** `nn.Linear(..., bias=True)`
     （`modeling_qwen2.py:189`）⇒ 无论 config 怎么写，从 config 建的 Qwen2 都带 q/k/v bias；
     `Qwen2Config(attention_bias=False)` 存下了键但**建模代码根本不读它**（实测：模块 `q_proj.bias is not None` → `True`）。
   * 生产 checkpoint **本身就有** bias：`model.safetensors` 头里 338 个张量中 84 个是
     `layers.*.{q,k,v}_proj.bias`（28 层 × 3），且**没有** `lm_head.weight`。
   ⇒ 生产模型**带** q/k/v bias，引擎侧 `attn_bias=True` 才是对的（338 = 254 + 84）。
   旧自测把 bias 删掉，恰好把微型模型做成了**与生产结构不同**的样子 —— 这是要修掉的错误，不是"模拟真实配置"。

## 1. 事先声明的协议 S*（两边**完全相同**）
  序列 = F13 prompt(8192) + 历史前 48 个 token（共 8240）
  * **C1**：4 个 chunk 覆盖 prompt —— `[0:2048, 2048:4096, 4096:6144, 6144:8192]`
  * **C2**：再喂 48 个 token 的**一个 chunk** `[8192:8240]`
  * 起点 `pre = 8240`；随后两臂（**仅 `XB_STAGE=ab`**）：
    - **Serial**：依次喂 `[220,16,16,1154]`（每次 1 个 token，共 4 次）
    - **Block**：一次喂 `[220,16,16,1154]`
  > ⚠️ 这是 **teacher-forced 数值对照**，**不是**自由生成，也**不是**完整 SpecDec 验收；
  > 也**不是**原始自由生成 AR/Spec 的历史。
  > ⚠️ 上一轮 HF 探针（`hfsb_*`）的日程与本协议**相同**（见 §0.2），但它**没有**自研侧的同路径对照，
  > 因此**不**用于跨实现比较。

## 2. 阶段闸门（`XB_STAGE`，默认 `c1`）
  * `c1`（默认）：只跑 C1 → 落盘 → 停。`stage = "c1_only_stopped_by_gate"`
  * `c2`：C1 + C2 → 落盘 → 停。`stage = "c2_only_stopped_by_gate"`
  * `ab`：C1 + C2 + 两臂。`stage = "complete"`
  闸门**不是**自动向前推进的：本工具是**单侧**运行，同一次运行内不可能知道另一侧 C1 是否逐位相同。
  因此"是否继续"只能是**人类显式授权的新运行**，并记录 `authorized_by`。
  落盘产物里记 `stage_gate.executed`，比较工具据此**拒绝**把"未执行"当成"相同"。
  ★ 措辞纪律：C1 只能说成 **"第一个已检查且发现差异的检查点"**，不能说"差异首次产生于 C1"
  （C1 之前没有已检查的检查点被比较过）。

## 3. 已知实现差异（**运行前**列出，不凭名称假设一致）
  1. **score 缩放**：HF `matmul(q,k^T) * scaling`（乘倒数）vs 自研 `matmul(...) / sqrt(head_dim)`（除）——
     数学等价、**bf16 下可能差 1 ULP**；
  2. **mask 施加**：HF 用 **bf16 加法掩码**（允许=0、禁止≈`finfo.min`）；自研用 `masked_fill(-inf)`；
     两边都构造**因果**掩码，且掩码位置语义不同（HF 在 `(1,1,q,kv)` 上、自研在 `(s,T)` 上按 `T-s` 定对角线）⇒需实测；
  3. **GQA**：HF `repeat_kv`（expand+reshape）vs 自研 `kv_repeat`（同模式），需实测而非假定；
  4. **RoPE**：HF `Qwen2RotaryEmbedding`（`position_embeddings`）vs 自研 `Build_Rope_cache` + `apply_rope`，
     cos/sin 的 dtype 与生成方式**需实测**；
  5. **RMSNorm**：两边**都是 fp32 内部**（已核对 HF 源码与自研实现）⇒ **不是**差异点；
  6. **LM head**：HF `lm_head(hidden)`（tied）vs 自研 `x @ embed.weight.T` —— 形状相同；
  7. **权重映射**：HF `state_dict` 去掉 `model.` 前缀、tied 时去掉 `lm_head.weight`、
     `load_state_dict(strict=True)` + 逐张量 `torch.equal` ⇒ **同名映射**（记录分组计数与**丢弃键清单**，
     不把 `missing=[]` 当唯一证据）。
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "g5" / "attn"
TAG = "xb_20261001"
TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
FAMILY = os.environ.get("XB_FAMILY", "F13").strip().upper()
CTX = int(os.environ.get("XB_CTX", "8192"))
CHUNK = int(os.environ.get("XB_CHUNK", "2048"))
PRE = int(os.environ.get("XB_PRE", "8240"))
TOKENS = [int(x) for x in os.environ.get("XB_TOKENS", "220,16,16,1154").split(",")]
IMPL = os.environ.get("XB_IMPL", "ours").strip().lower()
SELFTEST = os.environ.get("XB_SELFTEST", "1") == "1"
# ★ 阶段闸门：默认只到 C1（见模块 docstring §2）
STAGE = os.environ.get("XB_STAGE", "c1").strip().lower()
_STAGE_EXEC = {"c1": ["C1"], "c2": ["C1", "C2"], "ab": ["C1", "C2", "Serial", "Block"]}
_STOP_AFTER = {"c1": "C1", "c2": "C2", "ab": None}


def _newest(pat):
    return Path(sorted(glob.glob(str(OUTDIR / pat)))[-1])


def _sha(t: torch.Tensor, n: int = 16) -> str:
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:n]


def _row_fields(r) -> dict:
    x = r.detach().float().reshape(-1)
    t2 = torch.topk(x, 2).values
    am = int(torch.argmax(x))
    tied = [int(i) for i in torch.nonzero(x == x.amax()).reshape(-1)]
    return dict(argmax=am, max_score=float(x.max()), margin=float(t2[0] - t2[1]),
                n_tied_max=len(tied), finite=bool(torch.isfinite(x).all()), sha=_sha(r))


def _stage_gate_block() -> dict:
    return dict(
        requested=STAGE, default="c1", executed=_STAGE_EXEC[STAGE], stopped_after=_STOP_AFTER[STAGE],
        authorized_by=f"XB_STAGE={STAGE}",
        reason=("默认只执行到 C1 并落盘：在 C1 的跨实现结论被人工审阅并**显式授权**之前，"
                "不自动推进 C2 与两臂 —— 上一次正因为把下游阶段一起跑完，才产生了需要撤回的归因。"),
        advance_policy=("闸门**不会**自动向前推进：本工具是单侧运行，同一次运行内无法知道另一侧 C1 是否逐位相同。"
                        "推进只能是新的、显式授权的运行（XB_STAGE=c2 或 XB_STAGE=ab）。"),
        wording_rule="C1 只能说成「第一个已检查且发现差异的检查点」，不能说「差异首次产生于 C1」。",
    )


# ------------------------------------------------------------------ 自测
def _selftest() -> int:
    """微型模型自测，分三组：

    (a) **权重映射**：seed 在建模**之前**；HF 与引擎**结构相同**（bias 一并映射，**不删除**）；
    (b) **数值对齐**：记录 max|Δ| 与 argmax —— 只作信息，**不设 PASS 阈值**（不发明判据）；
    (c) **缓存结构与克隆**：布局 + HF/引擎两侧的克隆不变量 + 旧 bug 的回归。
    """
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from .align_ours import build_from_config
    from .cache_clone import (CloneError, clone_cache_full, clone_cache_old_xb,
                              clone_list_cache_full, require_clone_invariants,
                              require_list_clone_invariants)

    npass = nfail = 0

    def check(name, cond, extra=None):
        nonlocal npass, nfail
        if cond:
            npass += 1
            print(f"  [ok] {name}")
        else:
            nfail += 1
            print(f"  [FAIL] {name}" + (f"  {str(extra)[:200]}" if extra is not None else ""))

    def raises(fn, exc=CloneError):
        try:
            fn()
            return False
        except exc:
            return True

    SEED = 20261001
    torch.manual_seed(SEED)                    # ★ 种子在**两个模型初始化之前**
    cfg = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, max_position_embeddings=512, tie_word_embeddings=True)
    cfg._attn_implementation = "eager"
    hf = Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval()
    sa0 = hf.model.layers[0].self_attn
    bias_flags = [getattr(sa0, p).bias is not None for p in ("q_proj", "k_proj", "v_proj")]
    hf_bias = all(bias_flags)
    print(f"  （自测信息）seed={SEED} 微型 HF 实际 q/k/v bias={bias_flags}；"
          f"cfg.attention_bias={cfg.to_dict().get('attention_bias', '<未设置>')!r}")
    print("             实测：v5.16.1 `modeling_qwen2.py:189` **硬编码** bias=True、建模代码不读 config 的该键；")
    print("             生产 checkpoint 头里 84/338 个张量就是 q/k/v bias（无 lm_head.weight）")
    print("             ⇒ 一律以**实际模块**为准，并把 bias **映射**进来，而不是删掉。")
    if any(bias_flags) and not hf_bias:
        print("             ⚠️ q/k/v bias 三者不全一致：本自测只处理全有/全无两种情形。")
    ours = build_from_config(cfg, attn_bias_force=hf_bias).to(torch.bfloat16).eval()

    print("== (a) 权重映射：结构相同 + bias **映射**（不删除）==")
    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    dropped = []
    if cfg.to_dict().get("tie_word_embeddings", False) and "lm_head.weight" in sd:
        dropped.append("lm_head.weight")       # tied：引擎的 logits 直接读嵌入权重，故该键有意不用
        sd.pop("lm_head.weight")
    n_bias_keys = sum(1 for k in sd if k.endswith(("q_proj.bias", "k_proj.bias", "v_proj.bias")))
    ours.load_state_dict(sd, strict=True)
    cur = ours.state_dict()
    check("★ 同名映射 + strict=True 载入成功（无缺键/多键）", True)
    check("★ 引擎参数名集合 == 映射后的 HF 键集合", set(cur) == set(sd),
          sorted(set(cur) ^ set(sd))[:6])
    same = [k for k in sd if k in cur and torch.equal(cur[k], sd[k])]
    check(f"★ 逐张量位级一致 {len(same)} / {len(sd)}", len(same) == len(sd),
          [k for k in sd if k not in same][:4])
    check("★ 丢弃键清单 ⊆ {lm_head.weight}（没有偷偷丢 bias 等键）", set(dropped) <= {"lm_head.weight"},
          dropped)
    check("★ bias 处理与实际模块一致（HF 有就映射、HF 没有就都没有；**不删除**）",
          (n_bias_keys > 0) == hf_bias
          and (getattr(ours.layers[0].self_attn.q_proj, "bias", None) is not None) == hf_bias
          and (getattr(ours.layers[0].self_attn.k_proj, "bias", None) is not None) == hf_bias
          and (getattr(ours.layers[0].self_attn.v_proj, "bias", None) is not None) == hf_bias)
    groups = {}
    for k in sd:
        top = k.split(".")[0] if not k.startswith("layers") else "layers." + k.split(".")[2]
        groups[top] = groups.get(top, 0) + 1
    print(f"  映射分组计数：{json.dumps(groups, ensure_ascii=False)}"
          f"；bias 键 {n_bias_keys} 个；丢弃键 {dropped}")
    print("  ⚠️ 本自测只覆盖**微型模型的结构与映射**；生产权重的映射由 `lowmem_load.load_ours_bf16`"
          " 的 338/338 逐张量位级核对单独证明，两者范围不同，不能互相替代。")

    print("== (b) 数值对齐（只记录，**不设 PASS 阈值**）==")
    ids = torch.randint(0, 128, (1, 6))
    with torch.inference_mode():
        ho = hf(ids, use_cache=True)
        ol, oc = ours._run(ours.embed_tokens(ids), None, need_logits=True)   # 顺序：(logits, cache)
    d = (ho.logits[0, -1].float() - ol[0, -1].float()).abs()
    check("logits 形状/dtype 相同且差值有限",
          tuple(ho.logits.shape) == tuple(ol.shape) and bool(torch.isfinite(d).all()),
          f"{tuple(ho.logits.shape)} vs {tuple(ol.shape)}")
    am_hf, am_ours = int(ho.logits[0, -1].argmax()), int(ol[0, -1].argmax())
    print(f"  （信息，**不是**判据）末位 logits max|Δ|={float(d.max()):.6g} "
          f"mean|Δ|={float(d.mean()):.6g}；argmax HF={am_hf} OURS={am_ours} 相同={am_hf == am_ours}")

    print("== (c) 缓存结构与克隆 ==")
    hcache = ho.past_key_values
    hk, ok_k = hcache.layers[0].keys, oc[0]["k"]
    check("★ 缓存布局一致 (1, kv_heads, T, head_dim)",
          tuple(hk.shape) == tuple(ok_k.shape), f"{tuple(hk.shape)} vs {tuple(ok_k.shape)}")

    def happ(c, n):
        with torch.inference_mode():
            hf(torch.tensor([[am_hf]]), past_key_values=c, use_cache=True,
               position_ids=torch.tensor([[c.get_seq_length()]]),
               attention_mask=torch.ones(1, c.get_seq_length() + 1, dtype=torch.long))

    hcl_probe = clone_cache_full(hcache)
    hinfo = require_clone_invariants(hcache, hcl_probe, L=6, append_fn=happ, n=1)
    check("★ HF 侧（**本次使用的**克隆）8 项不变量通过",
          hinfo["appended"]["len_after"] == 7 and hinfo["appended"]["history_preserved"]
          and hinfo["appended"]["source_untouched"] and hinfo["appended"]["sibling_untouched"])
    hc1, hc2 = clone_cache_full(hcache), clone_cache_full(hcache)
    check("★ HF 侧两个臂副本之间不共享存储，且与源逐位相同",
          not any(getattr(hc1.layers[i], kk).data_ptr() == getattr(hc2.layers[i], kk).data_ptr()
                  for i in range(len(hcache.layers)) for kk in ("keys", "values"))
          and all(torch.equal(hc1.layers[i].keys, hcache.layers[i].keys)
                  for i in range(len(hcache.layers))))
    old = clone_cache_old_xb(hcache)
    check("★ 回归：旧 xb 克隆（只填 keys/values）被自测抓出",
          bool(old.layers[0].is_initialized) is False and old.get_seq_length() == 0
          and raises(lambda: require_clone_invariants(hcache, old, L=6)),
          f"is_initialized={bool(old.layers[0].is_initialized)} len={old.get_seq_length()}")

    def eappend(c, n):
        with torch.inference_mode():
            _, nc = ours._run(ours.embed_tokens(torch.full((1, n), 3)), c, need_logits=False)
        c[:] = nc

    ecl = clone_list_cache_full(oc)
    einfo = require_list_clone_invariants(oc, ecl, L=6, append_fn=eappend, n=1)
    check("★ 引擎侧（list[dict]）克隆不变量通过",
          einfo["appended"]["len_after"] == 7 and einfo["appended"]["history_preserved"]
          and einfo["appended"]["source_untouched"] and einfo["appended"]["sibling_untouched"])

    print(f"\n跨实现自测：PASS {npass} / FAIL {nfail}")
    return 0 if nfail == 0 else 1


# ------------------------------------------------------------------ 协议
def _schedule(seq_len: int) -> list:
    """声明的日程 S*：prompt 用 chunk 切分，剩下的（48 个 token）单独一个 chunk。"""
    sched = [(st, min(st + CHUNK, CTX)) for st in range(0, CTX, CHUNK)]
    if seq_len > CTX:
        sched.append((CTX, seq_len))
    return sched


def _save(rec: dict, out: dict, run_id: str) -> Path:
    tp = OUTDIR / f"{TAG}_{IMPL}_{run_id}.pt"
    torch.save(out, str(tp))
    rec["artifacts"] = dict(pt=str(tp), json=str(OUTDIR / f"{TAG}_{IMPL}_{run_id}.json"))
    p = OUTDIR / f"{TAG}_{IMPL}_{run_id}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[xb] RESULT_JSON={p}")
    return p


def _run_impl() -> int:
    import transformers
    if STAGE not in _STAGE_EXEC:
        print(f"[xb] !! 非法 XB_STAGE={STAGE!r}（可选 {sorted(_STAGE_EXEC)}）")
        return 4
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = dict(run_id=run_id, tool="xb_probe", impl=IMPL, family=FAMILY, ctx=CTX, chunk=CHUNK,
               pre=PRE, tokens=TOKENS, model=TARGET,
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               schedule=_schedule(PRE), stage_gate=_stage_gate_block(),
               versions=dict(torch=torch.__version__, transformers=transformers.__version__,
                             cuda=torch.version.cuda),
               semantics="同路径 teacher-forced 数值对照（不是自由生成/完整 SpecDec 验收）；"
                         "两边**各自独立**构建缓存，不互相导入。")
    print(f"[xb] run_id={run_id} impl={IMPL} stage={STAGE} executed={_STAGE_EXEC[STAGE]}", flush=True)
    print(f"RUN_ID={run_id}", flush=True)
    from . import g5_8k_spec_variance_probe as P5V
    from . import family_input as FI
    from . import model_self as M
    from . import spec_cache, spec_dec
    from . import lowmem_load
    from .cache_clone import (clone_cache_full, clone_list_cache_full, require_clone_invariants,
                              require_list_clone_invariants)
    ok, why = P5V.commit_guard()
    print(f"[xb] commit: {why}")
    if not ok:
        (OUTDIR / f"{TAG}_{IMPL}_{run_id}.json").write_text(json.dumps(rec, ensure_ascii=False),
                                                            encoding="utf-8")
        return 3
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True)
        ids_list, fam_meta = FI.load_family_ids(tok, FAMILY, ctx=CTX)
        cap = torch.load(_newest("f13i50_20261001_capture_ar_*.pt"), map_location="cpu")
        prefix48 = [int(x) for x in cap["prefix_tokens"]][:PRE - CTX]
        rec["input"] = dict(source_kind="manifest", family=fam_meta, ctx=CTX,
                            prefix_src=str(_newest("f13i50_20261001_capture_ar_*.pt")),
                            prefix_len=len(prefix48), tokens=TOKENS)
        seq = list(ids_list) + prefix48
        assert len(seq) == PRE
        rec["input"]["seq_sha16"] = hashlib.sha256(
            json.dumps(seq).encode("utf-8")).hexdigest()[:16]
        # ---- 唯一变量：reduction=True（显式），QK32/PROB32 关闭 ----
        b = torch.backends.cuda.matmul
        b.allow_bf16_reduced_precision_reduction = True
        rec["precision"] = dict(bf16_reduced_precision_reduction=bool(
            b.allow_bf16_reduced_precision_reduction), qk_fp32=False, prob_fp32=False,
            note="两边都显式 reduction=True；QK32/PROB32 关闭")
        out = dict(meta=dict(impl=IMPL, run_id=run_id, schedule=_schedule(PRE), stage_gate=STAGE))
        if IMPL == "ours":
            tgt, _tok2, info = lowmem_load.load_ours_bf16(TARGET)
            tgt.set_kv_format("fp16")            # base_bf16（KV 未量化）
            M.set_diag_preset("eager")
            M.set_attention_strategy("eager")    # 全 eager，与 HF 对齐
            rec["weights"] = dict(bitmatch=info["weights_bitmatch"],
                                  n_checked=info["n_params_checked"],
                                  mismatch=info["mismatch_keys"],
                                  scope="生产权重：338/338 逐张量位级（与 §自测的微型映射范围不同）",
                                  mapping="HF state_dict 去 `model.` 前缀 + tied 去 lm_head + strict 载入")
            rec["impl"] = dict(attn_implementation="eager(engine)", kv_format="fp16=base_bf16",
                               strategy=M.attention_strategy())
            ids = torch.tensor([seq], device="cuda")
            # ---------- C1：prompt 4 个 chunk ----------
            cache = None
            with torch.inference_mode(), M.attention_phase("prefill"):
                for st, en in _schedule(PRE):
                    if en > CTX:
                        break
                    piece = ids[:, st:en]
                    lg, cache = tgt._run(tgt.embed_tokens(piece), cache,
                                         need_logits=(en == CTX))
            rec["C1"] = dict(cache_len=int(spec_cache.cache_length(cache)), last_logits=_row_fields(lg[0, -1]))
            out["c1_logits"] = lg[0, -1].detach().float().cpu()
            out["c1_kv"] = [{k: v.detach().float().cpu() for k, v in d.items()} for d in cache]
            if STAGE == "c1":
                rec["stage"] = "c1_only_stopped_by_gate"
                rec["C1"]["note"] = ("C1 是**第一个已检查**的跨实现检查点；本次运行按闸门在此停止。"
                                     "C1 快照已落盘，可直接用于 C1 定位（无需重跑 8K）。")
                rec["resources"] = dict(pt_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                                        cuda_free_mib=torch.cuda.mem_get_info()[0] / 2 ** 20,
                                        note="顺序加载；显存只作诊断")
                rec["kv_digest"] = dict(c1_sha=[_sha(d["k"]) for d in out["c1_kv"]])
                _save(rec, out, run_id)
                print(f"[xb] C1 len={rec['C1']['cache_len']} "
                      f"argmax={rec['C1']['last_logits']['argmax']} | 闸门停止（未跑 C2/两臂）")
                return 0
            # ---------- C2：48 个 token 一个 chunk ----------
            with torch.inference_mode(), M.attention_phase("prefill"):
                lg2, cache = tgt._run(tgt.embed_tokens(ids[:, CTX:PRE]), cache, need_logits=True)
            rec["C2"] = dict(cache_len=int(spec_cache.cache_length(cache)), last_logits=_row_fields(lg2[0, -1]))
            out["c2_logits"] = lg2[0, -1].detach().float().cpu()
            out["c2_kv"] = [{k: v.detach().float().cpu() for k, v in d.items()} for d in cache]
            if STAGE == "c2":
                rec["stage"] = "c2_only_stopped_by_gate"
                rec["resources"] = dict(pt_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                                        cuda_free_mib=torch.cuda.mem_get_info()[0] / 2 ** 20,
                                        note="顺序加载；显存只作诊断")
                rec["kv_digest"] = dict(c1_sha=[_sha(d["k"]) for d in out["c1_kv"]],
                                        c2_sha=[_sha(d["k"]) for d in out["c2_kv"]])
                _save(rec, out, run_id)
                print(f"[xb] C1 len={rec['C1']['cache_len']} | C2 len={rec['C2']['cache_len']} "
                      f"argmax={rec['C2']['last_logits']['argmax']} | 闸门停止（未跑两臂）")
                return 0
            # ---------- 两臂（仅 XB_STAGE=ab）----------
            cs, cb = clone_list_cache_full(cache), clone_list_cache_full(cache)
            rec["arm_clone_check"] = dict(
                impl="engine list[dict]", method="cache_clone.clone_list_cache_full (deepcopy)",
                serial_arm=require_list_clone_invariants(cache, cs, L=PRE),
                block_arm=require_list_clone_invariants(cache, cb, L=PRE),
                old_impl_note="旧写法 [{'k': v.clone()}] 在只有张量键的实际缓存上同样正确；"
                              "HF 侧的旧 clone 才是 bug（见模块 docstring §0.1）")
            ser = []
            for j, t in enumerate(TOKENS):
                with torch.inference_mode(), M.attention_phase("decode"):
                    l, cs = spec_dec._step(tgt, t, cs)
                ser.append(l[0, -1].detach().float().cpu())
            with torch.inference_mode(), M.attention_phase("verify"):
                rows, cb = spec_dec._verify(tgt, cb, list(TOKENS))
            out["serial_rows"] = ser
            out["block_rows"] = [rows[j].detach().float().cpu() for j in range(len(TOKENS))]
            rec["serial"] = [_row_fields(x) for x in ser]
            rec["block"] = [_row_fields(x) for x in out["block_rows"]]
            rec["attention_stats"] = M.attention_stats()
        else:
            from transformers import AutoModelForCausalLM, DynamicCache
            from .hf_serial_block_probe import install_mask_recorder, _MASK_LOG, mask_allowed_from_actual
            install_mask_recorder()
            model = AutoModelForCausalLM.from_pretrained(TARGET, torch_dtype=torch.bfloat16,
                                                         attn_implementation="eager").to("cuda").eval()
            rec["impl"] = dict(attn_implementation=str(model.config._attn_implementation),
                               kv_format="DynamicCache bf16")
            assert str(model.config._attn_implementation) == "eager"
            ids = torch.tensor([seq], device="cuda")
            cache = DynamicCache()
            with torch.inference_mode():
                for st, en in _schedule(PRE):
                    if en > CTX:
                        break
                    piece = ids[:, st:en]
                    n = int(piece.shape[1])
                    o = model(piece, past_key_values=cache, use_cache=True,
                              position_ids=torch.arange(st, st + n, device="cuda").view(1, -1),
                              attention_mask=torch.ones(1, st + n, dtype=torch.long, device="cuda"))
            rec["C1"] = dict(cache_len=int(cache.get_seq_length()),
                             last_logits=_row_fields(o.logits[0, -1]))
            out["c1_logits"] = o.logits[0, -1].detach().float().cpu()
            out["c1_kv"] = [dict(k=cache.layers[i].keys.detach().float().cpu(),
                                 v=cache.layers[i].values.detach().float().cpu())
                            for i in range(len(cache.layers))]
            if STAGE == "c1":
                rec["stage"] = "c1_only_stopped_by_gate"
                rec["C1"]["note"] = ("C1 是**第一个已检查**的跨实现检查点；本次运行按闸门在此停止。"
                                     "C1 快照已落盘，可直接用于 C1 定位（无需重跑 8K）。")
                rec["resources"] = dict(pt_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                                        cuda_free_mib=torch.cuda.mem_get_info()[0] / 2 ** 20,
                                        note="顺序加载；显存只作诊断")
                rec["kv_digest"] = dict(c1_sha=[_sha(d["k"]) for d in out["c1_kv"]])
                _save(rec, out, run_id)
                print(f"[xb] C1 len={rec['C1']['cache_len']} "
                      f"argmax={rec['C1']['last_logits']['argmax']} | 闸门停止（未跑 C2/两臂）")
                return 0
            with torch.inference_mode():
                o2 = model(ids[:, CTX:PRE], past_key_values=cache, use_cache=True,
                           position_ids=torch.arange(CTX, PRE, device="cuda").view(1, -1),
                           attention_mask=torch.ones(1, PRE, dtype=torch.long, device="cuda"))
            rec["C2"] = dict(cache_len=int(cache.get_seq_length()),
                             last_logits=_row_fields(o2.logits[0, -1]))
            out["c2_logits"] = o2.logits[0, -1].detach().float().cpu()
            out["c2_kv"] = [dict(k=cache.layers[i].keys.detach().float().cpu(),
                                 v=cache.layers[i].values.detach().float().cpu())
                            for i in range(len(cache.layers))]
            if STAGE == "c2":
                rec["stage"] = "c2_only_stopped_by_gate"
                rec["resources"] = dict(pt_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                                        cuda_free_mib=torch.cuda.mem_get_info()[0] / 2 ** 20,
                                        note="顺序加载；显存只作诊断")
                rec["kv_digest"] = dict(c1_sha=[_sha(d["k"]) for d in out["c1_kv"]],
                                        c2_sha=[_sha(d["k"]) for d in out["c2_kv"]])
                _save(rec, out, run_id)
                print(f"[xb] C1 len={rec['C1']['cache_len']} | C2 len={rec['C2']['cache_len']} "
                      f"argmax={rec['C2']['last_logits']['argmax']} | 闸门停止（未跑两臂）")
                return 0
            # ---------- 两臂（仅 XB_STAGE=ab）：★ 用 cache_clone 的 deepcopy ──────────
            def happ(c, n):
                with torch.inference_mode():
                    model(torch.tensor([[TOKENS[0]]], device="cuda"), past_key_values=c, use_cache=True,
                          position_ids=torch.tensor([[c.get_seq_length()]], device="cuda"),
                          attention_mask=torch.ones(1, c.get_seq_length() + 1, dtype=torch.long,
                                                    device="cuda"))

            probe = clone_cache_full(cache)                     # 一次性探针（会被追加，不影响两臂）
            hinfo = require_clone_invariants(cache, probe, L=PRE, append_fn=happ, n=1)
            cs, cb = clone_cache_full(cache), clone_cache_full(cache)
            rec["arm_clone_check"] = dict(
                impl="HF DynamicCache", method="cache_clone.clone_cache_full (deepcopy)",
                invariants=require_clone_invariants(cache, cs, L=PRE),
                append_probe=dict(len_after=hinfo["appended"]["len_after"],
                                  history_preserved=hinfo["appended"]["history_preserved"],
                                  source_untouched=hinfo["appended"]["source_untouched"],
                                  sibling_untouched=hinfo["appended"]["sibling_untouched"]),
                old_impl_note="旧的手写 clone（只填 keys/values）会 is_initialized=False、get_seq_length=0，"
                              "第一次 update() 重新初始化 ⇒ 上一次运行的两臂并非从 8240 历史出发（已判 INVALID）")
            assert not any(cs.layers[i].keys.data_ptr() == cb.layers[i].keys.data_ptr()
                           for i in range(len(cs.layers))), "两臂共享存储"
            ser = []
            for j, t in enumerate(TOKENS):
                with torch.inference_mode():
                    oo = model(torch.tensor([[t]], device="cuda"), past_key_values=cs, use_cache=True,
                               position_ids=torch.tensor([[PRE + j]], device="cuda"),
                               attention_mask=torch.ones(1, PRE + j + 1, dtype=torch.long,
                                                        device="cuda"))
                ser.append(oo.logits[0, -1].detach().float().cpu())
            _MASK_LOG.clear()
            with torch.inference_mode():
                ob = model(torch.tensor([TOKENS], device="cuda"), past_key_values=cb, use_cache=True,
                           position_ids=torch.arange(PRE, PRE + len(TOKENS), device="cuda").view(1, -1),
                           attention_mask=torch.ones(1, PRE + len(TOKENS), dtype=torch.long,
                                                     device="cuda"))
            rec["mask_actual_serial"] = [mask_allowed_from_actual(dict(mask_obj=None)),
                                         "见 hf_serial_block_probe（本工具只记录 block 的实际 mask）"]
            ma = mask_allowed_from_actual(_MASK_LOG[0]) if _MASK_LOG else dict(kind="no_call")
            rec["mask_actual_block"] = ma
            out["serial_rows"] = ser
            out["block_rows"] = [ob.logits[0, j].detach().float().cpu() for j in range(len(TOKENS))]
            rec["serial"] = [_row_fields(x) for x in ser]
            rec["block"] = [_row_fields(x) for x in out["block_rows"]]
        rec["resources"] = dict(pt_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                                pt_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20,
                                cuda_free_mib=torch.cuda.mem_get_info()[0] / 2 ** 20,
                                note="顺序加载、不同时驻留；计时/显存只作诊断")
        rec["kv_digest"] = dict(
            c1_sha=[_sha(d["k"]) for d in out["c1_kv"]], c2_sha=[_sha(d["k"]) for d in out["c2_kv"]])
        rec["stage"] = "complete"
        _save(rec, out, run_id)
        print(f"[xb] C1 len={rec['C1']['cache_len']} argmax={rec['C1']['last_logits']['argmax']} | "
              f"C2 len={rec['C2']['cache_len']} argmax={rec['C2']['last_logits']['argmax']}")
        print(f"[xb] serial argmax={[x['argmax'] for x in rec['serial']]} "
              f"block argmax={[x['argmax'] for x in rec['block']]}")
        return 0
    except Exception as e:                                        # noqa: BLE001
        rec["stage"] = "crashed"
        rec["failure"] = str(e)
        rec["traceback"] = traceback.format_exc()
        p = OUTDIR / f"{TAG}_{IMPL}_{run_id}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[xb] !! 停止：{e}")
        print(f"[xb] RESULT_JSON={p}")
        return 2


if __name__ == "__main__":
    raise SystemExit(_selftest() if SELFTEST else _run_impl())
