"""lowmem_load.py — 与 `g4_b4_cache_tests.load_ours` **权重逐位相同**、但 CPU 峰值约减半的加载器。

为什么需要它
  `load_ours` 的 CPU 峰值由两部分构成（见 8K 探针的护栏推导）：
    HF 模型 bf16  : 1.54e9 × 2 B = 2.87 GiB
    **自建模型 fp32**: 1.54e9 × 4 B = **5.74 GiB**   ← `build_from_config` 用 nn.Linear 默认 fp32
    合计 ≈ 8.6 GiB，加余量 ≈ **9.1–9.6 GiB**；实测崩点就是 commit 可用 9.5 GiB（原生 0xC0000005）。
  这台机器上 commit 可用量长期在 13–16 GiB 之间波动（大部分是其它 23 个 GPU 应用的
  WDDM/分页文件提交量，无法由本进程释放）。**降低真实需求**比下调安全线更安全。

做法（**只改"什么时候转 bf16"，不改最终权重**）
  在 `load_state_dict` **之前**就把自建模型 `.to(bf16)`：fp32 5.74 GiB ⇒ bf16 2.87 GiB，
  峰值从 ~8.6 GiB 降到 ~5.7 GiB。
  为什么最终权重相同：`load_ours` 的路径是
    build(fp32) → `load_state_dict(hf_bf16)`（bf16→fp32 **无损**上转）→ `.to(cuda, bf16)`（fp32→bf16 舍入）
  本模块的路径是
    build(fp32) → `.to(bf16)`（未初始化的参数，随便舍入）→ `load_state_dict(hf_bf16)`（bf16→bf16 **精确**）
  两条路径的**终点都是"参数 = HF 的 bf16 state dict 值"**。

**不靠推理，靠校验**：载入后逐张量与 HF 的 state dict 做 `torch.equal`（bf16 逐位），
任何一张不相等都会记进返回的 `info["mismatch_keys"]`。调用方必须把 `info` 写进结果 JSON。
"""
from __future__ import annotations

import gc

import torch

from .align_ours import build_from_config
from .g4_b4_cache_tests import DTYPE, _from_pretrained


def load_ours_bf16(model_name: str, *, verify: bool = True):
    """返回 `(model_cuda_bf16, tokenizer, info)`；`info` 必须落盘。

    `info` 字段
      load_path            = "lowmem_bf16_build"
      weights_bitmatch     = 是否**全部**参数与 HF state dict 逐位相同（本次运行的等价性证据）
      mismatch_keys        = 不相等的参数名（应为空）
      n_params_checked     = 校验过的张量数
      cpu_peak_note        = 峰值说明（供报告引用）
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print(f"[lowmem] {model_name}: HF 权重在 CPU 以 bf16 载入（不上 GPU）", flush=True)
    cfg = _from_pretrained(AutoConfig, model_name, trust_remote_code=True)
    hf = _from_pretrained(AutoModelForCausalLM, model_name, config=cfg,
                          attn_implementation="eager", torch_dtype=DTYPE,
                          trust_remote_code=True)
    ours = build_from_config(cfg)
    # ---- 关键一行：在载入之前就降精度，把 5.74 GiB 变成 2.87 GiB ----
    ours = ours.to(DTYPE)
    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    if cfg.to_dict().get("tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    ours.load_state_dict(sd, strict=True)

    mismatch, checked = [], 0
    if verify:
        cur = ours.state_dict()
        with torch.no_grad():
            for k, v in sd.items():
                if k in cur:
                    checked += 1
                    if not torch.equal(cur[k], v):
                        mismatch.append(k)
        print(f"[lowmem] 权重校验：逐张量 torch.equal 检查 {checked} 个，"
              f"不一致 {len(mismatch)} 个{('：' + str(mismatch[:5])) if mismatch else ''}",
              flush=True)

    del hf, sd
    gc.collect()
    torch.cuda.empty_cache()
    print("[lowmem] 搬到 cuda ...", flush=True)
    ours = ours.to("cuda").eval()
    tok = _from_pretrained(AutoTokenizer, model_name, trust_remote_code=True)
    info = dict(load_path="lowmem_bf16_build", weights_bitmatch=(not mismatch),
                mismatch_keys=mismatch[:20], n_params_checked=checked,
                cpu_peak_note="先 .to(bf16) 再 load_state_dict：CPU 峰值 ~5.7 GiB（对比 load_ours ~8.6 GiB）")
    print(f"[lowmem] 完成：weights_bitmatch={info['weights_bitmatch']} "
          f"(checked={checked})", flush=True)
    return ours, tok, info
