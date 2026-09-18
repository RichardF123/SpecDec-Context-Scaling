from __future__ import annotations
import torch

def cache_length(cache):
    if cache is None:
        return 0
    return int(cache[0]["k"].shape[2])


def trim_cache(cache, new_len: int):
    if cache is None:
        if new_len == 0:
            return None
        raise ValueError("trim_cache(None,n > 0) 非法参数输入.")
    cur = cache_length(cache)
    if not isinstance(new_len, int) or isinstance(new_len, bool):
        raise ValueError(f"new_len 必须是int.")
    if new_len <= 0 or new_len > cur:
        raise ValueError(f"new_len {new_len} 越界 (current {cur})")

    out = []
    for i, layer in enumerate(cache):
        nl = {}
        for key, val in layer.items():
            if isinstance(val, torch.Tensor):
                nl[key] = val[:, :, :new_len]
            else:
                nl[key] = val
        out.append(nl)
    return out

def check_consistent(cache):
    if cache is None:
        return
    lens = []
    for i, layer in enumerate(cache):
        cur = []
        for key, val in layer.items():
            if isinstance(val, torch.Tensor) and val.dim() >= 3:
                cur.append((key, int(val.shape[2])))
        vals = {n for _, n in cur}
        if len(vals) != 1:
            raise RuntimeError(f"cache token dim mismatch at layer {i}: {cur}")
        lens.append(cur[0][1])
    if len(set(lens)) != 1:
        raise RuntimeError(f"cross-layer token length mismatch: {lens}")


if __name__ == "__main__":
    # python -c 或临时片段
    import torch
    from specdec.engine import spec_cache as sc
    fake = [{"k": torch.zeros(1,2,8,4), "v": torch.zeros(1,2,8,4)},
            {"k": torch.zeros(1,2,8,4), "v": torch.zeros(1,2,8,4)}]
    assert sc.cache_length(fake) == 8 and sc.cache_length(None) == 0
    assert sc.trim_cache(None, 0) is None
    t = sc.trim_cache(fake, 3); sc.check_consistent(t)
    assert sc.cache_length(t) == 3 and fake[0]["k"].shape[2] == 8   # 原 cache 未被改
    for bad in (-1, 9):
        try: sc.trim_cache(fake, bad); raise AssertionError("should raise")
        except ValueError: pass
    badc = [{"k": torch.zeros(1,2,8,4), "v": torch.zeros(1,2,7,4)}]
    try: sc.check_consistent(badc); raise AssertionError("should raise")
    except RuntimeError: pass
    print("spec_cache self-test OK")    