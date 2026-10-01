# Custom kernels

Four operators replace the non-GEMM math in the eager engine: RMSNorm, RoPE (`rotate_half`), SwiGLU (`silu(gate) * up`), and causal grouped-query attention with an online softmax. GEMMs, the KV-cache layout, FP8 scales, and the cuDNN prefill branch stay on the existing engine path.

The CUDA C++ extension is the implementation an engine pack loads, and the one a Windows machine can compile when `cl.exe` and `nvcc` match the installed PyTorch CUDA wheel. Triton implements the same formulas and is the cross-check on Linux. Official Triton has no Windows build, so a machine without it still loads the CUDA kernels.

```
kernels/                  # specdec/kernels/ inside the SpecDec repo
  ops.py                  # dispatch: backend = "torch" | "cuda" | "triton"
  reference.py            # Torch formulas the checks compare against
  patch.py                # install / restore on the live engine classes
  build.py                # JIT-compile the CUDA extension on first use
  sample_experiment.py    # one operator case, no model weights
  cuda/                   # rmsnorm, rope, swiglu, attention
  triton/                 # the same four
```

## Install

From the SpecDec repository root, with a CUDA build of PyTorch already installed:

```powershell
pip install -e .
pip install -r requirements-kernels.txt
```

`requirements-kernels.txt` is Triton, and it is Linux-only. Skip it on Windows. The CUDA extension compiles on the first `backend="cuda"` call, for the GPU that is visible at that moment. That JIT build needs `ninja` on `PATH` (`pip install ninja`). The build directory is `TORCH_EXTENSIONS_DIR` when that variable is set, otherwise `.cache/torch_extensions` under the repository root.

In the experiment checkout the same modules are importable as `kernels` when the experiment root is on `PYTHONPATH`.

## Call an operator

```python
import torch
from specdec.kernels import ops

x = torch.randn(2, 32, 1536, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(1536, device="cuda", dtype=torch.bfloat16)
y = ops.rmsnorm(x, weight, eps=1e-6, backend="cuda")
```

| function | tensors | notes |
| --- | --- | --- |
| `ops.rmsnorm(x, weight, eps=1e-6, backend="cuda")` | `x` is `(..., hidden)`, `weight` is `(hidden,)` | variance in fp32, store in `x.dtype` |
| `ops.apply_rope(x, cos, sin, backend="cuda")` | `x` is `(..., seq, dim)` or `(B, H, S, D)` | `cos` / `sin` broadcast to `(seq, dim)`; half-rotation matches `model_self.apply_rope` |
| `ops.swiglu(gate, up, backend="cuda")` | `gate` and `up` share a shape | `silu(gate) * up` |
| `ops.attention(q, k, v, backend="cuda")` | `q` is `(B, H, S, D)`, `k` and `v` are `(B, KV, T, D)` | `D` is 64 or 128 |

`backend` is `"torch"`, `"cuda"`, or `"triton"`. `ops.triton_available()` is false when Triton is not installed.

Query row `i` sees key `j` when `j <= i + (T - S)`. A decode step is `S = 1`, so every cached key is visible. Square prefill is the ordinary causal mask. The kernel keeps the dots and the softmax in fp32 and does not allocate the score matrix.

Qwen2.5 shapes used by the engine are the target `(hidden 1536, 12 query heads, 2 kv heads, head dim 128, intermediate 8960)` and the draft `(hidden 896, 14 / 2, head dim 64, intermediate 4864)`.

## Turn a pack on inside the engine

`patch.install` replaces methods on the live classes. Cache layout, FP8 quant, and the attention-phase counter stay as they are.

```python
from specdec.engine.model_self import set_attention_strategy
from specdec.kernels import patch

set_attention_strategy("eager")
patch.install("all")
# existing generate or spec_decode call
patch.restore()
```

| pack | RMSNorm | RoPE | SwiGLU | eager attention |
| --- | --- | --- | --- | --- |
| `torch` | engine | engine | engine | engine |
| `elementwise` | CUDA | CUDA | CUDA | engine |
| `attention` | engine | engine | engine | CUDA |
| `all` | CUDA | CUDA | CUDA | CUDA |

`patch.current_pack()` returns the active name. `patch.install("torch")` and `patch.restore()` put the original methods back.

The default attention strategy is `eager`, so an attention pack runs on prefill and decode. If the strategy is `cudnn_prefill`, prefill blocks that the engine already sends to cuDNN stay on that path. Decode, draft proposal, verify, and repair stay eager, and the attention pack replaces that eager call.

The CUDA and Triton attention kernels accumulate QK in fp32. Eager bf16 `torch.matmul` rounds the score matrix to bf16, so a near-tie can take a different argmax and every later token follows it. `reference.attention_fp32` is the formula check. `reference.attention` is the eager-engine check.

## 8 GiB allocator cap

On a larger card, set the caching allocator to 8 GiB before the first allocation. This is what an 8 GiB card does when the reserved pool hits the cap. It does not reproduce Windows WDDM paging.

```python
import torch

props = torch.cuda.get_device_properties(0)
torch.cuda.set_per_process_memory_fraction(min(1.0, (8 * 1024**3) / props.total_memory))
```

Run one pack per process so the cap starts from an empty pool.

## Sample experiment

No weights and no download. From the SpecDec repository root:

```powershell
python -m specdec.kernels.sample_experiment
```

From the experiment checkout:

```powershell
python -m kernels.sample_experiment
```

The first CUDA call compiles the extension and can take about a minute. Later calls reuse it.

The script does four things:

1. Caps the caching allocator at 8 GiB.
2. Compares CUDA, and Triton when it imports, with the Torch formulas.
   - fp32 RMSNorm, hidden 1536, shape `(4, 128, 1536)`
   - bf16 RoPE, 12 heads, head dim 128, sequence 32
   - bf16 SwiGLU, intermediate 8960, shape `(2, 32, 8960)`
   - fp32 attention, 4 / 2 heads, head dim 64, `S = T = 64`
   - bf16 decode attention, 12 / 2 heads, head dim 128, `S = 1`, `T = 128`
3. Times that decode row: 5 warmup launches, then the median of 20 CUDA events, for `"torch"` and `"cuda"`.
4. Installs the `all` pack and restores the engine methods.

An fp32 row passes when max abs error is at most `1e-4`. A bf16 row passes when max abs error is at most `2e-2`, or at most 2% of the output magnitude. Exit code 0 means every row passed.

The same attention call with a long cache is the row that materializes a large score tensor on the eager path:

```python
q = torch.randn(1, 12, 2048, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 2, 8192, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
out = ops.attention(q, k, v, backend="cuda")
```

Keep the 8 GiB cap on, and run that row in its own process. `"torch"` allocates the score tensor. `"cuda"` and `"triton"` do not.

An engine ablation uses the same packs as the operator sample, one fresh process each, strategy `eager`:

```python
patch.install("torch")        # baseline
patch.install("elementwise")  # RMSNorm, RoPE, SwiGLU
patch.install("attention")    # attention only
patch.install("all")
```
