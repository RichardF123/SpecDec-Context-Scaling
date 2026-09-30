"""Build the CUDA C++ extension.

Windows (instructor RTX 4060, Torch cu124): the same `load()` call works when
`cl.exe` and `nvcc` match that wheel. The extension is compiled for the GPU
that is visible at build time. Triton is not imported here, so a machine
without Triton can still load the CUDA kernels.
"""

from __future__ import annotations

import os
from pathlib import Path

_MOD = None


def load():
    global _MOD
    if _MOD is not None:
        return _MOD
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build the custom kernels")
    major, minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    from torch.utils.cpp_extension import load as cpp_load

    root = Path(__file__).resolve().parent
    # specdec/kernels/ -> repository root is parents[1] of this directory.
    # TORCH_EXTENSIONS_DIR keeps the JIT output off the source tree when set.
    env_cache = os.environ.get("TORCH_EXTENSIONS_DIR")
    cache = Path(env_cache) if env_cache else root.parents[1] / ".cache" / "torch_extensions"
    cache.mkdir(parents=True, exist_ok=True)
    src = root / "cuda"
    _MOD = cpp_load(
        name="specdec_custom_kernels",
        sources=[
            str(src / "bind.cpp"),
            str(src / "rmsnorm.cu"),
            str(src / "rope.cu"),
            str(src / "swiglu.cu"),
            str(src / "attention.cu"),
        ],
        extra_include_paths=[str(src)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        build_directory=str(cache),
        verbose=False,
        with_cuda=True,
    )
    return _MOD
