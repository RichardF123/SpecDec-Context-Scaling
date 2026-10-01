#include <torch/extension.h>

#include "dtype.cuh"

// y = silu(gate) * up, with silu(g) = g * sigmoid(g) computed in fp32.

template <typename T>
__global__ void swiglu_kernel(const T* __restrict__ gate, const T* __restrict__ up, T* __restrict__ y, long long n) {
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        const float g = load_as_float(gate + i);
        const float u = load_as_float(up + i);
        const float sig = 1.f / (1.f + expf(-g));
        store_from_float(y + i, g * sig * u);
    }
}

torch::Tensor swiglu_cuda(torch::Tensor gate, torch::Tensor up) {
    TORCH_CHECK(gate.is_cuda() && up.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(gate.is_contiguous() && up.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up shapes differ");
    TORCH_CHECK(gate.scalar_type() == up.scalar_type(), "dtype mismatch");
    auto y = torch::empty_like(gate);
    const long long n = gate.numel();
    if (n == 0) {
        return y;
    }
    const int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    if (blocks > 2048) {
        blocks = 2048;
    }
    if (gate.scalar_type() == torch::kFloat32) {
        swiglu_kernel<float><<<blocks, threads>>>(gate.data_ptr<float>(), up.data_ptr<float>(), y.data_ptr<float>(), n);
    } else if (gate.scalar_type() == torch::kBFloat16) {
        auto* g = reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr());
        auto* u = reinterpret_cast<const __nv_bfloat16*>(up.data_ptr());
        auto* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
        swiglu_kernel<__nv_bfloat16><<<blocks, threads>>>(g, u, yp, n);
    } else {
        TORCH_CHECK(false, "swiglu supports float32 and bfloat16");
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, cudaGetErrorString(cudaGetLastError()));
    return y;
}
