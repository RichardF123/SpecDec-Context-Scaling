#include <torch/extension.h>

#include "dtype.cuh"

// y = x * cos + rotate_half(x) * sin
// rotate_half(x) = cat(-x[..., d/2:], x[..., :d/2], dim=-1)
// Output is contiguous (B, H, S, D). Input may be strided: the engine
// transposes Q/K to (B, H, S, D) before RoPE. cos/sin are contiguous (S, D).

template <typename T>
__global__ void rope_kernel(const T* __restrict__ x, const T* __restrict__ cos, const T* __restrict__ sin,
                            T* __restrict__ y, int heads, int seq, int head_dim, long long n, long long stride_b,
                            long long stride_h, long long stride_s, long long stride_d) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    const int half = head_dim / 2;
    const int d = static_cast<int>(idx % head_dim);
    const long long sd = static_cast<long long>(head_dim) * seq;
    const int s = static_cast<int>((idx / head_dim) % seq);
    const int h = static_cast<int>((idx / sd) % heads);
    const int b = static_cast<int>(idx / (sd * heads));

    const long long base = static_cast<long long>(b) * stride_b + static_cast<long long>(h) * stride_h +
                           static_cast<long long>(s) * stride_s;
    const long long in = base + static_cast<long long>(d) * stride_d;
    const int partner_d = (d < half) ? (d + half) : (d - half);
    const long long in_p = base + static_cast<long long>(partner_d) * stride_d;
    const float xv = load_as_float(x + in);
    float pv = load_as_float(x + in_p);
    if (d < half) {
        pv = -pv;
    }
    const long long cs = static_cast<long long>(s) * head_dim + d;
    const float yv = xv * load_as_float(cos + cs) + pv * load_as_float(sin + cs);
    store_from_float(y + idx, yv);
}

torch::Tensor rope_cuda(torch::Tensor x, torch::Tensor cos, torch::Tensor sin) {
    TORCH_CHECK(x.is_cuda() && cos.is_cuda() && sin.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(x.dim() == 4, "x is (batch, heads, seq, head_dim)");
    TORCH_CHECK(x.scalar_type() == cos.scalar_type() && x.scalar_type() == sin.scalar_type(), "dtype mismatch");
    TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(), "cos and sin must be contiguous (S, D)");
    const int seq = static_cast<int>(x.size(2));
    const int head_dim = static_cast<int>(x.size(3));
    const int heads = static_cast<int>(x.size(1));
    TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even");
    TORCH_CHECK(cos.numel() == static_cast<long long>(seq) * head_dim, "cos must be (seq, head_dim)");
    TORCH_CHECK(sin.numel() == cos.numel(), "sin shape differs from cos");

    auto y = torch::empty_like(x, torch::MemoryFormat::Contiguous);
    const long long n = x.numel();
    if (n == 0) {
        return y;
    }
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const long long stride_b = x.stride(0);
    const long long stride_h = x.stride(1);
    const long long stride_s = x.stride(2);
    const long long stride_d = x.stride(3);

    if (x.scalar_type() == torch::kFloat32) {
        rope_kernel<float><<<blocks, threads>>>(x.data_ptr<float>(), cos.data_ptr<float>(), sin.data_ptr<float>(),
                                                 y.data_ptr<float>(), heads, seq, head_dim, n, stride_b, stride_h,
                                                 stride_s, stride_d);
    } else if (x.scalar_type() == torch::kBFloat16) {
        auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
        auto* cp = reinterpret_cast<const __nv_bfloat16*>(cos.data_ptr());
        auto* sp = reinterpret_cast<const __nv_bfloat16*>(sin.data_ptr());
        auto* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
        rope_kernel<__nv_bfloat16><<<blocks, threads>>>(xp, cp, sp, yp, heads, seq, head_dim, n, stride_b, stride_h,
                                                         stride_s, stride_d);
    } else {
        TORCH_CHECK(false, "rope supports float32 and bfloat16");
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, cudaGetErrorString(cudaGetLastError()));
    return y;
}
