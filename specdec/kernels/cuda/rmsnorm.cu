#include <torch/extension.h>

#include "dtype.cuh"

// y = weight * cast(x * rsqrt(mean(x^2) + eps))
// One block, one row. Matches RMSNORM.forward in model_self.py.

template <typename T>
__global__ void rmsnorm_kernel(const T* __restrict__ x, const T* __restrict__ weight, T* __restrict__ y,
                               int hidden, float eps) {
    const long long row = blockIdx.x;
    const T* row_x = x + row * hidden;
    T* row_y = y + row * hidden;

    float local = 0.f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = load_as_float(row_x + i);
        local += v * v;
    }

    constexpr int nwarps = 8;  // blockDim == 256
    __shared__ float warp_sum[nwarps];
    float v = local;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off);
    }
    if ((threadIdx.x & 31) == 0) {
        warp_sum[threadIdx.x >> 5] = v;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float w = (threadIdx.x < nwarps) ? warp_sum[threadIdx.x] : 0.f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            w += __shfl_xor_sync(0xffffffff, w, off);
        }
        if (threadIdx.x == 0) {
            warp_sum[0] = w;
        }
    }
    __syncthreads();

    const float inv = rsqrtf(warp_sum[0] / static_cast<float>(hidden) + eps);
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float xv = load_as_float(row_x + i);
        float wv = load_as_float(weight + i);
        store_rmsnorm(row_y + i, xv * inv, wv);
    }
}

torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "x and weight dtype differ");
    TORCH_CHECK(x.size(-1) == weight.numel(), "weight must match the last dimension");
    const int hidden = static_cast<int>(x.size(-1));
    TORCH_CHECK(hidden > 0, "empty hidden size");
    const int rows = static_cast<int>(x.numel() / hidden);
    auto y = torch::empty_like(x);
    if (rows == 0) {
        return y;
    }
    const float eps_f = static_cast<float>(eps);
    if (x.scalar_type() == torch::kFloat32) {
        rmsnorm_kernel<float><<<rows, 256>>>(x.data_ptr<float>(), weight.data_ptr<float>(), y.data_ptr<float>(),
                                              hidden, eps_f);
    } else if (x.scalar_type() == torch::kBFloat16) {
        auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
        auto* wp = reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr());
        auto* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
        rmsnorm_kernel<__nv_bfloat16><<<rows, 256>>>(xp, wp, yp, hidden, eps_f);
    } else {
        TORCH_CHECK(false, "rmsnorm supports float32 and bfloat16");
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, cudaGetErrorString(cudaGetLastError()));
    return y;
}
