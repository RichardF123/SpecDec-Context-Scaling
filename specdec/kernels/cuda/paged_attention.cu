#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "dtype.cuh"

// Paged KV decode attention v1.
//
// Scope intentionally narrow:
//   q:       [B, H, 1, D]
//   K pages: each physical page is [B, KV, PAGE, D]
//   V pages: same
//   D:       64 or 128
//   dtype:   float32 or bfloat16
//
// The Python wrapper passes device-side arrays of raw CUDA addresses:
//   k_ptrs[p] = data_ptr(K_page_p)
//   v_ptrs[p] = data_ptr(V_page_p)
//
// Logical page order == physical page-list order in paged_v0.
// No materialization of full K/V is performed.
//
// This is decode-only (S == 1). All cached keys are visible.
// Online softmax is accumulated in fp32.

template <typename T, int D>
__global__ void paged_decode_kernel(
    const T* __restrict__ q,
    const int64_t* __restrict__ k_ptrs,
    const int64_t* __restrict__ v_ptrs,
    T* __restrict__ out,
    int B,
    int H,
    int KV,
    int page_size,
    int length)
{
    constexpr int PER = D / 32;

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;

    const int warp_id =
        blockIdx.x * warps_per_block + warp_in_block;

    const int total = B * H;
    if (warp_id >= total) {
        return;
    }

    const int b = warp_id / H;
    const int h = warp_id - b * H;

    const int n_rep = H / KV;
    const int kv = h / n_rep;

    // q layout: [B, H, 1, D]
    const T* qp =
        q + ((static_cast<long long>(b) * H + h) * D);

    float qv[PER];

#pragma unroll
    for (int p = 0; p < PER; ++p) {
        qv[p] = load_as_float(qp + lane + p * 32);
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;

    float acc[PER];

#pragma unroll
    for (int p = 0; p < PER; ++p) {
        acc[p] = 0.0f;
    }

    const float scale = rsqrtf(static_cast<float>(D));

    for (int j = 0; j < length; ++j) {
        const int page = j / page_size;
        const int slot = j - page * page_size;

        const T* k_page =
            reinterpret_cast<const T*>(k_ptrs[page]);
        const T* v_page =
            reinterpret_cast<const T*>(v_ptrs[page]);

        // page layout: [B, KV, PAGE, D]
        const long long base =
            (((static_cast<long long>(b) * KV + kv)
               * page_size + slot) * D);

        const T* kj = k_page + base;
        const T* vj = v_page + base;

        float partial = 0.0f;

#pragma unroll
        for (int p = 0; p < PER; ++p) {
            partial +=
                qv[p] *
                load_as_float(kj + lane + p * 32);
        }

#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            partial +=
                __shfl_xor_sync(
                    0xffffffff,
                    partial,
                    off);
        }

        const float score = partial * scale;

        const float m_new =
            fmaxf(m_i, score);

        const float alpha =
            __expf(m_i - m_new);

        const float pv =
            __expf(score - m_new);

        l_i =
            l_i * alpha + pv;

#pragma unroll
        for (int p = 0; p < PER; ++p) {
            acc[p] =
                acc[p] * alpha
                + pv *
                load_as_float(
                    vj + lane + p * 32);
        }

        m_i = m_new;
    }

    T* op =
        out +
        ((static_cast<long long>(b) * H + h) * D);

    if (l_i == 0.0f) {
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            store_from_float(
                op + lane + p * 32,
                0.0f);
        }
    } else {
        const float inv =
            1.0f / l_i;

#pragma unroll
        for (int p = 0; p < PER; ++p) {
            store_from_float(
                op + lane + p * 32,
                acc[p] * inv);
        }
    }
}


template <typename T, int D>
static void launch_paged_decode(
    const T* q,
    const int64_t* k_ptrs,
    const int64_t* v_ptrs,
    T* out,
    int B,
    int H,
    int KV,
    int page_size,
    int length)
{
    constexpr int WARPS = 4;

    const int total = B * H;

    const int blocks =
        (total + WARPS - 1) / WARPS;

    paged_decode_kernel<T, D>
        <<<blocks, WARPS * 32>>>(
            q,
            k_ptrs,
            v_ptrs,
            out,
            B,
            H,
            KV,
            page_size,
            length);
}


torch::Tensor paged_attention_decode_cuda(
    torch::Tensor q,
    torch::Tensor k_ptrs,
    torch::Tensor v_ptrs,
    int64_t kv_heads,
    int64_t page_size,
    int64_t length)
{
    TORCH_CHECK(
        q.is_cuda(),
        "q must be CUDA");

    TORCH_CHECK(
        k_ptrs.is_cuda() &&
        v_ptrs.is_cuda(),
        "pointer tables must be CUDA");

    TORCH_CHECK(
        q.dim() == 4,
        "q must be rank-4 [B,H,1,D]");

    TORCH_CHECK(
        q.size(2) == 1,
        "paged_attention_decode is decode-only: q_len must be 1");

    TORCH_CHECK(
        k_ptrs.scalar_type() == torch::kInt64 &&
        v_ptrs.scalar_type() == torch::kInt64,
        "pointer tables must be int64");

    TORCH_CHECK(
        k_ptrs.dim() == 1 &&
        v_ptrs.dim() == 1,
        "pointer tables must be rank-1");

    TORCH_CHECK(
        k_ptrs.numel() == v_ptrs.numel(),
        "K/V pointer table length mismatch");

    TORCH_CHECK(
        kv_heads > 0,
        "kv_heads must be positive");

    TORCH_CHECK(
        page_size > 0,
        "page_size must be positive");

    TORCH_CHECK(
        length > 0,
        "length must be positive");

    const int B =
        static_cast<int>(q.size(0));

    const int H =
        static_cast<int>(q.size(1));

    const int D =
        static_cast<int>(q.size(3));

    const int KV =
        static_cast<int>(kv_heads);

    TORCH_CHECK(
        H % KV == 0,
        "query heads must be divisible by KV heads");

    TORCH_CHECK(
        D == 64 || D == 128,
        "head_dim must be 64 or 128");

    const int64_t required_pages =
        (length + page_size - 1) / page_size;

    TORCH_CHECK(
        k_ptrs.numel() >= required_pages,
        "not enough pages for logical length");

    TORCH_CHECK(
        q.scalar_type() == torch::kFloat32 ||
        q.scalar_type() == torch::kBFloat16,
        "paged attention supports float32 and bfloat16");

    auto out =
        torch::empty_like(q);

    const int64_t* kp =
        k_ptrs.data_ptr<int64_t>();

    const int64_t* vp =
        v_ptrs.data_ptr<int64_t>();

    if (q.scalar_type() == torch::kFloat32) {
        const float* qp =
            q.data_ptr<float>();

        float* op =
            out.data_ptr<float>();

        if (D == 64) {
            launch_paged_decode<float, 64>(
                qp,
                kp,
                vp,
                op,
                B,
                H,
                KV,
                static_cast<int>(page_size),
                static_cast<int>(length));
        } else {
            launch_paged_decode<float, 128>(
                qp,
                kp,
                vp,
                op,
                B,
                H,
                KV,
                static_cast<int>(page_size),
                static_cast<int>(length));
        }
    } else {
        const __nv_bfloat16* qp =
            reinterpret_cast<const __nv_bfloat16*>(
                q.data_ptr());

        __nv_bfloat16* op =
            reinterpret_cast<__nv_bfloat16*>(
                out.data_ptr());

        if (D == 64) {
            launch_paged_decode<__nv_bfloat16, 64>(
                qp,
                kp,
                vp,
                op,
                B,
                H,
                KV,
                static_cast<int>(page_size),
                static_cast<int>(length));
        } else {
            launch_paged_decode<__nv_bfloat16, 128>(
                qp,
                kp,
                vp,
                op,
                B,
                H,
                KV,
                static_cast<int>(page_size),
                static_cast<int>(length));
        }
    }

    const cudaError_t err =
        cudaGetLastError();

    TORCH_CHECK(
        err == cudaSuccess,
        cudaGetErrorString(err));

    return out;
}
