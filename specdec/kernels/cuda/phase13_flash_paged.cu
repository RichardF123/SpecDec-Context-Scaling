
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

#include "dtype.cuh"

// PHASE13_FLASH_PAGED
//
// Page-native, GQA-aware tiled online-softmax attention for q_len=1..4.
//
// This is deliberately "Flash-style" rather than a claim of reproducing the
// official FlashAttention implementation.  Key ideas:
//   1. keep K/V paged -- no full-history materialization;
//   2. tile K/V through shared memory;
//   3. share each K/V tile across query heads that map to the same KV head;
//   4. maintain online softmax state (m_i, l_i) in registers;
//   5. support BF16 K/V and asymmetric K-BF16 + V-E4M3FN.
//
// RTX 4060 target geometry:
//   target: H=12, KV=2, D=128  -> 6 query heads share one KV head
//   draft:  H=14, KV=2, D=64   -> 7 query heads share one KV head
//
// A block has eight warps.  Each warp owns one (query-head, query-position)
// row; all warps in the block share the same physical K/V tile.

namespace {

constexpr int TILE_T = 16;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS = WARPS_PER_BLOCK * 32;

__device__ __forceinline__ float p13_decode_e4m3fn(uint8_t x) {
    const int sign = (x >> 7) & 1;
    const int exp = (x >> 3) & 0xF;
    const int mant = x & 0x7;
    float value;
    if (exp == 0) {
        value = static_cast<float>(mant) * 0.001953125f;
    } else if (exp == 0xF && mant == 0x7) {
        value = NAN;
    } else {
        value = ldexpf(
            1.0f + static_cast<float>(mant) * 0.125f,
            exp - 7);
    }
    return sign ? -value : value;
}

template <int D, bool VFP8>
__global__ void flash_paged_kernel(
    const __nv_bfloat16* __restrict__ q,
    const int64_t* __restrict__ k_ptrs,
    const int64_t* __restrict__ v_ptrs,
    const int64_t* __restrict__ s_ptrs,
    __nv_bfloat16* __restrict__ out,
    int B,
    int H,
    int KV,
    int S,
    int page_size,
    int length,
    int scale_groups)
{
    constexpr int PER = D / 32;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto* k_tile =
        reinterpret_cast<__nv_bfloat16*>(smem_raw);
    auto* v_tile =
        k_tile + TILE_T * D;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    const int n_rep = H / KV;
    const int rows_per_kv = n_rep * S;
    const int row_groups =
        (rows_per_kv + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    int block = static_cast<int>(blockIdx.x);
    const int row_group = block % row_groups;
    block /= row_groups;
    const int kv = block % KV;
    const int b = block / KV;

    const int local_row =
        row_group * WARPS_PER_BLOCK + warp;
    const bool active = local_row < rows_per_kv;

    int h = 0;
    int qi = 0;
    int limit = -1;

    float qv[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) qv[p] = 0.0f;

    if (active) {
        const int h_local = local_row / S;
        qi = local_row - h_local * S;
        h = kv * n_rep + h_local;
        limit = qi + (length - S);

        const auto* qp =
            q + ((static_cast<long long>(b) * H + h) * S + qi) * D;

#pragma unroll
        for (int p = 0; p < PER; ++p) {
            qv[p] = __bfloat162float(
                qp[lane + p * 32]);
        }
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) acc[p] = 0.0f;

    const float qk_scale =
        rsqrtf(static_cast<float>(D));

    for (int tile_start = 0;
         tile_start < length;
         tile_start += TILE_T) {

        // Cooperative K/V tile load.  The tile is shared across all query
        // heads mapped to this KV head.
        const int elems = TILE_T * D;
        for (int e = threadIdx.x; e < elems; e += blockDim.x) {
            const int tj = e / D;
            const int d = e - tj * D;
            const int j = tile_start + tj;

            __nv_bfloat16 kval = __float2bfloat16(0.0f);
            __nv_bfloat16 vval = __float2bfloat16(0.0f);

            if (j < length) {
                const int page = j / page_size;
                const int slot = j - page * page_size;

                const auto* k_page =
                    reinterpret_cast<const __nv_bfloat16*>(
                        k_ptrs[page]);

                const long long base =
                    (((static_cast<long long>(b) * KV + kv)
                       * page_size + slot) * D);

                kval = k_page[base + d];

                if constexpr (!VFP8) {
                    const auto* v_page =
                        reinterpret_cast<const __nv_bfloat16*>(
                            v_ptrs[page]);
                    vval = v_page[base + d];
                } else {
                    const auto* v_page =
                        reinterpret_cast<const uint8_t*>(
                            v_ptrs[page]);
                    const auto* s_page =
                        reinterpret_cast<const float*>(
                            s_ptrs[page]);

                    const int group_size = D / scale_groups;
                    const long long sbase =
                        (((static_cast<long long>(b) * KV + kv)
                           * page_size + slot) * scale_groups);

                    const float raw =
                        p13_decode_e4m3fn(v_page[base + d]);
                    const float scaled =
                        raw * s_page[sbase + d / group_size];

                    // Match the final V-FP8 reference: restored V is rounded
                    // to BF16 before FP32 accumulation.
                    vval = __float2bfloat16(scaled);
                }
            }

            k_tile[e] = kval;
            v_tile[e] = vval;
        }

        __syncthreads();

        if (active) {
            const int visible =
                min(TILE_T, limit - tile_start + 1);

            for (int tj = 0; tj < visible; ++tj) {
                float partial = 0.0f;

#pragma unroll
                for (int p = 0; p < PER; ++p) {
                    const int d = lane + p * 32;
                    partial +=
                        qv[p] *
                        __bfloat162float(
                            k_tile[tj * D + d]);
                }

#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    partial += __shfl_xor_sync(
                        0xffffffff, partial, off);
                }

                const float score =
                    partial * qk_scale;

                const float m_new =
                    fmaxf(m_i, score);
                const float alpha =
                    __expf(m_i - m_new);
                const float pv =
                    __expf(score - m_new);

                l_i = l_i * alpha + pv;

#pragma unroll
                for (int p = 0; p < PER; ++p) {
                    const int d = lane + p * 32;
                    acc[p] =
                        acc[p] * alpha +
                        pv * __bfloat162float(
                            v_tile[tj * D + d]);
                }

                m_i = m_new;
            }
        }

        __syncthreads();
    }

    if (active) {
        auto* op =
            out + ((static_cast<long long>(b) * H + h) * S + qi) * D;

        const float inv =
            l_i == 0.0f ? 0.0f : 1.0f / l_i;

#pragma unroll
        for (int p = 0; p < PER; ++p) {
            op[lane + p * 32] =
                __float2bfloat16(acc[p] * inv);
        }
    }
}

template <int D, bool VFP8>
void launch_flash_paged(
    const __nv_bfloat16* q,
    const int64_t* k_ptrs,
    const int64_t* v_ptrs,
    const int64_t* s_ptrs,
    __nv_bfloat16* out,
    int B,
    int H,
    int KV,
    int S,
    int page_size,
    int length,
    int scale_groups)
{
    const int n_rep = H / KV;
    const int rows_per_kv = n_rep * S;
    const int row_groups =
        (rows_per_kv + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    const int blocks =
        B * KV * row_groups;

    const size_t shared_bytes =
        static_cast<size_t>(2 * TILE_T * D) *
        sizeof(__nv_bfloat16);

    flash_paged_kernel<D, VFP8>
        <<<blocks, THREADS, shared_bytes>>>(
            q,
            k_ptrs,
            v_ptrs,
            s_ptrs,
            out,
            B,
            H,
            KV,
            S,
            page_size,
            length,
            scale_groups);
}

void common_checks(
    const torch::Tensor& q,
    const torch::Tensor& k_ptrs,
    const torch::Tensor& v_ptrs,
    int64_t kv_heads,
    int64_t page_size,
    int64_t length)
{
    TORCH_CHECK(
        q.is_cuda() && q.is_contiguous(),
        "q must be contiguous CUDA");

    TORCH_CHECK(
        q.scalar_type() == torch::kBFloat16,
        "Phase13 Flash-paged attention requires BF16 q");

    TORCH_CHECK(
        q.dim() == 4,
        "q must be [B,H,S,D]");

    TORCH_CHECK(
        q.size(2) >= 1 && q.size(2) <= 4,
        "Phase13 Flash-paged supports q_len 1..4");

    TORCH_CHECK(
        k_ptrs.is_cuda() && v_ptrs.is_cuda(),
        "pointer tables must be CUDA");

    TORCH_CHECK(
        k_ptrs.scalar_type() == torch::kInt64 &&
        v_ptrs.scalar_type() == torch::kInt64,
        "pointer tables must be int64");

    TORCH_CHECK(
        kv_heads > 0,
        "kv_heads must be positive");

    TORCH_CHECK(
        page_size > 0,
        "page_size must be positive");

    TORCH_CHECK(
        length >= q.size(2),
        "logical length must include current query block");

    TORCH_CHECK(
        q.size(1) % kv_heads == 0,
        "query heads must be divisible by KV heads");

    TORCH_CHECK(
        q.size(3) == 64 || q.size(3) == 128,
        "head_dim must be 64 or 128");

    const int64_t required_pages =
        (length + page_size - 1) / page_size;

    TORCH_CHECK(
        k_ptrs.numel() >= required_pages &&
        v_ptrs.numel() >= required_pages,
        "insufficient paged pointer table length");
}

} // anonymous namespace


torch::Tensor phase13_flash_paged_bf16_cuda(
    torch::Tensor q,
    torch::Tensor k_ptrs,
    torch::Tensor v_ptrs,
    int64_t kv_heads,
    int64_t page_size,
    int64_t length)
{
    common_checks(
        q, k_ptrs, v_ptrs,
        kv_heads, page_size, length);

    auto out = torch::empty_like(q);

    const int B = static_cast<int>(q.size(0));
    const int H = static_cast<int>(q.size(1));
    const int S = static_cast<int>(q.size(2));
    const int D = static_cast<int>(q.size(3));
    const int KV = static_cast<int>(kv_heads);

    const auto* qp =
        reinterpret_cast<const __nv_bfloat16*>(
            q.data_ptr());

    auto* op =
        reinterpret_cast<__nv_bfloat16*>(
            out.data_ptr());

    const int64_t* kp =
        k_ptrs.data_ptr<int64_t>();

    const int64_t* vp =
        v_ptrs.data_ptr<int64_t>();

    if (D == 64) {
        launch_flash_paged<64, false>(
            qp, kp, vp, nullptr, op,
            B, H, KV, S,
            static_cast<int>(page_size),
            static_cast<int>(length),
            0);
    } else {
        launch_flash_paged<128, false>(
            qp, kp, vp, nullptr, op,
            B, H, KV, S,
            static_cast<int>(page_size),
            static_cast<int>(length),
            0);
    }

    const cudaError_t err =
        cudaGetLastError();

    TORCH_CHECK(
        err == cudaSuccess,
        cudaGetErrorString(err));

    return out;
}


torch::Tensor phase13_flash_paged_vfp8_cuda(
    torch::Tensor q,
    torch::Tensor k_ptrs,
    torch::Tensor v_ptrs,
    torch::Tensor s_ptrs,
    int64_t kv_heads,
    int64_t page_size,
    int64_t length,
    int64_t scale_groups)
{
    common_checks(
        q, k_ptrs, v_ptrs,
        kv_heads, page_size, length);

    TORCH_CHECK(
        s_ptrs.is_cuda() &&
        s_ptrs.scalar_type() == torch::kInt64,
        "scale pointer table must be CUDA int64");

    TORCH_CHECK(
        scale_groups > 0 &&
        q.size(3) % scale_groups == 0,
        "invalid V-FP8 scale geometry");

    const int64_t required_pages =
        (length + page_size - 1) / page_size;

    TORCH_CHECK(
        s_ptrs.numel() >= required_pages,
        "insufficient scale pointer table length");

    auto out = torch::empty_like(q);

    const int B = static_cast<int>(q.size(0));
    const int H = static_cast<int>(q.size(1));
    const int S = static_cast<int>(q.size(2));
    const int D = static_cast<int>(q.size(3));
    const int KV = static_cast<int>(kv_heads);

    const auto* qp =
        reinterpret_cast<const __nv_bfloat16*>(
            q.data_ptr());

    auto* op =
        reinterpret_cast<__nv_bfloat16*>(
            out.data_ptr());

    const int64_t* kp =
        k_ptrs.data_ptr<int64_t>();

    const int64_t* vp =
        v_ptrs.data_ptr<int64_t>();

    const int64_t* sp =
        s_ptrs.data_ptr<int64_t>();

    if (D == 64) {
        launch_flash_paged<64, true>(
            qp, kp, vp, sp, op,
            B, H, KV, S,
            static_cast<int>(page_size),
            static_cast<int>(length),
            static_cast<int>(scale_groups));
    } else {
        launch_flash_paged<128, true>(
            qp, kp, vp, sp, op,
            B, H, KV, S,
            static_cast<int>(page_size),
            static_cast<int>(length),
            static_cast<int>(scale_groups));
    }

    const cudaError_t err =
        cudaGetLastError();

    TORCH_CHECK(
        err == cudaSuccess,
        cudaGetErrorString(err));

    return out;
}
