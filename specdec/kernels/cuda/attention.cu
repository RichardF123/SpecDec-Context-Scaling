#include <torch/extension.h>

#include "dtype.cuh"

// Causal GQA attention with online softmax.
// q: (B, H, S, D), k/v: (B, KV, T, D), D in {64, 128}.
// Query row i may see keys j <= i + (T - S). That is the engine mask:
//   triu(..., diagonal=(T - S) + 1), including square prefill and s == 1.
// No score tensor and no kv_repeat expansion. Dots and softmax stay in fp32.
// Four warps in a block own four consecutive query rows of one head and share
// each key/value tile through shared memory.

template <typename T, int D>
__global__ void gqa_attn_kernel(const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
                                T* __restrict__ o, int H, int KV, int S, int Tlen) {
    constexpr int NQ = 4;
    constexpr int TILE = 8;
    constexpr int PER = D / 32;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int i0 = blockIdx.x * NQ;
    const int i = i0 + warp;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int n_rep = H / KV;
    const int kv = h / n_rep;
    const bool active = i < S;
    const int limit = active ? (i + (Tlen - S)) : -1;

    float qv[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        qv[p] = 0.f;
    }
    if (active) {
        const T* qp = q + ((static_cast<long long>(b) * H + h) * S + i) * D;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            qv[p] = load_as_float(qp + lane + p * 32);
        }
    }

    float m_i = -INFINITY;
    float l_i = 0.f;
    float acc[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        acc[p] = 0.f;
    }

    const int i_last = min(i0 + NQ - 1, S - 1);
    const int j_max = i_last + (Tlen - S);
    const float scale = rsqrtf(static_cast<float>(D));

    __shared__ float ksm[TILE][D];
    __shared__ float vsm[TILE][D];

    const T* kbase = k + (static_cast<long long>(b) * KV + kv) * Tlen * D;
    const T* vbase = v + (static_cast<long long>(b) * KV + kv) * Tlen * D;

    for (int j0 = 0; j0 <= j_max; j0 += TILE) {
        const int valid_n = min(TILE, j_max - j0 + 1);
        for (int t = threadIdx.x; t < valid_n * D; t += blockDim.x) {
            const int jj = t / D;
            const int dd = t - jj * D;
            const int j = j0 + jj;
            ksm[jj][dd] = load_as_float(kbase + static_cast<long long>(j) * D + dd);
            vsm[jj][dd] = load_as_float(vbase + static_cast<long long>(j) * D + dd);
        }
        __syncthreads();

        if (active && limit >= j0) {
            const int nloc = min(valid_n, limit - j0 + 1);
            for (int jj = 0; jj < nloc; ++jj) {
                float partial = 0.f;
#pragma unroll
                for (int p = 0; p < PER; ++p) {
                    partial += qv[p] * ksm[jj][lane + p * 32];
                }
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    partial += __shfl_xor_sync(0xffffffff, partial, off);
                }
                const float score = partial * scale;
                const float m_new = fmaxf(m_i, score);
                const float alpha = __expf(m_i - m_new);
                const float pv = __expf(score - m_new);
                l_i = l_i * alpha + pv;
#pragma unroll
                for (int p = 0; p < PER; ++p) {
                    acc[p] = acc[p] * alpha + pv * vsm[jj][lane + p * 32];
                }
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    if (active) {
        T* op = o + ((static_cast<long long>(b) * H + h) * S + i) * D;
        if (l_i == 0.f) {
#pragma unroll
            for (int p = 0; p < PER; ++p) {
                store_from_float(op + lane + p * 32, 0.f);
            }
        } else {
            const float inv = 1.f / l_i;
#pragma unroll
            for (int p = 0; p < PER; ++p) {
                store_from_float(op + lane + p * 32, acc[p] * inv);
            }
        }
    }
}

// One warp per query row. No block barrier, so short rows (decode s=1, verify s=k)
// are limited by the key scan instead of by syncthreads on a single block.
template <typename T, int D>
__global__ void gqa_attn_warp_kernel(const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
                                     T* __restrict__ o, int B, int H, int KV, int S, int Tlen) {
    constexpr int PER = D / 32;
    constexpr int UNROLL = 4;
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int warp_id = blockIdx.x * warps_per_block + warp_in_block;
    const int queries = H * S;
    if (warp_id >= B * queries) {
        return;
    }
    const int b = warp_id / queries;
    const int rem = warp_id - b * queries;
    const int h = rem / S;
    const int i = rem - h * S;
    const int n_rep = H / KV;
    const int kv = h / n_rep;
    const int limit = i + (Tlen - S);
    const float scale = rsqrtf(static_cast<float>(D));

    const T* qp = q + ((static_cast<long long>(b) * H + h) * S + i) * D;
    float qv[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        qv[p] = load_as_float(qp + lane + p * 32);
    }
    float m_i = -INFINITY;
    float l_i = 0.f;
    float acc[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        acc[p] = 0.f;
    }
    const T* kbase = k + (static_cast<long long>(b) * KV + kv) * Tlen * D;
    const T* vbase = v + (static_cast<long long>(b) * KV + kv) * Tlen * D;

    int j = 0;
    for (; j + UNROLL - 1 <= limit; j += UNROLL) {
        float scores[UNROLL];
        float vv[UNROLL][PER];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const T* kj = kbase + static_cast<long long>(j + u) * D;
            const T* vj = vbase + static_cast<long long>(j + u) * D;
            float partial = 0.f;
#pragma unroll
            for (int p = 0; p < PER; ++p) {
                partial += qv[p] * load_as_float(kj + lane + p * 32);
                vv[u][p] = load_as_float(vj + lane + p * 32);
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                partial += __shfl_xor_sync(0xffffffff, partial, off);
            }
            scores[u] = partial * scale;
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const float m_new = fmaxf(m_i, scores[u]);
            const float alpha = __expf(m_i - m_new);
            const float pv = __expf(scores[u] - m_new);
            l_i = l_i * alpha + pv;
#pragma unroll
            for (int p = 0; p < PER; ++p) {
                acc[p] = acc[p] * alpha + pv * vv[u][p];
            }
            m_i = m_new;
        }
    }
    for (; j <= limit; ++j) {
        const T* kj = kbase + static_cast<long long>(j) * D;
        const T* vj = vbase + static_cast<long long>(j) * D;
        float partial = 0.f;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            partial += qv[p] * load_as_float(kj + lane + p * 32);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            partial += __shfl_xor_sync(0xffffffff, partial, off);
        }
        const float score = partial * scale;
        const float m_new = fmaxf(m_i, score);
        const float alpha = __expf(m_i - m_new);
        const float pv = __expf(score - m_new);
        l_i = l_i * alpha + pv;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            acc[p] = acc[p] * alpha + pv * load_as_float(vj + lane + p * 32);
        }
        m_i = m_new;
    }

    T* op = o + ((static_cast<long long>(b) * H + h) * S + i) * D;
    if (l_i == 0.f) {
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            store_from_float(op + lane + p * 32, 0.f);
        }
    } else {
        const float inv = 1.f / l_i;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            store_from_float(op + lane + p * 32, acc[p] * inv);
        }
    }
}

template <typename T, int D>
void launch_attn(const T* q, const T* k, const T* v, T* o, int B, int H, int KV, int S, int Tlen) {
    if (S <= 32) {
        constexpr int warps_per_block = 4;
        const int total = B * H * S;
        const int blocks = (total + warps_per_block - 1) / warps_per_block;
        gqa_attn_warp_kernel<T, D><<<blocks, 32 * warps_per_block>>>(q, k, v, o, B, H, KV, S, Tlen);
        return;
    }
    constexpr int NQ = 4;
    dim3 block(32 * NQ);
    dim3 grid((S + NQ - 1) / NQ, H, B);
    gqa_attn_kernel<T, D><<<grid, block>>>(q, k, v, o, H, KV, S, Tlen);
}

torch::Tensor attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v are rank-4");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "dtype mismatch");
    const int B = static_cast<int>(q.size(0));
    const int H = static_cast<int>(q.size(1));
    const int S = static_cast<int>(q.size(2));
    const int D = static_cast<int>(q.size(3));
    const int KV = static_cast<int>(k.size(1));
    const int Tlen = static_cast<int>(k.size(2));
    TORCH_CHECK(k.size(0) == B && v.size(0) == B, "batch mismatch");
    TORCH_CHECK(k.size(2) == Tlen && v.size(2) == Tlen, "T mismatch");
    TORCH_CHECK(k.size(3) == D && v.size(3) == D, "head_dim mismatch");
    TORCH_CHECK(KV > 0 && H % KV == 0, "query heads must divide KV heads");
    TORCH_CHECK(D == 64 || D == 128, "head_dim must be 64 or 128");
    TORCH_CHECK(S > 0 && Tlen > 0, "empty sequence");
    auto o = torch::empty_like(q);

    if (q.scalar_type() == torch::kFloat32) {
        const float* qp = q.data_ptr<float>();
        const float* kp = k.data_ptr<float>();
        const float* vp = v.data_ptr<float>();
        float* op = o.data_ptr<float>();
        if (D == 64) {
            launch_attn<float, 64>(qp, kp, vp, op, B, H, KV, S, Tlen);
        } else {
            launch_attn<float, 128>(qp, kp, vp, op, B, H, KV, S, Tlen);
        }
    } else if (q.scalar_type() == torch::kBFloat16) {
        auto* qp = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
        auto* kp = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
        auto* vp = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
        auto* op = reinterpret_cast<__nv_bfloat16*>(o.data_ptr());
        if (D == 64) {
            launch_attn<__nv_bfloat16, 64>(qp, kp, vp, op, B, H, KV, S, Tlen);
        } else {
            launch_attn<__nv_bfloat16, 128>(qp, kp, vp, op, B, H, KV, S, Tlen);
        }
    } else {
        TORCH_CHECK(false, "attention supports float32 and bfloat16");
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, cudaGetErrorString(cudaGetLastError()));
    return o;
}
