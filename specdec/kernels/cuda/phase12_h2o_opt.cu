
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <cmath>
#include <cstdint>

#include "dtype.cuh"

// Experimental Phase-1 H2O execution path.
// - BF16/base and asymmetric V-FP8.
// - q_len 1..4.
// - Same warp-per-query online-softmax attention math as final paged path.
// - H2O mass is emitted after the final online-softmax normalizer is known,
//   by revisiting K once. This removes full-K materialization / kv_repeat /
//   framework matmul+softmax, but is NOT a zero-extra-QK single-pass fusion.

__device__ __forceinline__ float p12_decode_e4m3fn(uint8_t x) {
    const int sign = (x >> 7) & 1;
    const int exp = (x >> 3) & 0xF;
    const int mant = x & 0x7;
    float value;
    if (exp == 0) {
        value = static_cast<float>(mant) * 0.001953125f;
    } else if (exp == 0xF && mant == 0x7) {
        value = NAN;
    } else {
        value = ldexpf(1.0f + static_cast<float>(mant) * 0.125f, exp - 7);
    }
    return sign ? -value : value;
}

template <int D>
__global__ void p12_bf16_attn_h2o_kernel(
    const __nv_bfloat16* __restrict__ q,
    const int64_t* __restrict__ k_ptrs,
    const int64_t* __restrict__ v_ptrs,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ mass,
    int B, int H, int KV, int S, int page_size, int length)
{
    constexpr int PER = D / 32;
    constexpr int UNROLL = 4;

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int warp_id = blockIdx.x * warps_per_block + warp_in_block;
    const int queries = H * S;
    if (warp_id >= B * queries) return;

    const int b = warp_id / queries;
    const int rem = warp_id - b * queries;
    const int h = rem / S;
    const int i = rem - h * S;
    const int n_rep = H / KV;
    const int kv = h / n_rep;
    const int limit = i + (length - S);
    const float scale = rsqrtf(static_cast<float>(D));

    const __nv_bfloat16* qp =
        q + ((static_cast<long long>(b) * H + h) * S + i) * D;

    float qv[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        qv[p] = __bfloat162float(qp[lane + p * 32]);
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) acc[p] = 0.0f;

    int j = 0;
    for (; j + UNROLL - 1 <= limit; j += UNROLL) {
        float scores[UNROLL];
        float vv[UNROLL][PER];

#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int jj = j + u;
            const int page = jj / page_size;
            const int slot = jj - page * page_size;
            const auto* k_page =
                reinterpret_cast<const __nv_bfloat16*>(k_ptrs[page]);
            const auto* v_page =
                reinterpret_cast<const __nv_bfloat16*>(v_ptrs[page]);
            const long long base =
                (((static_cast<long long>(b) * KV + kv)
                   * page_size + slot) * D);
            const auto* kj = k_page + base;
            const auto* vj = v_page + base;

            float partial = 0.0f;
#pragma unroll
            for (int p = 0; p < PER; ++p) {
                const int d = lane + p * 32;
                partial += qv[p] * __bfloat162float(kj[d]);
                vv[u][p] = __bfloat162float(vj[d]);
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
        const int page = j / page_size;
        const int slot = j - page * page_size;
        const auto* k_page =
            reinterpret_cast<const __nv_bfloat16*>(k_ptrs[page]);
        const auto* v_page =
            reinterpret_cast<const __nv_bfloat16*>(v_ptrs[page]);
        const long long base =
            (((static_cast<long long>(b) * KV + kv)
               * page_size + slot) * D);
        const auto* kj = k_page + base;
        const auto* vj = v_page + base;

        float partial = 0.0f;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            partial += qv[p] *
                __bfloat162float(kj[lane + p * 32]);
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
            acc[p] = acc[p] * alpha +
                pv * __bfloat162float(vj[lane + p * 32]);
        }
        m_i = m_new;
    }

    auto* op =
        out + ((static_cast<long long>(b) * H + h) * S + i) * D;
    const float inv = l_i == 0.0f ? 0.0f : 1.0f / l_i;
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        op[lane + p * 32] = __float2bfloat16(acc[p] * inv);
    }

    // Probability mass uses the final online-softmax (m_i,l_i).
    if (l_i > 0.0f) {
        const float norm = 1.0f /
            (l_i * static_cast<float>(B * H));
        for (int jj = 0; jj <= limit; ++jj) {
            const int page = jj / page_size;
            const int slot = jj - page * page_size;
            const auto* k_page =
                reinterpret_cast<const __nv_bfloat16*>(k_ptrs[page]);
            const long long base =
                (((static_cast<long long>(b) * KV + kv)
                   * page_size + slot) * D);
            const auto* kj = k_page + base;
            float partial = 0.0f;
#pragma unroll
            for (int p = 0; p < PER; ++p) {
                partial += qv[p] *
                    __bfloat162float(kj[lane + p * 32]);
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                partial += __shfl_xor_sync(0xffffffff, partial, off);
            }
            if (lane == 0) {
                const float prob =
                    __expf(partial * scale - m_i) * norm;
                atomicAdd(
                    mass + static_cast<long long>(i) * length + jj,
                    prob);
            }
        }
    }
}

template <int D>
__global__ void p12_vfp8_attn_h2o_kernel(
    const __nv_bfloat16* __restrict__ q,
    const int64_t* __restrict__ k_ptrs,
    const int64_t* __restrict__ v_ptrs,
    const int64_t* __restrict__ s_ptrs,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ mass,
    int B, int H, int KV, int S, int page_size, int length,
    int scale_groups)
{
    constexpr int PER = D / 32;
    constexpr int UNROLL = 4;

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int warp_id = blockIdx.x * warps_per_block + warp_in_block;
    const int queries = H * S;
    if (warp_id >= B * queries) return;

    const int b = warp_id / queries;
    const int rem = warp_id - b * queries;
    const int h = rem / S;
    const int i = rem - h * S;
    const int n_rep = H / KV;
    const int kv = h / n_rep;
    const int limit = i + (length - S);
    const int group_size = D / scale_groups;
    const float scale = rsqrtf(static_cast<float>(D));

    const auto* qp =
        q + ((static_cast<long long>(b) * H + h) * S + i) * D;
    float qv[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        qv[p] = __bfloat162float(qp[lane + p * 32]);
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) acc[p] = 0.0f;

    int j = 0;
    for (; j + UNROLL - 1 <= limit; j += UNROLL) {
        float scores[UNROLL];
        float vv[UNROLL][PER];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int jj = j + u;
            const int page = jj / page_size;
            const int slot = jj - page * page_size;
            const auto* k_page =
                reinterpret_cast<const __nv_bfloat16*>(k_ptrs[page]);
            const auto* v_page =
                reinterpret_cast<const uint8_t*>(v_ptrs[page]);
            const auto* sc_page =
                reinterpret_cast<const float*>(s_ptrs[page]);
            const long long base =
                (((static_cast<long long>(b) * KV + kv)
                   * page_size + slot) * D);
            const long long sb =
                (((static_cast<long long>(b) * KV + kv)
                   * page_size + slot) * scale_groups);
            const auto* kj = k_page + base;
            const auto* vj = v_page + base;
            const auto* sj = sc_page + sb;

            float partial = 0.0f;
#pragma unroll
            for (int p = 0; p < PER; ++p) {
                const int d = lane + p * 32;
                partial += qv[p] * __bfloat162float(kj[d]);
                const float raw = p12_decode_e4m3fn(vj[d]);
                const float scaled = raw * sj[d / group_size];
                vv[u][p] =
                    __bfloat162float(__float2bfloat16(scaled));
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
        const int page = j / page_size;
        const int slot = j - page * page_size;
        const auto* k_page =
            reinterpret_cast<const __nv_bfloat16*>(k_ptrs[page]);
        const auto* v_page =
            reinterpret_cast<const uint8_t*>(v_ptrs[page]);
        const auto* sc_page =
            reinterpret_cast<const float*>(s_ptrs[page]);
        const long long base =
            (((static_cast<long long>(b) * KV + kv)
               * page_size + slot) * D);
        const long long sb =
            (((static_cast<long long>(b) * KV + kv)
               * page_size + slot) * scale_groups);
        const auto* kj = k_page + base;
        const auto* vj = v_page + base;
        const auto* sj = sc_page + sb;

        float partial = 0.0f;
        float vv[PER];
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int d = lane + p * 32;
            partial += qv[p] * __bfloat162float(kj[d]);
            const float raw = p12_decode_e4m3fn(vj[d]);
            const float scaled = raw * sj[d / group_size];
            vv[p] = __bfloat162float(__float2bfloat16(scaled));
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
            acc[p] = acc[p] * alpha + pv * vv[p];
        }
        m_i = m_new;
    }

    auto* op =
        out + ((static_cast<long long>(b) * H + h) * S + i) * D;
    const float inv = l_i == 0.0f ? 0.0f : 1.0f / l_i;
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        op[lane + p * 32] = __float2bfloat16(acc[p] * inv);
    }

    if (l_i > 0.0f) {
        const float norm = 1.0f /
            (l_i * static_cast<float>(B * H));
        for (int jj = 0; jj <= limit; ++jj) {
            const int page = jj / page_size;
            const int slot = jj - page * page_size;
            const auto* k_page =
                reinterpret_cast<const __nv_bfloat16*>(k_ptrs[page]);
            const long long base =
                (((static_cast<long long>(b) * KV + kv)
                   * page_size + slot) * D);
            const auto* kj = k_page + base;
            float partial = 0.0f;
#pragma unroll
            for (int p = 0; p < PER; ++p) {
                partial += qv[p] *
                    __bfloat162float(kj[lane + p * 32]);
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                partial += __shfl_xor_sync(0xffffffff, partial, off);
            }
            if (lane == 0) {
                const float prob =
                    __expf(partial * scale - m_i) * norm;
                atomicAdd(
                    mass + static_cast<long long>(i) * length + jj,
                    prob);
            }
        }
    }
}

template <int D>
static void launch_bf16(
    const __nv_bfloat16* q,
    const int64_t* kp, const int64_t* vp,
    __nv_bfloat16* out, float* mass,
    int B,int H,int KV,int S,int page_size,int length)
{
    constexpr int WARPS=4;
    const int total=B*H*S;
    const int blocks=(total+WARPS-1)/WARPS;
    p12_bf16_attn_h2o_kernel<D><<<blocks,WARPS*32>>>(
        q,kp,vp,out,mass,B,H,KV,S,page_size,length);
}

template <int D>
static void launch_vfp8(
    const __nv_bfloat16* q,
    const int64_t* kp, const int64_t* vp, const int64_t* sp,
    __nv_bfloat16* out, float* mass,
    int B,int H,int KV,int S,int page_size,int length,int groups)
{
    constexpr int WARPS=4;
    const int total=B*H*S;
    const int blocks=(total+WARPS-1)/WARPS;
    p12_vfp8_attn_h2o_kernel<D><<<blocks,WARPS*32>>>(
        q,kp,vp,sp,out,mass,B,H,KV,S,page_size,length,groups);
}

std::vector<torch::Tensor> phase12_paged_attention_h2o_bf16_cuda(
    torch::Tensor q, torch::Tensor k_ptrs, torch::Tensor v_ptrs,
    int64_t kv_heads, int64_t page_size, int64_t length)
{
    TORCH_CHECK(q.is_cuda() && q.is_contiguous(), "q must be contiguous CUDA");
    TORCH_CHECK(q.scalar_type()==torch::kBFloat16, "q must be BF16");
    TORCH_CHECK(q.dim()==4 && q.size(2)>=1 && q.size(2)<=4, "1<=q_len<=4");
    TORCH_CHECK(k_ptrs.is_cuda() && v_ptrs.is_cuda(), "pointer tables CUDA");
    TORCH_CHECK(k_ptrs.scalar_type()==torch::kInt64 &&
                v_ptrs.scalar_type()==torch::kInt64, "pointer tables int64");
    const int B=q.size(0), H=q.size(1), S=q.size(2), D=q.size(3);
    const int KV=static_cast<int>(kv_heads);
    TORCH_CHECK(KV>0 && H%KV==0, "invalid GQA");
    TORCH_CHECK(D==64 || D==128, "D must be 64/128");
    TORCH_CHECK(length>=S && page_size>0, "invalid length/page size");

    auto out=torch::empty_like(q);
    auto mass=torch::zeros({S,length},q.options().dtype(torch::kFloat32));
    const auto* qp=reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
    auto* op=reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    const auto* kp=k_ptrs.data_ptr<int64_t>();
    const auto* vp=v_ptrs.data_ptr<int64_t>();
    if (D==64) launch_bf16<64>(qp,kp,vp,op,mass.data_ptr<float>(),B,H,KV,S,page_size,length);
    else launch_bf16<128>(qp,kp,vp,op,mass.data_ptr<float>(),B,H,KV,S,page_size,length);
    auto err=cudaGetLastError();
    TORCH_CHECK(err==cudaSuccess,cudaGetErrorString(err));
    return {out,mass};
}

std::vector<torch::Tensor> phase12_paged_attention_h2o_vfp8_cuda(
    torch::Tensor q, torch::Tensor k_ptrs, torch::Tensor v_ptrs,
    torch::Tensor s_ptrs, int64_t kv_heads, int64_t page_size,
    int64_t length, int64_t scale_groups)
{
    TORCH_CHECK(q.is_cuda() && q.is_contiguous(), "q must be contiguous CUDA");
    TORCH_CHECK(q.scalar_type()==torch::kBFloat16, "q must be BF16");
    TORCH_CHECK(q.dim()==4 && q.size(2)>=1 && q.size(2)<=4, "1<=q_len<=4");
    const int B=q.size(0), H=q.size(1), S=q.size(2), D=q.size(3);
    const int KV=static_cast<int>(kv_heads);
    TORCH_CHECK(KV>0 && H%KV==0, "invalid GQA");
    TORCH_CHECK(D==64 || D==128, "D must be 64/128");

    auto out=torch::empty_like(q);
    auto mass=torch::zeros({S,length},q.options().dtype(torch::kFloat32));
    const auto* qp=reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
    auto* op=reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    const auto* kp=k_ptrs.data_ptr<int64_t>();
    const auto* vp=v_ptrs.data_ptr<int64_t>();
    const auto* sp=s_ptrs.data_ptr<int64_t>();
    if (D==64) launch_vfp8<64>(qp,kp,vp,sp,op,mass.data_ptr<float>(),B,H,KV,S,page_size,length,scale_groups);
    else launch_vfp8<128>(qp,kp,vp,sp,op,mass.data_ptr<float>(),B,H,KV,S,page_size,length,scale_groups);
    auto err=cudaGetLastError();
    TORCH_CHECK(err==cudaSuccess,cudaGetErrorString(err));
    return {out,mass};
}

// ---------------- page-native compaction ----------------

__global__ void p12_compact_bf16(
    const int64_t* sk,const int64_t* sv,
    const int64_t* dk,const int64_t* dv,
    const int64_t* idx,int B,int KV,int D,int page_size,int n)
{
    long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long total=(long long)B*KV*n*D;
    if(z>=total)return;
    const int d=z%D; z/=D;
    const int t=z%n; z/=n;
    const int kv=z%KV; z/=KV;
    const int b=(int)z;
    const int st=(int)idx[t], sp=st/page_size, ss=st%page_size;
    const int dp=t/page_size, ds=t%page_size;
    const auto* skp=reinterpret_cast<const __nv_bfloat16*>(sk[sp]);
    const auto* svp=reinterpret_cast<const __nv_bfloat16*>(sv[sp]);
    auto* dkp=reinterpret_cast<__nv_bfloat16*>(dk[dp]);
    auto* dvp=reinterpret_cast<__nv_bfloat16*>(dv[dp]);
    const long long so=(((long long)b*KV+kv)*page_size+ss)*D+d;
    const long long oo=(((long long)b*KV+kv)*page_size+ds)*D+d;
    dkp[oo]=skp[so]; dvp[oo]=svp[so];
}

__global__ void p12_compact_vfp8(
    const int64_t* sk,const int64_t* sv,
    const int64_t* dk,const int64_t* dv,
    const int64_t* idx,int B,int KV,int D,int page_size,int n)
{
    long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long total=(long long)B*KV*n*D;
    if(z>=total)return;
    const int d=z%D; z/=D;
    const int t=z%n; z/=n;
    const int kv=z%KV; z/=KV;
    const int b=(int)z;
    const int st=(int)idx[t], sp=st/page_size, ss=st%page_size;
    const int dp=t/page_size, ds=t%page_size;
    const auto* skp=reinterpret_cast<const __nv_bfloat16*>(sk[sp]);
    const auto* svp=reinterpret_cast<const uint8_t*>(sv[sp]);
    auto* dkp=reinterpret_cast<__nv_bfloat16*>(dk[dp]);
    auto* dvp=reinterpret_cast<uint8_t*>(dv[dp]);
    const long long so=(((long long)b*KV+kv)*page_size+ss)*D+d;
    const long long oo=(((long long)b*KV+kv)*page_size+ds)*D+d;
    dkp[oo]=skp[so]; dvp[oo]=svp[so];
}

__global__ void p12_compact_scale(
    const int64_t* ss,const int64_t* ds,const int64_t* idx,
    int B,int KV,int G,int page_size,int n)
{
    long long z=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long total=(long long)B*KV*n*G;
    if(z>=total)return;
    const int g=z%G; z/=G;
    const int t=z%n; z/=n;
    const int kv=z%KV; z/=KV;
    const int b=(int)z;
    const int st=(int)idx[t], sp=st/page_size, sl=st%page_size;
    const int dp=t/page_size, dl=t%page_size;
    const auto* src=reinterpret_cast<const float*>(ss[sp]);
    auto* dst=reinterpret_cast<float*>(ds[dp]);
    const long long so=(((long long)b*KV+kv)*page_size+sl)*G+g;
    const long long oo=(((long long)b*KV+kv)*page_size+dl)*G+g;
    dst[oo]=src[so];
}

void phase12_paged_compact_bf16_cuda(
    torch::Tensor sk,torch::Tensor sv,torch::Tensor dk,torch::Tensor dv,
    torch::Tensor indices,int64_t B,int64_t KV,int64_t D,int64_t page_size)
{
    TORCH_CHECK(indices.is_cuda() && indices.scalar_type()==torch::kInt64,
                "indices CUDA int64");
    const int n=indices.numel();
    const long long total=B*KV*n*D;
    constexpr int TH=256;
    const int blocks=(total+TH-1)/TH;
    p12_compact_bf16<<<blocks,TH>>>(
        sk.data_ptr<int64_t>(),sv.data_ptr<int64_t>(),
        dk.data_ptr<int64_t>(),dv.data_ptr<int64_t>(),
        indices.data_ptr<int64_t>(),B,KV,D,page_size,n);
    auto err=cudaGetLastError();
    TORCH_CHECK(err==cudaSuccess,cudaGetErrorString(err));
}

void phase12_paged_compact_vfp8_cuda(
    torch::Tensor sk,torch::Tensor sv,torch::Tensor ss,
    torch::Tensor dk,torch::Tensor dv,torch::Tensor ds,
    torch::Tensor indices,int64_t B,int64_t KV,int64_t D,
    int64_t G,int64_t page_size)
{
    TORCH_CHECK(indices.is_cuda() && indices.scalar_type()==torch::kInt64,
                "indices CUDA int64");
    const int n=indices.numel();
    constexpr int TH=256;
    const long long total=B*KV*n*D;
    const int blocks=(total+TH-1)/TH;
    p12_compact_vfp8<<<blocks,TH>>>(
        sk.data_ptr<int64_t>(),sv.data_ptr<int64_t>(),
        dk.data_ptr<int64_t>(),dv.data_ptr<int64_t>(),
        indices.data_ptr<int64_t>(),B,KV,D,page_size,n);

    const long long stotal=B*KV*n*G;
    const int sblocks=(stotal+TH-1)/TH;
    p12_compact_scale<<<sblocks,TH>>>(
        ss.data_ptr<int64_t>(),ds.data_ptr<int64_t>(),
        indices.data_ptr<int64_t>(),B,KV,G,page_size,n);
    auto err=cudaGetLastError();
    TORCH_CHECK(err==cudaSuccess,cudaGetErrorString(err));
}
