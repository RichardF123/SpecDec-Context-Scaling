#pragma once

#include <cuda_bf16.h>
#include <type_traits>

__device__ __forceinline__ float load_as_float(const float* p) { return *p; }

__device__ __forceinline__ float load_as_float(const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}

__device__ __forceinline__ void store_from_float(float* p, float v) { *p = v; }

__device__ __forceinline__ void store_from_float(__nv_bfloat16* p, float v) {
    *p = __float2bfloat16(v);
}

// Match model_self.RMSNORM: the fp32 normalized value is rounded to the
// activation dtype before it is multiplied by the weight.
template <typename T>
__device__ __forceinline__ void store_rmsnorm(T* p, float normed, float weight) {
    if constexpr (std::is_same<T, float>::value) {
        *p = normed * weight;
    } else {
        float n = __bfloat162float(__float2bfloat16(normed));
        float w = weight;
        *p = __float2bfloat16(n * w);
    }
}
