#include <torch/extension.h>
#include <vector>

// CUDA C++ entry points.

torch::Tensor rmsnorm_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    double eps);

torch::Tensor rope_cuda(
    torch::Tensor x,
    torch::Tensor cos,
    torch::Tensor sin);

torch::Tensor swiglu_cuda(
    torch::Tensor gate,
    torch::Tensor up);

torch::Tensor attention_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v);

torch::Tensor paged_attention_decode_cuda(
    torch::Tensor q,
    torch::Tensor k_ptrs,
    torch::Tensor v_ptrs,
    int64_t kv_heads,
    int64_t page_size,
    int64_t length);

// FINAL_PAGED_V2_Q1_Q4_VFP8
torch::Tensor paged_attention_block_cuda(
    torch::Tensor q,
    torch::Tensor k_ptrs,
    torch::Tensor v_ptrs,
    int64_t kv_heads,
    int64_t page_size,
    int64_t length);

torch::Tensor paged_attention_vfp8_block_cuda(
    torch::Tensor q,
    torch::Tensor k_ptrs,
    torch::Tensor v_ptrs,
    torch::Tensor s_ptrs,
    int64_t kv_heads,
    int64_t page_size,
    int64_t length,
    int64_t scale_groups);


// PHASE12_H2O_OPT
std::vector<torch::Tensor> phase12_paged_attention_h2o_bf16_cuda(
    torch::Tensor q, torch::Tensor k_ptrs, torch::Tensor v_ptrs,
    int64_t kv_heads, int64_t page_size, int64_t length);

std::vector<torch::Tensor> phase12_paged_attention_h2o_vfp8_cuda(
    torch::Tensor q, torch::Tensor k_ptrs, torch::Tensor v_ptrs,
    torch::Tensor s_ptrs, int64_t kv_heads, int64_t page_size,
    int64_t length, int64_t scale_groups);

void phase12_paged_compact_bf16_cuda(
    torch::Tensor sk, torch::Tensor sv,
    torch::Tensor dk, torch::Tensor dv,
    torch::Tensor indices, int64_t batch,
    int64_t kv_heads, int64_t head_dim, int64_t page_size);

void phase12_paged_compact_vfp8_cuda(
    torch::Tensor sk, torch::Tensor sv, torch::Tensor ss,
    torch::Tensor dk, torch::Tensor dv, torch::Tensor ds,
    torch::Tensor indices, int64_t batch, int64_t kv_heads,
    int64_t head_dim, int64_t scale_groups, int64_t page_size);


// PHASE13_FLASH_PAGED
torch::Tensor phase13_flash_paged_bf16_cuda(
    torch::Tensor q,
    torch::Tensor k_ptrs,
    torch::Tensor v_ptrs,
    int64_t kv_heads,
    int64_t page_size,
    int64_t length);

torch::Tensor phase13_flash_paged_vfp8_cuda(
    torch::Tensor q,
    torch::Tensor k_ptrs,
    torch::Tensor v_ptrs,
    torch::Tensor s_ptrs,
    int64_t kv_heads,
    int64_t page_size,
    int64_t length,
    int64_t scale_groups);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "rmsnorm",
        &rmsnorm_cuda,
        "RMSNorm (fp32 reduction, dtype store)");

    m.def(
        "apply_rope",
        &rope_cuda,
        "rotate_half RoPE");

    m.def(
        "swiglu",
        &swiglu_cuda,
        "silu(gate) * up");

    m.def(
        "attention",
        &attention_cuda,
        "causal GQA online-softmax attention");

    m.def(
        "paged_attention_decode",
        &paged_attention_decode_cuda,
        "paged GQA decode attention, q_len=1");

    m.def(
        "paged_attention_block",
        &paged_attention_block_cuda,
        "paged GQA attention, q_len=1..4, BF16/FP32");
    m.def(
        "paged_attention_vfp8_block",
        &paged_attention_vfp8_block_cuda,
        "paged GQA attention, q_len=1..4, K BF16 + V FP8");
    m.def(
        "phase12_paged_attention_h2o_bf16",
        &phase12_paged_attention_h2o_bf16_cuda,
        "Phase12 paged BF16 attention + H2O score");
    m.def(
        "phase12_paged_attention_h2o_vfp8",
        &phase12_paged_attention_h2o_vfp8_cuda,
        "Phase12 paged V-FP8 attention + H2O score");
    m.def(
        "phase12_paged_compact_bf16",
        &phase12_paged_compact_bf16_cuda,
        "Phase12 page-native BF16 compaction");
    m.def(
        "phase12_paged_compact_vfp8",
        &phase12_paged_compact_vfp8_cuda,
        "Phase12 page-native V-FP8 compaction");

    m.def(
        "phase13_flash_paged_bf16",
        &phase13_flash_paged_bf16_cuda,
        "Phase13 page-native tiled BF16 attention");
    m.def(
        "phase13_flash_paged_vfp8",
        &phase13_flash_paged_vfp8_cuda,
        "Phase13 page-native tiled V-FP8 attention");

}
