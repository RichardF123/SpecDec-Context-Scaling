#include <torch/extension.h>

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
}
