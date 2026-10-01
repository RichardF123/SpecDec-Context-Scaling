#include <torch/extension.h>

// CUDA C++ entry points. The same sources build on Windows with MSVC + nvcc
// when the toolkit matches the installed PyTorch CUDA wheel. Triton is not
// part of this translation unit.

torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, double eps);
torch::Tensor rope_cuda(torch::Tensor x, torch::Tensor cos, torch::Tensor sin);
torch::Tensor swiglu_cuda(torch::Tensor gate, torch::Tensor up);
torch::Tensor attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm", &rmsnorm_cuda, "RMSNorm (fp32 reduction, dtype store)");
    m.def("apply_rope", &rope_cuda, "rotate_half RoPE");
    m.def("swiglu", &swiglu_cuda, "silu(gate) * up");
    m.def("attention", &attention_cuda, "causal GQA online-softmax attention");
}
