#include <torch/extension.h>

// Forward declaration
torch::Tensor fast_short_seq_attn(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool is_causal,
    double sm_scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_short_seq_attn", &fast_short_seq_attn,
          "Fast attention for short sequences with half2 optimization");
}
