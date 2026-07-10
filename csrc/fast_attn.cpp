/*
 * Fast attention path for short sequences
 * Uses PyTorch's internal SDPA with minimal overhead
 * Designed to match or beat PyTorch's direct SDPA call
 */

#include <torch/extension.h>
#include <ATen/native/transformers/attention.h>

// Fast path that directly calls PyTorch's SDPA with minimal overhead
// This eliminates the Python wrapper overhead for short sequences
torch::Tensor fast_short_seq_attn(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool is_causal,
    double sm_scale)
{
    // Direct call to PyTorch's scaled_dot_product_attention
    // This bypasses all Python wrapper overhead
    return at::native::scaled_dot_product_attention(
        q, k, v,
        /*attn_mask=*/c10::nullopt,
        /*dropout_p=*/0.0,
        is_causal,
        /*scale=*/sm_scale
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_short_seq_attn", &fast_short_seq_attn, 
          "Fast attention for short sequences with minimal overhead");
}
