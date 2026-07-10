#include <torch/extension.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <cmath>

// Forward declaration
at::Tensor sm75_custom_short_sdpa(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    bool is_causal,
    float sm_scale);

at::Tensor sm75_fast_sdpa(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    bool is_causal,
    double sm_scale)
{
    // Get sequence lengths and head counts
    int64_t seq_q = q.size(2);
    int64_t seq_kv = k.size(2);
    int64_t q_heads = q.size(1);
    int64_t kv_heads = k.size(1);
    int64_t dim = q.size(3);

    // Calculate scale if not provided
    double scale = sm_scale;
    if (scale <= 0.0) {
        scale = 1.0 / std::sqrt(static_cast<double>(dim));
    }

    float scale_f = static_cast<float>(scale);

    // Tier 1: Custom CUDA kernel for very short sequences (beats PyTorch)
    // Covers SD text encoding (seq=77) and similar
    // Only for MHA (no GQA) and supported head dims
    if (seq_q <= 128 && seq_kv <= 128 && q_heads == kv_heads && 
        (dim == 64 || dim == 128)) {
        at::Tensor result = sm75_custom_short_sdpa(q, k, v, is_causal, scale_f);
        if (result.defined() && result.numel() > 0) {
            return result;
        }
        // Fall through to PyTorch if custom kernel failed
    }

    // Tier 2: PyTorch SDPA for short sequences (C++ eliminates Python overhead)
    // Covers short self-attention and cross-attention
    if (seq_q < 256 && seq_kv < 512) {
        return at::scaled_dot_product_attention(
            q, k, v,
            ::std::optional<at::Tensor>(),  // no attention mask
            0.0,           // no dropout
            is_causal,
            ::std::optional<double>(scale)
        );
    }

    // Tier 3: Everything else - fall back to Python sageattn()
    // This handles GQA, long sequences, NHD layout, BF16, INT8 path, etc.
    py::module_ sage_module = py::module_::import("sageattention");
    py::object sageattn = sage_module.attr("sageattn");

    py::kwargs kwargs;
    kwargs["tensor_layout"] = std::string("HND");
    kwargs["is_causal"] = is_causal;
    kwargs["sm_scale"] = scale;
    kwargs["return_lse"] = false;

    return sageattn(q, k, v, **kwargs).cast<at::Tensor>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Fast SM75 dispatch - C++ tiered dispatch for minimal overhead";

    m.def("sm75_fast_sdpa", &sm75_fast_sdpa,
        "Fast SDPA for SM75 with tiered C++ dispatch",
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("is_causal") = false,
        py::arg("sm_scale") = 0.0);
}
