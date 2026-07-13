#include <torch/extension.h>

torch::Tensor sm75_short_sdpa_v2(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    bool is_causal, double sm_scale, std::string layout /* "HND" or "NHD" */
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SM75 short SDPA v2 - stride-aware kernel supporting NQ != NK";
    m.def("sm75_short_sdpa_v2", &sm75_short_sdpa_v2,
        "SM75 short SDPA v2 kernel",
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("is_causal") = false,
        py::arg("sm_scale") = 0.0,
        py::arg("layout") = "HND");
}