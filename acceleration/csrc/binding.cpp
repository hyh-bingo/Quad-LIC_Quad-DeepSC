#include <torch/extension.h>

torch::Tensor leaky_relu_add_cuda(
    const torch::Tensor& input,
    const torch::Tensor& identity,
    double negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.attr("version") = "0.1.2";
    module.attr("supports_channels_last") = true;
    module.def("leaky_relu_add", &leaky_relu_add_cuda);
}
