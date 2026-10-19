#include <torch/extension.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

template <typename scalar_t>
__global__ void leaky_relu_add_kernel(
    const scalar_t* input,
    const scalar_t* identity,
    scalar_t* output,
    int64_t elements,
    double negative_slope) {
    using compute_t = at::opmath_type<scalar_t>;
    const auto slope = static_cast<compute_t>(negative_slope);
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < elements; i += stride) {
        const auto value = static_cast<compute_t>(input[i]);
        const scalar_t activated = static_cast<scalar_t>(value > 0 ? value : value * slope);
        output[i] = static_cast<scalar_t>(
            static_cast<compute_t>(activated) + static_cast<compute_t>(identity[i]));
    }
}

torch::Tensor leaky_relu_add_cuda(
    const torch::Tensor& input,
    const torch::Tensor& identity,
    double negative_slope) {
    TORCH_CHECK(input.is_cuda() && identity.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(input.device() == identity.device(), "Inputs must have the same CUDA device");
    TORCH_CHECK(input.scalar_type() == identity.scalar_type(), "Input dtypes must match");
    TORCH_CHECK(input.sizes() == identity.sizes(), "Input shapes must match");
    TORCH_CHECK(input.strides() == identity.strides(), "Input strides must match");
    const bool contiguous = input.is_contiguous();
    const bool channels_last = input.dim() == 4 && input.is_contiguous(at::MemoryFormat::ChannelsLast);
    TORCH_CHECK(contiguous || channels_last,
                "Inputs must be default-contiguous or dense 4D channels_last tensors");
    const c10::cuda::CUDAGuard guard(input.device());
    auto output = torch::empty_strided(input.sizes(), input.strides(), input.options());
    const int64_t elements = input.numel();
    if (elements == 0) {
        return output;
    }
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<int64_t>((elements + threads - 1) / threads, 4096));
    const auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
        "quad_leaky_relu_add", [&] {
            leaky_relu_add_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                input.data_ptr<scalar_t>(), identity.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(), elements, negative_slope);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
