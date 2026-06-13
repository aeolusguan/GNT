#include <torch/extension.h>

torch::Tensor projection_distance_cuda(
    torch::Tensor poses,
    torch::Tensor depths,
    torch::Tensor scales,
    torch::Tensor masks,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj,
    int64_t stride);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("projection_distance", &projection_distance_cuda, "geometry projection distance CUDA reduction");
}
