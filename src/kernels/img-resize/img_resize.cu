#include "common.cuh"

__global__ void resize_f32_kernel(float *src, float *dst, const double scale_x, const double scale_y, const int src_h,
                                  const int src_w, const int dst_h, const int dst_w, const int dst_N)
{
    const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_x = idx % dst_w;
    const int dst_y = idx / dst_w;
    if (idx >= dst_N) // 越界
        return;
    // +0.5: 像素通常认为是一个 小方格，而不是一个点
    // (dx, dy) 是像素的左上角坐标，(dx + 0.5, dy + 0.5) 是像素的中心坐标
    // -0.5: 保证源图和目标图的像素中心对齐
    const float src_x = (dst_x + 0.5) * scale_x - 0.5;
    const float src_y = (dst_y + 0.5) * scale_y - 0.5;
}

#define TORCH_BINDING_RESIZE(tag, th_type, element_type, n_pack)                                                   \
    torch::Tensor img_resize_##tag(torch::Tensor src, const int dst_h, const int dst_w)                            \
    {                                                                                                              \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                   \
        auto          options     = torch::TensorOptions().dtype(src.dtype()).device(torch::kCUDA, 0);             \
        torch::Tensor dst         = torch::zeros({dst_h, dst_w}, options);                                         \
        const int     N           = dst_h * dst_w;                                                                 \
        const int     src_h       = src.size(0);                                                                   \
        const int     src_w       = src.size(1);                                                                   \
        const double  inv_scale_x = (double)dst_w / src_w;                                                         \
        const double  inv_scale_y = (double)dst_h / src_h;                                                         \
        const double  scale_x     = 1. / inv_scale_x;                                                              \
        const double  scale_y     = 1. / inv_scale_y;                                                              \
        dim3          block(THREADS);                                                                              \
        dim3          grid(divUp(N, THREADS));                                                                     \
        resize_##tag##_kernel<<<grid, block>>>(reinterpret_cast<element_type *>(src.data_ptr()),                   \
                                               reinterpret_cast<element_type *>(dst.data_ptr()), scale_x, scale_y, \
                                               src_h, src_w, dst_h, dst_w, N);                                     \
        return dst;                                                                                                \
    }

TORCH_BINDING_RESIZE(f32, torch::kFloat32, float, 1)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(img_resize_f32)
}