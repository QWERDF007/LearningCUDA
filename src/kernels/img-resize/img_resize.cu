#include "common.cuh"

__global__ void resize_f32_kernel(flaot *src, float *dst, const float fx, const float fy, const int srcH,
                                  const int srcW, const int dstH, const int dstW, const int dstN)
{
    const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_x = idx % dstW;
    const int dst_y = idx / dstW;
    if (dst_idx >= dst_N) // 越界
        return;
    // +0.5: 像素通常认为是一个 小方格，而不是一个点
    // (dx, dy) 是像素的左上角坐标，(dx + 0.5, dy + 0.5) 是像素的中心坐标
    const float src_x = (dst_x + 0.5) * fx - 0.5;
    const float src_y = (dst_y + 0.5) * fy - 0.5;
}

#define TORCH_BINDING_RESIZE(tag, th_type, element_type, n_pack)                                                       \
    torch::Tensor resize_##tag(torch::Tensor src, const int dstH, const int dstW)                                      \
    {                                                                                                                  \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                       \
        auto          options = torch::TensorOptions().dtype(src.dtype()).device(torch::kCUDA, 0);                     \
        torch::Tensor dst     = torch::zeros({dstH, dstW}, options);                                                   \
        const int     srcH    = src.size(0);                                                                           \
        const int     srcW    = src.size(1);                                                                           \
        const float   fx      = (float)dstW / srcW;                                                                    \
        const float   fy      = (float)dstH / srcH;                                                                    \
        dim3          block(THREADS);                                                                                  \
        dim3          grid();                                                                                          \
        resize_##tag##_kernel<<<grid, block>>>(reinterpret_cast<element_type *>(src.data_ptr()),                       \
                                               reinterpret_cast<element_type *>(dst.data_ptr()), 1 / fx, 1 / fy, srcH, \
                                               srcW, dstH, dstW);                                                      \
    }