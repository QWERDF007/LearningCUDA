#include "common.cuh"

template<typename T, typename CT, typename KT, int CH>
__global__ void sobel_kernel(const T *__restrict__ src, CT *__restrict__ dst, const KT *__restrict__ kernel,
                             const int ksh, const int ksw, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    const int y = tid / W;
    const int x = tid % W;

    const int half_w = ksw / 2;
    const int half_h = ksh / 2;

    const int base = tid * CH;

    // 处理每个通道
#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        float sum = 0;
        // 卷积计算
        for (int ky = -half_h; ky <= half_h; ++ky)
        {
            int yy = border_reflect_101(y + ky, H);
            for (int kx = -half_w; kx <= half_w; ++kx)
            {
                int xx   = border_reflect_101(x + kx, W);
                KT  kval = kernel[(ky + half_h) * ksw + (kx + half_w)];
                sum += src[(yy * W + xx) * CH + c] * kval;
            }
        }
        // dst[base + c] = saturate_cast<CT>(sum); // cv2.Sobel(img)
        dst[base + c] = saturate_cast<uint8_t>(fabsf(sum)); // 等价于 cv2.convertScaleAbs(cv2.Sobel(img))
    }
}

#define TORCH_BINDING_SOBEL(torch_type, element_type, cal_type, kernel_type, n_pack)                                   \
    void sobel_##element_type##_##cal_type##_##kernel_type(torch::Tensor src, torch::Tensor dst, torch::Tensor kernel, \
                                                           const int ksh, const int ksw)                               \
    {                                                                                                                  \
        CHECK_TORCH_TENSOR_DTYPE(src, torch_type)                                                                      \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                                 \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                                 \
        CHECK_TORCH_TENSOR_DEVICE(kernel)                                                                              \
                                                                                                                       \
        const int H  = src.size(0);                                                                                    \
        const int W  = dst.size(1);                                                                                    \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                               \
        const int N  = H * W;                                                                                          \
        dim3      block(THREADS);                                                                                      \
        dim3      grid(divUp(N, THREADS));                                                                             \
                                                                                                                       \
        sobel_kernel<element_type, cal_type, kernel_type, 1><<<grid, block>>>(                                         \
            reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<cal_type *>(dst.data_ptr()),            \
            reinterpret_cast<kernel_type *>(kernel.data_ptr()), ksh, ksw, H, W, N);                                    \
    }

TORCH_BINDING_SOBEL(torch::kUInt8, uint8_t, int16_t, int8_t, 1)
TORCH_BINDING_SOBEL(torch::kUInt8, uint8_t, int16_t, float, 1)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(sobel_uint8_t_int16_t_int8_t)
    TORCH_BINDING_COMMON_EXTENSION(sobel_uint8_t_int16_t_float)
}