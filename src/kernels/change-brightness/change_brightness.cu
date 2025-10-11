#include "common.cuh"

template<typename T, typename CT, int CH>
__global__ void change_brightness_kernel(T *src, T *dst, const double alpha, const double beta, const int H,
                                         const int W, const int step, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    const int x = tid % W;
    const int y = tid / W;

    const int base = y * step + x * CH;

#pragma unroll
    for (int i = 0; i < CH; ++i)
    {
        const int idx = base + i;

        CT value = src[idx] * alpha + beta;
        if constexpr (std::is_same_v<CT, double>)
        {
            dst[idx] = saturate_cast<T>(fabs(value));
        }
        else if constexpr (std::is_same_v<CT, float>)
        {
            dst[idx] = saturate_cast<T>(fabsf(value));
        }
    }
}

#define TORCH_BINDING_CHANGE_BRIGHTNESS(tag, th_type, element_type, cal_type, n_pack)                                  \
    void tag##_##element_type##_##cal_type(torch::Tensor src, torch::Tensor dst, const double alpha,                   \
                                           const double beta)                                                          \
    {                                                                                                                  \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                       \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                       \
        const int H    = src.size(0);                                                                                  \
        const int W    = src.size(1);                                                                                  \
        const int CH   = src.dim() == 2 ? 1 : src.size(2);                                                             \
        const int N    = H * W;                                                                                        \
        const int step = W * CH;                                                                                       \
        dim3      block(THREADS);                                                                                      \
        dim3      grid(divUp(N, THREADS));                                                                             \
        if (CH == 1)                                                                                                   \
        {                                                                                                              \
            tag##_kernel<element_type, cal_type, 1><<<grid, block>>>(reinterpret_cast<element_type *>(src.data_ptr()), \
                                                                     reinterpret_cast<element_type *>(dst.data_ptr()), \
                                                                     alpha, beta, H, W, step, N);                      \
        }                                                                                                              \
        else if (CH == 3)                                                                                              \
        {                                                                                                              \
            tag##_kernel<element_type, cal_type, 3><<<grid, block>>>(reinterpret_cast<element_type *>(src.data_ptr()), \
                                                                     reinterpret_cast<element_type *>(dst.data_ptr()), \
                                                                     alpha, beta, H, W, step, N);                      \
        }                                                                                                              \
    }

TORCH_BINDING_CHANGE_BRIGHTNESS(change_brightness, torch::kUInt8, uint8_t, float, 1)
TORCH_BINDING_CHANGE_BRIGHTNESS(change_brightness, torch::kUInt8, uint8_t, double, 1)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(change_brightness_uint8_t_float)
    TORCH_BINDING_COMMON_EXTENSION(change_brightness_uint8_t_double)
}