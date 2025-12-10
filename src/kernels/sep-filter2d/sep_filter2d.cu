#include "common.cuh"

template<typename T, typename CT, typename KT, int CH>
__global__ void filter_h_kernel(const T *__restrict__ src, CT *__restrict__ dst, const KT *__restrict__ kernel,
                                const int ksize, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    const int y = tid / W;
    const int x = tid % W;

    const int anchor = ksize / 2;

    const int base = tid * CH;

    // 处理每个通道
#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        float sum = 0;
        // 卷积计算
        for (int k = 0; k < ksize; ++k)
        {
            int xx = border_reflect_101(x + k - anchor, W);
            sum += src[(y * W + xx) * CH + c] * kernel[k];
        }
        dst[base + c] = saturate_cast<CT>(sum); // cv2.Sobel(img)
    }
}

template<typename T, typename CT, typename KT, int CH>
__global__ void filter_v_kernel(const T *__restrict__ src, CT *__restrict__ dst, const KT *__restrict__ kernel,
                                const int ksize, const int H, const int W, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    const int y = tid / W;
    const int x = tid % W;

    const int anchor = ksize / 2;

    const int base = tid * CH;

    // 处理每个通道
#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        float sum = 0;
        // 卷积计算
        for (int k = 0; k < ksize; ++k)
        {
            int yy = border_reflect_101(y + k - anchor, H);
            sum += src[(yy * W + x) * CH + c] * kernel[k];
        }
        dst[base + c] = saturate_cast<CT>(sum); // cv2.Sobel(img)
    }
}

template<typename T, typename CT>
__global__ void elementwise_add_kernel(T *a, T *b, T *c, const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    CT v   = a[idx] + b[idx];
    c[idx] = saturate_cast<T>(v);
}

#define TORCH_BINDING_FILTER(torch_type, element_type, cal_type, kernel_type, n_pack)                                  \
    void sep_filter2D_##element_type##_##cal_type##_##kernel_type(torch::Tensor src, torch::Tensor dst,                \
                                                                  torch::Tensor tmp, torch::Tensor kernelx,            \
                                                                  torch::Tensor kernely, const int ksw, const int ksh) \
    {                                                                                                                  \
        CHECK_TORCH_TENSOR_DTYPE(src, torch_type)                                                                      \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                                 \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                                 \
        CHECK_TORCH_TENSOR_DEVICE(tmp)                                                                                 \
        CHECK_TORCH_TENSOR_DEVICE(kernelx)                                                                             \
        CHECK_TORCH_TENSOR_DEVICE(kernely)                                                                             \
                                                                                                                       \
        const int H  = src.size(0);                                                                                    \
        const int W  = src.size(1);                                                                                    \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                               \
        const int N  = H * W;                                                                                          \
        dim3      block(THREADS);                                                                                      \
        dim3      grid(divUp(N, THREADS));                                                                             \
        filter_h_kernel<element_type, cal_type, kernel_type, 1><<<grid, block>>>(                                      \
            reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<cal_type *>(tmp.data_ptr()),            \
            reinterpret_cast<kernel_type *>(kernelx.data_ptr()), ksw, H, W, N);                                        \
        filter_v_kernel<cal_type, cal_type, kernel_type, 1><<<grid, block>>>(                                          \
            reinterpret_cast<cal_type *>(tmp.data_ptr()), reinterpret_cast<cal_type *>(dst.data_ptr()),                \
            reinterpret_cast<kernel_type *>(kernely.data_ptr()), ksh, H, W, N);                                        \
    }

#define TORCH_BINDING_FILTER_LAPLACIAN(torch_type, element_type, cal_type, dst_type, kernel_type, n_pack)             \
    void laplacian_##element_type##_##cal_type##_##dst_type##_##kernel_type(                                          \
        torch::Tensor src, torch::Tensor dst, torch::Tensor tmp0, torch::Tensor tmp1, torch::Tensor tmp2,             \
        torch::Tensor kernelx, torch::Tensor kernely, const int ksw, const int ksh)                                   \
    {                                                                                                                 \
        CHECK_TORCH_TENSOR_DTYPE(src, torch_type)                                                                     \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                                \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                                \
        CHECK_TORCH_TENSOR_DEVICE(tmp0)                                                                               \
        CHECK_TORCH_TENSOR_DEVICE(tmp1)                                                                               \
        CHECK_TORCH_TENSOR_DEVICE(tmp2)                                                                               \
        CHECK_TORCH_TENSOR_DEVICE(kernelx)                                                                            \
        CHECK_TORCH_TENSOR_DEVICE(kernely)                                                                            \
                                                                                                                      \
        const int H  = src.size(0);                                                                                   \
        const int W  = src.size(1);                                                                                   \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                              \
        const int N  = H * W;                                                                                         \
        dim3      block(THREADS);                                                                                     \
        dim3      grid(divUp(N, THREADS));                                                                            \
        filter_h_kernel<element_type, cal_type, kernel_type, 1><<<grid, block>>>(                                     \
            reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<cal_type *>(tmp1.data_ptr()),          \
            reinterpret_cast<kernel_type *>(kernelx.data_ptr()), ksw, H, W, N);                                       \
        filter_v_kernel<cal_type, dst_type, kernel_type, 1><<<grid, block>>>(                                         \
            reinterpret_cast<cal_type *>(tmp1.data_ptr()), reinterpret_cast<dst_type *>(tmp0.data_ptr()),             \
            reinterpret_cast<kernel_type *>(kernely.data_ptr()), ksh, H, W, N);                                       \
        filter_h_kernel<element_type, cal_type, kernel_type, 1><<<grid, block>>>(                                     \
            reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<cal_type *>(tmp1.data_ptr()),          \
            reinterpret_cast<kernel_type *>(kernely.data_ptr()), ksh, H, W, N);                                       \
        filter_v_kernel<cal_type, dst_type, kernel_type, 1><<<grid, block>>>(                                         \
            reinterpret_cast<cal_type *>(tmp1.data_ptr()), reinterpret_cast<dst_type *>(dst.data_ptr()),              \
            reinterpret_cast<kernel_type *>(kernelx.data_ptr()), ksw, H, W, N);                                       \
        elementwise_add_kernel<dst_type, cal_type><<<grid, block>>>(reinterpret_cast<dst_type *>(tmp0.data_ptr()),    \
                                                                    reinterpret_cast<dst_type *>(dst.data_ptr()),     \
                                                                    reinterpret_cast<dst_type *>(dst.data_ptr()), N); \
    }

TORCH_BINDING_FILTER(torch::kUInt8, uint8_t, int16_t, int8_t, 1)
TORCH_BINDING_FILTER(torch::kUInt8, uint8_t, uint8_t, int8_t, 1)
TORCH_BINDING_FILTER_LAPLACIAN(torch::kUInt8, uint8_t, int16_t, int16_t, int8_t, 1)
TORCH_BINDING_FILTER_LAPLACIAN(torch::kUInt8, uint8_t, int32_t, int16_t, int8_t, 1)
TORCH_BINDING_FILTER_LAPLACIAN(torch::kUInt8, uint8_t, float, int16_t, int8_t, 1)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(sep_filter2D_uint8_t_int16_t_int8_t)
    TORCH_BINDING_COMMON_EXTENSION(sep_filter2D_uint8_t_uint8_t_int8_t)
    TORCH_BINDING_COMMON_EXTENSION(laplacian_uint8_t_int16_t_int16_t_int8_t)
    TORCH_BINDING_COMMON_EXTENSION(laplacian_uint8_t_int32_t_int16_t_int8_t)
    TORCH_BINDING_COMMON_EXTENSION(laplacian_uint8_t_float_int16_t_int8_t)
}
