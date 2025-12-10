#include "common.cuh"

template<typename T, typename CT, typename KT, int CH>
__global__ void filter_2D_ScaleAbs_kernel(const T *__restrict__ src, CT *__restrict__ dst,
                                          const KT *__restrict__ kernel, const int ksh, const int ksw, const int H,
                                          const int W, const int N)
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
        dst[base + c] = saturate_cast<uint8_t>(fabsf(sum)); // 等价于 cv2.convertScaleAbs(cv2.Sobel(img))
    }
}

template<typename T, typename CT, typename KT, int CH>
__global__ void filter_2D_kernel(const T *__restrict__ src, CT *__restrict__ dst, const KT *__restrict__ kernel,
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
        dst[base + c] = saturate_cast<CT>(sum); // cv2.Sobel(img)
    }
}

template<typename T, typename CT, typename WT, int CH>
__global__ void gaussian_blur_kernel(const T *src, T *dst, const WT *weights, const int ks_h, const int ks_w,
                                     const int img_h, const int img_w, const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const int x = idx % img_w;
    const int y = idx / img_w;

    const int radius_h = ks_h / 2;
    const int radius_w = ks_w / 2;

    const int base = idx * CH;

// 处理每个通道
#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        CT sum = 0;

        // 遍历卷积核窗口
        for (int ky = -radius_h; ky <= radius_h; ++ky)
        {
            const int yy = border_reflect_101(y + ky, img_h);
            for (int kx = -radius_w; kx <= radius_w; ++kx)
            {
                const int xx = border_reflect_101(x + kx, img_w);

                // 使用2D高斯权重
                const int weight_y = ky + radius_h;
                const int weight_x = kx + radius_w;
                const WT  weight   = weights[weight_y * ks_w + weight_x];
                sum += weight * src[(yy * img_w + xx) * CH + c];
            }
        }

        dst[base + c] = saturate_cast<T>(sum);
    }
}

template<typename T, int CH>
__global__ void copyTo_kernel(const T *src, T *dst, const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        dst[idx + c] = src[idx + c];
    }
}

#define TORCH_BINDING_FILTER(tag, torch_type, element_type, cal_type, kernel_type, n_pack)                             \
    void tag##_##element_type##_##cal_type##_##kernel_type(torch::Tensor src, torch::Tensor dst, torch::Tensor kernel, \
                                                           const int ksh, const int ksw)                               \
    {                                                                                                                  \
        CHECK_TORCH_TENSOR_DTYPE(src, torch_type)                                                                      \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                                 \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                                 \
        CHECK_TORCH_TENSOR_DEVICE(kernel)                                                                              \
                                                                                                                       \
        const int H  = src.size(0);                                                                                    \
        const int W  = src.size(1);                                                                                    \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                               \
        const int N  = H * W;                                                                                          \
        dim3      block(THREADS);                                                                                      \
        dim3      grid(divUp(N, THREADS));                                                                             \
                                                                                                                       \
        filter_2D_ScaleAbs_kernel<element_type, cal_type, kernel_type, 1><<<grid, block>>>(                            \
            reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<cal_type *>(dst.data_ptr()),            \
            reinterpret_cast<kernel_type *>(kernel.data_ptr()), ksh, ksw, H, W, N);                                    \
    }

#define TORCH_BINDING_GAUSSIAN_LAPLACIAN(tag, th_type, element_type, cal_type, kernel_type, n_pack)                 \
    void tag##_##element_type##_##cal_type##_##kernel_type(torch::Tensor src, torch::Tensor dst, torch::Tensor tmp, \
                                                           torch::Tensor gaussian_kernel, torch::Tensor kernel,     \
                                                           const int ksh, const int ksw)                            \
    {                                                                                                               \
        CHECK_TORCH_TENSOR_DTYPE(src, th_type)                                                                      \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                              \
                                                                                                                    \
        const int H  = src.size(0);                                                                                 \
        const int W  = src.size(1);                                                                                 \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                            \
        const int N  = H * W;                                                                                       \
        dim3      block(THREADS);                                                                                   \
        dim3      grid(divUp(N, THREADS));                                                                          \
        if (ksh == 1 && ksw == 1)                                                                                   \
        {                                                                                                           \
            copyTo_kernel<element_type, 1><<<grid, block>>>(reinterpret_cast<const element_type *>(src.data_ptr()), \
                                                            reinterpret_cast<element_type *>(tmp.data_ptr()), N);   \
            filter_2D_ScaleAbs_kernel<element_type, cal_type, kernel_type, 1><<<grid, block>>>(                     \
                reinterpret_cast<element_type *>(tmp.data_ptr()), reinterpret_cast<cal_type *>(dst.data_ptr()),     \
                reinterpret_cast<kernel_type *>(kernel.data_ptr()), 3, 3, H, W, N);                                 \
        }                                                                                                           \
        else                                                                                                        \
        {                                                                                                           \
            gaussian_blur_kernel<element_type, float, float, 1>                                                     \
                <<<grid, block>>>(reinterpret_cast<const element_type *>(src.data_ptr()),                           \
                                  reinterpret_cast<element_type *>(tmp.data_ptr()),                                 \
                                  reinterpret_cast<const float *>(gaussian_kernel.data_ptr()), ksh, ksw, H, W, N);  \
            filter_2D_ScaleAbs_kernel<element_type, cal_type, kernel_type, 1><<<grid, block>>>(                     \
                reinterpret_cast<element_type *>(tmp.data_ptr()), reinterpret_cast<cal_type *>(dst.data_ptr()),     \
                reinterpret_cast<kernel_type *>(kernel.data_ptr()), ksh, ksw, H, W, N);                             \
        }                                                                                                           \
    }

TORCH_BINDING_FILTER(sobel, torch::kUInt8, uint8_t, int16_t, int8_t, 1)
TORCH_BINDING_FILTER(sobel, torch::kUInt8, uint8_t, uint8_t, int8_t, 1)
TORCH_BINDING_FILTER(scharr, torch::kUInt8, uint8_t, int16_t, int8_t, 1)
TORCH_BINDING_FILTER(scharr, torch::kUInt8, uint8_t, uint8_t, int8_t, 1)
TORCH_BINDING_FILTER(laplacian, torch::kUInt8, uint8_t, int16_t, int8_t, 1)
TORCH_BINDING_FILTER(laplacian, torch::kUInt8, uint8_t, uint8_t, int8_t, 1)
TORCH_BINDING_GAUSSIAN_LAPLACIAN(gaussian_laplacian, torch::kUInt8, uint8_t, uint8_t, int8_t, 1)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(sobel_uint8_t_int16_t_int8_t)
    TORCH_BINDING_COMMON_EXTENSION(sobel_uint8_t_uint8_t_int8_t)
    TORCH_BINDING_COMMON_EXTENSION(scharr_uint8_t_int16_t_int8_t)
    TORCH_BINDING_COMMON_EXTENSION(scharr_uint8_t_uint8_t_int8_t)
    TORCH_BINDING_COMMON_EXTENSION(laplacian_uint8_t_int16_t_int8_t)
    TORCH_BINDING_COMMON_EXTENSION(laplacian_uint8_t_uint8_t_int8_t)
    TORCH_BINDING_COMMON_EXTENSION(gaussian_laplacian_uint8_t_uint8_t_int8_t)
}