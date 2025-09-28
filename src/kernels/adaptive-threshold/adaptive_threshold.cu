#include "common.cuh"

template<typename T, typename CT, int CH>
__global__ void blur_kernel(T *src, T *dst, const int ks_h, const int ks_w, const int img_h, const int img_w,
                            const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const int x = idx % img_w;
    const int y = idx / img_w;

    const int    half_w = ks_w / 2;
    const int    half_h = ks_h / 2;
    const double count  = ks_h * ks_w;

    // const int base = y * img_w * CH + x * CH;
    const int base = idx * CH;

// 处理每个通道
#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        CT sum = 0;

        // 遍历卷积核窗口
        for (int ky = -half_h; ky <= half_h; ++ky)
        {
            int yy = border_replicate(y + ky, img_h);
            for (int kx = -half_w; kx <= half_w; ++kx)
            {
                int xx = border_replicate(x + kx, img_w);
                sum += src[(yy * img_w + xx) * CH + c];
            }
        }

        dst[base + c] = saturate_cast<T>(sum / count);
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
            const int yy = border_replicate(y + ky, img_h);
            for (int kx = -radius_w; kx <= radius_w; ++kx)
            {
                const int xx = border_replicate(x + kx, img_w);

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
__global__ void adaptive_threshold_binary_kernel(T *src, T *mean, T *dst, const T maxval, const T delta, const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    const int base = idx * CH;

#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        dst[base + c] = src[base + c] > mean[base + c] - delta ? maxval : 0;
    }
}

template<typename T, int CH>
__global__ void adaptive_threshold_binary_inv_kernel(T *src, T *mean, T *dst, const T maxval, const T delta,
                                                     const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    const int base = idx * CH;

#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        dst[base + c] = src[base + c] > mean[base + c] - delta ? 0 : maxval;
    }
}

template<typename T, typename CT, int CH>
__global__ void sum_kernel(T *src, CT *dst, const int ks_h, const int ks_w, const int img_h, const int img_w,
                           const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const int x = idx % img_w;
    const int y = idx / img_w;

    const int half_w = ks_w / 2;
    const int half_h = ks_h / 2;

    // const int base = y * img_w * CH + x * CH;
    const int base = idx * CH;

// 处理每个通道
#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        CT sum = 0;

        // 遍历卷积核窗口
        for (int ky = -half_h; ky <= half_h; ++ky)
        {
            int yy = border_replicate(y + ky, img_h);
            for (int kx = -half_w; kx <= half_w; ++kx)
            {
                int xx = border_replicate(x + kx, img_w);
                sum += src[(yy * img_w + xx) * CH + c];
            }
        }

        dst[base + c] = sum;
    }
}

template<typename T, typename CT, int CH>
__global__ void adaptive_threshold_binary_percentage_kernel(T *src, CT *sum, T *dst, const T maxval, const int ws,
                                                            const double percentage, const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    const int base = idx * CH;

#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        dst[base + c] = src[base + c] * ws < sum[base + c] * percentage ? 0 : maxval;
    }
}

template<typename T, typename CT, int CH>
__global__ void adaptive_threshold_binary_inv_percentage_kernel(T *src, CT *sum, T *dst, const T maxval, const int ws,
                                                                const double percentage, const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    const int base = idx * CH;

#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        dst[base + c] = src[base + c] * ws < sum[base + c] * percentage ? maxval : 0;
    }
}

#define TORCH_BINDING_ADAPTIVE_THRESHOLD_MEAN(tag, th_type, element_type, cal_type, n_pack)                            \
    torch::Tensor tag##_##element_type##_##cal_type##_mean(torch::Tensor src, torch::Tensor mean, torch::Tensor dst,   \
                                                           const double maxval, const int ksz, const double delta)     \
    {                                                                                                                  \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                       \
        CHECK_TORCH_TENSOR_DTYPE(mean, (th_type))                                                                      \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                       \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                                 \
        CHECK_TORCH_TENSOR_DEVICE(mean)                                                                                \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                                 \
        CHECK_TORCH_TENSOR_SAME_SIZE(src, mean)                                                                        \
        CHECK_TORCH_TENSOR_SAME_SIZE(src, dst)                                                                         \
        const int H      = src.size(0);                                                                                \
        const int W      = src.size(1);                                                                                \
        const int CH     = src.dim() == 2 ? 1 : src.size(2);                                                           \
        const int N      = H * W;                                                                                      \
        const int idelta = strstr(#tag, "inv") ? floor(delta) : ceil(delta);                                           \
        dim3      block(THREADS);                                                                                      \
        dim3      grid(divUp(N, THREADS));                                                                             \
        if (CH == 1)                                                                                                   \
        {                                                                                                              \
            blur_kernel<element_type, cal_type, 1><<<grid, block>>>(reinterpret_cast<element_type *>(src.data_ptr()),  \
                                                                    reinterpret_cast<element_type *>(mean.data_ptr()), \
                                                                    ksz, ksz, H, W, N);                                \
            tag##_kernel<element_type, 1><<<grid, block>>>(                                                            \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(mean.data_ptr()),   \
                reinterpret_cast<element_type *>(dst.data_ptr()), static_cast<element_type>(maxval),                   \
                static_cast<element_type>(idelta), N);                                                                 \
        }                                                                                                              \
        else if (CH == 3)                                                                                              \
        {                                                                                                              \
            blur_kernel<element_type, cal_type, 3><<<grid, block>>>(reinterpret_cast<element_type *>(src.data_ptr()),  \
                                                                    reinterpret_cast<element_type *>(mean.data_ptr()), \
                                                                    ksz, ksz, H, W, N);                                \
            tag##_kernel<element_type, 3><<<grid, block>>>(                                                            \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(mean.data_ptr()),   \
                reinterpret_cast<element_type *>(dst.data_ptr()), static_cast<element_type>(maxval),                   \
                static_cast<element_type>(idelta), N);                                                                 \
        }                                                                                                              \
        return dst;                                                                                                    \
    }

#define TORCH_BINDING_ADAPTIVE_THRESHOLD_GAUSSIAN(tag, th_type, element_type, cal_type, weight_type, n_pack)           \
    torch::Tensor tag##_##element_type##_##cal_type##_gaussian(torch::Tensor src, torch::Tensor mean,                  \
                                                               torch::Tensor dst, torch::Tensor weights,               \
                                                               const double maxval, const int ksz, const double delta) \
    {                                                                                                                  \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                       \
        CHECK_TORCH_TENSOR_DTYPE(mean, (th_type))                                                                      \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                       \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                                 \
        CHECK_TORCH_TENSOR_DEVICE(mean)                                                                                \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                                 \
        CHECK_TORCH_TENSOR_SAME_SIZE(src, mean)                                                                        \
        CHECK_TORCH_TENSOR_SAME_SIZE(src, dst)                                                                         \
        const int H      = src.size(0);                                                                                \
        const int W      = src.size(1);                                                                                \
        const int CH     = src.dim() == 2 ? 1 : src.size(2);                                                           \
        const int N      = H * W;                                                                                      \
        const int idelta = strstr(#tag, "inv") ? floor(delta) : ceil(delta);                                           \
        dim3      block(THREADS);                                                                                      \
        dim3      grid(divUp(N, THREADS));                                                                             \
        if (CH == 1)                                                                                                   \
        {                                                                                                              \
            gaussian_blur_kernel<element_type, weight_type, cal_type, 1><<<grid, block>>>(                             \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(mean.data_ptr()),   \
                reinterpret_cast<weight_type *>(weights.data_ptr()), ksz, ksz, H, W, N);                               \
            tag##_kernel<element_type, 1><<<grid, block>>>(                                                            \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(mean.data_ptr()),   \
                reinterpret_cast<element_type *>(dst.data_ptr()), static_cast<element_type>(maxval),                   \
                static_cast<element_type>(idelta), N);                                                                 \
        }                                                                                                              \
        else if (CH == 3)                                                                                              \
        {                                                                                                              \
            gaussian_blur_kernel<element_type, cal_type, weight_type, 3><<<grid, block>>>(                             \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(mean.data_ptr()),   \
                reinterpret_cast<weight_type *>(weights.data_ptr()), ksz, ksz, H, W, N);                               \
            tag##_kernel<element_type, 3><<<grid, block>>>(                                                            \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(mean.data_ptr()),   \
                reinterpret_cast<element_type *>(dst.data_ptr()), static_cast<element_type>(maxval),                   \
                static_cast<element_type>(idelta), N);                                                                 \
        }                                                                                                              \
        return dst;                                                                                                    \
    }

#define TORCH_BINDING_ADAPTIVE_THRESHOLD_PERCENTAGE(tag, th_type, element_type, cal_type, n_pack)                     \
    torch::Tensor tag##_##element_type##_##cal_type##_percentage(torch::Tensor src, torch::Tensor sum,                \
                                                                 torch::Tensor dst, const double maxval,              \
                                                                 const int ksz, const double percentage)              \
    {                                                                                                                 \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                      \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                      \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                                \
        CHECK_TORCH_TENSOR_DEVICE(sum)                                                                                \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                                \
        CHECK_TORCH_TENSOR_SAME_SIZE(src, sum)                                                                        \
        CHECK_TORCH_TENSOR_SAME_SIZE(src, dst)                                                                        \
        const int H  = src.size(0);                                                                                   \
        const int W  = src.size(1);                                                                                   \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                              \
        const int N  = H * W;                                                                                         \
        dim3      block(THREADS);                                                                                     \
        dim3      grid(divUp(N, THREADS));                                                                            \
        if (CH == 1)                                                                                                  \
        {                                                                                                             \
            sum_kernel<element_type, cal_type, 1><<<grid, block>>>(reinterpret_cast<element_type *>(src.data_ptr()),  \
                                                                   reinterpret_cast<cal_type *>(sum.data_ptr()), ksz, \
                                                                   ksz, H, W, N);                                     \
            tag##_percentage_kernel<element_type, cal_type, 1><<<grid, block>>>(                                      \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<cal_type *>(sum.data_ptr()),       \
                reinterpret_cast<element_type *>(dst.data_ptr()), static_cast<element_type>(maxval), ksz * ksz,       \
                percentage, N);                                                                                       \
        }                                                                                                             \
        else if (CH == 3)                                                                                             \
        {                                                                                                             \
            sum_kernel<element_type, cal_type, 3><<<grid, block>>>(reinterpret_cast<element_type *>(src.data_ptr()),  \
                                                                   reinterpret_cast<cal_type *>(sum.data_ptr()), ksz, \
                                                                   ksz, H, W, N);                                     \
            tag##_percentage_kernel<element_type, cal_type, 3><<<grid, block>>>(                                      \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<cal_type *>(sum.data_ptr()),       \
                reinterpret_cast<element_type *>(dst.data_ptr()), static_cast<element_type>(maxval), ksz * ksz,       \
                percentage, N);                                                                                       \
        }                                                                                                             \
        return dst;                                                                                                   \
    }

TORCH_BINDING_ADAPTIVE_THRESHOLD_MEAN(adaptive_threshold_binary, torch::kUInt8, uint8_t, float, 1)
TORCH_BINDING_ADAPTIVE_THRESHOLD_MEAN(adaptive_threshold_binary_inv, torch::kUInt8, uint8_t, float, 1)

TORCH_BINDING_ADAPTIVE_THRESHOLD_GAUSSIAN(adaptive_threshold_binary, torch::kUInt8, uint8_t, float, float, 1)
TORCH_BINDING_ADAPTIVE_THRESHOLD_GAUSSIAN(adaptive_threshold_binary_inv, torch::kUInt8, uint8_t, float, float, 1)

TORCH_BINDING_ADAPTIVE_THRESHOLD_PERCENTAGE(adaptive_threshold_binary, torch::kUInt8, uint8_t, uint32_t, 1)
TORCH_BINDING_ADAPTIVE_THRESHOLD_PERCENTAGE(adaptive_threshold_binary_inv, torch::kUInt8, uint8_t, uint32_t, 1)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(adaptive_threshold_binary_uint8_t_float_mean)
    TORCH_BINDING_COMMON_EXTENSION(adaptive_threshold_binary_inv_uint8_t_float_mean)
    TORCH_BINDING_COMMON_EXTENSION(adaptive_threshold_binary_uint8_t_float_gaussian)
    TORCH_BINDING_COMMON_EXTENSION(adaptive_threshold_binary_inv_uint8_t_float_gaussian)
    TORCH_BINDING_COMMON_EXTENSION(adaptive_threshold_binary_uint8_t_uint32_t_percentage)
    TORCH_BINDING_COMMON_EXTENSION(adaptive_threshold_binary_inv_uint8_t_uint32_t_percentage)
}