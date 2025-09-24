#include "common.cuh"

/**
 * @brief 计算sigma值的辅助函数
 * 根据核大小计算合适的sigma值，模拟OpenCV的行为
 * 
 * @param kernel_size 核大小
 * @return 计算得到的sigma值
 */
template<typename T>
__device__ __host__ T get_gaussian_sigma(int kernel_size)
{
    // OpenCV默认sigma计算公式：sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    return static_cast<T>(0.3 * (static_cast<T>(kernel_size - 1) * 0.5 - 1) + 0.8);
}

/**
 * @brief 计算2D高斯函数：exp(-(x²/2σₓ²) - (y²/2σᵧ²))
 * 根据给定的偏移量和sigma值计算2D高斯权重
 * 
 * @tparam T 数值类型（float, double等）
 * @param dx 水平方向偏移量
 * @param dy 垂直方向偏移量
 * @param sigma_x 水平方向标准差
 * @param sigma_y 垂直方向标准差
 * @return 计算得到的2D高斯权重值
 */
template<typename T>
__device__ __forceinline__ T compute_gaussian_weight_2d(int dx, int dy, T sigma_x, T sigma_y)
{
    T exp_x = static_cast<T>(dx * dx) / (static_cast<T>(2) * sigma_x * sigma_x);
    T exp_y = static_cast<T>(dy * dy) / (static_cast<T>(2) * sigma_y * sigma_y);
    return exp(-(exp_x + exp_y));
}

// float特化版本，使用 __expf 以获得更好的性能
template<>
__device__ __forceinline__ float compute_gaussian_weight_2d<float>(int dx, int dy, float sigma_x, float sigma_y)
{
    // float weight = expf(-(kx * kx) / (2.0f * sigma_x * sigma_x) - (ky * ky) / (2.0f * sigma_y * sigma_y));
    float exp_x = static_cast<float>(dx * dx) / (2.0f * sigma_x * sigma_x);
    float exp_y = static_cast<float>(dy * dy) / (2.0f * sigma_y * sigma_y);
    return __expf(-(exp_x + exp_y));
}

/**
 * @brief 计算一维高斯权重
 * 根据给定的核大小和sigma值计算高斯权重数组
 * 
 * @param weights 输出权重数组指针
 * @param kernel_size 核大小（奇数）
 * @param sigma 高斯分布的标准差
 */
__device__ void compute_gaussian_weights_1d(float *weights, int kernel_size, float sigma)
{
    int   half = kernel_size / 2;
    float sum  = 0.0f;

    // 计算高斯权重
    for (int i = 0; i < kernel_size; ++i)
    {
        int x      = i - half;
        weights[i] = expf(-(x * x) / (2.0f * sigma * sigma));
        sum += weights[i];
    }

    // 归一化权重
    for (int i = 0; i < kernel_size; ++i)
    {
        weights[i] /= sum;
    }
}

/**
 * @brief 计算二维高斯权重
 * 根据给定的核大小和sigma值计算2D高斯权重矩阵
 * 
 * @param weights 输出权重矩阵指针（按行优先存储）
 * @param ks_h 核高度（奇数）
 * @param ks_w 核宽度（奇数）
 * @param sigma_x 水平方向标准差
 * @param sigma_y 垂直方向标准差
 */
__device__ void compute_gaussian_weights_2d(float *weights, int ks_h, int ks_w, float sigma_x, float sigma_y)
{
    int   half_h = ks_h / 2;
    int   half_w = ks_w / 2;
    float sum    = 0.0f;

    // 计算2D高斯权重
    for (int y = 0; y < ks_h; ++y)
    {
        for (int x = 0; x < ks_w; ++x)
        {
            int   dy              = y - half_h;
            int   dx              = x - half_w;
            float weight          = compute_gaussian_weight_2d<float>(dx, dy, sigma_x, sigma_y);
            weights[y * ks_w + x] = weight;
            sum += weight;
        }
    }

    // 归一化权重
    for (int i = 0; i < ks_h * ks_w; ++i)
    {
        weights[i] /= sum;
    }
}

/**
 * @brief 高斯模糊核函数（动态计算权重版本）
 * 在遍历卷积窗口时直接计算2D高斯权重
 * 
 * @tparam T 像素数据类型
 * @tparam CT 计算中间值类型
 * @tparam CH 图像通道数
 * @param src 输入图像数据指针
 * @param dst 输出图像数据指针
 * @param ks_h 卷积核高度
 * @param ks_w 卷积核宽度
 * @param sigma_x 水平方向标准差
 * @param sigma_y 垂直方向标准差
 * @param img_h 图像高度
 * @param img_w 图像宽度
 * @param N 总像素数量
 */
template<typename T, typename CT, typename WT, int CH>
__global__ void gaussian_blur_dynamic_kernel(T *src, T *dst, WT *weights, const int ks_h, const int ks_w,
                                             const double sigma_x, const double sigma_y, const int img_h,
                                             const int img_w, const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const int x = idx % img_w;
    const int y = idx / img_w;

    const int half_w = ks_w / 2;
    const int half_h = ks_h / 2;
    const int base   = idx * CH;

// 处理每个通道
#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        CT sum        = 0;
        CT weight_sum = 0;

        // 遍历卷积核窗口，动态计算权重
        for (int ky = -half_h; ky <= half_h; ++ky)
        {
            int yy = reflect_101(y + ky, img_h);
            for (int kx = -half_w; kx <= half_w; ++kx)
            {
                int xx = reflect_101(x + kx, img_w);

                // 使用模板函数计算2D高斯权重
                CT weight = compute_gaussian_weight_2d<CT>(kx, ky, sigma_x, sigma_y);

                sum += weight * src[(yy * img_w + xx) * CH + c];
                weight_sum += weight;
            }
        }

        // 归一化结果
        dst[base + c] = saturate_cast<T>(sum / weight_sum);
    }
}

/**
 * @brief 使用预计算2D权重的高斯模糊核函数
 * 支持任意数据类型和通道数的高斯模糊处理
 * 
 * @tparam T 像素数据类型（如uint8_t, float等）
 * @tparam CT 计算中间值类型（如int, float等，用于避免溢出）
 * @tparam CH 图像通道数（1=灰度图，3=RGB，4=RGBA等）
 * @param src 输入图像数据指针
 * @param dst 输出图像数据指针  
 * @param weights 二维权重矩阵指针（按行优先存储）
 * @param ks_h 卷积核高度
 * @param ks_w 卷积核宽度
 * @param img_h 图像高度
 * @param img_w 图像宽度
 * @param N 总像素数量（img_h * img_w）
 */
template<typename T, typename CT, typename WT, int CH>
__global__ void gaussian_blur_kernel(T *src, T *dst, WT *weights, const int ks_h, const int ks_w, const double sigma_x,
                                     const double sigma_y, const int img_h, const int img_w, const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const int x = idx % img_w;
    const int y = idx / img_w;

    const int half_w = ks_w / 2;
    const int half_h = ks_h / 2;
    const int base   = idx * CH;

// 处理每个通道
#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        CT sum = 0;

        // 遍历卷积核窗口
        for (int ky = -half_h; ky <= half_h; ++ky)
        {
            int yy = reflect_101(y + ky, img_h);
            for (int kx = -half_w; kx <= half_w; ++kx)
            {
                int xx = reflect_101(x + kx, img_w);

                // 使用2D高斯权重
                int weight_y = ky + half_h;
                int weight_x = kx + half_w;
                WT  weight   = weights[weight_y * ks_w + weight_x];
                sum += weight * src[(yy * img_w + xx) * CH + c];
            }
        }

        dst[base + c] = saturate_cast<T>(sum);
    }
}

/**
 * @brief 高斯模糊（可分离）- 水平一维卷积
 * @tparam T 像素类型
 * @tparam CT 计算类型
 * @tparam WT 权重类型
 * @tparam CH 通道数（1或3）
 */
template<typename T, typename CT, typename WT, int CH>
__global__ void gaussian_blur_sep_h_kernel(T *src, float *tmp, WT *weights_x, const int ks_w, const int img_h,
                                           const int img_w, const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const int x      = idx % img_w;
    const int y      = idx / img_w;
    const int half_w = ks_w / 2;
    const int base   = idx * CH;

#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        CT sum = 0;
        for (int kx = -half_w; kx <= half_w; ++kx)
        {
            int xx     = reflect_101(x + kx, img_w);
            int w_x    = kx + half_w;
            WT  weight = weights_x[w_x];
            sum += weight * src[(y * img_w + xx) * CH + c];
        }
        // 写入中间结果；对整数类型会产生一次中间舍入/饱和
        tmp[base + c] = saturate_cast<float>(sum);
    }
}

/**
 * @brief 高斯模糊（可分离）- 垂直一维卷积
 * @tparam T 像素类型
 * @tparam CT 计算类型
 * @tparam WT 权重类型
 * @tparam CH 通道数（1或3）
 */
template<typename T, typename CT, typename WT, int CH>
__global__ void gaussian_blur_sep_v_kernel(float *tmp, T *dst, WT *weights_y, const int ks_h, const int img_h,
                                           const int img_w, const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const int x      = idx % img_w;
    const int y      = idx / img_w;
    const int half_h = ks_h / 2;
    const int base   = idx * CH;

#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        CT sum = 0;
        for (int ky = -half_h; ky <= half_h; ++ky)
        {
            int yy     = reflect_101(y + ky, img_h);
            int w_y    = ky + half_h;
            WT  weight = weights_y[w_y];
            sum += weight * tmp[(yy * img_w + x) * CH + c];
        }
        dst[base + c] = saturate_cast<T>(sum);
    }
}

/**
 * @brief 高斯模糊绑定宏模板
 * 为不同数据类型创建高斯模糊函数的Python绑定
 * 
 * @param tag 函数标签
 * @param th_type PyTorch张量类型
 * @param element_type 元素数据类型
 * @param cal_type 计算类型
 * @param n_pack 打包数量
 */
#define TORCH_BINDING_GAUSSIAN_BLUR(tag, th_type, element_type, cal_type, weight_type, n_pack)                      \
    torch::Tensor tag##_##element_type##_##cal_type(torch::Tensor src, torch::Tensor dst, torch::Tensor weights,    \
                                                    const int ksh, const int ksw, double sigma_x, double sigma_y)   \
    {                                                                                                               \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                    \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(weights)                                                                          \
        const int H  = src.size(0);                                                                                 \
        const int W  = src.size(1);                                                                                 \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                            \
        const int N  = H * W;                                                                                       \
        dim3      block(THREADS);                                                                                   \
        dim3      grid(divUp(N, THREADS));                                                                          \
        if (sigma_x == 0.0 && sigma_y == 0.0)                                                                       \
        {                                                                                                           \
            sigma_x = get_gaussian_sigma<double>(ksw);                                                              \
            sigma_y = get_gaussian_sigma<double>(ksh);                                                              \
        }                                                                                                           \
        else if (sigma_y == 0.0)                                                                                    \
        {                                                                                                           \
            sigma_y = sigma_x;                                                                                      \
        }                                                                                                           \
        if (CH == 1)                                                                                                \
        {                                                                                                           \
            tag##_kernel<element_type, cal_type, weight_type, 1><<<grid, block>>>(                                  \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()), \
                reinterpret_cast<weight_type *>(weights.data_ptr()), ksh, ksw, sigma_x, sigma_y, H, W, N);          \
        }                                                                                                           \
        else if (CH == 3)                                                                                           \
        {                                                                                                           \
            tag##_kernel<element_type, cal_type, weight_type, 3><<<grid, block>>>(                                  \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()), \
                reinterpret_cast<weight_type *>(weights.data_ptr()), ksh, ksw, sigma_x, sigma_y, H, W, N);          \
        }                                                                                                           \
        return dst;                                                                                                 \
    }

// 分离版绑定：接受 X/Y 一维权重，先水平后垂直
#define TORCH_BINDING_GAUSSIAN_BLUR_SEP(tag, th_type, element_type, cal_type, weight_type, n_pack)                   \
    torch::Tensor tag##_##element_type##_##cal_type(torch::Tensor src, torch::Tensor tmp, torch::Tensor dst,         \
                                                    torch::Tensor weights_x, torch::Tensor weights_y, const int ksh, \
                                                    const int ksw)                                                   \
    {                                                                                                                \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                     \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                     \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                               \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                               \
        CHECK_TORCH_TENSOR_DEVICE(tmp)                                                                               \
        CHECK_TORCH_TENSOR_DEVICE(weights_x)                                                                         \
        CHECK_TORCH_TENSOR_DEVICE(weights_y)                                                                         \
        const int H  = src.size(0);                                                                                  \
        const int W  = src.size(1);                                                                                  \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                             \
        const int N  = H * W;                                                                                        \
        dim3      block(THREADS);                                                                                    \
        dim3      grid(divUp(N, THREADS));                                                                           \
        if (CH == 1)                                                                                                 \
        {                                                                                                            \
            gaussian_blur_sep_h_kernel<element_type, cal_type, weight_type, 1><<<grid, block>>>(                     \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<float *>(tmp.data_ptr()),         \
                reinterpret_cast<weight_type *>(weights_x.data_ptr()), ksw, H, W, N);                                \
            gaussian_blur_sep_v_kernel<element_type, cal_type, weight_type, 1><<<grid, block>>>(                     \
                reinterpret_cast<float *>(tmp.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()),         \
                reinterpret_cast<weight_type *>(weights_y.data_ptr()), ksh, H, W, N);                                \
        }                                                                                                            \
        else if (CH == 3)                                                                                            \
        {                                                                                                            \
            gaussian_blur_sep_h_kernel<element_type, cal_type, weight_type, 3><<<grid, block>>>(                     \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<float *>(tmp.data_ptr()),         \
                reinterpret_cast<weight_type *>(weights_x.data_ptr()), ksw, H, W, N);                                \
            gaussian_blur_sep_v_kernel<element_type, cal_type, weight_type, 3><<<grid, block>>>(                     \
                reinterpret_cast<float *>(tmp.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()),         \
                reinterpret_cast<weight_type *>(weights_y.data_ptr()), ksh, H, W, N);                                \
        }                                                                                                            \
        return dst;                                                                                                  \
    }

// 生成不同数据类型的高斯模糊函数
TORCH_BINDING_GAUSSIAN_BLUR(gaussian_blur, torch::kFloat32, float, float, float, 1)
TORCH_BINDING_GAUSSIAN_BLUR(gaussian_blur, torch::kFloat32, float, double, float, 1)
TORCH_BINDING_GAUSSIAN_BLUR(gaussian_blur, torch::kUInt8, uint8_t, float, float, 1)
TORCH_BINDING_GAUSSIAN_BLUR(gaussian_blur, torch::kUInt8, uint8_t, double, float, 1)

// 分离版函数
TORCH_BINDING_GAUSSIAN_BLUR_SEP(gaussian_blur_sep, torch::kFloat32, float, float, float, 1)
TORCH_BINDING_GAUSSIAN_BLUR_SEP(gaussian_blur_sep, torch::kFloat32, float, double, float, 1)
TORCH_BINDING_GAUSSIAN_BLUR_SEP(gaussian_blur_sep, torch::kUInt8, uint8_t, float, float, 1)
TORCH_BINDING_GAUSSIAN_BLUR_SEP(gaussian_blur_sep, torch::kUInt8, uint8_t, double, float, 1)

/**
 * @brief Python绑定模块
 * 将所有CUDA函数绑定到Python接口
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // 基础版本（每个线程计算权重）
    TORCH_BINDING_COMMON_EXTENSION(gaussian_blur_float_float)
    TORCH_BINDING_COMMON_EXTENSION(gaussian_blur_float_double)
    TORCH_BINDING_COMMON_EXTENSION(gaussian_blur_uint8_t_float)
    TORCH_BINDING_COMMON_EXTENSION(gaussian_blur_uint8_t_double)

    // 分离版本（先水平后垂直）
    TORCH_BINDING_COMMON_EXTENSION(gaussian_blur_sep_float_float)
    TORCH_BINDING_COMMON_EXTENSION(gaussian_blur_sep_float_double)
    TORCH_BINDING_COMMON_EXTENSION(gaussian_blur_sep_uint8_t_float)
    TORCH_BINDING_COMMON_EXTENSION(gaussian_blur_sep_uint8_t_double)
}