#include "common.cuh"

/**
 * @brief 中值滤波核函数模板
 * 
 * @tparam T 像素数据类型（如uint8_t, float等）
 * @tparam CT 计算中间值类型（如int, float等，用于避免溢出）
 * @tparam CH 图像通道数（1=灰度图，3=RGB，4=RGBA等）
 * @tparam WS 滤波窗口大小（ks_h * ks_w）
 * @param src 输入图像数据指针
 * @param dst 输出图像数据指针  
 * @param ks_h 滤波核高度
 * @param ks_w 滤波核宽度
 * @param img_h 图像高度
 * @param img_w 图像宽度
 * @param N 总像素数量（img_h * img_w）
 */
template<typename T, typename CT, int CH, int WS>
__global__ void median_blur_kernel(T *src, T *dst, const int ks_h, const int ks_w, const int img_h, const int img_w,
                                   const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const int x = idx % img_w;
    const int y = idx / img_w;

    const int half_w     = ks_w / 2;
    const int half_h     = ks_h / 2;
    const int median_idx = WS / 2;

    const int base = idx * CH;

// 处理每个通道
#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        // 临时数组存储卷积核窗口内的所有像素值
        T   values[WS];
        int count = 0;

        // 遍历卷积核窗口
        for (int ky = -half_h; ky <= half_h; ++ky)
        {
            int yy = border_replicate(y + ky, img_h);
            for (int kx = -half_w; kx <= half_w; ++kx)
            {
                int xx          = border_replicate(x + kx, img_w);
                values[count++] = src[(yy * img_w + xx) * CH + c];
            }
        }

        // 冒泡排序找中值（对于小数组效率可接受）
        for (int i = 0; i < count - 1; ++i)
        {
            for (int j = 0; j < count - 1 - i; ++j)
            {
                if (values[j] > values[j + 1])
                {
                    T temp        = values[j];
                    values[j]     = values[j + 1];
                    values[j + 1] = temp;
                }
            }
        }

        dst[base + c] = values[median_idx];
    }
}

// 获取直方图的中值 (扫描256桶)
__device__ __forceinline__ uint8_t histogram_median(const int *hist, int total)
{
    int sum = 0;
    int mid = (total + 1) / 2;
    for (int i = 0; i < 256; i++)
    {
        sum += hist[i];
        if (sum >= mid)
            return (uint8_t)i;
    }
    return 255; // fallback
}

// 滑动窗口直方图中值滤波 (灰度图)
template<int RADIUS>
__global__ void median_filter_hist_u8(const uint8_t *src, uint8_t *dst, int width, int height, int stride)
{
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height)
        return;

    const int K           = 2 * RADIUS + 1;
    const int window_size = K * K;

    // 直方图存放在寄存器/局部内存
    int hist[256];
#pragma unroll
    for (int i = 0; i < 256; i++) hist[i] = 0;

    // 初始化窗口 (x=0) - 正确的窗口范围应该是 [-RADIUS, +RADIUS]
    for (int dy = -RADIUS; dy <= RADIUS; ++dy)
    {
        int yy = y + dy;
        if (yy < 0)
            yy = 0;
        if (yy >= height)
            yy = height - 1;
        const uint8_t *row = src + yy * stride;
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) // 修复：使用正确的窗口范围
        {
            int xx = 0 + dx; // 对于 x=0 的情况
            if (xx < 0)
                xx = 0;
            if (xx >= width)
                xx = width - 1;
            hist[row[xx]]++;
        }
    }

    // 第一个像素
    dst[y * stride + 0] = histogram_median(hist, window_size);

    // 滑动窗口 - 重新实现以确保正确性
    for (int x = 1; x < width; ++x)
    {
        // 当前位置 x 的窗口范围是 [x-RADIUS, x+RADIUS]
        // 前一个位置 x-1 的窗口范围是 [x-1-RADIUS, x-1+RADIUS] = [x-RADIUS-1, x+RADIUS-1]
        // 所以需要移除列 (x-RADIUS-1)，添加列 (x+RADIUS)

        int remove_col = x - RADIUS - 1;
        int add_col    = x + RADIUS;

        for (int dy = -RADIUS; dy <= RADIUS; ++dy)
        {
            int yy = y + dy;
            if (yy < 0)
                yy = 0;
            if (yy >= height)
                yy = height - 1;
            const uint8_t *row = src + yy * stride;

            // 移除左侧列（应用边界复制）
            int remove_col_clamped = remove_col;
            if (remove_col_clamped < 0)
                remove_col_clamped = 0;
            if (remove_col_clamped >= width)
                remove_col_clamped = width - 1;
            hist[row[remove_col_clamped]]--;

            // 添加右侧列（应用边界复制）
            int add_col_clamped = add_col;
            if (add_col_clamped < 0)
                add_col_clamped = 0;
            if (add_col_clamped >= width)
                add_col_clamped = width - 1;
            hist[row[add_col_clamped]]++;
        }

        dst[y * stride + x] = histogram_median(hist, window_size);
    }
}

// Host wrapper
void median_filter_gray_hist(torch::Tensor src, torch::Tensor dst, const int ksz)
{
    CHECK_TORCH_TENSOR_DTYPE(src, torch::kUInt8)
    CHECK_TORCH_TENSOR_DTYPE(dst, torch::kUInt8)
    CHECK_TORCH_TENSOR_DEVICE(src)
    CHECK_TORCH_TENSOR_DEVICE(dst)

    const int H      = src.size(0);
    const int W      = src.size(1);
    const int CH     = src.dim() == 2 ? 1 : src.size(2);
    const int stride = W * CH;

    dim3 block(1, 32); // 每个warp处理一行
    dim3 grid(1, (H + block.y - 1) / block.y);

    if (ksz == 3)
    {
        median_filter_hist_u8<1><<<grid, block>>>(reinterpret_cast<uint8_t *>(src.data_ptr()),
                                                  reinterpret_cast<uint8_t *>(dst.data_ptr()), W, H, stride);
    }
    else if (ksz == 5)
    {
        median_filter_hist_u8<2><<<grid, block>>>(reinterpret_cast<uint8_t *>(src.data_ptr()),
                                                  reinterpret_cast<uint8_t *>(dst.data_ptr()), W, H, stride);
    }
    else if (ksz == 7)
    {
        median_filter_hist_u8<3><<<grid, block>>>(reinterpret_cast<uint8_t *>(src.data_ptr()),
                                                  reinterpret_cast<uint8_t *>(dst.data_ptr()), W, H, stride);
    }
    else if (ksz == 9)
    {
        median_filter_hist_u8<4><<<grid, block>>>(reinterpret_cast<uint8_t *>(src.data_ptr()),
                                                  reinterpret_cast<uint8_t *>(dst.data_ptr()), W, H, stride);
    }
    else if (ksz == 15)
    {
        median_filter_hist_u8<7><<<grid, block>>>(reinterpret_cast<uint8_t *>(src.data_ptr()),
                                                  reinterpret_cast<uint8_t *>(dst.data_ptr()), W, H, stride);
    }
}

#define TORCH_BINDING_MEDIAN_BLUR_TEMPLATE(tag, th_type, element_type, cal_type, n_pack)                    \
    torch::Tensor tag##_##element_type##_##cal_type(torch::Tensor src, torch::Tensor dst, const int ksz)    \
    {                                                                                                       \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                            \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                            \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                      \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                      \
        const int H  = src.size(0);                                                                         \
        const int W  = src.size(1);                                                                         \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                    \
        const int N  = H * W;                                                                               \
        dim3      block(THREADS);                                                                           \
        dim3      grid(divUp(N, THREADS));                                                                  \
        if (CH == 1)                                                                                        \
        {                                                                                                   \
            if (ksz == 3)                                                                                   \
            {                                                                                               \
                tag##_kernel<element_type, cal_type, 1, 9>                                                  \
                    <<<grid, block>>>(reinterpret_cast<element_type *>(src.data_ptr()),                     \
                                      reinterpret_cast<element_type *>(dst.data_ptr()), ksz, ksz, H, W, N); \
            }                                                                                               \
            else if (ksz == 5)                                                                              \
            {                                                                                               \
                tag##_kernel<element_type, cal_type, 1, 25>                                                 \
                    <<<grid, block>>>(reinterpret_cast<element_type *>(src.data_ptr()),                     \
                                      reinterpret_cast<element_type *>(dst.data_ptr()), ksz, ksz, H, W, N); \
            }                                                                                               \
            else if (ksz == 7)                                                                              \
            {                                                                                               \
                tag##_kernel<element_type, cal_type, 1, 49>                                                 \
                    <<<grid, block>>>(reinterpret_cast<element_type *>(src.data_ptr()),                     \
                                      reinterpret_cast<element_type *>(dst.data_ptr()), ksz, ksz, H, W, N); \
            }                                                                                               \
            else if (ksz == 9)                                                                              \
            {                                                                                               \
                tag##_kernel<element_type, cal_type, 1, 81>                                                 \
                    <<<grid, block>>>(reinterpret_cast<element_type *>(src.data_ptr()),                     \
                                      reinterpret_cast<element_type *>(dst.data_ptr()), ksz, ksz, H, W, N); \
            }                                                                                               \
            else if (ksz == 15)                                                                             \
            {                                                                                               \
                tag##_kernel<element_type, cal_type, 1, 225>                                                \
                    <<<grid, block>>>(reinterpret_cast<element_type *>(src.data_ptr()),                     \
                                      reinterpret_cast<element_type *>(dst.data_ptr()), ksz, ksz, H, W, N); \
            }                                                                                               \
        }                                                                                                   \
        else if (CH == 3)                                                                                   \
        {                                                                                                   \
            if (ksz == 3)                                                                                   \
            {                                                                                               \
                tag##_kernel<element_type, cal_type, 3, 9>                                                  \
                    <<<grid, block>>>(reinterpret_cast<element_type *>(src.data_ptr()),                     \
                                      reinterpret_cast<element_type *>(dst.data_ptr()), ksz, ksz, H, W, N); \
            }                                                                                               \
            else if (ksz == 5)                                                                              \
            {                                                                                               \
                tag##_kernel<element_type, cal_type, 3, 25>                                                 \
                    <<<grid, block>>>(reinterpret_cast<element_type *>(src.data_ptr()),                     \
                                      reinterpret_cast<element_type *>(dst.data_ptr()), ksz, ksz, H, W, N); \
            }                                                                                               \
            else if (ksz == 7)                                                                              \
            {                                                                                               \
                tag##_kernel<element_type, cal_type, 3, 49>                                                 \
                    <<<grid, block>>>(reinterpret_cast<element_type *>(src.data_ptr()),                     \
                                      reinterpret_cast<element_type *>(dst.data_ptr()), ksz, ksz, H, W, N); \
            }                                                                                               \
            else if (ksz == 9)                                                                              \
            {                                                                                               \
                tag##_kernel<element_type, cal_type, 3, 81>                                                 \
                    <<<grid, block>>>(reinterpret_cast<element_type *>(src.data_ptr()),                     \
                                      reinterpret_cast<element_type *>(dst.data_ptr()), ksz, ksz, H, W, N); \
            }                                                                                               \
            else if (ksz == 15)                                                                             \
            {                                                                                               \
                tag##_kernel<element_type, cal_type, 3, 225>                                                \
                    <<<grid, block>>>(reinterpret_cast<element_type *>(src.data_ptr()),                     \
                                      reinterpret_cast<element_type *>(dst.data_ptr()), ksz, ksz, H, W, N); \
            }                                                                                               \
        }                                                                                                   \
                                                                                                            \
        return dst;                                                                                         \
    }

TORCH_BINDING_MEDIAN_BLUR_TEMPLATE(median_blur, torch::kFloat32, float, float, 1)
TORCH_BINDING_MEDIAN_BLUR_TEMPLATE(median_blur, torch::kUInt8, uint8_t, uint8_t, 1)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(median_blur_float_float)
    TORCH_BINDING_COMMON_EXTENSION(median_blur_uint8_t_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(median_filter_gray_hist)
}