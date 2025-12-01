#include "common.cuh"

/**
 * @brief Resize 区域坐标和插值系数计算 (bilinear/bicubic/lanczos)
 */
template<typename CT>
__device__ __forceinline__ void cal_interpolation(const int dst_x, const int dst_y, const double scale_x,
                                                  const double scale_y, int &sx, int &sy, CT &fx, CT &fy)
{
    CT src_x = (dst_x + 0.5) * scale_x - 0.5;
    CT src_y = (dst_y + 0.5) * scale_y - 0.5;

    if constexpr (std::is_same_v<CT, double>)
    {
        sx = __double2int_rd(src_x);
        sy = __double2int_rd(src_y);
    }
    else if constexpr (std::is_same_v<CT, float>)
    {
        sx = __float2int_rd(src_x);
        sy = __float2int_rd(src_y);
    }
    else
    {
        sx = __double2int_rd(src_x);
        sy = __double2int_rd(src_y);
    }

    fx = src_x - sx;
    fy = src_y - sy;
}

/**
 * @brief Resize Linear 区域坐标和插值系数计算
 */
template<typename CT>
__device__ __forceinline__ void cal_bilinear_interpolation(const int src_w, const int src_h, const int dst_x,
                                                           const int dst_y, const double scale_x, const double scale_y,
                                                           int &sx, int &sy, CT &fx, CT &fy, int &sx1, int &sy1)
{
    cal_interpolation<CT>(dst_x, dst_y, scale_x, scale_y, sx, sy, fx, fy);

    if (sx < 0)
    {
        sx = 0;
        fx = 0.0;
    }
    else if (sx >= src_w - 1)
    {
        sx = src_w - 1;
        fx = 0.0;
    }

    if (sy < 0)
    {
        sy = 0;
        fy = 0.0;
    }
    else if (sy >= src_h - 1)
    {
        sy = src_h - 1;
        fy = 0.0;
    }

    sx1 = min(sx + 1, src_w - 1);
    sy1 = min(sy + 1, src_h - 1);
}

/**
 * @brief 双线性插值图像缩放CUDA核函数
 * 
 * @tparam T 数据类型 (如 float, uint8_t)
 * @tparam CT 计算类型 (如 float, double)
 * @tparam CH 通道数 (1或3)
 * @param src 源图像数据指针
 * @param dst 目标图像数据指针
 * @param scale_x X轴缩放比例 (src_w / dst_w)
 * @param scale_y Y轴缩放比例 (src_h / dst_h)
 * @param src_h 源图像高度
 * @param src_w 源图像宽度
 * @param src_line_width 源图像行宽度 (src_w * ch)
 * @param dst_h 目标图像高度
 * @param dst_w 目标图像宽度
 * @param dst_line_width 目标图像行宽度 (dst_w * ch)
 * @param dst_N 目标图像总像素数 (dst_h * dst_w)
 * @note +0.5: 像素通常认为是一个 小方格，而不是一个点
 *      (dx, dy) 是像素的左上角坐标，(dx + 0.5, dy + 0.5) 是像素的中心坐标
 *      -0.5: 保证源图和目标图的像素中心对齐
 */
template<typename T, typename CT, int CH>
__global__ void resize_bilinear_kernel(T *src, T *dst, const double scale_x, const double scale_y, const int src_h,
                                       const int src_w, const int src_line_width, const int dst_h, const int dst_w,
                                       const int dst_line_width, const int dst_N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_N) // 越界
        return;
    const int dst_x = idx % dst_w;
    const int dst_y = idx / dst_w;

    int sx, sy;
    CT  fx, fy;
    int sx1, sy1;
    cal_bilinear_interpolation<CT>(src_w, src_h, dst_x, dst_y, scale_x, scale_y, sx, sy, fx, fy, sx1, sy1);

    CT w1 = (1 - fx) * (1 - fy);
    CT w2 = fx * (1 - fy);
    CT w3 = (1 - fx) * fy;
    CT w4 = fx * fy;

    T *v1 = src + sy * src_line_width + sx * CH;
    T *v2 = src + sy * src_line_width + sx1 * CH;
    T *v3 = src + sy1 * src_line_width + sx * CH;
    T *v4 = src + sy1 * src_line_width + sx1 * CH;

    const int dst_base = dst_y * dst_line_width + dst_x * CH;

#pragma unroll
    for (int i = 0; i < CH; ++i)
    {
        // S0 = (1 - fx) * v1[i] + fx * v2[i];
        // S1 = (1 - fx) * v3[i] + fx * v4[i];
        // S2 = (1 - fy) * S0 + fy * S1;
        // S2 = (1 - fx) * (1 - fy) * v1[i] + fx * (1 - fy) * v2[i] + (1 - fx) * fy * v3[i] + fx * fy * v4[i]
        CT v = w1 * v1[i] + w2 * v2[i] + w3 * v3[i] + w4 * v4[i];

        dst[dst_base + i] = saturate_cast<T>(v);
    }
}

template<typename T, typename CT, int CH>
__global__ void u8_resize_bilinear_kernel(uint8_t *src, uint8_t *dst, const double scale_x, const double scale_y,
                                          const int src_h, const int src_w, const int src_line_width, const int dst_h,
                                          const int dst_w, const int dst_line_width, const int dst_N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_N)
        return;

    const int dst_x = idx % dst_w;
    const int dst_y = idx / dst_w;

    int sx, sy;
    CT  fx, fy;
    int sx1, sy1;
    cal_bilinear_interpolation<CT>(src_w, src_h, dst_x, dst_y, scale_x, scale_y, sx, sy, fx, fy, sx1, sy1);

    short alpha0 = saturate_cast<short>((1.0f - fx) * (float)INTER_RESIZE_COEF_SCALE);
    short alpha1 = INTER_RESIZE_COEF_SCALE - alpha0;

    short beta0 = saturate_cast<short>((1.0f - fy) * (float)INTER_RESIZE_COEF_SCALE);
    short beta1 = INTER_RESIZE_COEF_SCALE - beta0;

    // 获取源数据指针
    uint8_t *row0 = src + sy * src_line_width;
    uint8_t *row1 = src + sy1 * src_line_width;

    const int dst_base = dst_y * dst_line_width + dst_x * CH;

#pragma unroll
    for (int i = 0; i < CH; ++i)
    {
        // 水平插值（模拟HResizeLinear的行为）
        // WT t0 = S0[sx]*a0 + S0[sx + cn]*a1;
        int hval0 = row0[sx * CH + i] * alpha0 + row0[sx1 * CH + i] * alpha1;
        int hval1 = row1[sx * CH + i] * alpha0 + row1[sx1 * CH + i] * alpha1;

        // 垂直插值 - 模拟VResizeLinear<uchar>的公式
        // dst[x] = uchar(( ((b0 * (S0[x] >> 4)) >> 16) + ((b1 * (S1[x] >> 4)) >> 16) + 2)>>2);
        int term1  = (beta0 * (hval0 >> 4)) >> 16; // 第一项
        int term2  = (beta1 * (hval1 >> 4)) >> 16; // 第二项
        int result = (term1 + term2 + 2) >> 2;     // 加法、加2、右移2位

        // 使用与OpenCV完全相同的转换方式：直接uchar()转换
        dst[dst_base + i] = (uint8_t)result;
    }
}

template<typename T, typename CT = double, int CH>
__global__ void resize_bilinear_2D_kernel(T *src, T *dst, const double scale_x, const double scale_y, const int src_h,
                                          const int src_w, const int dst_h, const int dst_w)
{
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst_w || dst_y >= dst_h) // 越界
        return;

    int sx, sy;
    CT  fx, fy;
    int sx1, sy1;
    cal_bilinear_interpolation<CT>(src_w, src_h, dst_x, dst_y, scale_x, scale_y, sx, sy, fx, fy, sx1, sy1);

    CT w1 = (1 - fx) * (1 - fy);
    CT w2 = fx * (1 - fy);
    CT w3 = (1 - fx) * fy;
    CT w4 = fx * fy;

    const int dst_line_width = dst_w * CH;
    const int src_line_width = src_w * CH;

    T *v1 = src + sy * src_line_width + sx * CH;
    T *v2 = src + sy * src_line_width + sx1 * CH;
    T *v3 = src + sy1 * src_line_width + sx * CH;
    T *v4 = src + sy1 * src_line_width + sx1 * CH;

    const int dst_base = dst_y * dst_line_width + dst_x * CH;

#pragma unroll
    for (int i = 0; i < CH; ++i)
    {
        dst[dst_base + i] = w1 * v1[i] + w2 * v2[i] + w3 * v3[i] + w4 * v4[i];
    }
}

// ====== Kernel：shared memory tile 版本 ======
// T: 像素存储类型(如 uint8_t/float32)
// CH: 通道数（1/3/4）
// 计算全部使用 float，提高精度与吞吐
template<typename T, typename CalType, int CH>
__global__ void resize_bilinear_shared_kernel(const T *__restrict__ src, T *__restrict__ dst, const double scale_x,
                                              const double scale_y, int src_h, int src_w, int src_line_width, int dst_h,
                                              int dst_w, int dst_line_width)
{
    const int dx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h)
        return;

    // ---- 本 block 覆盖的 dst 坐标范围 ----
    const int bx0 = blockIdx.x * blockDim.x;
    const int by0 = blockIdx.y * blockDim.y;
    const int bx1 = min(bx0 + blockDim.x - 1, dst_w - 1);
    const int by1 = min(by0 + blockDim.y - 1, dst_h - 1);

    // ---- 将 dst block 的四角映射到 src，求上界范围（含 halo）----
    const CalType src_x_min = (bx0 + 0.5) * scale_x - 0.5;
    const CalType src_x_max = (bx1 + 0.5) * scale_x - 0.5;
    const CalType src_y_min = (by0 + 0.5) * scale_y - 0.5;
    const CalType src_y_max = (by1 + 0.5) * scale_y - 0.5;

    int sx0 = __double2int_rd(min(src_x_min, src_x_max)); // floor(min)
    int sy0 = __double2int_rd(min(src_y_min, src_y_max));
    int sx1 = __double2int_rd(max(src_x_min, src_x_max)) + 1; // +1 以便能取到 x1
    int sy1 = __double2int_rd(max(src_y_min, src_y_max)) + 1;

    // clamp 到边界（并再 +1 作为右/下侧 halo）
    sx0 = max(0, sx0);
    sy0 = max(0, sy0);
    sx1 = min(src_w - 1, sx1);
    sy1 = min(src_h - 1, sy1);

    const int tileW = sx1 - sx0 + 1; // 实际需要加载的宽（含 halo）
    const int tileH = sy1 - sy0 + 1;

    // ---- 共享内存布局：float，连续存放 CH 通道 ----
    // 大小在 host 启动时提供：maxTileW * maxTileH * CH * sizeof(float)
    extern __shared__ T s_tile[];
    T                  *tile = s_tile; // [tileH][tileW][CH]

    // ---- 协作加载到共享内存 ----
    // 采用 2D 循环，保证所有元素都能被加载
    for (int ty = threadIdx.y; ty < tileH; ty += blockDim.y)
    {
        const int sy      = sy0 + ty;
        const T  *src_row = src + sy * src_line_width + sx0 * CH;
        for (int tx = threadIdx.x; tx < tileW; tx += blockDim.x)
        {
            const T *sp = src_row + tx * CH;
#pragma unroll
            for (int c = 0; c < CH; ++c)
            {
                tile[(ty * tileW + tx) * CH + c] = sp[c];
            }
        }
    }
    __syncthreads();

    // ---- 计算当前线程的插值 ----
    const CalType src_x = (dx + 0.5) * scale_x - 0.5;
    const CalType src_y = (dy + 0.5) * scale_y - 0.5;

    int     x0 = __double2int_rd(src_x);
    int     y0 = __double2int_rd(src_y);
    CalType fx = src_x - x0;
    CalType fy = src_y - y0;

    // ---- 边界处理 (OpenCV INTER_LINEAR 一致) ----
    if (x0 < 0)
    {
        x0 = 0;
        fx = 0.0f;
    }
    else if (x0 >= src_w - 1)
    {
        x0 = src_w - 1;
        fx = 0.0f;
    }

    if (y0 < 0)
    {
        y0 = 0;
        fy = 0.0f;
    }
    else if (y0 >= src_h - 1)
    {
        y0 = src_h - 1;
        fy = 0.0f;
    }

    const int x1 = min(x0 + 1, src_w - 1);
    const int y1 = min(y0 + 1, src_h - 1);

    // 转换为共享内存局部索引，确保在tile边界内
    const int lx0 = max(0, min(x0 - sx0, tileW - 1));
    const int ly0 = max(0, min(y0 - sy0, tileH - 1));
    const int lx1 = max(0, min(x1 - sx0, tileW - 1));
    const int ly1 = max(0, min(y1 - sy0, tileH - 1));

    // 双线性权重
    const CalType w1 = (1.f - fx) * (1.f - fy);
    const CalType w2 = fx * (1.f - fy);
    const CalType w3 = (1.f - fx) * fy;
    const CalType w4 = fx * fy;

    const int dst_base = dy * dst_line_width + dx * CH;

#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        const T v1 = tile[(ly0 * tileW + lx0) * CH + c];
        const T v2 = tile[(ly0 * tileW + lx1) * CH + c];
        const T v3 = tile[(ly1 * tileW + lx0) * CH + c];
        const T v4 = tile[(ly1 * tileW + lx1) * CH + c];

        dst[dst_base + c] = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
    }
}

/**
 * @param scale_x X轴缩放比例 (src_w / dst_w)
 * @param scale_y Y轴缩放比例 (src_h / dst_h)
 */
template<typename T, typename CT, int CH>
__global__ void resize_nearest_kernel(T *src, T *dst, const double scale_x, const double scale_y, const int src_h,
                                      const int src_w, const int src_line_width, const int dst_h, const int dst_w,
                                      const int dst_line_width, const int dst_N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_N)
        return;
    const int dst_x = idx % dst_w;
    const int dst_y = idx / dst_w;

    CT src_x = dst_x * scale_x;
    CT src_y = dst_y * scale_y;

    int sx, sy;
    if constexpr (std::is_same_v<CT, double>)
    {
        sx = __double2int_rd(src_x);
        sy = __double2int_rd(src_y);
    }
    else if constexpr (std::is_same_v<CT, float>)
    {
        sx = __float2int_rd(src_x);
        sy = __float2int_rd(src_y);
    }
    else
    {
        sx = __double2int_rd(src_x);
        sy = __double2int_rd(src_y);
    }

    sx = max(0, min(sx, src_w - 1));
    sy = max(0, min(sy, src_h - 1));

    const int src_base = sy * src_line_width + sx * CH;
    const int dst_base = dst_y * dst_line_width + dst_x * CH;
#pragma unroll
    for (int i = 0; i < CH; ++i)
    {
        dst[dst_base + i] = src[src_base + i];
    }
}

/**
 * @brief 双三次插值辅助函数 - 计算三次插值权重
 * @param x 插值位置 [0, 1)
 * @return 四个插值权重
 */
template<typename CT>
__device__ __forceinline__ void interpolateCubic(CT x, CT coeffs[4])
{
    const CT A = -0.75;

    coeffs[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
    coeffs[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
    coeffs[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

/**
 * @brief 图像缩放CUDA核函数, 双三次插值
 * @param scale_x X轴缩放比例 (src_w / dst_w)
 * @param scale_y Y轴缩放比例 (src_h / dst_h)
 */
template<typename T, typename CT, int CH>
__global__ void resize_bicubic_kernel(T *src, T *dst, const double scale_x, const double scale_y, const int src_h,
                                      const int src_w, const int src_line_width, const int dst_h, const int dst_w,
                                      const int dst_line_width, const int dst_N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_N)
        return;

    const int dst_x = idx % dst_w;
    const int dst_y = idx / dst_w;

    int sx, sy;
    CT  fx, fy;
    cal_interpolation<CT>(dst_x, dst_y, scale_x, scale_y, sx, sy, fx, fy);

    // 计算 bicubic 权重
    CT alpha[4], beta[4];
    interpolateCubic<CT>(fx, alpha);
    interpolateCubic<CT>(fy, beta);

    const int dst_base = dst_y * dst_line_width + dst_x * CH;

#pragma unroll
    for (int ch = 0; ch < CH; ++ch)
    {
        CT sum = 0.0;

#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            int syj     = max(0, min(sy + j - 1, src_h - 1));
            CT  row_sum = 0.0;

#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                int sxi = max(0, min(sx + i - 1, src_w - 1));
                row_sum += alpha[i] * src[syj * src_line_width + sxi * CH + ch];
            }

            sum += beta[j] * row_sum;
        }

        dst[dst_base + ch] = saturate_cast<T>(sum);
    }
}

/**
 * @brief 图像缩放CUDA核函数, 双三次插值
 * @param scale_x X轴缩放比例 (src_w / dst_w)
 * @param scale_y Y轴缩放比例 (src_h / dst_h)
 */
template<typename T, typename CT, int CH>
__global__ void u8_resize_bicubic_kernel(uint8_t *src, uint8_t *dst, const double scale_x, const double scale_y,
                                         const int src_h, const int src_w, const int src_line_width, const int dst_h,
                                         const int dst_w, const int dst_line_width, const int dst_N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_N)
        return;

    const int dst_x = idx % dst_w;
    const int dst_y = idx / dst_w;

    int   sx, sy;
    float fx, fy;
    cal_interpolation<float>(dst_x, dst_y, scale_x, scale_y, sx, sy, fx, fy);

    float alpha[4], beta[4];
    interpolateCubic<float>(fx, alpha);
    interpolateCubic<float>(fy, beta);

    short ialpha[4], ibeta[4];
#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        ialpha[k] = saturate_cast<short>(alpha[k] * INTER_RESIZE_COEF_SCALE);
        ibeta[k]  = saturate_cast<short>(beta[k] * INTER_RESIZE_COEF_SCALE);
    }

    const int dst_base = dst_y * dst_line_width + dst_x * CH;

#pragma unroll
    for (int ch = 0; ch < CH; ++ch)
    {
        int sum = 0;

#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            int syj     = max(0, min(sy + j - 1, src_h - 1));
            int row_sum = 0;

#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                int sxi = max(0, min(sx + i - 1, src_w - 1));
                row_sum += ialpha[i] * src[syj * src_line_width + sxi * CH + ch];
            }

            sum += ibeta[j] * row_sum;
        }

        dst[dst_base + ch] = saturate_cast<uint8_t>((sum + DELTA) >> SHIFT);
    }
}

// __constant__ double s45      = 0.70710678118654752440084436210485;
__constant__ double cs[8][2] = {
    {                                  1,                                   0},
    {-0.70710678118654752440084436210485, -0.70710678118654752440084436210485},
    {                                  0,                                   1},
    { 0.70710678118654752440084436210485, -0.70710678118654752440084436210485},
    {                                 -1,                                   0},
    { 0.70710678118654752440084436210485,  0.70710678118654752440084436210485},
    {                                  0,                                  -1},
    {-0.70710678118654752440084436210485,  0.70710678118654752440084436210485}
};

template<typename CT>
__device__ __forceinline__ void interpolateLanczos4(CT x, CT *coeffs)
{
    CT     sum = 0;
    double y0  = -(x + 3) * CV_PI * 0.25;
    float  s0, c0;
    // sincos(y0, &s0, &c0);
    __sincosf(y0, &s0, &c0);
    // , s0 = std::sin(y0), c0 = std::cos(y0);
    for (int i = 0; i < 8; i++)
    {
        CT y0_ = (x + 3 - i);
        if (fabs(y0_) >= 1e-6f)
        {
            double y  = -y0_ * CV_PI * 0.25;
            coeffs[i] = (CT)((cs[i][0] * s0 + cs[i][1] * c0) / (y * y));
        }
        else
        {
            // special handling for 'x' values:
            // - ~0.0: 0 0 0 1 0 0 0 0
            // - ~1.0: 0 0 0 0 1 0 0 0
            coeffs[i] = 1e30f;
        }
        sum += coeffs[i];
    }

    sum = 1.0 / sum;
    for (int i = 0; i < 8; i++) coeffs[i] *= sum;
}

/**
 * @brief 图像缩放CUDA核函数, Lanczos4 插值
 * @param scale_x X轴缩放比例 (src_w / dst_w)
 * @param scale_y Y轴缩放比例 (src_h / dst_h)
 */
template<typename T, typename CT, int CH>
__global__ void resize_lanczos_kernel(T *src, T *dst, const double scale_x, const double scale_y, const int src_h,
                                      const int src_w, const int src_line_width, const int dst_h, const int dst_w,
                                      const int dst_line_width, const int dst_N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_N)
        return;

    const int dst_x = idx % dst_w;
    const int dst_y = idx / dst_w;

    int sx, sy;
    CT  fx, fy;
    cal_interpolation<CT>(dst_x, dst_y, scale_x, scale_y, sx, sy, fx, fy);

    // 计算 Lanczos4 权重
    CT alpha[8], beta[8];
    interpolateLanczos4<CT>(fx, alpha);
    interpolateLanczos4<CT>(fy, beta);

    const int dst_base = dst_y * dst_line_width + dst_x * CH;

#pragma unroll
    for (int ch = 0; ch < CH; ++ch)
    {
        CT sum = 0.0;

#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            int syj     = max(0, min(sy + j - 3, src_h - 1)); // Lanczos4中心在第4个位置(索引3)
            CT  row_sum = 0.0;

#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                int sxi = max(0, min(sx + i - 3, src_w - 1)); // Lanczos4中心在第4个位置(索引3)
                row_sum += alpha[i] * src[syj * src_line_width + sxi * CH + ch];
            }

            sum += beta[j] * row_sum;
        }

        dst[dst_base + ch] = saturate_cast<T>(sum);
    }
}

/**
 * @param scale_x X轴缩放比例 (src_w / dst_w)
 * @param scale_y Y轴缩放比例 (src_h / dst_h)
 */
template<typename T, typename CT, int CH>
__global__ void u8_resize_lanczos_kernel(uint8_t *src, uint8_t *dst, const double scale_x, const double scale_y,
                                         const int src_h, const int src_w, const int src_line_width, const int dst_h,
                                         const int dst_w, const int dst_line_width, const int dst_N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_N)
        return;

    const int dst_x = idx % dst_w;
    const int dst_y = idx / dst_w;

    int   sx, sy;
    float fx, fy;
    cal_interpolation<float>(dst_x, dst_y, scale_x, scale_y, sx, sy, fx, fy);

    float alpha[8], beta[8];
    interpolateLanczos4<float>(fx, alpha);
    interpolateLanczos4<float>(fy, beta);

    short ialpha[8], ibeta[8];
#pragma unroll
    for (int k = 0; k < 8; ++k)
    {
        ialpha[k] = saturate_cast<short>(alpha[k] * INTER_RESIZE_COEF_SCALE);
        ibeta[k]  = saturate_cast<short>(beta[k] * INTER_RESIZE_COEF_SCALE);
    }

    const int dst_base = dst_y * dst_line_width + dst_x * CH;
#pragma unroll
    for (int ch = 0; ch < CH; ++ch)
    {
        int sum = 0;

#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            int syj     = max(0, min(sy + j - 3, src_h - 1)); // Lanczos4中心在第4个位置(索引3)
            int row_sum = 0;

#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                int sxi = max(0, min(sx + i - 3, src_w - 1)); // Lanczos4中心在第4个位置(索引3)
                row_sum += ialpha[i] * src[syj * src_line_width + sxi * CH + ch];
            }

            sum += ibeta[j] * row_sum;
        }

        dst[dst_base + ch] = saturate_cast<uint8_t>((sum + DELTA) >> SHIFT);
    }
}

/**
 * @brief Resize Area 区域坐标和插值系数计算 (缩小 downscale)
 */
template<typename CT>
__device__ __forceinline__ void cal_area_interpolation(const int ssize, const CT dx, const double scale, CT &fsx1,
                                                       CT &fsx2, CT &cellWidthX, int &sx1, int &sx2)
{
    // 计算目标像素对应的源区域
    fsx1       = dx * scale;                     // 左边界
    fsx2       = fsx1 + scale;                   // 右边界
    cellWidthX = min(scale, (CT)(ssize - fsx1)); // 区域宽度

    // 计算整数边界
    sx1 = __double2int_ru(fsx1); // 向上取整
    sx2 = __double2int_rd(fsx2); // 向下取整
    sx2 = min(sx2, ssize - 1);
    sx1 = min(sx1, sx2);
}

/**
 * @brief Resize Area 水平插值计算
 */
template<typename T, typename CT, int CH>
__device__ __forceinline__ CT HResizeArea(T *src, const int sy, const int src_w, const int src_line_width, const int ch,
                                          const CT fsx1, const CT fsx2, const CT cellWidthX, const int sx1,
                                          const int sx2)
{
    CT row_sum = 0.0;

    // 左边界部分像素
    if (sx1 - fsx1 > 1e-3)
    {
        int sx = sx1 - 1;
        if (sx >= 0)
        {
            CT alpha = (CT)((sx1 - fsx1) / cellWidthX);
            row_sum += src[sy * src_line_width + sx * CH + ch] * alpha;
        }
    }

    // 中间完整像素
    for (int sx = sx1; sx < sx2; sx++)
    {
        CT alpha = (CT)(1.0 / cellWidthX);
        row_sum += src[sy * src_line_width + sx * CH + ch] * alpha;
    }

    // 右边界部分像素
    if (fsx2 - sx2 > 1e-3 && sx2 < src_w)
    {
        CT alpha = (CT)((fsx2 - sx2) / cellWidthX);
        row_sum += src[sy * src_line_width + sx2 * CH + ch] * alpha;
    }

    return row_sum;
}

/**
 * @brief 在 src_w / dst_w >= 1 && src_h / dst_h >= 1 时用 (缩小 downscale)
 * @param scale_x X轴缩放比例 (src_w / dst_w)
 * @param scale_y Y轴缩放比例 (src_h / dst_h)
 */
template<typename T, typename CT, int CH>
__global__ void resize_area_kernel(T *src, T *dst, const double scale_x, const double scale_y, const int src_h,
                                   const int src_w, const int src_line_width, const int dst_h, const int dst_w,
                                   const int dst_line_width, const int dst_N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_N)
        return;

    const int dst_x = idx % dst_w;
    const int dst_y = idx / dst_w;

    // 计算目标像素在源图像中对应的区域范围
    CT  fsx1, fsx2, cellWidthX;
    int sx1, sx2;
    cal_area_interpolation<CT>(src_w, dst_x, scale_x, fsx1, fsx2, cellWidthX, sx1, sx2);

    CT  fsy1, fsy2, cellHeightY;
    int sy1, sy2;
    cal_area_interpolation<CT>(src_h, dst_y, scale_y, fsy1, fsy2, cellHeightY, sy1, sy2);

    const int dst_base = dst_y * dst_line_width + dst_x * CH;

// 对每个通道进行插值
#pragma unroll
    for (int ch = 0; ch < CH; ++ch)
    {
        CT sum = 0.0;

        // 垂直方向遍历源区域
        // 处理上边界部分像素
        if (sy1 - fsy1 > 1e-3)
        {
            int sy = sy1 - 1;
            if (sy >= 0)
            {
                CT beta = (CT)((sy1 - fsy1) / cellHeightY);
                CT row_sum
                    = HResizeArea<T, CT, CH>(src, sy, src_w, src_line_width, ch, fsx1, fsx2, cellWidthX, sx1, sx2);

                sum += row_sum * beta;
            }
        }

        // 处理中间完整行
        for (int sy = sy1; sy < sy2; sy++)
        {
            CT beta    = (CT)(1.0 / cellHeightY);
            CT row_sum = HResizeArea<T, CT, CH>(src, sy, src_w, src_line_width, ch, fsx1, fsx2, cellWidthX, sx1, sx2);

            sum += row_sum * beta;
        }

        // 处理下边界部分像素
        if (fsy2 - sy2 > 1e-3 && sy2 < src_h)
        {
            int sy      = sy2;
            CT  beta    = (CT)((fsy2 - sy2) / cellHeightY);
            CT  row_sum = HResizeArea<T, CT, CH>(src, sy, src_w, src_line_width, ch, fsx1, fsx2, cellWidthX, sx1, sx2);

            sum += row_sum * beta;
        }

        dst[dst_base + ch] = saturate_cast<T>(sum);
    }
}

/**
 * @brief 只在 scale_x >= 1 && scale_y >= 1 &&  scale_x - iscale_x < DBL_EPSILON && scale_y - iscale_y < DBL_EPSILON 时用
 * @param scale_x X轴缩放比例 (src_w / dst_w)
 * @param scale_y Y轴缩放比例 (src_h / dst_h)
 */
template<typename T, typename CT, int CH>
__global__ void resize_area_fast_kernel(T *src, T *dst, const double scale_x, const double scale_y, const int src_h,
                                        const int src_w, const int src_line_width, const int dst_h, const int dst_w,
                                        const int dst_line_width, const int dst_N)
{
}

/**
 * @brief Resize Area 区域坐标和插值系数计算 (放大 upscale)
 */
template<typename CT>
__device__ __forceinline__ void cal_area_bilinear_interpolation(const int src_w, const int src_h, const int dst_x,
                                                                const int dst_y, const double scale_x,
                                                                const double scale_y, int &sx, int &sy, CT &fx, CT &fy,
                                                                int &sx1, int &sy1)
{
    CT src_x = dst_x * scale_x;
    CT src_y = dst_y * scale_y;

    if constexpr (std::is_same_v<CT, double>)
    {
        sx = __double2int_rd(src_x);
        sy = __double2int_rd(src_y);
    }
    else if constexpr (std::is_same_v<CT, float>)
    {
        sx = __float2int_rd(src_x);
        sy = __float2int_rd(src_y);
    }
    else
    {
        sx = __double2int_rd(src_x);
        sy = __double2int_rd(src_y);
    }

    fx = (CT)((dst_x + 1) - (sx + 1) / scale_x);
    fy = (CT)((dst_y + 1) - (sy + 1) / scale_y);

    if constexpr (std::is_same_v<CT, double>)
    {
        fx = fx <= 0 ? 0.0 : fx - __double2int_rd(fx);
        fy = fy <= 0 ? 0.0 : fy - __double2int_rd(fy);
    }
    else if constexpr (std::is_same_v<CT, float>)
    {
        fx = fx <= 0 ? 0.0 : fx - __float2int_rd(fx);
        fy = fy <= 0 ? 0.0 : fy - __float2int_rd(fy);
    }
    else
    {
        fx = fx <= 0 ? 0.0 : fx - __double2int_rd(fx);
        fy = fy <= 0 ? 0.0 : fy - __double2int_rd(fy);
    }

    // ---- 边界处理 ----
    if (sx < 0)
    {
        sx = 0;
        fx = 0.0;
    }
    else if (sx >= src_w - 1)
    {
        sx = src_w - 1;
        fx = 0.0;
    }

    if (sy < 0)
    {
        sy = 0;
        fy = 0.0;
    }
    else if (sx >= src_h - 1)
    {
        sy = src_h - 1;
        fy = 0.0;
    }

    sx1 = min(sx + 1, src_w - 1);
    sy1 = min(sy + 1, src_h - 1);
}

/**
 * @brief 在 src_w / dst_w < 1 && src_h / dst_h < 1 时用 (放大 upscale)
 * @param scale_x X轴缩放比例 (src_w / dst_w)
 * @param scale_y Y轴缩放比例 (src_h / dst_h)
 */
template<typename T, typename CT, int CH>
__global__ void resize_area_bilinear_kernel(T *src, T *dst, const double scale_x, const double scale_y, const int src_h,
                                            const int src_w, const int src_line_width, const int dst_h, const int dst_w,
                                            const int dst_line_width, const int dst_N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_N)
        return;

    const int dst_x = idx % dst_w;
    const int dst_y = idx / dst_w;

    int sx, sy;
    CT  fx, fy;
    int sx1, sy1;
    cal_area_bilinear_interpolation<CT>(src_w, src_h, dst_x, dst_y, scale_x, scale_y, sx, sy, fx, fy, sx1, sy1);

    // 下面和双线性插值一样的处理
    CT w1 = (1 - fx) * (1 - fy);
    CT w2 = fx * (1 - fy);
    CT w3 = (1 - fx) * fy;
    CT w4 = fx * fy;

    T *v1 = src + sy * src_line_width + sx * CH;
    T *v2 = src + sy * src_line_width + sx1 * CH;
    T *v3 = src + sy1 * src_line_width + sx * CH;
    T *v4 = src + sy1 * src_line_width + sx1 * CH;

    const int dst_base = dst_y * dst_line_width + dst_x * CH;

#pragma unroll
    for (int i = 0; i < CH; ++i)
    {
        CT v = w1 * v1[i] + w2 * v2[i] + w3 * v3[i] + w4 * v4[i];

        dst[dst_base + i] = saturate_cast<T>(v);
    }
}

template<typename T, typename CT, int CH>
__global__ void u8_resize_area_bilinear_kernel(uint8_t *src, uint8_t *dst, const double scale_x, const double scale_y,
                                               const int src_h, const int src_w, const int src_line_width,
                                               const int dst_h, const int dst_w, const int dst_line_width,
                                               const int dst_N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_N)
        return;

    const int dst_x = idx % dst_w;
    const int dst_y = idx / dst_w;

    int   sx, sy;
    float fx, fy;
    int   sx1, sy1;
    cal_area_bilinear_interpolation<float>(src_w, src_h, dst_x, dst_y, scale_x, scale_y, sx, sy, fx, fy, sx1, sy1);

    short alpha0 = saturate_cast<short>((1.0f - fx) * (float)INTER_RESIZE_COEF_SCALE);
    short alpha1 = INTER_RESIZE_COEF_SCALE - alpha0;

    short beta0 = saturate_cast<short>((1.0f - fy) * (float)INTER_RESIZE_COEF_SCALE);
    short beta1 = INTER_RESIZE_COEF_SCALE - beta0;

    // 获取源数据指针
    uint8_t *row0 = src + sy * src_line_width;
    uint8_t *row1 = src + sy1 * src_line_width;

    const int dst_base = dst_y * dst_line_width + dst_x * CH;

#pragma unroll
    for (int i = 0; i < CH; ++i)
    {
        // 水平插值（模拟HResizeLinear的行为）
        // WT t0 = S0[sx]*a0 + S0[sx + cn]*a1;
        int hval0 = row0[sx * CH + i] * alpha0 + row0[sx1 * CH + i] * alpha1;
        int hval1 = row1[sx * CH + i] * alpha0 + row1[sx1 * CH + i] * alpha1;

        // 垂直插值 - 模拟VResizeLinear<uchar>的公式
        // dst[x] = uchar(( ((b0 * (S0[x] >> 4)) >> 16) + ((b1 * (S1[x] >> 4)) >> 16) + 2)>>2);
        int term1  = (beta0 * (hval0 >> 4)) >> 16; // 第一项
        int term2  = (beta1 * (hval1 >> 4)) >> 16; // 第二项
        int result = (term1 + term2 + 2) >> 2;     // 加法、加2、右移2位

        // 使用与OpenCV完全相同的转换方式：直接uchar()转换
        dst[dst_base + i] = (uint8_t)result;
    }
}

template<typename T, typename CT, int CH>
__global__ void resize_nearest_bitexact_kernel(T *src, T *dst, const double scale_x, const double scale_y,
                                               const int src_h, const int src_w, const int src_line_width,
                                               const int dst_h, const int dst_w, const int dst_line_width,
                                               const int dst_N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_N)
        return;

    const int dst_x = idx % dst_w;
    const int dst_y = idx / dst_w;

    const int ifx  = ((src_w << 16) + dst_w / 2) / dst_w; // X轴缩放因子（16位定点）
    const int ifx0 = ifx / 2 - 1;                         // X轴偏移修正（使用中心像素坐标）
    const int ify  = ((src_h << 16) + dst_h / 2) / dst_h; // Y轴缩放因子
    const int ify0 = ify / 2 - 1;                         // Y轴偏移修正

    int sx = min((ifx * dst_x + ifx0) >> 16, src_w - 1); // 源图X坐标
    int sy = min((ify * dst_y + ify0) >> 16, src_h - 1); // 源图Y坐标

    if (scale_x == 2 || scale_x == 10)
    {
        sx += 1;
    }

    if (scale_y == 2 || scale_y == 10)
    {
        sy += 1;
    }

    // printf("idx: %d, dst: (%d, %d), src: (%d, %d)\n", idx, dst_x, dst_y, sx, sy);

    const int src_base = sy * src_line_width + sx * CH;
    const int dst_base = dst_y * dst_line_width + dst_x * CH;

#pragma unroll
    for (int i = 0; i < CH; ++i)
    {
        dst[dst_base + i] = src[src_base + i];
    }
}

struct ufixedpoint16
{
    static const int fixedShift = 8;

    static __device__ __forceinline__ uint16_t toFixedPoint(double _val)
    {
        return _val > 0 ? (uint16_t)round(_val * double((1 << fixedShift))) : 0;
    }
};

template<typename T, typename CT, int CH>
__global__ void resize_bilinear_bitexact_kernel(T *src, T *dst, const double scale_x, const double scale_y,
                                                const int src_h, const int src_w, const int src_line_width,
                                                const int dst_h, const int dst_w, const int dst_line_width,
                                                const int dst_N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_N) // 越界
        return;
    const int dst_x = idx % dst_w;
    const int dst_y = idx / dst_w;

    int sx, sy;
    CT  fx, fy;
    int sx1, sy1;

    cal_bilinear_interpolation<CT>(src_w, src_h, dst_x, dst_y, scale_x, scale_y, sx, sy, fx, fy, sx1, sy1);

    uint16_t coeff_x1 = ufixedpoint16::toFixedPoint(fx);
    // uint16_t coeff_x0 = ufixedpoint16::toFixedPoint(1 - fx);
    uint16_t coeff_x0 = 256 - coeff_x1;
    uint16_t coeff_y1 = ufixedpoint16::toFixedPoint(fy);
    // uint16_t coeff_y0 = ufixedpoint16::toFixedPoint(1 - fy);
    uint16_t coeff_y0 = 256 - coeff_y1;

    T *v1 = src + sy * src_line_width + sx * CH;
    T *v2 = src + sy * src_line_width + sx1 * CH;
    T *v3 = src + sy1 * src_line_width + sx * CH;
    T *v4 = src + sy1 * src_line_width + sx1 * CH;

    const int dst_base = dst_y * dst_line_width + dst_x * CH;

    // const uint32_t round_x = (1u << (ufixedpoint16::fixedShift - 1));
    const uint32_t round_y = (1u << (ufixedpoint16::fixedShift * 2 - 1));

#pragma unroll
    for (int i = 0; i < CH; ++i)
    {
        uint32_t s0   = coeff_x0 * v1[i] + coeff_x1 * v2[i];
        uint32_t s1   = coeff_x0 * v3[i] + coeff_x1 * v4[i];
        uint32_t temp = coeff_y0 * s0 + coeff_y1 * s1;
        uint16_t v    = (uint16_t)((temp + round_y) >> (ufixedpoint16::fixedShift * 2));

        dst[dst_base + i] = saturate_cast<T>(int(v));
    }
}

#define TORCH_BINDING_RESIZE(tag, th_type, element_type, cal_type, n_pack)                                            \
    torch::Tensor tag##_##element_type##_##cal_type(torch::Tensor src, const int dst_h, const int dst_w)              \
    {                                                                                                                 \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                      \
        const int     N             = dst_h * dst_w;                                                                  \
        const int     src_h         = src.size(0);                                                                    \
        const int     src_w         = src.size(1);                                                                    \
        const int     src_ch        = src.dim() == 2 ? 1 : src.size(2);                                               \
        const int     src_line_size = src_w * src_ch;                                                                 \
        const int     dst_line_size = dst_w * src_ch;                                                                 \
        auto          options       = torch::TensorOptions().dtype(src.dtype()).device(torch::kCUDA, 0);              \
        torch::Tensor dst                                                                                             \
            = src.dim() == 2 ? torch::zeros({dst_h, dst_w}, options) : torch::zeros({dst_h, dst_w, src_ch}, options); \
        const double inv_scale_x = (double)dst_w / src_w;                                                             \
        const double inv_scale_y = (double)dst_h / src_h;                                                             \
        const double scale_x     = 1. / inv_scale_x;                                                                  \
        const double scale_y     = 1. / inv_scale_y;                                                                  \
        dim3         block(THREADS);                                                                                  \
        dim3         grid(divUp(N, THREADS));                                                                         \
        if (src_ch == 1)                                                                                              \
        {                                                                                                             \
            tag##_kernel<element_type, cal_type, 1><<<grid, block>>>(                                                 \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()),   \
                scale_x, scale_y, src_h, src_w, src_line_size, dst_h, dst_w, dst_line_size, N);                       \
        }                                                                                                             \
        else if (src_ch == 3)                                                                                         \
        {                                                                                                             \
            tag##_kernel<element_type, cal_type, 3><<<grid, block>>>(                                                 \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()),   \
                scale_x, scale_y, src_h, src_w, src_line_size, dst_h, dst_w, dst_line_size, N);                       \
        }                                                                                                             \
        return dst;                                                                                                   \
    }

#define TORCH_BINDING_RESIZE_2D(tag, th_type, element_type, cal_type, n_pack)                                         \
    torch::Tensor tag##_2D_##element_type##_##cal_type(torch::Tensor src, const int dst_h, const int dst_w)           \
    {                                                                                                                 \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                      \
        const int     src_h   = src.size(0);                                                                          \
        const int     src_w   = src.size(1);                                                                          \
        const int     src_ch  = src.dim() == 2 ? 1 : src.size(2);                                                     \
        auto          options = torch::TensorOptions().dtype(src.dtype()).device(torch::kCUDA, 0);                    \
        torch::Tensor dst                                                                                             \
            = src.dim() == 2 ? torch::zeros({dst_h, dst_w}, options) : torch::zeros({dst_h, dst_w, src_ch}, options); \
        const double inv_scale_x = (double)dst_w / src_w;                                                             \
        const double inv_scale_y = (double)dst_h / src_h;                                                             \
        const double scale_x     = 1. / inv_scale_x;                                                                  \
        const double scale_y     = 1. / inv_scale_y;                                                                  \
        dim3         block(BLOCK_SIZE_X, BLOCK_SIZE_Y);                                                               \
        dim3         grid(divUp(dst_w, block.x), divUp(dst_h, block.y));                                              \
        if (src_ch == 1)                                                                                              \
        {                                                                                                             \
            tag##_2D_kernel<element_type, cal_type, 1><<<grid, block>>>(                                              \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()),   \
                scale_x, scale_y, src_h, src_w, dst_h, dst_w);                                                        \
        }                                                                                                             \
        else if (src_ch == 3)                                                                                         \
        {                                                                                                             \
            tag##_2D_kernel<element_type, cal_type, 3><<<grid, block>>>(                                              \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()),   \
                scale_x, scale_y, src_h, src_w, dst_h, dst_w);                                                        \
        }                                                                                                             \
        return dst;                                                                                                   \
    }

#define TORCH_BINDING_RESIZE_2D_SHARED(tag, th_type, element_type, cal_type, n_pack)                                  \
    torch::Tensor tag##_2D_##element_type##_##cal_type(torch::Tensor src, const int dst_h, const int dst_w)           \
    {                                                                                                                 \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                      \
        const int     src_h         = src.size(0);                                                                    \
        const int     src_w         = src.size(1);                                                                    \
        const int     src_ch        = src.dim() == 2 ? 1 : src.size(2);                                               \
        const int     src_line_size = src_w * src_ch;                                                                 \
        const int     dst_line_size = dst_w * src_ch;                                                                 \
        auto          options       = torch::TensorOptions().dtype(src.dtype()).device(torch::kCUDA, 0);              \
        torch::Tensor dst                                                                                             \
            = src.dim() == 2 ? torch::zeros({dst_h, dst_w}, options) : torch::zeros({dst_h, dst_w, src_ch}, options); \
        const double inv_scale_x = (double)dst_w / src_w;                                                             \
        const double inv_scale_y = (double)dst_h / src_h;                                                             \
        const double scale_x     = 1. / inv_scale_x;                                                                  \
        const double scale_y     = 1. / inv_scale_y;                                                                  \
        dim3         block(BLOCK_SIZE_X, BLOCK_SIZE_Y);                                                               \
        dim3         grid(divUp(dst_w, block.x), divUp(dst_h, block.y));                                              \
                                                                                                                      \
        auto ceilf_int = [](float v)                                                                                  \
        {                                                                                                             \
            return static_cast<int>(ceilf(v));                                                                        \
        };                                                                                                            \
        const int maxTileW = std::min(src_w, ceilf_int(block.x * scale_x) + 2);                                       \
        const int maxTileH = std::min(src_h, ceilf_int(block.y * scale_y) + 2);                                       \
        size_t    shmem_bytes                                                                                         \
            = static_cast<size_t>(maxTileW) * static_cast<size_t>(maxTileH) * src_ch * sizeof(element_type);          \
        if (src_ch == 1)                                                                                              \
        {                                                                                                             \
            tag##_kernel<element_type, cal_type, 1><<<grid, block, shmem_bytes>>>(                                    \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()),   \
                scale_x, scale_y, src_h, src_w, src_line_size, dst_h, dst_w, dst_line_size);                          \
        }                                                                                                             \
        else if (src_ch == 3)                                                                                         \
        {                                                                                                             \
            tag##_kernel<element_type, cal_type, 3><<<grid, block, shmem_bytes>>>(                                    \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()),   \
                scale_x, scale_y, src_h, src_w, src_line_size, dst_h, dst_w, dst_line_size);                          \
        }                                                                                                             \
        return dst;                                                                                                   \
    }

TORCH_BINDING_RESIZE(resize_bilinear, torch::kFloat32, float, float, 1)
TORCH_BINDING_RESIZE(resize_bilinear, torch::kFloat32, float, double, 1)

TORCH_BINDING_RESIZE(resize_bilinear, torch::kUInt8, uint8_t, float, 1)
TORCH_BINDING_RESIZE(resize_bilinear, torch::kUInt8, uint8_t, double, 1)
TORCH_BINDING_RESIZE(u8_resize_bilinear, torch::kUInt8, uint8_t, float, 1)
TORCH_BINDING_RESIZE(resize_bilinear_bitexact, torch::kUInt8, uint8_t, float, 1)
TORCH_BINDING_RESIZE(resize_bilinear_bitexact, torch::kUInt8, uint8_t, double, 1)

TORCH_BINDING_RESIZE_2D(resize_bilinear, torch::kFloat32, float, float, 1)
TORCH_BINDING_RESIZE_2D(resize_bilinear, torch::kFloat32, float, double, 1)

TORCH_BINDING_RESIZE_2D_SHARED(resize_bilinear_shared, torch::kFloat32, float, float, 1)
TORCH_BINDING_RESIZE_2D_SHARED(resize_bilinear_shared, torch::kFloat32, float, double, 1)

TORCH_BINDING_RESIZE(resize_nearest, torch::kFloat32, float, float, 1)
TORCH_BINDING_RESIZE(resize_nearest, torch::kFloat32, float, double, 1)
TORCH_BINDING_RESIZE(resize_nearest, torch::kUInt8, uint8_t, float, 1)
TORCH_BINDING_RESIZE(resize_nearest, torch::kUInt8, uint8_t, double, 1)
TORCH_BINDING_RESIZE(resize_nearest_bitexact, torch::kUInt8, uint8_t, uint8_t, 1)

TORCH_BINDING_RESIZE(resize_bicubic, torch::kFloat32, float, float, 1)
TORCH_BINDING_RESIZE(resize_bicubic, torch::kFloat32, float, double, 1)
TORCH_BINDING_RESIZE(resize_bicubic, torch::kUInt8, uint8_t, float, 1)
TORCH_BINDING_RESIZE(resize_bicubic, torch::kUInt8, uint8_t, double, 1)
TORCH_BINDING_RESIZE(u8_resize_bicubic, torch::kUInt8, uint8_t, float, 1)

TORCH_BINDING_RESIZE(resize_lanczos, torch::kFloat32, float, float, 1)
TORCH_BINDING_RESIZE(resize_lanczos, torch::kFloat32, float, double, 1)
TORCH_BINDING_RESIZE(resize_lanczos, torch::kUInt8, uint8_t, float, 1)
TORCH_BINDING_RESIZE(resize_lanczos, torch::kUInt8, uint8_t, double, 1)
TORCH_BINDING_RESIZE(u8_resize_lanczos, torch::kUInt8, uint8_t, float, 1)

TORCH_BINDING_RESIZE(resize_area, torch::kFloat32, float, float, 1)
TORCH_BINDING_RESIZE(resize_area, torch::kFloat32, float, double, 1)
TORCH_BINDING_RESIZE(resize_area, torch::kUInt8, uint8_t, float, 1)
TORCH_BINDING_RESIZE(resize_area, torch::kUInt8, uint8_t, double, 1)
TORCH_BINDING_RESIZE(resize_area_bilinear, torch::kFloat32, float, float, 1)
TORCH_BINDING_RESIZE(resize_area_bilinear, torch::kFloat32, float, double, 1)
TORCH_BINDING_RESIZE(resize_area_bilinear, torch::kUInt8, uint8_t, float, 1)
TORCH_BINDING_RESIZE(resize_area_bilinear, torch::kUInt8, uint8_t, double, 1)
TORCH_BINDING_RESIZE(u8_resize_area_bilinear, torch::kUInt8, uint8_t, float, 1)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(resize_bilinear_float_float)
    TORCH_BINDING_COMMON_EXTENSION(resize_bilinear_float_double)
    TORCH_BINDING_COMMON_EXTENSION(resize_bilinear_uint8_t_float)
    TORCH_BINDING_COMMON_EXTENSION(resize_bilinear_uint8_t_double)
    TORCH_BINDING_COMMON_EXTENSION(u8_resize_bilinear_uint8_t_float)
    TORCH_BINDING_COMMON_EXTENSION(resize_bilinear_bitexact_uint8_t_float)
    TORCH_BINDING_COMMON_EXTENSION(resize_bilinear_bitexact_uint8_t_double)

    TORCH_BINDING_COMMON_EXTENSION(resize_bilinear_shared_2D_float_float)
    TORCH_BINDING_COMMON_EXTENSION(resize_bilinear_shared_2D_float_double)
    TORCH_BINDING_COMMON_EXTENSION(resize_bilinear_2D_float_float)
    TORCH_BINDING_COMMON_EXTENSION(resize_bilinear_2D_float_double)

    TORCH_BINDING_COMMON_EXTENSION(resize_nearest_float_float)
    TORCH_BINDING_COMMON_EXTENSION(resize_nearest_float_double)
    TORCH_BINDING_COMMON_EXTENSION(resize_nearest_uint8_t_float)
    TORCH_BINDING_COMMON_EXTENSION(resize_nearest_uint8_t_double)
    TORCH_BINDING_COMMON_EXTENSION(resize_nearest_bitexact_uint8_t_uint8_t)

    TORCH_BINDING_COMMON_EXTENSION(resize_bicubic_float_float)
    TORCH_BINDING_COMMON_EXTENSION(resize_bicubic_float_double)
    TORCH_BINDING_COMMON_EXTENSION(resize_bicubic_uint8_t_float)
    TORCH_BINDING_COMMON_EXTENSION(resize_bicubic_uint8_t_double)
    TORCH_BINDING_COMMON_EXTENSION(u8_resize_bicubic_uint8_t_float)

    TORCH_BINDING_COMMON_EXTENSION(resize_lanczos_float_float)
    TORCH_BINDING_COMMON_EXTENSION(resize_lanczos_float_double)
    TORCH_BINDING_COMMON_EXTENSION(resize_lanczos_uint8_t_float)
    TORCH_BINDING_COMMON_EXTENSION(resize_lanczos_uint8_t_double)
    TORCH_BINDING_COMMON_EXTENSION(u8_resize_lanczos_uint8_t_float)

    TORCH_BINDING_COMMON_EXTENSION(resize_area_float_float)
    TORCH_BINDING_COMMON_EXTENSION(resize_area_float_double)
    TORCH_BINDING_COMMON_EXTENSION(resize_area_uint8_t_float)
    TORCH_BINDING_COMMON_EXTENSION(resize_area_uint8_t_double)

    TORCH_BINDING_COMMON_EXTENSION(resize_area_bilinear_float_float)
    TORCH_BINDING_COMMON_EXTENSION(resize_area_bilinear_float_double)
    TORCH_BINDING_COMMON_EXTENSION(resize_area_bilinear_uint8_t_float)
    TORCH_BINDING_COMMON_EXTENSION(resize_area_bilinear_uint8_t_double)
    TORCH_BINDING_COMMON_EXTENSION(u8_resize_area_bilinear_uint8_t_float)
}