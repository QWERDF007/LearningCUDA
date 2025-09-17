#include "common.cuh"

// +0.5: 像素通常认为是一个 小方格，而不是一个点
// (dx, dy) 是像素的左上角坐标，(dx + 0.5, dy + 0.5) 是像素的中心坐标
// -0.5: 保证源图和目标图的像素中心对齐
template<typename T, typename C, int chs>
__global__ void resize_bilinear_kernel(T *src, T *dst, const double scale_x, const double scale_y, const int src_h,
                                       const int src_w, const int src_line_width, const int dst_h, const int dst_w,
                                       const int dst_line_width, const int dst_N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_N) // 越界
        return;
    const int dst_x = idx % dst_w;
    const int dst_y = idx / dst_w;

    C src_x = (dst_x + 0.5) * scale_x - 0.5;
    C src_y = (dst_y + 0.5) * scale_y - 0.5;

    int x0 = __double2int_rd(src_x);
    int y0 = __double2int_rd(src_y);

    C fx = src_x - x0;
    C fy = src_y - y0;

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

    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);

    C w1 = (1 - fx) * (1 - fy);
    C w2 = fx * (1 - fy);
    C w3 = (1 - fx) * fy;
    C w4 = fx * fy;

    T *v1 = src + y0 * src_line_width + x0 * chs;
    T *v2 = src + y0 * src_line_width + x1 * chs;
    T *v3 = src + y1 * src_line_width + x0 * chs;
    T *v4 = src + y1 * src_line_width + x1 * chs;

    const int dst_idx = dst_y * dst_line_width + dst_x * chs;

#pragma unroll chs
    for (int i = 0; i < chs; ++i)
    {
        C v = w1 * v1[i] + w2 * v2[i] + w3 * v3[i] + w4 * v4[i];
        if constexpr (std::is_same_v<T, unsigned char>)
        {
            dst[dst_idx + i] = saturate_cast<T>(v);
        }
        else
        {
            dst[dst_idx + i] = v;
        }
    }
}

template<typename T, typename C, int chs>
__global__ void u8_resize_bilinear_kernel(uint8_t *src, uint8_t *dst, const double scale_x, const double scale_y,
                                          const int src_h, const int src_w, const int src_line_width, const int dst_h,
                                          const int dst_w, const int dst_line_width, const int dst_N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_N) // 越界
        return;

    const int dst_x = idx % dst_w;
    const int dst_y = idx / dst_w;

    // ========== 阶段1：水平插值（完全按OpenCV方式） ==========
    float fx = (dst_x + 0.5) * scale_x - 0.5;
    int   sx = __float2int_rd(fx);
    fx -= sx;

    // 边界处理
    if (sx < 0)
    {
        fx = 0.0f;
        sx = 0;
    }
    else if (sx >= src_w - 1)
    {
        fx = 0.0f;
        sx = src_w - 1;
    }

    int sx1 = min(sx + 1, src_w - 1);

    short alpha0 = saturate_cast<short>((1.0f - fx) * (float)INTER_RESIZE_COEF_SCALE);
    short alpha1 = INTER_RESIZE_COEF_SCALE - alpha0;

    // ========== 阶段2：垂直插值（完全按OpenCV方式） ==========
    float fyy = (dst_y + 0.5) * scale_y - 0.5;
    int   sy  = __float2int_rd(fyy);
    fyy -= sy;

    if (sy < 0)
    {
        fyy = 0.0f;
        sy  = 0;
    }
    else if (sy >= src_h - 1)
    {
        fyy = 0.0f;
        sy  = src_h - 1;
    }

    int sy1 = min(sy + 1, src_h - 1);

    short beta0 = saturate_cast<short>((1.0f - fyy) * (float)INTER_RESIZE_COEF_SCALE);
    short beta1 = INTER_RESIZE_COEF_SCALE - beta0;

    // 获取源数据指针
    uint8_t *row0 = src + sy * src_line_width;
    uint8_t *row1 = src + sy1 * src_line_width;

    const int dst_idx = dst_y * dst_line_width + dst_x * chs;

#pragma unroll
    for (int i = 0; i < chs; ++i)
    {
        // 水平插值（模拟HResizeLinear的行为）
        // WT t0 = S0[sx]*a0 + S0[sx + cn]*a1;
        int hval0 = row0[sx * chs + i] * alpha0 + row0[sx1 * chs + i] * alpha1;
        int hval1 = row1[sx * chs + i] * alpha0 + row1[sx1 * chs + i] * alpha1;

        // 垂直插值 - 模拟VResizeLinear<uchar>的公式
        // dst[x] = uchar(( ((b0 * (S0[x] >> 4)) >> 16) + ((b1 * (S1[x] >> 4)) >> 16) + 2)>>2);
        int term1  = (beta0 * (hval0 >> 4)) >> 16; // 第一项
        int term2  = (beta1 * (hval1 >> 4)) >> 16; // 第二项
        int result = (term1 + term2 + 2) >> 2;     // 加法、加2、右移2位

        // 使用与OpenCV完全相同的转换方式：直接uchar()转换
        dst[dst_idx + i] = (uint8_t)result;
    }
}

template<typename T, typename C = double, int chs = 1>
__global__ void resize_bilinear_2D_kernel(T *src, T *dst, const double scale_x, const double scale_y, const int src_h,
                                          const int src_w, const int dst_h, const int dst_w)
{
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst_w || dst_y >= dst_h) // 越界
        return;

    C src_x = (dst_x + 0.5) * scale_x - 0.5;
    C src_y = (dst_y + 0.5) * scale_y - 0.5;

    int x0 = __double2int_rd(src_x);
    int y0 = __double2int_rd(src_y);

    C fx = src_x - x0;
    C fy = src_y - y0;

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

    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);

    C w1 = (1 - fx) * (1 - fy);
    C w2 = fx * (1 - fy);
    C w3 = (1 - fx) * fy;
    C w4 = fx * fy;

    const int dst_line_width = dst_w * chs;
    const int src_line_width = src_w * chs;

    T *v1 = src + y0 * src_line_width + x0 * chs;
    T *v2 = src + y0 * src_line_width + x1 * chs;
    T *v3 = src + y1 * src_line_width + x0 * chs;
    T *v4 = src + y1 * src_line_width + x1 * chs;

#pragma unroll chs
    for (int i = 0; i < chs; ++i)
    {
        dst[dst_y * dst_line_width + dst_x * chs + i] = w1 * v1[i] + w2 * v2[i] + w3 * v3[i] + w4 * v4[i];
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

TORCH_BINDING_RESIZE_2D(resize_bilinear, torch::kFloat32, float, float, 1)
TORCH_BINDING_RESIZE_2D(resize_bilinear, torch::kFloat32, float, double, 1)

TORCH_BINDING_RESIZE_2D_SHARED(resize_bilinear_shared, torch::kFloat32, float, float, 1)
TORCH_BINDING_RESIZE_2D_SHARED(resize_bilinear_shared, torch::kFloat32, float, double, 1)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(resize_bilinear_float_float)
    TORCH_BINDING_COMMON_EXTENSION(resize_bilinear_float_double)

    TORCH_BINDING_COMMON_EXTENSION(resize_bilinear_uint8_t_float)
    TORCH_BINDING_COMMON_EXTENSION(resize_bilinear_uint8_t_double)
    TORCH_BINDING_COMMON_EXTENSION(u8_resize_bilinear_uint8_t_float)

    TORCH_BINDING_COMMON_EXTENSION(resize_bilinear_shared_2D_float_float)
    TORCH_BINDING_COMMON_EXTENSION(resize_bilinear_shared_2D_float_double)
    TORCH_BINDING_COMMON_EXTENSION(resize_bilinear_2D_float_float)
    TORCH_BINDING_COMMON_EXTENSION(resize_bilinear_2D_float_double)
}