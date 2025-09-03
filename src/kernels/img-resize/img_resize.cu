#include "common.cuh"

__device__ __forceinline__ int clip(int x, int a, int b)
{
    return x >= a ? (x < b ? x : b - 1) : a;
}

template<typename T, typename C = double, int chs>
__global__ void resize_no_align_kernel(T *src, T *dst, const double scale_x, const double scale_y, const int src_h,
                                       const int src_w, const int dst_h, const int dst_w, const int dst_N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_N) // 越界
        return;
    const int dst_x = idx % dst_w;
    const int dst_y = idx / dst_w;

    C src_x = dst_x * scale_x;
    C src_y = dst_y * scale_y;

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

// +0.5: 像素通常认为是一个 小方格，而不是一个点
// (dx, dy) 是像素的左上角坐标，(dx + 0.5, dy + 0.5) 是像素的中心坐标
// -0.5: 保证源图和目标图的像素中心对齐
template<typename T, typename C = double, int chs = 1>
__global__ void resize_align_kernel(T *src, T *dst, const double scale_x, const double scale_y, const int src_h,
                                    const int src_w, const int dst_h, const int dst_w, const int dst_N)
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

template<typename T, typename C = double, int chs = 1>
__global__ void resize_align_2D_kernel(T *src, T *dst, const double scale_x, const double scale_y, const int src_h,
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

#define TORCH_BINDING_RESIZE(tag, th_type, element_type, cal_type, n_pack)                                            \
    torch::Tensor img_resize_##tag##_##element_type##_##cal_type(torch::Tensor src, const int dst_h, const int dst_w) \
    {                                                                                                                 \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                      \
        const int     N       = dst_h * dst_w;                                                                        \
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
        dim3         block(THREADS);                                                                                  \
        dim3         grid(divUp(N, THREADS));                                                                         \
        if (src_ch == 1)                                                                                              \
        {                                                                                                             \
            resize_##tag##_kernel<element_type, cal_type, 1><<<grid, block>>>(                                        \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()),   \
                scale_x, scale_y, src_h, src_w, dst_h, dst_w, N);                                                     \
        }                                                                                                             \
        else if (src_ch == 3)                                                                                         \
        {                                                                                                             \
            resize_##tag##_kernel<element_type, cal_type, 3><<<grid, block>>>(                                        \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()),   \
                scale_x, scale_y, src_h, src_w, dst_h, dst_w, N);                                                     \
        }                                                                                                             \
        return dst;                                                                                                   \
    }

#define TORCH_BINDING_RESIZE_2D(tag, th_type, element_type, cal_type, n_pack)                                         \
    torch::Tensor img_resize_2D_##tag##_##element_type##_##cal_type(torch::Tensor src, const int dst_h,               \
                                                                    const int dst_w)                                  \
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
            resize_##tag##_2D_kernel<element_type, cal_type, 1><<<grid, block>>>(                                     \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()),   \
                scale_x, scale_y, src_h, src_w, dst_h, dst_w);                                                        \
        }                                                                                                             \
        else if (src_ch == 3)                                                                                         \
        {                                                                                                             \
            resize_##tag##_2D_kernel<element_type, cal_type, 3><<<grid, block>>>(                                     \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<element_type *>(dst.data_ptr()),   \
                scale_x, scale_y, src_h, src_w, dst_h, dst_w);                                                        \
        }                                                                                                             \
        return dst;                                                                                                   \
    }

TORCH_BINDING_RESIZE(no_align, torch::kFloat32, float, float, 1)
TORCH_BINDING_RESIZE(no_align, torch::kFloat32, float, double, 1)
TORCH_BINDING_RESIZE(align, torch::kFloat32, float, float, 1)
TORCH_BINDING_RESIZE(align, torch::kFloat32, float, double, 1)
TORCH_BINDING_RESIZE_2D(align, torch::kFloat32, float, float, 1)
TORCH_BINDING_RESIZE_2D(align, torch::kFloat32, float, double, 1)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(img_resize_no_align_float_float)
    TORCH_BINDING_COMMON_EXTENSION(img_resize_no_align_float_double)
    TORCH_BINDING_COMMON_EXTENSION(img_resize_align_float_float)
    TORCH_BINDING_COMMON_EXTENSION(img_resize_align_float_double)
    TORCH_BINDING_COMMON_EXTENSION(img_resize_2D_align_float_float)
    TORCH_BINDING_COMMON_EXTENSION(img_resize_2D_align_float_double)
}