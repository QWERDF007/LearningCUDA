#include "common.cuh"

template<int chs>
__global__ void letter_box_kernel(uint8_t *src, float *dst, const double scale_x, const double scale_y, const int src_h,
                                  const int src_w, const int src_line_width, const int dst_h, const int dst_w,
                                  const int dst_line_width, const int dst_N, const int left, const int right,
                                  const int top, const int bottom, const int temp_w, const int temp_h)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_N) // 越界
        return;

    const int dst_x = idx % dst_w;
    const int dst_y = idx / dst_w;

    const int dst_idx = dst_y * dst_line_width + dst_x * chs;

    // 检查是否在填充区域
    if (dst_x < left || dst_x >= right || dst_y < top || dst_y >= bottom)
    {
#pragma unroll
        for (int i = 0; i < chs; ++i)
        {
            dst[dst_idx + i] = 114.0f / 255.0f; // 灰色填充值归一化
        }
        return;
    }

    // 在有效图像区域内，需要进行双线性插值
    // 计算在缩放后图像中的相对坐标
    const int temp_x = dst_x - left;
    const int temp_y = dst_y - top;

    // ========== 阶段1：水平插值（完全按OpenCV方式） ==========
    float fx = (temp_x + 0.5) * scale_x - 0.5;
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
    short alpha1 = saturate_cast<short>(fx * (float)INTER_RESIZE_COEF_SCALE);

    alpha1 = INTER_RESIZE_COEF_SCALE - alpha0;

    // ========== 阶段2：垂直插值（完全按OpenCV方式） ==========
    float fyy = (temp_y + 0.5) * scale_y - 0.5;
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
    short beta1 = saturate_cast<short>(fyy * (float)INTER_RESIZE_COEF_SCALE);

    beta1 = INTER_RESIZE_COEF_SCALE - beta0;

    // 获取源数据指针
    uint8_t *row0 = src + sy * src_line_width;
    uint8_t *row1 = src + sy1 * src_line_width;

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
        dst[dst_idx + i] = result / 255.0f;
    }
}

torch::Tensor img_letter_box(torch::Tensor src, const int dst_height, const int dst_width)
{
    CHECK_TORCH_TENSOR_DTYPE(src, torch::kUInt8)
    const int src_height = src.size(0);
    const int src_width  = src.size(1);
    const int src_ch     = src.dim() == 2 ? 1 : src.size(2);

    auto          options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
    torch::Tensor dst     = src.dim() == 2 ? torch::zeros({dst_height, dst_width}, options)
                                           : torch::zeros({dst_height, dst_width, src_ch}, options);

    const int src_line_size = src_width * src_ch;
    const int dst_line_size = dst_width * src_ch;
    const int dst_N         = dst_height * dst_width;

    const double r_w   = (double)dst_width / src_width;
    const double r_y   = (double)dst_height / src_height;
    const double ratio = min(r_w, r_y); // 取比例小的才不会超过要求的宽高

    // 填充之前，缩放之后的宽高
    const int temp_w = (int)round(src_width * ratio);
    const int temp_h = (int)round(src_height * ratio);

    // 从temp坐标到src坐标的缩放比例
    const double scale_x = (double)src_width / temp_w;
    const double scale_y = (double)src_height / temp_h;

    // 每边要填充的宽高
    int dw = (dst_width - temp_w) / 2;
    int dh = (dst_height - temp_h) / 2;

    // 计算填充参数
    const int pad_top    = (int)round(dh - 0.1);
    const int pad_bottom = dst_height - pad_top - temp_h;
    const int pad_left   = (int)round(dw - 0.1);
    const int pad_right  = dst_width - pad_left - temp_w;

    // 有效图像区域的边界（坐标范围）
    const int top    = pad_top;
    const int bottom = pad_top + temp_h;
    const int left   = pad_left;
    const int right  = pad_left + temp_w;

    dim3 block(THREADS);
    dim3 grid(divUp(dst_N, THREADS));

    uint8_t *src_ptr = (uint8_t *)src.data_ptr<uint8_t>();
    float   *dst_ptr = (float *)dst.data_ptr<float>();

    if (src_ch == 1)
    {
        letter_box_kernel<1><<<grid, block>>>(src_ptr, dst_ptr, scale_x, scale_y, src_height, src_width, src_line_size,
                                              dst_height, dst_width, dst_line_size, dst_N, left, right, top, bottom,
                                              temp_w, temp_h);
    }
    else if (src_ch == 3)
    {
        letter_box_kernel<3><<<grid, block>>>(src_ptr, dst_ptr, scale_x, scale_y, src_height, src_width, src_line_size,
                                              dst_height, dst_width, dst_line_size, dst_N, left, right, top, bottom,
                                              temp_w, temp_h);
    }

    return dst;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(img_letter_box)
}