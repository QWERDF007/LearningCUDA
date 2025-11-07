#include "common.cuh"

static const int hsv_shift = 12;

// int hrange = depth == CV_32F ? 360 : isFullRange ? 256 : 180;

__device__ __forceinline__ int sdiv_table(int v)
{
    // OpenCV: sdiv_table[i] = saturate_cast<int>((255 << hsv_shift)/(1.*i));
    return saturate_cast<int>((255 << hsv_shift) / (1. * v));
}

__device__ __forceinline__ int hdiv_table180(int v)
{
    // OpenCV: hdiv_table180[i] = saturate_cast<int>((180 << hsv_shift)/(6.*i));
    return saturate_cast<int>((180 << hsv_shift) / (6. * v));
}

template<typename T>
__global__ void bgr2hsv_kernel(T *src, T *dst, const int H, const int W, const int src_step, const int dst_step,
                               const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    const int base = tid * 3;

    T b = src[base];
    T g = src[base + 1];
    T r = src[base + 2];

    if constexpr (std::is_same_v<T, float>)
    {
        float v    = max(b, max(g, r));
        float vmin = min(b, min(g, r));
        float diff = v - vmin;

        float s = diff / (float)(fabsf(v) + FLT_EPSILON);
        diff    = (float)(60. / (diff + FLT_EPSILON));

        float h;
        if (v == r)
            h = (g - b) * diff;
        else if (v == g)
            h = (b - r) * diff + 120.f;
        else
            h = (r - g) * diff + 240.f;
        h += h < 0.f ? 360.f : 0.f;

        // dst[base]  = h * hscale; // hscale = 360.f * (1.f / 360.f);
        dst[base]     = h;
        dst[base + 1] = s;
        dst[base + 2] = v;
    }
    else
    {
        int v    = max(b, max(g, r));
        int vmin = min(b, min(g, r));

        const T   diff = saturate_cast<T>(v - vmin);
        const int vr   = v == r ? -1 : 0;
        const int vg   = v == g ? -1 : 0;

        const int s = (diff * sdiv_table(v) + (1 << (hsv_shift - 1))) >> hsv_shift;

        int h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
        h     = (h * hdiv_table180(diff) + (1 << (hsv_shift - 1))) >> hsv_shift;
        h += (h >> 31) & 180; // 使用位运算替代分支：h < 0 ? 180 : 0

        dst[base]     = saturate_cast<T>(h);
        dst[base + 1] = (T)s;
        dst[base + 2] = (T)v;
    }
}

template<typename T>
__global__ void inrange_kernel(const T *__restrict__ src, T *__restrict__ dst, const int lower_a, const int lower_b,
                               const int lower_c, const int upper_a, const int upper_b, const int upper_c, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    const int base = tid * 3;

    const bool cond1 = src[base] >= lower_a && src[base] <= upper_a;
    const bool cond2 = src[base + 1] >= lower_b && src[base + 1] <= upper_b;
    const bool cond3 = src[base + 2] >= lower_c && src[base + 2] <= upper_c;
    if (cond1 && cond2 && cond3)
    {
        dst[tid] = 255;
    }
    else
    {
        dst[tid] = 0;
    }
}

template<typename T>
__global__ void bitwise_and_kernel(const T *__restrict__ src, const T *__restrict__ mask, T *__restrict__ dst,
                                   const bool is_invert, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    const int base = tid * 3;

    bool cond = mask[tid] == 255;
    if (is_invert)
        cond = !cond;

#pragma unroll
    for (int c = 0; c < 3; ++c)
    {
        const int idx = base + c;
        dst[idx]      = cond ? src[idx] : 0;
    }
}

void extract_color_uint8_t(torch::Tensor src, torch::Tensor dst, torch::Tensor hsv, torch::Tensor mask, const int lower,
                           const int upper, const bool is_invert)
{
    CHECK_TORCH_TENSOR_DTYPE(src, (torch::kUInt8))
    CHECK_TORCH_TENSOR_DTYPE(dst, (torch::kUInt8))
    CHECK_TORCH_TENSOR_DTYPE(hsv, (torch::kUInt8))
    CHECK_TORCH_TENSOR_DTYPE(mask, (torch::kUInt8))
    CHECK_TORCH_TENSOR_DEVICE(src)
    CHECK_TORCH_TENSOR_DEVICE(dst)
    CHECK_TORCH_TENSOR_DEVICE(hsv)
    CHECK_TORCH_TENSOR_DEVICE(mask)

    const int H    = src.size(0);
    const int W    = src.size(1);
    const int CH   = src.dim() == 2 ? 1 : src.size(2);
    const int N    = H * W;
    const int step = W * CH;

    dim3 block(THREADS);
    dim3 grid(divUp(N, THREADS));
    bgr2hsv_kernel<uint8_t><<<grid, block>>>(reinterpret_cast<uint8_t *>(src.data_ptr()),
                                             reinterpret_cast<uint8_t *>(hsv.data_ptr()), H, W, step, step, N);
    inrange_kernel<uint8_t><<<grid, block>>>(reinterpret_cast<uint8_t *>(hsv.data_ptr()),
                                             reinterpret_cast<uint8_t *>(mask.data_ptr()), lower, 43, 46, upper, 255,
                                             255, N);
    bitwise_and_kernel<uint8_t><<<grid, block>>>(reinterpret_cast<uint8_t *>(src.data_ptr()),
                                                 reinterpret_cast<uint8_t *>(mask.data_ptr()),
                                                 reinterpret_cast<uint8_t *>(dst.data_ptr()), is_invert, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(extract_color_uint8_t)
}