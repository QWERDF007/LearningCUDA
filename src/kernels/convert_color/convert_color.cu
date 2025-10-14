#include "common.cuh"

#include <cuda_runtime.h>

// 颜色通道辅助结构体 - 参考 OpenCV 实现
template<typename _Tp>
struct ColorChannel
{
    static __device__ __forceinline__ _Tp max()
    {
        return std::numeric_limits<_Tp>::max();
    }

    static __device__ __forceinline__ _Tp half()
    {
        return (_Tp)(max() / 2 + 1);
    }
};

template<>
struct ColorChannel<float>
{
    static __device__ __forceinline__ float max()
    {
        return 1.f;
    }

    static __device__ __forceinline__ float half()
    {
        return 0.5f;
    }
};

template<typename T>
__global__ void gray2bgr_kernel(T *src, T *dst, const int H, const int W, const int src_step, const int dst_step,
                                const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = y * dst_step + x * 3;

#pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        dst[base + i] = src[tid];
    }
}

template<typename T>
__global__ void gray2bgra_kernel(T *src, T *dst, const int H, const int W, const int src_step, const int dst_step,
                                 const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = y * dst_step + x * 4;

#pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        dst[base + i] = src[tid];
    }
    dst[base + 3] = ColorChannel<T>::max();
}

template<typename T>
__global__ void bgr2rgb_kernel(T *src, T *dst, const int H, const int W, const int src_step, const int dst_step,
                               const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = y * dst_step + x * 3;

    dst[base]     = src[base + 2];
    dst[base + 1] = src[base + 1];
    dst[base + 2] = src[base];
}

template<typename T>
__global__ void bgr2rgba_kernel(T *src, T *dst, const int H, const int W, const int src_step, const int dst_step,
                                const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int src_base = y * src_step + x * 3;
    const int dst_base = y * dst_step + x * 4;

    dst[dst_base]     = src[src_base + 2];
    dst[dst_base + 1] = src[src_base + 1];
    dst[dst_base + 2] = src[src_base];
    dst[dst_base + 3] = ColorChannel<T>::max();
}

#define CV_DESCALE(x, n) (((x) + (1 << ((n) - 1))) >> (n))

static const float B2YF = 0.114f;
static const float G2YF = 0.587f;
static const float R2YF = 0.299f;

static const int gray_shift = 15;

static const int RY15 = 9798;  // == R2YF*32768 + 0.5
static const int GY15 = 19235; // == G2YF*32768 + 0.5
static const int BY15 = 3735;  // == B2YF*32768 + 0.5

template<typename T>
__global__ void bgr2gray_kernel(T *src, T *dst, const int H, const int W, const int src_step, const int dst_step,
                                const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int src_base = y * src_step + x * 3;

    if constexpr (std::is_same_v<T, float>)
    {
        dst[tid] = src[src_base] * B2YF + src[src_base + 1] * G2YF + src[src_base + 2] * R2YF;
    }
    else
    {
        dst[tid] = CV_DESCALE(src[src_base] * BY15 + src[src_base + 1] * GY15 + src[src_base + 2] * RY15, gray_shift);
    }
}

template<typename T>
__global__ void rgb2gray_kernel(T *src, T *dst, const int H, const int W, const int src_step, const int dst_step,
                                const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int src_base = y * src_step + x * 3;

    if constexpr (std::is_same_v<T, float>)
    {
        dst[tid] = src[src_base] * R2YF + src[src_base + 1] * G2YF + src[src_base + 2] * B2YF;
    }
    else
    {
        dst[tid] = CV_DESCALE(src[src_base] * RY15 + src[src_base + 1] * GY15 + src[src_base + 2] * BY15, gray_shift);
    }
}

static const int R2Y = 4899; // == R2YF*16384
static const int G2Y = 9617; // == G2YF*16384
static const int B2Y = 1868; // == B2YF*16384

static const int yuv_shift = 14;

//to YCbCr
static const float YCBF = 0.564f; // == 1/2/(1-B2YF)
static const float YCRF = 0.713f; // == 1/2/(1-R2YF)
static const int   YCBI = 9241;   // == YCBF*16384
static const int   YCRI = 11682;  // == YCRF*16384

//to YUV
static const float B2UF = 0.492f;
static const float R2VF = 0.877f;
static const int   B2UI = 8061;  // == B2UF*16384
static const int   R2VI = 14369; // == R2VF*16384

template<typename T>
__global__ void bgr2YCrCb_kernel(T *src, T *dst, const int H, const int W, const int src_step, const int dst_step,
                                 const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = y * src_step + x * 3;

    if constexpr (std::is_same_v<T, float>)
    {
        const float delta = ColorChannel<T>::half();

        float Y  = src[base] * B2YF + src[base + 1] * G2YF + src[base + 2] * R2YF;
        float Cr = (src[base + 2] - Y) * YCRF + delta;
        float Cb = (src[base] - Y) * YCBF + delta;

        dst[base]     = Y;
        dst[base + 1] = Cr;
        dst[base + 2] = Cb;
    }
    else
    {
        const int delta = ColorChannel<T>::half() * (1 << yuv_shift);

        int Y  = CV_DESCALE(src[base] * B2Y + src[base + 1] * G2Y + src[base + 2] * R2Y, yuv_shift);
        int Cr = CV_DESCALE((src[base + 2] - Y) * YCRI + delta, yuv_shift);
        int Cb = CV_DESCALE((src[base] - Y) * YCBI + delta, yuv_shift);

        dst[base]     = saturate_cast<T>(Y);
        dst[base + 1] = saturate_cast<T>(Cr);
        dst[base + 2] = saturate_cast<T>(Cb);
    }
}

template<typename T>
__global__ void bgr2YUV_kernel(T *src, T *dst, const int H, const int W, const int src_step, const int dst_step,
                               const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = y * src_step + x * 3;

    if constexpr (std::is_same_v<T, float>)
    {
        const float delta = ColorChannel<T>::half();

        float Y  = src[base] * B2YF + src[base + 1] * G2YF + src[base + 2] * R2YF;
        float Cr = (src[base + 2] - Y) * R2VF + delta;
        float Cb = (src[base] - Y) * B2UF + delta;

        dst[base]     = Y;
        dst[base + 2] = Cr;
        dst[base + 1] = Cb;
    }
    else
    {
        const int delta = ColorChannel<T>::half() * (1 << yuv_shift);

        int Y  = CV_DESCALE(src[base] * B2Y + src[base + 1] * G2Y + src[base + 2] * R2Y, yuv_shift);
        int Cr = CV_DESCALE((src[base + 2] - Y) * R2VI + delta, yuv_shift);
        int Cb = CV_DESCALE((src[base] - Y) * B2UI + delta, yuv_shift);

        dst[base]     = saturate_cast<T>(Y);
        dst[base + 2] = saturate_cast<T>(Cr);
        dst[base + 1] = saturate_cast<T>(Cb);
    }
}

template<typename T>
__global__ void rgb2YCrCb_kernel(T *src, T *dst, const int H, const int W, const int src_step, const int dst_step,
                                 const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = y * src_step + x * 3;

    if constexpr (std::is_same_v<T, float>)
    {
        const float delta = ColorChannel<T>::half();

        float Y  = src[base] * R2YF + src[base + 1] * G2YF + src[base + 2] * B2YF;
        float Cr = (src[base] - Y) * YCRF + delta;
        float Cb = (src[base + 2] - Y) * YCBF + delta;

        dst[base]     = Y;
        dst[base + 1] = Cr;
        dst[base + 2] = Cb;
    }
    else
    {
        int delta = ColorChannel<T>::half() * (1 << yuv_shift);

        int Y  = CV_DESCALE(src[base] * R2Y + src[base + 1] * G2Y + src[base + 2] * B2Y, yuv_shift);
        int Cr = CV_DESCALE((src[base] - Y) * YCRI + delta, yuv_shift);
        int Cb = CV_DESCALE((src[base + 2] - Y) * YCBI + delta, yuv_shift);

        dst[base]     = saturate_cast<T>(Y);
        dst[base + 1] = saturate_cast<T>(Cr);
        dst[base + 2] = saturate_cast<T>(Cb);
    }
}

template<typename T>
__global__ void rgb2YUV_kernel(T *src, T *dst, const int H, const int W, const int src_step, const int dst_step,
                               const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = y * src_step + x * 3;

    if constexpr (std::is_same_v<T, float>)
    {
        const float delta = ColorChannel<T>::half();

        float Y  = src[base] * R2YF + src[base + 1] * G2YF + src[base + 2] * B2YF;
        float Cr = (src[base] - Y) * R2VF + delta;
        float Cb = (src[base + 2] - Y) * B2UF + delta;

        dst[base]     = Y;
        dst[base + 2] = Cr;
        dst[base + 1] = Cb;
    }
    else
    {
        int delta = ColorChannel<T>::half() * (1 << yuv_shift);

        int Y  = CV_DESCALE(src[base] * R2Y + src[base + 1] * G2Y + src[base + 2] * B2Y, yuv_shift);
        int Cr = CV_DESCALE((src[base] - Y) * R2VI + delta, yuv_shift);
        int Cb = CV_DESCALE((src[base + 2] - Y) * B2UI + delta, yuv_shift);

        dst[base]     = saturate_cast<T>(Y);
        dst[base + 2] = saturate_cast<T>(Cr);
        dst[base + 1] = saturate_cast<T>(Cb);
    }
}

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
    const int x = tid % W;
    const int y = tid / W;

    const int base = y * src_step + x * 3;

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
__global__ void rgb2hsv_kernel(T *src, T *dst, const int H, const int W, const int src_step, const int dst_step,
                               const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = y * src_step + x * 3;

    T r = src[base];
    T g = src[base + 1];
    T b = src[base + 2];

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

__device__ void hls_f(float b, float g, float r, float &h, float &l, float &s)
{
    float vmax = max(b, max(g, r));
    float vmin = min(b, min(g, r));
    float diff = vmax - vmin;

    l = (vmax + vmin) * 0.5f;
    if (diff > FLT_EPSILON)
    {
        s    = l < 0.5f ? diff / (vmax + vmin) : diff / (2 - vmax - vmin);
        diff = 60.f / diff;

        if (vmax == r)
            h = (g - b) * diff;
        else if (vmax == g)
            h = (b - r) * diff + 120.f;
        else
            h = (r - g) * diff + 240.f;

        if (h < 0.f)
            h += 360.f;
    }
}

template<typename T>
__global__ void bgr2hls_kernel(T *src, T *dst, const int H, const int W, const int src_step, const int dst_step,
                               const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = y * src_step + x * 3;

    T b = src[base];
    T g = src[base + 1];
    T r = src[base + 2];

    float h = 0.f, l = 0.f, s = 0.f;

    if constexpr (std::is_same_v<T, float>)
    {
        hls_f(b, g, r, h, l, s);

        // dst[base]     = h * hscale; hscale = 360.f / 360.f;
        dst[base]     = h;
        dst[base + 1] = l;
        dst[base + 2] = s;
    }
    else
    {
        float b_f = b * (1.f / 255.f);
        float g_f = g * (1.f / 255.f);
        float r_f = r * (1.f / 255.f);

        hls_f(b_f, g_f, r_f, h, l, s);

        // dst[base]     = saturate_cast<T>(h * hscale); // hscale = 180 / 360.f;
        dst[base]     = saturate_cast<T>(h * 0.5f);
        dst[base + 1] = saturate_cast<T>(l * 255.f);
        dst[base + 2] = saturate_cast<T>(s * 255.f);
    }
}

template<typename T>
__global__ void rgb2hls_kernel(T *src, T *dst, const int H, const int W, const int src_step, const int dst_step,
                               const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = y * src_step + x * 3;

    T r = src[base];
    T g = src[base + 1];
    T b = src[base + 2];

    float h = 0.f, l = 0.f, s = 0.f;

    if constexpr (std::is_same_v<T, float>)
    {
        hls_f(b, g, r, h, l, s);

        // dst[base]     = h * hscale; hscale = 360.f / 360.f;
        dst[base]     = h;
        dst[base + 1] = l;
        dst[base + 2] = s;
    }
    else
    {
        float b_f = b * (1.f / 255.f);
        float g_f = g * (1.f / 255.f);
        float r_f = r * (1.f / 255.f);

        hls_f(b_f, g_f, r_f, h, l, s);

        // dst[base]     = saturate_cast<T>(h * hscale); // hscale = 180 / 360.f;
        dst[base]     = saturate_cast<T>(h * 0.5f);
        dst[base + 1] = saturate_cast<T>(l * 255.f);
        dst[base + 2] = saturate_cast<T>(s * 255.f);
    }
}

static const int xyz_shift = 12;

__constant__ int   sRGB2XYZ_D65_i[9] = {1689, 1465, 739, 871, 2929, 296, 79, 488, 3892};
__constant__ float sRGB2XYZ_D65_f[9]
    = {0.412453, 0.357580, 0.180423, 0.212671, 0.715160, 0.072169, 0.019334, 0.119193, 0.950227};

template<typename T>
__global__ void bgr2xyz_kernel(T *src, T *dst, const int H, const int W, const int src_step, const int dst_step,
                               const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = y * src_step + x * 3;

    T b = src[base];
    T g = src[base + 1];
    T r = src[base + 2];

    if constexpr (std::is_same_v<T, float>)
    {
        float X = saturate_cast<T>(b * sRGB2XYZ_D65_f[2] + g * sRGB2XYZ_D65_f[1] + r * sRGB2XYZ_D65_f[0]);
        float Y = saturate_cast<T>(b * sRGB2XYZ_D65_f[5] + g * sRGB2XYZ_D65_f[4] + r * sRGB2XYZ_D65_f[3]);
        float Z = saturate_cast<T>(b * sRGB2XYZ_D65_f[8] + g * sRGB2XYZ_D65_f[7] + r * sRGB2XYZ_D65_f[6]);

        dst[base]     = X;
        dst[base + 1] = Y;
        dst[base + 2] = Z;
    }
    else
    {
        int X = CV_DESCALE(b * sRGB2XYZ_D65_i[2] + g * sRGB2XYZ_D65_i[1] + r * sRGB2XYZ_D65_i[0], xyz_shift);
        int Y = CV_DESCALE(b * sRGB2XYZ_D65_i[5] + g * sRGB2XYZ_D65_i[4] + r * sRGB2XYZ_D65_i[3], xyz_shift);
        int Z = CV_DESCALE(b * sRGB2XYZ_D65_i[8] + g * sRGB2XYZ_D65_i[7] + r * sRGB2XYZ_D65_i[6], xyz_shift);

        dst[base]     = saturate_cast<T>(X);
        dst[base + 1] = saturate_cast<T>(Y);
        dst[base + 2] = saturate_cast<T>(Z);
    }
}

template<typename T>
__global__ void rgb2xyz_kernel(T *src, T *dst, const int H, const int W, const int src_step, const int dst_step,
                               const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int base = y * src_step + x * 3;

    T r = src[base];
    T g = src[base + 1];
    T b = src[base + 2];

    if constexpr (std::is_same_v<T, float>)
    {
        float X = saturate_cast<T>(b * sRGB2XYZ_D65_f[2] + g * sRGB2XYZ_D65_f[1] + r * sRGB2XYZ_D65_f[0]);
        float Y = saturate_cast<T>(b * sRGB2XYZ_D65_f[5] + g * sRGB2XYZ_D65_f[4] + r * sRGB2XYZ_D65_f[3]);
        float Z = saturate_cast<T>(b * sRGB2XYZ_D65_f[8] + g * sRGB2XYZ_D65_f[7] + r * sRGB2XYZ_D65_f[6]);

        dst[base]     = X;
        dst[base + 1] = Y;
        dst[base + 2] = Z;
    }
    else
    {
        int X = CV_DESCALE(b * sRGB2XYZ_D65_i[2] + g * sRGB2XYZ_D65_i[1] + r * sRGB2XYZ_D65_i[0], xyz_shift);
        int Y = CV_DESCALE(b * sRGB2XYZ_D65_i[5] + g * sRGB2XYZ_D65_i[4] + r * sRGB2XYZ_D65_i[3], xyz_shift);
        int Z = CV_DESCALE(b * sRGB2XYZ_D65_i[8] + g * sRGB2XYZ_D65_i[7] + r * sRGB2XYZ_D65_i[6], xyz_shift);

        dst[base]     = saturate_cast<T>(X);
        dst[base + 1] = saturate_cast<T>(Y);
        dst[base + 2] = saturate_cast<T>(Z);
    }
}

#define TORCH_BINDING_CVTCOLOR_TEMPLATE(tag, th_type, element_type, n_pack)                                           \
    void tag##_##element_type(torch::Tensor src, torch::Tensor dst)                                                   \
    {                                                                                                                 \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                      \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                      \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                                \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                                \
        const int H        = src.size(0);                                                                             \
        const int W        = src.size(1);                                                                             \
        const int src_CH   = src.dim() == 2 ? 1 : src.size(2);                                                        \
        const int src_step = W * src_CH;                                                                              \
        const int dst_CH   = dst.dim() == 2 ? 1 : dst.size(2);                                                        \
        const int dst_step = W * dst_CH;                                                                              \
        const int N        = H * W;                                                                                   \
        dim3      block(THREADS);                                                                                     \
        dim3      grid(divUp(N, THREADS));                                                                            \
        tag##_kernel<element_type><<<grid, block>>>(reinterpret_cast<element_type *>(src.data_ptr()),                 \
                                                    reinterpret_cast<element_type *>(dst.data_ptr()), H, W, src_step, \
                                                    dst_step, N);                                                     \
    }

TORCH_BINDING_CVTCOLOR_TEMPLATE(gray2bgr, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(gray2bgra, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(bgr2rgb, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(bgr2rgba, torch::kUInt8, uint8_t, 1)

TORCH_BINDING_CVTCOLOR_TEMPLATE(bgr2gray, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(rgb2gray, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(bgr2gray, torch::kFloat32, float, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(rgb2gray, torch::kFloat32, float, 1)

TORCH_BINDING_CVTCOLOR_TEMPLATE(gray2bgr, torch::kFloat32, float, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(gray2bgra, torch::kFloat32, float, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(bgr2rgb, torch::kFloat32, float, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(bgr2rgba, torch::kFloat32, float, 1)

TORCH_BINDING_CVTCOLOR_TEMPLATE(bgr2YCrCb, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(rgb2YCrCb, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(bgr2YCrCb, torch::kFloat32, float, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(rgb2YCrCb, torch::kFloat32, float, 1)

TORCH_BINDING_CVTCOLOR_TEMPLATE(bgr2YUV, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(rgb2YUV, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(bgr2YUV, torch::kFloat32, float, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(rgb2YUV, torch::kFloat32, float, 1)

TORCH_BINDING_CVTCOLOR_TEMPLATE(bgr2hsv, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(rgb2hsv, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(bgr2hsv, torch::kFloat32, float, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(rgb2hsv, torch::kFloat32, float, 1)

TORCH_BINDING_CVTCOLOR_TEMPLATE(bgr2hls, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(rgb2hls, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(bgr2hls, torch::kFloat32, float, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(rgb2hls, torch::kFloat32, float, 1)

TORCH_BINDING_CVTCOLOR_TEMPLATE(bgr2xyz, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(rgb2xyz, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(bgr2xyz, torch::kFloat32, float, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(rgb2xyz, torch::kFloat32, float, 1)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(gray2bgr_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(gray2bgra_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(bgr2rgb_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(bgr2rgba_uint8_t)

    TORCH_BINDING_COMMON_EXTENSION(bgr2gray_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(rgb2gray_uint8_t)

    TORCH_BINDING_COMMON_EXTENSION(bgr2gray_float)
    TORCH_BINDING_COMMON_EXTENSION(rgb2gray_float)

    TORCH_BINDING_COMMON_EXTENSION(gray2bgr_float)
    TORCH_BINDING_COMMON_EXTENSION(gray2bgra_float)
    TORCH_BINDING_COMMON_EXTENSION(bgr2rgb_float)
    TORCH_BINDING_COMMON_EXTENSION(bgr2rgba_float)

    TORCH_BINDING_COMMON_EXTENSION(bgr2YCrCb_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(rgb2YCrCb_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(bgr2YCrCb_float)
    TORCH_BINDING_COMMON_EXTENSION(rgb2YCrCb_float)

    TORCH_BINDING_COMMON_EXTENSION(bgr2YUV_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(rgb2YUV_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(bgr2YUV_float)
    TORCH_BINDING_COMMON_EXTENSION(rgb2YUV_float)

    TORCH_BINDING_COMMON_EXTENSION(bgr2hsv_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(rgb2hsv_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(bgr2hsv_float)
    TORCH_BINDING_COMMON_EXTENSION(rgb2hsv_float)

    TORCH_BINDING_COMMON_EXTENSION(bgr2hls_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(rgb2hls_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(bgr2hls_float)
    TORCH_BINDING_COMMON_EXTENSION(rgb2hls_float)

    TORCH_BINDING_COMMON_EXTENSION(bgr2xyz_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(rgb2xyz_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(bgr2xyz_float)
    TORCH_BINDING_COMMON_EXTENSION(rgb2xyz_float)
}