#include "common.cuh"

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

TORCH_BINDING_CVTCOLOR_TEMPLATE(gray2bgr, torch::kFloat32, float, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(gray2bgra, torch::kFloat32, float, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(bgr2rgb, torch::kFloat32, float, 1)
TORCH_BINDING_CVTCOLOR_TEMPLATE(bgr2rgba, torch::kFloat32, float, 1)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(gray2bgr_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(gray2bgra_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(bgr2rgb_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(bgr2rgba_uint8_t)

    TORCH_BINDING_COMMON_EXTENSION(gray2bgr_float)
    TORCH_BINDING_COMMON_EXTENSION(gray2bgra_float)
    TORCH_BINDING_COMMON_EXTENSION(bgr2rgb_float)
    TORCH_BINDING_COMMON_EXTENSION(bgr2rgba_float)
}