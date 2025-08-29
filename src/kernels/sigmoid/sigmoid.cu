#include "common.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

/**
 * @brief y = 1 / (1 + exp(-x))
 */
__device__ __forceinline__ float _sigmoid_f32(float v)
{
    v = fminf(fmaxf(v, MIN_EXP_F32), MAX_EXP_F32);
    return 1.0f / (1.0f + expf(-v));
}

__device__ __forceinline__ half _sigmoid_f16(half v)
{
    v = __hmin(__hmax(v, MIN_EXP_F16), MAX_EXP_F16);

    const half one = __float2half(1.0f);
    return one / (one + hexp(-v));
}

__global__ void sigmoid_f32_kernel(float *x, float *y, const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    y[idx] = _sigmoid_f32(x[idx]);
}

__global__ void sigmoid_f32x4_kernel(float *x, float *y, const int N)
{
    const int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= N)
        return;
    float4 reg_x = *reinterpret_cast<float4 *>(&x[idx]);
    float4 reg_y;

    reg_y.x = _sigmoid_f32(reg_x.x);
    reg_y.y = _sigmoid_f32(reg_x.y);
    reg_y.z = _sigmoid_f32(reg_x.z);
    reg_y.w = _sigmoid_f32(reg_x.w);

    *reinterpret_cast<float4 *>(&y[idx]) = reg_y;
}

__global__ void sigmoid_f16_kernel(half *x, half *y, const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    y[idx] = _sigmoid_f16(x[idx]);
}

__global__ void sigmoid_f16x2_kernel(half *x, half *y, int N)
{
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx >= N)
        return;
    half2 reg_x = *reinterpret_cast<half2 *>(&x[idx]);
    half2 reg_y;
    reg_y.x = _sigmoid_f16(reg_x.x);
    reg_y.y = _sigmoid_f16(reg_x.y);

    *reinterpret_cast<half2 *>(&y[idx]) = reg_y;
}

__global__ void sigmoid_f16x8_pack_kernel(half *x, half *y, int N)
{
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (idx + 7 >= N)
        return;

    half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.

    *reinterpret_cast<float4 *>(&pack_x[0]) = *reinterpret_cast<float4 *>(&x[idx]);
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        pack_y[i] = _sigmoid_f16(pack_x[i]);
    }
    *reinterpret_cast<float4 *>(&y[idx]) = *reinterpret_cast<float4 *>(&pack_y[0]);
}

#define TORCH_BINDING_SIGMOID(packed_type, torch_type, element_type, n_elements)                              \
    void sigmoid_##packed_type(torch::Tensor in, torch::Tensor out)                                           \
    {                                                                                                         \
        CHECK_TORCH_TENSOR_DTYPE(in, torch_type)                                                              \
        CHECK_TORCH_TENSOR_DTYPE(out, torch_type)                                                             \
        CHECK_TORCH_TENSOR_DEVICE(in)                                                                         \
        CHECK_TORCH_TENSOR_DEVICE(out)                                                                        \
                                                                                                              \
        const int H = in.size(0);                                                                             \
        const int W = in.size(1);                                                                             \
        const int N = H * W;                                                                                  \
        dim3      block(THREADS / n_elements);                                                                \
        dim3      grid(divUp(N, THREADS));                                                                    \
        sigmoid_##packed_type##_kernel<<<grid, block>>>(reinterpret_cast<element_type *>(in.data_ptr()),      \
                                                        reinterpret_cast<element_type *>(out.data_ptr()), N); \
    }

TORCH_BINDING_SIGMOID(f32, torch::kFloat32, float, 1)
TORCH_BINDING_SIGMOID(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_SIGMOID(f16, torch::kHalf, half, 1)
TORCH_BINDING_SIGMOID(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_SIGMOID(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(sigmoid_f32)
    TORCH_BINDING_COMMON_EXTENSION(sigmoid_f32x4)
    TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16)
    TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16x2)
    TORCH_BINDING_COMMON_EXTENSION(sigmoid_f16x8_pack)
}