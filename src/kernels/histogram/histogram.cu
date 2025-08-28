#include "common.cuh"

#include <cuda_runtime.h>

__global__ void histogram_u8_kernel(uint8_t *in, int32_t *out, const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    atomicAdd(&(out[in[idx]]), 1);
}

__global__ void histogram_u8_kernel2D(uint8_t *in, int32_t *out, const int img_h, const int img_w)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= img_w || y >= img_h)
        return;
    const int idx = x + y * img_w;
    atomicAdd(&(out[in[idx]]), 1);
}

__global__ void histogram_u8x4_kernel(uint8_t *in, int32_t *out, const int N)
{
    const int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= N)
        return;
    uchar4 in4 = *reinterpret_cast<uchar4 *>(&in[idx]);
    atomicAdd(&(out[in4.x]), 1);
    atomicAdd(&(out[in4.y]), 1);
    atomicAdd(&(out[in4.z]), 1);
    atomicAdd(&(out[in4.w]), 1);
}

__global__ void histogram_u8x4_kernel2D(uint8_t *in, int32_t *out, const int img_h, const int img_w)
{
    const int x = 4 * (threadIdx.x + blockIdx.x * blockDim.x);
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= img_w || y >= img_h)
        return;
    const int idx = x + y * img_w;

    // 检查是否有足够的像素进行向量化处理
    if (x + 3 < img_w)
    {
        // 完整的4像素向量化处理
        uchar4 in4 = *reinterpret_cast<uchar4 *>(&in[idx]);
        atomicAdd(&(out[in4.x]), 1);
        atomicAdd(&(out[in4.y]), 1);
        atomicAdd(&(out[in4.z]), 1);
        atomicAdd(&(out[in4.w]), 1);
    }
    else
    {
        // 处理剩余的像素（不足4个的情况）
        for (int i = 0; i < 4 && (x + i) < img_w; i++)
        {
            atomicAdd(&(out[in[idx + i]]), 1);
        }
    }
}

__global__ void histogram_i32_kernel(int32_t *in, int32_t *out, const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    atomicAdd(&(out[in[idx]]), 1);
}

__global__ void histogram_i32x4_kernel(int32_t *in, int32_t *out, const int N)
{
    const int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= N)
        return;
    int4 in4 = *reinterpret_cast<int4 *>(&in[idx]);
    atomicAdd(&(out[in4.x]), 1);
    atomicAdd(&(out[in4.y]), 1);
    atomicAdd(&(out[in4.z]), 1);
    atomicAdd(&(out[in4.w]), 1);
}

#define TORCH_BINDING_HISTOGRAM(packed_type, torch_type, element_type, n_elements)                         \
    torch::Tensor histogram_##packed_type(torch::Tensor in)                                                \
    {                                                                                                      \
        CHECK_TORCH_TENSOR_DTYPE(in, torch_type)                                                           \
        CHECK_TORCH_TENSOR_DEVICE(in)                                                                      \
                                                                                                           \
        torch::Tensor max_val = torch::max(in).cpu();                                                      \
        const int     M       = max_val.item().to<int>();                                                  \
        auto          options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);       \
        torch::Tensor out     = torch::zeros({M + 1}, options);                                            \
                                                                                                           \
        const int H = in.size(0);                                                                          \
        const int W = in.size(1);                                                                          \
        const int N = H * W;                                                                               \
        dim3      block(THREADS / n_elements);                                                             \
        dim3      grid(divUp(N, THREADS));                                                                 \
        histogram_##packed_type##_kernel<<<grid, block>>>(reinterpret_cast<element_type *>(in.data_ptr()), \
                                                          reinterpret_cast<int32_t *>(out.data_ptr()), N); \
        return out;                                                                                        \
    }

#define TORCH_BINDING_HISTOGRAM_2D(packed_type, torch_type, element_type, n_elements)                           \
    torch::Tensor histogram_##packed_type##_2D(torch::Tensor in)                                                \
    {                                                                                                           \
        CHECK_TORCH_TENSOR_DTYPE(in, torch_type)                                                                \
        CHECK_TORCH_TENSOR_DEVICE(in)                                                                           \
                                                                                                                \
        torch::Tensor max_val = torch::max(in).cpu();                                                           \
        const int     M       = max_val.item().to<int>();                                                       \
        auto          options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);            \
        torch::Tensor out     = torch::zeros({M + 1}, options);                                                 \
                                                                                                                \
        const int H = in.size(0);                                                                               \
        const int W = in.size(1);                                                                               \
        const int N = H * W;                                                                                    \
        dim3      block(BLOCK_SIZE_X, BLOCK_SIZE_Y);                                                            \
        dim3      grid(divUp(W, block.x *n_elements), divUp(H, block.y));                                       \
        histogram_##packed_type##_kernel2D<<<grid, block>>>(reinterpret_cast<element_type *>(in.data_ptr()),    \
                                                            reinterpret_cast<int32_t *>(out.data_ptr()), H, W); \
        return out;                                                                                             \
    }

TORCH_BINDING_HISTOGRAM(u8, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_HISTOGRAM(u8x4, torch::kUInt8, uint8_t, 4)
TORCH_BINDING_HISTOGRAM(i32, torch::kInt32, int32_t, 1)
TORCH_BINDING_HISTOGRAM(i32x4, torch::kInt32, int32_t, 4)
TORCH_BINDING_HISTOGRAM_2D(u8, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_HISTOGRAM_2D(u8x4, torch::kUInt8, uint8_t, 4)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(histogram_u8)
    TORCH_BINDING_COMMON_EXTENSION(histogram_u8x4)
    TORCH_BINDING_COMMON_EXTENSION(histogram_u8_2D)
    TORCH_BINDING_COMMON_EXTENSION(histogram_u8x4_2D)
    TORCH_BINDING_COMMON_EXTENSION(histogram_i32)
    TORCH_BINDING_COMMON_EXTENSION(histogram_i32x4)
}