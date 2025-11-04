#include "common.cuh"

template<typename T>
__global__ void elementwise_sub_kernel(T *a, T *b, T *c, const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    c[idx] = saturate_cast<T>(a[idx] - b[idx]);
}

template<typename T>
__global__ void elementwise_sub_2D_kernel(T *a, T *b, T *c, const int H, const int W, const int N)
{
    const int x   = blockIdx.x * blockDim.x + threadIdx.x;
    const int y   = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = y * W + x;
    if (idx >= N)
        return;
    c[idx] = saturate_cast<T>(a[idx] - b[idx]);
}

template<typename T>
__global__ void elementwise_sub_32bit_kernel(const T *__restrict__ a, const T *__restrict__ b, T *c, const int N)
{
    if constexpr (std::is_same_v<T, float>)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N)
            return;
        c[idx] = a[idx] - b[idx];
    }
    else if constexpr (std::is_same_v<T, uint8_t>)
    {
        const int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
        if (idx >= N)
            return;
        uint8_t pack_a[4], pack_b[4], pack_c[4];
        *reinterpret_cast<uint32_t *>(pack_a) = *reinterpret_cast<const uint32_t *>(&(a[idx]));
        *reinterpret_cast<uint32_t *>(pack_b) = *reinterpret_cast<const uint32_t *>(&(b[idx]));
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            pack_c[i] = saturate_cast<uint8_t>(pack_a[i] - pack_b[i]);
        }
        *reinterpret_cast<uint32_t *>(&(c[idx])) = *reinterpret_cast<uint32_t *>(pack_c);
    }
}

template<typename T>
__global__ void elementwise_sub_32bit_2D_kernel(const T *__restrict__ a, const T *__restrict__ b, T *c, const int H,
                                                const int W, const int N)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if constexpr (std::is_same_v<T, float>)
    {
        const int idx = y * W + x;
        if (idx >= N)
            return;
        c[idx] = a[idx] - b[idx];
    }
    else if constexpr (std::is_same_v<T, uint8_t>)
    {
        const int idx = y * W + 4 * x;
        if (idx >= N)
            return;

        uint8_t pack_a[4], pack_b[4], pack_c[4];
        *reinterpret_cast<uint32_t *>(pack_a) = *reinterpret_cast<const uint32_t *>(&(a[idx]));
        *reinterpret_cast<uint32_t *>(pack_b) = *reinterpret_cast<const uint32_t *>(&(b[idx]));
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            pack_c[i] = saturate_cast<uint8_t>(pack_a[i] - pack_b[i]);
        }
        *reinterpret_cast<uint32_t *>(&(c[idx])) = *reinterpret_cast<uint32_t *>(pack_c);
    }
}

template<typename T>
__global__ void elementwise_sub_128bit_kernel(const T *__restrict__ a, const T *__restrict__ b, T *c, const int N)
{
    if constexpr (std::is_same_v<T, float>)
    {
        const int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
        if (idx >= N)
            return;

        float4 pack_a = *reinterpret_cast<const float4 *>(&(a[idx]));
        float4 pack_b = *reinterpret_cast<const float4 *>(&(b[idx]));
        float4 pack_c;

        pack_c.x = pack_a.x - pack_b.x;
        pack_c.y = pack_a.y - pack_b.y;
        pack_c.z = pack_a.z - pack_b.z;
        pack_c.w = pack_a.w - pack_b.w;

        *reinterpret_cast<float4 *>(&(c[idx])) = pack_c;
    }
    else if constexpr (std::is_same_v<T, uint8_t>)
    {
        const int idx = 16 * (blockIdx.x * blockDim.x + threadIdx.x);
        if (idx >= N)
            return;

        uint8_t pack_a[16], pack_b[16], pack_c[16];
        *reinterpret_cast<uint4 *>(pack_a) = *reinterpret_cast<const uint4 *>(&(a[idx]));
        *reinterpret_cast<uint4 *>(pack_b) = *reinterpret_cast<const uint4 *>(&(b[idx]));
#pragma unroll
        for (int i = 0; i < 16; i++)
        {
            pack_c[i] = saturate_cast<T>(pack_a[i] - pack_b[i]);
        }
        *reinterpret_cast<uint4 *>(&(c[idx])) = *reinterpret_cast<uint4 *>(pack_c);

        // uint4 pack_a = __ldg(reinterpret_cast<const uint4 *>(&(a[idx])));
        // uint4 pack_b = __ldg(reinterpret_cast<const uint4 *>(&(b[idx])));
        // uint4 pack_c;

        // pack_c.x = __vsub4(pack_a.x, pack_b.x);
        // pack_c.y = __vsub4(pack_a.y, pack_b.y);
        // pack_c.z = __vsub4(pack_a.z, pack_b.z);
        // pack_c.w = __vsub4(pack_a.w, pack_b.w);

        // *reinterpret_cast<uint4 *>(&(c[idx])) = pack_c;
    }
}

template<typename T>
__global__ void elementwise_sub_128bit_2D_kernel(const T *__restrict__ a, const T *__restrict__ b, T *c, const int H,
                                                 const int W, const int N)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if constexpr (std::is_same_v<T, float>)
    {
        const int idx = y * W + 4 * x;
        if (idx >= N)
            return;

        float4 pack_a = *reinterpret_cast<const float4 *>(&(a[idx]));
        float4 pack_b = *reinterpret_cast<const float4 *>(&(b[idx]));
        float4 pack_c;

        pack_c.x = pack_a.x - pack_b.x;
        pack_c.y = pack_a.y - pack_b.y;
        pack_c.z = pack_a.z - pack_b.z;
        pack_c.w = pack_a.w - pack_b.w;

        *reinterpret_cast<float4 *>(&(c[idx])) = pack_c;
    }
    else if constexpr (std::is_same_v<T, uint8_t>)
    {
        const int idx = y * W + 16 * x;
        if (idx >= N)
            return;

        uint8_t pack_a[16], pack_b[16], pack_c[16];
        *reinterpret_cast<uint4 *>(pack_a) = *reinterpret_cast<const uint4 *>(&(a[idx]));
        *reinterpret_cast<uint4 *>(pack_b) = *reinterpret_cast<const uint4 *>(&(b[idx]));
#pragma unroll
        for (int i = 0; i < 16; i++)
        {
            pack_c[i] = saturate_cast<uint8_t>(pack_a[i] - pack_b[i]);
        }
        *reinterpret_cast<uint4 *>(&(c[idx])) = *reinterpret_cast<uint4 *>(pack_c);

        // uint4 pack_a = __ldg(reinterpret_cast<const uint4 *>(&(a[idx])));
        // uint4 pack_b = __ldg(reinterpret_cast<const uint4 *>(&(b[idx])));
        // uint4 pack_c;

        // pack_c.x = __vsub4(pack_a.x, pack_b.x);
        // pack_c.y = __vsub4(pack_a.y, pack_b.y);
        // pack_c.z = __vsub4(pack_a.z, pack_b.z);
        // pack_c.w = __vsub4(pack_a.w, pack_b.w);

        // *reinterpret_cast<uint4 *>(&(c[idx])) = pack_c;
    }
}

#define TORCH_BINDING_ELEM_SUB(torch_type, element_type, n_pack)                                                  \
    void elementwise_sub_##element_type(torch::Tensor a, torch::Tensor b, torch::Tensor c)                        \
    {                                                                                                             \
        CHECK_TORCH_TENSOR_DTYPE(a, torch_type)                                                                   \
        CHECK_TORCH_TENSOR_DTYPE(b, torch_type)                                                                   \
        CHECK_TORCH_TENSOR_DTYPE(c, torch_type)                                                                   \
                                                                                                                  \
        CHECK_TORCH_TENSOR_DEVICE(a)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(b)                                                                              \
        CHECK_TORCH_TENSOR_DEVICE(c)                                                                              \
                                                                                                                  \
        const int N = a.numel();                                                                                  \
        dim3      block(THREADS / n_pack);                                                                        \
        dim3      grid((N + THREADS - 1) / THREADS);                                                              \
        elementwise_sub_kernel<element_type><<<grid, block>>>(reinterpret_cast<element_type *>(a.data_ptr()),     \
                                                              reinterpret_cast<element_type *>(b.data_ptr()),     \
                                                              reinterpret_cast<element_type *>(c.data_ptr()), N); \
    }

#define TORCH_BINDING_ELEM_SUB_2D(torch_type, element_type, n_pack)                                         \
    void elementwise_sub_2D_##element_type(torch::Tensor a, torch::Tensor b, torch::Tensor c)               \
    {                                                                                                       \
        CHECK_TORCH_TENSOR_DTYPE(a, torch_type)                                                             \
        CHECK_TORCH_TENSOR_DTYPE(b, torch_type)                                                             \
        CHECK_TORCH_TENSOR_DTYPE(c, torch_type)                                                             \
                                                                                                            \
        CHECK_TORCH_TENSOR_DEVICE(a)                                                                        \
        CHECK_TORCH_TENSOR_DEVICE(b)                                                                        \
        CHECK_TORCH_TENSOR_DEVICE(c)                                                                        \
                                                                                                            \
        const int H = a.size(0);                                                                            \
        const int W = a.size(1);                                                                            \
        const int N = H * W;                                                                                \
        dim3      block(32, 32);                                                                            \
        dim3      grid(divUp(W, block.x *n_pack), divUp(H, block.y));                                       \
        elementwise_sub_2D_kernel<element_type><<<grid, block>>>(                                           \
            reinterpret_cast<element_type *>(a.data_ptr()), reinterpret_cast<element_type *>(b.data_ptr()), \
            reinterpret_cast<element_type *>(c.data_ptr()), H, W, N);                                       \
    }

#define TORCH_BINDING_ELEM_SUB_TEMPLATE(tag, torch_type, element_type, n_pack)                              \
    void elementwise_sub_##tag##_##element_type(torch::Tensor a, torch::Tensor b, torch::Tensor c)          \
    {                                                                                                       \
        CHECK_TORCH_TENSOR_DTYPE(a, torch_type)                                                             \
        CHECK_TORCH_TENSOR_DTYPE(b, torch_type)                                                             \
        CHECK_TORCH_TENSOR_DTYPE(c, torch_type)                                                             \
                                                                                                            \
        CHECK_TORCH_TENSOR_DEVICE(a)                                                                        \
        CHECK_TORCH_TENSOR_DEVICE(b)                                                                        \
        CHECK_TORCH_TENSOR_DEVICE(c)                                                                        \
                                                                                                            \
        const int N = a.numel();                                                                            \
        dim3      block(THREADS / n_pack);                                                                  \
        dim3      grid((N + THREADS - 1) / THREADS);                                                        \
        elementwise_sub_##tag##_kernel<element_type><<<grid, block>>>(                                      \
            reinterpret_cast<element_type *>(a.data_ptr()), reinterpret_cast<element_type *>(b.data_ptr()), \
            reinterpret_cast<element_type *>(c.data_ptr()), N);                                             \
    }

#define TORCH_BINDING_ELEM_SUB_TEMPLATE_2D(tag, torch_type, element_type, n_pack)                           \
    void elementwise_sub_##tag##_##element_type(torch::Tensor a, torch::Tensor b, torch::Tensor c)          \
    {                                                                                                       \
        CHECK_TORCH_TENSOR_DTYPE(a, torch_type)                                                             \
        CHECK_TORCH_TENSOR_DTYPE(b, torch_type)                                                             \
        CHECK_TORCH_TENSOR_DTYPE(c, torch_type)                                                             \
                                                                                                            \
        CHECK_TORCH_TENSOR_DEVICE(a)                                                                        \
        CHECK_TORCH_TENSOR_DEVICE(b)                                                                        \
        CHECK_TORCH_TENSOR_DEVICE(c)                                                                        \
                                                                                                            \
        const int H = a.size(0);                                                                            \
        const int W = a.size(1);                                                                            \
        const int N = H * W;                                                                                \
        dim3      block(32, 32);                                                                            \
        dim3      grid(divUp(W, block.x *n_pack), divUp(H, block.y));                                       \
        elementwise_sub_##tag##_kernel<element_type><<<grid, block>>>(                                      \
            reinterpret_cast<element_type *>(a.data_ptr()), reinterpret_cast<element_type *>(b.data_ptr()), \
            reinterpret_cast<element_type *>(c.data_ptr()), H, W, N);                                       \
    }

TORCH_BINDING_ELEM_SUB(torch::kFloat32, float, 1)
TORCH_BINDING_ELEM_SUB(torch::kUInt8, uint8_t, 1)

TORCH_BINDING_ELEM_SUB_2D(torch::kUInt8, uint8_t, 1)

TORCH_BINDING_ELEM_SUB_TEMPLATE(32bit, torch::kUInt8, uint8_t, 4)
TORCH_BINDING_ELEM_SUB_TEMPLATE(128bit, torch::kUInt8, uint8_t, 16)
TORCH_BINDING_ELEM_SUB_TEMPLATE(128bit, torch::kFloat32, float, 4)
TORCH_BINDING_ELEM_SUB_TEMPLATE_2D(32bit_2D, torch::kUInt8, uint8_t, 4)
TORCH_BINDING_ELEM_SUB_TEMPLATE_2D(128bit_2D, torch::kUInt8, uint8_t, 16)
TORCH_BINDING_ELEM_SUB_TEMPLATE_2D(128bit_2D, torch::kFloat32, float, 4)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(elementwise_sub_float)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_sub_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_sub_2D_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_sub_32bit_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_sub_128bit_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_sub_128bit_float)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_sub_32bit_2D_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_sub_128bit_2D_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_sub_128bit_2D_float)
}