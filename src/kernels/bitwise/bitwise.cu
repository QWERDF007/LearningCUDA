#include "common.cuh"

template<typename T, int CH>
__global__ void bitwise_and_kernel(const T *__restrict__ src1, const T *__restrict__ src2, T *__restrict__ dst,
                                   const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int base = tid * CH;
#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        const int idx = base + c;
        dst[idx]      = src1[idx] & src2[idx];
    }
}

template<typename T, int CH>
__global__ void bitwise_and_u8x4_kernel(const T *__restrict__ src1, const T *__restrict__ src2, T *__restrict__ dst,
                                        const int N)
{
    const int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= N)
        return;

    // 如果剩余元素足够4个，使用向量化处理
    if (idx + 3 < N)
    {
        const uchar4 s1 = *reinterpret_cast<const uchar4 *>(&(src1[idx]));
        const uchar4 s2 = *reinterpret_cast<const uchar4 *>(&(src2[idx]));

        *reinterpret_cast<uchar4 *>(&(dst[idx])) = make_uchar4(s1.x & s2.x, s1.y & s2.y, s1.z & s2.z, s1.w & s2.w);
    }
    // 处理剩余不足4个的元素
    else
    {
        for (int i = idx; i < N; i++)
        {
            dst[i] = src1[i] & src2[i];
        }
    }
}

template<typename T, int CH>
__global__ void bitwise_and_vec_kernel(const uint8_t *__restrict__ src1, const uint8_t *__restrict__ src2,
                                       uint8_t *__restrict__ dst,
                                       const int N) // N = 像素数
{
    const int pixel_per_thread = 4; // 每次处理4个像素（每像素CH通道）

    const int tid       = blockIdx.x * blockDim.x + threadIdx.x;
    const int start_pix = tid * pixel_per_thread;
    if (start_pix >= N)
        return;

    // 每像素CH通道，共处理 pixel_per_thread * CH 字节
    const int vec_bytes = pixel_per_thread * CH; // 4 or 12
    const int vec_idx   = start_pix * CH;        // 字节起始位置

    // 将指针 reinterpret_cast 成 uchar4*
    const uchar4 *s1 = reinterpret_cast<const uchar4 *>(src1);
    const uchar4 *s2 = reinterpret_cast<const uchar4 *>(src2);
    uchar4       *d  = reinterpret_cast<uchar4 *>(dst);

    // 对齐的字节数（每个uchar4是4字节）
    const int total_vec = (N * CH + 3) / 4;
    const int vec_tid   = (start_pix * CH) / 4;

    if (vec_tid >= total_vec)
        return;

#pragma unroll
    for (int i = 0; i < (vec_bytes + 3) / 4 && (vec_tid + i) < total_vec; ++i)
    {
        uchar4 a       = s1[vec_tid + i];
        uchar4 b       = s2[vec_tid + i];
        d[vec_tid + i] = make_uchar4(a.x & b.x, a.y & b.y, a.z & b.z, a.w & b.w);
    }
}

template<typename T, int CH>
__global__ void bitwise_not_kernel(const T *__restrict__ src, T *__restrict__ dst, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int base = tid * CH;
#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        const int idx = base + c;
        dst[idx]      = ~src[idx];
    }
}

template<typename T, int CH>
__global__ void bitwise_or_kernel(const T *__restrict__ src1, const T *__restrict__ src2, T *__restrict__ dst,
                                  const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int base = tid * CH;
#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        const int idx = base + c;
        dst[idx]      = src1[idx] | src2[idx];
    }
}

template<typename T, int CH>
__global__ void bitwise_xor_kernel(const T *__restrict__ src1, const T *__restrict__ src2, T *__restrict__ dst,
                                   const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int base = tid * CH;
#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        const int idx = base + c;
        dst[idx]      = src1[idx] ^ src2[idx];
    }
}

#define TORCH_BINDING_BITWISE_TEMPLATE(tag, th_type, element_type, n_pack)                                            \
    void bitwise_##tag##_##element_type(torch::Tensor src1, torch::Tensor src2, torch::Tensor dst)                    \
    {                                                                                                                 \
        CHECK_TORCH_TENSOR_DTYPE(src1, (th_type))                                                                     \
        CHECK_TORCH_TENSOR_DTYPE(src2, (th_type))                                                                     \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                      \
        const int H  = src1.size(0);                                                                                  \
        const int W  = src1.size(1);                                                                                  \
        const int CH = src1.dim() == 2 ? 1 : src1.size(2);                                                            \
        const int N  = H * W;                                                                                         \
        dim3      block(THREADS);                                                                                     \
        dim3      grid(divUp(N, THREADS *n_pack));                                                                    \
        if (CH == 1)                                                                                                  \
        {                                                                                                             \
            bitwise_##tag##_kernel<element_type, 1><<<grid, block>>>(                                                 \
                reinterpret_cast<element_type *>(src1.data_ptr()), reinterpret_cast<element_type *>(src2.data_ptr()), \
                reinterpret_cast<element_type *>(dst.data_ptr()), N);                                                 \
        }                                                                                                             \
        else if (CH == 3)                                                                                             \
        {                                                                                                             \
            bitwise_##tag##_kernel<element_type, 3><<<grid, block>>>(                                                 \
                reinterpret_cast<element_type *>(src1.data_ptr()), reinterpret_cast<element_type *>(src2.data_ptr()), \
                reinterpret_cast<element_type *>(dst.data_ptr()), N);                                                 \
        }                                                                                                             \
    }

#define TORCH_BINDING_BITWISE_NOT_TEMPLATE(tag, th_type, element_type, n_pack)                                         \
    void bitwise_##tag##_##element_type(torch::Tensor src, torch::Tensor dst)                                          \
    {                                                                                                                  \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                       \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                       \
        const int H  = src.size(0);                                                                                    \
        const int W  = src.size(1);                                                                                    \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                               \
        const int N  = H * W;                                                                                          \
        dim3      block(THREADS);                                                                                      \
        dim3      grid(divUp(N, THREADS *n_pack));                                                                     \
        if (CH == 1)                                                                                                   \
        {                                                                                                              \
            bitwise_##tag##_kernel<element_type, 1><<<grid, block>>>(reinterpret_cast<element_type *>(src.data_ptr()), \
                                                                     reinterpret_cast<element_type *>(dst.data_ptr()), \
                                                                     N);                                               \
        }                                                                                                              \
        else if (CH == 3)                                                                                              \
        {                                                                                                              \
            bitwise_##tag##_kernel<element_type, 3><<<grid, block>>>(reinterpret_cast<element_type *>(src.data_ptr()), \
                                                                     reinterpret_cast<element_type *>(dst.data_ptr()), \
                                                                     N);                                               \
        }                                                                                                              \
    }

TORCH_BINDING_BITWISE_TEMPLATE(and, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_BITWISE_TEMPLATE(and_u8x4, torch::kUInt8, uint8_t, 4)
TORCH_BINDING_BITWISE_TEMPLATE(and_vec, torch::kUInt8, uint8_t, 4)
TORCH_BINDING_BITWISE_TEMPLATE(or, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_BITWISE_TEMPLATE(xor, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_BITWISE_NOT_TEMPLATE(not, torch::kUInt8, uint8_t, 1)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(bitwise_and_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(bitwise_and_u8x4_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(bitwise_and_vec_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(bitwise_or_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(bitwise_xor_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(bitwise_not_uint8_t)
}