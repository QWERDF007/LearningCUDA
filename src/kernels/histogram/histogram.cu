#include "common.cuh"

#include <cuda_runtime.h>

/**
 * @brief 计算uint8类型数据的直方图（基础版本）
 * @param in 输入数据指针
 * @param out 输出直方图数组指针
 * @param N 输入数据总数
 */
__global__ void histogram_u8_kernel(uint8_t *in, int32_t *out, const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    atomicAdd(&(out[in[idx]]), 1);
}

/**
 * @brief 计算uint8类型数据的直方图（2D版本）
 * @param in 输入图像数据指针
 * @param out 输出直方图数组指针
 * @param img_h 图像高度
 * @param img_w 图像宽度
 */
__global__ void histogram_u8_kernel2D(uint8_t *in, int32_t *out, const int img_h, const int img_w)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= img_w || y >= img_h)
        return;
    const int idx = x + y * img_w;
    atomicAdd(&(out[in[idx]]), 1);
}

/**
 * @brief 计算uint8类型数据的直方图（向量化版本，每次处理4个元素）
 * @param in 输入数据指针
 * @param out 输出直方图数组指针
 * @param N 输入数据总数
 */
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

/**
 * @brief 计算uint8类型数据的直方图（2D向量化版本，每次处理4个元素）
 * @param in 输入图像数据指针
 * @param out 输出直方图数组指针
 * @param img_h 图像高度
 * @param img_w 图像宽度
 */
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

/**
 * @brief 计算int32类型数据的直方图（基础版本）
 * @param in 输入数据指针
 * @param out 输出直方图数组指针
 * @param N 输入数据总数
 */
__global__ void histogram_i32_kernel(int32_t *in, int32_t *out, const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    atomicAdd(&(out[in[idx]]), 1);
}

/**
 * @brief 计算int32类型数据的直方图（向量化版本，每次处理4个元素）
 * @param in 输入数据指针
 * @param out 输出直方图数组指针
 * @param N 输入数据总数
 */
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

/**
 * @brief 使用共享内存优化的uint8直方图计算核函数（向量化版本）
 * @param in 输入数据指针
 * @param out 输出直方图数组指针
 * @param N 输入数据总数
 */
__global__ void histogram_u8x4_kernel_shared(const uint8_t *__restrict__ src, int32_t *dst, const int N)
{
    // 每个线程处理多个元素
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= N)
        return;

    __shared__ uint32_t s_hist[256];

    // 初始化共享直方图
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads();

    for (; idx + 3 < N; idx += 4 * blockDim.x * gridDim.x)
    {
        uchar4 v = *reinterpret_cast<const uchar4 *>(&src[idx]);
        atomicAdd(&s_hist[v.x], 1);
        atomicAdd(&s_hist[v.y], 1);
        atomicAdd(&s_hist[v.z], 1);
        atomicAdd(&s_hist[v.w], 1);
    }
    // 处理剩余元素
    for (; idx < N; idx++)
    {
        atomicAdd(&s_hist[src[idx]], 1);
    }
    __syncthreads();

    // 将共享直方图写回全局
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
    {
        atomicAdd(&dst[i], s_hist[i]);
    }
}

/**
 * @brief 使用共享内存优化的uint8直方图计算
 * @param in 输入张量
 * @return 输出直方图张量
 */
torch::Tensor histogram_u8x4_shared(torch::Tensor in)
{
    CHECK_TORCH_TENSOR_DTYPE(in, torch::kUInt8)
    CHECK_TORCH_TENSOR_DEVICE(in)
    auto          options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);
    torch::Tensor out     = torch::zeros({256}, options);

    const int H = in.size(0);
    const int W = in.size(1);
    const int N = H * W;
    dim3      block(THREADS);
    dim3      grid(divUp(N, THREADS * 4));
    histogram_u8x4_kernel_shared<<<grid, block>>>(reinterpret_cast<uint8_t *>(in.data_ptr()),
                                                  reinterpret_cast<int32_t *>(out.data_ptr()), N);
    return out;
}

/**
 * @brief 使用warp级别优化的uint8直方图计算核函数
 * @tparam BLOCK_SIZE 线程块大小，必须是32的整数倍
 * @param in 输入数据指针
 * @param out 输出直方图数组指针
 * @param N 输入数据总数
 */
template<int BLOCK_SIZE>
__global__ void histogram_u8x4_warp_kernel(const uint8_t *__restrict__ in, int32_t *__restrict__ out, const int N)
{
    constexpr int WARP_SIZE       = 32;
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

    // 为每个 warp 分配一份 256-bin 的直方图
    __shared__ int s_hist[WARPS_PER_BLOCK * 256];

    // 清零共享内存直方图（分片初始化）
    for (int i = threadIdx.x; i < WARPS_PER_BLOCK * 256; i += BLOCK_SIZE) s_hist[i] = 0;
    __syncthreads();

    // 每个线程归属的 warp
    const int lane   = threadIdx.x & (WARP_SIZE - 1);
    const int warpId = threadIdx.x >> 5;

    int *warp_hist = s_hist + warpId * 256;

    // -------- 向量化加载 (uchar4) + grid-stride --------
    const int global_thread = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int total_threads = gridDim.x * BLOCK_SIZE;

    // 处理按 4 对齐的主干数据
    const int N4 = (N >> 2); // 能整除的 uchar4 数量
    for (int idx4 = global_thread; idx4 < N4; idx4 += total_threads)
    {
        // 由于 idx = 4 * idx4，因此天然 4 字节对齐；in 通常由 cudaMalloc 分配，足够对齐
        const uchar4 v = reinterpret_cast<const uchar4 *>(in)[idx4];

        // 更新 warp 私有直方图（共享内存里的原子加，延迟低且只在 warp 内竞争）
        atomicAdd(&warp_hist[v.x], 1);
        atomicAdd(&warp_hist[v.y], 1);
        atomicAdd(&warp_hist[v.z], 1);
        atomicAdd(&warp_hist[v.w], 1);
    }

    // 处理尾部不足 4 个元素的数据（最多 3 个）
    const int tail_start = N4 << 2;
    for (int i = tail_start + global_thread; i < N; i += total_threads)
    {
        const uint8_t val = in[i];
        atomicAdd(&warp_hist[val], 1);
    }

    __syncthreads();

    // -------- 归并各 warp 的局部直方图并写回全局 --------
    // 使用 block 内线程分担 256 个 bin 的合并工作
    for (int bin = threadIdx.x; bin < 256; bin += BLOCK_SIZE)
    {
        int sum = 0;
#pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) sum += s_hist[w * 256 + bin];

        if (sum)
            atomicAdd(&out[bin], sum);
    }
}

torch::Tensor histogram_u8x4_warp(torch::Tensor in)
{
    CHECK_TORCH_TENSOR_DTYPE(in, torch::kUInt8)
    CHECK_TORCH_TENSOR_DEVICE(in)
    auto          options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);
    torch::Tensor out     = torch::zeros({256}, options);

    const int H     = in.size(0);
    const int W     = in.size(1);
    const int N     = H * W;
    const int block = THREADS; // 512

    // Grid大小设置策略：
    // 1. 每个线程处理4个元素，所以有效工作量是 N/4
    // 2. 使用适度的grid大小以平衡负载和原子操作竞争
    // 3. 通常设置为SM数量的倍数，RTX 4090有128个SM
    const int elements_per_thread = 4;
    const int min_grid            = divUp(N, block * elements_per_thread);
    const int max_grid            = 128 * 4; // SM数量的4倍，经验值
    const int grid                = (min_grid < max_grid) ? min_grid : max_grid;

    histogram_u8x4_warp_kernel<THREADS><<<grid, block>>>(reinterpret_cast<const uint8_t *>(in.data_ptr()),
                                                         reinterpret_cast<int32_t *>(out.data_ptr()), N);
    return out;
}

#define TORCH_BINDING_HISTOGRAM_U8(packed_type, torch_type, element_type, n_elements)                      \
    torch::Tensor histogram_##packed_type(torch::Tensor in)                                                \
    {                                                                                                      \
        CHECK_TORCH_TENSOR_DTYPE(in, torch_type)                                                           \
        CHECK_TORCH_TENSOR_DEVICE(in)                                                                      \
                                                                                                           \
        auto          options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);       \
        torch::Tensor out     = torch::zeros({256}, options);                                              \
                                                                                                           \
        const int H = in.size(0);                                                                          \
        const int W = in.size(1);                                                                          \
        const int N = H * W;                                                                               \
        dim3      block(THREADS);                                                                          \
        dim3      grid(divUp(N, THREADS *n_elements));                                                     \
        histogram_##packed_type##_kernel<<<grid, block>>>(reinterpret_cast<element_type *>(in.data_ptr()), \
                                                          reinterpret_cast<int32_t *>(out.data_ptr()), N); \
        return out;                                                                                        \
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
        dim3      block(THREADS);                                                                          \
        dim3      grid(divUp(N, THREADS *n_elements));                                                     \
        histogram_##packed_type##_kernel<<<grid, block>>>(reinterpret_cast<element_type *>(in.data_ptr()), \
                                                          reinterpret_cast<int32_t *>(out.data_ptr()), N); \
        return out;                                                                                        \
    }

#define TORCH_BINDING_HISTOGRAM_2D_U8(packed_type, torch_type, element_type, n_elements)                        \
    torch::Tensor histogram_##packed_type##_2D(torch::Tensor in)                                                \
    {                                                                                                           \
        CHECK_TORCH_TENSOR_DTYPE(in, torch_type)                                                                \
        CHECK_TORCH_TENSOR_DEVICE(in)                                                                           \
                                                                                                                \
        auto          options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);            \
        torch::Tensor out     = torch::zeros({256}, options);                                                   \
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

TORCH_BINDING_HISTOGRAM_U8(u8, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_HISTOGRAM_U8(u8x4, torch::kUInt8, uint8_t, 4)
TORCH_BINDING_HISTOGRAM(i32, torch::kInt32, int32_t, 1)
TORCH_BINDING_HISTOGRAM(i32x4, torch::kInt32, int32_t, 4)
TORCH_BINDING_HISTOGRAM_2D_U8(u8, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_HISTOGRAM_2D_U8(u8x4, torch::kUInt8, uint8_t, 4)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(histogram_u8)
    TORCH_BINDING_COMMON_EXTENSION(histogram_u8x4)
    TORCH_BINDING_COMMON_EXTENSION(histogram_u8_2D)
    TORCH_BINDING_COMMON_EXTENSION(histogram_u8x4_2D)
    TORCH_BINDING_COMMON_EXTENSION(histogram_u8x4_shared)
    TORCH_BINDING_COMMON_EXTENSION(histogram_u8x4_warp)
    TORCH_BINDING_COMMON_EXTENSION(histogram_i32)
    TORCH_BINDING_COMMON_EXTENSION(histogram_i32x4)
}