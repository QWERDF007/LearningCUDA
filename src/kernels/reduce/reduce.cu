#include "common.cuh"

template<typename T, typename CT>
__global__ void reduce_sum_atomic_kernel(const T *__restrict__ src, CT *__restrict__ dst, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    const int stride = blockDim.x * gridDim.x; // 256 * 1024

    CT sum = 0;

    for (int i = tid; i < N; i += stride)
    {
        sum += src[i];
    }

    {
        atomicAdd(dst, sum);
    }
}

template<typename T, typename CT>
__global__ void reduce_sum_shared_kernel(const T *__restrict__ src, CT *__restrict__ dst, const int N)
{
    extern __shared__ char sdata_raw[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. 修正：将其转换为 CT 类型的指针
    CT *sdata = reinterpret_cast<CT *>(sdata_raw);

    // 1. 加载到 Shared Memory (同时处理边界)
    // 即使 idx >= N，也要填 0，保证归约逻辑正确
    if (idx < N)
    {
        sdata[tid] = static_cast<CT>(src[idx]);
    }
    else
    {
        sdata[tid] = 0;
    }
    __syncthreads();

    // 2. Shared Memory 归约
    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 3. 写回全局内存
    if (tid == 0)
    {
        dst[blockIdx.x] = sdata[0];
    }
}

#define TORCH_BINDING_REDUCE(tag, th_type, element_type, cal_type, n_pack)                                      \
    torch::Tensor reduce_##tag##_##element_type##_##cal_type(torch::Tensor src)                                 \
    {                                                                                                           \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                          \
                                                                                                                \
        auto          options = torch::TensorOptions().dtype(th_type).device(torch::kCUDA, 0);                  \
        torch::Tensor dst     = torch::zeros({1, 1}, options);                                                  \
                                                                                                                \
        const int H = src.size(0);                                                                              \
        const int W = src.size(1);                                                                              \
        const int N = H * W;                                                                                    \
                                                                                                                \
        dim3 block(THREADS);                                                                                    \
        dim3 grid(min(divUp(N, THREADS), 128));                                                                 \
                                                                                                                \
        reduce_##tag##_kernel<element_type, cal_type><<<grid, block>>>(                                         \
            reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<cal_type *>(dst.data_ptr()), N); \
                                                                                                                \
        return dst;                                                                                             \
    }

#define TORCH_BINDING_REDUCE_SHARED(tag, th_type, element_type, cal_type, n_pack)                                   \
    torch::Tensor reduce_##tag##_##element_type##_##cal_type(torch::Tensor src)                                     \
    {                                                                                                               \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                              \
        const int H = src.size(0);                                                                                  \
        const int W = src.size(1);                                                                                  \
        int       N = H * W;                                                                                        \
                                                                                                                    \
        dim3   block(THREADS);                                                                                      \
        dim3   grid(divUp(N, THREADS));                                                                             \
        size_t smem_size = THREADS * sizeof(cal_type);                                                              \
                                                                                                                    \
        auto          options = torch::TensorOptions().dtype(th_type).device(torch::kCUDA, 0);                      \
        torch::Tensor dst     = torch::empty({1, grid.x}, options);                                                 \
                                                                                                                    \
        {                                                                                                           \
            reduce_##tag##_kernel<element_type, cal_type><<<grid, block, smem_size>>>(                              \
                reinterpret_cast<element_type *>(src.data_ptr()), reinterpret_cast<cal_type *>(dst.data_ptr()), N); \
            N = grid.x;                                                                                             \
        }                                                                                                           \
                                                                                                                    \
        if (N == 1)                                                                                                 \
            return dst;                                                                                             \
                                                                                                                    \
        while (N > 1)                                                                                               \
        {                                                                                                           \
            dim3 next_grid(divUp(N, THREADS));                                                                      \
            reduce_##tag##_kernel<cal_type, cal_type><<<next_grid, block, smem_size>>>(                             \
                reinterpret_cast<cal_type *>(dst.data_ptr()), reinterpret_cast<cal_type *>(dst.data_ptr()), N);     \
            N = next_grid.x;                                                                                        \
        }                                                                                                           \
                                                                                                                    \
        return dst.slice(0, 0, 1);                                                                                  \
    }

TORCH_BINDING_REDUCE(sum_atomic, torch::kFloat32, float, float, 1)
TORCH_BINDING_REDUCE(sum_atomic, torch::kFloat64, float, double, 1)

TORCH_BINDING_REDUCE_SHARED(sum_shared, torch::kFloat32, float, float, 1)
TORCH_BINDING_REDUCE_SHARED(sum_shared, torch::kFloat64, float, double, 1)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(reduce_sum_atomic_float_float)
    TORCH_BINDING_COMMON_EXTENSION(reduce_sum_atomic_float_double)
    TORCH_BINDING_COMMON_EXTENSION(reduce_sum_shared_float_float)
    TORCH_BINDING_COMMON_EXTENSION(reduce_sum_shared_float_double)
}