#include "common.cuh"

template<typename T, int CH>
__global__ void get_channel_kernel(const T *__restrict__ src, T *__restrict__ dst, const int cidx, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    const int base = tid * CH;
    dst[tid]       = src[base + cidx];
}

template<typename T, int CH>
__global__ void channel_subtract_kernel(const T *__restrict__ src, T *__restrict__ dst, const int i, const int j,
                                        const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    const int base = tid * CH;
    dst[tid]       = saturate_cast<T>(src[base + i] - src[base + j]);
}

void get_single_channel_uint8_t(torch::Tensor src, torch::Tensor dst, const int c1, const int c2)
{
    CHECK_TORCH_TENSOR_DTYPE(src, (torch::kUInt8))
    CHECK_TORCH_TENSOR_DTYPE(dst, (torch::kUInt8))
    CHECK_TORCH_TENSOR_DEVICE(src)
    CHECK_TORCH_TENSOR_DEVICE(dst)

    const int H  = src.size(0);
    const int W  = src.size(1);
    const int CH = src.dim() == 2 ? 1 : src.size(2);
    const int N  = H * W;

    dim3 block(THREADS);
    dim3 grid(divUp(N, THREADS));
    get_channel_kernel<uint8_t, 3><<<grid, block>>>(reinterpret_cast<uint8_t *>(src.data_ptr()),
                                                    reinterpret_cast<uint8_t *>(dst.data_ptr()), c1, N);
}

void channel_subtract_uint8_t(torch::Tensor src, torch::Tensor dst, const int c1, const int c2)
{
    CHECK_TORCH_TENSOR_DTYPE(src, (torch::kUInt8))
    CHECK_TORCH_TENSOR_DTYPE(dst, (torch::kUInt8))
    CHECK_TORCH_TENSOR_DEVICE(src)
    CHECK_TORCH_TENSOR_DEVICE(dst)

    const int H  = src.size(0);
    const int W  = src.size(1);
    const int CH = src.dim() == 2 ? 1 : src.size(2);
    const int N  = H * W;

    dim3 block(THREADS);
    dim3 grid(divUp(N, THREADS));
    channel_subtract_kernel<uint8_t, 3><<<grid, block>>>(reinterpret_cast<uint8_t *>(src.data_ptr()),
                                                         reinterpret_cast<uint8_t *>(dst.data_ptr()), c1, c2, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(get_single_channel_uint8_t)
    TORCH_BINDING_COMMON_EXTENSION(channel_subtract_uint8_t)
}