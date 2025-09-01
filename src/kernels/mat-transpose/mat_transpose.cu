#include "common.cuh"

#define WARP_SIZE 256

/**
 * @brief 矩阵转置, y[col][row] = x[row][col], HW -> WH
 */
__global__ void mat_transpose_f32_kernel(float *x, float *y, const int H, const int W)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= H * W)
        return;
    const int row    = tid / W;
    const int col    = tid % W;
    y[col * H + row] = x[tid];
}

#define TORCH_BINDING_MAT_TRANSPOSE(tag, th_type, element_type, n_pack)                                      \
    void mat_transpose_##tag(torch::Tensor x, torch::Tensor y)                                               \
    {                                                                                                        \
        CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                                               \
        CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                                               \
        const int H = x.size(0);                                                                             \
        const int W = x.size(1);                                                                             \
        const int N = H * W;                                                                                 \
        dim3      block(WARP_SIZE);                                                                          \
        dim3      grid(divUp(N, WARP_SIZE) / n_pack);                                                        \
        mat_transpose_##tag##_kernel<<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),        \
                                                      reinterpret_cast<element_type *>(y.data_ptr()), H, W); \
    }

// 1d index
TORCH_BINDING_MAT_TRANSPOSE(f32, torch::kFloat32, float, 1)


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32)
}