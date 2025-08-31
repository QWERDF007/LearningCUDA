#include "common.cuh"

/**
 * @brief BORDER_REFLECT边界处理函数
 * 实现OpenCV的BORDER_REFLECT边界反射模式: fedcba|abcdefgh|hgfedcb
 * 当坐标超出边界时，通过镜像反射的方式将坐标映射回有效范围内
 * 
 * @param coord 输入坐标值，可能超出有效范围[0, size)
 * @param size 有效坐标范围的大小，坐标应在[0, size)范围内
 * @return 经过边界反射处理后的有效坐标值
 */
__device__ __forceinline__ int reflect(int coord, int size)
{
    if (coord < 0)
        return -coord - 1;
    if (coord >= size)
        return 2 * size - coord - 1;
    return coord;
}

/**
 * @brief BORDER_REFLECT_101边界处理函数
 * 实现OpenCV的BORDER_REFLECT_101边界反射模式: gfedcb|abcdefgh|gfedcba
 * 与BORDER_REFLECT的区别是边界像素不会重复，反射时跳过边界像素
 * 
 * @param coord 输入坐标值，可能超出有效范围[0, size)
 * @param size 有效坐标范围的大小，坐标应在[0, size)范围内
 * @return 经过边界反射处理后的有效坐标值
 */
__device__ __forceinline__ int reflect_101(int coord, int size)
{
    if (coord < 0)
        return -coord;
    if (coord >= size)
        return 2 * size - coord - 2;
    return coord;
}

/**
 * @brief BORDER_REFLECT_101边界处理函数的优化版本
 * 使用条件运算符替代if-else分支，可能在某些GPU架构上有更好的性能
 * 优化成 selp 指令（无显式分支），减少 divergence
 * 
 * @param coord 输入坐标值，可能超出有效范围[0, size)
 * @param size 有效坐标范围的大小，坐标应在[0, size)范围内
 * @return 经过边界反射处理后的有效坐标值
 */
__device__ __forceinline__ int reflect_101_no_brnach(int coord, int size)
{
    int res = coord;
    res     = (res < 0) ? -res : res;
    res     = (res >= size) ? 2 * size - res - 2 : res;
    return res;
}

__global__ void blur_u8_kernel(uint8_t *in, uint8_t *out, const int ks_w, const int ks_h, const int img_w,
                               const int img_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= img_w || y >= img_h)
        return;

    const int idx = y * img_w + x;

    int   half_w = ks_w / 2;
    int   half_h = ks_h / 2;
    int   sum    = 0;
    float count  = ks_w * ks_h; // 所有像素都会被处理

    for (int ky = -half_h; ky <= half_h; ++ky)
    {
        int yy = reflect_101(y + ky, img_h); // 使用BORDER_REFLECT_101
        for (int kx = -half_w; kx <= half_w; ++kx)
        {
            int xx = reflect_101(x + kx, img_w);
            sum += in[yy * img_w + xx];
        }
    }

    out[idx] = (uint8_t)roundf(sum / count);
}

__global__ void blur_u8_nb_kernel(uint8_t *in, uint8_t *out, const int ks_w, const int ks_h, const int img_w,
                                  const int img_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= img_w || y >= img_h)
        return;

    const int idx = y * img_w + x;

    int   half_w = ks_w / 2;
    int   half_h = ks_h / 2;
    int   sum    = 0;
    float count  = ks_w * ks_h;

    for (int ky = -half_h; ky <= half_h; ++ky)
    {
        int yy = reflect_101_no_brnach(y + ky, img_h);
        for (int kx = -half_w; kx <= half_w; ++kx)
        {
            int xx = reflect_101_no_brnach(x + kx, img_w);
            sum += in[yy * img_w + xx];
        }
    }

    out[idx] = (uint8_t)roundf(sum / count);
}

__global__ void blur_u8_kernel_shared(uint8_t *in, uint8_t *out, const int ks_w, const int ks_h, const int img_w,
                                      const int img_h)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x  = blockIdx.x * blockDim.x + tx;
    const int y  = blockIdx.y * blockDim.y + ty;

    const int half_w = ks_w / 2;
    const int half_h = ks_h / 2;

    // 动态共享内存
    extern __shared__ uint8_t smem[];

    // 共享内存的尺寸包含halo区域
    const int smem_w = blockDim.x + 2 * half_w;
    const int smem_h = blockDim.y + 2 * half_h;

    // 每个线程需要加载多个像素来填充整个共享内存区域
    const int total_pixels      = smem_w * smem_h;
    const int threads_per_block = blockDim.x * blockDim.y;
    const int thread_id         = ty * blockDim.x + tx;

    // 使用所有线程协作加载共享内存
    for (int i = thread_id; i < total_pixels; i += threads_per_block)
    {
        int smem_y = i / smem_w;
        int smem_x = i % smem_w;

        // 计算对应的全局坐标
        int global_x = blockIdx.x * blockDim.x + smem_x - half_w;
        int global_y = blockIdx.y * blockDim.y + smem_y - half_h;

        // 边界反射处理
        global_x = reflect_101_no_brnach(global_x, img_w);
        global_y = reflect_101_no_brnach(global_y, img_h);

        // 存储到共享内存
        smem[smem_y * smem_w + smem_x] = in[global_y * img_w + global_x];
    }

    __syncthreads();

    // 只有有效线程才进行计算
    if (x >= img_w || y >= img_h)
        return;

    int   sum   = 0;
    float count = ks_w * ks_h;

    // 计算当前线程在共享内存中的起始位置
    const int smem_start_x = tx;
    const int smem_start_y = ty;

#pragma unroll
    for (int ky = 0; ky < ks_h; ++ky)
    {
#pragma unroll
        for (int kx = 0; kx < ks_w; ++kx)
        {
            int smem_idx = (smem_start_y + ky) * smem_w + (smem_start_x + kx);
            sum += smem[smem_idx];
        }
    }
    out[y * img_w + x] = (uint8_t)roundf(sum / count);
}

void blur_u8_shared(torch::Tensor in, const int ksz, torch::Tensor out)
{
    CHECK_TORCH_TENSOR_DTYPE(in, torch::kUInt8)
    CHECK_TORCH_TENSOR_DTYPE(out, torch::kUInt8)
    CHECK_TORCH_TENSOR_DEVICE(in)
    CHECK_TORCH_TENSOR_DEVICE(out)

    const int H = in.size(0);
    const int W = in.size(1);
    const int N = H * W;
    dim3      block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3      grid(divUp(W, block.x), divUp(H, block.y));
    size_t    smem_size = (block.x + 2 * (ksz / 2)) * (block.y + 2 * (ksz / 2)) * sizeof(uint8_t);
    blur_u8_kernel_shared<<<grid, block, smem_size>>>(reinterpret_cast<uint8_t *>(in.data_ptr()),
                                                      reinterpret_cast<uint8_t *>(out.data_ptr()), ksz, ksz, W, H);
}

#define TORCH_BINDING_BLUR(packed_type, torch_type, element_type, n_elements)                                       \
    void blur_##packed_type(torch::Tensor in, const int ksz, torch::Tensor out)                                     \
    {                                                                                                               \
        CHECK_TORCH_TENSOR_DTYPE(in, torch_type)                                                                    \
        CHECK_TORCH_TENSOR_DTYPE(out, torch_type)                                                                   \
        CHECK_TORCH_TENSOR_DEVICE(in)                                                                               \
        CHECK_TORCH_TENSOR_DEVICE(out)                                                                              \
                                                                                                                    \
        const int H = in.size(0);                                                                                   \
        const int W = in.size(1);                                                                                   \
        const int N = H * W;                                                                                        \
        dim3      block(BLOCK_SIZE_X, BLOCK_SIZE_Y);                                                                \
        dim3      grid(divUp(W, block.x), divUp(H, block.y));                                                       \
                                                                                                                    \
        blur_##packed_type##_kernel<<<grid, block>>>(reinterpret_cast<element_type *>(in.data_ptr()),               \
                                                     reinterpret_cast<element_type *>(out.data_ptr()), ksz, ksz, W, \
                                                     H);                                                            \
    }

TORCH_BINDING_BLUR(u8, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_BLUR(u8_nb, torch::kUInt8, uint8_t, 1)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(blur_u8)
    TORCH_BINDING_COMMON_EXTENSION(blur_u8_nb)
    TORCH_BINDING_COMMON_EXTENSION(blur_u8_shared)
}