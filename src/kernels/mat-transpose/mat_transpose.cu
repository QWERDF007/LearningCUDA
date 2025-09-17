#include "common.cuh"

#define WARP_SIZE  256
#define TILE_DIM   32
#define BLOCK_ROWS 32

/**
 * @brief 矩阵转置核函数 (读连续)
 * @param A 输入矩阵指针 (H x W)
 * @param B 输出矩阵指针 (W x H)
 * @param H 输入矩阵的行数
 * @param W 输入矩阵的列数
 * 
 * B[col][row] = A[row][col]  ，即 HW -> WH
 * warp 读取合并, 但写入不合并
 * 
 */
__global__ void mat_transpose_f32_coalesced_read_kernel(float *A, float *B, const int H, const int W)
{
    // 计算全局线程索引
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= H * W)
        return;

    const int ay = tid / W; // A 行索引
    const int ax = tid % W; // A 列索引

    B[ax * H + ay] = A[tid];
}

/**
 * @brief 矩阵转置核函数 (写连续)
 * 
 * B[row][col] = A[col][row]
 * 同一个 warp 合并内存写入, 虽然存在非合并内存读取, 但利用只读数据可以缓存加速非合并的访问
 */
__global__ void mat_transpose_f32_coalesced_write_kernel(float *A, float *B, const int H, const int W)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= H * W)
        return;

    const int by = tid / H; // B 行索引
    const int bx = tid % H; // B 列索引

    // B[by * H + bx] = A[bx * W + by];
    B[tid] = A[bx * W + by];
}

__global__ void mat_transpose_f32x4_coalesced_read_kernel(float *A, float *B, const int H, const int W)
{
    // 计算全局线程索引
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ay  = (idx * 4) / W; // A 行索引
    const int ax  = (idx * 4) % W; // A 列索引
    if (ay >= H || ax + 3 >= W)
        return;
    float4 a = reinterpret_cast<float4 *>(A)[idx];

    B[ax * H + ay]       = a.x;
    B[(ax + 1) * H + ay] = a.y;
    B[(ax + 2) * H + ay] = a.z;
    B[(ax + 3) * H + ay] = a.w;
}

__global__ void mat_transpose_f32x4_coalesced_write_kernel(float *A, float *B, const int H, const int W)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int by  = (idx * 4) / H; // B 行索引
    const int bx  = (idx * 4) % H; // B 列索引
    if (bx >= H || by >= W)
        return;

    float4 a;
    a.x = A[bx * W + by];
    a.y = A[(bx + 1) * W + by];
    a.z = A[(bx + 2) * W + by];
    a.w = A[(bx + 3) * W + by];

    reinterpret_cast<float4 *>(B)[idx] = a;
}

/**
 * @brief 矩阵转置核函数 (合并读, 2D 布局)
 * 
 * B[col][row] = A[row][col]
 * 同一个 warp 合并读, 但非合并写入
 */
__global__ void mat_transpose_f32_coalesced_read_2d_kernel(float *A, float *B, const int H, const int W)
{
    const int ax = blockIdx.x * blockDim.x + threadIdx.x;
    const int ay = blockIdx.y * blockDim.y + threadIdx.y;
    if (ax >= W || ay >= H)
        return;
    B[ax * H + ay] = A[ay * W + ax];
}

/**
 * @brief 矩阵转置核函数 (合并写, 2D 布局)
 * 
 * 同一个 warp 合并写入, 但非合并读取
 */
__global__ void mat_transpose_f32_coalesced_write_2d_kernel(float *A, float *B, const int H, const int W)
{
    const int ax = blockIdx.x * blockDim.x + threadIdx.x; // 列索引
    const int ay = blockIdx.y * blockDim.y + threadIdx.y; // 行索引
    if (ax >= W || ay >= H)
        return;
    const int idx = ay * W + ax;
    const int by  = idx / H;
    const int bx  = idx % H;
    // B[by * H + bx] = A[bx * W + by];
    B[idx] = A[bx * W + by];
}

/**
 * @brief 2D 布局, 合并读, 一次处理 4 个元素
 */
__global__ void mat_transpose_f32x4_coalesced_read_2d_kernel(float *A, float *B, const int H, const int W)
{
    const int x  = blockIdx.x * blockDim.x + threadIdx.x;
    const int y  = blockIdx.y * blockDim.y + threadIdx.y;
    const int ax = 4 * x;
    const int ay = y;
    if (ax + 3 >= W || ay >= H)
        return;
    float4 a = reinterpret_cast<float4 *>(A)[ay * W / 4 + x];

    B[ax * H + ay]       = a.x;
    B[(ax + 1) * H + ay] = a.y;
    B[(ax + 2) * H + ay] = a.z;
    B[(ax + 3) * H + ay] = a.w;
}

/**
 * @brief 2D 布局, 合并写, 一次处理 4 个元素
 */
__global__ void mat_transpose_f32x4_coalesced_write_2d_kernel(float *A, float *B, const int H, const int W)
{
    const int x  = blockIdx.x * blockDim.x + threadIdx.x;
    const int y  = blockIdx.y * blockDim.y + threadIdx.y;
    const int ax = x;
    const int ay = y * 4;
    if (ax >= W || ay + 3 >= H)
        return;
    float4 a;
    a.x = A[ay * W + ax];
    a.y = A[(ay + 1) * W + ax];
    a.z = A[(ay + 2) * W + ax];
    a.w = A[(ay + 3) * W + ax];

    reinterpret_cast<float4 *>(B)[x * H / 4 + y] = a;
}

/**
 * @brief 矩阵转置核函数，使用二维线程索引（版本2）
 * 
 */
__global__ void mat_transpose_f32_diagnonal_2d_kernel(float *A, float *B, const int H, const int W)
{
    const int block_y    = blockIdx.x;
    const int block_x    = (blockIdx.x + blockIdx.y) % gridDim.x;
    const int global_col = threadIdx.x + blockDim.x * block_x;
    const int global_row = threadIdx.y + blockDim.y * block_y;
    if (global_col < W && global_row < H)
    {
        B[global_row * W + global_col] = A[global_col * H + global_row];
    }
}

/**
 * @brief 使用共享内存优化的矩阵转置核函数
 * @param A 输入矩阵指针 (H x W)
 * @param B 输出矩阵指针 (W x H)
 * @param H 输入矩阵的行数
 * @param W 输入矩阵的列数
 * 
 * 该核函数使用共享内存来优化矩阵转置操作，通过分块处理和内存合并访问提高性能
 * 采用分块策略，先将数据加载到共享内存，再以转置的方式写回全局内存
 */
__global__ void mat_transpose_f32_2d_shared_kernel(const float *A, float *B, int H, int W)
{
    // 声明共享内存，添加 +1 填充以避免 bank 冲突
    // TILE_DIM x (TILE_DIM + 1) 的二维数组
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // 计算输入矩阵 A (cols = W, rows = H) 中的全局坐标
    // x 对应列索引，y 对应行索引
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x; // 全局列索引
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y; // 全局行索引

    // 第一阶段：从全局内存读取数据到共享内存
    // 以 BLOCK_ROWS 为步长循环遍历 tile 内的行，总共覆盖 TILE_DIM 行
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        int y = yIndex + j; // 当前处理的行索引

        if (xIndex < W && y < H)
        {
            // 从全局内存读取数据到共享内存
            // A 按行优先存储：A[row * W + col]
            // 存储到共享内存：tile[threadIdx.y + j][threadIdx.x]
            tile[threadIdx.y + j][threadIdx.x] = A[y * W + xIndex];
        }
        else
        {
            // 越界处理：填充零值，避免读取到过时数据
            // 虽然非必要，但提高安全性
            tile[threadIdx.y + j][threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

    // 第二阶段：计算转置后的坐标索引
    // 通过交换 blockIdx.x 和 blockIdx.y 来转置块的坐标
    int transposed_xIndex = blockIdx.y * TILE_DIM + threadIdx.x; // 转置后的列索引
    int transposed_yIndex = blockIdx.x * TILE_DIM + threadIdx.y; // 转置后的行索引

    // 第三阶段：将转置后的数据从共享内存写入全局内存
    // 输出矩阵 B 的尺寸为 W x H，按行优先存储
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        int yOut = transposed_yIndex + j; // 输出矩阵 B 中的行索引（范围 0 到 W-1）
        int xOut = transposed_xIndex;     // 输出矩阵 B 中的列索引（范围 0 到 H-1）

        // 边界检查，确保不越界
        if (xOut < H && yOut < W)
        {
            // 从共享内存读取数据并写入全局内存
            // 注意：读取共享内存时交换了索引，实现转置
            // tile[threadIdx.x][threadIdx.y + j] 对应转置后的数据
            // B 按行优先存储：B[yOut * H + xOut]
            B[yOut * H + xOut] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

__global__ void mat_transpose_f32_2d_shared_2_kernel(const float *A, float *B, int H, int W)
{
    // 声明共享内存
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x; // 全局列索引
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y; // 全局行索引

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        int y = yIndex + j; // 当前处理的行索引

        if (xIndex < W && y < H)
        {
            tile[threadIdx.y + j][threadIdx.x] = A[y * W + xIndex];
        }
        else
        {
            tile[threadIdx.y + j][threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

    int transposed_xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    int transposed_yIndex = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        int yOut = transposed_yIndex + j;
        int xOut = transposed_xIndex;

        if (xOut < H && yOut < W)
        {
            B[yOut * H + xOut] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
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

#define TORCH_BINDING_MAT_TRANSPOSE_2D(tag, th_type, element_type, n_pack_h, n_pack_w)                       \
    void mat_transpose_##tag(torch::Tensor x, torch::Tensor y)                                               \
    {                                                                                                        \
        CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                                               \
        CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                                               \
        const int H = x.size(0);                                                                             \
        const int W = x.size(1);                                                                             \
        const int N = H * W;                                                                                 \
        dim3      block(TILE_DIM, BLOCK_ROWS);                                                               \
        dim3      grid(divUp(W, block.x *n_pack_w), divUp(H, block.y *n_pack_h));                            \
        mat_transpose_##tag##_kernel<<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),        \
                                                      reinterpret_cast<element_type *>(y.data_ptr()), H, W); \
    }

// 1d index
TORCH_BINDING_MAT_TRANSPOSE(f32_coalesced_read, torch::kFloat32, float, 1)
TORCH_BINDING_MAT_TRANSPOSE(f32_coalesced_write, torch::kFloat32, float, 1)
TORCH_BINDING_MAT_TRANSPOSE(f32x4_coalesced_read, torch::kFloat32, float, 4)
TORCH_BINDING_MAT_TRANSPOSE(f32x4_coalesced_write, torch::kFloat32, float, 4)

TORCH_BINDING_MAT_TRANSPOSE_2D(f32_coalesced_read_2d, torch::kFloat32, float, 1, 1)
TORCH_BINDING_MAT_TRANSPOSE_2D(f32_coalesced_write_2d, torch::kFloat32, float, 1, 1)
TORCH_BINDING_MAT_TRANSPOSE_2D(f32x4_coalesced_read_2d, torch::kFloat32, float, 1, 4)
TORCH_BINDING_MAT_TRANSPOSE_2D(f32x4_coalesced_write_2d, torch::kFloat32, float, 4, 1)
// TORCH_BINDING_MAT_TRANSPOSE_2D(f32_2d_2, torch::kFloat32, float, 1)
// TORCH_BINDING_MAT_TRANSPOSE_2D(f32_2d_3, torch::kFloat32, float, 1)
TORCH_BINDING_MAT_TRANSPOSE_2D(f32_2d_shared, torch::kFloat32, float, 1, 1)
TORCH_BINDING_MAT_TRANSPOSE_2D(f32_2d_shared_2, torch::kFloat32, float, 1, 1)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_coalesced_read)
    TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_coalesced_write)
    TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_coalesced_read)
    TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_coalesced_write)

    TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_coalesced_read_2d)
    TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_coalesced_write_2d)
    TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_coalesced_read_2d)
    TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_coalesced_write_2d)
    // TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_2d_2)
    // TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_2d_3)
    TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_2d_shared)
    TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_2d_shared_2)
}