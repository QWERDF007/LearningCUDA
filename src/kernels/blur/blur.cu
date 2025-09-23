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
 * @note 性能优化高度依赖于具体的硬件架构和软件环境
 * 在 win11 + wsl Ubuntu 22.04 + 4090 vs  Ubuntu 18.04 + 4090D 测试下, 4090 无分支版本更慢
 * 
 * @param coord 输入坐标值，可能超出有效范围[0, size)
 * @param size 有效坐标范围的大小，坐标应在[0, size)范围内
 * @return 经过边界反射处理后的有效坐标值
 */
__device__ __forceinline__ int reflect_101_no_branch(int coord, int size)
{
    int res = (coord < 0) ? -coord : coord;
    res     = (res >= size) ? 2 * size - res - 2 : res;
    return res;
}

/**
 * @brief 通用的多通道模糊核函数模板
 * 支持任意数据类型和通道数的模糊处理
 * 
 * @tparam T 像素数据类型（如uint8_t, float等）
 * @tparam CT 计算中间值类型（如int, float等，用于避免溢出）
 * @tparam CH 图像通道数（1=灰度图，3=RGB，4=RGBA等）
 * @param src 输入图像数据指针
 * @param dst 输出图像数据指针  
 * @param ks_h 卷积核高度
 * @param ks_w 卷积核宽度
 * @param img_h 图像高度
 * @param img_w 图像宽度
 * @param N 总像素数量（img_h * img_w）
 */
template<typename T, typename CT, int CH>
__global__ void blur_kernel(T *src, T *dst, const int ks_h, const int ks_w, const int img_h, const int img_w,
                            const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const int x = idx % img_w;
    const int y = idx / img_w;

    const int    half_w = ks_w / 2;
    const int    half_h = ks_h / 2;
    const double count  = ks_h * ks_w;

    // const int base = y * img_w * CH + x * CH;
    const int base = idx * CH;

// 处理每个通道
#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        CT sum = 0;

        // 遍历卷积核窗口
        for (int ky = -half_h; ky <= half_h; ++ky)
        {
            int yy = reflect_101(y + ky, img_h);
            for (int kx = -half_w; kx <= half_w; ++kx)
            {
                int xx = reflect_101(x + kx, img_w);
                sum += src[(yy * img_w + xx) * CH + c];
            }
        }

        dst[base + c] = saturate_cast<T>(sum / count);
    }
}

/**
 * @brief 基础的2D模糊核函数
 * 使用全局内存访问，每个线程处理一个像素
 * 
 * @param in 输入图像数据指针
 * @param out 输出图像数据指针
 * @param ks_w 卷积核宽度
 * @param ks_h 卷积核高度
 * @param img_w 图像宽度
 * @param img_h 图像高度
 */
__global__ void blur_u8_kernel(uint8_t *in, uint8_t *out, const int ks_w, const int ks_h, const int img_w,
                               const int img_h)
{
    // 计算当前线程对应的像素坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查，超出图像范围的线程直接返回
    if (x >= img_w || y >= img_h)
        return;

    const int idx = y * img_w + x;

    int   half_w = ks_w / 2;
    int   half_h = ks_h / 2;
    int   sum    = 0;
    float count  = ks_w * ks_h; // 所有像素都会被处理

    // 遍历卷积核窗口
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

/**
 * @brief 无分支版本的2D模糊核函数
 * 使用无分支的边界处理函数，可能在某些GPU架构上有更好的性能
 * 
 * @param in 输入图像数据指针
 * @param out 输出图像数据指针
 * @param ks_w 卷积核宽度
 * @param ks_h 卷积核高度
 * @param img_w 图像宽度
 * @param img_h 图像高度
 */
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

    // 使用无分支版本的边界处理函数
    for (int ky = -half_h; ky <= half_h; ++ky)
    {
        int yy = reflect_101_no_branch(y + ky, img_h);
        for (int kx = -half_w; kx <= half_w; ++kx)
        {
            int xx = reflect_101_no_branch(x + kx, img_w);
            sum += in[yy * img_w + xx];
        }
    }

    out[idx] = (uint8_t)roundf(sum / count);
}

/**
 * @brief 使用共享内存优化的2D模糊核函数
 * 通过共享内存减少全局内存访问次数，提高内存访问效率
 * 
 * @param in 输入图像数据指针
 * @param out 输出图像数据指针
 * @param ks_w 卷积核宽度
 * @param ks_h 卷积核高度
 * @param img_w 图像宽度
 * @param img_h 图像高度
 */
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
        global_x = reflect_101_no_branch(global_x, img_w);
        global_y = reflect_101_no_branch(global_y, img_h);

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

    // 在共享内存中进行卷积计算
    for (int ky = 0; ky < ks_h; ++ky)
    {
        for (int kx = 0; kx < ks_w; ++kx)
        {
            int smem_idx = (smem_start_y + ky) * smem_w + (smem_start_x + kx);
            sum += smem[smem_idx];
        }
    }
    out[y * img_w + x] = (uint8_t)roundf(sum / count);
}

/**
 * @brief 横向一维滤波核函数
 * 可分离滤波的第一步，对每行进行水平方向的滤波
 * 
 * @param in 输入图像数据指针
 * @param tmp 临时存储数组，存储水平滤波结果
 * @param ks_w 卷积核宽度
 * @param img_w 图像宽度
 * @param img_h 图像高度
 */
__global__ void blur_u8_h_kernel(uint8_t *in, int32_t *tmp, int ks_w, int img_w, int img_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= img_w || y >= img_h)
        return;

    int half_w = ks_w / 2;
    int sum    = 0;

    // 水平方向卷积
    for (int kx = -half_w; kx <= half_w; ++kx)
    {
        int xx = reflect_101_no_branch(x + kx, img_w);
        sum += in[y * img_w + xx];
    }

    // 存储原始和，不进行除法以避免精度损失
    tmp[y * img_w + x] = sum;
}

/**
 * @brief 纵向一维滤波核函数
 * 可分离滤波的第二步，对每列进行垂直方向的滤波
 * 
 * @param tmp 临时存储数组，包含水平滤波结果
 * @param out 输出图像数据指针
 * @param ks_w 卷积核宽度
 * @param ks_h 卷积核高度
 * @param img_w 图像宽度
 * @param img_h 图像高度
 */
__global__ void blur_u8_v_kernel(int32_t *tmp, uint8_t *out, int ks_w, int ks_h, int img_w, int img_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= img_w || y >= img_h)
        return;

    int half_h = ks_h / 2;
    int sum    = 0;

    // 垂直方向卷积
    for (int ky = -half_h; ky <= half_h; ++ky)
    {
        int yy = reflect_101_no_branch(y + ky, img_h);
        sum += tmp[yy * img_w + x];
    }

    // 现在计算总平均值：sum是水平和的垂直和，需要除以 ks_w * ks_h
    out[y * img_w + x] = (uint8_t)roundf((float)sum / (ks_w * ks_h));
}

/**
 * @brief 横向滤波的共享内存优化版本
 * 每个block处理一行，使用共享内存减少全局内存访问
 * 
 * @param in 输入图像数据指针
 * @param tmp 临时存储数组，存储水平滤波结果
 * @param ks_w 卷积核宽度
 * @param img_w 图像宽度
 * @param img_h 图像高度
 */
__global__ void blur_u8_h_shared_kernel(uint8_t *in, int32_t *tmp, int ks_w, int img_w, int img_h)
{
    extern __shared__ uint8_t smem[];

    int tx = threadIdx.x;
    int x  = blockIdx.x * blockDim.x + tx;
    int y  = blockIdx.y; // 每个block处理一行

    if (x >= img_w || y >= img_h)
        return;

    int half_w = ks_w / 2;

    // 每个 block 覆盖的 tile 宽度
    int tile_w = blockDim.x + 2 * half_w;

    // 将当前 block 覆盖的范围搬到共享内存
    // 注意：每个线程可能要搬多个像素，确保覆盖完整窗口
    for (int k = tx; k < tile_w; k += blockDim.x)
    {
        int gx  = blockIdx.x * blockDim.x + k - half_w;
        gx      = reflect_101_no_branch(gx, img_w);
        smem[k] = in[y * img_w + gx];
    }
    __syncthreads();

    // 计算卷积 - 存储原始和，不进行除法以避免精度损失
    int sum = 0;
    for (int k = 0; k < ks_w; ++k)
    {
        sum += smem[tx + k];
    }
    tmp[y * img_w + x] = sum;
}

/**
 * @brief 纵向滤波的共享内存优化版本
 * 使用共享内存减少全局内存访问
 * 
 * @param tmp 临时存储数组，包含水平滤波结果
 * @param out 输出图像数据指针
 * @param ks_w 卷积核宽度
 * @param ks_h 卷积核高度
 * @param img_w 图像宽度
 * @param img_h 图像高度
 */
__global__ void blur_u8_v_shared_kernel(int32_t *tmp, uint8_t *out, int ks_w, int ks_h, int img_w, int img_h)
{
    extern __shared__ int32_t smem2[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    if (x >= img_w || y >= img_h)
        return;

    int half_h = ks_h / 2;

    // 每个 block 覆盖的 tile 高度
    int tile_h = blockDim.y + 2 * half_h;

    // 搬运共享内存
    for (int k = ty; k < tile_h; k += blockDim.y)
    {
        int gy                     = blockIdx.y * blockDim.y + k - half_h;
        gy                         = reflect_101_no_branch(gy, img_h);
        smem2[k * blockDim.x + tx] = tmp[gy * img_w + x];
    }
    __syncthreads();

    // 卷积计算
    int sum = 0;
    for (int k = 0; k < ks_h; ++k)
    {
        sum += smem2[(ty + k) * blockDim.x + tx];
    }

    out[y * img_w + x] = (uint8_t)roundf((float)sum / (ks_w * ks_h));
}

/**
 * @brief 纵向滤波的共享内存优化版本（每个block处理一列）
 * 特殊网格配置，每个block处理一列，可能在某些情况下有更好的内存访问模式
 * 
 * @param tmp 临时存储数组，包含水平滤波结果
 * @param out 输出图像数据指针
 * @param ks_w 卷积核宽度
 * @param ks_h 卷积核高度
 * @param img_w 图像宽度
 * @param img_h 图像高度
 */
__global__ void blur_u8_v_shared_column_kernel(int32_t *tmp, uint8_t *out, int ks_w, int ks_h, int img_w, int img_h)
{
    extern __shared__ int32_t smem2[];

    int ty = threadIdx.y;                  // 使用threadIdx.y作为列内的垂直索引
    int y  = blockIdx.y * blockDim.y + ty; // 每个block处理一列
    int x  = blockIdx.x;                   // 每个block对应一列

    if (x >= img_w || y >= img_h)
        return;

    int half_h = ks_h / 2;

    // 每个 block 覆盖的 tile 高度
    int tile_h = blockDim.y + 2 * half_h;

    // 将当前 block 覆盖的列数据搬到共享内存
    // 注意：每个线程可能要搬多个像素，确保覆盖完整窗口
    for (int k = ty; k < tile_h; k += blockDim.y)
    {
        int gy   = blockIdx.y * blockDim.y + k - half_h;
        gy       = reflect_101_no_branch(gy, img_h);
        smem2[k] = tmp[gy * img_w + x];
    }
    __syncthreads();

    // 计算卷积 - 计算最终结果
    int sum = 0;
    for (int k = 0; k < ks_h; ++k)
    {
        sum += smem2[ty + k];
    }

    out[y * img_w + x] = (uint8_t)roundf((float)sum / (ks_w * ks_h));
}

/**
 * @brief 横向滑动窗口优化的滤波核函数
 * 每个线程处理一整行，使用滑动窗口技术减少重复计算
 * 
 * @param in 输入图像数据指针（使用restrict关键字优化）
 * @param tmp 临时存储数组（使用restrict关键字优化）
 * @param ks_w 卷积核宽度
 * @param img_w 图像宽度
 * @param img_h 图像高度
 */
__global__ void blur_u8_h_sw_kernel(const uint8_t *__restrict__ in, int32_t *__restrict__ tmp, int ks_w, int img_w,
                                    int img_h)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= img_h)
        return;

    int half = ks_w / 2;

    // x=0 位置初始化
    int sum = 0;
    for (int k = -half; k <= half; ++k)
    {
        int xx = reflect_101_no_branch(k, img_w);
        sum += in[y * img_w + xx];
    }
    tmp[y * img_w + 0] = sum;

    // 递推：x 从 1 到 img_w-1
    for (int x = 1; x < img_w; ++x)
    {
        int prev = reflect_101_no_branch(x - half - 1, img_w); // 移出
        int next = reflect_101_no_branch(x + half, img_w);     // 移入
        sum += (int)in[y * img_w + next] - (int)in[y * img_w + prev];
        tmp[y * img_w + x] = sum;
    }
}

/**
 * @brief 纵向滑动窗口优化的滤波核函数
 * 每个线程处理一整列，使用滑动窗口技术减少重复计算
 * 
 * @param tmp 临时存储数组（使用restrict关键字优化）
 * @param out 输出图像数据指针（使用restrict关键字优化）
 * @param ks_w 卷积核宽度
 * @param ks_h 卷积核高度
 * @param img_w 图像宽度
 * @param img_h 图像高度
 */
__global__ void blur_u8_v_sw_kernel(const int32_t *__restrict__ tmp, uint8_t *__restrict__ out, int ks_w, int ks_h,
                                    int img_w, int img_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= img_w)
        return;

    int         half = ks_h / 2;
    const float area = ks_w * ks_h;

    // y=0 位置初始化（对 tmp 做竖向窗口和）
    int sum = 0;
    for (int k = -half; k <= half; ++k)
    {
        int yy = reflect_101_no_branch(k, img_h);
        sum += tmp[yy * img_w + x];
    }
    out[0 * img_w + x] = (uint8_t)((sum + area / 2) / area);

    // 递推：y 从 1 到 img_h-1
    for (int y = 1; y < img_h; ++y)
    {
        int prev = reflect_101_no_branch(y - half - 1, img_h); // 移出
        int next = reflect_101_no_branch(y + half, img_h);     // 移入
        sum += tmp[next * img_w + x] - tmp[prev * img_w + x];  // 注意：用的是"上一 y 的 sum"来递推
        out[y * img_w + x] = (uint8_t)((sum + area / 2) / area);
    }
}

/**
 * @brief 使用共享内存的2D模糊函数接口
 * 
 * @param in 输入张量
 * @param ksz 卷积核大小（正方形）
 * @param out 输出张量
 */
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

/**
 * @brief 可分离滤波函数接口
 * 将2D卷积分解为两个1D卷积，提高计算效率
 * 
 * @param in 输入张量
 * @param ksz 卷积核大小（正方形）
 * @param out 输出张量
 * @param tmp 临时张量，用于存储中间结果
 */
void blur_u8_split(torch::Tensor in, const int ksz, torch::Tensor out, torch::Tensor tmp)
{
    CHECK_TORCH_TENSOR_DTYPE(in, torch::kUInt8)
    CHECK_TORCH_TENSOR_DTYPE(out, torch::kUInt8)
    CHECK_TORCH_TENSOR_DTYPE(tmp, torch::kInt32)
    CHECK_TORCH_TENSOR_DEVICE(in)
    CHECK_TORCH_TENSOR_DEVICE(out)
    CHECK_TORCH_TENSOR_DEVICE(tmp)

    const int H = in.size(0);
    const int W = in.size(1);
    const int N = H * W;
    dim3      block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3      grid(divUp(W, block.x), divUp(H, block.y));

    // 先进行水平滤波
    blur_u8_h_kernel<<<grid, block>>>(reinterpret_cast<uint8_t *>(in.data_ptr()),
                                      reinterpret_cast<int32_t *>(tmp.data_ptr()), ksz, W, H);
    // 再进行垂直滤波
    blur_u8_v_kernel<<<grid, block>>>(reinterpret_cast<int32_t *>(tmp.data_ptr()),
                                      reinterpret_cast<uint8_t *>(out.data_ptr()), ksz, ksz, W, H);
}

/**
 * @brief 宏定义：生成标准的模糊函数绑定
 * 
 * @param packed_type 打包类型名称
 * @param torch_type PyTorch张量类型
 * @param element_type 元素类型
 * @param n_elements 元素数量
 */
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

// 生成基础模糊函数和无分支版本
TORCH_BINDING_BLUR(u8, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_BLUR(u8_nb, torch::kUInt8, uint8_t, 1)

/**
 * @brief 可分离滤波的共享内存优化版本（版本1）
 * 使用特殊的网格配置，每个block处理一列
 * 
 * @param in 输入张量
 * @param ksz 卷积核大小（正方形）
 * @param out 输出张量
 * @param tmp 临时张量，用于存储中间结果
 */
void blur_u8_split_shared(torch::Tensor in, const int ksz, torch::Tensor out, torch::Tensor tmp)
{
    CHECK_TORCH_TENSOR_DTYPE(in, torch::kUInt8)
    CHECK_TORCH_TENSOR_DTYPE(out, torch::kUInt8)
    CHECK_TORCH_TENSOR_DTYPE(tmp, torch::kInt32)
    CHECK_TORCH_TENSOR_DEVICE(in)
    CHECK_TORCH_TENSOR_DEVICE(out)
    CHECK_TORCH_TENSOR_DEVICE(tmp)

    const int H = in.size(0);
    const int W = in.size(1);
    const int N = H * W;
    dim3      block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3      grid(divUp(W, block.x), divUp(H, block.y));

    // 横向滤波：每个block处理一行，需要特殊的网格配置
    dim3   h_grid(divUp(W, block.x), H); // H个block，每个处理一行
    size_t h_smem_size = (block.x + 2 * (ksz / 2)) * sizeof(uint8_t);
    blur_u8_h_shared_kernel<<<h_grid, dim3(block.x, 1), h_smem_size>>>(
        reinterpret_cast<uint8_t *>(in.data_ptr()), reinterpret_cast<int32_t *>(tmp.data_ptr()), ksz, W, H);

    // 纵向滤波的共享内存大小：需要额外的halo区域
    // size_t v_smem_size = (block.y + 2 * (ksz / 2)) * block.x * sizeof(int32_t);
    // blur_u8_v_shared_kernel<<<grid, block, v_smem_size>>>(reinterpret_cast<int32_t *>(tmp.data_ptr()),
    //                                                reinterpret_cast<uint8_t *>(out.data_ptr()), ksz, ksz, W, H);

    // 纵向滤波：每个block处理一列，使用特殊的网格配置
    dim3   v_grid(W, divUp(H, block.y)); // W个block，每个处理一列
    size_t v_smem_size = (block.y + 2 * (ksz / 2)) * sizeof(int32_t);
    blur_u8_v_shared_column_kernel<<<v_grid, dim3(1, block.y), v_smem_size>>>(
        reinterpret_cast<int32_t *>(tmp.data_ptr()), reinterpret_cast<uint8_t *>(out.data_ptr()), ksz, ksz, W, H);
}

/**
 * @brief 可分离滤波的共享内存优化版本（版本2）
 * 使用标准的网格配置
 * 
 * @param in 输入张量
 * @param ksz 卷积核大小（正方形）
 * @param out 输出张量
 * @param tmp 临时张量，用于存储中间结果
 */
void blur_u8_split_shared2(torch::Tensor in, const int ksz, torch::Tensor out, torch::Tensor tmp)
{
    CHECK_TORCH_TENSOR_DTYPE(in, torch::kUInt8)
    CHECK_TORCH_TENSOR_DTYPE(out, torch::kUInt8)
    CHECK_TORCH_TENSOR_DTYPE(tmp, torch::kInt32)
    CHECK_TORCH_TENSOR_DEVICE(in)
    CHECK_TORCH_TENSOR_DEVICE(out)
    CHECK_TORCH_TENSOR_DEVICE(tmp)

    const int H = in.size(0);
    const int W = in.size(1);
    const int N = H * W;
    dim3      block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3      grid(divUp(W, block.x), divUp(H, block.y));

    // 横向滤波：每个block处理一行，需要特殊的网格配置
    dim3   h_grid(divUp(W, block.x), H); // H个block，每个处理一行
    size_t h_smem_size = (block.x + 2 * (ksz / 2)) * sizeof(uint8_t);
    blur_u8_h_shared_kernel<<<h_grid, dim3(block.x, 1), h_smem_size>>>(
        reinterpret_cast<uint8_t *>(in.data_ptr()), reinterpret_cast<int32_t *>(tmp.data_ptr()), ksz, W, H);

    // 纵向滤波的共享内存大小：需要额外的halo区域
    size_t v_smem_size = (block.y + 2 * (ksz / 2)) * block.x * sizeof(int32_t);
    blur_u8_v_shared_kernel<<<grid, block, v_smem_size>>>(reinterpret_cast<int32_t *>(tmp.data_ptr()),
                                                          reinterpret_cast<uint8_t *>(out.data_ptr()), ksz, ksz, W, H);

    // 纵向滤波：每个block处理一列，使用特殊的网格配置
    // dim3   v_grid(W, divUp(H, block.y)); // W个block，每个处理一列
    // size_t v_smem_size = (block.y + 2 * (ksz / 2)) * sizeof(int32_t);
    // blur_u8_v_shared_column_kernel<<<v_grid, dim3(1, block.y), v_smem_size>>>(
    //     reinterpret_cast<int32_t *>(tmp.data_ptr()), reinterpret_cast<uint8_t *>(out.data_ptr()), ksz, ksz, W, H);
}

/**
 * @brief 使用滑动窗口优化的可分离滤波函数
 * 每个线程处理一整行或一整列，使用滑动窗口技术减少重复计算
 * 
 * @param in 输入张量
 * @param ksz 卷积核大小（正方形）
 * @param out 输出张量
 * @param tmp 临时张量，用于存储中间结果
 */
void blur_u8_split_sw(torch::Tensor in, const int ksz, torch::Tensor out, torch::Tensor tmp)
{
    CHECK_TORCH_TENSOR_DTYPE(in, torch::kUInt8)
    CHECK_TORCH_TENSOR_DTYPE(out, torch::kUInt8)
    CHECK_TORCH_TENSOR_DTYPE(tmp, torch::kInt32)
    CHECK_TORCH_TENSOR_DEVICE(in)
    CHECK_TORCH_TENSOR_DEVICE(out)
    CHECK_TORCH_TENSOR_DEVICE(tmp)

    const int H = in.size(0);
    const int W = in.size(1);
    const int N = H * W;

    // 横向滑动窗口：每个线程处理一整行
    // 需要确保有足够的线程来覆盖所有行
    dim3 h_block(1, BLOCK_SIZE_X * BLOCK_SIZE_Y); // 使用32个线程处理行
    dim3 h_grid(1, divUp(H, h_block.y));          // 确保覆盖所有行
    blur_u8_h_sw_kernel<<<h_grid, h_block>>>(reinterpret_cast<uint8_t *>(in.data_ptr()),
                                             reinterpret_cast<int32_t *>(tmp.data_ptr()), ksz, W, H);

    // 纵向滑动窗口：每个线程处理一整列
    // 需要确保有足够的线程来覆盖所有列
    dim3 v_block(BLOCK_SIZE_X * BLOCK_SIZE_Y, 1); // 使用32个线程处理列
    dim3 v_grid(divUp(W, v_block.x), 1);          // 确保覆盖所有列
    blur_u8_v_sw_kernel<<<v_grid, v_block>>>(reinterpret_cast<int32_t *>(tmp.data_ptr()),
                                             reinterpret_cast<uint8_t *>(out.data_ptr()), ksz, ksz, W, H);
}

#define TORCH_BINDING_BLUR_TEMPLATE(tag, th_type, element_type, cal_type, n_pack)                                      \
    torch::Tensor tag##_##element_type##_##cal_type(torch::Tensor src, torch::Tensor dst, const int ksz)               \
    {                                                                                                                  \
        CHECK_TORCH_TENSOR_DTYPE(src, (th_type))                                                                       \
        CHECK_TORCH_TENSOR_DTYPE(dst, (th_type))                                                                       \
        CHECK_TORCH_TENSOR_DEVICE(src)                                                                                 \
        CHECK_TORCH_TENSOR_DEVICE(dst)                                                                                 \
        const int H  = src.size(0);                                                                                    \
        const int W  = src.size(1);                                                                                    \
        const int CH = src.dim() == 2 ? 1 : src.size(2);                                                               \
        const int N  = H * W;                                                                                          \
        dim3      block(THREADS);                                                                                      \
        dim3      grid(divUp(N, THREADS));                                                                             \
        if (CH == 1)                                                                                                   \
        {                                                                                                              \
            tag##_kernel<element_type, cal_type, 1><<<grid, block>>>(reinterpret_cast<element_type *>(src.data_ptr()), \
                                                                     reinterpret_cast<element_type *>(dst.data_ptr()), \
                                                                     ksz, ksz, H, W, N);                               \
        }                                                                                                              \
        else if (CH == 3)                                                                                              \
        {                                                                                                              \
            tag##_kernel<element_type, cal_type, 3><<<grid, block>>>(reinterpret_cast<element_type *>(src.data_ptr()), \
                                                                     reinterpret_cast<element_type *>(dst.data_ptr()), \
                                                                     ksz, ksz, H, W, N);                               \
        }                                                                                                              \
        return dst;                                                                                                    \
    }

TORCH_BINDING_BLUR_TEMPLATE(blur, torch::kFloat32, float, float, 1)
TORCH_BINDING_BLUR_TEMPLATE(blur, torch::kFloat32, float, double, 1)
TORCH_BINDING_BLUR_TEMPLATE(blur, torch::kUInt8, uint8_t, float, 1)
TORCH_BINDING_BLUR_TEMPLATE(blur, torch::kUInt8, uint8_t, double, 1)
TORCH_BINDING_BLUR_TEMPLATE(blur, torch::kUInt8, uint8_t, int32_t, 1)

/**
 * @brief Python绑定模块
 * 将所有CUDA函数绑定到Python接口
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(blur_u8)
    TORCH_BINDING_COMMON_EXTENSION(blur_u8_nb)
    TORCH_BINDING_COMMON_EXTENSION(blur_u8_shared)
    TORCH_BINDING_COMMON_EXTENSION(blur_u8_split)
    TORCH_BINDING_COMMON_EXTENSION(blur_u8_split_shared)
    TORCH_BINDING_COMMON_EXTENSION(blur_u8_split_shared2)
    TORCH_BINDING_COMMON_EXTENSION(blur_u8_split_sw)

    TORCH_BINDING_COMMON_EXTENSION(blur_float_float)
    TORCH_BINDING_COMMON_EXTENSION(blur_float_double)
    TORCH_BINDING_COMMON_EXTENSION(blur_uint8_t_float)
    TORCH_BINDING_COMMON_EXTENSION(blur_uint8_t_double)
    TORCH_BINDING_COMMON_EXTENSION(blur_uint8_t_int32_t)
}