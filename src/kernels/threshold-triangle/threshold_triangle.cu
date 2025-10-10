#include "common.cuh"

__global__ void histogram_shared_kernel(uint8_t *img, uint32_t *global_hist, const int N)
{
    // shared histogram 256 bins
    __shared__ uint32_t s_hist[256];

    const int tid = threadIdx.x;
    // initialize shared hist (threads step through bins)
    for (int i = tid; i < 256; i += blockDim.x)
    {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // each thread processes multiple pixels in strided loop
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride)
    {
        uint8_t v = img[i];
        atomicAdd(&s_hist[v], 1u);
    }
    __syncthreads();

    // merge shared histogram into global histogram (threads cooperatively)
    for (int i = tid; i < 256; i += blockDim.x)
    {
        uint32_t val = s_hist[i];
        if (val)
            atomicAdd(&global_hist[i], val);
    }
}

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

/**
 * @brief
 * 
 * @note launch this kernel with <<<1,256>>>
 */
__global__ void threshold_triangle_kernel(const uint32_t *hists, int *threshold)
{
    const int tid = threadIdx.x;
    const int N   = 256;

    // 共享内存存储直方图和中间结果
    __shared__ uint32_t s_hist[256];         // 直方图数据
    __shared__ uint32_t s_max_val[256];      // 用于找最大值的归约
    __shared__ int      s_max_idx[256];      // 最大值对应的索引
    __shared__ double   s_distance[256];     // 到直线的距离
    __shared__ int      s_distance_idx[256]; // 距离对应的索引

    // 第一步：加载直方图到共享内存
    if (tid < 256)
    {
        s_hist[tid]    = hists[tid];
        s_max_val[tid] = s_hist[tid];
        s_max_idx[tid] = tid;
    }
    else
    {
        s_hist[tid]    = 0;
        s_max_val[tid] = 0;
        s_max_idx[tid] = 0;
    }
    __syncthreads();

    // 第二步：并行归约找到直方图最大值及其位置
    for (int stride = 128; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            uint32_t a  = s_max_val[tid];
            uint32_t b  = s_max_val[tid + stride];
            int      ib = s_max_idx[tid + stride];
            if (b > a)
            {
                s_max_val[tid] = b;
                s_max_idx[tid] = ib;
            }
        }
        __syncthreads();
    }

    // 线程块协作获取最大值信息，并找到有效范围
    __shared__ uint32_t max_hist_val;
    __shared__ int      max_hist_pos;
    __shared__ int      left_bound;
    __shared__ int      right_bound;
    __shared__ bool     isflipped;

    // 临时共享内存用于并行查找边界
    __shared__ int s_left_candidates[256];
    __shared__ int s_right_candidates[256];

    if (tid == 0)
    {
        max_hist_val = s_max_val[0];
        max_hist_pos = s_max_idx[0];
    }

    // 并行查找第一个非零值（left_bound）
    if (tid < 256)
    {
        // 每个线程检查自己的位置是否是有效的左边界候选
        s_left_candidates[tid] = (s_hist[tid] > 0) ? tid : N; // N作为无效值
    }
    __syncthreads();

    // 并行归约找最小的有效索引（第一个非零值）
    for (int stride = 128; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            int a                  = s_left_candidates[tid];
            int b                  = s_left_candidates[tid + stride];
            s_left_candidates[tid] = (a < b) ? a : b; // 找最小值
        }
        __syncthreads();
    }

    // 并行查找最后一个非零值（right_bound）
    if (tid < 256)
    {
        // 每个线程检查自己的位置是否是有效的右边界候选
        s_right_candidates[tid] = (s_hist[tid] > 0) ? tid : -1; // -1作为无效值
    }
    __syncthreads();

    // 并行归约找最大的有效索引（最后一个非零值）
    for (int stride = 128; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            int a                   = s_right_candidates[tid];
            int b                   = s_right_candidates[tid + stride];
            s_right_candidates[tid] = (a > b) ? a : b; // 找最大值
        }
        __syncthreads();
    }

    // 线程0处理结果并设置最终边界
    if (tid == 0)
    {
        left_bound  = s_left_candidates[0];
        right_bound = s_right_candidates[0];

        // OpenCV: 向外扩展一位
        if (left_bound > 0)
            left_bound--;
        if (right_bound < N - 1)
            right_bound++;

        // 判断是否需要翻转（OpenCV 的关键逻辑）
        isflipped = (max_hist_pos - left_bound < right_bound - max_hist_pos);
    }
    __syncthreads();

    // 第三步：如果需要翻转，所有线程参与翻转直方图
    if (isflipped && tid < 128)
    {
        int      i = tid;
        int      j = N - 1 - tid;
        // 交换 s_hist[i] 和 s_hist[j]
        uint32_t temp = s_hist[i];
        s_hist[i]     = s_hist[j];
        s_hist[j]     = temp;
    }
    __syncthreads();

    // 线程0更新翻转后的边界和峰值位置
    if (tid == 0 && isflipped)
    {
        // 保存原始 right_bound
        int old_right = right_bound;
        // 按照 OpenCV 的逻辑更新
        left_bound   = N - 1 - old_right;
        max_hist_pos = N - 1 - max_hist_pos;
    }
    __syncthreads();

    // 第四步：Triangle算法核心 - 使用 OpenCV 的简化距离公式
    // 直线从 (max_ind, max) 到 (left_bound, 0)
    // 简化距离公式: tempdist = a*i + b*h[i]
    // 其中 a = max, b = left_bound - max_ind

    double distance = 0.0;

    // 只在 left_bound+1 到 max_ind 之间计算（OpenCV 的做法）
    if (tid >= left_bound + 1 && tid <= max_hist_pos)
    {
        // OpenCV 的简化距离公式
        double a = static_cast<double>(max_hist_val);
        double b = static_cast<double>(left_bound - max_hist_pos);
        distance = a * tid + b * s_hist[tid];
    }
    else
    {
        distance = 0.0;
    }

    // 第五步：准备归约找最大距离
    s_distance[tid]     = distance;
    s_distance_idx[tid] = tid;
    __syncthreads();

    // 第六步：并行归约找到最大距离及其对应的阈值
    for (int stride = 128; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            double a  = s_distance[tid];
            double b  = s_distance[tid + stride];
            int    ib = s_distance_idx[tid + stride];
            if (b > a)
            {
                s_distance[tid]     = b;
                s_distance_idx[tid] = ib;
            }
        }
        __syncthreads();
    }

    // 第七步：线程0输出结果，应用 OpenCV 的后处理
    if (tid == 0)
    {
        int thresh = s_distance_idx[0];

        // OpenCV: thresh--
        thresh--;

        // 如果之前翻转了，需要映射回去
        if (isflipped)
            thresh = N - 1 - thresh;

        threshold[0] = thresh;
    }
}

__global__ void threshold_kernel(uint8_t *src, uint8_t *dst, int *thresh, const uint8_t maxval, const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    dst[idx] = src[idx] > thresh[0] ? maxval : 0;
}

__global__ void threshold_u8x4_kernel(uint8_t *src, uint8_t *dst, int *thresh, const uint8_t maxval, const int N)
{
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= N)
        return;

    // 检查是否有足够的像素进行向量化处理
    if (idx + 3 < N)
    {
        // 完整的4像素向量化处理
        uchar4 in4 = *reinterpret_cast<uchar4 *>(&src[idx]);
        uchar4 out4;

        out4.x = in4.x > thresh[0] ? maxval : 0;
        out4.y = in4.y > thresh[0] ? maxval : 0;
        out4.z = in4.z > thresh[0] ? maxval : 0;
        out4.w = in4.w > thresh[0] ? maxval : 0;

        *reinterpret_cast<uchar4 *>(&dst[idx]) = out4;
    }
    else
    {
        // 处理剩余的像素（不足4个的情况）
        for (int i = 0; i < 4 && (idx + i) < N; i++)
        {
            dst[idx + i] = src[idx + i] > thresh[0] ? maxval : 0;
        }
    }
}

// Python绑定函数
void threshold_triangle_u8(torch::Tensor src, torch::Tensor dst, torch::Tensor thresh, const uint8_t maxval)
{
    CHECK_TORCH_TENSOR_DTYPE(src, torch::kUInt8)
    CHECK_TORCH_TENSOR_DTYPE(dst, torch::kUInt8)
    CHECK_TORCH_TENSOR_DEVICE(src)
    CHECK_TORCH_TENSOR_DEVICE(dst)

    auto          options = torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA, 0);
    torch::Tensor hists   = torch::zeros({256}, options);

    const int H = src.size(0);
    const int W = src.size(1);
    const int N = H * W;

    dim3 blocks(THREADS);
    dim3 grids(divUp(N, THREADS));
    histogram_shared_kernel<<<grids, blocks>>>(reinterpret_cast<uint8_t *>(src.data_ptr()),
                                               reinterpret_cast<uint32_t *>(hists.data_ptr()), N);

    threshold_triangle_kernel<<<1, 256>>>(reinterpret_cast<uint32_t *>(hists.data_ptr()),
                                          reinterpret_cast<int *>(thresh.data_ptr()));
    threshold_kernel<<<grids, blocks>>>(reinterpret_cast<uint8_t *>(src.data_ptr()),
                                        reinterpret_cast<uint8_t *>(dst.data_ptr()),
                                        reinterpret_cast<int *>(thresh.data_ptr()), maxval, N);
}

void threshold_triangle_u8_warp(torch::Tensor src, torch::Tensor dst, torch::Tensor thresh, const uint8_t maxval)
{
    CHECK_TORCH_TENSOR_DTYPE(src, torch::kUInt8)
    CHECK_TORCH_TENSOR_DTYPE(dst, torch::kUInt8)
    CHECK_TORCH_TENSOR_DEVICE(src)
    CHECK_TORCH_TENSOR_DEVICE(dst)

    auto          options = torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA, 0);
    torch::Tensor hists   = torch::zeros({256}, options);

    const int H = src.size(0);
    const int W = src.size(1);
    const int N = H * W;

    const int elements_per_thread = 4;
    const int block               = THREADS;
    const int min_grid            = divUp(N, block * elements_per_thread);
    const int max_grid            = 128 * 4; // SM数量的4倍，经验值
    const int grid                = (min_grid < max_grid) ? min_grid : max_grid;

    histogram_u8x4_warp_kernel<THREADS><<<grid, block>>>(reinterpret_cast<const uint8_t *>(src.data_ptr()),
                                                         reinterpret_cast<int32_t *>(hists.data_ptr()), N);

    dim3 blocks(THREADS / 4);
    dim3 grids(divUp(N, THREADS));
    threshold_triangle_kernel<<<1, 256>>>(reinterpret_cast<uint32_t *>(hists.data_ptr()),
                                          reinterpret_cast<int *>(thresh.data_ptr()));
    threshold_u8x4_kernel<<<grids, blocks>>>(reinterpret_cast<uint8_t *>(src.data_ptr()),
                                             reinterpret_cast<uint8_t *>(dst.data_ptr()),
                                             reinterpret_cast<int *>(thresh.data_ptr()), maxval, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(threshold_triangle_u8)
    TORCH_BINDING_COMMON_EXTENSION(threshold_triangle_u8_warp)
}