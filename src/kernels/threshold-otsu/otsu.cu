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
__global__ void otsu_gpu_kernel(const uint32_t *hists, int *threshold)
{
    const int         tid = threadIdx.x;
    __shared__ double s_scan_count[256];
    __shared__ double s_scan_weight[256];

    // 加载直方图到共享内存 (每个线程1个灰度级)
    if (tid < 256)
    {
        s_scan_count[tid]  = static_cast<double>(hists[tid]);              // 频次
        s_scan_weight[tid] = static_cast<double>(tid * s_scan_count[tid]); // 灰度值 x 频次
    }
    else
    {
        s_scan_count[tid]  = 0.0;
        s_scan_weight[tid] = 0.0;
    }
    __syncthreads();

    // 使用 Hillis-Steele 计算累前缀和, 使得
    // s_scan_count[i] = 从 0 到 i 的像素总数
    // s_scan_weight[i] = 从 0 到 i 的加权像素总和
    for (int offset = 1; offset < 256; offset <<= 1)
    {
        double val_count  = 0.0;
        double val_weight = 0.0;
        if (tid >= offset)
        {
            val_count  = s_scan_count[tid - offset];
            val_weight = s_scan_weight[tid - offset];
        }
        __syncthreads();
        s_scan_count[tid] += val_count;
        s_scan_weight[tid] += val_weight;
        __syncthreads();
    }

    // 像素总数和加权像素总和
    double total   = s_scan_count[255];
    double sum_all = s_scan_weight[255];

    // Each thread computes its wB and sumB (for threshold = tid)
    // 每个线程计算以 tid 为阈值的类间方差
    // wB 背景类权重
    // sumB 背景类加权和
    double wB   = s_scan_count[tid];  // inclusive count up to tid
    double sumB = s_scan_weight[tid]; // inclusive weighted sum up to tid

    //  σ²ᵦ = wB × wF × (μB - μF)²
    // 因为只要找到最大值的位置, 因此没有归一化, 忽略常数因子 1 / total²
    // Avoid cases where wB==0 or wF==0
    double varBetween = 0.0;
    if (wB > 0.0 && total - wB > 0.0)
    {
        double wF   = total - wB;            // 前景类权重
        double mB   = sumB / wB;             // 背景类均值 // mB = sumB / wB = Σ(i × hist[i]) / Σ(hist[i])
        double mF   = (sum_all - sumB) / wF; // 前景类均值
        double diff = mB - mF;               // 类间均值差
        varBetween  = wB * wF * diff * diff; // 类间方差
    }
    else
    {
        varBetween = -1.0; // invalid / sentinel so won't win
    }

    // Block-level reduction to find max varBetween and its index (threshold)
    // Use shared arrays to store (var,index), then reduce.
    __shared__ double s_var[256]; // 每个阈值的类间方差
    __shared__ int    s_idx[256]; // 对应的阈值索引
    s_var[tid] = varBetween;
    s_idx[tid] = tid;
    __syncthreads();

    // 二叉树归约模式，通过log₂(256) = 8次迭代找到最大值
    /***
     * stride=128:  [0vs128, 1vs129, 2vs130, ..., 127vs255] → 128个比较
     * stride=64:   [0vs64,  1vs65,  2vs66,  ..., 63vs127]  → 64个比较  
     * stride=32:   [0vs32,  1vs33,  2vs34,  ..., 31vs63]   → 32个比较
     * stride=16:   [0vs16,  1vs17,  2vs18,  ..., 15vs31]   → 16个比较
     * stride=8:    [0vs8,   1vs9,   2vs10,  ..., 7vs15]    → 8个比较
     * stride=4:    [0vs4,   1vs5,   2vs6,   3vs7]          → 4个比较
     * stride=2:    [0vs2,   1vs3]                          → 2个比较
     * stride=1:    [0vs1]                                  → 1个比较
     ***/
    for (int stride = 128; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            double a  = s_var[tid];
            double b  = s_var[tid + stride];
            int    ib = s_idx[tid + stride];
            if (b > a)
            {
                s_var[tid] = b;
                s_idx[tid] = ib;
            }
        }
        __syncthreads();
    }

    // thread 0 writes result
    if (tid == 0)
    {
        threshold[0] = s_idx[0];
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
void threshold_otsu_u8(torch::Tensor src, torch::Tensor dst, torch::Tensor thresh, const uint8_t maxval)
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

    otsu_gpu_kernel<<<1, 256>>>(reinterpret_cast<uint32_t *>(hists.data_ptr()),
                                reinterpret_cast<int *>(thresh.data_ptr()));
    threshold_kernel<<<grids, blocks>>>(reinterpret_cast<uint8_t *>(src.data_ptr()),
                                        reinterpret_cast<uint8_t *>(dst.data_ptr()),
                                        reinterpret_cast<int *>(thresh.data_ptr()), maxval, N);
}

void threshold_otsu_u8_warp(torch::Tensor src, torch::Tensor dst, torch::Tensor thresh, const uint8_t maxval)
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
    otsu_gpu_kernel<<<1, 256>>>(reinterpret_cast<uint32_t *>(hists.data_ptr()),
                                reinterpret_cast<int *>(thresh.data_ptr()));
    threshold_u8x4_kernel<<<grids, blocks>>>(reinterpret_cast<uint8_t *>(src.data_ptr()),
                                             reinterpret_cast<uint8_t *>(dst.data_ptr()),
                                             reinterpret_cast<int *>(thresh.data_ptr()), maxval, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(threshold_otsu_u8)
    TORCH_BINDING_COMMON_EXTENSION(threshold_otsu_u8_warp)
}