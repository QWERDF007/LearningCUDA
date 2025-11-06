#include "common.cuh"

__global__ void histogram_shared_kernel(const uint8_t *__restrict__ src, uint32_t *dst, const int N)
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

__global__ void equalize_lut_kernel(const uint32_t *__restrict__ hist, uint8_t *__restrict__ lut, const int total)
{
    __shared__ uint32_t s_hist[256];

    int tid     = threadIdx.x;
    s_hist[tid] = hist[tid];
    __syncthreads();

    // 前缀和（串行版即可，256 元素很少）
    if (tid == 0)
    {
        uint32_t sum = 0;
        int      i   = 0;
        while (!s_hist[i]) ++i; // 跳过空的前缀

        // 若整幅图像为恒定灰度，按照 OpenCV 行为设置输出为该灰度值
        if (s_hist[i] == (uint32_t)total)
        {
            // 将 LUT 的所有条目设置为该灰度值 i
            for (int j = 0; j < 256; ++j)
            {
                lut[j] = i;
            }
        }
        else
        {
            const int k = i;
            // OpenCV 等价逻辑：首个非零 bin 映射为 0，然后从下一 bin 开始累计
            float     scale = 255.0f / (float)(total - s_hist[i]);
            for (lut[i++] = 0; i < 256; ++i)
            {
                sum += s_hist[i];
                lut[i] = saturate_cast<uint8_t>(sum * scale);
            }
            // 对前面的空灰度值（未出现的）填充 0
            // for (int j = 0; j < k; ++j) lut[j] = 0;
        }
    }
}

__global__ void equalize_lut_opt_kernel(const uint32_t *__restrict__ hist, uint8_t *__restrict__ lut, const int total)
{
    __shared__ uint32_t s_cdf[256]; // 累积分布函数缓冲
    int                 tid = threadIdx.x;

    // 1️⃣ 拷贝到共享内存
    s_cdf[tid] = hist[tid];
    __syncthreads();

    // 2️⃣ 并行前缀和 (Blelloch scan)
    // ---- 上升阶段（reduce）----
    for (int offset = 1; offset < 256; offset <<= 1)
    {
        int i = (tid + 1) * offset * 2 - 1;
        if (i < 256)
            s_cdf[i] += s_cdf[i - offset];
        __syncthreads();
    }

    // ---- 下降阶段（down-sweep）----
    if (tid == 0)
        s_cdf[255] = 0; // 最后一个置 0（exclusive scan）
    __syncthreads();

    for (int offset = 128; offset >= 1; offset >>= 1)
    {
        int i = (tid + 1) * offset * 2 - 1;
        if (i < 256)
        {
            uint32_t t        = s_cdf[i - offset];
            s_cdf[i - offset] = s_cdf[i];
            s_cdf[i] += t;
        }
        __syncthreads();
    }

    // 3️⃣ 查找首个非零值 (cdf_min)
    //    我们让 tid==0 串行扫描（256很少，不值得并行）
    __shared__ uint32_t cdf_min;
    __shared__ float    scale;
    if (tid == 0)
    {
        int i = 0;
        while (i < 256 && hist[i] == 0) ++i;
        cdf_min = (i < 256 ? s_cdf[i] + hist[i] : 0);
        scale   = 255.0f / (float)(total - cdf_min);
    }
    __syncthreads();

    // 4️⃣ 并行生成 LUT
    if (cdf_min == (uint32_t)total)
    {
        // 全图灰度相同
        lut[tid] = (uint8_t)tid;
    }
    else if (total > 0)
    {
        uint32_t sum = s_cdf[tid] + hist[tid];
        lut[tid]     = saturate_cast<uint8_t>((sum - cdf_min) * scale);
    }
    else
    {
        lut[tid] = 0;
    }
}

__global__ void apply_lut_kernel(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst,
                                 const uint8_t *__restrict__ lut, const int N)
{
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= N)
        return;

    for (; idx + 3 < N; idx += 4 * blockDim.x * gridDim.x)
    {
        uchar4 v = *reinterpret_cast<const uchar4 *>(&src[idx]);

        v.x = lut[v.x];
        v.y = lut[v.y];
        v.z = lut[v.z];
        v.w = lut[v.w];

        *reinterpret_cast<uchar4 *>(&dst[idx]) = v;
    }

    // 处理剩余像素
    for (; idx < N; idx++)
    {
        dst[idx] = lut[src[idx]];
    }
}

void equalize_hist(torch::Tensor src, torch::Tensor dst)
{
    CHECK_TORCH_TENSOR_DTYPE(src, torch::kUInt8)
    CHECK_TORCH_TENSOR_DTYPE(dst, torch::kUInt8)
    CHECK_TORCH_TENSOR_DEVICE(src)
    CHECK_TORCH_TENSOR_DEVICE(dst)

    auto hist_options = torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA, 0);
    auto lut_options  = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, 0);

    torch::Tensor hist = torch::zeros({256}, hist_options);
    torch::Tensor lut  = torch::zeros({256}, lut_options);

    const int H = src.size(0);
    const int W = src.size(1);
    const int N = H * W;

    dim3 block(THREADS);
    dim3 grid(divUp(N, THREADS * 4));

    histogram_shared_kernel<<<grid, block>>>(reinterpret_cast<uint8_t *>(src.data_ptr()),
                                             reinterpret_cast<uint32_t *>(hist.data_ptr()), N);
    equalize_lut_kernel<<<1, 256>>>(reinterpret_cast<uint32_t *>(hist.data_ptr()),
                                    reinterpret_cast<uint8_t *>(lut.data_ptr()), N);
    apply_lut_kernel<<<grid, block>>>(reinterpret_cast<uint8_t *>(src.data_ptr()),
                                      reinterpret_cast<uint8_t *>(dst.data_ptr()),
                                      reinterpret_cast<uint8_t *>(lut.data_ptr()), N);
}

void equalize_hist_opt(torch::Tensor src, torch::Tensor dst)
{
    CHECK_TORCH_TENSOR_DTYPE(src, torch::kUInt8)
    CHECK_TORCH_TENSOR_DTYPE(dst, torch::kUInt8)
    CHECK_TORCH_TENSOR_DEVICE(src)
    CHECK_TORCH_TENSOR_DEVICE(dst)

    auto hist_options = torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA, 0);
    auto lut_options  = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, 0);

    torch::Tensor hist = torch::zeros({256}, hist_options);
    torch::Tensor lut  = torch::zeros({256}, lut_options);

    const int H = src.size(0);
    const int W = src.size(1);
    const int N = H * W;

    dim3 block(THREADS);
    dim3 grid(divUp(N, THREADS * 4));

    histogram_shared_kernel<<<grid, block>>>(reinterpret_cast<uint8_t *>(src.data_ptr()),
                                             reinterpret_cast<uint32_t *>(hist.data_ptr()), N);
    equalize_lut_opt_kernel<<<1, 256>>>(reinterpret_cast<uint32_t *>(hist.data_ptr()),
                                        reinterpret_cast<uint8_t *>(lut.data_ptr()), N);
    apply_lut_kernel<<<grid, block>>>(reinterpret_cast<uint8_t *>(src.data_ptr()),
                                      reinterpret_cast<uint8_t *>(dst.data_ptr()),
                                      reinterpret_cast<uint8_t *>(lut.data_ptr()), N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(equalize_hist)
    TORCH_BINDING_COMMON_EXTENSION(equalize_hist_opt)
}