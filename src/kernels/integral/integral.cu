#include "common.cuh"

template<typename T, typename CT, int CH>
__global__ void integral_row_prefix(T *src, CT *buf, int src_h, int src_w, int dst_w)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= src_h)
        return;

    // dst 有 +1 行 +1 列，所以写入时行偏移要 +1
    // 对于多通道图像，每行有 dst_w * CH 个元素
    CT      *out_row = buf + (y + 1) * dst_w * CH;
    const T *in_row  = src + y * src_w * CH;

#pragma unroll
    for (int ch = 0; ch < CH; ch++) // 第一列置 0
    {
        out_row[ch] = 0;
    }

    // 前缀和
    for (int x = 0; x < src_w; x++)
    {
#pragma unroll
        for (int ch = 0; ch < CH; ch++)
        {
            out_row[(x + 1) * CH + ch] = out_row[x * CH + ch] + static_cast<CT>(in_row[x * CH + ch]);
        }
    }
}

// 列方向前缀和
template<typename CT, int CH>
__global__ void integral_col_prefix(CT *dst, int src_h, int src_w, int dst_w)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= src_w + 1)
        return; // dst 宽 = src_w+1

    // 每列 CH 通道一起处理
    for (int ch = 0; ch < CH; ch++)
    {
        CT sum = 0;
        for (int y = 0; y <= src_h; y++)
        {
            int idx = (y * dst_w + x) * CH + ch;
            sum += dst[idx];
            dst[idx] = sum;
        }
    }
}

// Python绑定函数
void integral_u8(torch::Tensor src, torch::Tensor dst)
{
    CHECK_TORCH_TENSOR_DTYPE(src, torch::kUInt8);
    CHECK_TORCH_TENSOR_DTYPE(dst, torch::kUInt32);
    CHECK_TORCH_TENSOR_DEVICE(src);
    CHECK_TORCH_TENSOR_DEVICE(dst);

    const int ndim  = src.dim();
    const int src_h = src.size(0);
    const int src_w = src.size(1);
    const int dst_h = dst.size(0);
    const int dst_w = dst.size(1);

    // 初始化第一行为0
    cudaMemset(dst.data_ptr<uint32_t>(), 0, dst_w * (ndim == 2 ? 1 : 3) * sizeof(uint32_t));

    if (ndim == 2)
    {
        // 单通道
        // 第一步：计算行前缀和
        const dim3 block_size_row(THREADS);
        const dim3 grid_size_row(divUp(src_h, THREADS));
        integral_row_prefix<uint8_t, uint32_t, 1>
            <<<grid_size_row, block_size_row>>>(src.data_ptr<uint8_t>(), dst.data_ptr<uint32_t>(), src_h, src_w, dst_w);

        // 第二步：计算列前缀和
        const dim3 block_size_col(THREADS);
        const dim3 grid_size_col(divUp(dst_w, THREADS));
        integral_col_prefix<uint32_t, 1>
            <<<grid_size_col, block_size_col>>>(dst.data_ptr<uint32_t>(), src_h, src_w, dst_w);
    }
    else if (ndim == 3 && src.size(2) == 3)
    {
        // 三通道
        // 第一步：计算行前缀和
        const dim3 block_size_row(THREADS);
        const dim3 grid_size_row(divUp(src_h, THREADS));
        integral_row_prefix<uint8_t, uint32_t, 3>
            <<<grid_size_row, block_size_row>>>(src.data_ptr<uint8_t>(), dst.data_ptr<uint32_t>(), src_h, src_w, dst_w);

        // 第二步：计算列前缀和
        const dim3 block_size_col(THREADS);
        const dim3 grid_size_col(divUp(dst_w, THREADS));
        integral_col_prefix<uint32_t, 3>
            <<<grid_size_col, block_size_col>>>(dst.data_ptr<uint32_t>(), src_h, src_w, dst_w);
    }
}

void integral_f32(torch::Tensor src, torch::Tensor dst)
{
    CHECK_TORCH_TENSOR_DTYPE(src, torch::kFloat32);
    CHECK_TORCH_TENSOR_DTYPE(dst, torch::kFloat32);
    CHECK_TORCH_TENSOR_DEVICE(src);
    CHECK_TORCH_TENSOR_DEVICE(dst);

    const int ndim  = src.dim();
    const int src_h = src.size(0);
    const int src_w = src.size(1);
    const int dst_h = dst.size(0);
    const int dst_w = dst.size(1);

    // 初始化第一行为0
    cudaMemset(dst.data_ptr<float>(), 0, dst_w * (ndim == 2 ? 1 : 3) * sizeof(float));

    if (ndim == 2)
    {
        // 单通道
        // 第一步：计算行前缀和
        const dim3 block_size_row(THREADS);
        const dim3 grid_size_row(divUp(src_h, THREADS));
        integral_row_prefix<float, float, 1>
            <<<grid_size_row, block_size_row>>>(src.data_ptr<float>(), dst.data_ptr<float>(), src_h, src_w, dst_w);

        // 第二步：计算列前缀和
        const dim3 block_size_col(THREADS);
        const dim3 grid_size_col(divUp(dst_w, THREADS));
        integral_col_prefix<float, 1><<<grid_size_col, block_size_col>>>(dst.data_ptr<float>(), src_h, src_w, dst_w);
    }
    else if (ndim == 3 && src.size(2) == 3)
    {
        // 三通道
        // 第一步：计算行前缀和
        const dim3 block_size_row(THREADS);
        const dim3 grid_size_row(divUp(src_h, THREADS));
        integral_row_prefix<float, float, 3>
            <<<grid_size_row, block_size_row>>>(src.data_ptr<float>(), dst.data_ptr<float>(), src_h, src_w, dst_w);

        // 第二步：计算列前缀和
        const dim3 block_size_col(THREADS);
        const dim3 grid_size_col(divUp(dst_w, THREADS));
        integral_col_prefix<float, 3><<<grid_size_col, block_size_col>>>(dst.data_ptr<float>(), src_h, src_w, dst_w);
    }
}

// 通用调用函数，根据数据类型自动选择
torch::Tensor integral_cuda(torch::Tensor src)
{
    const int ndim  = src.dim();
    const int src_h = src.size(0);
    const int src_w = src.size(1);

    torch::Tensor dst;

    if (src.dtype() == torch::kUInt8)
    {
        if (ndim == 2)
        {
            dst = torch::zeros({src_h + 1, src_w + 1}, torch::kUInt32).cuda();
        }
        else if (ndim == 3)
        {
            dst = torch::zeros({src_h + 1, src_w + 1, src.size(2)}, torch::kUInt32).cuda();
        }
        integral_u8(src, dst);
    }
    else if (src.dtype() == torch::kFloat32)
    {
        if (ndim == 2)
        {
            dst = torch::zeros({src_h + 1, src_w + 1}, torch::kFloat32).cuda();
        }
        else if (ndim == 3)
        {
            dst = torch::zeros({src_h + 1, src_w + 1, src.size(2)}, torch::kFloat32).cuda();
        }
        integral_f32(src, dst);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported data type for integral computation");
    }

    return dst;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(integral_u8);
    TORCH_BINDING_COMMON_EXTENSION(integral_f32);
    TORCH_BINDING_COMMON_EXTENSION(integral_cuda);
}