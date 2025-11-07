#include "common.cuh"

#define CLIP(x, a, b) (max((a), min((b), (x))))

// ------------------------------------
// Step 1: 每个 tile 计算直方图 + 裁剪 + 生成 LUT
// ------------------------------------
__global__ void clahe_compute_lut_kernel(const uint8_t *__restrict__ src, uint8_t *__restrict__ lut, int width,
                                         int height, int tileW, int tileH, int gridX, int gridY, int clipLimit)
{
    int tx  = blockIdx.x;
    int ty  = blockIdx.y;
    int tid = threadIdx.x;

    if (tx >= gridX || ty >= gridY)
        return;

    __shared__ uint32_t hist[256];
    // 初始化直方图
    for (int i = tid; i < 256; i += blockDim.x)
    {
        hist[i] = 0;
    }
    __syncthreads();

    // 当前 tile 区域范围（与 OpenCV 对齐）
    int x0 = tx * tileW;
    int y0 = ty * tileH;
    int x1 = min(x0 + tileW, width);
    int y1 = min(y0 + tileH, height);

    // 统计局部直方图
    for (int y = y0 + threadIdx.x; y < y1; y += blockDim.x)
    {
        for (int x = x0; x < x1; ++x)
        {
            atomicAdd(&hist[src[y * width + x]], 1);
        }
    }
    __syncthreads();

    // OpenCV 风格的裁剪和重新分配逻辑
    if (clipLimit > 0)
    {
        // Step 1: 计算被裁剪的像素总数（使用原子操作累加）
        __shared__ uint32_t clipped_total;
        if (tid == 0)
            clipped_total = 0;
        __syncthreads();

        for (int i = tid; i < 256; i += blockDim.x)
        {
            if (hist[i] > (uint32_t)clipLimit)
            {
                uint32_t excess = hist[i] - clipLimit;
                atomicAdd(&clipped_total, excess);
                hist[i] = clipLimit;
            }
        }
        __syncthreads();

        // Step 2: 计算平均分配和余数
        uint32_t redistBatch = clipped_total / 256;
        uint32_t residual    = clipped_total - redistBatch * 256;

        // Step 3: 先平均分配
        for (int i = tid; i < 256; i += blockDim.x)
        {
            hist[i] += redistBatch;
        }
        __syncthreads();

        // Step 4: 分配余数（与 OpenCV 逻辑对齐：residualStep = MAX(histSize / residual, 1)）
        // OpenCV 使用串行方式处理余数，这里也采用串行方式以保证完全一致
        if (tid == 0 && residual > 0)
        {
            int residualStep = max(256 / (int)residual, 1);
            for (int i = 0; i < 256 && residual > 0; i += residualStep, residual--)
            {
                hist[i]++;
            }
        }
        __syncthreads();
    }

    // CDF + LUT 计算（与 OpenCV 对齐：使用 (histSize - 1) / total）
    __shared__ uint8_t tileLUT[256];
    if (tid == 0)
    {
        uint32_t sum   = 0;
        uint32_t total = (x1 - x0) * (y1 - y0);
        float    scale = 255.0f / (float)total; // 等价于 (256 - 1) / total

        for (int i = 0; i < 256; ++i)
        {
            sum += hist[i];
            tileLUT[i] = saturate_cast<uint8_t>(sum * scale);
        }
    }
    __syncthreads();

    // 写回到全局 LUT 表
    for (int i = tid; i < 256; i += blockDim.x)
    {
        // 每个 tile 有自己的一组 LUT
        lut[(ty * gridX + tx) * 256 + i] = tileLUT[i];
    }
}

// ------------------------------------
// Step 2: 应用 LUT + 双线性插值（与 OpenCV 对齐）
// ------------------------------------
__global__ void clahe_apply_kernel(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst,
                                   const uint8_t *__restrict__ lut, int width, int height, int tileW, int tileH,
                                   int gridX, int gridY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    // 与 OpenCV 对齐：使用 inv_tw/inv_th 和 -0.5f 偏移
    float inv_tw = 1.0f / (float)tileW;
    float inv_th = 1.0f / (float)tileH;

    float txf = (float)x * inv_tw - 0.5f;
    float tyf = (float)y * inv_th - 0.5f;

    // 使用 cvFloor 等价逻辑（向下取整）
    // cvFloor(x) = floor(x)，在 CUDA 中直接使用 floorf
    int tx1 = (int)floorf(txf);
    int ty1 = (int)floorf(tyf);
    int tx2 = tx1 + 1;
    int ty2 = ty1 + 1;

    // 计算插值权重（与 OpenCV 对齐）
    float xa  = txf - (float)tx1;
    float xa1 = 1.0f - xa;
    float ya  = tyf - (float)ty1;
    float ya1 = 1.0f - ya;

    // 边界裁剪（与 OpenCV 对齐：std::max/min）
    tx1 = max(tx1, 0);
    tx2 = min(tx2, gridX - 1);
    ty1 = max(ty1, 0);
    ty2 = min(ty2, gridY - 1);

    int     idx = y * width + x;
    uint8_t val = src[idx];

    // 计算 LUT 偏移量（与 OpenCV 对齐：lut_step = lut_.step / sizeof(T)）
    int lut_step = 256; // 每个 tile 的 LUT 大小
    int ind1     = tx1 * lut_step;
    int ind2     = tx2 * lut_step;

    // 获取周围4个 tile 的 LUT（与 OpenCV 对齐）
    const uint8_t *lutPlane1 = lut + (ty1 * gridX * lut_step);
    const uint8_t *lutPlane2 = lut + (ty2 * gridX * lut_step);

    // 双线性插值（与 OpenCV 公式完全对齐）
    float res = (lutPlane1[ind1 + val] * xa1 + lutPlane1[ind2 + val] * xa) * ya1
              + (lutPlane2[ind1 + val] * xa1 + lutPlane2[ind2 + val] * xa) * ya;

    dst[idx] = saturate_cast<uint8_t>(res);
}

void clahe(torch::Tensor src, torch::Tensor dst, const double clipLimit = 40, const int tileGridX = 8,
           const int tileGridY = 8)
{
    CHECK_TORCH_TENSOR_DTYPE(src, torch::kUInt8)
    CHECK_TORCH_TENSOR_DTYPE(dst, torch::kUInt8)
    CHECK_TORCH_TENSOR_DEVICE(src)
    CHECK_TORCH_TENSOR_DEVICE(dst)

    auto lut_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, 0);

    torch::Tensor lut = torch::zeros({tileGridX * tileGridY * 256}, lut_options);

    const int H = src.size(0);
    const int W = src.size(1);
    const int N = H * W;

    // 计算 tile 大小（与 OpenCV 对齐：使用 divUp 确保覆盖整个图像）
    int       tileW         = divUp(W, tileGridX);
    int       tileH         = divUp(H, tileGridY);
    const int tileSizeTotal = tileW * tileH;

    const int histSize = 256; // CV_8UC1 ? 256 : 65536

    // 将 clipLimit 从相对值转换为绝对值（与 OpenCV 对齐）
    // OpenCV: clipLimit = static_cast<int>(clipLimit_ * tileSizeTotal / histSize);
    int clipLimit_abs = 0;
    if (clipLimit > 0.0)
    {
        clipLimit_abs = static_cast<int>(clipLimit * tileSizeTotal / histSize);
        clipLimit_abs = max(clipLimit_abs, 1); // 至少为 1
    }

    dim3 blockHist(THREADS);
    dim3 gridHist(tileGridX, tileGridY);
    clahe_compute_lut_kernel<<<gridHist, blockHist>>>(reinterpret_cast<uint8_t *>(src.data_ptr()),
                                                      reinterpret_cast<uint8_t *>(lut.data_ptr()), W, H, tileW, tileH,
                                                      tileGridX, tileGridY, clipLimit_abs);

    dim3 blockApply(16, 16);
    dim3 gridApply(divUp(W, blockApply.x), divUp(H, blockApply.y));
    clahe_apply_kernel<<<gridApply, blockApply>>>(
        reinterpret_cast<uint8_t *>(src.data_ptr()), reinterpret_cast<uint8_t *>(dst.data_ptr()),
        reinterpret_cast<uint8_t *>(lut.data_ptr()), W, H, tileW, tileH, tileGridX, tileGridY);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(clahe)
}