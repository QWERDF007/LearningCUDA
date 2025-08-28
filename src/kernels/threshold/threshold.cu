#include "common.cuh"

#include <cuda_runtime.h>

/**
 * @brief 基础的uint8_t阈值化核函数
 * 
 * 对输入数组中的每个uint8_t元素进行阈值化处理：
 * - 如果元素值大于阈值th，则输出255
 * - 否则输出0
 * 
 * @param in 输入数组指针
 * @param th 阈值
 * @param out 输出数组指针
 * @param N 数组元素总数
 */
__global__ void threshold_u8_kernel(uint8_t *in, const uint8_t th, uint8_t *out, const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    out[idx] = in[idx] > th ? 255 : 0;
}

/**
 * @brief 2D布局的uint8_t阈值化核函数
 * 
 * 使用2D线程块布局处理图像数据，每个线程处理一个像素
 * 
 * @param in 输入图像数据指针
 * @param th 阈值
 * @param out 输出图像数据指针
 * @param img_w 图像宽度
 * @param img_h 图像高度
 */
__global__ void threshold_u8_kernel2D(uint8_t *in, const uint8_t th, uint8_t *out, const int img_w, const int img_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= img_w || y >= img_h)
        return;

    int idx  = x + y * img_w;
    out[idx] = in[idx] > th ? 255 : 0;
}

/**
 * @brief 向量化的uint8_t阈值化核函数
 * 
 * 使用uchar4向量类型，每个线程同时处理4个uint8_t元素，提高内存访问效率
 * 
 * @param in 输入数组指针
 * @param th 阈值
 * @param out 输出数组指针
 * @param N 数组元素总数
 */
__global__ void threshold_u8x4_kernel(uint8_t *in, const uint8_t th, uint8_t *out, const int N)
{
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= N)
        return;
    uchar4 in4 = *reinterpret_cast<uchar4 *>(&in[idx]);
    uchar4 out4;

    out4.x = in4.x > th ? 255 : 0;
    out4.y = in4.y > th ? 255 : 0;
    out4.z = in4.z > th ? 255 : 0;
    out4.w = in4.w > th ? 255 : 0;

    *reinterpret_cast<uchar4 *>(&out[idx]) = out4;
}

/**
 * @brief 2D布局的向量化uint8_t阈值化核函数
 * 
 * 结合2D线程块布局和向量化处理，每个线程处理4个连续的像素
 * 包含边界检查以处理宽度不是4的倍数的情况
 * 
 * @param in 输入图像数据指针
 * @param th 阈值
 * @param out 输出图像数据指针
 * @param img_w 图像宽度
 * @param img_h 图像高度
 */
__global__ void threshold_u8x4_kernel2D(uint8_t *in, const uint8_t th, uint8_t *out, const int img_w, const int img_h)
{
    // 每个线程处理4个像素，所以x坐标需要乘以4
    int x = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= img_w || y >= img_h)
        return;
    int idx = x + y * img_w;
    // 检查是否有足够的像素进行向量化处理
    if (x + 3 < img_w)
    {
        // 完整的4像素向量化处理
        uchar4 in4 = *reinterpret_cast<uchar4 *>(&in[idx]);
        uchar4 out4;

        out4.x = in4.x > th ? 255 : 0;
        out4.y = in4.y > th ? 255 : 0;
        out4.z = in4.z > th ? 255 : 0;
        out4.w = in4.w > th ? 255 : 0;

        *reinterpret_cast<uchar4 *>(&out[idx]) = out4;
    }
    else
    {
        // 处理剩余的像素（不足4个的情况）
        for (int i = 0; i < 4 && (x + i) < img_w; i++)
        {
            out[idx + i] = in[idx + i] > th ? 255 : 0;
        }
    }
}

/**
 * @brief 基础的float阈值化核函数
 * 
 * 对输入数组中的每个float元素进行阈值化处理：
 * - 如果元素值大于阈值th，则输出255.0
 * - 否则输出0.0
 * 
 * @param in 输入数组指针
 * @param th 阈值
 * @param out 输出数组指针
 * @param N 数组元素总数
 */
__global__ void threshold_f32_kernel(float *in, const float th, float *out, const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    out[idx] = in[idx] > th ? 255 : 0;
}

/**
 * @brief 2D布局的float阈值化核函数
 * 
 * 使用2D线程块布局处理float类型的图像数据，每个线程处理一个像素
 * 
 * @param in 输入图像数据指针
 * @param th 阈值
 * @param out 输出图像数据指针
 * @param img_w 图像宽度
 * @param img_h 图像高度
 */
__global__ void threshold_f32_kernel2D(float *in, const float th, float *out, const int img_w, const int img_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= img_w || y >= img_h)
        return;

    int idx  = x + y * img_w;
    out[idx] = in[idx] > th ? 255 : 0;
}

/**
 * @brief 向量化的float阈值化核函数
 * 
 * 使用float4向量类型，每个线程同时处理4个float元素，提高内存访问效率
 * 
 * @param in 输入数组指针
 * @param th 阈值
 * @param out 输出数组指针
 * @param N 数组元素总数
 */
__global__ void threshold_f32x4_kernel(float *in, const float th, float *out, const int N)
{
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= N)
        return;
    float4 in4 = *reinterpret_cast<float4 *>(&in[idx]);
    float4 out4;

    out4.x = in4.x > th ? 255 : 0;
    out4.y = in4.y > th ? 255 : 0;
    out4.z = in4.z > th ? 255 : 0;
    out4.w = in4.w > th ? 255 : 0;

    *reinterpret_cast<float4 *>(&out[idx]) = out4;
}

/**
 * @brief 2D布局的向量化float阈值化核函数
 * 
 * 结合2D线程块布局和向量化处理，每个线程处理4个连续的float像素
 * 包含边界检查以处理宽度不是4的倍数的情况
 * 
 * @param in 输入图像数据指针
 * @param th 阈值
 * @param out 输出图像数据指针
 * @param img_w 图像宽度
 * @param img_h 图像高度
 */
__global__ void threshold_f32x4_kernel2D(float *in, const float th, float *out, const int img_w, const int img_h)
{
    // 每个线程处理4个像素，所以x坐标需要乘以4
    int x = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= img_w || y >= img_h)
        return;
    int idx = x + y * img_w;
    // 检查是否有足够的像素进行向量化处理
    if (x + 3 < img_w)
    {
        // 完整的4像素向量化处理
        float4 in4 = *reinterpret_cast<float4 *>(&in[idx]);
        float4 out4;

        out4.x = in4.x > th ? 255 : 0;
        out4.y = in4.y > th ? 255 : 0;
        out4.z = in4.z > th ? 255 : 0;
        out4.w = in4.w > th ? 255 : 0;

        *reinterpret_cast<float4 *>(&out[idx]) = out4;
    }
    else
    {
        // 处理剩余的像素（不足4个的情况）
        for (int i = 0; i < 4 && (x + i) < img_w; i++)
        {
            out[idx + i] = in[idx + i] > th ? 255 : 0;
        }
    }
}

#define TORCH_BINDING_THRESHOLD_2D(packed_type, torch_type, element_type, n_elements)                                \
    void threshold_##packed_type##_2D(torch::Tensor in, element_type th, torch::Tensor out)                          \
    {                                                                                                                \
        CHECK_TORCH_TENSOR_DTYPE(in, torch_type)                                                                     \
        CHECK_TORCH_TENSOR_DTYPE(out, torch_type)                                                                    \
        CHECK_TORCH_TENSOR_DEVICE(in)                                                                                \
        CHECK_TORCH_TENSOR_DEVICE(out)                                                                               \
                                                                                                                     \
        const int H = in.size(0);                                                                                    \
        const int W = in.size(1);                                                                                    \
        dim3      block(BLOCK_SIZE_X, BLOCK_SIZE_Y);                                                                 \
        /* 由于每个线程处理n_elements个像素，所以网格的x维度需要除以n_elements */                                    \
        dim3      grid(divUp(W, block.x *n_elements), divUp(H, block.y));                                            \
        threshold_##packed_type##_kernel2D<<<grid, block>>>(reinterpret_cast<element_type *>(in.data_ptr()), th,     \
                                                            reinterpret_cast<element_type *>(out.data_ptr()), W, H); \
    }

#define TORCH_BINDING_THRESHOLD(packed_type, torch_type, element_type, n_elements)                              \
    void threshold_##packed_type(torch::Tensor in, element_type th, torch::Tensor out)                          \
    {                                                                                                           \
        CHECK_TORCH_TENSOR_DTYPE(in, torch_type)                                                                \
        CHECK_TORCH_TENSOR_DTYPE(out, torch_type)                                                               \
        CHECK_TORCH_TENSOR_DEVICE(in)                                                                           \
        CHECK_TORCH_TENSOR_DEVICE(out)                                                                          \
                                                                                                                \
        const int H = in.size(0);                                                                               \
        const int W = in.size(1);                                                                               \
        const int N = H * W;                                                                                    \
        dim3      block(THREADS / n_elements);                                                                  \
        dim3      grid(divUp(N, THREADS));                                                                      \
        threshold_##packed_type##_kernel<<<grid, block>>>(reinterpret_cast<element_type *>(in.data_ptr()), th,  \
                                                          reinterpret_cast<element_type *>(out.data_ptr()), N); \
    }

TORCH_BINDING_THRESHOLD(u8, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_THRESHOLD(u8x4, torch::kUInt8, uint8_t, 4)
TORCH_BINDING_THRESHOLD(f32, torch::kFloat32, float, 1)
TORCH_BINDING_THRESHOLD(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_THRESHOLD_2D(u8, torch::kUInt8, uint8_t, 1)
TORCH_BINDING_THRESHOLD_2D(u8x4, torch::kUInt8, uint8_t, 4)
TORCH_BINDING_THRESHOLD_2D(f32, torch::kFloat32, float, 1)
TORCH_BINDING_THRESHOLD_2D(f32x4, torch::kFloat32, float, 4)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(threshold_u8)
    TORCH_BINDING_COMMON_EXTENSION(threshold_u8x4)
    TORCH_BINDING_COMMON_EXTENSION(threshold_f32)
    TORCH_BINDING_COMMON_EXTENSION(threshold_f32x4)
    TORCH_BINDING_COMMON_EXTENSION(threshold_u8_2D)
    TORCH_BINDING_COMMON_EXTENSION(threshold_u8x4_2D)
    TORCH_BINDING_COMMON_EXTENSION(threshold_f32_2D)
    TORCH_BINDING_COMMON_EXTENSION(threshold_f32x4_2D)
}