
#include "common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <iostream>

#define THREADS 512

#define FLOAT4(value)      (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value)       (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

/**
 * @brief 逐元素加法操作, 一次处理1个float元素
 * 
 * @param a 输入数组A的指针
 * @param b 输入数组B的指针  
 * @param c 输出数组C的指针，存储结果
 * @param N 数组元素总数
 */
__global__ void elementwise_add_f32_kernel(float *a, float *b, float *c, const int N)
{
    // 计算当前线程的全局索引
    // blockIdx.x: 当前线程块在网格中的x维索引
    // blockDim.x: 线程块在x维的大小（每个块的线程数）
    // threadIdx.x: 当前线程在线程块中的x维索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 边界检查：确保不会越界访问数组
    if (idx >= N)
        return;
    // 元素相加, 写回结果
    c[idx] = a[idx] + b[idx];
}

/**
 * @brief 向量化的逐元素加法操作（float4版本）
 * 使用float4向量类型一次处理4个float元素，提高内存带宽利用率和计算效率
 * 
 * @param a 输入数组A的指针
 * @param b 输入数组B的指针  
 * @param c 输出数组C的指针，存储结果
 * @param N 数组元素总数
 */
__global__ void elementwise_add_f32x4_kernel(float *a, float *b, float *c, int N)
{
    // 计算当前线程处理的起始索引，每个线程处理4个连续的float元素
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);

    if (idx >= N)
        return;
    // 使用FLOAT4宏将连续的4个float元素加载为一个float4向量
    // 这样可以减少内存访问次数，提高内存带宽利用率
    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);
    float4 reg_c;

    // 对float4向量的每个分量执行加法运算
    // x, y, z, w分别对应4个连续的float元素
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;

    // 将计算结果写回到输出数组，同样使用向量化存储
    FLOAT4(c[idx]) = reg_c;
}

/**
 * @brief 半精度浮点数逐元素加法操作（half版本）
 * 使用CUDA内置的half数据类型进行计算，相比float32可以节省一半的内存带宽
 * 适用于对精度要求不高但对性能要求较高的场景
 * 
 * @param a 输入数组A的指针，指向half类型数据
 * @param b 输入数组B的指针，指向half类型数据
 * @param c 输出数组C的指针，存储half类型结果
 * @param N 数组元素总数
 */
__global__ void elementwise_add_f16_kernel(half *a, half *b, half *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;
    // 使用CUDA内置的__hadd函数执行half精度加法运算
    // __hadd是专门为half类型优化的加法函数，比直接使用+运算符更高效
    c[idx] = __hadd(a[idx], b[idx]);
}

/**
 * @brief 半精度浮点数逐元素加法操作（half2版本）
 * 使用half2向量类型一次处理2个half元素，提高内存带宽利用率和计算效率
 * 相比单个half处理，可以减少内存访问次数并提高并行度
 * 
 * @param a 输入数组A的指针，指向half类型数据
 * @param b 输入数组B的指针，指向half类型数据
 * @param c 输出数组C的指针，存储half类型结果
 * @param N 数组元素总数
 */
__global__ void elementwise_add_f16x2_kernel(half *a, half *b, half *c, int N)
{
    // 计算当前线程处理的起始索引，每个线程处理2个连续的half元素
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

    if (idx >= N)
        return;
    // 使用HALF2宏将连续的2个half元素加载为一个half2向量
    // 这样可以减少内存访问次数，提高内存带宽利用率
    half2 reg_a = HALF2(a[idx]);
    half2 reg_b = HALF2(b[idx]);
    half2 reg_c;

    // 对half2向量的每个分量执行加法运算
    // x, y分别对应2个连续的half元素
    // 使用CUDA内置的__hadd函数执行half精度加法运算
    reg_c.x = __hadd(reg_a.x, reg_b.x); // 第1个元素相加
    reg_c.y = __hadd(reg_a.y, reg_b.y); // 第2个元素相加

    // 将计算结果写回到输出数组，同样使用向量化存储
    HALF2(c[idx]) = reg_c;
}

/**
 * @brief 半精度浮点数逐元素加法操作（half2x4版本，处理8个half元素）
 * 使用4个half2向量同时处理8个half元素，进一步提高内存带宽利用率和计算并行度
 * 每个线程处理8个连续的half元素，相比f16x2版本可以减少线程数量，提高GPU占用率
 * 
 * @param a 输入数组A的指针，指向half类型数据
 * @param b 输入数组B的指针，指向half类型数据  
 * @param c 输出数组C的指针，存储half类型结果
 * @param N 数组元素总数
 */
__global__ void elementwise_add_f16x8_kernel(half *a, half *b, half *c, int N)
{
    // 计算当前线程处理的起始索引，每个线程处理8个half元素
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);

    // 从输入数组a中加载4个half2向量（共8个half元素）
    // 使用HALF2宏进行向量化内存访问，提高内存带宽利用率
    half2 reg_a_0 = HALF2(a[idx + 0]); // 加载第1-2个元素
    half2 reg_a_1 = HALF2(a[idx + 2]); // 加载第3-4个元素
    half2 reg_a_2 = HALF2(a[idx + 4]); // 加载第5-6个元素
    half2 reg_a_3 = HALF2(a[idx + 6]); // 加载第7-8个元素

    // 从输入数组b中加载4个half2向量（共8个half元素）
    half2 reg_b_0 = HALF2(b[idx + 0]); // 加载第1-2个元素
    half2 reg_b_1 = HALF2(b[idx + 2]); // 加载第3-4个元素
    half2 reg_b_2 = HALF2(b[idx + 4]); // 加载第5-6个元素
    half2 reg_b_3 = HALF2(b[idx + 6]); // 加载第7-8个元素

    // 声明4个half2结果向量，用于存储计算结果
    half2 reg_c_0, reg_c_1, reg_c_2, reg_c_3;

    // 对第1个half2向量执行逐元素加法运算
    reg_c_0.x = __hadd(reg_a_0.x, reg_b_0.x); // 第1个元素相加
    reg_c_0.y = __hadd(reg_a_0.y, reg_b_0.y); // 第2个元素相加

    // 对第2个half2向量执行逐元素加法运算
    reg_c_1.x = __hadd(reg_a_1.x, reg_b_1.x); // 第3个元素相加
    reg_c_1.y = __hadd(reg_a_1.y, reg_b_1.y); // 第4个元素相加

    // 对第3个half2向量执行逐元素加法运算
    reg_c_2.x = __hadd(reg_a_2.x, reg_b_2.x); // 第5个元素相加
    reg_c_2.y = __hadd(reg_a_2.y, reg_b_2.y); // 第6个元素相加

    // 对第4个half2向量执行逐元素加法运算
    reg_c_3.x = __hadd(reg_a_3.x, reg_b_3.x); // 第7个元素相加
    reg_c_3.y = __hadd(reg_a_3.y, reg_b_3.y); // 第8个元素相加

    // 边界检查并写回结果：确保不会越界访问内存
    // 由于每个线程处理8个元素，需要分别检查每对元素是否在有效范围内
    if ((idx + 0) < N)
    {
        HALF2(c[idx + 0]) = reg_c_0; // 写回第1-2个元素的结果
    }
    if ((idx + 2) < N)
    {
        HALF2(c[idx + 2]) = reg_c_1; // 写回第3-4个元素的结果
    }
    if ((idx + 4) < N)
    {
        HALF2(c[idx + 4]) = reg_c_2; // 写回第5-6个元素的结果
    }
    if ((idx + 6) < N)
    {
        HALF2(c[idx + 6]) = reg_c_3; // 写回第7-8个元素的结果
    }
}

/**
 * @brief 优化的向量化逐元素加法操作（half精度，打包版本）
 * 使用128位内存访问一次处理8个half元素，通过打包数组和循环展开提高性能
 * 相比于直接使用half2向量，这种方法可以更好地利用内存带宽和寄存器
 * 
 * @param a 输入数组A的指针（half精度）
 * @param b 输入数组B的指针（half精度）
 * @param c 输出数组C的指针，存储结果（half精度）
 * @param N 数组元素总数
 */
__global__ void elementwise_add_f16x8_pack_kernel(half *a, half *b, half *c, int N)
{
    // 计算当前线程处理的起始索引，每个线程处理8个连续的half元素
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);

    // 声明临时寄存器数组，存储在.local内存空间中，可寻址
    // 每个数组包含8个half元素，总共8x16位=128位
    half pack_a[8], pack_b[8], pack_c[8]; // 8x16 bits=128 bits.

    // 使用128位内存访问指令一次性加载8个half元素
    // LDST128BITS宏将连续的8个half元素重新解释为128位数据进行加载
    // 这样可以减少内存访问次数，提高内存带宽利用率
    LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // 从数组a加载128位数据
    LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]); // 从数组b加载128位数据

    // 使用编译器指令展开循环，减少循环开销
#pragma unroll
    for (int i = 0; i < 8; i += 2)
    {
        // 使用__hadd2指令对half2向量执行并行加法运算
        // 每次迭代处理2个half元素，总共4次迭代处理8个元素
        // HALF2宏将连续的2个half元素重新解释为一个half2向量
        HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
    }

    // 边界检查和结果写回
    // 如果当前线程处理的所有8个元素都在有效范围内
    if ((idx + 7) < N)
    {
        // 使用128位内存访问指令一次性存储8个half元素的结果
        // 这样可以减少内存访问次数，提高内存带宽利用率
        LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
    }
    else
    {
        // 如果部分元素超出边界，则逐个处理剩余的有效元素
        // 使用标量加法运算确保不会越界访问内存
        for (int i = 0; idx + i < N; i++)
        {
            c[idx + i] = __hadd(a[idx + i], b[idx + i]);
        }
    }
}

/**
 * @brief PyTorch绑定宏, 该宏用于自动生成不同数据类型的逐元素加法函数，减少代码重复
 * 
 * @param packed_type 打包数据类型标识符（如f32, f32x4, f16等），用于生成函数名和核函数名
 * @param torch_type PyTorch 张量数据类型（如torch::kFloat32），用于类型检查
 * @param element_type 底层元素数据类型（如float, half等），用于指针类型转换
 * @param n_elements 每个打包类型包含的元素数量，用于计算线程块大小
 */
#define TORCH_BINDING_ELEM_ADD(packed_type, torch_type, element_type, n_elements)                                   \
    void elementwise_add_##packed_type(torch::Tensor a, torch::Tensor b, torch::Tensor c)                           \
    {                                                                                                               \
        CHECK_TORCH_TENSOR_DTYPE(a, torch_type)                                                                     \
        CHECK_TORCH_TENSOR_DTYPE(b, torch_type)                                                                     \
        CHECK_TORCH_TENSOR_DTYPE(c, torch_type)                                                                     \
                                                                                                                    \
        CHECK_TORCH_TENSOR_DEVICE(a)                                                                                \
        CHECK_TORCH_TENSOR_DEVICE(b)                                                                                \
        CHECK_TORCH_TENSOR_DEVICE(c)                                                                                \
                                                                                                                    \
        const int N = a.numel();                                                                                    \
        dim3      block(THREADS / n_elements);                                                                      \
        dim3      grid((N + THREADS - 1) / THREADS);                                                                \
        elementwise_add_##packed_type##_kernel<<<grid, block>>>(reinterpret_cast<element_type *>(a.data_ptr()),     \
                                                                reinterpret_cast<element_type *>(b.data_ptr()),     \
                                                                reinterpret_cast<element_type *>(c.data_ptr()), N); \
    }

TORCH_BINDING_ELEM_ADD(f32, torch::kFloat32, float, 1)
TORCH_BINDING_ELEM_ADD(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_ELEM_ADD(f16, torch::kHalf, half, 1)
TORCH_BINDING_ELEM_ADD(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_ELEM_ADD(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_ELEM_ADD(f16x8_pack, torch::kHalf, half, 8)

/**
 * Python绑定模块定义
 * PYBIND11_MODULE宏用于创建Python可调用的模块
 * TORCH_EXTENSION_NAME是PyTorch自动生成的模块名称
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32x4)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x2)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8_pack)
}