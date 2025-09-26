#pragma once

#include "border.cuh"
#include "saturate.cuh"

#include <torch/extension.h>

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 16
#define THREADS      256

#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)

/**
 * @brief 128位内存访问宏：将任意类型的值重新解释为float4向量进行128位内存访问
 * 通过将连续的内存区域重新解释为float4类型，实现一次性读写16个字节（128位）的数据，
 * 提高内存带宽利用率和访问效率。常用于向量化操作中的批量数据加载和存储。
 * 
 * @param value 要进行128位访问的内存位置的引用
 * @return float4& 返回对应内存位置的float4引用，可用于读写操作
 */
#define LDST128BITS(value) (*reinterpret_cast<float4 *>(&(value)))

/**
 * 字符串化宏：将参数转换为字符串字面量
 * 用于在编译时将函数名或变量名转换为对应的字符串
 */
#define STRINGFY(str) #str

/**
 * @brief PyTorch绑定通用扩展宏：简化函数绑定过程
 * 自动将C++函数绑定到Python模块，使用函数名作为Python中的名称和文档字符串
 * 
 * @param func 要绑定的C++函数名
 * 
 */
#define TORCH_BINDING_COMMON_EXTENSION(func) m.def(STRINGFY(func), &func, STRINGFY(func));

/**
 * @brief PyTorch张量数据类型检查宏：验证张量的数据类型
 * 简化类型检查代码，提供统一的错误信息格式
 * 
 * @param T 要检查的张量对象
 * @param th_type 期望的PyTorch数据类型（如torch::kFloat32）
 * 
 */
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type) TORCH_CHECK((T).dtype() == th_type, "Tensor " #T " must be " #th_type);

/**
 * @brief PyTorch张量设备检查宏：验证张量的设备
 * 简化设备检查代码，提供统一的错误信息格式
 * 
 * @param T 要检查的张量对象
 * 
 */
#define CHECK_TORCH_TENSOR_DEVICE(T) TORCH_CHECK((T).device().is_cuda(), "Tensor " #T " must be on CUDA device");

inline static int divUp(int a, int b)
{
    return (a + b - 1) / b;
}

// OpenCV的常量定义
#define CV_PI  3.1415926535897932384626433832795
#define CV_2PI 6.283185307179586476925286766559

// OpenCV的定点算术常量
static const int INTER_RESIZE_COEF_BITS    = 11;
static const int INTER_RESIZE_COEF_SCALE   = 1 << INTER_RESIZE_COEF_BITS;                       // 2048
static const int SHIFT                     = INTER_RESIZE_COEF_BITS * 2;                        // 22
static const int DELTA                     = 1 << (SHIFT - 1);                                  // 2097152
static const int INTER_RESIZE_COEF_SCALE_2 = INTER_RESIZE_COEF_SCALE * INTER_RESIZE_COEF_SCALE; // 4194304
