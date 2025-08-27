#pragma once

#include <torch/extension.h>

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
