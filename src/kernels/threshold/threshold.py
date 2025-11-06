
import time
import re
from pathlib import Path
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

import cv2
import numpy as np

# 禁用梯度计算，因为我们只进行推理，不需要反向传播
torch.set_grad_enabled(False)

# 获取当前Python文件的路径对象
file = Path(__file__)
# 将 src/common 加入到 sys.path，便于导入 helper
import sys
sys.path.append(str(file.parent.parent.parent / "common"))
from helper import compute_accuracy_info

# 构建编译所需的源文件列表
sources = [
    str(file.parent / "threshold.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

# 动态编译并加载CUDA核函数作为Python模块
# 这种JIT（即时编译）方式允许在运行时编译CUDA代码
lib = load(
    name="threshold_lib",                      # 模块名称
    sources=sources,                           # CUDA源文件列表
    extra_include_paths=extra_include_paths,   # 添加头文件搜索路径
    extra_cuda_cflags=[                        # 额外的CUDA编译器标志
        "-O3",                                 # 启用最高级别的优化
        "-U__CUDA_NO_HALF_OPERATORS__",        # 取消定义，启用half精度运算符
        "-U__CUDA_NO_HALF_CONVERSIONS__",      # 取消定义，启用half精度转换
        "-U__CUDA_NO_HALF2_OPERATORS__",       # 取消定义，启用half2精度运算符
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",  # 取消定义，启用bfloat16转换
        "--expt-relaxed-constexpr",            # 启用实验性的宽松constexpr支持
        "--expt-extended-lambda",              # 启用实验性的扩展lambda支持
        "--use_fast_math",                     # 使用快速数学库，可能牺牲精度换取性能
    ],
    extra_cflags=["-std=c++17"],               # 额外的C++编译器标志，使用C++17标准（Linux用-std）
)


def run_benchmark(
    perf_func: callable,                 # 要测试的函数，可调用对象
    a: torch.Tensor,                     # 输入张量
    thresh,                              # 阈值
    maxval,                              # 最大值
    tag: str,                            # 测试标签，用于输出识别
    out: Optional[torch.Tensor] = None,  # 可选的输出张量，如果提供则作为输出容器
    warmup: int = 20,                    # 预热次数，用于GPU预热和缓存优化
    iters: int = 1000,                   # 正式测试的迭代次数
    show_all: bool = False,              # 是否显示完整的输出张量
    op: int = cv2.THRESH_BINARY,
):
    """
    性能基准测试函数
    用于测量CUDA核函数的执行时间性能
    """
    if out is not None:
        out.fill_(0)
    
    for i in range(warmup):
        perf_func(a, out, thresh, maxval)
    
    torch.cuda.synchronize()
    
    start = time.time()
    
    if out is not None:
        for i in range(iters):
            perf_func(a, out, thresh, maxval)
    else:
        out = torch.zeros_like(a)
        for i in range(iters):
            perf_func(a, out, thresh, maxval)
    
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000  # 总时间，转换为毫秒
    mean_time = total_time / iters  # 平均每次迭代的时间

    out_np = out.cpu().numpy()
    _, expected = cv2.threshold(a.cpu().numpy(), thresh, maxval, op)

    mismatch_info = compute_accuracy_info(out_np, expected)

    out_info = f"out_{tag}" 
    print(f"{out_info:>40}: {mismatch_info}, iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
    
    return out, mean_time

# 定义测试用的张量高度列表
Hs = [1024, 2048, 4096]
# 定义测试用的张量宽度列表  
Ws = [1024, 2048, 4096, 46000]
OPs = ['THRESH_BINARY', 'THRESH_BINARY_INV', 'THRESH_TRUNC', 'THRESH_TOZERO', 'THRESH_TOZERO_INV']
# 生成所有可能的(高度, 宽度)组合
Sizes = [(H, W, op) for H in Hs for W in Ws for op in OPs]

# 如果直接运行此脚本，则执行完整测试
if __name__ == "__main__":
    # 遍历所有尺寸组合进行性能测试
    for H, W, op in Sizes:
        print("-" * 85)
        print(" " * 40 + f"H={H}, W={W}, ch=1, op={op}")

        # 创建uint8类型的测试张量
        a = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
        out = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()

        op_val = eval(f'cv2.{op}')
        
        # 运行uint8类型的基础性能基准测试（仅对THRESH_BINARY测试所有变种以避免重复）
        if op_val == cv2.THRESH_BINARY:
            run_benchmark(lib.threshold_u8, a, 128, 255, "u8", out, op=op_val)
            run_benchmark(lib.threshold_u8x4, a, 128, 255, "u8x4", out, op=op_val)
            run_benchmark(lib.threshold_u8x16_pack, a, 128, 255, "u8x16_pack", out, op=op_val)
            run_benchmark(lib.threshold_u8_2D, a, 128, 255, "u8_2D", out, op=op_val)
            run_benchmark(lib.threshold_u8x4_2D, a, 128, 255, "u8x4_2D", out, op=op_val)
            run_benchmark(lambda x, _, th, maxval: F.threshold(x, th, maxval), a, 128, 255, "u8_torch", op=op_val)
        

        # 测试threshold_binary系列函数 (uint8_t) - 根据op类型选择对应函数
        if op_val == cv2.THRESH_BINARY:
            run_benchmark(lib.threshold_binary_uint8_t, a, 128, 255, "u8_binary", out, op=op_val)
        elif op_val == cv2.THRESH_BINARY_INV:
            run_benchmark(lib.threshold_binary_inv_uint8_t, a, 128, 255, "u8_binary_inv", out, op=op_val)
        elif op_val == cv2.THRESH_TRUNC:
            run_benchmark(lib.threshold_trunc_uint8_t, a, 128, 255, "u8_trunc", out, op=op_val)
        elif op_val == cv2.THRESH_TOZERO:
            run_benchmark(lib.threshold_tozero_uint8_t, a, 128, 255, "u8_tozero", out, op=op_val)
        elif op_val == cv2.THRESH_TOZERO_INV:
            run_benchmark(lib.threshold_tozero_inv_uint8_t, a, 128, 255, "u8_tozero_inv", out, op=op_val)
        
        print(" " * 40 + f"H={H}, W={W}, ch=3, op={op}")
        # 测试3通道图像 (uint8_t)
        a_3ch = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8).cuda().contiguous()
        out_3ch = torch.zeros_like(a_3ch).cuda().contiguous()
        
        if op_val == cv2.THRESH_BINARY:
            run_benchmark(lib.threshold_binary_uint8_t, a_3ch, 128, 255, "u8_binary", out_3ch, op=op_val)
        elif op_val == cv2.THRESH_BINARY_INV:
            run_benchmark(lib.threshold_binary_inv_uint8_t, a_3ch, 128, 255, "u8_binary_inv", out_3ch, op=op_val)
        elif op_val == cv2.THRESH_TRUNC:
            run_benchmark(lib.threshold_trunc_uint8_t, a_3ch, 128, 255, "u8_trunc", out_3ch, op=op_val)
        elif op_val == cv2.THRESH_TOZERO:
            run_benchmark(lib.threshold_tozero_uint8_t, a_3ch, 128, 255, "u8_tozero", out_3ch, op=op_val)
        elif op_val == cv2.THRESH_TOZERO_INV:
            run_benchmark(lib.threshold_tozero_inv_uint8_t, a_3ch, 128, 255, "u8_tozero_inv", out_3ch, op=op_val)

        print("-" * 85)

        if W > 4096:
            continue

        print(" " * 40 + f"H={H}, W={W}, ch=1, op={op}")

        # 创建float32类型的测试张量
        a = torch.randn((H, W), dtype=torch.float32).cuda().contiguous()
        out = torch.randn((H, W), dtype=torch.float32).cuda().contiguous()

        # 运行float32类型的基础性能基准测试（仅对THRESH_BINARY测试所有变种以避免重复）
        if op_val == cv2.THRESH_BINARY:
            run_benchmark(lib.threshold_f32, a, 0.5, 1.0, "f32", out, op=op_val)
            run_benchmark(lib.threshold_f32_2D, a, 0.5, 1.0, "f32_2D", out, op=op_val)
            run_benchmark(lib.threshold_f32x4, a, 0.5, 1.0, "f32x4", out, op=op_val)
            run_benchmark(lib.threshold_f32x4_2D, a, 0.5, 1.0, "f32x4_2D", out, op=op_val)
            run_benchmark(lambda x, _, th, maxval: F.threshold(x, th, maxval), a, 0.5, 1.0, "f32_torch", op=op_val)

        # 测试threshold_binary系列函数 (float) - 根据op类型选择对应函数
        if op_val == cv2.THRESH_BINARY:
            run_benchmark(lib.threshold_binary_float, a, 0.5, 1.0, "f32_binary", out, op=op_val)
        elif op_val == cv2.THRESH_BINARY_INV:
            run_benchmark(lib.threshold_binary_inv_float, a, 0.5, 1.0, "f32_binary_inv", out, op=op_val)
        elif op_val == cv2.THRESH_TRUNC:
            run_benchmark(lib.threshold_trunc_float, a, 0.5, 1.0, "f32_trunc", out, op=op_val)
        elif op_val == cv2.THRESH_TOZERO:
            run_benchmark(lib.threshold_tozero_float, a, 0.5, 1.0, "f32_tozero", out, op=op_val)
        elif op_val == cv2.THRESH_TOZERO_INV:
            run_benchmark(lib.threshold_tozero_inv_float, a, 0.5, 1.0, "f32_tozero_inv", out, op=op_val)
        
        print(" " * 40 + f"H={H}, W={W}, ch=3, op={op}")
        # 测试3通道图像 (float)
        a_3ch_f32 = torch.randn((H, W, 3), dtype=torch.float32).cuda().contiguous()
        out_3ch_f32 = torch.zeros_like(a_3ch_f32).cuda().contiguous()
        
        if op_val == cv2.THRESH_BINARY:
            run_benchmark(lib.threshold_binary_float, a_3ch_f32, 0.5, 1.0, "f32_binary", out_3ch_f32, op=op_val)
        elif op_val == cv2.THRESH_BINARY_INV:
            run_benchmark(lib.threshold_binary_inv_float, a_3ch_f32, 0.5, 1.0, "f32_binary_inv", out_3ch_f32, op=op_val)
        elif op_val == cv2.THRESH_TRUNC:
            run_benchmark(lib.threshold_trunc_float, a_3ch_f32, 0.5, 1.0, "f32_trunc", out_3ch_f32, op=op_val)
        elif op_val == cv2.THRESH_TOZERO:
            run_benchmark(lib.threshold_tozero_float, a_3ch_f32, 0.5, 1.0, "f32_tozero", out_3ch_f32, op=op_val)
        elif op_val == cv2.THRESH_TOZERO_INV:
            run_benchmark(lib.threshold_tozero_inv_float, a_3ch_f32, 0.5, 1.0, "f32_tozero_inv", out_3ch_f32, op=op_val)

        print("-" * 85)