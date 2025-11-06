
import time
import re
import random
from pathlib import Path
from functools import partial
from typing import Optional
from collections import defaultdict

import cv2

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

import numpy as np
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)

file = Path(__file__)
# 将 src/common 加入到 sys.path，便于导入 helper
import sys
sys.path.append(str(file.parent.parent.parent / "common"))
from helper import compute_accuracy_info

sources = [
    str(file.parent / "adaptive_threshold.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

lib = load(
    name="adaptive_threshold_lib",
    sources=sources,
    extra_include_paths=extra_include_paths,
    extra_cuda_cflags=[                      
        "-O3",                                 
        "-U__CUDA_NO_HALF_OPERATORS__",        
        "-U__CUDA_NO_HALF_CONVERSIONS__",      
        "-U__CUDA_NO_HALF2_OPERATORS__",       
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",  
        "--expt-relaxed-constexpr",            
        "--expt-extended-lambda",              
        "--use_fast_math",                     
    ],
    extra_cflags=["-std=c++17"],               
)

def compute_gaussian_kernel_opencv(kernel_size):
    # 使用OpenCV的getGaussianKernel函数计算一维高斯核
    kernel_1d_x = cv2.getGaussianKernel(kernel_size, 0)
    kernel_1d_y = cv2.getGaussianKernel(kernel_size, 0)

    kernel_2d = np.outer(kernel_1d_y, kernel_1d_x)
    
    # 转换为PyTorch Tensor并移动到GPU
    kernel_tensor = torch.from_numpy(kernel_2d.astype(np.float32)).cuda().contiguous()
    
    return kernel_tensor


def adaptive_threshold_percentage_python_wrapper(a_tensor, mean_tensor, out_tensor, maxval, ksz, percentage, threshold_type):
    """
    Python wrapper函数，模拟CUDA函数接口用于benchmark测试
    """
    a_np = a_tensor.cpu().numpy()
    result = adaptive_threshold_percentage(a_np, ksz, percentage, threshold_type)
    out_tensor.copy_(torch.from_numpy(result).cuda())
    return out_tensor


def adaptive_threshold_percentage(img, block_size, percentage, binary_method):
    """
    基于灰度占比的自适应二值化
    img: 输入图像 (numpy array)
    block_size: 块大小
    percentage: 百分比阈值
    binary_method: 二值化方法 (cv2.THRESH_BINARY 或 cv2.THRESH_BINARY_INV)
    """
    # 处理三通道图像
    if len(img.shape) == 3 and img.shape[2] == 3:
        # 分别对每个通道进行处理
        result_channels = []
        for c in range(3):
            channel_result = adaptive_threshold_percentage(img[:, :, c], block_size, percentage, binary_method)
            result_channels.append(channel_result)
        return np.stack(result_channels, axis=2)
    
    # 单通道处理
    if len(img.shape) == 3:
        img_gray = img[:, :, 0]  # 取第一个通道
    else:
        img_gray = img.copy()
    
    rows, cols = img_gray.shape
    output = np.zeros_like(img_gray)
    
    # 计算积分图
    integral = cv2.integral(img_gray)
    
    s2 = block_size // 2
    
    high = 0 if binary_method == cv2.THRESH_BINARY_INV else 255
    low = 255 if binary_method == cv2.THRESH_BINARY_INV else 0
    
    for i in range(rows):
        y1 = max(0, i - s2)
        y2 = min(rows - 1, i + s2)
        
        for j in range(cols):
            x1 = max(0, j - s2)
            x2 = min(cols - 1, j + s2)
            count = (x2 - x1 + 1) * (y2 - y1 + 1)
            # count = (x2 - x1) * (y2 - y1)
            # count = block_size * block_size
            
            # 使用积分图计算区域和
            # 注意：cv2.integral() 返回的积分图比原图大1
            region_sum = (integral[y2+1, x2+1] - integral[y1, x2+1] - integral[y2+1, x1] + integral[y1, x1])
            # region_sum = (integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1])
            
            # 应用阈值：current_pixel * count < region_sum * percentage
            output[i, j] = low if img_gray[i, j] * count < region_sum * percentage else high
    
    return output


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    mean: torch.Tensor,
    ksz: int,
    maxval,
    tag: str,
    out: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    method = cv2.ADAPTIVE_THRESH_MEAN_C, # cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    threshold_type = cv2.THRESH_BINARY, # cv2.THRESH_BINARY or cv2.THRESH_BINARY_INV
    delta = 0,
    percentage = None,  # 新增：百分比参数
    warmup: int = 20,
    iters: int = 1000,
    show_all: bool = False,
):
    """
    性能基准测试函数
    用于测量CUDA核函数的执行时间性能
    """
    # warmup
    if percentage is not None:
        for i in range(warmup):
            perf_func(a, mean, out, maxval, ksz, percentage)
    elif weights is None:
        for i in range(warmup):
            perf_func(a, mean, out, maxval, ksz, delta)
    else:
        for i in range(warmup):
            perf_func(a, mean, out, weights, maxval, ksz, delta)
    
    torch.cuda.synchronize()
    
    start = time.time()
    
    if percentage is not None:
        for i in range(iters):
            perf_func(a, mean, out, maxval, ksz, percentage)
    elif weights is None:
        for i in range(iters):
            perf_func(a, mean, out, maxval, ksz, delta)
    else:
        for i in range(iters):
            perf_func(a, mean, out, weights, maxval, ksz, delta)
        
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000 
    mean_time = total_time / iters 

    a_np = a.cpu().numpy()
    
    # 根据是否使用百分比方法选择不同的期望计算方式
    if percentage is not None:
        expected = adaptive_threshold_percentage(a_np, ksz, percentage, threshold_type)
    else:
        if len(a_np.shape) == 3 and a_np.shape[2] == 3:  # 三通道图像
            # 分别对每个通道进行自适应二值化
            expected_channels = []
            for c in range(3):
                channel_result = cv2.adaptiveThreshold(a_np[:, :, c], maxval, method, threshold_type, ksz, C=delta)
                expected_channels.append(channel_result)
            expected = np.stack(expected_channels, axis=2)
        else:  # 单通道图像
            expected = cv2.adaptiveThreshold(a_np, maxval, method, threshold_type, ksz, C=delta)
    
    out_np = out.cpu().numpy()

    mismatch_info = compute_accuracy_info(out_np, expected)

    out_info = f"out_{tag}" 
    print(f"{out_info:>40}: {mismatch_info}, iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
    
    return out, mean_time, tag

Hs = [1024]
Ws = [1024]
# Ks = [3, 5, 7, 9, 15]
Ks = [3, 9, 15]

# methods = ['ADAPTIVE_THRESH_MEAN_C', 'ADAPTIVE_THRESH_GAUSSIAN_C', 'PERCENTAGE']
methods = ['PERCENTAGE']

Sizes = [(H, W, ksz) for H in Hs for W in Ws for ksz in Ks]
for method in methods:
    for H, W, ksz in Sizes:
        print("-" * 85)
        print(" " * 40 + f"H={H}, W={W}, ksz={ksz}, ch=1, method={method}")

        delta = random.uniform(0, 5)
        percentage = random.uniform(0.1, 1)  # 百分比参数，通常在0.8-1.2之间

        if method == 'PERCENTAGE':
            method_val = None 
            weights = None
        else:
            method_val = eval(f'cv2.{method}')
            weights = compute_gaussian_kernel_opencv(ksz)

        a = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
        if method == 'PERCENTAGE':
            mean = torch.zeros_like(a, dtype=torch.uint32).cuda().contiguous()
        else:
            mean = torch.zeros_like(a, dtype=torch.uint8).cuda().contiguous()
        out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
        
        if method == 'PERCENTAGE':
            run_benchmark(lib.adaptive_threshold_binary_uint8_t_uint32_t_percentage, 
                a, mean, ksz, 255, "u8_adaptive_threshold_percentage", out, threshold_type=cv2.THRESH_BINARY, percentage=percentage)
            run_benchmark(lib.adaptive_threshold_binary_inv_uint8_t_uint32_t_percentage, 
                a, mean, ksz, 255, "u8_adaptive_threshold_percentage_inv", out, threshold_type=cv2.THRESH_BINARY_INV, percentage=percentage)
        elif method_val == cv2.ADAPTIVE_THRESH_MEAN_C:
            run_benchmark(lib.adaptive_threshold_binary_uint8_t_float_mean, 
                a, mean, ksz, 255, "u8_adaptive_threshold", out, method=method_val, threshold_type=cv2.THRESH_BINARY, delta=delta)
            run_benchmark(lib.adaptive_threshold_binary_inv_uint8_t_float_mean, 
                    a, mean, ksz, 255, "u8_adaptive_threshold_inv", out, method=method_val, threshold_type=cv2.THRESH_BINARY_INV, delta=delta)
        elif method_val == cv2.ADAPTIVE_THRESH_GAUSSIAN_C:
            run_benchmark(lib.adaptive_threshold_binary_uint8_t_float_gaussian, 
                a, mean, ksz, 255, "u8_adaptive_threshold", out, weights, method=method_val, threshold_type=cv2.THRESH_BINARY, delta=delta)
            run_benchmark(lib.adaptive_threshold_binary_inv_uint8_t_float_gaussian, 
                    a, mean, ksz, 255, "u8_adaptive_threshold_inv", out, weights, method=method_val, threshold_type=cv2.THRESH_BINARY_INV, delta=delta)
        
        print(" " * 40 + f"H={H}, W={W}, ksz={ksz}, ch=3, method={method}")

        a = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8).cuda().contiguous()
        if method == 'PERCENTAGE':
            mean = torch.zeros_like(a, dtype=torch.uint32).cuda().contiguous()
        else:
            mean = torch.zeros_like(a, dtype=torch.uint8).cuda().contiguous()
        out = torch.zeros((H, W, 3), dtype=torch.uint8).cuda().contiguous()
        
        if method == 'PERCENTAGE':
            run_benchmark(lib.adaptive_threshold_binary_uint8_t_uint32_t_percentage, 
                a, mean, ksz, 255, "u8_adaptive_threshold_percentage", out, threshold_type=cv2.THRESH_BINARY, percentage=percentage)
            run_benchmark(lib.adaptive_threshold_binary_inv_uint8_t_uint32_t_percentage, 
                a, mean, ksz, 255, "u8_adaptive_threshold_percentage_inv", out, threshold_type=cv2.THRESH_BINARY_INV, percentage=percentage)
        elif method_val == cv2.ADAPTIVE_THRESH_MEAN_C:
            run_benchmark(lib.adaptive_threshold_binary_uint8_t_float_mean, 
                a, mean, ksz, 255, "u8_adaptive_threshold", out, method=method_val, threshold_type=cv2.THRESH_BINARY, delta=delta)
            run_benchmark(lib.adaptive_threshold_binary_inv_uint8_t_float_mean, 
                a, mean, ksz, 255, "u8_adaptive_threshold_inv", out, method=method_val, threshold_type=cv2.THRESH_BINARY_INV, delta=delta)
        elif method_val == cv2.ADAPTIVE_THRESH_GAUSSIAN_C:
            run_benchmark(lib.adaptive_threshold_binary_uint8_t_float_gaussian, 
                a, mean, ksz, 255, "u8_adaptive_threshold", out, weights, method=method_val, threshold_type=cv2.THRESH_BINARY, delta=delta)
            run_benchmark(lib.adaptive_threshold_binary_inv_uint8_t_float_gaussian, 
                a, mean, ksz, 255, "u8_adaptive_threshold_inv", out, weights, method=method_val, threshold_type=cv2.THRESH_BINARY_INV, delta=delta)