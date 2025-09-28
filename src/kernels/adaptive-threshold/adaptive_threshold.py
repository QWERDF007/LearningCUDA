
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

sources = [
    str(file.parent / "adaptive_threshold.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

lib = load(
    name="blur_lib",
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
            
            # 使用积分图计算区域和
            # 注意：cv2.integral() 返回的积分图比原图大1
            region_sum = (integral[y2+1, x2+1] - integral[y1, x2+1] - 
                         integral[y2+1, x1] + integral[y1, x1])
            
            # 应用阈值：current_pixel * count < region_sum * percentage
            if img_gray[i, j] * count < region_sum * percentage:
                output[i, j] = low
            else:
                output[i, j] = high
    
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
    
    # 精度检测逻辑 - 参照img_resize.py
    decimal = 8
    mismatch_info = ''
    
    # 存储特定精度的不匹配百分比
    mismatch_1e1 = '0%'  # 1e-1精度的不匹配百分比
    mismatch_1e3 = '0%'  # 1e-3精度的不匹配百分比
    mismatch_1e6 = '0%'  # 1e-6精度的不匹配百分比
    passed_decimal = None  # 通过测试的前一个精度
    max_abs_diff_all = 0  # 所有未通过测试中的最大绝对差异

    # np.testing.assert_array_almost_equal(out_np, expected, decimal)

    for i in range(12):
        try:
            np.testing.assert_array_almost_equal(out_np, expected, decimal)
            if passed_decimal is None:
                passed_decimal = decimal
            break
        except AssertionError as e:
            msg = str(e)
            
            # 提取不匹配百分比
            match = re.search(r"Mismatched elements:\s*(\d+)\s*/\s*(\d+)\s*\(([\d\.eE+-]+%)\)", msg)
            percent_str = None
            if match:
                percent_str = match.group(3)
            
            # 提取最大绝对差异
            match = re.search(r"Max absolute difference:\s*([0-9.eE+-]+)", msg)
            if match:
                max_abs_diff = float(match.group(1))
                max_abs_diff_all = max(max_abs_diff_all, max_abs_diff)
            
            # 记录特定精度的不匹配百分比
            if decimal == 3:  # 1e-3
                mismatch_1e3 = percent_str if percent_str else "0%"
            elif decimal == 6:  # 1e-6
                mismatch_1e6 = percent_str if percent_str else "0%"
            elif decimal == 1: # 1e-1
                mismatch_1e1 = percent_str if percent_str else "0%"
            
            decimal -= 1
    
    # 如果在1e-3和1e-6之前就通过了测试，设置这两个精度的不匹配百分比为0
    if passed_decimal is not None:
        if passed_decimal > 3 and mismatch_1e3 is None:
            mismatch_1e3 = "0%"
        if passed_decimal > 6 and mismatch_1e6 is None:
            mismatch_1e6 = "0%"
        elif passed_decimal > 6 and mismatch_1e1 is None:
            mismatch_1e1 = "0%"
    
    # 构建mismatch_info字符串
    info_parts = []
    if passed_decimal is not None:
        sign = '-'
        if passed_decimal <= 0:
            passed_decimal = -passed_decimal
            sign = ''
        info_parts.append(f"passed: 1e{sign}{passed_decimal}")
    if mismatch_1e1 is not None:
        info_parts.append(f"1e-1: {mismatch_1e1}")
    if mismatch_1e3 is not None:
        info_parts.append(f"1e-3: {mismatch_1e3}")
    if mismatch_1e6 is not None:
        info_parts.append(f"1e-6: {mismatch_1e6}")
    if max_abs_diff_all:
        info_parts.append(f"max_diff: {max_abs_diff_all}")
    mismatch_info = ", ".join(info_parts)

    out_info = f"out_{tag}" 
    sign = '-'
    if decimal <= 0:
        decimal = -decimal
        sign = ''
    print(f"{out_info:>40}: {mismatch_info}, iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
    
    return out, mean_time, tag

Hs = [1024]
Ws = [1024]
# Ks = [3, 5, 7, 9, 15]
Ks = [9, 15]

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