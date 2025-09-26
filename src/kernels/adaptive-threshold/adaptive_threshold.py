
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
    warmup: int = 20,
    iters: int = 1000,
    show_all: bool = False,
):
    """
    性能基准测试函数
    用于测量CUDA核函数的执行时间性能
    """
    # warmup
    if weights is None:
        for i in range(warmup):
            perf_func(a, mean, out, maxval, ksz, delta)
    else:
        for i in range(warmup):
            perf_func(a, mean, out, weights, maxval, ksz, delta)
    
    torch.cuda.synchronize()
    
    start = time.time()
    
    if weights is None:
        for i in range(iters):
            perf_func(a, mean, out, maxval, ksz, delta)
    else:
        for i in range(warmup):
            perf_func(a, mean, out, weights, maxval, ksz, delta)
        
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000 
    mean_time = total_time / iters 

    a_np = a.cpu().numpy()
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

# methods = ['ADAPTIVE_THRESH_MEAN_C', 'ADAPTIVE_THRESH_GAUSSIAN_C']
methods = ['ADAPTIVE_THRESH_GAUSSIAN_C']

Sizes = [(H, W, ksz) for H in Hs for W in Ws for ksz in Ks]
for method in methods:
    for H, W, ksz in Sizes:
        print("-" * 85)
        print(" " * 40 + f"H={H}, W={W}, ksz={ksz}, ch=1, method={method}")

        delta = random.uniform(0, 5)

        method_val = eval(f'cv2.{method}')
        
        weights = compute_gaussian_kernel_opencv(ksz)

        a = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
        mean = torch.zeros_like(a).cuda().contiguous()
        out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
        if method_val == cv2.ADAPTIVE_THRESH_MEAN_C:
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
        mean = torch.zeros_like(a).cuda().contiguous()
        out = torch.zeros((H, W, 3), dtype=torch.uint8).cuda().contiguous()
        if method_val == cv2.ADAPTIVE_THRESH_MEAN_C:
            run_benchmark(lib.adaptive_threshold_binary_uint8_t_float_mean, 
                a, mean, ksz, 255, "u8_adaptive_threshold", out, method=method_val, threshold_type=cv2.THRESH_BINARY, delta=delta)
            run_benchmark(lib.adaptive_threshold_binary_inv_uint8_t_float_mean, 
                    a, mean, ksz, 255, "u8_adaptive_threshold_inv", out, method=method_val, threshold_type=cv2.THRESH_BINARY_INV, delta=delta)
        elif method_val == cv2.ADAPTIVE_THRESH_GAUSSIAN_C:
            run_benchmark(lib.adaptive_threshold_binary_uint8_t_float_gaussian, 
                a, mean, ksz, 255, "u8_adaptive_threshold", out, weights, method=method_val, threshold_type=cv2.THRESH_BINARY, delta=delta)
            run_benchmark(lib.adaptive_threshold_binary_inv_uint8_t_float_gaussian, 
                    a, mean, ksz, 255, "u8_adaptive_threshold_inv", out, weights, method=method_val, threshold_type=cv2.THRESH_BINARY_INV, delta=delta)