
import time
import re
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

def get_gaussian_sigma(kernel_size):
    """
    计算sigma值，模拟OpenCV的默认行为
    与CUDA代码中的get_gaussian_sigma函数保持一致
    """
    return 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

def compute_gaussian_kernel_opencv(kernel_size, sigma=None):
    """
    使用OpenCV计算高斯核并转换为PyTorch Tensor
    
    Args:
        kernel_size: 核大小（奇数）或者元组 (ksize_x, ksize_y)
        sigma: 标准差，可以是单个值、元组 (sigma_x, sigma_y) 或 None
    
    Returns:
        torch.Tensor: 高斯核权重，形状为 (kernel_size_y, kernel_size_x)
    """
    # 处理kernel_size参数
    if isinstance(kernel_size, (tuple, list)):
        ksize_x, ksize_y = kernel_size
    else:
        ksize_x = ksize_y = kernel_size
    
    # 处理sigma参数
    if sigma is None:
        sigma_x = get_gaussian_sigma(ksize_x)
        sigma_y = get_gaussian_sigma(ksize_y)
    elif isinstance(sigma, (tuple, list)):
        sigma_x, sigma_y = sigma
    else:
        sigma_x = sigma_y = sigma
    
    # 使用OpenCV的getGaussianKernel函数计算一维高斯核
    kernel_1d_x = cv2.getGaussianKernel(ksize_x, sigma_x)
    kernel_1d_y = cv2.getGaussianKernel(ksize_y, sigma_y)
    
    # 计算2D高斯核：外积
    kernel_2d = np.outer(kernel_1d_y, kernel_1d_x)
    
    # 转换为PyTorch Tensor并移动到GPU
    kernel_tensor = torch.from_numpy(kernel_2d.astype(np.float32)).cuda().contiguous()
    
    return kernel_tensor

file = Path(__file__)

sources = [
    str(file.parent / "gaussian_blur.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

lib = load(
    name="gaussian_blur_lib",
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


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    weights: torch.Tensor,
    kernel_size: int,
    sigma: tuple,
    tag: str,
    out: Optional[torch.Tensor] = None,
    tmp: Optional[torch.Tensor] = None,
    warmup: int = 20,
    iters: int = 1000,
    show_all: bool = False,
):
    """
    性能基准测试函数
    用于测量CUDA核函数的执行时间性能
    """
    ksz_h, ksz_w = kernel_size
    sigma_x, sigma_y = sigma
    # warmup
    for i in range(warmup):
        _ = perf_func(a, out, weights, ksz_h, ksz_w, sigma_x, sigma_y)
    
    torch.cuda.synchronize()
    
    start = time.time()
    
    
    for i in range(iters):
        perf_func(a, out, weights, ksz_h, ksz_w, sigma_x, sigma_y)
    
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000 
    mean_time = total_time / iters 
    
    if 'cv' in tag:
        expected = cv2.GaussianBlur(a, (ksz_w, ksz_h), sigmaX=sigma_x, sigmaY=sigma_y)
        out_np = out
    else:
        expected = cv2.GaussianBlur(a.cpu().numpy(), (ksz_w, ksz_h), sigmaX=sigma_x, sigmaY=sigma_y)
        out_np = out.cpu().numpy()
    
    # 精度检测逻辑 - 参照img_resize.py
    decimal = 8
    mismatch_info = ''
    ok = False
    
    # 存储特定精度的不匹配百分比
    mismatch_1e1 = '0%'  # 1e-1精度的不匹配百分比
    mismatch_1e3 = '0%'  # 1e-3精度的不匹配百分比
    mismatch_1e6 = '0%'  # 1e-6精度的不匹配百分比
    passed_decimal = None  # 通过测试的前一个精度
    max_abs_diff_all = 0  # 所有未通过测试中的最大绝对差异
    
    for i in range(12):
        try:
            np.testing.assert_array_almost_equal(out_np, expected, decimal)
            ok = True
            if passed_decimal is None:
                passed_decimal = decimal
            break
        except AssertionError as e:
            msg = str(e)
            
            # 提取不匹配百分比
            match = re.search(r"Mismatched elements:\s*(\d+)\s*/\s*(\d+)\s*\(([\d\.]+%)\)", msg)
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

Hs = [1024, 2048]
Ws = [1024, 2048]
Ks = [3, 5, 7, 9, 15]
SigmaX = [0.1, 0.3, 1, 2]
SigmaY = [0.1, 0.3, 1, 2]

# Hs = [4096]
# Ws = [4096]
# Ks = [7]

Sizes = [(H, W, (ksz, ksz), (sigmax, sigmay)) for H in Hs for W in Ws for ksz in Ks for sigmax in SigmaX for sigmay in SigmaY]

# 存储结果的数据结构
# results[tag][ksz] = {'mean_time': [], 'total_time': [], 'sizes': []}
results = defaultdict(lambda: defaultdict(lambda: {'mean_time': [], 'total_time': [], 'sizes': []}))

for H, W, ksz, sigma in Sizes:
    print("-" * 85)
    print(" " * 40 + f"H={H}, W={W}, ksz={ksz}, sigma={sigma}, ch=1")

    # 计算高斯核权重
    gaussian_weights = compute_gaussian_kernel_opencv(ksz, sigma)

    a = torch.randn((H, W), dtype=torch.float32).cuda().contiguous()
    a_np = a.cpu().numpy()
    out = torch.zeros((H, W), dtype=torch.float32).cuda().contiguous()

    run_benchmark(lib.gaussian_blur_float_float, a, gaussian_weights, ksz, sigma, "f32_gaussian_blur_float", out)
    run_benchmark(lib.gaussian_blur_float_double, a, gaussian_weights, ksz, sigma, "f32_gaussian_blur_double", out)

    print(" " * 40 + f"H={H}, W={W}, ksz={ksz}, sigma={sigma}, ch=3")

    a = torch.randn((H, W, 3), dtype=torch.float32).cuda().contiguous()
    a_np = a.cpu().numpy()
    out = torch.zeros((H, W, 3), dtype=torch.float32).cuda().contiguous()

    run_benchmark(lib.gaussian_blur_float_float, a, gaussian_weights, ksz, sigma, "f32_gaussian_blur_float", out)
    run_benchmark(lib.gaussian_blur_float_double, a, gaussian_weights, ksz, sigma, "f32_gaussian_blur_double", out)
