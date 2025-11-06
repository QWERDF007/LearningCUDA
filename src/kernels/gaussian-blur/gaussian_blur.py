
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

file = Path(__file__)
# 将 src/common 加入到 sys.path，便于导入 helper
import sys
sys.path.append(str(file.parent.parent.parent / "common"))
from helper import compute_accuracy_info

def get_gaussian_sigma(kernel_size):
    """
    计算sigma值，模拟OpenCV的默认行为
    与CUDA代码中的get_gaussian_sigma函数保持一致
    """
    return 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

def compute_gaussian_kernel_opencv(kernel_size, sigma=None, ktype=6, norm=False):
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
    kernel_1d_x = cv2.getGaussianKernel(ksize_x, sigma_x, ktype=ktype)
    kernel_1d_y = cv2.getGaussianKernel(ksize_y, sigma_y, ktype=ktype)

    # 计算2D高斯核：外积
    if not norm:
        kernel_2d = np.outer(kernel_1d_y, kernel_1d_x)
    else:
        kernel_1d_x_norm = np.round(kernel_1d_x / kernel_1d_x[0])
        kernel_1d_y_norm = np.round(kernel_1d_y / kernel_1d_y[0])
        
        kernel_2d_norm = np.outer(kernel_1d_x_norm, kernel_1d_y_norm)
        kernel_2d = kernel_2d_norm / np.sum(kernel_2d_norm)
    
    # 转换为PyTorch Tensor并移动到GPU
    kernel_tensor = torch.from_numpy(kernel_2d.astype(np.float32)).cuda().contiguous()
    
    return kernel_tensor, sigma_x, sigma_y


def compute_gaussian_kernel1d_opencv(kernel_size, sigma=None, ktype=6):
    """
    使用OpenCV计算一维高斯核（X/Y）并转换为PyTorch Tensor（GPU）
    返回 (weights_x, weights_y, sigma_x, sigma_y)
    """
    if isinstance(kernel_size, (tuple, list)):
        ksize_x, ksize_y = kernel_size
    else:
        ksize_x = ksize_y = kernel_size

    if sigma is None:
        sigma_x = get_gaussian_sigma(ksize_x)
        sigma_y = get_gaussian_sigma(ksize_y)
    elif isinstance(sigma, (tuple, list)):
        sigma_x, sigma_y = sigma
    else:
        sigma_x = sigma_y = sigma

    kx = cv2.getGaussianKernel(ksize_x, sigma_x, ktype=ktype).astype(np.float32)
    ky = cv2.getGaussianKernel(ksize_y, sigma_y, ktype=ktype).astype(np.float32)

    wx = torch.from_numpy(kx.reshape(-1)).cuda().contiguous()
    wy = torch.from_numpy(ky.reshape(-1)).cuda().contiguous()

    return wx, wy, sigma_x, sigma_y




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
    weights_sep: Optional[tuple] = None,
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
        if 'sep' in tag:
            wx, wy = weights_sep
            _ = perf_func(a, tmp, out, wx, wy, ksz_h, ksz_w)
        else:
            _ = perf_func(a, out, weights, ksz_h, ksz_w, sigma_x, sigma_y)
    
    torch.cuda.synchronize()
    
    start = time.time()
    
    
    for i in range(iters):
        if 'sep' in tag:
            wx, wy = weights_sep
            perf_func(a, tmp, out, wx, wy, ksz_h, ksz_w)
        else:
            perf_func(a, out, weights, ksz_h, ksz_w, sigma_x, sigma_y)
    
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000 
    mean_time = total_time / iters 
    
    expected = cv2.GaussianBlur(a.cpu().numpy(), (ksz_w, ksz_h), sigmaX=sigma_x, sigmaY=sigma_y)
    out_np = out.cpu().numpy()
    
    mismatch_info = compute_accuracy_info(out_np, expected)

    out_info = f"out_{tag}" 
    
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

Sizes = [(H, W, (ksz, ksz), None) for H in Hs for W in Ws for ksz in Ks]
# Sizes = [(H, W, (ksz, ksz), (sigmax, sigmay)) for H in Hs for W in Ws for ksz in Ks for sigmax in SigmaX for sigmay in SigmaY]

# 存储结果的数据结构
# results[tag][ksz] = {'mean_time': [], 'total_time': [], 'sizes': []}
results = defaultdict(lambda: defaultdict(lambda: {'mean_time': [], 'total_time': [], 'sizes': []}))

for H, W, ksz, sigma in Sizes:
    print("-" * 85)

    # 计算高斯核权重（2D 和 1D）
    gaussian_weights, *sigma = compute_gaussian_kernel_opencv(ksz, sigma)
    wx, wy, *_ = compute_gaussian_kernel1d_opencv(ksz, sigma)
    
    print(" " * 40 + f"H={H}, W={W}, ksz={ksz}, sigma={sigma}, ch=1")

    a = torch.randn((H, W), dtype=torch.float32).cuda().contiguous()
    a_np = a.cpu().numpy()
    out = torch.zeros((H, W), dtype=torch.float32).cuda().contiguous()
    tmp = torch.zeros_like(out)

    run_benchmark(lib.gaussian_blur_float_float, a, gaussian_weights, ksz, sigma, "f32_gaussian_blur_float", out)
    run_benchmark(lib.gaussian_blur_float_double, a, gaussian_weights, ksz, sigma, "f32_gaussian_blur_double", out)
    run_benchmark(lib.gaussian_blur_sep_float_float, a, gaussian_weights, ksz, sigma, "f32_gaussian_blur_sep_float", out, tmp, (wx, wy))
    run_benchmark(lib.gaussian_blur_sep_float_double, a, gaussian_weights, ksz, sigma, "f32_gaussian_blur_sep_double", out, tmp, (wx, wy))

    a = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    a_np = a.cpu().numpy()
    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.float32).cuda().contiguous()

    run_benchmark(lib.gaussian_blur_uint8_t_float, a, gaussian_weights, ksz, sigma, "u8_gaussian_blur_float", out)
    run_benchmark(lib.gaussian_blur_uint8_t_double, a, gaussian_weights, ksz, sigma, "u8_gaussian_blur_double", out)
    run_benchmark(lib.gaussian_blur_sep_uint8_t_float, a, gaussian_weights, ksz, sigma, "u8_gaussian_blur_sep_float", out, tmp, (wx, wy))
    run_benchmark(lib.gaussian_blur_sep_uint8_t_double, a, gaussian_weights, ksz, sigma, "u8_gaussian_blur_sep_double", out, tmp, (wx, wy))

    print(" " * 40 + f"H={H}, W={W}, ksz={ksz}, sigma={sigma}, ch=3")

    a = torch.randn((H, W, 3), dtype=torch.float32).cuda().contiguous()
    a_np = a.cpu().numpy()
    out = torch.zeros((H, W, 3), dtype=torch.float32).cuda().contiguous()
    tmp = torch.zeros_like(out)

    run_benchmark(lib.gaussian_blur_float_float, a, gaussian_weights, ksz, sigma, "f32_gaussian_blur_float", out)
    run_benchmark(lib.gaussian_blur_float_double, a, gaussian_weights, ksz, sigma, "f32_gaussian_blur_double", out)
    run_benchmark(lib.gaussian_blur_sep_float_float, a, gaussian_weights, ksz, sigma, "f32_gaussian_blur_sep_float", out, tmp, (wx, wy))
    run_benchmark(lib.gaussian_blur_sep_float_double, a, gaussian_weights, ksz, sigma, "f32_gaussian_blur_sep_double", out, tmp, (wx, wy))

    a = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8).cuda().contiguous()
    a_np = a.cpu().numpy()
    out = torch.zeros((H, W, 3), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W, 3), dtype=torch.float32).cuda().contiguous()

    run_benchmark(lib.gaussian_blur_uint8_t_float, a, gaussian_weights, ksz, sigma, "u8_gaussian_blur_float", out)
    run_benchmark(lib.gaussian_blur_uint8_t_double, a, gaussian_weights, ksz, sigma, "u8_gaussian_blur_double", out)
    run_benchmark(lib.gaussian_blur_sep_uint8_t_float, a, gaussian_weights, ksz, sigma, "u8_gaussian_blur_sep_float", out, tmp, (wx, wy))
    run_benchmark(lib.gaussian_blur_sep_uint8_t_double, a, gaussian_weights, ksz, sigma, "u8_gaussian_blur_sep_double", out, tmp, (wx, wy))
