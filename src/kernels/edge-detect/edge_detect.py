
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
    str(file.parent / "edge_detect.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

lib = load(
    name="edge_detect_lib",
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

def get_sobel_kernel(ksize, dx, dy, normalize=False):
    # kx, ky 返回的是列向量 (kx: ksize x 1, ky: ksize x 1)
    kx, ky = cv2.getDerivKernels(dx, dy, ksize, normalize=normalize)
    # 生成 2D kernel（外积）
    sobel_kernel = np.outer(kx.reshape(-1), ky.reshape(-1))
    return sobel_kernel.T



def get_scharr_kernel(dx, dy, normalize=False):
    """
    返回 3x3 Scharr 卷积核。
    dx, dy 必须满足 (dx,dy)=(1,0) 或 (0,1)。
    如果 normalize=True，会把核除以 32（常用的归一化因子）。
    """
    assert (dx == 1 and dy == 0) or (dx == 0 and dy == 1), "Scharr 仅支持 dx+dy=1 的情况"

    # 使用与 OpenCV 等价的可分解因子：
    # 对 dx=1（求 x 方向导数）：kx = [-1, 0, 1], ky = [3, 10, 3]
    # 对 dy=1（求 y 方向导数）：kx = [3, 10, 3], ky = [-1, 0, 1]
    if dx == 1:
        kx = np.array([-1, 0, 1], dtype=np.float32)
        ky = np.array([3, 10, 3], dtype=np.float32)
    else:
        kx = np.array([3, 10, 3], dtype=np.float32)
        ky = np.array([-1, 0, 1], dtype=np.float32)

    kernel = np.outer(ky, kx)  # ky[:,None] * kx[None,:]

    if normalize:
        kernel = kernel / 32.0

    return kernel.astype(np.float32)

def get_laplacian_kernel(ksize):
    if ksize == 1:
        return np.array([0, 1, 0, 1, -4, 1, 0, 1, 0], dtype=np.float32).reshape(3,3)
    elif ksize == 3:
        return np.array([2, 0, 2, 0, -8, 0, 2, 0, 2], dtype=np.float32).reshape(3,3)
    else:
        kx, ky = cv2.getDerivKernels(2, 0, ksize, normalize=False)
        sobel_kernel = np.outer(ky, kx)
        sobel_kernel2 = np.outer(kx, ky)
        return sobel_kernel.T, sobel_kernel2.T


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    kernel: torch.Tensor,
    dx: int,
    dy: int,
    ksize: int,
    ksh,
    ksw,
    tag: str,
    out: Optional[torch.Tensor] = None,
    tmp: torch.Tensor = None,
    kernely: torch.Tensor = None,
    warmup: int = 20,
    iters: int = 1000,
    is_scharr = False,
    is_laplacian = False,
    show_all: bool = False,
):
    """
    性能基准测试函数
    用于测量CUDA核函数的执行时间性能
    """
    # warmup
    if tmp is not None and kernely is not None:
        for i in range(warmup):
            perf_func(a, out, tmp, kernel, kernely, ksh, ksw)
    else:
        for i in range(warmup):
            perf_func(a, out, kernel, ksh, ksw)
    
    torch.cuda.synchronize()
    
    start = time.time()

    if tmp is not None and kernely is not None:
        for i in range(iters):
            perf_func(a, out, tmp, kernel, kernely, ksh, ksw)
    else:
        for i in range(iters):
            perf_func(a, out, kernel, ksh, ksw)
        
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000 
    mean_time = total_time / iters 

    a_np = a.cpu().numpy()
    out_np = out.cpu().numpy()
    
    if is_scharr:
        expected = cv2.Scharr(a_np, cv2.CV_16SC1, dx, dy, scale=1, delta=0)
    elif is_laplacian:
        expected = cv2.Laplacian(a_np, cv2.CV_16SC1, ksize=ksize, scale=1, delta=0, borderType=cv2.BORDER_REFLECT_101)
    else:
        expected = cv2.Sobel(a_np, cv2.CV_16SC1, dx, dy, ksize=ksize, scale=1, delta=0)
    expected = cv2.convertScaleAbs(expected)
    
    mismatch_info = compute_accuracy_info(out_np, expected)

    out_info = tag
    print(f"{out_info:>25}: {mismatch_info}, iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
    
    return out, mean_time, tag


Hs = [4096]
Ws = [4096]
Ks = [1, 3, 5, 7]
Sizes = [(H, W, K) for H in Hs for W in Ws for K in Ks]

for H, W, K in Sizes:
    print("-" * 85)
    print(" " * 40 + f"H={H}, W={W}, K={K}, ch=1")

    a = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    out = torch.zeros((H,W), dtype=torch.int16).cuda().contiguous()
    
    sobel_kernel_x = get_sobel_kernel(K, 1, 0)
    sobel_kernel_y = get_sobel_kernel(K, 0, 1)
    sobel_kernel_xy = get_sobel_kernel(K, 1, 1)

    sobel_kernel_x_tensor = torch.from_numpy(sobel_kernel_x.astype(np.int8)).cuda().contiguous()
    sobel_kernel_y_tensor = torch.from_numpy(sobel_kernel_y.astype(np.int8)).cuda().contiguous()
    sobel_kernel_xy_tensor = torch.from_numpy(sobel_kernel_xy.astype(np.int8)).cuda().contiguous()

    if K == 1:
        run_benchmark(lib.sobel_uint8_t_int16_t_int8_t, a, sobel_kernel_x_tensor, 1, 0, K, 1, 3,  'sobel x', out, iters=1000)
        run_benchmark(lib.sobel_uint8_t_int16_t_int8_t, a, sobel_kernel_y_tensor, 0, 1, K, 3, 1, 'sobel y', out, iters=1000)
        run_benchmark(lib.sobel_uint8_t_int16_t_int8_t, a, sobel_kernel_xy_tensor, 1, 1, K, 3, 3, 'sobel xy', out, iters=1000)
    else:
        run_benchmark(lib.sobel_uint8_t_int16_t_int8_t, a, sobel_kernel_x_tensor, 1, 0, K, K, K,  'sobel x', out, iters=1000)
        run_benchmark(lib.sobel_uint8_t_int16_t_int8_t, a, sobel_kernel_y_tensor, 0, 1, K, K, K, 'sobel y', out, iters=1000)
        run_benchmark(lib.sobel_uint8_t_int16_t_int8_t, a, sobel_kernel_xy_tensor, 1, 1, K, K, K, 'sobel xy', out, iters=1000)
   
    print("-" * 85)

    if K == 3:
        scharr_kernel_x = get_scharr_kernel(1, 0, False)
        scharr_kernel_y = get_scharr_kernel(0, 1, False)
        scharr_kernel_x_tensor = torch.from_numpy(scharr_kernel_x.astype(np.int8)).cuda().contiguous()
        scharr_kernel_y_tensor = torch.from_numpy(scharr_kernel_y.astype(np.int8)).cuda().contiguous()
        run_benchmark(lib.scharr_uint8_t_int16_t_int8_t, a, scharr_kernel_x_tensor, 1, 0, 3, 3, 3, 'scharr x', out, is_scharr=True, iters=1000)
        run_benchmark(lib.scharr_uint8_t_int16_t_int8_t, a, scharr_kernel_y_tensor, 0, 1, 3, 3, 3, 'scharr y', out, is_scharr=True, iters=1000)

    print("-" * 85)
    
    if K == 1:
        laplacian_kernel = get_laplacian_kernel(K)
        laplacian_kernel_tensor = torch.from_numpy(laplacian_kernel.astype(np.int8)).cuda().contiguous()
        run_benchmark(lib.laplacian_uint8_t_int16_t_int8_t, a, laplacian_kernel_tensor, 0, 0, 1, 3, 3, 'laplacian', out, 
            is_laplacian=True, iters=1000)
    elif K == 3:
        laplacian_kernel = get_laplacian_kernel(K)
        laplacian_kernel_tensor = torch.from_numpy(laplacian_kernel.astype(np.int8)).cuda().contiguous()
        run_benchmark(lib.laplacian_uint8_t_int16_t_int8_t, a, laplacian_kernel_tensor, 0, 0, K, K, K, 'laplacian', out, 
            is_laplacian=True, iters=1000)
    