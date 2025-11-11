
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
    str(file.parent / "sep_filter2d.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

lib = load(
    name="sep_filter2d_lib",
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
    return kx, ky

def get_laplacian_kernel(ksize):
    kx, ky = cv2.getDerivKernels(2, 0, ksize, normalize=False)
    return kx, ky


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    tmp: torch.Tensor,
    tmp1: torch.Tensor,
    tmp2: torch.Tensor,
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
    ksize: int,
    ksw: int,
    ksh: int,
    tag: str,
    out: Optional[torch.Tensor] = None,
    dx=0,
    dy=0,
    is_sobel=False,
    warmup: int = 20,
    iters: int = 1000,
    show_all: bool = False,
):
    """
    性能基准测试函数
    用于测量CUDA核函数的执行时间性能
    """
    kernel_x_tensor = torch.from_numpy(kernel_x.astype(np.int8)).cuda().contiguous()
    kernel_y_tensor = torch.from_numpy(kernel_y.astype(np.int8)).cuda().contiguous()
    # warmup
    if tmp2 is not None:
        for i in range(warmup):
            perf_func(a, out, tmp, tmp1, tmp2, kernel_x_tensor, kernel_y_tensor, ksw, ksh)
    else:
        for i in range(warmup):
            perf_func(a, out, tmp, kernel_x_tensor, kernel_y_tensor, ksw, ksh)
    
    torch.cuda.synchronize()
    
    start = time.time()

    if tmp2 is not None:
        for i in range(iters):
            perf_func(a, out, tmp, tmp1, tmp2, kernel_x_tensor, kernel_y_tensor, ksw, ksh)
    else:
        for i in range(iters):
            perf_func(a, out, tmp, kernel_x_tensor, kernel_y_tensor, ksw, ksh)
        
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000 
    mean_time = total_time / iters 

    a_np = a.cpu().numpy()
    out_np = out.cpu().numpy()
    
    # expected = cv2.sepFilter2D(a_np, cv2.CV_16SC1, kernelX=kernel_x, kernelY=kernel_y, delta=0)

    # cv2.filter2D(a_np, cv2.CV_16SC1, kernel)
    if is_sobel:
        expected = cv2.Sobel(a_np, cv2.CV_16SC1, dx, dy, ksize=ksize, scale=1, delta=0)
    else:
        expected = cv2.Laplacian(a_np, cv2.CV_16SC1, ksize=ksize, scale=1, delta=0)
    
    mismatch_info = compute_accuracy_info(out_np.astype(expected.dtype), expected)

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
    tmp = torch.zeros((H,W), dtype=torch.int16).cuda().contiguous()
    out = torch.zeros((H,W), dtype=torch.int16).cuda().contiguous()
    
    if K == 1:
        kx, ky = get_sobel_kernel(K, 1, 0, False)
        run_benchmark(lib.sep_filter2D_uint8_t_int16_t_int8_t, a, tmp, None, None, kx, ky, K, 3, 1, 'sobel', out, dx=1, dy=0, is_sobel=True)
        kx, ky = get_sobel_kernel(K, 0, 1, False)
        run_benchmark(lib.sep_filter2D_uint8_t_int16_t_int8_t, a, tmp, None, None, kx, ky, K, 1, 3, 'sobel', out, dx=0, dy=1, is_sobel=True)
        kx, ky = get_sobel_kernel(K, 1, 1, False)
        run_benchmark(lib.sep_filter2D_uint8_t_int16_t_int8_t, a, tmp, None, None, kx, ky, K, 3, 3, 'sobel', out, dx=1, dy=1, is_sobel=True)
    else:
        kx, ky = get_sobel_kernel(K, 1, 0, False)
        run_benchmark(lib.sep_filter2D_uint8_t_int16_t_int8_t, a, tmp, None, None, kx, ky, K, K, K, 'sobel', out, dx=1, dy=0, is_sobel=True)
        kx, ky = get_sobel_kernel(K, 0, 1, False)
        run_benchmark(lib.sep_filter2D_uint8_t_int16_t_int8_t, a, tmp, None, None, kx, ky, K, K, K, 'sobel', out, dx=0, dy=1, is_sobel=True)
        kx, ky = get_sobel_kernel(K, 1, 1, False)
        run_benchmark(lib.sep_filter2D_uint8_t_int16_t_int8_t, a, tmp, None, None, kx, ky, K, K, K, 'sobel', out, dx=1, dy=1, is_sobel=True)

    if K >= 5:
        print("-" * 85)
        tmp0 = torch.zeros((H,W), dtype=torch.int16).cuda().contiguous()
        tmp1 = torch.zeros((H,W), dtype=torch.int16).cuda().contiguous()
        tmp2 = torch.zeros((H,W), dtype=torch.int16).cuda().contiguous()
        out = torch.zeros((H,W), dtype=torch.int16).cuda().contiguous()
        kx, ky = get_laplacian_kernel(K)
        run_benchmark(lib.laplacian_uint8_t_int16_t_int16_t_int8_t, a, tmp0, tmp1, tmp2, kx, ky, K, K, K, 'laplacian', out, dx=0, dy=0)

        tmp0 = torch.zeros((H,W), dtype=torch.int32).cuda().contiguous()
        tmp1 = torch.zeros((H,W), dtype=torch.int32).cuda().contiguous()
        tmp2 = torch.zeros((H,W), dtype=torch.int32).cuda().contiguous()
        out = torch.zeros((H,W), dtype=torch.int16).cuda().contiguous()
        run_benchmark(lib.laplacian_uint8_t_int32_t_int16_t_int8_t, a, tmp0, tmp1, tmp2, kx, ky, K, K, K, 'laplacian', out, dx=0, dy=0)

        tmp0 = torch.zeros((H,W), dtype=torch.float32).cuda().contiguous()
        tmp1 = torch.zeros((H,W), dtype=torch.float32).cuda().contiguous()
        tmp2 = torch.zeros((H,W), dtype=torch.float32).cuda().contiguous()
        out = torch.zeros((H,W), dtype=torch.int16).cuda().contiguous()
        run_benchmark(lib.laplacian_uint8_t_float_int16_t_int8_t, a, tmp0, tmp1, tmp2, kx, ky, K, K, K, 'laplacian', out, dx=0, dy=0)