
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
    str(file.parent / "sobel.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

lib = load(
    name="sobel_lib",
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

def get_kernel(ksize, dx, dy, normalize=False):
    # kx, ky 返回的是列向量 (kx: ksize x 1, ky: ksize x 1)
    kx, ky = cv2.getDerivKernels(dx, dy, ksize, normalize=normalize)
    # 生成 2D kernel（外积）
    sobel_kernel = np.outer(kx.reshape(-1), ky.reshape(-1))
    return sobel_kernel.T


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
    warmup: int = 20,
    iters: int = 1000,
    show_all: bool = False,
):
    """
    性能基准测试函数
    用于测量CUDA核函数的执行时间性能
    """
    # warmup
    for i in range(warmup):
        perf_func(a, out, kernel, ksh, ksw)
    
    torch.cuda.synchronize()
    
    start = time.time()

    for i in range(iters):
        perf_func(a, out, kernel, ksh, ksw)
        
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000 
    mean_time = total_time / iters 

    a_np = a.cpu().numpy()
    out_np = out.cpu().numpy()
    
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

    kernel_x = get_kernel(K, 1, 0)
    kernel_y = get_kernel(K, 0, 1)
    kernel_xy = get_kernel(K, 1, 1)

    # print(kernel_x)
    # print(kernel_y)
    # print(kernel_xy)

    kernel_x_tensor = torch.from_numpy(kernel_x.astype(np.int8)).cuda().contiguous()
    kernel_y_tensor = torch.from_numpy(kernel_y.astype(np.int8)).cuda().contiguous()
    kernel_xy_tensor = torch.from_numpy(kernel_xy.astype(np.int8)).cuda().contiguous()

    if K == 1:
        run_benchmark(lib.sobel_uint8_t_int16_t_int8_t, a, kernel_x_tensor, 1, 0, K, 1, 3,  'sobel x', out, iters=1000)
        run_benchmark(lib.sobel_uint8_t_int16_t_int8_t, a, kernel_y_tensor, 0, 1, K, 3, 1, 'sobel y', out, iters=1000)
        run_benchmark(lib.sobel_uint8_t_int16_t_int8_t, a, kernel_xy_tensor, 1, 1, K, 3, 3, 'sobel xy', out, iters=1000)
    else:
        run_benchmark(lib.sobel_uint8_t_int16_t_int8_t, a, kernel_x_tensor, 1, 0, K, K, K,  'sobel x', out, iters=1000)
        run_benchmark(lib.sobel_uint8_t_int16_t_int8_t, a, kernel_y_tensor, 0, 1, K, K, K, 'sobel y', out, iters=1000)
        run_benchmark(lib.sobel_uint8_t_int16_t_int8_t, a, kernel_xy_tensor, 1, 1, K, K, K, 'sobel xy', out, iters=1000)
   
    print("-" * 85)