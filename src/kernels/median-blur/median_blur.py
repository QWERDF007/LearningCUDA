
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

sources = [
    str(file.parent / "median_blur.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

lib = load(
    name="median_blur_lib",
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
    ksz: int,
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
        _ = perf_func(a, out, ksz)
    
    torch.cuda.synchronize()
    
    start = time.time()
    
    for i in range(iters):
        perf_func(a, out, ksz)
    
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000 
    mean_time = total_time / iters 

    expected = cv2.medianBlur(a.cpu().numpy(), ksz)
    out_np = out.cpu().numpy()
    
    mismatch_info = compute_accuracy_info(out_np, expected)

    out_info = f"out_{tag}" 
    
    print(f"{out_info:>40}: {mismatch_info}, iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
    
    return out, mean_time, tag

Hs = [1024]
Ws = [1024]
Ks = [3, 5, 7, 9, 15]

# Hs = [4096]
# Ws = [4096]
# Ks = [7]

Sizes = [(H, W, ksz) for H in Hs for W in Ws for ksz in Ks]

# 存储结果的数据结构
# results[tag][ksz] = {'mean_time': [], 'total_time': [], 'sizes': []}
results = defaultdict(lambda: defaultdict(lambda: {'mean_time': [], 'total_time': [], 'sizes': []}))

for H, W, ksz in Sizes:
    print("-" * 85)
    print(" " * 40 + f"H={H}, W={W}, ksz={ksz}, ch=1")

    if ksz < 7:

        a = torch.randn((H, W), dtype=torch.float32).cuda().contiguous()
        a_np = a.cpu().numpy()
        out = torch.zeros((H, W), dtype=torch.float32).cuda().contiguous()

        run_benchmark(lib.median_blur_float_float, a, ksz, "f32_median_blur_float", out)

    a = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    a_np = a.cpu().numpy()
    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()

    run_benchmark(lib.median_blur_uint8_t_uint8_t, a, ksz, "u8_median_blur_u8", out)
    run_benchmark(lib.median_filter_gray_hist, a, ksz, "u8_median_blur_hist", out)

    print(" " * 40 + f"H={H}, W={W}, ksz={ksz}, ch=3")

    if ksz < 7:

        a = torch.randn((H, W, 3), dtype=torch.float32).cuda().contiguous()
        a_np = a.cpu().numpy()
        out = torch.zeros((H, W, 3), dtype=torch.float32).cuda().contiguous()

        run_benchmark(lib.median_blur_float_float, a, ksz, "f32_median_blur_float", out)

    a = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8).cuda().contiguous()
    a_np = a.cpu().numpy()
    out = torch.zeros((H, W, 3), dtype=torch.uint8).cuda().contiguous()

    run_benchmark(lib.median_blur_uint8_t_uint8_t, a, ksz, "u8_median_blur_u8", out)
