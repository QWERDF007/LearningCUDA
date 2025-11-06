
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
    str(file.parent / "bitwise.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

lib = load(
    name="bitwise_lib",
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
    b: torch.Tensor,
    tag: str,
    op: str,
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
    if op == 'not':
        for i in range(warmup):
            perf_func(a, out)
    else:
        for i in range(warmup):
            perf_func(a, b, out)
    
    torch.cuda.synchronize()
    
    start = time.time()

    if op == 'not':
        for i in range(iters):
            perf_func(a, out)
    else:
        for i in range(iters):
            perf_func(a, b, out)
        
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000 
    mean_time = total_time / iters 

    a_np = a.cpu().numpy()
    b_np = b.cpu().numpy()
    out_np = out.cpu().numpy()
    
    if op == 'not':
        expected = eval(f'cv2.bitwise_{op}(a_np)')
    else:
        expected = eval(f'cv2.bitwise_{op}(a_np, b_np)')
    
    mismatch_info = compute_accuracy_info(out_np, expected)

    out_info = tag
    print(f"{out_info:>25}: {mismatch_info}, iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
    
    return out, mean_time, tag

Hs = [1773, 4096]
Ws = [1367, 4096]
Sizes = [(H, W) for H in Hs for W in Ws]

for H, W in Sizes:
    print("-" * 85)
    print(" " * 40 + f"H={H}, W={W}, ch=1")

    a = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    b = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    out = torch.zeros_like(a).cuda().contiguous()
    
    run_benchmark(lib.bitwise_and_uint8_t, a, b, 'bitwise_and', 'and', out, iters=1000)
    run_benchmark(lib.bitwise_and_u8x4_uint8_t, a, b, 'bitwise_and_u8x4', 'and', out, iters=1000)
    run_benchmark(lib.bitwise_and_vec_uint8_t, a, b, 'bitwise_and_vec', 'and', out, iters=1000)
    run_benchmark(lib.bitwise_or_uint8_t, a, b,  'bitwise_or', 'or', out, iters=1000)
    run_benchmark(lib.bitwise_xor_uint8_t, a, b, 'bitwise_xor', 'xor', out, iters=1000)
    run_benchmark(lib.bitwise_not_uint8_t, a, b, 'bitwise_not', 'not', out, iters=1000)

    print("-" * 85)
    print(" " * 40 + f"H={H}, W={W}, ch=3")

    a = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8).cuda().contiguous()
    b = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8).cuda().contiguous()
    out = torch.zeros_like(a).cuda().contiguous()

    run_benchmark(lib.bitwise_and_uint8_t, a, b, 'bitwise_and', 'and', out, iters=1000)
    run_benchmark(lib.bitwise_and_vec_uint8_t, a, b, 'bitwise_and_vec', 'and', out, iters=1000)
    run_benchmark(lib.bitwise_or_uint8_t, a, b,  'bitwise_or', 'or', out, iters=1000)
    run_benchmark(lib.bitwise_xor_uint8_t, a, b, 'bitwise_xor', 'xor', out, iters=1000)
    run_benchmark(lib.bitwise_not_uint8_t, a, b, 'bitwise_not', 'not', out, iters=1000)

    print("-" * 85)