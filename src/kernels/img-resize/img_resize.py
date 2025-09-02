
import time
import math
from pathlib import Path
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

import numpy as np
import cv2

torch.set_grad_enabled(False)

file = Path(__file__)

sources = [
    str(file.parent / "img_resize.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

lib = load(
    name="img_resize_lib",
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
    dH: int,
    dW: int,
    tag: str,
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
        _ = perf_func(a, dH, dW)
    
    torch.cuda.synchronize()
    
    start = time.time()
    
    for i in range(iters):
        out = perf_func(a, dH, dW)
    
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000 
    mean_time = total_time / iters 

    expected = cv2.resize(a.cpu().numpy(), (dW, dH), interpolation=cv2.INTER_LINEAR)
    out_np = out.cpu().numpy()
    decimal = 8
    for i in range(10):
        try:
            np.testing.assert_array_almost_equal(out_np, expected, decimal)
            break
        except:
            decimal -= 1
   

    out_info = f"out_{tag}" 
    
    print(f"{out_info:>18}: (1e-{decimal}), iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
    
    return out, mean_time

Hs = [312, 1024, 4096]
Ws = [485, 2048, 4096]
Ss = [0.3, 0.5, 1.6, 2.0]
Sizes = [(H, W, S) for H in Hs for W in Ws for S in Ss]

for H, W, S in Sizes:
    print("-" * 85)
    print(" " * 40 + f"H={H}, W={W}, S={S}")

    a = torch.randn((H, W), dtype=torch.float32).cuda().contiguous()
    dH = math.ceil(H * S)
    dW = math.ceil(W * S)
    
    run_benchmark(lib.img_resize_f32, a, dH, dW, "f32")

    print("-" * 85)
    