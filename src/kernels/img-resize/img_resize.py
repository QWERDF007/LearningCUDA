
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
    baseline: float = None,
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
    ok = False
    for i in range(12):
        try:
            np.testing.assert_array_almost_equal(out_np, expected, decimal)
            ok = True
        except:
            decimal -= 1
        if ok:
            break
   

    out_info = f"out_{tag}" 
    sign = '-'
    if decimal <= 0:
        decimal = -decimal
        sign = ''
    speedup = mean_time / baseline if baseline is not None else 1.0
    print(f"{out_info:>30}: (1e{sign}{decimal}), iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms, {speedup:.3f}x")
    
    if show_all:
        print(out)
    
    return out, mean_time

Hs = [2, 45, 151, 224, 1024]
Ws = [2, 200, 320, 448, 2048]
Ss = [0.3, 0.5, 0.9, 1.7, 2.0, 2.4]
Sizes = [(H, W, S) for H in Hs for W in Ws for S in Ss]

for H, W, S in Sizes:
    print("-" * 85)
    dH = max(1, int(H * S))
    dW = max(1, int(W * S))
    print(" " * 40 + f"H={H}, W={W}, S={S}, dH={dH}, dW={dW}")
    print(" " * 55 + "ch=1")
    print("-" * 85)

    a = torch.randn((H, W), dtype=torch.float32).cuda().contiguous()
    
    _, baseline = run_benchmark(lib.img_resize_no_align_float_float, a, dH, dW, "f32_no_align_float")
    run_benchmark(lib.img_resize_no_align_float_double, a, dH, dW, "f32_no_align_double", baseline=baseline)
    run_benchmark(lib.img_resize_align_float_float, a, dH, dW, "f32_align_float", baseline=baseline)
    run_benchmark(lib.img_resize_2D_align_float_float, a, dH, dW, "f32_2D_align_float", baseline=baseline)
    run_benchmark(lib.img_resize_align_float_double, a, dH, dW, "f32_align_double", baseline=baseline)
    run_benchmark(lib.img_resize_2D_align_float_double, a, dH, dW, "f32_2D_align_double", baseline=baseline)

    print("-" * 85)
    print(" " * 55 + "ch=3")

    a = torch.randn((H, W, 3), dtype=torch.float32).cuda().contiguous()

    _, baseline = run_benchmark(lib.img_resize_no_align_float_float, a, dH, dW, "f32_no_align_float")
    run_benchmark(lib.img_resize_no_align_float_double, a, dH, dW, "f32_no_align_double", baseline=baseline)
    run_benchmark(lib.img_resize_align_float_float, a, dH, dW, "f32_align_float", baseline=baseline)
    run_benchmark(lib.img_resize_2D_align_float_float, a, dH, dW, "f32_2D_align_float", baseline=baseline)
    run_benchmark(lib.img_resize_align_float_double, a, dH, dW, "f32_align_double", baseline=baseline)
    run_benchmark(lib.img_resize_2D_align_float_double, a, dH, dW, "f32_2D_align_double", baseline=baseline)

    print("-" * 85)
    