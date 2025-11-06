
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
    str(file.parent / "elementwise_sub.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

lib = load(
    name="elementwise_sub_lib",
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

    if out is not None:
        for i in range(warmup):
            perf_func(a, b, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, b)
    
    torch.cuda.synchronize()
    
    start = time.time()

    if out is not None:
        for i in range(iters):
            perf_func(a, b, out)
    else:
        for i in range(iters):
            out = perf_func(a, b)
        
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000 
    mean_time = total_time / iters 

    a_np = a.cpu().numpy()
    b_np = b.cpu().numpy()
    
    # 根据是否使用百分比方法选择不同的期望计算方式
    expected = cv2.subtract(a_np, b_np)
    # expected = a_np - b_np
    
    out_np = out.cpu().numpy()
    
    mismatch_info = compute_accuracy_info(out_np, expected)

    out_info = tag 
    
    print(f"{out_info:>25}: {mismatch_info}, iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
    
    return out, mean_time, tag

Hs = [1024, 2048, 4096,8192]
Ws = [1024, 2048, 4096,8192]
Sizes = [(H, W) for H in Hs for W in Ws]

for H, W in Sizes:
    print("-" * 85)
    print(" " * 40 + f"H={H}, W={W}")

    # 创建uint8类型的测试张量
    a_u8 = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    b_u8 = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    c_u8 = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()

    run_benchmark(partial(torch.sub, out=c_u8), a_u8, b_u8, "torch_sub_u8")

    c_u8 = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.elementwise_sub_uint8_t, a_u8, b_u8, "sub_u8", c_u8)

    c_u8 = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.elementwise_sub_2D_uint8_t, a_u8, b_u8, "sub_2D_u8", c_u8)

    c_u8 = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.elementwise_sub_32bit_uint8_t, a_u8, b_u8, "32bit_sub_u8", c_u8)

    c_u8 = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.elementwise_sub_128bit_uint8_t, a_u8, b_u8, "128bit_sub_u8", c_u8)

    c_u8 = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.elementwise_sub_32bit_2D_uint8_t, a_u8, b_u8, "32bit_sub_2D_u8", c_u8)

    c_u8 = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.elementwise_sub_128bit_2D_uint8_t, a_u8, b_u8, "128bit_sub_2D_u8", c_u8)
    
    print("-" * 85)

    # 创建uint8类型的测试张量
    a_f32 = torch.randn((H, W), dtype=torch.float32).cuda().contiguous()
    b_f32 = torch.randn((H, W), dtype=torch.float32).cuda().contiguous()
    c_f32 = torch.randn((H, W), dtype=torch.float32).cuda().contiguous()

    run_benchmark(partial(torch.sub, out=c_f32), a_f32, b_f32, "torch_sub_f32")
    run_benchmark(lib.elementwise_sub_float, a_f32, b_f32, "sub_f32", c_f32)
    
    c_f32 = torch.randn((H, W), dtype=torch.float32).cuda().contiguous()
    run_benchmark(lib.elementwise_sub_128bit_float, a_f32, b_f32, "128bit_sub_f32", c_f32)
    
    c_f32 = torch.randn((H, W), dtype=torch.float32).cuda().contiguous()
    run_benchmark(lib.elementwise_sub_128bit_2D_float, a_f32, b_f32, "128bit_sub_2D_f32", c_f32)

    print("-" * 85)