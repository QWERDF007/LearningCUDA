
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

file = Path(__file__)
# 将 src/common 加入到 sys.path，便于导入 helper
import sys
sys.path.append(str(file.parent.parent.parent / "common"))
from helper import compute_accuracy_info

torch.set_grad_enabled(False)

sources = [
    str(file.parent / "otsu.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

lib = load(
    name="blur_lib",
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
    thresh: torch.Tensor,
    maxval,
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
        perf_func(a, out, thresh, maxval)
    
    torch.cuda.synchronize()
    
    start = time.time()

    for i in range(iters):
        perf_func(a, out, thresh, maxval)
        
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000 
    mean_time = total_time / iters 

    a_np = a.cpu().numpy()
    
    # 根据是否使用百分比方法选择不同的期望计算方式
    t,  expected = cv2.threshold(a_np, 128, maxval, cv2.THRESH_OTSU)
    
    out_np = out.cpu().numpy()
    
    # 使用抽离的精度检测函数
    mismatch_info = compute_accuracy_info(out_np, expected)

    out_info = f"out_{tag} (cv:{t} | cuda:{thresh.cpu().numpy().item()})" 
    print(f"{out_info:>40}: {mismatch_info}, iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
    
    return out, mean_time, tag

Hs = [1024, 2048, 4096]
Ws = [1024, 2048, 4096, 46000]
Sizes = [(H, W) for H in Hs for W in Ws]

for H, W in Sizes:
    print("-" * 85)
    print(" " * 40 + f"H={H}, W={W}, ch=1")

    rand_val = torch.randint(0, 256, (1,), dtype=torch.uint8).item()
    a = torch.randint(0, 256, (H, W), dtype=torch.int16)  # 先用int16防止溢出
    a = (a - rand_val).clamp(0, 255).to(torch.uint8).cuda().contiguous()
    thresh = torch.zeros(1, dtype=torch.int32).cuda().contiguous()
    out = torch.zeros_like(a).cuda().contiguous()

    run_benchmark(lib.threshold_otsu_u8, a, thresh, 255, 'u8_otsu', out)
    run_benchmark(lib.threshold_otsu_u8_warp, a, thresh, 255, 'u8_otsu_warp', out)
    print("-" * 85)