
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
    str(file.parent / "blur.cu")
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
    ksz: int,
    tag: str,
    out: Optional[torch.Tensor] = None,
    tmp: Optional[torch.Tensor] = None,
    warmup: int = 20,
    iters: int = 1000,
    show_all: bool = False,
):
    """
    性能基准测试函数
    用于测量CUDA核函数的执行时间性能
    """
    # warmup
    if 'cv' in tag:
        for i in range(warmup):
            _ = perf_func(a, (ksz, ksz))
    elif 'split' in tag:
        for i in range(warmup):
            _ = perf_func(a, ksz, out, tmp)
    elif 'sep' in tag:
        for i in range(warmup):
            _ = perf_func(a, ksz, out, tmp)
    else:
        for i in range(warmup):
            _ = perf_func(a, out, ksz)
    
    torch.cuda.synchronize()
    
    start = time.time()
    
    if 'cv' in tag:
        for i in range(iters):
            out = perf_func(a, (ksz, ksz))
    elif 'split' in tag:
        for i in range(iters):
            perf_func(a, ksz, out, tmp)
    elif 'sep' in tag:
        for i in range(iters):
            perf_func(a, ksz, out, tmp)
    else:
        for i in range(iters):
            perf_func(a, out, ksz)
    
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000 
    mean_time = total_time / iters 

    if 'cv' in tag:
        expected = cv2.blur(a, (ksz, ksz))
        out_np = out
    else:
        expected = cv2.blur(a.cpu().numpy(), (ksz, ksz))
        out_np = out.cpu().numpy()
    
    mismatch_info = compute_accuracy_info(out_np, expected)

    out_info = f"out_{tag}" 
    
    print(f"{out_info:>40}: {mismatch_info}, iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
    
    return out, mean_time, tag

Hs = [1024, 2048, 4096]
Ws = [1024, 2048, 4096]
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

    a = torch.randn((H, W), dtype=torch.float32).cuda().contiguous()
    a_np = a.cpu().numpy()
    out = torch.zeros((H, W), dtype=torch.float32).cuda().contiguous()
    tmp_f32 = torch.zeros((H, W, 3), dtype=torch.float32).cuda().contiguous()
    tmp_f64 = torch.zeros((H, W, 3), dtype=torch.float64).cuda().contiguous()

    run_benchmark(lib.blur_float_float, a, ksz, "f32_blur_float", out)
    run_benchmark(lib.blur_float_double, a, ksz, "f32_blur_double", out)
    run_benchmark(lib.blur_sep_float_float, a, ksz, "f32_blur_sep_float", out, tmp_f32)
    run_benchmark(lib.blur_sep_float_double, a, ksz, "f32_blur_sep_double", out, tmp_f64)

    a = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    a_np = a.cpu().numpy()
    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp_f32 = torch.zeros((H, W, 3), dtype=torch.float32).cuda().contiguous()
    tmp_f64 = torch.zeros((H, W, 3), dtype=torch.float64).cuda().contiguous()
    tmp_int32 = torch.zeros((H, W, 3), dtype=torch.int32).cuda().contiguous()

    run_benchmark(lib.blur_uint8_t_float, a, ksz, "u8_blur_float", out)
    run_benchmark(lib.blur_uint8_t_double, a, ksz, "u8_blur_double", out)
    run_benchmark(lib.blur_uint8_t_int32_t, a, ksz, "u8_blur_int32", out)
    run_benchmark(lib.blur_sep_uint8_t_float, a, ksz, "u8_blur_sep_float", out, tmp_f32)
    run_benchmark(lib.blur_sep_uint8_t_double, a, ksz, "u8_blur_sep_double", out, tmp_f64)
    run_benchmark(lib.blur_sep_uint8_t_int32_t, a, ksz, "u8_blur_sep_int32", out, tmp_int32)

    print(" " * 40 + f"H={H}, W={W}, ksz={ksz}, ch=3")

    a = torch.randn((H, W, 3), dtype=torch.float32).cuda().contiguous()
    a_np = a.cpu().numpy()
    out = torch.zeros((H, W, 3), dtype=torch.float32).cuda().contiguous()
    tmp_f32 = torch.zeros((H, W, 3), dtype=torch.float32).cuda().contiguous()
    tmp_f64 = torch.zeros((H, W, 3), dtype=torch.float64).cuda().contiguous()

    run_benchmark(lib.blur_float_float, a, ksz, "f32_blur_float", out)
    run_benchmark(lib.blur_float_double, a, ksz, "f32_blur_double", out)
    run_benchmark(lib.blur_sep_float_float, a, ksz, "f32_blur_sep_float", out, tmp_f32)
    run_benchmark(lib.blur_sep_float_double, a, ksz, "f32_blur_sep_double", out, tmp_f64)

    a = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8).cuda().contiguous()
    a_np = a.cpu().numpy()
    out = torch.zeros((H, W, 3), dtype=torch.uint8).cuda().contiguous()
    tmp_f32 = torch.zeros((H, W, 3), dtype=torch.float32).cuda().contiguous()
    tmp_f64 = torch.zeros((H, W, 3), dtype=torch.float64).cuda().contiguous()
    tmp_int32 = torch.zeros((H, W, 3), dtype=torch.int32).cuda().contiguous()

    run_benchmark(lib.blur_uint8_t_float, a, ksz, "u8_blur_float", out)
    run_benchmark(lib.blur_uint8_t_double, a, ksz, "u8_blur_double", out)
    run_benchmark(lib.blur_uint8_t_int32_t, a, ksz, "u8_blur_int32", out)
    run_benchmark(lib.blur_sep_uint8_t_float, a, ksz, "u8_blur_sep_float", out, tmp_f32)
    run_benchmark(lib.blur_sep_uint8_t_double, a, ksz, "u8_blur_sep_double", out, tmp_f64)
    run_benchmark(lib.blur_sep_uint8_t_int32_t, a, ksz, "u8_blur_sep_int32", out, tmp_int32)
