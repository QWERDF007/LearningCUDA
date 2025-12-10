
import time
import math
import re
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
# 将 src/common 加入到 sys.path，便于导入 helper
import sys
sys.path.append(str(file.parent.parent.parent / "common"))
from helper import compute_accuracy_info

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
    interpolation: int,
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
    expected = cv2.resize(a.cpu().numpy(), (dW, dH), interpolation=interpolation)
    out_np = out.cpu().numpy()

    mismatch_info = compute_accuracy_info(out_np, expected)

    out_info = f"out_{tag}" 
    
    print(f"{out_info:>40}: {mismatch_info}, iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
    
    return out, mean_time, tag

# Hs = [2, 45, 151, 224, 1024]
# Ws = [2, 200, 320, 448, 2048]
Hs = [1024]
Ws = [1024]
Ss = [0.3, 0.5, 0.9, 1.7, 2.0, 2.4]
# Ss = [0.1]
Sizes = [(H, W, S) for H in Hs for W in Ws for S in Ss]
INTER_TYPES =[
    'INTER_LINEAR', 'INTER_NEAREST', 'INTER_CUBIC', 'INTER_LANCZOS4', 'INTER_AREA', 
    'INTER_NEAREST_EXACT', 'INTER_LINEAR_EXACT', 'ALL'
]
INTER_TYPE = 'INTER_LINEAR_EXACT'


for H, W, S in Sizes:
    print("-" * 85)
    print(" " * 40 + f"H={H}, W={W}, S={S}")
    print("-" * 85)
    dH = max(1, int(H * S))
    dW = max(1, int(W * S))
    print(" " * 40 + f"dH={dH}, dW={dW}, ch=1")
    print("-" * 85)

    a = torch.randn((H, W), dtype=torch.float32).cuda().contiguous()
    
    if INTER_TYPE == 'INTER_LINEAR' or INTER_TYPE == 'ALL':
        run_benchmark(lib.resize_bilinear_float_float, a, dH, dW, "f32_bilinear_float", interpolation=cv2.INTER_LINEAR)
        run_benchmark(lib.resize_bilinear_float_double, a, dH, dW, "f32_bilinear_double", interpolation=cv2.INTER_LINEAR)
        run_benchmark(lib.resize_bilinear_2D_float_float, a, dH, dW, "f32_bilinear_2D_float", interpolation=cv2.INTER_LINEAR)
        run_benchmark(lib.resize_bilinear_2D_float_double, a, dH, dW, "f32_bilinear_2D_double", interpolation=cv2.INTER_LINEAR)
        # run_benchmark(lib.resize_bilinear_shared_2D_float_float, a, dH, dW, "f32_bilinear_shared_float", interpolation=cv2.INTER_LINEAR)
        # run_benchmark(lib.resize_bilinear_shared_2D_float_double, a, dH, dW, "f32_bilinear_shared_double", interpolation=cv2.INTER_LINEAR)

    if INTER_TYPE == 'INTER_NEAREST' or INTER_TYPE == 'ALL':
        run_benchmark(lib.resize_nearest_float_float, a, dH, dW, "f32_nearest_float", interpolation=cv2.INTER_NEAREST)
        run_benchmark(lib.resize_nearest_float_double, a, dH, dW, "f32_nearest_double", interpolation=cv2.INTER_NEAREST)

    if INTER_TYPE == 'INTER_CUBIC' or INTER_TYPE == 'ALL':
        run_benchmark(lib.resize_bicubic_float_float, a, dH, dW, "f32_bicubic_float", interpolation=cv2.INTER_CUBIC)
        run_benchmark(lib.resize_bicubic_float_double, a, dH, dW, "f32_bicubic_double", interpolation=cv2.INTER_CUBIC)

    if INTER_TYPE == 'INTER_LANCZOS4' or INTER_TYPE == 'ALL':
        run_benchmark(lib.resize_lanczos_float_float, a, dH, dW, "f32_lanczos_float", interpolation=cv2.INTER_LANCZOS4)
        run_benchmark(lib.resize_lanczos_float_double, a, dH, dW, "f32_lanczos_double", interpolation=cv2.INTER_LANCZOS4)

    if (INTER_TYPE == 'INTER_AREA' or INTER_TYPE == 'ALL'):
        if S < 1:
            run_benchmark(lib.resize_area_float_float, a, dH, dW, "f32_area_float", interpolation=cv2.INTER_AREA)
            run_benchmark(lib.resize_area_float_double, a, dH, dW, "f32_area_double", interpolation=cv2.INTER_AREA)
        else:
            run_benchmark(lib.resize_area_bilinear_float_float, a, dH, dW, "f32_area_float", interpolation=cv2.INTER_AREA)
            run_benchmark(lib.resize_area_bilinear_float_double, a, dH, dW, "f32_area_double", interpolation=cv2.INTER_AREA)
    
    a = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()

    if INTER_TYPE == 'INTER_LINEAR' or INTER_TYPE == 'ALL':
        run_benchmark(lib.resize_bilinear_uint8_t_float, a, dH, dW, "u8_bilinear_float", interpolation=cv2.INTER_LINEAR)
        run_benchmark(lib.resize_bilinear_uint8_t_double, a, dH, dW, "u8_bilinear_double", interpolation=cv2.INTER_LINEAR)
        run_benchmark(lib.u8_resize_bilinear_uint8_t_float, a, dH, dW, "u8x_bilinear_float", interpolation=cv2.INTER_LINEAR)

    if INTER_TYPE == 'INTER_NEAREST' or INTER_TYPE == 'ALL':
        run_benchmark(lib.resize_nearest_uint8_t_float, a, dH, dW, "u8_nearest_float", interpolation=cv2.INTER_NEAREST)
        run_benchmark(lib.resize_nearest_uint8_t_double, a, dH, dW, "u8_nearest_double", interpolation=cv2.INTER_NEAREST)

    if INTER_TYPE == 'INTER_CUBIC' or INTER_TYPE == 'ALL':
        run_benchmark(lib.resize_bicubic_uint8_t_float, a, dH, dW, "u8_bicubic_float", interpolation=cv2.INTER_CUBIC)
        run_benchmark(lib.resize_bicubic_uint8_t_double, a, dH, dW, "u8_bicubic_double", interpolation=cv2.INTER_CUBIC)
        run_benchmark(lib.u8_resize_bicubic_uint8_t_float, a, dH, dW, "u8x_bicubic_float", interpolation=cv2.INTER_CUBIC)

    if INTER_TYPE == 'INTER_LANCZOS4' or INTER_TYPE == 'ALL':
        run_benchmark(lib.resize_lanczos_uint8_t_float, a, dH, dW, "u8_lanczos_float", interpolation=cv2.INTER_LANCZOS4)
        run_benchmark(lib.resize_lanczos_uint8_t_double, a, dH, dW, "u8_lanczos_double", interpolation=cv2.INTER_LANCZOS4)
        run_benchmark(lib.u8_resize_lanczos_uint8_t_float, a, dH, dW, "u8x_lanczos_float", interpolation=cv2.INTER_LANCZOS4)

    if (INTER_TYPE == 'INTER_AREA' or INTER_TYPE == 'ALL'):
        if S < 1:
            run_benchmark(lib.resize_area_uint8_t_float, a, dH, dW, "u8_area_float", interpolation=cv2.INTER_AREA)
            run_benchmark(lib.resize_area_uint8_t_double, a, dH, dW, "u8_area_double", interpolation=cv2.INTER_AREA)
        else:
            run_benchmark(lib.resize_area_bilinear_uint8_t_float, a, dH, dW, "u8_area_float", interpolation=cv2.INTER_AREA)
            run_benchmark(lib.resize_area_bilinear_uint8_t_double, a, dH, dW, "u8_area_double", interpolation=cv2.INTER_AREA)
            run_benchmark(lib.u8_resize_area_bilinear_uint8_t_float, a, dH, dW, "u8x_area_float", interpolation=cv2.INTER_AREA)

    if INTER_TYPE == 'INTER_NEAREST_EXACT' or INTER_TYPE == 'ALL':
        run_benchmark(lib.resize_nearest_bitexact_uint8_t_uint8_t, a, dH, dW, "u8_nearest_bitexact_u8", interpolation=cv2.INTER_NEAREST_EXACT)
    
    if INTER_TYPE == 'INTER_LINEAR_EXACT' or INTER_TYPE == 'ALL':
        run_benchmark(lib.resize_bilinear_bitexact_uint8_t_float, a, dH, dW, "u8_bilinear_bitexact_float", interpolation=cv2.INTER_LINEAR_EXACT)
        run_benchmark(lib.resize_bilinear_bitexact_uint8_t_double, a, dH, dW, "u8_bilinear_bitexact_double", interpolation=cv2.INTER_LINEAR_EXACT)


    print("-" * 85)
    print(" " * 40 + f"dH={dH}, dW={dW}, ch=3")

    a = torch.randn((H, W, 3), dtype=torch.float32).cuda().contiguous()

    if INTER_TYPE == 'INTER_LINEAR' or INTER_TYPE == 'ALL':
        run_benchmark(lib.resize_bilinear_float_float, a, dH, dW, "f32_bilinear_float", interpolation=cv2.INTER_LINEAR)
        run_benchmark(lib.resize_bilinear_float_double, a, dH, dW, "f32_bilinear_double", interpolation=cv2.INTER_LINEAR)
        # run_benchmark(lib.resize_bilinear_2D_float_float, a, dH, dW, "f32_bilinear_2D_float", interpolation=cv2.INTER_LINEAR)
        # run_benchmark(lib.resize_bilinear_2D_float_double, a, dH, dW, "f32_bilinear_2D_double", interpolation=cv2.INTER_LINEAR)
        # # run_benchmark(lib.img_resize_2D_align_shared_float_float, a, dH, dW, "f32_align_shared_float")
        # # run_benchmark(lib.img_resize_2D_align_shared_float_double, a, dH, dW, "f32_align_shared_double")

    if INTER_TYPE == 'INTER_NEAREST' or INTER_TYPE == 'ALL':
        run_benchmark(lib.resize_nearest_float_float, a, dH, dW, "f32_nearest_float", interpolation=cv2.INTER_NEAREST)
        run_benchmark(lib.resize_nearest_float_double, a, dH, dW, "f32_nearest_double", interpolation=cv2.INTER_NEAREST)

    if INTER_TYPE == 'INTER_CUBIC' or INTER_TYPE == 'ALL':
        run_benchmark(lib.resize_bicubic_float_float, a, dH, dW, "f32_bicubic_float", interpolation=cv2.INTER_CUBIC)
        run_benchmark(lib.resize_bicubic_float_double, a, dH, dW, "f32_bicubic_double", interpolation=cv2.INTER_CUBIC)

    if INTER_TYPE == 'INTER_LANCZOS4' or INTER_TYPE == 'ALL':
        run_benchmark(lib.resize_lanczos_float_float, a, dH, dW, "f32_lanczos_float", interpolation=cv2.INTER_LANCZOS4)
        run_benchmark(lib.resize_lanczos_float_double, a, dH, dW, "f32_lanczos_double", interpolation=cv2.INTER_LANCZOS4)

    if (INTER_TYPE == 'INTER_AREA' or INTER_TYPE == 'ALL'):
        if S < 1:
            run_benchmark(lib.resize_area_float_float, a, dH, dW, "f32_area_float", interpolation=cv2.INTER_AREA)
            run_benchmark(lib.resize_area_float_double, a, dH, dW, "f32_area_double", interpolation=cv2.INTER_AREA)
        else:
            run_benchmark(lib.resize_area_bilinear_float_float, a, dH, dW, "f32_area_float", interpolation=cv2.INTER_AREA)
            run_benchmark(lib.resize_area_bilinear_float_double, a, dH, dW, "f32_area_double", interpolation=cv2.INTER_AREA)

    a = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8).cuda().contiguous()

    if INTER_TYPE == 'INTER_LINEAR' or INTER_TYPE == 'ALL':
        run_benchmark(lib.resize_bilinear_uint8_t_float, a, dH, dW, "u8_bilinear_float", interpolation=cv2.INTER_LINEAR)
        run_benchmark(lib.resize_bilinear_uint8_t_double, a, dH, dW, "u8_bilinear_double", interpolation=cv2.INTER_LINEAR)
        run_benchmark(lib.u8_resize_bilinear_uint8_t_float, a, dH, dW, "u8x_bilinear_float", interpolation=cv2.INTER_LINEAR)

    if INTER_TYPE == 'INTER_NEAREST' or INTER_TYPE == 'ALL':
        run_benchmark(lib.resize_nearest_uint8_t_float, a, dH, dW, "u8_nearest_float", interpolation=cv2.INTER_NEAREST)
        run_benchmark(lib.resize_nearest_uint8_t_double, a, dH, dW, "u8_nearest_double", interpolation=cv2.INTER_NEAREST)

    if INTER_TYPE == 'INTER_CUBIC' or INTER_TYPE == 'ALL':
        run_benchmark(lib.resize_bicubic_uint8_t_float, a, dH, dW, "u8_bicubic_float", interpolation=cv2.INTER_CUBIC)
        run_benchmark(lib.resize_bicubic_uint8_t_double, a, dH, dW, "u8_bicubic_double", interpolation=cv2.INTER_CUBIC)
        run_benchmark(lib.u8_resize_bicubic_uint8_t_float, a, dH, dW, "u8x_bicubic_float", interpolation=cv2.INTER_CUBIC)

    if INTER_TYPE == 'INTER_LANCZOS4' or INTER_TYPE == 'ALL':
        run_benchmark(lib.resize_lanczos_uint8_t_float, a, dH, dW, "u8_lanczos_float", interpolation=cv2.INTER_LANCZOS4)
        run_benchmark(lib.resize_lanczos_uint8_t_double, a, dH, dW, "u8_lanczos_double", interpolation=cv2.INTER_LANCZOS4)
        run_benchmark(lib.u8_resize_lanczos_uint8_t_float, a, dH, dW, "u8x_lanczos_float", interpolation=cv2.INTER_LANCZOS4)

    if (INTER_TYPE == 'INTER_AREA' or INTER_TYPE == 'ALL'):
        if S < 1:
            run_benchmark(lib.resize_area_uint8_t_float, a, dH, dW, "u8_area_float", interpolation=cv2.INTER_AREA)
            run_benchmark(lib.resize_area_uint8_t_double, a, dH, dW, "u8_area_double", interpolation=cv2.INTER_AREA)
        else:
            run_benchmark(lib.resize_area_bilinear_uint8_t_float, a, dH, dW, "u8_area_float", interpolation=cv2.INTER_AREA)
            run_benchmark(lib.resize_area_bilinear_uint8_t_double, a, dH, dW, "u8_area_double", interpolation=cv2.INTER_AREA)
            run_benchmark(lib.u8_resize_area_bilinear_uint8_t_float, a, dH, dW, "u8x_area_float", interpolation=cv2.INTER_AREA)

    if INTER_TYPE == 'INTER_NEAREST_EXACT' or INTER_TYPE == 'ALL':
        run_benchmark(lib.resize_nearest_bitexact_uint8_t_uint8_t, a, dH, dW, "u8_nearest_bitexact_u8", interpolation=cv2.INTER_NEAREST_EXACT)
    
    if INTER_TYPE == 'INTER_LINEAR_EXACT' or INTER_TYPE == 'ALL':
        run_benchmark(lib.resize_bilinear_bitexact_uint8_t_float, a, dH, dW, "u8_bilinear_bitexact_float", interpolation=cv2.INTER_LINEAR_EXACT)
        run_benchmark(lib.resize_bilinear_bitexact_uint8_t_double, a, dH, dW, "u8_bilinear_bitexact_double", interpolation=cv2.INTER_LINEAR_EXACT)


    print("-" * 85)
    