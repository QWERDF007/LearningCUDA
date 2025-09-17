
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
    decimal = 8
    mismatch_info = ''
    ok = False
    
    # 存储特定精度的不匹配百分比
    mismatch_1e3 = '0%'  # 1e-3精度的不匹配百分比
    mismatch_1e6 = '0%'  # 1e-6精度的不匹配百分比
    passed_decimal = None  # 通过测试的前一个精度
    max_abs_diff_all = 0  # 所有未通过测试中的最大绝对差异
    
    for i in range(12):
        try:
            np.testing.assert_array_almost_equal(out_np, expected, decimal)
            ok = True
            if passed_decimal is None:
                passed_decimal = decimal
            break
        except AssertionError as e:
            msg = str(e)
            
            # 提取不匹配百分比
            match = re.search(r"Mismatched elements:\s*(\d+)\s*/\s*(\d+)\s*\(([\d\.]+%)\)", msg)
            percent_str = None
            if match:
                percent_str = match.group(3)
            
            # 提取最大绝对差异
            match = re.search(r"Max absolute difference:\s*([0-9.eE+-]+)", msg)
            if match:
                max_abs_diff = float(match.group(1))
                max_abs_diff_all = max(max_abs_diff_all, max_abs_diff)
            
            # 记录特定精度的不匹配百分比
            if decimal == 3:  # 1e-3
                mismatch_1e3 = percent_str if percent_str else "0%"
            elif decimal == 6:  # 1e-6
                mismatch_1e6 = percent_str if percent_str else "0%"
            
            decimal -= 1
    
    # 如果在1e-3和1e-6之前就通过了测试，设置这两个精度的不匹配百分比为0
    if passed_decimal is not None:
        if passed_decimal > 3 and mismatch_1e3 is None:
            mismatch_1e3 = "0%"
        if passed_decimal > 6 and mismatch_1e6 is None:
            mismatch_1e6 = "0%"
    
    # 构建mismatch_info字符串
    info_parts = []
    if passed_decimal is not None:
        sign = '-'
        if passed_decimal <= 0:
            passed_decimal = -passed_decimal
            sign = ''
        info_parts.append(f"passed: 1e{sign}{passed_decimal}")
    if mismatch_1e3 is not None:
        info_parts.append(f"1e-3: {mismatch_1e3}")
    if mismatch_1e6 is not None:
        info_parts.append(f"1e-6: {mismatch_1e6}")
    if max_abs_diff_all:
        info_parts.append(f"max_diff: {max_abs_diff_all}")
    mismatch_info = ", ".join(info_parts)
   

    out_info = f"out_{tag}" 
    sign = '-'
    if decimal <= 0:
        decimal = -decimal
        sign = ''
    print(f"{out_info:>40}: {mismatch_info}, iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
    
    return out, mean_time, tag

# Hs = [2, 45, 151, 224, 1024]
# Ws = [2, 200, 320, 448, 2048]
Hs = [2048]
Ws = [1024]
Ss = [0.3, 0.5, 0.9, 1.7, 2.0, 2.4]
Sizes = [(H, W, S) for H in Hs for W in Ws for S in Ss]

for H, W, S in Sizes:
    print("-" * 85)
    print(" " * 40 + f"H={H}, W={W}, S={S}")
    print("-" * 85)
    dH = max(1, int(H * S))
    dW = max(1, int(W * S))
    print(" " * 40 + f"dH={dH}, dW={dW}, ch=1")
    print("-" * 85)

    a = torch.randn((H, W), dtype=torch.float32).cuda().contiguous()
    
    run_benchmark(lib.resize_bilinear_float_float, a, dH, dW, "f32_bilinear_float", interpolation=cv2.INTER_LINEAR)
    run_benchmark(lib.resize_bilinear_float_double, a, dH, dW, "f32_bilinear_double", interpolation=cv2.INTER_LINEAR)
    run_benchmark(lib.resize_bilinear_2D_float_float, a, dH, dW, "f32_bilinear_2D_float", interpolation=cv2.INTER_LINEAR)
    run_benchmark(lib.resize_bilinear_2D_float_double, a, dH, dW, "f32_bilinear_2D_double", interpolation=cv2.INTER_LINEAR)
    run_benchmark(lib.resize_bilinear_shared_2D_float_float, a, dH, dW, "f32_bilinear_shared_float", interpolation=cv2.INTER_LINEAR)
    run_benchmark(lib.resize_bilinear_shared_2D_float_double, a, dH, dW, "f32_bilinear_shared_double", interpolation=cv2.INTER_LINEAR)
    
    a = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()

    run_benchmark(lib.resize_bilinear_uint8_t_float, a, dH, dW, "u8_bilinear_float", interpolation=cv2.INTER_LINEAR)
    run_benchmark(lib.resize_bilinear_uint8_t_double, a, dH, dW, "u8_bilinear_double", interpolation=cv2.INTER_LINEAR)
    run_benchmark(lib.u8_resize_bilinear_uint8_t_float, a, dH, dW, "u8x_bilinear_float", interpolation=cv2.INTER_LINEAR)

    print("-" * 85)
    print(" " * 40 + f"dH={dH}, dW={dW}, ch=3")

    a = torch.randn((H, W, 3), dtype=torch.float32).cuda().contiguous()

    run_benchmark(lib.resize_bilinear_float_float, a, dH, dW, "f32_bilinear_float", interpolation=cv2.INTER_LINEAR)
    run_benchmark(lib.resize_bilinear_float_double, a, dH, dW, "f32_bilinear_double", interpolation=cv2.INTER_LINEAR)
    run_benchmark(lib.resize_bilinear_2D_float_float, a, dH, dW, "f32_bilinear_2D_float", interpolation=cv2.INTER_LINEAR)
    run_benchmark(lib.resize_bilinear_2D_float_double, a, dH, dW, "f32_bilinear_2D_double", interpolation=cv2.INTER_LINEAR)
    # run_benchmark(lib.img_resize_2D_align_shared_float_float, a, dH, dW, "f32_align_shared_float")
    # run_benchmark(lib.img_resize_2D_align_shared_float_double, a, dH, dW, "f32_align_shared_double")

    a = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8).cuda().contiguous()

    run_benchmark(lib.resize_bilinear_uint8_t_float, a, dH, dW, "u8_bilinear_float", interpolation=cv2.INTER_LINEAR)
    run_benchmark(lib.resize_bilinear_uint8_t_double, a, dH, dW, "u8_bilinear_double", interpolation=cv2.INTER_LINEAR)
    run_benchmark(lib.u8_resize_bilinear_uint8_t_float, a, dH, dW, "u8x_bilinear_float", interpolation=cv2.INTER_LINEAR)

    print("-" * 85)
    