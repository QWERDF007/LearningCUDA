
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
    
    # 精度检测逻辑 - 参照img_resize.py
    decimal = 8
    mismatch_info = ''
    
    # 存储特定精度的不匹配百分比
    mismatch_1e1 = '0%'  # 1e-1精度的不匹配百分比
    mismatch_1e3 = '0%'  # 1e-3精度的不匹配百分比
    mismatch_1e6 = '0%'  # 1e-6精度的不匹配百分比
    passed_decimal = None  # 通过测试的前一个精度
    max_abs_diff_all = 0  # 所有未通过测试中的最大绝对差异

    # np.testing.assert_array_almost_equal(out_np, expected, decimal)

    for i in range(12):
        try:
            np.testing.assert_array_almost_equal(out_np, expected, decimal)
            if passed_decimal is None:
                passed_decimal = decimal
            break
        except AssertionError as e:
            msg = str(e)
            
            # 提取不匹配百分比
            match = re.search(r"Mismatched elements:\s*(\d+)\s*/\s*(\d+)\s*\(([\d\.eE+-]+%)\)", msg)
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
            elif decimal == 1: # 1e-1
                mismatch_1e1 = percent_str if percent_str else "0%"
            
            decimal -= 1
    
    # 如果在1e-3和1e-6之前就通过了测试，设置这两个精度的不匹配百分比为0
    if passed_decimal is not None:
        if passed_decimal > 3 and mismatch_1e3 is None:
            mismatch_1e3 = "0%"
        if passed_decimal > 6 and mismatch_1e6 is None:
            mismatch_1e6 = "0%"
        elif passed_decimal > 6 and mismatch_1e1 is None:
            mismatch_1e1 = "0%"
    
    # 构建mismatch_info字符串
    info_parts = []
    if passed_decimal is not None:
        sign = '-'
        if passed_decimal <= 0:
            passed_decimal = -passed_decimal
            sign = ''
        info_parts.append(f"passed: 1e{sign}{passed_decimal}")
    if mismatch_1e1 is not None:
        info_parts.append(f"1e-1: {mismatch_1e1}")
    if mismatch_1e3 is not None:
        info_parts.append(f"1e-3: {mismatch_1e3}")
    if mismatch_1e6 is not None:
        info_parts.append(f"1e-6: {mismatch_1e6}")
    if max_abs_diff_all:
        info_parts.append(f"max_diff: {max_abs_diff_all}")
    mismatch_info = ", ".join(info_parts)

    out_info = tag 
    sign = '-'
    if decimal <= 0:
        decimal = -decimal
        sign = ''
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