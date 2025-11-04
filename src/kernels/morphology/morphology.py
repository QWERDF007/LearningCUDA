
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
    str(file.parent / "morphology.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

lib = load(
    name="morphology_lib",
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
    op,
    kernel,
    tag: str,
    out: Optional[torch.Tensor] = None,
    tmp: Optional[torch.Tensor] = None,
    tmpT: Optional[torch.Tensor] = None,
    is_separable: bool = False,
    is_open_close: bool = False,
    is_tophat_blackhat: bool = False,
    warmup: int = 20,
    iters: int = 1000,
    show_all: bool = False,
):
    """
    性能基准测试函数
    用于测量CUDA核函数的执行时间性能
    """
    # warmup

    ksh, ksw = kernel.shape[:2]
    kernel_tensor = torch.from_numpy(kernel).cuda().contiguous()

    if is_separable:
        if is_open_close:
            for i in range(warmup):
                perf_func(a, out, tmp, tmpT, ksh, ksw)
        elif is_tophat_blackhat:
            for i in range(warmup):
                perf_func(a, out, tmp, tmpT, ksh, ksw)
        else:
            for i in range(warmup):
                perf_func(a, out, tmp, tmpT, ksh, ksw)
    else:
        if is_open_close:
            for i in range(warmup):
                perf_func(a, out, tmp, kernel_tensor, ksh, ksw)
        elif is_tophat_blackhat:
            for i in range(warmup):
                perf_func(a, out, tmp, tmpT, kernel_tensor, ksh, ksw)
        else:
            for i in range(warmup):
                perf_func(a, out, kernel_tensor, ksh, ksw)
    
    torch.cuda.synchronize()
    
    start = time.time()

    if is_separable:
        if is_open_close:
            for i in range(iters):
                perf_func(a, out, tmp, tmpT, ksh, ksw)
        elif is_tophat_blackhat:
            for i in range(iters):
                perf_func(a, out, tmp, tmpT, ksh, ksw)
        else:
            for i in range(iters):
                perf_func(a, out, tmp, tmpT, ksh, ksw)
    else:
        if is_open_close:
            for i in range(iters):
                perf_func(a, out, tmp, kernel_tensor, ksh, ksw)
        elif is_tophat_blackhat:
            for i in range(iters):
                perf_func(a, out, tmp, tmpT, kernel_tensor, ksh, ksw)
        else:
            for i in range(iters):
                perf_func(a, out, kernel_tensor, ksh, ksw)
        
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000 
    mean_time = total_time / iters 

    a_np = a.cpu().numpy()
    
    # 根据是否使用百分比方法选择不同的期望计算方式
    expected = cv2.morphologyEx(a_np, op, kernel)
    
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

def basic_test(H, W, kernel):
    

    a = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.erode_uint8_t_uint8_t, a, cv2.MORPH_ERODE, kernel, 'MORPH_ERODE', out)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.dilate_uint8_t_uint8_t, a, cv2.MORPH_DILATE, kernel, 'MORPH_DILATE', out)

    # out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    # run_benchmark(lib.erode_no_cond_uint8_t_uint8_t, a, cv2.MORPH_ERODE, kernel, 'MORPH_ERODE', out)

    # out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    # run_benchmark(lib.dilate_no_cond_uint8_t_uint8_t, a, cv2.MORPH_DILATE, kernel, 'MORPH_DILATE', out)

    # out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    # run_benchmark(lib.erode_shared_uint8_t_uint8_t, a, cv2.MORPH_ERODE, kernel, 'MORPH_ERODE', out)

    print("-" * 85)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.erode_separable_uint8_t, a, cv2.MORPH_ERODE, kernel, 'SEP  ERODE', 
        out, tmp, tmpT, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.dilate_separable_uint8_t, a, cv2.MORPH_DILATE, kernel, 'SEP DILATE', 
        out, tmp, tmpT, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.erode_separable_shared_uint8_t, a, cv2.MORPH_ERODE, kernel, 'SEP_SHARED  ERODE', 
        out, tmp, tmpT, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.dilate_separable_shared_uint8_t, a, cv2.MORPH_DILATE, kernel, 'SEP_SHARED DILATE', 
        out, tmp, tmpT, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.erode_separable2_shared_uint8_t, a, cv2.MORPH_ERODE, kernel, 'SEP2_SHARED  ERODE', 
        out, tmp, tmpT, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.dilate_separable2_shared_uint8_t, a, cv2.MORPH_DILATE, kernel, 'SEP2_SHARED DILATE', 
        out, tmp, tmpT, is_separable=True)
    
    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.erode_separable_shared_vec4_u8_uint8_t, a, cv2.MORPH_ERODE, kernel, 'SEP_VEC4_CW  ERODE', 
        out, tmp, tmpT, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.dilate_separable_shared_vec4_u8_uint8_t, a, cv2.MORPH_DILATE, kernel, 'SEP_VEC4_CW DILATE', 
        out, tmp, tmpT, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.erode_separable2_shared_vec4_u8_uint8_t, a, cv2.MORPH_ERODE, kernel, 'SEP2_VEC4_CW  ERODE', 
        out, tmp, tmpT, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.dilate_separable2_shared_vec4_u8_uint8_t, a, cv2.MORPH_DILATE, kernel, 'SEP2_VEC4_CW DILATE', 
        out, tmp, tmpT, is_separable=True)


    print("-" * 85)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.erode_separable_T_uint8_t, a, cv2.MORPH_ERODE, kernel, 'SEP_T  ERODE', 
        out, tmp, tmpT, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.dilate_separable_T_uint8_t, a, cv2.MORPH_DILATE, kernel, 'SEP_T DILATE', 
        out, tmp, tmpT, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.erode_separable_T_shared_uint8_t, a, cv2.MORPH_ERODE, kernel, 'SEP_SHARED_T  ERODE', 
        out, tmp, tmpT, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.dilate_separable_T_shared_uint8_t, a, cv2.MORPH_DILATE, kernel, 'SEP_SHARED_T DILATE',
        out, tmp, tmpT, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.erode_separable_T_shared_vec4_uint8_t, a, cv2.MORPH_ERODE, kernel, 'SEP_VEC4_T  ERODE', 
        out, tmp, tmpT, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.dilate_separable_T_shared_vec4_uint8_t, a, cv2.MORPH_DILATE, kernel, 'SEP_VEC4_T DILATE', 
        out, tmp, tmpT, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.erode_separable_T_shared_vec4_u8_uint8_t, a, cv2.MORPH_ERODE, kernel, 'SEP_VEC4_CW_T  ERODE', 
        out, tmp, tmpT, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.dilate_separable_T_shared_vec4_u8_uint8_t, a, cv2.MORPH_DILATE, kernel, 'SEP_VEC4_CW_T DILATE', 
        out, tmp, tmpT, is_separable=True)

def open_close_test(H, W, kernel):
    a = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.open_uint8_t_uint8_t, a, cv2.MORPH_OPEN, kernel, 'MORPH_OPEN', out, tmp, is_open_close=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.close_uint8_t_uint8_t, a, cv2.MORPH_CLOSE, kernel, 'MORPH_CLOSE', out, tmp, is_open_close=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.open_separable_uint8_t, a, cv2.MORPH_OPEN, kernel, 'SEP  OPEN', out, tmp, tmpT, 
        is_open_close=True, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.close_separable_uint8_t, a, cv2.MORPH_CLOSE, kernel, 'SEP CLOSE', out, tmp, tmpT, 
        is_open_close=True, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.open_separable_shared_uint8_t, a, cv2.MORPH_OPEN, kernel, 'SEP_SHARED  OPEN', out, tmp, tmpT, 
        is_open_close=True, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.close_separable_shared_uint8_t, a, cv2.MORPH_CLOSE, kernel, 'SEP_SHARED CLOSE', out, tmp, tmpT, 
        is_open_close=True, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.open_separable2_shared_uint8_t, a, cv2.MORPH_OPEN, kernel, 'SEP2_SHARED  OPEN', out, tmp, tmpT, 
        is_open_close=True, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.close_separable2_shared_uint8_t, a, cv2.MORPH_CLOSE, kernel, 'SEP2_SHARED CLOSE', out, tmp, tmpT, 
        is_open_close=True, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.open_separable_shared_vec4_u8_uint8_t, a, cv2.MORPH_OPEN, kernel, 'SEP_VEC4_CW  OPEN', out, tmp, tmpT, 
        is_open_close=True, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.close_separable_shared_vec4_u8_uint8_t, a, cv2.MORPH_CLOSE, kernel, 'SEP_VEC4_CW CLOSE', out, tmp, tmpT, 
        is_open_close=True, is_separable=True)
    
    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.open_separable_T_shared_vec4_u8_uint8_t, a, cv2.MORPH_OPEN, kernel, 'SEP_VEC4_CW_T  OPEN', out, tmp, tmpT, 
        is_open_close=True, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.close_separable_T_shared_vec4_u8_uint8_t, a, cv2.MORPH_CLOSE, kernel, 'SEP_VEC4_CW_T CLOSE', out, tmp, tmpT, 
        is_open_close=True, is_separable=True)


def tophat_blackhat_test(H, W, kernel):
    a = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.tophat_uint8_t_uint8_t, a, cv2.MORPH_TOPHAT, kernel, 'MORPH_TOPHAT', out, tmp, tmpT, is_tophat_blackhat=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.blackhat_uint8_t_uint8_t, a, cv2.MORPH_BLACKHAT, kernel, 'MORPH_BLACKHAT', out, tmp, tmpT, is_tophat_blackhat=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.tophat_separable_uint8_t, a, cv2.MORPH_TOPHAT, kernel, 'SEP   TOPHAT', out, tmp, tmpT, 
        is_tophat_blackhat=True, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.blackhat_separable_uint8_t, a, cv2.MORPH_BLACKHAT, kernel, 'SEP BLACKHAT', out, tmp, tmpT, 
        is_tophat_blackhat=True, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.tophat_separable_T_shared_vec4_u8_uint8_t, a, cv2.MORPH_TOPHAT, kernel, 'SEP_VEC4_CW_T   TOPHAT', out, tmp, tmpT, 
        is_tophat_blackhat=True, is_separable=True)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    tmpT = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.blackhat_separable_T_shared_vec4_u8_uint8_t, a, cv2.MORPH_BLACKHAT, kernel, 'SEP_VEC4_CW_T BLACKHAT', out, tmp, tmpT, 
        is_tophat_blackhat=True, is_separable=True)

Hs = [4096]
Ws = [46000]
Ks = [11]
Sizes = [(H, W, K) for H in Hs for W in Ws for K in Ks]

for H, W, K in Sizes:
    print("-" * 85)
    print(" " * 40 + f"H={H}, W={W}, K={K}")
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (K, K))

    # basic_test(H, W, kernel)
    # open_close_test(H, W, kernel)
    tophat_blackhat_test(H, W, kernel)
    