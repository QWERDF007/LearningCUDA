
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
    str(file.parent / "convert_color.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

lib = load(
    name="convert_color_lib",
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
    code,
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
        perf_func(a, out)
    
    torch.cuda.synchronize()
    
    start = time.time()

    for i in range(iters):
        perf_func(a, out)
        
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000 
    mean_time = total_time / iters 

    a_np = a.cpu().numpy()
    
    # 根据是否使用百分比方法选择不同的期望计算方式
    expected = cv2.cvtColor(a_np, code)
    
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

Hs = [1024, 2048]
Ws = [1024, 2048]
Sizes = [(H, W) for H in Hs for W in Ws]

for H, W in Sizes:
    print("-" * 85)
    print(" " * 40 + f"H={H}, W={W}")

    a = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()

    out = torch.zeros((H, W, 3), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.gray2bgr_uint8_t, a, cv2.COLOR_GRAY2BGR, 'COLOR_GRAY2BGR [uint8]', out)

    out = torch.zeros((H, W, 4), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.gray2bgra_uint8_t, a, cv2.COLOR_GRAY2BGRA, 'COLOR_GRAY2BGRA [uint8]', out)

    a = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8).cuda().contiguous()

    out = torch.zeros((H, W, 3), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.bgr2rgb_uint8_t, a, cv2.COLOR_BGR2RGB, 'COLOR_BGR2RGB [uint8]', out)

    run_benchmark(lib.bgr2YCrCb_uint8_t, a, cv2.COLOR_BGR2YCrCb, 'COLOR_BGR2YCrCb [uint8]', out)
    run_benchmark(lib.rgb2YCrCb_uint8_t, a, cv2.COLOR_RGB2YCrCb, 'COLOR_RGB2YCrCb [uint8]', out)

    run_benchmark(lib.bgr2yuv_uint8_t, a, cv2.COLOR_BGR2YUV, 'COLOR_BGR2YUV [uint8]', out)
    run_benchmark(lib.rgb2yuv_uint8_t, a, cv2.COLOR_RGB2YUV, 'COLOR_RGB2YUV [uint8]', out)

    run_benchmark(lib.bgr2hsv_uint8_t, a, cv2.COLOR_BGR2HSV, 'COLOR_BGR2HSV [uint8]', out)
    run_benchmark(lib.hsv2bgr_uint8_t, a, cv2.COLOR_HSV2BGR, 'COLOR_HSV2BGR [uint8]', out)
    run_benchmark(lib.rgb2hsv_uint8_t, a, cv2.COLOR_RGB2HSV, 'COLOR_RGB2HSV [uint8]', out)

    run_benchmark(lib.bgr2hls_uint8_t, a, cv2.COLOR_BGR2HLS, 'COLOR_BGR2HLS [uint8]', out)
    run_benchmark(lib.hls2bgr_uint8_t, a, cv2.COLOR_HLS2BGR, 'COLOR_HLS2BGR [uint8]', out)
    run_benchmark(lib.rgb2hls_uint8_t, a, cv2.COLOR_RGB2HLS, 'COLOR_RGB2HLS [uint8]', out)

    run_benchmark(lib.bgr2xyz_uint8_t, a, cv2.COLOR_BGR2XYZ, 'COLOR_BGR2XYZ [uint8]', out)
    run_benchmark(lib.xyz2bgr_uint8_t, a, cv2.COLOR_XYZ2BGR, 'COLOR_XYZ2BGR [uint8]', out)
    run_benchmark(lib.rgb2xyz_uint8_t, a, cv2.COLOR_RGB2XYZ, 'COLOR_RGB2XYZ [uint8]', out)

    run_benchmark(lib.YCrCb2bgr_uint8_t, a, cv2.COLOR_YCrCb2BGR, 'COLOR_YCrCb2BGR [uint8]', out)
    run_benchmark(lib.yuv2bgr_uint8_t, a, cv2.COLOR_YUV2BGR, 'COLOR_YUV2BGR [uint8]', out)

    run_benchmark(lib.bgr2lab_uint8_t, a, cv2.COLOR_BGR2Lab, 'COLOR_BGR2Lab [uint8]', out)

    out = torch.zeros((H, W, 4), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.bgr2rgba_uint8_t, a, cv2.COLOR_BGR2RGBA, 'COLOR_BGR2RGBA [uint8]', out)
    run_benchmark(lib.bgr2bgra_uint8_t, a, cv2.COLOR_BGR2BGRA, 'COLOR_BGR2BGRA [uint8]', out)

    out = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.bgr2gray_uint8_t, a, cv2.COLOR_BGR2GRAY, 'COLOR_BGR2GRAY [uint8]', out)
    run_benchmark(lib.rgb2gray_uint8_t, a, cv2.COLOR_RGB2GRAY, 'COLOR_RGB2GRAY [uint8]', out)

    # float32 测试 - OpenCV 使用 [0.0, 1.0] 范围
    a = torch.rand((H, W), dtype=torch.float32).cuda().contiguous()

    out = torch.zeros((H, W, 3), dtype=torch.float32).cuda().contiguous()
    run_benchmark(lib.gray2bgr_float, a, cv2.COLOR_GRAY2BGR, 'COLOR_GRAY2BGR [float32]', out)

    out = torch.zeros((H, W, 4), dtype=torch.float32).cuda().contiguous()
    run_benchmark(lib.gray2bgra_float, a, cv2.COLOR_GRAY2BGRA, 'COLOR_GRAY2BGRA [float32]', out)

    a = torch.rand((H, W, 3), dtype=torch.float32).cuda().contiguous()

    out = torch.zeros((H, W, 3), dtype=torch.float32).cuda().contiguous()
    run_benchmark(lib.bgr2rgb_float, a, cv2.COLOR_BGR2RGB, 'COLOR_BGR2RGB [float32]', out)

    run_benchmark(lib.bgr2YCrCb_float, a, cv2.COLOR_BGR2YCrCb, 'COLOR_BGR2YCrCb [float32]', out)
    run_benchmark(lib.rgb2YCrCb_float, a, cv2.COLOR_RGB2YCrCb, 'COLOR_RGB2YCrCb [float32]', out)

    run_benchmark(lib.bgr2yuv_float, a, cv2.COLOR_BGR2YUV, 'COLOR_BGR2YUV [float32]', out)
    run_benchmark(lib.rgb2yuv_float, a, cv2.COLOR_RGB2YUV, 'COLOR_RGB2YUV [float32]', out)

    run_benchmark(lib.bgr2hsv_float, a, cv2.COLOR_BGR2HSV, 'COLOR_BGR2HSV [float32]', out)
    run_benchmark(lib.hsv2bgr_float, a, cv2.COLOR_HSV2BGR, 'COLOR_HSV2BGR [float32]', out)
    run_benchmark(lib.rgb2hsv_float, a, cv2.COLOR_RGB2HSV, 'COLOR_RGB2HSV [float32]', out)

    run_benchmark(lib.bgr2hls_float, a, cv2.COLOR_BGR2HLS, 'COLOR_BGR2HLS [float32]', out)
    run_benchmark(lib.hls2bgr_float, a, cv2.COLOR_HLS2BGR, 'COLOR_HLS2BGR [float32]', out)
    run_benchmark(lib.rgb2hls_float, a, cv2.COLOR_RGB2HLS, 'COLOR_RGB2HLS [float32]', out)

    run_benchmark(lib.bgr2xyz_float, a, cv2.COLOR_BGR2XYZ, 'COLOR_BGR2XYZ [float32]', out)
    run_benchmark(lib.xyz2bgr_float, a, cv2.COLOR_XYZ2BGR, 'COLOR_XYZ2BGR [float32]', out)
    run_benchmark(lib.rgb2xyz_float, a, cv2.COLOR_RGB2XYZ, 'COLOR_RGB2XYZ [float32]', out)

    run_benchmark(lib.YCrCb2bgr_float, a, cv2.COLOR_YCrCb2BGR, 'COLOR_YCrCb2BGR [float32]', out)
    run_benchmark(lib.yuv2bgr_float, a, cv2.COLOR_YUV2BGR, 'COLOR_YUV2BGR [float32]', out)

    run_benchmark(lib.bgr2lab_float, a, cv2.COLOR_BGR2Lab, 'COLOR_BGR2Lab [float32]', out)

    out = torch.zeros((H, W, 4), dtype=torch.float32).cuda().contiguous()
    run_benchmark(lib.bgr2rgba_float, a, cv2.COLOR_BGR2RGBA, 'COLOR_BGR2RGBA [float32]', out)

    out = torch.zeros((H, W), dtype=torch.float32).cuda().contiguous()
    run_benchmark(lib.bgr2gray_float, a, cv2.COLOR_BGR2GRAY, 'COLOR_BGR2GRAY [float32]', out)
    run_benchmark(lib.rgb2gray_float, a, cv2.COLOR_RGB2GRAY, 'COLOR_RGB2GRAY [float32]', out)
    
    print("-" * 85)