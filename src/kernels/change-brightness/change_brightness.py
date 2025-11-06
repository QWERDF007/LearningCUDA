
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
    str(file.parent / "change_brightness.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

lib = load(
    name="change_brightness_lib",
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
    alpha,
    beta,
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
        perf_func(a, out, alpha, beta)
    
    torch.cuda.synchronize()
    
    start = time.time()

    for i in range(iters):
        perf_func(a, out, alpha, beta)
        
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000 
    mean_time = total_time / iters 

    a_np = a.cpu().numpy()
    
    # 根据是否使用百分比方法选择不同的期望计算方式
    # C++代码转换为Python:
    # src_img.convertTo(ret_img, CV_32F, gain_value, offset_value);
    # cv::convertScaleAbs(ret_img, ret_img);
    # ret_img.convertTo(ret_img, CV_8U);
    
    # 方法1：分步实现（完全对应C++逻辑）
    # ret_img = a_np.astype(np.float32) * alpha + beta  # convertTo with gain/offset
    # expected = cv2.convertScaleAbs(ret_img)           # convertScaleAbs + convertTo(CV_8U)
    
    # 方法2：一步实现（推荐，OpenCV内部优化）
    expected = cv2.convertScaleAbs(a_np, alpha=alpha, beta=beta)
    
    out_np = out.cpu().numpy()
    
    mismatch_info = compute_accuracy_info(out_np, expected)

    out_info = f"out_{tag} (α:{alpha:.3f} | β:{beta:.1f})" 
    
    print(f"{out_info:>40}: {mismatch_info}, iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
    
    return out, mean_time, tag

Hs = [1024, 2048, 4096]
Ws = [1024, 2048, 4096]
Sizes = [(H, W) for H in Hs for W in Ws]

for H, W in Sizes:
    print("-" * 85)
    # 生成随机的 alpha (增益) 和 beta (偏移) 值
    # alpha 范围：0.5 到 2.0 (调整对比度/增益)
    # beta 范围：-50 到 50 (调整亮度/偏移)
    alpha = random.uniform(0.2, 80.0)
    beta = random.uniform(-255, 255)
    
    print(" " * 30 + f"H={H}, W={W}, ch=1 | α={alpha:.3f}, β={beta:.1f}")

    a = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    out = torch.zeros_like(a).cuda().contiguous()

    run_benchmark(lib.change_brightness_uint8_t_float, a, alpha, beta, 'u8_float', out)
    run_benchmark(lib.change_brightness_uint8_t_double, a, alpha, beta, 'u8_double', out)

    print("-" * 85)

    print(" " * 30 + f"H={H}, W={W}, ch=3 | α={alpha:.3f}, β={beta:.1f}")

    a = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8).cuda().contiguous()
    out = torch.zeros_like(a).cuda().contiguous()

    run_benchmark(lib.change_brightness_uint8_t_float, a, alpha, beta, 'u8_float', out)
    run_benchmark(lib.change_brightness_uint8_t_double, a, alpha, beta, 'u8_double', out)
    
    print("-" * 85)