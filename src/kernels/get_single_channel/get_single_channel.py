
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
    str(file.parent / "get_single_channel.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

lib = load(
    name="get_single_channel_lib",
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

color_range_map = {
    "红": (0, 10),
    "橙": (11, 25),
    "黄": (26, 34),
    "绿": (35, 77),
    "青": (78, 99),
    "蓝": (100, 124),
    "紫": (125, 155),
}

def get_single_channel(img, channel_type: str, channel1: int, channel2: int):
    channels = cv2.split(img)
    if channel_type == "两通道之差":
        res = cv2.subtract(channels[channel1], channels[channel2])
    else:
        res = channels[channel1]
    return res


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    channel_type: str,
    c1: int,
    c2: int,
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
        perf_func(a, out, c1, c2)
    
    torch.cuda.synchronize()
    
    start = time.time()

    for i in range(iters):
        perf_func(a, out, c1, c2)
        
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000 
    mean_time = total_time / iters 

    a_np = a.cpu().numpy()
    out_np = out.cpu().numpy()
    
    expected = get_single_channel(a_np, channel_type, c1, c2)
    
    mismatch_info = compute_accuracy_info(out_np, expected)

    out_info = tag
    print(f"{out_info:>25}: {mismatch_info}, iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
    
    return out, mean_time, tag

Hs = [4096]
Ws = [4096]
Channels = [('蓝色通道', 0), ('绿色通道', 1), ('红色通道', 2), ('两通道之差', -1)]
Sizes = [(H, W, ci) for H in Hs for W in Ws for ci in Channels]

for H, W, ci in Sizes:
    print("-" * 85)
    print(" " * 40 + f"H={H}, W={W}, ch=3")

    c, i = ci

    a = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8).cuda().contiguous()
    out = torch.zeros((H,W), dtype=torch.uint8).cuda().contiguous()

    if i == -1:
        run_benchmark(lib.channel_subtract_uint8_t, a, c, 0, 1, c, out, iters=1000)
        run_benchmark(lib.channel_subtract_uint8_t, a, c, 1, 1, c, out, iters=1000)
        run_benchmark(lib.channel_subtract_uint8_t, a, c, 2, 1, c, out, iters=1000)
    else:
        run_benchmark(lib.get_single_channel_uint8_t, a, c, i, 0, c, out, iters=1000)
   
    print("-" * 85)