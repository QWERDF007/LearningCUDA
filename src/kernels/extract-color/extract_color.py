
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
    str(file.parent / "extract_color.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

lib = load(
    name="extract_color_lib",
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

def extract_color(img, color, left_offset, right_offset, is_invert):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros((h,w), dtype=np.uint8)
    res = np.zeros_like(img, dtype=np.uint8)
    color_range = color_range_map[color]
    low = color_range[0] + left_offset
    up = color_range[1] + right_offset
    lowerb = np.array([low, 43, 46], dtype=np.uint8)
    upperb = np.array([up, 255, 255], dtype=np.uint8)

    # 提取颜色区域
    mask[:] = cv2.inRange(hsv, lowerb, upperb)

    # 反转掩码
    if is_invert:
        mask[:] = cv2.bitwise_not(mask)

    # 应用掩码
    res[:] = cv2.bitwise_and(img, img, mask=mask)

    return res


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    hsv: torch.Tensor,
    mask: torch.Tensor,
    color: str,
    left_offset: int,
    right_offset: int,
    tag: str,
    out: Optional[torch.Tensor] = None,
    is_invert: bool = False,
    warmup: int = 20,
    iters: int = 1000,
    show_all: bool = False,
):
    """
    性能基准测试函数
    用于测量CUDA核函数的执行时间性能
    """
    color_range = color_range_map[color]
    low = color_range[0] + left_offset
    up = color_range[1] + right_offset
    
    # warmup
    for i in range(warmup):
        perf_func(a, out, hsv, mask, low, up, is_invert)
    
    torch.cuda.synchronize()
    
    start = time.time()

    for i in range(iters):
        perf_func(a, out, hsv, mask, low, up, is_invert)
        
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000 
    mean_time = total_time / iters 

    a_np = a.cpu().numpy()
    out_np = out.cpu().numpy()
    
    expected = extract_color(a_np, color, left, right, is_invert)
    
    mismatch_info = compute_accuracy_info(out_np, expected)

    out_info = tag
    print(f"{out_info:>25}: {mismatch_info}, iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
    
    return out, mean_time, tag

# img = cv2.imread("/data2/wt/testImages/yolo/dog.jpg", cv2.IMREAD_COLOR)
# H, W = img.shape[:2]
# img_tensor = torch.from_numpy(img).cuda().contiguous()
# out = torch.zeros_like(img_tensor, dtype=torch.uint8).cuda().contiguous()
# hsv = torch.zeros_like(img_tensor, dtype=torch.uint8).cuda().contiguous()
# mask = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()

# left = 0
# right = 0
# color = '红'

# color_range = color_range_map[color]
# low = color_range[0] + left
# up = color_range[1] + right

# res_cpu = extract_color(img, color, left, right)

# lib.extract_color_uint8_t(img_tensor, out, hsv, mask, low, up)

# torch.cuda.synchronize()

# res_gpu = out.cpu().numpy()

# cv2.imshow("img", img)
# cv2.imshow("cpu", res_cpu)
# cv2.imshow("gpu", res_gpu)
# cv2.waitKey(0)


Hs = [4096]
Ws = [4096]
Colors = ["红", "橙", "黄", "绿", "青", "蓝", "紫"]
Sizes = [(H, W, color) for H in Hs for W in Ws for color in Colors]

for H, W, color in Sizes:
    print("-" * 85)
    print(" " * 40 + f"H={H}, W={W}, color={color}, ch=3")

    left = random.randint(0, 20)
    right = random.randint(0, 25)

    a = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8).cuda().contiguous()
    hsv = torch.zeros((H, W, 3), dtype=torch.uint8).cuda().contiguous()
    mask = torch.zeros((H, W), dtype=torch.uint8).cuda().contiguous()
    out = torch.zeros_like(a).cuda().contiguous()

    run_benchmark(lib.extract_color_uint8_t, a, hsv, mask, color, left, right, 'extract color', out, is_invert=False, iters=1000)
   
    print("-" * 85)