
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
    str(file.parent / "letter_box.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

lib = load(
    name="letter_box_lib",
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



def cv_baseline(img, input_w, input_h):
    """
    对输入图像进行预处理，包括缩放和填充
    
    Args:
        img: 输入图像 (numpy array)
        input_w: 目标宽度
        input_h: 目标高度
        
    Returns:
        处理后的图像 (numpy array)
    """
    r_w = input_w / img.shape[1]  # img.cols -> img.shape[1]
    r_h = input_h / img.shape[0]  # img.rows -> img.shape[0]
    ratio = min(r_w, r_h)  # 取比例小的才不会超过要求的宽高
    
    temp_w = int(round(img.shape[1] * ratio))  # 填充之前，缩放之后的宽高
    temp_h = int(round(img.shape[0] * ratio))
    
    dw = (input_w - temp_w) // 2  # 每边要填充的宽高
    dh = (input_h - temp_h) // 2
    
    # 先缩放
    re = cv2.resize(img, (temp_w, temp_h), interpolation=cv2.INTER_LINEAR)
    
    # 计算填充参数
    top = int(round(dh - 0.1))
    bottom = input_h - top - temp_h
    left = int(round(dw - 0.1))
    right = input_w - left - temp_w
    
    # 再填充 - 使用灰色(114, 114, 114)填充
    out = cv2.copyMakeBorder(re, top, bottom, left, right, 
                            cv2.BORDER_CONSTANT, value=(114, 114, 114))
    # BGR -> RGB, HWC -> CHW
    if len(out.shape) == 3:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        out = out.transpose(2, 0, 1)  # HWC -> CHW
    out = out.astype(np.float32) / 255.0
    return out
    



def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    dH: int,
    dW: int,
    tag: str,
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
    
    # 使用cv_baseline作为预期结果
    # 将uint8输入转换为numpy数组进行基准测试
    a_np = a.cpu().numpy()
    if a.dtype == torch.uint8:
        expected = cv_baseline(a_np, dW, dH)
    else:
        # 如果不是uint8，先转换为uint8再进行基准测试
        a_uint8 = (a_np * 255).astype(np.uint8) if a_np.dtype == np.float32 else a_np.astype(np.uint8)
        expected = cv_baseline(a_uint8, dW, dH)
    out_np = out.cpu().numpy()

    mismatch_info = compute_accuracy_info(out_np, expected)
    
    out_info = f"out_{tag}" 
    
    print(f"{out_info:>25}: {mismatch_info}, iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
    
    return out, mean_time, tag

# Hs = [2, 45, 151, 224, 1024]
# Ws = [2, 200, 320, 448, 2048]
Hs = [320, 640, 704]
Ws = [318, 611, 700]
Ss = [0.3, 0.5, 0.9, 1.7, 2.0, 2.4]
Sizes = [(H, W, S) for H in Hs for W in Ws for S in Ss]

for H, W, S in Sizes:
    print("-" * 85)
    print(" " * 40 + f"H={H}, W={W}, S={S}")
    print("-" * 85)
    dH = max(1, int(H * S))
    dW = max(1, int(W * S))
    print(" " * 40 + f"dH={dH}, dW={dW}, ch=1")

    a = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.img_letter_box, a, dH, dW, "letter_box_cuda")
    run_benchmark(lib.img_letter_box_shared, a, dH, dW, "letter_box_cuda_shared")
    
    print(" " * 40 + f"dH={dH}, dW={dW}, ch=3")

    a = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8).cuda().contiguous()
    run_benchmark(lib.img_letter_box, a, dH, dW, "letter_box_cuda")
    run_benchmark(lib.img_letter_box_shared, a, dH, dW, "letter_box_cuda_shared")
    
    print("-" * 85)
    
