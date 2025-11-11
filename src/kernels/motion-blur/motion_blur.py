
import time
import re
import random
import math
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
    str(file.parent / "motion_blur.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

lib = load(
    name="motion_blur_lib",
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

def genMotionDSF(angle: float, dist: int) -> np.ndarray:
    """
    生成运动模糊核（Directional Spread Function）
    等价于 C++ 版本的 genMotionDSF
    """
    # 运动角度转换到-90~90°
    angle = angle - int(angle / 180) * 180
    if angle > 90:
        angle -= 180
    elif (angle < -90):
        angle += 180

    sin_val = math.sin(angle * math.pi / 180.0)
    cos_val = math.cos(angle * math.pi / 180.0)
    width = int(np.ceil(abs(dist * 0.5 * cos_val)))
    height = int(np.ceil(abs(dist * 0.5 * sin_val)))

    kernel = np.zeros((height * 2 + 1, width * 2 + 1), dtype=np.float32)

    for i in range(int(dist * 0.5) + 1):
        offset_x = round(i * cos_val)
        offset_y = round(i * sin_val)
        kernel[height + offset_y, width + offset_x] = 1.0
        kernel[height - offset_y, width - offset_x] = 1.0

    kernel /= np.sum(kernel)
    return kernel

def Conv2DFT(img: np.ndarray,
             kernel: np.ndarray,
             eps: float = 1e-6,
             conv_type = 1) -> np.ndarray:
    """
    基于 DFT 的 2D 卷积实现，与 OpenCV C++ 版本一致
    """
    # 确保是 float32
    img = img.astype(np.float32)
    kernel = kernel.astype(np.float32)

    # DFT 最佳尺寸
    dft_M = cv2.getOptimalDFTSize(img.shape[0] + kernel.shape[0] - 1)
    dft_N = cv2.getOptimalDFTSize(img.shape[1] + kernel.shape[1] - 1)

    # 填充 image
    imagePad = np.zeros((dft_M, dft_N), np.float32)
    imagePad[:img.shape[0], :img.shape[1]] = img

    # 填充 kernel
    kernelPad = np.zeros((dft_M, dft_N), np.float32)
    kernelPad[:kernel.shape[0], :kernel.shape[1]] = kernel

    # DFT
    dft_img = cv2.dft(imagePad, flags=0, nonzeroRows=img.shape[0])
    dft_kernel = cv2.dft(kernelPad, flags=0, nonzeroRows=kernel.shape[0])

    # 频域乘积
    dft_mult = cv2.mulSpectrums(dft_img, dft_kernel, 0, conjB=False)

    # 逆 DFT
    conv = cv2.idft(dft_mult, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT, nonzeroRows=imagePad.shape[0])

    # 根据卷积模式裁剪
    if conv_type == 0: # full
        r0, c0 = 0, 0
        h, w = img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1
    elif conv_type == 1: # same
        r0, c0 = int((kernel.shape[0] + 0.5) / 2), int((kernel.shape[1] + 0.5) / 2)
        h, w = img.shape[0], img.shape[1]
    elif conv_type == 2: # valid
        r0, c0 = int((kernel.shape[0] + 0.5) / 2), int((kernel.shape[1] + 0.5) / 2)
        h = img.shape[0] - kernel.shape[0] + 1
        w = img.shape[1] - kernel.shape[1] + 1
    else:  # same
        r0, c0 = int((kernel.shape[0] + 0.5) / 2), int((kernel.shape[1] + 0.5) / 2)
        h, w = img.shape[0], img.shape[1]

    convRes = conv[r0:r0 + h, c0:c0 + w]
    convRes = np.clip(convRes, 0, 255)
    convRes = convRes.astype(np.uint8)
    return convRes

def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    kernel: np.ndarray,
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

    kernel_tensor = torch.from_numpy(kernel).cuda().contiguous()
    KH, KW = kernel_tensor.shape[:2]
    # warmup
    for i in range(warmup):
        perf_func(a, out, kernel_tensor, KH, KW)
    
    torch.cuda.synchronize()
    
    start = time.time()

    for i in range(iters):
        perf_func(a, out, kernel_tensor, KH, KW)
        
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000 
    mean_time = total_time / iters 

    a_np = a.cpu().numpy()
    out_np = out.cpu().numpy()
    
    expected = Conv2DFT(a_np, kernel)
    
    mismatch_info = compute_accuracy_info(out_np.astype(expected.dtype), expected)

    out_info = tag
    print(f"{out_info:>25}: {mismatch_info}, iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
    
    return out, mean_time, tag


Hs = [4096]
Ws = [4096]
Ks = [10, 30, 50]
Sizes = [(H, W, K) for H in Hs for W in Ws for K in Ks]

for H, W, K in Sizes:
    print("-" * 85)
    angle = random.randint(-90, 90)
    print(" " * 40 + f"H={H}, W={W}, angle={angle} K={K}, ch=1")

    a = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    kernel = genMotionDSF(angle, K)
    print('kernel')
    print(kernel.shape)
    out = torch.zeros((H,W), dtype=torch.uint8).cuda().contiguous()
    
    run_benchmark(lib.motion_blur, a, kernel, 'motion_blur', out)