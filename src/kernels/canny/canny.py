
import time
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
    str(file.parent / "canny.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

lib = load(
    name="canny_lib",
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

def get_sobel_kernel(ksize, dx, dy, normalize=False):
    # kx, ky 返回的是列向量 (kx: ksize x 1, ky: ksize x 1)
    kx, ky = cv2.getDerivKernels(dx, dy, ksize, normalize=normalize)
    # return kx, ky
    sobel_kernel = np.outer(kx.reshape(-1), ky.reshape(-1))
    return sobel_kernel.T

def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    kx, 
    ky,
    low, 
    high,
    apertureSize,
    L2gradient,
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
        perf_func(a, out, kx, ky, low, high, apertureSize, L2gradient)
    
    torch.cuda.synchronize()
    
    start = time.time()

    for i in range(iters):
        perf_func(a, out, kx, ky, low, high, apertureSize, L2gradient)
        
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000 
    mean_time = total_time / iters 

    a_np = a.cpu().numpy()
    
    t0 = time.time()
    # 根据是否使用百分比方法选择不同的期望计算方式
    expected = cv2.Canny(a_np, low, high, apertureSize=apertureSize, L2gradient=L2gradient)
    t1 = time.time()
    print(f"{(t1 - t0)*1000:.3f} ms")
    
    out_np = out.cpu().numpy()
    
    mismatch_info = compute_accuracy_info(out_np, expected)

    out_info = tag 
    
    print(f"{out_info:>18}: {mismatch_info}, iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
    
    return out, mean_time, tag

img = cv2.imread("/data2/wt/testImages/yolo/dog.jpg", cv2.IMREAD_GRAYSCALE)
img_tensor = torch.from_numpy(img).cuda().contiguous()
cpu_canny_res = cv2.Canny(img, 100, 150, apertureSize=3, L2gradient=False)

canny_tensor = torch.zeros_like(img_tensor).cuda().contiguous()

kx = get_sobel_kernel(3, 1, 0)
ky = get_sobel_kernel(3, 0, 1)

kx_tensor = torch.from_numpy(kx.astype(np.int8)).cuda().contiguous()
ky_tensor = torch.from_numpy(ky.astype(np.int8)).cuda().contiguous()

lib.cudaCanny(img_tensor, canny_tensor, kx_tensor, ky_tensor, 100, 150, 3, False)

torch.cuda.synchronize()

gpu_res = canny_tensor.cpu().numpy()

cv2.imwrite("cpu.png", cpu_canny_res)
cv2.imwrite("gpu.png", gpu_res)



for s in range(2,9):
    img = cv2.imread("/data2/wt/testImages/yolo/dog.jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    H, W = img.shape[:2]
    print("-" * 85)
    print(" " * 40 + f"H={H}, W={W}, S={s}")

    img_tensor = torch.from_numpy(img).cuda().contiguous()
    out = torch.zeros_like(img_tensor).cuda().contiguous()
    
    kx = get_sobel_kernel(3, 1, 0)
    ky = get_sobel_kernel(3, 0, 1)
    kx_tensor = torch.from_numpy(kx.astype(np.int8)).cuda().contiguous()
    ky_tensor = torch.from_numpy(ky.astype(np.int8)).cuda().contiguous()

    run_benchmark(lib.cudaCanny, img_tensor, kx_tensor, ky_tensor, 100, 150, 3, False, 'canny', out)
    
    print("-" * 85)