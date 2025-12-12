
import time
from pathlib import Path
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

import cv2
import numpy as np

torch.set_grad_enabled(False)

file = Path(__file__)
# 将 src/common 加入到 sys.path，便于导入 helper
import sys
sys.path.append(str(file.parent.parent.parent / "common"))
from helper import compute_accuracy_info

sources = [
    str(file.parent / "connected_component.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

lib = load(
    name="connected_component_lib",
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
    tag: str,
    connectivity: int,
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
        perf_func(a, out, connectivity)
    
    torch.cuda.synchronize()
    
    start = time.time()
    
    for i in range(iters):
        perf_func(a, out, connectivity)
    
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000 
    mean_time = total_time / iters 

    ret, cpu_labels = cv2.connectedComponents(a.cpu().numpy(), connectivity=connectivity)
    gpu_labels = out.cpu().numpy()
    print(len(np.unique(cpu_labels)))
    print(len(np.unique(gpu_labels)))
    cpu_labels[cpu_labels > 0] = 255
    gpu_labels[gpu_labels > 0] = 255

    mismatch_info = compute_accuracy_info(cpu_labels, gpu_labels)

    out_info = f"out_{tag}" 
    
    print(f"{out_info:>30}: {mismatch_info}, iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
    
    return out, mean_time

Hs = [1024]
Ws = [1024]
Sizes = [(H, W) for H in Hs for W in Ws]

for H, W in Sizes:
    print("-" * 85)
    print(" " * 40 + f"H={H}, W={W}")

    a = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    a[a>=128] = 255
    a[a<128] = 0
    out = torch.zeros_like(a, dtype=torch.uint32).cuda().contiguous()
    
    run_benchmark(lib.connectedComponent, a, 'connectedComponent-4', 4, out)
    run_benchmark(lib.connectedComponent, a, 'connectedComponent-8', 8, out)

    print("-" * 85)
    