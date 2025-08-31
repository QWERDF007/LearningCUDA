
import time
from pathlib import Path
from functools import partial
from typing import Optional

import cv2

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

import numpy as np

torch.set_grad_enabled(False)

file = Path(__file__)

sources = [
    str(file.parent / "blur.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

lib = load(
    name="blur_lib",
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
    ksz: int,
    tag: str,
    out: Optional[torch.Tensor] = None,
    tmp: Optional[torch.Tensor] = None,
    warmup: int = 20,
    iters: int = 1000,
    show_all: bool = False,
):
    """
    性能基准测试函数
    用于测量CUDA核函数的执行时间性能
    """
    # warmup
    if 'cv' in tag:
        for i in range(warmup):
            _ = perf_func(a, (ksz, ksz))
    elif 'split' in tag:
        for i in range(warmup):
            _ = perf_func(a, ksz, out, tmp)
    else:
        for i in range(warmup):
            _ = perf_func(a, ksz, out)
    
    torch.cuda.synchronize()
    
    start = time.time()
    
    if 'cv' in tag:
        for i in range(iters):
            out = perf_func(a, (ksz, ksz))
    elif 'split' in tag:
        for i in range(iters):
            perf_func(a, ksz, out, tmp)
    else:
        for i in range(iters):
            perf_func(a, ksz, out)
            
    
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000 
    mean_time = total_time / iters 

    if 'cv' in tag:
        expected = cv2.blur(a, (ksz, ksz))
        out_np = out
    else:
        expected = cv2.blur(a.cpu().numpy(), (ksz, ksz))
        out_np = out.cpu().numpy()
    
    # np.testing.assert_array_equal(out_np, expected)
    try:
        np.testing.assert_array_almost_equal(out_np, expected)
        logic = True
    except:
        logic = False
   

    out_info = f"out_{tag}" 
    
    print(f"{out_info:>18}: ({logic}), iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
    
    return out, mean_time

Hs = [1024, 2048, 4096]
Ws = [1024, 2048, 4096]
Ks = [3, 5, 7, 9, 15]
Sizes = [(H, W, ksz) for H in Hs for W in Ws for ksz in Ks]

for H, W, ksz in Sizes:
    print("-" * 85)
    print(" " * 40 + f"H={H}, W={W}, ksz={ksz}")

    a = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    a_np = a.cpu().numpy()
    out = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.int32).cuda().contiguous()

    
    run_benchmark(lib.blur_u8, a, ksz, "u8", out)
    run_benchmark(lib.blur_u8_nb, a, ksz, "u8_no_branch", out)
    run_benchmark(lib.blur_u8_shared, a, ksz, "u8_shared", out)
    run_benchmark(lib.blur_u8_split, a, ksz, "u8_split", out, tmp)
    # run_benchmark(cv2.blur, a_np, ksz, "u8_cv", None)

    print("-" * 85)
    