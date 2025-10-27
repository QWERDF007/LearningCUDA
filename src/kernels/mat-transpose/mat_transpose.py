
import time
from pathlib import Path
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

import numpy as np

torch.set_grad_enabled(False)

file = Path(__file__)

sources = [
    str(file.parent / "mat_transpose.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

lib = load(
    name="mat_transpose_lib",
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
    if out is not None:
        for i in range(warmup):
            _ = perf_func(a, out)
    else:
        for i in range(warmup):
            _ = perf_func(a)
    
    torch.cuda.synchronize()
    
    start = time.time()
    
    if out is not None:
        for i in range(iters):
            perf_func(a, out)
    else:
        for i in range(iters):
            out = perf_func(a)
    
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000 
    mean_time = total_time / iters 

    expected = a.T.cpu().numpy()
    out_np = out.cpu().numpy()
    decimal = 8
    for i in range(10):
        try:
            np.testing.assert_array_almost_equal(out_np, expected, decimal)
            break
        except:
            decimal -= 1
   

    out_info = f"out_{tag}" 
    
    print(f"{out_info:>30}: (1e-{decimal}), iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
    
    return out, mean_time

Hs = [1024, 2048, 4096, 8192]
Ws = [1024, 2048, 4096, 8192]
Sizes = [(H, W) for H in Hs for W in Ws]

for H, W in Sizes:
    print("-" * 85)
    print(" " * 40 + f"H={H}, W={W}")

    a = torch.randn((H, W), dtype=torch.float32).cuda().contiguous()
    out = torch.randn((W, H), dtype=torch.float32).cuda().contiguous()
    
    run_benchmark(lib.mat_transpose_f32_coalesced_read, a, "f32_coalesced_read", out)
    run_benchmark(lib.mat_transpose_f32_coalesced_write, a, "f32_coalesced_write", out)
    run_benchmark(lib.mat_transpose_f32x4_coalesced_read, a, "f32x4_coalesced_read", out)
    run_benchmark(lib.mat_transpose_f32x4_coalesced_write, a, "f32x4_coalesced_write", out)

    run_benchmark(lib.mat_transpose_f32_coalesced_read_2d, a, "f32_coalesced_read_2d", out)
    run_benchmark(lib.mat_transpose_f32_coalesced_write_2d, a, "f32_coalesced_write_2d", out)
    run_benchmark(lib.mat_transpose_f32x4_coalesced_read_2d, a, "f32x4_coalesced_read_2d", out)
    run_benchmark(lib.mat_transpose_f32x4_coalesced_write_2d, a, "f32x4_coalesced_write_2d", out)
    # run_benchmark(lib.mat_transpose_f32_2d_1, a, "f32_2d_1", out)
    # run_benchmark(lib.mat_transpose_f32_2d_2, a, "f32_2d_2", out)
    # run_benchmark(lib.mat_transpose_f32_2d_3, a, "f32_2d_3", out)
    run_benchmark(lib.mat_transpose_shared_float, a, "f32_shared", out)
    run_benchmark(lib.mat_transpose_shared_2_float, a, "f32_shared_2", out)

    a = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    out = torch.zeros((W, H), dtype=torch.uint8).cuda().contiguous()

    run_benchmark(lib.mat_transpose_u8_coalesced_read, a, "u8_coalesced_read", out)
    run_benchmark(lib.mat_transpose_u8_coalesced_write, a, "u8_coalesced_write", out)
    run_benchmark(lib.mat_transpose_u8x4_coalesced_read, a, "u8x4_coalesced_read", out)
    run_benchmark(lib.mat_transpose_u8x4_coalesced_write, a, "u8x4_coalesced_write", out)
    run_benchmark(lib.mat_transpose_u8x16_coalesced_write, a, "u8x16_coalesced_write", out)

    run_benchmark(lib.mat_transpose_u8x4_coalesced_read_2d, a, "u8x4_coalesced_read_2d", out)
    run_benchmark(lib.mat_transpose_u8x4_coalesced_write_2d, a, "u8x4_coalesced_write_2d", out)
    run_benchmark(lib.mat_transpose_u8x16_coalesced_write_2d, a, "u8x16_coalesced_write_2d", out)

    run_benchmark(lib.mat_transpose_shared_uint8_t, a, "u8_shared", out)
    run_benchmark(lib.mat_transpose_shared_2_uint8_t, a, "u8_shared_2", out)


    print("-" * 85)
    