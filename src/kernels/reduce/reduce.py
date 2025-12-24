
import time
import re
from pathlib import Path
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

import cv2
import numpy as np

# 禁用梯度计算，因为我们只进行推理，不需要反向传播
torch.set_grad_enabled(False)

# 获取当前Python文件的路径对象
file = Path(__file__)
# 将 src/common 加入到 sys.path，便于导入 helper
import sys
sys.path.append(str(file.parent.parent.parent / "common"))
from helper import compute_accuracy_info

# 构建编译所需的源文件列表
sources = [
    str(file.parent / "reduce.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

lib = load(
    name="reduce_lib",
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
        _ = perf_func(a)
    
    torch.cuda.synchronize()
    
    start = time.time()
    
    for i in range(iters):
        out = perf_func(a)
    
    torch.cuda.synchronize()
    
    end = time.time()
    
    total_time = (end - start) * 1000 
    mean_time = total_time / iters 
    expected = np.sum(a.cpu().numpy())
    out_np : np.ndarray = out.cpu().numpy()

    if out_np.shape[1] != 1:
        out_np = out_np[0, 0]

    mismatch_info = compute_accuracy_info(out_np, expected)

    out_info = f"out_{tag}" 
    
    print(f"{out_info:>40}: {mismatch_info}, iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
    
    return out, mean_time, tag

Hs = [1024, 4096, 40961]
Ws = [4096]
Sizes = [(H, W) for H in Hs for W in Ws]

# 如果直接运行此脚本，则执行完整测试
if __name__ == "__main__":
    # 遍历所有尺寸组合进行性能测试
    for H, W in Sizes:
        print("-" * 85)
        print(" " * 40 + f"H={H}, W={W}")

        a = torch.randn((H, W), dtype=torch.float32).cuda().contiguous()

        run_benchmark(lib.reduce_sum_atomic_float_float, a, "reduce_sum_atomic_float")
        run_benchmark(lib.reduce_sum_atomic_float_double, a, "reduce_sum_atomic_double")
        run_benchmark(lib.reduce_sum_shared_float_float, a, "reduce_sum_shared_float")
        run_benchmark(lib.reduce_sum_shared_float_double, a, "reduce_sum_shared_double")
        run_benchmark(lib.reduce_sum_shared_atomic_float_float, a, "reduce_sum_shared_atomic_float")
        run_benchmark(lib.reduce_sum_shared_atomic_float_double, a, "reduce_sum_shared_atomic_double")
        run_benchmark(lib.reduce_sum_shared_atomic2_float_float, a, "reduce_sum_shared_atomic2_float")
        run_benchmark(lib.reduce_sum_shared_atomic2_float_double, a, "reduce_sum_shared_atomic2_double")
        print("-" * 85)