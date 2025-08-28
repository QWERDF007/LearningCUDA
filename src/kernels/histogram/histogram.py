
import time
from pathlib import Path
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

import numpy as np

# 禁用梯度计算，因为我们只进行推理，不需要反向传播
torch.set_grad_enabled(False)

# 获取当前Python文件的路径对象
file = Path(__file__)

# 构建编译所需的源文件列表
sources = [
    str(file.parent / "histogram.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

# 动态编译并加载CUDA核函数作为Python模块
# 这种JIT（即时编译）方式允许在运行时编译CUDA代码
lib = load(
    name="histogram_lib",                      # 模块名称
    sources=sources,                           # CUDA源文件列表
    extra_include_paths=extra_include_paths,   # 添加头文件搜索路径
    extra_cuda_cflags=[                        # 额外的CUDA编译器标志
        "-O3",                                 # 启用最高级别的优化
        "-U__CUDA_NO_HALF_OPERATORS__",        # 取消定义，启用half精度运算符
        "-U__CUDA_NO_HALF_CONVERSIONS__",      # 取消定义，启用half精度转换
        "-U__CUDA_NO_HALF2_OPERATORS__",       # 取消定义，启用half2精度运算符
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",  # 取消定义，启用bfloat16转换
        "--expt-relaxed-constexpr",            # 启用实验性的宽松constexpr支持
        "--expt-extended-lambda",              # 启用实验性的扩展lambda支持
        "--use_fast_math",                     # 使用快速数学库，可能牺牲精度换取性能
    ],
    extra_cflags=["-std=c++17"],               # 额外的C++编译器标志，使用C++17标准（Linux用-std）
)


def run_benchmark(
    perf_func: callable,                 # 要测试的函数，可调用对象
    a: torch.Tensor,                     # 输入张量
    bins: int,                           # 直方图桶数, 测试结果用
    tag: str,                            # 测试标签，用于输出识别
    warmup: int = 20,                    # 预热次数，用于GPU预热和缓存优化
    iters: int = 1000,                   # 正式测试的迭代次数
    show_all: bool = False,              # 是否显示完整的输出张量
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
    
    total_time = (end - start) * 1000  # 总时间，转换为毫秒
    mean_time = total_time / iters  # 平均每次迭代的时间

    try:
        np.testing.assert_array_equal(out.cpu().numpy(), np.histogram(a.cpu().numpy(), bins)[0])
        logic = True
    except:
        logic = False

    out_info = f"out_{tag}" 
    
    print(f"{out_info:>18}: ({logic}), iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
    
    return out, mean_time

# 定义测试用的张量高度列表
Hs = [1024, 2048, 4096]
# 定义测试用的张量宽度列表  
Ws = [1024, 2048, 4096]
# 生成所有可能的(高度, 宽度)组合
Sizes = [(H, W) for H in Hs for W in Ws]

# 遍历所有尺寸组合进行性能测试
for H, W in Sizes:
    print("-" * 85)
    print(" " * 40 + f"H={H}, W={W}")

    bins = 256

    # 创建uint8类型的测试张量
    a = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()

    
    # 运行uint8类型的性能基准测试
    run_benchmark(lib.histogram_u8, a, bins, "u8")
    run_benchmark(lib.histogram_u8x4, a, bins, "u8x4")
    run_benchmark(lib.histogram_u8_2D, a, bins, "u8_2D")
    run_benchmark(lib.histogram_u8x4_2D, a, bins, "u8x4_2D")

    print("-" * 85)

    if W > 4096:
        continue

    # 创建int32类型的测试张量
    a = torch.randint(0, 1000, (H, W), dtype=torch.int32).cuda().contiguous()
    bins = a.max().item() + 1

    run_benchmark(lib.histogram_i32, a, bins, "i32")
    run_benchmark(lib.histogram_i32x4, a, bins, "i32x4")

    print("-" * 85)
    
    

    