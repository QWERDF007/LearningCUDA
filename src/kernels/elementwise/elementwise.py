
import time
from pathlib import Path
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

import numpy as np

# 禁用梯度计算，因为我们只进行推理，不需要反向传播
torch.set_grad_enabled(False)

# 获取当前Python文件的路径对象
file = Path(__file__)

# 构建编译所需的源文件列表
sources = [
    str(file.parent / "elementwise.cu")
]

extra_include_paths = [
    str(file.parent.parent.parent / "common")
]

# 动态编译并加载CUDA核函数作为Python模块
# 这种JIT（即时编译）方式允许在运行时编译CUDA代码
lib = load(
    name="elementwise_lib",                    # 模块名称
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
    a: torch.Tensor,                     # 输入张量A
    b: torch.Tensor,                     # 输入张量B
    tag: str,                            # 测试标签，用于输出识别
    out: Optional[torch.Tensor] = None,  # 可选的输出张量，如果提供则作为输出容器
    warmup: int = 10,                    # 预热次数，用于GPU预热和缓存优化
    iters: int = 1000,                   # 正式测试的迭代次数
    show_all: bool = False,              # 是否显示完整的输出张量
):
    """
    性能基准测试函数
    用于测量CUDA核函数的执行时间性能
    """
    # 如果提供了输出张量，将其清零
    if out is not None:
        out.fill_(0)
    
    # GPU预热阶段：让GPU进入工作状态，避免首次执行的开销影响测量
    if out is not None:
        # 如果有输出张量，使用三参数版本的函数
        for i in range(warmup):
            perf_func(a, b, out)
    else:
        # 如果没有输出张量，使用两参数版本的函数
        for i in range(warmup):
            _ = perf_func(a, b)
    
    # 同步CUDA流，确保所有预热操作完成
    torch.cuda.synchronize()
    
    # 记录开始时间
    start = time.time()
    
    # 正式测试阶段：执行多次迭代以获得稳定的性能数据
    if out is not None:
        # 如果有输出张量，使用三参数版本的函数
        for i in range(iters):
            perf_func(a, b, out)
    else:
        # 如果没有输出张量，使用两参数版本的函数
        for i in range(iters):
            out = perf_func(a, b)
    
    # 同步CUDA流，确保所有计算完成后再记录结束时间
    torch.cuda.synchronize()
    
    # 记录结束时间
    end = time.time()
    
    # 计算性能统计数据
    total_time = (end - start) * 1000  # 总时间，转换为毫秒
    mean_time = total_time / iters  # 平均每次迭代的时间
    
    # 准备输出信息
    out_info = f"out_{tag}"  # 输出标签
    # 获取输出张量的前两个元素，用于验证计算正确性
    a_val = a.flatten().detach().cpu().numpy()
    b_val = b.flatten().detach().cpu().numpy()
    out_val = out.flatten().detach().cpu().numpy()

    res_val = a_val + b_val
    
    decimal = 8
    for i in range(10):
        try:
            np.testing.assert_array_almost_equal(out_val, res_val, decimal)
            break
        except:
            decimal -= 1

    a_val = a.flatten().detach().cpu().numpy().tolist()[0]
    b_val = b.flatten().detach().cpu().numpy().tolist()[0]
    out_val = out.flatten().detach().cpu().numpy().tolist()[0]

    # 将数值四舍五入到8位小数
    # out_val = [round(v, 8) for v in out_val]
    a_val = round(a_val, 4)
    b_val = round(b_val, 4)
    out_val = round(out_val, 4)
    
    # 打印性能测试结果
    print(f"{out_info:>18}: a + b = {a_val} + {b_val} = c = {out_val} in decimal: (1e-{decimal}), iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    # 如果需要，打印完整的输出张量
    if show_all:
        print(out)
    
    # 返回输出张量和平均时间
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

    # 创建uint8类型的测试张量
    a_u8 = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    b_u8 = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    c_u8 = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    
    # 运行uint8类型的性能基准测试
    run_benchmark(lib.elementwise_add_u8, a_u8, b_u8, "u8", c_u8)
    run_benchmark(lib.elementwise_add_u8x4, a_u8, b_u8, "u8x4", c_u8)
    run_benchmark(lib.elementwise_add_u8x4v, a_u8, b_u8, "u8x4v", c_u8)
    run_benchmark(lib.elementwise_add_u8x16_pack, a_u8, b_u8, "u8x16pack", c_u8)
    run_benchmark(partial(torch.add, out=c_u8), a_u8, b_u8, "u8_torch")

    print("-" * 85)
    
    # 创建测试用的随机张量
    a = torch.randn((H, W), dtype=torch.float32).cuda().contiguous()
    b = torch.randn((H, W), dtype=torch.float32).cuda().contiguous()
    c = torch.randn((H, W), dtype=torch.float32).cuda().contiguous()
    
    # 运行性能基准测试
    run_benchmark(lib.elementwise_add_f32, a, b, "f32", c)
    run_benchmark(lib.elementwise_add_f32x4, a, b, "f32x4", c)
    run_benchmark(partial(torch.add, out=c), a, b, "f32_torch")
    
    print("-" * 85)

    a_f16 = a.half().contiguous()
    b_f16 = b.half().contiguous()
    c_f16 = c.half().contiguous()
    run_benchmark(lib.elementwise_add_f16, a_f16, b_f16, "f16", c_f16)
    run_benchmark(lib.elementwise_add_f16x2, a_f16, b_f16, "f16x2", c_f16)
    run_benchmark(lib.elementwise_add_f16x8, a_f16, b_f16, "f16x8", c_f16)
    run_benchmark(lib.elementwise_add_f16x8_pack, a_f16, b_f16, "f16x8pack", c_f16)

    # 使用PyTorch内置的torch.add函数作为性能基准对比
    # partial函数用于预设输出张量参数，使其与自定义kernel函数接口保持一致
    run_benchmark(partial(torch.add, out=c_f16), a_f16, b_f16, "f16_torch")
    
    print("-" * 85)

    