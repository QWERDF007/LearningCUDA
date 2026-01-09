
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

def compare_stats(gpu_stats: np.ndarray, cv_stats: np.ndarray, num_gpu: int, num_cv: int,
                  gpu_labels: np.ndarray = None, cv_labels: np.ndarray = None) -> dict:
    """
    比较 GPU 和 OpenCV 的统计信息
    由于顺序可能不一致，按 area 排序后比较
    
    Args:
        gpu_stats: GPU 统计结果 [x, y, w, h, area]
        cv_stats: OpenCV 统计结果 [x, y, w, h, area] (跳过背景)
        num_gpu: GPU 检测到的连通域数量
        num_cv: OpenCV 检测到的连通域数量 (不含背景)
        gpu_labels: GPU 标签图 (可选，用于诊断)
        cv_labels: OpenCV 标签图 (可选，用于诊断)
    
    Returns:
        比较结果字典
    """
    result = {
        "num_match": num_gpu == num_cv,
        "num_gpu": num_gpu,
        "num_cv": num_cv,
        "stats_match": False,
        "area_match": False,
        "bbox_match": False,
    }
    
    if num_gpu == 0 and num_cv == 0:
        result["stats_match"] = True
        result["area_match"] = True
        result["bbox_match"] = True
        return result
    
    if num_gpu != num_cv:
        return result
    
    # 提取有效数据
    gpu_valid = gpu_stats[:num_gpu]
    cv_valid = cv_stats[1:num_cv + 1]  # 跳过背景 (index 0)
    
    # 按 area (第5列) 排序，面积相同时按 x (第1列)、y (第2列) 排序
    gpu_sort_keys = np.lexsort((gpu_valid[:, 1], gpu_valid[:, 0], gpu_valid[:, 4]))  # 先按area，再按x，再按y
    cv_sort_keys = np.lexsort((cv_valid[:, 1], cv_valid[:, 0], cv_valid[:, 4]))
    gpu_sorted = gpu_valid[gpu_sort_keys]
    cv_sorted = cv_valid[cv_sort_keys]
    
    # 比较 area
    area_match = np.array_equal(gpu_sorted[:, 4], cv_sorted[:, 4])
    result["area_match"] = area_match
    
    # 比较 bbox (x, y, w, h)
    bbox_match = np.array_equal(gpu_sorted[:, :4], cv_sorted[:, :4])
    result["bbox_match"] = bbox_match
    
    # if not bbox_match:
    #     print(f"\n{'Index':<8} {'GPU bbox (x,y,w,h)':<30} {'CV bbox (x,y,w,h)':<30} {'Area':<10}")
    #     print("-" * 80)
    #     mismatch_count = 0
    #     for i in range(min(len(gpu_sorted), len(cv_sorted))):
    #         gpu_bbox = gpu_sorted[i, :4]
    #         cv_bbox = cv_sorted[i, :4]
    #         if not np.array_equal(gpu_bbox, cv_bbox):
    #             mismatch_count += 1
    #             area = gpu_sorted[i, 4]
    #             print(f"{i:<8} {str(tuple(gpu_bbox)):<30} {str(tuple(cv_bbox)):<30} {area:<10}")
    #     print(f"共 {mismatch_count} 个 bbox 不匹配")
    
    result["stats_match"] = area_match and bbox_match
    
    return result


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    tag: str,
    connectivity: int,
    out: Optional[torch.Tensor] = None,
    stats: Optional[torch.Tensor] = None,
    with_stats: bool = False,
    warmup: int = 20,
    iters: int = 1000,
    show_all: bool = False,
):
    """
    性能基准测试函数
    用于测量CUDA核函数的执行时间性能
    
    Args:
        perf_func: 要测试的函数
        a: 输入图像
        tag: 测试标签
        connectivity: 连通性 (4 或 8)
        out: 输出标签图像
        stats: 统计信息输出 (仅 with_stats=True 时使用)
        with_stats: 是否测试带统计信息的版本
        warmup: 预热次数
        iters: 迭代次数
        show_all: 是否显示所有输出
    """
    num_labels = 0
    num_labels_tensor = None
    
    if with_stats:
        # 带统计信息的版本，返回的是 tensor
        for i in range(warmup):
            num_labels_tensor = perf_func(a, out, stats, connectivity)
        
        torch.cuda.synchronize()
        start = time.time()
        
        for i in range(iters):
            num_labels_tensor = perf_func(a, out, stats, connectivity)
        
        torch.cuda.synchronize()
        end = time.time()
        
        # 同步后获取连通域数量
        num_labels = num_labels_tensor.cpu().item()
    else:
        # 不带统计信息的版本
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

    # 获取 OpenCV 结果进行对比
    a_np = a.cpu().numpy()
    gpu_labels = out.cpu().numpy()
    
    if with_stats:
        ret_cv, cpu_labels, cv_stats, _ = cv2.connectedComponentsWithStats(a_np, connectivity=connectivity)
        num_cv = ret_cv - 1  # 不含背景
        
        gpu_stats = stats.cpu().numpy()
        stats_result = compare_stats(gpu_stats, cv_stats, num_labels, num_cv, 
                                      gpu_labels, cpu_labels)
        
        # 如果面积不匹配，进行详细诊断
        if not stats_result["area_match"]:
            diagnose_label_difference(gpu_labels, cpu_labels)
        
        stats_info = (f"num_match={stats_result['num_match']}, "
                      f"area_match={stats_result['area_match']}, "
                      f"bbox_match={stats_result['bbox_match']}, "
                      f"gpu={stats_result['num_gpu']}, cv={stats_result['num_cv']}")
    else:
        ret_cv, cpu_labels = cv2.connectedComponents(a_np, connectivity=connectivity)
    
    # 标签数量对比
    num_gpu_labels = len(np.unique(gpu_labels))
    num_cv_labels = len(np.unique(cpu_labels))
    
    # 二值化后对比准确性
    cpu_binary = (cpu_labels > 0).astype(np.uint8) * 255
    gpu_binary = (gpu_labels > 0).astype(np.uint8) * 255
    mismatch_info = compute_accuracy_info(cpu_binary, gpu_binary)

    out_info = f"out_{tag}" 
    
    if with_stats:
        print(f"{out_info:>40}: {mismatch_info}, {stats_info}")
        print(f"{' ':>40}  iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    else:
        print(f"{out_info:>40}: {mismatch_info}, labels(gpu={num_gpu_labels}, cv={num_cv_labels})")
        print(f"{' ':>40}  iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
        if with_stats:
            print(f"Stats (前10个):\n{stats[:min(10, num_labels)].cpu().numpy()}")
    
    return out, mean_time


Hs = [1024]
Ws = [1024]
Sizes = [(H, W) for H in Hs for W in Ws]

for H, W in Sizes:
    print("-" * 100)
    print(" " * 45 + f"H={H}, W={W}")
    print("-" * 100)

    a = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    a[a >= 128] = 255
    a[a < 128] = 0
    
    out = torch.zeros_like(a, dtype=torch.uint32).cuda().contiguous()
    max_components = H*W+1
    stats = torch.zeros((max_components, 5), dtype=torch.int32).cuda().contiguous()
    
    # 不带统计信息的测试
    run_benchmark(lib.connectedComponent, a, 'connectedComponent-4', 4, out, with_stats=False)
    run_benchmark(lib.connectedComponent, a, 'connectedComponent-8', 8, out, with_stats=False)
    
    # 带统计信息的测试 (基础版本)
    out.zero_()
    stats.zero_()
    run_benchmark(lib.connectedComponentWithStats, a, 'WithStats-4 (basic)', 4, out, stats, with_stats=True)
    
    out.zero_()
    stats.zero_()
    run_benchmark(lib.connectedComponentWithStats, a, 'WithStats-8 (basic)', 8, out, stats, with_stats=True)

    # 带统计信息的测试 (优化版本)
    out.zero_()
    stats.zero_()
    run_benchmark(lib.connectedComponentWithStatsOptimized, a, 'WithStats-4 (optimized)', 4, out, stats, with_stats=True)
    
    out.zero_()
    stats.zero_()
    run_benchmark(lib.connectedComponentWithStatsOptimized, a, 'WithStats-8 (optimized)', 8, out, stats, with_stats=True)

    # 带统计信息的测试 (优化版本)
    out.zero_()
    stats.zero_()
    run_benchmark(lib.connectedComponentWithStatsOptimizedFast, a, 'WithStats-4 (optimized-fast)', 4, out, stats, with_stats=True)
    
    out.zero_()
    stats.zero_()
    run_benchmark(lib.connectedComponentWithStatsOptimizedFast, a, 'WithStats-8 (optimized-fast)', 8, out, stats, with_stats=True)

    print("-" * 100)
