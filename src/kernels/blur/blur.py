
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

    # torch.cuda.cudart().cudaProfilerStart()
    # if 'cv' in tag:
    #     out = perf_func(a, (ksz, ksz))
    # elif 'split' in tag:
    #     perf_func(a, ksz, out, tmp)
    # else:
    #     perf_func(a, ksz, out)
    # torch.cuda.cudart().cudaProfilerStop() 
    
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
    
    np.testing.assert_array_equal(out_np, expected)
    try:
        np.testing.assert_array_almost_equal(out_np, expected)
        logic = True
    except:
        logic = False
   

    out_info = f"out_{tag}" 
    
    print(f"{out_info:>20}: ({logic}), iters: {iters}, time: {total_time:.4f}ms, avg: {mean_time:.4f}ms")
    
    if show_all:
        print(out)
    
    return tag, total_time, mean_time

Hs = [1024, 2048, 4096]
Ws = [1024, 2048, 4096]
Ks = [3, 5, 7, 9, 15]

# Hs = [4096]
# Ws = [4096]
# Ks = [7]

Sizes = [(H, W, ksz) for H in Hs for W in Ws for ksz in Ks]

# 存储结果的数据结构
# results[tag][ksz] = {'mean_time': [], 'total_time': [], 'sizes': []}
results = defaultdict(lambda: defaultdict(lambda: {'mean_time': [], 'total_time': [], 'sizes': []}))

for H, W, ksz in Sizes:
    print("-" * 85)
    print(" " * 40 + f"H={H}, W={W}, ksz={ksz}")

    a = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    a_np = a.cpu().numpy()
    out = torch.randint(0, 256, (H, W), dtype=torch.uint8).cuda().contiguous()
    tmp = torch.zeros((H, W), dtype=torch.int32).cuda().contiguous()

    results_list = []
    # 运行基准测试并收集结果
    results_list.append(run_benchmark(lib.blur_u8, a, ksz, "u8", out))
    results_list.append(run_benchmark(lib.blur_u8_nb, a, ksz, "u8_no_branch", out))
    results_list.append(run_benchmark(lib.blur_u8_shared, a, ksz, "u8_shared", out))
    results_list.append(run_benchmark(lib.blur_u8_split, a, ksz, "u8_split", out, tmp))
    results_list.append(run_benchmark(lib.blur_u8_split_shared, a, ksz, "u8_split_shared", out, tmp))
    results_list.append(run_benchmark(lib.blur_u8_split_shared2, a, ksz, "u8_split_shared2", out, tmp))
    results_list.append(run_benchmark(lib.blur_u8_split_sw, a, ksz, "u8_split_sw", out, tmp))
    # tag5, total5, mean5 = run_benchmark(cv2.blur, a_np, ksz, "u8_cv", None)

    # 存储结果
    size_label = f"{H}x{W}"
    for tag, total_time, mean_time in results_list:
        results[tag][ksz]['mean_time'].append(mean_time)
        results[tag][ksz]['total_time'].append(total_time)
        results[tag][ksz]['sizes'].append(size_label)

    print("-" * 85)

# 绘制性能图表
def plot_performance_charts():
    """绘制性能对比图表"""
    plt.rcParams['font.size'] = 12
    
    # 获取所有的ksz值并排序
    all_ksz = sorted(set(Ks))
    
    # 为每个图像尺寸创建子图
    unique_sizes = sorted(set([f"{H}x{W}" for H in Hs for W in Ws]))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'brown']
    markers = ['o', 's', '^', 'D', 'v', 'x', 'p']
    
    # 为每个尺寸单独绘制图表
    for size in unique_sizes:
        # 收集当前尺寸的数据
        size_data = {}
        for tag, tag_data in results.items():
            mean_times = []
            total_times = []
            ksz_values = []
            
            for ksz in all_ksz:
                if ksz in tag_data:
                    size_indices = [i for i, s in enumerate(tag_data[ksz]['sizes']) if s == size]
                    if size_indices:
                        mean_for_ksz = np.mean([tag_data[ksz]['mean_time'][i] for i in size_indices])
                        total_for_ksz = np.mean([tag_data[ksz]['total_time'][i] for i in size_indices])
                        mean_times.append(mean_for_ksz)
                        total_times.append(total_for_ksz)
                        ksz_values.append(ksz)
            
            if mean_times:
                size_data[tag] = {
                    'ksz_values': ksz_values,
                    'mean_times': mean_times,
                    'total_times': total_times
                }
        
        # 绘制 Mean Time 组合图（折线图+柱状图）
        fig_mean, (ax_mean_line, ax_mean_bar) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Mean Time 折线图
        ax_mean_line.set_title(f'Mean Time vs Kernel Size ({size}) - Line Chart', fontweight='bold', fontsize=16)
        ax_mean_line.set_xlabel('Kernel Size (ksz)', fontsize=14)
        ax_mean_line.set_ylabel('Mean Time (ms)', fontsize=14)
        ax_mean_line.grid(True, alpha=0.3)
        
        for color_idx, (tag, data) in enumerate(size_data.items()):
            color = colors[color_idx % len(colors)]
            marker = markers[color_idx % len(markers)]
            ax_mean_line.plot(data['ksz_values'], data['mean_times'], 
                             color=color, marker=marker, linewidth=3, markersize=10,
                             label=tag)
        
        ax_mean_line.legend(fontsize=12)
        ax_mean_line.set_xticks(all_ksz)
        
        # Mean Time 柱状图
        ax_mean_bar.set_title(f'Mean Time vs Kernel Size ({size}) - Bar Chart', fontweight='bold', fontsize=16)
        ax_mean_bar.set_xlabel('Kernel Size (ksz)', fontsize=14)
        ax_mean_bar.set_ylabel('Mean Time (ms)', fontsize=14)
        ax_mean_bar.grid(True, alpha=0.3, axis='y')
        
        # 计算柱状图的位置
        bar_width = 0.15
        tag_list = list(size_data.keys())
        x_positions = np.arange(len(all_ksz))
        
        for color_idx, (tag, data) in enumerate(size_data.items()):
            color = colors[color_idx % len(colors)]
            # 为每个tag创建对应的y值数组
            y_values = []
            for ksz in all_ksz:
                if ksz in data['ksz_values']:
                    idx = data['ksz_values'].index(ksz)
                    y_values.append(data['mean_times'][idx])
                else:
                    y_values.append(0)
            
            offset = (color_idx - len(tag_list)/2 + 0.5) * bar_width
            ax_mean_bar.bar(x_positions + offset, y_values, bar_width, 
                           color=color, alpha=0.8, label=tag)
        
        ax_mean_bar.set_xticks(x_positions)
        ax_mean_bar.set_xticklabels(all_ksz)
        ax_mean_bar.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'blur_mean_time_{size}.png', dpi=100, bbox_inches='tight')
        plt.show()
        print(f"Mean Time 组合图已保存为 'blur_mean_time_{size}.png'")
        plt.close()
        
        # 绘制 Total Time 组合图（折线图+柱状图）
        fig_total, (ax_total_line, ax_total_bar) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Total Time 折线图
        ax_total_line.set_title(f'Total Time vs Kernel Size ({size}) - Line Chart', fontweight='bold', fontsize=16)
        ax_total_line.set_xlabel('Kernel Size (ksz)', fontsize=14)
        ax_total_line.set_ylabel('Total Time (ms)', fontsize=14)
        ax_total_line.grid(True, alpha=0.3)
        
        for color_idx, (tag, data) in enumerate(size_data.items()):
            color = colors[color_idx % len(colors)]
            marker = markers[color_idx % len(markers)]
            ax_total_line.plot(data['ksz_values'], data['total_times'], 
                              color=color, marker=marker, linewidth=3, markersize=10,
                              label=tag)
        
        ax_total_line.legend(fontsize=12)
        ax_total_line.set_xticks(all_ksz)
        
        # Total Time 柱状图
        ax_total_bar.set_title(f'Total Time vs Kernel Size ({size}) - Bar Chart', fontweight='bold', fontsize=16)
        ax_total_bar.set_xlabel('Kernel Size (ksz)', fontsize=14)
        ax_total_bar.set_ylabel('Total Time (ms)', fontsize=14)
        ax_total_bar.grid(True, alpha=0.3, axis='y')
        
        for color_idx, (tag, data) in enumerate(size_data.items()):
            color = colors[color_idx % len(colors)]
            # 为每个tag创建对应的y值数组
            y_values = []
            for ksz in all_ksz:
                if ksz in data['ksz_values']:
                    idx = data['ksz_values'].index(ksz)
                    y_values.append(data['total_times'][idx])
                else:
                    y_values.append(0)
            
            offset = (color_idx - len(tag_list)/2 + 0.5) * bar_width
            ax_total_bar.bar(x_positions + offset, y_values, bar_width, 
                            color=color, alpha=0.8, label=tag)
        
        ax_total_bar.set_xticks(x_positions)
        ax_total_bar.set_xticklabels(all_ksz)
        ax_total_bar.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'blur_total_time_{size}.png', dpi=100, bbox_inches='tight')
        plt.show()
        print(f"Total Time 组合图已保存为 'blur_total_time_{size}.png'")
        plt.close()
        
    print(f"\n所有图表已生成完成!")
    
# 运行完所有测试后绘制图表
plot_performance_charts()
    