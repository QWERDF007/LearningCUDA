# ConnectedComponent

## 介绍

包含以下内容：
- [x] connectedComponent (基础版连通域检测)
- [x] connectedComponentWithStats (基础版连通域分析)
- [x] connectedComponentWithStatsOptimized (warp优化版连通域分析)
- [x] connectedComponentWithStatsOptimizedFast (更快的, 但精度略差)

## 测试

硬件：

- cpu: Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
- gpu: NVIDIA GeForce RTX 4090 D

运行：

```bash
# 只测试Ada架构 不指定默认编译所有架构 耗时较长: Volta, Ampere, Ada, Hopper, ...
export TORCH_CUDA_ARCH_LIST=Ada
python3 convert_color.py
```

输出：

```
----------------------------------------------------------------------------------------------------
                                             H=1024, W=1024
----------------------------------------------------------------------------------------------------
                out_connectedComponent-4: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, labels(gpu=69111, cv=69111)
                                          iters: 1000, time: 35.1498ms, avg: 0.0351ms
                out_connectedComponent-8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, labels(gpu=3563, cv=3563)
                                          iters: 1000, time: 50.8785ms, avg: 0.0509ms
                 out_WithStats-4 (basic): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=True, gpu=69110, cv=69110
                                          iters: 1000, time: 113.7774ms, avg: 0.1138ms
                 out_WithStats-8 (basic): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=True, gpu=3562, cv=3562
                                          iters: 1000, time: 333.9252ms, avg: 0.3339ms
             out_WithStats-4 (optimized): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=True, gpu=69110, cv=69110
                                          iters: 1000, time: 94.6422ms, avg: 0.0946ms
             out_WithStats-8 (optimized): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=True, gpu=3562, cv=3562
                                          iters: 1000, time: 90.0698ms, avg: 0.0901ms
        out_WithStats-4 (optimized-fast): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=False, gpu=69110, cv=69110
                                          iters: 1000, time: 88.1555ms, avg: 0.0882ms
        out_WithStats-8 (optimized-fast): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=False, gpu=3562, cv=3562
                                          iters: 1000, time: 86.1235ms, avg: 0.0861ms
----------------------------------------------------------------------------------------------------

```



