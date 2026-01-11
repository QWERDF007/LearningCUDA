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
                out_connectedComponent-4: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, labels(gpu=69318, cv=69318)
                                          iters: 1000, time: 34.5492ms, avg: 0.0345ms
                out_connectedComponent-8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, labels(gpu=3565, cv=3565)
                                          iters: 1000, time: 51.2171ms, avg: 0.0512ms
                 out_WithStats-4 (basic): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=True, gpu=69317, cv=69317
                                          iters: 1000, time: 116.5986ms, avg: 0.1166ms
                 out_WithStats-8 (basic): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=True, gpu=3564, cv=3564
                                          iters: 1000, time: 334.1296ms, avg: 0.3341ms
             out_WithStats-4 (optimized): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=True, gpu=69317, cv=69317
                                          iters: 1000, time: 94.2075ms, avg: 0.0942ms
             out_WithStats-8 (optimized): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=True, gpu=3564, cv=3564
                                          iters: 1000, time: 90.2987ms, avg: 0.0903ms
        out_WithStats-4 (optimized-fast): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=False, gpu=69317, cv=69317
                                          iters: 1000, time: 88.0456ms, avg: 0.0880ms
        out_WithStats-8 (optimized-fast): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=False, gpu=3564, cv=3564
                                          iters: 1000, time: 86.6599ms, avg: 0.0867ms
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
                                             H=1024, W=4096
----------------------------------------------------------------------------------------------------
                out_connectedComponent-4: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, labels(gpu=276846, cv=276846)
                                          iters: 1000, time: 98.3522ms, avg: 0.0984ms
                out_connectedComponent-8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, labels(gpu=14166, cv=14166)
                                          iters: 1000, time: 131.1023ms, avg: 0.1311ms
                 out_WithStats-4 (basic): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=True, gpu=276845, cv=276845
                                          iters: 1000, time: 512.0316ms, avg: 0.5120ms
                 out_WithStats-8 (basic): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=True, gpu=14165, cv=14165
                                          iters: 1000, time: 1252.4996ms, avg: 1.2525ms
             out_WithStats-4 (optimized): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=True, gpu=276845, cv=276845
                                          iters: 1000, time: 426.9133ms, avg: 0.4269ms
             out_WithStats-8 (optimized): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=True, gpu=14165, cv=14165
                                          iters: 1000, time: 366.9281ms, avg: 0.3669ms
        out_WithStats-4 (optimized-fast): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=False, gpu=276845, cv=276845
                                          iters: 1000, time: 426.8470ms, avg: 0.4268ms
        out_WithStats-8 (optimized-fast): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=False, gpu=14165, cv=14165
                                          iters: 1000, time: 366.2536ms, avg: 0.3663ms
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
                                             H=4096, W=1024
----------------------------------------------------------------------------------------------------
                out_connectedComponent-4: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, labels(gpu=276770, cv=276770)
                                          iters: 1000, time: 102.5202ms, avg: 0.1025ms
                out_connectedComponent-8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, labels(gpu=14003, cv=14003)
                                          iters: 1000, time: 161.8321ms, avg: 0.1618ms
                 out_WithStats-4 (basic): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=True, gpu=276769, cv=276769
                                          iters: 1000, time: 517.9090ms, avg: 0.5179ms
                 out_WithStats-8 (basic): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=True, gpu=14002, cv=14002
                                          iters: 1000, time: 1285.2428ms, avg: 1.2852ms
             out_WithStats-4 (optimized): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=True, gpu=276769, cv=276769
                                          iters: 1000, time: 430.9335ms, avg: 0.4309ms
             out_WithStats-8 (optimized): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=True, gpu=14002, cv=14002
                                          iters: 1000, time: 398.7246ms, avg: 0.3987ms
        out_WithStats-4 (optimized-fast): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=False, gpu=276769, cv=276769
                                          iters: 1000, time: 431.8182ms, avg: 0.4318ms
        out_WithStats-8 (optimized-fast): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=False, gpu=14002, cv=14002
                                          iters: 1000, time: 397.5368ms, avg: 0.3975ms
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
                                             H=4096, W=4096
----------------------------------------------------------------------------------------------------
                out_connectedComponent-4: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, labels(gpu=1101214, cv=1101214)
                                          iters: 1000, time: 598.4516ms, avg: 0.5985ms
                out_connectedComponent-8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, labels(gpu=54953, cv=54953)
                                          iters: 1000, time: 711.5393ms, avg: 0.7115ms
                 out_WithStats-4 (basic): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=True, gpu=1101213, cv=1101213
                                          iters: 1000, time: 2385.5875ms, avg: 2.3856ms
                 out_WithStats-8 (basic): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=True, gpu=54952, cv=54952
                                          iters: 1000, time: 5894.9156ms, avg: 5.8949ms
             out_WithStats-4 (optimized): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=True, gpu=1101213, cv=1101213
                                          iters: 1000, time: 1975.2607ms, avg: 1.9753ms
             out_WithStats-8 (optimized): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=True, gpu=54952, cv=54952
                                          iters: 1000, time: 2110.0657ms, avg: 2.1101ms
        out_WithStats-4 (optimized-fast): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=False, gpu=1101213, cv=1101213
                                          iters: 1000, time: 1981.5209ms, avg: 1.9815ms
        out_WithStats-8 (optimized-fast): passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, num_match=True, area_match=True, bbox_match=False, gpu=54952, cv=54952
                                          iters: 1000, time: 2106.2386ms, avg: 2.1062ms
----------------------------------------------------------------------------------------------------

```



