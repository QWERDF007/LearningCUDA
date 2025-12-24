# Histogram

## 介绍

包含以下内容：
- [x] reduce_sum_atomic (原子加版本)
- [x] reduce_sum_shared (共享内存版本)
- [x] reduce_sum_shared_atomic (共享内存原子加版本)
- [x] reduce_sum_shared_atomic2 (每个thread处理多个数据版本)

## 测试

硬件：

- cpu: Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
- gpu: NVIDIA GeForce RTX 4090 D

运行：

```bash
# 只测试Ada架构 不指定默认编译所有架构 耗时较长: Volta, Ampere, Ada, Hopper, ...
export TORCH_CUDA_ARCH_LIST=Ada
python3 reduce.py
```

输出：

```
-------------------------------------------------------------------------------------
                                        H=1024, W=4096
             out_reduce_sum_atomic_float: passed: 1e-2, 1e-1: 0%, 1e-3: 100%, 1e-6: 100%, max_diff: 0.00427246, iters: 1000, time: 49.8905ms, avg: 0.0499ms
            out_reduce_sum_atomic_double: passed: 1e-3, 1e-1: 0%, 1e-3: 0%, 1e-6: 100%, max_diff: 0.00080284, iters: 1000, time: 60.6604ms, avg: 0.0607ms
             out_reduce_sum_shared_float: passed: 1e-3, 1e-1: 0%, 1e-3: 0%, 1e-6: 100%, max_diff: 0.00061035, iters: 1000, time: 31.7438ms, avg: 0.0317ms
            out_reduce_sum_shared_double: passed: 1e-3, 1e-1: 0%, 1e-3: 0%, 1e-6: 100%, max_diff: 0.00080284, iters: 1000, time: 77.5728ms, avg: 0.0776ms
      out_reduce_sum_shared_atomic_float: passed: 1e-2, 1e-1: 0%, 1e-3: 100%, 1e-6: 100%, max_diff: 0.00183105, iters: 1000, time: 29.3171ms, avg: 0.0293ms
     out_reduce_sum_shared_atomic_double: passed: 1e-3, 1e-1: 0%, 1e-3: 0%, 1e-6: 100%, max_diff: 0.00079049, iters: 1000, time: 77.4844ms, avg: 0.0775ms
     out_reduce_sum_shared_atomic2_float: passed: 1e-2, 1e-1: 0%, 1e-3: 100%, 1e-6: 100%, max_diff: 0.00268555, iters: 1000, time: 9.4035ms, avg: 0.0094ms
    out_reduce_sum_shared_atomic2_double: passed: 1e-3, 1e-1: 0%, 1e-3: 0%, 1e-6: 100%, max_diff: 0.00085603, iters: 1000, time: 27.2365ms, avg: 0.0272ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=4096
             out_reduce_sum_atomic_float: passed: 1e-1, 1e-1: 0%, 1e-3: 100%, 1e-6: 100%, max_diff: 0.03320312, iters: 1000, time: 61.7175ms, avg: 0.0617ms
            out_reduce_sum_atomic_double: passed: 1e-3, 1e-1: 0%, 1e-3: 0%, 1e-6: 100%, max_diff: 0.00116559, iters: 1000, time: 136.1647ms, avg: 0.1362ms
             out_reduce_sum_shared_float: passed: 1e-3, 1e-1: 0%, 1e-3: 0%, 1e-6: 100%, max_diff: 0.00097656, iters: 1000, time: 111.0144ms, avg: 0.1110ms
            out_reduce_sum_shared_double: passed: 1e-3, 1e-1: 0%, 1e-3: 0%, 1e-6: 100%, max_diff: 0.00116559, iters: 1000, time: 291.7795ms, avg: 0.2918ms
      out_reduce_sum_shared_atomic_float: passed: 1e-1, 1e-1: 0%, 1e-3: 100%, 1e-6: 100%, max_diff: 0.02636719, iters: 1000, time: 108.0854ms, avg: 0.1081ms
     out_reduce_sum_shared_atomic_double: passed: 1e-3, 1e-1: 0%, 1e-3: 0%, 1e-6: 100%, max_diff: 0.00101842, iters: 1000, time: 300.1623ms, avg: 0.3002ms
     out_reduce_sum_shared_atomic2_float: passed: 1e-2, 1e-1: 0%, 1e-3: 100%, 1e-6: 100%, max_diff: 0.01171875, iters: 1000, time: 20.5810ms, avg: 0.0206ms
    out_reduce_sum_shared_atomic2_double: passed: 1e-3, 1e-1: 0%, 1e-3: 0%, 1e-6: 100%, max_diff: 0.00106611, iters: 1000, time: 74.3606ms, avg: 0.0744ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=40961, W=4096
             out_reduce_sum_atomic_float: passed: 1e-1, 1e-1: 0%, 1e-3: 100%, 1e-6: 100%, max_diff: 0.02612305, iters: 1000, time: 749.4347ms, avg: 0.7494ms
            out_reduce_sum_atomic_double: passed: 1e-2, 1e-1: 0%, 1e-3: 100%, 1e-6: 100%, max_diff: 0.01130304, iters: 1000, time: 1133.4615ms, avg: 1.1335ms
             out_reduce_sum_shared_float: passed: 1e-2, 1e-1: 0%, 1e-3: 100%, 1e-6: 100%, max_diff: 0.00787354, iters: 1000, time: 1105.6111ms, avg: 1.1056ms
            out_reduce_sum_shared_double: passed: 1e-2, 1e-1: 0%, 1e-3: 100%, 1e-6: 100%, max_diff: 0.01130304, iters: 1000, time: 2864.5215ms, avg: 2.8645ms
      out_reduce_sum_shared_atomic_float: passed: 1e-1, 1e-1: 0%, 1e-3: 100%, 1e-6: 100%, max_diff: 0.04125977, iters: 1000, time: 1093.2441ms, avg: 1.0932ms
     out_reduce_sum_shared_atomic_double: passed: 1e-2, 1e-1: 0%, 1e-3: 100%, 1e-6: 100%, max_diff: 0.01165947, iters: 1000, time: 2975.8921ms, avg: 2.9759ms
     out_reduce_sum_shared_atomic2_float: passed: 1e-2, 1e-1: 0%, 1e-3: 100%, 1e-6: 100%, max_diff: 0.00816345, iters: 1000, time: 706.5763ms, avg: 0.7066ms
    out_reduce_sum_shared_atomic2_double: passed: 1e-2, 1e-1: 0%, 1e-3: 100%, 1e-6: 100%, max_diff: 0.01192699, iters: 1000, time: 718.2019ms, avg: 0.7182ms
-------------------------------------------------------------------------------------
```

## FAQ
