# cvtColor

## 介绍

包含以下内容：
- [x] bitwise_and_uint8_t
- [x] bitwise_and_u8x4_uint8_t
- [x] bitwise_and_vec_uint8_t
- [x] bitwise_or_uint8_t
- [x] bitwise_xor_uint8_t
- [x] bitwise_not_uint8_t

## 测试

硬件：

- cpu: Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
- gpu: NVIDIA GeForce RTX 4090 D

运行：

```bash
# 只测试Ada架构 不指定默认编译所有架构 耗时较长: Volta, Ampere, Ada, Hopper, ...
export TORCH_CUDA_ARCH_LIST=Ada
python3 bitwise.py
```

输出：

```
-------------------------------------------------------------------------------------
                                        H=1773, W=1367, ch=1
              bitwise_and: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.1820ms, avg: 0.0062ms
         bitwise_and_u8x4: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 3.1867ms, avg: 0.0032ms
          bitwise_and_vec: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 3.1359ms, avg: 0.0031ms
               bitwise_or: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.1808ms, avg: 0.0062ms
              bitwise_xor: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.1808ms, avg: 0.0062ms
              bitwise_not: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.1736ms, avg: 0.0062ms
-------------------------------------------------------------------------------------
                                        H=1773, W=1367, ch=3
              bitwise_and: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 8.0924ms, avg: 0.0081ms
          bitwise_and_vec: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.3867ms, avg: 0.0134ms
               bitwise_or: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 8.0957ms, avg: 0.0081ms
              bitwise_xor: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 8.0938ms, avg: 0.0081ms
              bitwise_not: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.0875ms, avg: 0.0071ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1773, W=4096, ch=1
              bitwise_and: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 15.2433ms, avg: 0.0152ms
         bitwise_and_u8x4: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.1786ms, avg: 0.0062ms
          bitwise_and_vec: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.0658ms, avg: 0.0061ms
               bitwise_or: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 15.2433ms, avg: 0.0152ms
              bitwise_xor: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 15.2438ms, avg: 0.0152ms
              bitwise_not: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 15.2326ms, avg: 0.0152ms
-------------------------------------------------------------------------------------
                                        H=1773, W=4096, ch=3
              bitwise_and: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 21.1897ms, avg: 0.0212ms
          bitwise_and_vec: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 37.2078ms, avg: 0.0372ms
               bitwise_or: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 21.1968ms, avg: 0.0212ms
              bitwise_xor: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 21.1985ms, avg: 0.0212ms
              bitwise_not: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 17.9181ms, avg: 0.0179ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=1367, ch=1
              bitwise_and: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 12.1720ms, avg: 0.0122ms
         bitwise_and_u8x4: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 5.1250ms, avg: 0.0051ms
          bitwise_and_vec: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 5.0137ms, avg: 0.0050ms
               bitwise_or: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 12.1725ms, avg: 0.0122ms
              bitwise_xor: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 12.1739ms, avg: 0.0122ms
              bitwise_not: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 12.1646ms, avg: 0.0122ms
-------------------------------------------------------------------------------------
                                        H=4096, W=1367, ch=3
              bitwise_and: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 16.6836ms, avg: 0.0167ms
          bitwise_and_vec: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 28.8689ms, avg: 0.0289ms
               bitwise_or: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 16.6841ms, avg: 0.0167ms
              bitwise_xor: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 16.6838ms, avg: 0.0167ms
              bitwise_not: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 14.2021ms, avg: 0.0142ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=4096, ch=1
              bitwise_and: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 33.0875ms, avg: 0.0331ms
         bitwise_and_u8x4: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 12.1162ms, avg: 0.0121ms
          bitwise_and_vec: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 11.8272ms, avg: 0.0118ms
               bitwise_or: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 33.0875ms, avg: 0.0331ms
              bitwise_xor: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 33.0889ms, avg: 0.0331ms
              bitwise_not: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 33.0806ms, avg: 0.0331ms
-------------------------------------------------------------------------------------
                                        H=4096, W=4096, ch=3
              bitwise_and: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 162.6024ms, avg: 0.1626ms
          bitwise_and_vec: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 161.9580ms, avg: 0.1620ms
               bitwise_or: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 162.6368ms, avg: 0.1626ms
              bitwise_xor: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 162.5905ms, avg: 0.1626ms
              bitwise_not: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 108.6929ms, avg: 0.1087ms
-------------------------------------------------------------------------------------
```



