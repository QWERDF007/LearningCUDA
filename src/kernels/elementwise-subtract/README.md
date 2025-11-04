# Elementwise

## 介绍

包含以下内容：
- [x] elementwise_sub_kernel
- [x] elementwise_sub_2D_kernel (2D 布局版本)
- [x] elementwise_sub_32bit_kernel (向量化版本)
- [x] elementwise_sub_32bit_2D_kernel (2D 布局向量化版本)
- [x] elementwise_sub_128bit_kernel (向量化版本)
- [x] elementwise_sub_128bit_2D_kernel (2D 布局向量化版本)
- [x] PyTorch bindings

## 测试

硬件：

- cpu: Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
- gpu: NVIDIA GeForce RTX 4090 D

运行：

```bash
# 只测试Ada架构 不指定默认编译所有架构 耗时较长: Volta, Ampere, Ada, Hopper, ...
export TORCH_CUDA_ARCH_LIST=Ada
python3 elementwise_sub.py
```

输出：
```
-------------------------------------------------------------------------------------
                                        H=1024, W=1024
             torch_sub_u8: passed: 1e3, 1e-1: 49.7%, 1e-3: 49.7%, 1e-6: 49.7%, max_diff: 128.0, iters: 1000, time: 3.9382ms, avg: 0.0039ms
                   sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 3.8688ms, avg: 0.0039ms
                sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.7746ms, avg: 0.0048ms
             32bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 3.6814ms, avg: 0.0037ms
            128bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 3.7158ms, avg: 0.0037ms
          32bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 2.9027ms, avg: 0.0029ms
         128bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 2.5597ms, avg: 0.0026ms
-------------------------------------------------------------------------------------
            torch_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.3080ms, avg: 0.0043ms
                  sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.2131ms, avg: 0.0042ms
           128bit_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.0729ms, avg: 0.0041ms
        128bit_sub_2D_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.4825ms, avg: 0.0045ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1024, W=2048
             torch_sub_u8: passed: 1e3, 1e-1: 49.8%, 1e-3: 49.8%, 1e-6: 49.8%, max_diff: 128.0, iters: 1000, time: 3.8769ms, avg: 0.0039ms
                   sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 5.9927ms, avg: 0.0060ms
                sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.9994ms, avg: 0.0080ms
             32bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 5.6987ms, avg: 0.0057ms
            128bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 5.8203ms, avg: 0.0058ms
          32bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.0333ms, avg: 0.0040ms
         128bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 3.7971ms, avg: 0.0038ms
-------------------------------------------------------------------------------------
            torch_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.6757ms, avg: 0.0067ms
                  sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.6187ms, avg: 0.0066ms
           128bit_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.4461ms, avg: 0.0064ms
        128bit_sub_2D_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.6936ms, avg: 0.0067ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1024, W=4096
             torch_sub_u8: passed: 1e3, 1e-1: 49.8%, 1e-3: 49.8%, 1e-6: 49.8%, max_diff: 128.0, iters: 1000, time: 5.7502ms, avg: 0.0058ms
                   sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 10.1306ms, avg: 0.0101ms
                sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 14.2612ms, avg: 0.0143ms
             32bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 9.8374ms, avg: 0.0098ms
            128bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 9.9647ms, avg: 0.0100ms
          32bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.3269ms, avg: 0.0063ms
         128bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 5.3113ms, avg: 0.0053ms
-------------------------------------------------------------------------------------
            torch_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 11.4923ms, avg: 0.0115ms
                  sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 11.5132ms, avg: 0.0115ms
           128bit_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 11.2867ms, avg: 0.0113ms
        128bit_sub_2D_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 11.5464ms, avg: 0.0115ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1024, W=8192
             torch_sub_u8: passed: 1e3, 1e-1: 49.8%, 1e-3: 49.8%, 1e-6: 49.8%, max_diff: 128.0, iters: 1000, time: 9.8619ms, avg: 0.0099ms
                   sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 18.3792ms, avg: 0.0184ms
                sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 26.9237ms, avg: 0.0269ms
             32bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 18.0836ms, avg: 0.0181ms
            128bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 18.2319ms, avg: 0.0182ms
          32bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 10.9346ms, avg: 0.0109ms
         128bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 8.1799ms, avg: 0.0082ms
-------------------------------------------------------------------------------------
            torch_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 108.0005ms, avg: 0.1080ms
                  sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 101.0511ms, avg: 0.1011ms
           128bit_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 104.3441ms, avg: 0.1043ms
        128bit_sub_2D_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 106.2324ms, avg: 0.1062ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=1024
             torch_sub_u8: passed: 1e3, 1e-1: 49.9%, 1e-3: 49.9%, 1e-6: 49.9%, max_diff: 128.0, iters: 1000, time: 3.7117ms, avg: 0.0037ms
                   sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 5.7063ms, avg: 0.0057ms
                sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.5850ms, avg: 0.0076ms
             32bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 5.4140ms, avg: 0.0054ms
            128bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 5.5461ms, avg: 0.0055ms
          32bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 3.8295ms, avg: 0.0038ms
         128bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 3.7141ms, avg: 0.0037ms
-------------------------------------------------------------------------------------
            torch_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.3424ms, avg: 0.0063ms
                  sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.2931ms, avg: 0.0063ms
           128bit_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.1195ms, avg: 0.0061ms
        128bit_sub_2D_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.3808ms, avg: 0.0064ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=2048
             torch_sub_u8: passed: 1e3, 1e-1: 49.8%, 1e-3: 49.8%, 1e-6: 49.8%, max_diff: 128.0, iters: 1000, time: 5.4431ms, avg: 0.0054ms
                   sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 9.6383ms, avg: 0.0096ms
                sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.5880ms, avg: 0.0136ms
             32bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 9.3348ms, avg: 0.0093ms
            128bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 9.9785ms, avg: 0.0100ms
          32bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.3362ms, avg: 0.0063ms
         128bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 5.1992ms, avg: 0.0052ms
-------------------------------------------------------------------------------------
            torch_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 11.4834ms, avg: 0.0115ms
                  sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 11.5139ms, avg: 0.0115ms
           128bit_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 11.2832ms, avg: 0.0113ms
        128bit_sub_2D_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 11.5159ms, avg: 0.0115ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=4096
             torch_sub_u8: passed: 1e3, 1e-1: 49.8%, 1e-3: 49.8%, 1e-6: 49.8%, max_diff: 128.0, iters: 1000, time: 9.8605ms, avg: 0.0099ms
                   sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 18.3773ms, avg: 0.0184ms
                sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 26.9299ms, avg: 0.0269ms
             32bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 18.0852ms, avg: 0.0181ms
            128bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 18.2314ms, avg: 0.0182ms
          32bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 10.9525ms, avg: 0.0110ms
         128bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 8.1806ms, avg: 0.0082ms
-------------------------------------------------------------------------------------
            torch_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 105.8123ms, avg: 0.1058ms
                  sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 105.5770ms, avg: 0.1056ms
           128bit_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 105.3030ms, avg: 0.1053ms
        128bit_sub_2D_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 106.0112ms, avg: 0.1060ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=8192
             torch_sub_u8: passed: 1e3, 1e-1: 49.8%, 1e-3: 49.8%, 1e-6: 49.8%, max_diff: 128.0, iters: 1000, time: 17.2253ms, avg: 0.0172ms
                   sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 34.8835ms, avg: 0.0349ms
                sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 52.2749ms, avg: 0.0523ms
             32bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 34.7044ms, avg: 0.0347ms
            128bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 34.7476ms, avg: 0.0347ms
          32bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 20.2487ms, avg: 0.0202ms
         128bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.9043ms, avg: 0.0139ms
-------------------------------------------------------------------------------------
            torch_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 216.5537ms, avg: 0.2166ms
                  sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 215.7469ms, avg: 0.2157ms
           128bit_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 216.3761ms, avg: 0.2164ms
        128bit_sub_2D_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 216.3591ms, avg: 0.2164ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=1024
             torch_sub_u8: passed: 1e3, 1e-1: 49.8%, 1e-3: 49.8%, 1e-6: 49.8%, max_diff: 128.0, iters: 1000, time: 5.4634ms, avg: 0.0055ms
                   sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 9.6784ms, avg: 0.0097ms
                sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.6068ms, avg: 0.0136ms
             32bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 9.3839ms, avg: 0.0094ms
            128bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 9.5270ms, avg: 0.0095ms
          32bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.0465ms, avg: 0.0060ms
         128bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 5.0638ms, avg: 0.0051ms
-------------------------------------------------------------------------------------
            torch_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 10.9460ms, avg: 0.0109ms
                  sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 10.9978ms, avg: 0.0110ms
           128bit_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 10.7594ms, avg: 0.0108ms
        128bit_sub_2D_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 11.0033ms, avg: 0.0110ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=2048
             torch_sub_u8: passed: 1e3, 1e-1: 49.8%, 1e-3: 49.8%, 1e-6: 49.8%, max_diff: 128.0, iters: 1000, time: 9.6264ms, avg: 0.0096ms
                   sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 18.3768ms, avg: 0.0184ms
                sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 26.9237ms, avg: 0.0269ms
             32bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 18.0805ms, avg: 0.0181ms
            128bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 18.2292ms, avg: 0.0182ms
          32bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 11.0724ms, avg: 0.0111ms
         128bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 8.1806ms, avg: 0.0082ms
-------------------------------------------------------------------------------------
            torch_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 108.0630ms, avg: 0.1081ms
                  sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 101.0902ms, avg: 0.1011ms
           128bit_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 104.4228ms, avg: 0.1044ms
        128bit_sub_2D_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 102.2644ms, avg: 0.1023ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=4096
             torch_sub_u8: passed: 1e3, 1e-1: 49.8%, 1e-3: 49.8%, 1e-6: 49.8%, max_diff: 128.0, iters: 1000, time: 17.3194ms, avg: 0.0173ms
                   sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 34.8926ms, avg: 0.0349ms
                sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 52.4583ms, avg: 0.0525ms
             32bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 34.7061ms, avg: 0.0347ms
            128bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 34.7621ms, avg: 0.0348ms
          32bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 20.3958ms, avg: 0.0204ms
         128bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.9773ms, avg: 0.0140ms
-------------------------------------------------------------------------------------
            torch_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 216.5146ms, avg: 0.2165ms
                  sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 215.8043ms, avg: 0.2158ms
           128bit_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 216.6514ms, avg: 0.2167ms
        128bit_sub_2D_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 216.1620ms, avg: 0.2162ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=8192
             torch_sub_u8: passed: 1e3, 1e-1: 49.8%, 1e-3: 49.8%, 1e-6: 49.8%, max_diff: 128.0, iters: 1000, time: 101.2130ms, avg: 0.1012ms
                   sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 129.0348ms, avg: 0.1290ms
                sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 191.2913ms, avg: 0.1913ms
             32bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 107.3418ms, avg: 0.1073ms
            128bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 101.2166ms, avg: 0.1012ms
          32bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 108.4826ms, avg: 0.1085ms
         128bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 102.3328ms, avg: 0.1023ms
-------------------------------------------------------------------------------------
            torch_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 432.4727ms, avg: 0.4325ms
                  sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 431.4590ms, avg: 0.4315ms
           128bit_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 432.2741ms, avg: 0.4323ms
        128bit_sub_2D_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 433.0704ms, avg: 0.4331ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=8192, W=1024
             torch_sub_u8: passed: 1e3, 1e-1: 49.8%, 1e-3: 49.8%, 1e-6: 49.8%, max_diff: 128.0, iters: 1000, time: 9.4230ms, avg: 0.0094ms
                   sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 18.3790ms, avg: 0.0184ms
                sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 26.9201ms, avg: 0.0269ms
             32bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 18.0805ms, avg: 0.0181ms
            128bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 18.2304ms, avg: 0.0182ms
          32bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 10.9944ms, avg: 0.0110ms
         128bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 8.2188ms, avg: 0.0082ms
-------------------------------------------------------------------------------------
            torch_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 108.0637ms, avg: 0.1081ms
                  sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 101.1248ms, avg: 0.1011ms
           128bit_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 104.1875ms, avg: 0.1042ms
        128bit_sub_2D_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 102.2749ms, avg: 0.1023ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=8192, W=2048
             torch_sub_u8: passed: 1e3, 1e-1: 49.8%, 1e-3: 49.8%, 1e-6: 49.8%, max_diff: 128.0, iters: 1000, time: 17.3140ms, avg: 0.0173ms
                   sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 34.9071ms, avg: 0.0349ms
                sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 52.3119ms, avg: 0.0523ms
             32bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 34.7016ms, avg: 0.0347ms
            128bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 34.7466ms, avg: 0.0347ms
          32bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 20.2658ms, avg: 0.0203ms
         128bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.8414ms, avg: 0.0138ms
-------------------------------------------------------------------------------------
            torch_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 216.7540ms, avg: 0.2168ms
                  sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 215.8005ms, avg: 0.2158ms
           128bit_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 216.6193ms, avg: 0.2166ms
        128bit_sub_2D_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 216.1632ms, avg: 0.2162ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=8192, W=4096
             torch_sub_u8: passed: 1e3, 1e-1: 49.8%, 1e-3: 49.8%, 1e-6: 49.8%, max_diff: 128.0, iters: 1000, time: 101.1381ms, avg: 0.1011ms
                   sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 129.0588ms, avg: 0.1291ms
                sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 188.9727ms, avg: 0.1890ms
             32bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 107.3093ms, avg: 0.1073ms
            128bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 101.2211ms, avg: 0.1012ms
          32bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 108.4359ms, avg: 0.1084ms
         128bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 102.4473ms, avg: 0.1024ms
-------------------------------------------------------------------------------------
            torch_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 432.2491ms, avg: 0.4322ms
                  sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 432.1249ms, avg: 0.4321ms
           128bit_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 432.7149ms, avg: 0.4327ms
        128bit_sub_2D_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 431.7672ms, avg: 0.4318ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=8192, W=8192
             torch_sub_u8: passed: 1e3, 1e-1: 49.8%, 1e-3: 49.8%, 1e-6: 49.8%, max_diff: 128.0, iters: 1000, time: 215.9348ms, avg: 0.2159ms
                   sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 254.4580ms, avg: 0.2545ms
                sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 382.0610ms, avg: 0.3821ms
             32bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 216.0189ms, avg: 0.2160ms
            128bit_sub_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 215.9104ms, avg: 0.2159ms
          32bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 216.1524ms, avg: 0.2162ms
         128bit_sub_2D_u8: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 216.1291ms, avg: 0.2161ms
-------------------------------------------------------------------------------------
            torch_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 868.0992ms, avg: 0.8681ms
                  sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 863.0035ms, avg: 0.8630ms
           128bit_sub_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 868.0401ms, avg: 0.8680ms
        128bit_sub_2D_f32: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 864.2673ms, avg: 0.8643ms
-------------------------------------------------------------------------------------
```