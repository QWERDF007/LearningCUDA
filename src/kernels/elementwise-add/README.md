# Elementwise

## 介绍

包含以下内容：
- [x] elementwise_add_f32_kernel
- [x]  elementwise_add_f32x4_kernel (float4向量化版本)
- [x]  elementwise_add_f16_kernel (fp16版本)
- [x]  elementwise_add_f16x2_kernel (fp16向量化版本)
- [x]  elementwise_add_f16x8_kernel (fp16向量化版本)
- [x]  elementwise_add_f16x8_pack_kernel (fp16向量化版本, pack)
- [x] elementwise_add_u8_kernel
- [x] elementwise_add_u8x4_kernel (uchar4向量化版本)
- [x] elementwise_add_u8x4v_kernel (uint32_t向量化, SIMD指令优化版本)
- [x] elementwise_add_u8x16_pack_kernel (uint32_t向量化, SIMD指令优化版本, pack)
- [x]  PyTorch bindings

## 测试

硬件：

- cpu: Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
- gpu: NVIDIA GeForce RTX 4090 D

运行：

```bash
# 只测试Ada架构 不指定默认编译所有架构 耗时较长: Volta, Ampere, Ada, Hopper, ...
export TORCH_CUDA_ARCH_LIST=Ada
python3 elementwise_add.py
```

输出：

```
-------------------------------------------------------------------------------------
                                        H=1024, W=1024
            out_u8: a + b = 12 + 93 = c = 105 in decimal: (1e-8), iters: 1000, time: 3.7346ms, avg: 0.0037ms
          out_u8x4: a + b = 12 + 93 = c = 105 in decimal: (1e-8), iters: 1000, time: 2.6031ms, avg: 0.0026ms
         out_u8x4v: a + b = 12 + 93 = c = 105 in decimal: (1e-8), iters: 1000, time: 2.6002ms, avg: 0.0026ms
     out_u8x16pack: a + b = 12 + 93 = c = 105 in decimal: (1e-8), iters: 1000, time: 2.6057ms, avg: 0.0026ms
      out_u8_torch: a + b = 12 + 93 = c = 105 in decimal: (1e-8), iters: 1000, time: 3.8989ms, avg: 0.0039ms
-------------------------------------------------------------------------------------
           out_f32: a + b = -0.6717 + -0.59 = c = -1.2617 in decimal: (1e-8), iters: 1000, time: 4.0693ms, avg: 0.0041ms
         out_f32x4: a + b = -0.6717 + -0.59 = c = -1.2617 in decimal: (1e-8), iters: 1000, time: 4.2663ms, avg: 0.0043ms
     out_f32_torch: a + b = -0.6717 + -0.59 = c = -1.2617 in decimal: (1e-8), iters: 1000, time: 4.2601ms, avg: 0.0043ms
-------------------------------------------------------------------------------------
           out_f16: a + b = -0.6719 + -0.5898 = c = -1.2617 in decimal: (1e-7), iters: 1000, time: 3.7563ms, avg: 0.0038ms
         out_f16x2: a + b = -0.6719 + -0.5898 = c = -1.2617 in decimal: (1e-7), iters: 1000, time: 2.9137ms, avg: 0.0029ms
         out_f16x8: a + b = -0.6719 + -0.5898 = c = -1.2617 in decimal: (1e-7), iters: 1000, time: 3.3686ms, avg: 0.0034ms
     out_f16x8pack: a + b = -0.6719 + -0.5898 = c = -1.2617 in decimal: (1e-7), iters: 1000, time: 2.7924ms, avg: 0.0028ms
     out_f16_torch: a + b = -0.6719 + -0.5898 = c = -1.2617 in decimal: (1e-7), iters: 1000, time: 3.7689ms, avg: 0.0038ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1024, W=2048
            out_u8: a + b = 100 + 238 = c = 82 in decimal: (1e-8), iters: 1000, time: 6.0003ms, avg: 0.0060ms
          out_u8x4: a + b = 100 + 238 = c = 82 in decimal: (1e-8), iters: 1000, time: 3.6073ms, avg: 0.0036ms
         out_u8x4v: a + b = 100 + 238 = c = 82 in decimal: (1e-8), iters: 1000, time: 3.6056ms, avg: 0.0036ms
     out_u8x16pack: a + b = 100 + 238 = c = 82 in decimal: (1e-8), iters: 1000, time: 3.6075ms, avg: 0.0036ms
      out_u8_torch: a + b = 100 + 238 = c = 82 in decimal: (1e-8), iters: 1000, time: 3.9515ms, avg: 0.0040ms
-------------------------------------------------------------------------------------
           out_f32: a + b = -0.1744 + -1.6864 = c = -1.8608 in decimal: (1e-8), iters: 1000, time: 6.6385ms, avg: 0.0066ms
         out_f32x4: a + b = -0.1744 + -1.6864 = c = -1.8608 in decimal: (1e-8), iters: 1000, time: 6.6156ms, avg: 0.0066ms
     out_f32_torch: a + b = -0.1744 + -1.6864 = c = -1.8608 in decimal: (1e-8), iters: 1000, time: 6.6130ms, avg: 0.0066ms
-------------------------------------------------------------------------------------
           out_f16: a + b = -0.1744 + -1.6865 = c = -1.8613 in decimal: (1e-7), iters: 1000, time: 6.1343ms, avg: 0.0061ms
         out_f16x2: a + b = -0.1744 + -1.6865 = c = -1.8613 in decimal: (1e-7), iters: 1000, time: 4.1411ms, avg: 0.0041ms
         out_f16x8: a + b = -0.1744 + -1.6865 = c = -1.8613 in decimal: (1e-7), iters: 1000, time: 5.6083ms, avg: 0.0056ms
     out_f16x8pack: a + b = -0.1744 + -1.6865 = c = -1.8613 in decimal: (1e-7), iters: 1000, time: 3.9802ms, avg: 0.0040ms
     out_f16_torch: a + b = -0.1744 + -1.6865 = c = -1.8613 in decimal: (1e-7), iters: 1000, time: 3.9861ms, avg: 0.0040ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1024, W=4096
            out_u8: a + b = 100 + 157 = c = 1 in decimal: (1e-8), iters: 1000, time: 10.5083ms, avg: 0.0105ms
          out_u8x4: a + b = 100 + 157 = c = 1 in decimal: (1e-8), iters: 1000, time: 5.7056ms, avg: 0.0057ms
         out_u8x4v: a + b = 100 + 157 = c = 1 in decimal: (1e-8), iters: 1000, time: 5.7070ms, avg: 0.0057ms
     out_u8x16pack: a + b = 100 + 157 = c = 1 in decimal: (1e-8), iters: 1000, time: 5.7309ms, avg: 0.0057ms
      out_u8_torch: a + b = 100 + 157 = c = 1 in decimal: (1e-8), iters: 1000, time: 5.6689ms, avg: 0.0057ms
-------------------------------------------------------------------------------------
           out_f32: a + b = -1.6631 + -1.781 = c = -3.4441 in decimal: (1e-8), iters: 1000, time: 11.7152ms, avg: 0.0117ms
         out_f32x4: a + b = -1.6631 + -1.781 = c = -3.4441 in decimal: (1e-8), iters: 1000, time: 11.4145ms, avg: 0.0114ms
     out_f32_torch: a + b = -1.6631 + -1.781 = c = -3.4441 in decimal: (1e-8), iters: 1000, time: 11.3993ms, avg: 0.0114ms
-------------------------------------------------------------------------------------
           out_f16: a + b = -1.6631 + -1.7812 = c = -3.4453 in decimal: (1e-7), iters: 1000, time: 10.7234ms, avg: 0.0107ms
         out_f16x2: a + b = -1.6631 + -1.7812 = c = -3.4453 in decimal: (1e-7), iters: 1000, time: 6.5563ms, avg: 0.0066ms
         out_f16x8: a + b = -1.6631 + -1.7812 = c = -3.4453 in decimal: (1e-7), iters: 1000, time: 10.3359ms, avg: 0.0103ms
     out_f16x8pack: a + b = -1.6631 + -1.7812 = c = -3.4453 in decimal: (1e-7), iters: 1000, time: 6.3903ms, avg: 0.0064ms
     out_f16_torch: a + b = -1.6631 + -1.7812 = c = -3.4453 in decimal: (1e-7), iters: 1000, time: 6.3777ms, avg: 0.0064ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=1024
            out_u8: a + b = 56 + 6 = c = 62 in decimal: (1e-8), iters: 1000, time: 6.0055ms, avg: 0.0060ms
          out_u8x4: a + b = 56 + 6 = c = 62 in decimal: (1e-8), iters: 1000, time: 3.6068ms, avg: 0.0036ms
         out_u8x4v: a + b = 56 + 6 = c = 62 in decimal: (1e-8), iters: 1000, time: 3.6056ms, avg: 0.0036ms
     out_u8x16pack: a + b = 56 + 6 = c = 62 in decimal: (1e-8), iters: 1000, time: 3.6085ms, avg: 0.0036ms
      out_u8_torch: a + b = 56 + 6 = c = 62 in decimal: (1e-8), iters: 1000, time: 3.9096ms, avg: 0.0039ms
-------------------------------------------------------------------------------------
           out_f32: a + b = -0.83 + -0.803 = c = -1.6331 in decimal: (1e-8), iters: 1000, time: 6.6495ms, avg: 0.0066ms
         out_f32x4: a + b = -0.83 + -0.803 = c = -1.6331 in decimal: (1e-8), iters: 1000, time: 6.6240ms, avg: 0.0066ms
     out_f32_torch: a + b = -0.83 + -0.803 = c = -1.6331 in decimal: (1e-8), iters: 1000, time: 6.6152ms, avg: 0.0066ms
-------------------------------------------------------------------------------------
           out_f16: a + b = -0.8301 + -0.8032 = c = -1.6328 in decimal: (1e-7), iters: 1000, time: 6.1109ms, avg: 0.0061ms
         out_f16x2: a + b = -0.8301 + -0.8032 = c = -1.6328 in decimal: (1e-7), iters: 1000, time: 4.1139ms, avg: 0.0041ms
         out_f16x8: a + b = -0.8301 + -0.8032 = c = -1.6328 in decimal: (1e-7), iters: 1000, time: 5.5821ms, avg: 0.0056ms
     out_f16x8pack: a + b = -0.8301 + -0.8032 = c = -1.6328 in decimal: (1e-7), iters: 1000, time: 3.9816ms, avg: 0.0040ms
     out_f16_torch: a + b = -0.8301 + -0.8032 = c = -1.6328 in decimal: (1e-7), iters: 1000, time: 3.9897ms, avg: 0.0040ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=2048
            out_u8: a + b = 176 + 112 = c = 32 in decimal: (1e-8), iters: 1000, time: 10.5443ms, avg: 0.0105ms
          out_u8x4: a + b = 176 + 112 = c = 32 in decimal: (1e-8), iters: 1000, time: 5.6975ms, avg: 0.0057ms
         out_u8x4v: a + b = 176 + 112 = c = 32 in decimal: (1e-8), iters: 1000, time: 5.6970ms, avg: 0.0057ms
     out_u8x16pack: a + b = 176 + 112 = c = 32 in decimal: (1e-8), iters: 1000, time: 5.7325ms, avg: 0.0057ms
      out_u8_torch: a + b = 176 + 112 = c = 32 in decimal: (1e-8), iters: 1000, time: 5.6667ms, avg: 0.0057ms
-------------------------------------------------------------------------------------
           out_f32: a + b = -0.0827 + 1.1446 = c = 1.0619 in decimal: (1e-8), iters: 1000, time: 11.7137ms, avg: 0.0117ms
         out_f32x4: a + b = -0.0827 + 1.1446 = c = 1.0619 in decimal: (1e-8), iters: 1000, time: 11.4198ms, avg: 0.0114ms
     out_f32_torch: a + b = -0.0827 + 1.1446 = c = 1.0619 in decimal: (1e-8), iters: 1000, time: 11.3933ms, avg: 0.0114ms
-------------------------------------------------------------------------------------
           out_f16: a + b = -0.0827 + 1.1445 = c = 1.0615 in decimal: (1e-7), iters: 1000, time: 10.7210ms, avg: 0.0107ms
         out_f16x2: a + b = -0.0827 + 1.1445 = c = 1.0615 in decimal: (1e-7), iters: 1000, time: 6.5553ms, avg: 0.0066ms
         out_f16x8: a + b = -0.0827 + 1.1445 = c = 1.0615 in decimal: (1e-7), iters: 1000, time: 10.2780ms, avg: 0.0103ms
     out_f16x8pack: a + b = -0.0827 + 1.1445 = c = 1.0615 in decimal: (1e-7), iters: 1000, time: 6.3858ms, avg: 0.0064ms
     out_f16_torch: a + b = -0.0827 + 1.1445 = c = 1.0615 in decimal: (1e-7), iters: 1000, time: 6.3744ms, avg: 0.0064ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=4096
            out_u8: a + b = 79 + 168 = c = 247 in decimal: (1e-8), iters: 1000, time: 19.5112ms, avg: 0.0195ms
          out_u8x4: a + b = 79 + 168 = c = 247 in decimal: (1e-8), iters: 1000, time: 9.8538ms, avg: 0.0099ms
         out_u8x4v: a + b = 79 + 168 = c = 247 in decimal: (1e-8), iters: 1000, time: 9.8298ms, avg: 0.0098ms
     out_u8x16pack: a + b = 79 + 168 = c = 247 in decimal: (1e-8), iters: 1000, time: 9.8705ms, avg: 0.0099ms
      out_u8_torch: a + b = 79 + 168 = c = 247 in decimal: (1e-8), iters: 1000, time: 9.8138ms, avg: 0.0098ms
-------------------------------------------------------------------------------------
           out_f32: a + b = 1.3065 + 1.6185 = c = 2.925 in decimal: (1e-8), iters: 1000, time: 104.7621ms, avg: 0.1048ms
         out_f32x4: a + b = 1.3065 + 1.6185 = c = 2.925 in decimal: (1e-8), iters: 1000, time: 105.5467ms, avg: 0.1055ms
     out_f32_torch: a + b = 1.3065 + 1.6185 = c = 2.925 in decimal: (1e-8), iters: 1000, time: 105.4876ms, avg: 0.1055ms
-------------------------------------------------------------------------------------
           out_f16: a + b = 1.3066 + 1.6182 = c = 2.9258 in decimal: (1e-7), iters: 1000, time: 19.1398ms, avg: 0.0191ms
         out_f16x2: a + b = 1.3066 + 1.6182 = c = 2.9258 in decimal: (1e-7), iters: 1000, time: 10.9715ms, avg: 0.0110ms
         out_f16x8: a + b = 1.3066 + 1.6182 = c = 2.9258 in decimal: (1e-7), iters: 1000, time: 19.9537ms, avg: 0.0200ms
     out_f16x8pack: a + b = 1.3066 + 1.6182 = c = 2.9258 in decimal: (1e-7), iters: 1000, time: 11.2245ms, avg: 0.0112ms
     out_f16_torch: a + b = 1.3066 + 1.6182 = c = 2.9258 in decimal: (1e-7), iters: 1000, time: 11.1990ms, avg: 0.0112ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=1024
            out_u8: a + b = 196 + 205 = c = 145 in decimal: (1e-8), iters: 1000, time: 10.4630ms, avg: 0.0105ms
          out_u8x4: a + b = 196 + 205 = c = 145 in decimal: (1e-8), iters: 1000, time: 5.7018ms, avg: 0.0057ms
         out_u8x4v: a + b = 196 + 205 = c = 145 in decimal: (1e-8), iters: 1000, time: 5.6987ms, avg: 0.0057ms
     out_u8x16pack: a + b = 196 + 205 = c = 145 in decimal: (1e-8), iters: 1000, time: 5.7333ms, avg: 0.0057ms
      out_u8_torch: a + b = 196 + 205 = c = 145 in decimal: (1e-8), iters: 1000, time: 5.6729ms, avg: 0.0057ms
-------------------------------------------------------------------------------------
           out_f32: a + b = 1.4847 + 0.5894 = c = 2.0741 in decimal: (1e-8), iters: 1000, time: 11.7013ms, avg: 0.0117ms
         out_f32x4: a + b = 1.4847 + 0.5894 = c = 2.0741 in decimal: (1e-8), iters: 1000, time: 11.3933ms, avg: 0.0114ms
     out_f32_torch: a + b = 1.4847 + 0.5894 = c = 2.0741 in decimal: (1e-8), iters: 1000, time: 11.3838ms, avg: 0.0114ms
-------------------------------------------------------------------------------------
           out_f16: a + b = 1.4844 + 0.5894 = c = 2.0742 in decimal: (1e-7), iters: 1000, time: 10.7245ms, avg: 0.0107ms
         out_f16x2: a + b = 1.4844 + 0.5894 = c = 2.0742 in decimal: (1e-7), iters: 1000, time: 6.5742ms, avg: 0.0066ms
         out_f16x8: a + b = 1.4844 + 0.5894 = c = 2.0742 in decimal: (1e-7), iters: 1000, time: 10.3838ms, avg: 0.0104ms
     out_f16x8pack: a + b = 1.4844 + 0.5894 = c = 2.0742 in decimal: (1e-7), iters: 1000, time: 6.4187ms, avg: 0.0064ms
     out_f16_torch: a + b = 1.4844 + 0.5894 = c = 2.0742 in decimal: (1e-7), iters: 1000, time: 6.3841ms, avg: 0.0064ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=2048
            out_u8: a + b = 56 + 108 = c = 164 in decimal: (1e-8), iters: 1000, time: 19.5229ms, avg: 0.0195ms
          out_u8x4: a + b = 56 + 108 = c = 164 in decimal: (1e-8), iters: 1000, time: 9.8517ms, avg: 0.0099ms
         out_u8x4v: a + b = 56 + 108 = c = 164 in decimal: (1e-8), iters: 1000, time: 9.8226ms, avg: 0.0098ms
     out_u8x16pack: a + b = 56 + 108 = c = 164 in decimal: (1e-8), iters: 1000, time: 9.8722ms, avg: 0.0099ms
      out_u8_torch: a + b = 56 + 108 = c = 164 in decimal: (1e-8), iters: 1000, time: 9.8267ms, avg: 0.0098ms
-------------------------------------------------------------------------------------
           out_f32: a + b = 1.3343 + -0.6276 = c = 0.7067 in decimal: (1e-8), iters: 1000, time: 104.7757ms, avg: 0.1048ms
         out_f32x4: a + b = 1.3343 + -0.6276 = c = 0.7067 in decimal: (1e-8), iters: 1000, time: 105.5374ms, avg: 0.1055ms
     out_f32_torch: a + b = 1.3343 + -0.6276 = c = 0.7067 in decimal: (1e-8), iters: 1000, time: 105.5188ms, avg: 0.1055ms
-------------------------------------------------------------------------------------
           out_f16: a + b = 1.334 + -0.6274 = c = 0.7065 in decimal: (1e-7), iters: 1000, time: 19.1293ms, avg: 0.0191ms
         out_f16x2: a + b = 1.334 + -0.6274 = c = 0.7065 in decimal: (1e-7), iters: 1000, time: 10.9847ms, avg: 0.0110ms
         out_f16x8: a + b = 1.334 + -0.6274 = c = 0.7065 in decimal: (1e-7), iters: 1000, time: 19.8286ms, avg: 0.0198ms
     out_f16x8pack: a + b = 1.334 + -0.6274 = c = 0.7065 in decimal: (1e-7), iters: 1000, time: 11.2231ms, avg: 0.0112ms
     out_f16_torch: a + b = 1.334 + -0.6274 = c = 0.7065 in decimal: (1e-7), iters: 1000, time: 11.2069ms, avg: 0.0112ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=4096
            out_u8: a + b = 246 + 18 = c = 8 in decimal: (1e-8), iters: 1000, time: 37.5638ms, avg: 0.0376ms
          out_u8x4: a + b = 246 + 18 = c = 8 in decimal: (1e-8), iters: 1000, time: 18.1212ms, avg: 0.0181ms
         out_u8x4v: a + b = 246 + 18 = c = 8 in decimal: (1e-8), iters: 1000, time: 18.1236ms, avg: 0.0181ms
     out_u8x16pack: a + b = 246 + 18 = c = 8 in decimal: (1e-8), iters: 1000, time: 18.1458ms, avg: 0.0181ms
      out_u8_torch: a + b = 246 + 18 = c = 8 in decimal: (1e-8), iters: 1000, time: 18.1103ms, avg: 0.0181ms
-------------------------------------------------------------------------------------
           out_f32: a + b = 1.8163 + -0.5666 = c = 1.2497 in decimal: (1e-8), iters: 1000, time: 215.7533ms, avg: 0.2158ms
         out_f32x4: a + b = 1.8163 + -0.5666 = c = 1.2497 in decimal: (1e-8), iters: 1000, time: 216.4311ms, avg: 0.2164ms
     out_f32_torch: a + b = 1.8163 + -0.5666 = c = 1.2497 in decimal: (1e-8), iters: 1000, time: 216.4567ms, avg: 0.2165ms
-------------------------------------------------------------------------------------
           out_f16: a + b = 1.8164 + -0.5664 = c = 1.25 in decimal: (1e-7), iters: 1000, time: 106.3800ms, avg: 0.1064ms
         out_f16x2: a + b = 1.8164 + -0.5664 = c = 1.25 in decimal: (1e-7), iters: 1000, time: 104.8839ms, avg: 0.1049ms
         out_f16x8: a + b = 1.8164 + -0.5664 = c = 1.25 in decimal: (1e-7), iters: 1000, time: 106.2915ms, avg: 0.1063ms
     out_f16x8pack: a + b = 1.8164 + -0.5664 = c = 1.25 in decimal: (1e-7), iters: 1000, time: 105.5863ms, avg: 0.1056ms
     out_f16_torch: a + b = 1.8164 + -0.5664 = c = 1.25 in decimal: (1e-7), iters: 1000, time: 105.4378ms, avg: 0.1054ms
-------------------------------------------------------------------------------------
```



