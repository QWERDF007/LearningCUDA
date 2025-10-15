# cvtColor

## 介绍

包含以下内容：
- [x] BGR2RGB, BGR2RGBA
- [x] BGR2Gray, Gray2BGR, Gray2BGRA
- [x] BGR2YCrCb, YCrCb2BGR
- [x] BGR2YUV, YUV2BGR
- [x] BGR2XYZ, XZY2BGR
- [x] BGR2HSV, HSV2BGR
- [x] BGR2HLS, HLS2BGR
- [x] BGR2Lab

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
-------------------------------------------------------------------------------------
                                        H=1024, W=1024
   COLOR_GRAY2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 5.9047ms, avg: 0.0059ms
  COLOR_GRAY2BGRA [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.3686ms, avg: 0.0074ms
    COLOR_BGR2RGB [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.3407ms, avg: 0.0063ms
  COLOR_BGR2YCrCb [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.6866ms, avg: 0.0047ms
  COLOR_RGB2YCrCb [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.6840ms, avg: 0.0047ms
    COLOR_BGR2YUV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.6823ms, avg: 0.0047ms
    COLOR_RGB2YUV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.6835ms, avg: 0.0047ms
    COLOR_BGR2HSV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 43.2279ms, avg: 0.0432ms
    COLOR_HSV2BGR [uint8]: passed: 1e0, 1e-1: 33.3%, 1e-3: 33.3%, 1e-6: 33.3%, max_diff: 1.0, iters: 1000, time: 6.5918ms, avg: 0.0066ms
    COLOR_RGB2HSV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 43.2281ms, avg: 0.0432ms
    COLOR_BGR2HLS [uint8]: passed: 1e0, 1e-1: 0.0501%, 1e-3: 0.0501%, 1e-6: 0.0501%, max_diff: 1.0, iters: 1000, time: 5.4967ms, avg: 0.0055ms
    COLOR_HLS2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.3002ms, avg: 0.0063ms
    COLOR_RGB2HLS [uint8]: passed: 1e0, 1e-1: 0.0499%, 1e-3: 0.0499%, 1e-6: 0.0499%, max_diff: 1.0, iters: 1000, time: 5.4979ms, avg: 0.0055ms
    COLOR_BGR2XYZ [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.6704ms, avg: 0.0047ms
    COLOR_XYZ2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.6716ms, avg: 0.0047ms
    COLOR_RGB2XYZ [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.6687ms, avg: 0.0047ms
  COLOR_YCrCb2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.5412ms, avg: 0.0045ms
    COLOR_YUV2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.5414ms, avg: 0.0045ms
    COLOR_BGR2Lab [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 8.7149ms, avg: 0.0087ms
   COLOR_BGR2RGBA [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.2987ms, avg: 0.0073ms
   COLOR_BGR2BGRA [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.2980ms, avg: 0.0073ms
   COLOR_BGR2GRAY [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.2431ms, avg: 0.0042ms
   COLOR_RGB2GRAY [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.2427ms, avg: 0.0042ms
 COLOR_GRAY2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 19.8934ms, avg: 0.0199ms
COLOR_GRAY2BGRA [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 28.1909ms, avg: 0.0282ms
  COLOR_BGR2RGB [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 20.7844ms, avg: 0.0208ms
COLOR_BGR2YCrCb [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 5.9604645e-08, iters: 1000, time: 7.5784ms, avg: 0.0076ms
COLOR_RGB2YCrCb [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 5.9604645e-08, iters: 1000, time: 7.5788ms, avg: 0.0076ms
  COLOR_BGR2YUV [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 7.5762ms, avg: 0.0076ms
  COLOR_RGB2YUV [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 7.5784ms, avg: 0.0076ms
  COLOR_BGR2HSV [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.59%, max_diff: 3.0517578e-05, iters: 1000, time: 7.7670ms, avg: 0.0078ms
  COLOR_HSV2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 8.1091ms, avg: 0.0081ms
  COLOR_RGB2HSV [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.59%, max_diff: 3.0517578e-05, iters: 1000, time: 7.7817ms, avg: 0.0078ms
  COLOR_BGR2HLS [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.58%, max_diff: 3.0517578e-05, iters: 1000, time: 7.6389ms, avg: 0.0076ms
  COLOR_HLS2BGR [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 8.0411ms, avg: 0.0080ms
  COLOR_RGB2HLS [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.58%, max_diff: 3.0517578e-05, iters: 1000, time: 7.6394ms, avg: 0.0076ms
  COLOR_BGR2XYZ [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 8.0168ms, avg: 0.0080ms
  COLOR_XYZ2BGR [float32]: passed: 1e-6, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 4.7683716e-07, iters: 1000, time: 8.0123ms, avg: 0.0080ms
  COLOR_RGB2XYZ [float32]: passed: 1e-6, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 2.3841858e-07, iters: 1000, time: 8.0163ms, avg: 0.0080ms
COLOR_YCrCb2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.6210ms, avg: 0.0076ms
  COLOR_YUV2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.6199ms, avg: 0.0076ms
 COLOR_BGR2RGBA [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 29.2895ms, avg: 0.0293ms
 COLOR_BGR2GRAY [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.5910ms, avg: 0.0046ms
 COLOR_RGB2GRAY [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.5912ms, avg: 0.0046ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1024, W=2048
   COLOR_GRAY2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 10.0822ms, avg: 0.0101ms
  COLOR_GRAY2BGRA [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 12.9485ms, avg: 0.0129ms
    COLOR_BGR2RGB [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 10.9394ms, avg: 0.0109ms
  COLOR_BGR2YCrCb [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.7977ms, avg: 0.0078ms
  COLOR_RGB2YCrCb [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.8132ms, avg: 0.0078ms
    COLOR_BGR2YUV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.7939ms, avg: 0.0078ms
    COLOR_RGB2YUV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.8135ms, avg: 0.0078ms
    COLOR_BGR2HSV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 84.7197ms, avg: 0.0847ms
    COLOR_HSV2BGR [uint8]: passed: 1e0, 1e-1: 33.4%, 1e-3: 33.4%, 1e-6: 33.4%, max_diff: 1.0, iters: 1000, time: 11.4300ms, avg: 0.0114ms
    COLOR_RGB2HSV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 84.7237ms, avg: 0.0847ms
    COLOR_BGR2HLS [uint8]: passed: 1e0, 1e-1: 0.0505%, 1e-3: 0.0505%, 1e-6: 0.0505%, max_diff: 1.0, iters: 1000, time: 9.3682ms, avg: 0.0094ms
    COLOR_HLS2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 10.7739ms, avg: 0.0108ms
    COLOR_RGB2HLS [uint8]: passed: 1e0, 1e-1: 0.05%, 1e-3: 0.05%, 1e-6: 0.05%, max_diff: 1.0, iters: 1000, time: 9.3684ms, avg: 0.0094ms
    COLOR_BGR2XYZ [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.5965ms, avg: 0.0076ms
    COLOR_XYZ2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.6070ms, avg: 0.0076ms
    COLOR_RGB2XYZ [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.5943ms, avg: 0.0076ms
  COLOR_YCrCb2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.4277ms, avg: 0.0074ms
    COLOR_YUV2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.4284ms, avg: 0.0074ms
    COLOR_BGR2Lab [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 15.6009ms, avg: 0.0156ms
   COLOR_BGR2RGBA [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 12.5813ms, avg: 0.0126ms
   COLOR_BGR2BGRA [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 12.5878ms, avg: 0.0126ms
   COLOR_BGR2GRAY [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.7360ms, avg: 0.0067ms
   COLOR_RGB2GRAY [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.7375ms, avg: 0.0067ms
 COLOR_GRAY2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 38.2650ms, avg: 0.0383ms
COLOR_GRAY2BGRA [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 54.5993ms, avg: 0.0546ms
  COLOR_BGR2RGB [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 40.1952ms, avg: 0.0402ms
COLOR_BGR2YCrCb [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 5.9604645e-08, iters: 1000, time: 13.5176ms, avg: 0.0135ms
COLOR_RGB2YCrCb [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 5.9604645e-08, iters: 1000, time: 13.5181ms, avg: 0.0135ms
  COLOR_BGR2YUV [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 13.5176ms, avg: 0.0135ms
  COLOR_RGB2YUV [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 13.5181ms, avg: 0.0135ms
  COLOR_BGR2HSV [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.59%, max_diff: 3.0517578e-05, iters: 1000, time: 13.6499ms, avg: 0.0136ms
  COLOR_HSV2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 14.1506ms, avg: 0.0142ms
  COLOR_RGB2HSV [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.59%, max_diff: 3.0517578e-05, iters: 1000, time: 13.8896ms, avg: 0.0139ms
  COLOR_BGR2HLS [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.6%, max_diff: 3.0517578e-05, iters: 1000, time: 13.5913ms, avg: 0.0136ms
  COLOR_HLS2BGR [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 14.0536ms, avg: 0.0141ms
  COLOR_RGB2HLS [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.58%, max_diff: 3.0517578e-05, iters: 1000, time: 13.5925ms, avg: 0.0136ms
  COLOR_BGR2XYZ [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 14.0212ms, avg: 0.0140ms
  COLOR_XYZ2BGR [float32]: passed: 1e-6, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 4.7683716e-07, iters: 1000, time: 14.0224ms, avg: 0.0140ms
  COLOR_RGB2XYZ [float32]: passed: 1e-6, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 2.3841858e-07, iters: 1000, time: 14.0264ms, avg: 0.0140ms
COLOR_YCrCb2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.6240ms, avg: 0.0136ms
  COLOR_YUV2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.6249ms, avg: 0.0136ms
 COLOR_BGR2RGBA [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 56.7737ms, avg: 0.0568ms
 COLOR_BGR2GRAY [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.4897ms, avg: 0.0075ms
 COLOR_RGB2GRAY [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.4897ms, avg: 0.0075ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=1024
   COLOR_GRAY2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 10.0808ms, avg: 0.0101ms
  COLOR_GRAY2BGRA [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 12.9535ms, avg: 0.0130ms
    COLOR_BGR2RGB [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 10.9439ms, avg: 0.0109ms
  COLOR_BGR2YCrCb [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.7899ms, avg: 0.0078ms
  COLOR_RGB2YCrCb [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.8118ms, avg: 0.0078ms
    COLOR_BGR2YUV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.7903ms, avg: 0.0078ms
    COLOR_RGB2YUV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.8115ms, avg: 0.0078ms
    COLOR_BGR2HSV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 84.6937ms, avg: 0.0847ms
    COLOR_HSV2BGR [uint8]: passed: 1e0, 1e-1: 33.4%, 1e-3: 33.4%, 1e-6: 33.4%, max_diff: 1.0, iters: 1000, time: 11.4329ms, avg: 0.0114ms
    COLOR_RGB2HSV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 84.6975ms, avg: 0.0847ms
    COLOR_BGR2HLS [uint8]: passed: 1e0, 1e-1: 0.0495%, 1e-3: 0.0495%, 1e-6: 0.0495%, max_diff: 1.0, iters: 1000, time: 9.3725ms, avg: 0.0094ms
    COLOR_HLS2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 10.7760ms, avg: 0.0108ms
    COLOR_RGB2HLS [uint8]: passed: 1e0, 1e-1: 0.0494%, 1e-3: 0.0494%, 1e-6: 0.0494%, max_diff: 1.0, iters: 1000, time: 9.3763ms, avg: 0.0094ms
    COLOR_BGR2XYZ [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.6039ms, avg: 0.0076ms
    COLOR_XYZ2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.6067ms, avg: 0.0076ms
    COLOR_RGB2XYZ [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.5998ms, avg: 0.0076ms
  COLOR_YCrCb2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.4263ms, avg: 0.0074ms
    COLOR_YUV2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.4289ms, avg: 0.0074ms
    COLOR_BGR2Lab [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 15.5997ms, avg: 0.0156ms
   COLOR_BGR2RGBA [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 12.5909ms, avg: 0.0126ms
   COLOR_BGR2BGRA [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 12.5952ms, avg: 0.0126ms
   COLOR_BGR2GRAY [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.7353ms, avg: 0.0067ms
   COLOR_RGB2GRAY [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.7358ms, avg: 0.0067ms
 COLOR_GRAY2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 38.2545ms, avg: 0.0383ms
COLOR_GRAY2BGRA [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 54.5976ms, avg: 0.0546ms
  COLOR_BGR2RGB [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 40.2150ms, avg: 0.0402ms
COLOR_BGR2YCrCb [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 5.9604645e-08, iters: 1000, time: 13.5193ms, avg: 0.0135ms
COLOR_RGB2YCrCb [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 5.9604645e-08, iters: 1000, time: 13.5193ms, avg: 0.0135ms
  COLOR_BGR2YUV [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 13.5226ms, avg: 0.0135ms
  COLOR_RGB2YUV [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 13.5233ms, avg: 0.0135ms
  COLOR_BGR2HSV [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.59%, max_diff: 3.0517578e-05, iters: 1000, time: 13.6518ms, avg: 0.0137ms
  COLOR_HSV2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 14.1644ms, avg: 0.0142ms
  COLOR_RGB2HSV [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.59%, max_diff: 3.0517578e-05, iters: 1000, time: 13.8965ms, avg: 0.0139ms
  COLOR_BGR2HLS [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.6%, max_diff: 3.0517578e-05, iters: 1000, time: 13.5939ms, avg: 0.0136ms
  COLOR_HLS2BGR [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 14.0555ms, avg: 0.0141ms
  COLOR_RGB2HLS [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.58%, max_diff: 3.0517578e-05, iters: 1000, time: 13.5958ms, avg: 0.0136ms
  COLOR_BGR2XYZ [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 14.0374ms, avg: 0.0140ms
  COLOR_XYZ2BGR [float32]: passed: 1e-6, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 4.7683716e-07, iters: 1000, time: 14.0338ms, avg: 0.0140ms
  COLOR_RGB2XYZ [float32]: passed: 1e-6, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 2.3841858e-07, iters: 1000, time: 14.0569ms, avg: 0.0141ms
COLOR_YCrCb2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.6306ms, avg: 0.0136ms
  COLOR_YUV2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.6375ms, avg: 0.0136ms
 COLOR_BGR2RGBA [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 56.7775ms, avg: 0.0568ms
 COLOR_BGR2GRAY [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.4914ms, avg: 0.0075ms
 COLOR_RGB2GRAY [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.4894ms, avg: 0.0075ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=2048
   COLOR_GRAY2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 18.4145ms, avg: 0.0184ms
  COLOR_GRAY2BGRA [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 24.1101ms, avg: 0.0241ms
    COLOR_BGR2RGB [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 20.0167ms, avg: 0.0200ms
  COLOR_BGR2YCrCb [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.8936ms, avg: 0.0139ms
  COLOR_RGB2YCrCb [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.9480ms, avg: 0.0139ms
    COLOR_BGR2YUV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.8979ms, avg: 0.0139ms
    COLOR_RGB2YUV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.9577ms, avg: 0.0140ms
    COLOR_BGR2HSV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 167.7623ms, avg: 0.1678ms
    COLOR_HSV2BGR [uint8]: passed: 1e0, 1e-1: 33.4%, 1e-3: 33.4%, 1e-6: 33.4%, max_diff: 1.0, iters: 1000, time: 20.9503ms, avg: 0.0210ms
    COLOR_RGB2HSV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 167.7406ms, avg: 0.1677ms
    COLOR_BGR2HLS [uint8]: passed: 1e0, 1e-1: 0.0479%, 1e-3: 0.0479%, 1e-6: 0.0479%, max_diff: 1.0, iters: 1000, time: 17.0212ms, avg: 0.0170ms
    COLOR_HLS2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 19.7232ms, avg: 0.0197ms
    COLOR_RGB2HLS [uint8]: passed: 1e0, 1e-1: 0.0478%, 1e-3: 0.0478%, 1e-6: 0.0478%, max_diff: 1.0, iters: 1000, time: 17.0238ms, avg: 0.0170ms
    COLOR_BGR2XYZ [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.4881ms, avg: 0.0135ms
    COLOR_XYZ2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.4885ms, avg: 0.0135ms
    COLOR_RGB2XYZ [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.4869ms, avg: 0.0135ms
  COLOR_YCrCb2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.2401ms, avg: 0.0132ms
    COLOR_YUV2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.2408ms, avg: 0.0132ms
    COLOR_BGR2Lab [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 29.3856ms, avg: 0.0294ms
   COLOR_BGR2RGBA [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 23.1712ms, avg: 0.0232ms
   COLOR_BGR2BGRA [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 23.1616ms, avg: 0.0232ms
   COLOR_BGR2GRAY [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 11.6899ms, avg: 0.0117ms
   COLOR_RGB2GRAY [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 11.6920ms, avg: 0.0117ms
 COLOR_GRAY2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 75.0155ms, avg: 0.0750ms
COLOR_GRAY2BGRA [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 96.6594ms, avg: 0.0967ms
  COLOR_BGR2RGB [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 109.1778ms, avg: 0.1092ms
COLOR_BGR2YCrCb [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 5.9604645e-08, iters: 1000, time: 102.8571ms, avg: 0.1029ms
COLOR_RGB2YCrCb [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 5.9604645e-08, iters: 1000, time: 102.8199ms, avg: 0.1028ms
  COLOR_BGR2YUV [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 102.8619ms, avg: 0.1029ms
  COLOR_RGB2YUV [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 102.8521ms, avg: 0.1029ms
  COLOR_BGR2HSV [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.6%, max_diff: 3.0517578e-05, iters: 1000, time: 102.9012ms, avg: 0.1029ms
  COLOR_HSV2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 103.0638ms, avg: 0.1031ms
  COLOR_RGB2HSV [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.6%, max_diff: 3.0517578e-05, iters: 1000, time: 103.0643ms, avg: 0.1031ms
  COLOR_BGR2HLS [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.59%, max_diff: 3.0517578e-05, iters: 1000, time: 102.8888ms, avg: 0.1029ms
  COLOR_HLS2BGR [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 103.0962ms, avg: 0.1031ms
  COLOR_RGB2HLS [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.59%, max_diff: 3.0517578e-05, iters: 1000, time: 102.9074ms, avg: 0.1029ms
  COLOR_BGR2XYZ [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 103.0989ms, avg: 0.1031ms
  COLOR_XYZ2BGR [float32]: passed: 1e-6, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 4.7683716e-07, iters: 1000, time: 103.0948ms, avg: 0.1031ms
  COLOR_RGB2XYZ [float32]: passed: 1e-6, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 2.3841858e-07, iters: 1000, time: 103.1272ms, avg: 0.1031ms
COLOR_YCrCb2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 102.8612ms, avg: 0.1029ms
  COLOR_YUV2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 102.8731ms, avg: 0.1029ms
 COLOR_BGR2RGBA [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 128.2799ms, avg: 0.1283ms
 COLOR_BGR2GRAY [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.2630ms, avg: 0.0133ms
 COLOR_RGB2GRAY [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.2644ms, avg: 0.0133ms
-------------------------------------------------------------------------------------
```



