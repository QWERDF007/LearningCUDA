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
   COLOR_GRAY2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 5.8899ms, avg: 0.0059ms
  COLOR_GRAY2BGRA [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.3538ms, avg: 0.0074ms
    COLOR_BGR2RGB [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.3374ms, avg: 0.0063ms
  COLOR_BGR2YCrCb [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.6666ms, avg: 0.0047ms
  COLOR_RGB2YCrCb [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.6763ms, avg: 0.0047ms
    COLOR_BGR2YUV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.6685ms, avg: 0.0047ms
    COLOR_RGB2YUV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.6754ms, avg: 0.0047ms
    COLOR_BGR2HSV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 43.4082ms, avg: 0.0434ms
    COLOR_HSV2BGR [uint8]: passed: 1e0, 1e-1: 33.4%, 1e-3: 33.4%, 1e-6: 33.4%, max_diff: 1.0, iters: 1000, time: 6.6257ms, avg: 0.0066ms
    COLOR_RGB2HSV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 43.4163ms, avg: 0.0434ms
    COLOR_BGR2HLS [uint8]: passed: 1e0, 1e-1: 0.0494%, 1e-3: 0.0494%, 1e-6: 0.0494%, max_diff: 1.0, iters: 1000, time: 5.4922ms, avg: 0.0055ms
    COLOR_HLS2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.3002ms, avg: 0.0063ms
    COLOR_RGB2HLS [uint8]: passed: 1e0, 1e-1: 0.0494%, 1e-3: 0.0494%, 1e-6: 0.0494%, max_diff: 1.0, iters: 1000, time: 5.4882ms, avg: 0.0055ms
    COLOR_BGR2XYZ [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.6601ms, avg: 0.0047ms
    COLOR_XYZ2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.6580ms, avg: 0.0047ms
    COLOR_RGB2XYZ [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.6580ms, avg: 0.0047ms
  COLOR_YCrCb2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.5335ms, avg: 0.0045ms
    COLOR_YUV2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.5350ms, avg: 0.0045ms
    COLOR_BGR2Lab [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 8.7059ms, avg: 0.0087ms
   COLOR_BGR2RGBA [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.3056ms, avg: 0.0073ms
   COLOR_BGR2BGRA [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.2951ms, avg: 0.0073ms
   COLOR_BGR2GRAY [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.2372ms, avg: 0.0042ms
   COLOR_RGB2GRAY [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.2367ms, avg: 0.0042ms
 COLOR_GRAY2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 19.9099ms, avg: 0.0199ms
COLOR_GRAY2BGRA [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 28.1923ms, avg: 0.0282ms
  COLOR_BGR2RGB [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 20.7915ms, avg: 0.0208ms
COLOR_BGR2YCrCb [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 5.9604645e-08, iters: 1000, time: 7.5672ms, avg: 0.0076ms
COLOR_RGB2YCrCb [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 5.9604645e-08, iters: 1000, time: 7.5626ms, avg: 0.0076ms
  COLOR_BGR2YUV [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 7.5631ms, avg: 0.0076ms
  COLOR_RGB2YUV [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 7.5567ms, avg: 0.0076ms
  COLOR_BGR2HSV [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.56%, max_diff: 3.0517578e-05, iters: 1000, time: 7.6511ms, avg: 0.0077ms
  COLOR_HSV2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 8.1139ms, avg: 0.0081ms
  COLOR_RGB2HSV [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.57%, max_diff: 3.0517578e-05, iters: 1000, time: 7.7827ms, avg: 0.0078ms
  COLOR_BGR2HLS [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.6%, max_diff: 3.0517578e-05, iters: 1000, time: 7.6475ms, avg: 0.0076ms
  COLOR_HLS2BGR [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 8.0504ms, avg: 0.0081ms
  COLOR_RGB2HLS [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.59%, max_diff: 3.0517578e-05, iters: 1000, time: 7.6482ms, avg: 0.0076ms
  COLOR_BGR2XYZ [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 8.0202ms, avg: 0.0080ms
  COLOR_XYZ2BGR [float32]: passed: 1e-6, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 4.7683716e-07, iters: 1000, time: 8.0259ms, avg: 0.0080ms
  COLOR_RGB2XYZ [float32]: passed: 1e-6, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 2.3841858e-07, iters: 1000, time: 8.0311ms, avg: 0.0080ms
COLOR_YCrCb2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.6339ms, avg: 0.0076ms
  COLOR_YUV2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.6306ms, avg: 0.0076ms
  COLOR_BGR2Lab [float32]: passed: 1e-1, 1e-1: 0%, 1e-3: 0.255%, 1e-6: 0.255%, max_diff: 0.015625, iters: 1000, time: 38.9812ms, avg: 0.0390ms
 COLOR_BGR2RGBA [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 29.2835ms, avg: 0.0293ms
 COLOR_BGR2GRAY [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.5791ms, avg: 0.0046ms
 COLOR_RGB2GRAY [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 4.5774ms, avg: 0.0046ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1024, W=2048
   COLOR_GRAY2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 10.0679ms, avg: 0.0101ms
  COLOR_GRAY2BGRA [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 12.9356ms, avg: 0.0129ms
    COLOR_BGR2RGB [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 10.9320ms, avg: 0.0109ms
  COLOR_BGR2YCrCb [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.7653ms, avg: 0.0078ms
  COLOR_RGB2YCrCb [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.8087ms, avg: 0.0078ms
    COLOR_BGR2YUV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.7622ms, avg: 0.0078ms
    COLOR_RGB2YUV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.8096ms, avg: 0.0078ms
    COLOR_BGR2HSV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 84.8193ms, avg: 0.0848ms
    COLOR_HSV2BGR [uint8]: passed: 1e0, 1e-1: 33.4%, 1e-3: 33.4%, 1e-6: 33.4%, max_diff: 1.0, iters: 1000, time: 11.3883ms, avg: 0.0114ms
    COLOR_RGB2HSV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 84.8191ms, avg: 0.0848ms
    COLOR_BGR2HLS [uint8]: passed: 1e0, 1e-1: 0.0496%, 1e-3: 0.0496%, 1e-6: 0.0496%, max_diff: 1.0, iters: 1000, time: 9.3815ms, avg: 0.0094ms
    COLOR_HLS2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 10.8001ms, avg: 0.0108ms
    COLOR_RGB2HLS [uint8]: passed: 1e0, 1e-1: 0.0505%, 1e-3: 0.0505%, 1e-6: 0.0505%, max_diff: 1.0, iters: 1000, time: 9.3832ms, avg: 0.0094ms
    COLOR_BGR2XYZ [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.5636ms, avg: 0.0076ms
    COLOR_XYZ2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.5712ms, avg: 0.0076ms
    COLOR_RGB2XYZ [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.5645ms, avg: 0.0076ms
  COLOR_YCrCb2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.4294ms, avg: 0.0074ms
    COLOR_YUV2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.4320ms, avg: 0.0074ms
    COLOR_BGR2Lab [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 15.6028ms, avg: 0.0156ms
   COLOR_BGR2RGBA [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 12.5735ms, avg: 0.0126ms
   COLOR_BGR2BGRA [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 12.5744ms, avg: 0.0126ms
   COLOR_BGR2GRAY [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.7301ms, avg: 0.0067ms
   COLOR_RGB2GRAY [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.7313ms, avg: 0.0067ms
 COLOR_GRAY2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 38.2533ms, avg: 0.0383ms
COLOR_GRAY2BGRA [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 54.5821ms, avg: 0.0546ms
  COLOR_BGR2RGB [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 40.1177ms, avg: 0.0401ms
COLOR_BGR2YCrCb [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 5.9604645e-08, iters: 1000, time: 13.5033ms, avg: 0.0135ms
COLOR_RGB2YCrCb [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 5.9604645e-08, iters: 1000, time: 13.5024ms, avg: 0.0135ms
  COLOR_BGR2YUV [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 13.5036ms, avg: 0.0135ms
  COLOR_RGB2YUV [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 13.5033ms, avg: 0.0135ms
  COLOR_BGR2HSV [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.59%, max_diff: 3.0517578e-05, iters: 1000, time: 13.6328ms, avg: 0.0136ms
  COLOR_HSV2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 14.1218ms, avg: 0.0141ms
  COLOR_RGB2HSV [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.59%, max_diff: 3.0517578e-05, iters: 1000, time: 13.8817ms, avg: 0.0139ms
  COLOR_BGR2HLS [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.6%, max_diff: 3.0517578e-05, iters: 1000, time: 13.6199ms, avg: 0.0136ms
  COLOR_HLS2BGR [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 14.0281ms, avg: 0.0140ms
  COLOR_RGB2HLS [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.59%, max_diff: 3.0517578e-05, iters: 1000, time: 13.5894ms, avg: 0.0136ms
  COLOR_BGR2XYZ [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 14.0154ms, avg: 0.0140ms
  COLOR_XYZ2BGR [float32]: passed: 1e-6, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 4.7683716e-07, iters: 1000, time: 14.0166ms, avg: 0.0140ms
  COLOR_RGB2XYZ [float32]: passed: 1e-6, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 2.3841858e-07, iters: 1000, time: 14.0173ms, avg: 0.0140ms
COLOR_YCrCb2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.6080ms, avg: 0.0136ms
  COLOR_YUV2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.6080ms, avg: 0.0136ms
  COLOR_BGR2Lab [float32]: passed: 1e-1, 1e-1: 0%, 1e-3: 0.258%, 1e-6: 0.258%, max_diff: 0.015625, iters: 1000, time: 75.6893ms, avg: 0.0757ms
 COLOR_BGR2RGBA [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 56.7601ms, avg: 0.0568ms
 COLOR_BGR2GRAY [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.5114ms, avg: 0.0075ms
 COLOR_RGB2GRAY [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.5111ms, avg: 0.0075ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=1024
   COLOR_GRAY2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 10.0787ms, avg: 0.0101ms
  COLOR_GRAY2BGRA [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 12.9473ms, avg: 0.0129ms
    COLOR_BGR2RGB [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 10.9384ms, avg: 0.0109ms
  COLOR_BGR2YCrCb [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.7598ms, avg: 0.0078ms
  COLOR_RGB2YCrCb [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.8094ms, avg: 0.0078ms
    COLOR_BGR2YUV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.7598ms, avg: 0.0078ms
    COLOR_RGB2YUV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.8089ms, avg: 0.0078ms
    COLOR_BGR2HSV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 85.0065ms, avg: 0.0850ms
    COLOR_HSV2BGR [uint8]: passed: 1e0, 1e-1: 33.4%, 1e-3: 33.4%, 1e-6: 33.4%, max_diff: 1.0, iters: 1000, time: 11.4067ms, avg: 0.0114ms
    COLOR_RGB2HSV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 85.0086ms, avg: 0.0850ms
    COLOR_BGR2HLS [uint8]: passed: 1e0, 1e-1: 0.049%, 1e-3: 0.049%, 1e-6: 0.049%, max_diff: 1.0, iters: 1000, time: 9.3896ms, avg: 0.0094ms
    COLOR_HLS2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 10.8085ms, avg: 0.0108ms
    COLOR_RGB2HLS [uint8]: passed: 1e0, 1e-1: 0.0486%, 1e-3: 0.0486%, 1e-6: 0.0486%, max_diff: 1.0, iters: 1000, time: 9.3911ms, avg: 0.0094ms
    COLOR_BGR2XYZ [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.5715ms, avg: 0.0076ms
    COLOR_XYZ2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.5781ms, avg: 0.0076ms
    COLOR_RGB2XYZ [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.5679ms, avg: 0.0076ms
  COLOR_YCrCb2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.4368ms, avg: 0.0074ms
    COLOR_YUV2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.4382ms, avg: 0.0074ms
    COLOR_BGR2Lab [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 15.6193ms, avg: 0.0156ms
   COLOR_BGR2RGBA [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 12.5875ms, avg: 0.0126ms
   COLOR_BGR2BGRA [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 12.5909ms, avg: 0.0126ms
   COLOR_BGR2GRAY [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.7337ms, avg: 0.0067ms
   COLOR_RGB2GRAY [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 6.7339ms, avg: 0.0067ms
 COLOR_GRAY2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 38.2421ms, avg: 0.0382ms
COLOR_GRAY2BGRA [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 54.5893ms, avg: 0.0546ms
  COLOR_BGR2RGB [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 40.1695ms, avg: 0.0402ms
COLOR_BGR2YCrCb [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 5.9604645e-08, iters: 1000, time: 13.5133ms, avg: 0.0135ms
COLOR_RGB2YCrCb [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 5.9604645e-08, iters: 1000, time: 13.5093ms, avg: 0.0135ms
  COLOR_BGR2YUV [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 13.5105ms, avg: 0.0135ms
  COLOR_RGB2YUV [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 13.5109ms, avg: 0.0135ms
  COLOR_BGR2HSV [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.59%, max_diff: 3.0517578e-05, iters: 1000, time: 13.6611ms, avg: 0.0137ms
  COLOR_HSV2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 14.1416ms, avg: 0.0141ms
  COLOR_RGB2HSV [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.58%, max_diff: 3.0517578e-05, iters: 1000, time: 13.8872ms, avg: 0.0139ms
  COLOR_BGR2HLS [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.59%, max_diff: 3.0517578e-05, iters: 1000, time: 13.5939ms, avg: 0.0136ms
  COLOR_HLS2BGR [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 14.0316ms, avg: 0.0140ms
  COLOR_RGB2HLS [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.59%, max_diff: 3.0517578e-05, iters: 1000, time: 13.5987ms, avg: 0.0136ms
  COLOR_BGR2XYZ [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 14.0288ms, avg: 0.0140ms
  COLOR_XYZ2BGR [float32]: passed: 1e-6, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 4.7683716e-07, iters: 1000, time: 14.0328ms, avg: 0.0140ms
  COLOR_RGB2XYZ [float32]: passed: 1e-6, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 2.3841858e-07, iters: 1000, time: 14.0319ms, avg: 0.0140ms
COLOR_YCrCb2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.6158ms, avg: 0.0136ms
  COLOR_YUV2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.6132ms, avg: 0.0136ms
  COLOR_BGR2Lab [float32]: passed: 1e-1, 1e-1: 0%, 1e-3: 0.257%, 1e-6: 0.257%, max_diff: 0.015625, iters: 1000, time: 75.7630ms, avg: 0.0758ms
 COLOR_BGR2RGBA [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 56.7672ms, avg: 0.0568ms
 COLOR_BGR2GRAY [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.5128ms, avg: 0.0075ms
 COLOR_RGB2GRAY [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 7.5142ms, avg: 0.0075ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=2048
   COLOR_GRAY2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 18.4186ms, avg: 0.0184ms
  COLOR_GRAY2BGRA [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 24.0984ms, avg: 0.0241ms
    COLOR_BGR2RGB [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 20.0112ms, avg: 0.0200ms
  COLOR_BGR2YCrCb [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.8884ms, avg: 0.0139ms
  COLOR_RGB2YCrCb [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.9456ms, avg: 0.0139ms
    COLOR_BGR2YUV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.8860ms, avg: 0.0139ms
    COLOR_RGB2YUV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.9458ms, avg: 0.0139ms
    COLOR_BGR2HSV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 168.3791ms, avg: 0.1684ms
    COLOR_HSV2BGR [uint8]: passed: 1e0, 1e-1: 33.4%, 1e-3: 33.4%, 1e-6: 33.4%, max_diff: 1.0, iters: 1000, time: 21.0173ms, avg: 0.0210ms
    COLOR_RGB2HSV [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 168.3798ms, avg: 0.1684ms
    COLOR_BGR2HLS [uint8]: passed: 1e0, 1e-1: 0.0489%, 1e-3: 0.0489%, 1e-6: 0.0489%, max_diff: 1.0, iters: 1000, time: 17.0143ms, avg: 0.0170ms
    COLOR_HLS2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 19.7315ms, avg: 0.0197ms
    COLOR_RGB2HLS [uint8]: passed: 1e0, 1e-1: 0.0488%, 1e-3: 0.0488%, 1e-6: 0.0488%, max_diff: 1.0, iters: 1000, time: 17.0145ms, avg: 0.0170ms
    COLOR_BGR2XYZ [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.4602ms, avg: 0.0135ms
    COLOR_XYZ2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.4656ms, avg: 0.0135ms
    COLOR_RGB2XYZ [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.4599ms, avg: 0.0135ms
  COLOR_YCrCb2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.2136ms, avg: 0.0132ms
    COLOR_YUV2BGR [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.2134ms, avg: 0.0132ms
    COLOR_BGR2Lab [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 29.4371ms, avg: 0.0294ms
   COLOR_BGR2RGBA [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 23.1745ms, avg: 0.0232ms
   COLOR_BGR2BGRA [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 23.1507ms, avg: 0.0232ms
   COLOR_BGR2GRAY [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 11.6963ms, avg: 0.0117ms
   COLOR_RGB2GRAY [uint8]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 11.6971ms, avg: 0.0117ms
 COLOR_GRAY2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 75.0089ms, avg: 0.0750ms
COLOR_GRAY2BGRA [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 96.6799ms, avg: 0.0967ms
  COLOR_BGR2RGB [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 109.1506ms, avg: 0.1092ms
COLOR_BGR2YCrCb [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 5.9604645e-08, iters: 1000, time: 102.8686ms, avg: 0.1029ms
COLOR_RGB2YCrCb [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 5.9604645e-08, iters: 1000, time: 102.7753ms, avg: 0.1028ms
  COLOR_BGR2YUV [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 102.8790ms, avg: 0.1029ms
  COLOR_RGB2YUV [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 102.8020ms, avg: 0.1028ms
  COLOR_BGR2HSV [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.59%, max_diff: 3.0517578e-05, iters: 1000, time: 102.9224ms, avg: 0.1029ms
  COLOR_HSV2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 103.1516ms, avg: 0.1032ms
  COLOR_RGB2HSV [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.59%, max_diff: 3.0517578e-05, iters: 1000, time: 102.9990ms, avg: 0.1030ms
  COLOR_BGR2HLS [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.59%, max_diff: 3.0517578e-05, iters: 1000, time: 102.9782ms, avg: 0.1030ms
  COLOR_HLS2BGR [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 103.1117ms, avg: 0.1031ms
  COLOR_RGB2HLS [float32]: passed: 1e-4, 1e-1: 0%, 1e-3: 0%, 1e-6: 2.58%, max_diff: 3.0517578e-05, iters: 1000, time: 102.7114ms, avg: 0.1027ms
  COLOR_BGR2XYZ [float32]: passed: 1e-7, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 1.1920929e-07, iters: 1000, time: 103.0960ms, avg: 0.1031ms
  COLOR_XYZ2BGR [float32]: passed: 1e-6, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 4.7683716e-07, iters: 1000, time: 103.2126ms, avg: 0.1032ms
  COLOR_RGB2XYZ [float32]: passed: 1e-6, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, max_diff: 2.3841858e-07, iters: 1000, time: 103.1272ms, avg: 0.1031ms
COLOR_YCrCb2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 102.9162ms, avg: 0.1029ms
  COLOR_YUV2BGR [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 102.9315ms, avg: 0.1029ms
  COLOR_BGR2Lab [float32]: passed: 1e-1, 1e-1: 0%, 1e-3: 0.254%, 1e-6: 0.254%, max_diff: 0.015625, iters: 1000, time: 157.0268ms, avg: 0.1570ms
 COLOR_BGR2RGBA [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 128.3655ms, avg: 0.1284ms
 COLOR_BGR2GRAY [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.2740ms, avg: 0.0133ms
 COLOR_RGB2GRAY [float32]: passed: 1e-8, 1e-1: 0%, 1e-3: 0%, 1e-6: 0%, iters: 1000, time: 13.2773ms, avg: 0.0133ms
-------------------------------------------------------------------------------------
```



