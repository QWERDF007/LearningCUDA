# MatTranspose

## 介绍

包含以下内容：


## 测试

硬件：

- cpu: Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
- gpu: NVIDIA GeForce RTX 4090D

运行：

```bash
# 只测试Ada架构 不指定默认编译所有架构 耗时较长: Volta, Ampere, Ada, Hopper, ...
export TORCH_CUDA_ARCH_LIST=Ada
python3 mat_transpose.py
```

输出：

```
-------------------------------------------------------------------------------------
                                        H=1024, W=1024
        out_f32_coalesced_read: (1e-8), iters: 1000, time: 30.0863ms, avg: 0.0301ms
       out_f32_coalesced_write: (1e-8), iters: 1000, time: 15.3019ms, avg: 0.0153ms
      out_f32x4_coalesced_read: (1e-8), iters: 1000, time: 30.7240ms, avg: 0.0307ms
     out_f32x4_coalesced_write: (1e-8), iters: 1000, time: 13.8249ms, avg: 0.0138ms
     out_f32_coalesced_read_2d: (1e-8), iters: 1000, time: 30.3469ms, avg: 0.0303ms
    out_f32_coalesced_write_2d: (1e-8), iters: 1000, time: 8.5692ms, avg: 0.0086ms
   out_f32x4_coalesced_read_2d: (1e-8), iters: 1000, time: 31.3807ms, avg: 0.0314ms
  out_f32x4_coalesced_write_2d: (1e-8), iters: 1000, time: 8.9958ms, avg: 0.0090ms
                out_f32_shared: (1e-8), iters: 1000, time: 4.9276ms, avg: 0.0049ms
              out_f32_shared_2: (1e-8), iters: 1000, time: 8.0442ms, avg: 0.0080ms
         out_u8_coalesced_read: (1e-8), iters: 1000, time: 30.5045ms, avg: 0.0305ms
        out_u8_coalesced_write: (1e-8), iters: 1000, time: 14.1685ms, avg: 0.0142ms
       out_u8x4_coalesced_read: (1e-8), iters: 1000, time: 30.3121ms, avg: 0.0303ms
      out_u8x4_coalesced_write: (1e-8), iters: 1000, time: 13.0308ms, avg: 0.0130ms
     out_u8x16_coalesced_write: (1e-8), iters: 1000, time: 6.9325ms, avg: 0.0069ms
    out_u8x4_coalesced_read_2d: (1e-8), iters: 1000, time: 30.7467ms, avg: 0.0307ms
   out_u8x4_coalesced_write_2d: (1e-8), iters: 1000, time: 8.8301ms, avg: 0.0088ms
  out_u8x16_coalesced_write_2d: (1e-8), iters: 1000, time: 3.9382ms, avg: 0.0039ms
                 out_u8_shared: (1e-8), iters: 1000, time: 4.6847ms, avg: 0.0047ms
               out_u8_shared_2: (1e-8), iters: 1000, time: 5.2259ms, avg: 0.0052ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1024, W=2048
        out_f32_coalesced_read: (1e-8), iters: 1000, time: 56.2370ms, avg: 0.0562ms
       out_f32_coalesced_write: (1e-8), iters: 1000, time: 29.6135ms, avg: 0.0296ms
      out_f32x4_coalesced_read: (1e-8), iters: 1000, time: 56.8798ms, avg: 0.0569ms
     out_f32x4_coalesced_write: (1e-8), iters: 1000, time: 28.4817ms, avg: 0.0285ms
     out_f32_coalesced_read_2d: (1e-8), iters: 1000, time: 56.9887ms, avg: 0.0570ms
    out_f32_coalesced_write_2d: (1e-8), iters: 1000, time: 15.9411ms, avg: 0.0159ms
   out_f32x4_coalesced_read_2d: (1e-8), iters: 1000, time: 57.7459ms, avg: 0.0577ms
  out_f32x4_coalesced_write_2d: (1e-8), iters: 1000, time: 15.5101ms, avg: 0.0155ms
                out_f32_shared: (1e-8), iters: 1000, time: 8.4484ms, avg: 0.0084ms
              out_f32_shared_2: (1e-8), iters: 1000, time: 14.8275ms, avg: 0.0148ms
         out_u8_coalesced_read: (1e-8), iters: 1000, time: 59.5691ms, avg: 0.0596ms
        out_u8_coalesced_write: (1e-8), iters: 1000, time: 26.4838ms, avg: 0.0265ms
       out_u8x4_coalesced_read: (1e-8), iters: 1000, time: 58.8088ms, avg: 0.0588ms
      out_u8x4_coalesced_write: (1e-8), iters: 1000, time: 26.9928ms, avg: 0.0270ms
     out_u8x16_coalesced_write: (1e-8), iters: 1000, time: 9.9974ms, avg: 0.0100ms
    out_u8x4_coalesced_read_2d: (1e-8), iters: 1000, time: 58.3372ms, avg: 0.0583ms
   out_u8x4_coalesced_write_2d: (1e-8), iters: 1000, time: 15.9881ms, avg: 0.0160ms
  out_u8x16_coalesced_write_2d: (1e-8), iters: 1000, time: 6.5942ms, avg: 0.0066ms
                 out_u8_shared: (1e-8), iters: 1000, time: 8.0063ms, avg: 0.0080ms
               out_u8_shared_2: (1e-8), iters: 1000, time: 9.1789ms, avg: 0.0092ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1024, W=4096
        out_f32_coalesced_read: (1e-8), iters: 1000, time: 113.0865ms, avg: 0.1131ms
       out_f32_coalesced_write: (1e-8), iters: 1000, time: 56.4253ms, avg: 0.0564ms
      out_f32x4_coalesced_read: (1e-8), iters: 1000, time: 111.5744ms, avg: 0.1116ms
     out_f32x4_coalesced_write: (1e-8), iters: 1000, time: 57.6739ms, avg: 0.0577ms
     out_f32_coalesced_read_2d: (1e-8), iters: 1000, time: 113.4248ms, avg: 0.1134ms
    out_f32_coalesced_write_2d: (1e-8), iters: 1000, time: 37.7598ms, avg: 0.0378ms
   out_f32x4_coalesced_read_2d: (1e-8), iters: 1000, time: 111.8433ms, avg: 0.1118ms
  out_f32x4_coalesced_write_2d: (1e-8), iters: 1000, time: 29.6099ms, avg: 0.0296ms
                out_f32_shared: (1e-8), iters: 1000, time: 15.3646ms, avg: 0.0154ms
              out_f32_shared_2: (1e-8), iters: 1000, time: 28.0786ms, avg: 0.0281ms
         out_u8_coalesced_read: (1e-8), iters: 1000, time: 107.9211ms, avg: 0.1079ms
        out_u8_coalesced_write: (1e-8), iters: 1000, time: 58.8493ms, avg: 0.0588ms
       out_u8x4_coalesced_read: (1e-8), iters: 1000, time: 108.9399ms, avg: 0.1089ms
      out_u8x4_coalesced_write: (1e-8), iters: 1000, time: 58.3320ms, avg: 0.0583ms
     out_u8x16_coalesced_write: (1e-8), iters: 1000, time: 19.9585ms, avg: 0.0200ms
    out_u8x4_coalesced_read_2d: (1e-8), iters: 1000, time: 110.2397ms, avg: 0.1102ms
   out_u8x4_coalesced_write_2d: (1e-8), iters: 1000, time: 29.7043ms, avg: 0.0297ms
  out_u8x16_coalesced_write_2d: (1e-8), iters: 1000, time: 10.1063ms, avg: 0.0101ms
                 out_u8_shared: (1e-8), iters: 1000, time: 14.6325ms, avg: 0.0146ms
               out_u8_shared_2: (1e-8), iters: 1000, time: 16.9132ms, avg: 0.0169ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1024, W=8192
        out_f32_coalesced_read: (1e-8), iters: 1000, time: 218.9944ms, avg: 0.2190ms
       out_f32_coalesced_write: (1e-8), iters: 1000, time: 121.6369ms, avg: 0.1216ms
      out_f32x4_coalesced_read: (1e-8), iters: 1000, time: 217.9766ms, avg: 0.2180ms
     out_f32x4_coalesced_write: (1e-8), iters: 1000, time: 127.4862ms, avg: 0.1275ms
     out_f32_coalesced_read_2d: (1e-8), iters: 1000, time: 220.5184ms, avg: 0.2205ms
    out_f32_coalesced_write_2d: (1e-8), iters: 1000, time: 130.7600ms, avg: 0.1308ms
   out_f32x4_coalesced_read_2d: (1e-8), iters: 1000, time: 218.7734ms, avg: 0.2188ms
  out_f32x4_coalesced_write_2d: (1e-8), iters: 1000, time: 57.1566ms, avg: 0.0572ms
                out_f32_shared: (1e-8), iters: 1000, time: 29.4120ms, avg: 0.0294ms
              out_f32_shared_2: (1e-8), iters: 1000, time: 54.6365ms, avg: 0.0546ms
         out_u8_coalesced_read: (1e-8), iters: 1000, time: 214.8438ms, avg: 0.2148ms
        out_u8_coalesced_write: (1e-8), iters: 1000, time: 104.7525ms, avg: 0.1048ms
       out_u8x4_coalesced_read: (1e-8), iters: 1000, time: 215.6096ms, avg: 0.2156ms
      out_u8x4_coalesced_write: (1e-8), iters: 1000, time: 114.6500ms, avg: 0.1147ms
     out_u8x16_coalesced_write: (1e-8), iters: 1000, time: 44.8112ms, avg: 0.0448ms
    out_u8x4_coalesced_read_2d: (1e-8), iters: 1000, time: 215.8942ms, avg: 0.2159ms
   out_u8x4_coalesced_write_2d: (1e-8), iters: 1000, time: 57.5039ms, avg: 0.0575ms
  out_u8x16_coalesced_write_2d: (1e-8), iters: 1000, time: 17.4222ms, avg: 0.0174ms
                 out_u8_shared: (1e-8), iters: 1000, time: 27.8406ms, avg: 0.0278ms
               out_u8_shared_2: (1e-8), iters: 1000, time: 32.6011ms, avg: 0.0326ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=1024
        out_f32_coalesced_read: (1e-8), iters: 1000, time: 58.4998ms, avg: 0.0585ms
       out_f32_coalesced_write: (1e-8), iters: 1000, time: 28.5900ms, avg: 0.0286ms
      out_f32x4_coalesced_read: (1e-8), iters: 1000, time: 58.6574ms, avg: 0.0587ms
     out_f32x4_coalesced_write: (1e-8), iters: 1000, time: 29.3584ms, avg: 0.0294ms
     out_f32_coalesced_read_2d: (1e-8), iters: 1000, time: 59.7212ms, avg: 0.0597ms
    out_f32_coalesced_write_2d: (1e-8), iters: 1000, time: 15.7132ms, avg: 0.0157ms
   out_f32x4_coalesced_read_2d: (1e-8), iters: 1000, time: 59.0987ms, avg: 0.0591ms
  out_f32x4_coalesced_write_2d: (1e-8), iters: 1000, time: 15.8587ms, avg: 0.0159ms
                out_f32_shared: (1e-8), iters: 1000, time: 8.4577ms, avg: 0.0085ms
              out_f32_shared_2: (1e-8), iters: 1000, time: 14.8423ms, avg: 0.0148ms
         out_u8_coalesced_read: (1e-8), iters: 1000, time: 59.1328ms, avg: 0.0591ms
        out_u8_coalesced_write: (1e-8), iters: 1000, time: 27.7238ms, avg: 0.0277ms
       out_u8x4_coalesced_read: (1e-8), iters: 1000, time: 58.3122ms, avg: 0.0583ms
      out_u8x4_coalesced_write: (1e-8), iters: 1000, time: 28.7495ms, avg: 0.0287ms
     out_u8x16_coalesced_write: (1e-8), iters: 1000, time: 20.1697ms, avg: 0.0202ms
    out_u8x4_coalesced_read_2d: (1e-8), iters: 1000, time: 58.2497ms, avg: 0.0582ms
   out_u8x4_coalesced_write_2d: (1e-8), iters: 1000, time: 15.9931ms, avg: 0.0160ms
  out_u8x16_coalesced_write_2d: (1e-8), iters: 1000, time: 6.5908ms, avg: 0.0066ms
                 out_u8_shared: (1e-8), iters: 1000, time: 8.0202ms, avg: 0.0080ms
               out_u8_shared_2: (1e-8), iters: 1000, time: 9.2115ms, avg: 0.0092ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=2048
        out_f32_coalesced_read: (1e-8), iters: 1000, time: 115.4742ms, avg: 0.1155ms
       out_f32_coalesced_write: (1e-8), iters: 1000, time: 56.0131ms, avg: 0.0560ms
      out_f32x4_coalesced_read: (1e-8), iters: 1000, time: 114.5606ms, avg: 0.1146ms
     out_f32x4_coalesced_write: (1e-8), iters: 1000, time: 57.3153ms, avg: 0.0573ms
     out_f32_coalesced_read_2d: (1e-8), iters: 1000, time: 116.4083ms, avg: 0.1164ms
    out_f32_coalesced_write_2d: (1e-8), iters: 1000, time: 29.9993ms, avg: 0.0300ms
   out_f32x4_coalesced_read_2d: (1e-8), iters: 1000, time: 115.2554ms, avg: 0.1153ms
  out_f32x4_coalesced_write_2d: (1e-8), iters: 1000, time: 30.1604ms, avg: 0.0302ms
                out_f32_shared: (1e-8), iters: 1000, time: 15.3918ms, avg: 0.0154ms
              out_f32_shared_2: (1e-8), iters: 1000, time: 28.1632ms, avg: 0.0282ms
         out_u8_coalesced_read: (1e-8), iters: 1000, time: 111.0897ms, avg: 0.1111ms
        out_u8_coalesced_write: (1e-8), iters: 1000, time: 52.4726ms, avg: 0.0525ms
       out_u8x4_coalesced_read: (1e-8), iters: 1000, time: 110.5700ms, avg: 0.1106ms
      out_u8x4_coalesced_write: (1e-8), iters: 1000, time: 55.4829ms, avg: 0.0555ms
     out_u8x16_coalesced_write: (1e-8), iters: 1000, time: 37.6980ms, avg: 0.0377ms
    out_u8x4_coalesced_read_2d: (1e-8), iters: 1000, time: 113.5845ms, avg: 0.1136ms
   out_u8x4_coalesced_write_2d: (1e-8), iters: 1000, time: 29.9418ms, avg: 0.0299ms
  out_u8x16_coalesced_write_2d: (1e-8), iters: 1000, time: 9.9916ms, avg: 0.0100ms
                 out_u8_shared: (1e-8), iters: 1000, time: 14.6275ms, avg: 0.0146ms
               out_u8_shared_2: (1e-8), iters: 1000, time: 16.9384ms, avg: 0.0169ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=4096
        out_f32_coalesced_read: (1e-8), iters: 1000, time: 224.3357ms, avg: 0.2243ms
       out_f32_coalesced_write: (1e-8), iters: 1000, time: 113.2765ms, avg: 0.1133ms
      out_f32x4_coalesced_read: (1e-8), iters: 1000, time: 223.9058ms, avg: 0.2239ms
     out_f32x4_coalesced_write: (1e-8), iters: 1000, time: 115.8335ms, avg: 0.1158ms
     out_f32_coalesced_read_2d: (1e-8), iters: 1000, time: 226.2907ms, avg: 0.2263ms
    out_f32_coalesced_write_2d: (1e-8), iters: 1000, time: 58.9123ms, avg: 0.0589ms
   out_f32x4_coalesced_read_2d: (1e-8), iters: 1000, time: 224.6070ms, avg: 0.2246ms
  out_f32x4_coalesced_write_2d: (1e-8), iters: 1000, time: 58.5742ms, avg: 0.0586ms
                out_f32_shared: (1e-8), iters: 1000, time: 29.4113ms, avg: 0.0294ms
              out_f32_shared_2: (1e-8), iters: 1000, time: 54.6513ms, avg: 0.0547ms
         out_u8_coalesced_read: (1e-8), iters: 1000, time: 233.3591ms, avg: 0.2334ms
        out_u8_coalesced_write: (1e-8), iters: 1000, time: 110.0583ms, avg: 0.1101ms
       out_u8x4_coalesced_read: (1e-8), iters: 1000, time: 231.3004ms, avg: 0.2313ms
      out_u8x4_coalesced_write: (1e-8), iters: 1000, time: 112.6926ms, avg: 0.1127ms
     out_u8x16_coalesced_write: (1e-8), iters: 1000, time: 78.9182ms, avg: 0.0789ms
    out_u8x4_coalesced_read_2d: (1e-8), iters: 1000, time: 232.2440ms, avg: 0.2322ms
   out_u8x4_coalesced_write_2d: (1e-8), iters: 1000, time: 59.5241ms, avg: 0.0595ms
  out_u8x16_coalesced_write_2d: (1e-8), iters: 1000, time: 17.6911ms, avg: 0.0177ms
                 out_u8_shared: (1e-8), iters: 1000, time: 27.8332ms, avg: 0.0278ms
               out_u8_shared_2: (1e-8), iters: 1000, time: 32.5966ms, avg: 0.0326ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=8192
        out_f32_coalesced_read: (1e-8), iters: 1000, time: 440.4919ms, avg: 0.4405ms
       out_f32_coalesced_write: (1e-8), iters: 1000, time: 253.9489ms, avg: 0.2539ms
      out_f32x4_coalesced_read: (1e-8), iters: 1000, time: 441.6945ms, avg: 0.4417ms
     out_f32x4_coalesced_write: (1e-8), iters: 1000, time: 239.0270ms, avg: 0.2390ms
     out_f32_coalesced_read_2d: (1e-8), iters: 1000, time: 444.8729ms, avg: 0.4449ms
    out_f32_coalesced_write_2d: (1e-8), iters: 1000, time: 209.1734ms, avg: 0.2092ms
   out_f32x4_coalesced_read_2d: (1e-8), iters: 1000, time: 441.4580ms, avg: 0.4415ms
  out_f32x4_coalesced_write_2d: (1e-8), iters: 1000, time: 151.7391ms, avg: 0.1517ms
                out_f32_shared: (1e-8), iters: 1000, time: 172.5678ms, avg: 0.1726ms
              out_f32_shared_2: (1e-8), iters: 1000, time: 193.0933ms, avg: 0.1931ms
         out_u8_coalesced_read: (1e-8), iters: 1000, time: 439.2371ms, avg: 0.4392ms
        out_u8_coalesced_write: (1e-8), iters: 1000, time: 216.8221ms, avg: 0.2168ms
       out_u8x4_coalesced_read: (1e-8), iters: 1000, time: 438.8885ms, avg: 0.4389ms
      out_u8x4_coalesced_write: (1e-8), iters: 1000, time: 227.2577ms, avg: 0.2273ms
     out_u8x16_coalesced_write: (1e-8), iters: 1000, time: 166.3966ms, avg: 0.1664ms
    out_u8x4_coalesced_read_2d: (1e-8), iters: 1000, time: 440.4294ms, avg: 0.4404ms
   out_u8x4_coalesced_write_2d: (1e-8), iters: 1000, time: 116.9372ms, avg: 0.1169ms
  out_u8x16_coalesced_write_2d: (1e-8), iters: 1000, time: 33.7062ms, avg: 0.0337ms
                 out_u8_shared: (1e-8), iters: 1000, time: 54.4055ms, avg: 0.0544ms
               out_u8_shared_2: (1e-8), iters: 1000, time: 63.7071ms, avg: 0.0637ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=1024
        out_f32_coalesced_read: (1e-8), iters: 1000, time: 119.9894ms, avg: 0.1200ms
       out_f32_coalesced_write: (1e-8), iters: 1000, time: 55.8913ms, avg: 0.0559ms
      out_f32x4_coalesced_read: (1e-8), iters: 1000, time: 119.5879ms, avg: 0.1196ms
     out_f32x4_coalesced_write: (1e-8), iters: 1000, time: 56.6711ms, avg: 0.0567ms
     out_f32_coalesced_read_2d: (1e-8), iters: 1000, time: 121.5177ms, avg: 0.1215ms
    out_f32_coalesced_write_2d: (1e-8), iters: 1000, time: 29.6102ms, avg: 0.0296ms
   out_f32x4_coalesced_read_2d: (1e-8), iters: 1000, time: 120.4317ms, avg: 0.1204ms
  out_f32x4_coalesced_write_2d: (1e-8), iters: 1000, time: 31.4264ms, avg: 0.0314ms
                out_f32_shared: (1e-8), iters: 1000, time: 15.3935ms, avg: 0.0154ms
              out_f32_shared_2: (1e-8), iters: 1000, time: 28.1501ms, avg: 0.0282ms
         out_u8_coalesced_read: (1e-8), iters: 1000, time: 141.4382ms, avg: 0.1414ms
        out_u8_coalesced_write: (1e-8), iters: 1000, time: 52.5045ms, avg: 0.0525ms
       out_u8x4_coalesced_read: (1e-8), iters: 1000, time: 123.2307ms, avg: 0.1232ms
      out_u8x4_coalesced_write: (1e-8), iters: 1000, time: 54.1704ms, avg: 0.0542ms
     out_u8x16_coalesced_write: (1e-8), iters: 1000, time: 55.2065ms, avg: 0.0552ms
    out_u8x4_coalesced_read_2d: (1e-8), iters: 1000, time: 120.7950ms, avg: 0.1208ms
   out_u8x4_coalesced_write_2d: (1e-8), iters: 1000, time: 31.7764ms, avg: 0.0318ms
  out_u8x16_coalesced_write_2d: (1e-8), iters: 1000, time: 10.2725ms, avg: 0.0103ms
                 out_u8_shared: (1e-8), iters: 1000, time: 14.6406ms, avg: 0.0146ms
               out_u8_shared_2: (1e-8), iters: 1000, time: 16.9532ms, avg: 0.0170ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=2048
        out_f32_coalesced_read: (1e-8), iters: 1000, time: 239.3668ms, avg: 0.2394ms
       out_f32_coalesced_write: (1e-8), iters: 1000, time: 111.7969ms, avg: 0.1118ms
      out_f32x4_coalesced_read: (1e-8), iters: 1000, time: 237.8392ms, avg: 0.2378ms
     out_f32x4_coalesced_write: (1e-8), iters: 1000, time: 113.1785ms, avg: 0.1132ms
     out_f32_coalesced_read_2d: (1e-8), iters: 1000, time: 241.6050ms, avg: 0.2416ms
    out_f32_coalesced_write_2d: (1e-8), iters: 1000, time: 57.9355ms, avg: 0.0579ms
   out_f32x4_coalesced_read_2d: (1e-8), iters: 1000, time: 239.9397ms, avg: 0.2399ms
  out_f32x4_coalesced_write_2d: (1e-8), iters: 1000, time: 61.3303ms, avg: 0.0613ms
                out_f32_shared: (1e-8), iters: 1000, time: 29.3860ms, avg: 0.0294ms
              out_f32_shared_2: (1e-8), iters: 1000, time: 54.6384ms, avg: 0.0546ms
         out_u8_coalesced_read: (1e-8), iters: 1000, time: 239.8844ms, avg: 0.2399ms
        out_u8_coalesced_write: (1e-8), iters: 1000, time: 110.0748ms, avg: 0.1101ms
       out_u8x4_coalesced_read: (1e-8), iters: 1000, time: 230.2251ms, avg: 0.2302ms
      out_u8x4_coalesced_write: (1e-8), iters: 1000, time: 112.8461ms, avg: 0.1128ms
     out_u8x16_coalesced_write: (1e-8), iters: 1000, time: 111.2657ms, avg: 0.1113ms
    out_u8x4_coalesced_read_2d: (1e-8), iters: 1000, time: 230.6025ms, avg: 0.2306ms
   out_u8x4_coalesced_write_2d: (1e-8), iters: 1000, time: 59.7625ms, avg: 0.0598ms
  out_u8x16_coalesced_write_2d: (1e-8), iters: 1000, time: 17.8361ms, avg: 0.0178ms
                 out_u8_shared: (1e-8), iters: 1000, time: 27.7700ms, avg: 0.0278ms
               out_u8_shared_2: (1e-8), iters: 1000, time: 32.4712ms, avg: 0.0325ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=4096
        out_f32_coalesced_read: (1e-8), iters: 1000, time: 453.7754ms, avg: 0.4538ms
       out_f32_coalesced_write: (1e-8), iters: 1000, time: 346.4248ms, avg: 0.3464ms
      out_f32x4_coalesced_read: (1e-8), iters: 1000, time: 453.8348ms, avg: 0.4538ms
     out_f32x4_coalesced_write: (1e-8), iters: 1000, time: 263.6554ms, avg: 0.2637ms
     out_f32_coalesced_read_2d: (1e-8), iters: 1000, time: 456.4884ms, avg: 0.4565ms
    out_f32_coalesced_write_2d: (1e-8), iters: 1000, time: 214.4032ms, avg: 0.2144ms
   out_f32x4_coalesced_read_2d: (1e-8), iters: 1000, time: 453.8207ms, avg: 0.4538ms
  out_f32x4_coalesced_write_2d: (1e-8), iters: 1000, time: 152.0851ms, avg: 0.1521ms
                out_f32_shared: (1e-8), iters: 1000, time: 171.3092ms, avg: 0.1713ms
              out_f32_shared_2: (1e-8), iters: 1000, time: 193.4407ms, avg: 0.1934ms
         out_u8_coalesced_read: (1e-8), iters: 1000, time: 465.7547ms, avg: 0.4658ms
        out_u8_coalesced_write: (1e-8), iters: 1000, time: 223.0422ms, avg: 0.2230ms
       out_u8x4_coalesced_read: (1e-8), iters: 1000, time: 458.7927ms, avg: 0.4588ms
      out_u8x4_coalesced_write: (1e-8), iters: 1000, time: 226.3591ms, avg: 0.2264ms
     out_u8x16_coalesced_write: (1e-8), iters: 1000, time: 219.9821ms, avg: 0.2200ms
    out_u8x4_coalesced_read_2d: (1e-8), iters: 1000, time: 451.6590ms, avg: 0.4517ms
   out_u8x4_coalesced_write_2d: (1e-8), iters: 1000, time: 116.8683ms, avg: 0.1169ms
  out_u8x16_coalesced_write_2d: (1e-8), iters: 1000, time: 33.9646ms, avg: 0.0340ms
                 out_u8_shared: (1e-8), iters: 1000, time: 54.4202ms, avg: 0.0544ms
               out_u8_shared_2: (1e-8), iters: 1000, time: 63.7169ms, avg: 0.0637ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=8192
        out_f32_coalesced_read: (1e-8), iters: 1000, time: 890.0614ms, avg: 0.8901ms
       out_f32_coalesced_write: (1e-8), iters: 1000, time: 636.8608ms, avg: 0.6369ms
      out_f32x4_coalesced_read: (1e-8), iters: 1000, time: 890.2879ms, avg: 0.8903ms
     out_f32x4_coalesced_write: (1e-8), iters: 1000, time: 544.3771ms, avg: 0.5444ms
     out_f32_coalesced_read_2d: (1e-8), iters: 1000, time: 896.2910ms, avg: 0.8963ms
    out_f32_coalesced_write_2d: (1e-8), iters: 1000, time: 362.6521ms, avg: 0.3627ms
   out_f32x4_coalesced_read_2d: (1e-8), iters: 1000, time: 890.6789ms, avg: 0.8907ms
  out_f32x4_coalesced_write_2d: (1e-8), iters: 1000, time: 309.4425ms, avg: 0.3094ms
                out_f32_shared: (1e-8), iters: 1000, time: 347.3904ms, avg: 0.3474ms
              out_f32_shared_2: (1e-8), iters: 1000, time: 389.1168ms, avg: 0.3891ms
         out_u8_coalesced_read: (1e-8), iters: 1000, time: 887.2592ms, avg: 0.8873ms
        out_u8_coalesced_write: (1e-8), iters: 1000, time: 434.4218ms, avg: 0.4344ms
       out_u8x4_coalesced_read: (1e-8), iters: 1000, time: 884.4633ms, avg: 0.8845ms
      out_u8x4_coalesced_write: (1e-8), iters: 1000, time: 444.1831ms, avg: 0.4442ms
     out_u8x16_coalesced_write: (1e-8), iters: 1000, time: 447.9926ms, avg: 0.4480ms
    out_u8x4_coalesced_read_2d: (1e-8), iters: 1000, time: 879.8180ms, avg: 0.8798ms
   out_u8x4_coalesced_write_2d: (1e-8), iters: 1000, time: 229.7294ms, avg: 0.2297ms
  out_u8x16_coalesced_write_2d: (1e-8), iters: 1000, time: 65.7377ms, avg: 0.0657ms
                 out_u8_shared: (1e-8), iters: 1000, time: 107.3706ms, avg: 0.1074ms
               out_u8_shared_2: (1e-8), iters: 1000, time: 126.0297ms, avg: 0.1260ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=8192, W=1024
        out_f32_coalesced_read: (1e-8), iters: 1000, time: 257.5111ms, avg: 0.2575ms
       out_f32_coalesced_write: (1e-8), iters: 1000, time: 109.7653ms, avg: 0.1098ms
      out_f32x4_coalesced_read: (1e-8), iters: 1000, time: 256.7806ms, avg: 0.2568ms
     out_f32x4_coalesced_write: (1e-8), iters: 1000, time: 110.7612ms, avg: 0.1108ms
     out_f32_coalesced_read_2d: (1e-8), iters: 1000, time: 258.4462ms, avg: 0.2584ms
    out_f32_coalesced_write_2d: (1e-8), iters: 1000, time: 59.1629ms, avg: 0.0592ms
   out_f32x4_coalesced_read_2d: (1e-8), iters: 1000, time: 256.6340ms, avg: 0.2566ms
  out_f32x4_coalesced_write_2d: (1e-8), iters: 1000, time: 65.4550ms, avg: 0.0655ms
                out_f32_shared: (1e-8), iters: 1000, time: 29.4039ms, avg: 0.0294ms
              out_f32_shared_2: (1e-8), iters: 1000, time: 54.6489ms, avg: 0.0546ms
         out_u8_coalesced_read: (1e-8), iters: 1000, time: 253.5679ms, avg: 0.2536ms
        out_u8_coalesced_write: (1e-8), iters: 1000, time: 105.2351ms, avg: 0.1052ms
       out_u8x4_coalesced_read: (1e-8), iters: 1000, time: 246.6748ms, avg: 0.2467ms
      out_u8x4_coalesced_write: (1e-8), iters: 1000, time: 105.9525ms, avg: 0.1060ms
     out_u8x16_coalesced_write: (1e-8), iters: 1000, time: 109.4449ms, avg: 0.1094ms
    out_u8x4_coalesced_read_2d: (1e-8), iters: 1000, time: 247.2160ms, avg: 0.2472ms
   out_u8x4_coalesced_write_2d: (1e-8), iters: 1000, time: 63.6849ms, avg: 0.0637ms
  out_u8x16_coalesced_write_2d: (1e-8), iters: 1000, time: 18.4839ms, avg: 0.0185ms
                 out_u8_shared: (1e-8), iters: 1000, time: 27.8146ms, avg: 0.0278ms
               out_u8_shared_2: (1e-8), iters: 1000, time: 32.4714ms, avg: 0.0325ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=8192, W=2048
        out_f32_coalesced_read: (1e-8), iters: 1000, time: 467.0730ms, avg: 0.4671ms
       out_f32_coalesced_write: (1e-8), iters: 1000, time: 379.6425ms, avg: 0.3796ms
      out_f32x4_coalesced_read: (1e-8), iters: 1000, time: 465.9891ms, avg: 0.4660ms
     out_f32x4_coalesced_write: (1e-8), iters: 1000, time: 307.4250ms, avg: 0.3074ms
     out_f32_coalesced_read_2d: (1e-8), iters: 1000, time: 470.1378ms, avg: 0.4701ms
    out_f32_coalesced_write_2d: (1e-8), iters: 1000, time: 410.8031ms, avg: 0.4108ms
   out_f32x4_coalesced_read_2d: (1e-8), iters: 1000, time: 466.2976ms, avg: 0.4663ms
  out_f32x4_coalesced_write_2d: (1e-8), iters: 1000, time: 151.9308ms, avg: 0.1519ms
                out_f32_shared: (1e-8), iters: 1000, time: 167.5382ms, avg: 0.1675ms
              out_f32_shared_2: (1e-8), iters: 1000, time: 188.4525ms, avg: 0.1885ms
         out_u8_coalesced_read: (1e-8), iters: 1000, time: 477.6933ms, avg: 0.4777ms
        out_u8_coalesced_write: (1e-8), iters: 1000, time: 218.1919ms, avg: 0.2182ms
       out_u8x4_coalesced_read: (1e-8), iters: 1000, time: 461.1545ms, avg: 0.4612ms
      out_u8x4_coalesced_write: (1e-8), iters: 1000, time: 219.6858ms, avg: 0.2197ms
     out_u8x16_coalesced_write: (1e-8), iters: 1000, time: 221.8468ms, avg: 0.2218ms
    out_u8x4_coalesced_read_2d: (1e-8), iters: 1000, time: 461.4358ms, avg: 0.4614ms
   out_u8x4_coalesced_write_2d: (1e-8), iters: 1000, time: 118.5842ms, avg: 0.1186ms
  out_u8x16_coalesced_write_2d: (1e-8), iters: 1000, time: 34.4331ms, avg: 0.0344ms
                 out_u8_shared: (1e-8), iters: 1000, time: 54.1577ms, avg: 0.0542ms
               out_u8_shared_2: (1e-8), iters: 1000, time: 63.4477ms, avg: 0.0634ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=8192, W=4096
        out_f32_coalesced_read: (1e-8), iters: 1000, time: 942.0230ms, avg: 0.9420ms
       out_f32_coalesced_write: (1e-8), iters: 1000, time: 849.0143ms, avg: 0.8490ms
      out_f32x4_coalesced_read: (1e-8), iters: 1000, time: 941.4093ms, avg: 0.9414ms
     out_f32x4_coalesced_write: (1e-8), iters: 1000, time: 695.4911ms, avg: 0.6955ms
     out_f32_coalesced_read_2d: (1e-8), iters: 1000, time: 944.7675ms, avg: 0.9448ms
    out_f32_coalesced_write_2d: (1e-8), iters: 1000, time: 550.8883ms, avg: 0.5509ms
   out_f32x4_coalesced_read_2d: (1e-8), iters: 1000, time: 939.8100ms, avg: 0.9398ms
  out_f32x4_coalesced_write_2d: (1e-8), iters: 1000, time: 313.0350ms, avg: 0.3130ms
                out_f32_shared: (1e-8), iters: 1000, time: 344.6176ms, avg: 0.3446ms
              out_f32_shared_2: (1e-8), iters: 1000, time: 388.1478ms, avg: 0.3881ms
         out_u8_coalesced_read: (1e-8), iters: 1000, time: 928.4806ms, avg: 0.9285ms
        out_u8_coalesced_write: (1e-8), iters: 1000, time: 422.5574ms, avg: 0.4226ms
       out_u8x4_coalesced_read: (1e-8), iters: 1000, time: 919.8883ms, avg: 0.9199ms
      out_u8x4_coalesced_write: (1e-8), iters: 1000, time: 423.8284ms, avg: 0.4238ms
     out_u8x16_coalesced_write: (1e-8), iters: 1000, time: 430.9692ms, avg: 0.4310ms
    out_u8x4_coalesced_read_2d: (1e-8), iters: 1000, time: 917.5880ms, avg: 0.9176ms
   out_u8x4_coalesced_write_2d: (1e-8), iters: 1000, time: 236.4566ms, avg: 0.2365ms
  out_u8x16_coalesced_write_2d: (1e-8), iters: 1000, time: 67.3809ms, avg: 0.0674ms
                 out_u8_shared: (1e-8), iters: 1000, time: 107.4162ms, avg: 0.1074ms
               out_u8_shared_2: (1e-8), iters: 1000, time: 126.0464ms, avg: 0.1260ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=8192, W=8192
        out_f32_coalesced_read: (1e-8), iters: 1000, time: 1776.4764ms, avg: 1.7765ms
       out_f32_coalesced_write: (1e-8), iters: 1000, time: 1594.6231ms, avg: 1.5946ms
      out_f32x4_coalesced_read: (1e-8), iters: 1000, time: 1774.6072ms, avg: 1.7746ms
     out_f32x4_coalesced_write: (1e-8), iters: 1000, time: 1311.7785ms, avg: 1.3118ms
     out_f32_coalesced_read_2d: (1e-8), iters: 1000, time: 1788.2750ms, avg: 1.7883ms
    out_f32_coalesced_write_2d: (1e-8), iters: 1000, time: 835.3102ms, avg: 0.8353ms
   out_f32x4_coalesced_read_2d: (1e-8), iters: 1000, time: 1774.4243ms, avg: 1.7744ms
  out_f32x4_coalesced_write_2d: (1e-8), iters: 1000, time: 630.1115ms, avg: 0.6301ms
                out_f32_shared: (1e-8), iters: 1000, time: 698.4296ms, avg: 0.6984ms
              out_f32_shared_2: (1e-8), iters: 1000, time: 780.5784ms, avg: 0.7806ms
         out_u8_coalesced_read: (1e-8), iters: 1000, time: 1775.7196ms, avg: 1.7757ms
        out_u8_coalesced_write: (1e-8), iters: 1000, time: 876.3318ms, avg: 0.8763ms
       out_u8x4_coalesced_read: (1e-8), iters: 1000, time: 1772.1145ms, avg: 1.7721ms
      out_u8x4_coalesced_write: (1e-8), iters: 1000, time: 865.2942ms, avg: 0.8653ms
     out_u8x16_coalesced_write: (1e-8), iters: 1000, time: 880.1672ms, avg: 0.8802ms
    out_u8x4_coalesced_read_2d: (1e-8), iters: 1000, time: 1769.2499ms, avg: 1.7692ms
   out_u8x4_coalesced_write_2d: (1e-8), iters: 1000, time: 462.8489ms, avg: 0.4628ms
  out_u8x16_coalesced_write_2d: (1e-8), iters: 1000, time: 150.5156ms, avg: 0.1505ms
                 out_u8_shared: (1e-8), iters: 1000, time: 382.5922ms, avg: 0.3826ms
               out_u8_shared_2: (1e-8), iters: 1000, time: 417.6118ms, avg: 0.4176ms
-------------------------------------------------------------------------------------
```

## FAQ

### 为什么共享内存添加 1 列填充能避免 bank conflict


```
__shared__ float tile[TILE_DIM][TILE_DIM + 1];
```

1. 共享内存Bank结构

    首先了解GPU共享内存的硬件组织结构：

    - 现代GPU（如V100、A100等）通常有 32个bank
    - 每个bank的宽度是 4字节（一个float）
    - 连续的32个4字节地址分别映射到32个不同的bank
    - Bank ID = (地址 / 4) % 32

    ```
    地址:     0    4    8   12   16  ...  124  128  132  136  ...
    Bank:     0    1    2    3    4  ...   31    0    1    2   ...
    ```

2. Bank冲突的产生

    当一个warp（32个线程）中的多个线程同时访问同一个bank的不同地址时，就会发生bank冲突，导致访问被序列化。
    没有填充的情况：

    内存布局分析：

    - tile[0][0] 地址偏移：0 * 32 + 0 = 0 → Bank 0
    - tile[0][1] 地址偏移：0 * 32 + 1 = 1 → Bank 1
    - tile[0][2] 地址偏移：0 * 32 + 2 = 2 → Bank 2
    - ...
    - tile[0][31] 地址偏移：0 * 32 + 31 = 31 → Bank 31

    这种情况下没有冲突，因为每个线程访问不同的bank。

    问题出现在转置访问时：

    内存布局分析：

    - tile[0][0] 地址偏移：0 * 32 + 0 = 0 → Bank 0
    - tile[1][0] 地址偏移：1 * 32 + 0 = 32 → Bank 0 ❌
    - tile[2][0] 地址偏移：2 * 32 + 0 = 64 → Bank 0 ❌
    - ...
    - tile[31][0] 地址偏移：31 * 32 + 0 = 992 → Bank 0 ❌

    结果：所有32个线程都访问Bank 0，产生32路bank冲突！

3. 添加填充的解决方案

    ```
    __shared__ float tile[32][33];  // 添加+1填充
    ```

    现在每行有33个元素而不是32个：

    转置访问时的内存布局：

    - tile[0][0] 地址偏移：0 * 33 + 0 = 0 → Bank 0
    - tile[1][0] 地址偏移：1 * 33 + 0 = 33 → Bank 1 ✅
    - tile[2][0] 地址偏移：2 * 33 + 0 = 66 → Bank 2 ✅
    - tile[3][0] 地址偏移：3 * 33 + 0 = 99 → Bank 3 ✅
    - ...
    - tile[31][0] 地址偏移：31 * 33 + 0 = 1023 → Bank 31 ✅

    计算验证：

    ```
    Bank ID = (地址偏移 / 4) % 32
    tile[i][0] 的 Bank ID = (i * 33) % 32

    i=0: (0 * 33) % 32 = 0
    i=1: (1 * 33) % 32 = 1  
    i=2: (2 * 33) % 32 = 2
    ...
    i=31: (31 * 33) % 32 = 31
    ```

    因为33和32互质，所以(i * 33) % 32会产生0到31的均匀分布！

4. 为什么选择+1而不是其他数值

    关键在于步长与bank数量互质：

    32个bank，步长是33, gcd(33, 32) = 1（互质）, 这保证了连续的行访问会映射到不同的bank

    如果使用其他数值：

    - +2：步长34，gcd(34, 32) = 2，仍会有冲突
    - +0：步长32，gcd(32, 32) = 32，最大冲突
    - +1：步长33，gcd(33, 32) = 1，无冲突 ✅

5. 实际代码中的体现

    ```c++
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // 32x33

    // 读取阶段 - 按行访问，无冲突
    tile[threadIdx.y + j][threadIdx.x] = A[y * W + xIndex];

    // 写入阶段 - 按列访问（转置），由于+1填充，也无冲突
    B[yOut * H + xOut] = tile[threadIdx.x][threadIdx.y + j];
    //                       ^^^^^^^^^^^^^ 这里是转置访问
    ```

6. 性能影响

    - 无填充：32路bank冲突 → 访问时间增加32倍
    - 有填充：无bank冲突 → 共享内存访问达到最大带宽
    - 代价：仅增加3%的共享内存使用量（33/32 ≈ 1.03）

这就是为什么一个简单的+1填充能带来如此显著的性能提升！这是GPU编程中的经典优化技巧，用很小的内存代价换取巨大的性能收益。