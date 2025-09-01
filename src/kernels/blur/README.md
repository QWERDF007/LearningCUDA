# Blur

## 介绍

包含以下内容：
- [x] blur_u8_kernel
- [x] blur_u8_nb_kernel (无分支版本)
- [x] blur_u8_kernel_shared (共享内存版本)
- [x] blur_u8_h_kernel + blur_u8_v_kernel (分离横向/纵向版本)
- [x] blur_u8_h_shared + blur_u8_v_shared (分离横向/纵向，共享内存版本)
- [x] blur_u8_v_shared_column (分离纵向，特殊网格版本)

## 测试

硬件：

- cpu: Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
- gpu: NVIDIA GeForce RTX 4090 D

运行：

```bash
# 只测试Ada架构 不指定默认编译所有架构 耗时较长: Volta, Ampere, Ada, Hopper, ...
export TORCH_CUDA_ARCH_LIST=Ada
python3 blur.py
```

输出：

```
-------------------------------------------------------------------------------------
                                        H=1024, W=1024, ksz=3
              out_u8: (True), iters: 1000, time: 7.0240ms, avg: 0.0070ms
    out_u8_no_branch: (True), iters: 1000, time: 6.7143ms, avg: 0.0067ms
       out_u8_shared: (True), iters: 1000, time: 8.8716ms, avg: 0.0089ms
        out_u8_split: (True), iters: 1000, time: 10.6962ms, avg: 0.0107ms
 out_u8_split_shared: (True), iters: 1000, time: 56.6301ms, avg: 0.0566ms
out_u8_split_shared2: (True), iters: 1000, time: 22.9008ms, avg: 0.0229ms
     out_u8_split_sw: (True), iters: 1000, time: 766.0849ms, avg: 0.7661ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1024, W=1024, ksz=5
              out_u8: (True), iters: 1000, time: 18.0686ms, avg: 0.0181ms
    out_u8_no_branch: (True), iters: 1000, time: 16.8228ms, avg: 0.0168ms
       out_u8_shared: (True), iters: 1000, time: 12.7451ms, avg: 0.0127ms
        out_u8_split: (True), iters: 1000, time: 10.9944ms, avg: 0.0110ms
 out_u8_split_shared: (True), iters: 1000, time: 57.6451ms, avg: 0.0576ms
out_u8_split_shared2: (True), iters: 1000, time: 23.4861ms, avg: 0.0235ms
     out_u8_split_sw: (True), iters: 1000, time: 767.2396ms, avg: 0.7672ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1024, W=1024, ksz=7
              out_u8: (True), iters: 1000, time: 23.7989ms, avg: 0.0238ms
    out_u8_no_branch: (True), iters: 1000, time: 22.1550ms, avg: 0.0222ms
       out_u8_shared: (True), iters: 1000, time: 16.3836ms, avg: 0.0164ms
        out_u8_split: (True), iters: 1000, time: 13.7603ms, avg: 0.0138ms
 out_u8_split_shared: (True), iters: 1000, time: 59.1938ms, avg: 0.0592ms
out_u8_split_shared2: (True), iters: 1000, time: 24.0409ms, avg: 0.0240ms
     out_u8_split_sw: (True), iters: 1000, time: 771.8940ms, avg: 0.7719ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1024, W=1024, ksz=9
              out_u8: (True), iters: 1000, time: 45.9714ms, avg: 0.0460ms
    out_u8_no_branch: (True), iters: 1000, time: 42.1381ms, avg: 0.0421ms
       out_u8_shared: (True), iters: 1000, time: 21.8213ms, avg: 0.0218ms
        out_u8_split: (True), iters: 1000, time: 14.5705ms, avg: 0.0146ms
 out_u8_split_shared: (True), iters: 1000, time: 58.9683ms, avg: 0.0590ms
out_u8_split_shared2: (True), iters: 1000, time: 24.6251ms, avg: 0.0246ms
     out_u8_split_sw: (True), iters: 1000, time: 773.9308ms, avg: 0.7739ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1024, W=1024, ksz=15
              out_u8: (True), iters: 1000, time: 101.7179ms, avg: 0.1017ms
    out_u8_no_branch: (True), iters: 1000, time: 93.1880ms, avg: 0.0932ms
       out_u8_shared: (True), iters: 1000, time: 53.3671ms, avg: 0.0534ms
        out_u8_split: (True), iters: 1000, time: 20.9394ms, avg: 0.0209ms
 out_u8_split_shared: (True), iters: 1000, time: 62.6209ms, avg: 0.0626ms
out_u8_split_shared2: (True), iters: 1000, time: 25.7583ms, avg: 0.0258ms
     out_u8_split_sw: (True), iters: 1000, time: 777.6799ms, avg: 0.7777ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1024, W=2048, ksz=3
              out_u8: (True), iters: 1000, time: 12.3549ms, avg: 0.0124ms
    out_u8_no_branch: (True), iters: 1000, time: 11.7490ms, avg: 0.0117ms
       out_u8_shared: (True), iters: 1000, time: 16.1855ms, avg: 0.0162ms
        out_u8_split: (True), iters: 1000, time: 18.3458ms, avg: 0.0183ms
 out_u8_split_shared: (True), iters: 1000, time: 117.6293ms, avg: 0.1176ms
out_u8_split_shared2: (True), iters: 1000, time: 42.5143ms, avg: 0.0425ms
     out_u8_split_sw: (True), iters: 1000, time: 1478.9169ms, avg: 1.4789ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1024, W=2048, ksz=5
              out_u8: (True), iters: 1000, time: 34.1074ms, avg: 0.0341ms
    out_u8_no_branch: (True), iters: 1000, time: 31.5685ms, avg: 0.0316ms
       out_u8_shared: (True), iters: 1000, time: 23.5388ms, avg: 0.0235ms
        out_u8_split: (True), iters: 1000, time: 18.7693ms, avg: 0.0188ms
 out_u8_split_shared: (True), iters: 1000, time: 120.3384ms, avg: 0.1203ms
out_u8_split_shared2: (True), iters: 1000, time: 43.7579ms, avg: 0.0438ms
     out_u8_split_sw: (True), iters: 1000, time: 1480.1021ms, avg: 1.4801ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1024, W=2048, ksz=7
              out_u8: (True), iters: 1000, time: 45.3255ms, avg: 0.0453ms
    out_u8_no_branch: (True), iters: 1000, time: 41.9564ms, avg: 0.0420ms
       out_u8_shared: (True), iters: 1000, time: 30.7453ms, avg: 0.0307ms
        out_u8_split: (True), iters: 1000, time: 24.1432ms, avg: 0.0241ms
 out_u8_split_shared: (True), iters: 1000, time: 123.1143ms, avg: 0.1231ms
out_u8_split_shared2: (True), iters: 1000, time: 44.7843ms, avg: 0.0448ms
     out_u8_split_sw: (True), iters: 1000, time: 1481.5454ms, avg: 1.4815ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1024, W=2048, ksz=9
              out_u8: (True), iters: 1000, time: 89.2954ms, avg: 0.0893ms
    out_u8_no_branch: (True), iters: 1000, time: 81.8710ms, avg: 0.0819ms
       out_u8_shared: (True), iters: 1000, time: 41.6081ms, avg: 0.0416ms
        out_u8_split: (True), iters: 1000, time: 26.0365ms, avg: 0.0260ms
 out_u8_split_shared: (True), iters: 1000, time: 123.6696ms, avg: 0.1237ms
out_u8_split_shared2: (True), iters: 1000, time: 45.8808ms, avg: 0.0459ms
     out_u8_split_sw: (True), iters: 1000, time: 1485.5576ms, avg: 1.4856ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1024, W=2048, ksz=15
              out_u8: (True), iters: 1000, time: 200.5296ms, avg: 0.2005ms
    out_u8_no_branch: (True), iters: 1000, time: 183.2490ms, avg: 0.1832ms
       out_u8_shared: (True), iters: 1000, time: 104.5363ms, avg: 0.1045ms
        out_u8_split: (True), iters: 1000, time: 38.3086ms, avg: 0.0383ms
 out_u8_split_shared: (True), iters: 1000, time: 131.6845ms, avg: 0.1317ms
out_u8_split_shared2: (True), iters: 1000, time: 48.1763ms, avg: 0.0482ms
     out_u8_split_sw: (True), iters: 1000, time: 1490.6137ms, avg: 1.4906ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1024, W=4096, ksz=3
              out_u8: (True), iters: 1000, time: 22.9664ms, avg: 0.0230ms
    out_u8_no_branch: (True), iters: 1000, time: 21.8837ms, avg: 0.0219ms
       out_u8_shared: (True), iters: 1000, time: 30.5402ms, avg: 0.0305ms
        out_u8_split: (True), iters: 1000, time: 33.5724ms, avg: 0.0336ms
 out_u8_split_shared: (True), iters: 1000, time: 243.1996ms, avg: 0.2432ms
out_u8_split_shared2: (True), iters: 1000, time: 81.5749ms, avg: 0.0816ms
     out_u8_split_sw: (True), iters: 1000, time: 2876.7912ms, avg: 2.8768ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1024, W=4096, ksz=5
              out_u8: (True), iters: 1000, time: 66.1345ms, avg: 0.0661ms
    out_u8_no_branch: (True), iters: 1000, time: 60.9634ms, avg: 0.0610ms
       out_u8_shared: (True), iters: 1000, time: 45.2278ms, avg: 0.0452ms
        out_u8_split: (True), iters: 1000, time: 34.4296ms, avg: 0.0344ms
 out_u8_split_shared: (True), iters: 1000, time: 245.9643ms, avg: 0.2460ms
out_u8_split_shared2: (True), iters: 1000, time: 84.2175ms, avg: 0.0842ms
     out_u8_split_sw: (True), iters: 1000, time: 2877.8455ms, avg: 2.8778ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1024, W=4096, ksz=7
              out_u8: (True), iters: 1000, time: 88.1782ms, avg: 0.0882ms
    out_u8_no_branch: (True), iters: 1000, time: 81.6429ms, avg: 0.0816ms
       out_u8_shared: (True), iters: 1000, time: 59.5973ms, avg: 0.0596ms
        out_u8_split: (True), iters: 1000, time: 45.0397ms, avg: 0.0450ms
 out_u8_split_shared: (True), iters: 1000, time: 253.1040ms, avg: 0.2531ms
out_u8_split_shared2: (True), iters: 1000, time: 86.2761ms, avg: 0.0863ms
     out_u8_split_sw: (True), iters: 1000, time: 2880.7309ms, avg: 2.8807ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1024, W=4096, ksz=9
              out_u8: (True), iters: 1000, time: 176.6105ms, avg: 0.1766ms
    out_u8_no_branch: (True), iters: 1000, time: 160.9933ms, avg: 0.1610ms
       out_u8_shared: (True), iters: 1000, time: 80.9531ms, avg: 0.0810ms
        out_u8_split: (True), iters: 1000, time: 48.8508ms, avg: 0.0489ms
 out_u8_split_shared: (True), iters: 1000, time: 258.2593ms, avg: 0.2583ms
out_u8_split_shared2: (True), iters: 1000, time: 88.6145ms, avg: 0.0886ms
     out_u8_split_sw: (True), iters: 1000, time: 2886.3072ms, avg: 2.8863ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=1024, W=4096, ksz=15
              out_u8: (True), iters: 1000, time: 399.3309ms, avg: 0.3993ms
    out_u8_no_branch: (True), iters: 1000, time: 363.9925ms, avg: 0.3640ms
       out_u8_shared: (True), iters: 1000, time: 206.9302ms, avg: 0.2069ms
        out_u8_split: (True), iters: 1000, time: 73.0739ms, avg: 0.0731ms
 out_u8_split_shared: (True), iters: 1000, time: 273.1771ms, avg: 0.2732ms
out_u8_split_shared2: (True), iters: 1000, time: 93.1060ms, avg: 0.0931ms
     out_u8_split_sw: (True), iters: 1000, time: 2893.8594ms, avg: 2.8939ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=1024, ksz=3
              out_u8: (True), iters: 1000, time: 12.3904ms, avg: 0.0124ms
    out_u8_no_branch: (True), iters: 1000, time: 11.7741ms, avg: 0.0118ms
       out_u8_shared: (True), iters: 1000, time: 16.2106ms, avg: 0.0162ms
        out_u8_split: (True), iters: 1000, time: 18.2545ms, avg: 0.0183ms
 out_u8_split_shared: (True), iters: 1000, time: 109.0198ms, avg: 0.1090ms
out_u8_split_shared2: (True), iters: 1000, time: 42.4781ms, avg: 0.0425ms
     out_u8_split_sw: (True), iters: 1000, time: 830.2641ms, avg: 0.8303ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=1024, ksz=5
              out_u8: (True), iters: 1000, time: 34.2507ms, avg: 0.0343ms
    out_u8_no_branch: (True), iters: 1000, time: 31.7137ms, avg: 0.0317ms
       out_u8_shared: (True), iters: 1000, time: 23.5300ms, avg: 0.0235ms
        out_u8_split: (True), iters: 1000, time: 18.7683ms, avg: 0.0188ms
 out_u8_split_shared: (True), iters: 1000, time: 111.3455ms, avg: 0.1113ms
out_u8_split_shared2: (True), iters: 1000, time: 43.7107ms, avg: 0.0437ms
     out_u8_split_sw: (True), iters: 1000, time: 830.7984ms, avg: 0.8308ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=1024, ksz=7
              out_u8: (True), iters: 1000, time: 45.4955ms, avg: 0.0455ms
    out_u8_no_branch: (True), iters: 1000, time: 42.1538ms, avg: 0.0422ms
       out_u8_shared: (True), iters: 1000, time: 30.7903ms, avg: 0.0308ms
        out_u8_split: (True), iters: 1000, time: 24.2193ms, avg: 0.0242ms
 out_u8_split_shared: (True), iters: 1000, time: 114.1438ms, avg: 0.1141ms
out_u8_split_shared2: (True), iters: 1000, time: 44.8039ms, avg: 0.0448ms
     out_u8_split_sw: (True), iters: 1000, time: 830.6475ms, avg: 0.8306ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=1024, ksz=9
              out_u8: (True), iters: 1000, time: 89.3805ms, avg: 0.0894ms
    out_u8_no_branch: (True), iters: 1000, time: 81.9664ms, avg: 0.0820ms
       out_u8_shared: (True), iters: 1000, time: 41.6038ms, avg: 0.0416ms
        out_u8_split: (True), iters: 1000, time: 26.0415ms, avg: 0.0260ms
 out_u8_split_shared: (True), iters: 1000, time: 114.0401ms, avg: 0.1140ms
out_u8_split_shared2: (True), iters: 1000, time: 46.0670ms, avg: 0.0461ms
     out_u8_split_sw: (True), iters: 1000, time: 833.7805ms, avg: 0.8338ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=1024, ksz=15
              out_u8: (True), iters: 1000, time: 201.2300ms, avg: 0.2012ms
    out_u8_no_branch: (True), iters: 1000, time: 183.5747ms, avg: 0.1836ms
       out_u8_shared: (True), iters: 1000, time: 104.5282ms, avg: 0.1045ms
        out_u8_split: (True), iters: 1000, time: 38.4371ms, avg: 0.0384ms
 out_u8_split_shared: (True), iters: 1000, time: 121.7790ms, avg: 0.1218ms
out_u8_split_shared2: (True), iters: 1000, time: 48.2063ms, avg: 0.0482ms
     out_u8_split_sw: (True), iters: 1000, time: 837.0440ms, avg: 0.8370ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=2048, ksz=3
              out_u8: (True), iters: 1000, time: 22.9959ms, avg: 0.0230ms
    out_u8_no_branch: (True), iters: 1000, time: 21.9204ms, avg: 0.0219ms
       out_u8_shared: (True), iters: 1000, time: 30.6046ms, avg: 0.0306ms
        out_u8_split: (True), iters: 1000, time: 33.4573ms, avg: 0.0335ms
 out_u8_split_shared: (True), iters: 1000, time: 231.7469ms, avg: 0.2317ms
out_u8_split_shared2: (True), iters: 1000, time: 81.6226ms, avg: 0.0816ms
     out_u8_split_sw: (True), iters: 1000, time: 1538.0924ms, avg: 1.5381ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=2048, ksz=5
              out_u8: (True), iters: 1000, time: 66.2782ms, avg: 0.0663ms
    out_u8_no_branch: (True), iters: 1000, time: 61.3632ms, avg: 0.0614ms
       out_u8_shared: (True), iters: 1000, time: 45.1415ms, avg: 0.0451ms
        out_u8_split: (True), iters: 1000, time: 34.3816ms, avg: 0.0344ms
 out_u8_split_shared: (True), iters: 1000, time: 235.5702ms, avg: 0.2356ms
out_u8_split_shared2: (True), iters: 1000, time: 84.2397ms, avg: 0.0842ms
     out_u8_split_sw: (True), iters: 1000, time: 1541.9707ms, avg: 1.5420ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=2048, ksz=7
              out_u8: (True), iters: 1000, time: 88.5870ms, avg: 0.0886ms
    out_u8_no_branch: (True), iters: 1000, time: 81.8236ms, avg: 0.0818ms
       out_u8_shared: (True), iters: 1000, time: 59.6316ms, avg: 0.0596ms
        out_u8_split: (True), iters: 1000, time: 45.1095ms, avg: 0.0451ms
 out_u8_split_shared: (True), iters: 1000, time: 242.1968ms, avg: 0.2422ms
out_u8_split_shared2: (True), iters: 1000, time: 86.2713ms, avg: 0.0863ms
     out_u8_split_sw: (True), iters: 1000, time: 1543.5412ms, avg: 1.5435ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=2048, ksz=9
              out_u8: (True), iters: 1000, time: 176.7671ms, avg: 0.1768ms
    out_u8_no_branch: (True), iters: 1000, time: 161.4082ms, avg: 0.1614ms
       out_u8_shared: (True), iters: 1000, time: 81.0869ms, avg: 0.0811ms
        out_u8_split: (True), iters: 1000, time: 48.9223ms, avg: 0.0489ms
 out_u8_split_shared: (True), iters: 1000, time: 244.1845ms, avg: 0.2442ms
out_u8_split_shared2: (True), iters: 1000, time: 88.7425ms, avg: 0.0887ms
     out_u8_split_sw: (True), iters: 1000, time: 1546.7856ms, avg: 1.5468ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=2048, ksz=15
              out_u8: (True), iters: 1000, time: 399.5068ms, avg: 0.3995ms
    out_u8_no_branch: (True), iters: 1000, time: 364.2001ms, avg: 0.3642ms
       out_u8_shared: (True), iters: 1000, time: 206.8663ms, avg: 0.2069ms
        out_u8_split: (True), iters: 1000, time: 73.2396ms, avg: 0.0732ms
 out_u8_split_shared: (True), iters: 1000, time: 260.0300ms, avg: 0.2600ms
out_u8_split_shared2: (True), iters: 1000, time: 93.1625ms, avg: 0.0932ms
     out_u8_split_sw: (True), iters: 1000, time: 1550.8018ms, avg: 1.5508ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=4096, ksz=3
              out_u8: (True), iters: 1000, time: 44.2002ms, avg: 0.0442ms
    out_u8_no_branch: (True), iters: 1000, time: 42.1576ms, avg: 0.0422ms
       out_u8_shared: (True), iters: 1000, time: 59.1135ms, avg: 0.0591ms
        out_u8_split: (True), iters: 1000, time: 64.1286ms, avg: 0.0641ms
 out_u8_split_shared: (True), iters: 1000, time: 481.2815ms, avg: 0.4813ms
out_u8_split_shared2: (True), iters: 1000, time: 159.7438ms, avg: 0.1597ms
     out_u8_split_sw: (True), iters: 1000, time: 2942.3006ms, avg: 2.9423ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=4096, ksz=5
              out_u8: (True), iters: 1000, time: 130.8494ms, avg: 0.1308ms
    out_u8_no_branch: (True), iters: 1000, time: 120.4791ms, avg: 0.1205ms
       out_u8_shared: (True), iters: 1000, time: 88.5913ms, avg: 0.0886ms
        out_u8_split: (True), iters: 1000, time: 65.7640ms, avg: 0.0658ms
 out_u8_split_shared: (True), iters: 1000, time: 488.9462ms, avg: 0.4889ms
out_u8_split_shared2: (True), iters: 1000, time: 165.1750ms, avg: 0.1652ms
     out_u8_split_sw: (True), iters: 1000, time: 2942.7092ms, avg: 2.9427ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=4096, ksz=7
              out_u8: (True), iters: 1000, time: 174.8972ms, avg: 0.1749ms
    out_u8_no_branch: (True), iters: 1000, time: 161.3176ms, avg: 0.1613ms
       out_u8_shared: (True), iters: 1000, time: 116.9660ms, avg: 0.1170ms
        out_u8_split: (True), iters: 1000, time: 86.8442ms, avg: 0.0868ms
 out_u8_split_shared: (True), iters: 1000, time: 502.2018ms, avg: 0.5022ms
out_u8_split_shared2: (True), iters: 1000, time: 169.2915ms, avg: 0.1693ms
     out_u8_split_sw: (True), iters: 1000, time: 2945.1621ms, avg: 2.9452ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=4096, ksz=9
              out_u8: (True), iters: 1000, time: 351.5120ms, avg: 0.3515ms
    out_u8_no_branch: (True), iters: 1000, time: 320.5733ms, avg: 0.3206ms
       out_u8_shared: (True), iters: 1000, time: 160.0077ms, avg: 0.1600ms
        out_u8_split: (True), iters: 1000, time: 94.5706ms, avg: 0.0946ms
 out_u8_split_shared: (True), iters: 1000, time: 510.1144ms, avg: 0.5101ms
out_u8_split_shared2: (True), iters: 1000, time: 174.1557ms, avg: 0.1742ms
     out_u8_split_sw: (True), iters: 1000, time: 2950.8340ms, avg: 2.9508ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=2048, W=4096, ksz=15
              out_u8: (True), iters: 1000, time: 797.3354ms, avg: 0.7973ms
    out_u8_no_branch: (True), iters: 1000, time: 725.6138ms, avg: 0.7256ms
       out_u8_shared: (True), iters: 1000, time: 411.2606ms, avg: 0.4113ms
        out_u8_split: (True), iters: 1000, time: 142.9851ms, avg: 0.1430ms
 out_u8_split_shared: (True), iters: 1000, time: 541.5719ms, avg: 0.5416ms
out_u8_split_shared2: (True), iters: 1000, time: 182.7369ms, avg: 0.1827ms
     out_u8_split_sw: (True), iters: 1000, time: 2958.9097ms, avg: 2.9589ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=1024, ksz=3
              out_u8: (True), iters: 1000, time: 23.0384ms, avg: 0.0230ms
    out_u8_no_branch: (True), iters: 1000, time: 21.8897ms, avg: 0.0219ms
       out_u8_shared: (True), iters: 1000, time: 30.5364ms, avg: 0.0305ms
        out_u8_split: (True), iters: 1000, time: 33.3359ms, avg: 0.0333ms
 out_u8_split_shared: (True), iters: 1000, time: 213.7077ms, avg: 0.2137ms
out_u8_split_shared2: (True), iters: 1000, time: 81.6131ms, avg: 0.0816ms
     out_u8_split_sw: (True), iters: 1000, time: 946.2883ms, avg: 0.9463ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=1024, ksz=5
              out_u8: (True), iters: 1000, time: 66.3941ms, avg: 0.0664ms
    out_u8_no_branch: (True), iters: 1000, time: 61.4479ms, avg: 0.0614ms
       out_u8_shared: (True), iters: 1000, time: 45.1558ms, avg: 0.0452ms
        out_u8_split: (True), iters: 1000, time: 34.3935ms, avg: 0.0344ms
 out_u8_split_shared: (True), iters: 1000, time: 220.1402ms, avg: 0.2201ms
out_u8_split_shared2: (True), iters: 1000, time: 84.3096ms, avg: 0.0843ms
     out_u8_split_sw: (True), iters: 1000, time: 947.5827ms, avg: 0.9476ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=1024, ksz=7
              out_u8: (True), iters: 1000, time: 88.6929ms, avg: 0.0887ms
    out_u8_no_branch: (True), iters: 1000, time: 82.0639ms, avg: 0.0821ms
       out_u8_shared: (True), iters: 1000, time: 59.5949ms, avg: 0.0596ms
        out_u8_split: (True), iters: 1000, time: 45.2366ms, avg: 0.0452ms
 out_u8_split_shared: (True), iters: 1000, time: 222.9571ms, avg: 0.2230ms
out_u8_split_shared2: (True), iters: 1000, time: 86.2808ms, avg: 0.0863ms
     out_u8_split_sw: (True), iters: 1000, time: 948.7717ms, avg: 0.9488ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=1024, ksz=9
              out_u8: (True), iters: 1000, time: 177.2397ms, avg: 0.1772ms
    out_u8_no_branch: (True), iters: 1000, time: 161.6333ms, avg: 0.1616ms
       out_u8_shared: (True), iters: 1000, time: 81.0943ms, avg: 0.0811ms
        out_u8_split: (True), iters: 1000, time: 49.0408ms, avg: 0.0490ms
 out_u8_split_shared: (True), iters: 1000, time: 225.8837ms, avg: 0.2259ms
out_u8_split_shared2: (True), iters: 1000, time: 88.6996ms, avg: 0.0887ms
     out_u8_split_sw: (True), iters: 1000, time: 950.9752ms, avg: 0.9510ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=1024, ksz=15
              out_u8: (True), iters: 1000, time: 399.9743ms, avg: 0.4000ms
    out_u8_no_branch: (True), iters: 1000, time: 365.1247ms, avg: 0.3651ms
       out_u8_shared: (True), iters: 1000, time: 206.9628ms, avg: 0.2070ms
        out_u8_split: (True), iters: 1000, time: 73.3151ms, avg: 0.0733ms
 out_u8_split_shared: (True), iters: 1000, time: 238.4338ms, avg: 0.2384ms
out_u8_split_shared2: (True), iters: 1000, time: 93.1132ms, avg: 0.0931ms
     out_u8_split_sw: (True), iters: 1000, time: 955.7900ms, avg: 0.9558ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=2048, ksz=3
              out_u8: (True), iters: 1000, time: 44.3442ms, avg: 0.0443ms
    out_u8_no_branch: (True), iters: 1000, time: 42.2766ms, avg: 0.0423ms
       out_u8_shared: (True), iters: 1000, time: 59.2666ms, avg: 0.0593ms
        out_u8_split: (True), iters: 1000, time: 63.8802ms, avg: 0.0639ms
 out_u8_split_shared: (True), iters: 1000, time: 458.0774ms, avg: 0.4581ms
out_u8_split_shared2: (True), iters: 1000, time: 159.9045ms, avg: 0.1599ms
     out_u8_split_sw: (True), iters: 1000, time: 1657.8496ms, avg: 1.6578ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=2048, ksz=5
              out_u8: (True), iters: 1000, time: 131.0904ms, avg: 0.1311ms
    out_u8_no_branch: (True), iters: 1000, time: 120.6734ms, avg: 0.1207ms
       out_u8_shared: (True), iters: 1000, time: 88.3048ms, avg: 0.0883ms
        out_u8_split: (True), iters: 1000, time: 66.1216ms, avg: 0.0661ms
 out_u8_split_shared: (True), iters: 1000, time: 466.6293ms, avg: 0.4666ms
out_u8_split_shared2: (True), iters: 1000, time: 165.3550ms, avg: 0.1654ms
     out_u8_split_sw: (True), iters: 1000, time: 1658.7393ms, avg: 1.6587ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=2048, ksz=7
              out_u8: (True), iters: 1000, time: 174.9291ms, avg: 0.1749ms
    out_u8_no_branch: (True), iters: 1000, time: 161.7401ms, avg: 0.1617ms
       out_u8_shared: (True), iters: 1000, time: 116.9875ms, avg: 0.1170ms
        out_u8_split: (True), iters: 1000, time: 87.0867ms, avg: 0.0871ms
 out_u8_split_shared: (True), iters: 1000, time: 477.5729ms, avg: 0.4776ms
out_u8_split_shared2: (True), iters: 1000, time: 169.3716ms, avg: 0.1694ms
     out_u8_split_sw: (True), iters: 1000, time: 1660.5041ms, avg: 1.6605ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=2048, ksz=9
              out_u8: (True), iters: 1000, time: 352.0899ms, avg: 0.3521ms
    out_u8_no_branch: (True), iters: 1000, time: 321.1102ms, avg: 0.3211ms
       out_u8_shared: (True), iters: 1000, time: 160.0823ms, avg: 0.1601ms
        out_u8_split: (True), iters: 1000, time: 94.5907ms, avg: 0.0946ms
 out_u8_split_shared: (True), iters: 1000, time: 484.3283ms, avg: 0.4843ms
out_u8_split_shared2: (True), iters: 1000, time: 174.3824ms, avg: 0.1744ms
     out_u8_split_sw: (True), iters: 1000, time: 1663.8608ms, avg: 1.6639ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=2048, ksz=15
              out_u8: (True), iters: 1000, time: 798.5582ms, avg: 0.7986ms
    out_u8_no_branch: (True), iters: 1000, time: 725.9793ms, avg: 0.7260ms
       out_u8_shared: (True), iters: 1000, time: 412.0119ms, avg: 0.4120ms
        out_u8_split: (True), iters: 1000, time: 143.1255ms, avg: 0.1431ms
 out_u8_split_shared: (True), iters: 1000, time: 509.1536ms, avg: 0.5092ms
out_u8_split_shared2: (True), iters: 1000, time: 182.7912ms, avg: 0.1828ms
     out_u8_split_sw: (True), iters: 1000, time: 1670.9468ms, avg: 1.6709ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=4096, ksz=3
              out_u8: (True), iters: 1000, time: 86.9625ms, avg: 0.0870ms
    out_u8_no_branch: (True), iters: 1000, time: 82.9599ms, avg: 0.0830ms
       out_u8_shared: (True), iters: 1000, time: 116.3714ms, avg: 0.1164ms
        out_u8_split: (True), iters: 1000, time: 205.3022ms, avg: 0.2053ms
 out_u8_split_shared: (True), iters: 1000, time: 990.1118ms, avg: 0.9901ms
out_u8_split_shared2: (True), iters: 1000, time: 361.0995ms, avg: 0.3611ms
     out_u8_split_sw: (True), iters: 1000, time: 3201.8299ms, avg: 3.2018ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=4096, ksz=5
              out_u8: (True), iters: 1000, time: 259.6228ms, avg: 0.2596ms
    out_u8_no_branch: (True), iters: 1000, time: 239.3603ms, avg: 0.2394ms
       out_u8_shared: (True), iters: 1000, time: 174.6793ms, avg: 0.1747ms
        out_u8_split: (True), iters: 1000, time: 205.2400ms, avg: 0.2052ms
 out_u8_split_shared: (True), iters: 1000, time: 1006.5382ms, avg: 1.0065ms
out_u8_split_shared2: (True), iters: 1000, time: 367.1811ms, avg: 0.3672ms
     out_u8_split_sw: (True), iters: 1000, time: 3208.1208ms, avg: 3.2081ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=4096, ksz=7
              out_u8: (True), iters: 1000, time: 348.3889ms, avg: 0.3484ms
    out_u8_no_branch: (True), iters: 1000, time: 321.6810ms, avg: 0.3217ms
       out_u8_shared: (True), iters: 1000, time: 231.8807ms, avg: 0.2319ms
        out_u8_split: (True), iters: 1000, time: 239.3751ms, avg: 0.2394ms
 out_u8_split_shared: (True), iters: 1000, time: 1034.5507ms, avg: 1.0346ms
out_u8_split_shared2: (True), iters: 1000, time: 375.6738ms, avg: 0.3757ms
     out_u8_split_sw: (True), iters: 1000, time: 3206.3842ms, avg: 3.2064ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=4096, ksz=9
              out_u8: (True), iters: 1000, time: 701.5305ms, avg: 0.7015ms
    out_u8_no_branch: (True), iters: 1000, time: 638.8874ms, avg: 0.6389ms
       out_u8_shared: (True), iters: 1000, time: 317.7936ms, avg: 0.3178ms
        out_u8_split: (True), iters: 1000, time: 240.3796ms, avg: 0.2404ms
 out_u8_split_shared: (True), iters: 1000, time: 1045.8872ms, avg: 1.0459ms
out_u8_split_shared2: (True), iters: 1000, time: 378.6693ms, avg: 0.3787ms
     out_u8_split_sw: (True), iters: 1000, time: 3214.7088ms, avg: 3.2147ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        H=4096, W=4096, ksz=15
              out_u8: (True), iters: 1000, time: 1594.2912ms, avg: 1.5943ms
    out_u8_no_branch: (True), iters: 1000, time: 1450.9280ms, avg: 1.4509ms
       out_u8_shared: (True), iters: 1000, time: 821.8784ms, avg: 0.8219ms
        out_u8_split: (True), iters: 1000, time: 321.5244ms, avg: 0.3215ms
 out_u8_split_shared: (True), iters: 1000, time: 1102.1492ms, avg: 1.1021ms
out_u8_split_shared2: (True), iters: 1000, time: 393.1880ms, avg: 0.3932ms
     out_u8_split_sw: (True), iters: 1000, time: 3219.1825ms, avg: 3.2192ms
-------------------------------------------------------------------------------------
```

## FAQ

### 为什么滑动窗口 (u8_split_sw) 版本的比其他版本慢很多？

1. 严重的并行度不足：只能并行处理 H+W 个任务，而 GPU 设计用来处理数万个并行任务
2. 强制串行化：滑动窗口算法要求按顺序处理，每个线程处理一整行/列，无法充分利用 GPU 的并行性
3. GPU资源浪费：大量的计算单元处于空闲状态

这是一个典型的"算法理论优秀但不适合GPU架构"的例子。在 CPU 上滑动窗口可能更有优势，但在 GPU 上，简单的并行算法往往表现更好，因为它们能更好地利用 GPU 的大规模并行能力。

### 为什么分离 (u8_split) 版本比其他版本快？

1. 算法优势：分离卷积将 $O({ksz}^2)$ 复杂度降为 $O(ksz)$
2. 适合 GPU 架构：保持了全并行性，每个线程处理一个像素
3. 内存访问友好：横向滤波是连续访问，纵向滤波计算量小
4. 实现简洁：避免了复杂的优化带来的额外开销

在 GPU 上，算法复杂度的降低往往比内存访问优化更重要。分离卷积是一个完美的例子，通过数学分解实现了显著的性能提升。

### 为什么 out_u8_shared 比 out_u8_split 慢很多？

1. 算法复杂度碾压：O(ksz²) vs O(ksz×2)，这是决定性因素
2. 现代GPU架构：全局内存已经足够快，共享内存优势不明显
3. 实现复杂性：共享内存版本的管理开销抵消了访问速度优势