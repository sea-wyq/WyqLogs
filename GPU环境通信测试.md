# 基于Nccl-test进行GPU通信测试

## 参考文档
[nccl-test 使用指引-腾讯云开发者社区-腾讯云](https://cloud.tencent.com/developer/article/2361710)

测试示例
```bash
git clone https://github.com/NVIDIA/nccl-tests.git

make -j40

./all_reduce_perf -b 8 -e 512M -f 2 -g 1
# nThread 1 nGpus 1 minBytes 8 maxBytes 536870912 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0

# Using devices
#  Rank  0 Group  0 Pid   1340 on 821a844d28ee device  0 [0x1b] NVIDIA GeForce RTX 3090
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum      -1     4.90    0.00    0.00      0     0.16    0.05    0.00      0
          16             4     float     sum      -1     4.11    0.00    0.00      0     0.16    0.10    0.00      0
          32             8     float     sum      -1     3.93    0.01    0.00      0     0.16    0.20    0.00      0
          64            16     float     sum      -1     4.08    0.02    0.00      0     0.16    0.40    0.00      0
         128            32     float     sum      -1     4.07    0.03    0.00      0     0.16    0.82    0.00      0
         256            64     float     sum      -1     4.12    0.06    0.00      0     0.16    1.59    0.00      0
         512           128     float     sum      -1     3.98    0.13    0.00      0     0.16    3.24    0.00      0
        1024           256     float     sum      -1     4.06    0.25    0.00      0     0.16    6.33    0.00      0
        2048           512     float     sum      -1     4.00    0.51    0.00      0     0.16   12.91    0.00      0
        4096          1024     float     sum      -1     4.14    0.99    0.00      0     0.17   24.69    0.00      0
        8192          2048     float     sum      -1     4.05    2.02    0.00      0     0.16   51.26    0.00      0
       16384          4096     float     sum      -1     4.03    4.06    0.00      0     0.16  100.82    0.00      0
       32768          8192     float     sum      -1     4.04    8.11    0.00      0     0.16  204.93    0.00      0
       65536         16384     float     sum      -1     4.06   16.13    0.00      0     0.16  415.84    0.00      0
      131072         32768     float     sum      -1     4.05   32.36    0.00      0     0.16  830.62    0.00      0
      262144         65536     float     sum      -1     4.17   62.80    0.00      0     0.16  1641.48    0.00      0
      524288        131072     float     sum      -1     4.21  124.58    0.00      0     0.16  3207.64    0.00      0
     1048576        262144     float     sum      -1     5.07  206.91    0.00      0     0.16  6559.75    0.00      0
     2097152        524288     float     sum      -1     7.92  264.89    0.00      0     0.16  13197.94    0.00      0
     4194304       1048576     float     sum      -1    12.67  331.05    0.00      0     0.16  25653.24    0.00      0
     8388608       2097152     float     sum      -1    22.95  365.58    0.00      0     0.16  53244.10    0.00      0
    16777216       4194304     float     sum      -1    42.93  390.79    0.00      0     0.16  104368.37    0.00      0
    33554432       8388608     float     sum      -1    82.83  405.08    0.00      0     0.16  213450.59    0.00      0
    67108864      16777216     float     sum      -1    162.4  413.12    0.00      0     0.16  410451.77    0.00      0
   134217728      33554432     float     sum      -1    321.2  417.81    0.00      0     0.16  830298.35    0.00      0
   268435456      67108864     float     sum      -1    638.5  420.41    0.00      0     0.16  1675104.25    0.00      0
   536870912     134217728     float     sum      -1   1274.7  421.17    0.00      0     0.17  3253763.10    0.00      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 0
```