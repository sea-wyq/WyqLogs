# 基于Nccl-test进行GPU通信测试


如果出现nccl.h找不到，需要安装nccl库。

1. 检查依赖项

   sudo ldconfig  # 更新缓存，使系统能找到新添加的库
   ldd ./all_reduce_perf | grep nccl  

   ![image-2025-06-11](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-2025-06-11.png)
   ![image-1-2025-06-11](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-1-2025-06-11.png)

2. nccl 安装

   安装地址：https://developer.nvidia.com/nccl/nccl-download

   ```bash
   Network Installer for Ubuntu22.04

   $ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   $ sudo dpkg -i cuda-keyring_1.1-1_all.deb
   $ sudo apt-get update
   Network Installer for Ubuntu20.04

   $ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
   $ sudo dpkg -i cuda-keyring_1.1-1_all.deb
   $ sudo apt-get update
   Network Installer for RedHat/CentOS 9

   $ sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
   Network Installer for RedHat/CentOS 8

   $ sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo


   For Ubuntu: sudo apt install libnccl2=2.27.3-1+cuda12.4 libnccl-dev=2.27.3-1+cuda12.4
   For RHEL/Centos: sudo yum install libnccl-2.27.3-1+cuda12.4 libnccl-devel-2.27.3-1+cuda12.4 libnccl-static-2.27.3-1+cuda12.4

   ```



## 测试示例

克隆仓库

```bash
git clone https://github.com/NVIDIA/nccl-tests.git
```

编译

```bash

# 不支持多机通信算子测试
make -j40

# 支持多机通信算子测试
make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/lib/x86_64-linux-gnu
```

验证

```bash

./all_reduce_perf -b 8 -e 512M -f 2 -g 1


mpirun -np 2 -H a100-44,a100-43  --allow-run-as-root -bind-to none -map-by slot -mca coll_hcoll_enable 0 -mca pml ob1 -mca btl_tcp_if_include  bond0 -mca btl ^openib -x NCCL_IB_GID_INDEX=3 -x NCCL_SOCKET_IFNAME=bond0  -x NCCL_IB_HCA=^mlx5_8 -x NCCL_IB_TC=128 -x NCCL_IB_QPS_PER_CONNECTION=8 -x NCCL_DEBUG=INFO -x NCCL_ALGO=Ring ./all_reduce_perf -b 32M -e 8G  -f 2 -g 8


# nThread 1 nGpus 1 minBytes 8 maxBytes 536870912 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0

# nThread 1 nGpus 8 minBytes 536870912 maxBytes 17179869184 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid  88731 on    a100-44 device  0 [0000:27:00] NVIDIA A100-SXM4-80GB
#  Rank  1 Group  0 Pid  88731 on    a100-44 device  1 [0000:2a:00] NVIDIA A100-SXM4-80GB
#  Rank  2 Group  0 Pid  88731 on    a100-44 device  2 [0000:51:00] NVIDIA A100-SXM4-80GB
#  Rank  3 Group  0 Pid  88731 on    a100-44 device  3 [0000:57:00] NVIDIA A100-SXM4-80GB
#  Rank  4 Group  0 Pid  88731 on    a100-44 device  4 [0000:9e:00] NVIDIA A100-SXM4-80GB
#  Rank  5 Group  0 Pid  88731 on    a100-44 device  5 [0000:a4:00] NVIDIA A100-SXM4-80GB
#  Rank  6 Group  0 Pid  88731 on    a100-44 device  6 [0000:c7:00] NVIDIA A100-SXM4-80GB
#  Rank  7 Group  0 Pid  88731 on    a100-44 device  7 [0000:ca:00] NVIDIA A100-SXM4-80GB
#  Rank  8 Group  0 Pid  92057 on    a100-43 device  0 [0000:27:00] NVIDIA A100-SXM4-80GB
#  Rank  9 Group  0 Pid  92057 on    a100-43 device  1 [0000:2a:00] NVIDIA A100-SXM4-80GB
#  Rank 10 Group  0 Pid  92057 on    a100-43 device  2 [0000:51:00] NVIDIA A100-SXM4-80GB
#  Rank 11 Group  0 Pid  92057 on    a100-43 device  3 [0000:57:00] NVIDIA A100-SXM4-80GB
#  Rank 12 Group  0 Pid  92057 on    a100-43 device  4 [0000:9e:00] NVIDIA A100-SXM4-80GB
#  Rank 13 Group  0 Pid  92057 on    a100-43 device  5 [0000:a4:00] NVIDIA A100-SXM4-80GB
#  Rank 14 Group  0 Pid  92057 on    a100-43 device  6 [0000:c7:00] NVIDIA A100-SXM4-80GB
#  Rank 15 Group  0 Pid  92057 on    a100-43 device  7 [0000:ca:00] NVIDIA A100-SXM4-80GB
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
33554432        524288     float    none      -1   1047.8   32.02   30.02      0   1015.2   33.05   30.99    N/A
    67108864       1048576     float    none      -1   1914.8   35.05   32.86      0   1921.0   34.94   32.75    N/A
   134217728       2097152     float    none      -1   3651.2   36.76   34.46      0   3621.3   37.06   34.75    N/A
   268435456       4194304     float    none      -1   6970.0   38.51   36.11      0   6948.6   38.63   36.22    N/A
   536870912       8388608     float    none      -1    13653   39.32   36.87      0    13571   39.56   37.09    N/A
  1073741824      16777216     float    none      -1    26893   39.93   37.43      0    26875   39.95   37.46    N/A
  2147483648      33554432     float    none      -1    53381   40.23   37.71      0    53255   40.32   37.80    N/A
  4294967296      67108864     float    none      -1   106262   40.42   37.89      0   106112   40.48   37.95    N/A
  8589934592     134217728     float    none      -1   212152   40.49   37.96      0   211851   40.55   38.01    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 35.7955
```

## 参考文档
[nccl-test 使用指引-腾讯云开发者社区-腾讯云](https://cloud.tencent.com/developer/article/2361710)