
基于cuda 和roce 网卡进行通信测试


## 克隆仓库
```bash
https://github.com/linux-rdma/perftest
```
## 编译（支持--use_cuda）
```bash
./autogen.sh && ./configure CUDA_H_PATH=/usr/local/cuda/include/cuda.h && make -j
```
## 验证（发送数据先经过指定的GPU，在发送到网卡。）
```bash
./ib_write_bw --use_cuda=4  -d mlx5_5 -x 3
./ib_write_bw --use_cuda=5 -d mlx5_4 10.1.30.43 --report_gbits  -x 3
```