
单机分布式训练

![bert示例](./bert.py)

示例来源: <https://colab.research.google.com/drive/13FjI_uXaw8JJGjzjVX3qKSLyW9p3b6OV?usp=sharing#scrollTo=uD3K8T6B4YJp>

多机分布式训练

![bert示例](./bert_mn.py)

```bash
torchrun \
  --nnodes=2 \          # 总节点数
  --node_rank=0 \       # 当前节点序号（主节点为 0）
  --nproc_per_node=2 \  # 每个节点的进程数（= GPU 数）
  --master_addr=192.168.1.100 \  # 主节点 IP
  --master_port=29500 \  # 主节点通信端口（可自定义，确保未占用）
  train.py  # 你的脚本名

torchrun \
  --nnodes=2 \          # 总节点数（与主节点一致）
  --node_rank=1 \       # 当前节点序号（从节点为 1）
  --nproc_per_node=2 \  # 每个节点的进程数（= GPU 数）
  --master_addr=192.168.1.100 \  # 主节点 IP（与主节点一致）
  --master_port=29500 \  # 主节点端口（与主节点一致）
  train.py  # 你的脚本名
```

npu 单机多卡

![bert示例](./bert_npu.py)