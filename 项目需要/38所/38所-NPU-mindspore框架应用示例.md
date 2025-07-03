
## MindSpore框架进行分布式训练任务演示示例

该演示示例，是基于MNIST数据集完成手写数字分类任务。

## 环境准备

**镜像：**

```bash
swr.cn-south-1.myhuaweicloud.com/ascendhub/ascend-mindspore:24.0.RC1-A1-ubuntu20.04
```

针对同的服务器环境，可以参考mindspore 官方镜像地址：<https://www.hiascend.com/developer/ascendhub/detail/9de02a1a179b4018a4bf8e50c6c2339e>

**服务器环境：**

- mindsopre 版本： 2.3
- npu 驱动版本： 24.1.0
- cann 版本： 8
- npu显卡型号： 910B3

**训练脚本：**

该脚本是基于**数据并行**，来完成分布式训练任务。需要的训练脚本如下：

train_ddp.py

```bash
"""Distributed Data Parallel Example"""

import os
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn
from mindspore.communication import init, get_rank, get_group_size

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
init()
ms.set_seed(1)

class Network(nn.Cell):
    """Network"""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512, weight_init="normal", bias_init="zeros"),
            nn.ReLU(),
            nn.Dense(512, 512, weight_init="normal", bias_init="zeros"),
            nn.ReLU(),
            nn.Dense(512, 10, weight_init="normal", bias_init="zeros")
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

net = Network()

def create_dataset(batch_size):
    """create dataset"""
    dataset_path = os.getenv("DATA_PATH")
    rank_id = get_rank()
    rank_size = get_group_size()
    dataset = ds.MnistDataset(dataset_path, num_shards=rank_size, shard_id=rank_id)
    image_transforms = [
        ds.vision.Rescale(1.0 / 255.0, 0),
        ds.vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        ds.vision.HWC2CHW()
    ]
    label_transform = ds.transforms.TypeCast(ms.int32)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

data_set = create_dataset(32)
optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.CrossEntropyLoss()

def forward_fn(data, target):
    """forward propagation"""
    logits = net(data)
    loss = loss_fn(logits, target)
    return loss, logits

grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)
grad_reducer = nn.DistributedGradReducer(optimizer.parameters)

for epoch in range(10):
    i = 0
    for image, label in data_set:
        (loss_value, _), grads = grad_fn(image, label)
        grads = grad_reducer(grads)
        optimizer(grads)
        if i % 10 == 0:
            print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss_value))
        i += 1

```

## 单机多卡分布式训练

平台使用流程如下图：

1. 选择算法设计-分布式训练功能栏。
2. 填写任务名称并选择平台镜像，（下方提供的示例脚本，已经提前写入截图中所展示的平台镜像中，也可自行挂载示例代码。）
3. 根据选择的多卡套餐，填写并修改分布式训练命令，

![image-2025-05-07](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-2025-05-07.png)
![image-1-2025-05-07](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-1-2025-05-07.png)

根据上面选择的NPU套餐，填写训练命令。单机4卡场景下的执行命令如下：

```bash
export DATA_PATH=/home/MNIST_Data/train/ 
mpirun -n 4 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout python train_ddp.py
```

点击确认后，等待训练结果如下：

![image-2-2025-05-07](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-2-2025-05-07.png)

**遇到的问题**：

npu-smi info 命令显示device正在被使用。

错误 `dcmi model initialized failed, because the device is used. ret is -8020`

![image-2025-06-11](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-2025-06-11.png)

<https://www.hiascend.com/document/detail/zh/Atlas%20200I%20A2/24.1.RC3/ep/installationguide/Install_99.html>
