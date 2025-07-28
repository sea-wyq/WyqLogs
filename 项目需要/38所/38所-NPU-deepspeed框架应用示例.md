# 功能验证

## 实验镜像构建

```bash
FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.1.rc1-910b-ubuntu22.04-py3.10

ENV ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest 

RUN pip3 install torch==2.2.0 torchvision torch_npu==2.2.0  tensorboard deepspeed -i https://pypi.tuna.tsinghua.edu.cn/simple


ADD ./train/ /home/train/
```

```bash
FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/ascend-pytorch:24.0.0-A1-2.1.0-ubuntu20.04

USER root

RUN apt update && apt install -y vim 

RUN pip3 install deepspeed

ADD ./train/ /home/train/

```

注：torch和torch_npu的版本需要对应起来，否则会报错。

## 通过docker容器进行训练验证

```bash
docker run -it --rm \
    --shm-size=10g \
    --privileged \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    --device=/dev/davinci0 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    -v ./train.py:/home/train.py \
    registry.cnbita.com:5000/leinaoyun-arm/deepspeed:0.16-pytorch-npu_2.2 bash 
```

## 验证npu deepspeed 环境是否可用

>>> import torch
>>> print('torch:',torch.__version__)
torch: 2.2.0
>>> import torch_npu
>>> print('torch_npu:',torch.npu.is_available(),",version:",torch_npu.__version__)
torch_npu: True ,version: 2.2.0
>>> from deepspeed.accelerator import get_accelerator
>>> print('accelerator:', get_accelerator()._name)
accelerator: npu

```bash
import argparse
import os
import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.tensorboard import SummaryWriter


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--addr", default='127.0.0.1', type=str, help='master addr')
    parser.add_argument("--data_dir", type=str, help='dataset directory')
    parser.add_argument("--output_dir", default='.', type=str, help='output directory')
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=0.001, type=float, help="学习率")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD 动量")
    parser = deepspeed.add_config_arguments(parser)
    FLAGS = parser.parse_args()
    deepspeed.init_distributed()
    local_rank = FLAGS.local_rank if FLAGS.local_rank > -1 else int(os.getenv("LOCAL_RANK", 0))
    addr = FLAGS.addr
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = '29501'
    loc = 'npu:{}'.format(local_rank)
    torch.npu.set_device(loc)
    # dist.init_process_group(backend='hccl')  # hccl 是 Ascend NPU 设备上的通信后端
    ckpt_path = None
    if dist.get_rank() == 0 and ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset,batch_size=16,shuffle=True,num_workers=2)
    model = ToyModel().to(loc)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    ds_config = {
    "train_batch_size": 16,
    "steps_per_print": 2000,
    "optimizer": {
        "type": "Adam",
        "params": {
        "lr": 0.001,
        "betas": [
            0.8,
            0.999
        ],
        "eps": 1e-8,
        "weight_decay": 3e-7
        }
    },}
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=FLAGS.lr, momentum=FLAGS.momentum)
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=FLAGS, model=model, model_parameters=parameters, training_data=trainset, config=ds_config)

    loss_func = nn.CrossEntropyLoss().to(loc)
    tensorboard_dir = os.path.join(FLAGS.output_dir, 'tensorboard')
    output_ckpt_dir = os.path.join(FLAGS.output_dir, 'ckpt')
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(output_ckpt_dir, exist_ok=True)
    # if dist.get_rank() == 0:
    #     writer = SummaryWriter(tensorboard_dir)

    # 开始训练
    # model.train()
    global_step = 0
    for epoch in range(FLAGS.epochs):
        # trainloader.sampler.set_epoch(epoch)
        for data, label in trainloader:
            data, label = data.to(local_rank), label.to(local_rank)
            optimizer.zero_grad()
            prediction = model_engine(data)
            loss = loss_func(prediction, label)
            loss.backward()
            optimizer.step()
            if dist.get_rank() == 0:
                print(f"epoch={epoch+1}/{FLAGS.epochs}, step={global_step}, loss={loss}")
                # writer.add_scalar("loss", loss, global_step)
            global_step += 1

        if dist.get_rank() == 0:
            save_model_path = os.path.join(output_ckpt_dir, "%d.ckpt" % epoch)
            torch.save(model_engine.state_dict(), save_model_path)
            print(f'saved checkpoint to {save_model_path}')

    if dist.get_rank() == 0:
        save_model_path = os.path.join(output_ckpt_dir, "model.pth")
        torch.save(model_engine.state_dict(), save_model_path)
        # writer.add_graph(model, data)
        # writer.close()


if __name__ == '__main__':
    main()
```

export HCCL_LOG_LEVEL=4
export ASCEND_PROCESS_LOG_PATH=./hccl_logs
export TORCH_DISTRIBUTED_DEBUG=DETAIL

deepspeed --include localhost:0 train.py

```bash
deepspeed --include localhost:0,1 train_dp.py 单机双卡
deepspeed --include localhost:0 train_dp.py 单机单卡

....
epoch=2/10, step=5985, loss=1.3057286739349365
epoch=2/10, step=5986, loss=0.7879546880722046
epoch=2/10, step=5987, loss=1.607465386390686
epoch=2/10, step=5988, loss=1.8528125286102295
epoch=2/10, step=5989, loss=1.5395621061325073
epoch=2/10, step=5990, loss=1.613113522529602
epoch=2/10, step=5991, loss=1.7886117696762085
epoch=2/10, step=5992, loss=0.4235781133174896
epoch=2/10, step=5993, loss=1.0590382814407349
epoch=2/10, step=5994, loss=1.3112924098968506
```

参考文档：<https://www.deepspeed.ai/tutorials/accelerator-setup-guide/#huawei-ascend-npu>

## 结论

在38所环境验证结果如下：

arm架构的服务器，缺少头文件immintrin.h，无法只使用cpu进行训练。

immintrin.h是 Intel 处理器的 SIMD 指令集头文件，ARM 架构（如 aarch64）不支持该指令集，因此编译时会报错。

使用npu进行训练出现通信算子报错，在更换服务版本和镜像后，结果仍然无法正常训练。

但在其它环境验证过，npu deepspeed是可以正常训练，推测可能是38所得npu服务器环境存在问题。

Npu 安装的是最新的驱动。
