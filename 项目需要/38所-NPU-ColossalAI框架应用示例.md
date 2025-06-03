


通过docker容器进行训练验证

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
    registry.cnbita.com:5000/leinaoyun-arm/colossalai:0.4.9-pytorch-npu_2.2 bash
```


实验镜像构建：

```bash
FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.1.rc1-910b-ubuntu22.04-py3.10

ENV ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest 

RUN pip3 install torch==2.2.0 torchvision torch_npu==2.2.0  tensorboard colossalai==0.4.9
```


```bash
import argparse
import os
from pathlib import Path
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.booster.plugin.dp_plugin_base import DPPluginBase
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam

NUM_EPOCHS = 80
LEARNING_RATE = 1e-3


def build_dataloader(batch_size: int, coordinator: DistCoordinator, plugin: DPPluginBase):
    transform_train = transforms.Compose(
        [transforms.Pad(4), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32), transforms.ToTensor()]
    )
    transform_test = transforms.ToTensor()

    data_path = os.environ.get("DATA", "./data")
    with coordinator.priority_execution():
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_path, train=True, transform=transform_train, download=True
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_path, train=False, transform=transform_test, download=True
        )

    train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = plugin.prepare_dataloader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_dataloader, test_dataloader



def train_epoch(
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    criterion: nn.Module,
    train_dataloader: DataLoader,
    booster: Booster,
    coordinator: DistCoordinator,
):
    model.train()
    with tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]", disable=not coordinator.is_master()) as pbar:
        for images, labels in pbar:
            images = images.cuda()
            labels = labels.cuda()
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            booster.backward(loss, optimizer)
            optimizer.step()
            optimizer.zero_grad()

            # Print log info
            pbar.set_postfix({"loss": loss.item()})


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--plugin",
        type=str,
        default="torch_ddp",
        choices=["torch_ddp", "torch_ddp_fp16", "low_level_zero", "gemini"],
        help="plugin to use",
    )
    parser.add_argument("-r", "--resume", type=int, default=-1, help="resume from the epoch's checkpoint")
    parser.add_argument("-c", "--checkpoint", type=str, default="./checkpoint", help="checkpoint directory")
    parser.add_argument("-i", "--interval", type=int, default=5, help="interval of saving checkpoint")
    parser.add_argument(
        "--target_acc", type=float, default=None, help="target accuracy. Raise exception if not reached"
    )
    args = parser.parse_args()

    if args.interval > 0:
        Path(args.checkpoint).mkdir(parents=True, exist_ok=True)

    colossalai.launch_from_torch(config={})
    coordinator = DistCoordinator()

    global LEARNING_RATE
    LEARNING_RATE *= coordinator.world_size

    booster_kwargs = {}
    if args.plugin == "torch_ddp_fp16":
        booster_kwargs["mixed_precision"] = "fp16"
    if args.plugin.startswith("torch_ddp"):
        plugin = TorchDDPPlugin()
    elif args.plugin == "gemini":
        plugin = GeminiPlugin(initial_scale=2**5)
    elif args.plugin == "low_level_zero":
        plugin = LowLevelZeroPlugin(initial_scale=2**5)

    booster = Booster(plugin=plugin, **booster_kwargs)

    train_dataloader, test_dataloader = build_dataloader(100, coordinator, plugin)

    model = torchvision.models.resnet18(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = HybridAdam(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=1 / 3)

    model, optimizer, criterion, _, lr_scheduler = booster.boost(
        model, optimizer, criterion=criterion, lr_scheduler=lr_scheduler
    )

    if args.resume >= 0:
        booster.load_model(model, f"{args.checkpoint}/model_{args.resume}.pth")
        booster.load_optimizer(optimizer, f"{args.checkpoint}/optimizer_{args.resume}.pth")
        booster.load_lr_scheduler(lr_scheduler, f"{args.checkpoint}/lr_scheduler_{args.resume}.pth")


    start_epoch = args.resume if args.resume >= 0 else 0
    for epoch in range(start_epoch, NUM_EPOCHS):
        train_epoch(epoch, model, optimizer, criterion, train_dataloader, booster, coordinator)
        lr_scheduler.step()

        if args.interval > 0 and (epoch + 1) % args.interval == 0:
            booster.save_model(model, f"{args.checkpoint}/model_{epoch + 1}.pth")
            booster.save_optimizer(optimizer, f"{args.checkpoint}/optimizer_{epoch + 1}.pth")
            booster.save_lr_scheduler(lr_scheduler, f"{args.checkpoint}/lr_scheduler_{epoch + 1}.pth")


if __name__ == "__main__":
    main()

```


在38所环境验证结果如下：

colossalai命令在真正执行得时候也是通过torchrun 命令去执行脚本得。

```bash
colossalai run --nproc_per_node 1  train.py
                    ||
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29500 train.py -c ./ckpt-fp32
```
colossalai官方并没有提供npu训练得相关示例。无法进行验证。


colossalai  不支持cpu 运行，必须添加--nproc_per_node参数

Npu安装的是最新的驱动。