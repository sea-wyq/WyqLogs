import argparse
import os
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.booster.plugin.dp_plugin_base import DPPluginBase
from colossalai.cluster import DistCoordinator

# 设置NPU环境变量
os.environ.setdefault('DISTRIBUTED_BACKEND', 'hccl')
os.environ.setdefault('ASCEND_SLOG_PRINT_TO_STDOUT', '0')

NUM_EPOCHS = 80
LEARNING_RATE = 1e-3


class SyntheticCIFAR10(Dataset):
    def __init__(self, num_samples, is_train=True):
        self.num_samples = num_samples
        self.is_train = is_train
        # 生成模拟图像（32x32x3，0-255 uint8，模拟原始图像）
        self.images = np.random.randint(0, 256, size=(num_samples, 32, 32, 3), dtype=np.uint8)
        self.labels = np.random.randint(0, 10, size=num_samples)
        
        # 变换管道：先转PIL Image，再执行空间变换
        if is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),  # 转为PIL Image，支持Pad等操作
                transforms.Pad(4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32),
                transforms.ToTensor(),  # 转为Tensor (CHW格式)
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[idx]  # 形状为(32,32,3)，ndarray
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def build_dataloader(batch_size: int, coordinator: DistCoordinator, plugin: DPPluginBase):
    with coordinator.priority_execution():
        train_dataset = SyntheticCIFAR10(num_samples=50000, is_train=True)
        test_dataset = SyntheticCIFAR10(num_samples=10000, is_train=False)

    train_dataloader = plugin.prepare_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True
    )
    test_dataloader = plugin.prepare_dataloader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True
    )
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
            images = images.npu()
            labels = labels.npu()
            outputs = model(images)
            loss = criterion(outputs, labels)

            booster.backward(loss, optimizer)
            optimizer.step()
            optimizer.zero_grad()

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

    colossalai.launch_from_torch(backend='hccl')
    coordinator = DistCoordinator()

    if args.interval > 0 and coordinator.is_master():
        Path(args.checkpoint).mkdir(parents=True, exist_ok=True)

    global LEARNING_RATE
    LEARNING_RATE *= coordinator.world_size

    booster_kwargs = {}
    if args.plugin == "torch_ddp_fp16":
        booster_kwargs["mixed_precision"] = "fp16"

    if args.plugin.startswith("torch_ddp"):
        plugin = TorchDDPPlugin()
    elif args.plugin == "gemini":
        plugin = GeminiPlugin(initial_scale=2**5, device='npu')
    elif args.plugin == "low_level_zero":
        plugin = LowLevelZeroPlugin(initial_scale=2**5, device='npu')

    booster = Booster(plugin=plugin, **booster_kwargs)

    train_dataloader, test_dataloader = build_dataloader(100, coordinator, plugin)

    model = torchvision.models.resnet18(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
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

        if args.interval > 0 and (epoch + 1) % args.interval == 0 and coordinator.is_master():
            booster.save_model(model, f"{args.checkpoint}/model_{epoch + 1}.pth")
            booster.save_optimizer(optimizer, f"{args.checkpoint}/optimizer_{epoch + 1}.pth")
            booster.save_lr_scheduler(lr_scheduler, f"{args.checkpoint}/lr_scheduler_{epoch + 1}.pth")


if __name__ == "__main__":
    main()