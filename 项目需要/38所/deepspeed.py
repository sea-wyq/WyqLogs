import argparse
import os
import numpy as np

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from deepspeed.accelerator import get_accelerator
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer


def add_argument():
    parser = argparse.ArgumentParser(description="CIFAR (Synthetic Data)")

    parser.add_argument(
        "-e",
        "--epochs",
        default=30,
        type=int,
        help="number of total epochs (default: 30)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=2000,
        help="output logging information at a given interval",
    )

    # 混合精度训练配置
    parser.add_argument(
        "--dtype",
        default="fp16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="训练数据类型",
    )

    # ZeRO优化配置
    parser.add_argument(
        "--stage",
        default=0,
        type=int,
        choices=[0, 1, 2, 3],
        help="ZeRO优化阶段",
    )

    # MoE（混合专家模型）配置
    parser.add_argument(
        "--moe",
        default=False,
        action="store_true",
        help="使用DeepSpeed的混合专家模型",
    )
    parser.add_argument(
        "--ep-world-size", default=1, type=int, help="专家并行世界大小"
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        nargs="+",
        default=[1],
        help="专家数量列表（MoE相关）",
    )
    parser.add_argument(
        "--mlp-type",
        type=str,
        default="standard",
        help="MLP类型，仅当num-experts>1时有效，可选[standard, residual]",
    )
    parser.add_argument(
        "--top-k", default=1, type=int, help="gating选择的专家数量（支持1或2）"
    )
    parser.add_argument(
        "--min-capacity",
        default=0,
        type=int,
        help="专家的最小容量（与capacity_factor无关）",
    )
    parser.add_argument(
        "--noisy-gate-policy",
        default=None,
        type=str,
        help="带噪声的gating策略（仅支持top-1），可选None, RSample, Jitter",
    )
    parser.add_argument(
        "--moe-param-group",
        default=False,
        action="store_true",
        help="为MoE创建单独的参数组（使用ZeRO+MoE时需要）",
    )

    # 添加DeepSpeed配置参数
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    return args


# 模拟CIFAR10数据集
class SyntheticCIFAR10(Dataset):
    def __init__(self, num_samples=50000, image_shape=(3, 32, 32), num_classes=10):
        """
        生成模拟CIFAR10格式的数据
        - image_shape: CIFAR10图像格式为3通道32x32像素
        - num_samples: 样本数量（训练集默认50000，测试集10000）
        - num_classes: 类别数（CIFAR10为10类）
        """
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.num_classes = num_classes
        
        # 生成随机图像数据（模拟归一化后的像素值，范围[-1, 1]，与真实CIFAR预处理一致）
        self.images = np.random.uniform(-1.0, 1.0, size=(num_samples, *image_shape)).astype(np.float32)
        
        # 生成随机标签（0-9）
        self.labels = np.random.randint(0, num_classes, size=num_samples)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def create_moe_param_groups(model):
    """为每个专家创建单独的参数组"""
    parameters = {"params": [p for p in model.parameters()], "name": "parameters"}
    return split_params_into_different_moe_groups_for_optimizer(parameters)


def get_ds_config(args):
    """获取DeepSpeed配置字典"""
    ds_config = {
        "train_batch_size": 16,
        "steps_per_print": 2000,
        "logging": {
            "level": "WARNING"
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.001,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 0.001,
                "warmup_num_steps": 1000,
            },
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "bf16": {"enabled": args.dtype == "bf16"},
        "fp16": {
            "enabled": args.dtype == "fp16",
            "fp16_master_weights_and_grads": False,
            "loss_scale": 0,
            "loss_scale_window": 500,
            "hysteresis": 2,
            "min_loss_scale": 1,
            "initial_scale_power": 15,
        },
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": args.stage,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 50000000,
            "reduce_bucket_size": 50000000,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "cpu_offload": False,
        },
    }
    return ds_config


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # CIFAR为3通道输入
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 32x32经过两次卷积池化后为5x5
        self.fc2 = nn.Linear(120, 84)
        self.moe = args.moe
        
        if self.moe:
            fc3 = nn.Linear(84, 84)
            self.moe_layer_list = []
            for n_e in args.num_experts:
                self.moe_layer_list.append(
                    deepspeed.moe.layer.MoE(
                        hidden_size=84,
                        expert=fc3,
                        num_experts=n_e,
                        ep_size=args.ep_world_size,
                        use_residual=args.mlp_type == "residual",
                        k=args.top_k,
                        min_capacity=args.min_capacity,
                        noisy_gate_policy=args.noisy_gate_policy,
                    )
                )
            self.moe_layer_list = nn.ModuleList(self.moe_layer_list)
            self.fc4 = nn.Linear(84, 10)  # 10类输出
        else:
            self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # 展平特征图
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        if self.moe:
            for layer in self.moe_layer_list:
                x, _, _ = layer(x)
            x = self.fc4(x)
        else:
            x = self.fc3(x)
        return x


def main(args):
    # 初始化DeepSpeed分布式后端
    deepspeed.init_distributed()
    _local_rank = int(os.environ.get("LOCAL_RANK"))
    get_accelerator().set_device(_local_rank)

    # 使用模拟数据集（无需真实数据下载）
    # 模拟CIFAR10：训练集50000样本，测试集10000样本
    trainset = SyntheticCIFAR10(num_samples=50000)
    testset = SyntheticCIFAR10(num_samples=10000)

    # 初始化模型
    net = Net(args)
    parameters = filter(lambda p: p.requires_grad, net.parameters())

    # 如果使用MoE，为每个专家创建单独的参数组
    if args.moe_param_group:
        parameters = create_moe_param_groups(net)
    
    # 初始化DeepSpeed配置
    ds_config = get_ds_config(args)
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=parameters,
        training_data=trainset,
        config=ds_config,
    )

    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank = model_engine.local_rank

    # 确定目标数据类型（混合精度）
    target_dtype = None
    if model_engine.bfloat16_enabled():
        target_dtype = torch.bfloat16
    elif model_engine.fp16_enabled():
        target_dtype = torch.half

    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # 获取输入数据并移动到设备
            inputs, labels = data[0].to(local_device), data[1].to(local_device)

            # 转换数据类型（如果启用混合精度）
            if target_dtype is not None:
                inputs = inputs.to(target_dtype)

            # 前向传播
            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            model_engine.backward(loss)
            model_engine.step()

            # 打印训练日志
            running_loss += loss.item()
            if local_rank == 0 and i % args.log_interval == (args.log_interval - 1):
                print(
                    f"[{epoch + 1: d}, {i + 1: 5d}] loss: {running_loss / args.log_interval:.3f}"
                )
                running_loss = 0.0
    
    print("训练完成")


if __name__ == "__main__":
    args = add_argument()
    main(args)