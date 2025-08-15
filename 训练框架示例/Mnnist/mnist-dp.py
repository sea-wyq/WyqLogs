import torch
import torchvision
import torchvision.transforms as transforms
import deepspeed
import time
import os
import torch.nn as nn
from torch.distributed.elastic.utils.data import ElasticDistributedSampler
from deepspeed.accelerator import get_accelerator

class MNIST(torch.nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(1, 32, 3, 1, 1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(32, 64, 3, 1, 1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(2, 2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(14 * 14 * 64, 1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.2),
                                         torch.nn.Linear(1024, 10))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 14 * 14 * 64)
        x = self.dense(x)
        return x


def save_checkpoint(epoch, model, optimizer, path):
    torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimize_state_dict": optimizer.state_dict(),
}, path)

def load_checkpoint(path):
    checkpoint = torch.load(path)
    return checkpoint

if __name__ == "__main__":
    deepspeed.init_distributed()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)
    train_sampler = ElasticDistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=100,
                                                num_workers=2,
                                                pin_memory=True,
                                                sampler=train_sampler,)
    net =MNIST()
    ds_config = {
        "train_batch_size": 100,
        "optimizer": {
            "type": "Adam",
            "params": {
            "lr": 0.0001,
            "betas": [
                0.8,
                0.999
            ],
            "eps": 1e-8,
            "weight_decay": 3e-7
            }
        },
        "elasticity": {
            "enabled": True,
            "max_train_batch_size": 200,
            "micro_batch_sizes": [100,200],
            "min_gpus": 1,
            "max_gpus": 2,
            "min_time": 0,
            "version": 0.1,
            "ignore_non_elastic_batch_info": True,
        }
    }
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(model=net, model_parameters=net.parameters(), training_data=train_dataset, config=ds_config)

    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank = model_engine.local_rank
    criterion = nn.CrossEntropyLoss()
    ckp_path = "/app/checkpoint.pt"
    first_epoch = -1
    max_epoch = 10
    if os.path.exists(ckp_path):
        print(f"load checkpoint from {ckp_path}")
        checkpoint = load_checkpoint(ckp_path)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimize_state_dict"])
        first_epoch = checkpoint["epoch"]
    epochs = 10

    for epoch in range(first_epoch + 1, max_epoch):
        # train
        start =  time.time()
        number = 0
        sum_loss = 0.0
        for inputs, lables in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs.to(local_rank))
            lables = lables.to(local_rank)
            loss = criterion(outputs, lables)
            loss.backward()
            optimizer.step()
            sum_loss += loss.data
            end = time.time()
            number += 1
            if number % 100 == 0:
                print('epoch: [%d,%d], step: [%d,%d] loss:%.06f, step time:%.06f' %
                    (epoch + 1, 10, number+1, len(train_loader), sum_loss / len(train_loader),  (end-start)/number))
        save_checkpoint(epoch, net, optimizer, ckp_path)
    print('Finished Training')
