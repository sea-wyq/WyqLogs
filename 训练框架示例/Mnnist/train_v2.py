import os
import torch
import time
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP

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


def cleanup():
    dist.destroy_process_group()


def demo_basic():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"Running basic DDP example on rank {rank}.")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    train_dataset = datasets.MNIST(root='./data',
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                num_replicas=world_size,
                                                                rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=100,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    model = MNIST().to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    cost = torch.nn.CrossEntropyLoss().to(local_rank)
    optimizer = torch.optim.Adam(ddp_model.parameters())

    epochs = 10
    number = 0
    start =  time.time()
    for epoch in range(epochs):
        sum_loss = 0.0
        train_correct = 0
        for inputs, lables in train_loader:
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            lables = lables.to(local_rank)
            loss = cost(outputs, lables)
            loss.backward()
            optimizer.step()
            _, id = torch.max(outputs.data, 1)
            sum_loss += loss.data
            train_correct += torch.sum(id == lables.data)
            end = time.time()
            number += 1
        print('[%d,%d] loss:%.03f, correct:%.03f' %
              (epoch + 1, epochs, sum_loss / len(train_loader), 100 * train_correct / len(train_dataset)))
        print("train time: ",(end-start)/number )

    if rank == 0:
        torch.save(model.state_dict(), "mnist.pkl")

    cleanup()
  
if __name__ == "__main__":
    demo_basic()
    print("finished")


# torchrun --nnodes 2 --nproc_per_node 1 --node-rank 0  --master-addr 10.244.58.97  --master-port 12345 ./train.py
# torchrun --nnodes 2 --nproc_per_node 1 --node-rank 1  --master-addr 10.244.58.97  --master-port 12345 ./train.py


# torchrun --nnodes 2 --local-addr=$POD_IP --nproc_per_node 1 --rdzv-backend=c10d  --rdzv-endpoint=10.244.58.97:12345 ./train.py