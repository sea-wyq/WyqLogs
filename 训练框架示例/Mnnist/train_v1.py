import os
import torch
import time
import torch.distributed as dist
import argparse
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from model import MNIST
from torch.nn.parallel import DistributedDataParallel as DDP


def cleanup():
    dist.destroy_process_group()


def demo_basic(gpu, args):
    rank = args.nr * args.gpus + gpu
    print(f"Running basic DDP example on rank {rank}.")
    dist.init_process_group("gloo", rank=rank, world_size=args.world_size)
    
    train_dataset = datasets.MNIST(root='./data',
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                num_replicas=args.world_size,
                                                                rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=100,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    model = MNIST().to(gpu)
    ddp_model = DDP(model, device_ids=[gpu])

    cost = torch.nn.CrossEntropyLoss().to(gpu)
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
            lables = lables.to(gpu)
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

    if args.nr == 0:
        torch.save(model.state_dict(), "mnist.pkl")

    cleanup()
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes

    os.environ['MASTER_ADDR'] = '127.0.0.1'       # master ip
    os.environ['MASTER_PORT'] = '12345'           # master port

    mp.spawn(demo_basic,
            args=(args,),
            nprocs=args.gpus,
            join=True)
    print("finished")

# python3 train.py -n 2 -g 1 -nr 0 节点1执行
# python3 train.py -n 2 -g 1 -nr 1 节点2执行
# 双机单卡环境   