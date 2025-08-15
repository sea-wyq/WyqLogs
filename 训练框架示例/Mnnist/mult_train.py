import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from model import MNIST
import time
import torch.multiprocessing as mp


def main(rank, world_size):
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    model = MNIST().to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    cost = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    train_dataset = datasets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=False, sampler=train_sampler)
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False, sampler=test_sampler)

    epochs = 2
    start = time.time()
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        # train
        sum_loss = 0.0
        train_correct = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = cost(outputs, labels)
            loss.backward()
            optimizer.step()

            _, id = torch.max(outputs.data, 1)
            sum_loss += loss.data
            train_correct += torch.sum(id == labels.data)
        if rank == 0:
            print('[%d,%d] loss:%.03f, correct:%.03f' %
                  (epoch + 1, epochs, sum_loss / len(train_loader), 100 * train_correct / len(train_dataset)))
            print("train time: ", (time.time() - start) / len(train_loader))

        model.eval()
        test_correct = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, id = torch.max(outputs.data, 1)
            test_correct += torch.sum(id == labels.data)
        if rank == 0:
            print("correct:%.3f%%" % (100 * test_correct / len(test_dataset)))

    if rank == 0:
        torch.save(model.module.state_dict(), "mnist.pkl")


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)