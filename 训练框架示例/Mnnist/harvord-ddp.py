import time
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed
import horovod.torch as hvd

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

if __name__ == '__main__':  
    hvd.init()
    torch.manual_seed(1)
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(1)
    torch.set_num_threads(1)
    
    train_dataset = datasets.MNIST('./data', train=True, download=True,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.1307,), (0.3081,))
                                                ]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(
                                                train_dataset, 
                                                num_replicas=hvd.size(), 
                                                rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=100,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)
    model = MNIST().cuda()

    optimizer = optim.Adam(model.parameters(),lr= 1e-4 * hvd.size())
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         op=hvd.Average,
                                         gradient_predivide_factor=1.0)
    cost = torch.nn.CrossEntropyLoss()
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    
    model.train()
    for epoch in range(0, 10):
        train_sampler.set_epoch(epoch)
        start =  time.time()
        number = 0
        sum_loss = 0.0
        for inputs, lables in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.cuda())
            lables = lables.cuda()
            loss = cost(outputs, lables)
            loss.backward()
            optimizer.step()
            sum_loss += loss.data
            end = time.time()
            number += 1
            if number % 100 == 0:
                print('epoch: [%d,%d], step: [%d,%d] loss:%.06f, step time:%.06f' %
                    (epoch + 1, 10, number+1, len(train_loader), sum_loss / len(train_loader),  (end-start)/number))
    