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
    cost = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr= 1e-4 * hvd.size())
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         op=hvd.Average,
                                         gradient_predivide_factor=1.0)

    @hvd.elastic.run
    def train(state):
        for state.epoch in range(state.epoch, 10):
            state.model.train()
            train_sampler.set_epoch(state.epoch)
            steps_remaining = len(train_loader) - state.batch

            sum_loss = 0.0
            for state.batch, (data, target) in enumerate(train_loader):
                if state.batch >= steps_remaining:
                    break
                data, target = data.cuda(), target.cuda()
                state.optimizer.zero_grad()
                output = state.model(data)
                loss = cost(output, target)
                loss.backward()
                state.optimizer.step()
                sum_loss += loss.data
                if state.batch % 100 == 0:
                    print('epoch: [%d,%d], step: [%d,%d] loss:%.06f' %
                    (state.epoch + 1, 10, state.batch+1, len(train_loader), sum_loss / len(train_loader)))
                state.commit()
            state.batch = 0


    def on_state_reset():
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-4 * hvd.size()


    state = hvd.elastic.TorchState(model, optimizer, epoch=1, batch=0)
    state.register_reset_callbacks([on_state_reset])
    train(state)
