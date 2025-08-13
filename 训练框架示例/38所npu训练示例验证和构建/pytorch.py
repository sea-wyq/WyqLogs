import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch_npu

# 设置NPU环境
os.environ["ASCEND_VISIBLE_DEVICES"] = "0"  # 指定使用的NPU设备

# 检查NPU是否可用
if not torch.npu.is_available():
    raise RuntimeError("NPU设备不可用，请检查环境配置")
print(f"使用NPU设备: {torch.npu.current_device()}")
device = torch.device("npu:0")  # 设置NPU设备

# 模拟MNIST数据集
class SyntheticMNISTDataset(Dataset):
    def __init__(self, num_samples=60000, image_size=(1, 28, 28), num_classes=10):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        
        # 生成随机图像数据(0-1之间，模拟归一化后的像素)
        self.images = np.random.rand(num_samples, *image_size).astype(np.float32)
        
        # 生成随机标签(0-9)
        self.labels = np.random.randint(0, num_classes, size=num_samples)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# 创建数据集和数据加载器
train_dataset = SyntheticMNISTDataset(num_samples=60000)
test_dataset = SyntheticMNISTDataset(num_samples=10000)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # 对NPU传输更友好
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# 定义CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # 展平
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型、损失函数和优化器
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(epoch):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # 将数据移至NPU
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播、计算损失、反向传播、参数更新
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # 统计
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 打印批次信息
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    # 计算训练集准确率
    train_acc = 100. * correct / total
    print(f'Epoch {epoch}, Train Loss: {train_loss/len(train_loader):.6f}, '
          f'Train Accuracy: {correct}/{total} ({train_acc:.2f}%)')
    return train_acc

# 测试函数
def test():
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    # 不计算梯度
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_acc = 100. * correct / total
    print(f'Test Loss: {test_loss/len(test_loader):.6f}, '
          f'Test Accuracy: {correct}/{total} ({test_acc:.2f}%)\n')
    return test_acc

# 执行训练和测试
num_epochs = 5
print("开始在NPU上训练...")
for epoch in range(1, num_epochs + 1):
    train(epoch)
    test()

# 保存模型
torch.save(model.state_dict(), "mnist_cnn_npu.pth")
print("模型已保存为 mnist_cnn_npu.pth")