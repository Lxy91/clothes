import torch
import torch.nn as nn
from ReadDataset import get_data_loaders
from ResNet import ResNet18

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_class = 10  # FashionMNIST共10类
batch_size = 100

# 加载数据
_, _, test_loader = get_data_loaders(batch_size=batch_size, pic_path='dataset')

# 初始化模型（适配单通道输入）
model = ResNet18()
model.conv1 = nn.Conv2d(
    in_channels=1,  # 单通道灰度图
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)
model.fc = torch.nn.Linear(512, n_class)
model = model.to(device)

# 加载训练好的权重
model.load_state_dict(torch.load('checkpoint/resnet18_fashionmnist.pt')['model_state_dict'])

# 测试模型
total_sample = 0
right_sample = 0
model.eval()
with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        _, pred = torch.max(output, 1)
        correct_tensor = pred.eq(target.data.view_as(pred))
        total_sample += data.size(0)
        right_sample += correct_tensor.sum().item()

print(f"Test Accuracy: {100 * right_sample / total_sample:.2f}%")
