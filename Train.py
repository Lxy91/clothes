import torch
import numpy as np
import os
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from ReadDataset import get_data_loaders
from ResNet import ResNet18

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 确保checkpoint文件夹存在
os.makedirs('checkpoint', exist_ok=True)
checkpoint_path = 'checkpoint/resnet18_fashionmnist.pt'  # 修改 checkpoint 文件名

# 超参数（FashionMNIST为10类，图像尺寸28x28单通道）
batch_size = 128
n_class = 10  # FashionMNIST共10类
n_epochs = 150  # 适当减少训练轮次（FashionMNIST相对简单）
initial_lr = 0.1
patience = 10  # 早停参数


def main():
    # 加载数据
    train_loader, valid_loader, _ = get_data_loaders(
        batch_size=batch_size,
        pic_path='data'
    )

    # 初始化模型（适配单通道输入）
    model = ResNet18()
    # 关键修改：将输入通道从3改为1（FashionMNIST为灰度图）
    model.conv1 = nn.Conv2d(
        in_channels=1,  # 此处改为1通道
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    model.fc = torch.nn.Linear(512, n_class)  # 输出层保持10类
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)

    # 断点续训相关变量
    start_epoch = 1
    best_valid_loss = np.inf
    accuracy_history = []
    lr = initial_lr
    counter = 0  # 早停计数器

    # 尝试加载checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_valid_loss = checkpoint['best_valid_loss']
        accuracy_history = checkpoint['accuracy_history']
        lr = checkpoint['lr']
        counter = checkpoint['counter']
        print(f"成功加载checkpoint，从第 {start_epoch} 个epoch开始训练")

    # 训练循环
    for epoch in tqdm(range(start_epoch, n_epochs + 1)):
        # 动态调整学习率
        if counter >= patience:
            lr *= 0.5
            counter = 0
            print(f"学习率调整为: {lr}")

        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4
        )

        # 加载优化器状态（如果有）
        if os.path.exists(checkpoint_path) and epoch == start_epoch:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 训练阶段
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)

        # 验证阶段
        model.eval()
        valid_loss = 0.0
        total_sample = 0
        right_sample = 0

        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item() * data.size(0)

                # 计算准确率
                _, pred = torch.max(output, 1)
                correct_tensor = pred.eq(target.data.view_as(pred))
                total_sample += data.size(0)
                right_sample += correct_tensor.sum().item()

        # 计算平均损失
        train_loss /= len(train_loader.sampler)
        valid_loss /= len(valid_loader.sampler)
        current_accuracy = right_sample / total_sample
        accuracy_history.append(current_accuracy)

        # 打印训练信息
        print(f'\nEpoch: {epoch}')
        print(f'Training Loss: {train_loss:.6f}')
        print(f'Validation Loss: {valid_loss:.6f}')
        print(f'Validation Accuracy: {current_accuracy * 100:.2f}%')

        # 保存最佳模型
        if valid_loss <= best_valid_loss:
            print(f'验证损失下降 ({best_valid_loss:.6f} --> {valid_loss:.6f})，保存模型...')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_valid_loss': valid_loss,
                'accuracy_history': accuracy_history,
                'lr': lr,
                'counter': counter
            }, checkpoint_path)
            best_valid_loss = valid_loss
            counter = 0
        else:
            counter += 1
            print(f'验证损失未下降，计数器: {counter}/{patience}')


if __name__ == '__main__':
    main()
