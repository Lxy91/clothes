import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from CutOut import Cutout

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_data_loaders(batch_size=128, valid_size=0.2, num_workers=0, pic_path='data'):
    """
    获取FashionMNIST数据集的数据加载器
    batch_size: 每个批次加载的图像数量
    valid_size: 用于验证集的训练集比例
    num_workers: 数据加载的子进程数量
    pic_path: 数据集存储路径
    返回: 训练、验证、测试数据加载器
    """
    # 训练集数据增强（FashionMNIST为单通道灰度图）
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),  # 28x28图像，填充4后随机裁剪
        transforms.RandomHorizontalFlip(),     # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2860], std=[0.3530]),  # FashionMNIST均值和标准差
        Cutout(n_holes=1, length=8),  # 调整Cutout尺寸适应28x28图像
    ])

    # 测试集和验证集不使用数据增强
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2860], std=[0.3530]),
    ])

    # 加载FashionMNIST数据集（注意：数据集名称和通道数变化）
    train_data = datasets.FashionMNIST(
        pic_path, train=True, download=True, transform=transform_train
    )
    valid_data = datasets.FashionMNIST(
        pic_path, train=True, download=True, transform=transform_test
    )
    test_data = datasets.FashionMNIST(
        pic_path, train=False, download=True, transform=transform_test
    )

    # 划分训练集和验证集
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # 创建采样器
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers
    )

    return train_loader, valid_loader, test_loader