import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    # 3x3卷积，带padding，保持空间尺寸（当stride=1时）
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

# padding=dilation 是关键：保证输出尺寸正确，特别是使用空洞卷积时
# bias=False：因为后面会接BatchNorm


def conv1x1(in_planes, out_planes, stride=1):
    # 1x1卷积，用于改变通道数或下采样
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):  #基础残差块
    expansion = 1  # 输出通道扩张倍数

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        """
        参数说明：
        - inplanes: 输入通道数
        - planes: 中间通道数（也是输出通道数，因为expansion=1）
        - stride: 步长，用于下采样
        - downsample: 下采样层（当输入输出维度不匹配时需要）
        - groups: 分组卷积组数（BasicBlock不支持）
        - base_width: ResNet 的基础宽度参数，用于计算分组卷积的实际通道数（BasicBlock 中一般用不到，是为了和 Bottleneck 块统一接口）
        - dilation: 空洞卷积率（BasicBlock不支持）
        - norm_layer: 归一化层，默认 None（通常用nn.BatchNorm2d）；可自定义为 LayerNorm 等，增加灵活性
        """
        # 两个3x3卷积层
        self.conv1 = conv3x3(inplanes, planes, stride)  # stride>1，用于下采样
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)  # inplace=True节省内存
        self.conv2 = conv3x3(planes, planes)  # stride=1
        self.bn2 = norm_layer(planes)
        # stride可能为2，用于下采样
        self.downsample = downsample  # 如果需要，调整输入维度
        self.stride = stride

    # 前向传播实现
    def forward(self, x):
        identity = x  # 保存原始输入
        # 主路径：两个卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # 如果需要，调整残差连接的维度
        if self.downsample is not None:
            identity = self.downsample(x)
        # 残差连接：主路径输出 + 输入
        out += identity
        out = self.relu(out)

        return out


# Bottleneck残差块
class Bottleneck(nn.Module):

    expansion = 4  # 输出通道是中间通道的4倍

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        """
        参数说明：
        - width = int(planes * (base_width / 64.)) * groups
        这是ResNeXt引入的"基数"概念，控制中间层宽度
        """
        width = int(planes * (base_width / 64.)) * groups
        # 1x1卷积：降维
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # 3x3卷积：特征提取（下采样）
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        # 1x1卷积：升维
        self.conv3 = conv1x1(width, planes * self.expansion)  # 输出通道扩张4倍
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    # 前向传播实现
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        """
        参数说明：
        - block: BasicBlock或Bottleneck
        - layers: 每个阶段的残差块block数量，如[3,4,6,3]对应ResNet50
        - num_classes: 分类任务的类别数，默认1000（ImageNet）
        - zero_init_residual: 是否将残差块中最后一个批归一化（BN）层的参数初始化为 0
        - groups: 分组卷积组数（ResNeXt引入）
        - width_per_group：每个分组的卷积通道数（ResNeXt引入）
        - replace_stride_with_dilation: 用空洞卷积替代下采样
        - norm_layer=None: 表示归一化层的类型，默认None时会使用nn.BatchNorm2d（批归一化）
        """
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # 每个元素表示是否用空洞卷积替代对应阶段的下采样
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # 注释掉最大池化，避免小图像过度压缩
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 4个阶段，每个阶段包含多个残差块
        self.layer1 = self._make_layer(block, 64, layers[0])  # stride=1
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])  # 下采样
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])  # 下采样
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])  # 下采样
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化到1x1
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接层输出分类结果
        # 初始化卷积层和批归一化层的参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        #每个残差分支中的最后一个批归一化层进行零初始化，
        # 这可以帮助模型更快地收敛，避免梯度消失问题
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """
        创建一个stage（包含多个block）
        - planes: 该stage的基本通道数
        - blocks: block数量
        - stride: 第一个block的步长（用于下采样）
        - dilate: 是否使用空洞卷积替代下采样
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        # 空洞卷积逻辑
        if dilate:
            self.dilation *= stride  # 增加空洞率
            stride = 1  # 不使用stride下采样，改用空洞扩大感受野
        # 判断是否需要下采样（调整维度）
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),  # 调整通道和尺寸
                norm_layer(planes * block.expansion),
            )
        # 创建第一个block（可能包含下采样）
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        # 更新当前通道数为下一个stage的基本通道数
        self.inplanes = planes * block.expansion
        # 创建剩余的blocks（stride=1，无下采样）
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    # 前向传播实现

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # 注释掉最大池化，避免小图像过度压缩
        #x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    # 前向传播入口
    def forward(self, x):
        return self._forward_impl(x)


# 定义ResNet模型的基础结构
def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


# 定义ResNet18等不同的模型
def ResNet18(**kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)
