import torch
import torch.nn as nn

class CNN_Fruits_Model(nn.Module):
    """
    CNN水果分类模型

    该模型是一个用于水果图像分类的卷积神经网络，包含6个卷积层和2个全连接层。
    每个卷积层后都添加了批归一化和SE注意力机制，以提高模型性能。
    默认可分类6种水果。
    """
    def __init__(self, num_classes=6):
        """
        初始化模型

        参数:
            num_classes (int): 分类类别数，默认为6
        """
        super(CNN_Fruits_Model, self).__init__()

        # 第一个卷积块：卷积层 -> 批归一化 -> 激活 -> SE注意力 -> 池化
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)  # 输入3通道(RGB)，输出32通道，3x3卷积核，步长1，填充1
        self.bn1 = nn.BatchNorm2d(32)  # 批归一化，加速训练并提高稳定性
        self.pool1 = nn.MaxPool2d(2, 2)  # 最大池化，2x2窗口，步长2，将特征图尺寸减半

        # 第二个卷积块36
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)  # 输入32通道，输出64通道
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # 第三个卷积块
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)  # 输入64通道，输出128通道
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # 第四个卷积块
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)  # 输入128通道，输出256通道
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # 第五个卷积块
        self.conv5 = nn.Conv2d(256, 256, 3, 1, 1)  # 保持256通道不变
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(2, 2)

        # 第六个卷积块（无池化）
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)  # 保持256通道不变
        self.bn6 = nn.BatchNorm2d(256)

        # 为每个卷积层添加SE注意力模块
        self.se1 = SEBlock(32)
        self.se2 = SEBlock(64)
        self.se3 = SEBlock(128)
        self.se4 = SEBlock(256)
        self.se5 = SEBlock(256)
        self.se6 = SEBlock(256)

        # 全局平均池化层，将特征图转换为1x1
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 全连接层
        self.fc1 = nn.Linear(256, 256)  # 第一个全连接层，256输入，256输出
        self.fc2 = nn.Linear(256, num_classes)  # 输出层，256输入，num_classes输出
        # 激活函数和正则化
        self.relu = nn.ReLU()  # ReLU激活函数
        self.dropout = nn.Dropout(0.2)  # Dropout层，防止过拟合

    def forward(self, x):
        """
        前向传播

        参数:
            x (torch.Tensor): 输入图像张量，形状为(batch_size, 3, height, width)

        返回:
            torch.Tensor: 分类结果，形状为(batch_size, num_classes)
        """
        # 第一个卷积块
        x = self.relu(self.bn1(self.conv1(x)))  # 卷积 -> 批归一化 -> ReLU激活
        x = self.se1(x)  # SE注意力机制
        x = self.pool1(x)  # 池化

        # 第二个卷积块
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.se2(x)
        x = self.pool2(x)

        # 第三个卷积块
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.se3(x)
        x = self.pool3(x)

        # 第四个卷积块
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.se4(x)
        x = self.pool4(x)

        # 第五个卷积块
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.se5(x)
        x = self.pool5(x)

        # 第六个卷积块（无池化）
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.se6(x)

        # 全局平均池化并展平
        x = self.gap(x)  # 全局平均池化，输出形状为(batch_size, 256, 1, 1)
        x = x.view(x.size(0), -1)  # 展平，形状为(batch_size, 256)

        # 全连接层
        x = self.dropout(self.relu(self.fc1(x)))  # 全连接 -> ReLU -> Dropout
        x = self.fc2(x)  # 输出层

        return x

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation注意力模块

    SE模块通过学习每个通道的权重，增强重要特征通道，抑制不重要特征通道，
    从而提高模型的表示能力。
    """
    def __init__(self, channels, reduction=16):
        """
        初始化SE模块

        参数:
            channels (int): 输入特征通道数
            reduction (int): 降维比例，默认为16，用于控制中间层的通道数
        """
        super(SEBlock, self).__init__()
        hidden = max(channels // reduction, 1)  # 确保至少有1个通道

        # 全局平均池化，将每个通道的特征图转换为单个数值
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 两个全连接层：降维和升维
        self.fc1 = nn.Linear(channels, hidden)  # 降维
        self.fc2 = nn.Linear(hidden, channels)  # 升维

        # 激活函数
        self.relu = nn.ReLU()  # 用于降维后的激活
        self.sig = nn.Sigmoid()  # 用于生成0-1之间的通道权重

    def forward(self, x):
        """
        SE模块的前向传播

        参数:
            x (torch.Tensor): 输入特征图，形状为(batch_size, channels, height, width)

        返回:
            torch.Tensor: 经过SE模块增强后的特征图，形状与输入相同
        """
        b, c, _, _ = x.size()  # 获取批次大小和通道数

        # Squeeze：全局平均池化，形状从(b, c, h, w)变为(b, c, 1, 1)
        s = self.gap(x).view(b, c)

        # Excitation：通过两个全连接层学习通道权重
        s = self.relu(self.fc1(s))  # 降维并激活
        s = self.sig(self.fc2(s)).view(b, c, 1, 1)  # 升维并使用sigmoid激活，形状为(b, c, 1, 1)

        # Scale：将学习到的通道权重应用于原始特征图
        return x * s
