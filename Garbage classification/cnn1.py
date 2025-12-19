import torch
import torch.nn as nn

class CNN_Garbage_Model(nn.Module):
    def __init__(self):
        super(CNN_Garbage_Model, self).__init__()
        # 第一层卷积：输入3通道(RGB)，输出16通道，卷积核5x5，步长1，填充2
        self.conv1 = nn.Conv2d(3, 16, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(16)  # 添加BatchNorm2d
        # 第一层池化：2x2最大池化
        self.pool1 = nn.MaxPool2d(2, 2)
        # 第二层卷积：输入16通道，输出32通道，卷积核5x5，步长1，填充2
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(32)  # 添加BatchNorm2d
        # 第二层池化：2x2最大池化
        self.pool2 = nn.MaxPool2d(2, 2)
        # 第三层卷积：输入32通道，输出64通道，卷积核3x3，步长1，填充1
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)  # 添加BatchNorm2d
        # 第三层池化：2x2最大池化
        self.pool3 = nn.MaxPool2d(2, 2)
        # 第四层卷积：输入64通道，输出128通道，卷积核3x3，步长1，填充1
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(128)  # 添加BatchNorm2d
        # 第四层池化：2x2最大池化
        self.pool4 = nn.MaxPool2d(2, 2)
        # 第五层卷积：输入128通道，输出256通道，卷积核3x3，步长1，填充1
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(256)  # 添加BatchNorm2d
        # 第五层池化：2x2最大池化
        self.pool5 = nn.MaxPool2d(2, 2)
        # 全连接层：输入256*12*12=36864，输出1024
        self.fc1 = nn.Linear(256 * 12 * 12, 1024)
        # 全连接层：输入1024，输出512
        self.fc2 = nn.Linear(1024, 512)
        # 全连接层：输入512，输出6（垃圾分类有6个类别）
        self.fc3 = nn.Linear(512, 6)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 第一层卷积 -> BatchNorm -> 激活 -> 池化
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        # 第二层卷积 -> BatchNorm -> 激活 -> 池化
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        # 第三层卷积 -> BatchNorm -> 激活 -> 池化
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        # 第四层卷积 -> BatchNorm -> 激活 -> 池化
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        # 第五层卷积 -> BatchNorm -> 激活 -> 池化
        x = self.pool5(self.relu(self.bn5(self.conv5(x))))
        # 展平为一维张量
        x = x.view(-1, 256 * 12 * 12)
        # 全连接层1 -> 激活 -> Dropout
        x = self.dropout(self.relu(self.fc1(x)))
        # 全连接层2 -> 激活 -> Dropout
        x = self.dropout(self.relu(self.fc2(x)))
        # 全连接层3（输出层）
        x = self.fc3(x)
        return x

from torchsummary import summary
if __name__ == '__main__':
    model = CNN_Garbage_Model()
    summary(model , (3,400,400))
