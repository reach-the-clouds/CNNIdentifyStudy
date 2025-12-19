
import torch
import torch.nn as nn

class FNN_Model(nn.Module):
    """
    全连接神经网络模型，用于MNIST手写数字识别
    输入：28x28的图像张量
    输出：10个类别的概率分布
    """
    def __init__(self):
        super(FNN_Model, self).__init__()
        # 定义神经网络层
        self.fc1 = nn.Linear(28*28, 128)  # 输入层到隐藏层1
        self.fc2 = nn.Linear(128, 64)     # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(64, 10)      # 隐藏层2到输出层
        self.relu = nn.ReLU()              # 激活函数

    def forward(self, x):
        # 将输入展平为一维向量
        x = x.view(-1, 28*28)
        # 前向传播
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
