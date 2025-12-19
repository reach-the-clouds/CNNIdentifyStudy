# 导入必要的库
import numpy  # 用于数值计算
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch的神经网络模块

# 定义CNN模型类，继承自nn.Module
class CNN_Model(nn.Module):
    def __init__(self):
        # 调用父类的初始化方法
        super(CNN_Model, self).__init__()

        # 使用nn.Sequential构建卷积神经网络结构
        self.s = nn.Sequential(
            # 第一卷积层：输入通道1(灰度图)，输出通道10，卷积核大小3×3，添加padding=1保持特征图尺寸
            nn.Conv2d(1, 10, 3, padding=1),
            # ReLU激活函数，引入非线性
            nn.ReLU(),
            # 批归一化层，对10个通道的特征图进行归一化，加速训练并提高稳定性
            nn.BatchNorm2d(10),
            # 最大池化层，2×2窗口，步长为2，减小特征图尺寸
            nn.MaxPool2d(2, 2),

            # 第二卷积层：输入通道10，输出通道20，卷积核大小3×3，添加padding=1保持特征图尺寸
            nn.Conv2d(10, 20, 3, padding=1),
            # ReLU激活函数
            nn.ReLU(),
            # 批归一化层，对20个通道的特征图进行归一化
            nn.BatchNorm2d(20),
            # 最大池化层，2×2窗口，步长为2
            nn.MaxPool2d(2, 2),

            # 将多维特征图展平为一维向量
            nn.Flatten(),
            # 全连接层：输入特征数20*7*7（28×28经过两次2×2池化后变为7×7），输出100个特征
            nn.Linear(20*7*7, 100),
            # ReLU激活函数
            nn.ReLU(),
            # Dropout层，防止过拟合
            nn.Dropout(0.25),
            # 输出层：输入100个特征，输出10个类别（对应0-9的数字）
            nn.Linear(100, 10)
        )

    # 前向传播函数
    def forward(self, x):
        # 输入x通过定义的神经网络结构s进行处理
        return self.s(x)
from torchsummary import summary

# 单张图片预测功能
def predict_image(image_path):
    """
    使用CNN模型预测单张图片中的数字
    参数:
        image_path: 图片文件路径
    返回:
        预测的数字
    """
    # 加载模型
    model = CNN_Model()
    try:
        model.load_state_dict(torch.load('cnn.pt'))
        print("成功加载CNN模型权重")
    except FileNotFoundError:
        print("CNN模型权重文件未找到，请先运行10.cnn模型训练.py训练模型")
        return None

    # 加载图片
    image = Image.open(image_path)
    image = ImageOps.invert(image)  # 背景反转
    image = image.convert("L")  # 灰度图

    # 图片预处理
    transform = Compose([ToTensor(), Resize((28, 28))])
    image_tensor = transform(image).unsqueeze(0)  # 添加batch维度

    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

if __name__=='__main__':
    # 显示模型结构
    model = CNN_Model()
    summary(model, (1, 28, 28))

    # 示例：预测一张图片
    from PIL import Image, ImageOps
    from torchvision.transforms import ToTensor, Resize, Compose

    # 预测示例图片
    image_path = 'trueImgs/7.png'
    result = predict_image(image_path)
    if result is not None:
        print(f"预测结果: {result}")