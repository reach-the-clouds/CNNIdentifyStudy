# 垃圾分类的卷积神经网络模型
# 针对400*400尺寸输入图像优化的模型
import torch.nn as nn

class CNN_Garbage_Model(nn.Module):  # 重命名为垃圾分类模型
    def __init__(self):
        super(CNN_Garbage_Model , self).__init__() # 调用父类的构造函数
        self.s = nn.Sequential(
            # 1. 第一次卷积： （3， 400 ， 400）
            # 计算： 400 - 3 + 1 = 398， 得到16个 398*398的矩阵
            nn.Conv2d(3, 16, 3),  # 增加通道数以处理更大尺寸的图像
            nn.ReLU(),           # 激活函数
            nn.BatchNorm2d(16),  # 批量归一化
            nn.MaxPool2d(2, 2),  # 最大池化，池化之后宽高减半： 199*199
            
            # 2. 第二次卷积（16， 199， 199）
            # 计算： 199 - 5 + 1 = 195， 得到32个 195*195的矩阵
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # 池化后： 97*97
            
            # 3. 第三次卷积（32， 97， 97）
            # 计算： 97 - 3 + 1 = 95， 得到64个 95*95的矩阵
            nn.Conv2d(32, 64, 3),  # 减小卷积核大小，避免过度缩小特征图
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # 池化后： 47*47
            
            # 4. 第四次卷积（64， 47， 47）
            # 计算： 47 - 3 + 1 = 45， 得到128个 45*45的矩阵
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),  # 池化后： 22*22
            
            # 5. 展平
            nn.Flatten(),  # 把多维的变为1维
            
            # 6. 线性模型 - 输入特征数：128 * 22 * 22 = 61952
            nn.Linear(128 * 22 * 22, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # 添加Dropout防止过拟合
            
            # 7. 线性模型 - 减少过拟合风险
            nn.Linear(512, 256),
            nn.ReLU(),
            
            # 8. 输出层 - 6个类别（cardboard, glass, metal, paper, plastic, trash）
            nn.Linear(256, 6)
        )
    
    # 前向计算
    def forward(self, x):
        return self.s(x)
from torchsummary import summary
if __name__ == '__main__':
    model = CNN_Garbage_Model()
    summary(model , (3,400,400))