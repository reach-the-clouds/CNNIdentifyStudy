# 导入的模块
from torchvision.models import resnet18 , ResNet18_Weights
from torchsummary import summary
import  torch

# 创建模型
model = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)  # 创建模型的时候， 传入预训练的参数
# 遇到的问题： 咋门的最后的输出应该是6. 咋门的输入是（3，n ，n）这个不需要改
# resnet18 , 本身最后一个线性层的输出是1000（固定的） ，
# 修改最后一个线性层的输出为6即可
in_features = model.fc.in_features   # in_features就是512
print(f"线性层输入： {in_features}")
model.fc = torch.nn.Linear(in_features , 6)
summary(model , (3,400,400))  #  输入是3通道的 ， 默认的输出是1000