from  os import  path
import torch
from torchvision import transforms
from  torchvision.models import resnet18 , ResNet18_Weights

# 1. 张量转换器要和训练的时候的张量转换器保持一致（不要图像处理部分）
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((400, 400)),
    # 数据标准化
    transforms.Normalize(mean=[0.6357556, 0.6043181, 0.57092524],
                         std=[0.21566282, 0.2124977, 0.21848688])
])
#？？？？ 图片转换为张量
# 2. 定义模型，加载模型参数
model = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(512,6)
model.load_state_dict(torch.load("resnet_garbage_model.pth"))

# 3. 预测
model.eval()
with torch.no_grad():
    pass

#4. 输出预测的结果
pass


