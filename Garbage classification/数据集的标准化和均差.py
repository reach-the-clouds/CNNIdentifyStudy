import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
#Normalize 是预处理阶段的固定操作
#Normalize 对原始图片做预处理，将数据缩放到合理范围（比如减去数据集均值、除以标准差），让输入分布更规整；
#Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
# 假设你的数据集按文件夹组织（类似ImageNet）
dataset = ImageFolder(root="strong_garbage_datasets", transform=ToTensor())  # ToTensor()会把像素值转成[0,1]的Tensor

mean = 0.0
std = 0.0
total_images = 0

for img, _ in dataset:
    # img shape: [C, H, W]，C=3（RGB）
    mean += img.mean(dim=[1,2])  # 按H、W维度求均值，得到每个通道的均值
    std += img.std(dim=[1,2])    # 按H、W维度求标准差
    total_images += 1

# 计算所有图像的平均均值和标准差（三通道）
mean = mean / total_images  # 结果形如 tensor([0.521, 0.498, 0.473])
std = std / total_images    # 结果形如 tensor([0.213, 0.209, 0.205])

print("你的数据集均值：", mean.numpy())
print("你的数据集标准差：", std.numpy())