from  os import  path
from torchvision import transforms
from torchvision.datasets import ImageFolder
from  torchvision.models import resnet18 , ResNet18_Weights
import torch
from torch.utils.data import DataLoader, random_split  # 数据集加载器
from torch.optim import Adam  # 优化器
from torch.nn import CrossEntropyLoss  # 损失函数
from torch.utils.mobile_optimizer import optimize_for_mobile
import time

BATCH_SIZE = 10 # 批量大小
EPOCH = 1 # 纪元
LR = 0.001 # 学习率
# 数据集的根目录
ROOT_DIR = "./strong_garbage_datasets" # 自己的文件夹的名字

# 1.数据集
# 转换器
transform_train = transforms.Compose([
    transforms.ToTensor() ,
    transforms.Resize((400,400)),
    # 图片处理
    # transforms.RandomHorizontalFlip(p=0.5) , # 随机水平翻转
    # 数据标准化
    transforms.Normalize(mean=[0.6357556, 0.6043181, 0.57092524],
              std=[0.21566282, 0.2124977, 0.21848688])
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((400, 400)),
    # 数据标准化
    transforms.Normalize(mean=[0.6357556, 0.6043181, 0.57092524],
                         std=[0.21566282, 0.2124977, 0.21848688])
])
# 创建数据集（根目录包含类别子文件夹，不再依赖 train/test 子目录）
base_dataset = ImageFolder(ROOT_DIR, transform=transform_train)
total_len = len(base_dataset)
train_len = int(0.8 * total_len)
test_len = total_len - train_len

# 生成随机切分索引，保证列车/测试集不重叠
indices = torch.randperm(total_len)
train_indices = indices[:train_len].tolist()
test_indices = indices[train_len:].tolist()

# 为不同子集应用不同的预处理transform
train_dataset = ImageFolder(ROOT_DIR, transform=transform_train)
test_dataset = ImageFolder(ROOT_DIR, transform=transform_test)

from torch.utils.data import Subset
train_set = Subset(train_dataset, train_indices)
test_set = Subset(test_dataset, test_indices)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
# 2. 模型， 损失函数， 优化器
model = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)  # 创建模型的时候， 传入预训练的参数
model.fc = torch.nn.Linear(512 , 6)
loss_fun = torch.nn.CrossEntropyLoss() # 交叉熵损失函数， 一般用于分类的时候
opti = torch.optim.Adam(model.parameters() , lr=LR)

# 2. 超参数的设置（人定义的参数）
BATCH_SIZE = 10  # 减小批次大小，因为400*400图片更大，需要更多内存
EPOCH = 5  # 增加训练轮数以获得更好的效果 ， 大概需要20分钟才能训练好
LEARNING_RATE = 0.0001  # 学习

# 检测可用设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
model = model.to(device)

# 5. 评估函数
def eval(model, test_loader):
    # 开启评估模式 ： 不要修改模型的参数
    model.eval()
    # 不计算梯度值
    with torch.no_grad():
        total = 0  # 总数
        correct = 0  # 正确数
        for images, labels in test_loader:
            # 将数据移到指定设备
            images = images.to(device)
            labels = labels.to(device)
            # 预测数据（就是做题）
            outputs = model(images)
            # 计算预测结果
            _, predicted = torch.max(outputs, 1)
            # 更新计数
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return correct / total  # 正确率


# 6. 训练模型
result = 0
for epoch in range(EPOCH):
    # 记录训练时间
    start_time = time.time()
    # 开启训练模式
    model.train()  # 调整模型的参数
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        # 将数据移到指定设备
        images = images.to(device)
        labels = labels.to(device)

        opti.zero_grad()  # 梯度清零
        outputs = model(images)  # 前向计算(预测)
        loss = loss_fun(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        opti.step()  # 优化参数

        running_loss += loss.item()
        # 输出损失值
        if batch_idx % 100 == 0 or batch_idx == len(train_loader) - 1:
            avg_loss = running_loss / (batch_idx + 1)
            print(f"{epoch + 1}/{EPOCH}纪元, 批次 {batch_idx + 1}/{len(train_loader)}, 平均损失: {avg_loss:.6f}")

    # 记录训练耗时
    training_time = time.time() - start_time

    # 评估
    acct = eval(model, test_loader)  # 准确率
    print(f"第 {epoch + 1} 纪元, 训练耗时: {training_time:.2f} 秒, 测试准确率: {acct:.4f}")

    if acct > result:
        result = acct
        # 7 .保存训练好的模型
        print(f"保存新的最佳模型，当前准确率: {acct:.4f}")
        # 保存完整模型参数
        torch.save(model.state_dict(), "resnet_garbage_model.pth")
        # 导出为PyTorch Lite模型用于移动部署
        scripted_module = torch.jit.script(model)
        optimized_scripted_module = optimize_for_mobile(scripted_module)
        optimized_scripted_module._save_for_lite_interpreter("resnet_garbage_model.ptl")
print(f"垃圾分类模型训练结束，最高准确率： {result:.4f}")








