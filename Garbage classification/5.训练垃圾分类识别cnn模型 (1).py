# 1. 导入需要的模块
import torch
from torchvision.transforms import ToTensor  , Compose , Resize,Normalize# 张量转换类
from cnn1 import CNN_Garbage_Model  # 自定义的垃圾分类模型
from torch.utils.data import DataLoader , random_split # 数据集加载器
from torch.optim import Adam  # 优化器
from torch.nn import CrossEntropyLoss # 损失函数
from garbagedataset import GarbageDataset
from torch.utils.mobile_optimizer import optimize_for_mobile
import time
# 2. 超参数的设置（人定义的参数）
BATCH_SIZE = 10  # 减小批次大小，因为400*400图片更大，需要更多内存
EPOCH = 5  # 增加训练轮数以获得更好的效果 ， 大概需要20分钟才能训练好
LEARNING_RATE = 0.0001  # 学习率
TRAIN_RATIO = 0.8  # 训练集比例
TEST_RATIO = 0.2   # 测试集比例

# 检测可用设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
# 3. 数据集
to_1 = Compose([Resize((400,400)), ToTensor(),
                Normalize(mean= [0.6357556 , 0.6043181 , 0.57092524],
                          std=[0.21566282 ,0.2124977 , 0.21848688])]) # 张量转换器，先调整尺寸再转换为张量
garbage_map = {
    "cardboard": 0 ,
        "glass":2 ,
        "metal": 3  ,
        "paper": 4,
        "plastic":5 ,
        "trash": 1
}
def fun_label(label):
    return garbage_map[label]
# 数据集：  训练集，  测试集
dataset = GarbageDataset(label_path="labels.json", transform=to_1 , transform_label=fun_label)
# 把数据的80%作为训练集，  20%作为测试集
train_set , test_set = random_split(dataset , [TRAIN_RATIO, TEST_RATIO])
train_loader = DataLoader(train_set , shuffle= True , batch_size= BATCH_SIZE)
test_loader = DataLoader(test_set  , shuffle= False , batch_size=BATCH_SIZE)

# 4. 模型 ， 定义损失函数 ， 定义优化器
model = CNN_Garbage_Model().to(device)  # 将模型移到指定设备
loss_fun = CrossEntropyLoss()  # 交叉熵损失函数
opti = Adam(model.parameters() , lr = LEARNING_RATE) # 优化器
# 5. 评估函数
def eval(model , test_loader):
    # 开启评估模式 ： 不要修改模型的参数
    model.eval()
    # 不计算梯度值
    with torch.no_grad():
        total = 0  # 总数
        correct = 0  # 正确数
        for images , labels  in test_loader:
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
        return  correct  / total  # 正确率
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
        
        opti.zero_grad() # 梯度清零
        outputs = model(images)  # 前向计算(预测)
        loss = loss_fun(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        opti.step()      # 优化参数
        
        running_loss += loss.item()
        # 输出损失值
        if batch_idx % 100 == 0 or batch_idx == len(train_loader) - 1:
            avg_loss = running_loss / (batch_idx + 1)
            print(f"{epoch+1}/{EPOCH}纪元, 批次 {batch_idx+1}/{len(train_loader)}, 平均损失: {avg_loss:.6f}")
    
    # 记录训练耗时
    training_time = time.time() - start_time
    
    # 评估
    acct = eval(model, test_loader)  # 准确率
    print(f"第 {epoch+1} 纪元, 训练耗时: {training_time:.2f} 秒, 测试准确率: {acct:.4f}")
    
    if acct > result:
        result = acct
        # 7 .保存训练好的模型
        print(f"保存新的最佳模型，当前准确率: {acct:.4f}")
        # 保存完整模型参数
        torch.save(model.state_dict(), "garbage_model.pth")
        # 导出为PyTorch Lite模型用于移动部署
        scripted_module = torch.jit.script(model)
        optimized_scripted_module = optimize_for_mobile(scripted_module)
        optimized_scripted_module._save_for_lite_interpreter("garbage_model.ptl")
print(f"垃圾分类模型训练结束，最高准确率： {result:.4f}")






