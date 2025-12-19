# 导入模块
import torch
from CNN import CNN_Model
from torchvision.transforms import ToTensor
import torchvision
from torch.utils.data import DataLoader
from torch.optim import Adam # 优化器 ， 调整模型参数
from torch.nn import CrossEntropyLoss # 交叉熵损失函数，分类项目中使用的一个损失函数

# 1. 模型
model = CNN_Model()

# 2. 超参数
BATCH_SIZE = 64 # 数据加载的批量大小，增加批量大小提高训练效率
EPOCH = 10  # 纪元（训练的轮次），增加轮次以充分训练模型
LEARNING_RATE = 0.001 # 提高学习率，加快收敛速度

# 3. 数据集
to_1 = ToTensor()
train_data = torchvision.datasets.MNIST("" , train=True , download= False , transform=to_1)
test_data = torchvision.datasets.MNIST("" , train=False , download= False , transform=to_1)
train_loader = DataLoader(train_data , batch_size= BATCH_SIZE , shuffle=True) # shuffle打乱
test_loader = DataLoader(test_data , batch_size=BATCH_SIZE , shuffle=False)

# 4. 损失函数， 优化器
loss_fun = CrossEntropyLoss()
opti = Adam(model.parameters() , lr= LEARNING_RATE)
# 添加学习率调度器，每5个epoch将学习率降低为原来的0.5倍
scheduler = torch.optim.lr_scheduler.StepLR(opti, step_size=5, gamma=0.5)

# 5. 评估函数 :  预测对的数据量 / 总数据量
def eval(model , test_loader):
    # 开启评估模式， 不让它修改模型的参数
    model.eval()
    # 不计算梯度
    with torch.no_grad():
        total = 0  # 总数
        correct = 0  # 正确数
        for images , labels  in test_loader: # trueImgs(长度是20 ) , labels(长度是20)
            # 预测： 前向计算
            outputs = model(images) # 省略了forward
            # 检查预测和label是否匹配
            i = 0
            for output in outputs:
                if torch.argmax(output)  == labels[i]:   # 预测值和真实值是否相等
                    correct += 1 # 正确的数据加1
                total += 1 # 总数都要加1
                i = i + 1
    return  correct / total  # 正确率

# 6. 训练
result = 0
for i in range(EPOCH): #训练的轮次
    # 开启训练模型  ， 训练的过程中调整模型的参数
    model.train()
    j = 0  # 每隔多少次输出一次这个损失值
    for images , labels in train_loader:
        opti.zero_grad()      # 梯度清零
        outputs = model(images)   # 前向传播  ， 得到预测值
        loss = loss_fun(outputs ,labels) # 计算损失 # 计算真实值和预测值之间的差
        loss.backward() # 反向传播
        opti.step()   # 优化参数
        if j == 0  or j  % 100 ==0 :
            print(f"{i+1}/{EPOCH}纪元： loss :{loss.item()}")
        j = j + 1
    # 更新学习率
    scheduler.step()
    # 评估
    acct = eval(model , test_loader)
    print(f"第{i+1}轮训练后，测试集准确率：{acct:.4f}，当前学习率：{scheduler.get_last_lr()[0]:.6f}")
    if acct > result:
        result = acct
        # 7. 保存模型参数
        torch.save(model.state_dict() , 'cnn.pt')
        print(f"找到更好的模型，准确率提升至{result:.4f}，已保存模型参数")
print(f"CNN训练结束了， 模型最好准确率： {result:.4f}")
