# 导入模块
import torch

from fnn import FNN_Model
from torchvision.transforms import ToTensor
import torchvision
from torch.utils.data import DataLoader
from torch.optim import Adam # 优化器 ， 调整模型参数
from torch.nn import CrossEntropyLoss # 交叉熵损失函数，分类项目中使用的一个损失函数
# 1. 模型
model = FNN_Model()
# 2. 超参数
BATCH_SIZE = 20 # 数据加载的批量大小
EPOCH = 3  # 纪元（训练的轮次）
LEARNING_RATE = 0.0001 # 学习率 ， 是一个小的数据值， 优化器要使用的一个数据
# 3. 数据集
to_1 = ToTensor()
train_data = torchvision.datasets.MNIST("" , train=True , download= False , transform=to_1)
test_data = torchvision.datasets.MNIST("" , train=False , download= False , transform=to_1)
train_loader = DataLoader(train_data , batch_size= BATCH_SIZE , shuffle=True) # shuffle打乱
test_loader = DataLoader(test_data , batch_size=BATCH_SIZE , shuffle=False)

# 4. 损失函数， 优化器
loss_fun =CrossEntropyLoss()
opti = Adam(model.parameters() , lr= LEARNING_RATE)
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
    # 评估
    acct = eval(model , test_loader)
    if acct > result:
        result = acct
        # 7. 保存模型参数
        torch.save(model.state_dict() , 'fnn.pt')
print(f"训练结束了， 模型最好准去率： {result}")