import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# 1、散点输入，定义输入数据
data = [[-0.5, 7.7], [1.8, 98.5], [0.9, 57.8], [0.4, 39.2], [-1.4, -15.7], [-1.4, -37.3], [-1.8, -49.1], [1.5, 75.6], [0.4, 34.0], [0.8, 62.3]]
x_train_np = np.array([item[0] for item in data])
y_train_np = np.array([item[1] for item in data])

# 转换为PyTorch的张量
x_train = torch.from_numpy(x_train_np).float()
y_train = torch.from_numpy(y_train_np).float()

# 2、定义前向模型
model = nn.Linear(1, 1)  # 输入维度为1，输出维度为1

# 3、定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# 获取模型初始化过的参数
w = float(model.weight)
b = float(model.bias)

# 用来记录梯度
gd_path = []

# 4、开始迭代
num_iterations = 500
for n in range(1, num_iterations + 1):
    # 记录梯度
    gd_path.append((w, b))
    # 前向传播
    y_pred = model(x_train.unsqueeze(1))
    # 计算损失
    loss = criterion(y_pred.squeeze(1), y_train)
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 获取当前的参数值
    w = float(model.weight)
    b = float(model.bias)
print("\n".join((map(str, gd_path))))
print(f"模型函数： y =  {w} * x + {b}")