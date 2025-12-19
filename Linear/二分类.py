import torch
from torch import nn
import matplotlib.pyplot as plt

# 构建两个图表
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)


# ===================== 绘制坐标点的函数 =========================
def drawPoints(features, labels, ax):
    for feature, label in zip(features, labels):
        x, y = feature
        if label == 0:  # A簇类
            ax.plot([x.item()], [y.item()], 'ro')
        elif label == 1:  # B簇类
            ax.plot([x.item()], [y.item()], 'bo')


# ===================== 构建一个二分类坐标系 =========================
# 噪点
noise = 0.2
# 簇族A  标签：0
points_a = torch.normal(0.5, noise, (20, 2))  # 总共有20对xy，指向簇类A
labels_a = torch.zeros((20, 1))  # 总共有20个为0的簇族A标签
features_a = torch.cat([points_a, labels_a], dim=1)  # (20,3)
# 簇族B  标签：1
# points_b = torch.normal(1.5, noise, (20, 2))  # 总共有20对xy，指向簇类B
points_b_x = torch.normal(0.5, noise, (20, 1))
points_b_y = torch.normal(1.5, noise, (20, 1))
points_b = torch.cat([points_b_x, points_b_y], dim=1)
labels_b = torch.ones((20, 1))  # 总共有20个为1的簇族B标签
features_b = torch.cat([points_b, labels_b], dim=1)  # (20,3)
# ===================== 合并上述数据 =========================
data = torch.cat([features_a, features_b], dim=0)  # (40,3)
# 为了让训练的准确更高，因此我们要将数据进行随机打散
indices = torch.randperm(40)  # 0~39 下标随机打乱
data = data[indices]

# 对数据进行拆分，将坐标x，y和标签进行拆分
features = data[:, :2]  # (40,2)
labels = data[:, 2:]  # (40,1) 专业说法：真实值
# 绘制原始图像
drawPoints(features, labels, ax1)

# ===================== 构建模型 =========================
# 注意：构建模型需要 torch.nn
# Sequential 组装模型的对象
model = nn.Sequential(
    # 输入
    nn.Linear(2, 10),  # 10是隐藏层的神经元数量
    # 激活函数
    nn.Tanh(),  # Tanh 函数是sigmoid函数改进版（tanh会加快模型的训练速度）
    # 输出
    nn.Linear(10, 1),
    nn.Sigmoid()  # 由于该函数介于[0,1], >= 0.5 归为 1这个分类， < 0.5 归为0分类
)
# =================== 定义模型训练的配置 =========================
# 损失函数
criterion = nn.MSELoss()  # MSELoss 均方差损失（总距离公式）
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# =================== 训练模型 =========================
# 模型训练次数
epochs = 10000
# 开始训练
for epoch in range(epochs):
    # 优化器将所有的w，b的导数(专业词汇：梯度)清零
    optimizer.zero_grad()
    # 预测
    label_predict = model(features)  # label_predict :预测值
    # 计算损失
    loss = criterion(label_predict, labels)  # 衡量真实值与预测值之间差距（损失）
    loss.backward()  # 反向传播
    # 更新w、b的参数
    optimizer.step()

    print(f"{epoch + 1} / {epochs} -- loss:{loss.item():.4f}")  # :.4f保留四位小数
# =================== 绘制预测图 =========================
x = torch.linspace(0, 2, 20)  # (20,)
y = torch.linspace(0, 2, 20)  # (20,)
x, y = torch.meshgrid([x, y], indexing='ij')  # x (20,20) y (20,20)
x = x.reshape(400, 1)
y = y.reshape(400, 1)
features = torch.cat([x, y], dim=1)  # (400,2)
model.eval()  # 开始评估模式
predict_labels = model(features)
# >= 0.5 归为 1这个分类， < 0.5 归为0分类
predict_labels[predict_labels >= 0.5] = 1
predict_labels[predict_labels < 0.5] = 0

# 绘制上述的图像
drawPoints(features, predict_labels, ax2)

plt.show()