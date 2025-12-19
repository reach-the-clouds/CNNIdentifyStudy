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
        elif label == 2:  # C簇类
            ax.plot([x.item()], [y.item()], 'go')


# ===================== 构建一个三分类坐标系 =========================
def generate_features(x, y, noise=0.2, num=20, label=0):
    points_x = torch.normal(x, noise, (num, 1))
    points_y = torch.normal(y, noise, (num, 1))
    points = torch.cat([points_x, points_y], dim=1)
    labels = torch.ones((num, 1)) * label
    features = torch.cat([points, labels], dim=1)
    return features


# 簇族A  标签：0
features_a = generate_features(0.5, 0.5, label=0)
# 簇族B  标签：1
features_b = generate_features(1.5, 1.5, label=1)
# 簇族C  标签：2
features_c = generate_features(0.5, 1.5, label=2)
# ===================== 合并上述数据 =========================
data = torch.cat([features_a, features_b, features_c], dim=0)
indices = torch.randperm(data.shape[0])
data = data[indices]

features = data[:, :2]
labels = data[:, 2]  # 注意事项，在交叉熵中labels必须是一维
drawPoints(features, labels, ax1)

# ===================== 多分类的构建模型方法 =========================
model = nn.Sequential(
    # 输入层
    nn.Linear(2, 10),
    # 激活函数
    nn.Tanh(),
    # 输出层
    nn.Linear(10, 3),  # 输出层输出值为分类的数量
    nn.LogSoftmax(dim=-1)  # dim=-1 因为输出矩阵原因
)
# =================== 定义模型训练的配置 =========================
# 损失函数
criterion = nn.CrossEntropyLoss()  # 交叉熵损失（总距离公式）
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# =================== 训练模型 =========================
# 模型训练次数
epochs = 10000
# 开始训练
for epoch in range(epochs):
    # 优化器将所有的w，b的导数(专业词汇：梯度)清零
    optimizer.zero_grad()
    # 预测
    label_predict = model(features)
    # 计算损失
    loss = criterion(label_predict, labels.long())  # 此处labels需要变为整数
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
# 返回得分最高的值的下标（0，1，2）
predict_labels = torch.argmax(predict_labels, dim=-1)

# 绘制上述的图像
drawPoints(features, predict_labels, ax2)

plt.show()