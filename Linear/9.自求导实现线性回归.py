import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 1、散点输入
data = [[-0.5, 7.7], [1.8, 98.5], [0.9, 57.8], [0.4, 39.2], [-1.4, -15.7], [-1.4, -37.3], [-1.8, -49.1], [1.5, 75.6], [0.4, 34.0], [0.8, 62.3]]
# 转换为 NumPy 数组
data = np.array(data)
# 提取 x_data 和 y_data
x_data = data[:, 0]
y_data = data[:, 1]

# 2、参数初始化
w = 0
b = 0
learning_rate = 0.01

w_temp = w
b_temp = b


# 3、损失函数只是为了展示公式，实际会直接求解导数，代码无用
def loss_function(X, Y, w, b):
    predicted_y = np.dot(X, w) + b
    total_loss = np.mean((2 * (predicted_y - Y) ** 2))
    return total_loss

# 数据个数，求平均用
l_x_data = x_data.size

# 记录梯度
gd_path = []

# 构建网格点
w_values = np.linspace(-20, 80, 100)
b_values = np.linspace(-20, 80, 100)
W, B = np.meshgrid(w_values, b_values)
loss_values = np.zeros_like(W)

# 计算每个网格点上的损失值
for i, w in enumerate(w_values):
    for j, b in enumerate(b_values):
        loss_values[j, i] = loss_function(x_data, y_data, w, b)

w = w_temp
b = b_temp

# 创建图形对象和子图布局
fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, 2)

# 左上格子
ax2 = fig.add_subplot(gs[0, 0])
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_title("Data")

# 左下格子
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_xlabel("w")
ax3.set_ylabel("b")
ax3.set_title("Contour Plot")

# 整个右侧格子
ax1 = fig.add_subplot(gs[:, 1], projection='3d')
ax1.plot_surface(W, B, loss_values, cmap='viridis', alpha=0.8)
ax1.set_xlabel('w')
ax1.set_ylabel('b')
ax1.set_zlabel('Loss')
ax1.set_title("Surface Plot")

# 4、开始迭代
num_iterations = 500
for n in range(1, num_iterations + 1):
    gd_path.append((w, b))
    # 5、反向传播，手动计算损失函数关于自变量(模型参数)的梯度
    y_predict = w * x_data + b
    gradient_w = 2 * (y_predict - y_data).dot(x_data) / l_x_data
    gradient_b = 2 * (y_predict - y_data).sum() / l_x_data
    # 更新参数
    w = w - learning_rate * gradient_w
    b = b - learning_rate * gradient_b
    # 6、显示频率设置
    frequence_display = 10
    if n % frequence_display == 0 or n == 1:
        # loss = np.mean(np.square(y_predict - y_data))
        x_min, x_max = x_data.min(), x_data.max()
        y_min, y_max = round(w * x_min + b, 3), round(w * x_max + b, 3)

        # 7、梯度下降显示
        # 更新子图 1 数据并绘制
        ax2.clear()
        ax2.scatter(x_data, y_data)
        ax2.plot([x_min, x_max], [y_min, y_max], '-')
        ax2.set_title(f"Linear Regression: w={round(w, 3)}, b={round(b, 3)}")

        # 绘制当前w和b的位置
        ax1.scatter(w, b, loss_function(x_data, y_data, w, b), c='black', s=20)

        # 绘制俯视图等高线
        ax3.clear()
        ax3.contourf(W, B, loss_values, levels=20, cmap='viridis')
        ax3.scatter(w, b, c='black', s=20)

        # 绘制梯度下降路径
        if len(gd_path) > 0:
            gd_w, gd_b = zip(*gd_path)
            ax1.plot(gd_w, gd_b, [loss_function(x_data, y_data, np.array(gd_w[i]), np.array(gd_b[i])) for i in range(len(gd_w))],
                     c='black')
            ax3.plot(gd_w, gd_b)
plt.show()