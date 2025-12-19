import matplotlib.pyplot as plt
import numpy as np
# 1、散点输入
# 定义输入数据
data = [[0.8, 1.0], [1.7, 0.9], [2.7, 2.4], [3.2, 2.9], [3.7, 2.8], [4.2, 3.8], [4.2, 2.7]]
# 转换为 NumPy 数组
data = np.array(data)
# 提取 x_data 和 y_data
x_data = data[:, 0]
y_data = data[:, 1]
# 2、前向计算
# 预测对应的 y 的值
w = 0.8
b = 0
y_predicted = w * x_data + b  # 使用训练得到的 w 和 b 预测 y 值

# 3、单点误差
e = y_data - y_predicted
print(f"单点误差：{e}")

# 计算均方误差
e_bar = np.mean((y_data - y_predicted) ** 2)
print(f"均方误差：{e_bar}")
# 4、均方误差显示，绘图
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
# 计算线性回归直线的两个端点
y_lower_lr = w * 0 + b
y_upper_lr = w * 5 + b
# 绘制线性回归直线
ax1.set_xlim(0, 5)
ax1.set_ylim(0, 5)
ax1.set_xlabel("x axis label")
ax1.set_ylabel("y axis label")
ax1.scatter(x_data, y_data, color='b')  # 绘制训练数据散点图
ax1.plot([0, 5], [y_lower_lr, y_upper_lr], color='r', linewidth=3)  # 绘制线性回归直线

# 绘制损失函数的曲线
w_values = np.linspace(-2, 4, 100)
loss_values = [(np.mean((y_data - (w * x_data + b)) ** 2)) for w in w_values]
ax2.plot(w_values, loss_values, color='g', linewidth=2)
ax2.plot(w, e_bar, marker='o', markersize=8, color='r')  # 在损失函数曲线上标记对应的 w 点
ax2.set_xlabel("w")
ax2.set_ylabel("e")
# 连接散点和拟合线上对应的 x 轴相同的点
for x, y_true, y_pred in zip(x_data, y_data, y_predicted):
    ax1.plot([x, x], [y_true, y_pred], color='g', linestyle='--')
plt.subplots_adjust(wspace=0.3)
plt.show()
