import matplotlib.pyplot as plt
import numpy as np
import time


def loss_function(w, b, x_data, y_data):
    # 预测对应的 y 的值
    y_predicted = w * x_data + b  # 使用训练得到的 w 和 b 预测 y 值
    # 计算均方误差
    e_bar = np.mean((y_data - y_predicted) ** 2)
    return e_bar


# 定义输入数据
data = [[0.8, 1.0], [1.7, 0.9], [2.7, 2.4], [3.2, 2.9], [3.7, 2.8], [4.2, 3.8], [4.2, 2.7]]
# 转换为 NumPy 数组
data = np.array(data)
# 提取 x_data 和 y_data
x_data = data[:, 0]
y_data = data[:, 1]

# 计算线性回归直线的两个端点
w = 0
b = 0

# 迭代更新 w_old 的值并重新绘制曲线
w_old = w
min_fixed_value = 0.01  # 选项三中使用的小固定值
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
w_values = np.linspace(-10, 10, 200)  # w 取值范围
loss_values = [(np.mean((y_data - (w * x_data + b)) ** 2)) for w in w_values]  # 计算损失函数的值

num_iterations = 1
for i in range(num_iterations + 1):
    # 清空上次绘制的内容
    ax1.cla()
    ax2.cla()

    # 左侧子图：损失函数曲线和切线
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-800, 800)
    ax1.set_xlabel("w")
    ax1.set_ylabel("e")
    ax1.set_title("Loss Function")
    ax1.plot(w_values, loss_values, color='g', linewidth=2)  # 绘制损失函数曲线
    ax1.plot(w_old, loss_function(w_old, b, x_data, y_data), marker='o', markersize=8, color='r')  # 标记当前 w 值对应的损失

    # 计算切线斜率和截距
    tangent_point = np.mean((y_data - (w_old * x_data + b)) ** 2)  # 在 w_old 处对应的损失
    tangent_slope = -2 * np.mean(x_data * y_data) + 2 * w_old * np.mean(x_data ** 2)  # 切线的斜率为损失函数在 w_old 处的导数
    tangent_intercept = tangent_point - tangent_slope * w_old  # 切线的截距

    # 绘制切线
    tangent_line = tangent_slope * w_values + tangent_intercept
    ax1.plot(w_values, tangent_line, color='b', linestyle='--')  # 绘制切线

    # 右侧子图：散点和拟合直线
    ax2.set_xlim(0, 7)
    ax2.set_ylim(-15, 15)
    ax2.set_xlabel("x axis label")
    ax2.set_ylabel("y axis label")
    ax2.scatter(x_data, y_data, color='b')  # 绘制训练数据散点图
    y_lower_lr = w_old * 0 + b
    y_upper_lr = w_old * 5 + b
    ax2.plot([0, 7], [y_lower_lr, y_upper_lr], color='r', linewidth=3)  # 绘制线性回归直线
    ax2.set_title("Data Scatter and Fitted Line")

    # 根据不同选项更新 w_old 的值, 减去切线斜率乘以最小固定值
    w_new = w_old - tangent_slope * min_fixed_value
    w_old = w_new

    # 显示图形
    time.sleep(3)
    # 清空Jupyter输出
plt.show()
