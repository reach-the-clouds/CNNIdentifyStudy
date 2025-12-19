import matplotlib.pyplot as plt
import numpy as np


def math_linear_regression_train(x_tarin, y_train):
    x_mean = np.mean(x_tarin)  # 计算 x 值的均值
    y_mean = np.mean(y_train)  # 计算 y 值的均值
    numerator = 0.0
    denominator = 0.0
    # 根据数据计算 w
    for x_i, y_i in zip(x_tarin, y_train):
        numerator += (x_i - x_mean) * (y_i - y_mean)
        denominator += (x_i - x_mean) ** 2
    # 计算 w 值
    w = numerator / denominator
    # 计算 b 值
    b = y_mean - w * x_mean
    print(f"w={w}, b={b}")
    return w, b


# 散点输入
data = [[0.8, 1.0], [1.7, 0.9], [2.7, 2.4], [3.2, 2.9], [3.7, 2.8], [4.2, 3.8], [4.2, 2.7]]
# 转换为 NumPy 数组
data = np.array(data)
# 提取 x_data 和 y_data
x_data = data[:, 0]
y_data = data[:, 1]

# 2.求w/b数学解
w, b = math_linear_regression_train(x_data, y_data)

# 计算线性回归直线的两个端点
y_lower_lr = w * 0 + b
y_upper_lr = w * 5 + b

# 3.拟合曲线显示
plt.xlim((0, 5))
plt.ylim((0, 5))
plt.xlabel("x axis label")
plt.ylabel("y axis label")
plt.scatter(x_data, y_data, color='b')  # 绘制训练数据散点图
plt.plot([0, 5], [y_lower_lr, y_upper_lr], color='r', linewidth=3)  # 绘制线性回归直线
plt.show()
