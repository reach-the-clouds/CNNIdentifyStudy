import numpy as np
import matplotlib.pyplot as plt
# 1、散点输入
data = [[-0.5, 7.7], [1.8, 98.5], [0.9, 57.8], [0.4, 39.2], [-1.4, -15.7], [-1.4, -37.3], [-1.8, -49.1], [1.5, 75.6], [0.4, 34.0], [0.8, 62.3]]
print(type(data)) # <class 'list'>
data_np = np.array(data)
print(type(data_np)) # <class 'numpy.ndarray'>
x_data = data_np[:,0]
y_data = data_np[:,1]
# 2. 绘制散点图
plt.scatter(x_data , y_data , c='r' , linewidths=1)
plt.xlim([-2 , 2])
plt.ylim([-110 , 110])
plt.xlabel("X")
plt.ylabel("Y")

# 2、参数初始化
w = 0
b = 0
learning_rate = 0.01
# 数据个数，求平均用
l_x_data = x_data.size
# 记录梯度
gd_path = []
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
plt.plot([-2 , 2] ,[-2*w+b , 2*w + b] , 'b--')
print(f"w:{w} , b：{b}")
plt.show()