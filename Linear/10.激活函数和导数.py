import numpy as np
import matplotlib.pyplot as plt

class Sigmoid:
    def __init__(self):
        # 定义x的取值范围
        x = np.linspace(-10, 10, 100)

        # 计算Sigmoid函数的值
        y_sigmoid = self.sigmoid(x)

        # 计算Sigmoid函数导数的值
        y_derivative = self.sigmoid_derivative(x)

        # 绘制Sigmoid函数及其导数曲线
        plt.figure(figsize=(8, 6))
        plt.plot(x, y_sigmoid, label='Sigmoid')
        plt.plot(x, y_derivative, label='Sigmoid Derivative')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Sigmoid Function and its Derivative')
        plt.legend()
        plt.grid(True)
        plt.show()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))


class Tanh:
    def __init__(self):
        # 定义x的取值范围
        x = np.linspace(-10, 10, 100)

        # 计算tanh函数的值
        y_tanh = self.tanh(x)

        # 计算tanh函数导数的值
        y_derivative = self.tanh_derivative(x)

        # 绘制tanh函数及其导数曲线
        plt.figure(figsize=(8, 6))
        plt.plot(x, y_tanh, label='tanh')
        plt.plot(x, y_derivative, label='tanh Derivative')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('tanh Function and its Derivative')
        plt.legend()
        plt.grid(True)
        plt.show()

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2


class ReLU:
    def __init__(self):
        # 定义x的取值范围
        x = np.linspace(-10, 10, 100)

        # 计算ReLU函数的值
        y_relu = self.relu(x)

        # 计算ReLU函数导数的值
        y_derivative = self.relu_derivative(x)

        # 绘制ReLU函数及其导数曲线
        plt.figure(figsize=(8, 6))
        plt.plot(x, y_relu, label='ReLU')
        plt.plot(x, y_derivative, label='ReLU Derivative')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('ReLU Function and its Derivative')
        plt.legend()
        plt.grid(True)
        plt.show()

    def relu(self, x):
        return np.maximum(0, x) # 小于0 的就是0， 大于0 的就是x

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)


class Softmax:
    def __init__(self):
        x = np.linspace(-5, 5, 100)
        y_softmax = self.softmax(x)
        y_softmax_derivative = self.softmax_derivative(x)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, y_softmax)
        plt.title('Softmax Function')
        plt.xlabel('x')
        plt.ylabel('Softmax(x)')

        plt.subplot(1, 2, 2)
        plt.plot(x, y_softmax_derivative)
        plt.title('Softmax Derivative')
        plt.xlabel('x')
        plt.ylabel('Softmax\'(x)')

        plt.tight_layout()
        plt.show()

    def softmax(self, x):
        exp_vals = np.exp(x)
        return exp_vals / np.sum(exp_vals)

    def softmax_derivative(self, x):
        s = self.softmax(x)
        return np.diagflat(s) - np.outer(s, s)


Sigmoid()
Tanh()
ReLU()
Softmax()
