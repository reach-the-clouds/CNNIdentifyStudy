import numpy as np
import time

# 民宿的曝光度，决定了名宿的价格。 下面就是一组统计出来的数据。
# 请你帮我预测， 如果曝光度为8 的时候，名宿的价格怎么定？

def loss_function(w, b, x_data, y_data):
    """
    计算均方误差损失函数
    
    参数:
        w: 权重参数
        b: 偏置参数
        x_data: 输入特征数据（曝光度）
        y_data: 目标值数据（价格）
    
    返回:
        均方误差损失值
    """
    y_predicted = w * x_data + b  # 预测值
    loss = np.mean((y_data - y_predicted) ** 2)  # 均方误差
    return loss

def compute_gradients(w, b, x_data, y_data):
    """
    计算损失函数对w和b的梯度
    
    参数:
        w: 当前权重参数
        b: 当前偏置参数
        x_data: 输入特征数据
        y_data: 目标值数据
    
    返回:
        dw: w的梯度
        db: b的梯度
    """
    n = len(y_data)  # 数据点数量
    y_predicted = w * x_data + b  # 当前预测值
    
    # 计算w的梯度: dE/dw = -(2/n)Σx_i(y_i - (w*x_i + b))
    dw = -(2/n) * np.sum(x_data * (y_data - y_predicted))
    
    # 计算b的梯度: dE/db = -(2/n)Σ(y_i - (w*x_i + b))
    db = -(2/n) * np.sum(y_data - y_predicted)
    
    return dw, db

def gradient_descent(x_data, y_data, learning_rate=0.001, num_iterations=10000, tolerance=1e-6):
    """
    使用梯度下降法训练线性回归模型
    
    参数:
        x_data: 输入特征数据
        y_data: 目标值数据
        learning_rate: 学习率（步长）
        num_iterations: 最大迭代次数
        tolerance: 收敛阈值
    
    返回:
        w: 训练后的权重参数
        b: 训练后的偏置参数
        loss_history: 损失函数历史值
    """
    # 初始化参数
    w = 0.0
    b = 0.0
    loss_history = []
    
    for i in range(num_iterations):
        # 计算当前损失
        current_loss = loss_function(w, b, x_data, y_data)
        loss_history.append(current_loss)

        # 计算梯度
        dw, db = compute_gradients(w, b, x_data, y_data)
        
        # 更新参数
        w_new = w - learning_rate * dw
        b_new = b - learning_rate * db
        
        # 检查是否收敛
        if np.abs(w_new - w) < tolerance and np.abs(b_new - b) < tolerance:
            print(f"迭代在 {i+1} 次后收敛")
            break
        
        # 更新参数值
        w, b = w_new, b_new
        
        # 每100次迭代打印一次信息
        if (i+1) % 100 == 0:
            print(f"迭代次数: {i+1}, w: {w:.4f}, b: {b:.4f}, 损失值: {current_loss:.4f}")
    
    return w, b, loss_history

if __name__ == "__main__":
    # 定义输入数据
    data = [[0, 330], [1, 580], [2, 700], [3, 950], [4, 1300], [5, 1400], [6, 1520]]
    
    # 转换为 NumPy 数组
    data = np.array(data)
    x_data = data[:, 0]  # 曝光度
    y_data = data[:, 1]  # 价格
    
    # 训练模型
    learning_rate = 0.001
    num_iterations = 10000
    
    print("开始训练线性回归模型...")
    start_time = time.time()
    
    w, b, loss_history = gradient_descent(x_data, y_data, learning_rate, num_iterations)
    
    end_time = time.time()
    print(f"训练完成，耗时: {end_time - start_time:.4f} 秒")
    
    # 输出最终结果
    print("\n最终训练结果:")
    print(f"权重 w: {w:.4f}")
    print(f"偏置 b: {b:.4f}")
    print(f"最终损失值: {loss_function(w, b, x_data, y_data):.4f}")
    print(f"回归方程: y = {w:.4f} * x + {b:.4f}")
    
    # 预测曝光度为8时的价格
    exposure = 8
    predicted_price = w * exposure + b
    print(f"\n预测结果:")
    print(f"当曝光度为 {exposure} 时，民宿的价格预测为: {predicted_price:.2f}")
    
    # 预测曝光度为7时的价格（用于比较原始代码结果）
    exposure_7 = 7
    predicted_price_7 = w * exposure_7 + b
    print(f"当曝光度为 {exposure_7} 时，民宿的价格预测为: {predicted_price_7:.2f}")