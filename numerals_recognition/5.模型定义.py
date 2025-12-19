import torch
# 定义一个神经网络： 继承torch.nn.Module ， 然后写构造函数， forward（前向计算）函数。
#  前向计算： y = 3 x + 4 ,  知道x = 5 , 计算出y的过程就叫前向计算。
# 构造函数中需要调用super函数， 在构造函数中定义神经网络层
class MyModel(torch.nn.Module): # 继承
    # 构造函数
    def __init__(self):
        super(MyModel , self).__init__()
        print("---定义神经网络的层次结构--")
        pass

    # 前向计算的函数的名字，必须叫forward
    # 给模型传入参数，就会直接调用forward函数。
    # 训练过程中会调用到forward函数，模型使用过程中的预测也是调用forward函数
    def forward(self , x):
        return  x * x ;

if __name__ == '__main__':
    model = MyModel()
    predict = model(10) # 本质就是在调用forward
    print(f"预测值：{predict}")

