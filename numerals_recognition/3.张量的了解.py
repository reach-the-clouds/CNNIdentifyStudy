# AI模型能使用的数据类型就是张量 ， 模型训练的时候， 或者让模型预测的时候都需要把数据转换为张量。
import torch
a1 = [1,2,3,4,5,6]  # <class 'list'>
print(type(a1))
# 张量： tensor
tensor1 = torch.Tensor(a1)
print(type(tensor1)) # <class 'torch.Tensor'>
print(tensor1) #tensor([1., 2., 3., 4., 5., 6.])
# 张量的特征： shape
print(tensor1.shape) # torch.Size([6]) , 一维， 有6个数据值
a2 = [[1,2,3],
      [4,5,6],
      [7,8,9]]
tensor2 = torch.Tensor(a2)
print(tensor2.shape) # torch.Size([3, 3])  , 二维