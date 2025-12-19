# 前向计算 ： 模型， 输入数据
# 输入 ： （1, 28  , 28 ） , 输出  ： [0.6,2.6,0.03,.......]包含10个数据值 ， 找的是最大值的索引
import torch

from  fnn import  FNN_Model
from PIL import  Image , ImageOps
from torchvision.transforms import ToTensor , Resize , Compose
# 1. 模型
model = FNN_Model()
model.load_state_dict(torch.load('fnn.pt'))
# 2.输入数据
path = 'trueImgs/7.png'
image = Image.open(path) # Image类型
image = ImageOps.invert(image) # 背景反转
image = image.convert("L")# 灰度图
to_1 = Compose([ToTensor() , Resize((28,28))])
image_tensor = to_1(image)
print(image_tensor.shape) # 张量的形状

# 3. 计算输出
predit = model(image_tensor)
print(predit)
print(torch.argmax(predit))
