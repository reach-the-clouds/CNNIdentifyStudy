import torchvision  # 先安装， 然后导入， 最后使用
import matplotlib.pyplot as plt
# 1. 下载数据集
# 训练集 : MNIST(下载的文件的保存路径， 是否为训练集， 是否下载)
# train=True ,代表训练集，  download = True ，代表从网络下载 ，  download = False , 表示读取本地的数据
train_data= torchvision.datasets.MNIST("" , train=True , download = True)
# 测试集 train=False  ，代表测试集 ，
test_data = torchvision.datasets.MNIST("" , train=False  , download= True)
# 2. 读取一条数据
image, label = train_data[1000] # image - 图片， label - 图片的标记
print(image , label)#<PIL.Image.Image image mode=L size=28x28 at 0x1BD39FB8800> 5
# image.convert("RGB")
plt.imshow(image)
plt.title(label)
plt.show() # 显示

