import torchvision  # 先安装， 然后导入， 最后使用
import matplotlib.pyplot as plt
import os  # 导入os模块用于创建目录

# 1. 读取本地数据集
# 训练集 : MNIST(下载的文件的保存路径， 是否为训练集， 是否下载)
# train=True ,代表训练集，  download = True ，代表从网络下载 ，  download = False , 表示读取本地的数据
train_data= torchvision.datasets.MNIST("" , train=True , download = False)
# 测试集 train=False  ，代表测试集 ，
test_data = torchvision.datasets.MNIST("" , train=False  , download= False)

# 创建保存图片的目录（如果不存在）
os.makedirs("trueImgs", exist_ok=True)  # exist_ok=True表示如果目录已存在则不报错

# 2.保存24张图片
iter = iter(train_data) # 迭代器
for i in range(24):
    image , label = next(iter)
    #保存图片： 图片命名  {i}_{label}.png
    image.save(f"./trueImgs/{i}.png")
    # 图片显示
    plt.subplot(3,8, i+1 )  # 3行8列， i+1代表第几张图片。
    plt.imshow(image) # 图片的设置
    plt.title(label)
plt.show()

