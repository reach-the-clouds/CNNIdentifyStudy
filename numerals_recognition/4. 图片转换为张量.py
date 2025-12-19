from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os

# Image --  把file类型，转换为Image类型
# ToTensor  --  可以把Image转换为Tensor类型

# 定义图片路径，使用更可靠的相对路径
path = 'trueImgs/0.png'  # 文件路径

# 检查文件是否存在
if not os.path.exists(path):
    # 如果文件不存在，尝试使用images目录（根据之前代码推断）
    alternative_path = 'trueImgs/0.png'
    if os.path.exists(alternative_path):
        print(f"文件 {path} 不存在，使用 {alternative_path} 代替")
        path = alternative_path
    else:
        # 如果两个路径都不存在，则报错
        raise FileNotFoundError(f"找不到文件 {path} 或 {alternative_path}")

# 打开图片
image = Image.open(path)  # Image类型
image = image.convert("L")  # 转换为灰度图像
# 背景反转
image = ImageOps.invert(image)
print(image)  # 打印图像对象信息

# 定义图像转换流程：先转换为张量，再调整大小
# 注意：通常应该先调整大小，再转换为张量
to_1 = Compose([Resize((28,28)), ToTensor()])  # 转换张量的时候，修改大小
tensor_1 = to_1(image)  # 张量类型
print(tensor_1.shape)  # torch.Size([1, 28, 28]) ，三维数据
print(tensor_1)  # 打印张量数据

# 显示图像
plt.imshow(image, cmap='gray')  # 使用灰度色彩映射显示灰度图像
plt.show()