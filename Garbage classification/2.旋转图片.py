from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# 1. 定义元组， 用于表示要旋转的角度
deg = (45 , 90 , 135, 180  , 225 , 270 , 315)
# 2. 打开一张图片
image = Image.open("香蕉2.jpg")

i = 1
plt.subplot(2, 4 ,1) # 2行四列的第一张图片
plt.imshow(image)
plt.xlabel("source")

for d in deg:
    img_d = transforms.functional.rotate(image , d) # 旋转的是原图。
    i = i + 1
    plt.subplot(2, 4 , i )
    plt.imshow(img_d)
    plt.xlabel(f"deg:{d}")
plt.show()

