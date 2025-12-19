import cv2
import numpy as np

import os
# 打印当前工作目录（脚本运行时的基准目录）
print("当前工作目录：", os.getcwd())
# 打印该目录下的所有文件，确认是否有目标图片
print("目录下的文件：", os.listdir('.'))

if __name__ == "__main__":
    path = "img/1.jpeg"
    image_np = cv2.imread(path)
    img_shape = image_np.shape
    image_np_gray = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)  # image_np.copy()
    # 加权灰度化
    wr = 0.299
    wg = 0.587
    wb = 0.114
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            image_np_gray[i, j] = (int(wr * image_np[i, j][2]) + int(wg * image_np[i, j][1]) + int(
                wb * image_np[i, j][0]))
    cv2.imshow("image_np_gray", image_np_gray)
    cv2.waitKey(0)
