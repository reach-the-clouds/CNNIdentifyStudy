import os
import json
labels={} # 字典 : key:value  ,key1 : value1 .....
index = 0  # 作为字典数据的key.
# 遍历没每一张图片 ， 把图片及其对应的标记，保存到这个labels类型中。
sub_dir = os.scandir("strong_garbage_datasets")
for dir in sub_dir:
    if dir.is_dir():
        files = os.scandir(dir)
        for file in files:
            labels[index] = {
                'path': file.path, # 图片的路径
                'label': dir.name # 当前文件所在文件夹的名字，就是这个图片的类型。
            }
            index = index + 1
# 存储为一个json格式的文件
with open('labels.json' , 'w') as file:
    json.dump(labels , file) # 写入到文件中
    print(f"标注文件已经生成， 一共有{index}条数据。")

# ？？  strong_garbage_datasets 文件夹中的文件可以删除一些吗？
#  如果strong_garbage_datasets文件夹中的图片被删掉了一部分，就需要重新生成新的标注文件。
#  因为AI训练的时候， 读取的数据集依赖的是通过labels.json文件中保存的图片路径取找到硬盘的图片本身。