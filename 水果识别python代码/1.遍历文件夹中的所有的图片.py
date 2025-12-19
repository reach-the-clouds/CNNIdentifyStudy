import  os
# 遍历文件夹，统计出图片的数量
count = 0
sub_dir = os.scandir("fruits")
for dir in sub_dir:
    if dir.is_dir():
        files = os.scandir(dir)
        for file in files:
            print(file.path)
            count += 1
print(f"fruits文件夹中现在一共有{count}张图片")
