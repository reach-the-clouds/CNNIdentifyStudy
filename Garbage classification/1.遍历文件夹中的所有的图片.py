import  os
# 遍历文件夹，统计出图片的数量
count = 0
sub_dir = os.scandir("strong_garbage_datasets")
for dir in sub_dir:
    if dir.is_dir():
        files = os.scandir(dir)
        for file in files:
            print(file.path)
            count += 1
print(f"strong_garbage_datasets文件夹中现在一共有{count}张图片")
