# 数据集模块
from torch.utils.data import Dataset , random_split
import json
from PIL import Image
from torchvision.transforms import ToTensor , Resize , Compose
# 定义一个把英文标记，转换为数字标记的函数
def english_digit_label(label):
    label_dict = {  # 让AI学习的时候，记住的是数据值， 比如学习的是glass这个类别的图片， 他记住
                        # 这个是2.
        "cardboard": 0 ,
        "glass":2 ,
        "metal": 3  ,
        "paper": 4,
        "plastic":5 ,
        "trash": 1
    }
    return  label_dict[label] # 传入的是英文的类别， 返回数字
# 1， 封装一个类型
class  GarbageDataset(Dataset):
    # 1, 构造函数
    def __init__(self , label_path , transform=None , transform_label = None ):
        with open(label_path  , 'r') as file:
            self.data = json.load(file) # 数据
        self.transform = transform # 张量转换
        self.transform_label = transform_label # 标注转换（把英文的类型，转换数字）
    # 2, 长度函数
    def __len__(self):
        return  len(self.data)
    # 3， 获取一条数据的函数
    def __getitem__(self, item): # item就是索引
        # 将整数索引转换为字符串键，因为JSON中的键是字符串
        str_item = str(item)
        one_data = self.data[str_item]
        path =  one_data['path']
        label = one_data['label']
        image = Image.open(path)
        # 判断是否需要把英文的label转换为数字的label
        if self.transform_label is not None:
            label = self.transform_label(label)
        if self.transform is not None:
            image = self.transform(image)
        return  image , label # 返回值要求，必须返回两个， 第一个是图片转换的张量， 第二个就是标注
if __name__ == '__main__':
    to_1 = Compose([ToTensor(), Resize((400,400))])
    dataset = GarbageDataset("labels.json" ,transform= to_1 ,
                             transform_label=english_digit_label)
    print(f"长度： {dataset.__len__()}")
    #print(f"一条数据： {dataset.__getitem__('1')}")
    train_set , test_set = random_split(dataset  , [0.8 , 0.2]) # 数据集的分割
    print(f"训练集数量： {train_set.__len__()} , 测试集的数量： {test_set.__len__()}")