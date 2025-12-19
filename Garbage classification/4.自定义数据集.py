from  torch.utils.data import  Dataset  # 数据集类型

class MyDataset(Dataset): # 继承
    # 1. 构造函数
    def __init__(self , data , transform = None, transform_label = None):
        self.data = data
        self.transform = transform
        self.transform_label = transform_label
        pass
    # 2. __len__ 函数 ， 统计长度
    def __len__(self):
        # 数据集的长度
        return  len(self.data)

    # 3. __getitem__ 函数 ， 找到一条数据。（AI能直接使用的一条数据）
    def __getitem__(self, item): # item -- 索引
        one_data = self.data[item]
        pass
