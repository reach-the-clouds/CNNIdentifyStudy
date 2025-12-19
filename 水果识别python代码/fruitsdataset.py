from torch.utils.data import Dataset
import json
from PIL import Image
import os

_LABEL_MAP_CACHE = None


def english_digit_label(label):
    global _LABEL_MAP_CACHE
    if _LABEL_MAP_CACHE is None:
        base = os.path.dirname(os.path.abspath(__file__))
        strong = os.path.join(base, "strong_fruits_datasets")
        fruits = os.path.join(base, "fruits")
        target = strong if os.path.isdir(strong) and any(d.is_dir() for d in os.scandir(strong)) else fruits
        names = [d.name for d in os.scandir(target) if d.is_dir()]
        names = sorted(names)
        _LABEL_MAP_CACHE = {n: i for i, n in enumerate(names)}
    return _LABEL_MAP_CACHE[label]


class FruitsDataset(Dataset):
    def __init__(self, label_path=None, transform=None, transform_label=None, items=None):
        if items is not None:
            self.data = items
            self._from_items = True
        else:
            with open(label_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            self._from_items = False
        self.transform = transform
        self.transform_label = transform_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self._from_items:
            one = self.data[idx]
            path = one['path']
            label = one['label']
        else:
            k = str(idx)
            one = self.data[k]
            path = one['path']
            label = one['label']
        img = Image.open(path).convert('RGB')
        if self.transform_label is not None:
            label = self.transform_label(label)
        else:
            label = english_digit_label(label)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
