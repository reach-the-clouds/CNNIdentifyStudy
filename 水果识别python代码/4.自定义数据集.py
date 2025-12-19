import os
import json
import torch
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
from torch.utils.data import DataLoader
from fruitsdataset import FruitsDataset, english_digit_label

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def collect_dir(d):
    items = []
    if not os.path.isdir(d):
        return items
    for c in os.scandir(d):
        if not c.is_dir():
            continue
        for f in os.scandir(c.path):
            if f.is_file():
                items.append({"path": f.path, "label": c.name})
    return items

def build_label_map(items):
    present = sorted({v['label'] for v in items}, key=lambda n: english_digit_label(n))
    names = list(present)
    label_map = {n: i for i, n in enumerate(names)}
    return label_map, names

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    strong = collect_dir(os.path.join(root, "strong_fruits_datasets"))
    items = strong if len(strong) > 0 else collect_dir(os.path.join(root, "fruits"))
    label_map, names = build_label_map(items)
    with open(os.path.join(root, "fruits_classes.json"), 'w', encoding='utf-8') as f:
        json.dump({"names": names, "label_to_index": label_map}, f, ensure_ascii=False)
    to_ = Compose([Resize((400, 400)), ToTensor(), Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    def tl(x):
        return label_map[x]
    ds = FruitsDataset(items=items, transform=to_, transform_label=tl)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    for x, y in loader:
        print(x.shape, y.shape)
        break

if __name__ == "__main__":
    main()
