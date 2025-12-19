import os
import json

def collect(dir_):
    out = []
    if not os.path.isdir(dir_):
        return out
    for d in os.scandir(dir_):
        if not d.is_dir():
            continue
        for f in os.scandir(d.path):
            if f.is_file():
                out.append((f.path, d.name))
    return out

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    dirs = [os.path.join(base, "strong_fruits_datasets"), os.path.join(base, "fruits")]
    items = []
    for dd in dirs:
        items.extend(collect(dd))
    labels = {}
    for i, (p, l) in enumerate(items):
        labels[i] = {"path": p, "label": l}
    with open(os.path.join(base, "labels.json"), "w", encoding="utf-8") as fp:
        json.dump(labels, fp, ensure_ascii=False)
    print(len(labels))

if __name__ == "__main__":
    main()
