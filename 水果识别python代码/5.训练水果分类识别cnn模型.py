import json
import os
import warnings
import torch
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, RandomHorizontalFlip, ColorJitter,     RandomRotation, RandomResizedCrop, RandomErasing
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch.nn import CrossEntropyLoss
from torch.utils.mobile_optimizer import optimize_for_mobile
from fruitsdataset import FruitsDataset
from cnn_fruits import CNN_Fruits_Model

# 配置参数
USE_RESNET = False  # 是否使用ResNet作为骨干网络
BATCH_SIZE = 32     # 批处理大小
EPOCH = 5          # 训练轮数
LR = 0.01          # 学习率
TRAIN_RATIO = 0.8  # 训练集比例
TEST_RATIO = 0.2   # 测试集比例

# 设置设备（GPU优先）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 忽略一些警告信息
warnings.filterwarnings("ignore", category=UserWarning, message=r"The epoch parameter in `scheduler\.step\(\)`.*")
# 如果使用GPU，启用cudnn加速并增加批处理大小
if device.type == "cuda":
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    BATCH_SIZE = 64
else:
    BATCH_SIZE = BATCH_SIZE


def build_label_map(items):
    """
    构建标签映射

    参数:
        items (list): 包含图像路径和标签的字典列表

    返回:
        tuple: (标签到索引的映射字典, 类别名称列表)
    """
    from fruitsdataset import english_digit_label
    # 获取所有存在的标签并按英文数字顺序排序
    present = sorted({v['label'] for v in items}, key=lambda n: english_digit_label(n))
    names = list(present)
    # 创建标签到索引的映射
    label_map = {n: i for i, n in enumerate(names)}
    return label_map, names


# ImageNet数据集的均值和标准差
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
USE_IMAGENET_NORM = True  # 是否使用ImageNet的归一化参数


def compute_mean_std(items):
    """
    计算数据集的均值和标准差

    参数:
        items (list): 包含图像路径和标签的字典列表

    返回:
        tuple: (均值列表, 标准差列表)
    """
    # 创建临时数据集，只应用基本变换
    ds_tmp = FruitsDataset(items=items, transform=Compose([Resize((400, 400)), ToTensor()]), transform_label=lambda x: 0)
    loader = DataLoader(ds_tmp, batch_size=32, shuffle=False)

    # 初始化均值、方差和样本计数
    m = torch.zeros(3)
    v = torch.zeros(3)
    n = 0

    # 遍历数据集计算均值和方差
    for x, _ in loader:
        bs = x.size(0)
        n += bs
        m += x.mean(dim=[0, 2, 3]) * bs
        v += x.var(dim=[0, 2, 3], unbiased=False) * bs

    m /= max(n, 1)
    std = torch.sqrt(v / max(n, 1))
    return m.tolist(), std.tolist()


def main():
    """
    主函数：执行模型训练过程
    """
    # 获取当前脚本所在目录
    root = os.path.dirname(os.path.abspath(__file__))
    base = root

    def collect_dir(d):
        """
        收集目录下的所有图像文件及其标签

        参数:
            d (str): 数据集根目录路径

        返回:
            list: 包含图像路径和标签的字典列表
        """
        items = []
        if not os.path.isdir(d):
            return items
        # 遍历每个子目录（每个子目录代表一个类别）
        for c in os.scandir(d):
            if not c.is_dir():
                continue
            # 遍历子目录中的所有文件
            for f in os.scandir(c.path):
                if f.is_file():
                    items.append({"path": f.path, "label": c.name})
        return items

    # 尝试加载增强数据集，如果不存在则使用原始数据集
    strong_items = collect_dir(os.path.join(base, "strong_fruits_datasets"))
    items = strong_items if len(strong_items) > 0 else collect_dir(os.path.join(base, "fruits"))

    # 构建标签映射
    label_map, names = build_label_map(items)

    # 确定归一化参数
    if USE_IMAGENET_NORM:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        # 计算数据集的均值和标准差
        mean, std = compute_mean_std(items)
        # 保存归一化参数
        with open(os.path.join(root, "fruits_norm.json"), 'w', encoding='utf-8') as nf:
            json.dump({"mean": mean, "std": std}, nf)

    # 设置图像大小和数据变换
    img_size = 224
    # 训练集数据增强
    to_train = Compose([Resize((img_size, img_size)), RandomHorizontalFlip(),
                        RandomRotation(3), ColorJitter(0.05, 0.05, 0.05, 0.02), ToTensor(),
                        Normalize(mean=mean, std=std)])
    # 测试集数据变换（不进行数据增强）
    to_test = Compose([Resize((img_size, img_size)), ToTensor(), Normalize(mean=mean, std=std)])

    # 标签转换函数：将文本标签转换为索引
    def tl(x):
        return label_map[x]

    # 创建完整的数据集
    ds_train_full = FruitsDataset(items=items, transform=to_train, transform_label=tl)
    ds_test_full = FruitsDataset(items=items, transform=to_test, transform_label=tl)

    # 分割训练集和测试集
    tr, te = random_split(range(len(ds_train_full)), [int(TRAIN_RATIO * len(ds_train_full)),
                                                      len(ds_train_full) - int(TRAIN_RATIO * len(ds_train_full))])
    train_indices = tr.indices if hasattr(tr, 'indices') else tr
    test_indices = te.indices if hasattr(te, 'indices') else te

    # 统计每个类别的样本数量
    class_counts = {i: 0 for i in range(len(names))}
    labels_by_index = [it['label'] for it in items]
    for idx in train_indices:
        lbl = label_map[labels_by_index[idx]]
        class_counts[lbl] += 1
    print({names[i]: c for i, c in class_counts.items()})

    # 计算样本权重（用于处理类别不平衡）
    weights = []
    for idx in train_indices:
        lbl = label_map[labels_by_index[idx]]
        w = 1.0 / max(class_counts[lbl], 1)
        weights.append(w)

    # 是否使用加权采样器
    USE_WEIGHTED_SAMPLER = False
    sampler = WeightedRandomSampler(weights, num_samples=len(train_indices), replacement=True) if USE_WEIGHTED_SAMPLER else None

    # 创建子数据集
    from torch.utils.data import Subset
    train_set = Subset(ds_train_full, train_indices)
    test_set = Subset(ds_test_full, test_indices)

    # 配置数据加载器参数
    is_win = (os.name == 'nt')  # 是否为Windows系统
    num_workers = 0 if is_win else max(1, (os.cpu_count() or 2) // 2)  # 多进程加载数据
    pin_memory = device.type == 'cuda'  # 是否使用固定内存
    persistent = False if is_win else True  # 是否保持工作进程

    dl_kwargs = {
        'batch_size': BATCH_SIZE,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': persistent,
    }
    if num_workers > 0:
        dl_kwargs['prefetch_factor'] = 2  # 预取因子

    # 创建数据加载器
    train_loader = DataLoader(train_set,
                              sampler=sampler,
                              shuffle=(not USE_WEIGHTED_SAMPLER),
                              **dl_kwargs)
    test_loader = DataLoader(test_set,
                             shuffle=False,
                             **dl_kwargs)

    # 初始化模型
    model = CNN_Fruits_Model(num_classes=len(names)).to(device)

    # 定义损失函数（带标签平滑）
    loss_fun = CrossEntropyLoss(label_smoothing=0.05)

    # 定义优化器和学习率调度器
    opti = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, nesterov=True, weight_decay=5e-4)
    warm = torch.optim.lr_scheduler.LinearLR(opti, start_factor=0.2, total_iters=2)  # 预热阶段
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opti, T_max=EPOCH - 2)  # 余弦退火
    sched = torch.optim.lr_scheduler.SequentialLR(opti, schedulers=[warm, cos], milestones=[2])  # 组合调度器

    # 混合精度训练
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    def mixup_data(x, y, alpha=0.2):
        """
        Mixup数据增强方法

        参数:
            x (torch.Tensor): 输入图像
            y (torch.Tensor): 输入标签
            alpha (float): Beta分布参数

        返回:
            tuple: (混合图像, 原始标签A, 原始标签B, 混合比例)
        """
        if alpha <= 0:
            return x, y, y, 1.0
        # 从Beta分布中采样混合比例
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        # 随机排列索引
        index = torch.randperm(x.size(0))
        # 混合图像
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def evaluate_detailed(m, loader, num_classes, names):
        """
        详细评估模型性能，包括Top-1准确率、Top-5准确率和混淆矩阵

        参数:
            m (nn.Module): 模型
            loader (DataLoader): 数据加载器
            num_classes (int): 类别数
            names (list): 类别名称列表

        返回:
            tuple: (Top-1准确率, Top-5准确率, 混淆矩阵, 每个类别的准确率)
        """
        m.eval()
        # 初始化混淆矩阵
        cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
        tot = 0
        cor = 0
        top5 = 0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                o = m(x)
                _, p = torch.max(o, 1)

                # 更新混淆矩阵
                for i in range(y.size(0)):
                    cm[y[i].item(), p[i].item()] += 1

                tot += y.size(0)
                cor += (p == y).sum().item()

                # 计算Top-5准确率
                k = min(5, o.size(1))
                tk = torch.topk(o, k=k, dim=1).indices
                top5 += sum(int(y[i].item() in tk[i].tolist()) for i in range(y.size(0)))

        # 计算总体准确率
        acc = cor / max(tot, 1)
        top5_acc = top5 / max(tot, 1)

        # 计算每个类别的准确率
        per_class = {}
        for i in range(num_classes):
            row_sum = cm[i].sum().item()
            per_class[names[i]] = (cm[i, i].item() / row_sum) if row_sum > 0 else 0.0

        return acc, top5_acc, cm, per_class

    def eval_(m, loader):
        """
        简单评估模型性能

        参数:
            m (nn.Module): 模型
            loader (DataLoader): 数据加载器

        返回:
            float: 准确率
        """
        m.eval()
        with torch.no_grad():
            tot, cor = 0, 0
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                o = m(x)
                _, p = torch.max(o, 1)
                tot += y.size(0)
                cor += (p == y).sum().item()
            return cor / max(tot, 1)

    # 训练循环
    best = 0  # 最佳准确率
    for e in range(EPOCH):
        model.train()
        run = 0.0  # 累计损失

        # 训练阶段
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            # 梯度清零
            opti.zero_grad()

            # 应用Mixup数据增强（alpha=0表示不使用）
            mx, ya, yb, lam = mixup_data(x, y, alpha=0.0)

            # 前向传播（混合精度）
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                o = model(mx)
                # 计算混合损失
                loss = lam * loss_fun(o, ya) + (1 - lam) * loss_fun(o, yb)

            # 反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(opti)
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            # 更新参数
            scaler.step(opti)
            scaler.update()

            run += loss.item()

        # 评估阶段
        acc, top5_acc, cm, per_class = evaluate_detailed(model, test_loader, len(names), names)

        # 更新学习率
        sched.step()

        # 保存最佳模型
        if acc > best:
            best = acc
            # 保存模型权重
            torch.save(model.state_dict(), os.path.join(root, "fruits_model.pth"))
            # 转换并保存为移动端优化模型
            sm = torch.jit.script(model)
            opt_sm = optimize_for_mobile(sm)
            opt_sm._save_for_lite_interpreter(os.path.join(root, "fruits_model.ptl"))
            # 保存类别映射
            with open(os.path.join(root, "fruits_classes.json"), 'w', encoding='utf-8') as f:
                json.dump({"names": names, "label_to_index": label_map}, f, ensure_ascii=False)

        # 记录训练日志
        log_path = os.path.join(root, "train_log.txt")
        with open(log_path, 'a', encoding='utf-8') as lf:
            lf.write(f"epoch={e + 1} loss={run:.6f} acc1={acc:.4f} acc5={top5_acc:.4f}\n")
            # 记录表现最差的5个类别
            worst = sorted(per_class.items(), key=lambda kv: kv[1])[:5]
            lf.write("worst_classes: " + ", ".join([f"{k}:{v:.3f}" for k, v in worst]) + "\n")

        # 打印当前轮次结果
        print(f"{e + 1}/{EPOCH} loss={run:.6f} acc@1={acc:.4f} acc@5={top5_acc:.4f}")

    # 打印最佳准确率
    print(f"{best:.4f}")


if __name__ == "__main__":
    main()
