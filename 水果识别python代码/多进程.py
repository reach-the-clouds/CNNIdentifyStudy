# 导入必要的模块
import os  # 操作系统接口，用于文件系统操作
import sys  # 系统相关参数和函数
import argparse  # 命令行参数解析器
import concurrent.futures as cf  # 并行执行库，用于多进程处理
import importlib.util  # 动态导入模块的工具

# 尝试导入tqdm进度条库，如果导入失败则设为None
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

def _load_augmenter(base_dir: str):
    """
    动态加载水果数据增强类

    参数:
        base_dir: 基础目录路径

    返回:
        FruitsDataAugmenter类
    """
    # 构建数据增强脚本的完整路径
    p = os.path.join(base_dir, '2-1水果数据增强.py')
    # 根据文件路径创建模块规范
    spec = importlib.util.spec_from_file_location('augmenter', p)
    # 根据规范创建新模块
    m = importlib.util.module_from_spec(spec)
    # 执行模块以加载其内容
    spec.loader.exec_module(m)
    # 返回模块中的FruitsDataAugmenter类
    return m.FruitsDataAugmenter

def _iter_images_by_cat(root: str):
    """
    按类别遍历图像文件

    参数:
        root: 根目录路径

    返回:
        字典，键为类别名，值为该类别下的图像文件路径列表
    """
    cats = {}  # 存储各类别及其对应的文件列表
    # 遍历根目录下的所有条目
    for d in os.scandir(root):
        # 跳过非目录项
        if not d.is_dir():
            continue
        cat = d.name  # 类别名为目录名
        files = []  # 存储该类别下的所有图像文件
        # 遍历类别目录下的所有条目
        for fn in os.scandir(d.path):
            # 跳过子目录
            if fn.is_dir():
                continue
            # 获取文件扩展名并转为小写
            ext = os.path.splitext(fn.name)[1].lower()
            # 检查是否为支持的图像格式
            if ext in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                files.append(fn.path)  # 添加文件路径到列表
        # 如果该类别下有文件，则添加到字典中
        if files:
            cats[cat] = files
    return cats

def _worker(args_tuple):
    """
    工作进程函数，用于处理单个图像文件

    参数:
        args_tuple: 包含(base_dir, input_dir, output_dir, path, cat)的元组

    返回:
        处理结果，通常是生成的图像数量
    """
    # 解包参数元组
    base_dir, input_dir, output_dir, path, cat = args_tuple
    # 在工作进程中加载数据增强器类
    FruitsDataAugmenter = _load_augmenter(base_dir)
    # 创建数据增强器实例
    aug = FruitsDataAugmenter(input_dir=input_dir, output_dir=output_dir)
    # 处理单个图像文件并返回结果
    return aug.process_one(path, cat)

def main():
    """
    主函数：解析命令行参数，使用多进程处理图像数据
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加输入目录参数，默认为'./fruits'
    parser.add_argument('--input', default='./fruits')
    # 添加输出目录参数，默认为'./strong_fruits_datasets'
    parser.add_argument('--output', default='./strong_fruits_datasets')
    # 添加工作进程数参数，默认为CPU核心数或至少2
    parser.add_argument('--workers', type=int, default=max(2, (os.cpu_count() or 2)))
    # 解析命令行参数
    args = parser.parse_args()

    # 获取当前脚本所在目录
    base_dir = os.path.dirname(__file__)
    # 按类别获取所有图像文件
    cats = _iter_images_by_cat(args.input)
    # 如果没有找到图像，则退出
    if not cats:
        print('no images')
        return

    # 定义生成任务载荷的函数
    def _payloads(cat, files):
        # 为每个文件创建参数元组，供工作进程使用
        return [(base_dir, args.input, args.output, p, cat) for p in files]

    # 初始化统计变量
    files_total = 0  # 处理的文件总数
    outputs_total = 0  # 生成的图像总数

    # 创建进程池并行处理图像
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        # 按类别处理图像
        for cat, files in cats.items():
            # 为当前类别创建进度条
            bar = tqdm(total=len(files), desc=f'{cat}', unit='img') if tqdm else None
            # 使用进程池并行处理该类别的所有图像
            for n in ex.map(_worker, _payloads(cat, files)):
                files_total += 1  # 增加处理的文件计数
                outputs_total += int(n or 0)  # 增加生成的图像计数
                if bar:
                    bar.update(1)  # 更新进度条
            # 关闭进度条
            if bar:
                bar.close()
    # 打印总体处理结果
    print(f'done {files_total} files, generated {outputs_total} outputs')

# 当脚本直接运行时执行主函数
if __name__ == '__main__':
    main()
