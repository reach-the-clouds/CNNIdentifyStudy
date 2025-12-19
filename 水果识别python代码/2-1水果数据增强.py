import os
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms

class FruitsDataAugmenter:
    """
    水果数据增强类，用于对水果图像进行多种数据增强处理

    功能包括：
    - 图像旋转
    - 亮度调整
    - 对比度调整
    - 模糊处理
    - 水平/垂直翻转
    """

    def __init__(self, input_dir="./fruits",
                 output_dir="./strong_fruits_datasets",
                 rotate_angles=(45, 90, 135, 180, 225, 270, 315),
                 brightness_factors=(0.8, 1.2),
                 contrast_factors=(0.8, 1.2),
                 blur_radii=(1.0,),
                 enable_flip=True,
                 enable_brightness=True,
                 enable_contrast=True,
                 enable_blur=True,
                 target_size=(400, 400)):
        """
        初始化水果数据增强器

        参数:
            input_dir: 输入图像目录，默认为"./fruits"
            output_dir: 输出增强后图像的目录，默认为"./strong_fruits_datasets"
            rotate_angles: 旋转角度集合，默认为(45, 90, 135, 180, 225, 270, 315)
            brightness_factors: 亮度调整因子集合，默认为(0.8, 1.2)
            contrast_factors: 对比度调整因子集合，默认为(0.8, 1.2)
            blur_radii: 模糊半径集合，默认为(1.0,)
            enable_flip: 是否启用翻转功能，默认为True
            enable_brightness: 是否启用亮度调整，默认为True
            enable_contrast: 是否启用对比度调整，默认为True
            enable_blur: 是否启用模糊处理，默认为True
            target_size: 目标图像尺寸，默认为(400, 400)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.rotate_angles = rotate_angles
        self.brightness_factors = brightness_factors
        self.contrast_factors = contrast_factors
        self.blur_radii = blur_radii
        self.enable_flip = enable_flip
        self.enable_brightness = enable_brightness
        self.enable_contrast = enable_contrast
        self.enable_blur = enable_blur
        self.target_size = target_size
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        # 创建图像缩放转换器
        self.resize = transforms.Resize(self.target_size)

    def _ensure_dir(self, d):
        """
        确保目录存在，若不存在则创建

        参数:
            d: 要确保存在的目录路径
        """
        os.makedirs(d, exist_ok=True)

    def process_one(self, image_path, category):
        """
        对单个图像进行数据增强处理

        参数:
            image_path: 输入图像路径
            category: 图像所属类别，用于创建输出子目录

        返回:
            生成的增强图像数量（包括原始图像）
        """
        try:
            # 打开图像并转换为RGB格式
            img = Image.open(image_path).convert("RGB")
            # 调整图像尺寸
            img = self.resize(img)
            # 获取文件路径信息
            p = Path(image_path)
            # 创建输出目录
            out_dir = os.path.join(self.output_dir, category)
            self._ensure_dir(out_dir)
            # 获取文件名和扩展名
            base = p.stem
            ext = p.suffix
            img_dir = os.path.join(out_dir, base)
            self._ensure_dir(img_dir)
            img.save(os.path.join(img_dir, f"original{ext}"))
            n = 1  # 计数器，已保存原始图像

            # 旋转图像
            for angle in self.rotate_angles:
                r = transforms.functional.rotate(img, angle)
                r.save(os.path.join(img_dir, f"rot{angle}{ext}"))
                n += 1

            # 水平和垂直翻转
            if self.enable_flip:
                h = transforms.functional.hflip(img)  # 水平翻转
                h.save(os.path.join(img_dir, f"flipH{ext}"))
                n += 1
                v = transforms.functional.vflip(img)  # 垂直翻转
                v.save(os.path.join(img_dir, f"flipV{ext}"))
                n += 1

            # 亮度调整
            if self.enable_brightness:
                for bf in self.brightness_factors:
                    b = ImageEnhance.Brightness(img).enhance(bf)
                    b.save(os.path.join(img_dir, f"bright{bf}{ext}"))
                    n += 1

            # 对比度调整
            if self.enable_contrast:
                for cf in self.contrast_factors:
                    c = ImageEnhance.Contrast(img).enhance(cf)
                    c.save(os.path.join(img_dir, f"contrast{cf}{ext}"))
                    n += 1

            # 高斯模糊
            if self.enable_blur:
                for br in self.blur_radii:
                    bl = img.filter(ImageFilter.GaussianBlur(radius=br))
                    bl.save(os.path.join(img_dir, f"blur{br}{ext}"))
                    n += 1

            return n
        except Exception:
            # 如果处理过程中出现异常，返回0
            return 0

    def process_class(self, in_dir, out_dir):
        """
        处理一个类别的所有图像

        参数:
            in_dir: 输入类别目录路径
            out_dir: 输出类别目录路径

        返回:
            处理的图像总数
        """
        # 确保输出目录存在
        self._ensure_dir(out_dir)
        total = 0
        # 遍历输入目录中的所有文件
        for fn in os.listdir(in_dir):
            p = os.path.join(in_dir, fn)
            # 跳过子目录
            if os.path.isdir(p):
                continue
            # 获取文件名（不含扩展名）
            stem = os.path.splitext(fn)[0]
            base_out = os.path.join(out_dir, stem)
            # 处理单个图像并累加计数
            total += self.process_one(p, Path(in_dir).name)
        return total

    def process_all_categories(self):
        """
        处理所有类别的图像

        返回:
            处理的图像总数
        """
        # 检查输入目录是否存在
        if not os.path.exists(self.input_dir):
            return 0
        # 获取所有子目录（即所有类别）
        cats = [d for d in os.scandir(self.input_dir) if d.is_dir()]
        if not cats:
            return 0
        total = 0
        # 逐个类别处理
        for c in cats:
            # 创建对应的输出目录
            out_c = os.path.join(self.output_dir, c.name)
            # 处理该类别下的所有图像
            total += self.process_class(c.path, out_c)
        return total

if __name__ == "__main__":
    """
    主程序入口，创建数据增强器实例并处理所有类别的图像
    """
    # 创建水果数据增强器实例
    aug = FruitsDataAugmenter(
        input_dir="./fruits",  # 输入目录
        output_dir="./strong_fruits_datasets",  # 输出目录
        rotate_angles=(45, 90, 135, 180, 225, 270, 315),  # 旋转角度
        brightness_factors=(0.8, 1.2),  # 亮度调整因子
        contrast_factors=(0.8, 1.2),  # 对比度调整因子
        blur_radii=(1.0,),  # 模糊半径
        enable_flip=True,  # 启用翻转
        enable_brightness=True,  # 启用亮度调整
        enable_contrast=True,  # 启用对比度调整
        enable_blur=True,  # 启用模糊处理
    )
    # 处理所有类别的图像
    aug.process_all_categories()
