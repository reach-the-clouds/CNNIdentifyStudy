# 垃圾分类数据增强脚本
# 支持多种数据增强方法：旋转、翻转、亮度调整、对比度调整等
# 并将所有图片调整为统一的400*400大小
import os
import random
from PIL import Image, ImageEnhance
from torchvision import transforms
from pathlib import Path

class GarbageDataAugmenter:
    def __init__(self, input_dir="./Garbage classification", 
                 output_dir="./strong_garbage_datasets",
                 rotate_angles=(45, 90, 135, 180, 225, 270, 315),
                 brightness_factors=(0.8, 1.2),
                 contrast_factors=(0.8, 1.2),
                 enable_flip=True,
                 enable_brightness=True,
                 enable_contrast=True,
                 target_size=(400, 400)):
        """
        初始化数据增强器
        
        参数:
        - input_dir: 输入数据集目录
        - output_dir: 增强后的数据集保存目录
        - rotate_angles: 旋转角度元组
        - brightness_factors: 亮度调整因子范围
        - contrast_factors: 对比度调整因子范围
        - enable_flip: 是否启用翻转增强
        - enable_brightness: 是否启用亮度调整
        - enable_contrast: 是否启用对比度调整
        - target_size: 调整后的图片尺寸，默认(400, 400)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.rotate_angles = rotate_angles
        self.brightness_factors = brightness_factors
        self.contrast_factors = contrast_factors
        self.enable_flip = enable_flip
        self.enable_brightness = enable_brightness
        self.enable_contrast = enable_contrast
        self.target_size = target_size
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建一个调整大小的转换器
        self.resize_transform = transforms.Resize(self.target_size)
    
    def _ensure_dir(self, dir_path):
        """确保目录存在"""
        os.makedirs(dir_path, exist_ok=True)
    
    def augment_image(self, image_path, category):
        """对单张图片进行多种数据增强"""
        try:
            # 打开并转换为RGB格式
            image = Image.open(image_path)
            image = image.convert("RGB")
            
            # 调整图片大小为统一尺寸
            image = self.resize_transform(image)
            
            # 生成输出路径
            path = Path(image_path)
            category_output_dir = os.path.join(self.output_dir, category)
            self._ensure_dir(category_output_dir)
            
            # 保存原图（已调整大小）
            base_filename = path.stem
            ext = path.suffix
            original_output_path = os.path.join(category_output_dir, f"{base_filename}_original{ext}")
            image.save(original_output_path)
            
            augmented_count = 1  # 已保存原图
            
            # 1. 旋转增强
            for angle in self.rotate_angles:
                rotated_image = transforms.functional.rotate(image, angle)
                rotated_output_path = os.path.join(category_output_dir, 
                                                 f"{base_filename}_rot{angle}{ext}")
                rotated_image.save(rotated_output_path)
                augmented_count += 1
            
            # 2. 翻转增强
            if self.enable_flip:
                # 水平翻转
                h_flipped = transforms.functional.hflip(image)
                h_flipped_path = os.path.join(category_output_dir, 
                                            f"{base_filename}_flipH{ext}")
                h_flipped.save(h_flipped_path)
                augmented_count += 1
                
                # 垂直翻转
                v_flipped = transforms.functional.vflip(image)
                v_flipped_path = os.path.join(category_output_dir, 
                                            f"{base_filename}_flipV{ext}")
                v_flipped.save(v_flipped_path)
                augmented_count += 1
            
            # 3. 亮度调整
            if self.enable_brightness:
                for factor in self.brightness_factors:
                    enhancer = ImageEnhance.Brightness(image)
                    bright_image = enhancer.enhance(factor)
                    bright_output_path = os.path.join(category_output_dir, 
                                                     f"{base_filename}_bright{factor}{ext}")
                    bright_image.save(bright_output_path)
                    augmented_count += 1
            
            # 4. 对比度调整
            if self.enable_contrast:
                for factor in self.contrast_factors:
                    enhancer = ImageEnhance.Contrast(image)
                    contrast_image = enhancer.enhance(factor)
                    contrast_output_path = os.path.join(category_output_dir, 
                                                       f"{base_filename}_contrast{factor}{ext}")
                    contrast_image.save(contrast_output_path)
                    augmented_count += 1
            
            return augmented_count
            
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {str(e)}")
            return 0
    
    def process_all_categories(self):
        """处理所有类别"""
        total_images = 0
        
        # 检查输入目录是否存在
        if not os.path.exists(self.input_dir):
            print(f"错误: 输入目录 {self.input_dir} 不存在")
            return 0
        
        # 获取所有子目录（类别）
        try:
            categories = [d for d in os.scandir(self.input_dir) if d.is_dir()]
            if not categories:
                print(f"警告: 输入目录 {self.input_dir} 中没有找到子目录")
                return 0
                
            print(f"开始数据增强，找到 {len(categories)} 个类别...")
            print(f"所有图片将被调整为 {self.target_size} 尺寸并转换为RGB格式")
            
            # 处理每个类别
            for category in categories:
                category_name = category.name
                category_path = category.path
                category_count = 0
                
                print(f"  处理类别: {category_name}...")
                
                # 处理该类别下的所有图片
                try:
                    image_files = [f for f in os.scandir(category_path) 
                                  if f.is_file() and f.name.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    
                    for image_file in image_files:
                        count = self.augment_image(image_file.path, category_name)
                        category_count += count
                    
                    total_images += category_count
                    print(f"  类别 {category_name} 增强完成，生成 {category_count} 张图片")
                    
                except Exception as e:
                    print(f"  处理类别 {category_name} 时出错: {str(e)}")
            
            print(f"数据增强完成！总计生成 {total_images} 张图片")
            return total_images
            
        except Exception as e:
            print(f"处理数据时发生错误: {str(e)}")
            return 0

# 主程序
if __name__ == "__main__":
    # 创建数据增强器实例
    augmenter = GarbageDataAugmenter(
        input_dir="./Garbage classification",
        output_dir="./strong_garbage_datasets",
        rotate_angles=(45, 90, 135, 180, 225, 270, 315),
        brightness_factors=(0.8, 1.2),
        contrast_factors=(0.8, 1.2),
        enable_flip=True,
        enable_brightness=True,
        enable_contrast=True
    )
    
    # 开始增强
    augmenter.process_all_categories()
