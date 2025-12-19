import tkinter as tk
from tkinter import Label, Button
from torchvision.transforms import ToTensor, Resize, Compose, Normalize, Grayscale
from PIL import Image, ImageOps, ImageDraw, ImageTk
import torch
import os
from cnn1 import CNN_Garbage_Model
from tkinter import filedialog


class GarbageClassificationApp:
    # 构造函数
    def __init__(self, root):
        self.root = root
        self.root.title("垃圾分类")
        # 创建组件（ , Button ， Label）
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=20)
        # 创建打开一个文件夹选择图片的按钮
        self.open_button = Button(root, text="打开", command=self.on_open_image_clicked)
        self.open_button.pack(pady=20)
        # 创建识别按钮
        self.pre_button = Button(root, text="识别", command=self.on_pre_button_clicked)
        self.pre_button.pack(pady=10)
        # 创建一个显示识别结果的Label
        self.pre_label = Label(root, text="", width=20)
        self.pre_label.pack(pady=20)
        self.img_tensor = None  # 模型要使用的张量
        # 加载模型参数
        try:
            # 使用绝对路径确保正确加载模型文件
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 尝试加载两种可能的模型文件格式
            model_path = os.path.join(current_dir, "garbage_model.pth")
            if not os.path.exists(model_path):
                model_path = os.path.join(current_dir, "garbage_model.ptl")

            if not os.path.exists(model_path):
                print(f"错误: 找不到模型文件 {model_path}")
                self.pre_label.config(text="错误: 模型文件不存在")
                self.model = None
            else:
                if model_path.endswith(".pth"):
                    self.model = CNN_Garbage_Model()
                    state = torch.load(model_path, map_location=torch.device('cpu'))
                    self.model.load_state_dict(state)
                    self.model.eval()
                    print(f"成功加载模型: {model_path}")
                elif model_path.endswith(".ptl"):
                    self.model = torch.jit.load(model_path, map_location=torch.device('cpu'))
                    self.model.eval()
                    print(f"成功加载Lite模型: {model_path}")
                else:
                    raise RuntimeError("不支持的模型文件类型")
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            self.pre_label.config(text=f"错误: 模型加载失败")
            self.model = None

    # 预测的按钮
    def on_pre_button_clicked(self):
        try:
            # 检查是否已选择图片
            if not hasattr(self, 'image') or self.image is None:
                self.pre_label.config(text="请先选择一张图片")
                return

            # 检查模型是否已正确加载
            if self.model is None:
                self.pre_label.config(text="模型未加载，请重启程序")
                return

            # 转换图片为张量
            transform = Compose([
                Resize((400, 400)),
                ToTensor(),
                Normalize(mean=[0.6357556, 0.6043181, 0.57092524], std=[0.21566282, 0.2124977, 0.21848688])
            ])

            # 确保图片是RGB格式
            img_rgb = self.image.convert("RGB")
            img_tensor = transform(img_rgb).float()
            self.img_tensor = img_tensor.unsqueeze(0)  # 升维

            # 使用torch.no_grad()提高推理速度并减少内存使用
            with torch.no_grad():
                output = self.model(self.img_tensor)  # 预测
                predicted = torch.argmax(output, dim=1)  # 找到最大值的索引

            # 垃圾标签字典，与训练数据集中的类别对应
            garbage_labels = {
                0: "纸板",
                2: "玻璃",
                3: "金属",
                4: "纸张",
                5: "塑料",
                1: "其他垃圾"
            }

            # 获取预测结果
            garbage_name = garbage_labels[int(predicted.item())]
            print(f"预测结果索引: {predicted.item()}, 垃圾类型: {garbage_name}")
            self.pre_label.config(text=f"识别结果是: {garbage_name}")  # 显示识别结果

        except Exception as e:
            print(f"预测过程中出错: {str(e)}")
            self.pre_label.config(text=f"错误: 识别失败 - {str(e)}")

    # 打开文件夹，选择图片的按钮
    def on_open_image_clicked(self):
        # 打开文件对话框，筛选图片格式
        file_path = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=(("PNG图片", "*.png"), ("JPG图片", "*.jpg"), ("所有文件", "*.*"))
        )
        if file_path:
            # 加载并显示图片
            try:
                # 加载图片并调整大小以适应显示
                self.image = Image.open(file_path)
                print(f"成功加载图片: {os.path.basename(file_path)}")

                # 调整图片大小，保持宽高比
                max_size = (300, 300)
                self.image.thumbnail(max_size, Image.Resampling.LANCZOS)

                # 转换为PhotoImage对象
                img = ImageTk.PhotoImage(self.image)

                # 更新标签显示
                self.image_label.config(image=img)
                self.image_label.image = img  # 保留引用防止被垃圾回收

                # 清空之前的预测结果
                self.pre_label.config(text="")
            except Exception as e:
                error_msg = f"加载图片时出错: {str(e)}"
                print(error_msg)
                self.pre_label.config(text="错误: 无法加载图片")
                self.image = None


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("400x500")  # 略微增大窗口以适应可能更大的图片显示
    app = GarbageClassificationApp(root)
    root.mainloop()  # 程序一直运行。


