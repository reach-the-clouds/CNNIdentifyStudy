# 导入必要的库
import tkinter as tk  # Tkinter GUI库
from tkinter import Canvas, Button, Label, Frame  # Tkinter控件
from PIL import Image, ImageOps, ImageDraw  # 图像处理库
from torchvision.transforms import ToTensor, Resize, Compose  # 图像变换
from CNN import CNN_Model  # 自定义的CNN模型
import torch  # PyTorch深度学习框架

class HandWriteApp:
    """
    手写数字识别应用类
    提供GUI界面，允许用户手写数字并使用CNN模型进行识别
    """
    def __init__(self, root):
        """
        初始化应用程序
        参数:
            root: Tkinter根窗口对象
        """
        self.root = root
        self.root.title("手写数字识别")  # 设置窗口标题
        self.root.geometry("400x500")  # 设置窗口大小

        # 创建画布，用于用户手写数字
        # 画布大小为280x280像素，背景为白色
        self.canvas = Canvas(self.root, width=280, height=280, bg='white')
        self.canvas.pack(pady=10)  # 添加垂直边距

        # 创建用于绘图的PIL图像对象
        # 'L'模式表示灰度图像，初始为白色背景
        self.image = Image.new('L', (280, 280), color='white')
        self.draw = ImageDraw.Draw(self.image)  # 创建绘图对象

        # 初始化鼠标位置变量
        self.old_x = None
        self.old_y = None

        # 绑定鼠标事件
        # '<Button-1>' - 鼠标左键按下事件
        self.canvas.bind('<Button-1>', self.start_draw)
        # '<B1-Motion>' - 鼠标左键拖动事件
        self.canvas.bind('<B1-Motion>', self.draw_line)

        # 创建按钮框架，用于放置控制按钮
        button_frame = Frame(self.root)
        button_frame.pack(pady=10)  # 添加垂直边距

        # 创建清除按钮，用于清除画布内容
        self.clear_button = Button(button_frame, text="清除", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=10)  # 左对齐，添加水平边距

        # 创建识别按钮，用于触发数字识别
        self.predict_button = Button(button_frame, text="识别", command=self.predict_digit)
        self.predict_button.pack(side=tk.LEFT, padx=10)  # 左对齐，添加水平边距

        # 创建结果标签，用于显示识别结果
        self.result_label = Label(self.root, text="请在上方画布上手写数字", font=("Arial", 14))
        self.result_label.pack(pady=10)  # 添加垂直边距

        # 加载预训练的CNN模型
        self.model = CNN_Model()  # 创建模型实例
        try:
            # 尝试加载预训练的模型权重
            self.model.load_state_dict(torch.load("cnn.pt"))
            self.model.eval()  # 设置为评估模式
            print("成功加载CNN模型权重")
        except (FileNotFoundError, RuntimeError) as e:
            # 如果加载失败，显示错误信息
            print(f"加载CNN模型权重失败: {str(e)}")
            self.result_label.config(text="模型加载失败，请先运行10.cnn模型训练.py训练模型")

    def start_draw(self, event):
        """
        开始绘图事件处理函数
        当鼠标左键按下时被调用，记录初始坐标
        参数:
            event: 鼠标事件对象，包含坐标信息
        """
        self.old_x = event.x  # 记录鼠标x坐标
        self.old_y = event.y  # 记录鼠标y坐标

    def draw_line(self, event):
        """
        绘制线条事件处理函数
        当鼠标左键拖动时被调用，在画布和PIL图像上绘制线条
        参数:
            event: 鼠标事件对象，包含当前坐标信息
        """
        if self.old_x and self.old_y:  # 如果有初始坐标
            # 在Tkinter画布上绘制线条
            # width=15: 线条宽度
            # fill='black': 线条颜色为黑色
            # capstyle=tk.ROUND: 线条端点为圆形
            # smooth=tk.TRUE: 平滑线条
            # splinesteps=36: 平滑度参数
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, 
                                   width=15, fill='black', capstyle=tk.ROUND, 
                                   smooth=tk.TRUE, splinesteps=36)

            # 在PIL图像上绘制线条
            # fill=0: 线条颜色为黑色（在灰度图中0表示黑色）
            # width=15: 线条宽度
            self.draw.line([self.old_x, self.old_y, event.x, event.y], 
                           fill=0, width=15)

            # 更新坐标为当前位置，准备下一次绘制
            self.old_x = event.x
            self.old_y = event.y
    def clear_canvas(self):
        """
        清除画布函数
        清除画布上的所有内容，并重置PIL图像
        """
        self.canvas.delete("all")  # 清除画布上的所有元素
        # 创建新的白色背景图像
        self.image = Image.new('L', (280, 280), color='white')
        self.draw = ImageDraw.Draw(self.image)  # 创建新的绘图对象
        # 重置结果标签文本
        self.result_label.config(text="请在上方画布上手写数字")

    def predict_digit(self):
        """
        识别手写数字函数
        对画布上的手写数字进行预处理，并使用CNN模型进行识别
        """
        try:
            # 图像预处理
            image = self.image.resize((28, 28))  # 将图像大小调整为28x28像素
            image = ImageOps.invert(image)  # 反转颜色（黑变白，白变黑）
            # 创建图像变换组合
            transform = Compose([ToTensor()])
            # 应用变换并添加batch维度
            image_tensor = transform(image).unsqueeze(0)

            # 使用模型进行预测
            with torch.no_grad():  # 不计算梯度，节省内存
                outputs = self.model(image_tensor)  # 获取模型输出
                # 使用softmax函数将输出转换为概率分布
                probs = torch.nn.functional.softmax(outputs, dim=1)
                # 获取预测结果（概率最大的类别）
                prediction = torch.argmax(outputs, 1).item()
                # 获取预测置信度（预测类别的概率）
                confidence = probs[0][prediction].item()

            # 显示预测结果
            self.result_label.config(text=f"预测结果: {prediction} (置信度: {confidence:.2f})")
        except Exception as e:
            # 如果预测过程中出现错误，显示错误信息
            print(f"预测出错: {str(e)}")
            self.result_label.config(text="预测出错，请重试")

# 程序入口点
if __name__=='__main__':
    root = tk.Tk()  # 创建Tkinter根窗口
    app = HandWriteApp(root)  # 创建应用程序实例
    root.mainloop()  # 启动事件循环