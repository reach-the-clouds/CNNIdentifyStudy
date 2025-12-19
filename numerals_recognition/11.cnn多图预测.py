from CNN import CNN_Model
import torch
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor, Resize, Compose
import os
files=os.scandir("trueImgs")
list_tensor=[]
to_1=Compose([ToTensor(),Resize((28,28))])
for file in files:
    image =Image.open(file)
    image_l=image.convert("L")
    image_bg=ImageOps.invert(image_l)
    image_tensor=to_1(image_bg)
    list_tensor.append(image_tensor)
obj_tensor=torch.stack(list_tensor)
model=CNN_Model()
try:
    model.load_state_dict(torch.load("cnn.pt"))
    print("成功加载CNN模型权重")
except (FileNotFoundError, RuntimeError) as e:
    print(f"加载CNN模型权重失败: {str(e)}")
    print("请先运行10.cnn模型训练.py训练模型")
    print("如果模型结构已更改，请删除旧的cnn.pt文件并重新训练")
    exit(1)

model.eval()
with torch.no_grad():
    outputs=model(obj_tensor)
    # 使用softmax将输出转换为概率
    probs = torch.nn.functional.softmax(outputs, dim=1)
    # 获取预测结果
    predictions = torch.argmax(outputs, dim=1)

    # 打印每个图片的预测结果和概率
    print("图片预测结果：")
    for i, (pred, prob) in enumerate(zip(predictions, probs)):
        print(f"图片{i+1}: 预测数字={pred.item()}, 置信度={prob[pred].item():.4f}")

    print(f"所有图片的预测结果: {predictions.tolist()}")