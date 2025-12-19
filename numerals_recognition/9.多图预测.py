from fnn import FNN_Model
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
model=FNN_Model()
model.load_state_dict(torch.load("fnn.pt"))

model.eval()
with torch.no_grad():
    outputs=model(obj_tensor)
    print(outputs)
    print(f"{torch.argmax(outputs, dim=1)}")