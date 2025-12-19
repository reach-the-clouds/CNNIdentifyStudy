import os
import socket
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from cnn1 import CNN_Garbage_Model

app = Flask(__name__)
CORS(app)

current_dir = os.path.dirname(os.path.abspath(__file__))
pth = os.path.join(current_dir, "garbage_model.pth")
ptl = os.path.join(current_dir, "garbage_model.ptl")
device = torch.device("cpu")
HOST = os.environ.get("GARBAGE_HOST", "127.0.0.1")

def choose_port():
    env = os.environ.get("GARBAGE_PORT")
    candidates = []
    if env:
        try:
            candidates.append(int(env))
        except Exception:
            pass
    candidates += [8000, 8001, 8002, 9000, 8080, 5000]
    for p in candidates:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, p))
            s.close()
            return p
        except Exception:
            continue
    return 8000

model = None
if os.path.exists(pth):
    model = CNN_Garbage_Model()
    state = torch.load(pth, map_location=device)
    model.load_state_dict(state)
    model.eval()
elif os.path.exists(ptl):
    model = torch.jit.load(ptl, map_location=device)
    model.eval()

transform = Compose([
    Resize((400, 400)),
    ToTensor(),
    Normalize(mean=[0.6357556, 0.6043181, 0.57092524], std=[0.21566282, 0.2124977, 0.21848688])
])

label_map = {0: "纸板", 1: "其他垃圾", 2: "玻璃", 3: "金属", 4: "纸张", 5: "塑料"}
names = [label_map[i] for i in sorted(label_map.keys())]

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "model_not_loaded"}), 500
    if "file" not in request.files:
        return jsonify({"error": "file_required"}), 400
    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        pred = int(torch.argmax(out, dim=1).item())
    return jsonify({"label_index": pred, "label_name": label_map.get(pred)})

@app.route("/labels", methods=["GET"])
def labels():
    return jsonify({"names": names})

if __name__ == "__main__":
    app.run(host=HOST, port=choose_port())
