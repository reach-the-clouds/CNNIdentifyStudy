import os
import json
import socket
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from cnn_fruits import CNN_Fruits_Model

app = Flask(__name__)
CORS(app)

current_dir = os.path.dirname(os.path.abspath(__file__))
pth = os.path.join(current_dir, "fruits_model.pth")
ptl = os.path.join(current_dir, "fruits_model.ptl")
cls = os.path.join(current_dir, "fruits_classes.json")
device = torch.device("cpu")
HOST = os.environ.get("FRUITS_HOST", "127.0.0.1")


def choose_port():
    env = os.environ.get("FRUITS_PORT")
    candidates = []
    if env:
        try:
            candidates.append(int(env))
        except Exception:
            pass
    candidates += [8001, 8002, 8003, 9001, 8081, 5001]
    for p in candidates:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, p))
            s.close()
            return p
        except Exception:
            continue
    return 8001


names = []
idx2name = {}
if os.path.exists(cls):
    try:
        with open(cls, 'r', encoding='utf-8') as f:
            data = json.load(f)
            names = data.get("names", [])
            idx2name = {i: n for i, n in enumerate(names)}
    except Exception:
        names = []
        idx2name = {}

num_classes = len(names) if len(names) > 0 else 34

model = None
if os.path.exists(pth):
    model = CNN_Fruits_Model(num_classes=num_classes)
    state = torch.load(pth, map_location=device)
    model.load_state_dict(state)
    model.eval()
elif os.path.exists(ptl):
    model = torch.jit.load(ptl, map_location=device)
    model.eval()

transform = Compose([
    Resize((400, 400)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.route("/labels", methods=["GET"])
def labels():
    return jsonify({"names": names})


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
    return jsonify({"label_index": pred, "label_name": idx2name.get(pred)})


if __name__ == "__main__":
    app.run(host=HOST, port=choose_port())
