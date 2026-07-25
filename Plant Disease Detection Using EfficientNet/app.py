"""
Plant Disease Detection — Flask API Server
Run: python app.py
Endpoint: POST /predict  (multipart/form-data, field name = "file")
"""

import io
import json
import os

import torch
import torch.nn as nn
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH  = "plant_disease_efficientnet.pth"
CLASS_JSON  = "class_names.json"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE    = 224

# ─────────────────────────────────────────────
# DISEASE INFO DATABASE
# Maps class names → human-friendly info
# (extend as needed for all PlantVillage classes)
# ─────────────────────────────────────────────
DISEASE_INFO = {
    "healthy": {
        "status": "healthy",
        "description": "The plant appears healthy with no visible disease symptoms.",
        "treatment": "Continue regular care: proper watering, fertilisation, and sunlight.",
        "severity": "none",
    },
    "default_disease": {
        "status": "diseased",
        "description": "A disease has been detected on this plant leaf.",
        "treatment": "Isolate the plant, remove affected leaves, and consult a local agronomist for targeted treatment.",
        "severity": "moderate",
    },
    # Add specific entries per PlantVillage class name as needed, e.g.:
    # "Tomato___Late_blight": {
    #     "status": "diseased",
    #     "description": "Late blight caused by Phytophthora infestans.",
    #     "treatment": "Apply copper-based fungicide; remove infected foliage.",
    #     "severity": "high",
    # },
}

# Prettify raw class names like "Apple___Apple_scab" → "Apple — Apple Scab"
def prettify(name: str) -> str:
    parts = name.split("___")
    if len(parts) == 2:
        plant, condition = parts
        return f"{plant.replace('_', ' ')} — {condition.replace('_', ' ').title()}"
    return name.replace("_", " ").title()


def get_disease_info(class_name: str) -> dict:
    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]
    if "healthy" in class_name.lower():
        return DISEASE_INFO["healthy"]
    return DISEASE_INFO["default_disease"]


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
def load_model(path: str, num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()
    return model


# Load class names
if not os.path.exists(CLASS_JSON):
    raise FileNotFoundError(f"'{CLASS_JSON}' not found. Train the model first.")

with open(CLASS_JSON) as f:
    class_names = json.load(f)

num_classes = len(class_names)

# Load trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"'{MODEL_PATH}' not found. Train the model first.")

model = load_model(MODEL_PATH, num_classes)
print(f"Model loaded — {num_classes} classes — device: {DEVICE}")

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


def predict(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top5_probs, top5_idx = torch.topk(probs, min(5, num_classes))
    top_class = class_names[top5_idx[0].item()]
    confidence = top5_probs[0].item()

    top5 = [
        {"class": prettify(class_names[i.item()]), "probability": round(p.item(), 4)}
        for i, p in zip(top5_idx, top5_probs)
    ]

    info = get_disease_info(top_class)

    return {
        "predicted_class" : prettify(top_class),
        "raw_class"       : top_class,
        "confidence"      : round(confidence * 100, 2),
        "top5"            : top5,
        "disease_info"    : info,
    }


# ─────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        result = predict(file.read())
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/classes", methods=["GET"])
def get_classes():
    return jsonify({"classes": [prettify(c) for c in class_names], "count": num_classes})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE), "num_classes": num_classes})


if __name__ == "__main__":
    print("Starting Flask server at http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
