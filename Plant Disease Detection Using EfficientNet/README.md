# 🌿 LeafAI — Plant Disease Detection (EfficientNet-B3)

A deep learning project that detects plant diseases from leaf images using **EfficientNet-B3**, with a polished web UI served via Flask.

---

## 📁 Project Structure

```
plant_disease_detection/
├── model_train.py          # EfficientNet-B3 training script
├── app.py                  # Flask backend API
├── index.html              # Frontend UI
├── requirements.txt        # Python dependencies
├── class_names.json        # Auto-generated after training
└── plant_disease_efficientnet.pth  # Auto-generated after training
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
Download the **PlantVillage** dataset from Kaggle:
👉 https://www.kaggle.com/datasets/emmarex/plantdisease

Extract so the folder looks like:
```
dataset/
└── PlantVillage/
    ├── Apple___Apple_scab/
    ├── Apple___Black_rot/
    ├── Apple___healthy/
    ├── Tomato___Late_blight/
    ... (38 classes total)
```

### 3. Train the Model
```bash
python model_train.py
```
This will:
- Train EfficientNet-B3 for 20 epochs (GPU recommended)
- Save `plant_disease_efficientnet.pth`
- Save `class_names.json`

### 4. Run the Flask Server
```bash
python app.py
```

### 5. Open the UI
Visit: **http://localhost:5000**

---

## 🧠 Model Architecture

| Component    | Detail                         |
|-------------|-------------------------------|
| Base Model  | EfficientNet-B3 (ImageNet pretrained) |
| Input Size  | 224 × 224 RGB                 |
| Classifier  | Dropout(0.4) → Linear(1536 → 38) |
| Optimizer   | AdamW (lr=1e-4, decay=1e-5)   |
| Scheduler   | Cosine Annealing              |
| Loss        | CrossEntropy (label smooth=0.1) |
| Augmentation| Flip, Rotate, ColorJitter, RandomCrop |

---

## 🌱 Supported Plants & Diseases (PlantVillage — 38 classes)

Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato — with multiple disease variants per plant.

---

## 📡 API Endpoints

| Method | Endpoint   | Description                      |
|--------|-----------|----------------------------------|
| POST   | /predict  | Upload image → get diagnosis      |
| GET    | /classes  | List all supported disease classes|
| GET    | /health   | Server health check               |

**Example POST request:**
```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@leaf.jpg"
```

**Response:**
```json
{
  "predicted_class": "Tomato — Late Blight",
  "confidence": 94.32,
  "disease_info": {
    "status": "diseased",
    "description": "...",
    "treatment": "...",
    "severity": "high"
  },
  "top5": [...]
}
```

---

## ⚡ Tips
- Use a GPU for training (reduces training time from hours to minutes)
- If you get <85% accuracy, increase EPOCHS or reduce LR
- Add more disease entries in `DISEASE_INFO` dict in `app.py` for richer output
