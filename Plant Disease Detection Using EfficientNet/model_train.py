"""
Plant Disease Detection - EfficientNet Training Script
Dataset: PlantVillage (use via torchvision or download from Kaggle)
Model: EfficientNet-B3
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR    = r"C:\Users\Priya\Downloads\files\dataset\PlantVillage"   # Folder with class sub-folders
MODEL_SAVE  = "plant_disease_efficientnet.pth"
CLASS_JSON  = "class_names.json"

IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-4
WEIGHT_DECAY= 1e-5
VAL_SPLIT   = 0.15
NUM_WORKERS = 0
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# DATA AUGMENTATION & TRANSFORMS
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomRotation(degrees=30),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# DATASET LOADING
# ─────────────────────────────────────────────
full_dataset = datasets.ImageFolder(DATA_DIR)
class_names  = full_dataset.classes
num_classes  = len(class_names)
print(f"Classes found: {num_classes}")

# Save class names for inference
with open(CLASS_JSON, "w") as f:
    json.dump(class_names, f, indent=2)

# Train / Val split
val_size   = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

# Apply transforms independently
train_ds.dataset.transform = train_transform
val_ds.dataset.transform   = val_transform

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

# ─────────────────────────────────────────────
# MODEL — EfficientNet-B3 (pretrained)
# ─────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model

model = build_model(num_classes).to(DEVICE)

# ─────────────────────────────────────────────
# TRAINING SETUP
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ─────────────────────────────────────────────
# TRAIN / EVAL LOOP
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Val  ", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
    return total_loss / total, correct / total


best_val_acc = 0.0
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

print("\n" + "="*60)
print("       TRAINING EfficientNet-B3 — Plant Disease")
print("="*60)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss,   val_acc   = eval_epoch (model, val_loader,   criterion,           DEVICE)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    tag = " ← best" if val_acc > best_val_acc else ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "epoch"      : epoch,
            "model_state": model.state_dict(),
            "num_classes": num_classes,
        }, MODEL_SAVE)

    print(f"Epoch {epoch:02d}/{EPOCHS}  "
          f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
          f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}{tag}")

print(f"\nBest val accuracy: {best_val_acc:.4f}")
print(f"Model saved → {MODEL_SAVE}")
print(f"Class names  → {CLASS_JSON}")
