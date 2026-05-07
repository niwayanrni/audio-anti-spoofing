import sys
import os

# ======================
# FIX PATH (BIAR IMPORT AMAN)
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from aasist import AASIST_Style
from data_loader import AudioDataset
import numpy as np

# ======================
# LOAD DATA
# ======================
train_dataset = AudioDataset(os.path.join(BASE_DIR, "processed_data/train"))
val_dataset = AudioDataset(os.path.join(BASE_DIR, "processed_data/val"))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))

# ======================
# DEVICE
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ======================
# MODEL
# ======================
model = AASIST_Style().to(device)

# ======================
# LOSS & OPTIMIZER
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ======================
# LOGGING LOSS
# ======================
train_losses = []
val_losses = []

# ======================
# TRAINING
# ======================
epochs = 100

for epoch in range(epochs):

    # ===== TRAIN =====
    model.train()
    train_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # ===== VALIDATION =====
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# ======================
# SAVE MODEL (AMAN)
# ======================
os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)

model_path = os.path.join(BASE_DIR, "model", "best_model_loss_demo.pth")
torch.save(model.state_dict(), model_path)

# ======================
# SAVE LOSS
# ======================
os.makedirs(os.path.join(BASE_DIR, "visualisasi"), exist_ok=True)

np.save(os.path.join(BASE_DIR, "visualisasi", "train_loss.npy"), train_losses)
np.save(os.path.join(BASE_DIR, "visualisasi", "val_loss.npy"), val_losses)

print("🔥 Training selesai!")
print("Model disimpan di:", model_path)
print("Loss tersimpan di folder visualisasi/")