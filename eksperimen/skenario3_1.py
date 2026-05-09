import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from aasist import AASIST_Style
from data_loader import AudioDataset
import numpy as np


train_dataset = AudioDataset(
    "processed_data/train",
    use_noise=True
)

val_dataset = AudioDataset(
    "processed_data/val",
    use_noise=False
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset)) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = AASIST_Style().to(device)

# mulai dari baseline
model.load_state_dict(torch.load("model/best_model.pth"))

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005
)

# ===== SIMPAN LOSS =====
train_losses = []
val_losses = []

# ===== SIMPAN SCORE & LABEL =====
all_scores = []
all_labels = []

best_val_loss = float("inf")

epochs = 100

for epoch in range(epochs):

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

    # ===== SIMPAN TRAIN LOSS =====
    train_losses.append(train_loss)

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for x, y in val_loader:

            x, y = x.to(device), y.to(device)

            outputs = model(x)

            loss = criterion(outputs, y)

            val_loss += loss.item()

            # ===== SCORE =====
            probs = torch.softmax(outputs, dim=1)

            all_scores.extend(
                probs[:, 1].cpu().numpy()
            )

            all_labels.extend(
                y.cpu().numpy()
            )

    # ===== SIMPAN VAL LOSS =====
    val_losses.append(val_loss)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    # ===== SAVE BEST MODEL =====
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "model/best_model_noise1.pth"
        )

# ===== BUAT FOLDER =====
os.makedirs("visualisasi", exist_ok=True)

# ===== SAVE LOSS =====
np.save(
    "visualisasi/train_loss_noise1.npy",
    train_losses
)

np.save(
    "visualisasi/val_loss_noise1.npy",
    val_losses
)

# ===== SAVE SCORE & LABEL =====
np.save(
    "visualisasi/scores_noise1.npy",
    np.array(all_scores)
)

np.save(
    "visualisasi/labels_noise1.npy",
    np.array(all_labels)
)

print("🔥 Training skenario 3 selesai!")
print("Loss, scores, dan labels berhasil disimpan!")