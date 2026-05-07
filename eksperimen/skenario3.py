import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from aasist import AASIST_Style
from data_loader import AudioDataset

# =====================================================
# LOAD DATASET NOISY (SUDAH ADA NOISE, TANPA use_noise)
# =====================================================
train_dataset = AudioDataset("processed_data/train_noisy", use_noise=False)
val_dataset   = AudioDataset("processed_data/val", use_noise=False)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8)

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =====================================================
# MODEL
# =====================================================
model = AASIST_Style().to(device)

# mulai dari baseline
model.load_state_dict(torch.load("model/best_model.pth"))

# =====================================================
# LOSS & OPTIMIZER
# =====================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# =====================================================
# EARLY STOPPING
# =====================================================
best_val_loss = float("inf")
patience = 10
counter = 0

epochs = 100

# =====================================================
# TRAINING
# =====================================================
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

    # =================================================
    # VALIDATION
    # =================================================
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # =================================================
    # SAVE BEST MODEL
    # =================================================
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "model/best_model_noise.pth")
    else:
        counter += 1

    # =================================================
    # EARLY STOPPING
    # =================================================
    if counter >= patience:
        print("Early stopping triggered!")
        break

print("🔥 Training skenario 3 selesai!")