import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aasist import AASIST_Style
from data_loader import AudioDataset


train_dataset = AudioDataset(
    "processed_data/train_noisy",
    use_noise=False
)

val_dataset = AudioDataset(
    "processed_data/val",
    use_noise=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


model = AASIST_Style().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005
)


epochs = 100
patience = 10

best_val_loss = float("inf")
counter = 0
best_epoch = 0

train_losses = []
val_losses = []


for epoch in range(epochs):

    model.train()

    train_loss = 0

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(
            outputs,
            y
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()


    model.eval()

    val_loss = 0

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            loss = criterion(
                outputs,
                y
            )

            val_loss += loss.item()


    train_losses.append(train_loss)
    val_losses.append(val_loss)


    print(
        f"[Epoch {epoch+1:03d}/{epochs}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )


    if val_loss < best_val_loss:

        best_val_loss = val_loss
        counter = 0
        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            "model/best_model_noise.pth"
        )

    else:

        counter += 1


    if counter >= patience:

        print(
            f"\nEarly stopping at epoch {epoch+1}"
        )

        break


os.makedirs(
    "evaluasi",
    exist_ok=True
)

np.save(
    "evaluasi/train_loss_noise.npy",
    np.array(train_losses)
)

np.save(
    "evaluasi/val_loss_noise.npy",
    np.array(val_losses)
)


print("\n====================================")
print(f"Best model epoch : {best_epoch}")
print(f"Best val loss    : {best_val_loss:.4f}")
print("Model saved      : model/best_model_noise.pth")
print("Loss history saved in /evaluasi")
print("====================================")

print("🔥 Training skenario 3 selesai!")