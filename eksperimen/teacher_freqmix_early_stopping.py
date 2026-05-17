import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aasist import AASIST_Style
from data_loader import AudioDataset
from fkd_utils import frequency_mix


train_dataset = AudioDataset("processed_data/train")
val_dataset = AudioDataset("processed_data/val")

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

train_loader2 = DataLoader(
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



teacher = AASIST_Style().to(device)

teacher.load_state_dict(
    torch.load("model/best_model.pth")
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    teacher.parameters(),
    lr=0.001
)


epochs = 100
patience = 10

best_val_loss = float("inf")
counter = 0
best_epoch = 0


train_losses = []
val_losses = []


for epoch in range(epochs):

    teacher.train()

    train_loss = 0

    for (x1, y1), (x2, y2) in zip(train_loader, train_loader2):

        x1 = x1.to(device)
        x2 = x2.to(device)
        y1 = y1.to(device)

        optimizer.zero_grad()

        x_mix = frequency_mix(x1, x2)

        outputs = teacher(x_mix)

        loss = criterion(outputs, y1)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()


    teacher.eval()

    val_loss = 0

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(device)
            y = y.to(device)

            outputs = teacher(x)

            loss = criterion(outputs, y)

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
            teacher.state_dict(),
            "model/teacher_freqmix.pth"
        )

    else:

        counter += 1

    if counter >= patience:
        break

os.makedirs("evaluasi", exist_ok=True)

np.save(
    "evaluasi/train_loss_teacher_freqmix.npy",
    np.array(train_losses)
)

np.save(
    "evaluasi/val_loss_teacher_freqmix.npy",
    np.array(val_losses)
)


print("\n====================================")
print(f"Best model epoch : {best_epoch}")
print(f"Best val loss    : {best_val_loss:.4f}")
print("Model saved      : model/teacher_freqmix.pth")
print("Loss history saved in /evaluasi")
print("====================================")

print("🔥 Teacher Frequency Mix Training Selesai!")