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
    torch.load("model/teacher_freqmix.pth")
)

teacher.eval()


student = AASIST_Style().to(device)


ce_loss = nn.CrossEntropyLoss()

kl_loss = nn.KLDivLoss(
    reduction="batchmean"
)

optimizer = torch.optim.Adam(
    student.parameters(),
    lr=0.001
)

temperature = 4.0
alpha = 0.7


best_val_loss = float("inf")
best_epoch = 0


train_losses = []
val_losses = []


epochs = 100


for epoch in range(epochs):

    student.train()

    train_loss = 0

    for (x1, y1), (x2, y2) in zip(
        train_loader,
        train_loader2
    ):

        x1 = x1.to(device)
        x2 = x2.to(device)
        y1 = y1.to(device)

        optimizer.zero_grad()

        x_mix = frequency_mix(
            x1,
            x2
        )

        with torch.no_grad():

            teacher_out = teacher(
                x_mix
            )

        student_out = student(
            x1
        )

        loss_ce = ce_loss(
            student_out,
            y1
        )

        soft_teacher = torch.softmax(
            teacher_out / temperature,
            dim=1
        )

        soft_student = torch.log_softmax(
            student_out / temperature,
            dim=1
        )

        loss_kd = kl_loss(
            soft_student,
            soft_teacher
        )

        loss = (
            alpha * loss_kd
            +
            (1-alpha) * loss_ce
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()


    student.eval()

    val_loss = 0

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(device)
            y = y.to(device)

            outputs = student(x)

            loss = ce_loss(
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
        best_epoch = epoch + 1

        torch.save(
            student.state_dict(),
            "model/best_model_fkd_freqmix_noes.pth"
        )


os.makedirs(
    "evaluasi",
    exist_ok=True
)

np.save(
    "evaluasi/train_loss_fkd_freqmix_noes.npy",
    np.array(train_losses)
)

np.save(
    "evaluasi/val_loss_fkd_freqmix_noes.npy",
    np.array(val_losses)
)


print("\n====================================")
print(f"Best model epoch : {best_epoch}")
print(f"Best val loss    : {best_val_loss:.4f}")
print("Model saved      : model/best_model_fkd_freqmix_noes.pth")
print("Loss history saved in /evaluasi")
print("====================================")

print("🔥 Training FKD selesai!")