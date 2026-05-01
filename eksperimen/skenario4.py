import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from aasist import AASIST_Style
from data_loader import AudioDataset

# =====================================================
# LOAD DATA
# =====================================================
train_dataset = AudioDataset("processed_data/train", use_noise=True)
val_dataset = AudioDataset("processed_data/val", use_noise=False)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =====================================================
# TEACHER MODEL
# =====================================================
teacher = AASIST_Style().to(device)
teacher.load_state_dict(torch.load("best_model.pth"))
teacher.eval()

# =====================================================
# STUDENT MODEL
# =====================================================
student = AASIST_Style().to(device)

# =====================================================
# LOSS FUNCTION
# =====================================================
ce_loss = nn.CrossEntropyLoss()
kl_loss = nn.KLDivLoss(reduction="batchmean")

optimizer = torch.optim.Adam(student.parameters(), lr=0.001)

# =====================================================
# FKD PARAMETER
# =====================================================
temperature = 4.0
alpha = 0.7

# =====================================================
# EARLY STOPPING
# =====================================================
best_val_loss = float("inf")
patience = 10
counter = 0

# =====================================================
# TRAINING
# =====================================================
epochs = 100

for epoch in range(epochs):

    # ================= TRAIN =================
    student.train()
    train_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        # teacher output
        with torch.no_grad():
            teacher_out = teacher(x)

        # student output
        student_out = student(x)

        # CE loss
        loss_ce = ce_loss(student_out, y)

        # KD loss
        soft_teacher = torch.softmax(teacher_out / temperature, dim=1)
        soft_student = torch.log_softmax(student_out / temperature, dim=1)

        loss_kd = kl_loss(soft_student, soft_teacher)

        # total loss
        loss = alpha * loss_kd + (1 - alpha) * loss_ce

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ================= VALIDATION =================
    student.eval()
    val_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            outputs = student(x)
            loss = ce_loss(outputs, y)

            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # ================= SAVE BEST MODEL =================
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(student.state_dict(), "best_model_fkd_noise.pth")
    else:
        counter += 1

    # ================= EARLY STOPPING =================
    if counter >= patience:
        print("Early stopping triggered!")
        break

print("🔥 Training skenario 4 selesai!")