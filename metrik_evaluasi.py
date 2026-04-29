import torch
import numpy as np
from torch.utils.data import DataLoader
from aasist import AASIST_Style
from data_loader import AudioDataset
from sklearn.metrics import roc_curve

# ===== LOAD DATA =====
test_dataset = AudioDataset("processed_data/test")
test_loader = DataLoader(test_dataset, batch_size=8)

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== LOAD MODEL =====
model = AASIST_Style().to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

scores = []
labels = []

# ===== INFERENCE =====
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)

        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)

        spoof_score = probs[:, 1]

        scores.extend(spoof_score.cpu().numpy())
        labels.extend(y.numpy())

scores = np.array(scores)
labels = np.array(labels)

# ======================
# 🔥 EER
# ======================
fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
fnr = 1 - tpr

eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

# ======================
# 🔥 min t-DCF (simplified placeholder)
# ======================
# (nanti kita upgrade versi asli)
min_tdcf = np.min(fpr + fnr) / 2

# ======================
# PRINT HASIL
# ======================
print("===== HASIL EVALUASI =====")
print(f"EER       : {eer*100:.2f}%")
print(f"min t-DCF : {min_tdcf:.4f}")