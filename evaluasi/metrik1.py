import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from aasist import AASIST_Style
from data_loader import AudioDataset
from sklearn.metrics import roc_curve

MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model_fkd_v2.pth")
DATA_PATH = os.path.join(BASE_DIR, "processed_data", "test_noisy")

test_dataset = AudioDataset(DATA_PATH)
test_loader = DataLoader(test_dataset, batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AASIST_Style().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

scores, labels = [], []

start = time.time()

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        out = model(x)

        probs = torch.softmax(out, dim=1)
        scores.extend(probs[:, 1].cpu().numpy()) 
        labels.extend(y.numpy())

end = time.time()

scores = np.array(scores)
labels = np.array(labels)


fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
fnr = 1 - tpr


eer_idx = np.nanargmin(np.abs(fnr - fpr))
eer_new = np.mean([fpr[eer_idx], fnr[eer_idx]])


Pspoof = 0.05
Ptar = (1 - Pspoof) * 0.99
Pnon = (1 - Pspoof) * 0.01

Cmiss = 1
Cfa = 10

tdcf_curve = Cmiss * Ptar * fnr + Cfa * Pspoof * fpr
min_tdcf_new = np.min(tdcf_curve)

import os

os.makedirs("visualisasi", exist_ok=True)

np.save("visualisasi/scores_s4.npy", scores)
np.save("visualisasi/labels_s4.npy", labels)

print("Scores & labels disimpan!")

# ======================
# OUTPUT
# ======================
print("===== HASIL EVALUASI =====")

print("\n--- EER ---")
print(f"EER : {eer_new*100:.6f}%")

print("\n--- min t-DCF ---")
print(f"min t-DCF : {min_tdcf_new:.6f}")

print("\n--- Model Info ---")
print(f"Total Parameter: {total_params:,}")
print(f"Trainable Param: {trainable_params:,}")

print("\n--- Inference ---")
print(f"Total waktu : {end-start:.6f} detik")
print(f"Per data    : {(end-start)/len(test_dataset):.8f} detik")