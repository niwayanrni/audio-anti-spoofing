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

MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model_noise.pth")
DATA_PATH = os.path.join(BASE_DIR, "processed_data", "test_noisy")

test_dataset = AudioDataset(DATA_PATH)
test_loader = DataLoader(test_dataset, batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AASIST_Style().to(device)
model.load_state_dict(torch.load(MODEL_PATH))
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
        scores.extend(probs[:,1].cpu().numpy())
        labels.extend(y.numpy())

end = time.time()

scores = np.array(scores)
labels = np.array(labels)

fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
fnr = 1 - tpr
eer = fpr[np.nanargmin(np.abs(fnr-fpr))]
min_tdcf = np.min(fpr+fnr)/2

print("===== HASIL SKENARIO 3 =====")
print(f"EER            : {eer*100:.6f}%")
print(f"min t-DCF      : {min_tdcf:.6f}")
print(f"Total Parameter: {total_params:,}")
print(f"Trainable Param: {trainable_params:,}")
print(f"Total Inferensi: {end-start:.6f} detik")
print(f"Rata-rata/Data : {(end-start)/len(test_dataset):.8f} detik")