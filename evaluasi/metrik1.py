import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve

from aasist import AASIST_Style
from data_loader import AudioDataset


MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "best_model_fkd_freqmix_fixx.pth"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "processed_data",
    "test"
)


test_dataset = AudioDataset(
    DATA_PATH
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


model = AASIST_Style().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


if total_params >= 1e6:

    complexity = f"{total_params/1e6} M"

elif total_params >= 1e3:

    complexity = f"{total_params/1e3} K"

else:

    complexity = str(total_params)


scores = []
labels = []


start = time.time()

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()

        out = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        probs = torch.softmax(
            out,
            dim=1
        )

        scores.extend(
            probs[:,1].cpu().numpy()
        )

        labels.extend(
            y.numpy()
        )

end = time.time()


avg_inference = (
    (end-start)
    /
    len(test_dataset)
)

avg_ms = avg_inference * 1000


scores = np.array(scores)
labels = np.array(labels)


fpr, tpr, thresholds = roc_curve(
    labels,
    scores,
    pos_label=1
)

fnr = 1 - tpr


eer_idx = np.nanargmin(
    np.abs(
        fnr - fpr
    )
)

eer_new = np.mean(
    [
        fpr[eer_idx],
        fnr[eer_idx]
    ]
)


Pspoof = 0.05
Ptar = (1-Pspoof)*0.99
Pnon = (1-Pspoof)*0.01

Cmiss = 1
Cfa = 10


tdcf_curve = (
    Cmiss*Ptar*fnr
    +
    Cfa*Pspoof*fpr
)

min_tdcf_new = np.min(
    tdcf_curve
)


os.makedirs(
    "visualisasi",
    exist_ok=True
)

np.save(
    "visualisasi/scores_s4.npy",
    scores
)

np.save(
    "visualisasi/labels_s4.npy",
    labels
)


print("\n")
print("HASIL EVALUASI")
print("")

print(
    f"{'EER (%)':<15}"
    f"{'min t-DCF':<15}"
    f"{'Inference(ms)':<20}"
    f"{'Kompleksitas':<20}"
)

print(
    f"{eer_new*100:<15.6f}"
    f"{min_tdcf_new:<15.6f}"
    f"{avg_ms:<20.4f}"
    f"{complexity:<20}"
)

print("")

print("Scores & labels disimpan!")