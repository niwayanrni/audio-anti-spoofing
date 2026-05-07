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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCENARIOS = {
    "Skenario 1 (AASIST + Clean)": {
        "model": "model/best_model.pth",
        "data": "processed_data/test"
    },
    "Skenario 2 (AASIST + FKD + Clean)": {
        "model": "model/best_model_fkd.pth",
        "data": "processed_data/test"
    },
    "Skenario 3 (AASIST + Noise)": {
        "model": "model/best_model_noise.pth",
        "data": "processed_data/test_noisy"
    },
    "Skenario 4 (AASIST + FKD + Noise)": {
        "model": "model/best_model_fkd_noise.pth",
        "data": "processed_data/test_noisy"
    }
}

results = []


for name, cfg in SCENARIOS.items():

    print("\n==============================")
    print(name)
    print("==============================")

    model_path = os.path.join(BASE_DIR, cfg["model"])
    data_path = os.path.join(BASE_DIR, cfg["data"])

    model = AASIST_Style().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = AudioDataset(data_path)
    loader = DataLoader(dataset, batch_size=8)

    scores, labels = [], []

    start = time.time()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)

            probs = torch.softmax(out, dim=1)
            scores.extend(probs[:, 1].cpu().numpy())
            labels.extend(y.numpy())

    end = time.time()

    scores = np.array(scores)
    labels = np.array(labels)


    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = np.mean([fpr[idx], fnr[idx]])


    Pspoof = 0.05
    Ptar = (1 - Pspoof) * 0.99
    Cmiss = 1
    Cfa = 10

    tdcf = Cmiss * Ptar * fnr + Cfa * Pspoof * fpr
    min_tdcf = np.min(tdcf)

    # simpan hasil
    results.append((name, eer*100, min_tdcf))


    print(f"EER            : {eer*100:.6f}%")
    print(f"min t-DCF      : {min_tdcf:.6f}")
    print(f"Total Data     : {len(dataset)}")
    print(f"Total waktu    : {end-start:.4f} detik")
    print(f"Rata-rata/Data : {(end-start)/len(dataset):.8f} detik")


os.makedirs("visualisasi", exist_ok=True)
np.save("visualisasi/scores.npy", scores)
np.save("visualisasi/labels.npy", labels)


print("\n==============================")
print("RINGKASAN HASIL")
print("==============================")

for name, eer, tdcf in results:
    print(f"{name}")
    print(f"  EER      : {eer:.6f}%")
    print(f"  min-tDCF : {tdcf:.6f}")
    print("")