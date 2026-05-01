import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from aasist import AASIST_Style
from data_loader import AudioDataset
from sklearn.metrics import roc_curve

test_dataset = AudioDataset("processed_data/test")
test_loader = DataLoader(test_dataset, batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AASIST_Style().to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

scores = []
labels = []

start_time = time.time()

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)

        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)

        spoof_score = probs[:, 1]

        scores.extend(spoof_score.cpu().numpy())
        labels.extend(y.numpy())

end_time = time.time()

scores = np.array(scores)
labels = np.array(labels)

fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
fnr = 1 - tpr

eer_idx = np.nanargmin(np.abs(fnr - fpr))
eer = fpr[eer_idx]

far = fpr[eer_idx]
frr = fnr[eer_idx]

min_tdcf = np.min(fpr + fnr) / 2

total_infer_time = end_time - start_time
avg_infer_time = total_infer_time / len(test_dataset)

print("===== HASIL EVALUASI SKENARIO 1 =====")
print(f"EER             : {eer*100:.2f}%")
print(f"FAR             : {far*100:.2f}%")
print(f"FRR             : {frr*100:.2f}%")
print(f"min t-DCF       : {min_tdcf:.4f}")
print(f"Total Parameter : {total_params:,}")
print(f"Trainable Param : {trainable_params:,}")
print(f"Total Inferensi : {total_infer_time:.4f} detik")
print(f"Rata-rata/Data  : {avg_infer_time:.6f} detik")

model = AASIST_Style().to(device)
model.load_state_dict(torch.load("best_model_fkd.pth"))
model.eval()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

scores = []
labels = []

start_time = time.time()

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)

        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)

        spoof_score = probs[:, 1]

        scores.extend(spoof_score.cpu().numpy())
        labels.extend(y.numpy())

end_time = time.time()

scores = np.array(scores)
labels = np.array(labels)

fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
fnr = 1 - tpr

eer_idx = np.nanargmin(np.abs(fnr - fpr))
eer = fpr[eer_idx]

far = fpr[eer_idx]
frr = fnr[eer_idx]

min_tdcf = np.min(fpr + fnr) / 2

total_infer_time = end_time - start_time
avg_infer_time = total_infer_time / len(test_dataset)

print("\n===== HASIL EVALUASI SKENARIO 2 =====")
print(f"EER             : {eer*100:.2f}%")
print(f"FAR             : {far*100:.2f}%")
print(f"FRR             : {frr*100:.2f}%")
print(f"min t-DCF       : {min_tdcf:.4f}")
print(f"Total Parameter : {total_params:,}")
print(f"Trainable Param : {trainable_params:,}")
print(f"Total Inferensi : {total_infer_time:.4f} detik")
print(f"Rata-rata/Data  : {avg_infer_time:.6f} detik")

model = AASIST_Style().to(device)
model.load_state_dict(torch.load("best_model_fkd_freqmix.pth"))
model.eval()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

scores = []
labels = []

start_time = time.time()

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)

        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)

        spoof_score = probs[:, 1]

        scores.extend(spoof_score.cpu().numpy())
        labels.extend(y.numpy())

end_time = time.time()

scores = np.array(scores)
labels = np.array(labels)

fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
fnr = 1 - tpr

eer_idx = np.nanargmin(np.abs(fnr - fpr))
eer = fpr[eer_idx]

far = fpr[eer_idx]
frr = fnr[eer_idx]

min_tdcf = np.min(fpr + fnr) / 2

total_infer_time = end_time - start_time
avg_infer_time = total_infer_time / len(test_dataset)

print("\n===== HASIL EVALUASI SKENARIO 2 FKD + Frequency Mix =====")
print(f"EER             : {eer*100:.2f}%")
print(f"min t-DCF       : {min_tdcf:.4f}")
print(f"Total Parameter : {total_params:,}")
print(f"Trainable Param : {trainable_params:,}")
print(f"Total Inferensi : {total_infer_time:.4f} detik")
print(f"Rata-rata/Data  : {avg_infer_time:.6f} detik")