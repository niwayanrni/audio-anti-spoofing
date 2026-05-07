import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

scores = np.load(os.path.join(BASE_DIR, "visualisasi", "scores.npy"))
labels = np.load(os.path.join(BASE_DIR, "visualisasi", "labels.npy"))

fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
fnr = 1 - tpr

idx = np.nanargmin(np.abs(fnr - fpr))


plt.figure(figsize=(6,6))

plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0,1],[0,1],'--')

plt.scatter(fpr[idx], tpr[idx], color='red', label="EER")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()
plt.grid()

output_dir = os.path.join(BASE_DIR, "visualisasi", "hasil")
os.makedirs(output_dir, exist_ok=True)

plt.savefig(os.path.join(output_dir, "roc_curve.png"))

plt.show()