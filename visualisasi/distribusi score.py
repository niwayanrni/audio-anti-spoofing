import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

scores = np.load(os.path.join(BASE_DIR, "visualisasi", "scores.npy"))
labels = np.load(os.path.join(BASE_DIR, "visualisasi", "labels.npy"))

# ======================
# PLOT TERPISAH (BIAR KELIHATAN)
# ======================
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.hist(scores[labels==0], bins=30)
plt.title("Bonafide")
plt.xlabel("Score")
plt.ylabel("Jumlah")

plt.subplot(1,2,2)
plt.hist(scores[labels==1], bins=30)
plt.title("Spoof")
plt.xlabel("Score")
plt.ylabel("Jumlah")

plt.tight_layout()
plt.show()