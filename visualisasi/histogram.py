import os
import numpy as np
import matplotlib.pyplot as plt

scenario_name = "Skenario 4"

score_path = "visualisasi/scores_s4.npy"
label_path = "visualisasi/labels_s4.npy"

scores = np.load(score_path)
labels = np.load(label_path)

bonafide_scores = scores[labels == 0]
spoof_scores = scores[labels == 1]


plt.figure(figsize=(8, 5))

plt.hist(
    bonafide_scores,
    bins=30,
    alpha=0.7,
    label="Bonafide (0)"
)

plt.hist(
    spoof_scores,
    bins=30,
    alpha=0.7,
    label="Spoof (1)"
)


plt.title(
    f"Histogram Distribusi Score: {scenario_name}"
)

plt.xlabel("Score")
plt.ylabel("Jumlah Data")

plt.legend()

plt.grid(True)


os.makedirs(
    "visualisasi/hasil",
    exist_ok=True
)

save_path = os.path.join(
    "visualisasi/hasil",
    f"histogram_{scenario_name}.png"
)

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Histogram berhasil disimpan!")
print("Lokasi:", save_path)