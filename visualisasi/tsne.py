import sys
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


embedding_path = os.path.join(BASE_DIR, "embeddings", "embeddings.npy")
label_path = os.path.join(BASE_DIR, "embeddings", "labels.npy")

X = np.load(embedding_path)
y = np.load(label_path)

print("Shape:", X.shape)


tsne = TSNE(n_components=2, random_state=42)
X_2d = tsne.fit_transform(X)


plt.figure(figsize=(8, 6))

scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='coolwarm')

plt.title("t-SNE Embedding (Bonafide vs Spoof)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")

# legend manual
legend1 = plt.legend(handles=scatter.legend_elements()[0],
                     labels=["Bonafide", "Spoof"])
plt.gca().add_artist(legend1)

plt.grid()

# ======================
# FIX SAVE PATH
# ======================
output_dir = os.path.join(BASE_DIR, "visualisasi", "hasil")
os.makedirs(output_dir, exist_ok=True)

save_path = os.path.join(output_dir, "tsne.png")
plt.savefig(save_path)

print(f"Gambar disimpan di: {save_path}")

plt.show()