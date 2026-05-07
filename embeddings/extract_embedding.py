import sys
import os

# ======================
# FIX IMPORT PATH (WAJIB)
# ======================
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

# ======================
# CONFIG
# ======================
data_dir = r"processed_data/train"
output_dir = r"embeddings"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# LOAD MODEL
# ======================
from aasist import AASIST_Style

model = AASIST_Style()
model.load_state_dict(torch.load("model/best_model.pth", map_location=device))
model.to(device)
model.eval()

# ======================
# STORAGE
# ======================
embeddings = []
labels = []

label_map = {
    "bonafide": 0,
    "spoof": 1
}

# ======================
# LOOP DATA
# ======================
for label_name in os.listdir(data_dir):
    label_path = os.path.join(data_dir, label_name)

    if not os.path.isdir(label_path):
        continue

    print(f"Processing {label_name}...")

    for file in os.listdir(label_path)[:100]:  # batasi dulu biar ringan
        file_path = os.path.join(label_path, file)

        # load feature (.npy)
        spec = np.load(file_path)

        # ubah ke tensor (PENTING: tambah channel)
        x = torch.tensor(spec, dtype=torch.float32)\
                .unsqueeze(0)\
                .unsqueeze(0)\
                .to(device)

        with torch.no_grad():
            # 🔥 ambil embedding (bukan classifier output)
            feat = model.extract_embedding(x)

        embeddings.append(feat.cpu().numpy())
        labels.append(label_map[label_name])

print("Selesai ambil embedding!")

# ======================
# SAVE
# ======================
os.makedirs(output_dir, exist_ok=True)

embeddings = np.concatenate(embeddings, axis=0)
labels = np.array(labels)

np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
np.save(os.path.join(output_dir, "labels.npy"), labels)

print("Saved ke folder embeddings/")
print("Shape embeddings:", embeddings.shape)
print("Shape labels:", labels.shape)