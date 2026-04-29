import os
import numpy as np

data_dir = "processed_data/train"

lengths = []

for label in ["bonafide", "spoof"]:
    folder = os.path.join(data_dir, label)

    for file in os.listdir(folder):
        if file.endswith(".npy"):
            path = os.path.join(folder, file)
            spec = np.load(path)

            lengths.append(spec.shape[1])  # ambil time frame

# hasil statistik
print("Total data:", len(lengths))
print("Min length:", min(lengths))
print("Max length:", max(lengths))
print("Average length:", sum(lengths)/len(lengths))