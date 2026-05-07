import os
import matplotlib.pyplot as plt

data_dir = "processed_data/train"

labels = {"bonafide": 0, "spoof": 1}

for label in labels.keys():
    path = os.path.join(data_dir, label)
    labels[label] = len(os.listdir(path))

plt.bar(labels.keys(), labels.values())
plt.title("Distribusi Data")
plt.xlabel("Kelas")
plt.ylabel("Jumlah")
plt.show()