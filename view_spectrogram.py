import os
import numpy as np
import matplotlib.pyplot as plt

folder = "processed_data/train/bonafide/male1"

# ambil file pertama
files = os.listdir(folder)
file_path = os.path.join(folder, files[0])

print("File yang dibuka:", file_path)

spec = np.load(file_path)

plt.imshow(spec, aspect='auto', origin='lower')
plt.colorbar()
plt.title("Spectrogram")
plt.show()