import os
import numpy as np
import matplotlib.pyplot as plt

folder = r"processed_data/train/bonafide"

files = os.listdir(folder)
file_path = os.path.join(folder, files[0])  # ambil 1 file

spec = np.load(file_path)

plt.imshow(spec, aspect='auto', origin='lower')
plt.colorbar()
plt.title("Spectrogram")
plt.show()