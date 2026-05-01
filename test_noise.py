import os
import numpy as np
from noise_utils import add_noise

src_root = "processed_data/test"
dst_root = "processed_data/test_noisy"

classes = ["bonafide", "spoof"]

for cls in classes:
    src_folder = os.path.join(src_root, cls)
    dst_folder = os.path.join(dst_root, cls)

    os.makedirs(dst_folder, exist_ok=True)

    for file in os.listdir(src_folder):
        if file.endswith(".npy"):
            path = os.path.join(src_folder, file)

            x = np.load(path)

            x_noisy = add_noise(x, noise_level=0.02)

            save_path = os.path.join(dst_folder, file)
            np.save(save_path, x_noisy)

print("🔥 test_noisy berhasil dibuat!")