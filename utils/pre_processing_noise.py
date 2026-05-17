import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from noise_utils import add_noise

input_folder = "processed_data/train"
output_folder = "processed_data/train_noisy"

snr_db = 10

classes = [
    "bonafide",
    "spoof"
]

for cls in classes:

    input_path = os.path.join(
        input_folder,
        cls
    )

    output_path = os.path.join(
        output_folder,
        cls
    )

    os.makedirs(
        output_path,
        exist_ok=True
    )

    files = [

        f for f in os.listdir(
            input_path
        )

        if f.endswith(".npy")

    ]


    for file in files:

        file_path = os.path.join(
            input_path,
            file
        )

        data = np.load(
            file_path
        )


        noisy_data = add_noise(
            data,
            snr_db=snr_db
        )


        save_path = os.path.join(
            output_path,
            file
        )

        np.save(
            save_path,
            noisy_data
        )

print(f"🔥 Train noisy selesai dibuat ({snr_db} dB)")