import os
import librosa
import numpy as np

# ===== PATH =====
input_dir = "dataset_final"
output_dir = "processed_data"

labels = ["bonafide", "spoof"]
speakers = ["male1", "male2", "female"]
splits = ["train", "val", "test"]

# ===== PARAMETER STFT =====
n_fft = 512
hop_length = 256

for split in splits:
    for label in labels:
        for speaker in speakers:
            src = os.path.join(input_dir, split, label, speaker)
            dst = os.path.join(output_dir, split, label, speaker)

            os.makedirs(dst, exist_ok=True)

            for file in os.listdir(src):
                if not file.endswith(".wav"):
                    continue

                file_path = os.path.join(src, file)

                try:
                    # ===== LOAD AUDIO =====
                    y, sr = librosa.load(file_path, sr=16000)

                    # ===== STFT =====
                    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

                    # ===== MAGNITUDE =====
                    spec = np.abs(stft)

                    # ===== LOG SCALING =====
                    log_spec = np.log1p(spec)

                    # ===== NORMALIZATION =====
                    mean = np.mean(log_spec)
                    std = np.std(log_spec)
                    norm_spec = (log_spec - mean) / (std + 1e-6)

                    # ===== SAVE =====
                    save_path = os.path.join(dst, file.replace(".wav", ".npy"))
                    np.save(save_path, norm_spec)

                except Exception as e:
                    print(f"❌ Error di file: {file_path}")
                    print(e)

print("✅ Preprocessing selesai! Data siap untuk training")