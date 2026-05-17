import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ===== AUDIO PATH =====
bonafide_path = "dataset_final/train/bonafide/male1/0101BFD00002.wav"

spoof_path = "dataset_final/train/spoof/male1/0102A0100039.wav"

# ===== LOAD AUDIO =====
y_bona, sr = librosa.load(
    bonafide_path,
    sr=16000
)

y_spoof, sr = librosa.load(
    spoof_path,
    sr=16000
)

# ===== STFT =====
stft_bona = np.abs(
    librosa.stft(
        y_bona,
        n_fft=512,
        hop_length=256
    )
)

stft_spoof = np.abs(
    librosa.stft(
        y_spoof,
        n_fft=512,
        hop_length=256
    )
)

# ===== LOG SCALE =====
log_bona = np.log1p(stft_bona)

log_spoof = np.log1p(stft_spoof)

# ===== VISUALIZATION =====
fig, ax = plt.subplots(
    1,
    2,
    figsize=(10,4)
)

# ===== BONAFIDE =====
librosa.display.specshow(
    log_bona,
    sr=sr,
    hop_length=256,
    x_axis='time',
    y_axis='hz',
    ax=ax[0]
)

ax[0].set_title(
    "Bonafide Spectrogram"
)

# ===== SPOOF =====
librosa.display.specshow(
    log_spoof,
    sr=sr,
    hop_length=256,
    x_axis='time',
    y_axis='hz',
    ax=ax[1]
)

ax[1].set_title(
    "Spoof Spectrogram"
)

plt.tight_layout()

# ===== SAVE RESULT =====
os.makedirs(
    "visualisasi/hasil",
    exist_ok=True
)

plt.savefig(
    "visualisasi/hasil/bonafide_vs_spoof_spectrogram.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print("✅ Visualisasi bonafide vs spoof berhasil disimpan!")