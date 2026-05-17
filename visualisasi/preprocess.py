import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_path = "dataset_final/train/bonafide/male1/0101BFD00002.wav"

# ===== LOAD AUDIO =====
y, sr = librosa.load(
    audio_path,
    sr=16000
)

# ===== STFT =====
stft = librosa.stft(
    y,
    n_fft=512,
    hop_length=256
)

# ===== MAGNITUDE =====
spec = np.abs(stft)

# ===== LOG SCALING =====
log_spec = np.log1p(spec)

# ===== NORMALIZATION =====
mean = np.mean(log_spec)
std = np.std(log_spec)

norm_spec = (
    log_spec - mean
) / (std + 1e-6)

# ===== VISUALIZATION =====
fig, ax = plt.subplots(
    2,
    2,
    figsize=(10,7)
)

# ===== 1. WAVEFORM =====
librosa.display.waveshow(
    y,
    sr=sr,
    ax=ax[0,0]
)

ax[0,0].set_title(
    "Waveform Audio"
)

# ===== 2. MAGNITUDE SPECTROGRAM =====
img1 = librosa.display.specshow(
    spec,
    sr=sr,
    hop_length=256,
    x_axis='time',
    y_axis='hz',
    ax=ax[0,1]
)

ax[0,1].set_title(
    "Magnitude Spectrogram"
)

# ===== 3. LOG-SCALED SPECTROGRAM =====
img2 = librosa.display.specshow(
    log_spec,
    sr=sr,
    hop_length=256,
    x_axis='time',
    y_axis='hz',
    ax=ax[1,0]
)

ax[1,0].set_title(
    "Log-Scaled Spectrogram"
)

# ===== 4. NORMALIZED SPECTROGRAM =====
img3 = librosa.display.specshow(
    norm_spec,
    sr=sr,
    hop_length=256,
    x_axis='time',
    y_axis='hz',
    ax=ax[1,1]
)

ax[1,1].set_title(
    "Normalized Spectrogram"
)

# ===== COLORBAR =====
fig.colorbar(
    img1,
    ax=ax[0,1],
    format='%+2.0f'
)

fig.colorbar(
    img2,
    ax=ax[1,0],
    format='%+2.0f'
)

fig.colorbar(
    img3,
    ax=ax[1,1],
    format='%+2.0f'
)

plt.tight_layout()

# ===== SAVE RESULT =====
os.makedirs(
    "visualisasi/hasil",
    exist_ok=True
)

plt.savefig(
    "visualisasi/hasil/preprocess_result.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print("✅ Visualisasi preprocessing berhasil disimpan!")