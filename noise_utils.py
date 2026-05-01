import os
import random
import librosa
import numpy as np

noise_folder = "free_sound_noise"


def add_noise(audio, sr=16000, noise_level=0.02):
    """
    Menambahkan noise ke fitur spectrogram (.npy)

    Input:
    audio = numpy array shape (257,198)
    """

    # Ambil file noise wav
    noise_files = [f for f in os.listdir(noise_folder) if f.endswith(".wav")]

    # Pilih random noise
    noise_file = random.choice(noise_files)
    noise_path = os.path.join(noise_folder, noise_file)

    # Load noise waveform
    noise_wave, _ = librosa.load(noise_path, sr=sr)

    # Ubah noise ke spectrogram
    noise_spec = np.abs(librosa.stft(noise_wave, n_fft=512, hop_length=128))

    # Resize ke ukuran data utama
    target_h, target_w = audio.shape

    # Jika kurang lebar → ulangi
    if noise_spec.shape[1] < target_w:
        repeat = int(np.ceil(target_w / noise_spec.shape[1]))
        noise_spec = np.tile(noise_spec, (1, repeat))

    # Potong sesuai ukuran target
    noise_spec = noise_spec[:target_h, :target_w]

    # Jika tinggi kurang
    if noise_spec.shape[0] < target_h:
        pad_h = target_h - noise_spec.shape[0]
        noise_spec = np.pad(noise_spec, ((0, pad_h), (0, 0)), mode='constant')

    # Tambahkan noise ke spectrogram utama
    mixed_audio = audio + noise_level * noise_spec

    # Normalisasi
    mixed_audio = mixed_audio / (np.max(np.abs(mixed_audio)) + 1e-8)

    return mixed_audio.astype(np.float32)