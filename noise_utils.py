import os
import random
import librosa
import numpy as np

noise_folder = "free_sound_noise"

def add_noise(audio, sr=16000, noise_level=0.02):
    """
    Menambahkan noise random ke audio input.

    Parameter:
    audio       : numpy array audio clean
    sr          : sample rate
    noise_level : kekuatan noise (0.01 - 0.05 disarankan)

    Return:
    audio + noise
    """

    # Ambil semua file wav
    noise_files = [f for f in os.listdir(noise_folder) if f.endswith(".wav")]

    # Pilih noise random
    noise_file = random.choice(noise_files)
    noise_path = os.path.join(noise_folder, noise_file)

    # Load noise
    noise, _ = librosa.load(noise_path, sr=sr)

    # Jika noise lebih pendek, ulangi
    if len(noise) < len(audio):
        repeat = int(np.ceil(len(audio) / len(noise)))
        noise = np.tile(noise, repeat)

    # Potong sesuai panjang audio
    noise = noise[:len(audio)]

    # Campur clean + noise
    mixed_audio = audio + noise_level * noise

    # Normalisasi agar tidak clipping
    mixed_audio = mixed_audio / np.max(np.abs(mixed_audio) + 1e-8)

    return mixed_audio