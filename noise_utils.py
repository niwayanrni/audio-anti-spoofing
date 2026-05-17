import os
import random
import librosa
import numpy as np


noise_folder = "free_sound_noise"


def add_noise(audio, sr=16000, snr_db=10):

    noise_files = [

        f for f in os.listdir(
            noise_folder
        )

        if f.endswith(".wav")

    ]

    noise_file = random.choice(
        noise_files
    )

    noise_path = os.path.join(
        noise_folder,
        noise_file
    )


    noise_wave, _ = librosa.load(
        noise_path,
        sr=sr
    )


    noise_spec = np.abs(
        librosa.stft(
            noise_wave,
            n_fft=512,
            hop_length=128
        )
    )


    target_h, target_w = audio.shape


    if noise_spec.shape[1] < target_w:

        repeat = int(
            np.ceil(
                target_w /
                noise_spec.shape[1]
            )
        )

        noise_spec = np.tile(
            noise_spec,
            (1, repeat)
        )


    noise_spec = noise_spec[
        :target_h,
        :target_w
    ]


    if noise_spec.shape[0] < target_h:

        pad_h = (
            target_h -
            noise_spec.shape[0]
        )

        noise_spec = np.pad(
            noise_spec,
            ((0,pad_h),(0,0)),
            mode='constant'
        )


    signal_power = np.mean(
        audio**2
    )

    noise_power = np.mean(
        noise_spec**2
    )


    scale = np.sqrt(
        signal_power /
        (
            noise_power *
            (10**(snr_db/10))
        )
    )


    noisy_audio = (
        audio +
        scale*noise_spec
    )


    noisy_audio = (
        noisy_audio /
        (
            np.max(
                np.abs(noisy_audio)
            ) + 1e-8
        )
    )


    return noisy_audio.astype(
        np.float32
    )