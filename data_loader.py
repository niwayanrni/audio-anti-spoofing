import os
import numpy as np
import torch
from torch.utils.data import Dataset
from noise_utils import add_noise

class AudioDataset(Dataset):

    def __init__(
        self,
        root_dir,
        use_noise=False
    ):

        self.data = []
        self.labels = []

        self.use_noise = use_noise

        classes = {
            "bonafide": 0,
            "spoof": 1
        }

        for label in classes:

            label_path = os.path.join(
                root_dir,
                label
            )

            for file in os.listdir(label_path):

                if file.endswith(".npy"):

                    self.data.append(
                        os.path.join(label_path, file)
                    )

                    self.labels.append(
                        classes[label]
                    )

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        x = np.load(self.data[idx])

        MAX_LEN = 256

        # ===== PADDING / CROPPING =====
        if x.shape[1] < MAX_LEN:

            pad_width = MAX_LEN - x.shape[1]

            x = np.pad(
                x,
                ((0, 0), (0, pad_width)),
                mode='constant'
            )

        else:

            x = x[:, :MAX_LEN]

        # ===== NOISE =====
        if self.use_noise:

            x = add_noise(x)

        # ===== CHANNEL DIM =====
        x = np.expand_dims(
            x,
            axis=0
        )

        x = torch.tensor(
            x,
            dtype=torch.float32
        )

        y = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return x, y