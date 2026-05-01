import torch
import random

def frequency_mix(x1, x2):
    batch, c, f, t = x1.shape

    split = random.randint(1, f - 1)

    mixed = torch.cat((x1[:, :, :split, :],
                       x2[:, :, split:, :]), dim=2)

    return mixed