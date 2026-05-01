import numpy as np

x = np.load("processed_data/train/bonafide/0101BFD00002.npy")
print(x.shape)
print(x.dtype)
print(x[:3,:3])