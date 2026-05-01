import numpy as np
import os

path = "processed_data/train/bonafide"
file = os.listdir(path)[0]

x = np.load(os.path.join(path, file))

print("Nama file :", file)
print("Shape     :", x.shape)
print("Min       :", x.min())
print("Max       :", x.max())
print("Mean      :", x.mean())
print("Std       :", x.std())

print(x[:5, :5])