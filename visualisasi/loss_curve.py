import os
import numpy as np
import matplotlib.pyplot as plt

# ======================
# PATH
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_loss = np.load(os.path.join(BASE_DIR, "visualisasi", "train_loss.npy"))
val_loss = np.load(os.path.join(BASE_DIR, "visualisasi", "val_loss.npy"))

epochs = range(1, len(train_loss) + 1)

# ======================
# PLOT
# ======================
plt.figure(figsize=(8,5))

plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Validation Loss")

# garis best epoch
best_epoch = np.argmin(val_loss) + 1
plt.axvline(x=best_epoch, linestyle='--', label=f"Best Epoch: {best_epoch}")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid()

plt.show()