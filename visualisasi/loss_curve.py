import numpy as np
import matplotlib.pyplot as plt

train_loss = np.load(
    "visualisasi/saved_metric/train_loss_noise.npy"
)

val_loss = np.load(
    "visualisasi/saved_metric/val_loss_noise.npy"
)

epochs = range(
    1,
    len(train_loss) + 1
)

plt.figure(figsize=(14,6))

plt.subplot(1,2,1)

plt.plot(
    epochs,
    train_loss,
    linewidth=2
)

plt.xlabel("Epoch", fontsize=10)
plt.ylabel("Loss", fontsize=10)

plt.title(
    "Train Loss with Early Stopping",
    fontsize=13,
    fontweight='bold',
    pad=15
)

plt.grid(alpha=0.3)

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

plt.subplot(1,2,2)

plt.plot(
    epochs,
    val_loss,
    linewidth=2
)

plt.xlabel("Epoch", fontsize=10)
plt.ylabel("Loss", fontsize=10)

plt.title(
    "Validation Loss with Early Stopping",
    fontsize=13,
    fontweight='bold',
    pad=15
)

plt.grid(alpha=0.3)

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

plt.suptitle(
    "AASIST baseline",
    fontsize=16,
    fontweight='bold',
    y=1.02
)

plt.tight_layout()

plt.savefig(
    "visualisasi/hasil/loss_vall_skenario3.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()