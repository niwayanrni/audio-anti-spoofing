import numpy as np
import matplotlib.pyplot as plt

train_loss = np.load(
    "visualisasi/saved_metric/train_loss_fkd_noise_noES.npy"
)

val_loss = np.load(
    "visualisasi/saved_metric/val_loss_fkd_noise_noES.npy"
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
    "Train Loss without Early Stopping",
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
    "Validation Loss without Early Stopping",
    fontsize=13,
    fontweight='bold',
    pad=15
)

plt.grid(alpha=0.3)

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

plt.suptitle(
    "Skenario 4",
    fontsize=16,
    fontweight='bold',
    y=1.02
)

plt.tight_layout()

plt.savefig(
    "visualisasi/hasil/loss_vall_skenario4_full.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()