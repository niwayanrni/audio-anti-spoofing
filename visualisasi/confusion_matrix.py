import os
import sys
import torch
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)

from torch.utils.data import DataLoader

from aasist import AASIST_Style
from data_loader import AudioDataset


test_dataset = AudioDataset(
    "processed_data/test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = AASIST_Style().to(device)

model.load_state_dict(
    torch.load(
        "model/best_model_fkd_noise.pth",
        map_location=device
    )
)

model.eval()


all_preds = []
all_labels = []


with torch.no_grad():

    for x, y in test_loader:
        x = x.to(device)
        outputs = model(x)
        preds = torch.argmax(
            outputs,
            dim=1
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            y.numpy()
        )


cm = confusion_matrix(
    all_labels,
    all_preds
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Bonafide",
        "Spoof"
    ]
)


fig, ax = plt.subplots(
    figsize=(4.5,4.5)
)

disp.plot(
    cmap="Blues",
    values_format='d',
    ax=ax
)

cbar = disp.im_.colorbar

cbar.ax.tick_params(
    labelsize=9
)

pos = cbar.ax.get_position()

cbar.ax.set_position([
    pos.x0,
    pos.y0,
    pos.width * 0.85,
    pos.height
])

# kecilkan font luar saja
ax.set_title(
    "Confusion Matrix",
    fontsize=9
)

ax.set_xlabel(
    "Predicted label",
    fontsize=9
)

ax.set_ylabel(
    "True label",
    fontsize=9
)

ax.tick_params(
    labelsize=9
)

os.makedirs(
    "visualisasi/hasil",
    exist_ok=True
)

plt.savefig(
    "visualisasi/hasil/confusion_matrix_clean1.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()