import os
import matplotlib.pyplot as plt
import numpy as np

base_path = "dataset_final"

splits = ["train", "val", "test"]

bonafide_male = []
bonafide_female = []

spoof_male = []
spoof_female = []

for split in splits:

    # ===== BONAFIDE =====
    bona_male1 = len([
        f for f in os.listdir(
            os.path.join(base_path, split, "bonafide", "male1")
        )
        if f.endswith(".wav")
    ])

    bona_male2 = len([
        f for f in os.listdir(
            os.path.join(base_path, split, "bonafide", "male2")
        )
        if f.endswith(".wav")
    ])

    bona_female = len([
        f for f in os.listdir(
            os.path.join(base_path, split, "bonafide", "female")
        )
        if f.endswith(".wav")
    ])

    bonafide_male.append(
        bona_male1 + bona_male2
    )

    bonafide_female.append(
        bona_female
    )

    # ===== SPOOF =====
    spoof_male1 = len([
        f for f in os.listdir(
            os.path.join(base_path, split, "spoof", "male1")
        )
        if f.endswith(".wav")
    ])

    spoof_male2 = len([
        f for f in os.listdir(
            os.path.join(base_path, split, "spoof", "male2")
        )
        if f.endswith(".wav")
    ])

    spoof_female_count = len([
        f for f in os.listdir(
            os.path.join(base_path, split, "spoof", "female")
        )
        if f.endswith(".wav")
    ])

    spoof_male.append(
        spoof_male1 + spoof_male2
    )

    spoof_female.append(
        spoof_female_count
    )

# ===== PLOT =====
x = np.arange(len(splits))

width = 0.2

plt.figure(figsize=(9,5))

plt.bar(
    x - 1.5*width,
    bonafide_male,
    width,
    label="Bonafide Male"
)

plt.bar(
    x - 0.5*width,
    bonafide_female,
    width,
    label="Bonafide Female"
)

plt.bar(
    x + 0.5*width,
    spoof_male,
    width,
    label="Spoof Male"
)

plt.bar(
    x + 1.5*width,
    spoof_female,
    width,
    label="Spoof Female"
)

# ===== VALUE LABEL =====
for i, v in enumerate(bonafide_male):
    plt.text(
        x[i] - 1.5*width,
        v + 5,
        str(v),
        ha='center',
        fontsize=8
    )

for i, v in enumerate(bonafide_female):
    plt.text(
        x[i] - 0.5*width,
        v + 5,
        str(v),
        ha='center',
        fontsize=8
    )

for i, v in enumerate(spoof_male):
    plt.text(
        x[i] + 0.5*width,
        v + 5,
        str(v),
        ha='center',
        fontsize=8
    )

for i, v in enumerate(spoof_female):
    plt.text(
        x[i] + 1.5*width,
        v + 5,
        str(v),
        ha='center',
        fontsize=8
    )

# ===== LABEL =====
plt.xticks(
    x,
    ["Train", "Validation", "Test"]
)

plt.ylabel("Jumlah Audio")

plt.title(
    "Distribusi Dataset Berdasarkan Gender dan Label"
)

plt.legend()

plt.tight_layout()

# ===== SAVE =====
os.makedirs(
    "visualisasi/hasil",
    exist_ok=True
)

plt.savefig(
    "visualisasi/hasil/distribusi_dataset_gender_label.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print("✅ Visualisasi berhasil disimpan!")