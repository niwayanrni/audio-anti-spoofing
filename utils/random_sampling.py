import os
import random
import shutil

random.seed(42)

base_dir = "."
output_dir = "dataset_sampled"

labels = ["bonafide", "spoof"]
speakers = ["male1", "male2", "female"]

target_total_per_label = 1000
target_per_speaker = target_total_per_label // len(speakers)

for label in labels:
    for speaker in speakers:
        src_path = os.path.join(base_dir, label, speaker)
        dst_path = os.path.join(output_dir, label, speaker)

        os.makedirs(dst_path, exist_ok=True)

        files = os.listdir(src_path)

        k = min(len(files), target_per_speaker)
        sampled = random.sample(files, k)

        for f in sampled:
            shutil.copy(
                os.path.join(src_path, f),
                os.path.join(dst_path, f)
            )

print("✅ Sampling selesai!")