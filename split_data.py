import os
import random
import shutil

random.seed(42)

base_dir = "dataset_sampled"
output_dir = "dataset_final"

labels = ["bonafide", "spoof"]
speakers = ["male1", "male2", "female"]

train_ratio = 0.7
val_ratio = 0.15

def split_files(files):
    random.shuffle(files)
    n = len(files)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = files[:train_end]
    val = files[train_end:val_end]
    test = files[val_end:]

    return train, val, test

for label in labels:
    for speaker in speakers:
        src = os.path.join(base_dir, label, speaker)

        files = os.listdir(src)

        train, val, test = split_files(files)

        for split_name, split_files_list in zip(
            ["train", "val", "test"], [train, val, test]
        ):
            for f in split_files_list:
                src_path = os.path.join(src, f)
                dst_path = os.path.join(output_dir, split_name, label, speaker)

                os.makedirs(dst_path, exist_ok=True)

                shutil.copy(src_path, os.path.join(dst_path, f))

print("✅ Split selesai!")