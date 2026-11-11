from pathlib import Path
import time
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# ---------------- CONFIGURATION ----------------

ROOT = Path(__file__).resolve().parent

BATCH_SIZE = 8
NUM_WORKERS = 0
DEVICE = torch.device("cpu")

# Set to 100 for the first test.
# Change to None to process the complete split.
MAX_VIDEOS = None

OUTPUT_DIR = ROOT / "mobilenet_features_full"

CSV_FILES = {
    "train": ROOT / "train.csv",
    "validation": ROOT / "validation.csv",
    "test": ROOT / "test.csv",
}

# ---------------- MODEL ----------------

print("Loading pretrained MobileNetV2...")

weights = MobileNet_V2_Weights.DEFAULT
transform = weights.transforms()

model = mobilenet_v2(weights=weights)
model.classifier = nn.Identity()
model.eval()
model.to(DEVICE)

for parameter in model.parameters():
    parameter.requires_grad = False

print("Model loaded. Feature dimension: 1280")


# ---------------- DATASET ----------------

class FrameDataset(Dataset):

    def __init__(self, metadata):
        self.metadata = metadata.reset_index(drop=True)

    def __len__(self):
        return len(self.metadata) * 16

    def __getitem__(self, index):

        video_index = index // 16
        frame_index = index % 16

        row = self.metadata.iloc[video_index]

        frame_dir = ROOT / Path(row["frame_dir"])
        image_path = frame_dir / f"frame_{frame_index:02d}.jpg"

        image = Image.open(image_path).convert("RGB")
        image = transform(image)

        return image, video_index, frame_index


# ---------------- EXTRACTION ----------------


def extract_split(split_name, csv_path):

    print(f"\nProcessing {split_name}...")

    metadata = pd.read_csv(csv_path)

    if MAX_VIDEOS is not None:
        metadata = metadata.head(MAX_VIDEOS).copy()

    total_videos = len(metadata)

    output_dir = OUTPUT_DIR / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = output_dir / "features.npy"
    status_path = output_dir / "completed.npy"

    if features_path.exists():
        features = np.lib.format.open_memmap(
            features_path, mode="r+"
        )
        if features.shape != (total_videos, 16, 1280):
            raise ValueError("Feature array shape mismatch.")

        if status_path.exists():
            completed = np.load(status_path)
        else:
            completed = np.zeros(total_videos, dtype=bool)
    else:
        features = np.lib.format.open_memmap(
            features_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_videos, 16, 1280)
        )
        completed = np.zeros(total_videos, dtype=bool)

    if len(completed) != total_videos:
        raise ValueError("Checkpoint size mismatch.")

    pending = np.where(~completed)[0]

    print(f"Already completed: {completed.sum()}")
    print(f"Remaining: {len(pending)}")

    class PendingFrameDataset(Dataset):

        def __init__(self, metadata, indices):
            self.metadata = metadata
            self.indices = indices

        def __len__(self):
            return len(self.indices) * 16

        def __getitem__(self, index):
            video_id = int(self.indices[index // 16])
            frame_id = index % 16

            frame_dir = ROOT / Path(
                self.metadata.iloc[video_id]["frame_dir"]
            )

            image_path = frame_dir / f"frame_{frame_id:02d}.jpg"

            image = Image.open(image_path).convert("RGB")
            return transform(image), video_id, frame_id

    dataset = PendingFrameDataset(metadata, pending)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model.eval()
    start = time.time()
    processed = 0

    with torch.inference_mode():

        for images, video_ids, frame_ids in loader:

            output = model(images.to(DEVICE)).cpu().numpy()

            video_ids = video_ids.numpy()
            frame_ids = frame_ids.numpy()

            for i in range(len(output)):
                features[
                    video_ids[i], frame_ids[i]
                ] = output[i]

            # Each pending video's 16 frames are consecutive.
            for video_id in np.unique(video_ids):
                if frame_ids[video_ids == video_id].max() == 15:
                    completed[video_id] = True
                    processed += 1

            if processed > 0 and processed % 100 == 0:
                features.flush()
                np.save(status_path, completed)
                print(
                    f"Progress: {completed.sum()}/{total_videos}"
                )

    features.flush()
    np.save(status_path, completed)

    if not completed.all():
        raise RuntimeError("Some videos remain incomplete.")

    metadata.to_csv(output_dir / "metadata.csv", index=False)

    print(f"{split_name} completed.")
    print("Feature shape:", features.shape)
    print("Time:", round((time.time() - start) / 60, 2), "minutes")


# ---------------- MAIN ----------------

if __name__ == "__main__":

    for split_name, csv_path in CSV_FILES.items():

        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        extract_split(split_name, csv_path)

    print("\nAll feature extraction completed.")
