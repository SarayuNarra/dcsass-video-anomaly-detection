from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent

FEATURE_ROOT = ROOT / "mobilenet_features_full"

# Separate output directory. Experiment 2 is untouched.
MODEL_DIR = ROOT / "model_outputs_experiment4"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Reuse exactly the same normalization as Experiment 2.
EXPERIMENT2_DIR = ROOT / "model_outputs_experiment2"

SEED = 42

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001
PATIENCE = 5

INPUT_DIM = 1280
PROJECTION_DIM = 128
HIDDEN_DIM = 64
NUM_LAYERS = 1
NUM_CLASSES = 2

DEVICE = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Device:", DEVICE)
print("Experiment: CNN-BiLSTM V4 - Moderate Category-Aware Sampling")


# ==================================================
# 2. DATASET
# ==================================================

class VideoFeatureDataset(Dataset):

    def __init__(self, features, labels, mean, std):

        self.features = features

        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

        self.mean = torch.tensor(
            mean,
            dtype=torch.float32
        )

        self.std = torch.tensor(
            std,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        x = torch.tensor(
            self.features[index].copy(),
            dtype=torch.float32
        )

        x = (x - self.mean) / self.std

        return x, self.labels[index]


# ==================================================
# 3. LOAD FEATURES AND METADATA
# ==================================================

print("\nLoading extracted features...")

train_features = np.load(
    FEATURE_ROOT / "train" / "features.npy",
    mmap_mode="r"
)

val_features = np.load(
    FEATURE_ROOT / "validation" / "features.npy",
    mmap_mode="r"
)

train_meta = pd.read_csv(
    FEATURE_ROOT / "train" / "metadata.csv"
)

val_meta = pd.read_csv(
    FEATURE_ROOT / "validation" / "metadata.csv"
)

train_labels = train_meta["label"].to_numpy()
val_labels = val_meta["label"].to_numpy()

assert len(train_features) == len(train_meta)
assert len(val_features) == len(val_meta)

print("Training:", train_features.shape)
print("Validation:", val_features.shape)

print("\nTraining categories:", train_meta["category"].nunique())


# ==================================================
# 4. REUSE EXPERIMENT 2 NORMALIZATION
# ==================================================

print("\nLoading Experiment 2 normalization...")

normalization = np.load(
    EXPERIMENT2_DIR / "normalization.npz"
)

mean = normalization["mean"]
std = normalization["std"]

np.savez(
    MODEL_DIR / "normalization.npz",
    mean=mean,
    std=std
)

print("Normalization reused successfully.")



# ==================================================
# 5. MODERATE CATEGORY-AWARE SAMPLING
# ==================================================

print("\nPreparing moderate category-aware sampler...")

# Count samples in each category-label combination.
stratum_sizes = train_meta.groupby(
    ["category", "label"]
)["label"].transform("size")

# Moderate inverse-frequency weighting.
# Square root reduces the effect of rare combinations.
sample_weights = 1.0 / np.sqrt(
    stratum_sizes.to_numpy()
)

sample_weights = torch.as_tensor(
    sample_weights,
    dtype=torch.double
)

generator = torch.Generator()
generator.manual_seed(SEED)

train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_meta),
    replacement=True,
    generator=generator
)

stratum_counts = train_meta.groupby(
    ["category", "label"]
).size()

print("\nOriginal category-label distribution:")
print(stratum_counts.to_string())

print(
    "\nNumber of category-label combinations:",
    len(stratum_counts)
)

# Expected sampling share for each combination.
expected_counts = (
    len(train_meta)
    * np.sqrt(stratum_counts)
    / np.sqrt(stratum_counts).sum()
)

expected_table = pd.DataFrame({
    "Original_Count": stratum_counts,
    "Expected_Sampled_Count": expected_counts
})

print("\nExpected moderate sampling distribution:")
print(expected_table.round(2).to_string())


# ==================================================
# 6. DATA LOADERS
# ==================================================

train_dataset = VideoFeatureDataset(
    train_features,
    train_labels,
    mean,
    std
)

val_dataset = VideoFeatureDataset(
    val_features,
    val_labels,
    mean,
    std
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==================================================
# 7. CNN-BiLSTM MODEL
# ==================================================

class CNNLSTMClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(INPUT_DIM, PROJECTION_DIM),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.lstm = nn.LSTM(
            input_size=PROJECTION_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):

        x = self.projection(x)

        output, _ = self.lstm(x)

        x = output.mean(dim=1)

        return self.classifier(x)


model = CNNLSTMClassifier().to(DEVICE)

# No class weights here because sampling already
# balances category-label combinations.
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ==================================================
# 8. VALIDATION
# ==================================================

def evaluate(model, loader):

    model.eval()

    actual = []
    predicted = []
    total_loss = 0.0

    with torch.inference_mode():

        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            outputs = model(x)

            loss = criterion(outputs, y)

            total_loss += loss.item() * len(y)

            predictions = outputs.argmax(dim=1)

            actual.extend(y.numpy())
            predicted.extend(predictions.numpy())

    return {

        "loss": total_loss / len(loader.dataset),

        "accuracy": accuracy_score(
            actual, predicted
        ),

        "precision": precision_score(
            actual, predicted, zero_division=0
        ),

        "recall": recall_score(
            actual, predicted, zero_division=0
        ),

        "f1": f1_score(
            actual, predicted, zero_division=0
        )
    }


# ==================================================
# 9. TRAINING
# ==================================================

best_val_f1 = -1.0
best_epoch = 0
epochs_without_improvement = 0

history = []

print("\n========== EXPERIMENT 3 TRAINING ==========")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    epoch_start = time.time()

    for x, y in train_loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs, y)

        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item() * len(y)

        correct += (
            outputs.argmax(dim=1) == y
        ).sum().item()

        total += len(y)

    train_loss = total_loss / total
    train_accuracy = correct / total

    val_metrics = evaluate(model, val_loader)

    scheduler.step(val_metrics["f1"])

    current_lr = optimizer.param_groups[0]["lr"]

    record = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "learning_rate": current_lr
    }

    history.append(record)

    elapsed = time.time() - epoch_start

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val Acc: {val_metrics['accuracy']:.4f} | "
        f"Val F1: {val_metrics['f1']:.4f} | "
        f"Time: {elapsed:.1f}s"
    )

    # Save the checkpoint with the best validation F1.
    if val_metrics["f1"] > best_val_f1:

        best_val_f1 = val_metrics["f1"]
        best_epoch = epoch + 1

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            MODEL_DIR / "best_model.pth"
        )

        print("  Best F1 checkpoint saved.")

    else:

        epochs_without_improvement += 1

    pd.DataFrame(history).to_csv(
        MODEL_DIR / "training_history.csv",
        index=False
    )

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")
        break


# ==================================================
# 10. SAVE SUMMARY
# ==================================================

summary = {

    "experiment": "CNN-BiLSTM V4 - Moderate Category-Aware Sampling",

    "best_epoch": best_epoch,

    "best_validation_f1": float(best_val_f1),

    "epochs_completed": len(history),

    "sampling_strategy": "Inverse square root frequency of category-label combinations",

    "category_label_combinations": int(len(stratum_counts)),

    "configuration": {

        "projection_dim": PROJECTION_DIM,

        "hidden_dim": HIDDEN_DIM,

        "lstm_layers": NUM_LAYERS,

        "bidirectional": True,

        "dropout": 0.4,

        "batch_size": BATCH_SIZE,

        "learning_rate": LEARNING_RATE,

        "weight_decay": WEIGHT_DECAY,

        "loss": "CrossEntropyLoss without class weights"
    }
}

with open(
    MODEL_DIR / "training_summary.json",
    "w"
) as file:

    json.dump(summary, file, indent=4)


print("\n========== EXPERIMENT 3 COMPLETED ==========")

print("Best epoch:", best_epoch)

print("Best validation F1:", round(best_val_f1, 4))

print("Output directory:", MODEL_DIR)
