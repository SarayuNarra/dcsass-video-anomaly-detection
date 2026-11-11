
from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ---------------- CONFIG ----------------
ROOT = Path(__file__).resolve().parent
FEATURE_ROOT = ROOT / "mobilenet_features_full"
OUTPUT = ROOT / "model_outputs_experiment5"
OUTPUT.mkdir(exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 20
PATIENCE = 4
LR = 0.0003
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
print("Device:", device)

# ---------------- LOAD DATA ----------------
def load_split(name):
    features = np.load(
        FEATURE_ROOT / name / "features.npy",
        mmap_mode="r"
    )
    labels = pd.read_csv(ROOT / f"{name}.csv")["label"].values
    return features, labels

X_train, y_train = load_split("train")
X_val, y_val = load_split("validation")

print("Train:", X_train.shape)
print("Validation:", X_val.shape)

# Training-only normalization
mean = np.asarray(X_train.mean(axis=(0, 1)), dtype=np.float32)
std = np.asarray(X_train.std(axis=(0, 1)), dtype=np.float32)
std[std < 1e-6] = 1.0

np.save(OUTPUT / "mean.npy", mean)
np.save(OUTPUT / "std.npy", std)

class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = np.asarray(self.X[idx], dtype=np.float32)
        x = (x - mean) / std
        return torch.from_numpy(x), torch.tensor(
            self.y[idx], dtype=torch.long
        )

train_loader = DataLoader(
    FeatureDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    FeatureDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ---------------- MODEL ----------------
class CNNBiGRU(nn.Module):
    def __init__(self):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.projection(x)
        _, hidden = self.gru(x)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined = torch.cat(
            (forward_hidden, backward_hidden), dim=1
        )

        return self.classifier(combined)

model = CNNBiGRU().to(device)

# Weighted loss
counts = np.bincount(y_train, minlength=2)
weights = len(y_train) / (2.0 * counts)
class_weights = torch.tensor(weights, dtype=torch.float32)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0.0001
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)

# ---------------- TRAINING ----------------
best_f1 = -1
best_epoch = 0
patience_counter = 0
history = []

for epoch in range(1, EPOCHS + 1):

    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item() * y.size(0)
        train_correct += (outputs.argmax(1) == y).sum().item()
        train_total += y.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total

    # Validation
    model.eval()
    val_loss = 0
    actual, predicted = [], []

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)

            outputs = model(X)
            loss = criterion(outputs, y)

            val_loss += loss.item() * y.size(0)
            predictions = outputs.argmax(1)

            actual.extend(y.cpu().numpy())
            predicted.extend(predictions.cpu().numpy())

    val_loss /= len(y_val)
    val_acc = accuracy_score(actual, predicted)
    val_f1 = f1_score(actual, predicted, zero_division=0)
    val_precision = precision_score(
        actual, predicted, zero_division=0
    )
    val_recall = recall_score(
        actual, predicted, zero_division=0
    )

    scheduler.step(val_f1)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val F1: {val_f1:.4f}"
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1
    })

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_epoch = epoch
        patience_counter = 0

        torch.save({
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "epoch": epoch,
            "best_val_f1": best_f1
        }, OUTPUT / "best_model.pth")

        print("  Best model saved!")

    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print("Early stopping.")
        break

pd.DataFrame(history).to_csv(
    OUTPUT / "training_history.csv", index=False
)

print("\nEXPERIMENT 5 COMPLETED")
print("Best epoch:", best_epoch)
print("Best validation F1:", round(best_f1, 4))
print("Output:", OUTPUT)
