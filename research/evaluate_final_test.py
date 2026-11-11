from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score
)

# ==================================================
# 1. CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent

FEATURE_DIR = ROOT / "mobilenet_features_full"

# Experiment 2 is our selected model.
MODEL_DIR = ROOT / "model_outputs_experiment2"

DEVICE = torch.device("cpu")
BATCH_SIZE = 32

INPUT_DIM = 1280
PROJECTION_DIM = 128
HIDDEN_DIM = 64
NUM_CLASSES = 2

print("Device:", DEVICE)
print("Final evaluation: Experiment 2 CNN-BiLSTM")


# ==================================================
# 2. MODEL ARCHITECTURE
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
            num_layers=1,
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


# ==================================================
# 3. TEST DATASET
# ==================================================

class FeatureDataset(Dataset):

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
# 4. LOAD TEST FEATURES
# ==================================================

print("\nLoading untouched test dataset...")

features = np.load(
    FEATURE_DIR / "test" / "features.npy",
    mmap_mode="r"
)

metadata = pd.read_csv(
    FEATURE_DIR / "test" / "metadata.csv"
)

labels = metadata["label"].to_numpy()

assert len(features) == len(metadata), (
    "Feature and metadata counts do not match!"
)

print("Test feature shape:", features.shape)
print("Test videos:", len(metadata))
print("Test source groups:", metadata["source_group"].nunique())


# ==================================================
# 5. LOAD EXPERIMENT 2 NORMALIZATION
# ==================================================

print("\nLoading Experiment 2 normalization...")

normalization = np.load(
    MODEL_DIR / "normalization.npz"
)

mean = normalization["mean"]
std = normalization["std"]

dataset = FeatureDataset(
    features,
    labels,
    mean,
    std
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==================================================
# 6. LOAD SELECTED MODEL
# ==================================================

print("\nLoading selected Experiment 2 checkpoint...")

model = CNNLSTMClassifier()

model.load_state_dict(
    torch.load(
        MODEL_DIR / "best_model.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ==================================================
# 7. GENERATE TEST PREDICTIONS
# ==================================================

print("\nGenerating test predictions...")

actual = []
predicted = []
probabilities = []

with torch.inference_mode():

    for x, y in loader:

        x = x.to(DEVICE)

        outputs = model(x)

        probs = torch.softmax(outputs, dim=1)

        predictions = outputs.argmax(dim=1)

        actual.extend(y.numpy())
        predicted.extend(predictions.numpy())
        probabilities.extend(probs[:, 1].numpy())


actual = np.array(actual)
predicted = np.array(predicted)
probabilities = np.array(probabilities)


# ==================================================
# 8. OVERALL TEST METRICS
# ==================================================

print("\n========== FINAL TEST RESULTS ==========")

accuracy = accuracy_score(actual, predicted)

precision = precision_score(
    actual, predicted, zero_division=0
)

recall = recall_score(
    actual, predicted, zero_division=0
)

f1 = f1_score(
    actual, predicted, zero_division=0
)

macro_f1 = f1_score(
    actual, predicted,
    average="macro",
    zero_division=0
)

balanced_acc = balanced_accuracy_score(
    actual, predicted
)

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))
print("Macro F1:", round(macro_f1, 4))
print("Balanced Accuracy:", round(balanced_acc, 4))


# ==================================================
# 9. CONFUSION MATRIX
# ==================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    actual,
    predicted,
    labels=[0, 1]
)

print(cm)


# ==================================================
# 10. CLASSIFICATION REPORT
# ==================================================

print("\nClassification Report:")

print(
    classification_report(
        actual,
        predicted,
        labels=[0, 1],
        zero_division=0
    )
)


# ==================================================
# 11. ROC-AUC
# ==================================================

print("\nROC-AUC:")

if len(np.unique(actual)) == 2:

    auc = roc_auc_score(
        actual,
        probabilities
    )

    print(round(auc, 4))

else:

    auc = None
    print("Cannot calculate: only one class present.")


# ==================================================
# 12. CATEGORY-WISE EVALUATION
# ==================================================

print("\n========== CATEGORY-WISE TEST RESULTS ==========")

metadata["actual"] = actual
metadata["predicted"] = predicted
metadata["probability_1"] = probabilities

results = []

for category, group in metadata.groupby("category"):

    y_true = group["actual"]
    y_pred = group["predicted"]

    results.append({

        "Category": category,

        "Videos": len(group),

        "Label_0_Count": int((y_true == 0).sum()),

        "Label_1_Count": int((y_true == 1).sum()),

        "Accuracy": accuracy_score(y_true, y_pred),

        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),

        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),

        "F1": f1_score(
            y_true, y_pred, zero_division=0
        ),

        "Balanced_Accuracy": balanced_accuracy_score(
            y_true, y_pred
        )
    })


category_results = pd.DataFrame(results)

print(category_results.round(4).to_string(index=False))


# ==================================================
# 13. CATEGORY MACRO AVERAGES
# ==================================================

print("\n========== CATEGORY MACRO AVERAGES ==========")

print(
    "Mean Category Accuracy:",
    round(category_results["Accuracy"].mean(), 4)
)

print(
    "Mean Category F1:",
    round(category_results["F1"].mean(), 4)
)

print(
    "Mean Category Balanced Accuracy:",
    round(
        category_results["Balanced_Accuracy"].mean(),
        4
    )
)


# ==================================================
# 14. SAVE FINAL RESULTS
# ==================================================

category_results.to_csv(
    ROOT / "final_test_category_results.csv",
    index=False
)

metadata.to_csv(
    ROOT / "final_test_predictions.csv",
    index=False
)

summary = {
    "selected_model": "Experiment 2 CNN-BiLSTM",
    "test_videos": len(actual),
    "test_source_groups": int(
        metadata["source_group"].nunique()
    ),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "macro_f1": float(macro_f1),
    "balanced_accuracy": float(balanced_acc),
    "roc_auc": float(auc) if auc is not None else None
}

import json

with open(
    ROOT / "final_test_summary.json",
    "w"
) as file:

    json.dump(summary, file, indent=4)


print("\n========== FINAL EVALUATION COMPLETED ==========")

print("Saved files:")
print("1. final_test_category_results.csv")
print("2. final_test_predictions.csv")
print("3. final_test_summary.json")
