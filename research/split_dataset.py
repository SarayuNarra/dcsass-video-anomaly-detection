import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Load metadata
df = pd.read_csv("full_preprocessed_metadata.csv")

# Keep successfully processed videos
df = df[df["status"] == "success"].copy()

# Convert labels to integers
df["label"] = df["label"].astype(int)

# First split: 85% training + validation, 15% testing
split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.15,
    random_state=42
)

train_val_idx, test_idx = next(
    split1.split(df, groups=df["source_group"])
)

train_val = df.iloc[train_val_idx].copy()
test = df.iloc[test_idx].copy()

# Second split: 70% training, 15% validation
# 15 / 85 = 0.17647
split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.17647,
    random_state=42
)

train_idx, val_idx = next(
    split2.split(
        train_val,
        groups=train_val["source_group"]
    )
)

train = train_val.iloc[train_idx].copy()
val = train_val.iloc[val_idx].copy()

# Save datasets
train.to_csv("train.csv", index=False)
val.to_csv("validation.csv", index=False)
test.to_csv("test.csv", index=False)

# Display statistics
splits = {
    "Training": train,
    "Validation": val,
    "Testing": test
}

print("\n========== DATASET SPLIT ==========")

for name, data in splits.items():
    print(f"\n{name}")
    print("Videos:", len(data))
    print("Source groups:", data["source_group"].nunique())
    print("Label distribution:")
    print(data["label"].value_counts().sort_index())

# Verify no source group leakage
train_groups = set(train["source_group"])
val_groups = set(val["source_group"])
test_groups = set(test["source_group"])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

print("\nNo source-group leakage detected!")
print("Train, validation and test CSV files saved.")