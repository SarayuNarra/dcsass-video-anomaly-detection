import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent

exp1 = pd.read_csv(
    ROOT / "model_outputs_experiment1" / "training_history.csv"
)

exp2 = pd.read_csv(
    ROOT / "model_outputs_experiment2" / "training_history.csv"
)

def best_row(df, metric):
    return df.loc[df[metric].idxmax()]

results = []

for name, df in [
    ("Experiment 1", exp1),
    ("Experiment 2", exp2)
]:

    accuracy = best_row(df, "val_accuracy")
    f1 = best_row(df, "val_f1")

    results.append({
        "Experiment": name,
        "Best Accuracy": accuracy["val_accuracy"],
        "Accuracy Epoch": int(accuracy["epoch"]),
        "Best F1": f1["val_f1"],
        "F1 Epoch": int(f1["epoch"]),
        "Best Validation Loss": df["val_loss"].min(),
        "Final Train Accuracy": df.iloc[-1]["train_accuracy"],
        "Final Validation Accuracy": df.iloc[-1]["val_accuracy"]
    })

comparison = pd.DataFrame(results)

print("\n========== EXPERIMENT COMPARISON ==========")
print(comparison.to_string(index=False))

comparison.to_csv(
    ROOT / "experiment_comparison.csv",
    index=False
)

print("\nSaved experiment_comparison.csv")
