import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent

files = [
    "train.csv",
    "validation.csv",
    "test.csv"
]

for filename in files:

    path = ROOT / filename
    df = pd.read_csv(path)

    print("\n" + "=" * 55)
    print("FILE:", filename)
    print("=" * 55)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index())

    # Identify possible category column
    category_columns = [
        c for c in df.columns
        if c.lower() in [
            "category", "class", "activity",
            "crime_type", "video_category"
        ]
    ]

    for column in category_columns:

        print(f"\n{column} vs label:")

        print(pd.crosstab(
            df[column],
            df["label"]
        ).to_string())

    print("\nSample records:")
    print(df.head(3).to_string(index=False))

print("\nCSV LABEL VERIFICATION COMPLETED")
