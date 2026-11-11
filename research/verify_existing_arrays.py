import os
import numpy as np

BASE_DIR = "mobilenet_features_full"

splits = {
    "train": 11604,
    "validation": 2493,
    "test": 2493
}

EXPECTED_SHAPE = (16, 1280)

print("\nFEATURE ARRAY VERIFICATION")
print("=" * 55)

for split, expected_videos in splits.items():

    feature_path = os.path.join(
        BASE_DIR, split, "features.npy"
    )

    print(f"\nChecking {split}...")

    if not os.path.exists(feature_path):
        print("ERROR: Feature file missing!")
        continue

    features = np.load(
        feature_path,
        mmap_mode="r"
    )

    print("Shape:", features.shape)
    print("Data type:", features.dtype)

    assert features.shape == (
        expected_videos,
        *EXPECTED_SHAPE
    ), f"Unexpected shape in {split}"

    assert np.issubdtype(
        features.dtype, np.floating
    ), "Expected floating-point features"

    finite = np.isfinite(features).all()

    print("All values finite:", finite)

    assert finite, f"Invalid values in {split}"

    print("STATUS: PASSED")

print("\n" + "=" * 55)
print("FEATURE VERIFICATION COMPLETED")
