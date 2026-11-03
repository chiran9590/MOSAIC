
from pathlib import Path
import numpy as np

RAW_ROOT = Path(
    "data/raw/benchmark_full/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore"
)

OUTPUT_ROOT = Path("data/processed")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

TARGET_ROW = 148  # Zero-based index: original row 149


def process_file(file_path, output_path):
    print(f"\nProcessing: {file_path.name}")

    # Load the original FI-2010 matrix.
    data = np.loadtxt(file_path, dtype=np.float32)

    if data.shape[0] != 149:
        raise ValueError(
            f"Expected 149 rows, got {data.shape[0]} in {file_path}"
        )

    # First 144 rows are input features.
    # Transpose to: observations x features.
    X = data[:144].T.copy()

    # Select original label row 149.
    y = data[TARGET_ROW].astype(np.int64).copy()

    # Validate target classes.
    if not np.isin(y, [1, 2, 3]).all():
        raise ValueError(f"Unexpected target values in {file_path}")

    # Save features and labels separately.
    np.savez_compressed(
        output_path,
        X=X,
        y=y
    )

    values, counts = np.unique(y, return_counts=True)

    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)
    print("Label distribution:", dict(zip(values.tolist(), counts.tolist())))
    print("Saved:", output_path)


def main():
    groups = {
        "train": RAW_ROOT / "NoAuction_Zscore_Training",
        "test": RAW_ROOT / "NoAuction_Zscore_Testing",
    }

    for split, folder in groups.items():
        files = sorted(folder.glob("*.txt"))

        if len(files) != 9:
            raise ValueError(
                f"Expected 9 {split} files, found {len(files)}"
            )

        for file_path in files:
            fold = file_path.stem.split("_")[-1]

            output_path = OUTPUT_ROOT / f"{split}_cf_{fold}.npz"

            process_file(file_path, output_path)

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
