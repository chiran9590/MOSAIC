from pathlib import Path
import numpy as np

ROOT = Path("data/raw/benchmark_full/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore")

files = [
    ROOT / "NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_1.txt",
    ROOT / "NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_1.txt",
]

for file in files:
    print("\n" + "=" * 60)
    print("FILE:", file.name)

    data = np.loadtxt(file, dtype=np.float32)

    features = data[:144]
    labels = data[144:]

    constant_rows = np.where(np.ptp(features, axis=1) == 0)[0]

    print("Feature matrix shape:", features.shape)
    print("Label matrix shape:", labels.shape)
    print("Constant feature rows (1-based):", constant_rows + 1)

    print("\nLabel distributions:")

    for i, label in enumerate(labels):
        values, counts = np.unique(label, return_counts=True)
        distribution = dict(zip(values.astype(int), counts))
        print(f"Label row {i + 145}: {distribution}")