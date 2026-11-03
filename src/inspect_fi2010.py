
from pathlib import Path
import numpy as np

ROOT = Path("data/raw/benchmark_full")

files = sorted(ROOT.rglob("*NoAuction*ZScore*"))

print("=" * 55)
print("MOSAIC - FI-2010 DATASET STRUCTURE")
print("=" * 55)

for file in files[:6]:
    if not file.is_file():
        continue

    print(f"\nFile: {file.name}")

    try:
        data = np.loadtxt(file)

        print("Shape:", data.shape)
        print("Data type:", data.dtype)
        print("First 5 rows, first 8 columns:")
        print(data[:5, :8])

        print("Minimum:", np.min(data))
        print("Maximum:", np.max(data))

    except Exception as e:
        print("Error:", e)

print("\nInspection finished.")
