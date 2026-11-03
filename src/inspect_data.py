
from pathlib import Path
import zipfile
import numpy as np

ROOT = Path("data/raw")

print("\n" + "=" * 55)
print("MOSAIC | FI-2010 DATA INSPECTION")
print("=" * 55)

files = [p for p in ROOT.rglob("*") if p.is_file()]

if not files:
    print("No files found. Put the downloaded archive in data/raw/")
    raise SystemExit(1)

print(f"\nFiles found: {len(files)}")

for file in files:
    size_mb = file.stat().st_size / (1024 * 1024)

    print(f"\nFILE: {file.relative_to(ROOT)}")
    print(f"SIZE: {size_mb:.2f} MB")

    if file.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(file) as archive:
                names = archive.namelist()
                print("ZIP contents (first 15):")
                for name in names[:15]:
                    print("  ", name)
                print("Total entries:", len(names))
        except zipfile.BadZipFile:
            print("Could not read ZIP archive.")

    elif file.suffix.lower() == ".txt":
        try:
            sample = np.loadtxt(file, max_rows=5)
            print("Preview shape:", sample.shape)
            print("Preview:")
            print(sample)
        except Exception as error:
            print("Text preview failed:", error)

    elif file.suffix.lower() == ".npy":
        try:
            arr = np.load(file, mmap_mode="r")
            print("Array shape:", arr.shape)
            print("Data type:", arr.dtype)
        except Exception as error:
            print("Array preview failed:", error)

print("\n" + "=" * 55)
print("INSPECTION FINISHED")
print("=" * 55)
