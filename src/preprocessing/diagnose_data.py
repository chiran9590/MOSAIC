from pathlib import Path
from collections import Counter

ROOT = Path("data/raw/benchmark_full")

file_groups = {
    "TRAINING": sorted(ROOT.rglob("Train_Dst_NoAuction_ZScore_CF_*.txt")),
    "TESTING": sorted(ROOT.rglob("Test_Dst_NoAuction_ZScore_CF_*.txt")),
}

for group, files in file_groups.items():
    print(f"\n{'=' * 65}")
    print(group, "FILES:", len(files))

    for file in files:
        counts = Counter()

        with open(file, "r") as f:
            for line in f:
                counts[len(line.split())] += 1

        consistent = len(counts) == 1

        print(
            f"{file.name}: "
            f"rows={sum(counts.values())}, "
            f"widths={dict(counts)}, "
            f"consistent={consistent}"
        )