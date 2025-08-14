import re
import csv
from pathlib import Path

log_path = Path("python_code/run_band_combination_finder.sh.39561.out")
csv_path = Path("HyperspectralMultiCNN_band_accuracy.csv")

results = []

with open(log_path, "r", encoding="utf-8", errors="replace") as file:
    for line in file:
        match = re.search(r"bands: \((.*?)\) \| accuracy: ([0-9.]+)", line)
        if match:
            bands = match.group(1).replace(" ", "")
            accuracy = float(match.group(2))
            results.append([bands, round(accuracy, 3)])

with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["bands", "accuracy"])
    writer.writerows(results)

print(f"✅ Saved {len(results)} entries to {csv_path}")